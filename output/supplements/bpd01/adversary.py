"""Time-shared PersistentPGD adversary and multi-step adversarial stability evaluation.

In Goodfire's single-pass VPD, ``PersistentPGD`` shares adversarial mask sources
across the batch (``[1, 1, C + 1]``). On a weight-tied loop, an unconstrained
adversary can also resonate across passes (flipping ``s_{tau}`` in opposite
phases across ``tau``), even though any physical edit to a shared weight matrix
is time-shared across passes by construction.

``PassSharedPersistentPGD`` shares sources across both batch and passes by default
(``scope="shared_batch_and_pass"``, shape ``[1, 1, 1, C + 1]``), while also
supporting ``"shared_pass_per_batch"`` (``[B, 1, P, C + 1]``) and ``"per_pass"``
(``[B, T, P, C + 1]``) for controlled ablations and ``T = 1`` VPD equivalence.

``evaluate_pgd_stability_curve`` directly audits the Goodfire failure mode where
reconstruction KL rises sharply with adversary step count (e.g., ``0.83`` at
20 steps to ``25.26`` at 160 steps), contrasting time-shared vs pass-resonant PGD.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

import torch
from torch import Tensor

from mechinterp.bpd.components import PassMaskSchedule, RolloutTrajectory
from mechinterp.bpd.config import AdversaryConfig, AdversaryScope


class PassSharedPersistentPGD:
    """Persistent Adam-updated adversarial mask sources in ``[0, 1]``."""

    def __init__(
        self,
        c_per_module: Mapping[str, int],
        cfg: AdversaryConfig | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg or AdversaryConfig()
        self.module_order = sorted(c_per_module.keys())
        self.c_per_module = {k: int(c_per_module[k]) for k in self.module_order}
        self.device = torch.device(device)

        self.sources: dict[str, Tensor] = {}
        self.exp_avg: dict[str, Tensor] = {}
        self.exp_avg_sq: dict[str, Tensor] = {}
        self.step_count: int = 0

    def _desired_shape(
        self, bsz: int, n_passes: int, seq_len: int, c: int
    ) -> tuple[int, int, int, int]:
        scope: AdversaryScope = self.cfg.scope
        if scope == "shared_batch_and_pass":
            return (1, 1, 1, c + 1)
        if scope == "shared_pass_per_batch":
            return (bsz, 1, seq_len, c + 1)
        if scope == "per_pass":
            return (bsz, n_passes, seq_len, c + 1)
        raise ValueError(f"Unknown AdversaryScope: {scope!r}")

    def ensure_initialized(
        self,
        bsz: int,
        n_passes: int,
        seq_len: int,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        """Allocate or resize persistent source tensors when shapes change."""
        resized = False
        for name in self.module_order:
            c = self.c_per_module[name]
            shape = self._desired_shape(bsz, n_passes, seq_len, c)
            cur = self.sources.get(name)
            if cur is None or cur.shape != shape or cur.device != self.device:
                src = torch.rand(shape, generator=generator, device=self.device, dtype=torch.float32)
                self.sources[name] = src
                self.exp_avg[name] = torch.zeros_like(src)
                self.exp_avg_sq[name] = torch.zeros_like(src)
                resized = True
        if resized:
            self.step_count = 0

    def lr_at(self, step: int, total_steps: int) -> float:
        if total_steps <= 1:
            return self.cfg.lr
        frac = min(max(float(step) / float(total_steps), 0.0), 1.0)
        mult = 1.0 + (self.cfg.lr_final_frac - 1.0) * frac
        return float(self.cfg.lr * mult)

    def build_schedules(
        self,
        ci_lower: Mapping[str, Tensor],
        *,
        requires_grad_sources: bool = False,
    ) -> tuple[dict[str, PassMaskSchedule], dict[str, Tensor]]:
        """Construct ``m_comp = ci + (1 - ci) * s_comp``, ``m_delta = s_delta``."""
        first = ci_lower[self.module_order[0]]
        bsz, n_passes, seq_len, _ = first.shape
        self.ensure_initialized(bsz, n_passes, seq_len)

        leaf_sources: dict[str, Tensor] = {}
        schedules: dict[str, PassMaskSchedule] = {}
        for name in self.module_order:
            ci = ci_lower[name]
            src = self.sources[name].detach()
            if requires_grad_sources:
                src = src.clone().requires_grad_(True)
            leaf_sources[name] = src

            s_comp = src[..., :-1].expand(bsz, n_passes, seq_len, ci.shape[-1])
            s_delta = src[..., -1].expand(bsz, n_passes, seq_len)
            comp_mask = ci + (1.0 - ci) * s_comp.to(ci.dtype)
            schedules[name] = PassMaskSchedule(
                comp_mask=comp_mask,
                delta_mask=s_delta.to(ci.dtype),
            )
        return schedules, leaf_sources

    @torch.no_grad()
    def ascent_step(self, grads: Mapping[str, Tensor], *, lr: float) -> None:
        """Apply one Adam gradient-ascent update on ``sources`` and clamp to ``[0, 1]``."""
        self.step_count += 1
        b1, b2, eps = self.cfg.beta1, self.cfg.beta2, self.cfg.eps
        bc1 = 1.0 - (b1 ** self.step_count)
        bc2 = 1.0 - (b2 ** self.step_count)

        for name in self.module_order:
            g = grads[name].detach().float()
            # Negate gradient so standard Adam descent performs gradient ASCENT on recon loss
            neg_g = -g
            m = self.exp_avg[name]
            v = self.exp_avg_sq[name]
            m.mul_(b1).add_(neg_g, alpha=1.0 - b1)
            v.mul_(b2).addcmul_(neg_g, neg_g, value=1.0 - b2)
            m_hat = m / bc1
            v_hat = v / bc2
            step_vec = m_hat / (v_hat.sqrt() + eps)
            self.sources[name].sub_(step_vec, alpha=lr).clamp_(0.0, 1.0)

    def warmup_inner_steps(
        self,
        ci_lower_detached: Mapping[str, Tensor],
        eval_loss_fn: Callable[[dict[str, PassMaskSchedule]], Tensor],
        *,
        n_steps: int | None = None,
        lr: float | None = None,
    ) -> None:
        """Run ``inner_steps`` of PGD maximization with component/CI parameters frozen."""
        k = self.cfg.inner_steps if n_steps is None else int(n_steps)
        step_lr = self.cfg.lr if lr is None else float(lr)
        for _ in range(k):
            schedules, leaves = self.build_schedules(
                ci_lower_detached, requires_grad_sources=True
            )
            loss = eval_loss_fn(schedules)
            leaf_list = [leaves[name] for name in self.module_order]
            grads_tuple = torch.autograd.grad(loss, leaf_list, only_inputs=True)
            grads = {name: g for name, g in zip(self.module_order, grads_tuple)}
            self.ascent_step(grads, lr=step_lr)


@dataclass
class PGDStabilityReport:
    """Adversarial reconstruction loss across PGD step counts ``K_adv``."""

    steps_grid: tuple[int, ...]
    loss_by_steps: dict[int, float]
    scope: AdversaryScope

    @property
    def amplification_ratio(self) -> float:
        """Ratio between max-step adversarial loss and 0-step random baseline."""
        if not self.steps_grid:
            return 1.0
        base = max(self.loss_by_steps[self.steps_grid[0]], 1e-8)
        last = self.loss_by_steps[self.steps_grid[-1]]
        return float(last / base)


def evaluate_pgd_stability_curve(
    ci_lower: Mapping[str, Tensor],
    c_per_module: Mapping[str, int],
    eval_loss_fn: Callable[[dict[str, PassMaskSchedule]], Tensor],
    *,
    scope: AdversaryScope = "shared_batch_and_pass",
    steps_grid: tuple[int, ...] = (0, 2, 5, 20, 80, 160),
    step_size: float = 0.1,
    seed: int = 0,
) -> PGDStabilityReport:
    """Fresh Sign-PGD evaluation across ``steps_grid`` (Goodfire 20-vs-160 step audit).

    Starts from fresh uniform ``U(0, 1)`` sources and records reconstruction loss
    at every milestone in ``steps_grid`` using Sign-PGD ascent ``s <- clamp(s + alpha * sign(g), 0, 1)``.
    """
    names = sorted(c_per_module.keys())
    ci_det = {k: ci_lower[k].detach() for k in names}
    first = ci_det[names[0]]
    bsz, n_passes, seq_len, _ = first.shape
    dev = first.device

    gen = torch.Generator(device=dev).manual_seed(seed)
    sources: dict[str, Tensor] = {}
    for name in names:
        c = int(c_per_module[name])
        if scope == "shared_batch_and_pass":
            shape = (1, 1, 1, c + 1)
        elif scope == "shared_pass_per_batch":
            shape = (bsz, 1, seq_len, c + 1)
        elif scope == "per_pass":
            shape = (bsz, n_passes, seq_len, c + 1)
        else:
            raise ValueError(f"Unknown scope: {scope!r}")
        sources[name] = torch.rand(shape, generator=gen, device=dev, dtype=torch.float32)

    sorted_steps = tuple(sorted(set(int(s) for s in steps_grid)))
    max_step = sorted_steps[-1] if sorted_steps else 0
    loss_by_steps: dict[int, float] = {}

    def _make_sched(req_grad: bool) -> tuple[dict[str, PassMaskSchedule], list[Tensor]]:
        leaves: list[Tensor] = []
        sched: dict[str, PassMaskSchedule] = {}
        for name in names:
            ci = ci_det[name]
            src = sources[name].detach()
            if req_grad:
                src = src.clone().requires_grad_(True)
            leaves.append(src)
            s_comp = src[..., :-1].expand(bsz, n_passes, seq_len, ci.shape[-1])
            s_delta = src[..., -1].expand(bsz, n_passes, seq_len)
            sched[name] = PassMaskSchedule(
                comp_mask=ci + (1.0 - ci) * s_comp.to(ci.dtype),
                delta_mask=s_delta.to(ci.dtype),
            )
        return sched, leaves

    for k in range(max_step + 1):
        need_grad = k < max_step
        sched, leaves = _make_sched(req_grad=need_grad)
        if need_grad:
            loss = eval_loss_fn(sched)
            if k in sorted_steps:
                loss_by_steps[k] = float(loss.detach().item())
            grads = torch.autograd.grad(loss, leaves, only_inputs=True)
            with torch.no_grad():
                for name, g in zip(names, grads):
                    sources[name].add_(g.sign(), alpha=step_size).clamp_(0.0, 1.0)
        else:
            with torch.no_grad():
                loss = eval_loss_fn(sched)
            if k in sorted_steps:
                loss_by_steps[k] = float(loss.detach().item())

    return PGDStabilityReport(
        steps_grid=sorted_steps,
        loss_by_steps=loss_by_steps,
        scope=scope,
    )
