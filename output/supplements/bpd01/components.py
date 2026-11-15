"""Weight-tied and unrolled linear component wrappers for BPD.

Each physical shared ``nn.Linear`` owns a single factor pair ``(V, U)`` with
``W_hat = (V @ U).T`` (shape ``[d_out, d_in]``) and residual
``Delta_W = W_target - (V @ U).T``. Across ``T`` invocations in a recurrent
forward pass, ``TiedComponentLinear`` tracks the pass index ``tau``, records
per-pass inputs, and applies pass-specific component masks with optional
pass-conditional mean-write ablation and Post-RMSNorm / sphere norm preservation.

``UnrolledComponentLinear`` provides the falsifiable unrolled VPD baseline
where each pass ``tau`` owns an independent ``(V_tau, U_tau)`` factor pair
(``T * C`` total components per matrix).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any, Callable, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mechinterp.bpd.config import AblationConfig


class _LowerLeakyHardSigmoid(torch.autograd.Function):
    """Goodfire lower-leaky hard sigmoid: clamp(x, 0, 1) with resurrecting negative leak."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, alpha: float) -> Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return x.clamp(0.0, 1.0)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor, None]:
        grad_output = grad_outputs[0]
        (x,) = ctx.saved_tensors
        alpha: float = ctx.alpha
        zero = torch.zeros_like(grad_output)
        grad = torch.where(
            x <= 0.0,
            torch.where(grad_output < 0.0, alpha * grad_output, zero),
            torch.where(x <= 1.0, grad_output, zero),
        )
        return grad, None


def lower_leaky_sigmoid(x: Tensor, alpha: float = 0.01) -> Tensor:
    """Clamp to [0, 1] with resurrection gradient ``alpha`` when ``x <= 0`` and ``grad < 0``."""
    return _LowerLeakyHardSigmoid.apply(x, alpha)  # type: ignore[return-value]


def upper_leaky_sigmoid(x: Tensor, alpha: float = 0.01) -> Tensor:
    """Clamp to [0, 1] below 1, linear slope ``alpha`` above 1 for minimality pressure."""
    return torch.where(x > 1.0, 1.0 + alpha * (x - 1.0), x.clamp(0.0, 1.0))


@dataclass
class PassMaskSchedule:
    """Per-module masks across passes for a single forward execution.

    Shapes (with ``T`` passes, batch ``B``, positions ``P``, components ``C``):
      - ``comp_mask``: ``[B, T, P, C]`` in ``[0, 1]``
      - ``delta_mask``: ``[B, T, P]`` in ``[0, 1]`` (or ``None`` for unmasked delta=1, or 0)
      - ``routing_mask``: ``[B, T, P]`` bool (``True`` routes through components, ``False`` keeps target)
    """

    comp_mask: Tensor
    delta_mask: Tensor | None = None
    routing_mask: Tensor | None = None


@dataclass
class ComponentIntervention:
    """Additive or multiplicative intervention on selected components at selected passes."""

    components: tuple[int, ...]
    passes: frozenset[int] | None  # None means all passes
    mode: Literal["ablate", "scale", "add_alpha", "override_read"]
    scale: float = 0.0
    read_override: Tensor | None = None  # [d_in, len(components)] for rotation nulls


def _svd_or_random_factors(
    weight: Tensor,
    c: int,
    *,
    init_svd: bool,
) -> tuple[Tensor, Tensor]:
    """Initialize V [d_in, C] and U [C, d_out] so (V @ U).T approximates top-C SVD or random."""
    d_out, d_in = weight.shape
    if init_svd:
        w64 = weight.detach().double().cpu()
        u_svd, s_svd, vh_svd = torch.linalg.svd(w64, full_matrices=False)
        k = min(c, int(s_svd.numel()))
        v_init = torch.zeros((d_in, c), dtype=torch.float32)
        u_init = torch.zeros((c, d_out), dtype=torch.float32)
        sqrt_s = s_svd[:k].sqrt()
        # weight = u_svd @ diag(s) @ vh_svd => weight.T = vh_svd.T @ diag(s) @ u_svd.T = V @ U
        v_init[:, :k] = (vh_svd[:k].T * sqrt_s.unsqueeze(0)).float()
        u_init[:k, :] = (sqrt_s.unsqueeze(1) * u_svd[:, :k].T).float()
        if c > k:
            v_init[:, k:].normal_(0.0, 1.0 / math.sqrt(d_in))
            u_init[k:, :].normal_(0.0, 0.01 / math.sqrt(c))
        return v_init.to(weight.device), u_init.to(weight.device)

    v_init = torch.empty((d_in, c), device=weight.device, dtype=torch.float32).normal_(
        0.0, 1.0 / math.sqrt(d_in)
    )
    u_init = torch.empty((c, d_out), device=weight.device, dtype=torch.float32).normal_(
        0.0, 1.0 / math.sqrt(c)
    )
    return v_init, u_init


class TiedComponentLinear(nn.Module):
    """Weight-tied linear replacement owning one ``(V, U)`` pair across ``T`` passes."""

    W_target: Tensor
    bias: Tensor | None
    mean_comp_acts: Tensor
    mean_delta_out: Tensor
    mean_write_rms: Tensor
    calibrated_passes: Tensor

    def __init__(
        self,
        linear: nn.Linear,
        c: int,
        *,
        max_passes: int = 64,
        ablation: AblationConfig | None = None,
        init_svd: bool = True,
    ) -> None:
        super().__init__()
        if c < 1:
            raise ValueError(f"Component count C must be >= 1, got {c}")
        d_out, d_in = linear.weight.shape
        self.d_in = d_in
        self.d_out = d_out
        self.C = c
        self.max_passes = max(max_passes, 1)
        self.ablation = ablation or AblationConfig()

        self.register_buffer("W_target", linear.weight.detach().clone())
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        self.register_buffer("bias", bias)

        v_init, u_init = _svd_or_random_factors(linear.weight, c, init_svd=init_svd)
        self.V = nn.Parameter(v_init)
        self.U = nn.Parameter(u_init)

        # Pass-conditional neutral ablation statistics
        self.register_buffer("mean_comp_acts", torch.zeros(self.max_passes, c, dtype=torch.float32))
        self.register_buffer(
            "mean_delta_out", torch.zeros(self.max_passes, d_out, dtype=torch.float32)
        )
        self.register_buffer("mean_write_rms", torch.zeros(self.max_passes, dtype=torch.float32))
        self.register_buffer("calibrated_passes", torch.zeros(self.max_passes, dtype=torch.bool))

        # Execution state reset on every rollout
        self.mode: Literal["target", "component"] = "target"
        self.schedule: PassMaskSchedule | None = None
        self.intervention: ComponentIntervention | None = None
        self.pass_offset: int = 0
        self.calls: int = 0
        self.record_inputs: bool = False
        self.detach_recorded_inputs: bool = True
        self.update_running_means: bool = False
        self.recorded_inputs: list[Tensor] = []

    def reset_pass_state(self, *, pass_offset: int = 0) -> None:
        """Reset invocation counter and cached per-pass inputs before a rollout."""
        self.calls = 0
        self.pass_offset = pass_offset
        self.recorded_inputs = []

    def reconstructed_weight(self) -> Tensor:
        """Return ``(V @ U).T`` of shape ``[d_out, d_in]`` in float32."""
        return (self.V @ self.U).transpose(0, 1)

    def weight_delta(self) -> Tensor:
        """Return ``W_target.float() - (V @ U).T`` of shape ``[d_out, d_in]``."""
        return self.W_target.float() - self.reconstructed_weight()

    def component_acts(self, x: Tensor) -> Tensor:
        """Project ``x`` ``[..., d_in]`` onto input read basis ``V`` -> ``[..., C]`` (float32)."""
        return x.float() @ self.V

    def _ensure_pass_capacity(self, tau: int) -> None:
        if tau < self.mean_comp_acts.shape[0]:
            return
        new_cap = max(tau + 1, self.mean_comp_acts.shape[0] * 2)
        dev = self.mean_comp_acts.device
        new_comp = torch.zeros((new_cap, self.C), device=dev, dtype=torch.float32)
        new_delta = torch.zeros((new_cap, self.d_out), device=dev, dtype=torch.float32)
        new_rms = torch.zeros((new_cap,), device=dev, dtype=torch.float32)
        new_cal = torch.zeros((new_cap,), device=dev, dtype=torch.bool)
        old = self.mean_comp_acts.shape[0]
        new_comp[:old] = self.mean_comp_acts
        new_delta[:old] = self.mean_delta_out
        new_rms[:old] = self.mean_write_rms
        new_cal[:old] = self.calibrated_passes
        self.mean_comp_acts = new_comp
        self.mean_delta_out = new_delta
        self.mean_write_rms = new_rms
        self.calibrated_passes = new_cal
        self.max_passes = new_cap

    @torch.no_grad()
    def observe_clean_pass(self, x: Tensor, tau: int) -> None:
        """Update pass-conditional mean component activation, residual write, and write RMS."""
        self._ensure_pass_capacity(tau)
        xf = x.detach().float()
        acts_mean = (xf @ self.V).reshape(-1, self.C).mean(dim=0)
        delta_mean = F.linear(xf, self.weight_delta()).reshape(-1, self.d_out).mean(dim=0)
        clean_nobias = F.linear(xf, self.W_target.float()).reshape(-1, self.d_out)
        write_rms = clean_nobias.pow(2).mean(dim=-1).add(self.ablation.eps).sqrt().mean()
        mom = self.ablation.ema_momentum
        if bool(self.calibrated_passes[tau].item()):
            self.mean_comp_acts[tau].lerp_(acts_mean, mom)
            self.mean_delta_out[tau].lerp_(delta_mean, mom)
            self.mean_write_rms[tau].lerp_(write_rms, mom)
        else:
            self.mean_comp_acts[tau].copy_(acts_mean)
            self.mean_delta_out[tau].copy_(delta_mean)
            self.mean_write_rms[tau].copy_(write_rms)
            self.calibrated_passes[tau] = True

    def _apply_intervention(self, x: Tensor, target_out: Tensor, tau: int) -> Tensor:
        assert self.intervention is not None
        iv = self.intervention
        if iv.passes is not None and tau not in iv.passes:
            return target_out

        idx = list(iv.components)
        xf = x.float()
        v_sub = self.V[:, idx]  # [d_in, k]
        u_sub = self.U[idx, :]  # [k, d_out]
        comp_proj = xf @ v_sub  # [..., k]

        if iv.mode == "ablate":
            if self.ablation.mode == "pass_mean" and tau < self.mean_comp_acts.shape[0] and bool(
                self.calibrated_passes[tau].item()
            ):
                base_proj = self.mean_comp_acts[tau, idx]
            else:
                base_proj = torch.zeros_like(comp_proj)
            delta_write = (base_proj - comp_proj) @ u_sub
        elif iv.mode == "scale":
            if self.ablation.mode == "pass_mean" and tau < self.mean_comp_acts.shape[0] and bool(
                self.calibrated_passes[tau].item()
            ):
                base_proj = self.mean_comp_acts[tau, idx]
                delta_write = ((iv.scale - 1.0) * (comp_proj - base_proj)) @ u_sub
            else:
                delta_write = ((iv.scale - 1.0) * comp_proj) @ u_sub
        elif iv.mode == "add_alpha":
            delta_write = (iv.scale * comp_proj) @ u_sub
        elif iv.mode == "override_read":
            if iv.read_override is None:
                raise ValueError("override_read requires read_override tensor")
            r_raw = iv.read_override.to(device=xf.device, dtype=torch.float32)
            r_sub = r_raw[tau] if r_raw.ndim == 3 else r_raw
            rot_proj = xf @ r_sub
            if iv.scale == 0.0:
                # Replace true component write with rotated-read write
                delta_write = (-comp_proj + rot_proj) @ u_sub
            else:
                # Subtract rotated-read write of scale iv.scale (exact HUG-CAL-02 / TCA parity with add_alpha=-scale)
                delta_write = (-iv.scale * rot_proj) @ u_sub
        else:
            raise ValueError(f"Unsupported intervention mode: {iv.mode}")

        apply_sphere = self.ablation.sphere_rescale and (
            iv.mode in ("ablate", "scale") or (iv.mode == "override_read" and iv.scale == 0.0)
        )
        if apply_sphere:
            eps = self.ablation.eps
            if self.bias is not None:
                bias_f = self.bias.float()
                clean_nobias = target_out.float() - bias_f
                new_nobias = clean_nobias + delta_write
                clean_rms = clean_nobias.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
                new_rms = new_nobias.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
                out_f = new_nobias * (clean_rms / new_rms) + bias_f
            else:
                clean_nobias = target_out.float()
                new_nobias = clean_nobias + delta_write
                clean_rms = clean_nobias.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
                new_rms = new_nobias.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
                out_f = new_nobias * (clean_rms / new_rms)
        else:
            out_f = target_out.float() + delta_write
        return out_f.to(target_out.dtype)

    def forward(self, x: Tensor) -> Tensor:
        local_call = self.calls
        tau = self.pass_offset + local_call
        self.calls += 1

        if self.record_inputs:
            self.recorded_inputs.append(x.detach() if self.detach_recorded_inputs else x)

        if self.update_running_means:
            self.observe_clean_pass(x, tau)

        if self.mode == "target":
            target_out = F.linear(x, self.W_target.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)
            if self.intervention is not None:
                return self._apply_intervention(x, target_out, tau)
            return target_out

        if self.schedule is None:
            raise RuntimeError("Component mode requires an active PassMaskSchedule")

        mask_tau = self.schedule.comp_mask[:, local_call]  # [B, P, C] or broadcastable
        xf = x.float()
        comp_acts = xf @ self.V  # [B, P, C]

        is_cal = (
            self.ablation.mode == "pass_mean"
            and tau < self.mean_comp_acts.shape[0]
            and bool(self.calibrated_passes[tau].item())
        )
        if is_cal:
            mu_c = self.mean_comp_acts[tau]  # [C]
            masked_acts = mask_tau * comp_acts + (1.0 - mask_tau) * mu_c
        else:
            masked_acts = mask_tau * comp_acts

        comp_nobias = masked_acts @ self.U  # [B, P, d_out]

        delta_w = self.weight_delta()
        delta_raw = F.linear(xf, delta_w)
        if self.schedule.delta_mask is not None:
            d_mask = self.schedule.delta_mask[:, local_call].unsqueeze(-1)  # [B, P, 1]
            if is_cal:
                mu_d = self.mean_delta_out[tau]
                delta_contrib = d_mask * delta_raw + (1.0 - d_mask) * mu_d
            else:
                delta_contrib = d_mask * delta_raw
        else:
            delta_contrib = delta_raw

        write_nobias = comp_nobias + delta_contrib

        if self.ablation.sphere_rescale and is_cal:
            eps = self.ablation.eps
            ablation_frac = (1.0 - mask_tau.clamp(0.0, 1.0)).mean(dim=-1, keepdim=True)
            cur_rms = write_nobias.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
            target_rms = self.mean_write_rms[tau].clamp_min(eps)
            desired_rms = (1.0 - ablation_frac) * cur_rms + ablation_frac * target_rms
            # Only lift non-degenerate vectors; when ablation_frac == 0, scale is identically 1.
            valid = write_nobias.pow(2).mean(dim=-1, keepdim=True) > eps
            scale = torch.where(valid, desired_rms / cur_rms, torch.ones_like(cur_rms))
            write_nobias = write_nobias * scale

        comp_out = write_nobias + self.bias.float() if self.bias is not None else write_nobias

        if self.schedule.routing_mask is not None:
            r_mask = self.schedule.routing_mask[:, local_call].unsqueeze(-1)
            target_out = F.linear(xf, self.W_target.float(), self.bias.float() if self.bias is not None else None)
            comp_out = torch.where(r_mask, comp_out, target_out)

        return comp_out.to(x.dtype)


class UnrolledComponentLinear(nn.Module):
    """Unrolled VPD baseline: ``T`` independent ``(V_tau, U_tau)`` sets for one shared matrix."""

    W_target: Tensor
    bias: Tensor | None

    def __init__(
        self,
        linear: nn.Linear,
        c_per_pass: int,
        n_passes: int,
        *,
        init_svd: bool = True,
    ) -> None:
        super().__init__()
        if c_per_pass < 1 or n_passes < 1:
            raise ValueError("c_per_pass and n_passes must be >= 1")
        d_out, d_in = linear.weight.shape
        self.d_in = d_in
        self.d_out = d_out
        self.C = c_per_pass
        self.n_passes = n_passes

        self.register_buffer("W_target", linear.weight.detach().clone())
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        self.register_buffer("bias", bias)

        v_base, u_base = _svd_or_random_factors(linear.weight, c_per_pass, init_svd=init_svd)
        # Parameterize [T, d_in, C] and [T, C, d_out] so each pass can drift independently
        self.V = nn.Parameter(v_base.unsqueeze(0).repeat(n_passes, 1, 1).clone())
        self.U = nn.Parameter(u_base.unsqueeze(0).repeat(n_passes, 1, 1).clone())

        self.mode: Literal["target", "component"] = "target"
        self.schedule: PassMaskSchedule | None = None
        self.calls: int = 0
        self.record_inputs: bool = False
        self.recorded_inputs: list[Tensor] = []

    def reset_pass_state(self) -> None:
        self.calls = 0
        self.recorded_inputs = []

    def weight_deltas(self) -> Tensor:
        """Return ``[T, d_out, d_in]`` weight residuals across all unrolled passes."""
        w_hat = torch.bmm(self.V, self.U).transpose(1, 2)  # [T, d_out, d_in]
        return self.W_target.float().unsqueeze(0) - w_hat

    def forward(self, x: Tensor) -> Tensor:
        tau = self.calls
        if tau >= self.n_passes:
            raise RuntimeError(f"UnrolledComponentLinear exceeded n_passes={self.n_passes}")
        self.calls += 1
        if self.record_inputs:
            self.recorded_inputs.append(x.detach())

        if self.mode == "target":
            return F.linear(
                x,
                self.W_target.to(x.dtype),
                self.bias.to(x.dtype) if self.bias is not None else None,
            )

        if self.schedule is None:
            raise RuntimeError("Component mode requires an active PassMaskSchedule")

        xf = x.float()
        v_t = self.V[tau]  # [d_in, C]
        u_t = self.U[tau]  # [C, d_out]
        mask_t = self.schedule.comp_mask[:, tau]  # [B, P, C]
        comp_out = ((xf @ v_t) * mask_t) @ u_t
        if self.bias is not None:
            comp_out = comp_out + self.bias.float()
        delta_w = self.W_target.float() - (v_t @ u_t).transpose(0, 1)
        delta_raw = F.linear(xf, delta_w)
        if self.schedule.delta_mask is not None:
            comp_out = comp_out + self.schedule.delta_mask[:, tau].unsqueeze(-1) * delta_raw
        else:
            comp_out = comp_out + delta_raw

        if self.schedule.routing_mask is not None:
            r_mask = self.schedule.routing_mask[:, tau].unsqueeze(-1)
            target_out = F.linear(
                xf, self.W_target.float(), self.bias.float() if self.bias is not None else None
            )
            comp_out = torch.where(r_mask, comp_out, target_out)
        return comp_out.to(x.dtype)


@dataclass
class RolloutTrajectory:
    """Outputs of a multi-pass looped execution for BPD training and evaluation.

    - ``readouts``: dict mapping pass index ``tau`` (0-based) -> logits ``[B, P_out, V]``
      for every pass that exposes a behavioral readout.
    - ``boundary_states``: dict mapping pass index ``tau`` -> state tensor ``[B, P_s, D]``
      for passes (or all passes) used in loop-boundary state matching and visible projection.
    - ``visible_states``: optional dict mapping ``tau`` -> ``V_tau`` ``[B, P_s, D_vis]``
      (head-readable projection ``S_tau @ P_head``) for ``visible_only`` CI routing.
    - ``prompt_features``: optional tensor ``[B, P_s, D_prompt]`` encoding input clues ``x``.
    - ``module_inputs``: dict ``module_name`` -> ``[B, T, P, d_in]`` stacked pre-weight inputs.
    """

    readouts: dict[int, Tensor]
    boundary_states: dict[int, Tensor]
    module_inputs: dict[str, Tensor]
    visible_states: dict[int, Tensor] | None = None
    prompt_features: Tensor | None = None


RolloutFn = Callable[[nn.Module, Any, int, int, Any], tuple[dict[int, Tensor], dict[int, Tensor], dict[int, Tensor] | None, Tensor | None]]


class BPDModel(nn.Module):
    """Installs ``TiedComponentLinear`` modules on a weight-tied target model."""

    def __init__(
        self,
        target_model: nn.Module,
        c_per_module: dict[str, int],
        *,
        max_passes: int = 64,
        ablation: AblationConfig | None = None,
        init_svd: bool = True,
    ) -> None:
        super().__init__()
        if not c_per_module:
            raise ValueError("c_per_module must list at least one shared nn.Linear module")

        for param in target_model.parameters():
            param.requires_grad_(False)
        target_model.eval()

        linears: dict[str, nn.Linear] = {}
        seen_weight_ids: set[int] = set()
        for path in c_per_module:
            mod = target_model.get_submodule(path)
            if not isinstance(mod, nn.Linear):
                raise TypeError(f"Module {path!r} is {type(mod).__name__}, expected nn.Linear")
            wid = id(mod.weight)
            if wid in seen_weight_ids:
                raise ValueError(f"Duplicate shared physical weight listed at {path!r}")
            seen_weight_ids.add(wid)
            linears[path] = mod

        self.target_model = target_model
        self.module_order = sorted(c_per_module.keys())
        self.c_per_module = {k: int(c_per_module[k]) for k in self.module_order}
        self.wrappers: dict[str, TiedComponentLinear] = {}

        for path in self.module_order:
            wrapper = TiedComponentLinear(
                linears[path],
                self.c_per_module[path],
                max_passes=max_passes,
                ablation=ablation,
                init_svd=init_svd,
            )
            parent_path, _, attr = path.rpartition(".")
            parent = target_model.get_submodule(parent_path) if parent_path else target_model
            setattr(parent, attr, wrapper)
            self.wrappers[path] = wrapper

        # Register wrappers for proper .to(device) and state_dict() tracking
        self._registered_wrappers = nn.ModuleDict(
            {k.replace(".", "__"): self.wrappers[k] for k in self.module_order}
        )

    def component_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for name in self.module_order:
            w = self.wrappers[name]
            params.extend([w.V, w.U])
        return params

    def total_components(self) -> int:
        return sum(self.c_per_module.values())

    def reset_wrappers(self, *, pass_offset: int = 0) -> None:
        for w in self.wrappers.values():
            w.reset_pass_state(pass_offset=pass_offset)

    def clear_schedules(self) -> None:
        for w in self.wrappers.values():
            w.mode = "target"
            w.schedule = None
            w.record_inputs = False
            w.update_running_means = False

    @contextmanager
    def target_capture(
        self,
        *,
        pass_offset: int = 0,
        update_means: bool = False,
        detach_inputs: bool = True,
    ):
        """Run clean target weights while capturing per-pass pre-weight inputs."""
        self.clear_schedules()
        self.reset_wrappers(pass_offset=pass_offset)
        for w in self.wrappers.values():
            w.record_inputs = True
            w.detach_recorded_inputs = detach_inputs
            w.update_running_means = update_means
        try:
            yield
        finally:
            for w in self.wrappers.values():
                w.record_inputs = False
                w.update_running_means = False

    @contextmanager
    def masked_components(
        self,
        schedules: dict[str, PassMaskSchedule],
        *,
        pass_offset: int = 0,
        record_inputs: bool = False,
    ):
        """Run decomposed component weights under pass-indexed mask schedules."""
        self.clear_schedules()
        self.reset_wrappers(pass_offset=pass_offset)
        for name, w in self.wrappers.items():
            w.mode = "component"
            w.schedule = schedules[name]
            w.record_inputs = record_inputs
            w.detach_recorded_inputs = False
        try:
            yield
        finally:
            self.clear_schedules()

    @contextmanager
    def intervene(
        self,
        module_name: str,
        components: int | list[int] | tuple[int, ...],
        *,
        passes: set[int] | frozenset[int] | list[int] | tuple[int, ...] | None = None,
        mode: Literal["ablate", "scale", "add_alpha", "override_read"] = "ablate",
        scale: float = 0.0,
        read_override: Tensor | None = None,
        pass_offset: int = 0,
    ):
        """Apply a clean-target counterfactual intervention on selected components/passes."""
        if module_name not in self.wrappers:
            raise KeyError(f"Unknown module {module_name!r}")
        comp_tuple = (components,) if isinstance(components, int) else tuple(components)
        pass_set = None if passes is None else frozenset(passes)
        w = self.wrappers[module_name]
        if w.intervention is not None:
            raise RuntimeError(f"Nested intervention on {module_name!r}")
        self.clear_schedules()
        self.reset_wrappers(pass_offset=pass_offset)
        w.intervention = ComponentIntervention(
            components=comp_tuple,
            passes=pass_set,
            mode=mode,
            scale=scale,
            read_override=read_override,
        )
        try:
            yield
        finally:
            w.intervention = None

    def collect_recorded_inputs(self, expected_passes: int | None = None) -> dict[str, Tensor]:
        """Stack recorded per-pass inputs into ``[B, T, P, d_in]`` tensors per module."""
        out: dict[str, Tensor] = {}
        for name in self.module_order:
            w = self.wrappers[name]
            if expected_passes is not None and len(w.recorded_inputs) != expected_passes:
                raise RuntimeError(
                    f"Module {name!r} recorded {len(w.recorded_inputs)} passes, expected {expected_passes}"
                )
            stacked = torch.stack(w.recorded_inputs, dim=1)  # [B, T, P, d_in] or [B, T, d_in]
            if stacked.ndim == 3:
                stacked = stacked.unsqueeze(2)
            out[name] = stacked
        return out
