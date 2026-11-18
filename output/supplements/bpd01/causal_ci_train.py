"""Closed-loop training of a CI network for strictly causal top-k gating (B4-CAUSAL-CI-02).

B4-CAUSAL-CI-01 found that a CI trained on clean target trajectories fails when it gates
the rollout causally. Its logits are nearly tied at the k-th rank, the masks reshuffle
after one masked invocation, and the error compounds. This module trains the CI on the
policy it is scored with:

- gates come from ``preflight_target`` observations of the CI's own masked rollout,
  exactly as ``causal_ci.causal_ci_rollout`` computes them;
- the forward mask is the hard top-k of the CI logits, with a straight-through gradient
  through ``sigmoid(logit)``. The forward value equals the scored mask bit for bit;
- the loss is computed on the closed-loop readouts and H states, with gradients through
  the recurrence. Observation features are detached, and each masked invocation is
  activation-checkpointed.

Scheduled sampling: with probability ``teacher_prob``, a board's observation at
invocation tau is the clean target input instead of the preflight input. Training anneals
``teacher_prob`` from 1 (clean-informed gates) to 0 (fully causal).
"""
from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.utils.checkpoint as torch_ckpt

from mechinterp.bpd.causal_ci import causal_ci_rollout
from mechinterp.bpd.ci import TiedCINetwork
from mechinterp.bpd.components import BPDModel, PassMaskSchedule


def straight_through_topk(logits: Tensor, k: int) -> Tensor:
    """Hard top-k mask in the forward pass; d mask / d logit = sigmoid'(logit) in the backward."""
    idx = logits.detach().float().topk(k, dim=-1).indices
    hard = torch.zeros_like(logits, dtype=torch.float32).scatter_(-1, idx, 1.0)
    soft = torch.sigmoid(logits.float())
    return hard + (soft - soft.detach())


@torch.no_grad()
def clean_reference(bpd: BPDModel, adapter: Any, inputs: Tensor, n_passes: int
                    ) -> tuple[dict[int, Tensor], dict[int, Tensor], dict[str, Tensor]]:
    """Clean readouts, clean H states at every H invocation, and clean module inputs [B, T, P, d]."""
    model = adapter.model
    with bpd.target_capture(pass_offset=0, update_means=False, detach_inputs=True):
        readouts, states, _, _ = adapter.rollout(inputs, n_passes, 0, None)
    clean_in = bpd.collect_recorded_inputs(expected_passes=n_passes)
    for w in bpd.wrappers.values():
        w.recorded_inputs.clear()
    cyc = adapter.calls_per_h_cycle
    h_states = {t: s for t, s in states.items() if (t % cyc) >= model.l_cycles}
    return readouts, h_states, clean_in


def causal_topk_rollout_st(
    bpd: BPDModel,
    adapter: Any,
    ci_net: TiedCINetwork,
    inputs: Tensor,
    n_passes: int,
    k: int,
    *,
    clean_inputs: Mapping[str, Tensor] | None = None,
    teacher_prob: float = 0.0,
    generator: torch.Generator | None = None,
    use_checkpoint: bool = True,
) -> dict[str, Any]:
    """Differentiable closed-loop rollout under causal preflight top-k CI gates.

    With ``teacher_prob = 0`` the forward pass is identical to
    ``causal_ci_rollout(observation="preflight_target", selection="topk", k=k)``.
    Returns readouts ``{tau: [B, 81, 9]}`` and H states ``{tau: [B, P, D]}``, both carrying
    gradients to the CI parameters, plus the per-invocation teacher fraction used.
    """
    model = adapter.model
    dev = model.embed_tokens.weight.device
    inp = inputs.to(dev)
    bsz = int(inp.shape[0])
    with torch.no_grad():
        x_emb = model.encode_inputs(inp)
    z_h, z_l = model.init_states(bsz)
    cyc = adapter.calls_per_h_cycle
    order = list(bpd.module_order)
    readouts: dict[int, Tensor] = {}
    h_states: dict[int, Tensor] = {}
    teacher_frac: list[float] = []

    for tau in range(n_passes):
        is_l = (tau % cyc) < model.l_cycles
        z_main, z_inj = (z_l, z_h + x_emb) if is_l else (z_h, z_l)
        with torch.no_grad():
            with bpd.target_capture(pass_offset=tau, update_means=False, detach_inputs=True):
                model.run_l_level(z_main.detach(), z_inj.detach())
            obs = bpd.collect_recorded_inputs(expected_passes=1)
            for w in bpd.wrappers.values():
                w.recorded_inputs.clear()
            if clean_inputs is not None and teacher_prob > 0.0:
                use_clean = torch.rand(bsz, generator=generator, device="cpu") < teacher_prob
                teacher_frac.append(float(use_clean.float().mean()))
                sel = use_clean.to(dev).view(bsz, 1, 1, 1)
                obs = {m: torch.where(sel, clean_inputs[m][:, tau:tau + 1].to(dev, obs[m].dtype), obs[m])
                       for m in order}
            else:
                teacher_frac.append(0.0)
        ci_out = ci_net({m: obs[m].detach() for m in order}, pass_offset=tau)
        masks = [straight_through_topk(ci_out.pre_sigmoid[m], k) for m in order]
        zero_delta = torch.zeros(bsz, 1, masks[0].shape[2], device=dev)

        def _step(zm: Tensor, zi: Tensor, *mk: Tensor, _tau: int = tau, _zd: Tensor = zero_delta) -> Tensor:
            sched = {m: PassMaskSchedule(comp_mask=v, delta_mask=_zd) for m, v in zip(order, mk)}
            with bpd.masked_components(sched, pass_offset=_tau):
                return model.run_l_level(zm, zi)

        if use_checkpoint and torch.is_grad_enabled():
            new = torch_ckpt.checkpoint(_step, z_main, z_inj, *masks, use_reentrant=False)
        else:
            new = _step(z_main, z_inj, *masks)
        if is_l:
            z_l = new
        else:
            z_h = new
            h_states[tau] = new
            if adapter.is_readout_pass(tau):
                readouts[tau] = model.digit_logits(z_h)
    return {"readouts": readouts, "h_states": h_states, "teacher_fraction": teacher_frac}


def closed_loop_loss(
    out: Mapping[str, Any],
    clean_readouts: Mapping[int, Tensor],
    clean_h: Mapping[int, Tensor],
    *,
    state_weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Mean token KL(clean || arm) over the readouts plus relative H-state MSE over H invocations."""
    kls = []
    parts: dict[str, float] = {}
    for t, logits in out["readouts"].items():
        ref = F.log_softmax(clean_readouts[t].float(), dim=-1)
        arm = F.log_softmax(logits.float(), dim=-1)
        kl = (ref.exp() * (ref - arm)).sum(-1).mean()
        kls.append(kl)
        parts[f"kl_pass_{t}"] = float(kl.detach())
    rel = []
    for t, h in out["h_states"].items():
        ref = clean_h[t].float()
        rel.append(((h.float() - ref) ** 2).sum(-1).mean() / (ref ** 2).sum(-1).mean().clamp_min(1e-12))
    kl_mean = torch.stack(kls).mean()
    state = torch.stack(rel).mean()
    parts["kl_mean"] = float(kl_mean.detach())
    parts["state_rel_mse"] = float(state.detach())
    return kl_mean + state_weight * state, parts


@torch.no_grad()
def validate_causal_topk(bpd: BPDModel, adapter: Any, ci_net: TiedCINetwork, inputs: Tensor,
                         n_passes: int, k: int, chunk: int = 16) -> dict[str, float]:
    """Exact scored-policy validation: mean token KL(clean || causal top-k) per readout."""
    sums: dict[int, float] = {}
    n = 0
    for i in range(0, inputs.shape[0], chunk):
        x = inputs[i:i + chunk]
        clean, _, _, _ = adapter.rollout(x, n_passes, 0, None)
        res = causal_ci_rollout(bpd, adapter, ci_net, x, n_passes, observation="preflight_target",
                                selection="topk", k=k)
        for t, logits in res["readouts"].items():
            ref = F.log_softmax(clean[t].double(), dim=-1)
            arm = F.log_softmax(logits.double(), dim=-1)
            sums[t] = sums.get(t, 0.0) + float((ref.exp() * (ref - arm)).sum(-1).mean(-1).sum())
        n += int(x.shape[0])
    out = {f"kl_pass_{t}": v / n for t, v in sums.items()}
    out["kl_mean"] = sum(out.values()) / len(sums)
    return out


def teacher_schedule(step: int, n_steps: int, anneal_fraction: float) -> float:
    """Linear 1 -> 0 over the first ``anneal_fraction`` of training, then 0."""
    span = max(1, int(round(anneal_fraction * n_steps)))
    return max(0.0, 1.0 - step / span)


def train_closed_loop_ci(
    bpd: BPDModel,
    adapter: Any,
    ci_net: TiedCINetwork,
    get_batch: Callable[[int], Tensor],
    val_inputs: Tensor,
    *,
    n_passes: int,
    k: int,
    n_steps: int,
    lr: float,
    weight_decay: float = 0.0,
    anneal_fraction: float = 0.5,
    state_weight: float = 1.0,
    val_every: int = 48,
    grad_clip: float = 1.0,
    seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> tuple[TiedCINetwork, dict[str, Any]]:
    """AdamW with cosine decay to 0.1x; keeps the state with the lowest validation kl_mean."""
    for p in bpd.parameters():
        p.requires_grad_(False)
    ci_net.train()
    for p in ci_net.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(ci_net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(s, n_steps) / max(n_steps, 1))))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    history: list[dict[str, float]] = []
    vals: list[dict[str, float]] = []

    def _validate(step: int) -> dict[str, float]:
        ci_net.eval()
        v = validate_causal_topk(bpd, adapter, ci_net, val_inputs, n_passes, k)
        ci_net.train()
        v["step"] = step
        vals.append(v)
        if log:
            log(f"val step {step}: " + ", ".join(f"{a} {b:.4f}" for a, b in v.items() if a != "step"))
        return v

    best = _validate(0)
    best_state = copy.deepcopy(ci_net.state_dict())
    for step in range(n_steps):
        t_step = time.time()
        x = get_batch(step)
        p_teacher = teacher_schedule(step, n_steps, anneal_fraction)
        clean_ro, clean_h, clean_in = clean_reference(bpd, adapter, x, n_passes)
        out = causal_topk_rollout_st(bpd, adapter, ci_net, x, n_passes, k,
                                     clean_inputs=clean_in, teacher_prob=p_teacher, generator=gen)
        loss, parts = closed_loop_loss(out, clean_ro, clean_h, state_weight=state_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(ci_net.parameters(), grad_clip))
        opt.step()
        sched.step()
        rec = {"step": step, "loss": float(loss.detach()), "teacher_prob": p_teacher, "grad_norm": gnorm,
               "lr": float(opt.param_groups[0]["lr"]), "seconds": time.time() - t_step, **parts}
        history.append(rec)
        if log and (step % 10 == 0 or step == n_steps - 1):
            log(f"step {step}: loss {rec['loss']:.4f} kl {parts['kl_mean']:.4f} state {parts['state_rel_mse']:.4f}"
                f" teacher {p_teacher:.3f} gnorm {gnorm:.3f} {rec['seconds']:.1f}s")
        del out, loss, clean_ro, clean_h, clean_in
        if (step + 1) % val_every == 0 or step == n_steps - 1:
            v = _validate(step + 1)
            if v["kl_mean"] < best["kl_mean"]:
                best = v
                best_state = copy.deepcopy(ci_net.state_dict())
    ci_net.load_state_dict(best_state)
    ci_net.eval()
    return ci_net, {"history": history, "validation": vals, "best_validation": best}
