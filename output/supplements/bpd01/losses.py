"""BPD training objectives: weight faithfulness, per-pass trajectory KL/state loss,
stochastic subset routing, and pass-invariant + temporal-switch minimality.

At ``T = 1`` with ``pool_mode="mean"``, ``coeff_switch=0.0``, and uniform pass
tolerance, every term reduces algebraically and numerically to upstream VPD
(Goodfire 2026).
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from mechinterp.bpd.components import (
    PassMaskSchedule,
    RolloutTrajectory,
    TiedComponentLinear,
    UnrolledComponentLinear,
)
from mechinterp.bpd.config import MinimalityConfig


def faithfulness_loss(wrappers: Mapping[str, TiedComponentLinear]) -> Tensor:
    """Mean squared entrywise error ``||W_m - (V_m U_m)^T||_F^2 / sum_m (d_out_m * d_in_m)``.

    Counted **once** per shared physical matrix, matching upstream VPD at ``T = 1``
    without duplicating weight parameters across passes.
    """
    sq_err = None
    total_elems = 0
    for w in wrappers.values():
        delta = w.weight_delta()
        term = delta.pow(2).sum()
        sq_err = term if sq_err is None else sq_err + term
        total_elems += delta.numel()
    assert sq_err is not None and total_elems > 0
    return sq_err / float(total_elems)


def unrolled_faithfulness_loss(wrappers: Mapping[str, UnrolledComponentLinear]) -> Tensor:
    """Mean squared entrywise error across all ``T`` unrolled copies for Gate B1."""
    sq_err = None
    total_elems = 0
    for w in wrappers.values():
        deltas = w.weight_deltas()  # [T, d_out, d_in]
        term = deltas.pow(2).sum()
        sq_err = term if sq_err is None else sq_err + term
        total_elems += deltas.numel()
    assert sq_err is not None and total_elems > 0
    return sq_err / float(total_elems)


def kl_divergence_logits(pred_logits: Tensor, target_logits: Tensor) -> Tensor:
    """Exact token-averaged ``KL(softmax(target) || softmax(pred))`` in float32."""
    pred_lp = F.log_softmax(pred_logits.float(), dim=-1)
    targ_lp = F.log_softmax(target_logits.float(), dim=-1)
    targ_p = targ_lp.exp()
    kl_per_pos = (targ_p * (targ_lp - pred_lp)).sum(dim=-1)
    return kl_per_pos.mean()


def boundary_state_loss(pred_state: Tensor, target_state: Tensor, eps: float = 1e-6) -> Tensor:
    """Scale-normalized MSE + directional discrepancy at read-free loop boundaries.

    Normalizes by the target state's per-token RMS squared so read-free inner
    passes (such as ``z_L`` steps in ``TRM``) have dimensionless loss scale ``O(1)``
    commensurate with readout KL.
    """
    pf = pred_state.float()
    tf = target_state.float()
    denom = tf.pow(2).mean(dim=-1, keepdim=True).clamp_min(eps)
    rel_mse = ((pf - tf).pow(2) / denom).mean()
    cos = F.cosine_similarity(pf, tf, dim=-1, eps=eps)
    return 0.5 * rel_mse + 0.5 * (1.0 - cos).mean()


def trajectory_recon_loss(
    pred_traj: RolloutTrajectory,
    clean_traj: RolloutTrajectory,
    *,
    pass_tolerances: Mapping[int, float] | Tensor | None = None,
    state_boundary_weight: float = 1.0,
    eps: float = 1e-6,
) -> tuple[Tensor, dict[int, float]]:
    """Match the behavioral readout at every pass, or loop-boundary state where none exists.

    When ``pass_tolerances`` is supplied (from the norm/RMS-matched Haar rotation
    null), pass ``tau`` receives weight proportional to ``1 / (tol_tau + eps)``,
    normalized to have mean ``1.0`` across active passes so fragile and contractive
    passes contribute on an equalized sensitivity footing.
    """
    all_passes = sorted(set(clean_traj.readouts.keys()) | set(clean_traj.boundary_states.keys()))
    if not all_passes:
        raise ValueError("Clean trajectory has neither readouts nor boundary_states")

    per_pass_losses: list[Tensor] = []
    raw_weights: list[float] = []
    per_pass_scalars: dict[int, float] = {}

    for tau in all_passes:
        if tau in clean_traj.readouts:
            loss_tau = kl_divergence_logits(pred_traj.readouts[tau], clean_traj.readouts[tau])
        else:
            loss_tau = state_boundary_weight * boundary_state_loss(
                pred_traj.boundary_states[tau], clean_traj.boundary_states[tau], eps=eps
            )
        per_pass_losses.append(loss_tau)
        per_pass_scalars[tau] = float(loss_tau.detach().item())

        if pass_tolerances is None:
            raw_weights.append(1.0)
        elif isinstance(pass_tolerances, Tensor):
            tol = float(pass_tolerances[tau].item()) if tau < pass_tolerances.numel() else 1.0
            raw_weights.append(1.0 / (max(tol, 0.0) + eps))
        else:
            tol = float(pass_tolerances.get(tau, 1.0))
            raw_weights.append(1.0 / (max(tol, 0.0) + eps))

    mean_w = sum(raw_weights) / float(len(raw_weights))
    norm_weights = [w / max(mean_w, eps) for w in raw_weights]

    stacked = torch.stack(per_pass_losses)
    w_tensor = torch.tensor(norm_weights, device=stacked.device, dtype=stacked.dtype)
    total = (stacked * w_tensor).mean()
    return total, per_pass_scalars


def annealed_p(step: int, total_steps: int, p_start: float, p_end: float) -> float:
    """Linearly anneal ``L_p`` sparsity exponent from ``p_start`` to ``p_end``."""
    if total_steps <= 1:
        return float(p_end)
    frac = min(max(float(step) / float(total_steps), 0.0), 1.0)
    return float(p_start + (p_end - p_start) * frac)


def bpd_minimality_loss(
    ci_upper: Mapping[str, Tensor],
    *,
    step: int,
    total_steps: int,
    cfg: MinimalityConfig,
) -> tuple[Tensor, dict[str, float]]:
    """Pass-invariant component minimality plus temporal switch penalty along ``tau``.

    Properties:
      1. **Exact ``T = 1`` VPD reduction**: When ``T = 1`` and ``pool_mode="mean"``,
         ``pooled_lp`` equals ``(g + eps)^p`` and ``switch_loss == 0`` identically,
         matching Goodfire VPD's ``mean_c + beta * mean_c * log2(1 + sum_c)`` loss.
      2. **No ``T``-fold tax on stationary mechanisms**: Under ``pool_mode="mean"``
         (or ``"max"``), a stationary mechanism ``g_{b,p,tau,c} = 1`` for all ``tau``
         incurs the **same** level cost at ``T = 16`` as at ``T = 1`` and ``0`` switch cost.
      3. **Switch penalty discourages artificial time-splitting**: Splitting one
         all-pass mechanism into ``T`` single-pass components adds ``O(1)`` total
         variation switches along ``tau`` plus extra rank slots in ``log2(1 + sum_c)``.
    """
    p = annealed_p(step, total_steps, cfg.p_start, cfg.p_end)

    pooled_list: list[Tensor] = []
    switch_terms: list[Tensor] = []

    for name in sorted(ci_upper.keys()):
        g = ci_upper[name].float()  # [B, T, P, C]
        n_passes = g.shape[1]
        lp = (g.clamp_min(0.0) + cfg.eps).pow(p)  # [B, T, P, C]

        if cfg.pool_mode == "mean":
            pooled = lp.mean(dim=1)  # [B, P, C]
        elif cfg.pool_mode == "max":
            pooled = lp.amax(dim=1)  # [B, P, C]
        elif cfg.pool_mode == "sum_legacy":
            pooled = lp.sum(dim=1)  # [B, P, C] - penalizes T times (for ablation only)
        else:
            raise ValueError(f"Unknown pool_mode: {cfg.pool_mode!r}")

        pooled_list.append(pooled)

        if n_passes > 1 and cfg.coeff_switch > 0.0:
            diffs = (g[:, 1:] - g[:, :-1]).abs()  # [B, T-1, P, C]
            if cfg.switch_q != 1.0:
                diffs = (diffs + cfg.eps).pow(cfg.switch_q)
            # Sum across C, mean across batch, transitions (T-1), and positions P
            switch_terms.append(diffs.sum(dim=-1).mean())

    cat_lp = torch.cat(pooled_list, dim=-1)  # [B, P, C_total]
    sum_over_c = cat_lp.sum(dim=-1)  # [B, P]
    mean_over_c = cat_lp.mean(dim=-1)  # [B, P]

    # Upstream Goodfire VPD functional: l1 + beta * l2
    l1 = mean_over_c
    l2 = mean_over_c * torch.log2(1.0 + sum_over_c)
    level_raw = (l1 + cfg.beta * l2).mean()
    level_weighted = cfg.coeff_level * level_raw

    if switch_terms:
        switch_raw = torch.stack(switch_terms).sum()
        switch_weighted = cfg.coeff_switch * switch_raw
    else:
        switch_raw = torch.zeros_like(level_raw)
        switch_weighted = torch.zeros_like(level_raw)

    total = level_weighted + switch_weighted
    return total, {
        "p": p,
        "level_raw": float(level_raw.detach().item()),
        "level_weighted": float(level_weighted.detach().item()),
        "switch_raw": float(switch_raw.detach().item()),
        "switch_weighted": float(switch_weighted.detach().item()),
        "total": float(total.detach().item()),
    }


def sample_stochastic_schedules(
    ci_lower: Mapping[str, Tensor],
    *,
    route_subset: bool = True,
    share_noise_across_passes: bool = True,
    generator: torch.Generator | None = None,
) -> dict[str, PassMaskSchedule]:
    """Build stochastic mask schedules ``m = ci + (1 - ci) * u``, ``u ~ U(0, 1)``.

    By default ``share_noise_across_passes=True`` samples ``u`` once per ``(b, p, c)``
    shared across ``tau`` (consistent with a time-shared parameter perturbation), while
    also allowing pass-independent noise when ``share_noise_across_passes=False``.
    """
    names = sorted(ci_lower.keys())
    first = ci_lower[names[0]]
    bsz, n_passes, seq_len, _ = first.shape
    dev = first.device
    dt = first.dtype

    routing_mask: Tensor | None = None
    if route_subset and seq_len > 1:
        k = int(torch.randint(1, seq_len + 1, (1,), generator=generator, device=dev).item())
        scores = torch.rand((bsz, seq_len), generator=generator, device=dev)
        topk = scores.topk(k, dim=-1).indices
        pos_mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=dev)
        pos_mask.scatter_(1, topk, True)
        routing_mask = pos_mask.unsqueeze(1).expand(bsz, n_passes, seq_len)

    schedules: dict[str, PassMaskSchedule] = {}
    for name in names:
        ci = ci_lower[name]  # [B, T, P, C]
        c = ci.shape[-1]
        if share_noise_across_passes:
            u_comp = torch.rand((bsz, 1, seq_len, c), generator=generator, device=dev, dtype=dt)
            u_delta = torch.rand((bsz, 1, seq_len), generator=generator, device=dev, dtype=dt)
            u_comp = u_comp.expand(bsz, n_passes, seq_len, c)
            u_delta = u_delta.expand(bsz, n_passes, seq_len)
        else:
            u_comp = torch.rand((bsz, n_passes, seq_len, c), generator=generator, device=dev, dtype=dt)
            u_delta = torch.rand((bsz, n_passes, seq_len), generator=generator, device=dev, dtype=dt)

        comp_mask = ci + (1.0 - ci) * u_comp
        schedules[name] = PassMaskSchedule(
            comp_mask=comp_mask,
            delta_mask=u_delta,
            routing_mask=routing_mask,
        )
    return schedules


def deterministic_ci_schedules(
    ci_lower: Mapping[str, Tensor],
    *,
    delta_value: float = 0.0,
    threshold: float | None = None,
) -> dict[str, PassMaskSchedule]:
    """Construct deterministic evaluation schedules directly from ``ci_lower``."""
    schedules: dict[str, PassMaskSchedule] = {}
    for name, ci in ci_lower.items():
        bsz, n_passes, seq_len, _ = ci.shape
        comp_mask = (ci > threshold).to(ci.dtype) if threshold is not None else ci
        delta_mask = torch.full(
            (bsz, n_passes, seq_len),
            fill_value=float(delta_value),
            device=ci.device,
            dtype=ci.dtype,
        )
        schedules[name] = PassMaskSchedule(comp_mask=comp_mask, delta_mask=delta_mask)
    return schedules
