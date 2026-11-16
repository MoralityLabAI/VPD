"""BPD training objectives: weight faithfulness, per-pass trajectory KL/state loss
with fragility hinge, stochastic subset routing, and pass-invariant + max-over-tau
frequency + temporal-switch minimality.

At ``T = 1`` with ``pool_mode="mean"``, ``coeff_switch=0.0``, and ``hinge_kappa=0.0``,
the objective matches the four-term PyTorch adVersarial Parameter Decomposition (VPD)
port in ``MoralityLabAI/VPD`` ``experiments/nanbeige_looped/vpd_math.py`` at commit
``69509c39e96a912f99b1a83640250ac01797c571`` (the Goodfire VPD paper adds a fifth
deterministic CI-masked reconstruction term).
"""

from __future__ import annotations

import math
from typing import Literal, Mapping

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


def compute_pass_loss_shares(
    per_pass_losses: Mapping[int, float],
    *,
    eps: float = 1e-12,
) -> dict[int, float]:
    """Compute each active pass's share ``loss_tau / sum_t loss_t`` of total trajectory loss."""
    cleaned = {int(tau): max(float(val), 0.0) for tau, val in sorted(per_pass_losses.items())}
    total = sum(cleaned.values())
    if total <= eps:
        return {tau: 0.0 for tau in cleaned}
    return {tau: val / total for tau, val in cleaned.items()}


def trajectory_recon_loss(
    pred_traj: RolloutTrajectory,
    clean_traj: RolloutTrajectory,
    *,
    match_inner_boundary_states: bool = False,
    pass_null_losses: Mapping[int, float] | Tensor | None = None,
    hinge_kappa: float = 0.0,
    pass_tolerances: Mapping[int, float] | Tensor | None = None,
    state_boundary_weight: float = 1.0,
    eps: float = 1e-6,
) -> tuple[Tensor, dict[int, float]]:
    """Match behavioral readouts across passes with optional inner-state matching and fragility hinge.

    - When ``match_inner_boundary_states=False`` (default) and ``clean_traj.readouts`` is
      non-empty, only behavioral readout passes ``tau in clean_traj.readouts`` are matched.
      When ``match_inner_boundary_states=True`` (or ``clean_traj.readouts`` is empty),
      read-free passes in ``clean_traj.boundary_states`` are also matched via
      ``state_boundary_weight * boundary_state_loss``.
    - Fragility hinge: when ``pass_null_losses`` is supplied and ``hinge_kappa > 0``,
      each pass applies ``loss_tau = max(0, raw_loss_tau - hinge_kappa * null_loss_tau)``.
    """
    if not match_inner_boundary_states and clean_traj.readouts:
        active_passes = sorted(clean_traj.readouts.keys())
    else:
        active_passes = sorted(
            set(clean_traj.readouts.keys()) | set(clean_traj.boundary_states.keys())
        )
    if not active_passes:
        raise ValueError("Clean trajectory has neither readouts nor boundary_states")

    null_source = pass_null_losses if pass_null_losses is not None else pass_tolerances
    kappa = max(float(hinge_kappa), 0.0)

    per_pass_losses: list[Tensor] = []
    per_pass_scalars: dict[int, float] = {}

    for tau in active_passes:
        if tau in clean_traj.readouts:
            raw_tau = kl_divergence_logits(pred_traj.readouts[tau], clean_traj.readouts[tau])
        else:
            raw_tau = state_boundary_weight * boundary_state_loss(
                pred_traj.boundary_states[tau], clean_traj.boundary_states[tau], eps=eps
            )

        if null_source is not None and kappa > 0.0:
            if isinstance(null_source, Tensor):
                null_tau = float(null_source[tau].item()) if tau < null_source.numel() else 0.0
            else:
                null_tau = float(null_source.get(tau, 0.0))
            loss_tau = torch.clamp_min(raw_tau - kappa * max(null_tau, 0.0), 0.0)
        else:
            loss_tau = raw_tau

        per_pass_losses.append(loss_tau)
        per_pass_scalars[tau] = float(loss_tau.detach().item())

    stacked = torch.stack(per_pass_losses)
    total = stacked.mean()
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
    world_size: int = 1,
) -> tuple[Tensor, dict[str, float]]:
    """Pass-invariant component minimality with max-over-``tau`` frequency ``f_c`` + switch penalty.

    For each shared module ``m`` and component ``c`` with ``vals = (g_upper + eps)^p`` (``[B, T, P, C]``):
      - Activity level ``l_c``: mean over ``(B, T, P)`` (or max over ``T`` then mean over ``(B, P)``).
      - Frequency ``f_c``: computed over data ``(B, P)`` using the **maximum over ``tau``**,
        ``max_tau = max_tau vals_{b,tau,p,c}``, ``sum_c = sum_{b,p} max_tau``, ``f_c = sum_c / (B * P)``.
      - Module cost: ``sum_c [ l_c + beta * f_c * log2(1 + sum_c * world_size) ]``.

    Properties:
      1. **Exact ``T = 1`` parity with ``vpd_math.importance_minimality_loss``**: at ``T = 1``
         and ``pool_mode="mean"``, ``l_c == f_c == mean_c`` and ``switch_loss == 0``.
      2. **No ``T``-fold tax on stationary mechanisms**: a constant-in-``tau`` schedule has
         identical ``l_c``, ``f_c``, and ``0`` switch cost at any ``T >= 1``.
      3. **Anti-splitting even at ``coeff_switch = 0``**: splitting one all-pass stationary
         mechanism into two half-trajectory copies preserves ``sum_c l_c`` while doubling
         the max-over-``tau`` frequency description term ``sum_c f_c * log2(1 + sum_c)``.
    """
    p = annealed_p(step, total_steps, cfg.p_start, cfg.p_end)

    mod_level_terms: list[Tensor] = []
    switch_terms: list[Tensor] = []
    total_c = 0

    for name in sorted(ci_upper.keys()):
        g = ci_upper[name].float()  # [B, T, P, C]
        bsz, n_passes, seq_len, c_dim = g.shape
        total_c += int(c_dim)
        vals = (g.clamp_min(0.0) + cfg.eps).pow(p)  # [B, T, P, C]

        if cfg.pool_mode == "mean":
            l_c = vals.mean(dim=(0, 1, 2))  # [C]
            max_tau = vals.amax(dim=1)  # [B, P, C]
            sum_c = max_tau.sum(dim=(0, 1))  # [C]
            f_c = sum_c / float(max(bsz * seq_len, 1))  # [C]
        elif cfg.pool_mode == "max":
            max_tau = vals.amax(dim=1)  # [B, P, C]
            l_c = max_tau.mean(dim=(0, 1))  # [C]
            sum_c = max_tau.sum(dim=(0, 1))  # [C]
            f_c = sum_c / float(max(bsz * seq_len, 1))  # [C]
        elif cfg.pool_mode == "sum_legacy":
            sum_c = vals.sum(dim=(0, 1, 2))  # [C]
            l_c = vals.sum(dim=1).mean(dim=(0, 1))  # [C]
            f_c = sum_c / float(max(bsz * n_passes * seq_len, 1))  # [C]
        else:
            raise ValueError(f"Unknown pool_mode: {cfg.pool_mode!r}")

        mod_term = (l_c + cfg.beta * f_c * torch.log2(1.0 + sum_c * float(world_size))).sum()
        mod_level_terms.append(mod_term)

        if n_passes > 1 and cfg.coeff_switch > 0.0:
            diffs = (g[:, 1:] - g[:, :-1]).abs()  # [B, T-1, P, C]
            if cfg.switch_q != 1.0:
                diffs = (diffs + cfg.eps).pow(cfg.switch_q)
            switch_terms.append(diffs.sum(dim=-1).mean())

    level_raw = torch.stack(mod_level_terms).sum()
    if cfg.normalize_by_c and total_c > 0:
        level_raw = level_raw / float(total_c)
    level_weighted = cfg.coeff_level * level_raw

    if switch_terms:
        switch_raw = torch.stack(switch_terms).sum()
        if cfg.normalize_by_c and total_c > 0:
            switch_raw = switch_raw / float(total_c)
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
    route_mode: Literal["module_k_subset", "position_k_subset"] = "module_k_subset",
    share_noise_across_passes: bool = True,
    generator: torch.Generator | None = None,
) -> dict[str, PassMaskSchedule]:
    """Build stochastic mask schedules ``m = ci + (1 - ci) * u``, ``u ~ U(0, 1)``.

    ``route_mode="module_k_subset"`` samples ``k ~ Uniform{1..M}`` active modules per
    ``(batch, pos)`` (matching ``vpd_math.sample_uniform_k_subset_routing``), shared
    across passes when ``share_noise_across_passes=True``.
    """
    names = sorted(ci_lower.keys())
    first = ci_lower[names[0]]
    bsz, n_passes, seq_len, _ = first.shape
    dev = first.device
    dt = first.dtype
    n_mods = len(names)

    routing_by_mod: dict[str, Tensor | None] = {name: None for name in names}
    if route_subset:
        if route_mode == "module_k_subset":
            if n_mods > 1:
                b_dims = (bsz, 1, seq_len) if share_noise_across_passes else (bsz, n_passes, seq_len)
                k = torch.randint(1, n_mods + 1, b_dims, generator=generator, device=dev)
                noise = torch.rand((n_mods, *b_dims), generator=generator, device=dev)
                ranks = noise.argsort(dim=0).argsort(dim=0)
                for idx, name in enumerate(names):
                    rm = ranks[idx] < k
                    routing_by_mod[name] = rm.expand(bsz, n_passes, seq_len)
            else:
                all_true = torch.ones((bsz, n_passes, seq_len), dtype=torch.bool, device=dev)
                routing_by_mod[names[0]] = all_true
        elif route_mode == "position_k_subset":
            if seq_len > 1:
                k_pos = int(
                    torch.randint(1, seq_len + 1, (1,), generator=generator, device=dev).item()
                )
                scores = torch.rand((bsz, seq_len), generator=generator, device=dev)
                topk = scores.topk(k_pos, dim=-1).indices
                pos_mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=dev)
                pos_mask.scatter_(1, topk, True)
                shared_rm = pos_mask.unsqueeze(1).expand(bsz, n_passes, seq_len)
                for name in names:
                    routing_by_mod[name] = shared_rm
        else:
            raise ValueError(f"Unknown route_mode: {route_mode!r}")

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
            routing_mask=routing_by_mod[name],
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
