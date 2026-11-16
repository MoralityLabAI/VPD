"""Norm/RMS-matched Haar rotation nulls and per-pass sensitivity normalization.

Cross-pass sensitivity in weight-tied models can vary by more than three orders
of magnitude (e.g., ``F_1 / F_6 = 1,808.4`` on detached-ACT ``TRM`` in
LOOPSCOPE-02, and isotropic settling fragility in ``HUG-CAL-02`` on
``Huginn-0125``). Treating raw output shift as causal necessity systematically
inflates fragile passes and flags generic high-energy settling directions.

This module implements:
  1. ``sample_rms_matched_read_rotations``: random unit directions ``r_k`` in
     ``R^{d_in}`` rescaled so ``RMS(X_tau @ r_k) == RMS(X_tau @ v_c)`` on pass ``tau``.
  2. ``calibrate_pass_null_losses``: empirical per-pass rotation-null loss ``KL_null_tau``
     under time-shared RMS-matched random rank-1 writes, driving the fragility hinge
     ``loss_tau = max(0, KL_mask_tau - kappa * KL_null_tau)``.
  3. ``calibrate_pass_tolerances``: single-pass Jacobian sensitivity diagnostic.
  4. ``evaluate_component_null_report``: computes and reports **both**
     ``raw_importance`` and ``net_importance = raw_importance - rotation_null_mean``
     (plus ``z_rot``) per ``(module, pass, component)`` so fragile-pass nulls
     can never silently hide real mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from mechinterp.bpd.components import BPDModel, RolloutTrajectory
from mechinterp.bpd.losses import boundary_state_loss, kl_divergence_logits


def sample_rms_matched_read_rotations(
    x_pass: Tensor,
    v_vec: Tensor,
    n_rotations: int,
    *,
    orthogonal_to_v: bool = True,
    eps: float = 1e-8,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample ``n_rotations`` read vectors ``[d_in, n_rotations]`` matched in pass RMS to ``v_vec``.

    Given clean inputs ``x_pass`` (``[..., d_in]``) at pass ``tau`` and component
    read vector ``v_vec`` (``[d_in]``), draws isotropic Gaussian directions
    (optionally projected orthogonal to ``v_vec`` when ``d_in > 1``) and rescales
    each direction ``r_k`` so that:

        ``E[(x_pass @ r_k)^2] == E[(x_pass @ v_vec)^2]``

    This matches both weight-space isotropy (Haar on ``S^{d_in - 1}``) and exact
    empirical activation energy on pass ``tau`` (the ``HUG-CAL-02`` control).
    """
    if n_rotations < 1:
        raise ValueError("n_rotations must be >= 1")
    xf = x_pass.detach().float().reshape(-1, x_pass.shape[-1])  # [N, d_in]
    vf = v_vec.detach().float().reshape(-1)  # [d_in]
    d_in = vf.numel()
    dev = xf.device

    raw = torch.randn((d_in, n_rotations), generator=generator, device=dev, dtype=torch.float32)
    if orthogonal_to_v and d_in > 2 and float(vf.norm().item()) > eps:
        v_unit = vf / vf.norm().clamp_min(eps)
        proj = (v_unit.unsqueeze(0) @ raw)  # [1, K]
        raw = raw - v_unit.unsqueeze(1) * proj

    raw = raw / raw.norm(dim=0, keepdim=True).clamp_min(eps)

    target_rms = (xf @ vf).pow(2).mean().add(eps).sqrt()
    rot_rms = (xf @ raw).pow(2).mean(dim=0, keepdim=True).add(eps).sqrt()  # [1, K]
    return raw * (target_rms / rot_rms)


def sample_per_pass_rms_matched_rotations(
    x_all_passes: Tensor,
    v_vec: Tensor,
    n_rotations: int,
    *,
    orthogonal_to_v: bool = True,
    eps: float = 1e-8,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample ``n_rotations`` time-shared unit read directions rescaled to exact per-pass RMS.

    Returns tensor of shape ``[T, d_in, n_rotations]`` where direction ``k`` shares the
    same unit direction ``r_k`` across passes ``tau = 0..T-1`` while matching
    ``RMS(X_tau @ r_{tau, k}) == RMS(X_tau @ v_vec)`` at every pass ``tau``
    (the exact ``HUG-CAL-02`` multi-iteration rotation control).
    """
    if n_rotations < 1:
        raise ValueError("n_rotations must be >= 1")
    vf = v_vec.detach().float().reshape(-1)
    d_in = vf.numel()
    n_passes = x_all_passes.shape[1]
    dev = x_all_passes.device

    raw = torch.randn((d_in, n_rotations), generator=generator, device=dev, dtype=torch.float32)
    if orthogonal_to_v and d_in > 2 and float(vf.norm().item()) > eps:
        v_unit = vf / vf.norm().clamp_min(eps)
        raw = raw - v_unit.unsqueeze(1) * (v_unit.unsqueeze(0) @ raw)
    r_unit = raw / raw.norm(dim=0, keepdim=True).clamp_min(eps)  # [d_in, K]

    per_pass_rots: list[Tensor] = []
    for tau in range(n_passes):
        xf = x_all_passes[:, tau].detach().float().reshape(-1, d_in)
        t_rms = (xf @ vf).pow(2).mean().add(eps).sqrt()
        r_rms = (xf @ r_unit).pow(2).mean(dim=0, keepdim=True).add(eps).sqrt()  # [1, K]
        per_pass_rots.append(r_unit * (t_rms / r_rms))
    return torch.stack(per_pass_rots, dim=0)  # [T, d_in, K]


def _pass_effect_against_clean(
    pert_traj: RolloutTrajectory,
    clean_traj: RolloutTrajectory,
    *,
    eval_pass: int | None = None,
    state_boundary_weight: float = 1.0,
) -> float:
    """Measure output KL / boundary shift at ``eval_pass`` (or mean over trajectory passes)."""
    if eval_pass is not None:
        if eval_pass in clean_traj.readouts:
            return float(
                kl_divergence_logits(
                    pert_traj.readouts[eval_pass], clean_traj.readouts[eval_pass]
                ).item()
            )
        return float(
            (
                state_boundary_weight
                * boundary_state_loss(
                    pert_traj.boundary_states[eval_pass], clean_traj.boundary_states[eval_pass]
                )
            ).item()
        )

    all_passes = sorted(set(clean_traj.readouts.keys()) | set(clean_traj.boundary_states.keys()))
    vals: list[float] = []
    for tau in all_passes:
        if tau in clean_traj.readouts:
            vals.append(
                float(kl_divergence_logits(pert_traj.readouts[tau], clean_traj.readouts[tau]).item())
            )
        else:
            vals.append(
                float(
                    (
                        state_boundary_weight
                        * boundary_state_loss(
                            pert_traj.boundary_states[tau], clean_traj.boundary_states[tau]
                        )
                    ).item()
                )
            )
    return sum(vals) / float(max(len(vals), 1))


@torch.no_grad()
def calibrate_pass_tolerances(
    bpd_model: BPDModel,
    clean_traj: RolloutTrajectory,
    run_rollout_fn: Callable[[], RolloutTrajectory],
    *,
    n_rotations: int = 4,
    perturbation_scale: float = 1.0,
    eval_at_final_readout: bool = True,
    state_boundary_weight: float = 1.0,
    eps: float = 1e-6,
    seed: int = 0,
) -> dict[int, float]:
    """Estimate per-pass fragility tolerance ``tol_tau`` via RMS-matched random rank-1 writes.

    For each pass ``tau``, injects a random RMS-matched rank-1 perturbation into the
    shared module(s) exclusively on pass ``tau`` and measures the induced shift on
    the final readout (or trajectory readout). Fragile passes with ``1000x`` higher
    Jacobian transport receive proportionally larger ``tol_tau``.
    """
    first_name = bpd_model.module_order[0]
    n_passes = clean_traj.module_inputs[first_name].shape[1]
    dev = clean_traj.module_inputs[first_name].device
    gen = torch.Generator(device=dev).manual_seed(seed)

    final_pass: int | None = None
    if eval_at_final_readout and clean_traj.readouts:
        final_pass = max(clean_traj.readouts.keys())

    tolerances: dict[int, float] = {}
    for tau in range(n_passes):
        effects: list[float] = []
        for name in bpd_model.module_order:
            w = bpd_model.wrappers[name]
            x_tau = clean_traj.module_inputs[name][:, tau]  # [B, P, d_in]
            # Reference RMS from top component 0
            rots = sample_rms_matched_read_rotations(
                x_tau, w.V[:, 0], n_rotations, generator=gen, eps=eps
            )
            for k in range(n_rotations):
                r_col = rots[:, k : k + 1]  # [d_in, 1]
                with bpd_model.intervene(
                    name,
                    0,
                    passes={tau},
                    mode="override_read",
                    scale=perturbation_scale,
                    read_override=r_col,
                ):
                    pert_traj = run_rollout_fn()
                eff = _pass_effect_against_clean(
                    pert_traj,
                    clean_traj,
                    eval_pass=final_pass,
                    state_boundary_weight=state_boundary_weight,
                )
                effects.append(max(eff, 0.0))
        tolerances[tau] = float(sum(effects) / max(len(effects), 1)) + eps
    return tolerances


@torch.no_grad()
def calibrate_pass_null_losses(
    bpd_model: BPDModel,
    clean_traj: RolloutTrajectory,
    run_rollout_fn: Callable[[], RolloutTrajectory],
    *,
    n_rotations: int = 4,
    perturbation_scale: float = 0.10,
    match_inner_boundary_states: bool = False,
    state_boundary_weight: float = 1.0,
    eps: float = 1e-8,
    seed: int = 0,
) -> dict[int, float]:
    """Estimate per-pass null loss ``KL_null_tau`` for the fragility hinge.

    Samples time-shared unit read directions rescaled to exact per-pass RMS
    (``sample_per_pass_rms_matched_rotations``) on each shared module at amplitude
    ``perturbation_scale`` and records the induced readout KL (or boundary-state
    discrepancy on read-free inner passes when ``match_inner_boundary_states=True``)
    at every active pass ``tau``.
    """
    if not match_inner_boundary_states and clean_traj.readouts:
        active_passes = sorted(clean_traj.readouts.keys())
    else:
        active_passes = sorted(
            set(clean_traj.readouts.keys()) | set(clean_traj.boundary_states.keys())
        )
    if not active_passes:
        return {}

    first_name = bpd_model.module_order[0]
    dev = clean_traj.module_inputs[first_name].device
    gen = torch.Generator(device=dev).manual_seed(seed)

    accum: dict[int, list[float]] = {tau: [] for tau in active_passes}
    for name in bpd_model.module_order:
        w = bpd_model.wrappers[name]
        x_all = clean_traj.module_inputs[name]  # [B, T, P, d_in]
        rots = sample_per_pass_rms_matched_rotations(
            x_all, w.V[:, 0], n_rotations, generator=gen, eps=eps
        )
        for k in range(n_rotations):
            r_col = rots[:, :, k : k + 1]  # [T, d_in, 1]
            with bpd_model.intervene(
                name,
                0,
                passes=None,
                mode="override_read",
                scale=perturbation_scale,
                read_override=r_col,
            ):
                pert_traj = run_rollout_fn()
            for tau in active_passes:
                eff_tau = _pass_effect_against_clean(
                    pert_traj,
                    clean_traj,
                    eval_pass=tau,
                    state_boundary_weight=state_boundary_weight,
                )
                accum[tau].append(max(eff_tau, 0.0))

    return {
        tau: float(sum(vals) / max(len(vals), 1))
        for tau, vals in accum.items()
    }


@dataclass
class ComponentNullEntry:
    """Side-by-side raw, net-of-null, and exact rank p-value for one ``(module, pass, c)``."""

    module: str
    pass_idx: int  # -1 indicates all-pass pooled ablation
    component: int
    raw_importance: float
    rotation_null_mean: float
    rotation_null_std: float
    net_importance: float
    z_rot: float
    n_rotations: int = 0
    exceed_count: int = 0
    p_rank: float = 1.0
    p_rank_two_sided: float = 1.0


@dataclass
class ComponentNullReport:
    """Complete raw + rotation-null importance table (never suppresses raw values)."""

    entries: list[ComponentNullEntry]

    def for_module(self, module: str) -> list[ComponentNullEntry]:
        return [e for e in self.entries if e.module == module]

    def to_dicts(self) -> list[dict[str, float | int | str]]:
        return [
            {
                "module": e.module,
                "pass_idx": e.pass_idx,
                "component": e.component,
                "raw_importance": e.raw_importance,
                "rotation_null_mean": e.rotation_null_mean,
                "rotation_null_std": e.rotation_null_std,
                "net_importance": e.net_importance,
                "z_rot": e.z_rot,
                "n_rotations": e.n_rotations,
                "exceed_count": e.exceed_count,
                "p_rank": e.p_rank,
                "p_rank_two_sided": e.p_rank_two_sided,
            }
            for e in self.entries
        ]


def compute_rank_p_values(raw_val: float, null_vals: list[float] | Tensor) -> tuple[int, float, float]:
    """Compute exact Monte Carlo rank p-values ``(exceed_count, p_upper, p_two_sided)``."""
    nt = null_vals.double() if isinstance(null_vals, Tensor) else torch.tensor(null_vals, dtype=torch.float64)
    k = int(nt.numel())
    if k == 0:
        return 0, 1.0, 1.0
    exceed = int((nt >= float(raw_val)).sum().item())
    below = int((nt <= float(raw_val)).sum().item())
    p_up = float(1 + exceed) / float(k + 1)
    p_lo = float(1 + below) / float(k + 1)
    p_two = min(1.0, 2.0 * min(p_up, p_lo))
    return exceed, p_up, p_two


@torch.no_grad()
def evaluate_component_null_report(
    bpd_model: BPDModel,
    clean_traj: RolloutTrajectory,
    run_rollout_fn: Callable[[], RolloutTrajectory],
    *,
    modules: list[str] | None = None,
    components_by_module: dict[str, list[int]] | None = None,
    passes: list[int] | None = None,
    include_all_pass: bool = True,
    n_rotations: int = 8,
    eval_pass: int | None = None,
    state_boundary_weight: float = 1.0,
    eps: float = 1e-8,
    seed: int = 0,
) -> ComponentNullReport:
    """Evaluate raw ablation importance alongside RMS-matched Haar rotation null.

    For each selected ``(module, pass_idx, c)``:
      - ``raw_importance``: output KL / state discrepancy when subtracting component ``c``'s
        write ``(a_c u_c)`` (or replacing with pass-mean write).
      - ``rotation_null_mean``, ``rotation_null_std``: output KL / state discrepancy across
        ``n_rotations`` random read vectors ``r_k`` matched in empirical pass RMS to ``v_c``,
        writing along the exact same ``u_c``.
      - ``net_importance = raw_importance - rotation_null_mean``
      - ``z_rot = (raw_importance - rotation_null_mean) / (rotation_null_std + eps)``
    """
    if modules is not None:
        target_modules = list(modules)
    elif components_by_module is not None:
        target_modules = [m for m in bpd_model.module_order if m in components_by_module]
    else:
        target_modules = list(bpd_model.module_order)
    first_mod = target_modules[0]
    n_passes = clean_traj.module_inputs[first_mod].shape[1]
    dev = clean_traj.module_inputs[first_mod].device
    gen = torch.Generator(device=dev).manual_seed(seed)

    target_passes = list(range(n_passes)) if passes is None else list(passes)
    if eval_pass is None and clean_traj.readouts:
        eval_pass = max(clean_traj.readouts.keys())

    entries: list[ComponentNullEntry] = []

    for mod_name in target_modules:
        w = bpd_model.wrappers[mod_name]
        comp_indices = (
            components_by_module[mod_name]
            if components_by_module is not None and mod_name in components_by_module
            else list(range(w.C))
        )

        # Precompute per-pass RMS-matched rotations for each evaluated component
        rots_per_pass_comp: dict[tuple[int, int], Tensor] = {}
        for tau in range(n_passes):
            x_tau = clean_traj.module_inputs[mod_name][:, tau]
            for c in comp_indices:
                rots_per_pass_comp[(tau, c)] = sample_rms_matched_read_rotations(
                    x_tau, w.V[:, c], n_rotations, generator=gen, eps=eps
                )

        # 1. Per-pass raw vs rotation null
        for tau in target_passes:
            for c in comp_indices:
                # Subtract rank-1 component write (-1.0 * (x @ v_c) u_c) for direct parity with rotation null
                with bpd_model.intervene(
                    mod_name, c, passes={tau}, mode="add_alpha", scale=-1.0
                ):
                    raw_traj = run_rollout_fn()
                raw_imp = _pass_effect_against_clean(
                    raw_traj,
                    clean_traj,
                    eval_pass=eval_pass,
                    state_boundary_weight=state_boundary_weight,
                )

                rots = rots_per_pass_comp[(tau, c)]
                null_vals: list[float] = []
                for k in range(n_rotations):
                    r_col = rots[:, k : k + 1]
                    with bpd_model.intervene(
                        mod_name,
                        c,
                        passes={tau},
                        mode="override_read",
                        scale=1.0,
                        read_override=r_col,
                    ):
                        null_traj = run_rollout_fn()
                    null_vals.append(
                        _pass_effect_against_clean(
                            null_traj,
                            clean_traj,
                            eval_pass=eval_pass,
                            state_boundary_weight=state_boundary_weight,
                        )
                    )
                null_t = torch.tensor(null_vals, dtype=torch.float64)
                n_mean = float(null_t.mean().item())
                n_std = float(null_t.std(unbiased=False).item())
                net_imp = raw_imp - n_mean
                z_rot = net_imp / (n_std + eps)
                exceed, p_up, p_two = compute_rank_p_values(raw_imp, null_t)
                entries.append(
                    ComponentNullEntry(
                        module=mod_name,
                        pass_idx=tau,
                        component=c,
                        raw_importance=raw_imp,
                        rotation_null_mean=n_mean,
                        rotation_null_std=n_std,
                        net_importance=net_imp,
                        z_rot=z_rot,
                        n_rotations=len(null_vals),
                        exceed_count=exceed,
                        p_rank=p_up,
                        p_rank_two_sided=p_two,
                    )
                )

        # 2. All-pass sustaining ablation vs time-shared rotation null
        if include_all_pass and n_passes > 1:
            x_all = clean_traj.module_inputs[mod_name]  # [B, T, P, d_in]
            for c in comp_indices:
                with bpd_model.intervene(
                    mod_name, c, passes=None, mode="add_alpha", scale=-1.0
                ):
                    raw_traj = run_rollout_fn()
                raw_imp = _pass_effect_against_clean(
                    raw_traj,
                    clean_traj,
                    eval_pass=eval_pass,
                    state_boundary_weight=state_boundary_weight,
                )

                # Time-shared rotation direction matched to exact per-pass projection RMS (HUG-CAL-02)
                rots_shared = sample_per_pass_rms_matched_rotations(
                    x_all, w.V[:, c], n_rotations, generator=gen, eps=eps
                )
                null_vals = []
                for k in range(n_rotations):
                    r_col = rots_shared[:, :, k : k + 1]  # [T, d_in, 1]
                    with bpd_model.intervene(
                        mod_name,
                        c,
                        passes=None,
                        mode="override_read",
                        scale=1.0,
                        read_override=r_col,
                    ):
                        null_traj = run_rollout_fn()
                    null_vals.append(
                        _pass_effect_against_clean(
                            null_traj,
                            clean_traj,
                            eval_pass=eval_pass,
                            state_boundary_weight=state_boundary_weight,
                        )
                    )
                null_t = torch.tensor(null_vals, dtype=torch.float64)
                n_mean = float(null_t.mean().item())
                n_std = float(null_t.std(unbiased=False).item())
                net_imp = raw_imp - n_mean
                z_rot = net_imp / (n_std + eps)
                exceed, p_up, p_two = compute_rank_p_values(raw_imp, null_t)
                entries.append(
                    ComponentNullEntry(
                        module=mod_name,
                        pass_idx=-1,
                        component=c,
                        raw_importance=raw_imp,
                        rotation_null_mean=n_mean,
                        rotation_null_std=n_std,
                        net_importance=net_imp,
                        z_rot=z_rot,
                        n_rotations=len(null_vals),
                        exceed_count=exceed,
                        p_rank=p_up,
                        p_rank_two_sided=p_two,
                    )
                )

    return ComponentNullReport(entries=entries)
