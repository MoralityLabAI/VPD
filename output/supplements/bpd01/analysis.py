"""Loop-specific mechanistic diagnostics that standard single-pass VPD cannot yield:

  1. **Schedule extraction** (``extract_schedule``): which components are active at
     which pass ``tau``, per input and across the batch, with duty cycle, center-of-mass
     pass, temporal entropy, and switch rate.
  2. **Transient vs. Sustaining classification** (``classify_transient_sustaining``):
     separates components whose intermediate trajectory effect is rebuilt by later
     passes (``"transient"``) from components that hold a fixed point and only degrade
     the final output when ablated at every pass (``"sustaining"``), alongside
     single-pass irrecoverable ``"bottleneck"`` and ``"inert"`` components — reporting
     both raw and net-of-rotation-null metrics.
  3. **Component transition operator & interventional path composition**
     (``compute_transition_operator``, ``compose_interventional_path``): one-step
     attribution from ``(c', tau)`` to ``(c, tau + 1)`` pooled over passes, plus
     multi-step nonlinear ``do``-interventional path composition contrasted against
     first-order matrix products.
  4. **Sustaining checkpoint diff & time-shared weight edit**
     (``rank_sustaining_checkpoint_diffs``, ``apply_sustaining_weight_edit``):
     identifies sustaining components differing between two checkpoints (Gate B4:
     ``gaoxin492`` vs ``yagizdevre``) and applies a permanent rank-``k`` weight edit
     without any external refill network.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal

import torch
from torch import Tensor, nn

from mechinterp.bpd.ci import CIOutputs
from mechinterp.bpd.components import BPDModel, RolloutTrajectory
from mechinterp.bpd.losses import boundary_state_loss, kl_divergence_logits
from mechinterp.bpd.nulls import sample_per_pass_rms_matched_rotations


@dataclass
class ComponentKey:
    """Global identifier for component ``c`` in shared module ``module``."""

    module: str
    component: int


@dataclass
class ComponentScheduleReport:
    """Per-input and batch-pooled pass schedule ``[T, C_total]``."""

    keys: list[ComponentKey]
    per_input_schedule: Tensor  # [B, T, C_total]
    mean_schedule: Tensor  # [T, C_total]
    duty_cycle: Tensor  # [C_total] in [0, 1]
    center_of_mass_pass: Tensor  # [C_total] in [0, T - 1]
    temporal_entropy: Tensor  # [C_total] normalized to [0, 1]
    switch_rate: Tensor  # [C_total] mean |g_{tau+1} - g_tau|
    is_stationary: Tensor  # [C_total] bool


def extract_schedule(
    ci_out: CIOutputs,
    *,
    position_pool: Literal["mean", "max", "last"] = "mean",
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> ComponentScheduleReport:
    """Extract per-input ``[B, T, C_total]`` and summary temporal schedules from ``CIOutputs``."""
    keys: list[ComponentKey] = []
    parts: list[Tensor] = []

    for name in sorted(ci_out.lower_leaky.keys()):
        g = ci_out.lower_leaky[name].detach().float()  # [B, T, P, C]
        c_cnt = g.shape[-1]
        for c in range(c_cnt):
            keys.append(ComponentKey(module=name, component=c))
        if position_pool == "mean":
            pooled = g.mean(dim=2)  # [B, T, C]
        elif position_pool == "max":
            pooled = g.amax(dim=2)
        elif position_pool == "last":
            pooled = g[:, :, -1, :]
        else:
            raise ValueError(f"Unknown position_pool: {position_pool!r}")
        parts.append(pooled)

    per_input = torch.cat(parts, dim=-1)  # [B, T, C_total]
    mean_sched = per_input.mean(dim=0)  # [T, C_total]
    n_passes = per_input.shape[1]

    duty = (per_input > threshold).float().mean(dim=(0, 1))  # [C_total]

    tau_grid = torch.arange(n_passes, device=per_input.device, dtype=torch.float32).view(1, n_passes, 1)
    mass = per_input.sum(dim=1).clamp_min(eps)  # [B, C_total]
    com = ((per_input * tau_grid).sum(dim=1) / mass).mean(dim=0)  # [C_total]

    if n_passes > 1:
        probs = (per_input + eps) / (per_input + eps).sum(dim=1, keepdim=True)
        ent = -(probs * probs.log()).sum(dim=1) / math.log(float(n_passes))
        entropy = ent.mean(dim=0)
        switches = (per_input[:, 1:, :] - per_input[:, :-1, :]).abs().mean(dim=(0, 1))
    else:
        entropy = torch.ones_like(duty)
        switches = torch.zeros_like(duty)

    stationary = (duty >= 0.7) & (switches <= 0.15)
    return ComponentScheduleReport(
        keys=keys,
        per_input_schedule=per_input,
        mean_schedule=mean_sched,
        duty_cycle=duty,
        center_of_mass_pass=com,
        temporal_entropy=entropy,
        switch_rate=switches,
        is_stationary=stationary,
    )


def _trajectory_intermediate_and_final_shift(
    pert_traj: RolloutTrajectory,
    clean_traj: RolloutTrajectory,
    *,
    state_boundary_weight: float = 1.0,
) -> tuple[float, float]:
    """Return ``(mean_intermediate_shift, final_shift)`` against ``clean_traj``."""
    all_passes = sorted(set(clean_traj.readouts.keys()) | set(clean_traj.boundary_states.keys()))
    if not all_passes:
        return 0.0, 0.0
    final_tau = all_passes[-1]

    def _loss_at(tau: int) -> float:
        if tau in clean_traj.readouts:
            return float(
                kl_divergence_logits(pert_traj.readouts[tau], clean_traj.readouts[tau]).item()
            )
        return float(
            (
                state_boundary_weight
                * boundary_state_loss(
                    pert_traj.boundary_states[tau], clean_traj.boundary_states[tau]
                )
            ).item()
        )

    final_val = _loss_at(final_tau)
    inter_passes = [t for t in all_passes if t < final_tau]
    if not inter_passes:
        return final_val, final_val
    inter_val = max(_loss_at(t) for t in inter_passes)
    return inter_val, final_val


@dataclass
class TransientSustainingEntry:
    """Classification of a shared component's temporal causal role across passes."""

    module: str
    component: int
    role: Literal["transient", "sustaining", "bottleneck", "inert"]
    raw_max_single_traj: float
    raw_max_single_final: float
    raw_all_pass_final: float
    null_all_pass_final: float
    net_all_pass_final: float
    sustain_amplification: float


@torch.no_grad()
def classify_transient_sustaining(
    bpd_model: BPDModel,
    clean_traj: RolloutTrajectory,
    run_rollout_fn: Callable[[], RolloutTrajectory],
    *,
    components_by_module: dict[str, list[int]] | None = None,
    min_effect: float = 1e-2,
    sustain_ratio: float = 2.0,
    recovery_ratio: float = 0.35,
    n_rotations: int = 4,
    ablation_mode: Literal["ablate", "add_alpha"] = "ablate",
    state_boundary_weight: float = 1.0,
    eps: float = 1e-8,
    seed: int = 0,
) -> list[TransientSustainingEntry]:
    """Classify components into ``transient``, ``sustaining``, ``bottleneck``, or ``inert``.

    Definitions:
      - ``sustaining``: ``all_pass_final >= min_effect`` (and positive net-of-null when
        ``n_rotations > 0``) while ``all_pass_final >= sustain_ratio * max_single_final``
        — single-pass damage is repaired by the loop, so necessity appears only when
        ablated at every pass (holding the fixed point).
      - ``transient``: ``max_single_traj >= min_effect`` while both ``max_single_final``
        and ``all_pass_final`` remain ``<= max(min_effect, recovery_ratio * max_single_traj)``
        — intermediate trajectory work that later passes overwrite or rebuild.
      - ``bottleneck``: ``max_single_final >= min_effect`` (single-pass ablation already
        degrades the final readout).
      - ``inert``: below ``min_effect`` on both intermediate trajectory and final readout.
    """
    first_mod = bpd_model.module_order[0]
    n_passes = clean_traj.module_inputs[first_mod].shape[1]
    dev = clean_traj.module_inputs[first_mod].device
    gen = torch.Generator(device=dev).manual_seed(seed)
    scale_arg = 0.0 if ablation_mode == "ablate" else -1.0

    results: list[TransientSustainingEntry] = []
    for mod_name in bpd_model.module_order:
        w = bpd_model.wrappers[mod_name]
        comp_list = (
            components_by_module[mod_name]
            if components_by_module is not None and mod_name in components_by_module
            else list(range(w.C))
        )
        x_all = clean_traj.module_inputs[mod_name]

        for c in comp_list:
            max_single_traj = 0.0
            max_single_final = 0.0
            for tau in range(n_passes):
                with bpd_model.intervene(
                    mod_name, c, passes={tau}, mode=ablation_mode, scale=scale_arg
                ):
                    pert = run_rollout_fn()
                inter_v, final_v = _trajectory_intermediate_and_final_shift(
                    pert, clean_traj, state_boundary_weight=state_boundary_weight
                )
                max_single_traj = max(max_single_traj, inter_v)
                max_single_final = max(max_single_final, final_v)

            with bpd_model.intervene(
                mod_name, c, passes=None, mode=ablation_mode, scale=scale_arg
            ):
                all_pert = run_rollout_fn()
            _, all_pass_final = _trajectory_intermediate_and_final_shift(
                all_pert, clean_traj, state_boundary_weight=state_boundary_weight
            )

            null_all_final = 0.0
            if n_rotations > 0:
                rots = sample_per_pass_rms_matched_rotations(
                    x_all, w.V[:, c], n_rotations, generator=gen, eps=eps
                )
                n_vals: list[float] = []
                for k in range(n_rotations):
                    with bpd_model.intervene(
                        mod_name,
                        c,
                        passes=None,
                        mode="override_read",
                        scale=1.0,
                        read_override=rots[:, :, k : k + 1],
                    ):
                        n_traj = run_rollout_fn()
                    _, nf = _trajectory_intermediate_and_final_shift(
                        n_traj, clean_traj, state_boundary_weight=state_boundary_weight
                    )
                    n_vals.append(nf)
                null_all_final = sum(n_vals) / float(len(n_vals))

            net_all_final = all_pass_final - null_all_final
            amp = all_pass_final / max(max_single_final, eps)

            # Decision rule
            is_transient = (
                n_passes > 1
                and max_single_traj >= min_effect
                and max_single_final <= max(min_effect, recovery_ratio * max_single_traj)
                and all_pass_final <= max(min_effect, recovery_ratio * max_single_traj)
            )
            is_sustaining = (
                n_passes > 1
                and all_pass_final >= min_effect
                and amp >= sustain_ratio
                and (n_rotations == 0 or net_all_final > 0.0)
            )

            if is_transient:
                role: Literal["transient", "sustaining", "bottleneck", "inert"] = "transient"
            elif is_sustaining:
                role = "sustaining"
            elif max_single_final >= min_effect or all_pass_final >= min_effect:
                role = "bottleneck"
            else:
                role = "inert"

            results.append(
                TransientSustainingEntry(
                    module=mod_name,
                    component=c,
                    role=role,
                    raw_max_single_traj=max_single_traj,
                    raw_max_single_final=max_single_final,
                    raw_all_pass_final=all_pass_final,
                    null_all_pass_final=null_all_final,
                    net_all_pass_final=net_all_final,
                    sustain_amplification=amp,
                )
            )

    return results


@dataclass
class TransitionOperator:
    """Pooled one-step component transition matrix ``(c', tau) -> (c, tau + 1)``."""

    keys: list[ComponentKey]
    matrix: Tensor  # [C_sub, C_sub]: entry [i, j] is attribution from keys[i] at tau to keys[j] at tau+1
    per_pass_matrices: Tensor  # [T - 1, C_sub, C_sub]


@torch.no_grad()
def compute_transition_operator(
    bpd_model: BPDModel,
    clean_traj: RolloutTrajectory,
    run_rollout_fn: Callable[[], RolloutTrajectory],
    *,
    components_by_module: dict[str, list[int]] | None = None,
    ablation_mode: Literal["ablate", "add_alpha"] = "ablate",
) -> TransitionOperator:
    """Compute 1-step attribution ``(c', tau) -> (c, tau + 1)`` pooled over passes ``0..T-2``.

    Entry ``M[i, j]`` measures the RMS shift in target component ``j``'s read
    activation ``a_{j, tau+1} = X_{m_j, tau+1} @ v_{j}`` on pass ``tau + 1`` caused
    by ablating source component ``i`` exclusively on pass ``tau``, pooled over
    ``tau = 0, ..., T - 2``.
    """
    keys: list[ComponentKey] = []
    for mod_name in bpd_model.module_order:
        w = bpd_model.wrappers[mod_name]
        c_list = (
            components_by_module[mod_name]
            if components_by_module is not None and mod_name in components_by_module
            else list(range(w.C))
        )
        for c in c_list:
            keys.append(ComponentKey(module=mod_name, component=c))

    first_mod = bpd_model.module_order[0]
    n_passes = clean_traj.module_inputs[first_mod].shape[1]
    if n_passes < 2:
        raise ValueError("compute_transition_operator requires n_passes >= 2")

    dev = clean_traj.module_inputs[first_mod].device
    n_keys = len(keys)
    per_pass = torch.zeros((n_passes - 1, n_keys, n_keys), device=dev, dtype=torch.float32)
    scale_arg = 0.0 if ablation_mode == "ablate" else -1.0

    # Precompute clean component read projections a_{j, tau} = X_{m_j, tau} @ v_j
    clean_acts: list[Tensor] = []
    for key in keys:
        w = bpd_model.wrappers[key.module]
        a = clean_traj.module_inputs[key.module].float() @ w.V[:, key.component]  # [B, T, P]
        clean_acts.append(a)

    for tau in range(n_passes - 1):
        for src_idx, src_key in enumerate(keys):
            with bpd_model.intervene(
                src_key.module,
                src_key.component,
                passes={tau},
                mode=ablation_mode,
                scale=scale_arg,
            ):
                pert_traj = run_rollout_fn()

            for dst_idx, dst_key in enumerate(keys):
                w_dst = bpd_model.wrappers[dst_key.module]
                pert_x_next = pert_traj.module_inputs[dst_key.module][:, tau + 1].float()
                pert_a_next = pert_x_next @ w_dst.V[:, dst_key.component]  # [B, P]
                clean_a_next = clean_acts[dst_idx][:, tau + 1]  # [B, P]
                rms_shift = (pert_a_next - clean_a_next).pow(2).mean().sqrt()
                per_pass[tau, src_idx, dst_idx] = rms_shift

    pooled = per_pass.mean(dim=0)
    return TransitionOperator(keys=keys, matrix=pooled, per_pass_matrices=per_pass)


@dataclass
class InterventionalPathResult:
    """Nonlinear multi-step path mediation vs first-order matrix-product approximation."""

    path: list[ComponentKey]
    start_pass: int
    direct_source_effect: float
    interventional_mediated_effect: float
    first_order_chain_product: float
    transport_discrepancy: float


@torch.no_grad()
def compose_interventional_path(
    bpd_model: BPDModel,
    clean_traj: RolloutTrajectory,
    run_rollout_fn: Callable[[], RolloutTrajectory],
    transition_op: TransitionOperator,
    path: list[ComponentKey],
    *,
    start_pass: int = 0,
    ablation_mode: Literal["ablate", "add_alpha"] = "ablate",
) -> InterventionalPathResult:
    """Compose a multi-step component path ``(c_0, tau_0) -> ... -> (c_L, tau_0 + L)`` by intervention.

    Because nonlinear normalization and gating break first-order Jacobian products
    over multiple loop steps, this function evaluates path mediation directly via
    inclusion-exclusion ``do``-interventions on the source ``(c_0, tau_0)`` and
    intermediate relay nodes ``{(c_k, tau_0 + k)}_{k=1..L-1}``, measuring the
    non-additive causal interaction on sink ``(c_L, tau_0 + L)`` and contrasting
    it against the normalized first-order chain product from ``transition_op``.
    """
    if len(path) < 2:
        raise ValueError("Path must contain at least source and sink (len >= 2)")
    n_steps = len(path) - 1
    sink_pass = start_pass + n_steps
    first_mod = bpd_model.module_order[0]
    n_passes = clean_traj.module_inputs[first_mod].shape[1]
    if sink_pass >= n_passes:
        raise ValueError(f"sink_pass={sink_pass} exceeds trajectory n_passes={n_passes}")

    key_to_idx = {(k.module, k.component): i for i, k in enumerate(transition_op.keys)}
    scale_arg = 0.0 if ablation_mode == "ablate" else -1.0

    sink = path[-1]
    w_sink = bpd_model.wrappers[sink.module]
    clean_sink_act = (
        clean_traj.module_inputs[sink.module][:, sink_pass].float() @ w_sink.V[:, sink.component]
    )

    def _run_multi_ablation(nodes_and_passes: list[tuple[ComponentKey, int]]) -> Tensor:
        # Install temporary interventions across the requested (module, component, pass) pairs
        bpd_model.clear_schedules()
        bpd_model.reset_wrappers(pass_offset=0)
        saved: list[tuple[str, object]] = []
        # Group by module if multiple nodes hit different passes on the same module:
        # For distinct passes on the same module, apply sequential schedule mask on component mode
        # if needed, or direct intervention when modules/passes don't collide.
        by_mod: dict[str, list[tuple[int, int]]] = {}
        for k_node, p_idx in nodes_and_passes:
            by_mod.setdefault(k_node.module, []).append((k_node.component, p_idx))

        # Use exact component-mode schedule with mask=1 everywhere except 0 at target (comp, pass)
        bsz, _, seq_len, _ = clean_traj.module_inputs[first_mod].shape
        dev = clean_traj.module_inputs[first_mod].device
        from mechinterp.bpd.components import PassMaskSchedule

        schedules: dict[str, PassMaskSchedule] = {}
        for mod_name in bpd_model.module_order:
            w = bpd_model.wrappers[mod_name]
            cm = torch.ones((bsz, n_passes, seq_len, w.C), device=dev, dtype=torch.float32)
            dm = torch.ones((bsz, n_passes, seq_len), device=dev, dtype=torch.float32)
            for comp_i, pass_i in by_mod.get(mod_name, []):
                cm[:, pass_i, :, comp_i] = 0.0
            schedules[mod_name] = PassMaskSchedule(comp_mask=cm, delta_mask=dm)

        with bpd_model.masked_components(schedules, record_inputs=True):
            traj = run_rollout_fn()
        return traj.module_inputs[sink.module][:, sink_pass].float() @ w_sink.V[:, sink.component]

    src_spec = [(path[0], start_pass)]
    relays_spec = [(path[k], start_pass + k) for k in range(1, n_steps)]

    act_no_src = _run_multi_ablation(src_spec)
    direct_src_eff = float((act_no_src - clean_sink_act).pow(2).mean().sqrt().item())

    if not relays_spec:
        mediated = direct_src_eff
    else:
        act_no_relays = _run_multi_ablation(relays_spec)
        act_no_both = _run_multi_ablation(src_spec + relays_spec)
        # Effect of source when relays are intact vs when relays are already blocked
        eff_intact = (act_no_src - clean_sink_act)
        eff_blocked = (act_no_both - act_no_relays)
        mediated = float((eff_intact - eff_blocked).pow(2).mean().sqrt().item())

    # First-order pooled chain estimate (normalized by clean activation RMS at intermediate nodes)
    fo_prod = 1.0
    for k in range(n_steps):
        i_src = key_to_idx[(path[k].module, path[k].component)]
        i_dst = key_to_idx[(path[k + 1].module, path[k + 1].component)]
        step_shift = float(transition_op.matrix[i_src, i_dst].item())
        if k > 0:
            w_mid = bpd_model.wrappers[path[k].module]
            mid_act = (
                clean_traj.module_inputs[path[k].module][:, start_pass + k].float()
                @ w_mid.V[:, path[k].component]
            )
            mid_rms = max(float(mid_act.pow(2).mean().sqrt().item()), 1e-6)
            fo_prod *= step_shift / mid_rms
        else:
            fo_prod *= step_shift

    return InterventionalPathResult(
        path=path,
        start_pass=start_pass,
        direct_source_effect=direct_src_eff,
        interventional_mediated_effect=mediated,
        first_order_chain_product=fo_prod,
        transport_discrepancy=abs(mediated - fo_prod),
    )


@dataclass
class SustainingCheckpointDiff:
    """Comparison of a sustaining component's effective weight coefficient across two checkpoints."""

    module: str
    component: int
    coeff_reference: float
    coeff_target: float
    coeff_gap: float
    net_sustaining_importance: float
    priority_score: float


@torch.no_grad()
def rank_sustaining_checkpoint_diffs(
    bpd_reference: BPDModel,
    target_checkpoint_model: nn.Module,
    classifications: list[TransientSustainingEntry],
    *,
    eps: float = 1e-8,
) -> list[SustainingCheckpointDiff]:
    """Rank sustaining components by parameter projection gap between reference and target checkpoints.

    For each component ``(v_c, u_c)`` classified as ``"sustaining"`` on ``bpd_reference``
    (e.g. ``gaoxin492``), computes its rank-1 Frobenius alignment coefficient

        ``alpha(W) = <W, u_c v_c^T>_F / ||u_c v_c^T||_F^2``

    on both ``W_ref`` and ``W_target_ckpt`` (e.g. ``yagizdevre``) and ranks by
    ``|alpha_ref - alpha_tgt| * max(net_all_pass_final, 0)``.
    """
    diffs: list[SustainingCheckpointDiff] = []
    for entry in classifications:
        if entry.role != "sustaining":
            continue
        w_wrap = bpd_reference.wrappers[entry.module]
        v_c = w_wrap.V[:, entry.component].detach().float()  # [d_in]
        u_c = w_wrap.U[entry.component, :].detach().float()  # [d_out]
        denom = float((v_c.pow(2).sum() * u_c.pow(2).sum()).clamp_min(eps).item())

        w_ref = w_wrap.W_target.detach().float()  # [d_out, d_in]
        tgt_sub = target_checkpoint_model.get_submodule(entry.module)
        w_tgt = (
            tgt_sub.W_target.detach().float()
            if hasattr(tgt_sub, "W_target")
            else tgt_sub.weight.detach().float()
        ).to(v_c.device)

        # <W, u v^T>_F = u^T W v
        coeff_ref = float((u_c @ (w_ref @ v_c)).item()) / denom
        coeff_tgt = float((u_c @ (w_tgt @ v_c)).item()) / denom
        gap = coeff_ref - coeff_tgt
        score = abs(gap) * max(entry.net_all_pass_final, 0.0)
        diffs.append(
            SustainingCheckpointDiff(
                module=entry.module,
                component=entry.component,
                coeff_reference=coeff_ref,
                coeff_target=coeff_tgt,
                coeff_gap=gap,
                net_sustaining_importance=entry.net_all_pass_final,
                priority_score=score,
            )
        )

    diffs.sort(key=lambda d: d.priority_score, reverse=True)
    return diffs


@torch.no_grad()
def apply_sustaining_weight_edit(
    target_checkpoint_model: nn.Module,
    bpd_reference: BPDModel,
    selected_diffs: list[SustainingCheckpointDiff],
    *,
    interpolation: float = 1.0,
) -> dict[str, int]:
    """Apply a permanent time-shared rank-``k`` weight edit to ``target_checkpoint_model``.

    Updates physical weight matrix ``W_tgt`` in-place via:

        ``W_tgt <- W_tgt + interpolation * sum_{c in selected} (alpha_ref - alpha_tgt) u_c v_c^T``

    Requires zero runtime hooks or auxiliary refill networks at inference time.
    """
    edited_counts: dict[str, int] = {}
    for diff in selected_diffs:
        w_wrap = bpd_reference.wrappers[diff.module]
        v_c = w_wrap.V[:, diff.component].detach().float()  # [d_in]
        u_c = w_wrap.U[diff.component, :].detach().float()  # [d_out]
        delta_rank1 = (interpolation * diff.coeff_gap) * torch.outer(u_c, v_c)

        tgt_mod = target_checkpoint_model.get_submodule(diff.module)
        if hasattr(tgt_mod, "W_target"):
            tgt_w = tgt_mod.W_target
            tgt_w.copy_((tgt_w.float() + delta_rank1.to(tgt_w.device)).to(tgt_w.dtype))
        elif isinstance(tgt_mod, nn.Linear):
            tgt_w = tgt_mod.weight.data
            tgt_w.copy_((tgt_w.float() + delta_rank1.to(tgt_w.device)).to(tgt_w.dtype))
        else:
            raise TypeError(f"Cannot edit unsupported module type {type(tgt_mod).__name__}")
        edited_counts[diff.module] = edited_counts.get(diff.module, 0) + 1
    return edited_counts
