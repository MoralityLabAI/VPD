"""End-to-end tests for BPD's falsifiable gates (B1-B4), planted rank-1 positive
controls, transient vs. sustaining classification, interventional transition
composition, intra-checkpoint functional sustaining edits, and CLI smoke pipelines.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import Tensor, nn

from mechinterp.bpd.adapters.huginn import SettlingCoreLooper, evaluate_hug_cal02_with_positive_arm
from mechinterp.bpd.adapters.looped_chase import (
    DEFAULT_LOOPED_CHASE_CKPT_DIR,
    evaluate_planted_rank1_looped_chase,
    load_looped_chase_checkpoint,
    resolve_looped_chase_checkpoint,
)
from mechinterp.bpd.adapters.trm import build_mini_trm
from mechinterp.bpd.analysis import (
    ComponentKey,
    apply_intra_checkpoint_sustaining_edit,
    apply_random_component_control_edit,
    apply_rotation_null_control_edit,
    classify_transient_sustaining,
    compose_interventional_path,
    compute_transition_operator,
    extract_component_functional_profiles,
    match_components_by_function,
)
from mechinterp.bpd.ci import CIOutputs
from mechinterp.bpd.components import BPDModel, RolloutTrajectory
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.scripts.bench_unrolled_vs_tied import run_benchmark
from mechinterp.bpd.scripts.run_looped_chase_bpd import run_looped_chase_experiment
from mechinterp.bpd.scripts.run_trm_bpd import run_trm_bpd_pipeline
from mechinterp.loopscope01.scripts.loopscope_toy import LoopedChase


def test_gate_b1_stationary_tied_matches_unrolled_at_one_over_T_components() -> None:
    res = run_benchmark(
        n_passes=4,
        c_shared=4,
        n_steps=45,
        batch_size=16,
        seq_len=6,
        device="cpu",
        seed=7,
    )
    assert res["compression_ratio"] == 4.0
    assert res["gate_b1_passed"] is True


class _CanonicalRolesLooper(nn.Module):
    """Exact 4-pass weight-tied oracle with 1 transient, 1 sustaining, and 1 inert component.

    State ``h`` has dimension 6:
      - ``h[..., 0]``: transient channel. Component 0 reads ``h[..., 2]`` (nonzero only at
        pass 0 before decay) and writes to ``h[..., 0]``. On passes >= 1, prompt re-injection
        clamps ``h[..., 0]`` directly to ``x[..., 0]``, erasing pass-0 perturbations by pass 3!
      - ``h[..., 1]``: sustaining fixed-point channel. Each pass decays ``h[..., 1]`` to 0 unless
        Component 1 (reading prompt carrier ``h[..., 3]``) refreshes ``h[..., 1]`` toward ``+3.0``
        through a saturating ``tanh``. Dropping Component 1 at any single pass ``tau < 3`` is
        immediately restored on pass 3, AND even dropping at pass 3 alone only partially decays
        ``0.82 * 3.0 = 2.46`` (small logit shift), whereas dropping Component 1 at ALL 4 passes
        keeps ``h[..., 1] == 0.0`` (large final KL!).
      - Component 2 writes strictly into dead coordinate ``h[..., 5]`` unread by head -> inert.
    """

    def __init__(self) -> None:
        super().__init__()
        self.core = nn.Linear(6, 6, bias=False)
        with torch.no_grad():
            self.core.weight.zero_()
            self.core.weight[0, 2] = 4.0  # Comp A: transient
            self.core.weight[1, 3] = 2.5  # Comp B: sustaining
            self.core.weight[5, 4] = 0.5  # Comp C: inert (dead subspace)

    def rollout(self, x_sign: Tensor, n_passes: int = 4) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        bsz = x_sign.shape[0]
        h = torch.zeros(bsz, 1, 6)
        h[:, 0, 2] = x_sign  # Active ONLY entering pass 0
        h[:, 0, 3] = x_sign  # Persistent carrier every pass
        h[:, 0, 4] = 1.0

        readouts: dict[int, Tensor] = {}
        boundaries: dict[int, Tensor] = {}
        for tau in range(n_passes):
            w = self.core(h)
            if tau == 0:
                c0 = torch.tanh(w[:, :, 0]) * 3.0
            else:
                # Re-read loop overwrites transient channel from prompt x_sign!
                c0 = x_sign.unsqueeze(-1) * 3.0
            # Leaky integrator on sustaining channel 1: single-pass drop barely shifts attractor,
            # all-pass drop leaves c1 == 0!
            c1 = 0.85 * h[:, :, 1] + 0.45 * w[:, :, 1]
            c1 = c1.clamp(-3.0, 3.0)

            h_next = torch.zeros_like(h)
            h_next[:, :, 0] = c0
            h_next[:, :, 1] = c1
            h_next[:, :, 2] = 0.0  # Zeroed after pass 0
            h_next[:, :, 3] = x_sign.unsqueeze(-1)
            h_next[:, :, 4] = 1.0
            h_next[:, :, 5] = w[:, :, 5]
            h = h_next

            # Behavioral 4-way logits read channels 0 and 1
            logits = torch.stack(
                [h[:, :, 0] + h[:, :, 1], -h[:, :, 0] + h[:, :, 1], h[:, :, 0] - h[:, :, 1], -h[:, :, 0] - h[:, :, 1]],
                dim=-1,
            )
            readouts[tau] = logits
            boundaries[tau] = h
        return readouts, boundaries


def test_transient_vs_sustaining_and_intra_checkpoint_functional_edit() -> None:
    oracle = _CanonicalRolesLooper()
    bpd_ref = BPDModel(
        oracle,
        {"core": 3},
        max_passes=4,
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        init_svd=False,
    )
    # Set exact 3 rank-1 factors matching the 3 planted mechanisms
    with torch.no_grad():
        bpd_ref.wrappers["core"].V.zero_()
        bpd_ref.wrappers["core"].U.zero_()
        # Comp 0: transient (reads 2, writes 0, weight 4.0)
        bpd_ref.wrappers["core"].V[2, 0] = 2.0
        bpd_ref.wrappers["core"].U[0, 0] = 2.0
        # Comp 1: sustaining (reads 3, writes 1, weight 2.5)
        bpd_ref.wrappers["core"].V[3, 1] = 1.0
        bpd_ref.wrappers["core"].U[1, 1] = 2.5
        # Comp 2: inert (reads 4, writes 5, weight 0.5)
        bpd_ref.wrappers["core"].V[4, 2] = 1.0
        bpd_ref.wrappers["core"].U[2, 5] = 0.5

    signs = torch.tensor([1.0, -1.0] * 12)
    with bpd_ref.target_capture():
        ro, bs = oracle.rollout(signs, n_passes=4)
    mod_in = bpd_ref.collect_recorded_inputs(expected_passes=4)
    clean_traj_ref = RolloutTrajectory(readouts=ro, boundary_states=bs, module_inputs=mod_in)

    def _run_oracle() -> RolloutTrajectory:
        p_ro, p_bs = oracle.rollout(signs, n_passes=4)
        return RolloutTrajectory(readouts=p_ro, boundary_states=p_bs, module_inputs=mod_in)

    roles_ref = classify_transient_sustaining(
        bpd_ref,
        clean_traj_ref,
        _run_oracle,
        min_effect=0.02,
        sustain_ratio=2.5,
        n_rotations=4,
        ablation_mode="ablate",
        seed=0,
    )
    role_by_c = {e.component: e.role for e in roles_ref}
    assert role_by_c[0] == "transient"
    assert role_by_c[1] == "sustaining"
    assert role_by_c[2] == "inert"

    # Separate decomposition of degraded target checkpoint with PERMUTED component slot order
    degraded_base = _CanonicalRolesLooper()
    with torch.no_grad():
        degraded_base.core.weight[1, 3] = 0.5  # weakened sustaining mechanism (0.5 vs 2.5)

    degraded_for_bpd = copy.deepcopy(degraded_base)
    bpd_tgt = BPDModel(
        degraded_for_bpd,
        {"core": 3},
        max_passes=4,
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        init_svd=False,
    )
    with torch.no_grad():
        bpd_tgt.wrappers["core"].V.zero_()
        bpd_tgt.wrappers["core"].U.zero_()
        # Slot 0: inert; Slot 1: transient; Slot 2: sustaining (permuted vs bpd_ref!)
        bpd_tgt.wrappers["core"].V[4, 0] = 1.0
        bpd_tgt.wrappers["core"].U[0, 5] = 0.5
        bpd_tgt.wrappers["core"].V[2, 1] = 2.0
        bpd_tgt.wrappers["core"].U[1, 0] = 2.0
        bpd_tgt.wrappers["core"].V[3, 2] = 1.0
        bpd_tgt.wrappers["core"].U[2, 1] = 0.5

    with bpd_tgt.target_capture():
        ro_t, bs_t = degraded_for_bpd.rollout(signs, n_passes=4)
    mod_in_t = bpd_tgt.collect_recorded_inputs(expected_passes=4)
    clean_traj_tgt = RolloutTrajectory(readouts=ro_t, boundary_states=bs_t, module_inputs=mod_in_t)

    def _run_degraded() -> RolloutTrajectory:
        p_ro, p_bs = degraded_for_bpd.rollout(signs, n_passes=4)
        return RolloutTrajectory(readouts=p_ro, boundary_states=p_bs, module_inputs=mod_in_t)

    roles_tgt = classify_transient_sustaining(
        bpd_tgt,
        clean_traj_tgt,
        _run_degraded,
        min_effect=0.02,
        sustain_ratio=1.5,
        n_rotations=4,
        ablation_mode="ablate",
        seed=1,
    )

    # Construct functional schedules matching actual temporal usage: transient on pass 0, sustaining on all passes
    bsz = signs.shape[0]
    sched_ref = torch.zeros(bsz, 4, 1, 3)
    sched_ref[:, 0, :, 0] = 1.0
    sched_ref[:, :, :, 1] = 1.0
    ci_ref = CIOutputs(lower_leaky={"core": sched_ref}, upper_leaky={"core": sched_ref}, pre_sigmoid={"core": sched_ref})

    sched_tgt = torch.zeros(bsz, 4, 1, 3)
    sched_tgt[:, 0, :, 1] = 1.0
    sched_tgt[:, :, :, 2] = 1.0
    ci_tgt = CIOutputs(lower_leaky={"core": sched_tgt}, upper_leaky={"core": sched_tgt}, pre_sigmoid={"core": sched_tgt})

    prof_ref = extract_component_functional_profiles(
        bpd_ref, ci_ref, clean_traj_ref, _run_oracle, roles_ref
    )
    prof_tgt = extract_component_functional_profiles(
        bpd_tgt, ci_tgt, clean_traj_tgt, _run_degraded, roles_tgt
    )
    matches = match_components_by_function(prof_ref, prof_tgt)
    assert len(matches) == 1
    assert matches[0].ref_component == 1
    assert matches[0].tgt_component == 2  # Matched across permuted slots purely by function!

    # Edit ONLY within degraded_base's own decomposition (0.5 + 4.0 * 0.5 = 2.5)
    edited_tgt = copy.deepcopy(degraded_base)
    edit_rc = apply_intra_checkpoint_sustaining_edit(
        edited_tgt, bpd_tgt, matches, scale_delta=4.0
    )
    assert edit_rc.edited_modules == {"core": 1}

    # Controls of exact matched count and Frobenius norm
    rand_tgt = copy.deepcopy(degraded_base)
    rand_rc = apply_random_component_control_edit(
        rand_tgt, bpd_tgt, edit_rc, exclude_matches=matches, seed=7
    )
    rot_tgt = copy.deepcopy(degraded_base)
    rot_rc = apply_rotation_null_control_edit(
        rot_tgt, bpd_tgt, matches, edit_rc, clean_traj=clean_traj_tgt, seed=11
    )
    assert math.isclose(rand_rc.total_frobenius_norm, edit_rc.total_frobenius_norm, rel_tol=1e-5)
    assert math.isclose(rot_rc.total_frobenius_norm, edit_rc.total_frobenius_norm, rel_tol=1e-5)

    ref_ro_12, _ = oracle.rollout(signs, n_passes=12)
    edit_ro_12, _ = edited_tgt.rollout(signs, n_passes=12)
    rand_ro_12, _ = rand_tgt.rollout(signs, n_passes=12)
    rot_ro_12, _ = rot_tgt.rollout(signs, n_passes=12)
    assert torch.allclose(ref_ro_12[11], edit_ro_12[11], atol=1e-5)
    assert float((ref_ro_12[11] - rand_ro_12[11]).abs().max().item()) > 0.5
    assert float((ref_ro_12[11] - rot_ro_12[11]).abs().max().item()) > 0.5


def test_gate_b2_dual_negative_and_positive_control() -> None:
    """Part 1.6: Gate B2 evaluates 8 settling factors alongside a planted positive arm."""
    settler = SettlingCoreLooper(vocab_size=32, d_model=64, n_heads=4, n_core_blocks=4, seed=11)
    toks = torch.randint(0, 32, (24, 12), generator=torch.Generator().manual_seed(11))
    dual = evaluate_hug_cal02_with_positive_arm(
        settler, toks, n_passes=6, n_rotations=16, z_threshold=2.5, seed=11
    )
    assert dual.negative_arm.passed_negative_control is True
    assert dual.negative_margin >= 0.0
    assert dual.positive_arm.passed_positive_control is True
    assert dual.positive_arm.margin > 0.0
    assert dual.passed_gate_b2 is True


def test_planted_rank1_positive_control_on_real_looped_chase_checkpoint() -> None:
    """Part 1.1: Planted rank-1 component on real LoopedChase checkpoint beats registered z_threshold."""
    ckpt_candidate = DEFAULT_LOOPED_CHASE_CKPT_DIR / "lam0.1_s1.final.pt"
    if not ckpt_candidate.exists():
        pytest.skip(f"Real LoopedChase checkpoint not present at {ckpt_candidate}")
    ck_path, ck_sha = resolve_looped_chase_checkpoint("lam0.1_s1")
    lc_model = load_looped_chase_checkpoint(ck_path, device="cpu")
    res = evaluate_planted_rank1_looped_chase(
        lc_model,
        n_passes=8,
        z_threshold=2.5,
        checkpoint_label="lam0.1_s1",
        checkpoint_sha256=ck_sha,
    )
    assert res.passed_positive_control is True
    assert res.observed_z_rot >= 2.5
    assert res.margin > 0.0


def test_transition_operator_and_interventional_path_composition() -> None:
    torch.manual_seed(55)

    class _RelayLooper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = nn.Linear(4, 4, bias=False)
            with torch.no_grad():
                self.core.weight.zero_()
                self.core.weight[1, 0] = 2.0  # Comp 0: reads 0 -> writes 1
                self.core.weight[2, 1] = 2.0  # Comp 1: reads 1 -> writes 2

        def rollout(self, x: Tensor, n_passes: int = 4):
            h = x.clone()
            boundaries = {}
            for tau in range(n_passes):
                h = torch.tanh(h + self.core(h))
                boundaries[tau] = h
            return {}, boundaries

    relay = _RelayLooper()
    bpd = BPDModel(
        relay,
        {"core": 2},
        max_passes=4,
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        init_svd=False,
    )
    with torch.no_grad():
        bpd.wrappers["core"].V.zero_()
        bpd.wrappers["core"].U.zero_()
        bpd.wrappers["core"].V[0, 0] = 1.0
        bpd.wrappers["core"].U[0, 1] = 2.0
        bpd.wrappers["core"].V[1, 1] = 1.0
        bpd.wrappers["core"].U[1, 2] = 2.0

    x0 = torch.zeros(16, 1, 4)
    x0[:, 0, 0] = torch.linspace(-1.0, 1.0, 16)

    def _capture_relay() -> RolloutTrajectory:
        with bpd.target_capture():
            ro, bs = relay.rollout(x0, n_passes=4)
        return RolloutTrajectory(
            readouts=ro, boundary_states=bs, module_inputs=bpd.collect_recorded_inputs(4)
        )

    clean_traj = _capture_relay()

    def _run_with_inputs() -> RolloutTrajectory:
        for w in bpd.wrappers.values():
            w.record_inputs = True
            w.recorded_inputs = []
        ro, bs = relay.rollout(x0, n_passes=4)
        ins = bpd.collect_recorded_inputs(4)
        for w in bpd.wrappers.values():
            w.record_inputs = False
        return RolloutTrajectory(readouts=ro, boundary_states=bs, module_inputs=ins)

    op = compute_transition_operator(bpd, clean_traj, _run_with_inputs, ablation_mode="ablate")
    # Comp 0 (writes coord 1) strongly drives Comp 1 (reads coord 1) on the next pass!
    assert float(op.matrix[0, 1].item()) > float(op.matrix[1, 0].item()) + 0.1

    # Compose 2-step path (core, 0) at pass 0 -> (core, 1) at pass 1 -> (core, 1) at pass 2
    path = [
        ComponentKey("core", 0),
        ComponentKey("core", 1),
        ComponentKey("core", 1),
    ]
    res = compose_interventional_path(bpd, clean_traj, _run_with_inputs, op, path, start_pass=0)
    assert res.direct_source_effect > 0.01
    assert res.interventional_mediated_effect >= 0.0


def test_trm_and_looped_chase_end_to_end_smokes() -> None:
    ref = build_mini_trm(hidden_size=32, num_heads=4, l_layers=1, h_cycles=1, l_cycles=1, seq_len=9, seed=1)
    tgt = copy.deepcopy(ref)
    gen = torch.Generator().manual_seed(2)
    t_in = torch.randint(1, 11, (6, ref.seq_len), generator=gen)
    e_in = torch.randint(1, 11, (6, ref.seq_len), generator=gen)
    e_tgt = torch.randint(0, 9, (6, ref.seq_len), generator=gen)

    trm_res = run_trm_bpd_pipeline(
        ref,
        tgt,
        t_in,
        e_in,
        e_tgt,
        train_outer_steps=2,
        eval_horizons=(2, 4),
        c_per_mod=4,
        n_steps=6,
        pgd_steps_grid=(0, 2),
        switch_sweep=(0.0, 2e-4),
        top_k_edit=2,
        device="cpu",
        seed=0,
    )
    assert trm_res["total_invocations_trained"] == 4
    assert sum(trm_res["rollout_mode_counts"].values()) == 6  # type: ignore[arg-type]
    modes = trm_res["trajectory_state_matching_modes"]  # type: ignore[assignment]
    assert "outer_readout_only_default" in modes  # type: ignore[operator]
    assert "with_inner_boundary_states" in modes  # type: ignore[operator]
    edit_sec = trm_res["sustaining_edit_test"]  # type: ignore[assignment]
    assert "final_solve_rate" in edit_sec["intra_checkpoint_sustaining_edit"]["metrics"]  # type: ignore[index]
    assert "ever_solved_rate" in edit_sec["intra_checkpoint_sustaining_edit"]["metrics"]  # type: ignore[index]
    assert "matched_random_component_control" in edit_sec  # type: ignore[operator]
    assert "rotation_null_control" in edit_sec  # type: ignore[operator]

    mini_chase = LoopedChase(n_nodes=6, d=24, layers=1, heads=4)
    chase_res = run_looped_chase_experiment(
        mini_chase,
        label="unit_test_chase",
        n_passes=3,
        c_per_mod=4,
        n_steps=6,
        ci_head_steps=5,
        batch_size=8,
        device="cpu",
        seed=0,
    )
    assert "clock_gap_recon" in chase_res["gaps"]  # type: ignore[operator]
    assert "visible_gap_recon" in chase_res["gaps"]  # type: ignore[operator]


def test_compose_interventional_path_wires_perturbed_inputs_when_rollout_fn_is_stale() -> None:
    """Regression test for v1 bug where stale traj.module_inputs yielded exact 0.0 effects."""
    torch.manual_seed(91)

    class _TwoStepChain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = nn.Linear(4, 4, bias=False)
            with torch.no_grad():
                self.core.weight.zero_()
                self.core.weight[1, 0] = 2.0
                self.core.weight[2, 1] = 2.0
                self.core.weight[3, 2] = 1.5

        def rollout(self, x: Tensor, n_passes: int = 4):
            h = x.clone()
            boundaries = {}
            for tau in range(n_passes):
                h = torch.tanh(h + self.core(h))
                boundaries[tau] = h
            return {}, boundaries

    chain_mod = _TwoStepChain()
    bpd = BPDModel(
        chain_mod,
        {"core": 3},
        max_passes=4,
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        init_svd=False,
    )
    with torch.no_grad():
        bpd.wrappers["core"].V.zero_()
        bpd.wrappers["core"].U.zero_()
        bpd.wrappers["core"].V[0, 0] = 1.0
        bpd.wrappers["core"].U[0, 1] = 2.0
        bpd.wrappers["core"].V[1, 1] = 1.0
        bpd.wrappers["core"].U[1, 2] = 2.0
        bpd.wrappers["core"].V[2, 2] = 1.0
        bpd.wrappers["core"].U[2, 3] = 1.5

    x0 = torch.zeros(12, 1, 4)
    x0[:, 0, 0] = torch.linspace(-1.0, 1.0, 12)
    with bpd.target_capture():
        ro, bs = chain_mod.rollout(x0, n_passes=4)
    clean_traj = RolloutTrajectory(
        readouts=ro, boundary_states=bs, module_inputs=bpd.collect_recorded_inputs(4)
    )

    # Deliberately return stale clean_traj.module_inputs (exact v1 _ref_plain_rollout pattern!)
    def _stale_plain_rollout() -> RolloutTrajectory:
        p_ro, p_bs = chain_mod.rollout(x0, n_passes=4)
        return RolloutTrajectory(
            readouts=p_ro, boundary_states=p_bs, module_inputs=clean_traj.module_inputs
        )

    op = compute_transition_operator(bpd, clean_traj, _stale_plain_rollout, ablation_mode="ablate")
    path = [ComponentKey("core", 0), ComponentKey("core", 1), ComponentKey("core", 2)]
    res = compose_interventional_path(bpd, clean_traj, _stale_plain_rollout, op, path, start_pass=0)
    assert res.direct_source_effect > 0.05
    assert res.interventional_mediated_effect > 0.05


def test_nested_state_plus_clock_warm_start_exact_zero_gap_at_step_zero() -> None:
    """state_plus_clock warm-started from state_only has identical outputs at step 0."""
    from mechinterp.bpd.ci import TiedCINetwork
    from mechinterp.bpd.config import CIConfig

    torch.manual_seed(77)
    mod_dims = {"m0": 12, "m1": 8}
    c_map = {"m0": 4, "m1": 3}
    state_net = TiedCINetwork(
        mod_dims, c_map, CIConfig(variant="state_only", d_model=32, n_blocks=1, n_heads=4)
    )
    clock_net = TiedCINetwork(
        mod_dims, c_map, CIConfig(variant="state_plus_clock", d_model=32, n_blocks=1, n_heads=4)
    )
    clock_net.load_nested_state_only(state_net)

    mod_in = {"m0": torch.randn(3, 5, 6, 12), "m1": torch.randn(3, 5, 6, 8)}
    out_s = state_net(mod_in)
    out_c = clock_net(mod_in)
    for m in c_map:
        assert torch.allclose(out_s.pre_sigmoid[m], out_c.pre_sigmoid[m], atol=1e-6)
        assert torch.allclose(out_s.lower_leaky[m], out_c.lower_leaky[m], atol=1e-6)


def test_v2_strict_tgt_sustaining_unmatched_similarity_mcnemar_and_rank_p() -> None:
    """Verify strict tgt_net_sustaining > 0 filter, unmatched baseline, McNemar, and rank p."""
    from mechinterp.bpd.analysis import (
        ComponentFunctionalProfile,
        compute_mcnemar_discordance,
        compute_unmatched_similarity_baseline,
        summarize_control_draws,
    )
    from mechinterp.bpd.nulls import compute_rank_p_values

    ref_profs = [
        ComponentFunctionalProfile(
            module="m0",
            component=0,
            role="sustaining",
            mean_schedule=torch.tensor([1.0, 1.0, 1.0, 0.2]),
            duty_cycle=0.8,
            switch_rate=0.2,
            transition_signature=torch.tensor([0.9, 0.1]),
            readout_effect_curve=torch.tensor([0.1, 0.5]),
            raw_all_pass_final=0.6,
            net_all_pass_final=0.45,
            sustain_amplification=3.0,
        ),
        ComponentFunctionalProfile(
            module="m0",
            component=1,
            role="transient",
            mean_schedule=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            duty_cycle=0.25,
            switch_rate=0.33,
            transition_signature=torch.tensor([0.1, 0.9]),
            readout_effect_curve=torch.tensor([0.4, 0.01]),
            raw_all_pass_final=0.02,
            net_all_pass_final=-0.01,
            sustain_amplification=0.5,
        ),
    ]
    tgt_profs = [
        # Identical schedule/profile to ref comp 0, BUT negative net_all_pass_final (-0.12)!
        ComponentFunctionalProfile(
            module="m0",
            component=0,
            role="inert",
            mean_schedule=torch.tensor([1.0, 1.0, 1.0, 0.2]),
            duty_cycle=0.8,
            switch_rate=0.2,
            transition_signature=torch.tensor([0.9, 0.1]),
            readout_effect_curve=torch.tensor([0.1, 0.5]),
            raw_all_pass_final=0.05,
            net_all_pass_final=-0.12,
            sustain_amplification=0.8,
        ),
        # Positive sustaining target counterpart
        ComponentFunctionalProfile(
            module="m0",
            component=1,
            role="sustaining",
            mean_schedule=torch.tensor([0.95, 0.95, 0.9, 0.25]),
            duty_cycle=0.75,
            switch_rate=0.2,
            transition_signature=torch.tensor([0.85, 0.15]),
            readout_effect_curve=torch.tensor([0.12, 0.48]),
            raw_all_pass_final=0.4,
            net_all_pass_final=0.22,
            sustain_amplification=2.5,
        ),
    ]
    matches = match_components_by_function(
        ref_profs, tgt_profs, require_ref_sustaining=True, require_tgt_sustaining=True
    )
    assert len(matches) == 1
    assert matches[0].tgt_component == 1  # Strictly rejected component 0 with net = -0.12!
    assert matches[0].tgt_net_sustaining > 0.0

    sim_base = compute_unmatched_similarity_baseline(
        ref_profs, tgt_profs, matches, n_permutations=49, seed=0
    )
    assert sim_base.num_matched_pairs == 1
    assert sim_base.num_unmatched_pairs == 3

    mc = compute_mcnemar_discordance(
        [True, True, True, True, True, True, False, False],
        [False, False, False, False, False, False, False, False],
    )
    assert mc.n_a_only == 6 and mc.n_b_only == 0
    assert math.isclose(mc.exact_two_sided_p, 2.0 * (0.5 ** 6), rel_tol=1e-6)

    dist = summarize_control_draws(0.80, [0.50 + 0.01 * i for i in range(19)], higher_is_better=True)
    assert dist.exceed_or_equal_count == 0
    assert math.isclose(dist.empirical_p_rank, 1.0 / 20.0, rel_tol=1e-6)

    exc, p_up, p_two = compute_rank_p_values(5.0, torch.linspace(-1.0, 1.0, 199))
    assert exc == 0
    assert math.isclose(p_up, 1.0 / 200.0, rel_tol=1e-6)
    assert math.isclose(p_two, 2.0 / 200.0, rel_tol=1e-6)


def test_trm_activation_checkpointing_exact_forward_and_gradient_parity() -> None:
    """Verify call_idx-pinned checkpointing matches uncheckpointed forward & (V, U) grads."""
    from mechinterp.bpd.adapters.trm import TRMBPDAdapter, default_trm_c_per_module
    from mechinterp.bpd.components import PassMaskSchedule
    from mechinterp.bpd.losses import trajectory_recon_loss

    base_a = build_mini_trm(hidden_size=32, num_heads=4, l_layers=2, h_cycles=1, l_cycles=2, seq_len=9, seed=12)
    base_b = copy.deepcopy(base_a)
    c_map = default_trm_c_per_module(base_a, c=8)
    ad_a = TRMBPDAdapter(base_a, use_activation_checkpointing=False)
    ad_b = TRMBPDAdapter(base_b, use_activation_checkpointing=True)
    n_inv = ad_a.invocations_for_outer_steps(2)

    bpd_a = BPDModel(base_a, c_map, max_passes=n_inv, init_svd=True)
    bpd_b = BPDModel(base_b, c_map, max_passes=n_inv, init_svd=True)

    inp = torch.randint(1, 11, (4, base_a.seq_len), generator=torch.Generator().manual_seed(3))
    batch_a = ad_a.make_batch_with_clean_cache(inp, n_inv)
    batch_b = ad_b.make_batch_with_clean_cache(inp, n_inv)

    gen = torch.Generator().manual_seed(4)
    sched_a = {}
    sched_b = {}
    for m in bpd_a.module_order:
        cm = torch.rand(4, n_inv, base_a.total_len, 8, generator=gen)
        dm = torch.rand(4, n_inv, base_a.total_len, generator=gen)
        sched_a[m] = PassMaskSchedule(comp_mask=cm.clone(), delta_mask=dm.clone())
        sched_b[m] = PassMaskSchedule(comp_mask=cm.clone(), delta_mask=dm.clone())

    with torch.no_grad():
        ro_clean, bs_clean, _, _ = ad_a.rollout(batch_a, n_inv, 0, None)
    clean_t = RolloutTrajectory(readouts=ro_clean, boundary_states=bs_clean, module_inputs={})

    with bpd_a.masked_components(sched_a):
        ro_a, bs_a, _, _ = ad_a.rollout(batch_a, n_inv, 0, None)
    loss_a, _ = trajectory_recon_loss(
        RolloutTrajectory(readouts=ro_a, boundary_states=bs_a, module_inputs={}), clean_t
    )
    loss_a.backward()

    with bpd_b.masked_components(sched_b):
        ro_b, bs_b, _, _ = ad_b.rollout(batch_b, n_inv, 0, None)
    loss_b, _ = trajectory_recon_loss(
        RolloutTrajectory(readouts=ro_b, boundary_states=bs_b, module_inputs={}), clean_t
    )
    loss_b.backward()

    assert torch.allclose(loss_a, loss_b, atol=1e-5)
    for m in bpd_a.module_order:
        assert torch.allclose(bpd_a.wrappers[m].V.grad, bpd_b.wrappers[m].V.grad, atol=1e-5)  # type: ignore[arg-type]
        assert torch.allclose(bpd_a.wrappers[m].U.grad, bpd_b.wrappers[m].U.grad, atol=1e-5)  # type: ignore[arg-type]
