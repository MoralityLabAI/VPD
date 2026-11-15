"""End-to-end tests for BPD's falsifiable gates (B1-B4), transient vs. sustaining
classification, interventional transition composition, sustaining weight edits,
and CLI smoke pipelines on PureSudokuTRM and LoopedChase.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from mechinterp.bpd.adapters.trm import build_mini_trm
from mechinterp.bpd.analysis import (
    ComponentKey,
    apply_sustaining_weight_edit,
    classify_transient_sustaining,
    compose_interventional_path,
    compute_transition_operator,
    rank_sustaining_checkpoint_diffs,
)
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


def test_transient_vs_sustaining_and_sustaining_weight_edit() -> None:
    oracle = _CanonicalRolesLooper()
    bpd = BPDModel(
        oracle,
        {"core": 3},
        max_passes=4,
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        init_svd=False,
    )
    # Set exact 3 rank-1 factors matching the 3 planted mechanisms
    with torch.no_grad():
        bpd.wrappers["core"].V.zero_()
        bpd.wrappers["core"].U.zero_()
        # Comp 0: transient (reads 2, writes 0, weight 4.0)
        bpd.wrappers["core"].V[2, 0] = 2.0
        bpd.wrappers["core"].U[0, 0] = 2.0
        # Comp 1: sustaining (reads 3, writes 1, weight 2.5)
        bpd.wrappers["core"].V[3, 1] = 1.0
        bpd.wrappers["core"].U[1, 1] = 2.5
        # Comp 2: inert (reads 4, writes 5, weight 0.5)
        bpd.wrappers["core"].V[4, 2] = 1.0
        bpd.wrappers["core"].U[2, 5] = 0.5

    signs = torch.tensor([1.0, -1.0] * 12)
    with bpd.target_capture():
        ro, bs = oracle.rollout(signs, n_passes=4)
    mod_in = bpd.collect_recorded_inputs(expected_passes=4)
    clean_traj = RolloutTrajectory(readouts=ro, boundary_states=bs, module_inputs=mod_in)

    def _run_oracle() -> RolloutTrajectory:
        p_ro, p_bs = oracle.rollout(signs, n_passes=4)
        return RolloutTrajectory(readouts=p_ro, boundary_states=p_bs, module_inputs=mod_in)

    roles = classify_transient_sustaining(
        bpd,
        clean_traj,
        _run_oracle,
        min_effect=0.02,
        sustain_ratio=2.5,
        n_rotations=4,
        ablation_mode="ablate",
        seed=0,
    )
    role_by_c = {e.component: e.role for e in roles}
    assert role_by_c[0] == "transient"
    assert role_by_c[1] == "sustaining"
    assert role_by_c[2] == "inert"

    # Now test Gate B4's sustaining weight-edit mechanism on a degraded target checkpoint
    degraded = _CanonicalRolesLooper()
    with torch.no_grad():
        # Weaken the sustaining mechanism in degraded checkpoint -> collapses long-horizon margin
        degraded.core.weight[1, 3] = 0.2

    diffs = rank_sustaining_checkpoint_diffs(bpd, degraded, roles)
    assert len(diffs) == 1
    assert diffs[0].component == 1
    assert abs(diffs[0].coeff_gap - (1.0 - 0.2 / 2.5)) < 1e-4

    edited_counts = apply_sustaining_weight_edit(degraded, bpd, diffs, interpolation=1.0)
    assert edited_counts == {"core": 1}

    # Verify edited checkpoint recovers reference readout at long horizon T = 12 without hooks
    ref_ro_12, _ = oracle.rollout(signs, n_passes=12)
    edit_ro_12, _ = degraded.rollout(signs, n_passes=12)
    assert torch.allclose(ref_ro_12[11], edit_ro_12[11], atol=1e-5)


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
    assert "sustaining_edit_test" in trm_res

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
