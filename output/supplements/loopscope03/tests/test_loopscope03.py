"""Unit and receipt-invariant tests for LOOPSCOPE-03 Three-Ledger Iteration Pricing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mechinterp.loopscope03.scripts import conditional_refill as CR              # noqa: E402
from mechinterp.loopscope03.scripts import path_independence_and_scaling as PIS  # noqa: E402
from mechinterp.loopscope03.scripts import resume_and_fork as RF                 # noqa: E402

RESULTS_DIR = REPO_ROOT / "mechinterp" / "loopscope03" / "results"


def test_head_projector_and_sphere_calibration_geometry():
    torch.manual_seed(42)
    w = torch.randn(11, 512, dtype=torch.float32)
    v_head, p_head, p_perp = CR.compute_head_projector_f64(w)
    assert v_head.shape == (512, 11)
    # Orthonormality & orthogonality
    ident = v_head.T @ v_head
    assert float((ident - torch.eye(11)).abs().max().item()) < 1e-5
    assert float((w @ p_perp).abs().max().item()) < 1e-5

    # Sphere calibration preserves exact visible logits while setting ||v_proj + orth||_2 = sqrt(512)
    z_clean = CR.rms_calibrate_vec(torch.randn(4, 81, 512))
    v_coords = z_clean @ v_head
    v_proj = v_coords @ v_head.T
    rand_raw = torch.randn(4, 81, 512)
    orth_cal = CR.sphere_calibrate_orth(rand_raw, v_proj, p_perp)
    z_rebuilt = v_proj + orth_cal

    norms_sq = z_rebuilt.pow(2).sum(dim=-1)
    assert float((norms_sq - 512.0).abs().max().item()) < 2e-3
    logit_diff = float(((z_clean @ w.T) - (z_rebuilt @ w.T)).abs().max().item())
    assert logit_diff < 1e-4


def test_lockin_step_and_telescoping_identity():
    # Shape [U+1, N]: U=6
    exact = np.array(
        [
            [False, True, False, False],
            [False, False, False, False],  # board 1 destabilized at u=1, re-locks at u=2!
            [True, True, False, False],
            [True, True, True, False],
            [False, True, True, False],    # board 0 flipped out at u=4, re-locks at u=5!
            [True, True, True, False],
            [True, True, True, False],
        ],
        dtype=bool,
    )
    tau_lock = RF.compute_lockin_step(exact)
    tau_first = RF.compute_first_solve_step(exact)
    assert tau_lock.tolist() == [5, 2, 3, 7]
    assert tau_first.tolist() == [2, 0, 3, 7]


def test_stage1_conditional_refill_receipt_invariants():
    p = RESULTS_DIR / "stage1_conditional_refill.json"
    assert p.exists()
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["dataset_splits"]["overlap_with_test"] == 0
    for ck_key in ("primary_gaoxin492", "replication_yagizdevre"):
        ck = d[ck_key]
        par = ck["zero_leak_and_logit_parity"]
        assert par["max_f32_logit_diff"] < 1e-4
        assert par["max_hidden_perturbation_leak"] < 1e-4
        r2 = ck["val_100_r2_by_capacity"]
        # Transformer refill must explain substantially positive out-of-sample hidden variance
        assert r2["C5_tf_2L_512"]["all"]["r2_joint_hidden"] > r2["C1_linear_local"]["all"]["r2_joint_hidden"]


def test_stage2_3_resume_and_fork_receipt_invariants():
    p = RESULTS_DIR / "stage2_3_resume_and_fork.json"
    assert p.exists()
    d = json.loads(p.read_text(encoding="utf-8"))
    gate = d["stationarity_admission_gate"]
    assert gate["Nanbeige4.2-3B"]["admitted"] is False
    assert gate["TRM_gaoxin492"]["admitted"] is True
    assert gate["TRM_gaoxin492"]["state_transplant_linf_diff"] < 1e-5

    for ck_key in ("trm_primary_gaoxin492", "trm_replication_yagizdevre"):
        ck = d[ck_key]
        for split_key in ("resume_items256", "resume_bench100"):
            for step_rec in ck[split_key]["by_transplant_step"].values():
                assert step_rec["max_telescope_error"] < 1e-9
        for cv_key in ("cvoting_and_divergence_items256", "cvoting_and_divergence_bench100"):
            for sig_rec in ck[cv_key].values():
                assert abs(sig_rec["asymptotic_solved_pair_agreement"] - 1.0) < 1e-9


def test_huginn_cache_and_fork_receipt_invariants():
    p = RESULTS_DIR / "huginn_cache_and_fork.json"
    assert p.exists()
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["stationarity_transplant_linf_diff"] < 1e-5
    pv = d["path_dependent_variance_share_upper_bound_by_step"]
    assert pv["0"] > 0.95
    assert pv["8"] < 0.05
    assert pv["32"] < 0.005
    res = d["resume_to_step32_steady_token_at_t4"]
    assert abs((res["mean_tau_fresh"] - res["mean_tau_full"]) - (res["accrued_cache_loops"] + res["price_R_loops"])) < 1e-9


def test_digit_relabel_canonical_isomorphism():
    q1 = "9..1..4..005008002" + "." * 63
    q2 = "3..7..2..004006009" + "." * 63
    q3 = "3..7..3..004006009" + "." * 63
    assert PIS.digit_relabel_canonical(q1) == PIS.digit_relabel_canonical(q2)
    assert PIS.digit_relabel_canonical(q1) != PIS.digit_relabel_canonical(q3)


def test_stage4_scaling_uncapped_resume_and_pi_invariants():
    p = RESULTS_DIR / "stage4_path_independence_and_scaling.json"
    assert p.exists()
    d = json.loads(p.read_text(encoding="utf-8"))
    pool = d["disjoint_train_pool"]
    assert pool["total_pool_n"] == 8100
    assert all(v == 0 for v in pool["exact_overlaps_with_splits"].values())

    sc = d["primary_gaoxin492"]["data_scaling_curve_c5"]
    for n_key in ("N_100", "N_300", "N_900", "N_2700", "N_8100"):
        assert n_key in sc
    # Out-of-sample R^2 must strictly improve from N=100 to N=8100
    assert sc["N_8100"]["val_100_r2"]["all"]["r2_joint_hidden"] > sc["N_100"]["val_100_r2"]["all"]["r2_joint_hidden"]

    for ck_key in ("primary_gaoxin492", "replication_yagizdevre"):
        ck = d[ck_key]
        res = ck["uncapped_resume_items256_n8100"]
        for st_rec in res.values():
            st_coh = st_rec["cohorts"]["unselected_wb_stall"]
            assert st_coh["n"] == 183
            eff = st_coh["R_effect_VC_vs_Full"]
            disc = eff["discordance"]
            assert (
                disc["both_lock"] + disc["ref_b_only_locks"] + disc["arm_a_only_locks"] + disc["neither_locks"]
                == 183
            )
            assert eff["uncapped_delay_among_both_lock"]["n_both"] == disc["both_lock"]

        pi_rec = ck["unconditional_and_enforced_path_independence_items256"]["sigma_0.35"]
        for arm_name in ("clean", "vc_c5_n8100"):
            st_sum = pi_rec[arm_name]["wb_stall"]
            assert 0.0 <= st_sum["uncond_pi_board_s64"] <= 1.0
            assert 0.0 <= st_sum["uncond_pi_cell_s64"] <= 1.0
            # Exact board agreement decomposes into solved-pair + same-unsolved-trap-pair agreement
            assert (
                abs(
                    st_sum["uncond_pi_board_s64"]
                    - (st_sum["pair_both_solved_s64"] + st_sum["pair_same_unsolved_trap_s64"])
                )
                < 1e-9
            )
