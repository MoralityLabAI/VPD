"""Unit and mathematical-invariant tests for LOOPSCOPE-02."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from mechinterp.loopscope02.scripts import llm_sudoku as LLM
from mechinterp.loopscope02.scripts import score_loopscope02 as SC
from mechinterp.loopscope02.scripts import sudoku_prop as SP
from mechinterp.loopscope02.scripts import trm_sudoku as TRM

SOLVED_GRID = (
    "534678912"
    "672195348"
    "198342567"
    "859761423"
    "426853791"
    "713924856"
    "961537284"
    "287419635"
    "345286179"
)


def test_sudoku_prop_lattice_and_trace():
    # Remove 1 cell -> solved in 1 naked-single round
    g_list = list(SOLVED_GRID)
    g_list[0] = "."
    q1 = "".join(g_list)
    tr1 = SP.trace_propagation(q1)
    assert tr1.solved is True
    assert tr1.num_rounds == 1
    assert tr1.naked_round1[0] is True
    assert tr1.cell_round[0] == 1

    # Construct a 2-round chain: empty an entire row's 2 cells plus peers so cell A is round 1 and cell B is round 2
    # Classic easy puzzle solvable by propagation in multiple rounds:
    easy_q = (
        "53..7...."
        "6..195..."
        ".98....6."
        "8...6...3"
        "4..8.3..1"
        "7...2...6"
        ".6....28."
        "...419..5"
        "....8..79"
    )
    tr_easy = SP.trace_propagation(easy_q)
    assert tr_easy.solved is True
    assert tr_easy.num_rounds >= 3
    assert tr_easy.rounds_grids[-1] == tuple(int(c) for c in SOLVED_GRID)
    assert any(r == 1 for r in tr_easy.cell_round)
    assert any(r == 2 for r in tr_easy.cell_round)
    assert any(r >= 3 for r in tr_easy.cell_round)

    # Empty grid is stalled at round 0 (zero singles)
    tr_empty = SP.trace_propagation("." * 81)
    assert tr_empty.solved is False
    assert tr_empty.num_rounds == 0
    assert all(r == -1 for r in tr_empty.cell_round)

    # g_prop_belief returns valid distributions summing to 1 per cell
    prob_u = np.full((81, 9), 1.0 / 9.0, dtype=np.float32)
    gp = SP.g_prop_belief(prob_u, SP.parse_grid(q1))
    assert gp.shape == (81, 9)
    np.testing.assert_allclose(gp.sum(axis=-1), np.ones(81), atol=1e-5)
    assert gp[0, int(SOLVED_GRID[0]) - 1] == pytest.approx(1.0)


def test_prompt_and_tokenizer_single_digit_boundary():
    q = "." + SOLVED_GRID[1:]
    shot = {"question": q, "answer": SOLVED_GRID, "target_idx": 0}
    prompt = SP.format_target_prompt([shot, shot, shot], q, 0)
    assert prompt.endswith("Answer: ")

    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    for profile, spec in LLM.MODEL_LOCKS.items():
        home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        root = home / "hub" / ("models--" + spec["repo"].replace("/", "--")) / "snapshots" / spec["revision"]
        if not root.exists():
            pytest.skip(f"Cached snapshot not present for {profile}")
        if profile == "nanbeige":
            tok_raw = PreTrainedTokenizerFast(tokenizer_file=str(root / "tokenizer.json"))
            bos = spec["bos_token_id"]
            enc = lambda txt, tr=tok_raw, b=bos: [b] + tr.encode(txt, add_special_tokens=False)
        else:
            tok = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
            enc = lambda txt, tk=tok: tk.encode(txt, add_special_tokens=False)
        d_ids = LLM.verify_and_resolve_digit_tokens(enc, prompt)
        assert len(d_ids) == 9
        assert len(set(d_ids)) == 9


def test_pure_sudoku_trm_fiber_projection_and_tca_exactness():
    torch.manual_seed(1234)
    model = TRM.PureSudokuTRM(
        hidden_size=32, num_heads=4, l_layers=2, h_cycles=2, l_cycles=2, puzzle_emb_len=4, seq_len=81
    ).float().eval()
    with torch.no_grad():
        model.H_init.normal_(0, 0.1)
        model.L_init.normal_(0, 0.1)
        model.puzzle_weight.normal_(0, 0.1)

    q1 = "." + SOLVED_GRID[1:]
    q2 = SOLVED_GRID[:2] + "." + SOLVED_GRID[3:]

    # 1. Verify lm_head orthogonal subspace projection preserves digit_logits identically
    inp = TRM.encode_questions([q1, q2], "cpu")
    with torch.no_grad():
        x = model.encode_inputs(inp)
        zh, zl = model.init_states(2)
        zh, zl = model.act_step(zh, zl, x)
        orig_logits = model.digit_logits(zh)

        w_head = model.lm_head.weight.detach().float()
        _, _, vh = torch.linalg.svd(w_head, full_matrices=False)
        p_head = vh.T @ vh
        h_init_perp = (model.H_init.float() @ (torch.eye(32) - p_head)).view(1, 1, -1)
        board_proj = (zh[:, 4:, :] @ p_head) + h_init_perp
        zh_proj = torch.cat([zh[:, :4, :], board_proj], dim=1)
        proj_logits = model.digit_logits(zh_proj)
        assert float((orig_logits - proj_logits).abs().max().item()) < 1e-5
        # Orthogonal residual is non-zero before projection and matches H_init_perp after
        ortho_diff = ((zh_proj[:, 4:, :] - h_init_perp) @ (torch.eye(32) - p_head)).abs().max().item()
        assert ortho_diff < 1e-5

    # 2. Verify TCA first-order Delta matches exact finite-difference weight rank-1 perturbation
    facs = TRM.svd_factors(model, top_k=8)
    u_mat, s_vec, v_mat = facs["L0.down_proj"]

    def margin_val() -> float:
        x_e = model.encode_inputs(inp[:1])
        z_h, z_l = model.init_states(1)
        for _ in range(2):
            z_h, z_l = model.act_step(z_h, z_l, x_e)
        lg = model.digit_logits(z_h)[0, 0]
        td = int(SOLVED_GRID[0]) - 1
        zc = lg[td]
        oth = lg.clone()
        oth[td] = float("-inf")
        return float((zc - torch.logsumexp(oth, dim=0)).item())

    # Differentiate across ALL calls of L0.down_proj to test exact chain rule + closed-form rotation SD
    model.zero_grad(set_to_none=True)
    xs, gs = [], []

    def hk(_m, i_t, o_t):
        xs.append(i_t[0].detach())
        o_t.register_hook(lambda g: gs.append(g.detach()))

    h_hook = model.layers[0].down_proj.register_forward_hook(hk)
    try:
        x_e = model.encode_inputs(inp[:1])
        z_h, z_l = model.init_states(1)
        for _ in range(2):
            z_h, z_l = model.act_step(z_h, z_l, x_e)
        lg = model.digit_logits(z_h)[0, 0]
        td = int(SOLVED_GRID[0]) - 1
        zc = lg[td]
        oth = lg.clone()
        oth[td] = float("-inf")
        (zc - torch.logsumexp(oth, dim=0)).backward()
    finally:
        h_hook.remove()

    #Note: hooks fire in reverse during backward, so reverse gs to align with xs
    gs = list(reversed(gs))
    w_grad = model.layers[0].down_proj.weight.grad.detach().clone()
    pred_deltas = []
    grad_deltas = []
    fd_deltas = []
    eps = 2e-3
    w_orig = model.layers[0].down_proj.weight.detach().clone()
    for r in range(6):
        vr = v_mat[:, r]
        ur = u_mat[:, r]
        sr = float(s_vec[r].item())
        d_pred = 0.0
        for xb, ab in zip(xs, gs):
            d_pred += float((-sr * ((xb @ vr) * (ab @ ur)).sum()).item())
        pred_deltas.append(d_pred)
        grad_deltas.append(float((-sr * (w_grad * torch.outer(ur, vr)).sum()).item()))

        with torch.no_grad():
            model.layers[0].down_proj.weight.copy_(w_orig - eps * sr * torch.outer(ur, vr))
            m_minus = margin_val()
            model.layers[0].down_proj.weight.copy_(w_orig + eps * sr * torch.outer(ur, vr))
            m_plus = margin_val()
            model.layers[0].down_proj.weight.copy_(w_orig)
        fd_deltas.append((m_minus - m_plus) / (2.0 * eps))

    # Exact chain-rule identity against parameter gradient + float32 central finite differences
    np.testing.assert_allclose(pred_deltas, grad_deltas, rtol=1e-5, atol=1e-6)
    assert np.corrcoef(pred_deltas, fd_deltas)[0, 1] > 0.9999
    np.testing.assert_allclose(pred_deltas, fd_deltas, rtol=2e-2, atol=5e-4)

    # 3. Verify closed-form rotation SD matches Monte Carlo random read-vector rotation
    xb0, ab0 = xs[-1], gs[-1]
    din = v_mat.shape[0]
    au0 = ab0 @ u_mat[:, 0]
    xtau0 = torch.einsum("bpi,bp->bi", xb0, au0)
    sd_closed = float(s_vec[0].item()) * float(xtau0.norm().item()) / math.sqrt(din)
    rng = torch.Generator().manual_seed(99)
    mc = []
    for _ in range(2000):
        rv = torch.randn(din, generator=rng)
        rv = rv / rv.norm()
        mc.append(float((-s_vec[0].item() * ((xb0 @ rv) * au0).sum()).item()))
    assert abs(np.std(mc) - sd_closed) / max(sd_closed, 1e-8) < 0.05

    # 4. End-to-end trajectory + compactification helper smoke check
    traj = TRM.evaluate_trajectories_and_usage(
        model, [q1, q2], [SOLVED_GRID, SOLVED_GRID], [1, 0], n_steps=3, batch_size=2, record_usage=True
    )
    comp = TRM.compute_compactification_and_revisions(
        traj["probs_steps"], [q1, q2], [SOLVED_GRID, SOLVED_GRID], [True, True], probs_hcycles=traj["probs_hcycles"]
    )
    assert len(comp["exact_solve_curve_all"]) == 3
    tca_out = TRM.run_trm_tca(model, [q1, q2], [SOLVED_GRID, SOLVED_GRID], [0, 2], unroll_steps=2, chunk_size=2)
    assert len(tca_out["f1_over_flate_by_matrix"]) == 4


def test_qwen35_tiny_cpu_suite_hooks_and_tca(monkeypatch):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(7)
    cfg = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
    )
    model = Qwen3_5ForCausalLM(cfg).eval()
    model.requires_grad_(False)
    monkeypatch.setitem(LLM.MODEL_LOCKS["qwen4b"], "tca_layers", (0, 1, 2, 3))

    def fake_enc(txt: str) -> list[int]:
        base = [10, 11, 12, 13]
        if txt and txt[-1] in "123456789":
            return base + [20 + int(txt[-1])]
        return base

    q = "." + SOLVED_GRID[1:]
    shot = {"question": q, "answer": SOLVED_GRID, "target_idx": 0}
    items = [
        {"item_id": f"i{k}", "stratum": s, "question": q, "answer": SOLVED_GRID,
         "target_idx": 0, "target_digit": int(SOLVED_GRID[0]), "donor_item_id": f"i{(k+1)%4}"}
        for k, s in enumerate(SP.STRATA)
    ]
    bundle = {"bundle_sha256": "deadbeef", "shots": [shot, shot, shot], "items": items}
    res = LLM.run_qwen_suite("qwen4b", model, fake_enc, bundle, min_watts=0.0)
    assert len(res["items_target_correct"]) == 4
    assert len(res["tca_s1"]["zr_gt3_share"]) == 4


def test_score_loopscope02_determinism_and_mcnemar():
    assert SC.exact_mcnemar_p([1] * 10, [0] * 10) == pytest.approx(2.0 * (0.5 ** 10))
    assert SC.exact_mcnemar_p([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(1.0)
    assert SC.exact_mcnemar_p([1, 1, 0], [1, 1, 0]) == pytest.approx(1.0)

    n = 16
    items = [
        {"item_id": f"it{i}", "stratum": SP.STRATA[i % 4]}
        for i in range(n)
    ]
    bundle = {"bundle_sha256": "abc123", "items": items}
    trm = {
        "bundle_sha256": "abc123",
        "items_target_correct_by_step": [[0] * n, [1] * n],
        "items_target_open_correct": [0] * n,
        "items_target_donor_correct": [0] * n,
        "items_board_exact_by_step": [[0] * n, [1] * n],
        "items_board_exact_corrupt_step1": [0] * n,
        "items_board_exact_corrupt_repaired": [1] * n,
        "items_target_corrupt_step1_correct": [0] * n,
        "items_target_corrupt_repaired_correct": [1] * n,
        "items_board_exact_reset_zL": [0] * n,
        "items_board_exact_fiber_erased": [0] * n,
        "bench100_clean16_exact": [1] * 20,
        "bench100_reset_zL_exact": [0] * 20,
        "bench100_fiber_erased_exact": [0] * 20,
        "kick_ratio_curve_mean": [1.0, 0.4],
        "usage_stats": {"L0.o_proj": {"r_outer_zH_mean": 0.92, "r_outer_zL_mean": 0.90, "r_role_zL_vs_zH_mean": 0.55}},
        "tca": {"f1_over_flate_mean": 0.08, "zr_gt3_overall": 0.04, "zw_gt3_overall": 0.03},
        "replication_bench100_exact_step16": 0.75,
        "bench100_compactification": {
            "exact_solve_curve_all": [0.05, 0.78],
            "commutation_vs_shuffled_auc_prop": 0.98,
            "stalled_solve_rate_step16": 0.62,
            "stalled_solved_with_nonmonotone_flips_share": 0.95,
        },
        "items_compactification": {
            "within_board_round_concordance": 0.81,
            "within_board_round_concordance_shuf_null": 0.50,
        },
    }
    nanbeige = {
        "bundle_sha256": "abc123",
        "items_target_correct_by_loop": [[1, 0, 0, 0] * 4, [1, 0, 0, 0] * 4, [1, 0, 0, 0] * 4],
        "items_open2_correct": [1, 0, 0, 0] * 4,
        "items_donor2_correct": [1, 0, 0, 0] * 4,
        "items_patch_repaired2_correct": [0] * n,
        "items_donor2_top1_agree_with_loop2": [0] * n,
        "items_patch_repaired2_top1_agree_with_loop2": [0] * n,
        "rel_step_norm_mean_by_loop": [0.9, 0.8, 0.5],
        "mean_r_pass1_vs_pass2": 0.08,
        "tca_s1": {"layers": ["L6"], "zr_gt3_share_pass1_pass2": [[0.01, 0.02]]},
    }
    qwen4b = {
        "bundle_sha256": "abc123",
        "items_target_correct": [1, 0, 0, 0] * 4,
        "items_mid_lens_correct": [0] * n,
        "tca_s1": {"layers": ["L8"], "zr_gt3_share": [0.02]},
    }
    r1 = SC.score_all(bundle, trm, nanbeige, qwen4b)
    r2 = SC.score_all(bundle, trm, nanbeige, qwen4b)
    assert r1 == r2
    assert r1["verdicts"]["P3_svd_operator_reuse_vs_multiplexing"] == "Supported"
    assert r1["verdicts"]["P4_tca_attractor_masking_and_specificity"] == "Supported"
    assert r1["verdicts"]["P5_q11_compactification_and_beyond_prop_search"] == "Supported"


def test_committed_results_replay_exact_if_present():
    rdir = Path(__file__).resolve().parents[1] / "results"
    rep_path = rdir / "loopscope02_report.json"
    if not rep_path.exists():
        pytest.skip("Pre-run state: committed results not yet generated")
    bundle = json.loads((rdir / "benchmark_bundle.json").read_text(encoding="utf-8"))
    trm = json.loads((rdir / "trm_results.json").read_text(encoding="utf-8"))
    nanbeige = json.loads((rdir / "nanbeige_results.json").read_text(encoding="utf-8"))
    qwen4b = json.loads((rdir / "qwen4b_results.json").read_text(encoding="utf-8"))
    inst_path = rdir / "qwen3_4b_inst_results.json"
    qwen3_inst = json.loads(inst_path.read_text(encoding="utf-8")) if inst_path.exists() else None
    recomputed = SC.score_all(bundle, trm, nanbeige, qwen4b, qwen3_inst=qwen3_inst)
    committed = json.loads(rep_path.read_text(encoding="utf-8"))
    assert recomputed == committed


def test_fiber_regime_controls_invariants_if_present():
    rdir = Path(__file__).resolve().parents[1] / "results"
    ctrl_path = rdir / "fiber_regime_controls.json"
    if not ctrl_path.exists():
        pytest.skip("Pre-run state: fiber_regime_controls.json not yet generated")
    ctrl = json.loads(ctrl_path.read_text(encoding="utf-8"))

    assert ctrl["head_rank"] == 11
    assert ctrl["orth_fiber_dim"] == 501
    assert ctrl["orth_fiber_linear_probes_bench100"]["max_lm_head_logit_on_orth_fiber"] < 1e-4
    assert ctrl["orth_fiber_linear_probes_items256"]["max_lm_head_logit_on_orth_fiber"] < 1e-4

    # Contrast model (LoopedChase lambda=0.1): cursor P_head projection retains 1.000
    lam01 = ctrl["looped_chase_contrast"]["q4_mean_by_lambda"]["0.1"]
    assert lam01["clean"]["d9_16_mean"] == pytest.approx(1.0)
    assert lam01["cursor_fiber_erased_eff"]["d9_16_mean"] == pytest.approx(1.0)

    # TRM items256 manifold-preserving controls preserve 100% of prop-solvable boards (73/73)
    # while selectively degrading stalled boards (183) and recovering full clean solve at pca_k501
    it = ctrl["trm_primary_items256"]
    assert it["clean"]["n_solved_prop"] == 73
    assert it["clean"]["n_solved_stalled"] == 153
    for m_ctrl in ("fiber_norm_restored", "fiber_train_mean_filled", "pca_k0", "fiber_donor_swapped"):
        assert it[m_ctrl]["n_solved_prop"] == 73
        assert it[m_ctrl]["n_solved_stalled"] < it["clean"]["n_solved_stalled"]
        assert it[m_ctrl]["mcnemar_p_vs_clean_stalled"] < 0.01
    assert it["pca_k501"]["n_solved_all"] == it["clean"]["n_solved_all"]
    # S4 board_exact <= S4 target_cell mathematical invariant
    for cond_val in it.values():
        s4 = cond_val["by_stratum"]["S4_stalled"]
        assert s4["board_exact"] <= s4["target_cell"] + 1e-9


def test_fiber_replication_and_history_probes_invariants_if_present():
    rdir = Path(__file__).resolve().parents[1] / "results"
    ext_path = rdir / "fiber_replication_and_history_probes.json"
    if not ext_path.exists():
        pytest.skip("Pre-run state: fiber_replication_and_history_probes.json not yet generated")
    ext = json.loads(ext_path.read_text(encoding="utf-8"))

    # 1. Batch-size & SVD determinism on bench100: bfloat16 clean & norm_restored are 100/100 everywhere,
    #    whereas raw off-manifold fiber_erased reproduces 7/100 (gpu_f32_svd, bsz=50) vs 5/100 (cpu_f64_svd, bsz=64)
    ba = ext["batch_size_and_precision_determinism_bench100"]
    assert ba["bfloat16__gpu_f32_svd__bsz50__raw_fiber_erased"]["n_solved"] == 7
    assert ba["bfloat16__cpu_f64_svd__bsz64__raw_fiber_erased"]["n_solved"] == 5
    for bsz in (1, 50, 64, 100):
        assert ba[f"bfloat16__cpu_f64_svd__bsz{bsz}__clean"]["n_solved"] == 100
        assert ba[f"bfloat16__cpu_f64_svd__bsz{bsz}__fiber_norm_restored"]["n_solved"] == 100
        assert ba[f"bfloat16__gpu_f32_svd__bsz{bsz}__fiber_norm_restored"]["n_solved"] == 100

    # 2. Replication (`yagizdevre`) on items256 (73 WB-Prop + 183 WB-Stall = 256)
    rep_it = ext["trm_replication_items256"]
    assert rep_it["clean"]["n_solved_prop"] == 64
    assert rep_it["clean"]["n_solved_stalled"] == 120
    assert rep_it["clean"]["n_solved_all"] == 184
    assert rep_it["fiber_erased"]["n_solved_all"] == 0
    assert rep_it["pca_k501"]["n_solved_all"] == rep_it["clean"]["n_solved_all"]
    for m_ctrl in ("fiber_norm_restored", "fiber_train_mean_filled", "pca_k0", "fiber_donor_swapped", "fiber_step1_frozen"):
        assert rep_it[m_ctrl]["n_solved_prop"] >= rep_it["clean"]["n_solved_prop"]
    assert rep_it["fiber_donor_swapped"]["mcnemar_p_vs_clean_stalled"] < 0.03
    assert rep_it["fiber_step1_frozen"]["mcnemar_p_vs_clean_stalled"] < 0.001
    for cond_val in rep_it.values():
        s4 = cond_val["by_stratum"]["S4_stalled"]
        assert s4["board_exact"] <= s4["target_cell"] + 1e-9

    # 3. Differential search-history probes (Primary & Replication) with shuffled-label null
    for probe_key in ("differential_history_probes_primary", "differential_history_probes_replication"):
        dp = ext[probe_key]
        for step_key in ("step_2", "step_4", "step_8", "step_12"):
            st = dp[step_key]["items256"]["stalled"]
            pr = st["probes"]
            shuf = pr["shuffled_label_null"]
            assert 0.07 <= shuf["refuted_prev_digit_acc"] <= 0.16
            assert 0.42 <= shuf["cumulative_flip_auc"] <= 0.58
            assert 0.42 <= shuf["resolved_conflict_auc"] <= 0.58
            # Orthogonal top-64 exceeds visible_plus_clues on cumulative flip history across all steps & checkpoints
            assert pr["orth_top64"]["cumulative_flip_auc"] > pr["visible_plus_clues"]["cumulative_flip_auc"]
            # Visible runner-up logit exceeds orth_top64 on specific just-refuted 9-way digit identity
            assert st["runner_up_logit_heuristic"]["refuted_prev_digit_acc"] > pr["orth_top64"]["refuted_prev_digit_acc"]

