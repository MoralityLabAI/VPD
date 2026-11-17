"""B4-CAUSAL-CI-01 scoring path on the mini TRM (no pretrained data, no evaluation boards)."""
from __future__ import annotations

import copy

import torch

from mechinterp.bpd.adapters.trm import TRMBPDAdapter, build_mini_trm, default_trm_c_per_module
from mechinterp.bpd.ci import TiedCINetwork
from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import CIConfig
from mechinterp.bpd.scripts import audit_b4_causal_ci as S


def test_score_chunk_and_decide_on_mini(monkeypatch) -> None:
    monkeypatch.setattr(S, "N_PASSES", 6)
    monkeypatch.setattr(S, "READOUTS", (2, 5))
    monkeypatch.setattr(S, "K", 19)
    monkeypatch.setattr(S, "C", 32)
    monkeypatch.setattr(S, "N_BOOT", 200)
    monkeypatch.setattr(S, "N_FLIP", 500)
    torch.manual_seed(0)
    base = build_mini_trm(hidden_size=32, num_heads=4, l_layers=2, h_cycles=1,
                          l_cycles=2, puzzle_emb_len=4, seq_len=16, seed=113).float().eval()
    c_map = default_trm_c_per_module(base, c=32)
    bpd_svd = BPDModel(copy.deepcopy(base), c_map, max_passes=6, ablation=S.ZERO, init_svd=True)
    bpd_tr = BPDModel(copy.deepcopy(base), c_map, max_passes=6, ablation=S.NATIVE, init_svd=True)
    with torch.no_grad():
        for w in bpd_tr.wrappers.values():
            w.U.add_(0.01 * torch.randn_like(w.U))  # "trained" factors differ from SVD
    ad_svd = TRMBPDAdapter(bpd_svd.target_model, readout_at="outer_step")
    ad_tr = TRMBPDAdapter(bpd_tr.target_model, readout_at="outer_step")
    calib = torch.randint(1, 11, (4, 16), dtype=torch.long)
    S._calibrate_means(bpd_tr, ad_tr, calib)
    assert all(bool(w.calibrated_passes[:6].all()) for w in bpd_tr.wrappers.values())

    dims = {m: bpd_tr.wrappers[m].d_in for m in bpd_tr.module_order}
    ci = TiedCINetwork(dims, bpd_tr.c_per_module,
                       CIConfig(variant="state_only", d_model=32, n_blocks=1, n_heads=4,
                                mlp_hidden=64, max_passes=16, max_seq_len=32)).eval()
    inputs = torch.randint(1, 11, (3, 16), dtype=torch.long)
    res = S.score_chunk(bpd_svd, ad_svd, bpd_tr, ad_tr, ci, inputs, "cpu")

    arms = res["arms"]
    for name in S.BUDGET_ARMS + S.NATIVE_ARMS + S.FLOOR_ARMS:
        assert name in arms
        assert len(arms[name]["pass_5"]["kl64"]) == 3
    for floor in S.FLOOR_ARMS + ("target_repeat_trained_wrapper",):
        assert max(arms[floor]["pass_5"]["kl64"]) < 1e-8, floor
    # Causal and clean-informed CI gates agree at invocation 0.
    for m, rows in res["T3_vs_T2_disagreement"].items():
        assert rows[0] == [0.0, 0.0, 0.0]
        assert len(rows) == 6

    per_ckpt = {"a": {"arms": arms}, "b": {"arms": arms}}
    monkeypatch.setattr(S, "READOUTS", (20, 41, 62))
    d = S.decide({k: {"arms": {n: {**v, "pass_20": v["pass_2"], "pass_41": v["pass_5"], "pass_62": v["pass_5"]}
                               for n, v in arms.items()}} for k in per_ckpt})
    assert set(d["registered_pass20_decisions"]) == {
        "H1_gate_value_beyond_magnitude_selection",
        "H2_causal_BPD_vs_zero_fit_SVD_online",
        "H3_causal_CI_noninferior_to_clean_informed_CI",
    }
    assert d["by_checkpoint"]["a"]["pass_20"]["interpretable"] is True

    # The BPD-01 reducer formats the receipt and cross-checks contrast means against per-board KL.
    from mechinterp.bpd.scripts.reduce_bpd01 import reduce_causal

    renamed = {n: {**v, "pass_20": v["pass_2"], "pass_41": v["pass_5"], "pass_62": v["pass_5"]}
               for n, v in arms.items()}
    mods = list(res["T3_vs_T2_disagreement"])
    ck_entry = {
        "arms": renamed,
        "ci_refit_receipt": {"steps": 1, "batch_size": 3, "seed": 0, "ci_weights_sha256": "0" * 64,
                             "validation_metrics_best_state": {"recon_loss": 0.5}},
        "module_execution_order": mods, "ci_module_order": sorted(mods),
        "within_pass_dependency_grad_norm_tau0": {},
        "T3_vs_T2_disagreement_by_module_by_pass": {m: [0.0] * 63 for m in mods},
        "T4_vs_T2_disagreement_by_module_by_pass": {m: [0.0] * 63 for m in mods},
        "N1_mean_mask_by_chunk": [res["N1_mean_mask"]],
        "N1_fraction_above_0p5_by_chunk": [res["N1_fraction_above_0p5"]],
        "N2_mean_mask_by_module_by_pass": {m: [0.5] * 63 for m in mods},
    }
    rec = {"regime": "test", "tf32_matmul_allowed": False, "cuda_device": None, "torch_version": "x",
           "k": 19, "C": 32, "readouts": [20, 41, 62], "eval_board_indices": [0, 1, 2],
           "checkpoints": {"a": ck_entry, "b": ck_entry}, "decisions": d}
    out, md = reduce_causal(rec)
    assert out["registered_pass20_decisions"] == d["registered_pass20_decisions"]
    assert len(out["checkpoints"]["a"]["per_board_kl"]["T3"]["pass_20"]) == 3
    assert any(line.startswith("| H1_gate_value_beyond_magnitude_selection |") for line in md)
