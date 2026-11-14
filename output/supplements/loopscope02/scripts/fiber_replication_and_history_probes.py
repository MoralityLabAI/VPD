#!/usr/bin/env python3
"""LOOPSCOPE-02 v012 Extended Controls:
1. Full `yagizdevre/trm-sudoku-extreme-att` replication on `items256` (N=256: 73 WB-Prop,
   183 WB-Stall, 64 per S1..S4 stratum) and full `pca_k` grid on `bench100` + `items256`
   (both global-mean-centered PCA matching v010/v011 and step-mean-centered PCA).
2. Differential search-history, refuted-assignment, and resolved-conflict linear probes
   comparing:
     - `visible_plus_clues`: [11-d P_head coordinates, 9-d softmax probs, 9-d clue mask] (29-d)
     - `orth_top64`: top-64 train-PCA coordinates inside ker(lm_head) (64-d)
     - `orth_full501`: all 501 train-PCA coordinates inside ker(lm_head) (501-d)
     - `combined_vis_orth64`: concatenation of visible_plus_clues + orth_top64 (93-d)
     - `shuffled_label_null`: orth_top64 fit on randomly permuted training labels
   trained strictly on `train.csv` (200 boards, zero test leakage) and evaluated on held-out
   `items256` and `bench100` for BOTH `gaoxin492` and `yagizdevre`.
3. Exact batch-size & floating-point precision audit explaining `7/100` (bsz=50, float32 GPU SVD)
   vs `5/100` (bsz=64, float64 CPU SVD) on raw off-manifold `fiber_erased`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mechinterp.loopscope02.scripts import fiber_regime_controls as FRC  # noqa: E402
from mechinterp.loopscope02.scripts import score_loopscope02 as SC       # noqa: E402
from mechinterp.loopscope02.scripts import sudoku_prop as SP             # noqa: E402
from mechinterp.loopscope02.scripts import trm_sudoku as TS              # noqa: E402

RESULTS_DIR = REPO_ROOT / "mechinterp" / "loopscope02" / "results"

# Precompute peer cells (same row, col, or 3x3 box, excluding self) for each of 81 cells
PEERS: list[list[int]] = []
for _idx in range(81):
    _r, _c = divmod(_idx, 9)
    _br, _bc = (_r // 3) * 3, (_c // 3) * 3
    _p = set()
    for _j in range(9):
        _p.add(_r * 9 + _j)
        _p.add(_j * 9 + _c)
    for _dr in range(3):
        for _dc in range(3):
            _p.add((_br + _dr) * 9 + (_bc + _dc))
    _p.discard(_idx)
    PEERS.append(sorted(_p))
PEERS_ARR = np.array(PEERS, dtype=np.int64)  # [81, 20]


def cell_unit_conflicts(grid_digits_0_8: np.ndarray) -> np.ndarray:
    """Given [N, 81] digits in 0..8, return boolean [N, 81] True iff cell shares its digit with any peer."""
    peer_vals = grid_digits_0_8[:, PEERS_ARR]  # [N, 81, 20]
    return (peer_vals == grid_digits_0_8[:, :, None]).any(axis=-1)


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact Wilcoxon-Mann-Whitney ROC-AUC with tie handling."""
    y = y_true.astype(np.int64)
    pos = (y == 1)
    neg = (y == 0)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    s_sorted = y_score[order]
    n = len(s_sorted)
    i = 0
    while i < n:
        j = i + 1
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos = float(ranks[pos].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


@torch.no_grad()
def collect_clean_trajectories(
    model: TS.PureSudokuTRM,
    qs: list[str],
    ans: list[str],
    n_steps: int = 16,
    bsz: int = 64,
) -> dict:
    dev = next(model.parameters()).device
    n = len(qs)
    inp = TS.encode_questions(qs, dev)
    x_all = model.encode_inputs(inp)
    puz_len = model.puzzle_emb_len

    zh_steps = torch.zeros((n_steps, n, 81, 512), dtype=torch.float32)
    logits_steps = torch.zeros((n_steps, n, 81, 9), dtype=torch.float32)
    for b0 in range(0, n, bsz):
        b1 = min(n, b0 + bsz)
        zh, zl = model.init_states(b1 - b0)
        xb = x_all[b0:b1]
        for s in range(n_steps):
            zh, zl = model.act_step(zh, zl, xb)
            zh_steps[s, b0:b1] = zh[:, puz_len:, :].float().cpu()
            logits_steps[s, b0:b1] = model.digit_logits(zh).cpu()

    probs_steps = torch.softmax(logits_steps, dim=-1).numpy()  # [S, N, 81, 9]
    pred_digits = logits_steps.argmax(dim=-1).numpy()          # [S, N, 81] in 0..8
    empty_mask = np.array([[ch == "." for ch in q] for q in qs], dtype=bool)  # [N, 81]
    given_digits = np.array([[-1 if ch == "." else int(ch) - 1 for ch in q] for q in qs], dtype=np.int64)
    true_digits = np.array([[int(ch) - 1 for ch in a] for a in ans], dtype=np.int64)

    # Full-board merged grid at each step (preserving givens on clue cells)
    merged_steps = np.where(empty_mask[None, :, :], pred_digits, given_digits[None, :, :])  # [S, N, 81]
    conflicts_steps = np.stack([cell_unit_conflicts(merged_steps[s]) for s in range(n_steps)], axis=0)  # [S, N, 81]

    init_cands = np.zeros((n, 81, 9), dtype=np.float32)
    for bi, q in enumerate(qs):
        init_cands[bi] = SP.candidate_mask(SP.parse_grid(q)).astype(np.float32)

    return {
        "zh_steps": zh_steps,            # [S, N, 81, 512] torch float32 CPU
        "probs_steps": probs_steps,      # [S, N, 81, 9]
        "pred_digits": pred_digits,      # [S, N, 81]
        "merged_steps": merged_steps,    # [S, N, 81]
        "conflicts_steps": conflicts_steps,  # [S, N, 81]
        "empty_mask": empty_mask,        # [N, 81]
        "true_digits": true_digits,      # [N, 81]
        "init_cands": init_cands,        # [N, 81, 9]
    }


def build_feature_views(
    traj: dict,
    s_idx: int,
    v_head: np.ndarray,
    q_pca: np.ndarray,
    global_mean_orth: np.ndarray,
    p_orth: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return [N, 81, D_view] arrays for step s_idx."""
    zh = traj["zh_steps"][s_idx].numpy()       # [N, 81, 512]
    head_coords = zh @ v_head                  # [N, 81, 11]
    probs = traj["probs_steps"][s_idx]         # [N, 81, 9]
    cands = traj["init_cands"]                 # [N, 81, 9]
    vis = np.concatenate([head_coords, probs, cands], axis=-1).astype(np.float32)  # [N, 81, 29]

    orth_c = (zh @ p_orth) - global_mean_orth  # [N, 81, 512]
    orth501 = (orth_c @ q_pca).astype(np.float32)          # [N, 81, 501]
    orth64 = orth501[:, :, :64]                            # [N, 81, 64]
    comb = np.concatenate([vis, orth64], axis=-1)          # [N, 81, 93]
    return {
        "visible_plus_clues": vis,
        "orth_top64": orth64,
        "orth_full501": orth501,
        "combined_vis_orth64": comb,
    }


def fit_ridge_matrix(x: np.ndarray, y: np.ndarray, reg: float = 10.0) -> np.ndarray:
    """Standardize features (mean/std on train) + bias, fit ridge weights W."""
    mu = x.mean(axis=0, keepdims=True)
    sd = np.maximum(x.std(axis=0, keepdims=True), 1e-5)
    xn = (x - mu) / sd
    xa = np.concatenate([xn, np.ones((xn.shape[0], 1), dtype=np.float32)], axis=1)
    gram = xa.T @ xa + reg * np.eye(xa.shape[1], dtype=np.float32)
    gram[-1, -1] -= reg  # do not regularize intercept
    w = np.linalg.solve(gram.astype(np.float64), (xa.T @ y).astype(np.float64)).astype(np.float32)
    return mu, sd, w


def predict_ridge(x: np.ndarray, fit_tuple: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    mu, sd, w = fit_tuple
    xn = (x - mu) / sd
    xa = np.concatenate([xn, np.ones((xn.shape[0], 1), dtype=np.float32)], axis=1)
    return xa @ w


def run_differential_history_probes(
    train_traj: dict,
    test_splits: dict[str, tuple[dict, np.ndarray]],
    v_head: torch.Tensor,
    pca_fit: dict,
) -> dict:
    """Train strictly on train_traj; evaluate on test_splits (each (traj, prop_flags))."""
    v_h = v_head.cpu().numpy().astype(np.float32)                   # [512, 11]
    p_orth = (np.eye(512, dtype=np.float64) - (v_h.astype(np.float64) @ v_h.astype(np.float64).T)).astype(np.float32)
    q_pca = pca_fit["q_pca"].cpu().numpy().astype(np.float32)       # [512, 501]
    g_mean = pca_fit["global_mean_orth"].cpu().numpy().astype(np.float32)

    rng = np.random.default_rng(20261114)
    out = {}

    for s_idx in (1, 3, 7, 11):  # steps 2, 4, 8, 12
        step_key = f"step_{s_idx + 1}"
        out[step_key] = {}

        tr_views = build_feature_views(train_traj, s_idx, v_h, q_pca, g_mean, p_orth)
        tr_emp = train_traj["empty_mask"]
        tr_cur = train_traj["pred_digits"][s_idx]
        tr_prev = train_traj["pred_digits"][s_idx - 1]
        tr_s1 = train_traj["pred_digits"][0]
        tr_true = train_traj["true_digits"]

        # Define 5 differential search-history / hidden-state tasks
        # Task 1: Just-refuted digit d_{s-1} on empty cells that flipped at step s (pred[s] != pred[s-1])
        m_tr_flip_prev = tr_emp & (tr_cur != tr_prev)
        # Task 2: Abandoned step-1 digit d_1 on empty cells where pred[s] != pred[0]
        m_tr_flip_s1 = tr_emp & (tr_cur != tr_s1)
        # Task 3: Cumulative flip history binary indicator (any pred[0..s-1] != pred[s]) on empty cells
        any_flipped_tr = np.any(train_traj["pred_digits"][:s_idx] != tr_cur[None, :, :], axis=0)
        # Task 4: Resolved unit conflict history on currently conflict-free empty cells:
        #   cell has NO conflict at step s, did it have a unit conflict in steps 1..s-1?
        m_tr_conffree = tr_emp & (~train_traj["conflicts_steps"][s_idx])
        had_prior_conf_tr = np.any(train_traj["conflicts_steps"][:s_idx], axis=0)
        # Task 5: Ground-truth digit on cells where current lm_head argmax is WRONG (pred[s] != true)
        m_tr_wrong = tr_emp & (tr_cur != tr_true)

        # Fit models on train
        fitted = {}
        for vname, v_arr in tr_views.items():
            fitted[vname] = {}
            if m_tr_flip_prev.sum() >= 20:
                y1 = np.eye(9, dtype=np.float32)[tr_prev[m_tr_flip_prev]]
                fitted[vname]["refuted_prev_digit"] = fit_ridge_matrix(v_arr[m_tr_flip_prev], y1, reg=25.0)
            if m_tr_flip_s1.sum() >= 20:
                y2 = np.eye(9, dtype=np.float32)[tr_s1[m_tr_flip_s1]]
                fitted[vname]["abandoned_step1_digit"] = fit_ridge_matrix(v_arr[m_tr_flip_s1], y2, reg=25.0)
            y3 = any_flipped_tr[tr_emp].astype(np.float32)[:, None]
            fitted[vname]["cumulative_flip_binary"] = fit_ridge_matrix(v_arr[tr_emp], y3, reg=25.0)
            if m_tr_conffree.sum() >= 20:
                y4 = had_prior_conf_tr[m_tr_conffree].astype(np.float32)[:, None]
                fitted[vname]["resolved_conflict_binary"] = fit_ridge_matrix(v_arr[m_tr_conffree], y4, reg=25.0)
            if m_tr_wrong.sum() >= 20:
                y5 = np.eye(9, dtype=np.float32)[tr_true[m_tr_wrong]]
                fitted[vname]["true_digit_when_head_wrong"] = fit_ridge_matrix(v_arr[m_tr_wrong], y5, reg=25.0)

        # Shuffled-label null on orth_top64
        fitted["shuffled_label_null"] = {}
        v64 = tr_views["orth_top64"]
        if m_tr_flip_prev.sum() >= 20:
            perm = rng.permutation(int(m_tr_flip_prev.sum()))
            y1s = np.eye(9, dtype=np.float32)[tr_prev[m_tr_flip_prev][perm]]
            fitted["shuffled_label_null"]["refuted_prev_digit"] = fit_ridge_matrix(v64[m_tr_flip_prev], y1s, reg=25.0)
        if m_tr_flip_s1.sum() >= 20:
            perm = rng.permutation(int(m_tr_flip_s1.sum()))
            y2s = np.eye(9, dtype=np.float32)[tr_s1[m_tr_flip_s1][perm]]
            fitted["shuffled_label_null"]["abandoned_step1_digit"] = fit_ridge_matrix(v64[m_tr_flip_s1], y2s, reg=25.0)
        perm_emp = rng.permutation(int(tr_emp.sum()))
        y3s = any_flipped_tr[tr_emp][perm_emp].astype(np.float32)[:, None]
        fitted["shuffled_label_null"]["cumulative_flip_binary"] = fit_ridge_matrix(v64[tr_emp], y3s, reg=25.0)
        if m_tr_conffree.sum() >= 20:
            perm_cf = rng.permutation(int(m_tr_conffree.sum()))
            y4s = had_prior_conf_tr[m_tr_conffree][perm_cf].astype(np.float32)[:, None]
            fitted["shuffled_label_null"]["resolved_conflict_binary"] = fit_ridge_matrix(v64[m_tr_conffree], y4s, reg=25.0)
        if m_tr_wrong.sum() >= 20:
            perm_w = rng.permutation(int(m_tr_wrong.sum()))
            y5s = np.eye(9, dtype=np.float32)[tr_true[m_tr_wrong][perm_w]]
            fitted["shuffled_label_null"]["true_digit_when_head_wrong"] = fit_ridge_matrix(v64[m_tr_wrong], y5s, reg=25.0)

        # Evaluate on each test split (focusing on stalled boards as well as all boards)
        for sp_name, (te_traj, prop_flags) in test_splits.items():
            out[step_key][sp_name] = {}
            te_views = build_feature_views(te_traj, s_idx, v_h, q_pca, g_mean, p_orth)
            stalled_b = (~prop_flags)[:, None]  # [N, 1]

            for cohort_name, b_filter in [("stalled", stalled_b), ("all", np.ones_like(stalled_b, dtype=bool))]:
                te_emp = te_traj["empty_mask"] & b_filter
                te_cur = te_traj["pred_digits"][s_idx]
                te_prev = te_traj["pred_digits"][s_idx - 1]
                te_s1 = te_traj["pred_digits"][0]
                te_true = te_traj["true_digits"]

                m_te_flip_prev = te_emp & (te_cur != te_prev)
                m_te_flip_s1 = te_emp & (te_cur != te_s1)
                any_flipped_te = np.any(te_traj["pred_digits"][:s_idx] != te_cur[None, :, :], axis=0)
                m_te_conffree = te_emp & (~te_traj["conflicts_steps"][s_idx])
                had_prior_conf_te = np.any(te_traj["conflicts_steps"][:s_idx], axis=0)
                m_te_wrong = te_emp & (te_cur != te_true)

                # Also compute non-learned runner-up logit baseline for 9-class tasks!
                probs_te = te_traj["probs_steps"][s_idx]  # [N, 81, 9]
                probs_masked_cur = probs_te.copy()
                # Mask out current argmax to get 2nd-choice visible runner-up digit
                idx_n, idx_c = np.meshgrid(np.arange(probs_te.shape[0]), np.arange(81), indexing="ij")
                probs_masked_cur[idx_n, idx_c, te_cur] = -1.0
                runner_up_digit = probs_masked_cur.argmax(axis=-1)

                cohort_res = {
                    "counts": {
                        "n_empty": int(te_emp.sum()),
                        "n_flipped_from_prev": int(m_te_flip_prev.sum()),
                        "n_abandoned_step1": int(m_te_flip_s1.sum()),
                        "n_cum_flipped_pos": int(any_flipped_te[te_emp].sum()),
                        "n_conffree_cells": int(m_te_conffree.sum()),
                        "n_conffree_with_prior_conflict": int(had_prior_conf_te[m_te_conffree].sum()),
                        "n_head_currently_wrong": int(m_te_wrong.sum()),
                    },
                    "runner_up_logit_heuristic": {
                        "refuted_prev_digit_acc": float((runner_up_digit[m_te_flip_prev] == te_prev[m_te_flip_prev]).mean()) if m_te_flip_prev.sum() > 0 else None,
                        "abandoned_step1_digit_acc": float((runner_up_digit[m_te_flip_s1] == te_s1[m_te_flip_s1]).mean()) if m_te_flip_s1.sum() > 0 else None,
                        "true_digit_when_head_wrong_acc": float((runner_up_digit[m_te_wrong] == te_true[m_te_wrong]).mean()) if m_te_wrong.sum() > 0 else None,
                    },
                    "probes": {},
                }

                for model_key in ("visible_plus_clues", "orth_top64", "orth_full501", "combined_vis_orth64", "shuffled_label_null"):
                    view_key = "orth_top64" if model_key == "shuffled_label_null" else model_key
                    v_te = te_views[view_key]
                    fdict = fitted[model_key]
                    pres = {}

                    if "refuted_prev_digit" in fdict and m_te_flip_prev.sum() > 0:
                        pr = predict_ridge(v_te[m_te_flip_prev], fdict["refuted_prev_digit"]).argmax(axis=-1)
                        pres["refuted_prev_digit_acc"] = float((pr == te_prev[m_te_flip_prev]).mean())
                    if "abandoned_step1_digit" in fdict and m_te_flip_s1.sum() > 0:
                        pr = predict_ridge(v_te[m_te_flip_s1], fdict["abandoned_step1_digit"]).argmax(axis=-1)
                        pres["abandoned_step1_digit_acc"] = float((pr == te_s1[m_te_flip_s1]).mean())
                    if "cumulative_flip_binary" in fdict and te_emp.sum() > 0:
                        sc = predict_ridge(v_te[te_emp], fdict["cumulative_flip_binary"])[:, 0]
                        pres["cumulative_flip_auc"] = float(roc_auc_binary(any_flipped_te[te_emp], sc))
                    if "resolved_conflict_binary" in fdict and m_te_conffree.sum() > 0:
                        sc = predict_ridge(v_te[m_te_conffree], fdict["resolved_conflict_binary"])[:, 0]
                        pres["resolved_conflict_auc"] = float(roc_auc_binary(had_prior_conf_te[m_te_conffree], sc))
                    if "true_digit_when_head_wrong" in fdict and m_te_wrong.sum() > 0:
                        pr = predict_ridge(v_te[m_te_wrong], fdict["true_digit_when_head_wrong"]).argmax(axis=-1)
                        pres["true_digit_when_head_wrong_acc"] = float((pr == te_true[m_te_wrong]).mean())

                    cohort_res["probes"][model_key] = pres
                out[step_key][sp_name][cohort_name] = cohort_res

    return out


@torch.no_grad()
def audit_batch_size_determinism(
    qs: list[str],
    ans: list[str],
    dev: str,
) -> dict:
    """Pinpoint why raw off-manifold `fiber_erased` scored 7/100 (P5, bsz=50, GPU float32 SVD)
    vs 5/100 (controls, bsz=64, CPU float64 SVD) on bench100, and check clean & norm_restored."""
    out = {}
    for model_dtype_name, model_dtype in [("bfloat16", torch.bfloat16), ("float32", torch.float32)]:
        m, _ = TS.load_trm_checkpoint("gaoxin492", device=dev, dtype=model_dtype)
        inp_all = TS.encode_questions(qs, dev)
        ans_all = TS.encode_answers(ans, dev)
        giv_mask = (inp_all != 1)
        puz_len = m.puzzle_emb_len

        # Projector A: GPU float32 SVD (as in trm_sudoku.py)
        w_f32 = m.lm_head.weight.detach().float()
        _, _, vh32 = torch.linalg.svd(w_f32, full_matrices=False)
        p_gpu32 = vh32.T @ vh32
        # Projector B: CPU float64 SVD (as in fiber_regime_controls.py)
        _, p_cpu64_cpu = FRC.compute_head_projector(m.lm_head.weight)
        p_cpu64 = p_cpu64_cpu.to(dev)

        for svd_name, p_mat in [("gpu_f32_svd", p_gpu32), ("cpu_f64_svd", p_cpu64)]:
            i_minus_p = torch.eye(512, device=dev, dtype=torch.float32) - p_mat
            h_init_orth = (m.H_init.detach().float().to(dev) @ i_minus_p).view(1, 1, 512)
            for bsz in (1, 50, 64, 100):
                for mode in ("clean", "raw_fiber_erased", "fiber_norm_restored"):
                    if svd_name == "gpu_f32_svd" and mode == "clean":
                        continue
                    solved = np.zeros(len(qs), dtype=bool)
                    for b0 in range(0, len(qs), bsz):
                        b1 = min(len(qs), b0 + bsz)
                        xb = m.encode_inputs(inp_all[b0:b1])
                        zh, zl = m.init_states(b1 - b0)
                        zl_init = zl.clone()
                        for s in range(16):
                            zh, zl = m.act_step(zh, zl, xb)
                            if s < 15:
                                zb = zh[:, puz_len:, :].float()
                                zb_head = zb @ p_mat
                                if mode == "raw_fiber_erased":
                                    zl = zl_init.clone()
                                    zh = zh.clone()
                                    zh[:, puz_len:, :] = (zb_head + h_init_orth).to(model_dtype)
                                elif mode == "fiber_norm_restored":
                                    zh = zh.clone()
                                    on = zb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                                    hn = zb_head.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                                    zh[:, puz_len:, :] = (zb_head * (on / hn)).to(model_dtype)
                        preds = m.digit_logits(zh).argmax(dim=-1)
                        merged = torch.where(giv_mask[b0:b1], inp_all[b0:b1] - 2, preds)
                        solved[b0:b1] = (merged == ans_all[b0:b1]).all(dim=1).cpu().numpy()
                    key = f"{model_dtype_name}__{svd_name}__bsz{bsz}__{mode}"
                    out[key] = {
                        "n_solved": int(solved.sum()),
                        "solved_indices": [int(i) for i in np.where(solved)[0]],
                    }
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
    return out


@torch.no_grad()
def run_step_mean_pca_curve(
    model: TS.PureSudokuTRM,
    p_head: torch.Tensor,
    pca_fit: dict,
    qs: list[str],
    ans: list[str],
    prop_flags: list[bool],
    pca_ks: tuple[int, ...] = (0, 4, 8, 16, 32, 64, 128, 256, 501),
) -> dict:
    """Step-mean-centered orthogonal PCA (centers at step_mean_orth[s] rather than global_mean_orth)."""
    dev = next(model.parameters()).device
    dt = model.embed_tokens.weight.dtype
    puz_len = model.puzzle_emb_len
    n = len(qs)
    inp_all = TS.encode_questions(qs, dev)
    ans_all = TS.encode_answers(ans, dev)
    giv_mask = (inp_all != 1)
    x_all = model.encode_inputs(inp_all)

    p_head_d = p_head.to(device=dev, dtype=torch.float32)
    i_minus_p = torch.eye(512, device=dev, dtype=torch.float32) - p_head_d
    step_mean_orth = pca_fit["step_mean_orth"].to(dev)  # [16, 81, 512]
    q_pca = pca_fit["q_pca"].to(dev)                    # [512, 501]
    prop_mask = np.array(prop_flags, dtype=bool)
    bsz = 64

    # Clean reference
    clean_solved = np.zeros(n, dtype=bool)
    for b0 in range(0, n, bsz):
        b1 = min(n, b0 + bsz)
        zh, zl = model.init_states(b1 - b0)
        for _s in range(16):
            zh, zl = model.act_step(zh, zl, x_all[b0:b1])
        preds = model.digit_logits(zh).argmax(dim=-1)
        merged = torch.where(giv_mask[b0:b1], inp_all[b0:b1] - 2, preds)
        clean_solved[b0:b1] = (merged == ans_all[b0:b1]).all(dim=1).cpu().numpy()

    out = {}
    for k in pca_ks:
        qk = q_pca[:, :k]
        pk = qk @ qk.T
        solved = np.zeros(n, dtype=bool)
        for b0 in range(0, n, bsz):
            b1 = min(n, b0 + bsz)
            zh, zl = model.init_states(b1 - b0)
            for s in range(16):
                zh, zl = model.act_step(zh, zl, x_all[b0:b1])
                if s < 15:
                    zb = zh[:, puz_len:, :].float()
                    zb_head = zb @ p_head_d
                    mu_s = step_mean_orth[s][None, :, :]
                    orth_c = (zb @ i_minus_p) - mu_s
                    orth_rec = (orth_c @ pk) + mu_s
                    zh = zh.clone()
                    zh[:, puz_len:, :] = (zb_head + orth_rec).to(dt)
            preds = model.digit_logits(zh).argmax(dim=-1)
            merged = torch.where(giv_mask[b0:b1], inp_all[b0:b1] - 2, preds)
            solved[b0:b1] = (merged == ans_all[b0:b1]).all(dim=1).cpu().numpy()
        out[f"step_pca_k{k}"] = {
            "board_exact_all": float(solved.mean()),
            "board_exact_prop": float(solved[prop_mask].mean()),
            "board_exact_stalled": float(solved[~prop_mask].mean()),
            "n_solved_prop": int(solved[prop_mask].sum()),
            "n_solved_stalled": int(solved[~prop_mask].sum()),
            "mcnemar_p_vs_clean_stalled": float(SC.exact_mcnemar_p(clean_solved[~prop_mask].astype(int).tolist(), solved[~prop_mask].astype(int).tolist())),
        }
    return out


def main() -> None:
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running v012 extended controls on device={dev}...")

    bundle = json.loads((RESULTS_DIR / "benchmark_bundle.json").read_text(encoding="utf-8"))
    items = bundle["items"]
    id_to_idx = {it["item_id"]: idx for idx, it in enumerate(items)}
    items_qs = [it["question"] for it in items]
    items_ans = [it["answer"] for it in items]
    items_donors = [id_to_idx[it["donor_item_id"]] for it in items]
    items_targets = [int(it["target_idx"]) for it in items]
    items_strata = [it["stratum"] for it in items]
    items_prop_flags = [bool(it["prop_solved"]) for it in items]

    bench_qs, bench_ans, bench_prop_flags = FRC.load_bench100()
    rng = np.random.default_rng(20261113)
    bench_donors = [0] * len(bench_qs)
    for flag_val in (True, False):
        idxs = [i for i, f in enumerate(bench_prop_flags) if f == flag_val]
        perm = rng.permutation(idxs)
        rolled = np.roll(perm, -1)
        for src_i, dst_i in zip(perm, rolled):
            bench_donors[int(src_i)] = int(dst_i)

    train_qs, train_ans = FRC.load_train_boards(200)

    # 1. Batch-size & precision determinism audit on bench100
    t_b = time.time()
    batch_audit = audit_batch_size_determinism(bench_qs, bench_ans, dev=dev)
    print(f"Completed batch-size determinism audit in {time.time() - t_b:.1f}s")

    # 2. Primary TRM (`gaoxin492`): differential history probes + step-mean PCA check
    t_p = time.time()
    trm_main, main_sha = TS.load_trm_checkpoint("gaoxin492", device=dev, dtype=torch.bfloat16)
    v_main, p_main = FRC.compute_head_projector(trm_main.lm_head.weight)
    pca_main = FRC.fit_train_fiber_pca_and_probes(trm_main, p_main.to(dev), train_qs, train_ans)
    tr_traj_main = collect_clean_trajectories(trm_main, train_qs, train_ans)
    items_traj_main = collect_clean_trajectories(trm_main, items_qs, items_ans)
    bench_traj_main = collect_clean_trajectories(trm_main, bench_qs, bench_ans)
    diff_probes_main = run_differential_history_probes(
        tr_traj_main,
        {
            "items256": (items_traj_main, np.array(items_prop_flags, dtype=bool)),
            "bench100": (bench_traj_main, np.array(bench_prop_flags, dtype=bool)),
        },
        v_main,
        pca_main,
    )
    del trm_main, tr_traj_main, items_traj_main, bench_traj_main
    if dev == "cuda":
        torch.cuda.empty_cache()
    print(f"Completed primary differential history probes in {time.time() - t_p:.1f}s")

    # 3. Replication TRM (`yagizdevre`): FULL items256 + FULL bench100 PCA grid + step-mean PCA + differential probes
    t_r = time.time()
    trm_rep, rep_sha = TS.load_trm_checkpoint("yagizdevre", device=dev, dtype=torch.bfloat16)
    v_rep, p_rep = FRC.compute_head_projector(trm_rep.lm_head.weight)
    pca_rep = FRC.fit_train_fiber_pca_and_probes(trm_rep, p_rep.to(dev), train_qs, train_ans)

    rep_bench_full = FRC.run_trm_fiber_suite_on_split(
        trm_rep, p_rep, pca_rep, bench_qs, bench_ans, bench_donors, None, None, bench_prop_flags,
        pca_ks=(0, 4, 8, 16, 32, 64, 128, 256, 501),
    )
    rep_items_full = FRC.run_trm_fiber_suite_on_split(
        trm_rep, p_rep, pca_rep, items_qs, items_ans, items_donors, items_targets, items_strata, items_prop_flags,
        pca_ks=(0, 4, 8, 16, 32, 64, 128, 256, 501),
    )
    rep_bench_step_pca = run_step_mean_pca_curve(trm_rep, p_rep, pca_rep, bench_qs, bench_ans, bench_prop_flags)
    rep_items_step_pca = run_step_mean_pca_curve(trm_rep, p_rep, pca_rep, items_qs, items_ans, items_prop_flags)

    tr_traj_rep = collect_clean_trajectories(trm_rep, train_qs, train_ans)
    items_traj_rep = collect_clean_trajectories(trm_rep, items_qs, items_ans)
    bench_traj_rep = collect_clean_trajectories(trm_rep, bench_qs, bench_ans)
    diff_probes_rep = run_differential_history_probes(
        tr_traj_rep,
        {
            "items256": (items_traj_rep, np.array(items_prop_flags, dtype=bool)),
            "bench100": (bench_traj_rep, np.array(bench_prop_flags, dtype=bool)),
        },
        v_rep,
        pca_rep,
    )
    probe_rep_items_std = FRC.evaluate_probes_on_split(
        trm_rep, p_rep.to(dev), pca_rep["probes"], items_qs, items_ans, items_prop_flags
    )
    del trm_rep
    if dev == "cuda":
        torch.cuda.empty_cache()
    print(f"Completed replication items256 + bench100 + probes in {time.time() - t_r:.1f}s")

    out_payload = {
        "elapsed_sec": float(time.time() - t0),
        "primary_checkpoint_sha256": main_sha,
        "replication_checkpoint_sha256": rep_sha,
        "replication_pca_train_cumulative_variance": pca_rep["pca_cum_var"],
        "batch_size_and_precision_determinism_bench100": batch_audit,
        "trm_replication_bench100_full_pca": rep_bench_full,
        "trm_replication_items256": rep_items_full,
        "trm_replication_bench100_step_mean_pca": rep_bench_step_pca,
        "trm_replication_items256_step_mean_pca": rep_items_step_pca,
        "trm_replication_standard_probes_items256": probe_rep_items_std,
        "differential_history_probes_primary": diff_probes_main,
        "differential_history_probes_replication": diff_probes_rep,
    }
    out_file = RESULTS_DIR / "fiber_replication_and_history_probes.json"
    out_file.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_file} in {out_payload['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
