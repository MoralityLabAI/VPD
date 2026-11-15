#!/usr/bin/env python3
"""Deterministic statistical aggregator for the Dark Fibers local campaign (LOOPSCOPE-04 & ARG-03-N).

Reads local run receipts from `runs/dark-fibers/` and emits compact, provenance-locked,
git-trackable summary JSONs (with zero raw puzzle strings or bulky per-case tapes) plus
exact paired hypothesis tests (McNemar, exact two-sided sign test, Wilcoxon signed-rank,
Spearman rank correlation, and deterministic paired bootstrap 95% confidence intervals).
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from math import comb, erfc, sqrt
from pathlib import Path
from typing import Sequence

import numpy as np

from .arg03_roster import exact_mcnemar
from .core import sha256_file, write_json


def exact_sign_test(x: Sequence[float], y: Sequence[float]) -> dict:
    """Exact two-sided binomial sign test on paired differences (x_i - y_i)."""
    diffs = [float(a) - float(b) for a, b in zip(x, y)]
    pos = sum(d > 0.0 for d in diffs)
    neg = sum(d < 0.0 for d in diffs)
    ties = len(diffs) - pos - neg
    n = pos + neg
    if n == 0:
        p = 1.0
    else:
        lo = min(pos, neg)
        tail = sum(comb(n, k) for k in range(lo + 1)) / (2**n)
        p = min(1.0, 2.0 * tail)
    return {
        "n_pairs": len(diffs),
        "positive": pos,
        "negative": neg,
        "ties": ties,
        "win_rate_nonzero": (pos / n) if n else 0.5,
        "exact_sign_p_two_sided": float(p),
    }


def wilcoxon_signed_rank_normal(x: Sequence[float], y: Sequence[float]) -> dict:
    """Two-sided Wilcoxon signed-rank test (tie-corrected normal approximation, pure NumPy)."""
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    nz = d[d != 0.0]
    n = int(nz.size)
    if n == 0:
        return {"n_nonzero": 0, "w_plus": 0.0, "z": 0.0, "p_two_sided": 1.0}
    abs_d = np.abs(nz)
    order = np.argsort(abs_d, kind="mergesort")
    sorted_abs = abs_d[order]
    ranks = np.empty(n, dtype=np.float64)
    tie_groups: list[int] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_abs[j] == sorted_abs[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        if j - i > 1:
            tie_groups.append(j - i)
        i = j
    w_plus = float(ranks[nz > 0.0].sum())
    mean_w = n * (n + 1) / 4.0
    tie_corr = sum(t * (t * t - 1) for t in tie_groups) / 48.0
    var_w = max(n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr, 1e-12)
    diff = w_plus - mean_w
    cc = 0.5 * np.sign(diff) if diff != 0.0 else 0.0
    z = float((diff - cc) / sqrt(var_w))
    p = float(min(1.0, erfc(abs(z) / sqrt(2.0))))
    return {"n_nonzero": n, "w_plus": w_plus, "z": z, "p_two_sided": p}


def paired_bootstrap_ci(
    x: Sequence[float],
    y: Sequence[float] | None = None,
    *,
    n_boot: int = 4000,
    seed: int = 20261115,
    stat: str = "mean",
) -> dict:
    """Deterministic paired bootstrap 95% CI for x (if y is None) or (x - y)."""
    arr = np.asarray(x, dtype=np.float64)
    if y is not None:
        arr = arr - np.asarray(y, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "estimate": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = arr[idx]
    if stat == "median":
        est = float(np.median(arr))
        boots = np.median(samples, axis=1)
    else:
        est = float(np.mean(arr))
        boots = np.mean(samples, axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": n, "estimate": est, "ci95_low": float(lo), "ci95_high": float(hi)}


def _rankdata_2d(mat: np.ndarray) -> np.ndarray:
    """Exact tie-averaged 1-based ranks along axis=1 for shape [B, N]."""
    lt = (mat[:, :, None] < mat[:, None, :]).sum(axis=2)
    eq = (mat[:, :, None] == mat[:, None, :]).sum(axis=2)
    return lt + 0.5 * eq + 0.5


def spearman_corr(x: Sequence[float], y: Sequence[float], *, n_boot: int = 2000, seed: int = 20261115) -> dict:
    """Spearman rank correlation with fast vectorized tie-exact bootstrap 95% CI."""
    ax = np.asarray(x, dtype=np.float64)
    ay = np.asarray(y, dtype=np.float64)
    n = int(ax.size)
    if n < 3:
        return {"n": n, "rho": None, "ci95_low": None, "ci95_high": None}

    def _batch_rho(bx: np.ndarray, by: np.ndarray) -> np.ndarray:
        rx = _rankdata_2d(bx) - 0.5 * (n + 1)
        ry = _rankdata_2d(by) - 0.5 * (n + 1)
        num = (rx * ry).sum(axis=1)
        den = np.linalg.norm(rx, axis=1) * np.linalg.norm(ry, axis=1)
        return np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)

    rho_hat = float(_batch_rho(ax[None, :], ay[None, :])[0])
    rng = np.random.default_rng(seed)
    boots_list = []
    chunk = 500
    for c0 in range(0, n_boot, chunk):
        m = min(chunk, n_boot - c0)
        idx = rng.integers(0, n, size=(m, n))
        boots_list.append(_batch_rho(ax[idx], ay[idx]))
    boots = np.concatenate(boots_list, axis=0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": n, "rho": rho_hat, "ci95_low": float(lo), "ci95_high": float(hi)}


def _load_rep_stall_ids(repo_root: Path) -> set[str]:
    eval_csv = repo_root / "runs/dark-fibers/loopscope04/rep_public/eval.csv"
    if not eval_csv.is_file():
        return set()
    stall_ids: set[str] = set()
    with eval_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = (row.get("question") or "").replace("0", ".")
            if str(row.get("prop_solved", "")) == "0" or row.get("stratum") in ("WB-Stall_partial", "S4_stalled"):
                stall_ids.add(sha256(q.encode()).hexdigest())
    return stall_ids


def analyze_swap_receipt(path: Path, stall_ids: set[str] | None = None) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d["rows"]
    n = len(rows)

    def _col(arm: str, field: str, sub_rows: list | None = None) -> list:
        src = rows if sub_rows is None else sub_rows
        return [r["arms"][arm][field] for r in src]

    sw_l2 = _col("swapped", "first_successor_logit_l2")
    rd_l2 = _col("random_orth", "first_successor_logit_l2")
    sf_l2 = _col("self_control", "first_successor_logit_l2")
    sw_cos = _col("swapped", "first_successor_cos_to_donor")
    rd_cos = _col("random_orth", "first_successor_cos_to_donor")
    fork_dis = [r["fork_disagreement_at_patch_cells"] for r in rows]

    has_joint = "zl_only_swap" in rows[0]["arms"]
    paired_tests: dict[str, object] = {
        "step1_l2_swapped_vs_random": {
            "bootstrap_diff": paired_bootstrap_ci(sw_l2, rd_l2),
            "sign_test": exact_sign_test(sw_l2, rd_l2),
            "wilcoxon": wilcoxon_signed_rank_normal(sw_l2, rd_l2),
        },
        "step1_l2_swapped_vs_self": {
            "bootstrap_diff": paired_bootstrap_ci(sw_l2, sf_l2),
            "sign_test": exact_sign_test(sw_l2, sf_l2),
        },
        "donor_cos_swapped_vs_random": {
            "swapped_mean_ci": paired_bootstrap_ci(sw_cos),
            "random_mean_ci": paired_bootstrap_ci(rd_cos),
            "bootstrap_diff": paired_bootstrap_ci(sw_cos, rd_cos),
            "sign_test": exact_sign_test(sw_cos, rd_cos),
            "wilcoxon": wilcoxon_signed_rank_normal(sw_cos, rd_cos),
        },
        "exact_s32_swapped_vs_clean": exact_mcnemar(_col("swapped", "exact"), _col("clean", "exact")),
        "exact_s32_random_vs_clean": exact_mcnemar(_col("random_orth", "exact"), _col("clean", "exact")),
        "exact_s32_swapped_vs_random": exact_mcnemar(_col("swapped", "exact"), _col("random_orth", "exact")),
        "cycle_swapped_vs_clean": exact_mcnemar(
            _col("swapped", "oscillatory_limit_cycle"), _col("clean", "oscillatory_limit_cycle")
        ),
        "spearman_fork_disagreement_vs_swapped_l2": spearman_corr(fork_dis, sw_l2),
    }

    diverged_idx = [i for i, fd in enumerate(fork_dis) if fd > 0]
    paired_tests["diverged_forks_subset"] = {
        "n_diverged": len(diverged_idx),
        "n_identical": n - len(diverged_idx),
        "mean_swapped_cos_diverged": float(np.mean([sw_cos[i] for i in diverged_idx])) if diverged_idx else None,
        "mean_random_cos_diverged": float(np.mean([rd_cos[i] for i in diverged_idx])) if diverged_idx else None,
    }

    if has_joint:
        zl_cos = _col("zl_only_swap", "first_successor_cos_to_donor")
        jt_cos = _col("swapped_joint_zl", "first_successor_cos_to_donor")
        paired_tests["donor_cos_zl_only_mean_ci"] = paired_bootstrap_ci(zl_cos)
        paired_tests["donor_cos_joint_zl_mean_ci"] = paired_bootstrap_ci(jt_cos)
        paired_tests["donor_cos_joint_vs_zh_only"] = {
            "bootstrap_diff": paired_bootstrap_ci(jt_cos, sw_cos),
            "sign_test": exact_sign_test(jt_cos, sw_cos),
            "wilcoxon": wilcoxon_signed_rank_normal(jt_cos, sw_cos),
        }
        paired_tests["donor_cos_joint_vs_random"] = {
            "bootstrap_diff": paired_bootstrap_ci(jt_cos, rd_cos),
            "sign_test": exact_sign_test(jt_cos, rd_cos),
            "wilcoxon": wilcoxon_signed_rank_normal(jt_cos, rd_cos),
        }
        paired_tests["exact_s32_joint_vs_clean"] = exact_mcnemar(
            _col("swapped_joint_zl", "exact"), _col("clean", "exact")
        )
        paired_tests["exact_s32_zl_only_vs_clean"] = exact_mcnemar(
            _col("zl_only_swap", "exact"), _col("clean", "exact")
        )
        if diverged_idx:
            paired_tests["diverged_forks_subset"]["mean_zl_only_cos_diverged"] = float(
                np.mean([zl_cos[i] for i in diverged_idx])
            )
            paired_tests["diverged_forks_subset"]["mean_joint_cos_diverged"] = float(
                np.mean([jt_cos[i] for i in diverged_idx])
            )

    if stall_ids:
        st_rows = [r for r in rows if r["case_id"] in stall_ids]
        if st_rows:
            st_sw_cos = _col("swapped", "first_successor_cos_to_donor", st_rows)
            st_rd_cos = _col("random_orth", "first_successor_cos_to_donor", st_rows)
            wb_stall_tests: dict[str, object] = {
                "n": len(st_rows),
                "swapped_cos_ci": paired_bootstrap_ci(st_sw_cos),
                "random_cos_ci": paired_bootstrap_ci(st_rd_cos),
                "swapped_vs_random_diff_ci": paired_bootstrap_ci(st_sw_cos, st_rd_cos),
                "swapped_vs_random_sign": exact_sign_test(st_sw_cos, st_rd_cos),
                "swapped_vs_random_wilcoxon": wilcoxon_signed_rank_normal(st_sw_cos, st_rd_cos),
                "exact_s32_swapped_vs_clean": exact_mcnemar(
                    _col("swapped", "exact", st_rows), _col("clean", "exact", st_rows)
                ),
            }
            if has_joint:
                st_zl_cos = _col("zl_only_swap", "first_successor_cos_to_donor", st_rows)
                st_jt_cos = _col("swapped_joint_zl", "first_successor_cos_to_donor", st_rows)
                wb_stall_tests["zl_only_cos_ci"] = paired_bootstrap_ci(st_zl_cos)
                wb_stall_tests["joint_zl_cos_ci"] = paired_bootstrap_ci(st_jt_cos)
                wb_stall_tests["joint_vs_random_diff_ci"] = paired_bootstrap_ci(st_jt_cos, st_rd_cos)
                wb_stall_tests["joint_vs_random_sign"] = exact_sign_test(st_jt_cos, st_rd_cos)
                wb_stall_tests["joint_vs_zh_only_sign"] = exact_sign_test(st_jt_cos, st_sw_cos)
            paired_tests["wb_stall_paired"] = wb_stall_tests

    max_pre = max(
        max(r["arms"][a]["pre_intervention_full_head_max_error"] for a in r["arms"])
        for r in rows
    )
    max_cell_l2 = max(r["certificates"]["max_cell_l2_error"] for r in rows)

    return {
        "receipt_file": path.name,
        "receipt_sha256": sha256_file(path),
        "checkpoint_tag": d["checkpoint_tag"],
        "checkpoint_sha256": d["checkpoint_sha256"],
        "code_sha256": d["code_sha256"],
        "n_eval": n,
        "test_sha256": d["audit"]["test_sha256"],
        "patch_step": d["settings"]["patch_step"],
        "horizon": d["settings"]["horizon"],
        "sigma": d["settings"]["sigma"],
        "elapsed_seconds": d["elapsed_seconds"],
        "peak_vram_mib": round(d.get("peak_vram_bytes", 0) / (1024**2), 2),
        "max_pre_head_error": float(max_pre),
        "max_cell_l2_error": float(max_cell_l2),
        "stratified_summary": d["summary"],
        "paired_tests": paired_tests,
    }


def analyze_cache_receipt(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    ridge = d["cache"]
    delta_ridge = {
        "r2_overall": float(ridge["visible_trace"]["r2"] - ridge["snapshot"]["r2"]),
        "r2_zh_orth": float(ridge["visible_trace"]["r2_zh_orth"] - ridge["snapshot"]["r2_zh_orth"]),
        "r2_zl": float(ridge["visible_trace"]["r2_zl"] - ridge["snapshot"]["r2_zl"]),
        "mse_reduction": float(ridge["snapshot"]["mse"] - ridge["visible_trace"]["mse"]),
    }
    out = {
        "receipt_file": path.name,
        "receipt_sha256": sha256_file(path),
        "checkpoint_tag": d["checkpoint_tag"],
        "checkpoint_sha256": d["checkpoint_sha256"],
        "code_sha256": d["code_sha256"],
        "n_train": d["audit"]["train"],
        "n_eval": d["audit"]["test"],
        "train_sha256": d["audit"]["train_sha256"],
        "test_sha256": d["audit"]["test_sha256"],
        "steps": d["settings"]["steps"],
        "horizon": d["settings"]["horizon"],
        "alpha": d["settings"]["alpha"],
        "elapsed_seconds": d["elapsed_seconds"],
        "peak_vram_mib": round(d.get("peak_vram_bytes", 0) / (1024**2), 2),
        "cellwise_ridge": ridge,
        "cellwise_ridge_delta_trace_minus_snapshot": delta_ridge,
    }
    if "full_97_refill" in d:
        out["full_97_refill"] = d["full_97_refill"]
    return out


def analyze_gramian_receipt(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d["rows"]
    n = len(rows)
    eh = rows[0].get("eval_horizon", d["settings"]["eval_horizon"])
    kh = min(rows[0]["horizon"], eh)
    top_eigs = [float(r["eigenvalues_descending"][0]) for r in rows]

    angle_tests = {}
    if "causal_direction_tests" in rows[0]:
        for ak in sorted(rows[0]["causal_direction_tests"].keys(), key=lambda s: float(s.split("_")[1])):
            hi_cum = [r["causal_direction_tests"][ak]["directions"]["high_eig"][f"cum_h{kh}_rms_readout_l2"] for r in rows]
            lo_cum = [r["causal_direction_tests"][ak]["directions"]["low_eig"][f"cum_h{kh}_rms_readout_l2"] for r in rows]
            rd_cum = [r["causal_direction_tests"][ak]["directions"]["random_dir"][f"cum_h{kh}_rms_readout_l2"] for r in rows]
            hi_flips = [r["causal_direction_tests"][ak]["directions"]["high_eig"][f"board_flips_vs_clean_at_h{eh}"] for r in rows]
            lo_flips = [r["causal_direction_tests"][ak]["directions"]["low_eig"][f"board_flips_vs_clean_at_h{eh}"] for r in rows]
            hi_ex = [r["causal_direction_tests"][ak]["directions"]["high_eig"][f"exact_solve_at_h{eh}"] for r in rows]
            lo_ex = [r["causal_direction_tests"][ak]["directions"]["low_eig"][f"exact_solve_at_h{eh}"] for r in rows]
            cl_ex = [r["clean_exact_solve_at_eval_horizon"] for r in rows]

            angle_tests[ak] = {
                "cum_rms_high_vs_low": {
                    "high_mean_ci": paired_bootstrap_ci(hi_cum),
                    "low_mean_ci": paired_bootstrap_ci(lo_cum),
                    "diff_ci": paired_bootstrap_ci(hi_cum, lo_cum),
                    "ratio_of_means": float(np.mean(hi_cum) / max(np.mean(lo_cum), 1e-9)),
                    "sign_test": exact_sign_test(hi_cum, lo_cum),
                    "wilcoxon": wilcoxon_signed_rank_normal(hi_cum, lo_cum),
                },
                "cum_rms_high_vs_random": {
                    "random_mean_ci": paired_bootstrap_ci(rd_cum),
                    "diff_ci": paired_bootstrap_ci(hi_cum, rd_cum),
                    "ratio_of_means": float(np.mean(hi_cum) / max(np.mean(rd_cum), 1e-9)),
                    "sign_test": exact_sign_test(hi_cum, rd_cum),
                    "wilcoxon": wilcoxon_signed_rank_normal(hi_cum, rd_cum),
                },
                "board_flips_high_vs_low_sign": exact_sign_test(hi_flips, lo_flips),
                "exact_solve_high_vs_clean": exact_mcnemar(hi_ex, cl_ex),
                "exact_solve_low_vs_clean": exact_mcnemar(lo_ex, cl_ex),
                "exact_solve_high_vs_low": exact_mcnemar(hi_ex, lo_ex),
                "spearman_top_eig_vs_high_cum_rms": spearman_corr(top_eigs, hi_cum),
            }

    return {
        "receipt_file": path.name,
        "receipt_sha256": sha256_file(path),
        "checkpoint_tag": d["checkpoint_tag"],
        "checkpoint_sha256": d["checkpoint_sha256"],
        "code_sha256": d["code_sha256"],
        "n_eval": n,
        "test_sha256": d["audit"]["test_sha256"],
        "patch_step": d["settings"]["patch_step"],
        "probes": d["settings"]["probes"],
        "gramian_horizon": d["settings"]["gramian_horizon"],
        "eval_horizon": eh,
        "elapsed_seconds": d["elapsed_seconds"],
        "peak_vram_mib": round(d.get("peak_vram_bytes", 0) / (1024**2), 2),
        "stratified_summary": d["summary"],
        "paired_angle_tests": angle_tests,
    }


def _summarize_arg03_tape(tape_path: Path) -> dict:
    t_rows = [json.loads(ln) for ln in tape_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not t_rows:
        return {"n_rows": 0}
    max_leak = max(r["max_precontinuation_logprob_error"] for r in t_rows)
    max_leak_r = max(r["max_precontinuation_random_logprob_error"] for r in t_rows)
    max_leak_s = max(r["max_precontinuation_self_logprob_error"] for r in t_rows)
    tape_stats: dict[str, object] = {
        "n_rows": len(t_rows),
        "max_precontinuation_donor_logprob_error": float(max_leak),
        "max_precontinuation_random_logprob_error": float(max_leak_r),
        "max_precontinuation_self_logprob_error": float(max_leak_s),
        "mean_case_elapsed_seconds": float(np.mean([r.get("case_elapsed_seconds", 0.0) for r in t_rows])),
        "total_case_elapsed_seconds": float(np.sum([r.get("case_elapsed_seconds", 0.0) for r in t_rows])),
        "postloop_readout_distance_by_depth": {},
        "hidden_perturbation_norms_by_depth": {},
        "ranking_changes_by_depth": {},
    }
    for d_str in ("1", "2", "4", "8"):
        if d_str in t_rows[0].get("hidden_state_norms", {}):
            tape_stats["hidden_perturbation_norms_by_depth"][d_str] = {
                "mean_clean_l2": float(np.mean([r["hidden_state_norms"][d_str]["clean_l2"] for r in t_rows])),
                "mean_context_swap_delta_l2": float(
                    np.mean([r["hidden_state_norms"][d_str]["context_swap_delta_l2"] for r in t_rows])
                ),
                "mean_random_context_delta_l2": float(
                    np.mean([r["hidden_state_norms"][d_str]["random_context_delta_l2"] for r in t_rows])
                ),
                "mean_donor_full_delta_l2": float(
                    np.mean([r["hidden_state_norms"][d_str]["donor_full_delta_l2"] for r in t_rows])
                ),
            }
        if d_str in t_rows[0].get("postloop_readout_distance", {}):
            d_sw = [r["postloop_readout_distance"][d_str] for r in t_rows]
            d_rd = [r["postloop_random_readout_distance"][d_str] for r in t_rows]
            tape_stats["postloop_readout_distance_by_depth"][d_str] = {
                "context_swap_mean_ci": paired_bootstrap_ci(d_sw),
                "random_context_mean_ci": paired_bootstrap_ci(d_rd),
                "swap_minus_random_diff_ci": paired_bootstrap_ci(d_sw, d_rd),
                "swap_vs_random_sign_test": exact_sign_test(d_sw, d_rd),
                "swap_vs_random_wilcoxon": wilcoxon_signed_rank_normal(d_sw, d_rd),
            }
            rk_depth: dict[str, object] = {}
            for arm_k in ("context_swap", "random_context_norm_matched", "open_loop"):
                if arm_k not in t_rows[0]["ranking_changes_vs_clean"].get(d_str, {}):
                    continue
                top1_c = [r["ranking_changes_vs_clean"][d_str][arm_k]["top1_changed"] for r in t_rows]
                ov12 = [r["ranking_changes_vs_clean"][d_str][arm_k]["top12_overlap"] for r in t_rows]
                sp_r = [r["ranking_changes_vs_clean"][d_str][arm_k]["spearman_rho"] for r in t_rows]
                kl_v = [r["ranking_changes_vs_clean"][d_str][arm_k]["kl_a_to_b"] for r in t_rows]
                rk_depth[arm_k] = {
                    "top1_changed_rate": float(np.mean(top1_c)),
                    "mean_top12_overlap": float(np.mean(ov12)),
                    "spearman_rho_mean_ci": paired_bootstrap_ci(sp_r),
                    "kl_clean_to_arm_mean_ci": paired_bootstrap_ci(kl_v),
                }
            sp_sw = [r["ranking_changes_vs_clean"][d_str]["context_swap"]["spearman_rho"] for r in t_rows]
            sp_rd = [
                r["ranking_changes_vs_clean"][d_str]["random_context_norm_matched"]["spearman_rho"]
                for r in t_rows
            ]
            kl_sw = [r["ranking_changes_vs_clean"][d_str]["context_swap"]["kl_a_to_b"] for r in t_rows]
            kl_rd = [
                r["ranking_changes_vs_clean"][d_str]["random_context_norm_matched"]["kl_a_to_b"]
                for r in t_rows
            ]
            rk_depth["swap_vs_random_spearman_sign"] = exact_sign_test(sp_sw, sp_rd)
            rk_depth["swap_vs_random_kl_sign"] = exact_sign_test(kl_sw, kl_rd)
            tape_stats["ranking_changes_by_depth"][d_str] = rk_depth
    return tape_stats


def _sanitize_manifest_paths(obj):
    """Replace machine-local filesystem paths with portable basenames while preserving digests."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            clean_k = Path(k).name if ("\\" in str(k) or "/" in str(k)) and ":" in str(k) else k
            if k in ("source_csv", "source_runner") and isinstance(v, str) and ("\\" in v or "/" in v):
                out[clean_k] = Path(v).name
            elif k == "excluded_sha256" and isinstance(v, dict):
                out[clean_k] = {Path(pk).name: pv for pk, pv in v.items()}
            else:
                out[clean_k] = _sanitize_manifest_paths(v)
        return out
    if isinstance(obj, list):
        return [_sanitize_manifest_paths(x) for x in obj]
    return obj


def analyze_arg03(arg03_root: Path) -> dict:
    out: dict[str, object] = {}
    audit_p = arg03_root / "public/audit.json"
    if audit_p.is_file():
        out["roster_audit"] = _sanitize_manifest_paths(json.loads(audit_p.read_text(encoding="utf-8")))

    smoke_grade_p = arg03_root / "dev_smoke4/grade_smoke4.json"
    smoke_manifest_p = arg03_root / "dev_smoke4/manifest.json"
    smoke_tape_p = arg03_root / "dev_smoke4/answer_tape.jsonl"
    if smoke_grade_p.is_file() and smoke_manifest_p.is_file():
        smoke_m = _sanitize_manifest_paths(json.loads(smoke_manifest_p.read_text(encoding="utf-8")))
        out["dev_smoke4"] = {
            "manifest": smoke_m,
            "peak_vram_mib": round(smoke_m.get("peak_allocated_vram_bytes", 0) / (1024**2), 2),
            "graded": json.loads(smoke_grade_p.read_text(encoding="utf-8")),
            "tape_diagnostics": _summarize_arg03_tape(smoke_tape_p) if smoke_tape_p.is_file() else {},
        }

    dev_grade_p = arg03_root / "dev_full96/grade_dev96.json"
    dev_manifest_p = arg03_root / "dev_full96/manifest.json"
    dev_tape_p = arg03_root / "dev_full96/answer_tape.jsonl"
    vram_gate_p = arg03_root / "vram_gate_status.json"

    if dev_grade_p.is_file() and dev_manifest_p.is_file():
        manifest = _sanitize_manifest_paths(json.loads(dev_manifest_p.read_text(encoding="utf-8")))
        graded = json.loads(dev_grade_p.read_text(encoding="utf-8"))
        out["dev_full96"] = {
            "manifest": manifest,
            "peak_vram_mib": round(manifest.get("peak_allocated_vram_bytes", 0) / (1024**2), 2),
            "graded": graded,
            "tape_diagnostics": _summarize_arg03_tape(dev_tape_p) if dev_tape_p.is_file() else {},
        }
    elif vram_gate_p.is_file():
        out["vram_gate_status"] = _sanitize_manifest_paths(json.loads(vram_gate_p.read_text(encoding="utf-8")))

    test_dirs = [p.name for p in arg03_root.glob("*test*") if p.is_dir()]
    out["heldout_test_seal_verified"] = len(test_dirs) == 0
    return out


def build_campaign_summary(repo_root: Path) -> dict:
    ls_res = repo_root / "runs/dark-fibers/loopscope04/results"
    arg03_root = repo_root / "runs/dark-fibers/arg03"
    stall_ids = _load_rep_stall_ids(repo_root)

    swaps, caches, gramians = {}, {}, {}
    total_gpu_sec = 0.0
    max_trm_vram_mib = 0.0
    if ls_res.is_dir():
        for p in sorted(ls_res.glob("*.json")):
            d_head = json.loads(p.read_text(encoding="utf-8"))
            mode = d_head.get("mode")
            total_gpu_sec += float(d_head.get("elapsed_seconds", 0.0))
            max_trm_vram_mib = max(max_trm_vram_mib, d_head.get("peak_vram_bytes", 0) / (1024**2))
            stem = p.stem
            if mode == "swap":
                swaps[stem] = analyze_swap_receipt(p, stall_ids if "_rep_" in stem else None)
            elif mode == "cache":
                caches[stem] = analyze_cache_receipt(p)
            elif mode == "gramian":
                gramians[stem] = analyze_gramian_receipt(p)

    pilot_meta = repo_root / "runs/dark-fibers/loopscope04/pilot_public/roster_manifest.json"
    rep_meta = repo_root / "runs/dark-fibers/loopscope04/rep_public/roster_manifest.json"

    return {
        "schema": "darkfibers.campaign_summary.20261115.v1",
        "loopscope04": {
            "n_receipts": len(swaps) + len(caches) + len(gramians),
            "total_model_elapsed_seconds": round(total_gpu_sec, 2),
            "max_peak_vram_mib": round(max_trm_vram_mib, 2),
            "pilot_dataset_metadata": _sanitize_manifest_paths(json.loads(pilot_meta.read_text(encoding="utf-8"))) if pilot_meta.is_file() else None,
            "rep_dataset_metadata": _sanitize_manifest_paths(json.loads(rep_meta.read_text(encoding="utf-8"))) if rep_meta.is_file() else None,
            "exp_3a_swaps": swaps,
            "exp_3b_caches": caches,
            "exp_3c_gramians": gramians,
        },
        "arg03_nanbeige": analyze_arg03(arg03_root),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("papers/shared_weights_dark_fibers/campaign_20261115/campaign_summary.json"),
    )
    args = p.parse_args(argv)
    summary = build_campaign_summary(args.repo_root.resolve())
    write_json(args.out, summary)
    print(
        f"wrote {args.out} (loopscope04 receipts={summary['loopscope04']['n_receipts']}, "
        f"heldout_test_sealed={summary['arg03_nanbeige']['heldout_test_seal_verified']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
