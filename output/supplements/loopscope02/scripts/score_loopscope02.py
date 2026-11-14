"""Pure-CPU deterministic replay scorer for LOOPSCOPE-02.

Reads `benchmark_bundle.json`, `trm_results.json`, `nanbeige_results.json`,
`qwen4b_results.json` (plus optional `qwen3_4b_inst_results.json`), verifies
`bundle_sha256` alignment across all receipts, computes exact two-sided McNemar
p-values and 2,000-resample paired bootstrap 95% CIs (seed 20261113), and scores
pre-registered hypotheses P1-P5.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mechinterp.loopscope02.scripts import sudoku_prop as SP  # noqa: E402

BOOTSTRAP_SEED = 20261113
BOOTSTRAP_RESAMPLES = 2000


def exact_mcnemar_p(a: Sequence[int | bool], b: Sequence[int | bool]) -> float:
    """Exact two-sided binomial McNemar p-value on paired binary outcomes."""
    arr_a = np.asarray(a, dtype=bool)
    arr_b = np.asarray(b, dtype=bool)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"shape mismatch in McNemar: {arr_a.shape} vs {arr_b.shape}")
    n10 = int(np.sum(arr_a & ~arr_b))
    n01 = int(np.sum(~arr_a & arr_b))
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    log2_n = n * math.log(2.0)
    tail = 0.0
    for i in range(k + 1):
        log_c = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        tail += math.exp(log_c - log2_n)
    return float(min(1.0, 2.0 * tail))


def bootstrap_mean_ci(
    vals: Sequence[float | int],
    rng: np.random.Generator,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    n = arr.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(arr.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi)}


def paired_diff_summary(
    a: Sequence[int | bool],
    b: Sequence[int | bool],
    rng: np.random.Generator,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float]:
    """Summary of (a - b) on paired binary outcomes with bootstrap CI and exact McNemar p."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    diff = arr_a - arr_b
    ci = bootstrap_mean_ci(diff, rng, n_resamples=n_resamples)
    return {
        "mean_a": float(arr_a.mean()),
        "mean_b": float(arr_b.mean()),
        "diff": ci["mean"],
        "ci95_lo": ci["ci95_lo"],
        "ci95_hi": ci["ci95_hi"],
        "mcnemar_p": exact_mcnemar_p(a, b),
    }


def stratum_breakdown(
    correct: Sequence[int | bool],
    strata_labels: Sequence[str],
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    arr = np.asarray(correct, dtype=np.float64)
    out: dict[str, dict[str, float]] = {"overall": bootstrap_mean_ci(arr, rng)}
    for s in SP.STRATA:
        mask = np.array([lbl == s for lbl in strata_labels], dtype=bool)
        if np.any(mask):
            out[s] = bootstrap_mean_ci(arr[mask], rng)
    multistep_mask = np.array([lbl != "S1_naked1" for lbl in strata_labels], dtype=bool)
    if np.any(multistep_mask):
        out["multistep_S2_S4"] = bootstrap_mean_ci(arr[multistep_mask], rng)
    return out


def score_all(
    bundle: dict,
    trm: dict,
    nanbeige: dict,
    qwen4b: dict,
    qwen3_inst: dict | None = None,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    expected_sha = bundle["bundle_sha256"]
    for tag, rec in [("trm", trm), ("nanbeige", nanbeige), ("qwen4b", qwen4b)]:
        if rec.get("bundle_sha256") != expected_sha:
            raise ValueError(f"{tag} bundle_sha256 mismatch: {rec.get('bundle_sha256')} != {expected_sha}")
    if qwen3_inst is not None and qwen3_inst.get("bundle_sha256") != expected_sha:
        raise ValueError("qwen3_inst bundle_sha256 mismatch")

    rng = np.random.default_rng(seed)
    items = bundle["items"]
    strata = [it["stratum"] for it in items]
    multistep_idx = [i for i, s in enumerate(strata) if s != "S1_naked1"]

    trm_s1 = trm["items_target_correct_by_step"][0]
    trm_s16 = trm["items_target_correct_by_step"][-1]
    nb_t1 = nanbeige["items_target_correct_by_loop"][0]
    nb_t2 = nanbeige["items_target_correct_by_loop"][1]
    nb_tmax = nanbeige["items_target_correct_by_loop"][-1]
    qw4_fin = qwen4b["items_target_correct"]
    qw4_mid = qwen4b["items_mid_lens_correct"]

    accuracy_tables = {
        "trm_step1": stratum_breakdown(trm_s1, strata, rng),
        "trm_step16": stratum_breakdown(trm_s16, strata, rng),
        "nanbeige_loop1": stratum_breakdown(nb_t1, strata, rng),
        "nanbeige_loop2": stratum_breakdown(nb_t2, strata, rng),
        "nanbeige_loop_max": stratum_breakdown(nb_tmax, strata, rng),
        "qwen4b_final": stratum_breakdown(qw4_fin, strata, rng),
        "qwen4b_mid_lens": stratum_breakdown(qw4_mid, strata, rng),
    }
    if qwen3_inst is not None:
        accuracy_tables["qwen3_4b_inst_final"] = stratum_breakdown(qwen3_inst["items_target_correct"], strata, rng)

    # Key paired comparisons
    nb_vs_qw_all = paired_diff_summary(nb_t2, qw4_fin, rng)
    nb_vs_qw_multistep = paired_diff_summary(
        [nb_t2[i] for i in multistep_idx],
        [qw4_fin[i] for i in multistep_idx],
        rng,
    )
    nb_t2_vs_t1 = paired_diff_summary(nb_t2, nb_t1, rng)
    nb_tmax_vs_t2 = paired_diff_summary(nb_tmax, nb_t2, rng)
    trm_s16_vs_s1 = paired_diff_summary(trm_s16, trm_s1, rng)
    trm_vs_nb = paired_diff_summary(trm_s16, nb_t2, rng)
    trm_vs_qw = paired_diff_summary(trm_s16, qw4_fin, rng)

    # Counterfactual comparisons
    trm_vs_open = paired_diff_summary(trm_s16, trm["items_target_open_correct"], rng)
    trm_vs_donor = paired_diff_summary(trm_s16, trm["items_target_donor_correct"], rng)
    trm_board_s16 = trm["items_board_exact_by_step"][-1]
    trm_board_repair = paired_diff_summary(
        trm["items_board_exact_corrupt_repaired"],
        trm["items_board_exact_corrupt_step1"],
        rng,
    )
    trm_target_repair = paired_diff_summary(
        trm["items_target_corrupt_repaired_correct"],
        trm["items_target_corrupt_step1_correct"],
        rng,
    )
    trm_board_reset_zl = paired_diff_summary(trm_board_s16, trm["items_board_exact_reset_zL"], rng)
    trm_board_fiber = paired_diff_summary(trm_board_s16, trm["items_board_exact_fiber_erased"], rng)
    bench_reset_zl = paired_diff_summary(trm["bench100_clean16_exact"], trm["bench100_reset_zL_exact"], rng)
    bench_fiber = paired_diff_summary(trm["bench100_clean16_exact"], trm["bench100_fiber_erased_exact"], rng)

    nb_vs_open2 = paired_diff_summary(nb_t2, nanbeige["items_open2_correct"], rng)
    nb_vs_donor2 = paired_diff_summary(nb_t2, nanbeige["items_donor2_correct"], rng)
    nb_vs_patch2 = paired_diff_summary(nb_t2, nanbeige["items_patch_repaired2_correct"], rng)

    # P1 verdict: Capability scaling triangulation
    p1a_no_extrap = nb_tmax_vs_t2["diff"] <= 0.02
    p1a_llm_multistep_ceiling = nb_vs_qw_multistep["diff"] < 0.10
    p1b_trm_gain = trm_s16_vs_s1["diff"] >= 0.25
    best_llm_acc = max(nb_vs_qw_all["mean_a"], nb_vs_qw_all["mean_b"])
    p1b_trm_margin_vs_llm = (trm_s16_vs_s1["mean_a"] - best_llm_acc) >= 0.35
    p1b_sig = max(trm_vs_nb["mcnemar_p"], trm_vs_qw["mcnemar_p"]) < 1e-6

    if p1a_no_extrap and p1a_llm_multistep_ceiling and p1b_trm_gain and p1b_trm_margin_vs_llm and p1b_sig:
        p1_verdict = "Supported"
    elif trm_s16_vs_s1["diff"] >= 0.15 and (trm_s16_vs_s1["mean_a"] - best_llm_acc) >= 0.25:
        p1_verdict = "Weak"
    else:
        p1_verdict = "Not supported"

    # P2 verdict: ARG-02 causal counterfactuals & perturbation repair
    kick_final = float(trm["kick_ratio_curve_mean"][-1])
    trm_cf_ok = (
        trm_vs_open["diff"] >= 0.30
        and trm_vs_open["mcnemar_p"] < 1e-6
        and trm_vs_donor["diff"] >= 0.30
        and trm_vs_donor["mcnemar_p"] < 1e-6
    )
    trm_repair_ok = (
        trm_board_repair["diff"] >= 0.15
        or trm_target_repair["diff"] >= 0.15
        or kick_final < 1.0
    )
    nb_rel_norm_last = float(nanbeige["rel_step_norm_mean_by_loop"][-1])
    nb_non_convergent = (nb_rel_norm_last >= 0.10) and (nb_tmax_vs_t2["diff"] <= 0.02)

    if trm_cf_ok and trm_repair_ok and nb_non_convergent:
        p2_verdict = "Supported"
    elif trm_vs_open["diff"] >= 0.20 and trm_vs_donor["diff"] >= 0.20 and trm_repair_ok:
        p2_verdict = "Weak"
    else:
        p2_verdict = "Not supported"

    # P3 verdict: SVD weight-usage correlations (Nanbeige orthogonal pass multiplexing vs TRM outer reuse + role split)
    nb_mean_r = float(nanbeige["mean_r_pass1_vs_pass2"])
    trm_u = trm["usage_stats"]
    trm_r_outer_zh = float(np.mean([v["r_outer_zH_mean"] for v in trm_u.values()]))
    trm_r_outer_zl = float(np.mean([v["r_outer_zL_mean"] for v in trm_u.values()]))
    trm_r_role = float(np.mean([v["r_role_zL_vs_zH_mean"] for v in trm_u.values()]))

    if nb_mean_r < 0.30 and trm_r_outer_zh >= 0.80 and trm_r_role <= (trm_r_outer_zh - 0.10):
        p3_verdict = "Supported"
    elif nb_mean_r < 0.40 and trm_r_outer_zh >= 0.70 and trm_r_role < (trm_r_outer_zh - 0.05):
        p3_verdict = "Weak"
    else:
        p3_verdict = "Not supported"

    # P4 verdict: Unrolled TCA attractor masking (F1/F6 <= 0.25) + specificity (> 5x 0.27% Gaussian null = 1.35%)
    trm_f1_flate = float(trm["tca"]["f1_over_flate_mean"])
    trm_max_spec = max(float(trm["tca"]["zr_gt3_overall"]), float(trm["tca"]["zw_gt3_overall"]))
    if trm_f1_flate <= 0.25 and trm_max_spec >= 0.0135:
        p4_verdict = "Supported"
    elif trm_f1_flate <= 0.40 and trm_max_spec >= 0.0081:
        p4_verdict = "Weak"
    else:
        p4_verdict = "Not supported"

    # P5 verdict: Q11 G_prop compactification, deductive staircase, stalled non-monotone repair, and fiber carry
    b_comp = trm["bench100_compactification"]
    i_comp = trm["items_compactification"]
    auc_comm = b_comp["commutation_vs_shuffled_auc_prop"]
    concord = i_comp["within_board_round_concordance"]
    concord_null = i_comp["within_board_round_concordance_shuf_null"]
    stalled_solve = float(b_comp["stalled_solve_rate_step16"])
    stalled_nonmono = float(b_comp["stalled_solved_with_nonmonotone_flips_share"])

    # Dark-state carry test: either reset_zL OR full 501-dim orthogonal fiber erasure degrades solve rate >= 0.05 (p < 0.05)
    dark_carry_drop = max(
        bench_reset_zl["diff"],
        trm_board_reset_zl["diff"],
        bench_fiber["diff"],
        trm_board_fiber["diff"],
    )
    dark_carry_sig = (
        (bench_reset_zl["diff"] >= 0.05 and bench_reset_zl["mcnemar_p"] < 0.05)
        or (trm_board_reset_zl["diff"] >= 0.05 and trm_board_reset_zl["mcnemar_p"] < 0.05)
        or (bench_fiber["diff"] >= 0.05 and bench_fiber["mcnemar_p"] < 0.05)
        or (trm_board_fiber["diff"] >= 0.05 and trm_board_fiber["mcnemar_p"] < 0.05)
    )

    p5_checks = {
        "commutation_auc_ge_090": bool(auc_comm is not None and auc_comm >= 0.90),
        "within_board_staircase_concordance": bool(
            concord is not None and concord_null is not None and concord >= 0.65 and (concord - concord_null) >= 0.10
        ),
        "stalled_solve_ge_040": stalled_solve >= 0.40,
        "stalled_nonmonotone_ge_080": stalled_nonmono >= 0.80,
        "dark_fiber_or_zL_necessity": bool(dark_carry_sig),
    }
    p5_pass_count = sum(int(v) for v in p5_checks.values())
    if p5_pass_count == 5:
        p5_verdict = "Supported"
    elif p5_pass_count == 4:
        p5_verdict = "Weak"
    else:
        p5_verdict = "Not supported"

    return {
        "bundle_sha256": expected_sha,
        "bootstrap_seed": seed,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "verdicts": {
            "P1_capability_triangulation": p1_verdict,
            "P2_counterfactual_attractor_vs_virtual_depth": p2_verdict,
            "P3_svd_operator_reuse_vs_multiplexing": p3_verdict,
            "P4_tca_attractor_masking_and_specificity": p4_verdict,
            "P5_q11_compactification_and_beyond_prop_search": p5_verdict,
        },
        "accuracy_by_stratum": accuracy_tables,
        "paired_comparisons": {
            "nanbeige_loop2_vs_qwen4b_all": nb_vs_qw_all,
            "nanbeige_loop2_vs_qwen4b_multistep": nb_vs_qw_multistep,
            "nanbeige_loop2_vs_loop1": nb_t2_vs_t1,
            "nanbeige_loopmax_vs_loop2": nb_tmax_vs_t2,
            "trm_step16_vs_step1": trm_s16_vs_s1,
            "trm_step16_vs_nanbeige_loop2": trm_vs_nb,
            "trm_step16_vs_qwen4b": trm_vs_qw,
            "trm_clean16_vs_open16": trm_vs_open,
            "trm_clean16_vs_donor_zH16": trm_vs_donor,
            "trm_board_corrupt_repaired_vs_step1": trm_board_repair,
            "trm_target_corrupt_repaired_vs_step1": trm_target_repair,
            "trm_items_board_clean16_vs_reset_zL": trm_board_reset_zl,
            "trm_items_board_clean16_vs_fiber_erased": trm_board_fiber,
            "trm_bench100_clean16_vs_reset_zL": bench_reset_zl,
            "trm_bench100_clean16_vs_fiber_erased": bench_fiber,
            "nanbeige_loop2_vs_open2": nb_vs_open2,
            "nanbeige_loop2_vs_donor2": nb_vs_donor2,
            "nanbeige_loop2_vs_patch_repaired2": nb_vs_patch2,
        },
        "mechanistic_metrics": {
            "nanbeige_mean_r_pass1_vs_pass2": nb_mean_r,
            "nanbeige_rel_step_norm_final": nb_rel_norm_last,
            "nanbeige_donor2_top1_agree_loop2": float(np.mean(nanbeige["items_donor2_top1_agree_with_loop2"])),
            "nanbeige_patch2_top1_agree_loop2": float(np.mean(nanbeige["items_patch_repaired2_top1_agree_with_loop2"])),
            "trm_r_outer_zH_mean": trm_r_outer_zh,
            "trm_r_outer_zL_mean": trm_r_outer_zl,
            "trm_r_role_zL_vs_zH_mean": trm_r_role,
            "trm_kick_ratio_final": kick_final,
            "trm_tca_f1_over_flate_mean": trm_f1_flate,
            "trm_tca_max_spec_gt3": trm_max_spec,
            "nanbeige_tca_s1": nanbeige["tca_s1"],
            "qwen4b_tca_s1": qwen4b["tca_s1"],
            "trm_bench100_exact_step1": float(b_comp["exact_solve_curve_all"][0]),
            "trm_bench100_exact_step16": float(b_comp["exact_solve_curve_all"][-1]),
            "trm_replication_bench100_exact_step16": float(trm["replication_bench100_exact_step16"]),
            "p5_subchecks": p5_checks,
            "p5_max_dark_carry_drop": float(dark_carry_drop),
            "bench100_compactification": b_comp,
            "items_compactification": i_comp,
        },
    }


def render_markdown_summary(rep: dict) -> str:
    v = rep["verdicts"]
    acc = rep["accuracy_by_stratum"]
    pc = rep["paired_comparisons"]
    mm = rep["mechanistic_metrics"]
    lines = [
        "# LOOPSCOPE-02 Automated Replay Summary",
        "",
        f"- **Bundle SHA-256:** `{rep['bundle_sha256']}`",
        f"- **Bootstrap:** `{rep['bootstrap_resamples']}` resamples (`seed={rep['bootstrap_seed']}`)",
        "",
        "## Pre-Registered Verdicts",
        "",
        "| Hypothesis | Verdict | Key Quantitative Readout |",
        "|---|---|---|",
        f"| **P1** (Capability triangulation) | **{v['P1_capability_triangulation']}** | "
        f"TRM s1->s16 `{pc['trm_step16_vs_step1']['diff']:+.3f}`; "
        f"Nanbeige t=2 vs Qwen-4B (S2-S4) `{pc['nanbeige_loop2_vs_qwen4b_multistep']['diff']:+.3f}` "
        f"(p={pc['nanbeige_loop2_vs_qwen4b_multistep']['mcnemar_p']:.3g}) |",
        f"| **P2** (ARG-02 causal counterfactuals) | **{v['P2_counterfactual_attractor_vs_virtual_depth']}** | "
        f"TRM clean-open `{pc['trm_clean16_vs_open16']['diff']:+.3f}`, "
        f"clean-donor `{pc['trm_clean16_vs_donor_zH16']['diff']:+.3f}`, "
        f"board repair `{pc['trm_board_corrupt_repaired_vs_step1']['diff']:+.3f}` |",
        f"| **P3** (SVD operator reuse vs multiplexing) | **{v['P3_svd_operator_reuse_vs_multiplexing']}** | "
        f"Nanbeige r(P1,P2)=`{mm['nanbeige_mean_r_pass1_vs_pass2']:.3f}`; "
        f"TRM r_outer(zH)=`{mm['trm_r_outer_zH_mean']:.3f}`, r_role(zL,zH)=`{mm['trm_r_role_zL_vs_zH_mean']:.3f}` |",
        f"| **P4** (TCA attractor masking & specificity) | **{v['P4_tca_attractor_masking_and_specificity']}** | "
        f"TRM F1/F6=`{mm['trm_tca_f1_over_flate_mean']:.3f}`, max(|z|>3)=`{100*mm['trm_tca_max_spec_gt3']:.2f}%` |",
        f"| **P5** (Q11 compactification & search) | **{v['P5_q11_compactification_and_beyond_prop_search']}** | "
        f"Comm AUC=`{mm['bench100_compactification']['commutation_vs_shuffled_auc_prop']:.3f}`, "
        f"staircase=`{mm['items_compactification']['within_board_round_concordance']:.3f}`, "
        f"stalled solve=`{mm['bench100_compactification']['stalled_solve_rate_step16']:.3f}` |",
        "",
        "## Stratified Target-Cell Accuracy (256 Distinct Boards)",
        "",
        "| Model / Condition | Overall | S1 (Naked-1) | S2 (Prop-2) | S3 (Prop-3+) | S4 (Stalled) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    row_order = [
        ("Qwen3.5-4B-Base (mid-stack L16 lens)", "qwen4b_mid_lens"),
        ("Qwen3.5-4B-Base (final L32)", "qwen4b_final"),
        ("Qwen3-4B-Instruct-2507 (final L36)", "qwen3_4b_inst_final"),
        ("Nanbeige4.2-3B (loop 1)", "nanbeige_loop1"),
        ("Nanbeige4.2-3B (loop 2, trained)", "nanbeige_loop2"),
        ("Nanbeige4.2-3B (loop 8, over-looped)", "nanbeige_loop_max"),
        ("TRM 7M (outer step 1)", "trm_step1"),
        ("TRM 7M (outer step 16)", "trm_step16"),
    ]
    for label, key in row_order:
        if key not in acc:
            continue
        tb = acc[key]
        lines.append(
            f"| {label} | {tb['overall']['mean']:.3f} | {tb['S1_naked1']['mean']:.3f} | "
            f"{tb['S2_prop2']['mean']:.3f} | {tb['S3_prop3plus']['mean']:.3f} | {tb['S4_stalled']['mean']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("mechinterp/loopscope02/results"))
    args = parser.parse_args()

    rdir = args.results_dir
    bundle = json.loads((rdir / "benchmark_bundle.json").read_text(encoding="utf-8"))
    trm = json.loads((rdir / "trm_results.json").read_text(encoding="utf-8"))
    nanbeige = json.loads((rdir / "nanbeige_results.json").read_text(encoding="utf-8"))
    qwen4b = json.loads((rdir / "qwen4b_results.json").read_text(encoding="utf-8"))
    inst_path = rdir / "qwen3_4b_inst_results.json"
    qwen3_inst = json.loads(inst_path.read_text(encoding="utf-8")) if inst_path.exists() else None

    rep = score_all(bundle, trm, nanbeige, qwen4b, qwen3_inst=qwen3_inst)
    (rdir / "loopscope02_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (rdir / "loopscope02_report.md").write_text(render_markdown_summary(rep), encoding="utf-8")
    print("[SCORED]", json.dumps(rep["verdicts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
