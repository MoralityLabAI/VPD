"""Independent NumPy reducer for the BPD-01 receipts.

Reads the executed JSON receipts (never the models) and writes paired per-board
statistics and Markdown tables:

- Experiment A: per-board KL for the seven registered identity controls at passes
  20, 41 and 62, the zero/pass-mean x sphere factorial on the same frozen masks
  with board-paired contrasts, and the closure-mechanism decisions.
- Experiment B: static / clean-informed / online top-k against the identity floor,
  paired online-minus-oracle differences with bootstrap CIs and sign-flip p-values,
  the reserved-64 margin statistic (descriptive at n = 16), argmax flips and mask
  disagreement, separately for CUDA BF16 and CUDA FP32.
- Experiment C (when ``gate_b4_causal_ci.json`` holds decisions): registered H1-H3
  decisions, every paired contrast, per-board KL for all arms, causal vs
  clean-informed mask disagreement and CI density.
- Experiment C2 (when ``gate_b4_causal_ci_closed_loop.json`` holds decisions): the same
  tables for the closed-loop-trained CI, plus registered H4 (closed-loop vs refit T3).

A readout is interpretable for mechanism classification only when every identity
arm in that regime has per-board KL <= 1e-5 on all boards at that readout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

READOUTS = (20, 41, 62)
CLOSURE_TOL = 1e-5
MARGIN = 0.10
SEED = 20261117
N_BOOT = 20000
N_FLIP = 100000

CONTROL_KEYS = (
    ("1_same_dtype_target_repeat", "control_0_same_dtype_target_rerun"),
    ("2_target_fp32_linear", "control_1a_target_f32_linear_vs_bf16_target"),
    ("3_full_fp32_original_weights", "control_1b_full_f32_model_vs_bf16_target"),
    ("4_fp32_bf16_quantized_weights", "control_1c_loaded_bf16_weights_full_f32"),
    ("5_svd_all_512_residual_off", "control_2a_svd_start_all_512_on"),
    ("6_svd_all_512_residual_on", "control_2a_svd_start_all_512_plus_delta"),
    ("7_trained_all_residual_on", "control_2b_trained_all_512_plus_delta_on"),
)


def paired(d: np.ndarray, *, seed: int = SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = d.size
    boots = d[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    obs = float(d.mean())
    cnt = 0
    left = N_FLIP
    while left:
        b = min(4096, left)
        s = rng.integers(0, 2, size=(b, n)) * 2 - 1
        cnt += int((np.abs((s * d[None, :]).mean(axis=1)) >= abs(obs)).sum())
        left -= b
    return {
        "mean": obs,
        "median": float(np.median(d)),
        "bootstrap_ci95": [float(lo), float(hi)],
        "signflip_two_sided_p": (1 + cnt) / (N_FLIP + 1),
        "n_negative": int((d < 0).sum()),
        "n_positive": int((d > 0).sum()),
        "per_board": [float(x) for x in d],
    }


def margin_stat(S: np.ndarray, C: np.ndarray, O: np.ndarray, *, seed: int = SEED) -> dict[str, Any]:
    """Reserved-64 statistic M = (O - C) - 0.10 S, 97.5% bootstrap upper bound."""
    M = (O - C) - MARGIN * S
    rng = np.random.default_rng(seed + 1)
    boots = M[rng.integers(0, M.size, size=(N_BOOT, M.size))].mean(axis=1)
    return {
        "mean_M": float(M.mean()),
        "upper_97_5": float(np.quantile(boots, 0.975)),
        "below_zero": bool(np.quantile(boots, 0.975) < 0),
    }


def fmt(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e3:
        return f"{x:.2e}"
    return f"{x:.4f}"


def reduce_controls(ctrl: dict[str, Any], mech: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    out: dict[str, Any] = {}
    md: list[str] = []
    for ck_json, ck in (("gaoxin492_ref", "gaoxin492"), ("yagizdevre_tgt", "yagizdevre")):
        d = ctrl[ck_json]
        rows: dict[str, Any] = {}
        md.append(f"\n### {ck}: registered identity controls, KL(native BF16 target || arm)\n")
        for tau in (20, 62):
            md.append(f"\nPass {tau}, per board (boards 0-15), mean, max, boards > 1e-3:\n")
            md.append("| Control | " + " | ".join(str(b) for b in range(16)) + " | mean | max | n>1e-3 |")
            md.append("|---|" + "---:|" * 19)
            for label, key in CONTROL_KEYS:
                pb = np.array(d[key][f"pass_{tau}"]["per_board"], dtype=np.float64)
                rows.setdefault(label, {})[f"pass_{tau}"] = pb.tolist()
                md.append(
                    f"| {label} | " + " | ".join(fmt(v) for v in pb)
                    + f" | {fmt(pb.mean())} | {fmt(pb.max())} | {int((pb > 1e-3).sum())} |"
                )
        for label, key in CONTROL_KEYS:
            rows[label]["pass_41"] = d[key]["pass_41"]["per_board"]
        fac = d["svd_top60_ablation_2x2_factorial"]
        arms = {k: {t: np.array(v[f"pass_{t}"]["per_board"]) for t in READOUTS} for k, v in fac.items()}
        z0 = "mode_zero__sphere_rescale_false"
        z1 = "mode_zero__sphere_rescale_true"
        m0 = "mode_pass_mean__sphere_rescale_false"
        m1 = "mode_pass_mean__sphere_rescale_true"
        contrasts = {}
        for t in READOUTS:
            contrasts[f"pass_{t}"] = {
                "sphere_effect_in_zero_mode": paired(arms[z1][t] - arms[z0][t]),
                "mean_fill_effect_without_sphere": paired(arms[m0][t] - arms[z0][t]),
                "sphere_effect_in_mean_mode": paired(arms[m1][t] - arms[m0][t]),
                "interaction": paired((arms[m1][t] - arms[m0][t]) - (arms[z1][t] - arms[z0][t])),
            }
        md.append(f"\n{ck}: ablation factorial on the frozen clean-informed top-307 SVD masks (mean KL)\n")
        md.append("| Arm | pass 20 | pass 41 | pass 62 |")
        md.append("|---|---:|---:|---:|")
        for k in (z0, z1, m0, m1):
            md.append(f"| {k} | " + " | ".join(fmt(arms[k][t].mean()) for t in READOUTS) + " |")
        md.append("\nPaired contrasts at pass 20 (mean, bootstrap 95% CI, sign-flip p):\n")
        md.append("| Contrast | mean | 95% CI | p | boards + / - |")
        md.append("|---|---:|---|---:|---|")
        for name, st in contrasts["pass_20"].items():
            md.append(
                f"| {name} | {fmt(st['mean'])} | [{fmt(st['bootstrap_ci95'][0])}, {fmt(st['bootstrap_ci95'][1])}]"
                f" | {st['signflip_two_sided_p']:.4f} | {st['n_positive']} / {st['n_negative']} |"
            )
        mdec = mech["checkpoints"][ck]["decisions"]
        out[ck] = {
            "identity_controls_per_board": rows,
            "svd_relative_residual_fro": d["svd_start_relative_residual_fro"],
            "ablation_factorial_contrasts": contrasts,
            "mechanism_decisions": mdec,
        }
    return out, md


def reduce_online(rec: dict[str, Any], regime: str, identity_closed: dict[str, dict[str, bool]]) -> tuple[dict[str, Any], list[str]]:
    out: dict[str, Any] = {}
    md: list[str] = [f"\n### Online SVD, {regime}\n"]
    for ck, r in rec["metrics"].items():
        f64 = r["float64_kl_and_argmax_flips"]
        res: dict[str, Any] = {"identity_floor_closed": {}, "readouts": {}}
        md.append(f"\n{ck} (k = {r['k_keep']} of {r['components_per_module']}, zero ablation, residual off)\n")
        md.append("| Readout | repeat max | identity floor mean / max | static S | oracle C | online O | O-C mean [95% CI] | p | online better / worse | M upper 97.5% | interpretable |")
        md.append("|---|---:|---:|---:|---:|---:|---|---:|---|---:|---|")
        for t in READOUTS:
            S = np.array(f64["static_rank_k"][f"pass_{t}"]["kl64"]["per_board"])
            C = np.array(f64["clean_oracle_dynamic_topk"][f"pass_{t}"]["kl64"]["per_board"])
            O = np.array(f64["online_dynamic_topk"][f"pass_{t}"]["kl64"]["per_board"])
            F = np.array(f64["svd_all_on_identity_floor"][f"pass_{t}"]["kl64"]["per_board"])
            R = np.array(f64["same_dtype_target_rerun"][f"pass_{t}"]["kl64"]["per_board"])
            floor_closed = bool(F.max() <= CLOSURE_TOL and R.max() <= CLOSURE_TOL)
            interp = floor_closed and identity_closed[ck][f"pass_{t}"]
            D = paired(O - C)
            ms = margin_stat(S, C, O)
            flips = {
                a: f64[a][f"pass_{t}"]["argmax_flips_per_board"]
                for a in ("svd_all_on_identity_floor", "static_rank_k", "clean_oracle_dynamic_topk",
                          "online_dynamic_topk", "online_vs_oracle_rollout")
            }
            above_floor = {
                a: int((arr > np.maximum(F, CLOSURE_TOL)).sum()) for a, arr in (("static", S), ("oracle", C), ("online", O))
            }
            res["readouts"][f"pass_{t}"] = {
                "static_kl": S.tolist(), "oracle_kl": C.tolist(), "online_kl": O.tolist(),
                "identity_floor_kl": F.tolist(), "target_repeat_kl": R.tolist(),
                "means": {"static": float(S.mean()), "oracle": float(C.mean()), "online": float(O.mean()),
                          "floor": float(F.mean())},
                "online_minus_oracle": D,
                "reserved64_margin_statistic_descriptive": ms,
                "argmax_flips_per_board": flips,
                "boards_above_identity_floor": above_floor,
                "identity_floor_closed": floor_closed,
                "interpretable_for_mechanism_classification": interp,
            }
            md.append(
                f"| {t} | {fmt(R.max())} | {fmt(F.mean())} / {fmt(F.max())} | {fmt(S.mean())} | {fmt(C.mean())}"
                f" | {fmt(O.mean())} | {fmt(D['mean'])} [{fmt(D['bootstrap_ci95'][0])}, {fmt(D['bootstrap_ci95'][1])}]"
                f" | {D['signflip_two_sided_p']:.4f} | {D['n_negative']} / {D['n_positive']} | {fmt(ms['upper_97_5'])}"
                f" | {'yes' if interp else 'no'} |"
            )
        dis = r["oracle_mask_disagreement"]
        res["mask_disagreement_overall_mean"] = {m: v["overall_mean"] for m, v in dis.items()}
        res["mask_disagreement_by_pass_mean"] = {m: v["by_pass_mean"] for m, v in dis.items()}
        res["component_density"] = r["component_density"]
        md.append("\nOnline vs clean-informed mask disagreement (mean Hamming fraction over boards, tokens, components):\n")
        md.append("| Module | overall | pass 0 | pass 20 | pass 41 | pass 62 |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for m, v in dis.items():
            bp = v["by_pass_mean"]
            md.append(f"| {m} | {v['overall_mean']:.4f} | {bp[0]:.4f} | {bp[20]:.4f} | {bp[41]:.4f} | {bp[62]:.4f} |")
        md.append("\nArgmax cell flips vs clean target at pass 20 / 62 (sum over 16 boards x 81 cells):\n")
        md.append("| Arm | pass 20 | pass 62 |")
        md.append("|---|---:|---:|")
        for a in ("svd_all_on_identity_floor", "static_rank_k", "clean_oracle_dynamic_topk", "online_dynamic_topk"):
            md.append(
                f"| {a} | {sum(res['readouts']['pass_20']['argmax_flips_per_board'][a])}"
                f" | {sum(res['readouts']['pass_62']['argmax_flips_per_board'][a])} |"
            )
        out[ck] = res
    return out, md


CAUSAL_ARMS = ("S0", "S1", "S2", "T1", "T2", "T3", "T4", "N1", "N2")
CAUSAL_FLOORS = ("target_repeat", "svd_all_on_residual_off", "trained_all_on_residual_on")
CAUSAL_CONTRASTS = ("H1_T3_minus_T1", "H2_T3_minus_S2", "H3_M_T3_minus_T2_minus_0.10_S0", "T3_minus_T2",
                    "T4_minus_T3", "N2_minus_N1", "T2_minus_S1", "T1_minus_S2")


def reduce_causal(rec: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Experiment C tables from ``gate_b4_causal_ci.json``.

    The registered decisions are taken from the receipt. Per-board means are
    recomputed here and checked against the receipt's contrast means.
    """
    dec = rec["decisions"]
    out: dict[str, Any] = {"registered_pass20_decisions": dec["registered_pass20_decisions"],
                           "holm_adjusted_signflip_p_pass20": dec.get("holm_adjusted_signflip_p_pass20"),
                           "checkpoints": {}}
    md: list[str] = [
        f"\nRegime: {rec['regime']}, TF32 {'on' if rec['tf32_matmul_allowed'] else 'off'}, {rec['cuda_device']},"
        f" torch {rec['torch_version']}. Reserved 64 boards, k = {rec['k']} of {rec['C']}, readouts {rec['readouts']}.\n",
        "Registered pass-20 decisions:\n",
        "| Hypothesis | Decision |", "|---|---|",
    ]
    md += [f"| {k} | {v} |" for k, v in dec["registered_pass20_decisions"].items()]
    if dec.get("holm_adjusted_signflip_p_pass20"):
        md.append("\nHolm-adjusted one-sided sign-flip p (pass 20, H: arm difference < 0):\n")
        md.append("| Contrast | " + " | ".join(rec["checkpoints"]) + " |")
        md.append("|---|" + "---:|" * len(rec["checkpoints"]))
        for label, ps in dec["holm_adjusted_signflip_p_pass20"].items():
            md.append(f"| {label} | " + " | ".join(f"{ps[ck]:.4f}" for ck in rec["checkpoints"]) + " |")
    for ck, r in rec["checkpoints"].items():
        a = r["arms"]
        rr = r["ci_refit_receipt"]
        kl = {n: {t: np.array(a[n][f"pass_{t}"]["kl64"], dtype=np.float64) for t in READOUTS} for n in a}
        floor = {t: np.max([kl[f][t] for f in CAUSAL_FLOORS], axis=0) for t in READOUTS}
        dck = dec["by_checkpoint"][ck]
        for t in READOUTS:
            for label, diff in (("H1_T3_minus_T1", kl["T3"][t] - kl["T1"][t]),
                                ("H2_T3_minus_S2", kl["T3"][t] - kl["S2"][t]),
                                ("H3_M_T3_minus_T2_minus_0.10_S0", kl["T3"][t] - kl["T2"][t] - 0.10 * kl["S0"][t])):
                if abs(float(diff.mean()) - dck[f"pass_{t}"][label]["mean"]) > 1e-9:
                    raise ValueError(f"{ck} pass {t} {label}: receipt mean disagrees with per-board KL")
        out["checkpoints"][ck] = {
            "ci_weights_sha256": rr["ci_weights_sha256"],
            "refit_validation_recon_loss": rr["validation_metrics_best_state"]["recon_loss"],
            "per_board_kl": {n: {f"pass_{t}": kl[n][t].tolist() for t in READOUTS} for n in kl},
            "floor_max_per_board": {f"pass_{t}": floor[t].tolist() for t in READOUTS},
            "contrasts": dck,
            "module_execution_order": r["module_execution_order"],
            "ci_module_order": r["ci_module_order"],
            "within_pass_dependency_grad_norm_tau0": r["within_pass_dependency_grad_norm_tau0"],
        }
        md.append(f"\n### {ck}\n")
        md.append(f"CI refit: {rr['steps']} steps x batch {rr['batch_size']}, seed {rr['seed']},"
                  f" validation recon loss {rr['validation_metrics_best_state']['recon_loss']:.4f},"
                  f" weights SHA-256 `{rr['ci_weights_sha256'][:16]}...`.\n")
        md.append("Mean KL(clean FP32 target || arm) over 64 boards, and summed argmax cell flips:\n")
        md.append("| Arm | pass 20 | pass 41 | pass 62 | flips p20 | flips p62 |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for n in CAUSAL_FLOORS + CAUSAL_ARMS:
            md.append(f"| {n} | " + " | ".join(fmt(float(kl[n][t].mean())) for t in READOUTS)
                      + f" | {sum(a[n]['pass_20']['argmax_flips'])} | {sum(a[n]['pass_62']['argmax_flips'])} |")
        md.append("| floor max (any floor, any board) | " + " | ".join(fmt(float(floor[t].max())) for t in READOUTS)
                  + " | | |")
        md.append("| interpretable | " + " | ".join("yes" if dck[f"pass_{t}"]["interpretable"] else "no"
                                                   for t in READOUTS) + " | | |")
        for t in READOUTS:
            md.append(f"\nPaired contrasts at pass {t} (bootstrap 95% CI, seed 20261117301; one-sided sign-flip p for < 0):\n")
            md.append("| Contrast | mean | median | 95% CI | p | boards - / + | top-board share |")
            md.append("|---|---:|---:|---|---:|---|---:|")
            for label in CAUSAL_CONTRASTS:
                s = dck[f"pass_{t}"][label]
                md.append(f"| {label} | {fmt(s['mean'])} | {fmt(s['median'])} | [{fmt(s['lower_2_5'])},"
                          f" {fmt(s['upper_97_5'])}] | {s['signflip_one_sided_p_negative']:.4f}"
                          f" | {s['n_negative']} / {s['n_positive']} | {s['abs_effect_top_board_share']:.3f} |")
        for key, title in (("T3_vs_T2_disagreement_by_module_by_pass", "T3 (causal preflight) vs T2 (clean-informed)"),
                           ("T4_vs_T2_disagreement_by_module_by_pass", "T4 (causal lagged) vs T2 (clean-informed)")):
            md.append(f"\n{title} top-k mask disagreement (Hamming fraction, mean over boards and tokens):\n")
            md.append("| Module | inv 0 | inv 1 | inv 20 | inv 41 | inv 62 | mean |")
            md.append("|---|---:|---:|---:|---:|---:|---:|")
            for m, v in r[key].items():
                md.append(f"| {m} | " + " | ".join(f"{v[i]:.4f}" for i in (0, 1, 20, 41, 62)) + f" | {np.mean(v):.4f} |")
        md.append("\nContinuous CI density: N1 (clean-informed) mean mask and fraction > 0.5 by chunk;"
                  " N2 (causal) mean mask at invocations 0 / 20 / 41 / 62:\n")
        md.append("| Module | N1 mean | N1 > 0.5 | N2 inv 0 | N2 inv 20 | N2 inv 41 | N2 inv 62 |")
        md.append("|---|---:|---:|---:|---:|---:|---:|")
        for m, v in r["N2_mean_mask_by_module_by_pass"].items():
            n1 = float(np.mean([c[m] for c in r["N1_mean_mask_by_chunk"]]))
            n1a = float(np.mean([c[m] for c in r["N1_fraction_above_0p5_by_chunk"]]))
            md.append(f"| {m} | {n1:.4f} | {n1a:.4f} | " + " | ".join(f"{v[i]:.4f}" for i in (0, 20, 41, 62)) + " |")
        for t in (20, 62):
            md.append(f"\nPer-board KL at pass {t} (reserved-64 order):\n")
            md.append("| Board | floor max | " + " | ".join(CAUSAL_ARMS) + " |")
            md.append("|---:|---:|" + "---:|" * len(CAUSAL_ARMS))
            for i, b in enumerate(rec["eval_board_indices"]):
                md.append(f"| {b} | {fmt(float(floor[t][i]))} | "
                          + " | ".join(fmt(float(kl[n][t][i])) for n in CAUSAL_ARMS) + " |")
    return out, md


FIX_SEED = 20261118401


def closed_loop_fix_effect(new: dict[str, Any], old: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Registered H4 for B4-CAUSAL-CI-02: KL(T3 closed-loop CI) - KL(T3 refit CI), same boards."""
    if new["eval_board_indices"] != old["eval_board_indices"]:
        raise ValueError("closed-loop and refit receipts score different boards")
    out: dict[str, Any] = {"by_checkpoint": {}}
    md = ["\nH4, closed-loop vs clean-trajectory CI under causal top-307 gating "
          "(KL(T3 new) - KL(T3 refit); bootstrap 95% CI, 20,000 resamples, seed 20261118401):\n",
          "| Checkpoint | Readout | mean | 95% CI | boards new lower / higher | interpretable |",
          "|---|---|---:|---|---|---|"]
    for ck in new["checkpoints"]:
        rng = np.random.default_rng(FIX_SEED)
        rows = {}
        for t in READOUTS:
            d = (np.array(new["checkpoints"][ck]["arms"]["T3"][f"pass_{t}"]["kl64"], dtype=np.float64)
                 - np.array(old["checkpoints"][ck]["arms"]["T3"][f"pass_{t}"]["kl64"], dtype=np.float64))
            boots = d[rng.integers(0, d.size, size=(N_BOOT, d.size))].mean(axis=1)
            interp = bool(new["decisions"]["by_checkpoint"][ck][f"pass_{t}"]["interpretable"]
                          and old["decisions"]["by_checkpoint"][ck][f"pass_{t}"]["interpretable"])
            rows[f"pass_{t}"] = {"mean": float(d.mean()), "lower_2_5": float(np.quantile(boots, 0.025)),
                                 "upper_97_5": float(np.quantile(boots, 0.975)), "n_negative": int((d < 0).sum()),
                                 "n_positive": int((d > 0).sum()), "interpretable": interp,
                                 "per_board": d.tolist()}
            r = rows[f"pass_{t}"]
            md.append(f"| {ck} | {t} | {fmt(r['mean'])} | [{fmt(r['lower_2_5'])}, {fmt(r['upper_97_5'])}]"
                      f" | {r['n_negative']} / {r['n_positive']} | {'yes' if interp else 'no'} |")
        out["by_checkpoint"][ck] = rows
    p20 = [out["by_checkpoint"][ck]["pass_20"] for ck in out["by_checkpoint"]]
    if all(r["interpretable"] and r["upper_97_5"] < 0 for r in p20):
        verdict = "closed-loop CI better (upper bound < 0 on both checkpoints)"
    elif all(r["interpretable"] and r["lower_2_5"] > 0 for r in p20):
        verdict = "closed-loop CI worse (lower bound > 0 on both checkpoints)"
    else:
        verdict = "inconclusive"
    out["H4_registered_pass20_decision"] = verdict
    md.append(f"\nH4 registered pass-20 decision: **{verdict}**")
    return out, md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    base = Path("mechinterp/bpd/results/v2")
    ap.add_argument("--controls", type=Path, default=base / "gate_b4_v2_closure_controls_corrected.json")
    ap.add_argument("--mechanism", type=Path, default=base / "gate_b4_v2_closure_mechanism.json")
    ap.add_argument("--online-bf16", type=Path, default=base / "gate_b4_v2_online_svd.json")
    ap.add_argument("--online-fp32", type=Path, default=base / "gate_b4_v2_online_svd_cuda_fp32.json")
    ap.add_argument("--causal-ci", type=Path, default=base / "gate_b4_causal_ci.json")
    ap.add_argument("--causal-ci-closed-loop", type=Path, default=base / "gate_b4_causal_ci_closed_loop.json")
    ap.add_argument("--out-json", type=Path, default=base / "bpd01_summary.json")
    ap.add_argument("--out-md", type=Path, default=base / "bpd01_tables.md")
    args = ap.parse_args()

    ctrl = json.loads(args.controls.read_text(encoding="utf-8"))
    mech = json.loads(args.mechanism.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {"closure_tolerance_kl": CLOSURE_TOL, "bootstrap_resamples": N_BOOT,
                               "signflip_draws": N_FLIP, "seed": SEED}
    md: list[str] = ["# BPD-01 reduced tables", "",
                     "Generated by `mechinterp/bpd/scripts/reduce_bpd01.py` from the committed JSON receipts.",
                     "KL is token-mean KL(reference || arm) over 81 cells and 9 digits."]
    a_out, a_md = reduce_controls(ctrl, mech)
    summary["experiment_A"] = a_out
    md += ["", "## Experiment A: numerical closure (CUDA BF16, 16 boards, 63 invocations)"] + a_md

    closures = {
        ck: {
            "native_bf16": {t: v["closed"] for t, v in a_out[ck]["mechanism_decisions"]["identity_closure_by_regime"]["native_bf16"].items()},
            "fp32": {t: v["closed"] for t, v in a_out[ck]["mechanism_decisions"]["identity_closure_by_regime"]["fp32_original_fp32_weights"].items()},
        }
        for ck in a_out
    }
    summary["experiment_B"] = {}
    md += ["", "## Experiment B: online vs clean-informed SVD top-k"]
    for label, path, key in (("cuda_bf16_registered", args.online_bf16, "native_bf16"),
                             ("cuda_fp32_original_weights", args.online_fp32, "fp32")):
        if not path.exists():
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        b_out, b_md = reduce_online(rec, label, {ck: closures[ck][key] for ck in closures})
        summary["experiment_B"][label] = b_out
        md += b_md

    if args.causal_ci.exists():
        crec = json.loads(args.causal_ci.read_text(encoding="utf-8"))
        if "decisions" in crec:
            c_out, c_md = reduce_causal(crec)
            summary["experiment_C"] = c_out
            md += ["", "## Experiment C: refit CI, clean-informed vs causal gating (B4-CAUSAL-CI-01)"] + c_md
            if args.causal_ci_closed_loop.exists():
                c2 = json.loads(args.causal_ci_closed_loop.read_text(encoding="utf-8"))
                if "decisions" in c2:
                    c2_out, c2_md = reduce_causal(c2)
                    fx, fx_md = closed_loop_fix_effect(c2, crec)
                    c2_out["H4_closed_loop_vs_refit"] = fx
                    summary["experiment_C2"] = c2_out
                    md += ["", "## Experiment C2: closed-loop-trained CI under causal gating (B4-CAUSAL-CI-02)"]
                    md += fx_md + c2_md

    args.out_json.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
