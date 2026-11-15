#!/usr/bin/env python3
"""Publication figure generator for the LOOPSCOPE-04 & ARG-03-N Dark Fibers campaign.

Reads `papers/shared_weights_dark_fibers/campaign_20261115/campaign_summary.json` and renders
both vector `.pdf` and 300-DPI `.png` figures in `papers/shared_weights_dark_fibers/figures/`:
  1. `fig6_loopscope04_causal_quotient.{png,pdf}` (LOOPSCOPE-04 3A/3B/3C across both TRM checkpoints)
  2. `fig7_arg03_nanbeige_dev.{png,pdf}` (ARG-03-N Nanbeige4.2-3B development.jsonl N=96, when present)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Colorblind-safe Okabe-Ito palette matching papers/shared_weights_dark_fibers/build_figures.py
C_SLATE = "#4A5568"
C_BLUE = "#0072B2"
C_SKY = "#56B4E9"
C_VERMILION = "#D55E00"
C_TEAL = "#009E73"
C_AMBER = "#E69F00"
C_PURPLE = "#CC79A7"
C_DARK = "#1A202C"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.3,
            "axes.titlesize": 8.6,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.32,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_png_and_pdf(fig: plt.Figure, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png} and {out_pdf}")


def render_fig6(data: dict, out_png: Path) -> None:
    ls = data["loopscope04"]
    swaps = ls["exp_3a_swaps"]
    caches = ls["exp_3b_caches"]
    grams = ls["exp_3c_gramians"]

    fig, axes = plt.subplots(
        1, 3, figsize=(8.35, 3.05), gridspec_kw={"width_ratios": [1.05, 1.02, 1.05], "wspace": 0.38}
    )
    colors = {"gaoxin492": C_BLUE, "yagizdevre": C_VERMILION}
    disp_short = {"gaoxin492": "Prim. (gaoxin492)", "yagizdevre": "Repl. (yagizdevre)"}

    # --- Panel (a): 3A Causal Fiber Swaps on WB-Stall (t=4, n=80/120) ---
    ax = axes[0]
    arm_keys = ["swapped", "zl_only_swap", "swapped_joint_zl", "random_orth"]
    arm_labels = ["$z_H^\\perp$\n(501-D)", "$z_L$\n(board)", "Joint\n$z_H^\\perp{+}z_L$", "Random\neq.-rad."]
    x = np.arange(len(arm_keys))
    w = 0.36

    for idx_tag, tag in enumerate(("gaoxin492", "yagizdevre")):
        key = f"{tag}_rep_swap_t4_h32"
        if key not in swaps:
            continue
        rec = swaps[key]
        st_pt = rec["paired_tests"].get("wb_stall_paired")
        use_group = "wb_stall" if st_pt else "all"
        means, errs_lo, errs_hi = [], [], []
        for ak in arm_keys:
            val = rec["stratified_summary"][use_group]["arms"][ak]["mean_first_successor_cos_to_donor"]
            means.append(val)
            ci_obj = None
            if st_pt:
                ci_map = {
                    "swapped": st_pt.get("swapped_cos_ci"),
                    "random_orth": st_pt.get("random_cos_ci"),
                    "zl_only_swap": st_pt.get("zl_only_cos_ci"),
                    "swapped_joint_zl": st_pt.get("joint_zl_cos_ci"),
                }
                ci_obj = ci_map.get(ak)
            if ci_obj and ci_obj["ci95_low"] is not None:
                errs_lo.append(max(0.0, val - ci_obj["ci95_low"]))
                errs_hi.append(max(0.0, ci_obj["ci95_high"] - val))
            else:
                errs_lo.append(0.0)
                errs_hi.append(0.0)
        offset = (-0.5 + idx_tag) * w
        ax.bar(
            x + offset,
            means,
            width=w * 0.92,
            color=colors[tag],
            edgecolor="white",
            linewidth=0.35,
            label=disp_short[tag],
            zorder=3,
        )
        ax.errorbar(
            x + offset,
            means,
            yerr=[errs_lo, errs_hi],
            fmt="none",
            ecolor=C_DARK,
            elinewidth=0.65,
            capsize=1.8,
            zorder=4,
        )

    ax.axhline(0.0, color="#718096", lw=0.7, ls="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(arm_labels, fontsize=6.8)
    ax.set_ylim(-0.04, 0.78)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.set_ylabel("Step-$(t{+}1)$ cosine to donor shift")
    ax.set_title("(a) 3A: Stalled steering ($t{=}4$)", pad=7)
    ax.legend(
        loc="upper left",
        ncol=2,
        columnspacing=0.6,
        handletextpad=0.35,
        frameon=True,
        framealpha=0.95,
        edgecolor="#CBD5E0",
        fontsize=5.6,
        borderpad=0.22,
    )
    ax.grid(axis="y")
    ax.text(
        0.97,
        0.81,
        "WB-Prop: 40/40 Prim. | Joint > Rand ($p{<}10^{-6}$)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.2,
        bbox=dict(boxstyle="round,pad=0.18", fc="#EDF2F7", ec="#CBD5E0", alpha=0.95),
    )

    # --- Panel (b): 3B Observer-Relative Cache (Ridge + SeqRefill97 Snapshot vs. Trace) ---
    ax = axes[1]
    sizes = [90, 180, 360]
    for tag, marker in (("gaoxin492", "o"), ("yagizdevre", "s")):
        r2_snap, r2_tr, valid_n = [], [], []
        for n_tr in sizes:
            ck = f"{tag}_rep_cache_full97_n360" if n_tr == 360 else f"{tag}_rep_cache_ridge_n{n_tr}"
            if ck in caches:
                valid_n.append(n_tr)
                r2_snap.append(caches[ck]["cellwise_ridge"]["snapshot"]["r2_zh_orth"])
                r2_tr.append(caches[ck]["cellwise_ridge"]["visible_trace"]["r2_zh_orth"])
        if valid_n:
            ax.plot(
                valid_n,
                r2_snap,
                ls="--",
                marker=marker,
                markersize=3.2,
                color=colors[tag],
                lw=1.25,
                alpha=0.80,
                label=f"{tag[:4]} Ridge $C^{{\\mathrm{{now}}}}$",
            )
            ax.plot(
                valid_n,
                r2_tr,
                ls="-",
                marker=marker,
                markersize=3.4,
                color=colors[tag],
                lw=1.55,
                label=f"{tag[:4]} Ridge $C^{{\\mathrm{{trace}}}}$",
            )
        ck360 = f"{tag}_rep_cache_full97_n360"
        if ck360 in caches and "full_97_refill" in caches[ck360]:
            f97 = caches[ck360]["full_97_refill"]["models"]
            y_now = f97["c_now_97"]["heldout_r2"]["r2_board_zh_orth"]
            y_tr = f97["c_trace_97"]["heldout_r2"]["r2_board_zh_orth"]
            ax.scatter([346], [y_now], marker="D", s=22, facecolors="white", edgecolors=colors[tag], linewidths=1.1, zorder=5)
            ax.scatter([372], [y_tr], marker="D", s=24, color=colors[tag], zorder=5)

    ax.set_xticks(sizes)
    ax.set_xlim(70, 396)
    ax.set_ylim(0.16, 0.96)
    ax.set_xlabel("Disjoint training boards ($N_{\\mathrm{train}}$)")
    ax.set_ylabel("Held-out $z_H^\\perp$ $R^2$")
    ax.set_title("(b) 3B: Snapshot vs. trace", pad=7)
    ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.53), frameon=True, framealpha=0.95, edgecolor="#CBD5E0", fontsize=5.6, borderpad=0.20)
    ax.grid(axis="both")
    ax.text(
        198,
        0.845,
        "Diamonds @360: SeqRefill97 ($C^{\\mathrm{now}}/C^{\\mathrm{trace}}$)\nm=1 Stall: Prim .71$\\to$.78 | Repl .45$\\to$.54$\\to$.60",
        ha="center",
        va="center",
        fontsize=5.25,
        bbox=dict(boxstyle="round,pad=0.20", fc="#EDF2F7", ec="#CBD5E0", alpha=0.95),
    )

    # --- Panel (c): 3C Delayed Observability Gramian Nonlinear Validation ---
    ax = axes[2]
    angles = [0.04, 0.15, 0.35, 0.75]
    for tag in ("gaoxin492", "yagizdevre"):
        gk = f"{tag}_rep_gramian_t4_p8_k4_h12"
        if gk not in grams:
            continue
        at = grams[gk]["paired_angle_tests"]
        hi_vals, lo_vals, rd_vals, valid_ang = [], [], [], []
        for a_val in angles:
            ak = f"angle_{a_val:.2f}"
            if ak not in at:
                ak = f"angle_{a_val:g}"
            if ak in at:
                valid_ang.append(a_val)
                hi_vals.append(at[ak]["cum_rms_high_vs_low"]["high_mean_ci"]["estimate"])
                lo_vals.append(at[ak]["cum_rms_high_vs_low"]["low_mean_ci"]["estimate"])
                rd_vals.append(at[ak]["cum_rms_high_vs_random"]["random_mean_ci"]["estimate"])
        if valid_ang:
            ax.plot(
                valid_ang,
                hi_vals,
                ls="-",
                marker="o",
                markersize=3.3,
                color=colors[tag],
                lw=1.6,
                label=f"{tag[:4]} Top eig ($v_1$)",
            )
            ax.plot(
                valid_ang,
                rd_vals,
                ls="--",
                marker="^",
                markersize=2.8,
                color=colors[tag],
                alpha=0.75,
                lw=1.15,
                label=f"{tag[:4]} Random dir.",
            )
            ax.plot(
                valid_ang,
                lo_vals,
                ls=":",
                marker="v",
                markersize=2.9,
                color=colors[tag],
                alpha=0.70,
                lw=1.25,
                label=f"{tag[:4]} Bot. eig ($v_8$)",
            )

    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(angles)
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(["0.04", "0.15", "0.35", "0.75"]))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylim(0.0, 262.0)
    ax.set_xlabel("Finite geodesic angle $\\beta$ (rad)")
    ax.set_ylabel("4-step cumulative readout RMS $L_2$")
    ax.set_title("(c) 3C: Gramian $G_{4,4}$ causality", pad=7)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#CBD5E0", fontsize=5.3, ncol=2, borderpad=0.18, columnspacing=0.5)
    ax.grid(axis="both")
    ax.text(
        0.31,
        28.0,
        "$\\beta{=}0.04$: $v_1 > v_8$ in\n119/120 (99.2%) both",
        ha="center",
        va="center",
        fontsize=5.4,
        bbox=dict(boxstyle="round,pad=0.20", fc="#EDF2F7", ec="#CBD5E0", alpha=0.95),
    )

    _save_png_and_pdf(fig, out_png)


def render_fig7_arg03(data: dict, out_png: Path) -> bool:
    arg03 = data.get("arg03_nanbeige", {})
    if "dev_full96" not in arg03:
        return False
    dev = arg03["dev_full96"]
    graded = dev["graded"]
    res = graded["results"]
    td = dev.get("tape_diagnostics", {})
    denom = float(graded.get("rows_scored", 96))

    fig, axes = plt.subplots(
        1, 3, figsize=(8.35, 3.05), gridspec_kw={"width_ratios": [1.12, 0.96, 1.02], "wspace": 0.38}
    )

    # --- Panel (a): Verified future@12 accuracy across depths 1, 2, 4, 8 ---
    ax = axes[0]
    depths = [1, 2, 4, 8]
    xd = np.arange(len(depths))
    series = [
        ("baseline", "Clean recurrent", C_BLUE, "-", "o", 1.65),
        ("context_swap", "Context donor swap", C_VERMILION, "-", "s", 1.55),
        ("random_context_norm_matched", "Random eq.-norm ctx", C_PURPLE, "--", "^", 1.35),
        ("open_loop", "Open-loop linear", C_SLATE, ":", "v", 1.30),
        ("equal_compute", "Equal-compute 1-pass", C_TEAL, "-.", "D", 1.35),
    ]
    all_ys = []
    for arm_k, lbl, col, ls_style, mk, lw in series:
        ys, valid_x = [], []
        for idx_d, d in enumerate(depths):
            rk = f"{arm_k}@{d}"
            if rk in res:
                valid_x.append(idx_d)
                v = res[rk]["correct"] / denom
                ys.append(v)
                all_ys.append(v)
        if valid_x:
            ax.plot(valid_x, ys, color=col, linestyle=ls_style, marker=mk, markersize=3.4, linewidth=lw, label=lbl)

    cal = graded.get("dev_calibration", {})
    if cal:
        best_t_key = cal.get("selected_full_vocab_scalar_temp")
        if best_t_key and f"temperature_grid@{best_t_key}" in res:
            val_t = res[f"temperature_grid@{best_t_key}"]["correct"] / denom
            all_ys.append(val_t)
            ax.axhline(val_t, color=C_AMBER, ls="--", lw=1.0, label=f"1-pass {best_t_key} ({val_t:.2f})")
        if "dev_calibrated_loop1@per_pos_op" in res:
            val_op = res["dev_calibrated_loop1@per_pos_op"]["correct"] / denom
            all_ys.append(val_op)
            ax.axhline(val_op, color=C_SKY, ls=":", lw=1.1, label=f"1-pass pos-cal ({val_op:.2f})")

    y_top = min(1.05, max(all_ys + [0.55]) + 0.32)
    ax.set_xticks(xd)
    ax.set_xticklabels([f"t={d}" for d in depths])
    ax.set_ylim(0.0, y_top)
    ax.set_ylabel("Verified future@12 rate ($N{=}96$)")
    ax.set_title("(a) ARG-03-N dev accuracy", pad=7)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#CBD5E0", fontsize=5.3, ncol=2, borderpad=0.18, columnspacing=0.5)
    ax.grid(axis="both")

    # --- Panel (b): Delayed full-vocab readout L2 divergence (exact 0 at t=1) ---
    ax = axes[1]
    pr = td.get("postloop_readout_distance_by_depth", {})
    sw_m, sw_lo, sw_hi = [0.0], [0.0], [0.0]
    rd_m, rd_lo, rd_hi = [0.0], [0.0], [0.0]
    d_labels = ["t=1\n(patch)", "t=2", "t=4", "t=8"]
    for d_str in ("2", "4", "8"):
        if d_str in pr:
            c_sw = pr[d_str]["context_swap_mean_ci"]
            c_rd = pr[d_str]["random_context_mean_ci"]
            sw_m.append(c_sw["estimate"])
            sw_lo.append(max(0.0, c_sw["estimate"] - c_sw["ci95_low"]))
            sw_hi.append(max(0.0, c_sw["ci95_high"] - c_sw["estimate"]))
            rd_m.append(c_rd["estimate"])
            rd_lo.append(max(0.0, c_rd["estimate"] - c_rd["ci95_low"]))
            rd_hi.append(max(0.0, c_rd["ci95_high"] - c_rd["estimate"]))
    xb = np.arange(len(sw_m))
    ax.errorbar(
        xb, sw_m, yerr=[sw_lo, sw_hi], color=C_VERMILION, marker="s", markersize=3.5,
        linewidth=1.55, capsize=2.0, label="Context donor swap"
    )
    ax.errorbar(
        xb, rd_m, yerr=[rd_lo, rd_hi], color=C_PURPLE, linestyle="--", marker="^", markersize=3.4,
        linewidth=1.4, capsize=2.0, label="Random eq.-norm ctx"
    )
    max_y_b = max([m + h for m, h in zip(sw_m + rd_m, sw_hi + rd_hi)] + [1.0])
    ax.set_ylim(-0.03 * max_y_b, 1.28 * max_y_b)
    ax.set_xticks(xb)
    ax.set_xticklabels(d_labels[: len(sw_m)], fontsize=7.0)
    ax.set_ylabel("Tail readout log-prob $L_2$ diff.")
    ax.set_title("(b) Delayed fiber visibility", pad=7)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#CBD5E0", fontsize=5.7, borderpad=0.20)
    ax.grid(axis="both")
    max_leak = td.get("max_precontinuation_donor_logprob_error", 0.0)
    ax.text(
        1.85,
        0.16 * max_y_b,
        f"Pre-cont. leak @t=1:\nmax = {max_leak:.1e}",
        ha="center",
        va="center",
        fontsize=5.4,
        bbox=dict(boxstyle="round,pad=0.20", fc="#EDF2F7", ec="#CBD5E0", alpha=0.95),
    )

    # --- Panel (c): 64-program ranking stability vs. clean ---
    ax = axes[2]
    rk_by_d = td.get("ranking_changes_by_depth", {})
    d_post = [d for d in ("2", "4", "8") if d in rk_by_d]
    xc = np.arange(len(d_post))
    for arm_k, lbl, col, ls_s, mk in (
        ("context_swap", "Context swap", C_VERMILION, "-", "s"),
        ("random_context_norm_matched", "Random eq.-norm", C_PURPLE, "--", "^"),
        ("open_loop", "Open-loop", C_SLATE, ":", "v"),
    ):
        vals, elo, ehi = [], [], []
        for d_str in d_post:
            ci_o = rk_by_d[d_str][arm_k]["spearman_rho_mean_ci"]
            v = ci_o["estimate"]
            vals.append(v)
            elo.append(max(0.0, v - ci_o["ci95_low"]))
            ehi.append(max(0.0, ci_o["ci95_high"] - v))
        if vals:
            ax.errorbar(
                xc, vals, yerr=[elo, ehi], color=col, linestyle=ls_s, marker=mk,
                markersize=3.4, linewidth=1.45, capsize=1.8, label=lbl
            )
    ax.axhline(0.0, color="#718096", lw=0.65, ls="--", zorder=2)
    ax.set_xticks(xc)
    ax.set_xticklabels([f"t={d}" for d in d_post])
    ax.set_ylim(-0.38, 1.08)
    ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("64-program rank Spearman $\\rho$")
    ax.set_title("(c) Candidate ranking shift", pad=7)
    ax.legend(loc="center left", frameon=True, framealpha=0.95, edgecolor="#CBD5E0", fontsize=5.6, borderpad=0.20)
    ax.grid(axis="both")
    if "4" in rk_by_d and "context_swap" in rk_by_d["4"]:
        flip4 = 100.0 * rk_by_d["4"]["context_swap"]["top1_changed_rate"]
        ax.text(
            0.97,
            0.22,
            f"Swap $\\rho \\geq 0.995$\nTop-1 flip @t=4: {flip4:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.3,
            bbox=dict(boxstyle="round,pad=0.18", fc="#EDF2F7", ec="#CBD5E0", alpha=0.95),
        )

    _save_png_and_pdf(fig, out_png)
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--summary",
        type=Path,
        default=Path("papers/shared_weights_dark_fibers/campaign_20261115/campaign_summary.json"),
    )
    p.add_argument(
        "--out-png",
        type=Path,
        default=Path("papers/shared_weights_dark_fibers/figures/fig6_loopscope04_causal_quotient.png"),
    )
    p.add_argument(
        "--out-arg03-png",
        type=Path,
        default=Path("papers/shared_weights_dark_fibers/figures/fig7_arg03_nanbeige_dev.png"),
    )
    args = p.parse_args(argv)
    configure_style()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    render_fig6(data, args.out_png)
    render_fig7_arg03(data, args.out_arg03_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
