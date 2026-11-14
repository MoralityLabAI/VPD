#!/usr/bin/env python3
"""Deterministic vector + raster figure builder for:
From Shared Weights to Head-Orthogonal Fibers: Virtual Depth, Head-Readable
Staircases, and Post-Stall Search in Looped Transformers.

Reads committed LOOPSCOPE-02 JSON receipts directly from
../../mechinterp/loopscope02/results/ (including fiber_regime_controls.json)
and pairs them with audited summary tables from Paper I (v008, Sept 2026),
HUG-CAL-02 (Silico Exp #3), Paper II (RMP), and LOOPSCOPE-01 (Q1-Q9, Q11).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
LS02_RESULTS = REPO_ROOT / "mechinterp" / "loopscope02" / "results"
FIG_DIR = HERE / "figures"

# Colorblind-safe Okabe-Ito / academic palette
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
            "font.size": 8.4,
            "axes.titlesize": 8.9,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.3,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.32,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_both(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_fig1_triangulation(report: dict, trm: dict, nb: dict) -> None:
    """Fig 1: Stratified 9x9 Sudoku triangulation + SVD operator reuse vs multiplexing."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 2.95), gridspec_kw={"width_ratios": [1.30, 1.0], "wspace": 0.28}
    )

    strata = ["S1_naked1", "S2_prop2", "S3_prop3plus", "S4_stalled"]
    stratum_labels = ["S1\n(r = 1)", "S2\n(r = 2)", "S3\n(r >= 3)", "S4\n(Stalled)"]
    models = [
        ("nanbeige_loop1", "Nanbeige-3B (t=1)", C_SLATE),
        ("nanbeige_loop2", "Nanbeige-3B (t=2)", C_AMBER),
        ("qwen4b_final", "Qwen3.5-4B-Base", C_SKY),
        ("qwen3_4b_inst_final", "Qwen3-4B-Inst", C_BLUE),
        ("trm_step1", "TRM-7M (s=1)", C_PURPLE),
        ("trm_step16", "TRM-7M (s=16)", C_TEAL),
    ]

    x = np.arange(len(strata))
    width = 0.132
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2.0) * width

    acc = report["accuracy_by_stratum"]
    for idx, (key, label, color) in enumerate(models):
        means = [acc[key][s]["mean"] for s in strata]
        lo = [means[i] - acc[key][s]["ci95_lo"] for i, s in enumerate(strata)]
        hi = [acc[key][s]["ci95_hi"] - means[i] for i, s in enumerate(strata)]
        ax1.bar(
            x + offsets[idx],
            means,
            width=width * 0.92,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            label=label,
            zorder=3,
        )
        ax1.errorbar(
            x + offsets[idx],
            means,
            yerr=[lo, hi],
            fmt="none",
            ecolor=C_DARK,
            elinewidth=0.6,
            capsize=1.4,
            zorder=4,
        )

    ax1.axhline(1.0 / 9.0, color="#718096", linestyle="--", linewidth=0.8, zorder=2, label="1/9 uniform floor")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stratum_labels)
    ax1.set_ylim(0.0, 1.42)
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_ylabel("Target-cell accuracy (items256)")
    ax1.set_title("(a) Capability across deductive strata", pad=6)
    ax1.grid(axis="y")
    ax1.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.95, edgecolor="#CBD5E0", borderpad=0.28, columnspacing=0.8)

    # Panel (b): SVD rank-1 usage correlations (Nanbeige P1 vs P2 vs TRM outer & role)
    nb_layers = ["L6", "L12", "L17", "L22"]
    nb_r = [nb["usage_pass1_vs_pass2"][lay]["r_pass1_vs_pass2"] for lay in nb_layers]
    trm_mats = ["L0.o", "L0.down", "L1.o", "L1.down"]
    trm_keys = ["L0.o_proj", "L0.down_proj", "L1.o_proj", "L1.down_proj"]
    trm_outer = [trm["usage_stats"][k]["r_outer_zH_mean"] for k in trm_keys]
    trm_role = [trm["usage_stats"][k]["r_role_zL_vs_zH_mean"] for k in trm_keys]

    pos_nb = np.arange(4)
    pos_trm = np.arange(4) + 4.8
    ax2.bar(pos_nb, nb_r, width=0.58, color=C_AMBER, edgecolor="white", linewidth=0.4, label="Nanbeige r(Pass 1, Pass 2)", zorder=3)
    ax2.bar(pos_trm - 0.17, trm_outer, width=0.34, color=C_TEAL, edgecolor="white", linewidth=0.4, label="TRM r_outer(z_H, s=2..16)", zorder=3)
    ax2.bar(pos_trm + 0.17, trm_role, width=0.34, color=C_PURPLE, edgecolor="white", linewidth=0.4, label="TRM r_role(z_L vs z_H)", zorder=3)

    ax2.axvline(3.9, color="#A0AEC0", linestyle=":", linewidth=0.9)
    ax2.set_xticks(list(pos_nb) + list(pos_trm))
    ax2.set_xticklabels(nb_layers + trm_mats, rotation=28, ha="right")
    ax2.set_ylim(0.0, 1.38)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_ylabel("Top-64 SVD usage correlation (r)")
    ax2.set_title("(b) Virtual depth vs. stationary operator", pad=6)
    ax2.grid(axis="y")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#CBD5E0", borderpad=0.28)

    save_both(fig, "fig1_virtual_depth_triangulation")


def build_fig2_fragility_vs_specificity(trm: dict) -> None:
    """Fig 2: Silico HUG-CAL-02 matched-rotation null vs TRM Detached-ACT Lyapunov dichotomy."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.25, 2.85), gridspec_kw={"width_ratios": [1.0, 1.18], "wspace": 0.36}
    )

    categories = [
        "Heatmap\nt=1..8",
        "Heatmap\nt=13..16",
        "Exp #3\nraw",
        "Exp #3\nrot. null",
        "Net\nspec.",
    ]
    vals = [0.000, -0.770, -0.366, -0.368, 0.0007]
    colors = [C_SLATE, C_VERMILION, C_AMBER, C_BLUE, C_TEAL]
    x_a = np.arange(len(categories))
    bars = ax1.bar(x_a, vals, width=0.56, color=colors, edgecolor="white", linewidth=0.4, zorder=3)
    ax1.axhline(0.0, color=C_DARK, linewidth=0.65)
    ax1.axhline(-0.05, color="#718096", linestyle="--", linewidth=0.8, label="Registered -0.05 nat gate")
    ax1.errorbar([4], [0.0007], yerr=[[0.0065], [0.0065]], fmt="none", ecolor=C_DARK, elinewidth=1.0, capsize=2.5, zorder=4)
    for b, v in zip(bars, vals):
        y_txt = v - 0.052 if v < -0.02 else v + 0.036
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            y_txt,
            f"{v:+.3f}" if abs(v) < 0.02 else f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=6.9,
        )

    ax1.set_xticks(x_a)
    ax1.set_xticklabels(categories, fontsize=7.0)
    ax1.set_ylim(-0.90, 0.30)
    ax1.set_ylabel("Answer log-prob change (nats)")
    ax1.set_title("(a) Huginn-0125 rotation null", pad=6)
    ax1.grid(axis="y")
    ax1.legend(loc="upper center", frameon=True, framealpha=0.95, edgecolor="#CBD5E0", borderpad=0.25)

    # Panel (b): TRM 6-step unrolled TCA fragility (log scale) vs flat |z_read| > 3 specificity & finite kick
    steps_6 = np.arange(1, 7)
    frag_mat = np.array(trm["tca"]["frag_mean_by_step"])
    frag_mean = frag_mat.mean(axis=0)
    zr_mat = np.array(trm["tca"]["zr_gt3_share_by_step"]) * 100.0
    zr_mean = zr_mat.mean(axis=0)
    kick_6 = np.array(trm["kick_ratio_curve_mean"][:6]) * 100.0

    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)

    l1 = ax2.plot(steps_6, frag_mean, marker="o", markersize=4.4, linewidth=1.6, color=C_VERMILION, label="TCA fragility F_s (left, log)")
    ax2.set_yscale("log")
    ax2.set_ylabel("Mean TCA fragility ||X_s^T A_s||_F", color=C_VERMILION)
    ax2.tick_params(axis="y", labelcolor=C_VERMILION)
    ax2.set_ylim(8.0, 2.5e5)

    l2 = ax2_r.plot(steps_6, zr_mean, marker="s", markersize=4.1, linewidth=1.6, color=C_TEAL, label="TCA |z_read| > 3 (%) (right)")
    l3 = ax2_r.plot(steps_6, kick_6, marker="^", markersize=4.1, linewidth=1.3, linestyle="--", color=C_BLUE, label="Finite kick norm (%) (right)")
    ax2_r.axhline(0.27, color="#718096", linestyle=":", linewidth=0.85)
    ax2_r.set_ylabel("Specificity (%) / Kick ratio (%)", color=C_DARK)
    ax2_r.set_ylim(0.0, 112.0)

    ax2.set_xticks(steps_6)
    ax2.set_xlabel("Unrolled outer ACT step s")
    ax2.set_title("(b) TRM Jacobian vs. TCA z_read", pad=6)
    ax2.grid(axis="both")

    lines = l1 + l2 + l3
    labels = [ln.get_label() for ln in lines]
    ax2_r.legend(lines, labels, loc="upper right", frameon=True, framealpha=0.95, edgecolor="#CBD5E0", fontsize=6.7, borderpad=0.28)

    save_both(fig, "fig2_fragility_vs_specificity_tca")


def build_fig3_compactification_staircase(report: dict) -> None:
    """Fig 3: LOOPSCOPE-01 24-arm predictors + LOOPSCOPE-02 G_prop commutation & 48-cycle staircase."""
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(7.35, 2.65), gridspec_kw={"width_ratios": [1.05, 1.0, 0.98], "wspace": 0.36}
    )

    pred_names = [
        "Shuffled-op\nnull (Q11)",
        "Cursor kappa\n(Q7)",
        "Post-d8 drift\n(Q8)",
        "Fragility F1/F8\n(Q4/Q5)",
        "Loop clock C\n(Q9)",
        "Belief comm. E\n(Q11)",
    ]
    aucs = [0.071, 0.824, 0.882, 0.950, 0.958, 0.992]
    cols = [C_SLATE, C_SKY, C_BLUE, C_AMBER, C_PURPLE, C_TEAL]
    y_pos = np.arange(len(pred_names))
    ax1.barh(y_pos, aucs, height=0.60, color=cols, edgecolor="white", linewidth=0.4, zorder=3)
    ax1.axvline(0.5, color="#718096", linestyle=":", linewidth=0.8)
    for y, v in zip(y_pos, aucs):
        ax1.text(
            v + 0.022,
            y,
            f"{v:.3f}",
            va="center",
            ha="left",
            color=C_DARK,
            fontweight="bold" if v > 0.98 else "normal",
            fontsize=7.0,
        )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(pred_names, fontsize=7.0)
    ax1.set_xlim(0.0, 1.22)
    ax1.set_xticks([0.0, 0.5, 1.0])
    ax1.set_xlabel("Depth 9-16 failure ROC-AUC")
    ax1.set_title("(a) Pointer-chase AUC", pad=6)
    ax1.grid(axis="x")

    # Panel (b): LOOPSCOPE-02 G_prop commutation error rho_clean vs rho_shuf
    b_comp = report["mechanistic_metrics"]["bench100_compactification"]
    i_comp = report["mechanistic_metrics"]["items_compactification"]
    transitions = np.arange(1, len(b_comp["rho_clean_curve_prop"]) + 1)

    ax2.plot(transitions, b_comp["rho_shuf_curve_prop"], color=C_SLATE, linestyle="--", linewidth=1.3, label="Deranged G_prop")
    ax2.plot(transitions, b_comp["rho_clean_curve_prop"], color=C_TEAL, marker="o", markersize=3.2, linewidth=1.6, label="Exact (bench100)")
    ax2.plot(transitions, i_comp["rho_clean_curve_prop"], color=C_BLUE, marker="s", markersize=2.8, linewidth=1.3, label="Exact (items256)")
    ax2.set_xlim(1, 10)
    ax2.set_ylim(-0.04, 1.04)
    ax2.set_xlabel("Outer transition s -> s+1")
    ax2.set_ylabel("Belief commutation error rho_s")
    ax2.set_title("(b) TRM G_prop commutation", pad=6)
    ax2.grid(axis="both")
    ax2.text(
        5.5,
        0.74,
        "AUC = 1.000\n(both splits)",
        ha="center",
        fontsize=6.9,
        bbox=dict(boxstyle="round,pad=0.24", fc="#EDF2F7", ec="#CBD5E0"),
    )
    ax2.legend(loc="center right", bbox_to_anchor=(0.99, 0.36), frameon=True, framealpha=0.95, edgecolor="#CBD5E0", fontsize=6.5, borderpad=0.25)

    # Panel (c): 48-H-cycle deductive lock-in staircase
    rounds = ["r=1", "r=2", "r>=3", "Stall"]
    keys = ["r1", "r2", "r3plus", "stalled"]
    bench_cyc = [b_comp["staircase_lockin_cycle_mean"][k] for k in keys]
    items_cyc = [i_comp["staircase_lockin_cycle_mean"][k] for k in keys]

    xr = np.arange(len(rounds))
    ax3.bar(xr - 0.17, bench_cyc, width=0.33, color=C_TEAL, edgecolor="white", label="bench100 (C=0.618)")
    ax3.bar(xr + 0.17, items_cyc, width=0.33, color=C_BLUE, edgecolor="white", label="items256 (C=0.653)")
    ax3.set_xticks(xr)
    ax3.set_xticklabels(rounds, fontsize=7.3)
    ax3.set_ylim(0.0, 13.8)
    ax3.set_ylabel("Mean lock-in H-cycle (of 48)")
    ax3.set_title("(c) Lock-in staircase", pad=6)
    ax3.grid(axis="y")
    ax3.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#CBD5E0", fontsize=6.5, borderpad=0.25)

    save_both(fig, "fig3_compactification_staircase")


def build_fig4_dark_fiber_ablation(report: dict, ctrl: dict, ext: dict) -> None:
    """Fig 4: (a) Post-stall conflict repair, (b) Contrast & manifold controls, (c) Head-orth PCA dose curve."""
    fig = plt.figure(figsize=(7.65, 2.82))
    # Explicit horizontal coordinates so panel (a)'s right twin axis has generous room before panel (b)
    ax1 = fig.add_axes([0.065, 0.19, 0.235, 0.70])
    ax2 = fig.add_axes([0.415, 0.19, 0.275, 0.70])
    ax3 = fig.add_axes([0.765, 0.19, 0.220, 0.70])

    b_comp = report["mechanistic_metrics"]["bench100_compactification"]
    steps = np.arange(1, 17)

    ax1_r = ax1.twinx()
    ax1_r.spines["top"].set_visible(False)

    l1 = ax1.plot(steps, b_comp["exact_solve_curve_prop"], color=C_TEAL, marker="o", markersize=2.7, linewidth=1.4, label="Prop exact (49)")
    l2 = ax1.plot(steps, b_comp["exact_solve_curve_stalled"], color=C_BLUE, marker="s", markersize=2.7, linewidth=1.4, label="Stalled exact (51)")
    l3 = ax1_r.plot(steps, b_comp["conflicts_curve_stalled"], color=C_VERMILION, marker="^", markersize=2.7, linestyle="--", linewidth=1.25, label="Conflicts (right)")

    ax1.set_xlabel("Outer step s (21 calls/step)")
    ax1.set_ylabel("Exact board solve")
    ax1.set_ylim(-0.03, 1.09)
    ax1_r.set_ylabel("Violated units", color=C_VERMILION, labelpad=3)
    ax1_r.tick_params(axis="y", labelcolor=C_VERMILION, pad=2)
    ax1_r.set_ylim(-0.4, 15.5)
    ax1.set_xticks([1, 4, 8, 12, 16])
    ax1.set_title("(a) Conflict repair", pad=6)
    ax1.grid(axis="both")

    lines = l1 + l2 + l3
    ax1_r.legend(lines, [ln.get_label() for ln in lines], loc="center right", frameon=True, framealpha=0.95, edgecolor="#CBD5E0", fontsize=5.9, borderpad=0.22)

    # Panel (b): Contrast (LoopedChase lambda=0.1) vs TRM items256 Prop (73), WB-Stalled (183), S4 (64)
    lc01 = ctrl["looped_chase_contrast"]["q4_mean_by_lambda"]["0.1"]
    it = ctrl["trm_primary_items256"]

    groups = ["Chase\nlam=0.1", "Prop\n(n=73)", "Stalled\n(n=183)", "S4\n(n=64)"]
    clean_v = [
        lc01["clean"]["d9_16_mean"],
        it["clean"]["board_exact_prop"],
        it["clean"]["board_exact_stalled"],
        it["clean"]["by_stratum"]["S4_stalled"]["board_exact"],
    ]
    raw_v = [
        lc01["cursor_fiber_erased_eff"]["d9_16_mean"],
        it["fiber_erased"]["board_exact_prop"],
        it["fiber_erased"]["board_exact_stalled"],
        it["fiber_erased"]["by_stratum"]["S4_stalled"]["board_exact"],
    ]
    norm_v = [
        lc01["cursor_fiber_ln_preserved"]["d9_16_mean"],
        it["fiber_norm_restored"]["board_exact_prop"],
        it["fiber_norm_restored"]["board_exact_stalled"],
        it["fiber_norm_restored"]["by_stratum"]["S4_stalled"]["board_exact"],
    ]
    donor_v = [
        lc01["cursor_donor_fiber"]["d9_16_mean"],
        it["fiber_donor_swapped"]["board_exact_prop"],
        it["fiber_donor_swapped"]["board_exact_stalled"],
        it["fiber_donor_swapped"]["by_stratum"]["S4_stalled"]["board_exact"],
    ]
    frz_v = [
        lc01["cursor_step1_frozen"]["d9_16_mean"],
        it["fiber_step1_frozen"]["board_exact_prop"],
        it["fiber_step1_frozen"]["board_exact_stalled"],
        it["fiber_step1_frozen"]["by_stratum"]["S4_stalled"]["board_exact"],
    ]

    xg = np.arange(len(groups))
    w = 0.152
    series = [
        (clean_v, -2 * w, C_TEAL, "Clean"),
        (raw_v, -1 * w, C_VERMILION, "Raw P_head"),
        (norm_v, 0 * w, C_SKY, "Norm-restored"),
        (donor_v, 1 * w, C_BLUE, "Donor fiber"),
        (frz_v, 2 * w, C_PURPLE, "Step-1 frozen"),
    ]
    for vals, off, col, lbl in series:
        ax2.bar(xg + off, vals, width=w * 0.90, color=col, edgecolor="white", linewidth=0.3, label=lbl, zorder=3)

    ax2.set_xticks(xg)
    ax2.set_xticklabels(groups, fontsize=6.8)
    ax2.set_ylim(0.0, 1.48)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_ylabel("Accuracy / exact solve")
    ax2.set_title("(b) Manifold & donor controls", pad=6)
    ax2.grid(axis="y")
    ax2.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.95, edgecolor="#CBD5E0", fontsize=5.8, borderpad=0.22, columnspacing=0.6)

    # Panel (c): Top-k head-orthogonal PCA dose-response on items256 (Primary vs Replication)
    ks = [0, 4, 8, 16, 32, 64, 128, 256, 501]
    xk = np.arange(len(ks))
    rep_it = ext["trm_replication_items256"]
    prop_pca = [it[f"pca_k{k}"]["board_exact_prop"] for k in ks]
    stall_pca = [it[f"pca_k{k}"]["board_exact_stalled"] for k in ks]
    s4_pca = [it[f"pca_k{k}"]["by_stratum"]["S4_stalled"]["board_exact"] for k in ks]
    rep_stall_pca = [rep_it[f"pca_k{k}"]["board_exact_stalled"] for k in ks]

    ax3.plot(xk, prop_pca, color=C_TEAL, marker="o", markersize=2.8, linewidth=1.35, label="Prim. Prop (73)")
    ax3.plot(xk, stall_pca, color=C_BLUE, marker="s", markersize=2.8, linewidth=1.35, label="Prim. Stall (183)")
    ax3.plot(xk, rep_stall_pca, color=C_PURPLE, marker="D", markersize=2.6, linestyle="--", linewidth=1.25, label="Repl. Stall (183)")
    ax3.plot(xk, s4_pca, color=C_AMBER, marker="^", markersize=2.8, linewidth=1.35, label="Prim. S4 (64)")
    ax3.axhline(it["clean"]["board_exact_stalled"], color=C_BLUE, linestyle=":", linewidth=0.75)
    ax3.axhline(rep_it["clean"]["board_exact_stalled"], color=C_PURPLE, linestyle=":", linewidth=0.75)

    ax3.set_xticks([0, 2, 4, 5, 6, 8])
    ax3.set_xticklabels(["0", "8", "32", "64", "128", "501"], fontsize=6.7)
    ax3.set_xlabel("Retained orth. PCA dims k")
    ax3.set_ylabel("Exact board solve")
    ax3.set_ylim(0.04, 1.09)
    ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.set_title("(c) PCA dose curve", pad=6)
    ax3.grid(axis="both")
    ax3.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#CBD5E0", fontsize=5.5, borderpad=0.20)

    save_both(fig, "fig4_dark_fiber_ablation")


def main() -> None:
    configure_style()
    report = json.loads((LS02_RESULTS / "loopscope02_report.json").read_text(encoding="utf-8"))
    trm = json.loads((LS02_RESULTS / "trm_results.json").read_text(encoding="utf-8"))
    nb = json.loads((LS02_RESULTS / "nanbeige_results.json").read_text(encoding="utf-8"))
    ctrl = json.loads((LS02_RESULTS / "fiber_regime_controls.json").read_text(encoding="utf-8"))
    ext = json.loads((LS02_RESULTS / "fiber_replication_and_history_probes.json").read_text(encoding="utf-8"))

    build_fig1_triangulation(report, trm, nb)
    build_fig2_fragility_vs_specificity(trm)
    build_fig3_compactification_staircase(report)
    build_fig4_dark_fiber_ablation(report, ctrl, ext)
    print(f"Built 4 PDF + 4 PNG figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
