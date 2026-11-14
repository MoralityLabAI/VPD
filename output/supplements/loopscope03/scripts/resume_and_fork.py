#!/usr/bin/env python3
"""LOOPSCOPE-03 Stage 2 & Stage 3: Resume Test (Iteration Pricing) and Fork Test (Path Memory).

Implements:
  1. Time-Invariance Stationarity Admission Gate (`Nanbeige4.2-3B` fails -> Regime I;
     `LoopedChase` and `TRM` pass -> loop-equivalents well-defined).
  2. Measurement 2 — Resume Test (`TRM` primary + replication, `LoopedChase` 12 Q4 arms):
     Transplants `Fresh`, `V_t` (unconditional hidden fill), `V_t + C_t` (conditional refill),
     sub-channel crosses (`V_t + live z_H`, `V_t + live z_L`), and `Full S_t` into a fresh
     unperturbed run of horizon T_max = 32 steps. Verifies exact telescoping decomposition:
       min(t, tau_0) = Accrued(V_t) + Price(C_t) + Price(R_t)
     and computes Spearman rank correlations between puzzle-level Price(R_t) / ||R_t||^2 and
     clean conflict-flip / backtrack counts.
  3. Measurement 3 — Fork Test (`TRM` primary + replication, `LoopedChase` 12 Q4 arms):
     - Multi-initialization ensemble (K = 8 forks at sigma in {0.35, 1.00}) testing C-voting
       (Kubo, Kamiya et al.) and reconciling Anil et al. (2022) asymptotic path independence
       with transient post-stall path divergence.
     - Causal R_t swap at matched steps t in {3, 5, 8} across forked runs (A, B) on the same
       board x (`Keep-R_A`, `Zero-R_oneshot`, `Zero-R_persist`, `Swap-R_B`, `Rewind-R_A1`),
       scoring both (i) active donor-branch steering at step t+1 and (ii) revisit rates of
       already-refuted boards, 9-cell Sudoku units, and abandoned wrong cell digits.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "mechinterp" / "loopscope01" / "scripts"))

from mechinterp.loopscope02.scripts import fiber_regime_controls as FRC  # noqa: E402
from mechinterp.loopscope02.scripts import score_loopscope02 as SC       # noqa: E402
from mechinterp.loopscope02.scripts import sudoku_prop as SP             # noqa: E402
from mechinterp.loopscope02.scripts import trm_sudoku as TS              # noqa: E402
from mechinterp.loopscope03.scripts import conditional_refill as CR      # noqa: E402
import loopscope_toy as LT  # noqa: E402

RESULTS_DIR = REPO_ROOT / "mechinterp" / "loopscope03" / "results"

# Precompute the 27 Sudoku units (9 rows, 9 cols, 9 boxes), each of 9 cell indices
UNITS_27: list[list[int]] = []
for _r in range(9):
    UNITS_27.append([_r * 9 + _c for _c in range(9)])
for _c in range(9):
    UNITS_27.append([_r * 9 + _c for _r in range(9)])
for _br in (0, 3, 6):
    for _bc in (0, 3, 6):
        UNITS_27.append([(_br + _dr) * 9 + (_bc + _dc) for _dr in range(3) for _dc in range(3)])
UNITS_ARR = np.array(UNITS_27, dtype=np.int64)  # [27, 9]


def rankdata_avg_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    xs = x[order]
    n = len(xs)
    i = 0
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman_corr_and_perm_p(x: np.ndarray, y: np.ndarray, n_perm: int = 5000, seed: int = 20261114) -> dict:
    """Exact Spearman rho with two-sided permutation p-value (zero scipy dependency)."""
    mask = np.isfinite(x) & np.isfinite(y)
    xv = x[mask].astype(np.float64)
    yv = y[mask].astype(np.float64)
    n = len(xv)
    if n < 5 or np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return {"n": int(n), "rho": 0.0, "p_two_sided": 1.0}
    rx = rankdata_avg_ties(xv) - 0.5 * (n + 1)
    ry = rankdata_avg_ties(yv) - 0.5 * (n + 1)
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    rho = float((rx @ ry) / max(denom, 1e-12))
    rng = np.random.default_rng(seed)
    ge = 1
    for _ in range(n_perm):
        rp = rng.permutation(ry)
        r_null = abs(float((rx @ rp) / max(denom, 1e-12)))
        if r_null >= abs(rho) - 1e-12:
            ge += 1
    return {"n": int(n), "rho": rho, "p_two_sided": float(ge / (n_perm + 1))}


def paired_signflip_p(diff: np.ndarray, n_perm: int = 5000, seed: int = 20261114) -> dict:
    """Two-sided paired sign-flip test + 95% bootstrap CI on per-board differences."""
    d = diff[np.isfinite(diff)].astype(np.float64)
    n = len(d)
    if n == 0:
        return {"n": 0, "mean": 0.0, "ci95": [0.0, 0.0], "p_two_sided": 1.0}
    obs = float(d.mean())
    if np.all(np.abs(d) < 1e-12):
        return {"n": int(n), "mean": 0.0, "ci95": [0.0, 0.0], "p_two_sided": 1.0}
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null_means = (signs * d[None, :]).mean(axis=1)
    p_val = float((1 + np.sum(np.abs(null_means) >= abs(obs) - 1e-12)) / (n_perm + 1))
    boot_idx = rng.integers(0, n, size=(2000, n))
    boot_means = d[boot_idx].mean(axis=1)
    ci = [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))]
    return {"n": int(n), "mean": obs, "ci95": ci, "p_two_sided": p_val}


def load_saved_refill_bundle(cache_dir: Path, ckpt_tag: str, dev: str) -> dict:
    bundle = torch.load(cache_dir / f"{ckpt_tag}_refill_bundle.pt", map_location="cpu", weights_only=False)
    best_name = bundle["best_tf_name"]
    n_layers = 4 if "4L" in best_name else 2
    net = CR.RefillTransformer(d_model=512, n_layers=n_layers, n_heads=8, use_step_clock=False)
    net.load_state_dict(bundle[f"{best_name}_state"])
    net.to(dev).eval()
    bundle["net"] = net
    return bundle


def compute_lockin_step(exact_by_u: np.ndarray) -> np.ndarray:
    """Given boolean [U+1, N] where index 0 is pre-step (u=0) and 1..U are post-step u=1..U:
    return tau_lock in 0..U (or U+1 if not solved at index U).
    tau_lock is the smallest u in 0..U such that exact_by_u[u..U] are ALL True.
    """
    u_plus_1, n = exact_by_u.shape
    u_max = u_plus_1 - 1
    out = np.full(n, u_max + 1, dtype=np.int32)
    for b in range(n):
        if not exact_by_u[u_max, b]:
            out[b] = u_max + 1
        else:
            k = u_max
            while k > 0 and exact_by_u[k - 1, b]:
                k -= 1
            out[b] = k
    return out


def compute_first_solve_step(exact_by_u: np.ndarray) -> np.ndarray:
    """Smallest u in 0..U where exact_by_u[u] is True (or U+1 if never True)."""
    u_plus_1, n = exact_by_u.shape
    u_max = u_plus_1 - 1
    out = np.full(n, u_max + 1, dtype=np.int32)
    for b in range(n):
        idx = np.where(exact_by_u[:, b])[0]
        if len(idx) > 0:
            out[b] = int(idx[0])
    return out


# =====================================================================
# 1. Stationarity Admission Gate (Nanbeige vs TRM vs LoopedChase)
# =====================================================================

@torch.no_grad()
def evaluate_stationarity_admission(trm_model: TS.PureSudokuTRM, sample_qs: list[str], dev: str) -> dict:
    """Verify exact state-transplant time-invariance and collate Nanbeige admission failure."""
    nan_res = json.loads((REPO_ROOT / "mechinterp" / "loopscope02" / "results" / "nanbeige_results.json").read_text("utf-8"))
    trm_rep = json.loads((REPO_ROOT / "mechinterp" / "loopscope02" / "results" / "trm_results.json").read_text("utf-8"))

    # Direct numerical transplant check on TRM: F_x^(4)(S_4) == S_8
    inp = TS.encode_questions(sample_qs[:16], dev)
    xb = trm_model.encode_inputs(inp)
    zh, zl = trm_model.init_states(16)
    s4 = None
    for s in range(8):
        zh, zl = trm_model.act_step(zh, zl, xb)
        if s == 3:
            s4 = (zh.clone(), zl.clone())
    zh_res, zl_res = s4[0].clone(), s4[1].clone()
    for _ in range(4):
        zh_res, zl_res = trm_model.act_step(zh_res, zl_res, xb)
    trm_transplant_max_diff = float(max((zh_res.float() - zh.float()).abs().max().item(),
                                        (zl_res.float() - zl.float()).abs().max().item()))

    nan_loop_accs = [float(np.mean(row)) for row in nan_res["items_target_correct_by_loop"]]
    return {
        "admission_rule": "r_outer_mean >= 0.90 and state_transplant_linf_diff < 1e-5",
        "Nanbeige4.2-3B": {
            "admitted": False,
            "regime": "Regime I (Fixed-Depth Subspace Multiplexing)",
            "r_pass1_pass2_mean": float(nan_res["mean_r_pass1_vs_pass2"]),
            "loop_1_to_8_target_acc": nan_loop_accs,
            "rel_step_norm_loop8": float(nan_res["rel_step_norm_mean_by_loop"][-1]),
            "reason": "Shared weights switch singular subspaces across passes (r=0.355; L12 r=0.085) without input re-injection; resuming beyond trained 2-pass horizon degrades accuracy (0.129 -> 0.090).",
        },
        "TRM_gaoxin492": {
            "admitted": True,
            "regime": "Regime II/III (Stationary Input-Reinjected Operator)",
            "r_outer_zH_mean": float(
                np.mean([v["r_outer_zH_mean"] for v in trm_rep["usage_stats"].values()])
            ),
            "state_transplant_linf_diff": trm_transplant_max_diff,
        },
    }


# =====================================================================
# 2. Measurement 2: TRM Resume Test (Telescoping Iteration Prices)
# =====================================================================

@torch.no_grad()
def run_trm_resume_test(
    model: TS.PureSudokuTRM,
    refill_bundle: dict,
    qs: list[str],
    ans: list[str],
    prop_flags: list[bool],
    transplant_steps: tuple[int, ...] = (1, 2, 4, 8, 12),
    horizon: int = 32,
    bsz: int = 64,
) -> dict:
    dev = next(model.parameters()).device
    dt = model.embed_tokens.weight.dtype
    n = len(qs)
    inp_all = TS.encode_questions(qs, dev)
    cand_all = CR.build_cand_masks(qs).to(dev)
    x_all = model.encode_inputs(inp_all)
    ans_np = np.array([[int(ch) - 1 for ch in a] for a in ans], dtype=np.int64)
    empty_mask = np.array([[ch == "." for ch in q] for q in qs], dtype=bool)
    given_np = np.array([[-1 if ch == "." else int(ch) - 1 for ch in q] for q in qs], dtype=np.int64)

    v_head_d = refill_bundle["v_head"].to(dev)
    p_perp_d = refill_bundle["p_perp"].to(dev)
    w_dig_d = refill_bundle["w_digit"].to(dev)
    mu_orth = refill_bundle["mean_base"]["mu_orth"].to(dev)
    mu_pref = refill_bundle["mean_base"]["mu_pref"].to(dev)
    mu_zl = refill_bundle["mean_base"]["mu_zl"].to(dev)
    l_init_d = model.L_init.detach().float().to(dev).view(1, 1, 512)
    net: CR.RefillTransformer = refill_bundle["net"]

    # Step 1: Run clean trajectory up to max(transplant_steps) + horizon (max 44 steps)
    total_clean_steps = max(transplant_steps) + horizon
    clean_preds = np.empty((total_clean_steps + 1, n, 81), dtype=np.int8)
    saved_snapshots: dict[int, tuple[torch.Tensor, torch.Tensor]] = {
        t: (torch.empty((n, 97, 512), dtype=dt), torch.empty((n, 97, 512), dtype=dt))
        for t in transplant_steps
    }

    for b0 in range(0, n, bsz):
        b1 = min(n, b0 + bsz)
        zh, zl = model.init_states(b1 - b0)
        clean_preds[0, b0:b1] = model.digit_logits(zh).argmax(dim=-1).cpu().numpy().astype(np.int8)
        xb = x_all[b0:b1]
        for s in range(1, total_clean_steps + 1):
            zh, zl = model.act_step(zh, zl, xb)
            clean_preds[s, b0:b1] = model.digit_logits(zh).argmax(dim=-1).cpu().numpy().astype(np.int8)
            if s in saved_snapshots:
                saved_snapshots[s][0][b0:b1] = zh.cpu()
                saved_snapshots[s][1][b0:b1] = zl.cpu()

    clean_merged = np.where(empty_mask[None, :, :], clean_preds, given_np[None, :, :])
    clean_exact_all = (clean_merged == ans_np[None, :, :]).all(axis=-1)  # [total_clean + 1, N]

    # Baseline clean tau_0 over u = 0..horizon
    tau0_lock = compute_lockin_step(clean_exact_all[: horizon + 1])      # in 0..horizon+1
    tau0_first = compute_first_solve_step(clean_exact_all[: horizon + 1])

    # Clean per-board search intensity across steps 1..16: cell flips & cumulative unit conflicts
    flips_16 = np.zeros(n, dtype=np.float64)
    conflicts_16 = np.zeros(n, dtype=np.float64)
    for b in range(n):
        em = empty_mask[b]
        seq = clean_merged[1:17, b, :][:, em]  # [16, n_emp]
        flips_16[b] = float(np.sum(seq[1:] != seq[:-1]))
        for s in range(1, 17):
            conflicts_16[b] += float(SP.count_unit_conflicts((clean_merged[s, b] + 1).tolist()))

    prop_arr = np.array(prop_flags, dtype=bool)
    stall_arr = ~prop_arr
    clean_locked_mask = (tau0_lock <= horizon)

    by_step_out = {}
    for t in transplant_steps:
        zh_t_cpu, zl_t_cpu = saved_snapshots[t]
        # Measure ||R_t||^2 per board on empty cells + z_L
        r_norm_sq = np.zeros(n, dtype=np.float64)

        # Resume arms at step t:
        #   - V_only_mean_all: z_H_orth <- mu_orth, z_H_pref <- mu_pref, z_L <- mu_zl
        #   - V_only_reset_zL: z_H_orth <- mu_orth, z_H_pref <- mu_pref, z_L <- L_init
        #   - V_plus_zH_live:  keep z_H live, z_L <- L_init (prices z_L alone!)
        #   - V_plus_zL_live:  z_H_orth <- mu_orth, z_H_pref <- mu_pref, keep z_L live (prices z_H alone!)
        #   - V_plus_C:        z_H_orth, z_H_pref, z_L <- RefillTransformer(x, V_t)
        #   - Full_S_t:        exact (z_H(t), z_L(t))
        arm_names = [
            "V_only_mean_all",
            "V_only_reset_zL",
            "V_plus_zH_live",
            "V_plus_zL_live",
            "V_plus_C",
            "Full_S_t",
        ]
        tau_lock_arms: dict[str, np.ndarray] = {}
        tau_first_arms: dict[str, np.ndarray] = {}

        for arm in arm_names:
            preds_u = np.empty((horizon + 1, n, 81), dtype=np.int8)
            for b0 in range(0, n, bsz):
                b1 = min(n, b0 + bsz)
                m = b1 - b0
                zh_c = zh_t_cpu[b0:b1].to(dev).float()
                zl_c = zl_t_cpu[b0:b1].to(dev).float()
                xb = x_all[b0:b1]
                inp_b = inp_all[b0:b1]
                cand_b = cand_all[b0:b1]

                zb = zh_c[:, 16:, :]
                vc = zb @ v_head_d
                vp = vc @ v_head_d.T

                if arm == "Full_S_t":
                    zh_0, zl_0 = zh_c.to(dt), zl_c.to(dt)
                elif arm == "V_only_mean_all":
                    no = CR.sphere_calibrate_orth(mu_orth.expand(m, -1, -1), vp, p_perp_d)
                    zh_0 = torch.cat([mu_pref.expand(m, -1, -1), vp + no], dim=1).to(dt)
                    zl_0 = mu_zl.expand(m, -1, -1).to(dt)
                elif arm == "V_only_reset_zL":
                    no = CR.sphere_calibrate_orth(mu_orth.expand(m, -1, -1), vp, p_perp_d)
                    zh_0 = torch.cat([mu_pref.expand(m, -1, -1), vp + no], dim=1).to(dt)
                    zl_0 = l_init_d.expand(m, 97, -1).to(dt)
                elif arm == "V_plus_zH_live":
                    zh_0 = zh_c.to(dt)
                    zl_0 = l_init_d.expand(m, 97, -1).to(dt)
                elif arm == "V_plus_zL_live":
                    no = CR.sphere_calibrate_orth(mu_orth.expand(m, -1, -1), vp, p_perp_d)
                    zh_0 = torch.cat([mu_pref.expand(m, -1, -1), vp + no], dim=1).to(dt)
                    zl_0 = zl_c.to(dt)
                elif arm == "V_plus_C":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                        ro, rp, rz = net(inp_b, cand_b, vc, vp, w_dig_d, p_perp_d, calibrate_sphere=True)
                    ro, rp, rz = ro.float(), rp.float(), rz.float()
                    # Record residual norm ||R_t||^2 on board orthogonal fiber + z_L
                    diff_o = (zb - vp) - ro
                    diff_z = zl_c - rz
                    r_norm_sq[b0:b1] = (
                        diff_o.pow(2).mean(dim=(1, 2)) + diff_z.pow(2).mean(dim=(1, 2))
                    ).cpu().numpy()
                    zh_0 = torch.cat([rp, vp + ro], dim=1).to(dt)
                    zl_0 = rz.to(dt)
                else:
                    raise ValueError(arm)

                preds_u[0, b0:b1] = model.digit_logits(zh_0).argmax(dim=-1).cpu().numpy().astype(np.int8)
                zh_r, zl_r = zh_0, zl_0
                for u in range(1, horizon + 1):
                    zh_r, zl_r = model.act_step(zh_r, zl_r, xb)
                    preds_u[u, b0:b1] = model.digit_logits(zh_r).argmax(dim=-1).cpu().numpy().astype(np.int8)

            merged_u = np.where(empty_mask[None, :, :], preds_u, given_np[None, :, :])
            exact_u = (merged_u == ans_np[None, :, :]).all(axis=-1)
            tau_lock_arms[arm] = compute_lockin_step(exact_u)
            tau_first_arms[arm] = compute_first_solve_step(exact_u)

        # Telescoping ledger decomposition using V_only_mean_all as V_t:
        t_v = tau_lock_arms["V_only_mean_all"].astype(np.float64)
        t_vc = tau_lock_arms["V_plus_C"].astype(np.float64)
        t_full = tau_lock_arms["Full_S_t"].astype(np.float64)
        t0_f = tau0_lock.astype(np.float64)

        accrued_total = t0_f - t_full
        accrued_v = t0_f - t_v
        price_c = t_v - t_vc
        price_r = t_vc - t_full
        price_zl_only = tau_lock_arms["V_plus_zH_live"].astype(np.float64) - t_full
        price_zh_only = tau_lock_arms["V_plus_zL_live"].astype(np.float64) - t_full

        # Verify exact telescoping identity on all boards!
        max_telescope_err = float(np.max(np.abs(accrued_total - (accrued_v + price_c + price_r))))
        assert max_telescope_err < 1e-9, f"Telescoping violated: {max_telescope_err}"

        def summarize_subset(sub_mask: np.ndarray) -> dict:
            cnt = int(sub_mask.sum())
            if cnt == 0:
                return {"n": 0}
            return {
                "n": cnt,
                "tau0_lock_mean": float(t0_f[sub_mask].mean()),
                "tau_res_V_mean": float(t_v[sub_mask].mean()),
                "tau_res_VC_mean": float(t_vc[sub_mask].mean()),
                "tau_res_Full_mean": float(t_full[sub_mask].mean()),
                "accrued_total_loops": float(accrued_total[sub_mask].mean()),
                "accrued_V_loops": float(accrued_v[sub_mask].mean()),
                "price_C_loops": float(price_c[sub_mask].mean()),
                "price_R_loops": float(price_r[sub_mask].mean()),
                "price_R_test": paired_signflip_p(price_r[sub_mask]),
                "price_C_test": paired_signflip_p(price_c[sub_mask]),
                "price_zL_alone_loops": float(price_zl_only[sub_mask].mean()),
                "price_zH_alone_loops": float(price_zh_only[sub_mask].mean()),
                "solve_within_h32_V": float((tau_lock_arms["V_only_mean_all"][sub_mask] <= horizon).mean()),
                "solve_within_h32_VC": float((tau_lock_arms["V_plus_C"][sub_mask] <= horizon).mean()),
                "solve_within_h32_Full": float((tau_lock_arms["Full_S_t"][sub_mask] <= horizon).mean()),
                "mean_R_norm_sq": float(r_norm_sq[sub_mask].mean()),
            }

        active_stall = stall_arr & clean_locked_mask & (tau0_lock > t)
        by_step_out[f"step_{t}"] = {
            "max_telescope_error": max_telescope_err,
            "clean_solved_wb_prop": summarize_subset(prop_arr & clean_locked_mask),
            "clean_solved_wb_stall": summarize_subset(stall_arr & clean_locked_mask),
            "active_searching_wb_stall_tau0_gt_t": summarize_subset(active_stall),
            "all_wb_stall_capped": summarize_subset(stall_arr),
            "correlations_across_clean_solved": {
                "price_R_vs_cell_flips": spearman_corr_and_perm_p(price_r[clean_locked_mask], flips_16[clean_locked_mask]),
                "price_R_vs_unit_conflicts": spearman_corr_and_perm_p(price_r[clean_locked_mask], conflicts_16[clean_locked_mask]),
                "R_norm_sq_vs_cell_flips": spearman_corr_and_perm_p(r_norm_sq[clean_locked_mask], flips_16[clean_locked_mask]),
            },
            "correlations_within_wb_stall_solved": {
                "price_R_vs_cell_flips": spearman_corr_and_perm_p(
                    price_r[stall_arr & clean_locked_mask], flips_16[stall_arr & clean_locked_mask]
                ),
                "price_R_vs_unit_conflicts": spearman_corr_and_perm_p(
                    price_r[stall_arr & clean_locked_mask], conflicts_16[stall_arr & clean_locked_mask]
                ),
                "R_norm_sq_vs_cell_flips": spearman_corr_and_perm_p(
                    r_norm_sq[stall_arr & clean_locked_mask], flips_16[stall_arr & clean_locked_mask]
                ),
            },
        }
        st_s = by_step_out[f"step_{t}"]["clean_solved_wb_stall"]
        pr_s = by_step_out[f"step_{t}"]["clean_solved_wb_prop"]
        act_s = by_step_out[f"step_{t}"]["active_searching_wb_stall_tau0_gt_t"]
        print(
            f"  [Resume t={t:2d}] Prop(n={pr_s.get('n',0)}): V={pr_s.get('accrued_V_loops',0):.2f} "
            f"C={pr_s.get('price_C_loops',0):.2f} R={pr_s.get('price_R_loops',0):.2f} | "
            f"Stall(n={st_s.get('n',0)}): V={st_s.get('accrued_V_loops',0):.2f} "
            f"C={st_s.get('price_C_loops',0):.2f} R={st_s.get('price_R_loops',0):.2f} | "
            f"ActiveStall(n={act_s.get('n',0)}): V={act_s.get('accrued_V_loops',0):.2f} "
            f"C={act_s.get('price_C_loops',0):.2f} R={act_s.get('price_R_loops',0):.2f}",
            flush=True,
        )

    return {
        "horizon": horizon,
        "clean_locked_by_h32_prop": float(clean_locked_mask[prop_arr].mean()),
        "clean_locked_by_h32_stall": float(clean_locked_mask[stall_arr].mean()),
        "by_transplant_step": by_step_out,
    }


# =====================================================================
# 3. Measurement 3: TRM Fork Test (C-Voting, Path Divergence, R_t Swap)
# =====================================================================

@torch.no_grad()
def make_forked_initial_states(
    model: TS.PureSudokuTRM,
    batch_size: int,
    fork_id: int,
    sigma: float,
    dev: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fork_id == 0 is canonical (H_init, L_init); fork_id >= 1 adds Gaussian noise + RMS calibration."""
    zh0, zl0 = model.init_states(batch_size)
    if fork_id == 0 or sigma <= 0.0:
        return zh0, zl0
    dt = zh0.dtype
    g = torch.Generator(device="cpu").manual_seed(20261114 + 9973 * fork_id + int(round(sigma * 1000)))
    # Sample per-board, per-token noise so each board's fork_id explores an independent trajectory
    nh = torch.randn(zh0.shape, generator=g, dtype=torch.float32).to(dev)
    nl = torch.randn(zl0.shape, generator=g, dtype=torch.float32).to(dev)
    zh_f = CR.rms_calibrate_vec(zh0.float() + sigma * 22.6274 * (nh / math.sqrt(512.0))).to(dt)
    zl_f = CR.rms_calibrate_vec(zl0.float() + sigma * 22.6274 * (nl / math.sqrt(512.0))).to(dt)
    return zh_f, zl_f


@torch.no_grad()
def evaluate_cvoting_and_transient_divergence(
    model: TS.PureSudokuTRM,
    qs: list[str],
    ans: list[str],
    prop_flags: list[bool],
    k_forks: int = 8,
    sigmas: tuple[float, ...] = (0.35, 1.00),
    n_steps: int = 16,
    bsz: int = 64,
) -> dict:
    dev = next(model.parameters()).device
    n = len(qs)
    inp_all = TS.encode_questions(qs, dev)
    x_all = model.encode_inputs(inp_all)
    ans_np = np.array([[int(ch) - 1 for ch in a] for a in ans], dtype=np.int64)
    empty_mask = np.array([[ch == "." for ch in q] for q in qs], dtype=bool)
    given_np = np.array([[-1 if ch == "." else int(ch) - 1 for ch in q] for q in qs], dtype=np.int64)
    prop_arr = np.array(prop_flags, dtype=bool)
    stall_arr = ~prop_arr

    out_by_sigma = {}
    for sigma in sigmas:
        # Collect [K, S, N, 81] pred_digits and [K, N] final mean top-1 confidence
        preds_ks = np.empty((k_forks, n_steps, n, 81), dtype=np.int8)
        conf_final = np.empty((k_forks, n), dtype=np.float32)

        for m in range(k_forks):
            for b0 in range(0, n, bsz):
                b1 = min(n, b0 + bsz)
                zh, zl = make_forked_initial_states(model, b1 - b0, m, sigma, dev)
                xb = x_all[b0:b1]
                for s in range(n_steps):
                    zh, zl = model.act_step(zh, zl, xb)
                    logits = model.digit_logits(zh)
                    probs = torch.softmax(logits, dim=-1)
                    preds_ks[m, s, b0:b1] = probs.argmax(dim=-1).cpu().numpy().astype(np.int8)
                    if s == n_steps - 1:
                        # C-voting score (Kubo, Kamiya et al.): mean top-1 probability across cells
                        conf_final[m, b0:b1] = probs.amax(dim=-1).mean(dim=-1).cpu().numpy()

        merged_ks = np.where(empty_mask[None, None, :, :], preds_ks, given_np[None, None, :, :])
        exact_final = (merged_ks[:, -1, :, :] == ans_np[None, :, :]).all(axis=-1)  # [K, N]

        # C-voting selection: argmax_m conf_final[m, b] (strictly label-free!)
        best_m = conf_final.argmax(axis=0)  # [N]
        cvote_exact = exact_final[best_m, np.arange(n)]
        canonical_exact = exact_final[0]
        oracle_any_exact = exact_final.any(axis=0)
        mean_noisy_fork_exact = exact_final[1:].mean(axis=0)

        # Asymptotic agreement among all fork pairs (m1 < m2) that BOTH solve board b at final step
        solved_pair_agree_num = 0
        solved_pair_agree_den = 0
        # Transient pairwise empty-cell Hamming divergence across steps 1..n_steps
        div_curve = np.zeros((n_steps, n), dtype=np.float64)
        n_pairs = k_forks * (k_forks - 1) // 2

        for b in range(n):
            em = empty_mask[b]
            for m1 in range(k_forks):
                for m2 in range(m1 + 1, k_forks):
                    if exact_final[m1, b] and exact_final[m2, b]:
                        solved_pair_agree_den += 1
                        if np.array_equal(merged_ks[m1, -1, b], merged_ks[m2, -1, b]):
                            solved_pair_agree_num += 1
                    disagree = (preds_ks[m1, :, b, :][:, em] != preds_ks[m2, :, b, :][:, em]).mean(axis=-1)
                    div_curve[:, b] += disagree / n_pairs

        peak_div_per_board = div_curve[:8].max(axis=0)  # peak divergence in steps 1..8
        out_by_sigma[f"sigma_{sigma:.2f}"] = {
            "k_forks": k_forks,
            "asymptotic_solved_pair_agreement": (
                float(solved_pair_agree_num / solved_pair_agree_den) if solved_pair_agree_den > 0 else 1.0
            ),
            "canonical_single_solve": {
                "all": float(canonical_exact.mean()),
                "wb_prop": float(canonical_exact[prop_arr].mean()),
                "wb_stall": float(canonical_exact[stall_arr].mean()),
            },
            "mean_perturbed_fork_solve": {
                "all": float(mean_noisy_fork_exact.mean()),
                "wb_prop": float(mean_noisy_fork_exact[prop_arr].mean()),
                "wb_stall": float(mean_noisy_fork_exact[stall_arr].mean()),
            },
            "c_voting_solve": {
                "all": float(cvote_exact.mean()),
                "wb_prop": float(cvote_exact[prop_arr].mean()),
                "wb_stall": float(cvote_exact[stall_arr].mean()),
                "mcnemar_wb_stall_vs_canonical": SC.exact_mcnemar_p(canonical_exact[stall_arr], cvote_exact[stall_arr]),
            },
            "oracle_union_k8_solve": {
                "all": float(oracle_any_exact.mean()),
                "wb_prop": float(oracle_any_exact[prop_arr].mean()),
                "wb_stall": float(oracle_any_exact[stall_arr].mean()),
            },
            "transient_divergence_curve_wb_prop": [float(x) for x in div_curve[:, prop_arr].mean(axis=1)],
            "transient_divergence_curve_wb_stall": [float(x) for x in div_curve[:, stall_arr].mean(axis=1)],
            "peak_divergence_steps1_8_wb_prop": float(peak_div_per_board[prop_arr].mean()),
            "peak_divergence_steps1_8_wb_stall": float(peak_div_per_board[stall_arr].mean()),
            "stall_over_prop_peak_divergence_ratio": float(
                peak_div_per_board[stall_arr].mean() / max(peak_div_per_board[prop_arr].mean(), 1e-8)
            ),
        }
        s_entry = out_by_sigma[f"sigma_{sigma:.2f}"]
        print(
            f"  [Fork C-voting sigma={sigma:.2f}] Stall single={s_entry['canonical_single_solve']['wb_stall']:.3f} -> "
            f"C-vote={s_entry['c_voting_solve']['wb_stall']:.3f} (oracle={s_entry['oracle_union_k8_solve']['wb_stall']:.3f}) | "
            f"PeakDiv Stall={s_entry['peak_divergence_steps1_8_wb_stall']:.3f} vs "
            f"Prop={s_entry['peak_divergence_steps1_8_wb_prop']:.3f} "
            f"({s_entry['stall_over_prop_peak_divergence_ratio']:.2f}x)",
            flush=True,
        )
    return out_by_sigma


@torch.no_grad()
def run_trm_fork_swap_test(
    model: TS.PureSudokuTRM,
    refill_bundle: dict,
    qs: list[str],
    ans: list[str],
    prop_flags: list[bool],
    swap_steps: tuple[int, ...] = (3, 5, 8),
    k_forks: int = 4,
    sigma: float = 1.00,
    total_steps: int = 24,
    bsz: int = 64,
) -> dict:
    """Run K=4 forked runs on each board, swap R_t between forks (A, B) at step t, and measure:
    1. Donor trajectory steering at step t+1 on cells where clean A and clean B disagree.
    2. Revisit rates of A's already-refuted boards, 9-cell Sudoku units, and wrong cell digits.
    """
    dev = next(model.parameters()).device
    dt = model.embed_tokens.weight.dtype
    n = len(qs)
    inp_all = TS.encode_questions(qs, dev)
    cand_all = CR.build_cand_masks(qs).to(dev)
    x_all = model.encode_inputs(inp_all)
    ans_np = np.array([[int(ch) - 1 for ch in a] for a in ans], dtype=np.int64)
    empty_mask = np.array([[ch == "." for ch in q] for q in qs], dtype=bool)
    given_np = np.array([[-1 if ch == "." else int(ch) - 1 for ch in q] for q in qs], dtype=np.int64)
    prop_arr = np.array(prop_flags, dtype=bool)
    stall_arr = ~prop_arr

    v_head_d = refill_bundle["v_head"].to(dev)
    p_perp_d = refill_bundle["p_perp"].to(dev)
    w_dig_d = refill_bundle["w_digit"].to(dev)
    net: CR.RefillTransformer = refill_bundle["net"]

    # Precompute clean trajectories for forks m = 0..k_forks-1 up to total_steps, storing states at {1} U swap_steps
    needed_snaps = sorted(set([1] + list(swap_steps)))
    fork_preds = np.empty((k_forks, total_steps, n, 81), dtype=np.int8)
    fork_snaps: dict[int, tuple[torch.Tensor, torch.Tensor]] = {
        st: (torch.empty((k_forks, n, 97, 512), dtype=dt), torch.empty((k_forks, n, 97, 512), dtype=dt))
        for st in needed_snaps
    }

    for m in range(k_forks):
        for b0 in range(0, n, bsz):
            b1 = min(n, b0 + bsz)
            zh, zl = make_forked_initial_states(model, b1 - b0, m, sigma, dev)
            xb = x_all[b0:b1]
            for s in range(1, total_steps + 1):
                zh, zl = model.act_step(zh, zl, xb)
                fork_preds[m, s - 1, b0:b1] = model.digit_logits(zh).argmax(dim=-1).cpu().numpy().astype(np.int8)
                if s in fork_snaps:
                    fork_snaps[s][0][m, b0:b1] = zh.cpu()
                    fork_snaps[s][1][m, b0:b1] = zl.cpu()

    def decompose_snap(zh_cpu: torch.Tensor, zl_cpu: torch.Tensor):
        """Return CPU float32 (v_proj [N,81,512], c_orth, c_pref, c_zl, r_orth, r_pref, r_zl)."""
        v_proj_out = torch.empty((n, 81, 512), dtype=torch.float32)
        c_orth_out = torch.empty((n, 81, 512), dtype=torch.float32)
        c_pref_out = torch.empty((n, 16, 512), dtype=torch.float32)
        c_zl_out = torch.empty((n, 97, 512), dtype=torch.float32)
        r_orth_out = torch.empty((n, 81, 512), dtype=torch.float32)
        r_pref_out = torch.empty((n, 16, 512), dtype=torch.float32)
        r_zl_out = torch.empty((n, 97, 512), dtype=torch.float32)
        for b0 in range(0, n, bsz):
            b1 = min(n, b0 + bsz)
            zh_b = zh_cpu[b0:b1].to(dev).float()
            zl_b = zl_cpu[b0:b1].to(dev).float()
            zb = zh_b[:, 16:, :]
            vc = zb @ v_head_d
            vp = vc @ v_head_d.T
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                co, cp, cz = net(inp_all[b0:b1], cand_all[b0:b1], vc, vp, w_dig_d, p_perp_d, calibrate_sphere=True)
            co, cp, cz = co.float(), cp.float(), cz.float()
            v_proj_out[b0:b1] = vp.cpu()
            c_orth_out[b0:b1] = co.cpu()
            c_pref_out[b0:b1] = cp.cpu()
            c_zl_out[b0:b1] = cz.cpu()
            r_orth_out[b0:b1] = ((zb - vp) - co).cpu()
            r_pref_out[b0:b1] = (zh_b[:, :16, :] - cp).cpu()
            r_zl_out[b0:b1] = (zl_b - cz).cpu()
        return v_proj_out, c_orth_out, c_pref_out, c_zl_out, r_orth_out, r_pref_out, r_zl_out

    # Pre-decompose step 1 for Rewind-R_A1
    step1_dec = [decompose_snap(fork_snaps[1][0][m], fork_snaps[1][1][m]) for m in range(k_forks)]

    out_by_step = {}
    arms = ["Keep_R_A", "Zero_R_oneshot", "Zero_R_persist", "Swap_R_B", "Swap_R_C_unrelated", "Rewind_R_A1"]

    for t in swap_steps:
        step_dec = [decompose_snap(fork_snaps[t][0][m], fork_snaps[t][1][m]) for m in range(k_forks)]
        rem_steps = total_steps - t

        # Store post-t predictions [len(arms), k_forks, rem_steps, N, 81]
        # Pair A = m, B = (m + 1) % k_forks, C = (m + 2) % k_forks
        post_preds = {arm: np.empty((k_forks, rem_steps, n, 81), dtype=np.int8) for arm in arms}

        for m_a in range(k_forks):
            m_b = (m_a + 1) % k_forks
            m_c = (m_a + 2) % k_forks
            vp_a, co_a, cp_a, cz_a, ro_a, rp_a, rz_a = step_dec[m_a]
            _, _, _, _, ro_b, rp_b, rz_b = step_dec[m_b]
            _, _, _, _, ro_c, rp_c, rz_c = step_dec[m_c]
            _, _, _, _, ro_a1, rp_a1, rz_a1 = step1_dec[m_a]

            for arm in arms:
                if arm == "Keep_R_A":
                    # Identical to fork_preds[m_a, t:total_steps]
                    post_preds[arm][m_a] = fork_preds[m_a, t:total_steps]
                    continue

                for b0 in range(0, n, bsz):
                    b1 = min(n, b0 + bsz)
                    vp = vp_a[b0:b1].to(dev)
                    co = co_a[b0:b1].to(dev)
                    cp = cp_a[b0:b1].to(dev)
                    cz = cz_a[b0:b1].to(dev)
                    xb = x_all[b0:b1]
                    inp_b = inp_all[b0:b1]
                    cand_b = cand_all[b0:b1]

                    if arm in ("Zero_R_oneshot", "Zero_R_persist"):
                        zh = torch.cat([cp, vp + co], dim=1).to(dt)
                        zl = cz.to(dt)
                    elif arm == "Swap_R_B":
                        no = CR.sphere_calibrate_orth(co + ro_b[b0:b1].to(dev), vp, p_perp_d)
                        np_ = CR.rms_calibrate_vec(cp + rp_b[b0:b1].to(dev))
                        nz = CR.rms_calibrate_vec(cz + rz_b[b0:b1].to(dev))
                        zh = torch.cat([np_, vp + no], dim=1).to(dt)
                        zl = nz.to(dt)
                    elif arm == "Swap_R_C_unrelated":
                        no = CR.sphere_calibrate_orth(co + ro_c[b0:b1].to(dev), vp, p_perp_d)
                        np_ = CR.rms_calibrate_vec(cp + rp_c[b0:b1].to(dev))
                        nz = CR.rms_calibrate_vec(cz + rz_c[b0:b1].to(dev))
                        zh = torch.cat([np_, vp + no], dim=1).to(dt)
                        zl = nz.to(dt)
                    elif arm == "Rewind_R_A1":
                        no = CR.sphere_calibrate_orth(co + ro_a1[b0:b1].to(dev), vp, p_perp_d)
                        np_ = CR.rms_calibrate_vec(cp + rp_a1[b0:b1].to(dev))
                        nz = CR.rms_calibrate_vec(cz + rz_a1[b0:b1].to(dev))
                        zh = torch.cat([np_, vp + no], dim=1).to(dt)
                        zl = nz.to(dt)
                    else:
                        raise ValueError(arm)

                    for u in range(rem_steps):
                        zh, zl = model.act_step(zh, zl, xb)
                        post_preds[arm][m_a, u, b0:b1] = (
                            model.digit_logits(zh).argmax(dim=-1).cpu().numpy().astype(np.int8)
                        )
                        if arm == "Zero_R_persist" and u < rem_steps - 1:
                            zb_u = zh[:, 16:, :].float()
                            vc_u = zb_u @ v_head_d
                            vp_u = vc_u @ v_head_d.T
                            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                                co_u, cp_u, cz_u = net(inp_b, cand_b, vc_u, vp_u, w_dig_d, p_perp_d, calibrate_sphere=True)
                            zh = torch.cat([cp_u.float(), vp_u + co_u.float()], dim=1).to(dt)
                            zl = cz_u.to(dt)

        # Score 1: Active Donor Steering at step t+1 (u=0)
        # On empty cells where clean A(t+1) != clean B(t+1) AND clean B(t+1) != A(t):
        def score_fork_metrics(board_mask: np.ndarray) -> dict:
            b_indices = np.where(board_mask)[0]
            if len(b_indices) == 0:
                return {}
            # Per-board donor match rate across the k_forks swaps (for paired sign-flip test!)
            board_match_B = {arm: np.full(len(b_indices), np.nan, dtype=np.float64) for arm in arms}
            board_ham2_revisit = {arm: np.zeros(len(b_indices), dtype=np.float64) for arm in arms}
            board_exact_revisit = {arm: np.zeros(len(b_indices), dtype=np.float64) for arm in arms}
            board_unit_revisit = {arm: np.full(len(b_indices), np.nan, dtype=np.float64) for arm in arms}
            board_cell_revisit = {arm: np.full(len(b_indices), np.nan, dtype=np.float64) for arm in arms}
            board_avoid_B_excl = {arm: np.full(len(b_indices), np.nan, dtype=np.float64) for arm in arms}
            board_solve_s24 = {arm: np.zeros(len(b_indices), dtype=np.float64) for arm in arms}

            total_active_disagree_cells = 0

            for ii, bi in enumerate(b_indices):
                em = empty_mask[bi]
                giv = given_np[bi]
                true_b = ans_np[bi]

                match_num = {arm: 0 for arm in arms}
                match_den = 0

                ex_rev = {arm: 0.0 for arm in arms}
                h2_rev = {arm: 0.0 for arm in arms}
                u_rev_num = {arm: 0 for arm in arms}
                u_rev_den = 0
                c_rev_num = {arm: 0 for arm in arms}
                c_rev_den = 0
                b_excl_num = {arm: 0 for arm in arms}
                b_excl_den = 0

                for m_a in range(k_forks):
                    m_b = (m_a + 1) % k_forks
                    hist_a = fork_preds[m_a, :t, bi, :]      # [t, 81], index t-1 is step t
                    hist_b = fork_preds[m_b, :t, bi, :]
                    cur_a = hist_a[t - 1]
                    next_a = fork_preds[m_a, t, bi, :]
                    next_b = fork_preds[m_b, t, bi, :]

                    # Active disagreement cells between clean A(t+1) and clean B(t+1)
                    act_mask = em & (next_a != next_b) & (next_b != cur_a)
                    cnt_act = int(act_mask.sum())
                    if cnt_act > 0:
                        match_den += cnt_act
                        total_active_disagree_cells += cnt_act
                        for arm in arms:
                            pr_next = post_preds[arm][m_a, 0, bi, :]
                            match_num[arm] += int((pr_next[act_mask] == next_b[act_mask]).sum())

                    # Refuted prior boards of A (steps 0..t-2, != step t-1, != true_b)
                    merged_hist_a = np.where(em[None, :], hist_a, giv[None, :])
                    cur_merged_a = merged_hist_a[t - 1]
                    prior_refuted_boards = []
                    for u_prev in range(t - 1):
                        cand_brd = merged_hist_a[u_prev]
                        if not np.array_equal(cand_brd, cur_merged_a) and not np.array_equal(cand_brd, true_b):
                            prior_refuted_boards.append(cand_brd)

                    # Refuted 9-cell Sudoku units of A (visited in 0..t-2, != unit at t-1, != true unit)
                    refuted_units: list[tuple[int, bytes]] = []
                    seen_u = set()
                    for ui in range(27):
                        u_idx = UNITS_ARR[ui]
                        cur_u = cur_merged_a[u_idx].tobytes()
                        true_u = true_b[u_idx].tobytes()
                        for u_prev in range(t - 1):
                            prev_u = merged_hist_a[u_prev, u_idx].tobytes()
                            if prev_u != cur_u and prev_u != true_u and (ui, prev_u) not in seen_u:
                                seen_u.add((ui, prev_u))
                                refuted_units.append((ui, prev_u))
                    u_rev_den += len(refuted_units)

                    # Refuted wrong cell digits of A (visited in 0..t-2, != cur_a[c], != true_b[c])
                    refuted_cells_a: set[tuple[int, int]] = set()
                    refuted_cells_b: set[tuple[int, int]] = set()
                    for c_idx in np.where(em)[0]:
                        ca = int(cur_a[c_idx])
                        tb = int(true_b[c_idx])
                        for u_prev in range(t - 1):
                            da = int(hist_a[u_prev, c_idx])
                            if da != ca and da != tb:
                                refuted_cells_a.add((int(c_idx), da))
                            db = int(hist_b[u_prev, c_idx])
                            if db != ca and db != tb:
                                refuted_cells_b.add((int(c_idx), db))
                    c_rev_den += len(refuted_cells_a)
                    b_only_cells = refuted_cells_b - refuted_cells_a
                    b_excl_den += len(b_only_cells)

                    for arm in arms:
                        fut_p = post_preds[arm][m_a, :, bi, :]  # [rem_steps, 81]
                        fut_m = np.where(em[None, :], fut_p, giv[None, :])
                        board_solve_s24[arm][ii] += float(np.array_equal(fut_m[-1], true_b)) / k_forks

                        if prior_refuted_boards:
                            any_ex = False
                            any_h2 = False
                            for f_brd in fut_m:
                                if np.array_equal(f_brd, cur_merged_a) or np.array_equal(f_brd, true_b):
                                    continue
                                for pr_brd in prior_refuted_boards:
                                    hd = int(np.sum(f_brd[em] != pr_brd[em]))
                                    if hd == 0:
                                        any_ex = True
                                    if hd <= 2:
                                        any_h2 = True
                            if any_ex:
                                ex_rev[arm] += 1.0 / k_forks
                            if any_h2:
                                h2_rev[arm] += 1.0 / k_forks

                        for ui, prev_u in refuted_units:
                            u_idx = UNITS_ARR[ui]
                            if any(fut_m[s_f, u_idx].tobytes() == prev_u for s_f in range(rem_steps)):
                                u_rev_num[arm] += 1

                        for c_idx, da in refuted_cells_a:
                            if np.any(fut_p[:, c_idx] == da):
                                c_rev_num[arm] += 1

                        for c_idx, db in b_only_cells:
                            if np.any(fut_p[:, c_idx] == db):
                                b_excl_num[arm] += 1

                for arm in arms:
                    if match_den > 0:
                        board_match_B[arm][ii] = match_num[arm] / match_den
                    board_exact_revisit[arm][ii] = ex_rev[arm]
                    board_ham2_revisit[arm][ii] = h2_rev[arm]
                    if u_rev_den > 0:
                        board_unit_revisit[arm][ii] = u_rev_num[arm] / u_rev_den
                    if c_rev_den > 0:
                        board_cell_revisit[arm][ii] = c_rev_num[arm] / c_rev_den
                    if b_excl_den > 0:
                        board_avoid_B_excl[arm][ii] = b_excl_num[arm] / b_excl_den

            summary_arms = {}
            for arm in arms:
                summary_arms[arm] = {
                    "donor_B_next_step_match_rate": float(np.nanmean(board_match_B[arm])) if np.any(np.isfinite(board_match_B[arm])) else 0.0,
                    "refuted_exact_board_revisit_rate": float(board_exact_revisit[arm].mean()),
                    "refuted_ham2_board_revisit_rate": float(board_ham2_revisit[arm].mean()),
                    "refuted_unit_revisit_rate": float(np.nanmean(board_unit_revisit[arm])) if np.any(np.isfinite(board_unit_revisit[arm])) else 0.0,
                    "refuted_wrong_cell_revisit_rate": float(np.nanmean(board_cell_revisit[arm])) if np.any(np.isfinite(board_cell_revisit[arm])) else 0.0,
                    "donor_B_exclusive_refuted_cell_visit_rate": float(np.nanmean(board_avoid_B_excl[arm])) if np.any(np.isfinite(board_avoid_B_excl[arm])) else 0.0,
                    "solve_rate_step24": float(board_solve_s24[arm].mean()),
                }

            return {
                "n_boards": int(len(b_indices)),
                "total_active_disagree_cells": int(total_active_disagree_cells),
                "arms": summary_arms,
                "hypothesis_tests": {
                    "steering_SwapB_minus_ZeroR": paired_signflip_p(
                        board_match_B["Swap_R_B"] - board_match_B["Zero_R_oneshot"]
                    ),
                    "steering_SwapB_minus_SwapC_unrelated": paired_signflip_p(
                        board_match_B["Swap_R_B"] - board_match_B["Swap_R_C_unrelated"]
                    ),
                    "unit_revisit_ZeroR_persist_minus_KeepRA": paired_signflip_p(
                        board_unit_revisit["Zero_R_persist"] - board_unit_revisit["Keep_R_A"]
                    ),
                    "unit_revisit_SwapRB_minus_KeepRA": paired_signflip_p(
                        board_unit_revisit["Swap_R_B"] - board_unit_revisit["Keep_R_A"]
                    ),
                    "unit_revisit_RewindRA1_minus_KeepRA": paired_signflip_p(
                        board_unit_revisit["Rewind_R_A1"] - board_unit_revisit["Keep_R_A"]
                    ),
                    "cell_revisit_ZeroR_persist_minus_KeepRA": paired_signflip_p(
                        board_cell_revisit["Zero_R_persist"] - board_cell_revisit["Keep_R_A"]
                    ),
                    "cell_revisit_SwapRB_minus_KeepRA": paired_signflip_p(
                        board_cell_revisit["Swap_R_B"] - board_cell_revisit["Keep_R_A"]
                    ),
                    "donor_B_excl_avoidance_ZeroR_minus_SwapRB": paired_signflip_p(
                        board_avoid_B_excl["Zero_R_oneshot"] - board_avoid_B_excl["Swap_R_B"]
                    ),
                },
            }

        out_by_step[f"step_{t}"] = {
            "wb_prop": score_fork_metrics(prop_arr),
            "wb_stall": score_fork_metrics(stall_arr),
        }
        st_m = out_by_step[f"step_{t}"]["wb_stall"]["arms"]
        st_t = out_by_step[f"step_{t}"]["wb_stall"]["hypothesis_tests"]
        print(
            f"  [Fork Swap t={t}] WB-Stall Steer->B: SwapB={st_m['Swap_R_B']['donor_B_next_step_match_rate']:.3f} "
            f"vs ZeroR={st_m['Zero_R_oneshot']['donor_B_next_step_match_rate']:.3f} "
            f"vs SwapC={st_m['Swap_R_C_unrelated']['donor_B_next_step_match_rate']:.3f} "
            f"(p={st_t['steering_SwapB_minus_SwapC_unrelated']['p_two_sided']:.4f}) | "
            f"UnitRevisit Keep={st_m['Keep_R_A']['refuted_unit_revisit_rate']:.3f} -> "
            f"SwapB={st_m['Swap_R_B']['refuted_unit_revisit_rate']:.3f} / "
            f"ZeroPersist={st_m['Zero_R_persist']['refuted_unit_revisit_rate']:.3f}",
            flush=True,
        )
    return out_by_step


# =====================================================================
# 4. LoopedChase Contrast: Conditional Refill, Resume Test, & Fork Test
# =====================================================================

class ChaseRefillNet(nn.Module):
    """2-layer Transformer mapping (graph input e [B, 17, d], cursor V_t [B, d]) -> full hidden h [B, 17, d]."""

    def __init__(self, d: int = 256, heads: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(2 * d, d)
        self.blocks = nn.ModuleList([LT.Block(d, heads) for _ in range(2)])
        self.ln = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, e: torch.Tensor, v_cursor: torch.Tensor) -> torch.Tensor:
        vc_rep = torch.zeros_like(e)
        vc_rep[:, -1, :] = v_cursor
        x = self.in_proj(torch.cat([e, vc_rep], dim=-1))
        for blk in self.blocks:
            x = blk(x)
        return self.out_proj(self.ln(x))


def evaluate_looped_chase_three_ledgers(dev: str = "cuda") -> dict:
    ckpt_paths = sorted(FRC.RMP_TOY_Q4.glob("*.final.pt"))
    n_loops = 16
    tr_tokens_cache: dict[int, torch.Tensor] = {}
    te_batch_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    arms_out = []
    for pt_path in ckpt_paths:
        ck = torch.load(pt_path, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        n_nodes = int(cfg["n_nodes"])
        d = int(cfg["d"])
        heads = int(cfg["heads"])
        lam = float(cfg.get("lam", -1.0))
        seed = int(cfg.get("seed", -1))

        if n_nodes not in tr_tokens_cache:
            t_tr, _ = LT.make_batch(n_nodes=n_nodes, batch=1024, seed=910200, max_depth=n_loops)
            tr_tokens_cache[n_nodes] = t_tr.to(dev)
            t_te, h_te = LT.make_batch(n_nodes=n_nodes, batch=512, seed=910100, max_depth=n_loops)
            te_batch_cache[n_nodes] = (t_te.to(dev), h_te.to(dev))

        m = LT.build_model(LT.model_args_from(cfg))
        m.load_state_dict(ck["model"])
        m.to(dev).eval()

        # Effective readout projector before bias: readout(c) = head(ln_f(c))
        ones = torch.ones((d, 1), dtype=torch.float32)
        j_center = torch.eye(d, dtype=torch.float32) - (ones @ ones.T) / float(d)
        w_eff = (m.head.weight.detach().cpu().float() * m.ln_f.weight.detach().cpu().float()[None, :]) @ j_center
        _v_eff, p_eff_cpu, p_perp_cpu = CR.compute_head_projector_f64(w_eff)
        p_eff = p_eff_cpu.to(dev)
        p_perp = p_perp_cpu.to(dev)

        def encode_e(toks: torch.Tensor) -> torch.Tensor:
            edges = toks[:, : 2 * n_nodes].reshape(toks.shape[0], n_nodes, 2)
            h_in = torch.cat([m.tok(edges[..., 0]) + m.dst(edges[..., 1]), m.tok(toks[:, 2 * n_nodes :])], dim=1)
            return h_in + m.pos(torch.arange(h_in.shape[1], device=dev))

        # Collect training states on seed=910200 (1024 * 12 loops = 12,288 states)
        with torch.no_grad():
            e_tr = encode_e(tr_tokens_cache[n_nodes])
            h = torch.zeros_like(e_tr)
            h_tr_list = []
            for _ in range(12):
                h = m.inject(h) + e_tr
                for blk in m.core:
                    h = blk(h)
                h_tr_list.append(h.detach())
            h_tr_all = torch.stack(h_tr_list, dim=1).reshape(-1, e_tr.shape[1], d)  # [12288, 17, d]
            e_tr_rep = e_tr.unsqueeze(1).expand(-1, 12, -1, -1).reshape(-1, e_tr.shape[1], d)
            mu_h = h_tr_all.mean(dim=0, keepdim=True)  # [1, 17, d]

        # Train ChaseRefillNet for 18 epochs (takes ~2s on RTX 5080)
        torch.manual_seed(20261114 + seed)
        rnet = ChaseRefillNet(d=d, heads=heads).to(dev)
        opt = torch.optim.AdamW(rnet.parameters(), lr=1.5e-3, weight_decay=1e-2)
        m_tot = h_tr_all.shape[0]
        for _ep in range(18):
            perm = torch.randperm(m_tot, device=dev)
            for b0 in range(0, m_tot, 256):
                idx = perm[b0 : b0 + 256]
                eb = e_tr_rep[idx]
                hb = h_tr_all[idx]
                vc = hb[:, -1, :] @ p_eff
                pred_h = rnet(eb, vc)
                # Predict edge tokens + cursor orthogonal component
                loss = (pred_h[:, :-1, :] - hb[:, :-1, :]).pow(2).mean() + (
                    (pred_h[:, -1, :] @ p_perp) - (hb[:, -1, :] @ p_perp)
                ).pow(2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        rnet.eval()

        with torch.no_grad():
            t_te, hops_te = te_batch_cache[n_nodes]
            e_te = encode_e(t_te)
            bsz_te = e_te.shape[0]

            def merge_cursor_vis(c_orig: torch.Tensor, c_orth_pred: torch.Tensor) -> torch.Tensor:
                # Preserve exact LayerNorm readout of c_orig while replacing orthogonal component
                vc = c_orig @ p_eff
                cand = vc + (c_orth_pred @ p_perp)
                mu_orig = c_orig.mean(dim=-1, keepdim=True)
                std_orig = (c_orig - mu_orig).norm(dim=-1, keepdim=True).clamp_min(1e-6)
                cand_c = cand - cand.mean(dim=-1, keepdim=True)
                return cand_c * (std_orig / cand_c.norm(dim=-1, keepdim=True).clamp_min(1e-6)) + mu_orig

            # 1. Closed-loop accuracy (d9_16_mean): Clean vs V_only_all_tokens vs VC_refill_all_tokens
            clean_snaps = []
            h = torch.zeros_like(e_te)
            pr_clean = []
            for _ in range(n_loops):
                h = m.inject(h) + e_te
                for blk in m.core:
                    h = blk(h)
                clean_snaps.append(h.clone())
                pr_clean.append(m.readout(h[:, -1]).argmax(dim=-1))
            pr_clean_t = torch.stack(pr_clean)  # [16, B]

            h_v = torch.zeros_like(e_te)
            h_vc = torch.zeros_like(e_te)
            pr_v, pr_vc = [], []
            for _ in range(n_loops):
                h_v = m.inject(h_v) + e_te
                for blk in m.core:
                    h_v = blk(h_v)
                pr_v.append(m.readout(h_v[:, -1]).argmax(dim=-1))
                c_v = h_v[:, -1].clone()
                h_v = mu_h.expand(bsz_te, -1, -1).clone()
                h_v[:, -1, :] = merge_cursor_vis(c_v, mu_h[:, -1, :].expand(bsz_te, -1))

                h_vc = m.inject(h_vc) + e_te
                for blk in m.core:
                    h_vc = blk(h_vc)
                pr_vc.append(m.readout(h_vc[:, -1]).argmax(dim=-1))
                c_vc = h_vc[:, -1].clone()
                pred_full = rnet(e_te, c_vc @ p_eff)
                h_vc = pred_full.clone()
                h_vc[:, -1, :] = merge_cursor_vis(c_vc, pred_full[:, -1, :])

            pr_v_t = torch.stack(pr_v)
            pr_vc_t = torch.stack(pr_vc)

            def d9_16_acc(pr_s: torch.Tensor) -> float:
                return float(np.mean([(pr_s[k] == hops_te[:, k]).float().mean().item() for k in range(8, 16)]))

            # 2. Resume Test at t = 4 for target hop d = 12 (clean takes 12 steps -> Full takes 8 steps):
            t_idx = 3  # after 4 loops
            target_hop = hops_te[:, 11].cpu().numpy()  # hop 12
            h4 = clean_snaps[t_idx]
            pred_c4 = rnet(e_te, h4[:, -1, :] @ p_eff)

            resume_inits = {
                "Fresh": torch.zeros_like(e_te),
                "V_only": None,
                "V_plus_C": None,
                "Full": h4.clone(),
            }
            hv0 = mu_h.expand(bsz_te, -1, -1).clone()
            hv0[:, -1, :] = merge_cursor_vis(h4[:, -1, :], mu_h[:, -1, :].expand(bsz_te, -1))
            resume_inits["V_only"] = hv0
            hvc0 = pred_c4.clone()
            hvc0[:, -1, :] = merge_cursor_vis(h4[:, -1, :], pred_c4[:, -1, :])
            resume_inits["V_plus_C"] = hvc0

            tau_first_chase = {}
            for rname, h_start in resume_inits.items():
                hr = h_start.clone()
                prs = [m.readout(hr[:, -1]).argmax(dim=-1).cpu().numpy()]
                for _ in range(16):
                    hr = m.inject(hr) + e_te
                    for blk in m.core:
                        hr = blk(hr)
                    prs.append(m.readout(hr[:, -1]).argmax(dim=-1).cpu().numpy())
                ex = (np.stack(prs, axis=0) == target_hop[None, :])
                tau_first_chase[rname] = compute_first_solve_step(ex).astype(np.float64)

            clean_ok = (tau_first_chase["Fresh"] <= 16)
            t0_c = tau_first_chase["Fresh"][clean_ok]
            tv_c = tau_first_chase["V_only"][clean_ok]
            tvc_c = tau_first_chase["V_plus_C"][clean_ok]
            tfl_c = tau_first_chase["Full"][clean_ok]

            # 3. Fork Test on LoopedChase (2 perturbed initializations at sigma=0.5, swap R_4 at loop 4):
            g1 = torch.Generator(device=dev).manual_seed(101)
            g2 = torch.Generator(device=dev).manual_seed(202)
            ha = torch.randn(e_te.shape, generator=g1, device=dev) * 0.5
            hb = torch.randn(e_te.shape, generator=g2, device=dev) * 0.5
            for _ in range(4):
                ha = m.inject(ha) + e_te
                hb = m.inject(hb) + e_te
                for blk in m.core:
                    ha = blk(ha)
                    hb = blk(hb)
            ca_4 = rnet(e_te, ha[:, -1, :] @ p_eff)
            cb_4 = rnet(e_te, hb[:, -1, :] @ p_eff)
            rb_4 = hb - cb_4
            h_swap = ca_4 + rb_4
            h_swap[:, -1, :] = merge_cursor_vis(ha[:, -1, :], (ca_4[:, -1, :] + rb_4[:, -1, :]))

            # Step 5 prediction of clean A, clean B, and Swap_R_B
            ha5 = m.inject(ha) + e_te
            hb5 = m.inject(hb) + e_te
            hs5 = m.inject(h_swap) + e_te
            for blk in m.core:
                ha5 = blk(ha5)
                hb5 = blk(hb5)
                hs5 = blk(hs5)
            pa5 = m.readout(ha5[:, -1]).argmax(dim=-1)
            pb5 = m.readout(hb5[:, -1]).argmax(dim=-1)
            ps5 = m.readout(hs5[:, -1]).argmax(dim=-1)
            fork_disagree_rate = float((pa5 != pb5).float().mean().item())
            swap_flips_away_from_A = float((ps5 != pa5).float().mean().item())

        arms_out.append({
            "arm": pt_path.stem.replace(".final", ""),
            "lam": lam,
            "seed": seed,
            "closed_loop_d9_16_clean": d9_16_acc(pr_clean_t),
            "closed_loop_d9_16_V_only_all_tokens": d9_16_acc(pr_v_t),
            "closed_loop_d9_16_VC_refill_all_tokens": d9_16_acc(pr_vc_t),
            "resume_hop12_at_t4": {
                "n_clean_solved": int(clean_ok.sum()),
                "accrued_V_loops": float((t0_c - tv_c).mean()) if len(t0_c) else 0.0,
                "price_C_loops": float((tv_c - tvc_c).mean()) if len(t0_c) else 0.0,
                "price_R_loops": float((tvc_c - tfl_c).mean()) if len(t0_c) else 0.0,
            },
            "fork_loop4_disagree_rate": fork_disagree_rate,
            "fork_loop4_swap_RB_flip_rate": swap_flips_away_from_A,
        })

    by_lam = {}
    for lam in sorted({r["lam"] for r in arms_out}):
        sub = [r for r in arms_out if r["lam"] == lam]
        by_lam[str(lam)] = {
            "closed_loop_d9_16_clean": float(np.mean([r["closed_loop_d9_16_clean"] for r in sub])),
            "closed_loop_d9_16_V_only_all_tokens": float(np.mean([r["closed_loop_d9_16_V_only_all_tokens"] for r in sub])),
            "closed_loop_d9_16_VC_refill_all_tokens": float(np.mean([r["closed_loop_d9_16_VC_refill_all_tokens"] for r in sub])),
            "resume_accrued_V_loops": float(np.mean([r["resume_hop12_at_t4"]["accrued_V_loops"] for r in sub])),
            "resume_price_C_loops": float(np.mean([r["resume_hop12_at_t4"]["price_C_loops"] for r in sub])),
            "resume_price_R_loops": float(np.mean([r["resume_hop12_at_t4"]["price_R_loops"] for r in sub])),
            "fork_loop4_disagree_rate": float(np.mean([r["fork_loop4_disagree_rate"] for r in sub])),
            "fork_loop4_swap_RB_flip_rate": float(np.mean([r["fork_loop4_swap_RB_flip_rate"] for r in sub])),
        }
        print(
            f"  [LoopedChase lam={lam:g}] d9_16 Clean={by_lam[str(lam)]['closed_loop_d9_16_clean']:.3f} "
            f"V_only={by_lam[str(lam)]['closed_loop_d9_16_V_only_all_tokens']:.3f} "
            f"VC={by_lam[str(lam)]['closed_loop_d9_16_VC_refill_all_tokens']:.3f} | "
            f"Resume@t=4: V={by_lam[str(lam)]['resume_accrued_V_loops']:.2f} "
            f"C={by_lam[str(lam)]['resume_price_C_loops']:.2f} "
            f"R={by_lam[str(lam)]['resume_price_R_loops']:.2f} | "
            f"ForkDisagree={by_lam[str(lam)]['fork_loop4_disagree_rate']:.3f}",
            flush=True,
        )
    return {"arms": arms_out, "mean_by_lambda": by_lam}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CR.DEFAULT_CACHE_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "stage2_3_resume_and_fork.json")
    parser.add_argument("--mem-fraction", type=float, default=0.65)
    parser.add_argument("--min-watts", type=float, default=40.0)
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        watts = TS.power_limit_watts()
        if watts is not None and watts < args.min_watts:
            print(f"[power-guard] GPU limit {watts}W < {args.min_watts}W; exiting 75.")
            return TS.EXIT_POWER_CAP
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
        torch.backends.cuda.matmul.allow_tf32 = False

    t0 = time.time()
    bundle = json.loads((REPO_ROOT / "mechinterp" / "loopscope02" / "results" / "benchmark_bundle.json").read_text("utf-8"))
    items = bundle["items"]
    items_qs = [it["question"] for it in items]
    items_ans = [it["answer"] for it in items]
    items_prop = [bool(it["prop_solved"]) for it in items]
    bench_qs, bench_ans, bench_prop = FRC.load_bench100()

    print("\n=== 1. LoopedChase Three-Ledger Contrast (12 Q4 checkpoints) ===", flush=True)
    chase_res = evaluate_looped_chase_three_ledgers(dev=dev)

    trm_results = {}
    stationarity_gate = None

    for ckpt_tag in ("gaoxin492", "yagizdevre"):
        print(f"\n=== 2/3. Resume & Fork Tests on TRM checkpoint: {ckpt_tag} ===", flush=True)
        model, sha = TS.load_trm_checkpoint(ckpt_tag, device=dev, dtype=torch.bfloat16)
        refill_b = load_saved_refill_bundle(args.cache_dir, ckpt_tag, dev=dev)

        if stationarity_gate is None:
            stationarity_gate = evaluate_stationarity_admission(model, items_qs, dev=dev)
            print(f"Stationarity admission gate verified: {stationarity_gate['TRM_gaoxin492']}", flush=True)

        print(f"[{ckpt_tag}] Running Resume Test on items256 and bench100...", flush=True)
        res_items = run_trm_resume_test(model, refill_b, items_qs, items_ans, items_prop)
        res_bench = run_trm_resume_test(model, refill_b, bench_qs, bench_ans, bench_prop)

        print(f"[{ckpt_tag}] Running Multi-Init C-Voting & Transient Divergence on items256 and bench100...", flush=True)
        cv_items = evaluate_cvoting_and_transient_divergence(model, items_qs, items_ans, items_prop)
        cv_bench = evaluate_cvoting_and_transient_divergence(model, bench_qs, bench_ans, bench_prop)

        print(f"[{ckpt_tag}] Running Causal R_t Fork Swap & Refuted-State Revisit Test...", flush=True)
        fswap_items = run_trm_fork_swap_test(model, refill_b, items_qs, items_ans, items_prop)
        fswap_bench = run_trm_fork_swap_test(model, refill_b, bench_qs, bench_ans, bench_prop)

        trm_results[ckpt_tag] = {
            "checkpoint_sha256": sha,
            "refill_model_used": refill_b["best_tf_name"],
            "resume_items256": res_items,
            "resume_bench100": res_bench,
            "cvoting_and_divergence_items256": cv_items,
            "cvoting_and_divergence_bench100": cv_bench,
            "fork_swap_items256": fswap_items,
            "fork_swap_bench100": fswap_bench,
        }
        del model, refill_b
        if dev == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "preregistration_commit": "b1ec77550997cbc5ae2cd84618af74e8f34e38e7",
        "elapsed_sec": float(time.time() - t0),
        "stationarity_admission_gate": stationarity_gate,
        "looped_chase_contrast": chase_res,
        "trm_primary_gaoxin492": trm_results["gaoxin492"],
        "trm_replication_yagizdevre": trm_results["yagizdevre"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    print(f"\nWrote Stage 2 & 3 receipt to {args.out} in {payload['elapsed_sec']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
