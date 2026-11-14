#!/usr/bin/env python3
"""LOOPSCOPE-03 Stage 4 (Post-Review Diagnostic Controls):
Unconditional Path Independence, Enforced Path Independence, Refill Data-Size Scaling Curve,
Closed-Loop Covariate-Shift Audit, and Uncapped Resume Never-Lock / Median-Delay Estimators.

Addresses four methodological questions raised on v013:
  1. Unconditional Per-Board Path Independence & Enforced Path Independence (K = 8 forks, s = 1..64):
     - Measures unconditional steady-state exact-board agreement PI_board(b, s), empty-cell agreement
       PI_cell(b, s) = 1 - d_Hamming, and hidden-state pairwise cosine PI_hidden(b, s) across ALL
       C(8, 2) = 28 fork pairs (without conditioning on solve, which is tautological on unique-solution
       Sudoku).
     - Tests directly whether Conditional Refill H_t <- H_hat(x, V_t) enforces path independence across
       the K = 8 perturbed starts, and correlates clean per-board path independence against clean solve,
       s = 16 -> 64 retention/degradation (Anil et al., 2022), and per-board Conditional Refill gain.
  2. Training-Set Size Scaling Curve (N_train in {100, 300, 900, 2700, 8100} at fixed C5 2L-512):
     - Streams disjoint puzzles from `sapientinc/sudoku-extreme` `train.csv` (verified zero puzzle and
       digit-relabeled isomorphism overlap with `val_100`, `items256`, and `bench100`).
     - Evaluates held-out `val_100` R^2, closed-loop `items256`/`bench100` solve (s = 16, 32, 64), and
       covariate-shift-free single-transplant `Resume` across two orders of magnitude in training boards.
  3. Closed-Loop Rollout Drift vs. Periodic Refill Intervals (m in {1, 2, 4, 8}):
     - Separates every-step closed-loop covariate shift from periodic hidden-state projection and
       tracks per-step rollout conflict/entropy divergence from clean trajectories.
  4. Unselected & Uncapped Resume Reporting (t in {1, 2, 4, 8, 12}, H = 32):
     - Reports across fixed unselected denominators (`WB-Prop` n=73, `WB-Stall` n=183, clean-locking
       `WB-Stall`, and unresolved-at-t `WB-Stall`):
         (a) exact fraction that never lock within H = 32 (plus paired McNemar p), and
         (b) median, IQR, and mean delay strictly among boards that lock within H = 32 under BOTH
             compared arms (zero 33-cap conflation of delay with non-termination).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mechinterp.loopscope02.scripts import fiber_regime_controls as FRC  # noqa: E402
from mechinterp.loopscope02.scripts import score_loopscope02 as SC       # noqa: E402
from mechinterp.loopscope02.scripts import sudoku_prop as SP             # noqa: E402
from mechinterp.loopscope02.scripts import trm_sudoku as TS              # noqa: E402
from mechinterp.loopscope03.scripts import conditional_refill as CR      # noqa: E402
from mechinterp.loopscope03.scripts import resume_and_fork as RF         # noqa: E402

RESULTS_DIR = REPO_ROOT / "mechinterp" / "loopscope03" / "results"
DEFAULT_CACHE_DIR = Path(
    r"C:\Users\patri\.gemini\antigravity\brain\683081f5-5cee-4346-8423-d0c90341cef0\scratch\loopscope03_models"
)
SUDOKU_EXTREME_FULL_TRAIN_URL = (
    "https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/train.csv"
)


def digit_relabel_canonical(q: str) -> str:
    """Canonicalize 81-char Sudoku clue string up to 9! digit relabelings."""
    mapping: dict[str, str] = {}
    nxt = 1
    out = []
    for ch in q:
        if ch in ".0":
            out.append(".")
        else:
            if ch not in mapping:
                mapping[ch] = str(nxt)
                nxt += 1
            out.append(mapping[ch])
    return "".join(out)


def load_disjoint_train_pool(
    target_total: int,
    base_train_900: list[str],
    forbidden_sets: dict[str, list[str]],
    cache_dir: Path,
) -> tuple[list[str], dict]:
    """Build nested training puzzle pool of size `target_total` starting with `base_train_900` (900)
    followed by disjoint puzzles streamed from `sapientinc/sudoku-extreme/train.csv`.
    Verifies zero exact or digit-relabeled overlap with `val_100`, `items256`, and `bench100`.
    """
    assert len(base_train_900) == 900
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_cache = cache_dir / "sudoku_extreme_full_train_prefix2mb.csv"

    if not csv_cache.exists():
        req = urllib.request.Request(
            SUDOKU_EXTREME_FULL_TRAIN_URL,
            headers={"Range": "bytes=0-2400000"},
        )
        raw_bytes = urllib.request.urlopen(req, timeout=60).read()
        # Drop potentially partial final line
        text = raw_bytes.decode("utf-8", errors="ignore")
        complete_lines = text.splitlines()[:-1]
        csv_cache.write_bytes(("\n".join(complete_lines) + "\n").encode("utf-8"))

    prefix_sha256 = hashlib.sha256(csv_cache.read_bytes()).hexdigest()

    forbidden_exact: set[str] = set()
    forbidden_relab: set[str] = set()
    for qs in forbidden_sets.values():
        for q in qs:
            forbidden_exact.add(q)
            forbidden_relab.add(digit_relabel_canonical(q))

    seen_exact: set[str] = set(base_train_900)
    # Verify base_train_900 itself has zero overlap with forbidden test/val splits
    for q in base_train_900:
        assert q not in forbidden_exact, "base_train_900 overlaps forbidden split!"

    pool = list(base_train_900)
    skipped_overlap = 0
    streamed_rows = 0
    with csv_cache.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            streamed_rows += 1
            q = row.get("question") or row.get("puzzle")
            a = row.get("answer") or row.get("solution")
            if not q or not a or len(q) != 81 or len(a) != 81:
                continue
            q_norm = "".join("." if ch in ".0" else ch for ch in q)
            if q_norm in seen_exact:
                continue
            if q_norm in forbidden_exact or digit_relabel_canonical(q_norm) in forbidden_relab:
                skipped_overlap += 1
                continue
            seen_exact.add(q_norm)
            pool.append(q_norm)
            if len(pool) >= target_total:
                break

    if len(pool) < target_total:
        raise RuntimeError(f"Only collected {len(pool)} disjoint train puzzles, needed {target_total}")

    # Final programmatic overlap verification
    pool_set = set(pool)
    overlaps = {name: len(pool_set.intersection(set(qs))) for name, qs in forbidden_sets.items()}
    assert all(v == 0 for v in overlaps.values()), f"Overlap detected: {overlaps}"

    return pool, {
        "source_url": SUDOKU_EXTREME_FULL_TRAIN_URL,
        "cached_prefix_sha256": prefix_sha256,
        "streamed_rows_scanned": streamed_rows,
        "skipped_forbidden_or_relabeled": skipped_overlap,
        "total_pool_n": len(pool),
        "exact_overlaps_with_splits": overlaps,
    }


@torch.no_grad()
def collect_trm_states_memmap(
    model: TS.PureSudokuTRM,
    qs: list[str],
    cache_dir: Path,
    ckpt_tag: str,
    n_steps: int = 16,
    bsz: int = 256,
) -> dict:
    """Collect clean 16-step TRM trajectories into disk-backed uint16 (bf16 bitpattern) memmaps
    so 8,100 x 16 = 129,600 states use < 1 GB of Python RAM.
    """
    n = len(qs)
    meta_path = cache_dir / f"{ckpt_tag}_traj_{n}_meta.pt"
    zh_path = cache_dir / f"{ckpt_tag}_traj_{n}_zh.bf16.mmap"
    zl_path = cache_dir / f"{ckpt_tag}_traj_{n}_zl.bf16.mmap"
    expected_bytes = n * n_steps * 97 * 512 * 2

    if meta_path.exists() and zh_path.exists() and zl_path.exists():
        if zh_path.stat().st_size == expected_bytes and zl_path.stat().st_size == expected_bytes:
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            if meta.get("qs_sha256") == hashlib.sha256("\n".join(qs).encode("utf-8")).hexdigest():
                print(f"  [{ckpt_tag}] Reusing cached {n}-board trajectory memmaps...", flush=True)
                zh_mm = np.memmap(zh_path, dtype=np.uint16, mode="r", shape=(n, n_steps, 97, 512))
                zl_mm = np.memmap(zl_path, dtype=np.uint16, mode="r", shape=(n, n_steps, 97, 512))
                return {
                    "inp_tokens": meta["inp_tokens"],
                    "cand_masks": meta["cand_masks"],
                    "zh_mm": zh_mm,
                    "zl_mm": zl_mm,
                }

    print(f"  [{ckpt_tag}] Collecting clean 16-step trajectories for {n} train boards...", flush=True)
    t0 = time.time()
    dev = next(model.parameters()).device
    inp_cpu = TS.encode_questions(qs, "cpu")
    cand_cpu = CR.build_cand_masks(qs)

    zh_mm = np.memmap(zh_path, dtype=np.uint16, mode="w+", shape=(n, n_steps, 97, 512))
    zl_mm = np.memmap(zl_path, dtype=np.uint16, mode="w+", shape=(n, n_steps, 97, 512))

    for b0 in range(0, n, bsz):
        b1 = min(n, b0 + bsz)
        inp_b = inp_cpu[b0:b1].to(dev)
        xb = model.encode_inputs(inp_b)
        zh, zl = model.init_states(b1 - b0)
        zh_buf = np.empty((b1 - b0, n_steps, 97, 512), dtype=np.uint16)
        zl_buf = np.empty((b1 - b0, n_steps, 97, 512), dtype=np.uint16)
        for s in range(n_steps):
            zh, zl = model.act_step(zh, zl, xb)
            zh_buf[:, s] = zh.to(torch.bfloat16).cpu().view(torch.uint16).numpy()
            zl_buf[:, s] = zl.to(torch.bfloat16).cpu().view(torch.uint16).numpy()
        zh_mm[b0:b1] = zh_buf
        zl_mm[b0:b1] = zl_buf
        if b1 % 1024 == 0 or b1 == n:
            print(f"    collected {b1}/{n} boards ({time.time() - t0:.1f}s)", flush=True)

    zh_mm.flush()
    zl_mm.flush()
    torch.save(
        {
            "qs_sha256": hashlib.sha256("\n".join(qs).encode("utf-8")).hexdigest(),
            "inp_tokens": inp_cpu,
            "cand_masks": cand_cpu,
        },
        meta_path,
    )
    zh_ro = np.memmap(zh_path, dtype=np.uint16, mode="r", shape=(n, n_steps, 97, 512))
    zl_ro = np.memmap(zl_path, dtype=np.uint16, mode="r", shape=(n, n_steps, 97, 512))
    return {
        "inp_tokens": inp_cpu,
        "cand_masks": cand_cpu,
        "zh_mm": zh_ro,
        "zl_mm": zl_ro,
    }


@torch.no_grad()
def compute_mean_baseline_from_memmap(
    mm_data: dict,
    n_use: int,
    v_head: torch.Tensor,
    p_perp: torch.Tensor,
    dev: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Stream over `0:n_use` boards to compute exact sphere-calibrated global mean baseline."""
    n_s = mm_data["zh_mm"].shape[1]
    v_head_d = v_head.to(dev)
    p_perp_d = p_perp.to(dev)
    sum_orth = torch.zeros((81, 512), dtype=torch.float64, device=dev)
    sum_pref = torch.zeros((16, 512), dtype=torch.float64, device=dev)
    sum_zl = torch.zeros((97, 512), dtype=torch.float64, device=dev)
    count = 0
    chunk_b = 300
    for b0 in range(0, n_use, chunk_b):
        b1 = min(n_use, b0 + chunk_b)
        zh_u16 = np.array(mm_data["zh_mm"][b0:b1], copy=True)
        zl_u16 = np.array(mm_data["zl_mm"][b0:b1], copy=True)
        zh_t = torch.from_numpy(zh_u16).view(torch.bfloat16).to(dev).float()
        zl_t = torch.from_numpy(zl_u16).view(torch.bfloat16).to(dev).float()
        zb = zh_t[:, :, 16:, :]
        vc = zb @ v_head_d
        vp = vc @ v_head_d.T
        orth = zb - vp
        sum_orth += orth.double().sum(dim=(0, 1))
        sum_pref += zh_t[:, :, :16, :].double().sum(dim=(0, 1))
        sum_zl += zl_t.double().sum(dim=(0, 1))
        count += (b1 - b0) * n_s

    mu_orth = ((sum_orth / count).float().unsqueeze(0) @ p_perp_d).cpu().contiguous()
    mu_pref = CR.rms_calibrate_vec((sum_pref / count).float().unsqueeze(0)).cpu().contiguous()
    mu_zl = CR.rms_calibrate_vec((sum_zl / count).float().unsqueeze(0)).cpu().contiguous()
    return {"mu_orth": mu_orth, "mu_pref": mu_pref, "mu_zl": mu_zl}


def train_c5_from_memmap(
    mm_data: dict,
    n_use: int,
    val_data: dict[str, torch.Tensor],
    v_head: torch.Tensor,
    w_digit: torch.Tensor,
    p_perp: torch.Tensor,
    epochs: int,
    batch_size: int = 512,
    lr: float = 1.25e-3,
    dev: str = "cuda",
) -> tuple[CR.RefillTransformer, dict]:
    """Train C5_tf_2L_512 on `0:n_use` boards using chunked GPU staging so VRAM stays < 3.5 GB."""
    torch.manual_seed(20261114)
    net = CR.RefillTransformer(d_model=512, n_layers=2, n_heads=8, use_step_clock=False).to(dev)
    v_head_d = v_head.to(dev)
    w_dig_d = w_digit.to(dev)
    p_perp_d = p_perp.to(dev)

    n_s = mm_data["zh_mm"].shape[1]
    m_tot = n_use * n_s

    n_va = val_data["zh"].shape[0]
    m_va = n_va * n_s
    va_inp = val_data["inp_tokens"].unsqueeze(1).expand(-1, n_s, -1).reshape(m_va, 81).to(dev)
    va_cand = val_data["cand_masks"].unsqueeze(1).expand(-1, n_s, -1, -1).reshape(m_va, 81, 9).to(dev)
    va_zh = val_data["zh"].reshape(m_va, 97, 512).to(dev)
    va_zl = val_data["zl"].reshape(m_va, 97, 512).to(dev)
    with torch.no_grad():
        va_vc = va_zh[:, 16:, :].float() @ v_head_d

    steps_per_epoch = (m_tot + batch_size - 1) // batch_size
    total_steps = epochs * steps_per_epoch
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=3e-5)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    rng = np.random.default_rng(20261114 + n_use)
    t0 = time.time()

    # Stage up to 900 boards (14,400 states) onto GPU at a time after permuting board indices
    board_chunk = 900
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32 if dev == "cuda" else False
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    try:
        for ep in range(epochs):
            net.train()
            board_perm = rng.permutation(n_use)
            for c0 in range(0, n_use, board_chunk):
                b_idx = board_perm[c0 : min(n_use, c0 + board_chunk)]
                nb_c = len(b_idx)
                mc = nb_c * n_s
                # Read selected boards from memmap into contiguous CPU array then copy to GPU
                zh_u16 = np.array(mm_data["zh_mm"][b_idx], copy=True)
                zl_u16 = np.array(mm_data["zl_mm"][b_idx], copy=True)
                b_idx_t = torch.from_numpy(b_idx)

                c_inp = mm_data["inp_tokens"][b_idx_t].unsqueeze(1).expand(-1, n_s, -1).reshape(mc, 81).to(dev)
                c_cand = (
                    mm_data["cand_masks"][b_idx_t].unsqueeze(1).expand(-1, n_s, -1, -1).reshape(mc, 81, 9).to(dev)
                )
                c_zh = torch.from_numpy(zh_u16).view(torch.bfloat16).reshape(mc, 97, 512).to(dev)
                c_zl = torch.from_numpy(zl_u16).view(torch.bfloat16).reshape(mc, 97, 512).to(dev)
                with torch.no_grad():
                    c_vc = torch.empty((mc, 81, v_head_d.shape[1]), dtype=torch.float32, device=dev)
                    for i0 in range(0, mc, 1024):
                        i1 = min(mc, i0 + 1024)
                        c_vc[i0:i1] = c_zh[i0:i1, 16:, :].float() @ v_head_d

                sub_perm = torch.from_numpy(rng.permutation(mc)).to(dev)
                for b0 in range(0, mc, batch_size):
                    idx = sub_perm[b0 : b0 + batch_size]
                    inp_b = c_inp[idx]
                    cand_b = c_cand[idx]
                    zh_b = c_zh[idx].float()
                    zl_b = c_zl[idx].float()
                    vc = c_vc[idx]
                    vp = vc @ v_head_d.T
                    tgt_orth = zh_b[:, 16:, :] - vp
                    tgt_pref = zh_b[:, :16, :]

                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                        po, pp, pz = net(
                            inp_b,
                            cand_b,
                            vc,
                            vp,
                            w_dig_d,
                            p_perp_d,
                            calibrate_sphere=True,
                            v_head=v_head_d,
                        )
                    loss = (
                        (po - tgt_orth).pow(2).mean()
                        + 0.25 * (pp - tgt_pref).pow(2).mean()
                        + (pz - zl_b).pow(2).mean()
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                    sched.step()

                del c_inp, c_cand, c_zh, c_zl, c_vc

            eval_every = 2 if epochs <= 24 else 3
            if (ep + 1) % eval_every == 0 or ep == epochs - 1:
                net.eval()
                val_sum = 0.0
                val_cnt = 0
                with torch.no_grad():
                    for b0 in range(0, m_va, 512):
                        b1 = min(m_va, b0 + 512)
                        inp_b = va_inp[b0:b1]
                        cand_b = va_cand[b0:b1]
                        zh_b = va_zh[b0:b1].float()
                        zl_b = va_zl[b0:b1].float()
                        vc = va_vc[b0:b1]
                        vp = vc @ v_head_d.T
                        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                            po, pp, pz = net(
                                inp_b,
                                cand_b,
                                vc,
                                vp,
                                w_dig_d,
                                p_perp_d,
                                calibrate_sphere=True,
                                v_head=v_head_d,
                            )
                        v_loss = (
                            (po - (zh_b[:, 16:, :] - vp)).pow(2).mean()
                            + 0.25 * (pp - zh_b[:, :16, :]).pow(2).mean()
                            + (pz - zl_b).pow(2).mean()
                        )
                        val_sum += float(v_loss.item()) * (b1 - b0)
                        val_cnt += (b1 - b0)
                val_mse = val_sum / max(val_cnt, 1)
                if val_mse < best_val_loss:
                    best_val_loss = val_mse
                    best_epoch = ep + 1
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    finally:
        if dev == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    assert best_state is not None
    net.load_state_dict(best_state)
    net.eval()
    return net, {
        "n_train_boards": n_use,
        "n_train_states": m_tot,
        "epochs": epochs,
        "optimizer_steps": total_steps,
        "n_params": net.num_parameters(),
        "best_epoch": best_epoch,
        "best_val_objective": best_val_loss,
        "train_sec": float(time.time() - t0),
    }


# =====================================================================
# Part 1: Unconditional Per-Board Path Independence & Enforced PI
# =====================================================================

@torch.no_grad()
def run_k_fork_rollouts(
    model: TS.PureSudokuTRM,
    qs: list[str],
    ans: list[str],
    k_forks: int,
    sigma: float,
    max_steps: int,
    refill_net: CR.RefillTransformer | None,
    v_head: torch.Tensor,
    w_digit: torch.Tensor,
    p_perp: torch.Tensor,
    bsz: int = 128,
) -> dict:
    """Run K perturbed initializations per board for `max_steps` outer steps under either
    Clean (`refill_net=None`) or Enforced Path Independence (`VC_C5` every step).
    Returns predictions [K, S, N, 81], confidence [K, S_sel, N], and normalized hidden snapshots at s=16, 64.
    """
    dev = next(model.parameters()).device
    dt = model.embed_tokens.weight.dtype
    n = len(qs)
    inp_all = TS.encode_questions(qs, dev)
    cand_all = CR.build_cand_masks(qs).to(dev)
    x_all = model.encode_inputs(inp_all)

    v_head_d = v_head.to(dev)
    w_dig_d = w_digit.to(dev)
    p_perp_d = p_perp.to(dev)

    preds_ks = np.empty((k_forks, max_steps, n, 81), dtype=np.int8)
    conf_s16 = np.empty((k_forks, n), dtype=np.float32)
    conf_s64 = np.empty((k_forks, n), dtype=np.float32)

    # Store board-pooled normalized hidden vectors or per-token downsampled vectors?
    # Better: compute exact pairwise tokenwise cosine similarity across forks at s=16 and s=64!
    # Storing [K, N, 97, 512] in bf16 for 2 snapshots is only 8 * 256 * 97 * 512 * 2 = 203 MB!
    snap_steps = (16, max_steps)
    hid_s16 = torch.empty((k_forks, n, 194, 512), dtype=torch.bfloat16)
    hid_s64 = torch.empty((k_forks, n, 194, 512), dtype=torch.bfloat16)

    for m_idx in range(k_forks):
        for b0 in range(0, n, bsz):
            b1 = min(n, b0 + bsz)
            zh, zl = RF.make_forked_initial_states(model, b1 - b0, m_idx, sigma, dev)
            xb = x_all[b0:b1]
            inp_b = inp_all[b0:b1]
            cand_b = cand_all[b0:b1]

            for s in range(max_steps):
                zh, zl = model.act_step(zh, zl, xb)
                logits = model.digit_logits(zh)
                probs = torch.softmax(logits.float(), dim=-1)
                preds_ks[m_idx, s, b0:b1] = probs.argmax(dim=-1).cpu().numpy().astype(np.int8)

                if s + 1 == 16:
                    conf_s16[m_idx, b0:b1] = probs.amax(dim=-1).mean(dim=-1).cpu().numpy()
                    hid_s16[m_idx, b0:b1] = torch.cat([zh, zl], dim=1).to(torch.bfloat16).cpu()
                if s + 1 == max_steps:
                    conf_s64[m_idx, b0:b1] = probs.amax(dim=-1).mean(dim=-1).cpu().numpy()
                    hid_s64[m_idx, b0:b1] = torch.cat([zh, zl], dim=1).to(torch.bfloat16).cpu()

                if refill_net is not None and s < max_steps - 1:
                    zb = zh[:, 16:, :].float()
                    vc = zb @ v_head_d
                    vp = vc @ v_head_d.T
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                        ro, rp, rz = refill_net(
                            inp_b, cand_b, vc, vp, w_dig_d, p_perp_d, calibrate_sphere=True, v_head=v_head_d
                        )
                    zh = torch.cat([rp.float(), vp + ro.float()], dim=1).to(dt)
                    zl = rz.to(dt)

    return {
        "preds_ks": preds_ks,
        "conf_s16": conf_s16,
        "conf_s64": conf_s64,
        "hid_s16": hid_s16,
        "hid_s64": hid_s64,
    }


def score_unconditional_path_independence(
    rollout: dict,
    qs: list[str],
    ans: list[str],
    prop_flags: list[bool],
    dev: str = "cuda",
) -> dict:
    """Compute unconditional exact-board PI, empty-cell PI, hidden-state cosine PI, outcome-split
    rates, and C-voting across ALL C(K, 2) = 28 fork pairs without conditioning on solve.
    """
    preds_ks = rollout["preds_ks"]  # [K, S, N, 81]
    k_forks, max_steps, n, _ = preds_ks.shape
    ans_np = np.array([[int(ch) - 1 for ch in a] for a in ans], dtype=np.int64)
    empty_mask = np.array([[ch == "." for ch in q] for q in qs], dtype=bool)
    given_np = np.array([[-1 if ch == "." else int(ch) - 1 for ch in q] for q in qs], dtype=np.int64)
    prop_arr = np.array(prop_flags, dtype=bool)
    stall_arr = ~prop_arr

    merged_ks = np.where(empty_mask[None, None, :, :], preds_ks, given_np[None, None, :, :])
    exact_ks = (merged_ks == ans_np[None, None, :, :]).all(axis=-1)  # [K, S, N]

    n_pairs = k_forks * (k_forks - 1) // 2
    pi_cell_curve = np.zeros((max_steps, n), dtype=np.float64)
    pi_board_s16 = np.zeros(n, dtype=np.float64)
    pi_board_s64 = np.zeros(n, dtype=np.float64)
    pair_both_solved_s16 = np.zeros(n, dtype=np.float64)
    pair_same_unsolved_s16 = np.zeros(n, dtype=np.float64)
    pair_both_solved_s64 = np.zeros(n, dtype=np.float64)
    pair_same_unsolved_s64 = np.zeros(n, dtype=np.float64)

    for b in range(n):
        em = empty_mask[b]
        for m1 in range(k_forks):
            for m2 in range(m1 + 1, k_forks):
                agree_cells = (preds_ks[m1, :, b, :][:, em] == preds_ks[m2, :, b, :][:, em]).mean(axis=-1)
                pi_cell_curve[:, b] += agree_cells / n_pairs

                eq16 = bool(np.array_equal(merged_ks[m1, 15, b], merged_ks[m2, 15, b]))
                if eq16:
                    pi_board_s16[b] += 1.0 / n_pairs
                    if exact_ks[m1, 15, b]:
                        pair_both_solved_s16[b] += 1.0 / n_pairs
                    else:
                        pair_same_unsolved_s16[b] += 1.0 / n_pairs

                eq64 = bool(np.array_equal(merged_ks[m1, max_steps - 1, b], merged_ks[m2, max_steps - 1, b]))
                if eq64:
                    pi_board_s64[b] += 1.0 / n_pairs
                    if exact_ks[m1, max_steps - 1, b]:
                        pair_both_solved_s64[b] += 1.0 / n_pairs
                    else:
                        pair_same_unsolved_s64[b] += 1.0 / n_pairs

    # Compute hidden-state pairwise token cosine similarity at s=16 and s=64 on GPU
    def pairwise_hidden_cos(hid_bf16: torch.Tensor) -> np.ndarray:
        # hid_bf16: [K, N, 194, 512]
        out = np.zeros(n, dtype=np.float64)
        with torch.no_grad():
            for b0 in range(0, n, 64):
                b1 = min(n, b0 + 64)
                h = F.normalize(hid_bf16[:, b0:b1].to(dev).float(), dim=-1)  # [K, mb, 194, 512]
                acc = torch.zeros(b1 - b0, dtype=torch.float64, device=dev)
                for m1 in range(k_forks):
                    for m2 in range(m1 + 1, k_forks):
                        acc += (h[m1] * h[m2]).sum(dim=-1).mean(dim=-1).double() / n_pairs
                out[b0:b1] = acc.cpu().numpy()
        return out

    pi_hid_s16 = pairwise_hidden_cos(rollout["hid_s16"])
    pi_hid_s64 = pairwise_hidden_cos(rollout["hid_s64"])

    # Fork solve fractions and C-voting at s=16 and s=64
    fork_solve_frac_s16 = exact_ks[:, 15, :].mean(axis=0)          # [N]
    fork_solve_frac_s64 = exact_ks[:, max_steps - 1, :].mean(axis=0)
    outcome_split_s16 = (fork_solve_frac_s16 > 0.0) & (fork_solve_frac_s16 < 1.0)
    outcome_split_s64 = (fork_solve_frac_s64 > 0.0) & (fork_solve_frac_s64 < 1.0)

    best_m16 = rollout["conf_s16"].argmax(axis=0)
    best_m64 = rollout["conf_s64"].argmax(axis=0)
    cvote_s16 = exact_ks[best_m16, 15, np.arange(n)]
    cvote_s64 = exact_ks[best_m64, max_steps - 1, np.arange(n)]
    oracle_s16 = exact_ks[:, 15, :].any(axis=0)
    oracle_s64 = exact_ks[:, max_steps - 1, :].any(axis=0)
    canon_s16 = exact_ks[0, 15, :]
    canon_s64 = exact_ks[0, max_steps - 1, :]

    def summarize_mask(mask: np.ndarray) -> dict:
        return {
            "n": int(mask.sum()),
            "uncond_pi_board_s16": float(pi_board_s16[mask].mean()),
            "uncond_pi_board_s64": float(pi_board_s64[mask].mean()),
            "pair_both_solved_s16": float(pair_both_solved_s16[mask].mean()),
            "pair_same_unsolved_trap_s16": float(pair_same_unsolved_s16[mask].mean()),
            "pair_both_solved_s64": float(pair_both_solved_s64[mask].mean()),
            "pair_same_unsolved_trap_s64": float(pair_same_unsolved_s64[mask].mean()),
            "uncond_pi_cell_s16": float(pi_cell_curve[15, mask].mean()),
            "uncond_pi_cell_s64": float(pi_cell_curve[max_steps - 1, mask].mean()),
            "uncond_pi_hidden_cos_s16": float(pi_hid_s16[mask].mean()),
            "uncond_pi_hidden_cos_s64": float(pi_hid_s64[mask].mean()),
            "outcome_split_board_rate_s16": float(outcome_split_s16[mask].mean()),
            "outcome_split_board_rate_s64": float(outcome_split_s64[mask].mean()),
            "canonical_solve_s16": float(canon_s16[mask].mean()),
            "canonical_solve_s64": float(canon_s64[mask].mean()),
            "mean_fork_solve_s16": float(fork_solve_frac_s16[mask].mean()),
            "mean_fork_solve_s64": float(fork_solve_frac_s64[mask].mean()),
            "cvoting_solve_s16": float(cvote_s16[mask].mean()),
            "cvoting_solve_s64": float(cvote_s64[mask].mean()),
            "oracle_union_s16": float(oracle_s16[mask].mean()),
            "oracle_union_s64": float(oracle_s64[mask].mean()),
            "pi_cell_curve_s1_64": [float(x) for x in pi_cell_curve[:, mask].mean(axis=1)],
        }

    return {
        "summary": {
            "all": summarize_mask(np.ones(n, dtype=bool)),
            "wb_prop": summarize_mask(prop_arr),
            "wb_stall": summarize_mask(stall_arr),
        },
        "per_board": {
            "pi_cell_s16": pi_cell_curve[15],
            "pi_cell_s64": pi_cell_curve[max_steps - 1],
            "pi_board_s16": pi_board_s16,
            "pi_board_s64": pi_board_s64,
            "pi_hid_s16": pi_hid_s16,
            "pi_hid_s64": pi_hid_s64,
            "canon_s16": canon_s16.astype(np.float64),
            "canon_s64": canon_s64.astype(np.float64),
            "mean_fork_s16": fork_solve_frac_s16,
            "mean_fork_s64": fork_solve_frac_s64,
        },
    }


def analyze_per_board_pi_vs_solve_and_refill(
    clean_pi: dict,
    vc_pi: dict,
    prop_flags: list[bool],
) -> dict:
    """Test Anil et al. (2022) + Reviewer's 'better story in the data' per board:
      - Does clean per-board path independence predict solve and s=16->64 retention?
      - Does Conditional Refill (`VC_C5`) enforce higher path independence across the 8 forks?
      - Does Conditional Refill help low-PI (path-dependent) boards while hurting/leaving neutral
        high-PI boards?
    """
    cb = clean_pi["per_board"]
    vb = vc_pi["per_board"]
    prop_arr = np.array(prop_flags, dtype=bool)
    stall_arr = ~prop_arr

    pi16 = cb["pi_cell_s16"]
    pi64 = cb["pi_cell_s64"]
    clean_fork16 = cb["mean_fork_s16"]
    clean_fork64 = cb["mean_fork_s64"]
    vc_fork16 = vb["mean_fork_s16"]
    vc_fork64 = vb["mean_fork_s64"]
    clean_can16 = cb["canon_s16"]
    clean_can64 = cb["canon_s64"]
    vc_can16 = vb["canon_s16"]
    vc_can64 = vb["canon_s64"]

    delta_ood_clean = clean_fork64 - clean_fork16
    delta_vc_fork64 = vc_fork64 - clean_fork64
    delta_vc_can64 = vc_can64 - clean_can64

    # Quartile bins of clean PI_cell(b, s=16) on WB-Stall (n=183) AND across all items256 (n=256)
    def build_bins(mask: np.ndarray, score_vec: np.ndarray, n_bins: int = 4) -> list[dict]:
        idx = np.where(mask)[0]
        order = idx[np.argsort(score_vec[idx], kind="mergesort")]
        splits = np.array_split(order, n_bins)
        bins_out = []
        for bi_i, sub in enumerate(splits):
            bins_out.append({
                "bin": bi_i + 1,
                "n": int(len(sub)),
                "clean_pi_cell_mean": float(score_vec[sub].mean()),
                "clean_pi_cell_min": float(score_vec[sub].min()),
                "clean_pi_cell_max": float(score_vec[sub].max()),
                "vc_pi_cell_s64_mean": float(vb["pi_cell_s64"][sub].mean()),
                "clean_canon_s16": float(clean_can16[sub].mean()),
                "clean_canon_s64": float(clean_can64[sub].mean()),
                "vc_canon_s16": float(vc_can16[sub].mean()),
                "vc_canon_s64": float(vc_can64[sub].mean()),
                "clean_mean_fork_s16": float(clean_fork16[sub].mean()),
                "clean_mean_fork_s64": float(clean_fork64[sub].mean()),
                "vc_mean_fork_s16": float(vc_fork16[sub].mean()),
                "vc_mean_fork_s64": float(vc_fork64[sub].mean()),
                "delta_vc_minus_clean_canon_s64": float(delta_vc_can64[sub].mean()),
                "delta_vc_minus_clean_fork_s64": float(delta_vc_fork64[sub].mean()),
            })
        return bins_out

    return {
        "enforced_pi_shift_wb_stall": {
            "pi_cell_s16_clean_to_vc": [
                float(cb["pi_cell_s16"][stall_arr].mean()),
                float(vb["pi_cell_s16"][stall_arr].mean()),
            ],
            "pi_cell_s64_clean_to_vc": [
                float(cb["pi_cell_s64"][stall_arr].mean()),
                float(vb["pi_cell_s64"][stall_arr].mean()),
            ],
            "pi_board_s64_clean_to_vc": [
                float(cb["pi_board_s64"][stall_arr].mean()),
                float(vb["pi_board_s64"][stall_arr].mean()),
            ],
            "pi_hidden_cos_s64_clean_to_vc": [
                float(cb["pi_hid_s64"][stall_arr].mean()),
                float(vb["pi_hid_s64"][stall_arr].mean()),
            ],
            "paired_signflip_pi_cell_s64": RF.paired_signflip_p(
                vb["pi_cell_s64"][stall_arr] - cb["pi_cell_s64"][stall_arr]
            ),
        },
        "correlations_all_256": {
            "pi16_vs_clean_fork_s16": RF.spearman_corr_and_perm_p(pi16, clean_fork16),
            "pi16_vs_clean_fork_s64": RF.spearman_corr_and_perm_p(pi16, clean_fork64),
            "pi64_vs_clean_canon_s64": RF.spearman_corr_and_perm_p(pi64, clean_can64),
            "pi16_vs_ood_change_s16_to_s64": RF.spearman_corr_and_perm_p(pi16, delta_ood_clean),
            "pi64_vs_vc_gain_fork_s64": RF.spearman_corr_and_perm_p(pi64, delta_vc_fork64),
            "pi64_vs_vc_gain_canon_s64": RF.spearman_corr_and_perm_p(pi64, delta_vc_can64),
        },
        "correlations_wb_stall_183": {
            "pi16_vs_clean_fork_s16": RF.spearman_corr_and_perm_p(pi16[stall_arr], clean_fork16[stall_arr]),
            "pi16_vs_clean_fork_s64": RF.spearman_corr_and_perm_p(pi16[stall_arr], clean_fork64[stall_arr]),
            "pi64_vs_clean_canon_s64": RF.spearman_corr_and_perm_p(pi64[stall_arr], clean_can64[stall_arr]),
            "pi16_vs_ood_change_s16_to_s64": RF.spearman_corr_and_perm_p(
                pi16[stall_arr], delta_ood_clean[stall_arr]
            ),
            "pi64_vs_vc_gain_fork_s64": RF.spearman_corr_and_perm_p(
                pi64[stall_arr], delta_vc_fork64[stall_arr]
            ),
            "pi64_vs_vc_gain_canon_s64": RF.spearman_corr_and_perm_p(
                pi64[stall_arr], delta_vc_can64[stall_arr]
            ),
        },
        "wb_stall_quartiles_by_pi16": build_bins(stall_arr, pi16, n_bins=4),
        "wb_stall_quartiles_by_pi64": build_bins(stall_arr, pi64, n_bins=4),
        "all256_quartiles_by_pi16": build_bins(np.ones(len(prop_flags), dtype=bool), pi16, n_bins=4),
        "per_board_vectors": {
            "prop_solved": [bool(x) for x in prop_flags],
            "clean_pi_cell_s16": [round(float(x), 5) for x in pi16],
            "clean_pi_cell_s64": [round(float(x), 5) for x in pi64],
            "vc_pi_cell_s16": [round(float(x), 5) for x in vb["pi_cell_s16"]],
            "vc_pi_cell_s64": [round(float(x), 5) for x in vb["pi_cell_s64"]],
            "clean_fork_solve_s16": [round(float(x), 5) for x in clean_fork16],
            "clean_fork_solve_s64": [round(float(x), 5) for x in clean_fork64],
            "vc_fork_solve_s16": [round(float(x), 5) for x in vc_fork16],
            "vc_fork_solve_s64": [round(float(x), 5) for x in vc_fork64],
            "clean_canon_s16": [int(x) for x in clean_can16],
            "clean_canon_s64": [int(x) for x in clean_can64],
            "vc_canon_s16": [int(x) for x in vc_can16],
            "vc_canon_s64": [int(x) for x in vc_can64],
        },
    }


# =====================================================================
# Part 3: Uncapped Resume Estimators & Periodic-Refill Drift Audit
# =====================================================================

@torch.no_grad()
def run_uncapped_resume_audit(
    model: TS.PureSudokuTRM,
    refill_net: CR.RefillTransformer,
    mean_base: dict[str, torch.Tensor],
    v_head: torch.Tensor,
    w_digit: torch.Tensor,
    p_perp: torch.Tensor,
    qs: list[str],
    ans: list[str],
    prop_flags: list[bool],
    transplant_steps: tuple[int, ...] = (1, 2, 4, 8, 12),
    horizon: int = 32,
    bsz: int = 64,
) -> dict:
    """Evaluate single-step clean-state Resume without conflating delay and non-termination:
      - Separate `never_lock_rate_h32` (+ exact paired McNemar p) from
      - `median_delay_loops` & IQR strictly on boards that lock within H=32 under BOTH compared arms.
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

    v_head_d = v_head.to(dev)
    w_dig_d = w_digit.to(dev)
    p_perp_d = p_perp.to(dev)
    mu_orth = mean_base["mu_orth"].to(dev)
    mu_pref = mean_base["mu_pref"].to(dev)
    mu_zl = mean_base["mu_zl"].to(dev)

    total_clean_steps = max(transplant_steps) + horizon
    clean_preds = np.empty((total_clean_steps + 1, n, 81), dtype=np.int8)
    saved_snaps: dict[int, tuple[torch.Tensor, torch.Tensor]] = {
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
            if s in saved_snaps:
                saved_snaps[s][0][b0:b1] = zh.cpu()
                saved_snaps[s][1][b0:b1] = zl.cpu()

    clean_merged = np.where(empty_mask[None, :, :], clean_preds, given_np[None, :, :])
    clean_exact_all = (clean_merged == ans_np[None, :, :]).all(axis=-1)
    tau0_lock = RF.compute_lockin_step(clean_exact_all[: horizon + 1])

    flips_16 = np.zeros(n, dtype=np.float64)
    for b in range(n):
        em = empty_mask[b]
        seq = clean_merged[1:17, b, :][:, em]
        flips_16[b] = float(np.sum(seq[1:] != seq[:-1]))

    prop_arr = np.array(prop_flags, dtype=bool)
    stall_arr = ~prop_arr
    clean_locked_h32 = tau0_lock <= horizon

    def summarize_pair_delay(tau_a: np.ndarray, tau_b: np.ndarray, sub_mask: np.ndarray) -> dict:
        """Compare arm A (slower baseline, e.g. VC) minus arm B (faster ref, e.g. Full) strictly
        on boards in `sub_mask` where BOTH arms lock within `horizon` (zero cap mixing!).
        """
        lock_a = tau_a <= horizon
        lock_b = tau_b <= horizon
        m_sub = sub_mask
        n_sub = int(m_sub.sum())
        if n_sub == 0:
            return {"n_cohort": 0}

        both = m_sub & lock_a & lock_b
        b_only = m_sub & (~lock_a) & lock_b
        a_only = m_sub & lock_a & (~lock_b)
        neither = m_sub & (~lock_a) & (~lock_b)

        n_both = int(both.sum())
        if n_both > 0:
            d = (tau_a[both] - tau_b[both]).astype(np.float64)
            med = float(np.median(d))
            q25, q75 = float(np.percentile(d, 25)), float(np.percentile(d, 75))
            sf = RF.paired_signflip_p(d)
            n_pos = int((d > 0).sum())
            n_neg = int((d < 0).sum())
            n_zero = int((d == 0).sum())
        else:
            med, q25, q75 = 0.0, 0.0, 0.0
            sf = {"n": 0, "mean": 0.0, "ci95": [0.0, 0.0], "p_two_sided": 1.0}
            n_pos = n_neg = n_zero = 0

        return {
            "n_cohort": n_sub,
            "never_lock_arm_a_count": int((m_sub & (~lock_a)).sum()),
            "never_lock_arm_a_rate": float((~lock_a[m_sub]).mean()),
            "never_lock_arm_b_count": int((m_sub & (~lock_b)).sum()),
            "never_lock_arm_b_rate": float((~lock_b[m_sub]).mean()),
            "mcnemar_lock_a_vs_b": SC.exact_mcnemar_p(lock_b[m_sub], lock_a[m_sub]),
            "discordance": {
                "both_lock": n_both,
                "ref_b_only_locks": int(b_only.sum()),
                "arm_a_only_locks": int(a_only.sum()),
                "neither_locks": int(neither.sum()),
            },
            "uncapped_delay_among_both_lock": {
                "n_both": n_both,
                "median_loops": med,
                "iqr_loops": [q25, q75],
                "mean_loops": sf["mean"],
                "mean_ci95": sf["ci95"],
                "signflip_p_two_sided": sf["p_two_sided"],
                "n_delayed": n_pos,
                "n_faster": n_neg,
                "n_tied": n_zero,
            },
        }

    out_steps = {}
    for t in transplant_steps:
        zh_t_cpu, zl_t_cpu = saved_snaps[t]
        arms = ["V_only", "V_plus_C", "Full_S_t"]
        tau_lock: dict[str, np.ndarray] = {}
        ever_solve: dict[str, np.ndarray] = {}

        for arm in arms:
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
                    zh = zh_c.to(dt)
                    zl = zl_c.to(dt)
                elif arm == "V_only":
                    bo = CR.sphere_calibrate_orth(mu_orth.expand(m, -1, -1), vp, p_perp_d, v_head=v_head_d)
                    bp = mu_pref.expand(m, -1, -1)
                    bz = mu_zl.expand(m, -1, -1)
                    zh = torch.cat([bp, vp + bo], dim=1).to(dt)
                    zl = bz.to(dt)
                elif arm == "V_plus_C":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                        ro, rp, rz = refill_net(
                            inp_b, cand_b, vc, vp, w_dig_d, p_perp_d, calibrate_sphere=True, v_head=v_head_d
                        )
                    zh = torch.cat([rp.float(), vp + ro.float()], dim=1).to(dt)
                    zl = rz.to(dt)
                else:
                    raise ValueError(arm)

                preds_u[0, b0:b1] = model.digit_logits(zh).argmax(dim=-1).cpu().numpy().astype(np.int8)
                for u in range(1, horizon + 1):
                    zh, zl = model.act_step(zh, zl, xb)
                    preds_u[u, b0:b1] = model.digit_logits(zh).argmax(dim=-1).cpu().numpy().astype(np.int8)

            merged_u = np.where(empty_mask[None, :, :], preds_u, given_np[None, :, :])
            exact_u = (merged_u == ans_np[None, :, :]).all(axis=-1)
            tau_lock[arm] = RF.compute_lockin_step(exact_u)
            ever_solve[arm] = exact_u.any(axis=0)

        # Clean run not yet locked at step t (tau_lock(Full_S_t) > 0, without dropping eventual clean non-lockers!)
        not_locked_at_t = tau_lock["Full_S_t"] > 0
        cohorts = {
            "unselected_wb_prop": prop_arr,
            "unselected_wb_stall": stall_arr,
            "clean_locking_wb_stall": stall_arr & clean_locked_h32,
            "unresolved_at_t_all_wb_stall": stall_arr & not_locked_at_t,
            "active_clean_locking_wb_stall_tau0_gt_t": stall_arr & clean_locked_h32 & (tau0_lock > t),
        }

        cohort_summaries = {}
        for cname, cmask in cohorts.items():
            nc = int(cmask.sum())
            cohort_summaries[cname] = {
                "n": nc,
                "never_lock_rates_h32": {
                    arm: float((tau_lock[arm][cmask] > horizon).mean()) if nc > 0 else 0.0 for arm in arms
                },
                "ever_solve_rates_h32": {
                    arm: float(ever_solve[arm][cmask].mean()) if nc > 0 else 0.0 for arm in arms
                },
                "R_effect_VC_vs_Full": summarize_pair_delay(tau_lock["V_plus_C"], tau_lock["Full_S_t"], cmask),
                "C_effect_V_vs_VC": summarize_pair_delay(tau_lock["V_only"], tau_lock["V_plus_C"], cmask),
            }

        # Correlation on uncapped mutual-locking WB-Stall boards
        both_st = stall_arr & (tau_lock["V_plus_C"] <= horizon) & (tau_lock["Full_S_t"] <= horizon)
        uncapped_r_delay = (tau_lock["V_plus_C"][both_st] - tau_lock["Full_S_t"][both_st]).astype(np.float64)
        out_steps[f"step_{t}"] = {
            "cohorts": cohort_summaries,
            "uncapped_delay_vs_flips_wb_stall": RF.spearman_corr_and_perm_p(
                uncapped_r_delay, flips_16[both_st]
            ),
        }

    return out_steps


@torch.no_grad()
def run_periodic_refill_and_drift_audit(
    model: TS.PureSudokuTRM,
    refill_net: CR.RefillTransformer,
    v_head: torch.Tensor,
    w_digit: torch.Tensor,
    p_perp: torch.Tensor,
    qs: list[str],
    ans: list[str],
    prop_flags: list[bool],
    intervals: tuple[int, ...] = (1, 2, 4, 8),
    max_steps: int = 64,
    bsz: int = 64,
) -> dict:
    """Compare closed-loop refill applied every m in {1, 2, 4, 8} steps vs. clean (m=inf),
    plus track visible-logit entropy and board-conflict trajectory drift across s=1..64.
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

    v_head_d = v_head.to(dev)
    w_dig_d = w_digit.to(dev)
    p_perp_d = p_perp.to(dev)

    modes = [("clean", 0)] + [(f"refill_every_{m}", m) for m in intervals]
    out = {}

    for mode_name, m_int in modes:
        preds_all = np.empty((max_steps, n, 81), dtype=np.int8)
        ent_curve = np.zeros((max_steps, n), dtype=np.float64)

        for b0 in range(0, n, bsz):
            b1 = min(n, b0 + bsz)
            zh, zl = model.init_states(b1 - b0)
            xb = x_all[b0:b1]
            inp_b = inp_all[b0:b1]
            cand_b = cand_all[b0:b1]

            for s in range(max_steps):
                zh, zl = model.act_step(zh, zl, xb)
                logits = model.digit_logits(zh).float()
                probs = torch.softmax(logits, dim=-1)
                ent = -(probs * (probs + 1e-8).log()).sum(dim=-1)
                preds_all[s, b0:b1] = probs.argmax(dim=-1).cpu().numpy().astype(np.int8)
                ent_curve[s, b0:b1] = ent.mean(dim=-1).cpu().numpy()

                if m_int > 0 and ((s + 1) % m_int == 0) and s < max_steps - 1:
                    zb = zh[:, 16:, :].float()
                    vc = zb @ v_head_d
                    vp = vc @ v_head_d.T
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                        ro, rp, rz = refill_net(
                            inp_b, cand_b, vc, vp, w_dig_d, p_perp_d, calibrate_sphere=True, v_head=v_head_d
                        )
                    zh = torch.cat([rp.float(), vp + ro.float()], dim=1).to(dt)
                    zl = rz.to(dt)

        merged = np.where(empty_mask[None, :, :], preds_all, given_np[None, :, :])
        exact_steps = (merged == ans_np[None, :, :]).all(axis=-1)

        trapped_cycle = np.zeros(n, dtype=bool)
        for bi in range(n):
            if not exact_steps[-1, bi]:
                tail_b = [merged[s, bi].tobytes() for s in range(max_steps // 2, max_steps)]
                trapped_cycle[bi] = len(set(tail_b)) < len(tail_b)

        out[mode_name] = {
            "interval_steps": m_int,
            "wb_prop_s16": float(exact_steps[15, prop_arr].mean()),
            "wb_prop_s64": float(exact_steps[max_steps - 1, prop_arr].mean()),
            "wb_stall_s16": float(exact_steps[15, stall_arr].mean()),
            "wb_stall_s32": float(exact_steps[31, stall_arr].mean()),
            "wb_stall_s64": float(exact_steps[max_steps - 1, stall_arr].mean()),
            "wb_stall_ever_solved_s64": float(exact_steps[:, stall_arr].any(axis=0).mean()),
            "wb_stall_tail_cycle_rate_s64": float(trapped_cycle[stall_arr].mean()),
            "wb_stall_entropy_s16": float(ent_curve[15, stall_arr].mean()),
            "wb_stall_entropy_s64": float(ent_curve[max_steps - 1, stall_arr].mean()),
            "exact_s16_flags": exact_steps[15].tolist(),
            "exact_s64_flags": exact_steps[max_steps - 1].tolist(),
        }

    clean16 = np.array(out["clean"]["exact_s16_flags"], dtype=bool)
    clean64 = np.array(out["clean"]["exact_s64_flags"], dtype=bool)
    for k, v in out.items():
        s16 = np.array(v.pop("exact_s16_flags"), dtype=bool)
        s64 = np.array(v.pop("exact_s64_flags"), dtype=bool)
        v["mcnemar_s16_wb_stall_vs_clean"] = SC.exact_mcnemar_p(clean16[stall_arr], s16[stall_arr])
        v["mcnemar_s64_wb_stall_vs_clean"] = SC.exact_mcnemar_p(clean64[stall_arr], s64[stall_arr])

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "stage4_path_independence_and_scaling.json")
    parser.add_argument("--mem-fraction", type=float, default=0.72)
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

    t_start = time.time()
    all_1k_qs, _ = FRC.load_train_boards(1000)
    base_train_900 = all_1k_qs[:900]
    val_qs = all_1k_qs[900:1000]
    val_prop_flags = np.array([bool(SP.trace_propagation(q).solved) for q in val_qs], dtype=bool)

    bundle = json.loads(
        (REPO_ROOT / "mechinterp" / "loopscope02" / "results" / "benchmark_bundle.json").read_text("utf-8")
    )
    items = bundle["items"]
    items_qs = [it["question"] for it in items]
    items_ans = [it["answer"] for it in items]
    items_prop = [bool(it["prop_solved"]) for it in items]
    items_strata = [it["stratum"] for it in items]
    bench_qs, bench_ans, bench_prop = FRC.load_bench100()

    pool_8100, pool_meta = load_disjoint_train_pool(
        target_total=8100,
        base_train_900=base_train_900,
        forbidden_sets={"val_100": val_qs, "items256": items_qs, "bench100": bench_qs},
        cache_dir=args.cache_dir,
    )
    print(f"Loaded disjoint 8,100-board scaling pool: {pool_meta}", flush=True)

    # =====================================================================
    # 1. Primary Checkpoint (`gaoxin492`): Data-Size Scaling + Uncapped Resume + Path Independence
    # =====================================================================
    print("\n=== [1/2] Primary Checkpoint: gaoxin492 ===", flush=True)
    prim_model, prim_sha = TS.load_trm_checkpoint("gaoxin492", device=dev, dtype=torch.bfloat16)
    v_head_p, _, p_perp_p = CR.compute_head_projector_f64(prim_model.lm_head.weight)
    w_dig_p = (prim_model.lm_head.weight.detach().cpu().float()[2:11, :] @ v_head_p).contiguous()

    va_data_p = CR.collect_trm_states(prim_model, val_qs, n_steps=16, bsz=100)
    mm_data_p = collect_trm_states_memmap(prim_model, pool_8100, args.cache_dir, "gaoxin492", n_steps=16, bsz=256)

    # Epoch schedule giving ample convergence updates across every N_train
    scaling_specs = [
        (100, 36),
        (300, 36),
        (900, 30),
        (2700, 24),
        (8100, 18),
    ]
    saved_p_bundle = torch.load(args.cache_dir / "gaoxin492_refill_bundle.pt", map_location="cpu", weights_only=False)
    scaling_curve_primary: dict[str, dict] = {}
    prim_c5_8100: CR.RefillTransformer | None = None
    prim_mean_8100: dict[str, torch.Tensor] | None = None
    prim_c5_900 = CR.RefillTransformer(d_model=512, n_layers=2, n_heads=8, use_step_clock=False)
    prim_c5_900.load_state_dict(saved_p_bundle["C5_tf_2L_512_state"])
    prim_c5_900.to(dev).eval()

    for n_tr, n_ep in scaling_specs:
        ck_file = args.cache_dir / f"gaoxin492_C5_N{n_tr}.pt"
        mean_n = compute_mean_baseline_from_memmap(mm_data_p, n_tr, v_head_p, p_perp_p, dev=dev)
        if n_tr == 900:
            net_n = prim_c5_900
            meta_n = {"n_train_boards": 900, "n_train_states": 14400, "reused_stage1_c5": True}
            mean_n = saved_p_bundle["mean_base"]
        elif ck_file.exists():
            saved_ck = torch.load(ck_file, map_location="cpu", weights_only=False)
            net_n = CR.RefillTransformer(d_model=512, n_layers=2, n_heads=8, use_step_clock=False)
            net_n.load_state_dict(saved_ck["state"])
            net_n.to(dev).eval()
            meta_n = saved_ck["meta"]
        else:
            print(f"  [gaoxin492] Training C5_tf_2L_512 on N_train={n_tr} ({n_ep} epochs)...", flush=True)
            net_n, meta_n = train_c5_from_memmap(
                mm_data_p, n_tr, va_data_p, v_head_p, w_dig_p, p_perp_p, epochs=n_ep, batch_size=512, dev=dev
            )
            torch.save({"meta": meta_n, "state": {k: v.cpu() for k, v in net_n.state_dict().items()}}, ck_file)

        if n_tr == 8100:
            prim_c5_8100 = net_n
            prim_mean_8100 = mean_n

        # Evaluate val_100 R^2 (using fixed train_900 mean denominator so R^2 is strictly comparable across N!)
        def _wrap_n(inp, cand, vc, vp, s, calibrate_sphere=True, _net=net_n):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                return _net(
                    inp, cand, vc, vp, w_dig_p.to(dev), p_perp_p.to(dev),
                    calibrate_sphere=calibrate_sphere, v_head=v_head_p.to(dev),
                )

        r2_n = CR.evaluate_val_r2(
            _wrap_n, va_data_p, val_prop_flags, saved_p_bundle["mean_base"], v_head_p, p_perp_p, dev=dev
        )
        cl_items_n = CR.run_closed_loop_arm(
            prim_model, items_qs, items_ans, items_prop, items_strata, v_head_p, p_perp_p,
            "refill", "refill", "refill", _wrap_n, mean_n, max_steps=64, bsz=64, calibrate_sphere=True,
        )
        cl_bench_n = CR.run_closed_loop_arm(
            prim_model, bench_qs, bench_ans, bench_prop, None, v_head_p, p_perp_p,
            "refill", "refill", "refill", _wrap_n, mean_n, max_steps=64, bsz=64, calibrate_sphere=True,
        )
        # Clean-Reference McNemar against Stage 1 clean
        res_48_n = run_uncapped_resume_audit(
            prim_model, net_n, mean_n, v_head_p, w_dig_p, p_perp_p,
            items_qs, items_ans, items_prop, transplant_steps=(4, 8), horizon=32, bsz=64,
        )
        del cl_items_n["exact_s16_flags"], cl_items_n["exact_s64_flags"]
        del cl_bench_n["exact_s16_flags"], cl_bench_n["exact_s64_flags"]

        scaling_curve_primary[f"N_{n_tr}"] = {
            "train_meta": meta_n,
            "val_100_r2": r2_n,
            "items256_closed_loop": cl_items_n,
            "bench100_closed_loop": cl_bench_n,
            "items256_resume_t4_t8": res_48_n,
        }
        st_un4 = res_48_n["step_4"]["cohorts"]["unselected_wb_stall"]["R_effect_VC_vs_Full"]
        st_un8 = res_48_n["step_8"]["cohorts"]["unselected_wb_stall"]["R_effect_VC_vs_Full"]
        print(
            f"  [Scale N={n_tr:4d}] val_R2={r2_n['all']['r2_joint_hidden']:.4f} "
            f"(stall={r2_n['stalled']['r2_joint_hidden']:.4f}) | "
            f"CL Stall@16={cl_items_n['wb_stall']['solve_s16']:.3f} Stall@64={cl_items_n['wb_stall']['solve_s64']:.3f} | "
            f"Resume@8 never_lock VC={st_un8['never_lock_arm_a_rate']:.3f} vs Full={st_un8['never_lock_arm_b_rate']:.3f}, "
            f"med_delay={st_un8['uncapped_delay_among_both_lock']['median_loops']:.1f}",
            flush=True,
        )

    assert prim_c5_8100 is not None and prim_mean_8100 is not None

    # Full Uncapped Resume at t in {1, 2, 4, 8, 12} for BOTH N=900 and N=8,100 on primary
    print("  [gaoxin492] Running full uncapped Resume (t=1,2,4,8,12) & periodic-refill audit...", flush=True)
    prim_resume_n900 = run_uncapped_resume_audit(
        prim_model, prim_c5_900, saved_p_bundle["mean_base"], v_head_p, w_dig_p, p_perp_p,
        items_qs, items_ans, items_prop, transplant_steps=(1, 2, 4, 8, 12), horizon=32, bsz=64,
    )
    prim_resume_n8100 = run_uncapped_resume_audit(
        prim_model, prim_c5_8100, prim_mean_8100, v_head_p, w_dig_p, p_perp_p,
        items_qs, items_ans, items_prop, transplant_steps=(1, 2, 4, 8, 12), horizon=32, bsz=64,
    )
    prim_periodic_n8100 = run_periodic_refill_and_drift_audit(
        prim_model, prim_c5_8100, v_head_p, w_dig_p, p_perp_p, items_qs, items_ans, items_prop,
        intervals=(1, 2, 4, 8), max_steps=64, bsz=64,
    )

    # Unconditional Path Independence & Enforced Path Independence (K=8, s=1..64) on Primary
    print("  [gaoxin492] Running K=8 unconditional & enforced path-independence rollouts (s=1..64)...", flush=True)
    prim_pi_out = {}
    for sig in (0.35, 1.00):
        r_clean = run_k_fork_rollouts(
            prim_model, items_qs, items_ans, k_forks=8, sigma=sig, max_steps=64,
            refill_net=None, v_head=v_head_p, w_digit=w_dig_p, p_perp=p_perp_p, bsz=128,
        )
        r_vc = run_k_fork_rollouts(
            prim_model, items_qs, items_ans, k_forks=8, sigma=sig, max_steps=64,
            refill_net=prim_c5_8100, v_head=v_head_p, w_digit=w_dig_p, p_perp=p_perp_p, bsz=128,
        )
        sc_clean = score_unconditional_path_independence(r_clean, items_qs, items_ans, items_prop, dev=dev)
        sc_vc = score_unconditional_path_independence(r_vc, items_qs, items_ans, items_prop, dev=dev)
        rel = analyze_per_board_pi_vs_solve_and_refill(sc_clean, sc_vc, items_prop)
        prim_pi_out[f"sigma_{sig:.2f}"] = {
            "clean": sc_clean["summary"],
            "vc_c5_n8100": sc_vc["summary"],
            "per_board_analysis": rel,
        }
        print(
            f"    [Primary K=8 sigma={sig:.2f}] Uncond PI_board@64 Prop={sc_clean['summary']['wb_prop']['uncond_pi_board_s64']:.3f} "
            f"Stall={sc_clean['summary']['wb_stall']['uncond_pi_board_s64']:.3f} (VC->{sc_vc['summary']['wb_stall']['uncond_pi_board_s64']:.3f}) | "
            f"PI_cell@64 Stall={sc_clean['summary']['wb_stall']['uncond_pi_cell_s64']:.3f} (VC->{sc_vc['summary']['wb_stall']['uncond_pi_cell_s64']:.3f}) | "
            f"OutcomeSplit@16={sc_clean['summary']['wb_stall']['outcome_split_board_rate_s16']:.3f}",
            flush=True,
        )
        del r_clean, r_vc

    del prim_model, va_data_p, mm_data_p
    if dev == "cuda":
        torch.cuda.empty_cache()

    # =====================================================================
    # 2. Replication Checkpoint (`yagizdevre`): N=8,100 Scaling + Uncapped Resume + Path Independence
    # =====================================================================
    print("\n=== [2/2] Replication Checkpoint: yagizdevre ===", flush=True)
    repl_model, repl_sha = TS.load_trm_checkpoint("yagizdevre", device=dev, dtype=torch.bfloat16)
    v_head_r, _, p_perp_r = CR.compute_head_projector_f64(repl_model.lm_head.weight)
    w_dig_r = (repl_model.lm_head.weight.detach().cpu().float()[2:11, :] @ v_head_r).contiguous()

    va_data_r = CR.collect_trm_states(repl_model, val_qs, n_steps=16, bsz=100)
    mm_data_r = collect_trm_states_memmap(repl_model, pool_8100, args.cache_dir, "yagizdevre", n_steps=16, bsz=256)
    saved_r_bundle = torch.load(args.cache_dir / "yagizdevre_refill_bundle.pt", map_location="cpu", weights_only=False)

    repl_c5_900 = CR.RefillTransformer(d_model=512, n_layers=2, n_heads=8, use_step_clock=False)
    repl_c5_900.load_state_dict(saved_r_bundle["C5_tf_2L_512_state"])
    repl_c5_900.to(dev).eval()

    ck_r8100 = args.cache_dir / "yagizdevre_C5_N8100.pt"
    repl_mean_8100 = compute_mean_baseline_from_memmap(mm_data_r, 8100, v_head_r, p_perp_r, dev=dev)
    if ck_r8100.exists():
        saved_rck = torch.load(ck_r8100, map_location="cpu", weights_only=False)
        repl_c5_8100 = CR.RefillTransformer(d_model=512, n_layers=2, n_heads=8, use_step_clock=False)
        repl_c5_8100.load_state_dict(saved_rck["state"])
        repl_c5_8100.to(dev).eval()
        repl_meta_8100 = saved_rck["meta"]
    else:
        print("  [yagizdevre] Training C5_tf_2L_512 on N_train=8100 (18 epochs)...", flush=True)
        repl_c5_8100, repl_meta_8100 = train_c5_from_memmap(
            mm_data_r, 8100, va_data_r, v_head_r, w_dig_r, p_perp_r, epochs=18, batch_size=512, dev=dev
        )
        torch.save(
            {"meta": repl_meta_8100, "state": {k: v.cpu() for k, v in repl_c5_8100.state_dict().items()}},
            ck_r8100,
        )

    def _wrap_r8100(inp, cand, vc, vp, s, calibrate_sphere=True):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            return repl_c5_8100(
                inp, cand, vc, vp, w_dig_r.to(dev), p_perp_r.to(dev),
                calibrate_sphere=calibrate_sphere, v_head=v_head_r.to(dev),
            )

    repl_r2_8100 = CR.evaluate_val_r2(
        _wrap_r8100, va_data_r, val_prop_flags, saved_r_bundle["mean_base"], v_head_r, p_perp_r, dev=dev
    )
    repl_cl_items_8100 = CR.run_closed_loop_arm(
        repl_model, items_qs, items_ans, items_prop, items_strata, v_head_r, p_perp_r,
        "refill", "refill", "refill", _wrap_r8100, repl_mean_8100, max_steps=64, bsz=64, calibrate_sphere=True,
    )
    repl_cl_bench_8100 = CR.run_closed_loop_arm(
        repl_model, bench_qs, bench_ans, bench_prop, None, v_head_r, p_perp_r,
        "refill", "refill", "refill", _wrap_r8100, repl_mean_8100, max_steps=64, bsz=64, calibrate_sphere=True,
    )
    del repl_cl_items_8100["exact_s16_flags"], repl_cl_items_8100["exact_s64_flags"]
    del repl_cl_bench_8100["exact_s16_flags"], repl_cl_bench_8100["exact_s64_flags"]
    print(
        f"  [Repl Scale N=8100] val_R2={repl_r2_8100['all']['r2_joint_hidden']:.4f} "
        f"(stall={repl_r2_8100['stalled']['r2_joint_hidden']:.4f}) | "
        f"CL Stall@16={repl_cl_items_8100['wb_stall']['solve_s16']:.3f} Stall@64={repl_cl_items_8100['wb_stall']['solve_s64']:.3f}",
        flush=True,
    )

    print("  [yagizdevre] Running full uncapped Resume (t=1,2,4,8,12) & periodic-refill audit...", flush=True)
    repl_resume_n900 = run_uncapped_resume_audit(
        repl_model, repl_c5_900, saved_r_bundle["mean_base"], v_head_r, w_dig_r, p_perp_r,
        items_qs, items_ans, items_prop, transplant_steps=(1, 2, 4, 8, 12), horizon=32, bsz=64,
    )
    repl_resume_n8100 = run_uncapped_resume_audit(
        repl_model, repl_c5_8100, repl_mean_8100, v_head_r, w_dig_r, p_perp_r,
        items_qs, items_ans, items_prop, transplant_steps=(1, 2, 4, 8, 12), horizon=32, bsz=64,
    )
    repl_resume_bench_n8100 = run_uncapped_resume_audit(
        repl_model, repl_c5_8100, repl_mean_8100, v_head_r, w_dig_r, p_perp_r,
        bench_qs, bench_ans, bench_prop, transplant_steps=(2, 4, 8), horizon=32, bsz=64,
    )
    repl_periodic_n8100 = run_periodic_refill_and_drift_audit(
        repl_model, repl_c5_8100, v_head_r, w_dig_r, p_perp_r, items_qs, items_ans, items_prop,
        intervals=(1, 2, 4, 8), max_steps=64, bsz=64,
    )

    print("  [yagizdevre] Running K=8 unconditional & enforced path-independence rollouts (s=1..64)...", flush=True)
    repl_pi_out = {}
    for sig in (0.35, 1.00):
        r_clean = run_k_fork_rollouts(
            repl_model, items_qs, items_ans, k_forks=8, sigma=sig, max_steps=64,
            refill_net=None, v_head=v_head_r, w_digit=w_dig_r, p_perp=p_perp_r, bsz=128,
        )
        r_vc = run_k_fork_rollouts(
            repl_model, items_qs, items_ans, k_forks=8, sigma=sig, max_steps=64,
            refill_net=repl_c5_8100, v_head=v_head_r, w_digit=w_dig_r, p_perp=p_perp_r, bsz=128,
        )
        sc_clean = score_unconditional_path_independence(r_clean, items_qs, items_ans, items_prop, dev=dev)
        sc_vc = score_unconditional_path_independence(r_vc, items_qs, items_ans, items_prop, dev=dev)
        rel = analyze_per_board_pi_vs_solve_and_refill(sc_clean, sc_vc, items_prop)
        repl_pi_out[f"sigma_{sig:.2f}"] = {
            "clean": sc_clean["summary"],
            "vc_c5_n8100": sc_vc["summary"],
            "per_board_analysis": rel,
        }
        print(
            f"    [Repl K=8 sigma={sig:.2f}] Uncond PI_board@64 Prop={sc_clean['summary']['wb_prop']['uncond_pi_board_s64']:.3f} "
            f"Stall={sc_clean['summary']['wb_stall']['uncond_pi_board_s64']:.3f} (VC->{sc_vc['summary']['wb_stall']['uncond_pi_board_s64']:.3f}) | "
            f"PI_cell@64 Stall={sc_clean['summary']['wb_stall']['uncond_pi_cell_s64']:.3f} (VC->{sc_vc['summary']['wb_stall']['uncond_pi_cell_s64']:.3f}) | "
            f"OutcomeSplit@16={sc_clean['summary']['wb_stall']['outcome_split_board_rate_s16']:.3f}",
            flush=True,
        )
        del r_clean, r_vc

    payload = {
        "provenance": "post_review_diagnostic_controls_v014",
        "base_preregistration_commit": "b1ec77550997cbc5ae2cd84618af74e8f34e38e7",
        "elapsed_sec": float(time.time() - t_start),
        "disjoint_train_pool": pool_meta,
        "primary_gaoxin492": {
            "checkpoint_sha256": prim_sha,
            "data_scaling_curve_c5": scaling_curve_primary,
            "uncapped_resume_items256_n900": prim_resume_n900,
            "uncapped_resume_items256_n8100": prim_resume_n8100,
            "periodic_refill_and_drift_n8100": prim_periodic_n8100,
            "unconditional_and_enforced_path_independence_items256": prim_pi_out,
        },
        "replication_yagizdevre": {
            "checkpoint_sha256": repl_sha,
            "n8100_c5": {
                "train_meta": repl_meta_8100,
                "val_100_r2": repl_r2_8100,
                "items256_closed_loop": repl_cl_items_8100,
                "bench100_closed_loop": repl_cl_bench_8100,
            },
            "uncapped_resume_items256_n900": repl_resume_n900,
            "uncapped_resume_items256_n8100": repl_resume_n8100,
            "uncapped_resume_bench100_n8100": repl_resume_bench_n8100,
            "periodic_refill_and_drift_n8100": repl_periodic_n8100,
            "unconditional_and_enforced_path_independence_items256": repl_pi_out,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    print(f"\nWrote Stage 4 receipt to {args.out} in {payload['elapsed_sec']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
