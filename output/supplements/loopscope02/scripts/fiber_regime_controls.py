#!/usr/bin/env python3
"""LOOPSCOPE-02 Follow-up Controls: Head-Orthogonal Fiber Regime Boundary,
Dose-Response PCA, Manifold/Donor Controls, and Linear Probe Decodability.

Addresses four methodological questions:
  1. Per-stratum breakdown of head-space projection across propagation-solvable
     vs. stalled boards (both target-cell accuracy and 81-cell exact board solve),
     plus independent replication on `yagizdevre/trm-sudoku-extreme-att`.
  2. Contrast model (`LoopedChase`, 12 Q4 checkpoints across lambda in {0, 0.01, 0.1, 1}):
     inter-loop projection of carried cursor state onto the 16-d readout row space
     (erasing 240 of 256 dimensions at every loop) across depths 1..8 and held-out 9..16.
  3. Dose & manifold controls on 7M Sudoku TRM (`bench100` and `items256`):
       - `clean`
       - `reset_zL`
       - `fiber_erased` (raw H_init fill + reset_zL, matching registered P5)
       - `zH_fiber_only_erased` (raw H_init fill, keeping z_L live)
       - `fiber_norm_restored` (project onto 11-d P_head, restore exact clean cell L2 norm)
       - `fiber_train_mean_filled` (replace 501-d fiber with step-s train-split mean fiber)
       - `fiber_step1_frozen` (freeze 501-d fiber at board's own step-1 fiber for s >= 2)
       - `fiber_donor_swapped` (graft donor board's live step-s 501-d fiber for s >= 1)
       - `fiber_topk_pca` for k in {0, 4, 8, 16, 32, 64, 128, 256, 501} fitted strictly on
         `train.csv` boards (zero test-board leakage).
  4. Linear probe audit on the 501-d `lm_head`-nullspace fiber `(I - P_head) z_H(s)`
     (trained on `train.csv` boards, evaluated on held-out `bench100` empty cells):
     decodability of current `lm_head` digit commitment, final ground-truth digit,
     and valid candidate-digit elimination masks.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "mechinterp" / "loopscope01" / "scripts"))

from mechinterp.loopscope02.scripts import score_loopscope02 as SC  # noqa: E402
from mechinterp.loopscope02.scripts import sudoku_prop as SP  # noqa: E402
from mechinterp.loopscope02.scripts import trm_sudoku as TS   # noqa: E402
import loopscope_toy as LT  # noqa: E402


RMP_TOY_Q4 = Path(r"C:\Users\patri\Documents\Codex\LoopedTransformers-rmp\experiments\recurrence_monitorability_v0\loopscope_toy\ckpt")
RMP_TOY_Q5 = Path(r"C:\Users\patri\Documents\Codex\LoopedTransformers-rmp\experiments\recurrence_monitorability_v0\loopscope_toy_q5\ckpt")
RESULTS_DIR = REPO_ROOT / "mechinterp" / "loopscope02" / "results"


def compute_head_projector(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return orthonormal basis V_head [d, r] and projector P_head = V V^T [d, d] in float32 on CPU."""
    w = weight.detach().cpu().double()
    _u, s, vh = torch.linalg.svd(w, full_matrices=False)
    rank = int((s > 1e-8 * s[0]).sum().item())
    v = vh[:rank].T.contiguous().float()  # [d, rank]
    p = (v @ v.T).contiguous()            # [d, d]
    return v, p


@torch.no_grad()
def evaluate_looped_chase_contrast(dev: str = "cpu") -> dict:
    """Run P_head projection controls across all Q4 (and available Q5) LoopedChase checkpoints."""
    n_loops = 16
    batch_sz = 512
    # Deterministic derangement for donor fiber control
    rng = np.random.default_rng(20261113)
    perm = rng.permutation(batch_sz)
    donor_idx = torch.empty(batch_sz, dtype=torch.long, device=dev)
    donor_idx[torch.tensor(perm, device=dev)] = torch.tensor(np.roll(perm, -1), dtype=torch.long, device=dev)

    ckpt_paths = sorted(RMP_TOY_Q4.glob("*.final.pt"))
    q5_paths = sorted(RMP_TOY_Q5.glob("*.final.pt")) if RMP_TOY_Q5.exists() else []
    batch_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def eval_one_ckpt(pt_path: Path) -> dict:
        ck = torch.load(pt_path, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        n_nodes_cfg = int(cfg["n_nodes"])
        if n_nodes_cfg not in batch_cache:
            t_cpu, h_cpu = LT.make_batch(n_nodes=n_nodes_cfg, batch=batch_sz, seed=910100, max_depth=n_loops)
            batch_cache[n_nodes_cfg] = (t_cpu.to(dev), h_cpu.to(dev))
        tokens, hops = batch_cache[n_nodes_cfg]
        m = LT.build_model(LT.model_args_from(cfg))
        m.load_state_dict(ck["model"])
        m.to(dev).eval()

        # Effective linear readout map before bias: readout(h) = head(ln_f(h))
        d = cfg["d"]
        ones = torch.ones((d, 1), dtype=torch.float32)
        j_center = torch.eye(d, dtype=torch.float32) - (ones @ ones.T) / float(d)
        w_eff = (m.head.weight.detach().cpu().float() * m.ln_f.weight.detach().cpu().float()[None, :]) @ j_center
        _v_eff, p_eff_cpu = compute_head_projector(w_eff)
        _v_raw, p_raw_cpu = compute_head_projector(m.head.weight.detach().cpu())
        p_eff = p_eff_cpu.to(dev)
        p_raw = p_raw_cpu.to(dev)

        # Precompute clean trajectory so donor_fiber can graft exact step-t donor orthogonal component
        n = m.n_nodes
        edges = tokens[:, : 2 * n].reshape(tokens.shape[0], n, 2)
        h_in = torch.cat([m.tok(edges[..., 0]) + m.dst(edges[..., 1]), m.tok(tokens[:, 2 * n :])], dim=1)
        e = h_in + m.pos(torch.arange(h_in.shape[1], device=dev))

        clean_cursor_orth = []
        h = torch.zeros_like(e)
        clean_preds = []
        for _t in range(n_loops):
            h = m.inject(h) + e
            for blk in m.core:
                h = blk(h)
            clean_preds.append(m.readout(h[:, -1]).argmax(dim=-1))
            c = h[:, -1]
            clean_cursor_orth.append(c - (c @ p_eff))
        clean_preds_t = torch.stack(clean_preds)

        modes = [
            "cursor_fiber_erased_eff",
            "cursor_fiber_erased_raw",
            "cursor_fiber_ln_preserved",
            "cursor_donor_fiber",
            "cursor_step1_frozen",
            "all_tokens_fiber_erased",
        ]
        out_metrics = {}

        def score_preds(pr_stack: torch.Tensor) -> dict:
            acc_by_d = [(pr_stack[k] == hops[:, k]).float().mean().item() for k in range(n_loops)]
            return {
                "d1_8_mean": float(np.mean(acc_by_d[:8])),
                "d8": float(acc_by_d[7]),
                "d9_16_mean": float(np.mean(acc_by_d[8:])),
            }

        out_metrics["clean"] = score_preds(clean_preds_t)

        for mode in modes:
            h = torch.zeros_like(e)
            preds = []
            step1_orth = None
            for step in range(n_loops):
                h = m.inject(h) + e
                for blk in m.core:
                    h = blk(h)
                preds.append(m.readout(h[:, -1]).argmax(dim=-1))
                c = h[:, -1]
                if mode == "cursor_fiber_erased_eff":
                    h = h.clone()
                    h[:, -1] = c @ p_eff
                elif mode == "cursor_fiber_erased_raw":
                    h = h.clone()
                    h[:, -1] = c @ p_raw
                elif mode == "cursor_fiber_ln_preserved":
                    # Project onto p_eff while restoring exact pre-projection mean and L2 norm around mean
                    h = h.clone()
                    mu = c.mean(dim=-1, keepdim=True)
                    c_centered = c - mu
                    orig_norm = c_centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    proj = (c @ p_eff)
                    proj_c = proj - proj.mean(dim=-1, keepdim=True)
                    proj_scaled = proj_c * (orig_norm / proj_c.norm(dim=-1, keepdim=True).clamp_min(1e-8)) + mu
                    h[:, -1] = proj_scaled
                elif mode == "cursor_donor_fiber":
                    h = h.clone()
                    h[:, -1] = (c @ p_eff) + clean_cursor_orth[step][donor_idx]
                elif mode == "cursor_step1_frozen":
                    if step == 0:
                        step1_orth = c - (c @ p_eff)
                    else:
                        h = h.clone()
                        h[:, -1] = (c @ p_eff) + step1_orth
                elif mode == "all_tokens_fiber_erased":
                    h = h @ p_eff
            out_metrics[mode] = score_preds(torch.stack(preds))

        return {
            "arm": pt_path.stem.replace(".final", ""),
            "lam": float(cfg.get("lam", -1.0)),
            "seed": int(cfg.get("seed", -1)),
            "d": int(d),
            "head_rank": int(_v_eff.shape[1]),
            "orth_dim": int(d - _v_eff.shape[1]),
            "metrics": out_metrics,
        }

    q4_res = [eval_one_ckpt(p) for p in ckpt_paths]
    q5_res = [eval_one_ckpt(p) for p in q5_paths]

    # Summarize Q4 by lambda
    by_lam = {}
    for lam in sorted({r["lam"] for r in q4_res}):
        sub = [r for r in q4_res if r["lam"] == lam]
        by_lam[str(lam)] = {
            mode: {
                k: float(np.mean([r["metrics"][mode][k] for r in sub]))
                for k in ("d1_8_mean", "d8", "d9_16_mean")
            }
            for mode in sub[0]["metrics"].keys()
        }

    return {"q4_arms": q4_res, "q5_arms": q5_res, "q4_mean_by_lambda": by_lam}


def load_train_boards(n_train: int = 200) -> tuple[list[str], list[str]]:
    files = SP.resolve_dataset_files()
    qs, ans = [], []
    with open(files["train.csv"], "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question") or row.get("puzzle") or row[reader.fieldnames[1]]
            a = row.get("answer") or row.get("solution") or row[reader.fieldnames[2]]
            if len(q) == 81 and len(a) == 81:
                qs.append(q)
                ans.append(a)
                if len(qs) >= n_train:
                    break
    return qs, ans


def load_bench100() -> tuple[list[str], list[str], list[bool]]:
    files = SP.resolve_dataset_files()
    qs, ans, prop_flags = [], [], []
    with open(files["test_sudoku_bench.csv"], "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question") or row.get("puzzle") or row[reader.fieldnames[1]]
            a = row.get("answer") or row.get("solution") or row[reader.fieldnames[2]]
            if len(q) == 81 and len(a) == 81:
                qs.append(q)
                ans.append(a)
                prop_flags.append(bool(SP.trace_propagation(q).solved))
    return qs, ans, prop_flags


@torch.no_grad()
def fit_train_fiber_pca_and_probes(
    model: TS.PureSudokuTRM,
    p_head: torch.Tensor,
    train_qs: list[str],
    train_ans: list[str],
    n_steps: int = 16,
) -> dict:
    """Collect clean trajectories on train.csv (zero test leakage) to fit:
      1. Step-wise train mean fiber mu_orth[s] [1, 81, 512]
      2. Orthonormal top-501 PCA basis V_pca [512, 501] of centered (I - P_head) z_H
      3. Ridge linear probes on (I - P_head) z_H(s) for s in {1, 4, 16}.
    """
    dev = next(model.parameters()).device
    p_orth = torch.eye(512, device=dev, dtype=torch.float32) - p_head
    n = len(train_qs)
    inp = TS.encode_questions(train_qs, dev)
    x_all = model.encode_inputs(inp)
    puz_len = model.puzzle_emb_len

    # Collect board z_H across all 16 steps for train boards in batches of 50
    bsz = 50
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

    p_orth_cpu = p_orth.cpu()
    orth_steps = torch.einsum("snbd,de->snbe", zh_steps, p_orth_cpu)  # [16, N, 81, 512]
    step_mean_orth = orth_steps.mean(dim=1)  # [16, 81, 512]
    global_mean_orth = orth_steps.mean(dim=(0, 1, 2))  # [512]

    # Fit PCA on pooled centered orthogonal states (subsample 24,000 vectors across steps 1..15)
    flat = (orth_steps[:-1] - global_mean_orth).reshape(-1, 512)
    rng = np.random.default_rng(20261113)
    sub_idx = rng.choice(flat.shape[0], size=min(24000, flat.shape[0]), replace=False)
    sample = flat[sub_idx].double()
    # Project again onto p_orth in float64 to guarantee strict orthogonality to p_head
    p_orth64 = torch.eye(512, dtype=torch.float64) - (
        compute_head_projector(model.lm_head.weight)[0].double()
        @ compute_head_projector(model.lm_head.weight)[0].double().T
    )
    sample = sample @ p_orth64
    _u, s_vals, vh = torch.linalg.svd(sample, full_matrices=False)
    q64, _ = torch.linalg.qr(p_orth64 @ vh[:501].T.contiguous())
    q_pca = q64[:, :501].contiguous().float()  # [512, 501] orthonormal inside ker(lm_head)
    ev_cum = np.cumsum((s_vals[:501].numpy() ** 2)) / max(float((s_vals.numpy() ** 2).sum()), 1e-12)

    # Fit ridge probes at steps s_idx in {0, 3, 15} (steps 1, 4, 16) on empty cells only
    empty_masks = np.array([[ch == "." for ch in q] for q in train_qs], dtype=bool)  # [N, 81]
    true_digits = np.array([[int(ch) - 1 for ch in a] for a in train_ans], dtype=np.int64)  # [N, 81]
    # Initial clue candidate masks [N, 81, 9]
    init_cands = np.zeros((n, 81, 9), dtype=np.float32)
    for bi, q in enumerate(train_qs):
        giv = SP.parse_grid(q)
        init_cands[bi] = SP.candidate_mask(giv).astype(np.float32)

    probes = {}
    reg = 10.0
    for s_idx in (0, 3, 15):
        x_tr = orth_steps[s_idx].numpy()[empty_masks]  # [M, 512]
        # Add bias column
        x_aug = np.concatenate([x_tr, np.ones((x_tr.shape[0], 1), dtype=np.float32)], axis=1)
        gram = x_aug.T @ x_aug + reg * np.eye(x_aug.shape[1], dtype=np.float32)

        y_true = np.eye(9, dtype=np.float32)[true_digits[empty_masks]]
        y_head = np.eye(9, dtype=np.float32)[logits_steps[s_idx].argmax(dim=-1).numpy()[empty_masks]]
        y_cand = init_cands[empty_masks]

        w_true = np.linalg.solve(gram, x_aug.T @ y_true)
        w_head = np.linalg.solve(gram, x_aug.T @ y_head)
        w_cand = np.linalg.solve(gram, x_aug.T @ y_cand)
        probes[s_idx] = (w_true, w_head, w_cand)

    return {
        "step_mean_orth": step_mean_orth,       # [16, 81, 512] CPU
        "global_mean_orth": global_mean_orth,   # [512] CPU
        "q_pca": q_pca[:, :501],                # [512, 501] CPU
        "pca_cum_var": {str(k): float(ev_cum[min(k - 1, len(ev_cum) - 1)]) if k > 0 else 0.0 for k in (0, 4, 8, 16, 32, 64, 128, 256, 501)},
        "probes": probes,
    }


@torch.no_grad()
def evaluate_probes_on_split(
    model: TS.PureSudokuTRM,
    p_head: torch.Tensor,
    probes: dict,
    qs: list[str],
    ans: list[str],
    prop_flags: list[bool],
    n_steps: int = 16,
) -> dict:
    dev = next(model.parameters()).device
    p_orth = (torch.eye(512, device=dev, dtype=torch.float32) - p_head).cpu()
    n = len(qs)
    inp = TS.encode_questions(qs, dev)
    x_all = model.encode_inputs(inp)
    puz_len = model.puzzle_emb_len

    zh_steps = torch.zeros((n_steps, n, 81, 512), dtype=torch.float32)
    logits_steps = torch.zeros((n_steps, n, 81, 9), dtype=torch.float32)
    bsz = 50
    for b0 in range(0, n, bsz):
        b1 = min(n, b0 + bsz)
        zh, zl = model.init_states(b1 - b0)
        for s in range(n_steps):
            zh, zl = model.act_step(zh, zl, x_all[b0:b1])
            zh_steps[s, b0:b1] = zh[:, puz_len:, :].float().cpu()
            logits_steps[s, b0:b1] = model.digit_logits(zh).cpu()

    orth_steps = torch.einsum("snbd,de->snbe", zh_steps, p_orth)
    # Verify max absolute logit leak from orth_steps through lm_head is ~0
    head_w_cpu = model.lm_head.weight.detach().float().cpu()[:, :]
    max_head_logit_on_orth = float((orth_steps @ head_w_cpu.T).abs().max().item())

    empty_masks = np.array([[ch == "." for ch in q] for q in qs], dtype=bool)
    true_digits = np.array([[int(ch) - 1 for ch in a] for a in ans], dtype=np.int64)
    init_cands = np.zeros((n, 81, 9), dtype=np.float32)
    for bi, q in enumerate(qs):
        giv = SP.parse_grid(q)
        init_cands[bi] = SP.candidate_mask(giv).astype(np.float32)

    out = {"max_lm_head_logit_on_orth_fiber": max_head_logit_on_orth, "by_step": {}}
    prop_arr = np.array(prop_flags, dtype=bool)
    for s_idx in (0, 3, 15):
        w_true, w_head, w_cand = probes[s_idx]
        step_lbl = f"step_{s_idx + 1}"
        out["by_step"][step_lbl] = {}
        for sub_name, b_mask in [("all", np.ones(n, dtype=bool)), ("prop", prop_arr), ("stalled", ~prop_arr)]:
            em = empty_masks[b_mask]
            x_te = orth_steps[s_idx].numpy()[b_mask][em]
            x_aug = np.concatenate([x_te, np.ones((x_te.shape[0], 1), dtype=np.float32)], axis=1)
            pred_true = (x_aug @ w_true).argmax(axis=-1)
            pred_head = (x_aug @ w_head).argmax(axis=-1)
            score_cand = x_aug @ w_cand
            yt = true_digits[b_mask][em]
            yh = logits_steps[s_idx].argmax(dim=-1).numpy()[b_mask][em]
            yc = init_cands[b_mask][em]
            head_own_acc = float((yh == yt).mean())
            cand_acc = float(((score_cand > 0.5).astype(np.float32) == yc).mean())
            out["by_step"][step_lbl][sub_name] = {
                "fiber_probe_true_digit_acc": float((pred_true == yt).mean()),
                "fiber_probe_current_head_argmax_acc": float((pred_head == yh).mean()),
                "fiber_probe_candidate_mask_acc": cand_acc,
                "lm_head_true_digit_acc": head_own_acc,
            }
    return out


@torch.no_grad()
def run_trm_fiber_suite_on_split(
    model: TS.PureSudokuTRM,
    p_head: torch.Tensor,
    pca_fit: dict,
    qs: list[str],
    ans: list[str],
    donor_indices: list[int],
    target_indices: list[int] | None,
    strata: list[str] | None,
    prop_flags: list[bool],
    n_steps: int = 16,
    pca_ks: tuple[int, ...] = (0, 4, 8, 16, 32, 64, 128, 256, 501),
) -> dict:
    """Run all geometry, norm, donor, step1-frozen, and top-k PCA interventions on a Sudoku split."""
    dev = next(model.parameters()).device
    dt = model.embed_tokens.weight.dtype
    puz_len = model.puzzle_emb_len
    n = len(qs)
    inp_all = TS.encode_questions(qs, dev)
    ans_all = TS.encode_answers(ans, dev)
    giv_mask = (inp_all != 1)  # True where given clue is present [N, 81]
    x_all = model.encode_inputs(inp_all)

    p_head_d = p_head.to(device=dev, dtype=torch.float32)
    i_minus_p = torch.eye(512, device=dev, dtype=torch.float32) - p_head_d
    h_init_orth = (model.H_init.float().to(dev) @ i_minus_p).view(1, 1, 512)
    step_mean_orth = pca_fit["step_mean_orth"].to(dev)      # [16, 81, 512]
    global_mean_orth = pca_fit["global_mean_orth"].to(dev)  # [512]
    q_pca = pca_fit["q_pca"].to(dev)                        # [512, 501]

    # First run clean trajectories to record exact donor step-s orthogonal states
    bsz = 64
    clean_orth_steps = torch.zeros((n_steps, n, 81, 512), dtype=torch.bfloat16)
    for b0 in range(0, n, bsz):
        b1 = min(n, b0 + bsz)
        zh, zl = model.init_states(b1 - b0)
        for s in range(n_steps):
            zh, zl = model.act_step(zh, zl, x_all[b0:b1])
            zb = zh[:, puz_len:, :].float()
            clean_orth_steps[s, b0:b1] = (zb @ i_minus_p).to(torch.bfloat16).cpu()

    donor_t = torch.tensor(donor_indices, dtype=torch.long)

    conditions = [
        "clean",
        "reset_zL",
        "fiber_erased",            # reset_zL + H_init orth fill (registered P5)
        "zH_fiber_only_erased",    # keep z_L live + H_init orth fill
        "fiber_norm_restored",     # keep z_L live + project onto P_head & restore per-cell L2 norm
        "fiber_train_mean_filled", # keep z_L live + fill orth with step-s train mean orth
        "fiber_step1_frozen",      # keep z_L live + freeze orth at board's own step-1 orth
        "fiber_donor_swapped",     # keep z_L live + graft donor board's live step-s orth
    ] + [f"pca_k{k}" for k in pca_ks]

    results = {}
    prop_mask = np.array(prop_flags, dtype=bool)
    clean_board_exact: np.ndarray | None = None

    for cond in conditions:
        board_exact = np.zeros(n, dtype=bool)
        cell_acc_empty = np.zeros(n, dtype=np.float32)
        target_correct = np.zeros(n, dtype=bool) if target_indices is not None else None

        if cond.startswith("pca_k"):
            k_dim = int(cond.replace("pca_k", ""))
            qk = q_pca[:, :k_dim]  # [512, k]
            p_k = qk @ qk.T        # [512, 512]
        else:
            k_dim = -1
            p_k = None

        for b0 in range(0, n, bsz):
            b1 = min(n, b0 + bsz)
            zh, zl = model.init_states(b1 - b0)
            zl_init = zl.clone()
            xb = x_all[b0:b1]
            step1_orth = None

            for s in range(n_steps):
                zh, zl = model.act_step(zh, zl, xb)
                if s < n_steps - 1:
                    zb = zh[:, puz_len:, :].float()
                    zb_head = zb @ p_head_d

                    if cond == "clean":
                        pass
                    elif cond == "reset_zL":
                        zl = zl_init.clone()
                    elif cond == "fiber_erased":
                        zl = zl_init.clone()
                        zh = zh.clone()
                        zh[:, puz_len:, :] = (zb_head + h_init_orth).to(dt)
                    elif cond == "zH_fiber_only_erased":
                        zh = zh.clone()
                        zh[:, puz_len:, :] = (zb_head + h_init_orth).to(dt)
                    elif cond == "fiber_norm_restored":
                        zh = zh.clone()
                        orig_norm = zb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        head_norm = zb_head.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        zh[:, puz_len:, :] = (zb_head * (orig_norm / head_norm)).to(dt)
                    elif cond == "fiber_train_mean_filled":
                        zh = zh.clone()
                        zh[:, puz_len:, :] = (zb_head + step_mean_orth[s][None, :, :]).to(dt)
                    elif cond == "fiber_step1_frozen":
                        if s == 0:
                            step1_orth = zb @ i_minus_p
                        else:
                            zh = zh.clone()
                            zh[:, puz_len:, :] = (zb_head + step1_orth).to(dt)
                    elif cond == "fiber_donor_swapped":
                        zh = zh.clone()
                        d_orth = clean_orth_steps[s, donor_t[b0:b1]].to(device=dev, dtype=torch.float32)
                        zh[:, puz_len:, :] = (zb_head + d_orth).to(dt)
                    elif cond.startswith("pca_k"):
                        zh = zh.clone()
                        orth_c = (zb @ i_minus_p) - global_mean_orth
                        orth_rec = (orth_c @ p_k) + global_mean_orth
                        zh[:, puz_len:, :] = (zb_head + orth_rec).to(dt)

            preds = model.digit_logits(zh).argmax(dim=-1)  # [B, 81] in 0..8
            yb = ans_all[b0:b1]
            gm = giv_mask[b0:b1]
            merged = torch.where(gm, inp_all[b0:b1] - 2, preds)
            board_exact[b0:b1] = (merged == yb).all(dim=1).cpu().numpy()
            empty_m = (~gm)
            for bi in range(b1 - b0):
                em_i = empty_m[bi]
                cell_acc_empty[b0 + bi] = float((preds[bi, em_i] == yb[bi, em_i]).float().mean().item())
            if target_indices is not None:
                for bi in range(b1 - b0):
                    ti = target_indices[b0 + bi]
                    target_correct[b0 + bi] = bool((preds[bi, ti] == yb[bi, ti]).item())

        if cond == "clean":
            clean_board_exact = board_exact.copy()
        assert clean_board_exact is not None

        entry = {
            "n_all": int(n),
            "n_prop": int(prop_mask.sum()),
            "n_stalled": int((~prop_mask).sum()),
            "n_solved_all": int(board_exact.sum()),
            "n_solved_prop": int(board_exact[prop_mask].sum()),
            "n_solved_stalled": int(board_exact[~prop_mask].sum()),
            "board_exact_all": float(board_exact.mean()),
            "board_exact_prop": float(board_exact[prop_mask].mean()),
            "board_exact_stalled": float(board_exact[~prop_mask].mean()),
            "mcnemar_p_vs_clean_all": float(SC.exact_mcnemar_p(clean_board_exact.astype(int).tolist(), board_exact.astype(int).tolist())),
            "mcnemar_p_vs_clean_prop": float(SC.exact_mcnemar_p(clean_board_exact[prop_mask].astype(int).tolist(), board_exact[prop_mask].astype(int).tolist())),
            "mcnemar_p_vs_clean_stalled": float(SC.exact_mcnemar_p(clean_board_exact[~prop_mask].astype(int).tolist(), board_exact[~prop_mask].astype(int).tolist())),
            "empty_cell_acc_all": float(cell_acc_empty.mean()),
            "empty_cell_acc_prop": float(cell_acc_empty[prop_mask].mean()),
            "empty_cell_acc_stalled": float(cell_acc_empty[~prop_mask].mean()),
        }
        if target_correct is not None and strata is not None:
            entry["n_target_correct_all"] = int(target_correct.sum())
            entry["target_cell_all"] = float(target_correct.mean())
            entry["target_cell_wb_prop"] = float(target_correct[prop_mask].mean())
            entry["target_cell_wb_stalled"] = float(target_correct[~prop_mask].mean())
            entry["by_stratum"] = {}
            for st_name in ("S1_naked1", "S2_prop2", "S3_prop3plus", "S4_stalled"):
                sm = np.array([s == st_name for s in strata], dtype=bool)
                entry["by_stratum"][st_name] = {
                    "n": int(sm.sum()),
                    "n_target_correct": int(target_correct[sm].sum()),
                    "target_cell": float(target_correct[sm].mean()),
                    "n_board_exact": int(board_exact[sm].sum()),
                    "board_exact": float(board_exact[sm].mean()),
                    "empty_cell_acc": float(cell_acc_empty[sm].mean()),
                }
        results[cond] = entry

    return results


def main() -> None:
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running fiber_regime_controls on device={dev}...")

    # 1. Contrast case: LoopedChase
    t_lc = time.time()
    lc_out = evaluate_looped_chase_contrast(dev=dev)
    print(f"Completed LoopedChase contrast ({len(lc_out['q4_arms'])} Q4 + {len(lc_out['q5_arms'])} Q5 arms) in {time.time() - t_lc:.1f}s")

    # 2. Load Sudoku datasets
    bundle = json.loads((RESULTS_DIR / "benchmark_bundle.json").read_text(encoding="utf-8"))
    items = bundle["items"]
    id_to_idx = {it["item_id"]: idx for idx, it in enumerate(items)}
    items_qs = [it["question"] for it in items]
    items_ans = [it["answer"] for it in items]
    items_donors = [id_to_idx[it["donor_item_id"]] for it in items]
    items_targets = [int(it["target_idx"]) for it in items]
    items_strata = [it["stratum"] for it in items]
    items_prop_flags = [bool(it["prop_solved"]) for it in items]

    bench_qs, bench_ans, bench_prop_flags = load_bench100()
    # Deterministic within-regime donor pairing on bench100
    rng = np.random.default_rng(20261113)
    bench_donors = [0] * len(bench_qs)
    for flag_val in (True, False):
        idxs = [i for i, f in enumerate(bench_prop_flags) if f == flag_val]
        perm = rng.permutation(idxs)
        rolled = np.roll(perm, -1)
        for src_i, dst_i in zip(perm, rolled):
            bench_donors[int(src_i)] = int(dst_i)

    train_qs, train_ans = load_train_boards(200)

    # 3. Primary TRM (`gaoxin492/step_21700`)
    trm_main, main_sha = TS.load_trm_checkpoint("gaoxin492", device=dev, dtype=torch.bfloat16)
    _v_head, p_head = compute_head_projector(trm_main.lm_head.weight)
    pca_fit = fit_train_fiber_pca_and_probes(trm_main, p_head.to(dev), train_qs, train_ans)
    probe_bench = evaluate_probes_on_split(trm_main, p_head.to(dev), pca_fit["probes"], bench_qs, bench_ans, bench_prop_flags)
    probe_items = evaluate_probes_on_split(trm_main, p_head.to(dev), pca_fit["probes"], items_qs, items_ans, items_prop_flags)

    bench_suite = run_trm_fiber_suite_on_split(
        trm_main, p_head, pca_fit, bench_qs, bench_ans, bench_donors, None, None, bench_prop_flags
    )
    items_suite = run_trm_fiber_suite_on_split(
        trm_main, p_head, pca_fit, items_qs, items_ans, items_donors, items_targets, items_strata, items_prop_flags
    )
    del trm_main
    if dev == "cuda":
        torch.cuda.empty_cache()

    # 4. Replication TRM (`yagizdevre/trm-sudoku-extreme-att`) on bench100
    trm_rep, rep_sha = TS.load_trm_checkpoint("yagizdevre", device=dev, dtype=torch.bfloat16)
    _v_rep, p_head_rep = compute_head_projector(trm_rep.lm_head.weight)
    pca_fit_rep = fit_train_fiber_pca_and_probes(trm_rep, p_head_rep.to(dev), train_qs, train_ans)
    rep_bench_suite = run_trm_fiber_suite_on_split(
        trm_rep, p_head_rep, pca_fit_rep, bench_qs, bench_ans, bench_donors, None, None, bench_prop_flags,
        pca_ks=(0, 16, 64, 128, 501),
    )
    del trm_rep
    if dev == "cuda":
        torch.cuda.empty_cache()

    payload = {
        "elapsed_sec": float(time.time() - t0),
        "primary_checkpoint_sha256": main_sha,
        "replication_checkpoint_sha256": rep_sha,
        "head_rank": int(_v_head.shape[1]),
        "orth_fiber_dim": int(512 - _v_head.shape[1]),
        "pca_train_cumulative_variance": pca_fit["pca_cum_var"],
        "looped_chase_contrast": lc_out,
        "trm_primary_bench100": bench_suite,
        "trm_primary_items256": items_suite,
        "trm_replication_bench100": rep_bench_suite,
        "orth_fiber_linear_probes_bench100": probe_bench,
        "orth_fiber_linear_probes_items256": probe_items,
    }
    out_path = RESULTS_DIR / "fiber_regime_controls.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} in {payload['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
