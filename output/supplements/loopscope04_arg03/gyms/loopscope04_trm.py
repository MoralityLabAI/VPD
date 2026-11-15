#!/usr/bin/env python3
"""LOOPSCOPE-04 local gym: audited TRM sphere-fiber swaps, cache/history refill, and Gramian probes.

Uses the exact pinned PureSudokuTRM decoder from LOOPSCOPE-02 in float32.
Supports:
  - `swap`: sphere-calibrated 501-D board z_H orthogonal swaps (preserving all 11 lm_head logits,
    Post-RMSNorm cell radii, prefix tokens, and recipient z_L), plus separately labeled z_L-only
    and joint (z_H_orth + z_L) buffer ablations, recovery-time tracking, and limit-cycle detection.
  - `cache`: matched-width cellwise ridge baselines (snapshot vs. visible trace) AND optional
    capacity-matched full 97-token sequence-level Transformer refill models (C^now vs. C^trace)
    evaluated both by held-out R^2 and by causal single-transplant Resume / closed-loop rollouts.
  - `gramian`: sampled local dark-to-visible observability Gramian G_k plus finite-angle nonlinear
    causal validation comparing top-eigenvector, bottom-eigenvector, and random fiber directions.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from itertools import permutations
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import (
    detect_board_limit_cycle,
    project_visible,
    row_basis,
    sha256_file,
    trm_manifold_swap,
    write_json,
)
from .observability import causal_gramian_validation, causal_gramian_validation_batch, sampled_trm_gramian

S3_PERMS = tuple(permutations(range(3)))


def import_trm(root: Path):
    sys.path.insert(0, str(root.resolve()))
    from mechinterp.loopscope02.scripts import trm_sudoku as TS
    return TS


def import_sudoku_prop(root: Path):
    sys.path.insert(0, str(root.resolve()))
    from mechinterp.loopscope02.scripts import sudoku_prop as SP
    return SP


def canonical_digit_relabel(question: str) -> str:
    mapping: dict[str, str] = {}
    next_id = 1
    chars = []
    for c in question:
        if c in ".0":
            chars.append(".")
        else:
            if c not in mapping:
                mapping[c] = str(next_id)
                next_id += 1
            chars.append(mapping[c])
    return "".join(chars)


def _transpose_grid_str(q: str) -> str:
    return "".join(q[c * 9 + r] for r in range(9) for c in range(9))


def _directed_wl_hash(q: str, rounds: int = 3) -> str:
    g = [0 if ch in ".0" else int(ch) for ch in q]
    band_c = ["B"] * 3
    stack_c = ["S"] * 3
    row_c = [f"R:{sum(1 for c in range(9) if g[r * 9 + c] != 0)}" for r in range(9)]
    col_c = [f"C:{sum(1 for r in range(9) if g[r * 9 + c] != 0)}" for c in range(9)]
    dig_c = [f"D:{sum(1 for v in g if v == d)}" for d in range(1, 10)]

    def _h(obj: object) -> str:
        return sha256( repr(obj).encode("utf-8") ).hexdigest()[:16]

    for _ in range(rounds):
        new_row = [
            _h((
                row_c[r],
                band_c[r // 3],
                sorted(
                    (col_c[c], dig_c[g[r * 9 + c] - 1] if g[r * 9 + c] else "0")
                    for c in range(9)
                ),
            ))
            for r in range(9)
        ]
        new_col = [
            _h((
                col_c[c],
                stack_c[c // 3],
                sorted(
                    (row_c[r], dig_c[g[r * 9 + c] - 1] if g[r * 9 + c] else "0")
                    for r in range(9)
                ),
            ))
            for c in range(9)
        ]
        new_dig = [
            _h((
                dig_c[d - 1],
                sorted(
                    (row_c[r], col_c[c])
                    for r in range(9)
                    for c in range(9)
                    if g[r * 9 + c] == d
                ),
            ))
            for d in range(1, 10)
        ]
        new_band = [
            _h((band_c[b], sorted(new_row[3 * b : 3 * b + 3])))
            for b in range(3)
        ]
        new_stack = [
            _h((stack_c[s], sorted(new_col[3 * s : 3 * s + 3])))
            for s in range(3)
        ]
        row_c, col_c, dig_c, band_c, stack_c = new_row, new_col, new_dig, new_band, new_stack

    return sha256(
        repr((
            sorted(band_c),
            sorted(stack_c),
            sorted(row_c),
            sorted(col_c),
            sorted(dig_c),
        )).encode("utf-8")
    ).hexdigest()


def sudoku_wl_signature(question: str) -> str:
    """Full Sudoku symmetry-invariant Weisfeiler-Leman signature (bands, stacks, rows, cols, transpose, digits)."""
    q = "".join("." if c in ".0" else c for c in question)
    if len(q) != 81:
        raise ValueError("expected 81-char Sudoku puzzle")
    h1 = _directed_wl_hash(q)
    h2 = _directed_wl_hash(_transpose_grid_str(q))
    return h1 if h1 <= h2 else h2


def are_sudoku_isomorphic(q1: str, q2: str) -> bool:
    """Exact full-group Sudoku puzzle isomorphism test (2 * 6^8 grid symmetries x 9! digit permutations)."""
    a = "".join("." if c in ".0" else c for c in q1)
    b = "".join("." if c in ".0" else c for c in q2)
    if len(a) != 81 or len(b) != 81:
        raise ValueError("expected 81-char Sudoku strings")
    can_b = canonical_digit_relabel(b)
    if canonical_digit_relabel(a) == can_b:
        return True
    if a.count(".") != b.count("."):
        return False
    cnt_a = sorted(a.count(str(d)) for d in range(1, 10) if a.count(str(d)) > 0)
    cnt_b = sorted(b.count(str(d)) for d in range(1, 10) if b.count(str(d)) > 0)
    if cnt_a != cnt_b:
        return False
    if sudoku_wl_signature(a) != sudoku_wl_signature(b):
        return False

    b_row_cnts = [sum(1 for c in range(9) if b[r * 9 + c] != ".") for r in range(9)]
    b_col_cnts = [sum(1 for r in range(9) if b[r * 9 + c] != ".") for c in range(9)]
    b_band_multisets = [ tuple(sorted(b_row_cnts[3 * bi : 3 * bi + 3])) for bi in range(3) ]
    b_stack_multisets = [ tuple(sorted(b_col_cnts[3 * si : 3 * si + 3])) for si in range(3) ]

    for src in (a, _transpose_grid_str(a)):
        s_row_cnts = [sum(1 for c in range(9) if src[r * 9 + c] != ".") for r in range(9)]
        s_col_cnts = [sum(1 for r in range(9) if src[r * 9 + c] != ".") for c in range(9)]
        s_band_ms = [tuple(sorted(s_row_cnts[3 * bi : 3 * bi + 3])) for bi in range(3)]
        s_stack_ms = [tuple(sorted(s_col_cnts[3 * si : 3 * si + 3])) for si in range(3)]

        valid_col_maps: list[tuple[int, ...]] = []
        for sp in S3_PERMS:
            if any(s_stack_ms[sp[si]] != b_stack_multisets[si] for si in range(3)):
                continue
            stack_choices: list[list[tuple[int, int, int]]] = []
            ok = True
            for si in range(3):
                src_cols = [3 * sp[si] + j for j in range(3)]
                want = b_col_cnts[3 * si : 3 * si + 3]
                opts = [
                    tuple(src_cols[p[j]] for j in range(3))
                    for p in S3_PERMS
                    if [s_col_cnts[src_cols[p[j]]] for j in range(3)] == want
                ]
                if not opts:
                    ok = False
                    break
                stack_choices.append(opts)
            if ok:
                for c0 in stack_choices[0]:
                    for c1 in stack_choices[1]:
                        for c2 in stack_choices[2]:
                            valid_col_maps.append(c0 + c1 + c2)
        if not valid_col_maps:
            continue

        for bp in S3_PERMS:
            if any(s_band_ms[bp[bi]] != b_band_multisets[bi] for bi in range(3)):
                continue
            band_choices: list[list[tuple[int, int, int]]] = []
            ok = True
            for bi in range(3):
                src_rows = [3 * bp[bi] + j for j in range(3)]
                want = b_row_cnts[3 * bi : 3 * bi + 3]
                opts = [
                    tuple(src_rows[p[j]] for j in range(3))
                    for p in S3_PERMS
                    if [s_row_cnts[src_rows[p[j]]] for j in range(3)] == want
                ]
                if not opts:
                    ok = False
                    break
                band_choices.append(opts)
            if not ok:
                continue
            for r0 in band_choices[0]:
                for r1 in band_choices[1]:
                    for r2 in band_choices[2]:
                        rmap = r0 + r1 + r2
                        for cmap in valid_col_maps:
                            # Fast binary clue-mask check before digit relabeling
                            mask_match = True
                            for rr in range(9):
                                sr = rmap[rr] * 9
                                br = rr * 9
                                for cc in range(9):
                                    if (src[sr + cmap[cc]] == ".") != (b[br + cc] == "."):
                                        mask_match = False
                                        break
                                if not mask_match:
                                    break
                            if not mask_match:
                                continue
                            perm_q = "".join(src[rmap[rr] * 9 + cmap[cc]] for rr in range(9) for cc in range(9))
                            if canonical_digit_relabel(perm_q) == can_b:
                                return True
    return False


def _validate_qa_pair(path: Path, idx_label: int, q_raw: str | None, a_raw: str | None, extra: dict | None = None) -> dict:
    if not q_raw or not a_raw:
        raise ValueError(f"{path}:{idx_label}: expected question/answer columns")
    q = q_raw.replace("0", ".")
    a = a_raw
    if len(q) != 81 or len(a) != 81 or any(c not in ".123456789" for c in q) or any(c not in "123456789" for c in a):
        raise ValueError(f"{path}:{idx_label}: invalid 81-cell puzzle or answer")
    if any(q[i] not in (".", a[i]) for i in range(81)):
        raise ValueError("clue/answer mismatch")
    rec = {"question": q, "answer": a, "id": sha256(q.encode()).hexdigest()}
    if extra:
        for k in ("stratum", "prop_solved", "prop_rounds", "origin"):
            if k in extra:
                rec[k] = extra[k]
    return rec


def read_boards(path: Path, limit: int = 0) -> list[dict]:
    found: list[dict] = []
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected JSON list or dict with 'items'")
        for idx, row in enumerate(rows):
            q = row.get("question") or row.get("puzzle")
            a = row.get("answer") or row.get("solution")
            found.append(_validate_qa_pair(path, idx + 1, q, a, row))
            if limit and len(found) >= limit:
                break
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rd = csv.DictReader(handle)
            for idx, row in enumerate(rd):
                q = row.get("question") or row.get("puzzle")
                a = row.get("answer") or row.get("solution")
                found.append(_validate_qa_pair(path, idx + 2, q, a, row))
                if limit and len(found) >= limit:
                    break
    if not found:
        raise ValueError(f"empty dataset: {path}")
    return found


def _fast_grid_clue_profile(q: str) -> tuple:
    r_cnt = [sum(1 for c in range(9) if q[r * 9 + c] != ".") for r in range(9)]
    c_cnt = [sum(1 for r in range(9) if q[r * 9 + c] != ".") for c in range(9)]
    b_sig = tuple(sorted(tuple(sorted(r_cnt[3 * b : 3 * b + 3])) for b in range(3)))
    s_sig = tuple(sorted(tuple(sorted(c_cnt[3 * s : 3 * s + 3])) for s in range(3)))
    return (b_sig, s_sig)


def _fast_sudoku_invariant(question: str) -> tuple:
    """Necessary O(81) invariant under full Sudoku grid/transpose/digit automorphisms."""
    q = "".join("." if c in ".0" else c for c in question)
    dots = q.count(".")
    dig_mult = tuple(sorted(q.count(str(d)) for d in range(1, 10) if q.count(str(d)) > 0))
    p1 = _fast_grid_clue_profile(q)
    p2 = (p1[1], p1[0])  # transpose swaps bands and stacks
    return (dots, dig_mult, p1 if p1 <= p2 else p2)


def audit_boards(
    train: list[dict],
    test: list[dict],
    excluded: list[Path],
    *,
    check_full_automorphisms: bool = True,
) -> dict:
    tr = [r["question"] for r in train]
    te = [r["question"] for r in test]
    if len(set(tr)) != len(tr) or len(set(te)) != len(te):
        raise ValueError("duplicate puzzle in a split")
    can_tr = {canonical_digit_relabel(q) for q in tr}
    can_te = {canonical_digit_relabel(q) for q in te}
    if len(can_tr) != len(tr) or len(can_te) != len(te):
        raise ValueError("digit-relabel duplicate within a split")
    if can_tr & can_te:
        raise ValueError("digit-relabel overlap between train/eval")

    active_invariants = {_fast_sudoku_invariant(q) for q in tr + te} if check_full_automorphisms else set()

    banned_exact: set[str] = set()
    banned_relabel: set[str] = set()
    banned_by_wl: dict[str, list[str]] = {}
    for path in excluded:
        for r in read_boards(path):
            q = r["question"]
            banned_exact.add(q)
            banned_relabel.add(canonical_digit_relabel(q))
            if check_full_automorphisms and _fast_sudoku_invariant(q) in active_invariants:
                wl = sudoku_wl_signature(q)
                banned_by_wl.setdefault(wl, []).append(q)

    if any(q in banned_exact or canonical_digit_relabel(q) in banned_relabel for q in tr + te):
        raise ValueError("puzzle overlaps excluded prior experimental data")

    if check_full_automorphisms:
        seen_split_wl: dict[str, list[tuple[str, str]]] = {}
        for split_tag, seq in (("train", tr), ("test", te)):
            for q in seq:
                wl = sudoku_wl_signature(q)
                for prev_tag, prev_q in seen_split_wl.get(wl, ()):
                    if are_sudoku_isomorphic(q, prev_q):
                        if prev_tag == split_tag:
                            raise ValueError(f"Sudoku automorphism duplicate within {split_tag}")
                        raise ValueError("Sudoku automorphism overlap between train/eval")
                seen_split_wl.setdefault(wl, []).append((split_tag, q))
                for excl_q in banned_by_wl.get(wl, ()):
                    if are_sudoku_isomorphic(q, excl_q):
                        raise ValueError("puzzle overlaps excluded prior experimental data (Sudoku automorphism)")

    return {
        "train": len(tr),
        "test": len(te),
        "excluded_files": {str(p): sha256_file(p) for p in excluded},
        "audit_level": (
            "exact + digit-relabel + full Sudoku grid/transpose/digit automorphisms (WL + exact search)"
            if check_full_automorphisms
            else "exact plus digit-relabel"
        ),
        "train_sha256": sha256("\n".join(tr).encode()).hexdigest(),
        "test_sha256": sha256("\n".join(te).encode()).hexdigest(),
    }


def classify_sudoku_board(SP, question: str) -> dict:
    tr = SP.trace_propagation(question)
    if tr.solved:
        stratum = "WB-Prop"
    elif tr.num_rounds == 0:
        stratum = "S4_stalled"
    else:
        stratum = "WB-Stall_partial"
    return {
        "prop_solved": bool(tr.solved),
        "prop_rounds": int(tr.num_rounds),
        "wb_group": "WB-Prop" if tr.solved else "WB-Stall",
        "fine_stratum": stratum,
        "clues": int(sum(1 for c in question if c not in ".0")),
    }


def load_model(TS, checkpoint_file: Path, tag: str, device: str):
    expected = TS.TRM_CHECKPOINTS[tag]["sha256"]
    found = sha256_file(checkpoint_file)
    if found != expected:
        raise AssertionError(f"{tag}: checkpoint SHA mismatch {found} != {expected}")
    sd = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    model = TS.PureSudokuTRM.from_state_dict(sd, dtype=torch.float32).to(device).eval()
    return model, found


@torch.inference_mode()
def solve_logits(model, zh):
    return model.digit_logits(zh).argmax(dim=-1)[0].add(1).cpu().tolist()


def _lock_step_from_boards(boards: list[list[int]], answer: list[int], start_step: int) -> int | None:
    """Earliest step >= start_step from which all subsequent steps remain solved, or None."""
    if not boards or boards[-1] != answer:
        return None
    earliest = len(boards) - 1
    while earliest > 0 and boards[earliest - 1] == answer:
        earliest -= 1
    return start_step + earliest


@torch.inference_mode()
def run_swap_case(
    TS,
    model,
    row: dict,
    *,
    patch_step: int,
    horizon: int,
    seed: int,
    sigma: float,
    include_joint_ablations: bool = True,
):
    device = model.H_init.device
    x = model.encode_inputs(TS.encode_questions([row["question"]], device))
    zh_a, zl_a = model.init_states(1)
    zh_b, zl_b = model.init_states(1)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def perturb(z: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(tuple(z.shape), generator=g, dtype=torch.float32).to(device)
        return z + sigma * z.float().square().mean().sqrt() * noise

    zh_b, zl_b = perturb(zh_b), perturb(zl_b)
    for _ in range(patch_step):
        zh_a, zl_a = model.act_step(zh_a, zl_a, x)
        zh_b, zl_b = model.act_step(zh_b, zl_b, x)

    b_start = model.puzzle_emb_len
    h_swap, cert = trm_manifold_swap(zh_a, zh_b, model.lm_head.weight, board_start=b_start)
    g_rand = torch.Generator(device="cpu").manual_seed(seed + 9137)
    rand = torch.randn(tuple(zh_a.shape), generator=g_rand, dtype=torch.float32).to(device)
    h_random, cert_random = trm_manifold_swap(zh_a, rand, model.lm_head.weight, board_start=b_start)
    h_self, cert_self = trm_manifold_swap(zh_a, zh_a, model.lm_head.weight, board_start=b_start)

    states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "clean": (zh_a.clone(), zl_a.clone()),
        "swapped": (h_swap, zl_a.clone()),
        "random_orth": (h_random, zl_a.clone()),
        "self_control": (h_self, zl_a.clone()),
    }

    if include_joint_ablations:
        # Separately labeled z_L-only and joint (z_H_orth + z_L_board) ablations; prefix stays recipient-owned.
        zl_donor_board = zl_a.clone()
        zl_donor_board[:, b_start:, :] = zl_b[:, b_start:, :]
        rand_zl_raw = torch.randn(tuple(zl_a[:, b_start:, :].shape), generator=g_rand, dtype=torch.float32).to(device)
        rec_zl_norm = zl_a[:, b_start:, :].norm(dim=-1, keepdim=True)
        zl_rand_board = zl_a.clone()
        zl_rand_board[:, b_start:, :] = rand_zl_raw / rand_zl_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8) * rec_zl_norm

        states["zl_only_swap"] = (zh_a.clone(), zl_donor_board.clone())
        states["swapped_joint_zl"] = (h_swap.clone(), zl_donor_board.clone())
        states["random_joint_zl"] = (h_random.clone(), zl_rand_board.clone())

    # Crucial: at the *intervention instant* all 11 logits per board cell match clean across every readout-preserving arm.
    readouts = {
        name: model.lm_head(zh)[:, b_start:, :].float()
        for name, (zh, _zl) in states.items()
    }
    errors = {
        name: float((q - readouts["clean"]).abs().max().item())
        for name, q in readouts.items()
    }
    if max(errors.values()) > 1e-3:
        raise AssertionError(f"pre-intervention full-head mismatch: {errors}")

    # Also track donor's own continuation (NOT asserted equal to recipient at patch_step)
    donor_zh, donor_zl = zh_b.clone(), zl_b.clone()
    donor_pred_at_patch = solve_logits(model, donor_zh)
    rec_pred_at_patch = solve_logits(model, zh_a)

    # Batch all arms + donor into a single GPU forward loop for exact deterministic speed
    arm_names = list(states.keys())
    zh_batch = torch.cat([states[k][0] for k in arm_names] + [donor_zh], dim=0)
    zl_batch = torch.cat([states[k][1] for k in arm_names] + [donor_zl], dim=0)
    x_batch = x.expand(zh_batch.shape[0], -1, -1)

    trajectories: dict[str, list[list[int]]] = {k: [] for k in arm_names}
    donor_traj: list[list[int]] = []
    first_step_l2: dict[str, float] = {}
    donor_align_cos: dict[str, float] = {}
    s16_boards: dict[str, list[int]] = {}

    for step in range(patch_step + 1, horizon + 1):
        zh_batch, zl_batch = model.act_step(zh_batch, zl_batch, x_batch)
        d_logits = model.digit_logits(zh_batch)  # [B, 81, 9]
        preds = d_logits.argmax(dim=-1).add(1).cpu().tolist()
        for idx_k, k in enumerate(arm_names):
            trajectories[k].append(preds[idx_k])
            if step == 16:
                s16_boards[k] = preds[idx_k]
        donor_traj.append(preds[-1])
        if step == patch_step + 1:
            ref_l = d_logits[0]  # clean
            don_shift = (d_logits[-1] - ref_l).reshape(-1)
            don_norm = float(don_shift.norm().item())
            for idx_k, k in enumerate(arm_names):
                arm_shift = (d_logits[idx_k] - ref_l).reshape(-1)
                a_norm = float(arm_shift.norm().item())
                first_step_l2[k] = a_norm
                if a_norm > 1e-8 and don_norm > 1e-8:
                    donor_align_cos[k] = float(torch.dot(arm_shift, don_shift).item() / (a_norm * don_norm))
                else:
                    donor_align_cos[k] = 0.0

    answer = [int(c) for c in row["answer"]]
    clean_seq = trajectories["clean"]
    result = {}
    for k in arm_names:
        seq = trajectories[k]
        pred_end = seq[-1]
        pred_16 = s16_boards.get(k, pred_end)
        lock_s = _lock_step_from_boards(seq, answer, patch_step + 1)
        ever_s = any(b == answer for b in seq)
        rec_steps = next((idx + 1 for idx, (ba, bc) in enumerate(zip(seq, clean_seq)) if ba == bc), None)
        cyc = detect_board_limit_cycle(seq)
        result[k] = {
            "exact": bool(pred_end == answer),
            "exact_s16": bool(pred_16 == answer),
            "ever_solved": bool(ever_s),
            "lock_step": lock_s,
            "never_lock": bool(lock_s is None),
            "recovery_steps_to_clean": rec_steps,
            "correct_cells": int(sum(a == b for a, b in zip(pred_end, answer))),
            "first_successor_logit_l2": first_step_l2[k],
            "first_successor_cos_to_donor": donor_align_cos[k],
            "pre_intervention_full_head_max_error": errors[k],
            "oscillatory_limit_cycle": cyc["oscillatory_cycle"],
            "terminal_period": cyc["terminal_period"],
            "tail_churn_steps": cyc["tail_churn_steps"],
        }

    return {
        "case_id": row["id"],
        "patch_step": patch_step,
        "horizon": horizon,
        "sigma": sigma,
        "board_start": b_start,
        "head_rank": cert["head_rank"],
        "orthogonal_dim": cert["orthogonal_dim"],
        "fork_disagreement_at_patch_cells": int(sum(a != b for a, b in zip(rec_pred_at_patch, donor_pred_at_patch))),
        "donor_clean_exact_end": bool(donor_traj[-1] == answer),
        "certificates": {**cert, "random": cert_random, "self": cert_self},
        "arms": result,
    }


@torch.inference_mode()
def run_swap_batch(
    TS,
    model,
    rows: list[dict],
    *,
    patch_step: int,
    horizon: int,
    base_seed: int,
    sigma: float,
    chunk_size: int = 32,
    include_joint_ablations: bool = True,
) -> list[dict]:
    """Batched GPU swap rollout preserving exact per-board deterministic seeds."""
    device = model.H_init.device
    b_start = model.puzzle_emb_len
    all_out: list[dict] = []

    for c0 in range(0, len(rows), chunk_size):
        sub = rows[c0 : c0 + chunk_size]
        m = len(sub)
        x_m = model.encode_inputs(TS.encode_questions([r["question"] for r in sub], device))
        zh_a_list, zl_a_list, zh_b_list, zl_b_list, seeds = [], [], [], [], []
        for r in sub:
            s_i = base_seed ^ int(r["id"][:8], 16)
            seeds.append(s_i)
            zha, zla = model.init_states(1)
            zhb, zlb = model.init_states(1)
            g = torch.Generator(device="cpu").manual_seed(s_i)

            def _pert(z: torch.Tensor) -> torch.Tensor:
                nz = torch.randn(tuple(z.shape), generator=g, dtype=torch.float32).to(device)
                return z + sigma * z.float().square().mean().sqrt() * nz

            zh_a_list.append(zha)
            zl_a_list.append(zla)
            zh_b_list.append(_pert(zhb))
            zl_b_list.append(_pert(zlb))

        # Batched pre-patch rollout: [2*m, 97, 512]
        zh_pre = torch.cat(zh_a_list + zh_b_list, dim=0)
        zl_pre = torch.cat(zl_a_list + zl_b_list, dim=0)
        x_pre = torch.cat([x_m, x_m], dim=0)
        for _ in range(patch_step):
            zh_pre, zl_pre = model.act_step(zh_pre, zl_pre, x_pre)

        zh_a_m, zh_b_m = zh_pre[:m], zh_pre[m:]
        zl_a_m, zl_b_m = zl_pre[:m], zl_pre[m:]
        rec_preds_patch = model.digit_logits(zh_a_m).argmax(dim=-1).add(1).cpu().tolist()
        don_preds_patch = model.digit_logits(zh_b_m).argmax(dim=-1).add(1).cpu().tolist()

        arm_names = ["clean", "swapped", "random_orth", "self_control"]
        if include_joint_ablations:
            arm_names.extend(["zl_only_swap", "swapped_joint_zl", "random_joint_zl"])
        n_arms = len(arm_names) + 1  # + donor

        zh_cat, zl_cat, x_cat = [], [], []
        per_board_meta = []
        for bi in range(m):
            zha_i = zh_a_m[bi : bi + 1]
            zla_i = zl_a_m[bi : bi + 1]
            zhb_i = zh_b_m[bi : bi + 1]
            zlb_i = zl_b_m[bi : bi + 1]
            s_i = seeds[bi]

            h_swap, cert = trm_manifold_swap(zha_i, zhb_i, model.lm_head.weight, board_start=b_start)
            g_rand = torch.Generator(device="cpu").manual_seed(s_i + 9137)
            rand = torch.randn(tuple(zha_i.shape), generator=g_rand, dtype=torch.float32).to(device)
            h_random, cert_random = trm_manifold_swap(zha_i, rand, model.lm_head.weight, board_start=b_start)
            h_self, cert_self = trm_manifold_swap(zha_i, zha_i, model.lm_head.weight, board_start=b_start)

            st_i: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
                "clean": (zha_i.clone(), zla_i.clone()),
                "swapped": (h_swap, zla_i.clone()),
                "random_orth": (h_random, zla_i.clone()),
                "self_control": (h_self, zla_i.clone()),
            }
            if include_joint_ablations:
                zl_don_b = zla_i.clone()
                zl_don_b[:, b_start:, :] = zlb_i[:, b_start:, :]
                rand_zl_raw = torch.randn(tuple(zla_i[:, b_start:, :].shape), generator=g_rand, dtype=torch.float32).to(device)
                rec_zl_norm = zla_i[:, b_start:, :].norm(dim=-1, keepdim=True)
                zl_rand_b = zla_i.clone()
                zl_rand_b[:, b_start:, :] = rand_zl_raw / rand_zl_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8) * rec_zl_norm
                st_i["zl_only_swap"] = (zha_i.clone(), zl_don_b.clone())
                st_i["swapped_joint_zl"] = (h_swap.clone(), zl_don_b.clone())
                st_i["random_joint_zl"] = (h_random.clone(), zl_rand_b.clone())

            ro_i = {k: model.lm_head(st_i[k][0])[:, b_start:, :].float() for k in arm_names}
            errs_i = {k: float((ro_i[k] - ro_i["clean"]).abs().max().item()) for k in arm_names}
            if max(errs_i.values()) > 1e-3:
                raise AssertionError(f"pre-intervention full-head mismatch: {errs_i}")

            for k in arm_names:
                zh_cat.append(st_i[k][0])
                zl_cat.append(st_i[k][1])
            zh_cat.append(zhb_i.clone())
            zl_cat.append(zlb_i.clone())
            x_cat.append(x_m[bi : bi + 1].expand(n_arms, -1, -1))
            per_board_meta.append((cert, cert_random, cert_self, errs_i))

        zh_post = torch.cat(zh_cat, dim=0)
        zl_post = torch.cat(zl_cat, dim=0)
        x_post = torch.cat(x_cat, dim=0)

        trajs = [{k: [] for k in arm_names} for _ in range(m)]
        don_trajs: list[list[list[int]]] = [[] for _ in range(m)]
        first_l2 = [{} for _ in range(m)]
        don_cos = [{} for _ in range(m)]
        s16_b = [{} for _ in range(m)]

        for step in range(patch_step + 1, horizon + 1):
            zh_post, zl_post = model.act_step(zh_post, zl_post, x_post)
            dl_all = model.digit_logits(zh_post).view(m, n_arms, 81, 9)
            pr_all = dl_all.argmax(dim=-1).add(1).cpu().tolist()
            for bi in range(m):
                for idx_k, k in enumerate(arm_names):
                    trajs[bi][k].append(pr_all[bi][idx_k])
                    if step == 16:
                        s16_b[bi][k] = pr_all[bi][idx_k]
                don_trajs[bi].append(pr_all[bi][-1])
                if step == patch_step + 1:
                    ref_l = dl_all[bi, 0]
                    don_shift = (dl_all[bi, -1] - ref_l).reshape(-1)
                    don_norm = float(don_shift.norm().item())
                    for idx_k, k in enumerate(arm_names):
                        arm_shift = (dl_all[bi, idx_k] - ref_l).reshape(-1)
                        a_norm = float(arm_shift.norm().item())
                        first_l2[bi][k] = a_norm
                        if a_norm > 1e-8 and don_norm > 1e-8:
                            don_cos[bi][k] = float(torch.dot(arm_shift, don_shift).item() / (a_norm * don_norm))
                        else:
                            don_cos[bi][k] = 0.0

        for bi, r in enumerate(sub):
            ans = [int(c) for c in r["answer"]]
            clean_seq = trajs[bi]["clean"]
            cert, cert_random, cert_self, errs_i = per_board_meta[bi]
            res_arms = {}
            for k in arm_names:
                seq = trajs[bi][k]
                pred_end = seq[-1]
                pred_16 = s16_b[bi].get(k, pred_end)
                lock_s = _lock_step_from_boards(seq, ans, patch_step + 1)
                ever_s = any(bd == ans for bd in seq)
                rec_steps = next((idx + 1 for idx, (ba, bc) in enumerate(zip(seq, clean_seq)) if ba == bc), None)
                cyc = detect_board_limit_cycle(seq)
                res_arms[k] = {
                    "exact": bool(pred_end == ans),
                    "exact_s16": bool(pred_16 == ans),
                    "ever_solved": bool(ever_s),
                    "lock_step": lock_s,
                    "never_lock": bool(lock_s is None),
                    "recovery_steps_to_clean": rec_steps,
                    "correct_cells": int(sum(a == b for a, b in zip(pred_end, ans))),
                    "first_successor_logit_l2": first_l2[bi][k],
                    "first_successor_cos_to_donor": don_cos[bi][k],
                    "pre_intervention_full_head_max_error": errs_i[k],
                    "oscillatory_limit_cycle": cyc["oscillatory_cycle"],
                    "terminal_period": cyc["terminal_period"],
                    "tail_churn_steps": cyc["tail_churn_steps"],
                }
            all_out.append({
                "case_id": r["id"],
                "patch_step": patch_step,
                "horizon": horizon,
                "sigma": sigma,
                "board_start": b_start,
                "head_rank": cert["head_rank"],
                "orthogonal_dim": cert["orthogonal_dim"],
                "fork_disagreement_at_patch_cells": int(
                    sum(a != b for a, b in zip(rec_preds_patch[bi], don_preds_patch[bi]))
                ),
                "donor_clean_exact_end": bool(don_trajs[bi][-1] == ans),
                "certificates": {**cert, "random": cert_random, "self": cert_self},
                "arms": res_arms,
            })
    return all_out


@torch.inference_mode()
def collect_cache_rows(TS, model, boards: list[dict], n_steps: int, basis: torch.Tensor, batch_size: int = 64):
    """Construct matched-width snapshot/trace features and full board-cell hidden targets.

    Diagnostic target = [zH_board orthogonal, zL_board] (2*512 dimensions).
    """
    X_now, X_trace, X_ema, Y = [], [], [], []
    device = model.H_init.device
    eye10 = np.eye(10, dtype=np.float32)
    eye9 = np.eye(9, dtype=np.float32)
    row_coords = eye9[np.arange(81) // 9]
    col_coords = eye9[np.arange(81) % 9]

    for i0 in range(0, len(boards), batch_size):
        sub = boards[i0 : i0 + batch_size]
        m = len(sub)
        xb = model.encode_inputs(TS.encode_questions([r["question"] for r in sub], device))
        zh, zl = model.init_states(m)
        v_prev = (zh[:, model.puzzle_emb_len :, :].float() @ basis).cpu().numpy()  # [m, 81, 11]
        v_ema = v_prev.copy()
        bases = []
        for r in sub:
            clue = np.array([0 if c == "." else int(c) for c in r["question"]], dtype=np.int64)
            bases.append(np.concatenate([eye10[clue], row_coords, col_coords], axis=1))

        # Collect per-board across steps to preserve exact canonical row ordering
        bn_b = [[] for _ in range(m)]
        bt_b = [[] for _ in range(m)]
        be_b = [[] for _ in range(m)]
        by_b = [[] for _ in range(m)]

        for t in range(1, n_steps + 1):
            zh, zl = model.act_step(zh, zl, xb)
            board = zh[:, model.puzzle_emb_len :, :].float()
            v_t = (board @ basis).cpu().numpy()
            orth = (board - project_visible(board, basis)).cpu().numpy()
            zl_board = zl[:, model.puzzle_emb_len :, :].float().cpu().numpy()
            step = np.full((81, 1), t / n_steps, dtype=np.float32)
            for bi in range(m):
                prefix = np.concatenate([bases[bi], step, v_t[bi]], axis=1)
                bn_b[bi].append(np.concatenate([prefix, v_t[bi]], axis=1))
                bt_b[bi].append(np.concatenate([prefix, v_prev[bi]], axis=1))
                be_b[bi].append(np.concatenate([prefix, v_ema[bi]], axis=1))
                by_b[bi].append(np.concatenate([orth[bi], zl_board[bi]], axis=1))
            v_ema = 0.5 * v_ema + 0.5 * v_t
            v_prev = v_t

        for bi in range(m):
            X_now.extend(bn_b[bi])
            X_trace.extend(bt_b[bi])
            X_ema.extend(be_b[bi])
            Y.extend(by_b[bi])
    return np.concatenate(X_now), np.concatenate(X_trace), np.concatenate(X_ema), np.concatenate(Y)


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    xb = np.column_stack([np.ones((len(x64), 1)), x64])
    penalty = np.eye(xb.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(xb.T @ xb + penalty, xb.T @ y64)


def ridge_predict(x, w):
    return np.column_stack([np.ones((len(x), 1)), x.astype(np.float64)]) @ w


@torch.inference_mode()
def cache_compare(TS, model, train, test, steps, alpha):
    basis = row_basis(model.lm_head.weight).to(model.H_init.device)
    xn, xt, xe, yt = collect_cache_rows(TS, model, train, steps, basis)
    vn, vt, ve, yv = collect_cache_rows(TS, model, test, steps, basis)
    m_now = ridge_fit(xn, yt, alpha)
    m_trace = ridge_fit(xt, yt, alpha)
    m_ema = ridge_fit(xe, yt, alpha)
    den = max(float(((yv - yv.mean(0)) ** 2).sum()), 1e-12)
    den_zh = max(float(((yv[:, :512] - yv[:, :512].mean(0)) ** 2).sum()), 1e-12)
    den_zl = max(float(((yv[:, 512:] - yv[:, 512:].mean(0)) ** 2).sum()), 1e-12)
    result = {}
    for name, x, w in (
        ("snapshot", vn, m_now),
        ("visible_trace", vt, m_trace),
        ("visible_trace_ema", ve, m_ema),
    ):
        pred = ridge_predict(x, w)
        err = yv - pred
        mse = float((err ** 2).mean())
        r2 = 1.0 - float((err ** 2).sum()) / den
        r2_zh = 1.0 - float((err[:, :512] ** 2).sum()) / den_zh
        r2_zl = 1.0 - float((err[:, 512:] ** 2).sum()) / den_zl
        result[name] = {
            "mse": mse,
            "r2": r2,
            "r2_zh_orth": r2_zh,
            "r2_zl": r2_zl,
            "feature_width": int(x.shape[1]),
            "parameters": int(w.size),
            "training_cells": int(len(xn)),
            "validation_cells": int(len(vn)),
        }
    return result


class SeqRefill97(nn.Module):
    """Capacity-matched 97-token sequence reconstructor for C^now(x,t,V_t) vs C^trace(x,t,V_{0:t}).

    Both models use identical input dimension (46 per token), identical 2-layer Transformer encoder
    architecture (`d_model=256, nhead=8`), and identical parameter count, predicting [zH_97, zL_97].
    """

    def __init__(self, in_dim: int = 46, d_model: int = 256, nhead: int = 8, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 97, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x) + self.pos_emb
        h = self.encoder(h)
        return self.out_proj(self.out_norm(h))


def _build_seq97_features(
    clue_oh_97: torch.Tensor,  # [B, 97, 11] (10 clue classes + is_prefix flag)
    step_val: float,
    max_steps: int,
    vt_board: torch.Tensor,    # [B, 81, 11]
    vprev_board: torch.Tensor, # [B, 81, 11]
    vema_board: torch.Tensor,  # [B, 81, 11]
    *,
    is_trace: bool,
) -> torch.Tensor:
    b = clue_oh_97.shape[0]
    dev = clue_oh_97.device
    s_norm = step_val / max(float(max_steps), 1.0)
    step_feat = torch.tensor([s_norm, math.log1p(step_val) / math.log1p(max_steps)], device=dev, dtype=torch.float32)
    step_97 = step_feat.view(1, 1, 2).expand(b, 97, 2)
    pad16 = torch.zeros((b, 16, 11), device=dev, dtype=torch.float32)
    v_cur_97 = torch.cat([pad16, vt_board], dim=1)
    if is_trace:
        v_b1 = torch.cat([pad16, vprev_board], dim=1)
        v_b2 = torch.cat([pad16, vema_board], dim=1)
    else:
        # Identical input width & scaling with zero history information
        v_b1 = v_cur_97
        v_b2 = v_cur_97
    return torch.cat([clue_oh_97, step_97, v_cur_97, v_b1, v_b2], dim=-1)  # [B, 97, 46]


def _encode_clue_oh_97(questions: list[str], device: torch.device) -> torch.Tensor:
    b = len(questions)
    out = torch.zeros((b, 97, 11), device=device, dtype=torch.float32)
    out[:, :16, 10] = 1.0  # prefix indicator
    for bi, q in enumerate(questions):
        for ci, ch in enumerate(q):
            d = 0 if ch in ".0" else int(ch)
            out[bi, 16 + ci, d] = 1.0
    return out


@torch.no_grad()
def _collect_seq97_dataset(TS, model, boards: list[dict], n_steps: int, basis: torch.Tensor, batch_size: int = 64):
    """Collect sequence dataset ordered by [board, step] so board-level splits never leak."""
    dev = model.H_init.device
    qs = [r["question"] for r in boards]
    x_now_list, x_tr_list, y_list = [], [], []
    for i0 in range(0, len(qs), batch_size):
        q_sub = qs[i0 : i0 + batch_size]
        bsz = len(q_sub)
        clue_oh = _encode_clue_oh_97(q_sub, dev)
        xb = model.encode_inputs(TS.encode_questions(q_sub, dev))
        zh, zl = model.init_states(bsz)
        v_prev = zh[:, 16:, :].float() @ basis
        v_ema = v_prev.clone()
        bn_steps, bt_steps, by_steps = [], [], []
        for t in range(1, n_steps + 1):
            zh, zl = model.act_step(zh, zl, xb)
            board = zh[:, 16:, :].float()
            vt = board @ basis
            orth_board = board - project_visible(board, basis)
            zh_target = torch.cat([zh[:, :16, :].float(), orth_board], dim=1)
            target = torch.cat([zh_target, zl.float()], dim=-1)  # [B, 97, 1024]
            fn_now = _build_seq97_features(clue_oh, t, n_steps, vt, v_prev, v_ema, is_trace=False)
            fn_tr = _build_seq97_features(clue_oh, t, n_steps, vt, v_prev, v_ema, is_trace=True)
            bn_steps.append(fn_now.cpu())
            bt_steps.append(fn_tr.cpu())
            by_steps.append(target.cpu())
            v_ema = 0.5 * v_ema + 0.5 * vt
            v_prev = vt
        # [bsz, n_steps, 97, D] -> [bsz * n_steps, 97, D]
        x_now_list.append(torch.stack(bn_steps, dim=1).reshape(bsz * n_steps, 97, -1))
        x_tr_list.append(torch.stack(bt_steps, dim=1).reshape(bsz * n_steps, 97, -1))
        y_list.append(torch.stack(by_steps, dim=1).reshape(bsz * n_steps, 97, -1))
    return torch.cat(x_now_list, 0), torch.cat(x_tr_list, 0), torch.cat(y_list, 0)


def _train_seq97_model(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    n_steps: int,
    device: torch.device,
    seed: int = 20261115,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1.5e-3,
) -> tuple[SeqRefill97, dict]:
    n_boards = x_train.shape[0] // n_steps
    val_boards = max(2, min(n_boards // 10, 36)) if n_boards >= 20 else 0
    split_idx = (n_boards - val_boards) * n_steps
    x_fit, y_fit = x_train[:split_idx], y_train[:split_idx]
    x_iv, y_iv = (x_train[split_idx:], y_train[split_idx:]) if val_boards > 0 else (None, None)
    eff_bsz = min(batch_size, max(8, x_fit.shape[0] // 20))

    with torch.enable_grad():
        torch.manual_seed(seed)
        net = SeqRefill97(in_dim=x_train.shape[-1], d_model=256, nhead=8, num_layers=2, dropout=0.0).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        gen = torch.Generator(device="cpu").manual_seed(seed + 101)
        n = x_fit.shape[0]
        t0 = time.perf_counter()
        final_loss = 0.0
        best_val = float("inf")
        best_ep = epochs
        best_sd = None

        for ep in range(1, epochs + 1):
            net.train()
            perm = torch.randperm(n, generator=gen)
            ep_loss = 0.0
            steps_cnt = 0
            for i0 in range(0, n, eff_bsz):
                idx = perm[i0 : i0 + eff_bsz]
                xb = x_fit[idx].to(device)
                yb = y_fit[idx].to(device)
                pred = net(xb)
                loss = F.mse_loss(pred, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())
                steps_cnt += 1
            sched.step()
            final_loss = ep_loss / max(steps_cnt, 1)

            if x_iv is not None and y_iv is not None:
                net.eval()
                with torch.no_grad():
                    v_losses = []
                    for j0 in range(0, x_iv.shape[0], 64):
                        vp = net(x_iv[j0 : j0 + 64].to(device))
                        v_losses.append(float(F.mse_loss(vp, y_iv[j0 : j0 + 64].to(device)).item()))
                    v_mse = float(np.mean(v_losses))
                if v_mse < best_val:
                    best_val = v_mse
                    best_ep = ep
                    best_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        if best_sd is not None:
            net.load_state_dict({k: v.to(device) for k, v in best_sd.items()})
        net.eval()
    return net, {
        "n_parameters": int(sum(p.numel() for p in net.parameters())),
        "epochs": epochs,
        "selected_epoch": int(best_ep),
        "best_internal_val_mse": None if best_val == float("inf") else best_val,
        "batch_size": eff_bsz,
        "final_train_mse": final_loss,
        "train_seconds": round(time.perf_counter() - t0, 3),
    }


@torch.no_grad()
def _eval_seq97_r2(net: SeqRefill97, x_val: torch.Tensor, y_val: torch.Tensor, device: torch.device) -> dict:
    if x_val.shape[0] == 0:
        return {"n_snapshots": 0}
    preds = []
    for i0 in range(0, x_val.shape[0], 64):
        preds.append(net(x_val[i0 : i0 + 64].to(device)).cpu())
    p = torch.cat(preds, dim=0)
    err = y_val - p

    def _r2(y_slice: torch.Tensor, e_slice: torch.Tensor) -> float:
        den = float((y_slice - y_slice.mean(dim=(0, 1), keepdim=True)).square().sum().item())
        num = float(e_slice.square().sum().item())
        return 1.0 - num / max(den, 1e-12)

    return {
        "n_snapshots": int(x_val.shape[0]),
        "mse_overall": float(err.square().mean().item()),
        "r2_overall_97": _r2(y_val, err),
        "r2_prefix_16": _r2(y_val[:, :16, :], err[:, :16, :]),
        "r2_board_zh_orth": _r2(y_val[:, 16:, :512], err[:, 16:, :512]),
        "r2_board_zl": _r2(y_val[:, 16:, 512:], err[:, 16:, 512:]),
    }


@torch.no_grad()
def _apply_seq97_prediction_sphere(
    zh_clean: torch.Tensor,
    zl_clean: torch.Tensor,
    pred_97: torch.Tensor,
    head_weight: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Build Post-RMSNorm manifold-calibrated (zh_hat, zl_hat) preserving exact board lm_head logits and all 97 token L2 radii."""
    zh_pref_raw = pred_97[:, :16, :512].float()
    zh_board_raw = pred_97[:, 16:, :512].float()
    zl_raw = pred_97[:, :, 512:].float()

    pref_radius = zh_clean[:, :16, :].float().norm(dim=-1, keepdim=True)
    zh_pref = zh_pref_raw / zh_pref_raw.norm(dim=-1, keepdim=True).clamp_min(1e-7) * pref_radius

    rec_board = zh_clean[:, 16:, :].float()
    rv = project_visible(rec_board, basis)
    du = zh_board_raw - project_visible(zh_board_raw, basis)
    radius2 = (rec_board.square().sum(-1, keepdim=True) - rv.square().sum(-1, keepdim=True)).clamp_min(0.0)
    du_norm = du.norm(dim=-1, keepdim=True).clamp_min(1e-7)
    board_hat = rv + du / du_norm * radius2.sqrt()

    zh_hat = torch.cat([zh_pref, board_hat], dim=1).to(zh_clean.dtype)
    zl_radius = zl_clean.float().norm(dim=-1, keepdim=True)
    zl_hat = (zl_raw / zl_raw.norm(dim=-1, keepdim=True).clamp_min(1e-7) * zl_radius).to(zl_clean.dtype)

    w = head_weight.float().to(zh_clean.device)
    max_logit_err = float(((board_hat @ w.T) - (rec_board @ w.T)).abs().max().item())
    if max_logit_err > 1e-3:
        raise AssertionError(f"seq97 sphere projection logit leak: {max_logit_err}")
    return zh_hat, zl_hat, max_logit_err


@torch.no_grad()
def full_97_cache_and_causal_eval(
    TS,
    SP,
    model,
    train_boards: list[dict],
    eval_boards: list[dict],
    *,
    train_steps: int = 16,
    resume_steps: tuple[int, ...] = (2, 4, 8),
    horizon: int = 32,
    seed: int = 20261115,
) -> dict:
    """Train matched 97-token C^now and C^trace reconstructors and separate R^2 from causal continuation."""
    dev = model.H_init.device
    basis = row_basis(model.lm_head.weight).to(dev)
    xn_tr, xt_tr, y_tr = _collect_seq97_dataset(TS, model, train_boards, train_steps, basis)
    xn_ev, xt_ev, y_ev = _collect_seq97_dataset(TS, model, eval_boards, train_steps, basis)

    net_now, fit_now = _train_seq97_model(xn_tr, y_tr, n_steps=train_steps, device=dev, seed=seed)
    net_tr, fit_tr = _train_seq97_model(xt_tr, y_tr, n_steps=train_steps, device=dev, seed=seed)

    qs = [r["question"] for r in eval_boards]
    ans_all = [[int(c) for c in r["answer"]] for r in eval_boards]
    strata = [classify_sudoku_board(SP, q)["wb_group"] for q in qs]
    n_eval = len(qs)

    r2_now = _eval_seq97_r2(net_now, xn_ev, y_ev, dev)
    r2_tr = _eval_seq97_r2(net_tr, xt_ev, y_ev, dev)

    def _subset_snap_idx(board_indices: list[int]) -> torch.Tensor:
        rows = [bi * train_steps + s for bi in board_indices for s in range(train_steps)]
        return torch.tensor(rows, dtype=torch.long)

    idx_all = list(range(n_eval))
    idx_prop = [i for i, g in enumerate(strata) if g == "WB-Prop"]
    idx_stall = [i for i, g in enumerate(strata) if g == "WB-Stall"]

    stratified_r2 = {}
    for s_name, b_idx in (("wb_prop", idx_prop), ("wb_stall", idx_stall)):
        if b_idx:
            si = _subset_snap_idx(b_idx)
            rn = _eval_seq97_r2(net_now, xn_ev[si], y_ev[si], dev)
            rt = _eval_seq97_r2(net_tr, xt_ev[si], y_ev[si], dev)
            stratified_r2[s_name] = {
                "c_now_97": rn,
                "c_trace_97": rt,
                "delta_r2_trace_minus_now": {k: float(rt[k] - rn[k]) for k in rn if k.startswith("r2_")},
            }
        else:
            stratified_r2[s_name] = {"n_snapshots": 0}

    # Per-step training mean [train_steps, 97, 1024] for step-matched V_mean_sphere baseline
    y_tr_by_step = y_tr.view(-1, train_steps, 97, 1024).mean(dim=0).to(dev)  # [train_steps, 97, 1024]

    bsz_eval = 32
    clean_solve_16 = [False] * n_eval
    clean_solve_h = [False] * n_eval
    clean_lock: list[int | None] = [None] * n_eval

    closed_loop_arms = ("c_now_97", "c_trace_97", "v_mean_sphere")
    cl_solve_16 = {a: [False] * n_eval for a in closed_loop_arms}
    cl_solve_h = {a: [False] * n_eval for a in closed_loop_arms}
    cl_ever_h = {a: [False] * n_eval for a in closed_loop_arms}
    cl_cyc_h = {a: [False] * n_eval for a in closed_loop_arms}

    cl_m2_solve_16 = {a: [False] * n_eval for a in closed_loop_arms}
    cl_m2_solve_h = {a: [False] * n_eval for a in closed_loop_arms}
    cl_m2_ever_h = {a: [False] * n_eval for a in closed_loop_arms}
    cl_m2_cyc_h = {a: [False] * n_eval for a in closed_loop_arms}

    resume_data = {
        t: {
            a: {"solve_h": [False] * n_eval, "lock": [None] * n_eval, "step1_logit_l2": [0.0] * n_eval, "pre_logit_err": 0.0}
            for a in ("c_now_97", "c_trace_97", "v_mean_sphere")
        }
        for t in resume_steps
    }

    for i0 in range(0, n_eval, bsz_eval):
        i1 = min(n_eval, i0 + bsz_eval)
        b = i1 - i0
        q_sub = qs[i0:i1]
        ans_sub = ans_all[i0:i1]
        clue_oh = _encode_clue_oh_97(q_sub, dev)
        xb = model.encode_inputs(TS.encode_questions(q_sub, dev))

        zh_c, zl_c = model.init_states(b)
        v_prev = zh_c[:, 16:, :].float() @ basis
        v_ema = v_prev.clone()
        clean_boards_batch: list[list[list[int]]] = [[] for _ in range(b)]
        snaps: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        clean_next_logits: dict[int, torch.Tensor] = {}

        for s in range(1, horizon + 1):
            zh_c, zl_c = model.act_step(zh_c, zl_c, xb)
            d_log = model.digit_logits(zh_c)
            if (s - 1) in resume_steps:
                clean_next_logits[s - 1] = d_log.clone()
            preds = d_log.argmax(dim=-1).add(1).cpu().tolist()
            for bi in range(b):
                clean_boards_batch[bi].append(preds[bi])
            vt = zh_c[:, 16:, :].float() @ basis
            if s in resume_steps:
                snaps[s] = (zh_c.clone(), zl_c.clone(), vt.clone(), v_prev.clone(), v_ema.clone())
            v_ema = 0.5 * v_ema + 0.5 * vt
            v_prev = vt

        for bi in range(b):
            seq = clean_boards_batch[bi]
            clean_solve_16[i0 + bi] = (seq[min(15, len(seq) - 1)] == ans_sub[bi])
            clean_solve_h[i0 + bi] = (seq[-1] == ans_sub[bi])
            clean_lock[i0 + bi] = _lock_step_from_boards(seq, ans_sub[bi], 1)

        # Single-step Resume from each t in resume_steps
        for t in resume_steps:
            zh_t, zl_t, vt_t, vp_t, ve_t = snaps[t]
            t_clamp = min(t, train_steps)
            fn_now = _build_seq97_features(clue_oh, t_clamp, train_steps, vt_t, vp_t, ve_t, is_trace=False)
            fn_tr = _build_seq97_features(clue_oh, t_clamp, train_steps, vt_t, vp_t, ve_t, is_trace=True)
            p_now = net_now(fn_now)
            p_tr = net_tr(fn_tr)
            p_mean = y_tr_by_step[t_clamp - 1 : t_clamp].expand(b, -1, -1)

            for a_name, p_raw in (("c_now_97", p_now), ("c_trace_97", p_tr), ("v_mean_sphere", p_mean)):
                zh_r, zl_r, l_err = _apply_seq97_prediction_sphere(zh_t, zl_t, p_raw, model.lm_head.weight, basis)
                resume_data[t][a_name]["pre_logit_err"] = max(resume_data[t][a_name]["pre_logit_err"], l_err)
                seq_r: list[list[list[int]]] = [[] for _ in range(b)]
                for s in range(t + 1, horizon + 1):
                    zh_r, zl_r = model.act_step(zh_r, zl_r, xb)
                    dl = model.digit_logits(zh_r)
                    if s == t + 1:
                        l2_per_b = (dl - clean_next_logits[t]).reshape(b, -1).norm(dim=-1).cpu().tolist()
                        for bi in range(b):
                            resume_data[t][a_name]["step1_logit_l2"][i0 + bi] = float(l2_per_b[bi])
                    pr = dl.argmax(dim=-1).add(1).cpu().tolist()
                    for bi in range(b):
                        seq_r[bi].append(pr[bi])
                for bi in range(b):
                    resume_data[t][a_name]["solve_h"][i0 + bi] = (seq_r[bi][-1] == ans_sub[bi])
                    resume_data[t][a_name]["lock"][i0 + bi] = _lock_step_from_boards(seq_r[bi], ans_sub[bi], t + 1)

        # Closed-loop refill up to horizon for cadences m=1 (every step) and m=2 (every 2 steps)
        for cadence_m, s16_d, sh_d, ev_d, cyc_d in (
            (1, cl_solve_16, cl_solve_h, cl_ever_h, cl_cyc_h),
            (2, cl_m2_solve_16, cl_m2_solve_h, cl_m2_ever_h, cl_m2_cyc_h),
        ):
            for a_name in closed_loop_arms:
                zh_k, zl_k = model.init_states(b)
                vp_k = zh_k[:, 16:, :].float() @ basis
                ve_k = vp_k.clone()
                seq_cl: list[list[list[int]]] = [[] for _ in range(b)]
                for s in range(1, horizon + 1):
                    zh_k, zl_k = model.act_step(zh_k, zl_k, xb)
                    dl = model.digit_logits(zh_k)
                    pr = dl.argmax(dim=-1).add(1).cpu().tolist()
                    for bi in range(b):
                        seq_cl[bi].append(pr[bi])
                    vt_k = zh_k[:, 16:, :].float() @ basis
                    if s % cadence_m == 0:
                        s_clamped = min(s, train_steps)
                        if a_name == "c_now_97":
                            fk = _build_seq97_features(clue_oh, s_clamped, train_steps, vt_k, vp_k, ve_k, is_trace=False)
                            pk = net_now(fk)
                        elif a_name == "c_trace_97":
                            fk = _build_seq97_features(clue_oh, s_clamped, train_steps, vt_k, vp_k, ve_k, is_trace=True)
                            pk = net_tr(fk)
                        else:
                            pk = y_tr_by_step[s_clamped - 1 : s_clamped].expand(b, -1, -1)
                        zh_k, zl_k, _ = _apply_seq97_prediction_sphere(zh_k, zl_k, pk, model.lm_head.weight, basis)
                    ve_k = 0.5 * ve_k + 0.5 * vt_k
                    vp_k = vt_k
                for bi in range(b):
                    s16_d[a_name][i0 + bi] = (seq_cl[bi][min(15, len(seq_cl[bi]) - 1)] == ans_sub[bi])
                    sh_d[a_name][i0 + bi] = (seq_cl[bi][-1] == ans_sub[bi])
                    ev_d[a_name][i0 + bi] = any(bd == ans_sub[bi] for bd in seq_cl[bi])
                    cyc_d[a_name][i0 + bi] = detect_board_limit_cycle(seq_cl[bi])["oscillatory_cycle"]

    def _summarize_subset(indices: list[int]) -> dict:
        if not indices:
            return {"n": 0}
        denom = len(indices)
        c16 = sum(clean_solve_16[i] for i in indices) / denom
        ch = sum(clean_solve_h[i] for i in indices) / denom
        c_nl = sum(clean_lock[i] is None for i in indices) / denom
        cl_out = {}
        cl_m2_out = {}
        for a in closed_loop_arms:
            cl_out[a] = {
                "solve_s16": sum(cl_solve_16[a][i] for i in indices) / denom,
                f"solve_s{horizon}": sum(cl_solve_h[a][i] for i in indices) / denom,
                f"ever_solved_s{horizon}": sum(cl_ever_h[a][i] for i in indices) / denom,
                f"oscillatory_cycle_s{horizon}": sum(cl_cyc_h[a][i] for i in indices) / denom,
            }
            cl_m2_out[a] = {
                "solve_s16": sum(cl_m2_solve_16[a][i] for i in indices) / denom,
                f"solve_s{horizon}": sum(cl_m2_solve_h[a][i] for i in indices) / denom,
                f"ever_solved_s{horizon}": sum(cl_m2_ever_h[a][i] for i in indices) / denom,
                f"oscillatory_cycle_s{horizon}": sum(cl_m2_cyc_h[a][i] for i in indices) / denom,
            }
        res_out = {}
        for t in resume_steps:
            res_out[f"t_{t}"] = {}
            for a in ("c_now_97", "c_trace_97", "v_mean_sphere"):
                rd = resume_data[t][a]
                lost = sum((clean_lock[i] is not None) and (rd["lock"][i] is None) for i in indices)
                gained = sum((clean_lock[i] is None) and (rd["lock"][i] is not None) for i in indices)
                mutual_delays = [
                    rd["lock"][i] - clean_lock[i]
                    for i in indices
                    if (clean_lock[i] is not None) and (rd["lock"][i] is not None)
                ]
                res_out[f"t_{t}"][a] = {
                    f"solve_s{horizon}": sum(rd["solve_h"][i] for i in indices) / denom,
                    f"never_lock_h{horizon}": sum(rd["lock"][i] is None for i in indices) / denom,
                    "lost_locks_vs_clean": int(lost),
                    "gained_locks_vs_clean": int(gained),
                    "mutual_lock_n": int(len(mutual_delays)),
                    "median_delay_mutual": float(np.median(mutual_delays)) if mutual_delays else None,
                    "mean_step1_logit_l2": float(np.mean([rd["step1_logit_l2"][i] for i in indices])),
                    "max_pre_logit_err": float(rd["pre_logit_err"]),
                }
        return {
            "n": denom,
            "clean": {"solve_s16": c16, f"solve_s{horizon}": ch, f"never_lock_h{horizon}": c_nl},
            "closed_loop_every_step": cl_out,
            "closed_loop_m2": cl_m2_out,
            "single_step_resume": res_out,
        }

    return {
        "schema": "darkfibers.seq97_snapshot_vs_trace.v2",
        "models": {
            "c_now_97": {"fit": fit_now, "heldout_r2": r2_now},
            "c_trace_97": {"fit": fit_tr, "heldout_r2": r2_tr},
            "delta_r2_trace_minus_now": {
                k: float(r2_tr[k] - r2_now[k]) for k in r2_now if k.startswith("r2_")
            },
            "stratified_heldout_r2": stratified_r2,
        },
        "causal_outcomes": {
            "all": _summarize_subset(idx_all),
            "wb_prop": _summarize_subset(idx_prop),
            "wb_stall": _summarize_subset(idx_stall),
        },
    }


@torch.inference_mode()
def run_gramian_case(
    TS,
    model,
    row: dict,
    *,
    patch_step: int,
    horizon: int,
    probes: int,
    angle: float,
    seed: int,
    validate_causal: bool = True,
    eval_horizon: int = 12,
):
    device = model.H_init.device
    x = model.encode_inputs(TS.encode_questions([row["question"]], device))
    zh, zl = model.init_states(1)
    for _ in range(patch_step):
        zh, zl = model.act_step(zh, zl, x)
    if validate_causal:
        ans_digits = [int(c) for c in row["answer"]]
        out = causal_gramian_validation(
            model,
            zh,
            zl,
            x,
            answer_digits=ans_digits,
            board_start=model.puzzle_emb_len,
            probes=probes,
            fd_angle=angle,
            gramian_horizon=horizon,
            eval_horizon=eval_horizon,
            seed=seed,
        )
    else:
        out = sampled_trm_gramian(
            model,
            zh,
            zl,
            x,
            board_start=model.puzzle_emb_len,
            probes=probes,
            angle=angle,
            horizon=horizon,
            seed=seed,
        )
    out["case_id"] = row["id"]
    out["patch_step"] = patch_step
    return out


@torch.inference_mode()
def run_gramian_batch(
    TS,
    model,
    rows: list[dict],
    *,
    patch_step: int,
    horizon: int,
    probes: int,
    angle: float,
    base_seed: int,
    eval_horizon: int = 12,
    chunk_size: int = 8,
) -> list[dict]:
    device = model.H_init.device
    all_out: list[dict] = []
    for c0 in range(0, len(rows), chunk_size):
        sub = rows[c0 : c0 + chunk_size]
        m = len(sub)
        x_m = model.encode_inputs(TS.encode_questions([r["question"] for r in sub], device))
        zh_m, zl_m = model.init_states(m)
        for _ in range(patch_step):
            zh_m, zl_m = model.act_step(zh_m, zl_m, x_m)
        seeds = [base_seed ^ int(r["id"][:8], 16) for r in sub]
        ans_list = [[int(c) for c in r["answer"]] for r in sub]
        chunk_res = causal_gramian_validation_batch(
            model,
            zh_m,
            zl_m,
            x_m,
            answers_digits=ans_list,
            board_start=model.puzzle_emb_len,
            probes=probes,
            fd_angle=angle,
            gramian_horizon=horizon,
            eval_horizon=eval_horizon,
            seeds=seeds,
        )
        for r, item in zip(sub, chunk_res):
            item["case_id"] = r["id"]
            item["patch_step"] = patch_step
            all_out.append(item)
    return all_out


def summarize_swap_rows(SP, eval_boards: list[dict], swap_rows: list[dict]) -> dict:
    strata_map = {r["id"]: classify_sudoku_board(SP, r["question"]) for r in eval_boards}
    groups = {"all": swap_rows, "wb_prop": [], "wb_stall": [], "s4_stalled": []}
    for sr in swap_rows:
        info = strata_map[sr["case_id"]]
        if info["wb_group"] == "WB-Prop":
            groups["wb_prop"].append(sr)
        else:
            groups["wb_stall"].append(sr)
            if info["fine_stratum"] == "S4_stalled":
                groups["s4_stalled"].append(sr)

    summary = {}
    for g_name, rows in groups.items():
        if not rows:
            summary[g_name] = {"n": 0}
            continue
        n = len(rows)
        arm_names = list(rows[0]["arms"].keys())
        arm_stats = {}
        for a in arm_names:
            clean_ex = [r["arms"]["clean"]["exact"] for r in rows]
            arm_ex = [r["arms"][a]["exact"] for r in rows]
            lost = sum(c and (not x) for c, x in zip(clean_ex, arm_ex))
            gained = sum((not c) and x for c, x in zip(clean_ex, arm_ex))
            rec_vals = [r["arms"][a]["recovery_steps_to_clean"] for r in rows if r["arms"][a]["recovery_steps_to_clean"] is not None]
            arm_stats[a] = {
                "exact_rate": sum(arm_ex) / n,
                "exact_s16_rate": sum(r["arms"][a]["exact_s16"] for r in rows) / n,
                "ever_solved_rate": sum(r["arms"][a]["ever_solved"] for r in rows) / n,
                "never_lock_rate": sum(r["arms"][a]["never_lock"] for r in rows) / n,
                "oscillatory_cycle_rate": sum(r["arms"][a]["oscillatory_limit_cycle"] for r in rows) / n,
                "mean_first_successor_logit_l2": float(np.mean([r["arms"][a]["first_successor_logit_l2"] for r in rows])),
                "mean_first_successor_cos_to_donor": float(np.mean([r["arms"][a]["first_successor_cos_to_donor"] for r in rows])),
                "lost_vs_clean": int(lost),
                "gained_vs_clean": int(gained),
                "recovered_to_clean_count": int(len(rec_vals)),
                "median_recovery_steps": float(np.median(rec_vals)) if rec_vals else None,
                "max_pre_head_error": float(max(r["arms"][a]["pre_intervention_full_head_max_error"] for r in rows)),
            }
        summary[g_name] = {"n": n, "arms": arm_stats}
    return summary


def summarize_gramian_rows(SP, eval_boards: list[dict], gram_rows: list[dict]) -> dict:
    strata_map = {r["id"]: classify_sudoku_board(SP, r["question"]) for r in eval_boards}
    groups = {"all": gram_rows, "wb_prop": [], "wb_stall": []}
    for gr in gram_rows:
        info = strata_map[gr["case_id"]]
        groups["wb_prop" if info["wb_group"] == "WB-Prop" else "wb_stall"].append(gr)

    out = {}
    for g_name, rows in groups.items():
        if not rows:
            out[g_name] = {"n": 0}
            continue
        n = len(rows)
        rec = {
            "n": n,
            "mean_sampled_rank_1e-3": float(np.mean([r["sampled_rank_relative_1e-3"] for r in rows])),
            "median_condition_ratio_top_to_bottom": float(np.median([r["condition_ratio_top_to_bottom"] for r in rows])),
            "mean_eigenvalues_descending": np.mean([r["eigenvalues_descending"] for r in rows], axis=0).tolist(),
            "mean_per_lag_response": np.mean([r["per_lag_mean_response"] for r in rows], axis=0).tolist(),
            "max_pre_readout_error": float(max(r["certificate"]["max_readout_error"] for r in rows)),
            "max_pre_l2_error": float(max(r["certificate"]["max_l2_error"] for r in rows)),
        }
        if "causal_direction_tests" in rows[0]:
            angle_keys = list(rows[0]["causal_direction_tests"].keys())
            causal_sum = {}
            eh = rows[0]["eval_horizon"]
            kh = min(rows[0]["horizon"], eh)
            for ak in angle_keys:
                causal_sum[ak] = {}
                for d_name in ("high_eig", "low_eig", "random_dir"):
                    d_recs = [r["causal_direction_tests"][ak]["directions"][d_name] for r in rows]
                    causal_sum[ak][d_name] = {
                        "mean_lag1_readout_l2": float(np.mean([x["lag1_readout_l2"] for x in d_recs])),
                        f"mean_cum_h{kh}_rms_readout_l2": float(np.mean([x[f"cum_h{kh}_rms_readout_l2"] for x in d_recs])),
                        f"mean_lag{eh}_readout_l2": float(np.mean([x[f"lag{eh}_readout_l2"] for x in d_recs])),
                        f"mean_board_flips_at_h{eh}": float(np.mean([x[f"board_flips_vs_clean_at_h{eh}"] for x in d_recs])),
                        f"exact_solve_rate_at_h{eh}": float(np.mean([x[f"exact_solve_at_h{eh}"] for x in d_recs])),
                    }
                high_wins_h4 = sum(
                    r["causal_direction_tests"][ak]["directions"]["high_eig"][f"cum_h{kh}_rms_readout_l2"]
                    > r["causal_direction_tests"][ak]["directions"]["low_eig"][f"cum_h{kh}_rms_readout_l2"]
                    for r in rows
                )
                high_wins_rand = sum(
                    r["causal_direction_tests"][ak]["directions"]["high_eig"][f"cum_h{kh}_rms_readout_l2"]
                    > r["causal_direction_tests"][ak]["directions"]["random_dir"][f"cum_h{kh}_rms_readout_l2"]
                    for r in rows
                )
                causal_sum[ak]["high_beats_low_cum_rate"] = float(high_wins_h4 / n)
                causal_sum[ak]["high_beats_random_cum_rate"] = float(high_wins_rand / n)
            rec["causal_validation_summary"] = causal_sum
        out[g_name] = rec
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--checkpoint-file", type=Path, required=True)
    p.add_argument("--tag", choices=("gaoxin492", "yagizdevre"), default="gaoxin492")
    p.add_argument("--mode", choices=("swap", "cache", "gramian"), default="swap")
    p.add_argument("--train-csv", type=Path, help="Required for cache mode")
    p.add_argument("--eval-csv", type=Path, required=True, help="Fresh puzzles, disjoint from prior runs")
    p.add_argument("--exclude-csv", type=Path, action="append", default=[],
                   help="Prior evaluated puzzles (.csv or .json); repeat for each data file")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=4, help="pilot limit; 0 means all")
    p.add_argument("--train-limit", type=int, default=24)
    p.add_argument("--patch-step", type=int, default=4)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--steps", type=int, default=8, help="cache collection depth")
    p.add_argument("--alpha", type=float, default=50.0)
    p.add_argument("--sigma", type=float, default=0.35)
    p.add_argument("--probes", type=int, default=8, help="orthonormal kernel-sphere tangent directions")
    p.add_argument("--angle", type=float, default=0.04, help="central-difference angle in radians")
    p.add_argument("--gramian-horizon", type=int, default=4)
    p.add_argument("--eval-horizon", type=int, default=12, help="nonlinear continuation horizon for Gramian validation")
    p.add_argument("--full-97-refill", action="store_true",
                   help="In cache mode, also train matched 97-token C^now vs C^trace Transformer refillers + causal Resume")
    p.add_argument("--seed", type=int, default=20261115)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if (
        args.limit < 0
        or args.train_limit < 1
        or args.steps < 2
        or not (1 <= args.patch_step < args.horizon)
        or args.probes < 1
        or args.gramian_horizon < 1
    ):
        p.error("invalid limits/steps/horizon")
    if args.mode == "cache" and args.train_csv is None:
        p.error("--train-csv required in cache mode")
    if not args.exclude_csv:
        p.error("explicit --exclude-csv of previously evaluated boards is required")

    evaluation = read_boards(args.eval_csv, args.limit)
    train = read_boards(args.train_csv, args.train_limit) if args.mode == "cache" else []
    audit = audit_boards(train, evaluation, args.exclude_csv, check_full_automorphisms=True)

    manifest = {
        "schema": "darkfibers.loopscope04.local.v2",
        "audit": audit,
        "mode": args.mode,
        "checkpoint_tag": args.tag,
        "checkpoint_path": str(args.checkpoint_file),
        "checkpoint_sha256": sha256_file(args.checkpoint_file),
        "code_sha256": sha256_file(Path(__file__)),
        "device": args.device,
        "settings": {
            k: getattr(args, k)
            for k in ("patch_step", "horizon", "steps", "alpha", "sigma", "probes", "angle", "gramian_horizon", "eval_horizon", "seed")
        },
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device unavailable: use --device cpu")
    if args.device.startswith("cuda"):
        torch.cuda.init()
        cuda_dev = torch.device(args.device)
        if cuda_dev.index is None:
            cuda_dev = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.reset_peak_memory_stats(cuda_dev)

    TS = import_trm(args.repo_root)
    SP = import_sudoku_prop(args.repo_root)
    start = time.perf_counter()
    model, _pin = load_model(TS, args.checkpoint_file, args.tag, args.device)

    if args.mode == "cache":
        manifest["cache"] = cache_compare(TS, model, train, evaluation, args.steps, args.alpha)
        manifest["scope"] = (
            "cellwise ridge + full 97-token capacity-matched sequence refill + causal Resume"
            if args.full_97_refill
            else "cellwise ridge snapshot-vs-one-step-visible-history; not full C5"
        )
        if args.full_97_refill:
            manifest["full_97_refill"] = full_97_cache_and_causal_eval(
                TS,
                SP,
                model,
                train,
                evaluation,
                train_steps=args.steps,
                horizon=max(args.horizon, 32),
                seed=args.seed,
            )
    else:
        if args.mode == "gramian":
            gram_chunk = max(1, min(8, 64 // max(args.probes, 1)))
            manifest["rows"] = run_gramian_batch(
                TS,
                model,
                evaluation,
                patch_step=args.patch_step,
                horizon=args.gramian_horizon,
                probes=args.probes,
                angle=args.angle,
                base_seed=args.seed,
                eval_horizon=args.eval_horizon,
                chunk_size=gram_chunk,
            )
            manifest["summary"] = summarize_gramian_rows(SP, evaluation, manifest["rows"])
        else:
            manifest["rows"] = run_swap_batch(
                TS,
                model,
                evaluation,
                patch_step=args.patch_step,
                horizon=args.horizon,
                base_seed=args.seed,
                sigma=args.sigma,
                chunk_size=32,
            )
            manifest["summary"] = summarize_swap_rows(SP, evaluation, manifest["rows"])
        print(f"completed {len(manifest['rows'])} {args.mode} cases", flush=True)

    manifest["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    if args.device.startswith("cuda"):
        manifest["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated(args.device))
    write_json(args.out, manifest)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
