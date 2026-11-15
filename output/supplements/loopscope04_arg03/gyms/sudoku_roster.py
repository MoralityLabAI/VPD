#!/usr/bin/env python3
"""Deterministic unique-solution Sudoku roster builder & representative split exporter for LOOPSCOPE-04.

Supports two provenance modes:
  1. `pilot` (`--mode pilot`, default): generates novel unique-solution puzzles via exact
     count-up-to-two solver deletion, using either the fast permuted Latin base (`--generator permuted`)
     or independent randomized MRV grid synthesis across non-isomorphic solution classes (`--generator diverse`).
  2. `representative` (`--mode representative --source-csv <path> --start-row <offset>`):
     extracts solver-certified (`count_solutions == 1`), deductively stratified (`WB-Prop`, `WB-Stall`,
     `S4_stalled`) held-out puzzles from untouched tail rows of an external benchmark CSV, screening
     every candidate against all `--exclude-csv` datasets under exact, digit-relabel, AND full
     Sudoku grid/transpose/digit automorphisms (`are_sudoku_isomorphic`).
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

from .core import sha256_file, write_json
from .loopscope04_trm import (
    are_sudoku_isomorphic,
    canonical_digit_relabel,
    read_boards,
    sudoku_wl_signature,
)

FULL = (1 << 9) - 1


def generate_solution(rng: random.Random) -> str:
    bands = rng.sample(range(3), 3)
    stacks = rng.sample(range(3), 3)
    rows = [3 * b + i for b in bands for i in rng.sample(range(3), 3)]
    cols = [3 * b + i for b in stacks for i in rng.sample(range(3), 3)]
    digits = rng.sample(range(1, 10), 9)
    grid = [[digits[(r * 3 + r // 3 + c) % 9] for c in cols] for r in rows]
    if rng.randrange(2):
        grid = [[grid[c][r] for c in range(9)] for r in range(9)]
    return "".join(str(v) for row in grid for v in row)


def generate_diverse_solution(rng: random.Random) -> str:
    """Synthesize a genuinely random valid 9x9 Sudoku solution via randomized MRV backtracking."""
    grid = [0] * 81
    rmask, cmask, bmask = [0] * 9, [0] * 9, [0] * 9

    def solve() -> bool:
        best = -1
        best_mask = 0
        best_cnt = 10
        for i, v in enumerate(grid):
            if v:
                continue
            r, c = divmod(i, 9)
            b = (r // 3) * 3 + c // 3
            mask = FULL & ~(rmask[r] | cmask[c] | bmask[b])
            n = mask.bit_count()
            if n == 0:
                return False
            if n < best_cnt:
                best, best_mask, best_cnt = i, mask, n
                if n == 1:
                    break
        if best < 0:
            return True
        r, c = divmod(best, 9)
        b = (r // 3) * 3 + c // 3
        digits = [d for d in range(1, 10) if (best_mask & (1 << (d - 1)))]
        rng.shuffle(digits)
        for d in digits:
            bit = 1 << (d - 1)
            grid[best] = d
            rmask[r] |= bit
            cmask[c] |= bit
            bmask[b] |= bit
            if solve():
                return True
            rmask[r] ^= bit
            cmask[c] ^= bit
            bmask[b] ^= bit
            grid[best] = 0
        return False

    if not solve():
        raise RuntimeError("empty Sudoku failed to solve")
    return "".join(str(v) for v in grid)


def count_solutions(puzzle: str, limit: int = 2) -> int:
    grid = [0 if c in ".0" else int(c) for c in puzzle]
    rmask, cmask, bmask = [0] * 9, [0] * 9, [0] * 9
    for i, v in enumerate(grid):
        if not v:
            continue
        r, c = divmod(i, 9)
        b = (r // 3) * 3 + c // 3
        bit = 1 << (v - 1)
        if (rmask[r] | cmask[c] | bmask[b]) & bit:
            return 0
        rmask[r] |= bit
        cmask[c] |= bit
        bmask[b] |= bit
    answers = 0

    def search():
        nonlocal answers
        if answers >= limit:
            return
        best = -1
        best_mask = 0
        best_count = 10
        for i, v in enumerate(grid):
            if v:
                continue
            r, c = divmod(i, 9)
            b = (r // 3) * 3 + c // 3
            mask = FULL & ~(rmask[r] | cmask[c] | bmask[b])
            n = mask.bit_count()
            if n == 0:
                return
            if n < best_count:
                best, best_mask, best_count = i, mask, n
                if n == 1:
                    break
        if best < 0:
            answers += 1
            return
        r, c = divmod(best, 9)
        b = (r // 3) * 3 + c // 3
        mask = best_mask
        while mask and answers < limit:
            bit = mask & -mask
            mask -= bit
            grid[best] = bit.bit_length()
            rmask[r] |= bit
            cmask[c] |= bit
            bmask[b] |= bit
            search()
            rmask[r] ^= bit
            cmask[c] ^= bit
            bmask[b] ^= bit
            grid[best] = 0

    search()
    return answers


def delete_clues(solution: str, target: int, rng: random.Random) -> str | None:
    grid = list(solution)
    indexes = list(range(81))
    rng.shuffle(indexes)
    for i in indexes:
        if sum(c != "." for c in grid) <= target:
            break
        old = grid[i]
        grid[i] = "."
        if count_solutions("".join(grid)) != 1:
            grid[i] = old
    return "".join(grid) if sum(c != "." for c in grid) == target else None


def _build_exclusion_index(exclude_paths: list[Path]) -> tuple[set[str], dict[str, list[str]], dict[str, str]]:
    banned_relabel: set[str] = set()
    banned_wl: dict[str, list[str]] = {}
    excluded_sha: dict[str, str] = {}
    for path in exclude_paths:
        excluded_sha[str(path)] = sha256_file(path)
        for r in read_boards(path):
            q = r["question"]
            banned_relabel.add(canonical_digit_relabel(q))
            wl = sudoku_wl_signature(q)
            banned_wl.setdefault(wl, []).append(q)
    return banned_relabel, banned_wl, excluded_sha


def _is_novel_and_register(q: str, used_relabel: set[str], used_wl: dict[str, list[str]]) -> bool:
    sig = canonical_digit_relabel(q)
    if sig in used_relabel:
        return False
    wl = sudoku_wl_signature(q)
    for prev_q in used_wl.get(wl, ()):
        if are_sudoku_isomorphic(q, prev_q):
            return False
    used_relabel.add(sig)
    used_wl.setdefault(wl, []).append(q)
    return True


def run_representative(args) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from mechinterp.loopscope02.scripts import sudoku_prop as SP

    rng = random.Random(args.seed)
    used_relabel, used_wl, excluded_sha = _build_exclusion_index(args.exclude_csv)
    source_sha = sha256_file(args.source_csv)

    raw_tail: list[dict] = []
    with args.source_csv.open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for idx, row in enumerate(rd):
            if idx < args.start_row:
                continue
            q = (row.get("question") or row.get("puzzle") or "").replace("0", ".")
            a = row.get("answer") or row.get("solution") or ""
            if len(q) == 81 and len(a) == 81:
                raw_tail.append({"question": q, "answer": a, "source_row_idx": idx})

    if not raw_tail:
        raise RuntimeError(f"no candidate rows found in {args.source_csv} at >= {args.start_row}")
    rng.shuffle(raw_tail)

    # Stratified evaluation targets: 1/3 WB-Prop, 1/3 WB-Stall (partial singles), 1/3 S4_stalled (round-0 zero singles)
    q_prop = args.n_eval // 3
    q_s4 = args.n_eval // 3
    q_part = args.n_eval - q_prop - q_s4
    eval_buckets = {"WB-Prop": [], "WB-Stall_partial": [], "S4_stalled": []}
    eval_quotas = {"WB-Prop": q_prop, "WB-Stall_partial": q_part, "S4_stalled": q_s4}
    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    inspected = 0

    for cand in raw_tail:
        need_eval = any(len(eval_buckets[k]) < eval_quotas[k] for k in eval_quotas)
        need_train = len(train_rows) < args.n_train
        need_dev = len(dev_rows) < args.n_dev
        if not (need_eval or need_train or need_dev):
            break
        inspected += 1
        q, a = cand["question"], cand["answer"]
        tr = SP.trace_propagation(q)
        b_key = "WB-Prop" if tr.solved else ("S4_stalled" if tr.num_rounds == 0 else "WB-Stall_partial")

        # Only spend uniqueness-solver + isomorphism checks on needed buckets
        want_for_eval = len(eval_buckets[b_key]) < eval_quotas[b_key]
        if not (want_for_eval or need_dev or need_train):
            continue
        if count_solutions(q) != 1:
            continue
        if any(q[i] not in (".", a[i]) for i in range(81)) or count_solutions(a) != 1:
            continue
        if not _is_novel_and_register(q, used_relabel, used_wl):
            continue

        rec = {
            "question": q,
            "answer": a,
            "stratum": b_key,
            "prop_solved": int(tr.solved),
            "prop_rounds": int(tr.num_rounds),
            "origin": f"holdout_row_{cand['source_row_idx']}",
        }
        if want_for_eval:
            eval_buckets[b_key].append(rec)
        elif need_dev:
            dev_rows.append(rec)
        elif need_train:
            train_rows.append(rec)

    eval_rows = eval_buckets["WB-Prop"] + eval_buckets["WB-Stall_partial"] + eval_buckets["S4_stalled"]
    if len(eval_rows) < args.n_eval or len(train_rows) < args.n_train or len(dev_rows) < args.n_dev:
        raise RuntimeError(
            f"insufficient disjoint boards: train={len(train_rows)}/{args.n_train}, "
            f"dev={len(dev_rows)}/{args.n_dev}, eval_buckets={{k: len(v) for k, v in eval_buckets.items()}}"
        )

    splits_out = {"train": train_rows, "dev": dev_rows, "eval": eval_rows}
    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in splits_out.items():
            if not rows:
                continue
            path = args.out_dir / f"{name}.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=("question", "answer", "stratum", "prop_solved", "prop_rounds", "origin"))
                w.writeheader()
                w.writerows(rows)

    manifest = {
        "schema": "darkfibers.sudoku.representative_holdout.v1",
        "seed": args.seed,
        "source_csv": str(args.source_csv),
        "source_csv_sha256": source_sha,
        "source_start_row": args.start_row,
        "candidates_inspected": inspected,
        "counts": {k: len(v) for k, v in splits_out.items()},
        "eval_strata_counts": {k: len(v) for k, v in eval_buckets.items()},
        "exact_unique_solution_certified": True,
        "excluded_sha256": excluded_sha,
        "collision_audit": "exact + digit-relabel + full Sudoku grid/transpose/digit automorphisms (WL + exact search)",
        "puzzle_sha256": {
            k: sha256("".join(r["question"] for r in v).encode()).hexdigest()
            for k, v in splits_out.items()
            if v
        },
    }
    if not args.dry_run:
        manifest["csv_sha256"] = {
            k: sha256_file(args.out_dir / f"{k}.csv")
            for k, v in splits_out.items()
            if v
        }
        write_json(args.out_dir / "roster_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def run(args):
    if getattr(args, "mode", "pilot") == "representative":
        return run_representative(args)
    rng = random.Random(args.seed)
    used_relabel, used_wl, excluded = _build_exclusion_index(args.exclude_csv)
    gen_fn = generate_diverse_solution if getattr(args, "generator", "permuted") == "diverse" else generate_solution
    out = {}
    total = 0
    for name, n in (("train", args.n_train), ("eval", args.n_eval)):
        rows = []
        for _ in range(args.max_attempts):
            if len(rows) >= n:
                break
            total += 1
            answer = gen_fn(rng)
            target = rng.randint(args.min_clues, args.max_clues)
            question = delete_clues(answer, target, rng)
            if question is None:
                continue
            if count_solutions(question) != 1:
                raise AssertionError("generator uniqueness regression")
            if not _is_novel_and_register(question, used_relabel, used_wl):
                continue
            rows.append({"question": question, "answer": answer})
        if len(rows) < n:
            raise RuntimeError(f"generator exhausted ({name}: {len(rows)}/{n})")
        out[name] = rows
    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in out.items():
            path = args.out_dir / f"{name}.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=("question", "answer"))
                w.writeheader()
                w.writerows(rows)
    manifest = {
        "schema": "darkfibers.sudoku.fresh_pilot.v2",
        "seed": args.seed,
        "generator": getattr(args, "generator", "permuted"),
        "n_train": len(out["train"]),
        "n_eval": len(out["eval"]),
        "clue_range": [args.min_clues, args.max_clues],
        "exact_unique_solution_certified": True,
        "excluded_sha256": excluded,
        "collision_audit": "exact + digit-relabel + full Sudoku grid/transpose/digit automorphisms (WL + exact search)",
        "source_distribution": (
            "independent MRV-synthesized Sudoku solutions, unique clue deletion"
            if getattr(args, "generator", "permuted") == "diverse"
            else "permuted base Latin solved grid, unique clue deletion"
        ),
        "generator_attempts": total,
        "puzzle_sha256": {
            k: sha256("".join(row["question"] for row in v).encode()).hexdigest()
            for k, v in out.items()
        },
    }
    if not args.dry_run:
        manifest["csv_sha256"] = {k: sha256_file(args.out_dir / f"{k}.csv") for k in out}
        write_json(args.out_dir / "roster_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("pilot", "representative"), default="pilot")
    p.add_argument("--generator", choices=("permuted", "diverse"), default="permuted")
    p.add_argument("--source-csv", type=Path, help="External benchmark CSV for --mode representative")
    p.add_argument("--start-row", type=int, default=5000, help="Row offset into --source-csv")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--exclude-csv", type=Path, action="append", default=[], help="Previously evaluated puzzles (.csv or .json)")
    p.add_argument("--n-train", type=int, default=24)
    p.add_argument("--n-dev", type=int, default=0)
    p.add_argument("--n-eval", type=int, default=12)
    p.add_argument("--min-clues", type=int, default=32)
    p.add_argument("--max-clues", type=int, default=38)
    p.add_argument("--max-attempts", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261115)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if not args.exclude_csv:
        p.error("at least one --exclude-csv is required")
    if args.mode == "representative" and args.source_csv is None:
        p.error("--source-csv is required when --mode representative")
    if args.n_train < 1 or args.n_eval < 1 or args.n_dev < 0 or not (17 <= args.min_clues <= args.max_clues <= 81):
        p.error("invalid puzzle sizes/clues")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
