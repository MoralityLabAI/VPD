"""Exact 9x9 Sudoku constraint-propagation lattice G_prop and stratified dataset builder.

Defines synchronous naked-single + hidden-single elimination rounds, peer-conflict
counting, candidate masks, and deterministic stratified sampling from
`sapientinc/sudoku-extreme-1k` (`train.csv`, `test_sudoku_bench.csv`, `test_hard.csv`)
with SHA-256 verification on every input file.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

DATASET_REPO = "sapientinc/sudoku-extreme-1k"
EXPECTED_SHA256 = {
    "train.csv": "887dea10daba811d3a6a565bb138e31ef60a95fdb57045633e4762fc46357cbb",
    "test_sudoku_bench.csv": "0bd94e4843c3ec55e225c8798bf3f63083b2b2207e88ca996c3bbb4b8b1c18e1",
    "test_hard.csv": "5f9549fb2f38e2cbcb9a515733fdfb19d10c599f21d5c9e82ec20bca28e78c2c",
}
SPLIT_SEED = 20261113
STRATA = ("S1_naked1", "S2_prop2", "S3_prop3plus", "S4_stalled")
PER_STRATUM = 64

UNITS: list[list[int]] = []
for _r in range(9):
    UNITS.append([_r * 9 + _c for _c in range(9)])
for _c in range(9):
    UNITS.append([_r * 9 + _c for _r in range(9)])
for _br in range(0, 9, 3):
    for _bc in range(0, 9, 3):
        UNITS.append([(_br + _dr) * 9 + (_bc + _dc) for _dr in range(3) for _dc in range(3)])

PEERS: list[list[int]] = []
for _i in range(81):
    _r, _c = divmod(_i, 9)
    _br, _bc = (_r // 3) * 3, (_c // 3) * 3
    _ps = (
        {(_r * 9 + _cc) for _cc in range(9)}
        | {(_rr * 9 + _c) for _rr in range(9)}
        | {(_br + _dr) * 9 + (_bc + _dc) for _dr in range(3) for _dc in range(3)}
    )
    _ps.remove(_i)
    PEERS.append(sorted(_ps))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_grid(s: str) -> list[int]:
    if len(s) != 81:
        raise ValueError(f"expected 81-char Sudoku string, got len={len(s)}")
    return [0 if ch in ".0" else int(ch) for ch in s]


def format_grid(grid: Sequence[int]) -> str:
    return "".join("." if v == 0 else str(v) for v in grid)


def candidate_mask(grid: Sequence[int]) -> np.ndarray:
    """Return [81, 9] bool mask of locally consistent digits 1..9 per cell."""
    mask = np.zeros((81, 9), dtype=bool)
    for i in range(81):
        v = grid[i]
        if v != 0:
            mask[i, v - 1] = True
        else:
            used = {grid[p] for p in PEERS[i] if grid[p] != 0}
            for d in range(1, 10):
                if d not in used:
                    mask[i, d - 1] = True
    return mask


def step_propagation(grid: Sequence[int]) -> tuple[list[int], dict[int, int], bool]:
    """One synchronous round G_prop of naked + hidden single elimination.

    Returns (next_grid, assigned_cells_map, conflict_detected).
    """
    nxt = list(grid)
    if count_unit_conflicts(nxt) > 0:
        return nxt, {}, True
    cands: list[set[int]] = [set() for _ in range(81)]
    for i in range(81):
        if nxt[i] == 0:
            used = {nxt[p] for p in PEERS[i] if nxt[p] != 0}
            cands[i] = set(range(1, 10)) - used
            if not cands[i]:
                return nxt, {}, True
    assigns: dict[int, int] = {}
    conflict = False
    for i in range(81):
        if nxt[i] == 0 and len(cands[i]) == 1:
            assigns[i] = next(iter(cands[i]))
    for u in UNITS:
        for d in range(1, 10):
            if any(nxt[i] == d for i in u):
                continue
            places = [i for i in u if nxt[i] == 0 and d in cands[i]]
            if not places:
                conflict = True
            elif len(places) == 1:
                idx = places[0]
                if idx in assigns and assigns[idx] != d:
                    conflict = True
                else:
                    assigns[idx] = d
    if conflict:
        return nxt, {}, True
    for idx, d in assigns.items():
        nxt[idx] = d
    if count_unit_conflicts(nxt) > 0:
        return list(grid), {}, True
    return nxt, assigns, False


def count_unit_conflicts(grid: Sequence[int]) -> int:
    """Count duplicate non-zero digit occurrences across the 27 Sudoku units."""
    conflicts = 0
    for u in UNITS:
        vals = [grid[i] for i in u if grid[i] != 0]
        conflicts += len(vals) - len(set(vals))
    return conflicts


def count_cell_peer_conflicts(grid: Sequence[int]) -> int:
    """Number of cells whose non-zero digit matches at least one peer cell."""
    bad = 0
    for i in range(81):
        v = grid[i]
        if v != 0 and any(grid[p] == v for p in PEERS[i]):
            bad += 1
    return bad


def is_valid_solution(grid: Sequence[int], givens: Sequence[int]) -> bool:
    if any(v < 1 or v > 9 for v in grid):
        return False
    for i in range(81):
        if givens[i] != 0 and grid[i] != givens[i]:
            return False
    return count_unit_conflicts(grid) == 0


@dataclass(frozen=True)
class PropagationTrace:
    cell_round: tuple[int, ...]      # 0 = given, r>=1 = round deduced, -1 = stalled
    naked_round1: tuple[bool, ...]   # True iff empty cell has 8 distinct given peer digits at round 0
    rounds_grids: tuple[tuple[int, ...], ...]  # grid after 0, 1, ..., R rounds
    solved: bool
    num_rounds: int


def trace_propagation(question: str) -> PropagationTrace:
    g0 = parse_grid(question)
    naked1 = [False] * 81
    for i in range(81):
        if g0[i] == 0:
            used = {g0[p] for p in PEERS[i] if g0[p] != 0}
            naked1[i] = len(used) == 8
    cell_round = [0 if v != 0 else -1 for v in g0]
    grids = [tuple(g0)]
    cur = list(g0)
    rnd = 0
    while True:
        nxt, assigns, conflict = step_propagation(cur)
        if conflict or not assigns:
            break
        rnd += 1
        for idx in assigns:
            cell_round[idx] = rnd
        cur = nxt
        grids.append(tuple(cur))
    solved = all(v != 0 for v in cur) and count_unit_conflicts(cur) == 0
    return PropagationTrace(
        cell_round=tuple(cell_round),
        naked_round1=tuple(naked1),
        rounds_grids=tuple(grids),
        solved=solved,
        num_rounds=rnd,
    )


def g_prop_belief(prob: np.ndarray, givens: Sequence[int]) -> np.ndarray:
    """Apply G_prop to a discrete board projection of cell beliefs prob [81, 9].

    Projects prob to its hard argmax on cells with concentration >= 0.5 (keeping givens fixed);
    if consistent, applies one synchronous step_propagation round and returns target distribution:
    1-hot on newly deduced/given cells, and locally masked renormalized prob on remaining cells.
    """
    hard = list(givens)
    for i in range(81):
        if hard[i] == 0:
            top = int(np.argmax(prob[i]))
            if prob[i, top] >= 0.5:
                hard[i] = top + 1
    nxt, assigns, conflict = step_propagation(hard)
    if conflict:
        # Fall back to propagating from certified givens + conflict-free high-confidence subset
        nxt, assigns, _ = step_propagation(givens)
    out = np.zeros_like(prob, dtype=np.float64)
    mask = candidate_mask(nxt)
    for i in range(81):
        if nxt[i] != 0:
            out[i, nxt[i] - 1] = 1.0
        else:
            p = prob[i].astype(np.float64) * mask[i].astype(np.float64)
            s = float(p.sum())
            if s > 1e-12:
                out[i] = p / s
            else:
                m_sum = float(mask[i].sum())
                out[i] = mask[i].astype(np.float64) / max(m_sum, 1.0)
    return out.astype(np.float32)


def resolve_dataset_files() -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    paths: dict[str, Path] = {}
    for fname, expected in EXPECTED_SHA256.items():
        p = Path(hf_hub_download(repo_id=DATASET_REPO, filename=fname, repo_type="dataset"))
        actual = sha256_file(p)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {fname}: expected {expected}, got {actual}")
        paths[fname] = p
    return paths


def render_board_with_query(question: str, target_idx: int | None = None) -> str:
    """Render 9x9 Sudoku with row/col headers and optional '?' at target_idx."""
    lines = ["    C1 C2 C3   C4 C5 C6   C7 C8 C9"]
    for r in range(9):
        row_tokens = []
        for c in range(9):
            idx = r * 9 + c
            ch = question[idx]
            if target_idx is not None and idx == target_idx:
                if ch != ".":
                    raise ValueError("target_idx must point to an empty cell")
                tok = " ?"
            else:
                tok = f" {ch}"
            row_tokens.append(tok)
            if c in (2, 5):
                row_tokens.append("  ")
        lines.append(f"R{r + 1} " + "".join(row_tokens))
        if r in (2, 5):
            lines.append("")
    return "\n".join(lines)


def format_target_prompt(shots: Sequence[dict], question: str, target_idx: int) -> str:
    """3-shot prompt ending with 'Answer: ' so digits '1'..'9' are single next tokens."""
    blocks = [
        "Solve for the digit (1-9) at the cell marked '?' in the 9x9 Sudoku grid. "
        "Each row R1-R9, column C1-C9, and 3x3 box contains digits 1-9 once."
    ]
    for s in shots:
        r, c = divmod(int(s["target_idx"]), 9)
        grid_txt = render_board_with_query(s["question"], int(s["target_idx"]))
        ans_digit = s["answer"][int(s["target_idx"])]
        blocks.append(f"{grid_txt}\nCell ? (R{r + 1}C{c + 1}) Answer: {ans_digit}")
    r, c = divmod(target_idx, 9)
    grid_txt = render_board_with_query(question, target_idx)
    blocks.append(f"{grid_txt}\nCell ? (R{r + 1}C{c + 1}) Answer: ")
    return "\n\n".join(blocks)


def build_benchmark_bundle(seed: int = SPLIT_SEED, per_stratum: int = PER_STRATUM) -> dict:
    """Build deterministic 3-shot demos + 4x64 stratified target-cell items + 100 bench boards."""
    paths = resolve_dataset_files()
    rng = np.random.default_rng(seed)

    # 1. Select 3 deterministic shots from train.csv (1 naked1, 1 prop2, 1 prop3)
    with open(paths["train.csv"], "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    shots: list[dict] = []
    wanted = ["n1", "p2", "p3"]
    for row in train_rows:
        tr = trace_propagation(row["question"])
        kind = wanted[len(shots)]
        cand_cells = []
        for idx in range(81):
            if kind == "n1" and tr.naked_round1[idx]:
                cand_cells.append(idx)
            elif kind == "p2" and tr.cell_round[idx] == 2:
                cand_cells.append(idx)
            elif kind == "p3" and tr.cell_round[idx] >= 3:
                cand_cells.append(idx)
        if cand_cells:
            shots.append({
                "source": row["source"],
                "question": row["question"],
                "answer": row["answer"],
                "target_idx": int(cand_cells[0]),
                "demo_kind": kind,
            })
            if len(shots) == 3:
                break
    if len(shots) != 3:
        raise RuntimeError(f"expected 3 shots from train.csv, got {len(shots)}")

    # 2. Load 100 sudoku-bench boards + first 1,500 test_hard boards (disjoint questions)
    with open(paths["test_sudoku_bench.csv"], "r", encoding="utf-8") as f:
        bench_rows = list(csv.DictReader(f))
    with open(paths["test_hard.csv"], "r", encoding="utf-8") as f:
        hard_rows = []
        for idx, r in enumerate(csv.DictReader(f)):
            if idx >= 1500:
                break
            hard_rows.append(r)

    bench_boards = []
    for bi, r in enumerate(bench_rows):
        tr = trace_propagation(r["question"])
        bench_boards.append({
            "board_id": f"bench_{bi:03d}",
            "source": r["source"],
            "question": r["question"],
            "answer": r["answer"],
            "rating": int(r["rating"]),
            "prop_solved": tr.solved,
            "prop_rounds": tr.num_rounds,
            "cell_round": list(tr.cell_round),
        })

    # Pool candidate boards per stratum (bench first, then hard_rows; distinct question strings)
    seen_q: set[str] = set()
    pools: dict[str, list[dict]] = {s: [] for s in STRATA}
    for tag, rows in [("bench", bench_rows), ("hard", hard_rows)]:
        for ri, r in enumerate(rows):
            q = r["question"]
            if q in seen_q:
                continue
            seen_q.add(q)
            tr = trace_propagation(q)
            n1 = [i for i in range(81) if tr.naked_round1[i]]
            p2 = [i for i in range(81) if tr.cell_round[i] == 2]
            p3 = [i for i in range(81) if tr.cell_round[i] >= 3]
            st = [i for i in range(81) if tr.cell_round[i] == -1]
            rec_base = {
                "origin": f"{tag}_{ri:04d}",
                "question": q,
                "answer": r["answer"],
                "rating": int(r["rating"]),
                "prop_solved": tr.solved,
                "prop_rounds": tr.num_rounds,
            }
            if n1:
                pools["S1_naked1"].append({**rec_base, "cands": n1, "cell_round_map": tr.cell_round})
            if p2:
                pools["S2_prop2"].append({**rec_base, "cands": p2, "cell_round_map": tr.cell_round})
            if p3:
                pools["S3_prop3plus"].append({**rec_base, "cands": p3, "cell_round_map": tr.cell_round})
            if st and not tr.solved and tr.num_rounds == 0:
                # Strict zero-single stalled boards: G_prop makes zero progress at step 0
                pools["S4_stalled"].append({**rec_base, "cands": st, "cell_round_map": tr.cell_round})

    items: list[dict] = []
    used_questions: set[str] = set()
    # Sample S3, S2, S1, S4 in order of pool scarcity so all 256 items come from 256 distinct boards
    sample_order = ("S3_prop3plus", "S1_naked1", "S2_prop2", "S4_stalled")
    by_stratum: dict[str, list[dict]] = {}
    for s in sample_order:
        avail = [b for b in pools[s] if b["question"] not in used_questions]
        if len(avail) < per_stratum:
            raise RuntimeError(f"insufficient distinct boards for {s}: {len(avail)} < {per_stratum}")
        order = rng.permutation(len(avail))[:per_stratum]
        chosen = []
        for idx_pos, oi in enumerate(sorted(int(x) for x in order)):
            b = avail[oi]
            used_questions.add(b["question"])
            cands = b["cands"]
            target_idx = int(cands[int(rng.integers(len(cands)))])
            target_digit = int(b["answer"][target_idx])
            chosen.append({
                "item_id": f"{s}_{idx_pos:02d}",
                "stratum": s,
                "origin": b["origin"],
                "question": b["question"],
                "answer": b["answer"],
                "rating": b["rating"],
                "prop_solved": b["prop_solved"],
                "prop_rounds": b["prop_rounds"],
                "target_idx": target_idx,
                "target_digit": target_digit,
                "target_prop_round": int(b["cell_round_map"][target_idx]),
            })
        by_stratum[s] = chosen

    # Order canonically STRATA, and assign deterministic donor index (cyclic shift within stratum
    # choosing next item with a DIFFERENT target_digit so donor-shuffle never leaks target label)
    for s in STRATA:
        grp = by_stratum[s]
        n = len(grp)
        for i in range(n):
            donor = (i + 1) % n
            for off in range(1, n):
                cand = (i + off) % n
                if grp[cand]["target_digit"] != grp[i]["target_digit"]:
                    donor = cand
                    break
            grp[i]["donor_item_id"] = grp[donor]["item_id"]
            items.append(grp[i])

    payload = {
        "seed": seed,
        "per_stratum": per_stratum,
        "dataset_repo": DATASET_REPO,
        "sha256": EXPECTED_SHA256,
        "shots": shots,
        "items": items,
        "bench_boards": bench_boards,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    payload["bundle_sha256"] = digest
    return payload
