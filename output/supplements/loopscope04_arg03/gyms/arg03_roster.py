#!/usr/bin/env python3
"""Fresh ALG/ARG-03 roster export and independent EvalMath grade; no GPU required.

The model runner sees only public *.jsonl. Regeneration of hidden cases
occurs here, in the oracle-only process. Test graphs use an operator family
held out from dev (final operator R). All prior EvalMath/ARG-02 signatures
are excluded. Unlike ARG-02, this initial ARG-03 roster is positive-only.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import product
import json
from math import comb
from pathlib import Path
import random
import sys

import numpy as np

from .core import sha256_file, write_json

DEV_SEED = 2026111503
TEST_SEED = 2026111504
DEV_PER_FAMILY = 32
TEST_PER_FAMILY = 24
OPS = "INSR"


def oracle_context(evalmath_root: Path, compactification_root: Path):
    em_root = evalmath_root.resolve()
    em_src = em_root / "src"
    if (em_src / "spiral_ln").is_dir():
        sys.path.insert(0, str(em_src))
    sys.path.insert(0, str(em_root))
    sys.path.insert(0, str(compactification_root.resolve()))
    from spiral_ln import algebraic_reasoning_oracle as oracle
    from research.algernon.algebraic_reasoning_gym.arg02 import splits as arg02_splits

    return oracle, arg02_splits


def manifest_hash(data) -> str:
    return sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cases(oracle, arg02_splits):
    evalmath_sigs = {c.signature for seq in oracle.roster().values() for c in seq}
    arg02_dev_sigs = {c.signature for c in arg02_splits.dev_cases()}
    arg02_shot_sigs = {c.signature for c in arg02_splits.shot_cases()}
    forbidden = set(evalmath_sigs) | set(arg02_dev_sigs) | set(arg02_shot_sigs)
    out = {}
    plans = (
        ("development", DEV_SEED, oracle.CAL_FAMILIES, DEV_PER_FAMILY, False),
        ("test", TEST_SEED, oracle.ALL_FAMILIES, TEST_PER_FAMILY, True),
    )
    for name, seed, families, n_each, held_out in plans:
        group = []
        for family_i, family in enumerate(families):
            rng = random.Random(seed + 7919 * family_i)
            for trial in range(1, n_each * 1000 + 1):
                if len([c for c in group if c.family == family]) >= n_each:
                    break
                topology = rng.choice(oracle.TOPOLOGIES)
                graphs = [p for p in oracle.PROGRAMS if (p[2] == "R") == held_out]
                graph = rng.choice(graphs)
                c = oracle.make_case(
                    rng,
                    split=f"arg03_{name}",
                    family=family,
                    topology=topology,
                    graph=graph,
                    identifier=f"arg03:{name}:{family}:{trial:05d}",
                )
                if c.signature in forbidden:
                    continue
                forbidden.add(c.signature)
                group.append(c)
            if sum(c.family == family for c in group) < n_each:
                raise RuntimeError(f"insufficient fresh cases: {name}/{family}")
        out[name] = tuple(group)
    if {tuple(c.source_graph) for c in out["development"]} & {tuple(c.source_graph) for c in out["test"]}:
        raise AssertionError("development/test program-graph overlap")
    return out


def public(c) -> dict:
    data = dict(c.public)
    data["public_sha256"] = c.signature
    return data


def prepare(args):
    oracle, split = oracle_context(args.evalmath_root, args.compactification_root)
    roster = cases(oracle, split)
    evalmath_sigs = {c.signature for seq in oracle.roster().values() for c in seq}
    arg02_dev_sigs = {c.signature for c in split.dev_cases()}
    arg02_shot_sigs = {c.signature for c in split.shot_cases()}
    dev_sigs = [c.signature for c in roster["development"]]
    test_sigs = [c.signature for c in roster["test"]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, arr in roster.items():
        path = args.out_dir / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(public(c), sort_keys=True, separators=(",", ":")) + "\n" for c in arr),
            encoding="utf-8",
        )
        digests[name] = sha256_file(path)
    audit = {
        "schema": "darkfibers.arg03.new_roster.v2",
        "seeds": {"development": DEV_SEED, "test": TEST_SEED},
        "counts": {key: len(value) for key, value in roster.items()},
        "duplicate_signatures": {
            "development_internal": len(dev_sigs) - len(set(dev_sigs)),
            "test_internal": len(test_sigs) - len(set(test_sigs)),
        },
        "contamination_overlaps": {
            "development_vs_test_signatures": len(set(dev_sigs) & set(test_sigs)),
            "development_vs_evalmath": len(set(dev_sigs) & evalmath_sigs),
            "test_vs_evalmath": len(set(test_sigs) & evalmath_sigs),
            "development_vs_arg02_dev": len(set(dev_sigs) & arg02_dev_sigs),
            "test_vs_arg02_dev": len(set(test_sigs) & arg02_dev_sigs),
            "development_vs_arg02_shots": len(set(dev_sigs) & arg02_shot_sigs),
            "test_vs_arg02_shots": len(set(test_sigs) & arg02_shot_sigs),
            "development_vs_test_source_graphs": len(
                {tuple(c.source_graph) for c in roster["development"]}
                & {tuple(c.source_graph) for c in roster["test"]}
            ),
        },
        "public_sha256": digests,
        "model_input_fields": sorted(public(roster["test"][0])),
        "families": {"development": list(oracle.CAL_FAMILIES), "test": list(oracle.ALL_FAMILIES)},
        "held_out_program_rule": "test final operator R; development final operator not-R",
        "forbidden": "all EvalMath oracle.roster signatures and ARG-02 dev/shots",
        "oracle_code_sha256": sha256_file(Path(oracle.__file__)),
        "arg02_splits_sha256": sha256_file(Path(split.__file__)),
        "negative_controls": "not included; prepare a separate registered negative/collision suite",
    }
    if any(v != 0 for v in audit["duplicate_signatures"].values()) or any(
        v != 0 for v in audit["contamination_overlaps"].values()
    ):
        raise AssertionError(f"roster leakage detected: {audit}")
    write_json(args.out_dir / "audit.json", audit)
    print(json.dumps(audit, indent=2))
    return 0


def exact_mcnemar(a: list[bool], b: list[bool]) -> dict:
    gain = sum(x and not y for x, y in zip(a, b))
    loss = sum(y and not x for x, y in zip(a, b))
    n = gain + loss
    p = min(1.0, 2.0 * sum(comb(n, k) for k in range(min(gain, loss) + 1)) / (2**n)) if n else 1.0
    return {"gain": gain, "loss": loss, "paired_diff": gain - loss, "exact_two_sided_p": p}


def _op_subtensor_program_scores(op_logits_16x3x4: list, temps: tuple[float, float, float]) -> list[float]:
    """Score all 64 programs under per-position operator-restricted temperature/affine weights."""
    arr = np.asarray(op_logits_16x3x4, dtype=np.float64)  # [16, 3, 4]
    t_vec = np.asarray(temps, dtype=np.float64).reshape(1, 3, 1)
    scaled = arr / t_vec
    # Restricted log-softmax over the 4 valid operator tokens at each site
    m = scaled.max(axis=-1, keepdims=True)
    lp = (scaled - m) - np.log(np.exp(scaled - m).sum(axis=-1, keepdims=True))
    out = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                out.append(float(lp[0, 0, i] + lp[4 * i, 1, j] + lp[4 * i + j, 2, k]))
    return out


def _score_order(oracle, c, ordered: list[tuple[str, str, str]], scores: list[float]) -> tuple[bool, bool, bool]:
    if len(scores) != 64:
        raise ValueError("missing program scores")
    order = [ordered[i] for i in sorted(range(64), key=lambda i: (-scores[i], i))]
    search12 = oracle.search(c, order, budget=12)
    ver12 = oracle.grade(c, search12)
    search1 = oracle.search(c, order[:1], budget=1)
    ver1 = oracle.grade(c, search1)
    return bool(ver12["future_correct"]), bool(ver12["accepted"]), bool(ver1["future_correct"])


def _fit_dev_calibrations(oracle, ordered, tape: list[dict], case_map: dict) -> dict:
    """Fit single-pass scalar temperature and per-position operator temperatures strictly on development."""
    best_scalar_key = "T1"
    best_scalar_hits = -1
    grid_keys = sorted(
        tape[0].get("temperature_grid", tape[0].get("temperature_controls", {})).keys(),
        key=lambda k: float(k[1:]),
    )
    scalar_table = {}
    for tk in grid_keys:
        hits = 0
        for rec in tape:
            c = case_map[rec["case_id"]]
            src = rec.get("temperature_grid") or rec.get("temperature_controls") or {}
            if tk not in src:
                continue
            fc, _, _ = _score_order(oracle, c, ordered, src[tk])
            hits += int(fc)
        scalar_table[tk] = hits
        if hits > best_scalar_hits:
            best_scalar_hits = hits
            best_scalar_key = tk

    pos_candidates = (0.35, 0.5, 0.75, 1.0, 1.5, 2.0)
    best_pos_temps = (1.0, 1.0, 1.0)
    best_pos_hits = -1
    if "loop1_operator_site_logits" in tape[0]:
        for t1, t2, t3 in product(pos_candidates, repeat=3):
            hits = 0
            for rec in tape:
                c = case_map[rec["case_id"]]
                sc = _op_subtensor_program_scores(rec["loop1_operator_site_logits"], (t1, t2, t3))
                fc, _, _ = _score_order(oracle, c, ordered, sc)
                hits += int(fc)
            if hits > best_pos_hits:
                best_pos_hits = hits
                best_pos_temps = (float(t1), float(t2), float(t3))

    # Also compute median top1-vs-top2 program score margin at loops 1, 2, 4, 8 on development
    margin_by_depth = {}
    for d in ("1", "2", "4", "8"):
        margins = []
        for rec in tape:
            if d in rec.get("baseline", {}):
                s_sorted = sorted(rec["baseline"][d], reverse=True)
                margins.append(s_sorted[0] - s_sorted[1])
        if margins:
            margin_by_depth[d] = float(np.median(margins))

    return {
        "schema": "darkfibers.arg03.dev_calibration_lock.v1",
        "fitted_on_split": "development",
        "rows_fitted": len(tape),
        "selected_full_vocab_scalar_temp": best_scalar_key,
        "dev_scalar_temp_future12_counts": scalar_table,
        "selected_per_position_op_temps": list(best_pos_temps),
        "dev_per_position_op_future12": int(best_pos_hits),
        "dev_median_top1_top2_margin_by_depth": margin_by_depth,
    }


def grade(args):
    if args.fit_dev_calibration and args.split != "development":
        raise RuntimeError("calibration fitting is strictly forbidden outside the development split")

    oracle, split = oracle_context(args.evalmath_root, args.compactification_root)
    roster = cases(oracle, split)
    audit = json.loads((args.split_dir / "audit.json").read_text(encoding="utf-8"))
    expected = {"development": DEV_SEED, "test": TEST_SEED}
    if audit["seeds"] != expected:
        raise AssertionError("split seeds drift")
    if audit["oracle_code_sha256"] != sha256_file(Path(oracle.__file__)):
        raise AssertionError("oracle changed since export")
    if audit["arg02_splits_sha256"] != sha256_file(Path(split.__file__)):
        raise AssertionError("forbidden-signature generator changed since export")
    if audit["public_sha256"][args.split] != sha256_file(args.split_dir / f"{args.split}.jsonl"):
        raise AssertionError("public rows changed")

    ordered = list(oracle.PROGRAMS)
    canonical = [(a, b, c) for a in OPS for b in OPS for c in OPS]
    if ordered != canonical:
        raise AssertionError("program ordering differs from model score order")
    case_map = {c.identifier: c for c in roster[args.split]}
    tape = [json.loads(line) for line in args.tape.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({r["case_id"] for r in tape}) != len(tape):
        raise ValueError("duplicate model results")
    if args.full and set(r["case_id"] for r in tape) != set(case_map):
        raise ValueError("full held-out evaluation needs every case")

    scored_arm_groups = (
        "baseline",
        "context_swap",
        "random_context_norm_matched",
        "self_context_control",
        "donor_clean_perturbed",
        "open_loop",
        "equal_compute",
        "temperature_controls",
        "temperature_grid",
    )

    per_arm_fc: dict[tuple[str, str], dict[str, bool]] = {}
    per_arm_acc: dict[tuple[str, str], dict[str, bool]] = {}
    per_arm_top1: dict[tuple[str, str], dict[str, bool]] = {}

    cal_lock = None
    if args.fit_dev_calibration:
        cal_lock = _fit_dev_calibrations(oracle, ordered, tape, case_map)
        write_json(args.fit_dev_calibration, cal_lock)
    elif args.apply_dev_calibration:
        cal_lock = json.loads(args.apply_dev_calibration.read_text(encoding="utf-8"))

    for rec in tape:
        cid = rec["case_id"]
        c = case_map[cid]
        if rec["public_sha256"] != c.signature:
            raise AssertionError("case identity mismatch")
        for arm in scored_arm_groups:
            for depth, scores in rec.get(arm, {}).items():
                fc, acc, top1 = _score_order(oracle, c, ordered, scores)
                per_arm_fc.setdefault((arm, depth), {})[cid] = fc
                per_arm_acc.setdefault((arm, depth), {})[cid] = acc
                per_arm_top1.setdefault((arm, depth), {})[cid] = top1
        if cal_lock and "loop1_operator_site_logits" in rec:
            pt = tuple(cal_lock["selected_per_position_op_temps"])
            sc_cal = _op_subtensor_program_scores(rec["loop1_operator_site_logits"], pt)
            fc, acc, top1 = _score_order(oracle, c, ordered, sc_cal)
            per_arm_fc.setdefault(("dev_calibrated_loop1", "per_pos_op"), {})[cid] = fc
            per_arm_acc.setdefault(("dev_calibrated_loop1", "per_pos_op"), {})[cid] = acc
            per_arm_top1.setdefault(("dev_calibrated_loop1", "per_pos_op"), {})[cid] = top1

    summary = {
        "schema": "darkfibers.arg03.graded.v2",
        "split": args.split,
        "test_exploratory": not args.full,
        "rows_scored": len(tape),
        "tape_sha256": sha256_file(args.tape),
        "results": {},
    }
    if cal_lock is not None:
        summary["dev_calibration"] = cal_lock

    baseline1 = per_arm_fc.get(("baseline", "1"), {})
    baseline2 = per_arm_fc.get(("baseline", "2"), {})
    for (arm, depth), row in sorted(per_arm_fc.items()):
        ids = sorted(row)
        info = {
            "correct": sum(row.values()),
            "accepted_at_12": sum(per_arm_acc[(arm, depth)].values()),
            "top1_future_correct": sum(per_arm_top1[(arm, depth)].values()),
            "denominator": len(ids),
        }
        if (arm, depth) != ("baseline", "1") and baseline1 and set(ids) == set(baseline1):
            info["vs_baseline1"] = exact_mcnemar([row[c] for c in ids], [baseline1[c] for c in ids])
        if (arm, depth) != ("baseline", "2") and baseline2 and set(ids) == set(baseline2):
            info["vs_baseline2"] = exact_mcnemar([row[c] for c in ids], [baseline2[c] for c in ids])
        base_same = per_arm_fc.get(("baseline", depth))
        if arm != "baseline" and base_same and set(ids) == set(base_same):
            info["vs_baseline_same_depth"] = exact_mcnemar([row[c] for c in ids], [base_same[c] for c in ids])
        rand_same = per_arm_fc.get(("random_context_norm_matched", depth))
        if arm == "context_swap" and rand_same and set(ids) == set(rand_same):
            info["vs_random_context_same_depth"] = exact_mcnemar([row[c] for c in ids], [rand_same[c] for c in ids])
        summary["results"][f"{arm}@{depth}"] = info

    write_json(args.out, summary)
    print(json.dumps(summary, indent=2))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "grade"))
    p.add_argument("--evalmath-root", type=Path, required=True)
    p.add_argument("--compactification-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, help="Directory for new public split and audit")
    p.add_argument("--split-dir", type=Path, help="Existing audited public split")
    p.add_argument("--split", choices=("development", "test"), default="development")
    p.add_argument("--tape", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--fit-dev-calibration", type=Path, help="Write dev-fitted calibration JSON (development only)")
    p.add_argument("--apply-dev-calibration", type=Path, help="Apply frozen dev calibration JSON")
    p.add_argument("--full", action="store_true", help="Require complete split; appropriate for final test")
    args = p.parse_args(argv)
    if args.mode == "prepare":
        if args.out_dir is None:
            p.error("prepare requires --out-dir")
        return prepare(args)
    if any(v is None for v in (args.split_dir, args.tape, args.out)):
        p.error("grade requires --split-dir --tape --out")
    return grade(args)


if __name__ == "__main__":
    raise SystemExit(main())
