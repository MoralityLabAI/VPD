"""Public-clue duplicate classes under the eight rectangle grid symmetries."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json
from experiments.authz_mechinterp.nonogram_text_adapter_v1 import public_constraints


def transform_clues(constraints, *, transpose=False, flip_rows=False, flip_cols=False):
    if any(type(v) is not bool for v in (transpose, flip_rows, flip_cols)):
        raise ValueError("explicit Boolean symmetry flags required")
    c = public_constraints(constraints)
    rows, cols = c["num_rows"], c["num_cols"]
    rc, cc = c["row_clues"], c["col_clues"]
    if transpose:
        rows, cols, rc, cc = cols, rows, cc, rc
    if flip_rows:
        rc, cc = rc[::-1], [line[::-1] for line in cc]
    if flip_cols:
        rc, cc = [line[::-1] for line in rc], cc[::-1]
    return {"num_rows": rows, "num_cols": cols, "row_clues": rc, "col_clues": cc}


def symmetry_class(constraints):
    orbit = {json.dumps(transform_clues(constraints, transpose=t, flip_rows=r, flip_cols=c),
                        sort_keys=True, separators=(",", ":")) for t, r, c in product((False, True), repeat=3)}
    canonical = json.loads(min(orbit))
    return {"canonical_constraints": canonical, "distinct_orientations": len(orbit),
            "symmetry_class_sha256": content_sha256({"task": "nonogram", "equivalence": "grid_dihedral_v1", **canonical})}


def audit_questions(questions):
    if not isinstance(questions, list) or not questions:
        raise ValueError("nonempty question list required")
    seen, groups, entries = set(), {}, []
    for q in questions:
        if (q.get("schema") != "sch.vpd-nonogram-text-question.v1" or q.get("family") != "nonogram"
                or not isinstance(q.get("id"), str) or not q["id"] or q["id"] in seen):
            raise ValueError("unique explicit nonogram question identity required")
        seen.add(q["id"])
        c = public_constraints(q["constraints"])
        if q["public_instance_sha256"] != content_sha256({"task": "nonogram", **c}):
            raise ValueError("public clue fingerprint differs")
        result = symmetry_class(c)
        groups.setdefault(result["symmetry_class_sha256"], []).append(q["id"])
        entries.append({"id": q["id"], "public_instance_sha256": q["public_instance_sha256"], **result})
    return {"questions": len(questions), "symmetry_classes": len(groups),
            "duplicate_classes": [{"symmetry_class_sha256": k, "ids": sorted(v)} for k, v in sorted(groups.items()) if len(v) > 1],
            "entries": sorted(entries, key=lambda e: e["id"])}


def build_report(source_path):
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (source.get("schema") != "sch.vpd-nonogram-generator-canary.v1"
            or source.get("report_id") != content_sha256({k: v for k, v in source.items() if k != "report_id"})):
        raise ValueError("source canary receipt integrity differs")
    questions_path = source_path.parent / "questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if (content_sha256(questions) != source["questions_content_sha256"]
            or any(q.get("role") != "development_only" for q in questions)
            or sorted(q["id"] for q in questions) != sorted(r["id"] for r in source["checks"])):
        raise ValueError("source development question inventory differs")
    body = {"schema": "sch.vpd-nonogram-symmetry-duplicate-audit.v1", "result": audit_questions(questions),
            "source": {"path": str(source_path.resolve()), "sha256": sha256_file(source_path),
                       "report_id": source["report_id"], "questions_sha256": sha256_file(questions_path)},
            "limitations": ["Only rotations, reflections and transposition of ordered public clues are canonicalized; not every possible task equivalence or near-duplicate.",
                            "No grid reference is read, no solver runs and no uniqueness or difficulty claim is made.",
                            "Distinct symmetry classes are instance-level evidence, not independent task families or certification against historical/model exposure.",
                            "The saved source receipt and question hashes are checked; the generator is not re-executed in this audit.",
                            "All inputs are exposed development puzzles, not a fresh final evaluation pack."],
            "boundary": {"generator_executed": False, "private_graders_read": False, "model_inference_performed": False,
                         "training_performed": False, "additional_families_admitted": 0, "launch_authorized": False},
            "provenance": {"runner_sha256": sha256_file(Path(__file__)),
                           "adapter_sha256": sha256_file(Path(__file__).with_name("nonogram_text_adapter_v1.py"))}}
    return {**body, "report_id": content_sha256(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        observed = json.loads(args.verify.read_text(encoding="utf-8"))
        report = build_report(Path(observed["source"]["path"]))
        if observed != report:
            raise ValueError("symmetry audit does not reproduce")
    else:
        if not args.source or args.output.exists():
            parser.error("source required; refuse existing output")
        report = build_report(args.source)
        write_json(args.output, report)
    print(json.dumps({"report_id": report["report_id"], **{k: v for k, v in report["result"].items() if k != "entries"}}, indent=2))


if __name__ == "__main__":
    main()
