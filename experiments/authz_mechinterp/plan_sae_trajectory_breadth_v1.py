"""Q2 retained-row planner. Metadata only; never trains or launches compute."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json
from experiments.nanbeige_reasoning.positions import capture_positions

REGIONS = ("prompt", "early", "middle", "late")
DATA_SELECTION_SEED = 20260911


def validate_windows(row):
    prompt, generated = row["prompt_tokens"], row["generated_tokens"]
    if type(prompt) is not int or type(generated) is not int or prompt < 8 or generated < 1:
        raise ValueError("adequate integer trajectory lengths required")
    positions, regions = capture_positions(prompt, prompt+generated)
    if len(positions) != 32 or any(b-a != 8 for a, b in regions.values()):
        raise ValueError("four nonoverlapping eight-row windows required")
    if row["positions"] != positions or row["regions"] != {k: list(v) for k, v in regions.items()}:
        raise ValueError("positions/regions differ from frozen window recipe")
    if any(type(p) is not int for p in row["positions"]) or any(type(p) is not int for bounds in row["regions"].values() for p in bounds):
        raise ValueError("integer position metadata required")
    return regions


def build_plan(metadata):
    expected = metadata["expected_training_families"]
    seed = metadata["data_selection_seed"]
    records = metadata["records"]
    if not isinstance(expected, list) or len(expected) != 10 or any(not isinstance(f, str) or not f for f in expected) or len(set(expected)) != len(expected):
        raise ValueError("Q2 requires exactly ten explicit unique training families")
    if type(seed) is not int or seed < 0 or not records:
        raise ValueError("nonnegative selection seed and records required")
    ids, roles, training = set(), {}, {}
    for row in records:
        id, family, role = row["id"], row["family"], row["split"]
        if not isinstance(id, str) or not id or id in ids or not isinstance(family, str) or not family:
            raise ValueError("unique IDs and explicit families required")
        if role not in {"train", "eval", "discovery", "calibration", "validation"}:
            raise ValueError("unknown data role")
        ids.add(id)
        if family in roles and roles[family] != role:
            raise ValueError("family overlaps roles")
        roles[family] = role
        if role == "train":
            validate_windows(row)
            training.setdefault(family, []).append(row)
    if set(training) != set(expected) or any(len(rows) != 4 for rows in training.values()):
        raise ValueError("exactly four examples per declared training family required")
    narrow, broad, complete, choices = [], [], [], {}
    for family in sorted(expected):
        rows = sorted(training[family], key=lambda r: r["id"])
        chosen = min(rows, key=lambda r: (content_sha256({"seed": seed, "family": family, "id": r["id"]}), r["id"]))
        choices[family] = chosen["id"]
        for row in rows:
            for region in REGIONS:
                start, end = row["regions"][region]
                for position in range(start, end):
                    item = {"id": row["id"], "family": family, "position": position,
                            "region": region, "phase": "prompt" if region == "prompt" else "generated"}
                    complete.append(item)
                    if row["id"] == chosen["id"]:
                        narrow.append(item)
                    if position in (start, end-1):
                        broad.append(item)
    def describe(roster):
        return {"rows_per_pass": len(roster), "trajectory_count": len({r["id"] for r in roster}),
                "phase_counts_per_pass": dict(sorted(Counter(r["phase"] for r in roster).items())),
                "family_counts_per_pass": dict(sorted(Counter(r["family"] for r in roster).items())),
                "roster_sha256": content_sha256(roster), "roster": roster}
    arms = {"Q2-N": describe(narrow), "Q2-B": describe(broad)}
    for field in ("rows_per_pass", "phase_counts_per_pass", "family_counts_per_pass"):
        if arms["Q2-N"][field] != arms["Q2-B"][field]:
            raise ValueError("arm budgets differ")
    key = lambda r: (r["id"], r["position"])
    overlap = len(set(map(key, narrow)) & set(map(key, broad)))
    return {"data_selection_seed": seed, "narrow_trajectory_choices": choices, "arms": arms,
            "common_preprocessing_pool": {"rows_per_pass": len(complete), "roster_sha256": content_sha256(complete)},
            "shared_retained_rows_per_pass": overlap,
            "selection_rule": "Minimum SHA-256 of canonical seed/family/id object, independent of metadata order and activation values.",
            "alignment_requirement": "Use each roster unchanged at all sites and on both aligned passes; future array checks still required."}


def synthetic_metadata():
    expected = [f"synthetic-family-{i:02d}" for i in range(10)]
    records = []
    for family in expected:
        for example in range(4):
            prompt, generated = 100+example, 64+example*8
            positions, regions = capture_positions(prompt, prompt+generated)
            records.append({"id": f"{family}/example-{example}", "family": family, "split": "train",
                            "prompt_tokens": prompt, "generated_tokens": generated, "positions": positions,
                            "regions": {k: list(v) for k, v in regions.items()}})
    records.append({"id": "synthetic-holdout/example-0", "family": "synthetic-holdout", "split": "eval"})
    return {"expected_training_families": expected, "data_selection_seed": DATA_SELECTION_SEED, "records": records}


def build_report(metadata_path=None):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path else synthetic_metadata()
    source = {"kind": "metadata_file", "path": str(metadata_path.resolve()), "sha256": sha256_file(metadata_path)} if metadata_path else {"kind": "synthetic_demo", "fresh_pack": False}
    body = {"schema": "sch.vpd-sae-trajectory-breadth-plan.v1", "candidate": "Q2-v1", "source": source,
            "plan": build_plan(metadata), "metadata_content_sha256": content_sha256(metadata),
            "boundary": {"training_performed": False, "activation_arrays_loaded": False, "freshness_verified": False,
                         "launch_authorized": False, "initialization_equality_verified": False},
            "provenance": {"runner_sha256": sha256_file(Path(__file__)),
                           "window_recipe_sha256": sha256_file(Path(__file__).parents[1] / "nanbeige_reasoning/positions.py")}}
    return {**body, "report_id": content_sha256(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--metadata", type=Path)
    source.add_argument("--synthetic-demo", action="store_true")
    source.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.verify:
        if args.output:
            parser.error("--verify is read-only")
        observed = json.loads(args.verify.read_text(encoding="utf-8"))
        kind = observed["source"]["kind"]
        if kind not in {"synthetic_demo", "metadata_file"}:
            raise ValueError("unknown receipt source")
        report = build_report(Path(observed["source"]["path"]) if kind == "metadata_file" else None)
        if report != observed:
            raise ValueError("Q2 row plan does not reproduce")
    else:
        if not args.output:
            parser.error("--output required")
        report = build_report(args.metadata)
        write_json(args.output, report)
    print(json.dumps({"report_id": report["report_id"], "source_kind": report["source"]["kind"],
                      "arms": {name: {k: v for k, v in arm.items() if k != "roster"} for name, arm in report["plan"]["arms"].items()},
                      "shared_retained_rows_per_pass": report["plan"]["shared_retained_rows_per_pass"]}, indent=2))


if __name__ == "__main__":
    main()
