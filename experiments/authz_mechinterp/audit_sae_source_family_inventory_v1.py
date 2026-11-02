"""Question-only local family inventory; no fresh-holdout certification."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json

ELIGIBLE_SPLITS = {"train", "dev", "eval"}


def inventory(rows, minimum_groups=4):
    if type(minimum_groups) is not int or minimum_groups < 1:
        raise ValueError("positive integer group requirement required")
    counts, groups, revisions = Counter(), defaultdict(set), defaultdict(set)
    ids = set()
    for row in rows:
        for key in ("id", "family", "split", "group_id", "prompt_sha256", "source_revision"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"missing string metadata: {key}")
        if row["id"] in ids:
            raise ValueError("duplicate question ID")
        ids.add(row["id"])
        if row["split"] not in ELIGIBLE_SPLITS | {"quarantine", "historical_eval"}:
            raise ValueError("unknown source split")
        counts[row["family"], row["split"]] += 1
        groups[row["family"], row["split"]].add(row["group_id"])
        revisions[row["family"]].add(row["source_revision"])
    families = sorted(revisions)
    if not families:
        raise ValueError("empty source corpus")
    entries = [{"family": family, "source_revisions": sorted(revisions[family]),
                "splits": {split: {"rows": counts[family, split], "unique_declared_groups": len(groups[family, split])}
                           for split in sorted(ELIGIBLE_SPLITS | {"quarantine", "historical_eval"}) if counts[family, split]}}
               for family in families]
    eligible_by_split = {split: [family for family in families if len(groups[family, split]) >= minimum_groups]
                         for split in sorted(ELIGIBLE_SPLITS)}
    eligible_any = sorted(set().union(*map(set, eligible_by_split.values())))
    return {"rows": len(ids), "source_family_labels": len(families), "families": entries,
            "minimum_unique_declared_groups_per_role": minimum_groups,
            "eligible_families_by_source_split": eligible_by_split,
            "eligible_family_labels_any_allowed_split": eligible_any,
            "excluded_only_or_insufficient_family_labels": sorted(set(families)-set(eligible_any)),
            "F1_necessary_family_count_check": {
                "families_per_pack_required": 32, "packs_required": 2, "total_pairwise_disjoint_labels_required": 64,
                "available_label_upper_bound": len(eligible_any),
                "single_pack_count_possible": len(eligible_any) >= 32,
                "two_pack_count_possible": len(eligible_any) >= 64,
                "is_sufficient_feasibility_test": False}}


def build_report(directory):
    manifest_path, questions = directory / "manifest.json", directory / "questions.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "nanbeige.reasoning.dataset.v1" or sha256_file(questions) != manifest["files"]["questions.jsonl"]:
        raise ValueError("question corpus integrity differs")
    with questions.open(encoding="utf-8") as stream:
        result = inventory(json.loads(line) for line in stream if line.strip())
    if result["rows"] != manifest["records"]:
        raise ValueError("record count differs from manifest")
    body = {"schema": "sch.vpd-sae-source-family-inventory.v1", "inventory": result,
            "source": {"path": str(directory.resolve()), "manifest_sha256": sha256_file(manifest_path),
                       "questions_sha256": sha256_file(questions)},
            "limitations": ["A distinct source label is not proof of independent semantic families; this is an upper-bound count.",
                            "Group IDs are declared source grouping, not a new near-duplicate or contamination audit.",
                            "Only this local corpus is inventoried; other disks or datasets are not ruled out.",
                            "Historical exposure across prior runs and pretraining is not certified absent.",
                            "No family roles, examples or held-out pack were selected; graders were not read."],
            "boundary": {"fresh_pack_created": False, "training_performed": False, "model_inference_performed": False, "launch_authorized": False},
            "runner_sha256": sha256_file(Path(__file__))}
    return {**body, "report_id": content_sha256(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-directory", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        observed = json.loads(args.verify.read_text(encoding="utf-8"))
        report = build_report(Path(observed["source"]["path"]))
        if report != observed:
            raise ValueError("source inventory does not reproduce")
    else:
        if not args.dataset_directory:
            parser.error("--dataset-directory required")
        report = build_report(args.dataset_directory)
        write_json(args.output, report)
    inv = report["inventory"]
    print(json.dumps({"report_id": report["report_id"], "rows": inv["rows"], "family_labels": inv["source_family_labels"],
                      "eligible_by_split": {s: len(f) for s, f in inv["eligible_families_by_source_split"].items()},
                      "count_check": inv["F1_necessary_family_count_check"],
                      "excluded_only_or_insufficient": inv["excluded_only_or_insufficient_family_labels"]}, indent=2))


if __name__ == "__main__":
    main()
