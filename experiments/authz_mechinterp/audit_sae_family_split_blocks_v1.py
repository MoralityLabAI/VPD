"""Conservative family split-block review, not an independence certificate."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from experiments.authz_mechinterp import audit_sae_source_family_inventory_v1 as inventory
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json

BLOCKS = {
    "bracket_matching": ("dyck_language", "dyck_language_errors"),
    "lexical_ordering": ("word_sorting", "word_sorting_mistake"),
}
EVIDENCE = {
    "dyck_language": "intellect3_logic/05645/a9b60f7bd14a",
    "dyck_language_errors": "intellect3_logic/06128/236238adf018",
    "word_sorting": "intellect3_logic/07442/3af82de403ea",
    "word_sorting_mistake": "intellect3_logic/08260/4e9078710573",
}
NQUEENS_SHA = "83dfb8ba2381961e7c38b0f10434155ac049cc52f6a43cf94c3cde1552096174"
NQUEENS = Path("E:/Recovered_C_projects/Remote_clones/prime_intellect_research_environments/environments/reasoning/prolog/prolog/generators/nqueens.py")


def question_digest(prompt):
    # dataset_v1 hashes a canonical JSON string, not its raw UTF-8 contents.
    encoded = json.dumps(prompt, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def split_block(family):
    for block, members in BLOCKS.items():
        if family in members:
            return "reviewed:" + block
    return "unreviewed:" + family


def validate_role_blocks(records):
    """Enforce the two reviewed co-assignment rules; unknown families not certified."""
    roles = {}
    for row in records:
        family, role = row["family"], row["split"]
        if not isinstance(family, str) or not family or role not in {"train", "eval", "discovery", "calibration", "validation"}:
            raise ValueError("explicit family and supported experimental role required")
        block = split_block(family)
        if block in roles and roles[block] != role:
            raise ValueError(f"related-family split block crosses roles: {block}")
        roles[block] = role
    return roles


def nqueens_public_domain(path):
    if sha256_file(path) != NQUEENS_SHA:
        raise ValueError("reviewed N-Queens source changed")
    # Read literal bounds without importing or executing recovered source.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NQueensGenerator")
    assignment = next(n for n in cls.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "DIFFICULTY_PARAMS" for t in n.targets))
    bounds = ast.literal_eval(assignment.value)
    domains = {k: list(range(lo, hi+1)) for k, (lo, hi) in bounds.items()}
    return {"source_sha256": NQUEENS_SHA, "public_board_sizes_by_difficulty": domains,
            "distinct_problems_per_difficulty": {k: len(v) for k, v in domains.items()},
            "distinct_board_sizes_all_difficulties": len(set().union(*map(set, domains.values()))),
            "interpretation": "Reviewed generate method randomizes only board size; reference search is deterministic. Seeds and difficulty labels are not additional public constraints.",
            "four_distinct_examples_at_one_declared_difficulty_possible": all(len(v) >= 4 for v in domains.values())}


def build_report(directory, nqueens_path):
    source = inventory.build_report(directory)
    evidence = {}
    with (directory / "questions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] not in EVIDENCE.values():
                continue
            if row["split"] != "dev" or EVIDENCE[row["family"]] != row["id"]:
                raise ValueError("development evidence identity differs")
            if question_digest(row["prompt"]) != row["prompt_sha256"]:
                raise ValueError("development prompt hash differs")
            evidence[row["family"]] = {k: row[k] for k in ("id", "family", "split", "prompt_sha256", "source_revision")}
    if set(evidence) != set(EVIDENCE):
        raise ValueError("missing reviewed development evidence")
    labels = source["inventory"]["eligible_family_labels_any_allowed_split"]
    membership = {f: split_block(f) for f in labels}
    body = {"schema": "sch.vpd-sae-family-split-block-review.v1",
            "source": source["source"], "source_inventory_report_id": source["report_id"],
            "reviewed_blocks": {k: list(v) for k, v in BLOCKS.items()}, "development_evidence": evidence,
            "eligible_label_to_split_block": membership, "eligible_labels": len(labels),
            "split_blocks_after_two_conservative_merges": len(set(membership.values())),
            "remaining_unreviewed_singleton_blocks": sum(v.startswith("unreviewed:") for v in membership.values()),
            "nqueens_source_path": str(nqueens_path.resolve()), "nqueens": nqueens_public_domain(nqueens_path),
            "limitations": ["Co-assignment is a conservative design judgment based on public development task semantics, not proof of identical mechanisms or statistical dependence.",
                            "Singleton blocks are unreviewed; the block count is an upper bound under these rules, not a count of admitted independent families.",
                            "No final-evaluation prompts or graders were selected for semantic review; metadata inventory reads the existing corpus.",
                            "N-Queens has nine board sizes across all three difficulties, so the single-difficulty shortfall does not rule out four sizes across difficulties. Freshness and near-duplicates remain unverified.",
                            "The role checker is a reusable component, not integrated into the sealed protocol or existing row planner."],
            "boundary": {"new_families_admitted": 0, "fresh_pack_created": False, "recovery_code_executed": False,
                         "model_inference_performed": False, "training_performed": False, "launch_authorized": False},
            "runner_sha256": sha256_file(Path(__file__))}
    return {**body, "report_id": content_sha256(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-directory", type=Path)
    parser.add_argument("--nqueens-source", type=Path, default=NQUEENS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        observed = json.loads(args.verify.read_text(encoding="utf-8"))
        report = build_report(Path(observed["source"]["path"]), Path(observed["nqueens_source_path"]))
        if report != observed:
            raise ValueError("family split-block review does not reproduce")
    else:
        if not args.dataset_directory or args.output.exists():
            parser.error("dataset required and output must not already exist")
        report = build_report(args.dataset_directory, args.nqueens_source)
        write_json(args.output, report)
    print(json.dumps({k: report[k] for k in ("report_id", "eligible_labels", "split_blocks_after_two_conservative_merges", "remaining_unreviewed_singleton_blocks", "nqueens")}, indent=2))


if __name__ == "__main__":
    main()
