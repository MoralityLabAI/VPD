from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.activation_categories import (
    activation_category,
    category_contract,
)
from experiments.authz_mechinterp.build_activation_category_splits import (
    DEFAULT_SPLIT_WEIGHTS,
    SPLIT_ROLES,
    _allocate_counts,
    _pair_records,
    _split_summary,
    parse_split_weights,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    load_jsonl,
    sha256_file,
    stable_hash,
    write_json,
    write_jsonl,
)


SCHEMA = "sch.vpd-activation-category-split-manifest.v2"


def assign_template_roles(
    rows: list[dict[str, Any]], *, seed: int, weights: dict[str, int]
) -> dict[str, str]:
    _, pair_records = _pair_records(rows)
    templates = sorted({str(record["template_group"]) for record in pair_records.values()})
    counts = _allocate_counts(len(templates), weights)
    ordered = sorted(
        templates,
        key=lambda family: (
            stable_hash(seed, "activation-category-semantic-split", family),
            family,
        ),
    )
    assignments: dict[str, str] = {}
    offset = 0
    for role in SPLIT_ROLES:
        for family in ordered[offset : offset + counts[role]]:
            assignments[family] = role
        offset += counts[role]
    if set(assignments) != set(templates):
        raise ValueError("semantic split assignment does not cover every template group")
    return assignments


def build_semantic_split_bundle(
    rows: list[dict[str, Any]],
    *,
    source_sha256: str,
    output_dir: Path,
    seed: int,
    weights: dict[str, int] | None = None,
    minimum_episodes_per_category_per_split: int = 8,
) -> dict[str, Any]:
    normalized_weights = {
        role: int((weights or DEFAULT_SPLIT_WEIGHTS)[role]) for role in SPLIT_ROLES
    }
    if set(normalized_weights) != set(SPLIT_ROLES) or any(
        value < 1 for value in normalized_weights.values()
    ):
        raise ValueError("semantic split weights are incomplete or invalid")
    grouped, pair_records = _pair_records(rows)
    template_roles = assign_template_roles(rows, seed=seed, weights=normalized_weights)
    pair_roles = {
        pair_id: template_roles[str(record["template_group"])]
        for pair_id, record in pair_records.items()
    }
    role_rows = {
        role: sorted(
            [
                row
                for pair_id, pair_rows in grouped.items()
                if pair_roles[pair_id] == role
                for row in pair_rows
            ],
            key=lambda row: str(row["episode_id"]),
        )
        for role in SPLIT_ROLES
    }
    observed_categories = sorted({activation_category(row) for row in rows})
    observed_templates = sorted({str(row["template_group"]) for row in rows})
    if len(observed_templates) < len(SPLIT_ROLES):
        raise ValueError("semantic split needs at least one template group per role")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_entries: dict[str, dict[str, Any]] = {}
    episode_sets: dict[str, set[str]] = {}
    pair_sets: dict[str, set[str]] = {}
    template_sets: dict[str, set[str]] = {}
    for role in SPLIT_ROLES:
        split_rows = role_rows[role]
        summary = _split_summary(split_rows)
        if set(summary["activation_category_counts"]) != set(observed_categories):
            raise ValueError(f"semantic split omits an activation category: {role}")
        if min(summary["activation_category_counts"].values()) < (
            minimum_episodes_per_category_per_split
        ):
            raise ValueError(f"semantic split has insufficient category support: {role}")
        role_templates = set(summary["template_group_counts"])
        if not role_templates:
            raise ValueError(f"semantic split has no template groups: {role}")
        artifact = output_dir / f"{role}_episodes.jsonl"
        write_jsonl(artifact, split_rows)
        split_entries[role] = {
            "artifact": {
                "path": artifact.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(artifact),
            },
            **summary,
            "template_group_set_sha256": content_sha256(sorted(role_templates)),
        }
        episode_sets[role] = {str(row["episode_id"]) for row in split_rows}
        pair_sets[role] = {str(row["pair_id"]) for row in split_rows}
        template_sets[role] = role_templates
    for left_index, left in enumerate(SPLIT_ROLES):
        for right in SPLIT_ROLES[left_index + 1 :]:
            if episode_sets[left] & episode_sets[right]:
                raise ValueError("semantic split episode sets overlap")
            if pair_sets[left] & pair_sets[right]:
                raise ValueError("semantic split pair sets overlap")
            if template_sets[left] & template_sets[right]:
                raise ValueError("semantic split template-group sets overlap")
    manifest_body = {
        "schema": SCHEMA,
        "status": "complete",
        "source": {
            "dataset_sha256": source_sha256,
            "rows": len(rows),
            "pairs": len(pair_records),
            "episode_id_set_sha256": content_sha256(
                sorted(str(row["episode_id"]) for row in rows)
            ),
            "pair_id_set_sha256": content_sha256(sorted(pair_records)),
            "template_group_set_sha256": content_sha256(observed_templates),
        },
        "protocol": {
            "seed": seed,
            "atomic_units": ["template_group", "pair_id"],
            "assignment": "whole_template_group_to_exactly_one_role",
            "stratification": ["semantic_template_group"],
            "split_weights": normalized_weights,
            "roles": list(SPLIT_ROLES),
            "activation_category_contract_sha256": content_sha256(
                category_contract()
            ),
            "minimum_episodes_per_category_per_split": (
                minimum_episodes_per_category_per_split
            ),
            "role_uses": {
                "sae_training": "fixed_base_dictionary_learning_only",
                "discovery": "vpd_effect_features_and_clustering_only",
                "calibration": "cross_template_compartment_threshold_selection_only",
                "final_evaluation": "unseen_template_focused_intervention_evaluation_only",
            },
            "final_evaluation_boundary": (
                "pair_and_semantic_template_disjoint_from_all_training_selection_roles_"
                "algorithmically_held_out_not_secret_or_externally_authenticated"
            ),
        },
        "observed": {
            "activation_categories": observed_categories,
            "template_groups": observed_templates,
            "template_group_role_assignment": dict(sorted(template_roles.items())),
            "pair_category_signatures": sorted(
                {
                    record["category_signature_sha256"]
                    for record in pair_records.values()
                }
            ),
        },
        "splits": split_entries,
        "invariants": {
            "exact_episode_partition": True,
            "pair_atomic": True,
            "template_group_atomic": True,
            "episode_sets_pairwise_disjoint": True,
            "pair_sets_pairwise_disjoint": True,
            "template_group_sets_pairwise_disjoint": True,
            "all_categories_in_every_split": True,
            "final_evaluation_excluded_from_training_clustering_and_threshold_selection": True,
        },
    }
    manifest = {**manifest_body, "split_id": content_sha256(manifest_body)}
    write_json(output_dir / "split_manifest.json", manifest)
    return manifest


def verify_semantic_split_bundle(
    output_dir: Path, *, source_path: Path | None = None
) -> dict[str, Any]:
    manifest_path = output_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = dict(manifest)
    split_id = str(body.pop("split_id", ""))
    if content_sha256(body) != split_id:
        raise ValueError("semantic split manifest content address mismatch")
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("unexpected or incomplete semantic split manifest")
    if manifest["protocol"]["activation_category_contract_sha256"] != content_sha256(
        category_contract()
    ):
        raise ValueError("semantic split category contract differs from code")
    role_rows: dict[str, list[dict[str, Any]]] = {}
    episode_sets: dict[str, set[str]] = {}
    pair_sets: dict[str, set[str]] = {}
    template_sets: dict[str, set[str]] = {}
    for role in SPLIT_ROLES:
        entry = manifest["splits"][role]
        path = output_dir / entry["artifact"]["path"]
        if sha256_file(path) != entry["artifact"]["sha256"]:
            raise ValueError(f"semantic split artifact hash mismatch: {role}")
        rows = load_jsonl(path)
        summary = _split_summary(rows)
        for key, expected in summary.items():
            if entry.get(key) != expected:
                raise ValueError(f"semantic split summary differs: {role}:{key}")
        _pair_records(rows)
        role_rows[role] = rows
        episode_sets[role] = {str(row["episode_id"]) for row in rows}
        pair_sets[role] = {str(row["pair_id"]) for row in rows}
        template_sets[role] = {str(row["template_group"]) for row in rows}
        if content_sha256(sorted(template_sets[role])) != entry[
            "template_group_set_sha256"
        ]:
            raise ValueError(f"semantic split template set hash differs: {role}")
    for left_index, left in enumerate(SPLIT_ROLES):
        for right in SPLIT_ROLES[left_index + 1 :]:
            if episode_sets[left] & episode_sets[right]:
                raise ValueError("semantic split episode overlap")
            if pair_sets[left] & pair_sets[right]:
                raise ValueError("semantic split pair overlap")
            if template_sets[left] & template_sets[right]:
                raise ValueError("semantic split template-group overlap")
    all_rows = [row for role in SPLIT_ROLES for row in role_rows[role]]
    source = manifest["source"]
    episode_ids = sorted(str(row["episode_id"]) for row in all_rows)
    pair_ids = sorted({str(row["pair_id"]) for row in all_rows})
    templates = sorted({str(row["template_group"]) for row in all_rows})
    if len(all_rows) != int(source["rows"]) or len(pair_ids) != int(source["pairs"]):
        raise ValueError("semantic split union size differs from source")
    if content_sha256(episode_ids) != source["episode_id_set_sha256"]:
        raise ValueError("semantic split episode union differs")
    if content_sha256(pair_ids) != source["pair_id_set_sha256"]:
        raise ValueError("semantic split pair union differs")
    if content_sha256(templates) != source["template_group_set_sha256"]:
        raise ValueError("semantic split template union differs")
    if source_path is not None:
        if sha256_file(source_path) != source["dataset_sha256"]:
            raise ValueError("semantic split source dataset hash mismatch")
        source_rows = load_jsonl(source_path)
        expected = assign_template_roles(
            source_rows,
            seed=int(manifest["protocol"]["seed"]),
            weights={
                role: int(manifest["protocol"]["split_weights"][role])
                for role in SPLIT_ROLES
            },
        )
        observed = {
            family: role for role, families in template_sets.items() for family in families
        }
        if observed != expected:
            raise ValueError("semantic split assignment is not reproducible")
        if sorted(str(row["episode_id"]) for row in source_rows) != episode_ids:
            raise ValueError("semantic split source episodes differ")
    return {
        "schema": "sch.vpd-activation-category-split-verification.v2",
        "status": "pass",
        "split_id": split_id,
        "rows": len(all_rows),
        "pairs": len(pair_ids),
        "activation_categories": len(manifest["observed"]["activation_categories"]),
        "template_groups": len(templates),
        "exact_partition": True,
        "template_group_sets_pairwise_disjoint": True,
    }


def verify_semantic_split_artifact_role(
    dataset_path: Path, split_manifest_path: Path, split_role: str
) -> dict[str, Any]:
    if split_role not in SPLIT_ROLES:
        raise ValueError("unknown activation-category split role")
    verification = verify_semantic_split_bundle(split_manifest_path.parent)
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    entry = manifest["splits"][split_role]
    expected_path = (split_manifest_path.parent / entry["artifact"]["path"]).resolve()
    if dataset_path.resolve() != expected_path:
        raise ValueError("dataset path does not match its semantic split role")
    if sha256_file(dataset_path) != entry["artifact"]["sha256"]:
        raise ValueError("dataset hash does not match its semantic split role")
    return {
        "activation_category_split_id": manifest["split_id"],
        "activation_category_split_manifest_sha256": sha256_file(split_manifest_path),
        "activation_category_split_schema": SCHEMA,
        "activation_category_split_role": split_role,
        "activation_category_split_episode_id_set_sha256": entry[
            "episode_id_set_sha256"
        ],
        "activation_category_split_pair_id_set_sha256": entry["pair_id_set_sha256"],
        "activation_category_split_template_group_set_sha256": entry[
            "template_group_set_sha256"
        ],
        "activation_category_split_semantic_disjointness": "pass",
        "activation_category_split_verification_status": verification["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the semantic-template-disjoint V2 role split."
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=310829)
    parser.add_argument(
        "--split-weights",
        default="sae_training=1,discovery=1,calibration=1,final_evaluation=1",
    )
    parser.add_argument("--minimum-episodes-per-category", type=int, default=8)
    parser.add_argument("--verify-output", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.verify_output is not None:
        print(
            json.dumps(
                verify_semantic_split_bundle(
                    args.verify_output, source_path=args.source
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.input is None or args.output_dir is None:
        parser.error("--input and --output-dir are required unless verifying")
    rows = load_jsonl(args.input)
    result = build_semantic_split_bundle(
        rows,
        source_sha256=sha256_file(args.input),
        output_dir=args.output_dir,
        seed=args.seed,
        weights=parse_split_weights(args.split_weights),
        minimum_episodes_per_category_per_split=args.minimum_episodes_per_category,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
