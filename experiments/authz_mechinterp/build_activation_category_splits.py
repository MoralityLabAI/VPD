from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.activation_categories import (
    activation_category,
    category_contract,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    load_jsonl,
    sha256_file,
    stable_hash,
    write_json,
    write_jsonl,
)


SPLIT_ROLES = (
    "sae_training",
    "discovery",
    "calibration",
    "final_evaluation",
)
DEFAULT_SPLIT_WEIGHTS = {role: 1 for role in SPLIT_ROLES}


def parse_split_weights(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in value.split(","):
        key, separator, raw = item.strip().partition("=")
        if not separator or key not in SPLIT_ROLES or key in result:
            raise ValueError("split weights must name each role exactly once")
        weight = int(raw)
        if weight < 1:
            raise ValueError("split weights must be positive integers")
        result[key] = weight
    if set(result) != set(SPLIT_ROLES):
        raise ValueError(
            "split weights must name sae_training, discovery, calibration, and final_evaluation"
        )
    return {role: result[role] for role in SPLIT_ROLES}


def _allocate_counts(size: int, weights: dict[str, int]) -> dict[str, int]:
    total_weight = sum(weights.values())
    raw = {role: size * weights[role] / total_weight for role in SPLIT_ROLES}
    counts = {role: math.floor(raw[role]) for role in SPLIT_ROLES}
    remaining = size - sum(counts.values())
    ordering = sorted(
        SPLIT_ROLES,
        key=lambda role: (-(raw[role] - counts[role]), SPLIT_ROLES.index(role)),
    )
    for role in ordering[:remaining]:
        counts[role] += 1
    if any(counts[role] < 1 for role in SPLIT_ROLES):
        raise ValueError("every category-signature stratum needs at least one pair per split")
    return counts


def _pair_records(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    episode_ids = [str(row.get("episode_id", "")) for row in rows]
    if any(not value for value in episode_ids) or len(episode_ids) != len(set(episode_ids)):
        raise ValueError("episode IDs must be present and unique")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        template_group = str(row.get("template_group", ""))
        if not pair_id or not template_group:
            raise ValueError("every episode needs pair_id and template_group")
        grouped[pair_id].append(row)
    pair_records: dict[str, dict[str, Any]] = {}
    for pair_id, pair_rows in grouped.items():
        families = {str(row["template_group"]) for row in pair_rows}
        if len(families) != 1:
            raise ValueError(f"pair crosses template groups: {pair_id}")
        category_counts = Counter(activation_category(row) for row in pair_rows)
        pair_records[pair_id] = {
            "template_group": next(iter(families)),
            "category_counts": dict(sorted(category_counts.items())),
            "category_signature_sha256": content_sha256(
                dict(sorted(category_counts.items()))
            ),
        }
    return dict(grouped), pair_records


def assign_pair_roles(
    rows: list[dict[str, Any]], *, seed: int, weights: dict[str, int]
) -> dict[str, str]:
    _, pair_records = _pair_records(rows)
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pair_id, record in pair_records.items():
        strata[
            (record["template_group"], record["category_signature_sha256"])
        ].append(pair_id)
    assignments: dict[str, str] = {}
    for (family, signature), pair_ids in sorted(strata.items()):
        ordered = sorted(
            pair_ids,
            key=lambda pair_id: (
                stable_hash(seed, "activation-category-split", family, signature, pair_id),
                pair_id,
            ),
        )
        counts = _allocate_counts(len(ordered), weights)
        offset = 0
        for role in SPLIT_ROLES:
            for pair_id in ordered[offset : offset + counts[role]]:
                assignments[pair_id] = role
            offset += counts[role]
    if set(assignments) != set(pair_records):
        raise ValueError("split assignment does not cover every pair")
    return assignments


def _split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    episode_ids = sorted(str(row["episode_id"]) for row in rows)
    pair_ids = sorted({str(row["pair_id"]) for row in rows})
    category_counts = Counter(activation_category(row) for row in rows)
    family_counts = Counter(str(row["template_group"]) for row in rows)
    family_category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family_category_counts[str(row["template_group"])][activation_category(row)] += 1
    return {
        "rows": len(rows),
        "pairs": len(pair_ids),
        "episode_id_set_sha256": content_sha256(episode_ids),
        "pair_id_set_sha256": content_sha256(pair_ids),
        "activation_category_counts": dict(sorted(category_counts.items())),
        "template_group_counts": dict(sorted(family_counts.items())),
        "template_group_activation_category_counts": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(family_category_counts.items())
        },
    }


def build_split_bundle(
    rows: list[dict[str, Any]],
    *,
    source_sha256: str,
    output_dir: Path,
    seed: int,
    weights: dict[str, int] | None = None,
    minimum_episodes_per_category_per_split: int = 8,
) -> dict[str, Any]:
    normalized_weights = dict(weights or DEFAULT_SPLIT_WEIGHTS)
    if set(normalized_weights) != set(SPLIT_ROLES) or any(
        int(value) < 1 for value in normalized_weights.values()
    ):
        raise ValueError("split weights are incomplete or invalid")
    normalized_weights = {
        role: int(normalized_weights[role]) for role in SPLIT_ROLES
    }
    if minimum_episodes_per_category_per_split < 1:
        raise ValueError("minimum category support must be positive")
    grouped, pair_records = _pair_records(rows)
    pair_roles = assign_pair_roles(rows, seed=seed, weights=normalized_weights)
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
    observed_families = sorted({str(row["template_group"]) for row in rows})
    if len(observed_categories) < 2:
        raise ValueError("activation-category splitting requires at least two categories")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_entries: dict[str, dict[str, Any]] = {}
    role_episode_sets: dict[str, set[str]] = {}
    role_pair_sets: dict[str, set[str]] = {}
    for role in SPLIT_ROLES:
        split_rows = role_rows[role]
        summary = _split_summary(split_rows)
        if set(summary["activation_category_counts"]) != set(observed_categories):
            raise ValueError(f"split does not cover every activation category: {role}")
        if min(summary["activation_category_counts"].values()) < (
            minimum_episodes_per_category_per_split
        ):
            raise ValueError(f"split has insufficient category support: {role}")
        if set(summary["template_group_counts"]) != set(observed_families):
            raise ValueError(f"split does not cover every template group: {role}")
        path = output_dir / f"{role}_episodes.jsonl"
        write_jsonl(path, split_rows)
        split_entries[role] = {
            "artifact": {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(path),
            },
            **summary,
        }
        role_episode_sets[role] = {str(row["episode_id"]) for row in split_rows}
        role_pair_sets[role] = {str(row["pair_id"]) for row in split_rows}
    for left_index, left in enumerate(SPLIT_ROLES):
        for right in SPLIT_ROLES[left_index + 1 :]:
            if role_episode_sets[left] & role_episode_sets[right]:
                raise ValueError("split episode sets overlap")
            if role_pair_sets[left] & role_pair_sets[right]:
                raise ValueError("split pair sets overlap")
    all_episode_ids = sorted(str(row["episode_id"]) for row in rows)
    all_pair_ids = sorted(pair_records)
    manifest_body = {
        "schema": "sch.vpd-activation-category-split-manifest.v1",
        "status": "complete",
        "source": {
            "dataset_sha256": source_sha256,
            "rows": len(rows),
            "pairs": len(pair_records),
            "episode_id_set_sha256": content_sha256(all_episode_ids),
            "pair_id_set_sha256": content_sha256(all_pair_ids),
        },
        "protocol": {
            "seed": seed,
            "atomic_unit": "pair_id",
            "stratification": [
                "template_group",
                "within_pair_activation_category_histogram",
            ],
            "split_weights": normalized_weights,
            "roles": list(SPLIT_ROLES),
            "activation_category_contract_sha256": content_sha256(
                category_contract()
            ),
            "minimum_episodes_per_category_per_split": (
                minimum_episodes_per_category_per_split
            ),
            "final_evaluation_boundary": (
                "algorithmically_held_out_from_effect_threshold_selection_and_"
                "clustering_not_secret_or_externally_authenticated"
            ),
            "role_uses": {
                "sae_training": "fixed_base_dictionary_learning_only",
                "discovery": "vpd_effect_features_and_clustering_only",
                "calibration": "compartment_threshold_selection_only",
                "final_evaluation": "focused_intervention_evaluation_only",
            },
        },
        "observed": {
            "activation_categories": observed_categories,
            "template_groups": observed_families,
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
            "episode_sets_pairwise_disjoint": True,
            "pair_sets_pairwise_disjoint": True,
            "all_categories_in_every_split": True,
            "all_template_groups_in_every_split": True,
            "final_evaluation_excluded_from_threshold_selection": True,
            "sae_training_excluded_from_effect_and_threshold_selection": True,
        },
    }
    manifest = {
        **manifest_body,
        "split_id": content_sha256(manifest_body),
    }
    write_json(output_dir / "split_manifest.json", manifest)
    return manifest


def verify_split_bundle(
    output_dir: Path, *, source_path: Path | None = None
) -> dict[str, Any]:
    manifest_path = output_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = dict(manifest)
    split_id = str(body.pop("split_id", ""))
    if content_sha256(body) != split_id:
        raise ValueError("activation-category split manifest content address mismatch")
    if manifest.get("schema") != "sch.vpd-activation-category-split-manifest.v1":
        raise ValueError("unexpected activation-category split schema")
    if manifest.get("status") != "complete":
        raise ValueError("activation-category split is incomplete")
    protocol = manifest["protocol"]
    if protocol["activation_category_contract_sha256"] != content_sha256(
        category_contract()
    ):
        raise ValueError("activation-category split contract differs from code")
    role_rows: dict[str, list[dict[str, Any]]] = {}
    episode_sets: dict[str, set[str]] = {}
    pair_sets: dict[str, set[str]] = {}
    for role in SPLIT_ROLES:
        entry = manifest["splits"][role]
        path = output_dir / entry["artifact"]["path"]
        if sha256_file(path) != entry["artifact"]["sha256"]:
            raise ValueError(f"activation-category split artifact hash mismatch: {role}")
        rows = load_jsonl(path)
        summary = _split_summary(rows)
        for key, expected in summary.items():
            if entry.get(key) != expected:
                raise ValueError(f"activation-category split summary differs: {role}:{key}")
        _pair_records(rows)
        role_rows[role] = rows
        episode_sets[role] = {str(row["episode_id"]) for row in rows}
        pair_sets[role] = {str(row["pair_id"]) for row in rows}
    for left_index, left in enumerate(SPLIT_ROLES):
        for right in SPLIT_ROLES[left_index + 1 :]:
            if episode_sets[left] & episode_sets[right]:
                raise ValueError("activation-category split episode overlap")
            if pair_sets[left] & pair_sets[right]:
                raise ValueError("activation-category split pair overlap")
    all_rows = [row for role in SPLIT_ROLES for row in role_rows[role]]
    all_episode_ids = sorted(str(row["episode_id"]) for row in all_rows)
    all_pair_ids = sorted({str(row["pair_id"]) for row in all_rows})
    source = manifest["source"]
    if len(all_rows) != int(source["rows"]):
        raise ValueError("activation-category split row union differs from source")
    if len(all_pair_ids) != int(source["pairs"]):
        raise ValueError("activation-category split pair union differs from source")
    if content_sha256(all_episode_ids) != source["episode_id_set_sha256"]:
        raise ValueError("activation-category split episode union differs from source")
    if content_sha256(all_pair_ids) != source["pair_id_set_sha256"]:
        raise ValueError("activation-category split pair union differs from source")
    if source_path is not None:
        if sha256_file(source_path) != source["dataset_sha256"]:
            raise ValueError("activation-category split source dataset hash mismatch")
        source_rows = load_jsonl(source_path)
        expected_assignment = assign_pair_roles(
            source_rows,
            seed=int(protocol["seed"]),
            weights={
                role: int(protocol["split_weights"][role]) for role in SPLIT_ROLES
            },
        )
        observed_assignment = {
            str(row["pair_id"]): role
            for role in SPLIT_ROLES
            for row in role_rows[role]
        }
        if observed_assignment != expected_assignment:
            raise ValueError("activation-category split assignment is not reproducible")
        if sorted(str(row["episode_id"]) for row in source_rows) != all_episode_ids:
            raise ValueError("activation-category split source episodes differ")
    return {
        "schema": "sch.vpd-activation-category-split-verification.v1",
        "status": "pass",
        "split_id": split_id,
        "rows": len(all_rows),
        "pairs": len(all_pair_ids),
        "activation_categories": len(manifest["observed"]["activation_categories"]),
        "template_groups": len(manifest["observed"]["template_groups"]),
        "exact_partition": True,
    }


def verify_split_artifact_role(
    dataset_path: Path, split_manifest_path: Path, split_role: str
) -> dict[str, Any]:
    if split_role not in SPLIT_ROLES:
        raise ValueError("unknown activation-category split role")
    verification = verify_split_bundle(split_manifest_path.parent)
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    entry = manifest["splits"][split_role]
    expected_path = (split_manifest_path.parent / entry["artifact"]["path"]).resolve()
    if dataset_path.resolve() != expected_path:
        raise ValueError("dataset path does not match its frozen split role")
    if sha256_file(dataset_path) != entry["artifact"]["sha256"]:
        raise ValueError("dataset hash does not match its frozen split role")
    return {
        "activation_category_split_id": manifest["split_id"],
        "activation_category_split_manifest_sha256": sha256_file(
            split_manifest_path
        ),
        "activation_category_split_role": split_role,
        "activation_category_split_episode_id_set_sha256": entry[
            "episode_id_set_sha256"
        ],
        "activation_category_split_pair_id_set_sha256": entry[
            "pair_id_set_sha256"
        ],
        "activation_category_split_verification_status": verification["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify disjoint activation-category discovery, calibration, "
            "SAE-training, and final-evaluation episode packs."
        )
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=290829)
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
                verify_split_bundle(args.verify_output, source_path=args.source),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.input is None or args.output_dir is None:
        parser.error("--input and --output-dir are required unless verifying")
    rows = load_jsonl(args.input)
    manifest = build_split_bundle(
        rows,
        source_sha256=sha256_file(args.input),
        output_dir=args.output_dir,
        seed=args.seed,
        weights=parse_split_weights(args.split_weights),
        minimum_episodes_per_category_per_split=args.minimum_episodes_per_category,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
