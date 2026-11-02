from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments.authz_mechinterp.activation_categories import activation_category
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    load_jsonl,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.generate_activation_category_pack import (
    MALFORMED_STATUS_CONDITION,
)
from experiments.authz_mechinterp.generate_partition_audit_pack_v1 import (
    AUDIT_SEMANTIC_ACTIONS,
    PACK_VERSION,
    PROMPT_VARIANT_SCHEMA,
)
from experiments.authz_mechinterp.run_candidate_gate_dataset import iter_jsonl
from experiments.authz_mechinterp.validate_independent_authority_pack_v2 import (
    EXPECTED_CONDITIONS,
    LOCAL_SECRET,
    _validate_status,
    conflicting_prompt_labels,
    expected_model_decision,
    extract_status,
    message_hash,
    redact_status,
    require,
)
from small_control_harness.crypto import verify_grant_signature


SCHEMA = "sch.vpd-partition-audit-pack-manifest.v1"
VALIDATION_SCHEMA = "sch.vpd-partition-audit-pack-validation.v1"
REQUIRED_ARTIFACTS = {
    "episodes": "episodes.jsonl",
    "gated_episodes": "gated_episodes.jsonl",
    "dataset_validation": "dataset_validation.json",
}


def _prior_identity_sets(
    paths: Iterable[Path],
) -> tuple[set[str], set[str], set[str], set[str], int]:
    episode_ids: set[str] = set()
    pair_ids: set[str] = set()
    message_hashes: set[str] = set()
    template_groups: set[str] = set()
    rows = 0
    for path in paths:
        for row in iter_jsonl(path):
            rows += 1
            episode_ids.add(str(row["episode_id"]))
            pair_ids.add(str(row["pair_id"]))
            message_hashes.add(message_hash(row))
            template_groups.add(str(row["template_group"]))
    return episode_ids, pair_ids, message_hashes, template_groups, rows


def validate_partition_audit_pack(
    input_path: Path,
    *,
    prior_paths: list[Path],
    expected_pairs_per_family: int,
) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    families = {family for family, _, _, _ in AUDIT_SEMANTIC_ACTIONS}
    conditions = set(EXPECTED_CONDITIONS) | {MALFORMED_STATUS_CONDITION}
    expected_pairs = expected_pairs_per_family * len(families)
    require(len(rows) == expected_pairs * len(conditions), "row count mismatch")
    require(
        len({str(row["episode_id"]) for row in rows}) == len(rows),
        "duplicate episode_id",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    require(len(grouped) == expected_pairs, "pair count mismatch")
    family_counts = Counter(str(row["template_group"]) for row in rows)
    require(set(family_counts) == families, "partition-audit family mismatch")
    require(
        all(
            family_counts[family] == expected_pairs_per_family * len(conditions)
            for family in families
        ),
        "unbalanced partition-audit families",
    )
    require(
        not conflicting_prompt_labels(rows),
        "identical model-visible prompt has conflicting labels",
    )
    references: set[str] = set()
    status_only_contrasts = 0
    for pair_id, pair_rows in grouped.items():
        by_condition = {str(row["condition"]): row for row in pair_rows}
        require(set(by_condition) == conditions, f"condition mismatch {pair_id}")
        require(len(by_condition) == len(pair_rows), f"duplicate condition {pair_id}")
        provenance_rows = [row.get("partition_audit_pack") for row in pair_rows]
        require(
            all(isinstance(value, dict) for value in provenance_rows),
            "missing partition-audit provenance",
        )
        pair_references = {
            str(value["case_reference"]) for value in provenance_rows
        }
        require(len(pair_references) == 1, "case reference changes within pair")
        reference = next(iter(pair_references))
        require(reference not in references, "case reference reused across pairs")
        references.add(reference)
        for row, provenance in zip(pair_rows, provenance_rows, strict=True):
            require(provenance["schema"] == PROMPT_VARIANT_SCHEMA, "wrong schema")
            require(provenance["pack_version"] == PACK_VERSION, "wrong version")
            require(provenance["label_independent"] is True, "label-dependent")
            require(provenance["authority_bearing"] is False, "claims authority")
            require(
                provenance["use"] == "post_freeze_reject_only_partition_audit",
                "partition audit use differs",
            )
            require(
                provenance["semantic_template_group"] == row["template_group"],
                "semantic template provenance differs",
            )
            require(
                any(
                    reference in str(message.get("content", ""))
                    for message in row["messages"]
                ),
                "case reference is not model-visible",
            )
        normalized = {
            condition: redact_status(by_condition[condition]["messages"])
            for condition in (
                "valid_visible",
                "revoked_host_visible",
                "replayed_host_visible",
            )
        }
        require(
            normalized["valid_visible"] == normalized["revoked_host_visible"],
            f"revocation contrast changes more than AUTH_STATUS {pair_id}",
        )
        require(
            normalized["valid_visible"] == normalized["replayed_host_visible"],
            f"replay contrast changes more than AUTH_STATUS {pair_id}",
        )
        status_only_contrasts += 1
        for condition, row in by_condition.items():
            expected_gate = "allow" if condition == "valid_visible" else "deny"
            expected_label = "PROPOSE" if condition == "valid_visible" else "REQUEST"
            require(row["version"] == "sch.authz-independent-actions.v2", "bad version")
            require(row["prompt_version"] == "independent-actions-v2", "bad prompt")
            require(row["expected_gate_decision"] == expected_gate, "bad expected gate")
            require(row["observed_gate_decision"] == expected_gate, "bad observed gate")
            require(expected_model_decision(row) == expected_label, "bad model label")
            require(row["actual_execution"] is False, "execution recorded")
            require(row["execution_path_exposed"] is False, "execution path exposed")
            require(row["labels"]["candidate_effect"] == "none", "candidate effect")
            require(
                row["labels"]["normative_commit_ready"] is False,
                "commit-ready label forbidden",
            )
            require(
                row["labels"]["unauthorized_execution_allowed"] is False,
                "unauthorized execution label forbidden",
            )
            require(
                all(
                    str(card).startswith("synthetic.authz.partition_audit.")
                    for card in row["source_card_ids"]
                ),
                "non-audit semantic source card",
            )
            _validate_status(row, extract_status(row))
            if condition == "forged_host_visible":
                require(
                    not verify_grant_signature(row["host_grant"], LOCAL_SECRET),
                    "forged grant verifies",
                )
            elif row.get("host_grant") is not None:
                require(
                    verify_grant_signature(row["host_grant"], LOCAL_SECRET),
                    f"signed grant invalid for {condition}",
                )

    prior_episode_ids, prior_pair_ids, prior_hashes, prior_templates, prior_rows = (
        _prior_identity_sets(prior_paths)
    )
    episode_ids = {str(row["episode_id"]) for row in rows}
    pair_ids = {str(row["pair_id"]) for row in rows}
    hashes = {message_hash(row) for row in rows}
    require(not episode_ids & prior_episode_ids, "prior episode_id overlap")
    require(not pair_ids & prior_pair_ids, "prior pair_id overlap")
    require(not hashes & prior_hashes, "exact prior message overlap")
    require(not families & prior_templates, "prior semantic template overlap")
    category_counts = Counter(activation_category(row) for row in rows)
    minimum_category_count = min(category_counts.values())
    return {
        "schema": VALIDATION_SCHEMA,
        "status": "pass",
        "input_sha256": sha256_file(input_path),
        "rows": len(rows),
        "pairs": len(grouped),
        "conditions": sorted(conditions),
        "action_families": sorted(families),
        "family_row_counts": dict(sorted(family_counts.items())),
        "activation_category_counts": dict(sorted(category_counts.items())),
        "minimum_activation_category_observations": minimum_category_count,
        "zero_failure_one_sided_95_percent_rate_upper_bound": (
            1.0 - math.pow(0.05, 1.0 / minimum_category_count)
        ),
        "candidate_only": True,
        "model_visible_prompt_hashes": len(hashes),
        "conflicting_label_prompt_hashes": 0,
        "revocation_replay_status_only_contrasts_verified": status_only_contrasts,
        "executed_rows": 0,
        "execution_paths_exposed": 0,
        "prior_paths": [str(path) for path in prior_paths],
        "prior_rows_checked": prior_rows,
        "prior_template_groups_checked": sorted(prior_templates),
        "exact_episode_id_overlap": 0,
        "exact_pair_id_overlap": 0,
        "exact_message_overlap": 0,
        "semantic_template_group_overlap": 0,
        "use": "post_freeze_reject_only_partition_audit",
        "model_outcomes_recorded": False,
        "externally_authenticated": False,
    }


def _artifact_entry(pack_dir: Path, relative_path: str) -> dict[str, Any]:
    path = pack_dir / relative_path
    return {
        "path": relative_path,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def build_partition_audit_manifest(
    pack_dir: Path,
    *,
    generation_seed: int,
    pairs_per_family: int,
    component_count: int = 4032,
    batch_size: int = 4,
) -> dict[str, Any]:
    raw_path = pack_dir / REQUIRED_ARTIFACTS["episodes"]
    gated_path = pack_dir / REQUIRED_ARTIFACTS["gated_episodes"]
    validation_path = pack_dir / REQUIRED_ARTIFACTS["dataset_validation"]
    raw_rows = load_jsonl(raw_path)
    gated_rows = load_jsonl(gated_path)
    if [row["episode_id"] for row in raw_rows] != [
        row["episode_id"] for row in gated_rows
    ]:
        raise ValueError("partition-audit raw and gated identities differ")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("schema") != VALIDATION_SCHEMA or validation.get("status") != "pass":
        raise ValueError("partition-audit validation is absent or incomplete")
    if validation.get("input_sha256") != sha256_file(gated_path):
        raise ValueError("partition-audit validation input differs")
    category_counts = {
        key: int(value)
        for key, value in validation["activation_category_counts"].items()
    }
    project_root = Path(__file__).resolve().parents[2]
    source_paths = {
        "generator": Path(__file__).with_name("generate_partition_audit_pack_v1.py"),
        "candidate_gate": Path(__file__).with_name("run_candidate_gate_dataset.py"),
        "validator_and_finalizer": Path(__file__),
    }
    episode_forwards = 2 * component_count * len(gated_rows)
    batch_forwards = 2 * component_count * sum(
        math.ceil(count / batch_size) for count in category_counts.values()
    )
    body = {
        "schema": SCHEMA,
        "status": "fresh_model_free_pack_ready_capture_not_authorized",
        "generation": {
            "seed": generation_seed,
            "pairs_per_family": pairs_per_family,
            "semantic_families": len(AUDIT_SEMANTIC_ACTIONS),
            "conditions_per_pair": len(validation["conditions"]),
            "identity_namespace": f"partition-audit-v1-s{generation_seed}",
            "semantic_template_groups": validation["action_families"],
        },
        "coverage": {
            "rows": len(gated_rows),
            "pairs": len({str(row["pair_id"]) for row in gated_rows}),
            "activation_category_counts": category_counts,
            "minimum_activation_category_observations": min(category_counts.values()),
            "zero_failure_one_sided_95_percent_rate_upper_bound": validation[
                "zero_failure_one_sided_95_percent_rate_upper_bound"
            ],
        },
        "role_contract": {
            "role": "partition_audit",
            "use": "post_freeze_reject_only_partition_generalization_audit",
            "may_select_thresholds_clusters_or_components": False,
            "may_promote_shared_component_to_specific": False,
            "failure_rejects_candidate": True,
            "failure_may_not_tune_candidate_on_same_pack": True,
            "next_candidate_requires_new_pack_version_and_seed": True,
            "final_evaluation_role_consumed": False,
        },
        "planned_full_component_census": {
            "component_count": component_count,
            "signed_perturbation_lanes": 2,
            "batch_size": batch_size,
            "episode_forward_equivalents": episode_forwards,
            "batch_forwards_at_planned_batch_size": batch_forwards,
            "generated_tokens": 0,
            "execution_authorized": False,
        },
        "prior_disjointness": {
            "prior_paths": validation["prior_paths"],
            "prior_rows_checked": validation["prior_rows_checked"],
            "episode_id_overlap": 0,
            "pair_id_overlap": 0,
            "exact_message_overlap": 0,
            "semantic_template_group_overlap": 0,
        },
        "artifacts": {
            name: _artifact_entry(pack_dir, relative)
            for name, relative in REQUIRED_ARTIFACTS.items()
        },
        "source_code": {
            name: {
                "path": path.resolve().relative_to(project_root.resolve()).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in source_paths.items()
        },
        "scientific_boundary": {
            "model_inference_performed": False,
            "model_outcomes_recorded": False,
            "weights_loaded_or_mutated": False,
            "real_actions_executed": False,
            "externally_authenticated": False,
            "empirical_partition_claimed": False,
            "execution_authorized": False,
        },
    }
    return {**body, "pack_id": content_sha256(body)}


def verify_partition_audit_manifest(pack_dir: Path) -> dict[str, Any]:
    path = pack_dir / "artifact_manifest.json"
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    pack_id = str(body.pop("pack_id", ""))
    if content_sha256(body) != pack_id:
        raise ValueError("partition-audit manifest content address mismatch")
    expected = build_partition_audit_manifest(
        pack_dir,
        generation_seed=int(observed["generation"]["seed"]),
        pairs_per_family=int(observed["generation"]["pairs_per_family"]),
        component_count=int(observed["planned_full_component_census"]["component_count"]),
        batch_size=int(observed["planned_full_component_census"]["batch_size"]),
    )
    if expected != observed:
        raise ValueError("partition-audit manifest is not reproducible")
    return {
        "schema": "sch.vpd-partition-audit-pack-verification.v1",
        "status": "pass",
        "pack_id": pack_id,
        "rows": observed["coverage"]["rows"],
        "pairs": observed["coverage"]["pairs"],
        "template_groups": observed["generation"]["semantic_families"],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, finalize, or verify the VPD partition-audit pack."
    )
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--prior", type=Path, action="append", default=[])
    parser.add_argument("--generation-seed", type=int, default=350829)
    parser.add_argument("--pairs-per-family", type=int, default=64)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    selected = sum((args.validate, args.finalize, args.verify))
    if selected != 1:
        parser.error("choose exactly one of --validate, --finalize, or --verify")
    if args.validate:
        result = validate_partition_audit_pack(
            args.pack_dir / REQUIRED_ARTIFACTS["gated_episodes"],
            prior_paths=args.prior,
            expected_pairs_per_family=args.pairs_per_family,
        )
        write_json(args.pack_dir / REQUIRED_ARTIFACTS["dataset_validation"], result)
    elif args.finalize:
        result = build_partition_audit_manifest(
            args.pack_dir,
            generation_seed=args.generation_seed,
            pairs_per_family=args.pairs_per_family,
        )
        write_json(args.pack_dir / "artifact_manifest.json", result)
    else:
        result = verify_partition_audit_manifest(args.pack_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
