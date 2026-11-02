from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.activation_category_split_contract import (
    verify_activation_category_split_bundle,
)
from experiments.authz_mechinterp.benchmark_focused_sae_input_lineage import (
    verify_benchmark as verify_focused_input_lineage_benchmark,
)
from experiments.authz_mechinterp.benchmark_compartment_sae_screening import (
    verify_benchmark as verify_focused_screening_benchmark,
)
from experiments.authz_mechinterp.benchmark_activation_capture_semantics import (
    verify_benchmark as verify_capture_semantic_benchmark,
)
from experiments.authz_mechinterp.plan_compartment_capture_resources import (
    verify_capture_resource_plan,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-activation-category-protocol.v3"
EFFECT_ROLES = ("discovery", "calibration", "cluster_validation")
ROLE_USES = {
    "sae_training": "fixed_base_dictionary_learning_only",
    "discovery": "effect_features_and_final_cluster_assignment_only",
    "calibration": "cross_template_threshold_selection_only",
    "cluster_validation": "within_category_cluster_count_replication_only",
    "final_evaluation": "never",
}


def _verify_capacity_pack(pack_dir: Path) -> dict[str, Any]:
    manifest_path = pack_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = dict(manifest)
    pack_id = str(body.pop("pack_id", ""))
    if content_sha256(body) != pack_id:
        raise ValueError("capacity pack content address mismatch")
    if manifest.get("status") != "capacity_ready_design_locally_validated_no_model_outcomes":
        raise ValueError("unified protocol requires the validated capacity pack")
    for record in manifest.get("artifacts", {}).values():
        path = pack_dir / str(record["path"])
        if sha256_file(path) != str(record["sha256"]):
            raise ValueError("capacity pack artifact hash differs")
    return manifest


def build_protocol(
    *, predecessor_path: Path, pack_dir: Path, split_dir: Path
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if predecessor.get("schema") != "sch.vpd-activation-category-protocol.v2":
        raise ValueError("unified protocol predecessor is not V2")
    pack = _verify_capacity_pack(pack_dir)
    focused_lineage_benchmark_path = (
        project_root
        / "artifacts/vpd_focused_sae_input_lineage_benchmark_v1/report.json"
    )
    focused_lineage_benchmark = verify_focused_input_lineage_benchmark(
        focused_lineage_benchmark_path
    )
    focused_screening_benchmark_path = (
        project_root
        / "artifacts/vpd_compartment_sae_screening_benchmark_v2/report.json"
    )
    focused_screening_benchmark = verify_focused_screening_benchmark(
        focused_screening_benchmark_path
    )
    capture_semantic_benchmark_path = (
        project_root
        / "artifacts/vpd_activation_capture_semantic_benchmark_v1/report.json"
    )
    capture_semantic_benchmark = verify_capture_semantic_benchmark(
        capture_semantic_benchmark_path
    )
    split_path = split_dir / "split_manifest.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split_verification = verify_activation_category_split_bundle(
        split_dir,
        source_path=pack_dir / "gated_episodes.jsonl",
    )
    if split.get("schema") != "sch.vpd-activation-category-split-manifest.v3":
        raise ValueError("unified protocol requires a five-role V3 split")
    if split_verification.get("status") != "pass":
        raise ValueError("unified protocol split verification failed")
    if int(split["source"]["rows"]) != int(pack["coverage"]["rows"]) or int(
        split["source"]["pairs"]
    ) != int(pack["coverage"]["pairs"]):
        raise ValueError("unified protocol pack and split capacity differ")
    capture_resource_path = (
        project_root / "artifacts/vpd_compartment_capture_resources_v1/plan.json"
    )
    capacity_plan_path = (
        project_root / "artifacts/vpd_compartment_sae_capacity_v2/plan.json"
    )
    capture_resource_verification = verify_capture_resource_plan(
        capture_resource_path,
        capacity_plan_path=capacity_plan_path,
        pack_dir=pack_dir,
        split_dir=split_dir,
    )

    relative = lambda path: path.resolve().relative_to(project_root.resolve()).as_posix()
    roles: dict[str, dict[str, Any]] = {}
    for role in ROLE_USES:
        record = split["splits"][role]
        roles[role] = {
            "rows": int(record["rows"]),
            "pairs": int(record["pairs"]),
            "template_groups": sorted(record["template_group_counts"]),
            "optimizer_or_threshold_use": ROLE_USES[role],
        }

    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = relative(predecessor_path)
    result["data_pack"] = {
        "path": relative(pack_dir),
        "status": "capacity_ready_design_locally_validated_no_model_outcomes",
        "pack_id": str(pack["pack_id"]),
        "split_path": relative(split_path),
        "split_id": str(split["split_id"]),
        "split_schema": str(split["schema"]),
        "split_seed": int(split["protocol"]["seed"]),
        "generation_seed": int(pack["generation"]["seed"]),
        "rows": int(split["source"]["rows"]),
        "pairs": int(split["source"]["pairs"]),
        "template_groups": len(split["observed"]["template_groups"]),
        "activation_categories": len(split["observed"]["activation_categories"]),
        "roles": roles,
        "atomic_units": list(split["protocol"]["atomic_units"]),
        "assignment": str(split["protocol"]["assignment"]),
        "pair_episode_and_template_group_sets_are_disjoint": True,
        "all_declared_categories_in_every_role": True,
        "pair_constant_prompt_variant_is_label_independent": True,
        "cluster_validation_is_semantically_disjoint_and_outcome_free": True,
        "final_evaluation_is_semantically_and_algorithmically_held_out_not_secret": True,
    }
    campaign = result["effect_capture"]["campaign_plan"]
    campaign["path"] = "artifacts/vpd_effect_capture_campaign_v2/plan.json"
    campaign["stage_gate_benchmark_path"] = (
        "artifacts/vpd_effect_campaign_stage_gate_benchmark_v2/report.json"
    )
    component_count = 4032
    batch_size = 4
    effect_rows = sum(int(split["splits"][role]["rows"]) for role in EFFECT_ROLES)
    episode_forwards = 2 * component_count * effect_rows
    batch_forwards = 0
    for role in EFFECT_ROLES:
        category_counts = split["splits"][role]["activation_category_counts"]
        batches = sum(
            (int(count) + batch_size - 1) // batch_size
            for count in category_counts.values()
        )
        batch_forwards += 2 * component_count * batches
    campaign["episode_forward_equivalents"] = episode_forwards
    campaign["batch_forwards_at_planned_batch_size"] = batch_forwards
    result["sae_training"]["replica_training_plan"]["path"] = (
        "artifacts/vpd_compartment_sae_training_plan_v2/plan.json"
    )
    capture_resources = result["sae_training"]["capacity_support"][
        "capture_resources"
    ]
    capture_resources["resource_plan_id"] = capture_resource_verification[
        "resource_plan_id"
    ]
    capture_resources["role_resources"] = dict(
        capture_resource_verification["capture_roles"]
    )
    capture_resources["activation_rows_total"] = sum(
        int(record["activation_rows"])
        for record in capture_resource_verification["capture_roles"].values()
    )
    capture_resources["role_specific_episode_row_and_shard_counts_required"] = True
    result["sae_training"]["capture_semantic_admission"] = {
        "schema": "sch.vpd-prompt-window-capture-semantic-verification.v1",
        "source_episode_order_and_content_address_required": True,
        "observable_activation_category_recomputed_from_source": True,
        "prompt_window_coordinates_reproduced_without_model_inference": True,
        "required_before_dictionary_training": True,
        "required_before_dictionary_bundle_admission": True,
        "required_before_calibration_quality": True,
        "synthetic_benchmark_path": relative(capture_semantic_benchmark_path),
        "synthetic_benchmark_id": capture_semantic_benchmark["benchmark_id"],
    }
    result["focused_evaluation"]["input_lineage"] = {
        "schema": "sch.vpd-focused-sae-packet-effect-lineage.v1",
        "packet_bound_discovery_effect_archive_required": True,
        "packet_bound_discovery_merge_manifest_required": True,
        "category_packet_without_exact_effect_lineage_rule": "reject",
        "legacy_metadata_only_packet_status": "unattested_legacy_metadata_pilot",
        "legacy_metadata_only_packet_empirical_category_claim_allowed": False,
        "synthetic_benchmark_path": relative(focused_lineage_benchmark_path),
        "synthetic_benchmark_id": focused_lineage_benchmark["benchmark_id"],
    }
    result["focused_evaluation"]["screening_endpoint"][
        "synthetic_benchmark_path"
    ] = relative(focused_screening_benchmark_path)
    result["focused_evaluation"]["screening_endpoint"][
        "synthetic_benchmark_id"
    ] = focused_screening_benchmark["benchmark_id"]
    result["unified_lineage"] = {
        "schema": "sch.vpd-unified-compartment-lineage.v1",
        "split_id": str(split["split_id"]),
        "split_manifest_sha256": sha256_file(split_path),
        "effect_roles": list(EFFECT_ROLES),
        "sae_training_role": "sae_training",
        "final_evaluation_role": "final_evaluation",
        "all_effect_sae_and_evaluation_roles_share_one_exact_split": True,
        "cross_pack_role_reuse_forbidden": True,
        "model_inference_performed": False,
        "execution_authorized": False,
    }
    return result


def verify_protocol(
    path: Path, *, predecessor_path: Path, pack_dir: Path, split_dir: Path
) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = build_protocol(
        predecessor_path=predecessor_path,
        pack_dir=pack_dir,
        split_dir=split_dir,
    )
    if observed != expected:
        raise ValueError("unified VPD compartment protocol is not reproducible")
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v3",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "split_id": observed["data_pack"]["split_id"],
        "rows": observed["data_pack"]["rows"],
        "effect_episode_forward_equivalents": observed["effect_capture"][
            "campaign_plan"
        ]["episode_forward_equivalents"],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive or verify the unified capacity-scale VPD compartment protocol."
    )
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    common = {
        "predecessor_path": args.predecessor,
        "pack_dir": args.pack_dir,
        "split_dir": args.split_dir,
    }
    if args.verify is not None:
        result = verify_protocol(args.verify, **common)
    else:
        if args.output is None:
            parser.error("protocol derivation requires --output")
        result = build_protocol(**common)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
