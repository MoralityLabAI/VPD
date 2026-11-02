from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.audit_frozen_vpd_category_partition import (
    evaluate_frozen_partition,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    category_profile_stratified_robustness,
    content_sha256,
    load_effects,
    load_hierarchy_plan,
    load_jsonl,
    sha256_file,
    verify_bundle,
    write_json,
)
from experiments.authz_mechinterp.finalize_partition_audit_pack_v1 import (
    verify_partition_audit_manifest,
)
from experiments.authz_mechinterp.merge_vpd_partition_audit_effect_shards import (
    verify_partition_audit_effect_merge,
)


MEASUREMENT_IDENTITY_KEYS = (
    "model_id",
    "revision",
    "adapter_sha256",
    "runner_sha256",
    "effect_space_id",
    "phase",
    "estimator",
    "perturbation_epsilon",
    "selection_seed",
    "generated_tokens",
    "activation_category_contract_sha256",
    "activation_categories",
    "activation_category_uncertainty",
    "activation_category_condition_robustness",
    "activation_category_template_robustness",
)


def validate_cross_archive_measurement_identity(
    *, discovery_manifest: dict[str, Any], audit_manifest: dict[str, Any]
) -> dict[str, Any]:
    discovery = discovery_manifest.get("identity")
    audit = audit_manifest.get("identity")
    if not isinstance(discovery, dict) or not isinstance(audit, dict):
        raise ValueError("partition-audit effect manifest identity is absent")
    shared: dict[str, Any] = {}
    for key in MEASUREMENT_IDENTITY_KEYS:
        if key not in discovery or key not in audit:
            raise ValueError(
                f"partition-audit cross-archive identity omits required field: {key}"
            )
        if discovery[key] != audit[key]:
            raise ValueError(
                f"partition-audit cross-archive measurement identity differs: {key}"
            )
        shared[key] = copy.deepcopy(discovery[key])
    body = {
        "schema": "sch.vpd-partition-audit-cross-archive-measurement-identity.v1",
        "status": "pass",
        "shared_identity": shared,
        "dataset_identity_intentionally_distinct": True,
        "split_role_identity_intentionally_distinct": True,
    }
    return {**body, "identity_id": content_sha256(body)}


def _require_production_screens(compartments: dict[str, Any]) -> dict[str, Any]:
    pair = compartments["pair_block_direction_robustness_screen"]
    condition = compartments["condition_robustness_screen"]
    template = compartments["template_robustness_screen"]
    expected_modes = {
        "pair": "pair_block_mean_countsketch_direction_and_relative_norm_coverage",
        "condition": "condition_mean_countsketch_direction_and_relative_norm_coverage",
        "template": "template_group_mean_countsketch_direction_and_relative_norm_coverage",
    }
    if pair.get("mode") != expected_modes["pair"] or pair.get("required") is not True:
        raise ValueError("frozen bundle lacks the production pair-block screen")
    if (
        condition.get("mode") != expected_modes["condition"]
        or condition.get("required") is not True
    ):
        raise ValueError("frozen bundle lacks the production condition screen")
    if (
        template.get("mode") != expected_modes["template"]
        or template.get("required") is not True
    ):
        raise ValueError("frozen bundle lacks the production template screen")
    return {"pair": pair, "condition": condition, "template": template}


def build_partition_audit_report(
    *,
    bundle_dir: Path,
    hierarchy_plan_path: Path,
    discovery_effects_path: Path,
    discovery_merge_manifest_path: Path,
    audit_merge_dir: Path,
    pack_dir: Path,
    gate_plan_path: Path,
) -> dict[str, Any]:
    bundle_verification = verify_bundle(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    compartments = manifest.get("algorithm", {}).get(
        "activation_category_compartments", {}
    )
    if compartments.get("enabled") is not True:
        raise ValueError("partition audit requires a frozen category-aware bundle")
    if manifest.get("source", {}).get("effects_sha256") != sha256_file(
        discovery_effects_path
    ):
        raise ValueError("frozen bundle discovery effect archive differs")
    if manifest.get("source", {}).get(
        "effects_merge_manifest_sha256"
    ) != sha256_file(discovery_merge_manifest_path):
        raise ValueError("frozen bundle discovery merge manifest differs")
    if manifest["source"]["component_inventory_sha256"] != sha256_file(
        hierarchy_plan_path
    ):
        raise ValueError("frozen bundle component inventory differs")
    screens = _require_production_screens(compartments)
    admission = compartments["per_component_cross_archive_admission"]
    if admission.get("required") is not True:
        raise ValueError("frozen bundle lacks per-component cross-archive admission")

    pack = verify_partition_audit_manifest(pack_dir)
    audit_merge = verify_partition_audit_effect_merge(
        audit_merge_dir,
        pack_dir=pack_dir,
        gate_plan_path=gate_plan_path,
    )
    audit_manifest_path = audit_merge_dir / "effect_merge_manifest.json"
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    discovery_merge_manifest = json.loads(
        discovery_merge_manifest_path.read_text(encoding="utf-8")
    )
    measurement_identity = validate_cross_archive_measurement_identity(
        discovery_manifest=discovery_merge_manifest,
        audit_manifest=audit_manifest,
    )
    audit_effects_path = audit_merge_dir / audit_manifest["artifacts"][
        "activation_effects"
    ]["path"]
    if audit_merge["pack_id"] != pack["pack_id"]:
        raise ValueError("partition-audit merge and fresh pack differ")
    if audit_manifest["identity"]["dataset_sha256"] != json.loads(
        (pack_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )["artifacts"]["gated_episodes"]["sha256"]:
        raise ValueError("partition-audit merge dataset differs from fresh pack")

    rows = load_hierarchy_plan(hierarchy_plan_path)
    seed = int(manifest["algorithm"]["seed"])
    projection_dim = int(manifest["algorithm"]["effect_projection_dim"])
    _, discovery_profile = load_effects(
        discovery_effects_path,
        rows,
        seed=seed,
        projection_dim=projection_dim,
    )
    _, audit_profile = load_effects(
        audit_effects_path,
        rows,
        seed=seed,
        projection_dim=projection_dim,
    )
    if discovery_profile is None or audit_profile is None:
        raise ValueError("partition audit requires category effect arrays")
    if discovery_profile["category_names"] != audit_profile["category_names"]:
        raise ValueError("partition audit category contracts differ")
    category_names = list(discovery_profile["category_names"])
    if category_names != list(compartments["categories"]):
        raise ValueError("partition audit categories differ from frozen bundle")

    robustness = category_profile_stratified_robustness(
        audit_profile,
        minimum_pair_block_coverage=float(screens["pair"]["minimum_coverage"]),
        minimum_pair_block_direction_cosine=float(
            screens["pair"]["minimum_direction_cosine"]
        ),
        minimum_pair_block_relative_norm=float(
            screens["pair"]["minimum_relative_norm"]
        ),
        require_pair_block=True,
        minimum_condition_coverage=float(
            screens["condition"]["minimum_coverage"]
        ),
        minimum_condition_direction_cosine=float(
            screens["condition"]["minimum_direction_cosine"]
        ),
        minimum_condition_relative_norm=float(
            screens["condition"]["minimum_relative_norm"]
        ),
        require_condition=True,
        minimum_template_coverage=float(screens["template"]["minimum_coverage"]),
        minimum_template_direction_cosine=float(
            screens["template"]["minimum_direction_cosine"]
        ),
        minimum_template_relative_norm=float(
            screens["template"]["minimum_relative_norm"]
        ),
        require_template=True,
        minimum_sketch_dimension=max(
            int(screens[key]["minimum_countsketch_dimension"])
            for key in ("pair", "condition", "template")
        ),
    )
    assignments = load_jsonl(bundle_dir / manifest["artifacts"]["assignments"]["path"])
    assignment_by_id = {str(row["component_id"]): row for row in assignments}
    component_ids = [str(row["component_id"]) for row in rows]
    if set(assignment_by_id) != set(component_ids):
        raise ValueError("frozen bundle assignments and component inventory differ")
    frozen_categories = [
        str(assignment_by_id[component_id].get("activation_category", ""))
        for component_id in component_ids
    ]
    source = {
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "bundle_verification": bundle_verification,
        "assignment_sha256": sha256_file(
            bundle_dir / manifest["artifacts"]["assignments"]["path"]
        ),
        "component_inventory_sha256": sha256_file(hierarchy_plan_path),
        "discovery_effects_sha256": sha256_file(discovery_effects_path),
        "discovery_merge_manifest_sha256": sha256_file(
            discovery_merge_manifest_path
        ),
        "partition_audit_effects_sha256": sha256_file(audit_effects_path),
        "partition_audit_merge_manifest_sha256": sha256_file(audit_manifest_path),
        "partition_audit_merge_id": audit_merge["merge_id"],
        "partition_audit_pack_id": pack["pack_id"],
        "partition_audit_gate_plan_id": audit_merge["gate_plan_id"],
        "cross_archive_measurement_identity": measurement_identity,
        "production_driver_sha256": sha256_file(Path(__file__)),
    }
    return evaluate_frozen_partition(
        component_ids=component_ids,
        frozen_categories=frozen_categories,
        category_names=category_names,
        discovery_effects=discovery_profile["category_effect_vectors"],
        audit_effects=audit_profile["category_effect_vectors"],
        robustness_eligibility=robustness["eligible"],
        specificity_thresholds=dict(compartments["specificity_thresholds"]),
        minimum_direction_cosine=float(admission["minimum_direction_cosine"]),
        minimum_strength_ratio=float(admission["minimum_strength_ratio"]),
        source=source,
    )


def verify_report(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = copy.deepcopy(observed)
    audit_id = str(body.pop("partition_audit_id", ""))
    if content_sha256(body) != audit_id:
        raise ValueError("partition audit report content address mismatch")
    expected = build_partition_audit_report(**kwargs)
    if expected != observed:
        raise ValueError("partition audit report is not reproducible")
    return {
        "schema": "sch.vpd-production-partition-audit-verification.v1",
        "status": "pass",
        "partition_audit_id": audit_id,
        "audit_decision": observed["status"],
        "component_count": observed["summary"]["component_count"],
        "specific_failure_count": observed["summary"]["specific_failure_count"],
        "eligible_for_category_specific_sae_training": observed["summary"][
            "eligible_for_category_specific_sae_training"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate or verify a signed, post-freeze VPD category partition audit."
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--hierarchy-plan", type=Path, required=True)
    parser.add_argument("--discovery-effects", type=Path, required=True)
    parser.add_argument("--discovery-merge-manifest", type=Path, required=True)
    parser.add_argument("--audit-merge-dir", type=Path, required=True)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--gate-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = {
        "bundle_dir": args.bundle_dir,
        "hierarchy_plan_path": args.hierarchy_plan,
        "discovery_effects_path": args.discovery_effects,
        "discovery_merge_manifest_path": args.discovery_merge_manifest,
        "audit_merge_dir": args.audit_merge_dir,
        "pack_dir": args.pack_dir,
        "gate_plan_path": args.gate_plan,
    }
    if args.verify is not None:
        result = verify_report(args.verify, **inputs)
    else:
        if args.output is None:
            parser.error("--output is required unless --verify is used")
        result = build_partition_audit_report(**inputs)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
