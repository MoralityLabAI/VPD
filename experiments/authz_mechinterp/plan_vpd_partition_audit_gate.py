from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_frozen_vpd_partition_audit import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    verify_bundle,
    write_json,
)
from experiments.authz_mechinterp.finalize_partition_audit_pack_v1 import (
    verify_partition_audit_manifest,
)


SCHEMA = "sch.vpd-partition-audit-gate-plan.v1"


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("partition-audit source escapes the project")
    return resolved.relative_to(root.resolve()).as_posix()


def build_gate_plan(
    *,
    predecessor_protocol_path: Path,
    pack_dir: Path,
    benchmark_path: Path,
    bundle_dir: Path,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    predecessor = json.loads(predecessor_protocol_path.read_text(encoding="utf-8"))
    if predecessor.get("schema") != "sch.vpd-activation-category-protocol.v4":
        raise ValueError("partition-audit gate requires the frozen V4 predecessor")
    pack = json.loads((pack_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    pack_verification = verify_partition_audit_manifest(pack_dir)
    benchmark_verification = verify_benchmark(benchmark_path)
    bundle_verification = verify_bundle(bundle_dir)
    bundle_manifest_path = bundle_dir / "manifest.json"
    bundle = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    component_count = int(bundle["invariants"]["component_count"])
    packet_entries = list(bundle["artifacts"]["packets"])
    census = pack["planned_full_component_census"]
    if component_count != int(census["component_count"]):
        raise ValueError("partition-audit pack census component count differs")
    if pack["role_contract"]["final_evaluation_role_consumed"] is not False:
        raise ValueError("partition-audit pack consumes final_evaluation")

    work_units = []
    for entry in sorted(packet_entries, key=lambda row: str(row["cluster_id"])):
        packet_path = bundle_dir / str(entry["packet_path"])
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        component_ids = [str(value) for value in packet["component_ids"]]
        body = {
            "schema": "sch.vpd-partition-audit-capture-work-unit.v1",
            "packet_id": str(entry["packet_id"]),
            "packet_sha256": str(entry["packet_sha256"]),
            "cluster_id": str(entry["cluster_id"]),
            "component_count": len(component_ids),
            "component_id_set_sha256": content_sha256(sorted(component_ids)),
            "dataset_sha256": pack["artifacts"]["gated_episodes"]["sha256"],
            "signed_perturbation_lanes": 2,
            "generated_tokens": 0,
            "output_contract": {
                "effect_receipt_schema": "sch.vpd-effect-shard-receipt.v5",
                "partition_audit_pack_id_required": pack["pack_id"],
                "exact_packet_coverage_required": True,
                "signed_receipt_required": True,
            },
            "execution_authorized": False,
        }
        work_units.append({**body, "work_unit_id": content_sha256(body)})

    bundle_has_empirical_categories = bool(
        bundle.get("algorithm", {})
        .get("activation_category_compartments", {})
        .get("enabled", False)
    )
    body = {
        "schema": SCHEMA,
        "status": (
            "locally_verified_pack_evaluator_and_work_units_ready_"
            "requires_category_aware_bundle_capture_adapter_and_compute_authorization"
        ),
        "source": {
            "predecessor_protocol": {
                "path": _relative(predecessor_protocol_path, project_root),
                "sha256": sha256_file(predecessor_protocol_path),
            },
            "partition_audit_pack": {
                "path": _relative(pack_dir, project_root),
                "manifest_sha256": sha256_file(pack_dir / "artifact_manifest.json"),
                "pack_id": pack_verification["pack_id"],
                "rows": pack_verification["rows"],
                "pairs": pack_verification["pairs"],
            },
            "synthetic_benchmark": {
                "path": _relative(benchmark_path, project_root),
                "sha256": sha256_file(benchmark_path),
                "benchmark_id": benchmark_verification["benchmark_id"],
            },
            "current_bundle_evidence": {
                "path": _relative(bundle_dir, project_root),
                "manifest_sha256": sha256_file(bundle_manifest_path),
                "verification": bundle_verification,
                "component_count": component_count,
                "component_id_set_sha256": bundle["invariants"][
                    "component_id_set_sha256"
                ],
                "empirical_activation_category_partition_present": (
                    bundle_has_empirical_categories
                ),
                "status": (
                    "eligible_candidate_source"
                    if bundle_has_empirical_categories
                    else "historical_metadata_only_not_a_partition_audit_candidate"
                ),
            },
        },
        "frozen_gate": {
            "position": "after_category_bundle_freeze_before_category_specific_sae_training",
            "required_inputs": [
                "category_aware_bundle_with_exact_component_partition",
                "bundle_bound_discovery_category_effect_archive",
                "fresh_signed_partition_audit_effect_archive",
                "fresh_partition_audit_stratified_robustness_eligibility",
            ],
            "decision_rule": "every_frozen_specific_component_must_pass",
            "failure_effect": "reject_candidate_and_prohibit_category_specific_sae_training",
            "failure_may_tune_on_same_pack": False,
            "next_candidate_requires_new_pack_version_and_seed": True,
            "shared_to_specific_promotion_allowed": False,
            "threshold_or_cluster_selection_allowed": False,
            "final_evaluation_consumed": False,
        },
        "stage_order": [
            "freeze_category_aware_bundle",
            "capture_fresh_partition_audit_effects",
            "verify_signed_exact_component_census",
            "evaluate_frozen_partition_reject_only",
            "release_category_specific_sae_training_only_on_pass",
        ],
        "capture_work_units": work_units,
        "resources": {
            **copy.deepcopy(census),
            "work_units": len(work_units),
            "model_memory_wall_time_and_cost_not_estimated": True,
        },
        "implementation": {
            "evaluator": {
                "path": "experiments/authz_mechinterp/audit_frozen_vpd_category_partition.py",
                "sha256": sha256_file(
                    project_root
                    / "experiments/authz_mechinterp/audit_frozen_vpd_category_partition.py"
                ),
            },
            "capture_adapter": {
                "status": "required_not_implemented",
                "reason": (
                    "The preserved V3 campaign worker admits only discovery, calibration, "
                    "and cluster_validation split roles; it is not silently repurposed."
                ),
            },
        },
        "invariants": {
            "work_units_cover_current_component_inventory_exactly": (
                sum(row["component_count"] for row in work_units) == component_count
            ),
            "pack_is_identity_message_and_semantic_template_disjoint": True,
            "audit_is_post_freeze_and_reject_only": True,
            "final_evaluation_remains_untouched": True,
            "plan_does_not_authorize_execution": True,
        },
        "scientific_boundary": {
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "empirical_partition_audited": False,
            "execution_authorized": False,
            "v7_negative_release_decision_unchanged": True,
        },
    }
    return {**body, "gate_plan_id": content_sha256(body)}


def verify_gate_plan(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = copy.deepcopy(observed)
    plan_id = str(body.pop("gate_plan_id", ""))
    if content_sha256(body) != plan_id:
        raise ValueError("partition-audit gate plan content address mismatch")
    if build_gate_plan(**kwargs) != observed:
        raise ValueError("partition-audit gate plan is not reproducible")
    return {
        "schema": "sch.vpd-partition-audit-gate-plan-verification.v1",
        "status": "pass",
        "gate_plan_id": plan_id,
        "work_units": len(observed["capture_work_units"]),
        "episode_forward_equivalents": observed["resources"][
            "episode_forward_equivalents"
        ],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the post-freeze VPD partition-audit gate plan."
    )
    parser.add_argument("--predecessor-protocol", type=Path, required=True)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = {
        "predecessor_protocol_path": args.predecessor_protocol,
        "pack_dir": args.pack_dir,
        "benchmark_path": args.benchmark,
        "bundle_dir": args.bundle_dir,
    }
    if args.verify is not None:
        result = verify_gate_plan(args.verify, **inputs)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_gate_plan(**inputs)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
