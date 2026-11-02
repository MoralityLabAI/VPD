from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.audit_compartment_sae_cluster_routing_gap_v1 import (
    verify_audit,
)
from experiments.authz_mechinterp.benchmark_cluster_aware_sae_training_plan_v1 import (
    verify_benchmark as verify_planner_benchmark,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_episode_routing_v1 import (
    verify_benchmark as verify_router_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json


SCHEMA = "sch.vpd-activation-category-protocol.v21"
EXPECTED_V20_SHA256 = (
    "b4edb971141764c6c6a4ac34c7971a4968ac93400f8c0e3c4340ae3d91df1199"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V21 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V20_SHA256:
        raise ValueError("protocol V20 differs from its pinned SHA-256")
    predecessor = json.loads(path.read_text(encoding="utf-8"))
    if predecessor.get("schema") != "sch.vpd-activation-category-protocol.v20":
        raise ValueError("pinned predecessor is not protocol V20")
    if predecessor["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("protocol V20 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-predecessor-verification.v1",
        "status": "pass",
        "method": "exact_sha256_and_boundary_fields",
        "sha256": EXPECTED_V20_SHA256,
    }


def build_protocol(
    *,
    predecessor_path: Path,
    router_benchmark_path: Path,
    planner_benchmark_path: Path,
    gap_audit_path: Path,
    predecessor_sae_plan_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    router_verification = verify_router_benchmark(router_benchmark_path)
    planner_verification = verify_planner_benchmark(
        planner_benchmark_path, project_root=project_root
    )
    audit_verification = verify_audit(
        gap_audit_path, plan_path=predecessor_sae_plan_path
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    router_benchmark = json.loads(router_benchmark_path.read_text(encoding="utf-8"))
    planner_benchmark = json.loads(planner_benchmark_path.read_text(encoding="utf-8"))
    router_runtime = (
        project_root / "experiments/authz_mechinterp/route_vpd_cluster_episodes_v1.py"
    )
    planner_runtime = (
        project_root
        / "experiments/authz_mechinterp/plan_cluster_aware_sae_training_v1.py"
    )
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["cluster_episode_routing_v1"] = {
        "schema": "sch.vpd-cluster-episode-routing-contract.v1",
        "status": "required_between_partition_admission_and_cluster_aware_sae_planning",
        "partition_identity_v4_required": True,
        "canonical_membership_derived_cluster_ids": True,
        "exact_component_effect_and_episode_order_hashes_required": True,
        "score": "cluster_size_normalized_episode_rms_with_absolute_and_relative_margin",
        "pair_atomic_primary_routing": True,
        "ambiguous_low_response_and_pair_conflict_route": "shared_ambiguous",
        "discovery_and_cluster_validation_stability_required": True,
        "aggregate_agreement_alone_is_insufficient": True,
        "condition_stratified_stability_required": True,
        "sae_training_role_must_be_disjoint": True,
        "runtime": {
            "path": _relative(router_runtime, project_root),
            "sha256": sha256_file(router_runtime),
            "schema": "sch.vpd-cluster-episode-routing.v1",
        },
        "benchmark": {
            "path": _relative(router_benchmark_path, project_root),
            "sha256": sha256_file(router_benchmark_path),
            "benchmark_id": router_benchmark["benchmark_id"],
            "verification": router_verification,
            "controls": 16,
        },
        "predecessor_gap_audit": {
            "path": _relative(gap_audit_path, project_root),
            "sha256": sha256_file(gap_audit_path),
            "audit_id": audit_verification["audit_id"],
            "finding": audit_verification["finding"],
            "cluster_bound_training_jobs": audit_verification[
                "cluster_bound_training_jobs"
            ],
        },
        "empirical_receipt_present": False,
        "sae_training_authorized": False,
    }
    result["component_compartments"]["cluster_aware_sae_training_plan_v1"] = {
        "schema": "sch.vpd-cluster-aware-sae-training-plan-contract.v1",
        "status": "manifest_only_successor_planner_ready_empirical_inputs_absent",
        "requires_discovery_cluster_validation_and_sae_training_routing_receipts": True,
        "reproduces_routing_stability_before_planning": True,
        "specific_compartments_bind_exact_component_membership": True,
        "under_supported_routes_merge_to_shared": True,
        "cross_route_pair_conflicts_merge_to_shared": True,
        "pair_atomic_train_validation": True,
        "two_initialization_seeds_required": True,
        "resolved_job_capacity_must_match_profile": True,
        "runtime": {
            "path": _relative(planner_runtime, project_root),
            "sha256": sha256_file(planner_runtime),
            "schema": "sch.vpd-cluster-aware-sae-training-plan.v1",
        },
        "benchmark": {
            "path": _relative(planner_benchmark_path, project_root),
            "sha256": sha256_file(planner_benchmark_path),
            "benchmark_id": planner_benchmark["benchmark_id"],
            "verification": planner_verification,
            "controls": 16,
            "synthetic_plan_id": planner_benchmark["results"][
                "cluster_aware_sae_plan_id"
            ],
        },
        "empirical_plan_present": False,
        "sae_training_authorized": False,
    }
    result["unified_lineage"]["cluster_episode_routing_v1_required"] = True
    result["unified_lineage"]["cluster_aware_sae_training_plan_v1_required"] = True
    result["unified_lineage"]["category_only_plan_not_cluster_compartmentalized"] = True
    result["unified_lineage"]["duplicated_category_archives_prohibited"] = True
    result["unified_lineage"]["sae_training_role_routing_receipt_required"] = True
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v20_verification": predecessor_verification,
        "cluster_episode_routing_benchmark_verification": router_verification,
        "cluster_aware_sae_plan_benchmark_verification": planner_verification,
        "category_only_predecessor_gap_audit_verification": audit_verification,
        "router_runtime_sha256": sha256_file(router_runtime),
        "planner_runtime_sha256": sha256_file(planner_runtime),
        "bounded_predecessor_replay": True,
        "research_protocol_v21_is_distinct_from_historical_experiment_labels": True,
    }
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["real_effect_archive_emitted"] = False
    result["scientific_boundary"]["empirical_cluster_routing_receipt_emitted"] = False
    result["scientific_boundary"]["empirical_cluster_aware_sae_plan_emitted"] = False
    result["scientific_boundary"]["weights_loaded_or_mutated"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != build_protocol(**kwargs):
        raise ValueError("research protocol V21 is not reproducible")
    router = observed["component_compartments"]["cluster_episode_routing_v1"]
    planner = observed["component_compartments"][
        "cluster_aware_sae_training_plan_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v21",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "predecessor_sha256": sha256_file(kwargs["predecessor_path"]),
        "router_benchmark_id": router["benchmark"]["benchmark_id"],
        "planner_benchmark_id": planner["benchmark"]["benchmark_id"],
        "empirical_cluster_routing_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v20.json"
        ),
        "router_benchmark_path": (
            project_root / "artifacts/vpd_cluster_episode_routing_benchmark_v1/report.json"
        ),
        "planner_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_training_plan_benchmark_v1/report.json"
        ),
        "gap_audit_path": (
            project_root
            / "artifacts/vpd_compartment_sae_cluster_routing_gap_audit_v1/report.json"
        ),
        "predecessor_sae_plan_path": (
            project_root / "artifacts/vpd_compartment_sae_training_plan_v4/plan.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V21.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.output is not None:
        result = build_protocol(**inputs)
        write_json(args.output, result)
        protocol_sha256 = sha256_file(args.output)
    else:
        result = verify_protocol(args.verify, **inputs)
        protocol_sha256 = result["protocol_sha256"]
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result.get("status", "derived"),
                "protocol_sha256": protocol_sha256,
                "execution_authorized": result.get("execution_authorized", False),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
