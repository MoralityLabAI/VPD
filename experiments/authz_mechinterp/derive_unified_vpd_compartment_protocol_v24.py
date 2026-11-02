from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_three_role_capture_to_sae_integration_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json


SCHEMA = "sch.vpd-activation-category-protocol.v24"
EXPECTED_V23_SHA256 = (
    "8eda03fd5c982af058da674feff55c223b58598a8d38422fd8bb27afc34e5c49"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V24 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V23_SHA256:
        raise ValueError("protocol V23 differs from its pinned SHA-256")
    predecessor = json.loads(path.read_text(encoding="utf-8"))
    if predecessor.get("schema") != "sch.vpd-activation-category-protocol.v23":
        raise ValueError("pinned predecessor is not protocol V23")
    if predecessor["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("protocol V23 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-predecessor-verification.v1",
        "status": "pass",
        "method": "exact_sha256_and_boundary_fields",
        "sha256": EXPECTED_V23_SHA256,
    }


def build_protocol(
    *,
    predecessor_path: Path,
    integration_benchmark_path: Path,
    capture_router_adapter_path: Path,
    planner_v2_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    benchmark_verification = verify_benchmark(
        integration_benchmark_path, project_root=project_root
    )
    benchmark = json.loads(integration_benchmark_path.read_text(encoding="utf-8"))
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["cluster_aware_sae_training_plan_v2"] = {
        "schema": "sch.vpd-cluster-aware-sae-training-plan-contract.v2",
        "status": "three_role_synthetic_integration_ready_empirical_receipts_pending",
        "capture_router_adapter": {
            "path": _relative(capture_router_adapter_path, project_root),
            "sha256": sha256_file(capture_router_adapter_path),
            "schema": "sch.vpd-capture-receipt-router-input-adapter.v1",
            "verified_capture_artifacts_only": True,
        },
        "planner": {
            "path": _relative(planner_v2_path, project_root),
            "sha256": sha256_file(planner_v2_path),
            "plan_schema": "sch.vpd-cluster-aware-sae-training-plan.v2",
            "job_schema": "sch.vpd-cluster-aware-sae-training-job.v2",
            "same_pair_may_span_activation_category_compartments": True,
            "one_global_pair_split_prevents_cross_compartment_leakage": True,
            "exact_episode_and_pair_lists_persisted": True,
            "v1_preserved_as_synthetic_disjoint_pair_predecessor": True,
            "v1_forbidden_for_real_shared_pair_split": True,
        },
        "integration_benchmark": {
            "path": _relative(integration_benchmark_path, project_root),
            "sha256": sha256_file(integration_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"]["controls"],
            "split_roles": benchmark["design"]["split_roles"],
            "activation_categories": benchmark["design"]["activation_categories"],
            "v1_specific_compartments": benchmark["results"][
                "v1_specific_compartments"
            ],
            "v2_specific_compartments": benchmark["results"]["v2_summary"][
                "specific_compartments"
            ],
            "v2_shared_compartments": benchmark["results"]["v2_summary"][
                "shared_compartments"
            ],
            "v2_training_jobs": benchmark["results"]["v2_summary"][
                "training_jobs"
            ],
        },
        "empirical_capture_receipts_present": False,
        "empirical_routing_receipts_present": False,
        "empirical_cluster_aware_plan_present": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    result["unified_lineage"]["capture_receipt_to_router_adapter_ready"] = True
    result["unified_lineage"]["cluster_aware_sae_training_plan_v2_ready"] = True
    result["unified_lineage"]["cluster_aware_sae_training_plan_v1_real_shared_pair_use_forbidden"] = True
    result["unified_lineage"]["global_pair_split_across_category_compartments_required"] = True
    result["unified_lineage"]["three_role_capture_to_sae_integration_controls"] = 18
    result["unified_lineage"]["empirical_cluster_aware_sae_plan_ready"] = False
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v23_verification": predecessor_verification,
        "three_role_capture_to_sae_integration_verification": benchmark_verification,
        "capture_router_adapter_sha256": sha256_file(capture_router_adapter_path),
        "cluster_aware_sae_planner_v2_sha256": sha256_file(planner_v2_path),
        "bounded_predecessor_replay": True,
        "research_protocol_v24_is_distinct_from_historical_experiment_labels": True,
    }
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["real_effect_archive_emitted"] = False
    result["scientific_boundary"]["empirical_cluster_routing_receipts_emitted"] = False
    result["scientific_boundary"]["empirical_cluster_aware_sae_plan_emitted"] = False
    result["scientific_boundary"]["weights_loaded_or_mutated"] = False
    result["scientific_boundary"]["sae_dictionary_training_authorized"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != build_protocol(**kwargs):
        raise ValueError("research protocol V24 is not reproducible")
    contract = observed["component_compartments"][
        "cluster_aware_sae_training_plan_v2"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v24",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "integration_benchmark_id": contract["integration_benchmark"][
            "benchmark_id"
        ],
        "planner_v2_ready": True,
        "empirical_plan_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v23.json"
        ),
        "integration_benchmark_path": (
            project_root
            / "artifacts/vpd_three_role_capture_to_sae_integration_benchmark_v1/report.json"
        ),
        "capture_router_adapter_path": (
            project_root
            / "experiments/authz_mechinterp/load_vpd_capture_router_inputs_v1.py"
        ),
        "planner_v2_path": (
            project_root
            / "experiments/authz_mechinterp/plan_cluster_aware_sae_training_v2.py"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V24.")
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
