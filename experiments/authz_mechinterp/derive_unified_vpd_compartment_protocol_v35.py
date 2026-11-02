from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_materialization_v1 import (
    verify_benchmark as verify_materialization_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v34 import (
    canonical_inputs as protocol_v34_inputs,
    verify_protocol as verify_protocol_v34,
)


SCHEMA = "sch.vpd-activation-category-protocol.v35"
EXPECTED_V34_SHA256 = (
    "4a1d90e56b29f0eca513874aa62d5478731e6d5c529ec2a81f56f2fb2c35b370"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V35 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, *, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V34_SHA256:
        raise ValueError("protocol V34 differs from its pinned SHA-256")
    verification = verify_protocol_v34(path, **protocol_v34_inputs(project_root))
    if verification["execution_authorized"] is not False:
        raise ValueError("protocol V34 authorization boundary differs")
    return verification


def build_protocol(
    *,
    predecessor_path: Path,
    materializer_path: Path,
    materialization_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(
        predecessor_path, project_root=project_root
    )
    benchmark_verification = verify_materialization_benchmark(
        materialization_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    benchmark = json.loads(
        materialization_benchmark_path.read_text(encoding="utf-8")
    )
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["guarded_cluster_aware_sae_materialization_v1"] = {
        "schema": "sch.vpd-cluster-aware-sae-plan-materialization-contract.v1",
        "status": "eight_category_three_role_materializer_ready_synthetic_controls_only",
        "materializer": {
            "path": _relative(materializer_path, project_root),
            "sha256": sha256_file(materializer_path),
            "bundle_schema": "sch.vpd-guarded-cluster-routing-evidence-bundle.v1",
            "materialization_schema": "sch.vpd-cluster-aware-sae-plan-materialization.v1",
        },
        "benchmark": {
            "path": _relative(materialization_benchmark_path, project_root),
            "sha256": sha256_file(materialization_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
            "results": benchmark["results"]["summary"],
        },
        "exact_activation_categories": 8,
        "roles_per_category": 3,
        "guarded_v3_capture_receipts": 3,
        "adapter_receipts": 24,
        "routing_receipts": 24,
        "independent_stability_receipts": 8,
        "specific_compartments": 16,
        "shared_fallback_compartments": 1,
        "cluster_bound_training_jobs": 34,
        "global_pair_split_required": True,
        "synthetic_evidence_as_empirical_rejected": True,
        "persistent_empirical_bundle_present": False,
        "empirical_cluster_aware_plan_present": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    result["unified_lineage"]["guarded_cluster_aware_sae_materializer_ready"] = True
    result["unified_lineage"]["exact_eight_category_route_materialization_ready"] = True
    result["unified_lineage"]["global_pair_safe_cluster_bound_jobs_ready"] = True
    result["unified_lineage"]["persistent_empirical_routing_bundle_present"] = False
    result["unified_lineage"]["empirical_cluster_aware_plan_present"] = False
    result["unified_lineage"]["empirical_capture_ready"] = False
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v34_verification": predecessor_verification,
        "cluster_aware_materialization_benchmark_verification": benchmark_verification,
        "cluster_aware_materialization_derivation_sha256": sha256_file(Path(__file__)),
        "bounded_predecessor_replay": True,
    }
    result["scientific_boundary"]["synthetic_guarded_route_bundle_only"] = True
    result["scientific_boundary"]["exact_eight_category_materializer_present"] = True
    result["scientific_boundary"]["persistent_empirical_routing_bundle_present"] = False
    result["scientific_boundary"]["empirical_cluster_aware_plan_present"] = False
    result["scientific_boundary"]["model_framework_imported"] = False
    result["scientific_boundary"]["model_or_adapter_weights_opened"] = False
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["weights_loaded_or_mutated"] = False
    result["scientific_boundary"]["sae_dictionary_training_authorized"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != build_protocol(**kwargs):
        raise ValueError("research protocol V35 is not reproducible")
    component = observed["component_compartments"][
        "guarded_cluster_aware_sae_materialization_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v35",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "materialization_benchmark_id": component["benchmark"]["benchmark_id"],
        "materialization_controls": component["benchmark"]["controls"]["controls"],
        "activation_categories": component["exact_activation_categories"],
        "cluster_bound_training_jobs": component["cluster_bound_training_jobs"],
        "persistent_empirical_bundle_present": False,
        "empirical_cluster_aware_plan_present": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v34.json"
        ),
        "materializer_path": (
            project_root
            / "experiments/authz_mechinterp/materialize_vpd_cluster_aware_sae_plan_v1.py"
        ),
        "materialization_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_materialization_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V35.")
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
