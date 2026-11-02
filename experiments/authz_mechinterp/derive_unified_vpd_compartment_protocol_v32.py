from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_guarded_capture_job_allocations_v1 import (
    verify_benchmark as verify_allocation_benchmark,
)
from experiments.authz_mechinterp.benchmark_vpd_guarded_capture_job_binding_v1 import (
    verify_benchmark as verify_binding_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v31 import (
    canonical_inputs as protocol_v31_inputs,
    verify_protocol as verify_protocol_v31,
)
from experiments.authz_mechinterp.plan_vpd_guarded_capture_job_allocations_v1 import (
    verify_manifest as verify_allocation_manifest,
)


SCHEMA = "sch.vpd-activation-category-protocol.v32"
EXPECTED_V31_SHA256 = (
    "d36d6dc655b4879b391e604965f8a214ae4397cc9c569c1e70a0f66d35712104"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V32 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, *, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V31_SHA256:
        raise ValueError("protocol V31 differs from its pinned SHA-256")
    verification = verify_protocol_v31(path, **protocol_v31_inputs(project_root))
    if verification["execution_authorized"] is not False:
        raise ValueError("protocol V31 authorization boundary differs")
    return verification


def build_protocol(
    *,
    predecessor_path: Path,
    allocation_planner_path: Path,
    allocation_manifest_path: Path,
    allocation_benchmark_path: Path,
    binding_verifier_path: Path,
    binding_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(
        predecessor_path, project_root=project_root
    )
    allocation_verification = verify_allocation_manifest(
        allocation_manifest_path, project_root=project_root
    )
    allocation_benchmark_verification = verify_allocation_benchmark(
        allocation_benchmark_path, project_root=project_root
    )
    binding_benchmark_verification = verify_binding_benchmark(
        binding_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    allocation = json.loads(allocation_manifest_path.read_text(encoding="utf-8"))
    allocation_benchmark = json.loads(
        allocation_benchmark_path.read_text(encoding="utf-8")
    )
    binding_benchmark = json.loads(binding_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["guarded_capture_job_allocations_v1"] = {
        "schema": "sch.vpd-guarded-capture-job-allocation-contract.v1",
        "status": "exact_recovered_shape_allocations_ready_identity_bindings_incomplete",
        "planner": {
            "path": _relative(allocation_planner_path, project_root),
            "sha256": sha256_file(allocation_planner_path),
        },
        "manifest": {
            "path": _relative(allocation_manifest_path, project_root),
            "sha256": sha256_file(allocation_manifest_path),
            "allocation_manifest_id": allocation["allocation_manifest_id"],
            "verification": allocation_verification,
        },
        "benchmark": {
            "path": _relative(allocation_benchmark_path, project_root),
            "sha256": sha256_file(allocation_benchmark_path),
            "benchmark_id": allocation_benchmark["benchmark_id"],
            "verification": allocation_benchmark_verification,
            "controls": allocation_benchmark["summary"],
        },
        "roles": 4,
        "jobs": 64,
        "global_totals": allocation["global_totals"],
        "identity_bindings_complete": False,
        "regenerate_from_future_identity_complete_preflight": True,
        "capture_started": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    result["component_compartments"]["guarded_capture_job_binding_v1"] = {
        "schema": "sch.vpd-guarded-capture-job-binding-contract.v1",
        "status": "shape_and_identity_verifier_ready_real_plans_and_admission_absent",
        "verifier": {
            "path": _relative(binding_verifier_path, project_root),
            "sha256": sha256_file(binding_verifier_path),
        },
        "benchmark": {
            "path": _relative(binding_benchmark_path, project_root),
            "sha256": sha256_file(binding_benchmark_path),
            "benchmark_id": binding_benchmark["benchmark_id"],
            "verification": binding_benchmark_verification,
            "controls": binding_benchmark["summary"],
        },
        "exact_packet_membership_required": True,
        "exact_role_dataset_resource_and_sketch_shape_required": True,
        "identity_complete_plan_still_requires_runtime_admission": True,
        "actual_recovered_shape_capture_plans_present": False,
        "production_runner_integration_present": False,
        "capture_started": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    result["unified_lineage"]["guarded_job_allocations_ready"] = True
    result["unified_lineage"]["guarded_job_allocation_controls"] = allocation_benchmark[
        "summary"
    ]["controls"]
    result["unified_lineage"]["guarded_job_binding_ready"] = True
    result["unified_lineage"]["guarded_job_binding_controls"] = binding_benchmark[
        "summary"
    ]["controls"]
    result["unified_lineage"]["identity_bindings_complete"] = False
    result["unified_lineage"]["production_runner_integration_present"] = False
    result["unified_lineage"]["capture_started"] = False
    result["unified_lineage"]["empirical_capture_ready"] = False
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v31_verification": predecessor_verification,
        "allocation_manifest_verification": allocation_verification,
        "allocation_benchmark_verification": allocation_benchmark_verification,
        "binding_benchmark_verification": binding_benchmark_verification,
        "guarded_allocation_derivation_sha256": sha256_file(Path(__file__)),
        "allocation_manifest_must_be_regenerated_after_binding_completion": True,
        "bounded_predecessor_replay": True,
    }
    result["scientific_boundary"]["model_free_allocation_derivation_only"] = True
    result["scientific_boundary"]["actual_capture_plans_present"] = False
    result["scientific_boundary"]["identity_bindings_complete"] = False
    result["scientific_boundary"]["production_runner_integration_present"] = False
    result["scientific_boundary"]["persistent_output_root_created"] = False
    result["scientific_boundary"]["capture_started"] = False
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
        raise ValueError("research protocol V32 is not reproducible")
    allocation = observed["component_compartments"][
        "guarded_capture_job_allocations_v1"
    ]
    binding = observed["component_compartments"]["guarded_capture_job_binding_v1"]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v32",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "allocation_manifest_id": allocation["manifest"]["allocation_manifest_id"],
        "allocation_benchmark_id": allocation["benchmark"]["benchmark_id"],
        "binding_benchmark_id": binding["benchmark"]["benchmark_id"],
        "jobs": allocation["jobs"],
        "identity_bindings_complete": False,
        "production_runner_integration_present": False,
        "capture_started": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v31.json"
        ),
        "allocation_planner_path": (
            project_root
            / "experiments/authz_mechinterp/plan_vpd_guarded_capture_job_allocations_v1.py"
        ),
        "allocation_manifest_path": (
            project_root / "artifacts/vpd_guarded_capture_job_allocations_v1/manifest.json"
        ),
        "allocation_benchmark_path": (
            project_root
            / "artifacts/vpd_guarded_capture_job_allocations_benchmark_v1/report.json"
        ),
        "binding_verifier_path": (
            project_root
            / "experiments/authz_mechinterp/verify_vpd_guarded_capture_job_binding_v1.py"
        ),
        "binding_benchmark_path": (
            project_root
            / "artifacts/vpd_guarded_capture_job_binding_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V32.")
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
