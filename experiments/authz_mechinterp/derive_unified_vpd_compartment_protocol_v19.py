from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_plan_v1 import (
    verify_benchmark as verify_plan_benchmark,
)
from experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_runtime_v1 import (
    verify_benchmark as verify_runtime_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json


SCHEMA = "sch.vpd-activation-category-protocol.v19"
EXPECTED_V18_SHA256 = (
    "97c445d5ba9ab1ea833e1fd223e2e7d2e576c46451d88775a70488aec4f66e84"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V19 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_pinned_predecessor(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V18_SHA256:
        raise ValueError("protocol V18 differs from its pinned SHA-256")
    predecessor = json.loads(path.read_text(encoding="utf-8"))
    if predecessor.get("schema") != "sch.vpd-activation-category-protocol.v18":
        raise ValueError("pinned predecessor is not protocol V18")
    if predecessor["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("protocol V18 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-predecessor-verification.v1",
        "status": "pass",
        "method": "exact_sha256_and_boundary_fields",
        "sha256": EXPECTED_V18_SHA256,
    }


def build_protocol(
    *,
    predecessor_path: Path,
    plan_benchmark_path: Path,
    runtime_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_pinned_predecessor(predecessor_path)
    plan_verification = verify_plan_benchmark(plan_benchmark_path)
    runtime_verification = verify_runtime_benchmark(runtime_benchmark_path)
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    plan_benchmark = json.loads(plan_benchmark_path.read_text(encoding="utf-8"))
    runtime_benchmark = json.loads(runtime_benchmark_path.read_text(encoding="utf-8"))
    planner = project_root / "experiments/authz_mechinterp/plan_canonical_vpd_capture_job_v1.py"
    runner = project_root / "experiments/authz_mechinterp/run_canonical_vpd_capture_job_v1.py"
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    contract = result["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    contract["status"] = (
        "exact_capture_job_planner_and_lazy_worker_benchmarked_"
        "real_adapter_model_authorization_and_actual_capture_pending"
    )
    contract["canonical_capture_job_planner"] = {
        "path": _relative(planner, project_root),
        "sha256": sha256_file(planner),
        "schema": "sch.vpd-canonical-category-capture-job-plan.v1",
        "exact_model_adapter_split_component_and_resource_binding": True,
        "planner_can_create_authorization": False,
    }
    contract["canonical_capture_job_plan_benchmark"] = {
        "path": _relative(plan_benchmark_path, project_root),
        "sha256": sha256_file(plan_benchmark_path),
        "benchmark_id": plan_benchmark["benchmark_id"],
        "verification": plan_verification,
        "negative_controls": 9,
        "synthetic_inputs_only": True,
    }
    contract["canonical_capture_job_runner"] = {
        "path": _relative(runner, project_root),
        "sha256": sha256_file(runner),
        "preflight_before_framework_import": True,
        "exact_job_authorization_required": True,
        "canonical_category_capture_kernel_used": True,
        "provenance_complete_shard_assembler_used": True,
        "atomic_staging_and_failure_cleanup": True,
        "release_authority_emitted": False,
    }
    contract["canonical_capture_job_runtime_benchmark"] = {
        "path": _relative(runtime_benchmark_path, project_root),
        "sha256": sha256_file(runtime_benchmark_path),
        "benchmark_id": runtime_benchmark["benchmark_id"],
        "verification": runtime_verification,
        "negative_controls": 8,
        "synthetic_fake_peft_only": True,
        "actual_model_loaded": False,
    }
    contract["successor_effect_worker_cli_integration_present"] = True
    contract["successor_worker_model_loader_implementation_present"] = True
    contract["production_canonical_effect_capture_integration_present"] = True
    contract["exact_capture_authorization_verifier_present"] = True
    contract["actual_torch_peft_runtime_execution_verified"] = False
    contract["real_adapter_factor_bundle_receipt_present"] = False
    contract["actual_model_category_effect_capture_verified"] = False
    result["unified_lineage"]["canonical_capture_job_planner_benchmarked"] = True
    result["unified_lineage"]["lazy_successor_effect_worker_benchmarked"] = True
    result["unified_lineage"]["real_adapter_model_identity_and_authorization_required"] = True
    result["unified_lineage"]["actual_model_category_capture_receipt_required"] = True
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v18_verification": predecessor_verification,
        "canonical_capture_job_plan_benchmark_verification": plan_verification,
        "canonical_capture_job_runtime_benchmark_verification": runtime_verification,
        "canonical_capture_job_planner_sha256": sha256_file(planner),
        "canonical_capture_job_runner_sha256": sha256_file(runner),
        "bounded_predecessor_replay": True,
        "research_protocol_v19_is_distinct_from_historical_experiment_labels": True,
    }
    result["scientific_boundary"]["real_adapter_accessed"] = False
    result["scientific_boundary"]["torch_or_peft_loaded"] = False
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["real_effect_archive_emitted"] = False
    result["scientific_boundary"]["weights_loaded_or_mutated"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != build_protocol(**kwargs):
        raise ValueError("research protocol V19 is not reproducible")
    contract = observed["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v19",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "predecessor_sha256": sha256_file(kwargs["predecessor_path"]),
        "plan_benchmark_id": contract["canonical_capture_job_plan_benchmark"][
            "benchmark_id"
        ],
        "runtime_benchmark_id": contract[
            "canonical_capture_job_runtime_benchmark"
        ]["benchmark_id"],
        "successor_worker_model_loader_implementation_present": True,
        "actual_model_category_effect_capture_verified": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v18.json"
        ),
        "plan_benchmark_path": (
            project_root
            / "artifacts/vpd_canonical_capture_job_plan_benchmark_v1/report.json"
        ),
        "runtime_benchmark_path": (
            project_root
            / "artifacts/vpd_canonical_capture_job_runtime_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V19.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None:
        result = verify_protocol(args.verify, **inputs)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_protocol(**inputs)
        write_json(args.output, result)
    protocol_sha256 = result.get("protocol_sha256")
    if protocol_sha256 is None:
        protocol_sha256 = sha256_file(args.output)
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
