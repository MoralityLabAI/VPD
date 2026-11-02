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
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v32 import (
    canonical_inputs as protocol_v32_inputs,
    verify_protocol as verify_protocol_v32,
)
from experiments.authz_mechinterp.plan_vpd_guarded_capture_job_allocations_v1 import (
    verify_manifest as verify_allocation_manifest,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v24"
EXPECTED_V23_RUNBOOK_ID = (
    "79f888f86513ad58c6640f6d00c4337716566c26ace55263207300c1f0e5bb2f"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V24 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V23_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V23 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v23":
        raise ValueError("pinned predecessor is not runbook V23")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V23 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v32_path: Path,
    allocation_planner_path: Path,
    allocation_manifest_path: Path,
    allocation_benchmark_path: Path,
    binding_verifier_path: Path,
    binding_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v32(
        protocol_v32_path, **protocol_v32_inputs(project_root)
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
    predecessor = _load(predecessor_path)
    allocation = _load(allocation_manifest_path)
    allocation_benchmark = _load(allocation_benchmark_path)
    binding_benchmark = _load(binding_benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "exact_guarded_job_allocations_and_binding_verifier_ready_capture_blocked_"
        "on_identity_complete_plans_real_admission_reservation_runner_and_authority"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v32"] = {
        "path": _relative(protocol_v32_path, project_root),
        "sha256": sha256_file(protocol_v32_path),
        "verification": protocol_verification,
    }
    result["source"]["guarded_capture_job_allocation_planner_v1"] = {
        "path": _relative(allocation_planner_path, project_root),
        "sha256": sha256_file(allocation_planner_path),
    }
    result["source"]["guarded_capture_job_allocation_manifest_v1"] = {
        "path": _relative(allocation_manifest_path, project_root),
        "sha256": sha256_file(allocation_manifest_path),
        "allocation_manifest_id": allocation["allocation_manifest_id"],
        "verification": allocation_verification,
    }
    result["source"]["guarded_capture_job_allocation_benchmark_v1"] = {
        "path": _relative(allocation_benchmark_path, project_root),
        "sha256": sha256_file(allocation_benchmark_path),
        "benchmark_id": allocation_benchmark["benchmark_id"],
        "verification": allocation_benchmark_verification,
    }
    result["source"]["guarded_capture_job_binding_verifier_v1"] = {
        "path": _relative(binding_verifier_path, project_root),
        "sha256": sha256_file(binding_verifier_path),
    }
    result["source"]["guarded_capture_job_binding_benchmark_v1"] = {
        "path": _relative(binding_benchmark_path, project_root),
        "sha256": sha256_file(binding_benchmark_path),
        "benchmark_id": binding_benchmark["benchmark_id"],
        "verification": binding_benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["guarded_job_allocations"] = (
        "ready_64_exact_role_packet_allocations_from_recovered_shape"
    )
    result["current_result"]["guarded_job_allocation_benchmark"] = (
        "pass_18_of_18_exact_partition_totals_address_and_tamper_controls"
    )
    result["current_result"]["guarded_job_binding_verifier"] = (
        "ready_shape_identity_and_output_child_checks"
    )
    result["current_result"]["guarded_job_binding_benchmark"] = (
        "pass_15_of_15_shape_resource_identity_and_output_controls"
    )
    result["current_result"]["identity_bindings_complete"] = False
    result["current_result"]["actual_recovered_shape_capture_plans_present"] = False
    result["current_result"]["production_runner_integration_present"] = False
    result["current_result"]["capture_started"] = False
    result["current_result"]["empirical_capture_ready"] = False

    preflight_stage = result["stages"][0]
    preflight_stage["commands"].extend(
        [
            {
                "id": "verify_guarded_capture_job_allocations_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.plan_vpd_guarded_capture_job_allocations_v1",
                    "--verify",
                    "artifacts/vpd_guarded_capture_job_allocations_v1/manifest.json",
                ],
            },
            {
                "id": "verify_guarded_capture_job_allocation_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_guarded_capture_job_allocations_v1",
                    "--verify",
                    "artifacts/vpd_guarded_capture_job_allocations_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_guarded_capture_job_binding_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_guarded_capture_job_binding_v1",
                    "--verify",
                    "artifacts/vpd_guarded_capture_job_binding_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v32",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v32",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v32.json",
                ],
            },
        ]
    )
    preflight_stage["expected"]["guarded_job_allocations"] = (
        "64 jobs and exact four-role global totals; identity_bindings_complete false"
    )
    preflight_stage["expected"]["guarded_job_binding_benchmark"] = (
        "15 controls pass; identity-complete shape still does not authorize execution"
    )
    preflight_stage["stop_if"].extend(
        [
            "protocol V32, allocation manifest, allocation benchmark, or binding benchmark differs",
            "allocation manifest is reused after the empirical preflight ID changes",
            "plan component membership is not exactly one packet",
            "plan role, dataset, forward, episode, payload, category, or sketch shape differs",
            "plan output is not a new direct child of its reserved campaign root",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "exact role-packet allocations and plan-binding checks pass model-free controls; "
        "identity-complete plans, runtime admission, reservation, production runner, and "
        "capture authority remain absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "the recovered incomplete-preflight allocation manifest is treated as production-ready",
        "a completed preflight is not followed by allocation-manifest regeneration",
        "a job binding pass is treated as runtime admission or capture authority",
    ]
    result["completion_definition"]["guarded_job_allocations_ready"] = True
    result["completion_definition"]["guarded_job_allocation_benchmark_ready"] = True
    result["completion_definition"]["guarded_job_binding_verifier_ready"] = True
    result["completion_definition"]["guarded_job_binding_benchmark_ready"] = True
    result["completion_definition"]["identity_bindings_complete"] = False
    result["completion_definition"]["actual_capture_plans_present"] = False
    result["completion_definition"]["production_runner_integration_present"] = False
    result["completion_definition"]["capture_started"] = False
    result["completion_definition"]["empirical_capture_ready"] = False
    result["scientific_boundary"]["model_free_allocation_and_binding_fixtures_only"] = True
    result["scientific_boundary"]["identity_bindings_complete"] = False
    result["scientific_boundary"]["actual_capture_plans_present"] = False
    result["scientific_boundary"]["production_runner_integration_present"] = False
    result["scientific_boundary"]["real_guard_receipt_present"] = False
    result["scientific_boundary"]["persistent_output_root_created"] = False
    result["scientific_boundary"]["capture_started"] = False
    result["scientific_boundary"]["model_framework_imported"] = False
    result["scientific_boundary"]["model_or_adapter_weights_opened"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["sae_training_authorized"] = False
    result["scientific_boundary"]["final_evaluation_consumed"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V24 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v24",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "guarded_job_allocations_ready": True,
        "guarded_job_binding_verifier_ready": True,
        "identity_bindings_complete": False,
        "production_runner_integration_present": False,
        "capture_started": False,
        "empirical_capture_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v23/runbook.json"
        ),
        "protocol_v32_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v32.json"
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
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V24.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.output is not None:
        result = build_runbook(**inputs)
        write_json(args.output, result)
    else:
        result = verify_runbook(args.verify, **inputs)
    print(
        json.dumps(
            {key: result[key] for key in ("schema", "status", "runbook_id")},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
