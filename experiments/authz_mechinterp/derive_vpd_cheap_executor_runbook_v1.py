from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v1"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("cheap-executor runbook source escapes the project")
    return resolved.relative_to(root.resolve()).as_posix()


def _source(path: Path, root: Path, identifier: tuple[str, str] | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": _relative(path, root),
        "sha256": sha256_file(path),
    }
    if identifier is not None:
        field, name = identifier
        value[name] = str(_load(path)[field])
    return value


def build_runbook(
    *,
    training_plan_path: Path,
    release_gate_path: Path,
    runtime_benchmark_path: Path,
    protocol_path: Path,
    unified_receipt_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    plan = _load(training_plan_path)
    gate = _load(release_gate_path)
    benchmark = _load(runtime_benchmark_path)
    protocol = _load(protocol_path)
    receipt = _load(unified_receipt_path)

    benchmark_body = dict(benchmark)
    benchmark_id = str(benchmark_body.pop("benchmark_id", ""))
    if content_sha256(benchmark_body) != benchmark_id:
        raise ValueError("runtime benchmark content address mismatch")
    receipt_body = dict(receipt)
    receipt_id = str(receipt_body.pop("pipeline_receipt_id", ""))
    if content_sha256(receipt_body) != receipt_id:
        raise ValueError("unified receipt content address mismatch")
    runtime = protocol["sae_training"]["execution_runtime_v2"]
    if runtime["training_plan"]["training_plan_id"] != plan["training_plan_id"]:
        raise ValueError("protocol training plan lineage mismatch")
    if runtime["release_gate"]["release_gate_plan_id"] != gate["release_gate_plan_id"]:
        raise ValueError("protocol release gate lineage mismatch")
    if runtime["synthetic_benchmark"]["benchmark_id"] != benchmark_id:
        raise ValueError("protocol runtime benchmark lineage mismatch")

    training_jobs = list(plan["training_jobs"])
    if len(training_jobs) != 18:
        raise ValueError("cheap-executor runbook requires exactly 18 training jobs")
    guarded_jobs = list(gate["guarded_execution"]["guarded_jobs"])
    if len(guarded_jobs) != 22:
        raise ValueError("cheap-executor runbook requires exactly 22 guarded jobs")

    canary = next(
        job
        for job in training_jobs
        if job["compartment"] == "other_invalid"
        and int(job["initialization_seed"]) == 17
    )
    remaining_training = [
        job for job in training_jobs if job["job_id"] != canary["job_id"]
    ]
    remaining_training.sort(
        key=lambda job: (
            job["compartment"] in {"missing_or_untrusted_authority", "shared_cross_category"},
            int(job["expected_support"]["activation_rows"]),
            str(job["compartment"]),
            int(job["initialization_seed"]),
        )
    )

    scientific_release = gate["release_job"]
    local_python = ".venv/Scripts/python.exe"
    body = {
        "schema": SCHEMA,
        "status": "ready_for_low_cost_executor_preflight_paid_compute_not_authorized",
        "source": {
            "training_plan": _source(
                training_plan_path,
                project_root,
                ("training_plan_id", "training_plan_id"),
            ),
            "release_gate": _source(
                release_gate_path,
                project_root,
                ("release_gate_plan_id", "release_gate_plan_id"),
            ),
            "runtime_benchmark": _source(
                runtime_benchmark_path,
                project_root,
                ("benchmark_id", "benchmark_id"),
            ),
            "research_protocol_v9": _source(protocol_path, project_root),
            "unified_lineage_v7": _source(
                unified_receipt_path,
                project_root,
                ("pipeline_receipt_id", "pipeline_receipt_id"),
            ),
            "derivation_runner_sha256": sha256_file(Path(__file__)),
        },
        "current_result": {
            "historical_v7_release_decision": "reject_base_and_adapter",
            "base_separation": 0.692347,
            "adapter_separation": 0.580743,
            "adapter_minus_base": -0.111604,
            "adapter_minus_base_ci95": [-0.122397, -0.100996],
            "invalid_request_rate": 0.0,
            "valid_propose_rate": 1.0,
            "runtime_contract_scenarios_passed": int(benchmark["summary"]["passed"]),
            "runtime_contract_scenarios_total": int(benchmark["summary"]["scenarios"]),
            "full_local_tests_passed": 217,
            "full_local_subtests_passed": 3,
            "empirical_activation_capture_present": False,
            "empirical_sae_dictionary_present": False,
        },
        "executor_policy": {
            "default_action": "run_only_local_preflight_then_stop",
            "never_fabricate_missing_inputs_or_receipts": True,
            "never_generate_external_compute_authorization": True,
            "never_use_final_evaluation_for_tuning": True,
            "never_mutate_base_or_adapter_weights": True,
            "never_execute_real_actions": True,
            "do_not_repeat_historical_v7_paid_inference": True,
            "preserve_recovered_vs_reconstructed_provenance": True,
        },
        "placeholders": {
            "scientific_release_inputs": [
                "FROZEN_CATEGORY_BUNDLE_DIR",
                "COMPONENT_INVENTORY",
                "DISCOVERY_EFFECTS",
                "DISCOVERY_MERGE_MANIFEST",
                "PARTITION_AUDIT_MERGE_DIR",
                "PARTITION_AUDIT_REPORT",
            ],
            "paid_compute_inputs": [
                "SAE_TRAINING_RELEASE_RECEIPT",
                "SAE_COMPUTE_AUTHORIZATION_RECEIPT",
                "FROZEN_BASE_MODEL",
                "FROZEN_BASE_REVISION",
                "SAE_TRAINING_DEVICE",
            ],
        },
        "stages": [
            {
                "stage": 0,
                "id": "local_preflight",
                "authorization": "none_local_model_free",
                "safe_for_cheap_executor_now": True,
                "commands": [
                    {
                        "id": "verify_runbook",
                        "executable": local_python,
                        "arguments": [
                            "-m",
                            "experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v1",
                            "--verify",
                            "artifacts/vpd_cheap_executor_runbook_v1/runbook.json",
                        ],
                    },
                    {
                        "id": "verify_runtime_benchmark",
                        "executable": local_python,
                        "arguments": [
                            "-m",
                            "experiments.authz_mechinterp.benchmark_compartment_sae_runtime_v2",
                            "--verify",
                            "artifacts/vpd_compartment_sae_runtime_benchmark_v2/report.json",
                        ],
                    },
                    {
                        "id": "fast_runtime_tests",
                        "executable": local_python,
                        "arguments": [
                            "-m",
                            "pytest",
                            "tests/test_vpd_compartment_sae_runtime_v2.py",
                            "tests/test_vpd_compartment_sae_runtime_lineage_v2.py",
                            "-q",
                            "-k",
                            "not plan_v4_and_release_gate_v2_reproduce",
                        ],
                    },
                ],
                "expected": {
                    "runbook_verification": "pass",
                    "runtime_benchmark": "11/11 pass",
                    "fast_tests": "8 passed, 1 deselected",
                },
                "stop_if": ["any command exits nonzero", "any source hash changes"],
            },
            {
                "stage": 1,
                "id": "scientific_release_prerequisite",
                "authorization": "empirical_inputs_required_compute_remains_false",
                "safe_for_cheap_executor_now": False,
                "job_id": str(scientific_release["job_id"]),
                "required_placeholders": list(scientific_release["runtime_placeholders"]),
                "command": {
                    "executable": local_python,
                    "arguments": list(scientific_release["arguments"]),
                },
                "expected_output": str(scientific_release["output"]),
                "expected_schema": str(scientific_release["required_output_schema"]),
                "expected_status": str(scientific_release["required_output_status"]),
                "invariant": "receipt_compute_authorized_must_remain_false",
                "stop_if": [
                    "any required placeholder is unresolved",
                    "fresh partition audit is absent or fails",
                    "candidate or source hashes differ",
                ],
            },
            {
                "stage": 2,
                "id": "sae_training_capture",
                "authorization": "scientific_release_plus_external_exact_job_compute_authorization",
                "safe_for_cheap_executor_now": False,
                "job": plan["capture_jobs"][0],
                "expected_numeric_bytes": int(
                    plan["capture_jobs"][0]["expected_capture_resources"][
                        "array_bytes_all_layers"
                    ]
                ),
                "stop_if": [
                    "release receipt absent or not V2 pass",
                    "external authorization does not cover this exact job ID and action",
                    "model identity or revision differs",
                    "capture execution receipt fails independent validation",
                ],
            },
            {
                "stage": 3,
                "id": "canary_dictionary",
                "authorization": "scientific_release_plus_external_exact_job_compute_authorization",
                "safe_for_cheap_executor_now": False,
                "rationale": "exercise one full V2 training handoff before launching the other 17 replicas",
                "job": canary,
                "acceptance": [
                    "exit code zero",
                    "manifest schema sch.vpd-fixed-sae-dictionary-manifest.v2",
                    "manifest independently validates exact samples and hyperparameters",
                    "no base or adapter weights changed",
                ],
                "stop_if": ["any acceptance condition fails"],
            },
            {
                "stage": 4,
                "id": "remaining_dictionaries",
                "authorization": "scientific_release_plus_external_exact_job_compute_authorization",
                "safe_for_cheap_executor_now": False,
                "parallelism": "bounded_by_explicit_resource_budget; never infer concurrency",
                "jobs": remaining_training,
                "acceptance": "all 18 planned V2 manifests present and independently valid",
                "stop_if": [
                    "any job lacks exact authorization scope",
                    "any manifest or dictionary hash fails",
                    "any exact sample contract differs",
                ],
            },
            {
                "stage": 5,
                "id": "calibration_capture",
                "authorization": "scientific_release_plus_external_exact_job_compute_authorization",
                "safe_for_cheap_executor_now": False,
                "job": plan["capture_jobs"][1],
                "expected_numeric_bytes": int(
                    plan["capture_jobs"][1]["expected_capture_resources"][
                        "array_bytes_all_layers"
                    ]
                ),
                "cost_control": "run only after every training manifest validates",
                "stop_if": ["capture execution receipt fails independent validation"],
            },
            {
                "stage": 6,
                "id": "dictionary_bundle_admission",
                "authorization": "scientific_release_plus_external_exact_job_compute_authorization",
                "safe_for_cheap_executor_now": False,
                "job": plan["admission_job"],
                "acceptance": "pass_ready_for_calibration_quality_v2",
                "stop_if": ["job coverage is not exactly all 18 planned dictionaries"],
            },
            {
                "stage": 7,
                "id": "calibration_quality",
                "authorization": "scientific_release_plus_external_exact_job_compute_authorization",
                "safe_for_cheap_executor_now": False,
                "job": plan["quality_job"],
                "acceptance": "quality report status pass",
                "scientific_limit": "quality pass does not authorize release or focused interventions",
                "stop_if": [
                    "bundle admission differs",
                    "calibration capture differs",
                    "any compartment or replica gate fails",
                ],
            },
        ],
        "resource_summary": plan["resource_summary"],
        "completion_definition": {
            "cheap_executor_task_complete_after_stage_0": True,
            "paid_pipeline_complete_only_after_stage_7": True,
            "release_approved_after_stage_7": False,
            "next_required_evidence_after_quality": "held_out_signed_activation_and_parameter_interventions",
        },
        "scientific_boundary": {
            "model_inference_performed_by_runbook_derivation": False,
            "activation_capture_performed_by_runbook_derivation": False,
            "sae_training_performed_by_runbook_derivation": False,
            "weights_loaded_or_mutated": False,
            "execution_authorized": False,
        },
    }
    return {**body, "runbook_id": content_sha256(body)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if content_sha256(body) != identifier:
        raise ValueError("cheap-executor runbook content address mismatch")
    expected = build_runbook(**kwargs)
    if observed != expected:
        raise ValueError("cheap-executor runbook is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "guarded_jobs": 22,
        "safe_stage_now": 0,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "training_plan_path": project_root
        / "artifacts/vpd_compartment_sae_training_plan_v4/plan.json",
        "release_gate_path": project_root
        / "artifacts/vpd_compartment_sae_training_release_gate_v2/plan.json",
        "runtime_benchmark_path": project_root
        / "artifacts/vpd_compartment_sae_runtime_benchmark_v2/report.json",
        "protocol_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v9.json",
        "unified_receipt_path": project_root
        / "artifacts/vpd_unified_compartment_pipeline_v7/receipt.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive or verify the low-cost VPD executor handoff runbook."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None:
        result = verify_runbook(args.verify, **inputs)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_runbook(**inputs)
        write_json(args.output, result)
    print(
        json.dumps(
            {
                "schema": str(result["schema"]),
                "status": str(result["status"]),
                "runbook_id": str(result["runbook_id"]),
                "execution_authorized": bool(
                    result.get("execution_authorized", False)
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
