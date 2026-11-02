from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_guarded_capture_runtime_adapter_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v31 import (
    canonical_inputs as protocol_v31_inputs,
    verify_protocol as verify_protocol_v31,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v23"
EXPECTED_V22_RUNBOOK_ID = (
    "8b39c4b66bfbe6edcd3688d0aff9842bdfd3fecc318bcf3ac2bd3c9a04e6ff1d"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V23 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V22_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V22 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v22":
        raise ValueError("pinned predecessor is not runbook V22")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V22 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v31_path: Path,
    adapter_path: Path,
    adapter_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v31(
        protocol_v31_path, **protocol_v31_inputs(project_root)
    )
    benchmark_verification = verify_benchmark(
        adapter_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    benchmark = _load(adapter_benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "guarded_callback_adapter_ready_empirical_capture_blocked_on_real_"
        "admission_reservation_production_runner_integration_and_capture_authority"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v31"] = {
        "path": _relative(protocol_v31_path, project_root),
        "sha256": sha256_file(protocol_v31_path),
        "verification": protocol_verification,
    }
    result["source"]["guarded_capture_runtime_adapter_v1"] = {
        "path": _relative(adapter_path, project_root),
        "sha256": sha256_file(adapter_path),
    }
    result["source"]["guarded_capture_runtime_adapter_benchmark_v1"] = {
        "path": _relative(adapter_benchmark_path, project_root),
        "sha256": sha256_file(adapter_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["guarded_runtime_adapter"] = (
        "ready_for_injection_into_a_future_versioned_runner"
    )
    result["current_result"]["guarded_runtime_adapter_benchmark"] = (
        "pass_20_of_20_precall_job_payload_forward_episode_failure_and_receipt_controls"
    )
    result["current_result"]["historical_capture_worker_modified"] = False
    result["current_result"]["production_runner_integration_present"] = False
    result["current_result"]["real_guard_receipt_present"] = False
    result["current_result"]["capture_started"] = False
    result["current_result"]["empirical_capture_ready"] = False

    preflight_stage = result["stages"][0]
    preflight_stage["commands"].extend(
        [
            {
                "id": "verify_guarded_capture_runtime_adapter_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_guarded_capture_runtime_adapter_v1",
                    "--verify",
                    "artifacts/vpd_guarded_capture_runtime_adapter_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v31",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v31",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v31.json",
                ],
            },
        ]
    )
    preflight_stage["expected"]["guarded_runtime_adapter_benchmark"] = (
        "20 controls pass; all callbacks are synthetic and non-authorizing"
    )
    preflight_stage["stop_if"].extend(
        [
            "protocol V31, runtime adapter, or adapter benchmark differs",
            "job or payload reservation occurs after the first callback",
            "forward or actual episode accounting occurs after its callback",
            "a failed callback is removed from the consumed budget",
            "job completion receipt differs from the exact reserved allocation",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "campaign guard and injectable pre-call adapter pass synthetic controls; "
        "a versioned production runner, real admission, atomic reservation, and "
        "capture authority remain absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "a cheaper executor edits the pinned historical capture worker",
        "adapter readiness is treated as capture or model-loading authority",
        "a job receipt is treated as a completed role-level guard receipt",
    ]
    result["completion_definition"]["guarded_runtime_adapter_ready"] = True
    result["completion_definition"]["guarded_runtime_adapter_benchmark_ready"] = True
    result["completion_definition"]["production_runner_integration_present"] = False
    result["completion_definition"]["real_guard_receipt_present"] = False
    result["completion_definition"]["capture_started"] = False
    result["completion_definition"]["empirical_capture_ready"] = False
    result["scientific_boundary"]["synthetic_guarded_callbacks_only"] = True
    result["scientific_boundary"]["historical_capture_worker_modified"] = False
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
        raise ValueError("cheap-executor runbook V23 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v23",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "guarded_runtime_adapter_ready": True,
        "production_runner_integration_present": False,
        "real_guard_receipt_present": False,
        "capture_started": False,
        "empirical_capture_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v22/runbook.json"
        ),
        "protocol_v31_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v31.json"
        ),
        "adapter_path": (
            project_root
            / "experiments/authz_mechinterp/vpd_guarded_capture_runtime_adapter_v1.py"
        ),
        "adapter_benchmark_path": (
            project_root
            / "artifacts/vpd_guarded_capture_runtime_adapter_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V23.")
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
