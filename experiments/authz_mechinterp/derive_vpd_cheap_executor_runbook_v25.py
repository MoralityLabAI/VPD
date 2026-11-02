from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_runtime_v3 import (
    verify_benchmark as verify_runner_benchmark,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_output_reservation_v1 import (
    verify_benchmark as verify_reservation_benchmark,
)
from experiments.authz_mechinterp.benchmark_vpd_guarded_capture_runtime_adapter_v2 import (
    verify_benchmark as verify_adapter_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v33 import (
    canonical_inputs as protocol_v33_inputs,
    verify_protocol as verify_protocol_v33,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v25"
EXPECTED_V24_RUNBOOK_ID = (
    "7a40353f8c085840c587c0cb2fb239a27bf9eb26381a91690993f8c1557c6f6f"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V25 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V24_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V24 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v24":
        raise ValueError("pinned predecessor is not runbook V24")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V24 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v33_path: Path,
    adapter_v2_path: Path,
    adapter_v2_benchmark_path: Path,
    reservation_path: Path,
    reservation_benchmark_path: Path,
    runner_v3_path: Path,
    runner_v3_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v33(
        protocol_v33_path, **protocol_v33_inputs(project_root)
    )
    adapter_verification = verify_adapter_benchmark(
        adapter_v2_benchmark_path, project_root=project_root
    )
    reservation_verification = verify_reservation_benchmark(
        reservation_benchmark_path, project_root=project_root
    )
    runner_verification = verify_runner_benchmark(
        runner_v3_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    adapter_benchmark = _load(adapter_v2_benchmark_path)
    reservation_benchmark = _load(reservation_benchmark_path)
    runner_benchmark = _load(runner_v3_benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "guarded_capture_runner_v3_ready_synthetic_controls_pass_capture_blocked_"
        "on_bindings_plans_real_admission_reservation_and_authority"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v33"] = {
        "path": _relative(protocol_v33_path, project_root),
        "sha256": sha256_file(protocol_v33_path),
        "verification": protocol_verification,
    }
    result["source"]["guarded_capture_runtime_adapter_v2"] = {
        "path": _relative(adapter_v2_path, project_root),
        "sha256": sha256_file(adapter_v2_path),
    }
    result["source"]["guarded_capture_runtime_adapter_benchmark_v2"] = {
        "path": _relative(adapter_v2_benchmark_path, project_root),
        "sha256": sha256_file(adapter_v2_benchmark_path),
        "benchmark_id": adapter_benchmark["benchmark_id"],
        "verification": adapter_verification,
    }
    result["source"]["output_reservation_transition_v1"] = {
        "path": _relative(reservation_path, project_root),
        "sha256": sha256_file(reservation_path),
    }
    result["source"]["output_reservation_transition_benchmark_v1"] = {
        "path": _relative(reservation_benchmark_path, project_root),
        "sha256": sha256_file(reservation_benchmark_path),
        "benchmark_id": reservation_benchmark["benchmark_id"],
        "verification": reservation_verification,
    }
    result["source"]["canonical_capture_runner_v3"] = {
        "path": _relative(runner_v3_path, project_root),
        "sha256": sha256_file(runner_v3_path),
    }
    result["source"]["canonical_capture_runner_benchmark_v3"] = {
        "path": _relative(runner_v3_benchmark_path, project_root),
        "sha256": sha256_file(runner_v3_benchmark_path),
        "benchmark_id": runner_benchmark["benchmark_id"],
        "verification": runner_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["guarded_runtime_adapter_v2"] = (
        "pass_14_of_14_reservation_before_runtime_attachment"
    )
    result["current_result"]["output_reservation_transition_v1"] = (
        "pass_13_of_13_atomic_fail_closed_transition_controls"
    )
    result["current_result"]["canonical_capture_runner_v3"] = (
        "pass_18_of_18_fake_runtime_controls_production_path_unexercised"
    )
    result["current_result"]["production_runner_integration_present"] = True
    result["current_result"]["production_runner_empirically_exercised"] = False
    result["current_result"]["identity_bindings_complete"] = False
    result["current_result"]["actual_recovered_shape_capture_plans_present"] = False
    result["current_result"]["real_reservation_receipt_present"] = False
    result["current_result"]["capture_started"] = False
    result["current_result"]["empirical_capture_ready"] = False

    preflight_stage = result["stages"][0]
    preflight_stage["commands"].extend(
        [
            {
                "id": "verify_guarded_capture_runtime_adapter_benchmark_v2",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_guarded_capture_runtime_adapter_v2",
                    "--verify",
                    "artifacts/vpd_guarded_capture_runtime_adapter_benchmark_v2/report.json",
                ],
            },
            {
                "id": "verify_output_reservation_transition_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_output_reservation_v1",
                    "--verify",
                    "artifacts/vpd_cluster_to_sae_output_reservation_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_canonical_capture_runner_benchmark_v3",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_runtime_v3",
                    "--verify",
                    "artifacts/vpd_canonical_capture_job_runtime_benchmark_v3/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v33",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v33",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v33.json",
                ],
            },
        ]
    )
    preflight_stage["expected"]["guarded_runtime_adapter_v2"] = (
        "14 controls pass; budget reservation precedes runtime attachment"
    )
    preflight_stage["expected"]["output_reservation_transition_v1"] = (
        "13 controls pass; real admission must reverify before atomic root creation"
    )
    preflight_stage["expected"]["canonical_capture_runner_v3"] = (
        "18 controls pass on fake runtime; production path remains empirically unexercised"
    )
    preflight_stage["stop_if"].extend(
        [
            "protocol V33 or any guarded runner benchmark differs",
            "runtime loading begins before job and payload budget reservation",
            "a synthetic admission or reservation is accepted by the production runner",
            "a callback failure is not charged or leaves a committed output",
            "a fake-runtime benchmark is described as a real recovered-shape capture",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "production runner code path and fail-closed synthetic controls pass; identity-complete "
        "bindings, actual plans, real signed admission, atomic reservation, and explicit capture "
        "authority remain absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "production capture begins without exact allocation and output-reservation replay",
        "the runner's production path is called empirically exercised before a real receipt verifies",
        "runner readiness is treated as authorization to load weights or perform inference",
    ]
    result["completion_definition"]["guarded_runtime_adapter_v2_ready"] = True
    result["completion_definition"]["output_reservation_transition_ready"] = True
    result["completion_definition"]["canonical_capture_runner_v3_code_ready"] = True
    result["completion_definition"]["production_runner_integration_present"] = True
    result["completion_definition"]["production_runner_empirically_exercised"] = False
    result["completion_definition"]["identity_bindings_complete"] = False
    result["completion_definition"]["actual_capture_plans_present"] = False
    result["completion_definition"]["real_reservation_receipt_present"] = False
    result["completion_definition"]["capture_started"] = False
    result["completion_definition"]["empirical_capture_ready"] = False
    result["scientific_boundary"]["synthetic_guarded_runner_fixture_only"] = True
    result["scientific_boundary"]["production_runner_code_path_present"] = True
    result["scientific_boundary"]["production_runner_empirically_exercised"] = False
    result["scientific_boundary"]["identity_bindings_complete"] = False
    result["scientific_boundary"]["actual_capture_plans_present"] = False
    result["scientific_boundary"]["real_guard_receipt_present"] = False
    result["scientific_boundary"]["real_reservation_receipt_present"] = False
    result["scientific_boundary"]["persistent_output_root_created"] = False
    result["scientific_boundary"]["capture_started"] = False
    result["scientific_boundary"]["model_framework_imported"] = False
    result["scientific_boundary"]["model_or_adapter_weights_opened"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["sae_training_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V25 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v25",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "guarded_runtime_adapter_v2_ready": True,
        "output_reservation_transition_ready": True,
        "canonical_capture_runner_v3_code_ready": True,
        "production_runner_empirically_exercised": False,
        "identity_bindings_complete": False,
        "actual_capture_plans_present": False,
        "capture_started": False,
        "empirical_capture_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v24/runbook.json"
        ),
        "protocol_v33_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v33.json"
        ),
        "adapter_v2_path": (
            project_root
            / "experiments/authz_mechinterp/vpd_guarded_capture_runtime_adapter_v2.py"
        ),
        "adapter_v2_benchmark_path": (
            project_root
            / "artifacts/vpd_guarded_capture_runtime_adapter_benchmark_v2/report.json"
        ),
        "reservation_path": (
            project_root
            / "experiments/authz_mechinterp/vpd_cluster_to_sae_output_reservation_v1.py"
        ),
        "reservation_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_to_sae_output_reservation_benchmark_v1/report.json"
        ),
        "runner_v3_path": (
            project_root / "experiments/authz_mechinterp/run_canonical_vpd_capture_job_v3.py"
        ),
        "runner_v3_benchmark_path": (
            project_root
            / "artifacts/vpd_canonical_capture_job_runtime_benchmark_v3/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V25.")
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
