from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_capture_campaign_guard_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v30 import (
    canonical_inputs as protocol_v30_inputs,
    verify_protocol as verify_protocol_v30,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v22"
EXPECTED_V21_RUNBOOK_ID = (
    "3dc3f49680e15add3a1e2346662576472ea72673bbee602b157b5714099c82a7"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V22 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V21_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V21 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v21":
        raise ValueError("pinned predecessor is not runbook V21")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V21 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v30_path: Path,
    campaign_guard_path: Path,
    campaign_guard_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v30(
        protocol_v30_path, **protocol_v30_inputs(project_root)
    )
    benchmark_verification = verify_benchmark(
        campaign_guard_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    benchmark = _load(campaign_guard_benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "campaign_guard_ready_empirical_capture_blocked_on_real_admission_"
        "reservation_worker_integration_and_capture_authority"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v30"] = {
        "path": _relative(protocol_v30_path, project_root),
        "sha256": sha256_file(protocol_v30_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_capture_campaign_guard_v1"] = {
        "path": _relative(campaign_guard_path, project_root),
        "sha256": sha256_file(campaign_guard_path),
    }
    result["source"]["cluster_capture_campaign_guard_benchmark_v1"] = {
        "path": _relative(campaign_guard_benchmark_path, project_root),
        "sha256": sha256_file(campaign_guard_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["campaign_guard"] = (
        "ready_four_role_stage_and_resource_meter"
    )
    result["current_result"]["campaign_guard_benchmark"] = (
        "pass_22_of_22_stage_budget_time_cost_zero_action_and_receipt_controls"
    )
    result["current_result"]["real_guard_receipt_present"] = False
    result["current_result"]["capture_worker_guard_integration_present"] = False
    result["current_result"]["capture_started"] = False
    result["current_result"]["empirical_capture_ready"] = False

    preflight_stage = result["stages"][0]
    preflight_stage["commands"].extend(
        [
            {
                "id": "verify_cluster_capture_campaign_guard_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_capture_campaign_guard_v1",
                    "--verify",
                    "artifacts/vpd_cluster_capture_campaign_guard_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v30",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v30",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v30.json",
                ],
            },
        ]
    )
    preflight_stage["expected"]["campaign_guard_benchmark"] = (
        "22 controls pass; all counter simulations remain non-authorizing"
    )
    preflight_stage["stop_if"].extend(
        [
            "protocol V30, campaign guard, or guard benchmark differs",
            "role starts before every exact predecessor capture and gate receipt",
            "counter increment occurs after rather than before a compute call",
            "job, forward, episode, payload, time, or cost ceiling is exceeded",
            "generation, real actions, persistent mutation, or final evaluation is attempted",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "four-role stage/resource guard passes simulations; production worker "
        "integration, real admission, atomic reservation, and capture remain absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "capture worker does not increment the shared guard before every model call",
        "role receipt does not exactly equal its full preflight resource envelope",
        "guard success is treated as model inference or SAE-training authority",
    ]
    result["completion_definition"]["campaign_guard_ready"] = True
    result["completion_definition"]["campaign_guard_benchmark_ready"] = True
    result["completion_definition"]["capture_worker_guard_integration_present"] = False
    result["completion_definition"]["real_guard_receipt_present"] = False
    result["completion_definition"]["capture_started"] = False
    result["completion_definition"]["empirical_capture_ready"] = False
    result["scientific_boundary"]["synthetic_campaign_counter_fixture_only"] = True
    result["scientific_boundary"]["capture_worker_guard_integration_present"] = False
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
        raise ValueError("cheap-executor runbook V22 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v22",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "campaign_guard_ready": True,
        "capture_worker_guard_integration_present": False,
        "real_guard_receipt_present": False,
        "capture_started": False,
        "empirical_capture_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v21/runbook.json"
        ),
        "protocol_v30_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v30.json"
        ),
        "campaign_guard_path": (
            project_root
            / "experiments/authz_mechinterp/vpd_cluster_capture_campaign_guard_v1.py"
        ),
        "campaign_guard_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_capture_campaign_guard_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive or verify cheap-executor runbook V22."
    )
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
