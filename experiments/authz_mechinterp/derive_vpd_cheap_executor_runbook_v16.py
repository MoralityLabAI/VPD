from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_three_role_capture_to_sae_integration_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v24 import (
    canonical_inputs as protocol_v24_inputs,
    verify_protocol as verify_protocol_v24,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v16"
EXPECTED_V15_RUNBOOK_ID = (
    "60e3a3e8778a2efe435eeaa904d58c11621add35ed105345974fad26e5aa2699"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V16 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V15_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V15 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v15":
        raise ValueError("pinned predecessor is not runbook V15")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V15 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "method": "content_address_and_boundary_fields",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v24_path: Path,
    integration_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v24(
        protocol_v24_path, **protocol_v24_inputs(project_root)
    )
    benchmark_verification = verify_benchmark(
        integration_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    protocol = _load(protocol_v24_path)
    benchmark = _load(integration_benchmark_path)
    contract = protocol["component_compartments"][
        "cluster_aware_sae_training_plan_v2"
    ]
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "ready_for_local_preflight_three_role_capture_to_sae_integration_ready_"
        "empirical_receipts_and_authorization_pending"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v24"] = {
        "path": _relative(protocol_v24_path, project_root),
        "sha256": sha256_file(protocol_v24_path),
        "verification": protocol_verification,
    }
    result["source"]["three_role_capture_to_sae_integration_benchmark_v1"] = {
        "path": _relative(integration_benchmark_path, project_root),
        "sha256": sha256_file(integration_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["capture_receipt_router_adapter_v1"] = contract[
        "capture_router_adapter"
    ]
    result["source"]["cluster_aware_sae_planner_v2"] = contract["planner"]
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["three_role_capture_to_sae_integration"] = (
        "pass_18_of_18_all_eight_categories"
    )
    result["current_result"]["cluster_aware_sae_planner_v1_real_shared_pair_safe"] = False
    result["current_result"]["cluster_aware_sae_planner_v2_ready"] = True
    result["current_result"]["synthetic_v2_specific_compartments"] = 16
    result["current_result"]["synthetic_v2_shared_compartments"] = 1
    result["current_result"]["synthetic_v2_training_jobs"] = 34
    result["current_result"]["empirical_cluster_aware_sae_plan_present"] = False

    preflight = result["stages"][0]
    preflight["commands"].append(
        {
            "id": "verify_three_role_capture_to_sae_integration_v1",
            "executable": ".venv/Scripts/python.exe",
            "arguments": [
                "-m",
                "experiments.authz_mechinterp.benchmark_vpd_three_role_capture_to_sae_integration_v1",
                "--verify",
                "artifacts/vpd_three_role_capture_to_sae_integration_benchmark_v1/report.json",
            ],
        }
    )
    preflight["expected"]["three_role_capture_to_sae_integration"] = (
        "18 controls pass; V1 collapse reproduced; V2 preserves 16 specific plus one shared compartment"
    )
    preflight["stop_if"].append(
        "protocol V24 or three-role integration benchmark differs"
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "verified capture-receipt adapter, three-role routing, and shared-pair-safe "
        "cluster-aware SAE planner V2 pass synthetic controls; empirical capture, "
        "partition admission, routing receipts, and user authorization remain absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "integration benchmark does not pass all 18 controls",
        "planner uses V1 on a split whose pair IDs are shared across activation categories",
        "global pair split is absent or differs across compartments",
        "empirical capture, partition admission, or routing receipt is absent",
    ]
    result["completion_definition"]["capture_receipt_router_adapter_ready"] = True
    result["completion_definition"]["cluster_aware_sae_planner_v2_ready"] = True
    result["completion_definition"]["empirical_cluster_aware_sae_plan_ready"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["sae_training_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V16 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v16",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "cluster_aware_sae_planner_v2_ready": True,
        "empirical_cluster_aware_sae_plan_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v15/runbook.json"
        ),
        "protocol_v24_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v24.json"
        ),
        "integration_benchmark_path": (
            project_root
            / "artifacts/vpd_three_role_capture_to_sae_integration_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive or verify cheap-executor runbook V16."
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
