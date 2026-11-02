from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_canonical_vpd_sae_training_capture_plan_v2 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v22 import (
    canonical_inputs as protocol_v22_inputs,
    verify_protocol as verify_protocol_v22,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v14"
EXPECTED_V13_RUNBOOK_ID = (
    "79f1c87f1cfa84fb05d575be0265cdf7e41405e1e791306e1b36c9b41b0ea757"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V14 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V13_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V13 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v13":
        raise ValueError("pinned predecessor is not runbook V13")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V13 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "method": "content_address_and_boundary_fields",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v22_path: Path,
    capture_plan_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v22(
        protocol_v22_path, **protocol_v22_inputs(project_root)
    )
    benchmark_verification = verify_benchmark(capture_plan_benchmark_path)
    predecessor = _load(predecessor_path)
    protocol = _load(protocol_v22_path)
    benchmark = _load(capture_plan_benchmark_path)
    contract = protocol["component_compartments"][
        "sae_training_effect_capture_plan_v2"
    ]
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "ready_for_local_preflight_sae_training_capture_plan_ready_"
        "capture_worker_and_empirical_inputs_pending"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v22"] = {
        "path": _relative(protocol_v22_path, project_root),
        "sha256": sha256_file(protocol_v22_path),
        "verification": protocol_verification,
    }
    result["source"]["sae_training_capture_plan_benchmark_v2"] = {
        "path": _relative(capture_plan_benchmark_path, project_root),
        "sha256": sha256_file(capture_plan_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["sae_training_capture_planner_v2"] = contract["planner"]
    result["source"]["recovered_sae_training_split_v2"] = contract[
        "recovered_sae_training_split"
    ]
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["sae_training_capture_plan_benchmark"] = "pass_12_of_12"
    result["current_result"]["recovered_sae_training_split_v2"] = (
        "pass_6292_episodes_572_pairs"
    )
    result["current_result"]["sae_training_capture_planner_v2_ready"] = True
    result["current_result"]["sae_training_capture_worker_v2_ready"] = False
    result["current_result"]["empirical_sae_training_capture_plan_present"] = False
    result["current_result"]["empirical_sae_training_capture_authorization_present"] = False

    preflight = result["stages"][0]
    preflight["commands"].append(
        {
            "id": "verify_sae_training_capture_plan_benchmark_v2",
            "executable": ".venv/Scripts/python.exe",
            "arguments": [
                "-m",
                "experiments.authz_mechinterp.benchmark_canonical_vpd_sae_training_capture_plan_v2",
                "--verify",
                "artifacts/vpd_canonical_sae_training_capture_plan_benchmark_v2/report.json",
            ],
        }
    )
    preflight["expected"]["sae_training_capture_plan_benchmark"] = (
        "12 controls pass; V1 rejects role; V2 binds role, dataset, ceiling, and separate authority"
    )
    preflight["expected"]["recovered_sae_training_split"] = (
        "6292 episodes, 572 pairs, all eight categories, exact V3 hashes"
    )
    preflight["stop_if"].append(
        "protocol V22, SAE-training capture-plan benchmark, or recovered split differs"
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "exact SAE-training capture planning and authorization verification ready; "
        "V2 execution worker, real adapter/model receipts, and user authorization absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "capture plan does not use schema sch.vpd-canonical-category-capture-job-plan.v2",
        "capture authorization does not use exact V2 role/dataset/ceiling scope",
        "authorization grants SAE dictionary training rather than effect capture only",
        "V2 capture worker implementation or benchmark receipt is absent",
        "empirical SAE-training capture plan or exact user authorization is absent",
    ]
    result["completion_definition"]["sae_training_role_capture_planner_ready"] = True
    result["completion_definition"]["sae_training_role_capture_authorization_verifier_ready"] = True
    result["completion_definition"]["sae_training_role_capture_worker_ready"] = False
    result["completion_definition"]["empirical_sae_training_capture_ready"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V14 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v14",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "sae_training_role_capture_planner_ready": True,
        "sae_training_role_capture_worker_ready": False,
        "empirical_sae_training_capture_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v13/runbook.json"
        ),
        "protocol_v22_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v22.json"
        ),
        "capture_plan_benchmark_path": (
            project_root
            / "artifacts/vpd_canonical_sae_training_capture_plan_benchmark_v2/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive or verify cheap-executor runbook V14."
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
