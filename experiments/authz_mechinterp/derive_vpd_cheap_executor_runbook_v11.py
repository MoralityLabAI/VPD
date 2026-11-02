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
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v19 import (
    canonical_inputs as protocol_v19_inputs,
    verify_protocol as verify_protocol_v19,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v11"
EXPECTED_V10_RUNBOOK_ID = (
    "d2b30149591145310b25b73f36920c1916ac8a350e4676c277e7b2de0c087713"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V11 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_pinned_runbook_v10(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V10_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V10 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v10":
        raise ValueError("pinned predecessor is not runbook V10")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V10 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "method": "content_address_and_boundary_fields",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v19_path: Path,
    plan_benchmark_path: Path,
    runtime_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_pinned_runbook_v10(predecessor_path)
    protocol_verification = verify_protocol_v19(
        protocol_v19_path, **protocol_v19_inputs(project_root)
    )
    plan_verification = verify_plan_benchmark(plan_benchmark_path)
    runtime_verification = verify_runtime_benchmark(runtime_benchmark_path)
    predecessor = _load(predecessor_path)
    protocol = _load(protocol_v19_path)
    plan_benchmark = _load(plan_benchmark_path)
    runtime_benchmark = _load(runtime_benchmark_path)
    contract = protocol["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "ready_for_local_preflight_real_adapter_model_identity_authorization_"
        "and_actual_capture_pending"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v19"] = {
        "path": _relative(protocol_v19_path, project_root),
        "sha256": sha256_file(protocol_v19_path),
        "verification": protocol_verification,
    }
    result["source"]["canonical_capture_job_plan_benchmark_v1"] = {
        "path": _relative(plan_benchmark_path, project_root),
        "sha256": sha256_file(plan_benchmark_path),
        "benchmark_id": plan_benchmark["benchmark_id"],
        "verification": plan_verification,
    }
    result["source"]["canonical_capture_job_runtime_benchmark_v1"] = {
        "path": _relative(runtime_benchmark_path, project_root),
        "sha256": sha256_file(runtime_benchmark_path),
        "benchmark_id": runtime_benchmark["benchmark_id"],
        "verification": runtime_verification,
    }
    result["source"]["canonical_capture_job_planner_v1"] = contract[
        "canonical_capture_job_planner"
    ]
    result["source"]["canonical_capture_job_runner_v1"] = contract[
        "canonical_capture_job_runner"
    ]
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["canonical_capture_job_plan_benchmark"] = "pass"
    result["current_result"]["canonical_capture_job_runtime_benchmark"] = "pass"
    result["current_result"]["successor_effect_worker_cli_integration_present"] = True
    result["current_result"]["successor_worker_model_loader_implementation_present"] = True
    result["current_result"]["production_canonical_capture_integration_present"] = True
    result["current_result"]["actual_torch_peft_runtime_execution_verified"] = False
    result["current_result"]["real_adapter_factor_bundle_receipt_present"] = False
    result["current_result"]["actual_model_category_effect_capture_verified"] = False
    preflight = result["stages"][0]
    preflight["commands"].extend(
        [
            {
                "id": "verify_canonical_capture_job_plan_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_plan_v1",
                    "--verify",
                    "artifacts/vpd_canonical_capture_job_plan_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_canonical_capture_job_runtime_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_runtime_v1",
                    "--verify",
                    "artifacts/vpd_canonical_capture_job_runtime_benchmark_v1/report.json",
                ],
            },
        ]
    )
    preflight["expected"]["canonical_capture_job_plan_benchmark"] = (
        "exact model/adapter/split/component/resource plan; nine controls pass"
    )
    preflight["expected"]["canonical_capture_job_runtime_benchmark"] = (
        "fake-PEFT category capture and shard assembly; eight controls pass; no Torch import"
    )
    preflight["stop_if"].append(
        "protocol V19 or canonical capture planner/runtime evidence differs"
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "exact capture-job planner, authorization verifier, lazy model loader, canonical "
        "category worker, provenance-complete assembler, and receipt verifier implemented; "
        "real adapter/model identities, user authorization, actual Torch smoke, and empirical "
        "capture receipts absent"
    )
    gate["stop_if"] = [
        "raw LoRA factor rank is used as the component identity",
        "authenticated real-adapter factor-bundle extraction receipt is absent",
        "frozen immutable real-model identity manifest is absent",
        "exact V2/V3 semantic split or dataset hash differs",
        "pre-existing user authorization receipt does not bind the exact capture job plan ID and forward-pass ceiling",
        "Torch, Transformers, or PEFT would be imported before plan and authorization preflight passes",
        "authorized actual-Torch/PEFT canonical runtime smoke receipt is absent",
        "any canonical component, category, pair, condition, template, sketch, artifact, restoration, or receipt check fails",
        "output directory exists or is outside the exact plan",
    ]
    result["completion_definition"]["model_free_successor_worker_assembler_ready"] = True
    result["completion_definition"]["worker_batch_and_split_identity_ready"] = True
    result["completion_definition"]["worker_semantic_strata_sketches_ready"] = True
    result["completion_definition"]["successor_effect_worker_cli_ready"] = True
    result["completion_definition"]["successor_worker_model_loader_implementation_ready"] = True
    result["completion_definition"]["exact_capture_authorization_verifier_ready"] = True
    result["completion_definition"]["successor_worker_model_loader_ready"] = False
    result["completion_definition"]["actual_torch_peft_runtime_smoke_ready"] = False
    result["completion_definition"]["actual_model_category_capture_ready"] = False
    result["scientific_boundary"]["real_adapter_access_authorized"] = False
    result["scientific_boundary"]["model_or_torch_execution_authorized"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V11 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v11",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "successor_effect_worker_cli_ready": True,
        "successor_worker_model_loader_implementation_ready": True,
        "actual_torch_peft_runtime_smoke_ready": False,
        "actual_model_category_capture_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v10/runbook.json"
        ),
        "protocol_v19_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v19.json"
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
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V11.")
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
            {key: result[key] for key in ("schema", "status", "runbook_id")},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
