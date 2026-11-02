from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_budget_admission_policy_v1 import (
    verify_benchmark as verify_budget_benchmark,
)
from experiments.authz_mechinterp.build_cluster_aware_sae_budget_admission_policy_v1 import (
    canonical_inputs as budget_policy_inputs,
    verify_budget_policy,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v44 import (
    canonical_inputs as protocol_v44_inputs,
    verify_protocol as verify_protocol_v44,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v36"
EXPECTED_V35_RUNBOOK_ID = (
    "a2d6a5c4902830ccb18658820db14a4431c7d9bb01d2b5ba29a1b7dc39610917"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V36 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V35_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V35 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v35":
        raise ValueError("pinned predecessor is not runbook V35")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V35 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v44_path: Path,
    budget_policy_builder_path: Path,
    budget_policy_path: Path,
    budget_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v44(
        protocol_v44_path, **protocol_v44_inputs(project_root)
    )
    policy_verification = verify_budget_policy(
        budget_policy_path, **budget_policy_inputs(project_root)
    )
    benchmark_verification = verify_budget_benchmark(
        budget_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    benchmark = _load(budget_benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "nonexecuting_category_compartment_budget_admission_ready_"
        "empirical_inputs_training_and_execution_absent"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v44"] = {
        "path": _relative(protocol_v44_path, project_root),
        "sha256": sha256_file(protocol_v44_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_aware_sae_budget_policy_builder_v1"] = {
        "path": _relative(budget_policy_builder_path, project_root),
        "sha256": sha256_file(budget_policy_builder_path),
    }
    result["source"]["cluster_aware_sae_budget_policy_v1"] = {
        "path": _relative(budget_policy_path, project_root),
        "sha256": sha256_file(budget_policy_path),
        "budget_policy_id": policy_verification["budget_policy_id"],
        "verification": policy_verification,
    }
    result["source"]["cluster_aware_sae_budget_policy_benchmark_v1"] = {
        "path": _relative(budget_benchmark_path, project_root),
        "sha256": sha256_file(budget_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    current = result["current_result"]
    current["cluster_aware_sae_budget_admission_policy_v1"] = (
        "pass_32_of_32_category_compartment_shared_global_binding_authority_and_tamper_controls"
    )
    current["budget_policy_activation_categories"] = 8
    current["budget_policy_compartments"] = 17
    current["budget_policy_jobs_ceiling"] = 34
    current["budget_policy_planned_steps_ceiling"] = 68000
    current["budget_policy_planned_activation_rows_ceiling"] = 20480
    current["shared_fallback_charged_to_categories"] = False
    current["budget_admission_transfers_execution_authority"] = False
    current["training_performed"] = False
    current["persistent_dictionary_output_created"] = False
    current["empirical_cluster_aware_blueprint_present"] = False
    current["real_scientific_release_receipt_present"] = False
    current["real_compute_authorization_present"] = False

    stage = result["stages"][0]
    stage["commands"].extend(
        [
            {
                "id": "verify_cluster_aware_sae_budget_admission_policy_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.build_cluster_aware_sae_budget_admission_policy_v1",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_budget_admission_policy_v1/policy.json",
                ],
            },
            {
                "id": "verify_cluster_aware_sae_budget_admission_policy_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_budget_admission_policy_v1",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_budget_admission_policy_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v44",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v44",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v44.json",
                ],
            },
        ]
    )
    stage["expected"]["cluster_aware_sae_budget_admission_policy_v1"] = (
        "32 controls pass; exact 8-category/17-compartment/34-job reference shape is within 68000-step and 20480-row ceilings; shared fallback charged once; execution authority false"
    )
    stage["stop_if"].extend(
        [
            "protocol V44, budget policy, or budget benchmark differs",
            "any category, specific compartment, shared fallback, or global ceiling is broadened",
            "shared fallback resources are charged into category-specific budgets",
            "budget admission is described as execution, compute, or training authority",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "nonexecuting exact category, specific-compartment, shared-fallback, and global budget policy passes controls; "
        "empirical blueprint, real receipts, training, and execution authority are absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "a budget-admissible synthetic reference proposal is treated as empirical evidence",
        "a shared fallback request is double-counted or charged to categories",
        "a budget ceiling is treated as permission to consume that resource",
    ]
    completion = result["completion_definition"]
    completion["cluster_aware_sae_budget_admission_policy_ready"] = True
    completion["budget_policy_activation_categories"] = 8
    completion["budget_policy_compartments"] = 17
    completion["budget_policy_jobs_ceiling"] = 34
    completion["budget_policy_planned_steps_ceiling"] = 68000
    completion["budget_policy_planned_activation_rows_ceiling"] = 20480
    completion["shared_fallback_charged_to_categories"] = False
    completion["budget_admission_transfers_execution_authority"] = False
    completion["training_performed"] = False
    completion["persistent_dictionary_output_created"] = False
    completion["empirical_cluster_aware_blueprint_present"] = False
    completion["real_scientific_release_receipt_present"] = False
    completion["real_compute_authorization_present"] = False
    boundary = result["scientific_boundary"]
    boundary["temporary_budget_reference_proposal_only"] = True
    boundary["cluster_aware_sae_budget_admission_policy_ready"] = True
    boundary["budget_admission_is_empirical_evidence"] = False
    boundary["shared_fallback_charged_to_categories"] = False
    boundary["budget_admission_transfers_execution_authority"] = False
    boundary["terminal_failure_retry_authorized"] = False
    boundary["production_campaign_executed"] = False
    boundary["production_trainer_callback_invoked"] = False
    boundary["training_performed"] = False
    boundary["persistent_dictionary_output_created"] = False
    boundary["empirical_cluster_aware_blueprint_present"] = False
    boundary["model_framework_imported"] = False
    boundary["model_or_adapter_weights_opened"] = False
    boundary["model_inference_performed"] = False
    boundary["execution_authorized"] = False
    boundary["sae_training_authorized"] = False
    boundary["paid_compute_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V36 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v36",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "cluster_aware_budget_admission_policy_ready": True,
        "categories": 8,
        "compartments": 17,
        "jobs_ceiling": 34,
        "planned_steps_ceiling": 68000,
        "planned_activation_rows_ceiling": 20480,
        "shared_fallback_charged_to_categories": False,
        "budget_admission_transfers_execution_authority": False,
        "training_performed": False,
        "persistent_dictionary_output_created": False,
        "empirical_cluster_aware_blueprint_present": False,
        "real_scientific_release_receipt_present": False,
        "real_compute_authorization_present": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v35/runbook.json"
        ),
        "protocol_v44_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v44.json"
        ),
        "budget_policy_builder_path": (
            project_root
            / "experiments/authz_mechinterp/build_cluster_aware_sae_budget_admission_policy_v1.py"
        ),
        "budget_policy_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_budget_admission_policy_v1/policy.json"
        ),
        "budget_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_budget_admission_policy_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V36.")
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
