from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_empirical_budget_binding_readiness_v1 import (
    verify_benchmark as verify_binding_benchmark,
)
from experiments.authz_mechinterp.build_cluster_aware_sae_empirical_budget_binding_readiness_v1 import (
    canonical_inputs as binding_inputs,
    verify_binding_readiness,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v45 import (
    canonical_inputs as protocol_v45_inputs,
    verify_protocol as verify_protocol_v45,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v37"
EXPECTED_V36_RUNBOOK_ID = (
    "69ed3ae8195993372a58bc02231daaa223181383a472ef110c893ebf08804de5"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V37 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V36_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V36 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v36":
        raise ValueError("pinned predecessor is not runbook V36")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V36 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v45_path: Path,
    binding_builder_path: Path,
    binding_readiness_path: Path,
    binding_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v45(
        protocol_v45_path, **protocol_v45_inputs(project_root)
    )
    readiness_verification = verify_binding_readiness(
        binding_readiness_path, **binding_inputs(project_root)
    )
    benchmark_verification = verify_binding_benchmark(
        binding_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    benchmark = _load(binding_benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "empirical_blueprint_budget_binding_readiness_blocked_50_bindings_"
        "missing_training_and_execution_absent"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v45"] = {
        "path": _relative(protocol_v45_path, project_root),
        "sha256": sha256_file(protocol_v45_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_aware_sae_empirical_budget_binding_builder_v1"] = {
        "path": _relative(binding_builder_path, project_root),
        "sha256": sha256_file(binding_builder_path),
    }
    result["source"]["cluster_aware_sae_empirical_budget_binding_readiness_v1"] = {
        "path": _relative(binding_readiness_path, project_root),
        "sha256": sha256_file(binding_readiness_path),
        "binding_readiness_id": readiness_verification["binding_readiness_id"],
        "verification": readiness_verification,
    }
    result["source"]["cluster_aware_sae_empirical_budget_binding_benchmark_v1"] = {
        "path": _relative(binding_benchmark_path, project_root),
        "sha256": sha256_file(binding_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    current = result["current_result"]
    current["cluster_aware_sae_empirical_budget_binding_readiness_v1"] = (
        "pass_29_of_29_exact_missing_placeholder_phase_isolation_authority_and_tamper_controls"
    )
    current["empirical_budget_binding_phases"] = 6
    current["empirical_budget_binding_required_bindings"] = 50
    current["empirical_budget_binding_supplied_bindings"] = 0
    current["empirical_budget_binding_missing_bindings"] = 50
    current["empirical_budget_binding_syntax_complete"] = False
    current["empirical_budget_binding_verified"] = False
    current["signed_authority_verified"] = False
    current["durable_runtime_transition_verified"] = False
    current["training_performed"] = False
    current["persistent_dictionary_output_created"] = False
    current["empirical_cluster_aware_blueprint_present"] = False
    current["real_scientific_release_receipt_present"] = False
    current["real_compute_authorization_present"] = False

    stage = result["stages"][0]
    stage["commands"].extend(
        [
            {
                "id": "verify_cluster_aware_sae_empirical_budget_binding_readiness_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.build_cluster_aware_sae_empirical_budget_binding_readiness_v1",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_empirical_budget_binding_readiness_v1/readiness.json",
                ],
            },
            {
                "id": "verify_cluster_aware_sae_empirical_budget_binding_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_empirical_budget_binding_readiness_v1",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_empirical_budget_binding_readiness_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v45",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v45",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v45.json",
                ],
            },
        ]
    )
    stage["expected"]["cluster_aware_sae_empirical_budget_binding_readiness_v1"] = (
        "29 controls pass; exactly 50 bindings across 6 phases remain missing; placeholders rejected; syntax completion transfers no authority"
    )
    stage["stop_if"].extend(
        [
            "protocol V45, binding worksheet, or binding benchmark differs",
            "any missing or placeholder binding is treated as complete",
            "syntactic completion is described as cross-artifact verification or signed authority",
            "binding readiness is described as execution, compute, or training authority",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "nonexecuting 50-binding readiness worksheet passes controls; all real empirical graph, signed authority, durable transition, training, and execution inputs remain absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "a synthetic complete binding fixture is persisted or presented as empirical",
        "an ID or digest is accepted without independent content authentication",
        "a nonce output worker or runtime binding is accepted as fresh without durable verification",
    ]
    completion = result["completion_definition"]
    completion["cluster_aware_sae_empirical_budget_binding_readiness_ready"] = True
    completion["empirical_budget_binding_phases"] = 6
    completion["empirical_budget_binding_required_bindings"] = 50
    completion["empirical_budget_binding_missing_bindings"] = 50
    completion["empirical_budget_binding_syntax_complete"] = False
    completion["empirical_budget_binding_verified"] = False
    completion["signed_authority_verified"] = False
    completion["durable_runtime_transition_verified"] = False
    completion["training_performed"] = False
    completion["persistent_dictionary_output_created"] = False
    completion["empirical_cluster_aware_blueprint_present"] = False
    completion["real_scientific_release_receipt_present"] = False
    completion["real_compute_authorization_present"] = False
    boundary = result["scientific_boundary"]
    boundary["temporary_complete_binding_fixture_in_memory_only"] = True
    boundary["cluster_aware_sae_empirical_budget_binding_readiness_ready"] = True
    boundary["empirical_budget_binding_syntax_complete"] = False
    boundary["empirical_budget_binding_verified"] = False
    boundary["signed_authority_verified"] = False
    boundary["durable_runtime_transition_verified"] = False
    boundary["synthetic_complete_binding_fixture_persisted"] = False
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
        raise ValueError("cheap-executor runbook V37 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v37",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "empirical_budget_binding_readiness_ready": True,
        "binding_phases": 6,
        "required_bindings": 50,
        "missing_bindings": 50,
        "syntax_complete": False,
        "budget_binding_verified": False,
        "signed_authority_verified": False,
        "durable_runtime_transition_verified": False,
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
            project_root / "artifacts/vpd_cheap_executor_runbook_v36/runbook.json"
        ),
        "protocol_v45_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v45.json"
        ),
        "binding_builder_path": (
            project_root
            / "experiments/authz_mechinterp/build_cluster_aware_sae_empirical_budget_binding_readiness_v1.py"
        ),
        "binding_readiness_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_empirical_budget_binding_readiness_v1/readiness.json"
        ),
        "binding_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_empirical_budget_binding_readiness_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V37.")
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
