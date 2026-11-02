from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_compartment_resource_reconciliation_v1 import (
    verify_benchmark as verify_resource_benchmark,
)
from experiments.authz_mechinterp.build_cluster_aware_sae_compartment_resource_reconciliation_v1 import (
    canonical_inputs as resource_inputs,
    verify_resource_reconciliation,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v43 import (
    canonical_inputs as protocol_v43_inputs,
    verify_protocol as verify_protocol_v43,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v35"
EXPECTED_V34_RUNBOOK_ID = (
    "9e93f62742b81b7080ba6d75fa12f58a20bf882b5f8ae3190194c6e03d7d9b06"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V35 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V34_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V34 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v34":
        raise ValueError("pinned predecessor is not runbook V34")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V34 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v43_path: Path,
    resource_builder_path: Path,
    resource_ledger_path: Path,
    resource_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v43(
        protocol_v43_path, **protocol_v43_inputs(project_root)
    )
    resource_verification = verify_resource_reconciliation(
        resource_ledger_path, **resource_inputs(project_root)
    )
    benchmark_verification = verify_resource_benchmark(
        resource_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    benchmark = _load(resource_benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "reconstructed_exact_compartment_resource_and_terminal_failure_"
        "reconciliation_ready_raw_receipts_empirical_inputs_training_retry_and_execution_absent"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v43"] = {
        "path": _relative(protocol_v43_path, project_root),
        "sha256": sha256_file(protocol_v43_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_aware_sae_resource_reconciliation_builder_v1"] = {
        "path": _relative(resource_builder_path, project_root),
        "sha256": sha256_file(resource_builder_path),
    }
    result["source"]["cluster_aware_sae_resource_reconciliation_v1"] = {
        "path": _relative(resource_ledger_path, project_root),
        "sha256": sha256_file(resource_ledger_path),
        "resource_reconciliation_id": resource_verification[
            "resource_reconciliation_id"
        ],
        "verification": resource_verification,
    }
    result["source"]["cluster_aware_sae_resource_reconciliation_benchmark_v1"] = {
        "path": _relative(resource_benchmark_path, project_root),
        "sha256": sha256_file(resource_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    current = result["current_result"]
    current["cluster_aware_sae_resource_reconciliation_v1"] = (
        "pass_24_of_24_exact_aggregation_provenance_failure_authority_and_tamper_controls"
    )
    current["resource_reconciliation_reconstructed_not_raw"] = True
    current["raw_campaign_receipts_recovered"] = False
    current["synthetic_campaign_activation_categories"] = 8
    current["synthetic_campaign_compartments"] = 17
    current["synthetic_campaign_execution_jobs"] = 34
    current["synthetic_campaign_planned_steps"] = resource_verification[
        "planned_steps"
    ]
    current["synthetic_campaign_planned_activation_rows"] = resource_verification[
        "planned_activation_rows"
    ]
    current["shared_fallback_double_counted"] = False
    current["terminal_failure_retry_authorized"] = False
    current["training_performed"] = False
    current["persistent_dictionary_output_created"] = False
    current["empirical_cluster_aware_blueprint_present"] = False
    current["real_scientific_release_receipt_present"] = False
    current["real_compute_authorization_present"] = False

    stage = result["stages"][0]
    stage["commands"].extend(
        [
            {
                "id": "verify_cluster_aware_sae_resource_reconciliation_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.build_cluster_aware_sae_compartment_resource_reconciliation_v1",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_compartment_resource_reconciliation_v1/ledger.json",
                ],
            },
            {
                "id": "verify_cluster_aware_sae_resource_reconciliation_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_compartment_resource_reconciliation_v1",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_compartment_resource_reconciliation_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v43",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v43",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v43.json",
                ],
            },
        ]
    )
    stage["expected"]["cluster_aware_sae_resource_reconciliation_v1"] = (
        "24 controls pass; 8 categories/17 compartments/34 jobs reconcile to 68000 planned steps and 20480 planned rows; shared fallback separate; retry false"
    )
    stage["stop_if"].extend(
        [
            "protocol V43, resource ledger, or reconciliation benchmark differs",
            "reconstructed summary evidence is presented as a recovered raw campaign receipt",
            "shared fallback resources are double-counted into all category totals",
            "terminal failure reconciliation grants retry or execution authority",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "reconstructed nonexecuting exact compartment resource and failure reconciliation passes controls; "
        "raw campaign receipts, empirical blueprint, real receipts, retry, training, and execution authority are absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "planned resource totals are described as observed training consumption",
        "a synthetic terminal failure is reused as real-job authority",
        "specific and shared dictionary widths are silently collapsed",
    ]
    completion = result["completion_definition"]
    completion["cluster_aware_sae_resource_reconciliation_ready"] = True
    completion["resource_reconciliation_reconstructed_not_raw"] = True
    completion["raw_campaign_receipts_recovered"] = False
    completion["synthetic_campaign_planned_steps"] = resource_verification[
        "planned_steps"
    ]
    completion["synthetic_campaign_planned_activation_rows"] = resource_verification[
        "planned_activation_rows"
    ]
    completion["shared_fallback_double_counted"] = False
    completion["terminal_failure_retry_authorized"] = False
    completion["training_performed"] = False
    completion["persistent_dictionary_output_created"] = False
    completion["empirical_cluster_aware_blueprint_present"] = False
    completion["real_scientific_release_receipt_present"] = False
    completion["real_compute_authorization_present"] = False
    boundary = result["scientific_boundary"]
    boundary["temporary_resource_reconciliation_fixture_only"] = True
    boundary["cluster_aware_compartment_resource_reconciliation_ready"] = True
    boundary["resource_reconciliation_reconstructed_not_raw"] = True
    boundary["raw_campaign_receipts_recovered"] = False
    boundary["shared_fallback_double_counted"] = False
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
        raise ValueError("cheap-executor runbook V35 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v35",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "cluster_aware_resource_reconciliation_ready": True,
        "categories": 8,
        "compartments": 17,
        "jobs": 34,
        "planned_steps": observed["current_result"][
            "synthetic_campaign_planned_steps"
        ],
        "planned_activation_rows": observed["current_result"][
            "synthetic_campaign_planned_activation_rows"
        ],
        "raw_campaign_receipts_recovered": False,
        "shared_fallback_double_counted": False,
        "retry_authorized": False,
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
            project_root / "artifacts/vpd_cheap_executor_runbook_v34/runbook.json"
        ),
        "protocol_v43_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v43.json"
        ),
        "resource_builder_path": (
            project_root
            / "experiments/authz_mechinterp/build_cluster_aware_sae_compartment_resource_reconciliation_v1.py"
        ),
        "resource_ledger_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_compartment_resource_reconciliation_v1/ledger.json"
        ),
        "resource_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_compartment_resource_reconciliation_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V35.")
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
