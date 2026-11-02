from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_inert_callback_v1 import (
    verify_benchmark as verify_callback_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v41 import (
    canonical_inputs as protocol_v41_inputs,
    verify_protocol as verify_protocol_v41,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v33"
EXPECTED_V32_RUNBOOK_ID = (
    "30bcec255ef18f654134794c14a52d88599264d3c8f387fdca2a991f75c9bacf"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V33 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V32_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V32 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v32":
        raise ValueError("pinned predecessor is not runbook V32")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V32 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v41_path: Path,
    callback_execution_path: Path,
    callback_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v41(
        protocol_v41_path, **protocol_v41_inputs(project_root)
    )
    benchmark_verification = verify_callback_benchmark(
        callback_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    benchmark = _load(callback_benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "cluster_aware_sae_synthetic_inert_callback_receipts_ready_"
        "production_trainer_empirical_blueprint_real_receipts_and_execution_absent"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v41"] = {
        "path": _relative(protocol_v41_path, project_root),
        "sha256": sha256_file(protocol_v41_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_aware_sae_inert_callback_execution_v1"] = {
        "path": _relative(callback_execution_path, project_root),
        "sha256": sha256_file(callback_execution_path),
    }
    result["source"]["cluster_aware_sae_inert_callback_benchmark_v1"] = {
        "path": _relative(callback_benchmark_path, project_root),
        "sha256": sha256_file(callback_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    current = result["current_result"]
    current["cluster_aware_sae_inert_callback_v1"] = (
        "pass_21_of_21_claim_order_success_failure_charge_replay_tamper_and_no_training_controls"
    )
    current["durable_callback_execution_claim_implemented"] = True
    current["synthetic_inert_callback_invoked"] = True
    current["synthetic_callback_failure_charge_persisted"] = True
    current["production_trainer_callback_invoked"] = False
    current["training_performed"] = False
    current["persistent_dictionary_output_created"] = False
    current["empirical_cluster_aware_blueprint_present"] = False
    current["real_scientific_release_receipt_present"] = False
    current["real_compute_authorization_present"] = False

    stage = result["stages"][0]
    stage["commands"].extend(
        [
            {
                "id": "verify_cluster_aware_sae_inert_callback_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_inert_callback_v1",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_inert_callback_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v41",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v41",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v41.json",
                ],
            },
        ]
    )
    stage["expected"]["cluster_aware_sae_inert_callback_v1"] = (
        "21 controls pass; durable claim precedes callback; success and failed charges persist; no model, training, or output"
    )
    stage["stop_if"].extend(
        [
            "protocol V41 or inert-callback benchmark differs",
            "a callback runs before its durable execution claim exists",
            "a failed or invalid callback is not charged terminally",
            "a synthetic callback loads a model, trains weights, or writes output",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "durable synthetic inert-callback success and charged-failure receipts pass controls; "
        "production trainer, empirical blueprint, real receipts, and execution authority are absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "synthetic callback evidence is treated as production training authority",
        "a terminal callback failure is deleted or retried",
        "an inert callback reports model loading, training, or dictionary output",
    ]
    completion = result["completion_definition"]
    completion["cluster_aware_sae_inert_callback_ready"] = True
    completion["durable_callback_execution_claim_ready"] = True
    completion["synthetic_inert_callback_invoked"] = True
    completion["synthetic_callback_failure_charge_persisted"] = True
    completion["production_trainer_callback_invoked"] = False
    completion["training_performed"] = False
    completion["persistent_dictionary_output_created"] = False
    completion["empirical_cluster_aware_blueprint_present"] = False
    completion["real_scientific_release_receipt_present"] = False
    completion["real_compute_authorization_present"] = False
    boundary = result["scientific_boundary"]
    boundary["temporary_inert_callback_fixture_only"] = True
    boundary["cluster_aware_inert_callback_ready"] = True
    boundary["durable_callback_execution_claim_implemented"] = True
    boundary["synthetic_inert_callback_invoked"] = True
    boundary["synthetic_callback_failure_charge_persisted"] = True
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
        raise ValueError("cheap-executor runbook V33 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v33",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "cluster_aware_inert_callback_ready": True,
        "durable_callback_execution_claim_implemented": True,
        "synthetic_inert_callback_invoked": True,
        "synthetic_callback_failure_charge_persisted": True,
        "production_trainer_callback_invoked": False,
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
            project_root / "artifacts/vpd_cheap_executor_runbook_v32/runbook.json"
        ),
        "protocol_v41_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v41.json"
        ),
        "callback_execution_path": (
            project_root
            / "experiments/authz_mechinterp/execute_cluster_aware_sae_inert_callback_v1.py"
        ),
        "callback_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_inert_callback_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V33.")
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
