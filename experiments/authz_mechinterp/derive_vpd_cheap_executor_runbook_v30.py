from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_admission_v1 import (
    verify_benchmark as verify_admission_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v38 import (
    canonical_inputs as protocol_v38_inputs,
    verify_protocol as verify_protocol_v38,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v30"
EXPECTED_V29_RUNBOOK_ID = (
    "d4d7e3d9099b5f454569e8d752fc16081c2f1d5d8b959c5a985f88dc3d3cb3f1"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V30 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V29_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V29 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v29":
        raise ValueError("pinned predecessor is not runbook V29")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V29 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v38_path: Path,
    admission_verifier_path: Path,
    admission_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v38(
        protocol_v38_path, **protocol_v38_inputs(project_root)
    )
    benchmark_verification = verify_admission_benchmark(
        admission_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    benchmark = _load(admission_benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "cluster_aware_sae_signed_admission_and_dry_run_ready_synthetic_controls_pass_"
        "durable_nonce_output_reservation_empirical_blueprint_and_execution_absent"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v38"] = {
        "path": _relative(protocol_v38_path, project_root),
        "sha256": sha256_file(protocol_v38_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_aware_sae_signed_admission_v1"] = {
        "path": _relative(admission_verifier_path, project_root),
        "sha256": sha256_file(admission_verifier_path),
    }
    result["source"]["cluster_aware_sae_admission_benchmark_v1"] = {
        "path": _relative(admission_benchmark_path, project_root),
        "sha256": sha256_file(admission_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["cluster_aware_sae_signed_admission_v1"] = (
        "pass_24_of_24_signature_scope_replay_evidence_kind_and_nonexecution_controls"
    )
    result["current_result"]["cluster_aware_sae_dry_run_cli_implemented"] = True
    result["current_result"]["durable_nonce_consumption_implemented"] = False
    result["current_result"]["atomic_output_reservation_implemented"] = False
    result["current_result"]["production_trainer_invocation_implemented"] = False
    result["current_result"]["empirical_cluster_aware_blueprint_present"] = False
    result["current_result"]["scientific_release_receipt_present"] = False
    result["current_result"]["explicit_compute_authorization_present"] = False
    result["current_result"]["identity_bindings_complete"] = False
    result["current_result"]["capture_started"] = False

    stage = result["stages"][0]
    stage["commands"].extend(
        [
            {
                "id": "verify_cluster_aware_sae_admission_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_admission_v1",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_admission_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v38",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v38",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v38.json",
                ],
            },
        ]
    )
    stage["expected"]["cluster_aware_sae_signed_admission_v1"] = (
        "24 controls pass; Ed25519 release and exact-job compute receipts verify; dry run never reserves or invokes"
    )
    stage["stop_if"].extend(
        [
            "protocol V38 or signed-admission benchmark differs",
            "a signature, time, revocation, nonce, blueprint, job, sample, resource, device, or output binding differs",
            "a synthetic blueprint is treated as eligible for real release",
            "the dry-run CLI consumes a nonce, creates output, attaches a runtime, or invokes training",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "signed cluster-aware release and exact-job compute admission plus dry-run CLI pass synthetic controls; "
        "durable nonce consumption, atomic output reservation, empirical blueprint, real receipts, and trainer invocation are absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "dry-run admission is treated as durable execution admission",
        "a nonce is not consumed atomically with output/job reservation",
        "synthetic keys or receipts are relabeled as real authority",
    ]
    result["completion_definition"]["cluster_aware_sae_signed_admission_ready"] = True
    result["completion_definition"]["cluster_aware_sae_dry_run_cli_ready"] = True
    result["completion_definition"]["durable_nonce_consumption_implemented"] = False
    result["completion_definition"]["atomic_output_reservation_implemented"] = False
    result["completion_definition"]["production_trainer_invocation_implemented"] = False
    result["completion_definition"]["empirical_cluster_aware_blueprint_present"] = (
        False
    )
    result["completion_definition"]["scientific_release_receipt_present"] = False
    result["completion_definition"]["explicit_compute_authorization_present"] = False
    result["completion_definition"]["identity_bindings_complete"] = False
    result["completion_definition"]["capture_started"] = False
    result["scientific_boundary"]["synthetic_signed_admission_fixture_only"] = True
    result["scientific_boundary"]["cluster_aware_signed_admission_ready"] = True
    result["scientific_boundary"]["cluster_aware_dry_run_cli_implemented"] = True
    result["scientific_boundary"]["durable_nonce_consumption_implemented"] = False
    result["scientific_boundary"]["atomic_output_reservation_implemented"] = False
    result["scientific_boundary"]["production_trainer_invocation_implemented"] = False
    result["scientific_boundary"]["empirical_cluster_aware_blueprint_present"] = (
        False
    )
    result["scientific_boundary"]["model_framework_imported"] = False
    result["scientific_boundary"]["model_or_adapter_weights_opened"] = False
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["sae_training_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V30 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v30",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "cluster_aware_signed_admission_ready": True,
        "cluster_aware_dry_run_cli_implemented": True,
        "durable_nonce_consumption_implemented": False,
        "atomic_output_reservation_implemented": False,
        "production_trainer_invocation_implemented": False,
        "empirical_cluster_aware_blueprint_present": False,
        "scientific_release_receipt_present": False,
        "explicit_compute_authorization_present": False,
        "identity_bindings_complete": False,
        "capture_started": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v29/runbook.json"
        ),
        "protocol_v38_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v38.json"
        ),
        "admission_verifier_path": (
            project_root
            / "experiments/authz_mechinterp/admit_cluster_aware_sae_training_v1.py"
        ),
        "admission_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_admission_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V30.")
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
