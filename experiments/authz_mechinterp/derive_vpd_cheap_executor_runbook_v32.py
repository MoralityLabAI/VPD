from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_inert_attachment_v1 import (
    verify_benchmark as verify_attachment_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v40 import (
    canonical_inputs as protocol_v40_inputs,
    verify_protocol as verify_protocol_v40,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v32"
EXPECTED_V31_RUNBOOK_ID = (
    "91108681b156f3ad3e6ac7449d32fa8e052147dd04ea0ba5109fe39fb69e476e"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V32 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V31_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V31 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v31":
        raise ValueError("pinned predecessor is not runbook V31")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V31 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v40_path: Path,
    attachment_path: Path,
    attachment_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v40(
        protocol_v40_path, **protocol_v40_inputs(project_root)
    )
    benchmark_verification = verify_attachment_benchmark(
        attachment_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    benchmark = _load(attachment_benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "cluster_aware_sae_inert_attachment_ready_synthetic_controls_pass_"
        "production_runtime_trainer_empirical_blueprint_real_receipts_and_execution_absent"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v40"] = {
        "path": _relative(protocol_v40_path, project_root),
        "sha256": sha256_file(protocol_v40_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_aware_sae_inert_attachment_v1"] = {
        "path": _relative(attachment_path, project_root),
        "sha256": sha256_file(attachment_path),
    }
    result["source"]["cluster_aware_sae_inert_attachment_benchmark_v1"] = {
        "path": _relative(attachment_benchmark_path, project_root),
        "sha256": sha256_file(attachment_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["cluster_aware_sae_inert_attachment_v1"] = (
        "pass_20_of_20_claim_order_ledger_replay_factory_failure_active_runtime_and_no_callback_controls"
    )
    result["current_result"]["durable_attachment_claim_implemented"] = True
    result["current_result"]["worker_ledger_reconstruction_implemented"] = True
    result["current_result"]["synthetic_inert_runtime_attached"] = True
    result["current_result"]["production_runtime_attach_implemented"] = False
    result["current_result"]["trainer_callback_invoked"] = False
    result["current_result"]["empirical_cluster_aware_blueprint_present"] = False
    result["current_result"]["real_scientific_release_receipt_present"] = False
    result["current_result"]["real_compute_authorization_present"] = False

    stage = result["stages"][0]
    stage["commands"].extend(
        [
            {
                "id": "verify_cluster_aware_sae_inert_attachment_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_inert_attachment_v1",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_inert_attachment_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v40",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v40",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v40.json",
                ],
            },
        ]
    )
    stage["expected"]["cluster_aware_sae_inert_attachment_v1"] = (
        "20 controls pass; durable claim precedes factory; exact ledger reconstructed; inert runtime only; callback absent"
    )
    stage["stop_if"].extend(
        [
            "protocol V40 or inert-attachment benchmark differs",
            "a runtime factory runs before its durable attachment claim exists",
            "ledger reconstruction differs from the signed transition",
            "an active runtime is accepted or a trainer callback is invoked",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "durable inert-runtime attachment claim and exact worker-ledger reconstruction pass synthetic controls; "
        "production runtime attach, trainer invocation, empirical blueprint, real receipts, and execution authority are absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "inert attachment evidence is treated as production model loading authority",
        "a failed attachment claim tombstone is deleted or retried",
        "the worker trainer callback is invoked from the inert attachment path",
    ]
    result["completion_definition"]["cluster_aware_sae_inert_attachment_ready"] = (
        True
    )
    result["completion_definition"]["durable_attachment_claim_ready"] = True
    result["completion_definition"]["worker_ledger_reconstruction_ready"] = True
    result["completion_definition"]["synthetic_inert_runtime_attached"] = True
    result["completion_definition"]["production_runtime_attach_implemented"] = False
    result["completion_definition"]["trainer_callback_invoked"] = False
    result["completion_definition"]["empirical_cluster_aware_blueprint_present"] = (
        False
    )
    result["completion_definition"]["real_scientific_release_receipt_present"] = False
    result["completion_definition"]["real_compute_authorization_present"] = False
    result["scientific_boundary"]["temporary_inert_attachment_fixture_only"] = True
    result["scientific_boundary"]["cluster_aware_inert_attachment_ready"] = True
    result["scientific_boundary"]["durable_attachment_claim_implemented"] = True
    result["scientific_boundary"]["worker_ledger_reconstruction_implemented"] = True
    result["scientific_boundary"]["synthetic_inert_runtime_attached"] = True
    result["scientific_boundary"]["production_runtime_attach_implemented"] = False
    result["scientific_boundary"]["trainer_callback_invoked"] = False
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
        raise ValueError("cheap-executor runbook V32 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v32",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "cluster_aware_inert_attachment_ready": True,
        "durable_attachment_claim_implemented": True,
        "worker_ledger_reconstruction_implemented": True,
        "synthetic_inert_runtime_attached": True,
        "production_runtime_attach_implemented": False,
        "trainer_callback_invoked": False,
        "empirical_cluster_aware_blueprint_present": False,
        "real_scientific_release_receipt_present": False,
        "real_compute_authorization_present": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v31/runbook.json"
        ),
        "protocol_v40_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v40.json"
        ),
        "attachment_path": (
            project_root
            / "experiments/authz_mechinterp/attach_cluster_aware_sae_inert_runtime_v1.py"
        ),
        "attachment_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_inert_attachment_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V32.")
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
