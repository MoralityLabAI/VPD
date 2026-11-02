from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_transition_v1 import (
    verify_benchmark as verify_transition_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v39 import (
    canonical_inputs as protocol_v39_inputs,
    verify_protocol as verify_protocol_v39,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v31"
EXPECTED_V30_RUNBOOK_ID = (
    "8f3cbc0bff18b1cdc8f85e71b220f28269f2f3db44fac32fdf8883b1689d1d61"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V31 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V30_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V30 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v30":
        raise ValueError("pinned predecessor is not runbook V30")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V30 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v39_path: Path,
    transition_path: Path,
    transition_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v39(
        protocol_v39_path, **protocol_v39_inputs(project_root)
    )
    benchmark_verification = verify_transition_benchmark(
        transition_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    benchmark = _load(transition_benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "cluster_aware_sae_durable_transition_ready_synthetic_controls_pass_"
        "runtime_attach_trainer_empirical_blueprint_real_receipts_and_execution_absent"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v39"] = {
        "path": _relative(protocol_v39_path, project_root),
        "sha256": sha256_file(protocol_v39_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_aware_sae_transition_v1"] = {
        "path": _relative(transition_path, project_root),
        "sha256": sha256_file(transition_path),
    }
    result["source"]["cluster_aware_sae_transition_benchmark_v1"] = {
        "path": _relative(transition_benchmark_path, project_root),
        "sha256": sha256_file(transition_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["cluster_aware_sae_transition_v1"] = (
        "pass_20_of_20_campaign_compute_nonce_job_root_ledger_replay_and_tombstone_controls"
    )
    result["current_result"]["release_campaign_nonce_reservation_implemented"] = True
    result["current_result"][
        "atomic_compute_nonce_and_job_root_reservation_implemented"
    ] = True
    result["current_result"]["worker_ledger_reservation_implemented"] = True
    result["current_result"]["runtime_attach_implemented"] = False
    result["current_result"]["trainer_invocation_implemented"] = False
    result["current_result"]["empirical_cluster_aware_blueprint_present"] = False
    result["current_result"]["real_scientific_release_receipt_present"] = False
    result["current_result"]["real_compute_authorization_present"] = False
    result["current_result"]["capture_started"] = False

    stage = result["stages"][0]
    stage["commands"].extend(
        [
            {
                "id": "verify_cluster_aware_sae_transition_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_transition_v1",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_transition_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v39",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v39",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v39.json",
                ],
            },
        ]
    )
    stage["expected"]["cluster_aware_sae_transition_v1"] = (
        "20 controls pass; release campaign and compute nonce/job root are durable; worker ledger reserved; no runtime or trainer"
    )
    stage["stop_if"].extend(
        [
            "protocol V39 or durable-transition benchmark differs",
            "a release nonce switches campaigns or a compute nonce switches jobs",
            "a partial tombstone is deleted, resumed, or treated as replayable",
            "a runtime attaches or exact dictionary output is created during transition",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "durable release-campaign and atomic compute-nonce/job-root transition plus worker-ledger reservation pass synthetic controls; "
        "runtime attach, trainer invocation, empirical blueprint, real receipts, and execution authority are absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "a durable transition is treated as permission to attach a runtime",
        "a terminal tombstone is rolled back without a separately designed recovery protocol",
        "synthetic transition receipts are relabeled as empirical execution",
    ]
    result["completion_definition"]["cluster_aware_sae_durable_transition_ready"] = (
        True
    )
    result["completion_definition"]["release_campaign_nonce_reservation_ready"] = True
    result["completion_definition"][
        "atomic_compute_nonce_and_job_root_reservation_ready"
    ] = True
    result["completion_definition"]["worker_ledger_reservation_ready"] = True
    result["completion_definition"]["runtime_attach_implemented"] = False
    result["completion_definition"]["trainer_invocation_implemented"] = False
    result["completion_definition"]["empirical_cluster_aware_blueprint_present"] = (
        False
    )
    result["completion_definition"]["real_scientific_release_receipt_present"] = False
    result["completion_definition"]["real_compute_authorization_present"] = False
    result["completion_definition"]["capture_started"] = False
    result["scientific_boundary"]["temporary_transition_fixture_only"] = True
    result["scientific_boundary"]["cluster_aware_durable_transition_ready"] = True
    result["scientific_boundary"]["release_campaign_nonce_reservation_implemented"] = (
        True
    )
    result["scientific_boundary"][
        "atomic_compute_nonce_and_job_root_reservation_implemented"
    ] = True
    result["scientific_boundary"]["worker_ledger_reservation_implemented"] = True
    result["scientific_boundary"]["runtime_attach_implemented"] = False
    result["scientific_boundary"]["trainer_invocation_implemented"] = False
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
        raise ValueError("cheap-executor runbook V31 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v31",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "cluster_aware_durable_transition_ready": True,
        "release_campaign_nonce_reservation_implemented": True,
        "atomic_compute_nonce_and_job_root_reservation_implemented": True,
        "worker_ledger_reservation_implemented": True,
        "runtime_attach_implemented": False,
        "trainer_invocation_implemented": False,
        "empirical_cluster_aware_blueprint_present": False,
        "real_scientific_release_receipt_present": False,
        "real_compute_authorization_present": False,
        "capture_started": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v30/runbook.json"
        ),
        "protocol_v39_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v39.json"
        ),
        "transition_path": (
            project_root
            / "experiments/authz_mechinterp/reserve_cluster_aware_sae_training_transition_v1.py"
        ),
        "transition_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_transition_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V31.")
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
