from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_semantic_dispatch_slice_v2 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v50 import (
    canonical_inputs as protocol_v50_inputs,
    verify_protocol as verify_protocol_v50,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v42"
EXPECTED_V41_RUNBOOK_ID = (
    "733555f2967840041f005380985305b61b265c8c15615543284258646f2019ad"
)
EXPECTED_V50_SHA256 = (
    "55dc12c6f948c6fc8ecaeb1d9d06d037fbc5a72ab2fd0a5990b453bd7271a951"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V42 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V41_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V41 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v41":
        raise ValueError("pinned predecessor is not runbook V41")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V41 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v50_path: Path,
    release_runtime_verifier_path: Path,
    dispatcher_path: Path,
    benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    if sha256_file(protocol_v50_path) != EXPECTED_V50_SHA256:
        raise ValueError("protocol V50 differs from its pinned SHA-256")
    protocol_verification = verify_protocol_v50(
        protocol_v50_path, **protocol_v50_inputs(project_root)
    )
    benchmark_verification = verify_benchmark(
        benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    benchmark = _load(benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "synthetic_budget_bound_signed_release_and_runtime_admission_verified_"
        "durable_transition_production_attachment_empirical_training_absent"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v50"] = {
        "path": _relative(protocol_v50_path, project_root),
        "sha256": sha256_file(protocol_v50_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_aware_sae_release_runtime_verifier_v2"] = {
        "path": _relative(release_runtime_verifier_path, project_root),
        "sha256": sha256_file(release_runtime_verifier_path),
    }
    result["source"]["cluster_aware_sae_semantic_dispatcher_v2"] = {
        "path": _relative(dispatcher_path, project_root),
        "sha256": sha256_file(dispatcher_path),
    }
    result["source"]["cluster_aware_sae_semantic_dispatch_benchmark_v2"] = {
        "path": _relative(benchmark_path, project_root),
        "sha256": sha256_file(benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "semantic_dispatch_slice_id": benchmark["results"][
            "semantic_dispatch_slice_id"
        ],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    current = result["current_result"]
    current["cluster_aware_sae_semantic_dispatch_slice_v2"] = (
        "pass_25_of_25_budget_signature_registry_nonce_reproduction_path_and_authority_controls"
    )
    current["semantic_dispatch_slice_nodes"] = 7
    current["semantic_dispatch_full_reproduction_comparisons"] = 5
    current["semantic_dispatch_cryptographic_receipt_verifications"] = 2
    current["semantic_dispatch_execution_jobs"] = 34
    current["semantic_dispatch_closed_terminal_joins"] = 2
    current["semantic_dispatch_terminal_gap_count"] = 2
    current["synthetic_signed_receipts_verified"] = True
    current["real_signed_authority_dispatched"] = False
    current["semantic_dispatch_synthetic_fixture_persisted"] = False
    current["semantic_dispatch_empirical_graph_present"] = False
    current["end_to_end_semantic_dispatch_complete"] = False
    current["nonce_consumed"] = False
    current["output_reserved"] = False
    current["durable_transition_dispatched"] = False
    current["production_runtime_attachment_dispatched"] = False
    current["semantic_dispatch_allowed"] = False
    current["training_performed"] = False
    current["persistent_dictionary_output_created"] = False

    stage = result["stages"][0]
    stage["commands"].extend(
        [
            {
                "id": "verify_cluster_aware_sae_semantic_dispatch_slice_benchmark_v2",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_semantic_dispatch_slice_v2",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_semantic_dispatch_slice_benchmark_v2/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v50",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v50",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v50.json",
                ],
            },
        ]
    )
    stage["expected"]["cluster_aware_sae_semantic_dispatch_slice_v2"] = (
        "25 controls pass; seven representations are dispatched; five full "
        "reproductions and two Ed25519 receipt verifications preserve 34 jobs; "
        "two budget joins close and two runtime gaps remain; no authority transfers"
    )
    stage["stop_if"].extend(
        [
            "protocol V50 or semantic dispatch slice V2 benchmark differs",
            "the signed release omits an exact budget ID or file hash",
            "proposal or admission full reproduction is skipped",
            "registry status validity revocation or nonce checks are skipped",
            "synthetic signatures are treated as real authority",
            "a nonce output durable transition or production runtime is created",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "the synthetic materialization-to-budget-to-signed-release-to-runtime-"
        "admission slice is semantically verified; end-to-end dispatch remains "
        "blocked on real user-signed authority, durable nonce/output transition, "
        "and a separately approved production runtime attachment"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "a synthetic signature is presented as real user authorization",
        "a runtime admission is treated as durable nonce or output reservation",
        "production attachment verifier remains absent",
    ]
    completion = result["completion_definition"]
    completion["cluster_aware_sae_semantic_dispatch_slice_v2_verified"] = True
    completion["semantic_dispatch_slice_nodes"] = 7
    completion["semantic_dispatch_full_reproduction_comparisons"] = 5
    completion["semantic_dispatch_cryptographic_receipt_verifications"] = 2
    completion["semantic_dispatch_execution_jobs"] = 34
    completion["semantic_dispatch_closed_terminal_joins"] = 2
    completion["semantic_dispatch_terminal_gap_count"] = 2
    completion["synthetic_signed_receipts_verified"] = True
    completion["real_signed_authority_dispatched"] = False
    completion["semantic_dispatch_synthetic_fixture_persisted"] = False
    completion["semantic_dispatch_empirical_graph_present"] = False
    completion["end_to_end_semantic_dispatch_complete"] = False
    completion["nonce_consumed"] = False
    completion["output_reserved"] = False
    completion["durable_transition_dispatched"] = False
    completion["production_runtime_attachment_dispatched"] = False
    completion["semantic_dispatch_allowed"] = False
    completion["training_performed"] = False
    completion["persistent_dictionary_output_created"] = False
    boundary = result["scientific_boundary"]
    boundary["cluster_aware_sae_semantic_dispatch_slice_v2_verified"] = True
    boundary["semantic_dispatch_slice_nodes"] = 7
    boundary["semantic_dispatch_full_reproduction_comparisons"] = 5
    boundary["semantic_dispatch_cryptographic_receipt_verifications"] = 2
    boundary["semantic_dispatch_closed_terminal_joins"] = 2
    boundary["semantic_dispatch_terminal_gap_count"] = 2
    boundary["synthetic_signed_receipts_verified"] = True
    boundary["real_signed_authority_dispatched"] = False
    boundary["semantic_dispatch_synthetic_fixture_persisted"] = False
    boundary["semantic_dispatch_allowed"] = False
    boundary["end_to_end_semantic_dispatch_complete"] = False
    boundary["nonce_consumed"] = False
    boundary["output_reserved"] = False
    boundary["durable_transition_dispatched"] = False
    boundary["production_runtime_attachment_dispatched"] = False
    boundary["empirical_cross_artifact_graph_present"] = False
    boundary["semantic_verifiers_invoked_on_empirical_inputs"] = False
    boundary["production_campaign_executed"] = False
    boundary["production_trainer_callback_invoked"] = False
    boundary["training_performed"] = False
    boundary["persistent_dictionary_output_created"] = False
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
        raise ValueError("cheap-executor runbook V42 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v42",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "semantic_dispatch_slice_v2_verified": True,
        "dispatched_nodes": 7,
        "full_reproduction_comparisons": 5,
        "cryptographic_receipt_verifications": 2,
        "execution_jobs": 34,
        "closed_terminal_joins": 2,
        "terminal_gap_count": 2,
        "production_runtime_attachment_verifier_present": False,
        "empirical_graph_present": False,
        "semantic_dispatch_allowed": False,
        "training_performed": False,
        "persistent_dictionary_output_created": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root
        / "artifacts/vpd_cheap_executor_runbook_v41/runbook.json",
        "protocol_v50_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v50.json",
        "release_runtime_verifier_path": project_root
        / "experiments/authz_mechinterp/admit_cluster_aware_sae_training_v2.py",
        "dispatcher_path": project_root
        / "experiments/authz_mechinterp/dispatch_cluster_aware_sae_semantic_slice_v2.py",
        "benchmark_path": project_root
        / "artifacts/vpd_cluster_aware_sae_semantic_dispatch_slice_benchmark_v2/report.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V42.")
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
