from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_semantic_graph_contract_v2 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.build_cluster_aware_sae_semantic_graph_contract_v2 import (
    canonical_inputs as contract_inputs,
    verify_contract,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v48 import (
    canonical_inputs as protocol_v48_inputs,
    verify_protocol as verify_protocol_v48,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v40"
EXPECTED_V39_RUNBOOK_ID = (
    "0533d2b242ccb8a566a8dee69146728214584abed4ff6f46a4251529d41fb68c"
)
EXPECTED_V48_SHA256 = (
    "bc526292ca5e84a65124129215239aa96d568331716e7abf5ec2ccbd49e536d4"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V40 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V39_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V39 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v39":
        raise ValueError("pinned predecessor is not runbook V39")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V39 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v48_path: Path,
    contract_builder_path: Path,
    contract_path: Path,
    benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    if sha256_file(protocol_v48_path) != EXPECTED_V48_SHA256:
        raise ValueError("protocol V48 differs from its pinned SHA-256")
    protocol_verification = verify_protocol_v48(
        protocol_v48_path, **protocol_v48_inputs(project_root)
    )
    contract_verification = verify_contract(
        contract_path, **contract_inputs(project_root)
    )
    benchmark_verification = verify_benchmark(
        benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    contract = _load(contract_path)
    benchmark = _load(benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "corrected_semantic_graph_v2_contract_verified_three_terminal_joins_"
        "empirical_dispatch_training_and_execution_absent"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v48"] = {
        "path": _relative(protocol_v48_path, project_root),
        "sha256": sha256_file(protocol_v48_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_aware_sae_semantic_graph_contract_builder_v2"] = {
        "path": _relative(contract_builder_path, project_root),
        "sha256": sha256_file(contract_builder_path),
    }
    result["source"]["cluster_aware_sae_semantic_graph_contract_v2"] = {
        "path": _relative(contract_path, project_root),
        "sha256": sha256_file(contract_path),
        "semantic_graph_contract_id": contract["semantic_graph_contract_id"],
        "verification": contract_verification,
    }
    result["source"]["cluster_aware_sae_semantic_graph_contract_benchmark_v2"] = {
        "path": _relative(benchmark_path, project_root),
        "sha256": sha256_file(benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    current = result["current_result"]
    current["cluster_aware_sae_semantic_graph_contract_v2"] = (
        "pass_36_of_36_corrected_interface_content_address_gap_and_authority_controls"
    )
    current["v1_cross_artifact_structural_graph_preserved"] = True
    current["v1_cross_artifact_semantic_graph_verified"] = False
    current["semantic_graph_v2_logical_nodes"] = 21
    current["semantic_graph_v2_implemented_node_representations"] = 20
    current["semantic_graph_v2_semantic_joins"] = 38
    current["semantic_graph_v2_terminal_gap_nodes"] = 1
    current["semantic_graph_v2_terminal_gap_joins"] = 3
    current["semantic_graph_v1_mismatches_accounted"] = 28
    current["blueprint_bound_budget_v2_verified"] = True
    current["budget_admission_to_signed_release_join_present"] = False
    current["budget_admission_to_runtime_admission_join_present"] = False
    current["production_runtime_attachment_verifier_present"] = False
    current["empirical_cross_artifact_graph_present"] = False
    current["semantic_dispatch_allowed"] = False
    current["training_performed"] = False
    current["persistent_dictionary_output_created"] = False

    stage = result["stages"][0]
    stage["commands"].extend(
        [
            {
                "id": "verify_cluster_aware_sae_semantic_graph_contract_v2",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.build_cluster_aware_sae_semantic_graph_contract_v2",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_semantic_graph_contract_v2/contract.json",
                ],
            },
            {
                "id": "verify_cluster_aware_sae_semantic_graph_contract_benchmark_v2",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_semantic_graph_contract_v2",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_semantic_graph_contract_benchmark_v2/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v48",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v48",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v48.json",
                ],
            },
        ]
    )
    stage["expected"]["cluster_aware_sae_semantic_graph_contract_v2"] = (
        "36 controls pass; all 28 V1 mismatches accounted; 21 logical nodes and "
        "38 semantic joins declared; three terminal joins remain; dispatch stays disabled"
    )
    stage["stop_if"].extend(
        [
            "protocol V48, semantic graph V2 contract, or benchmark differs",
            "capture preflight canonical digest is treated as a file hash",
            "global pair split is treated as an independent persisted node",
            "budget proposal or admission loses exact blueprint ID or file-hash binding",
            "budget admission is presented as bound into signed release or runtime admission V1",
            "synthetic inert runtime attachment is presented as production attachment",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "corrected semantic interface contract V2 is implemented and benchmarked; "
        "semantic dispatch remains withheld until a fail-closed dispatcher invokes "
        "the real compound verifiers and the three terminal joins have explicit successors"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "a dispatcher compares only schema or ID instead of reproducing full artifacts",
        "required companion inputs are omitted or substituted",
        "a terminal join is silently treated as implemented",
        "synthetic evidence is presented as empirical",
    ]
    completion = result["completion_definition"]
    completion["cluster_aware_sae_semantic_graph_contract_v2_verified"] = True
    completion["semantic_graph_v2_logical_nodes"] = 21
    completion["semantic_graph_v2_implemented_node_representations"] = 20
    completion["semantic_graph_v2_semantic_joins"] = 38
    completion["semantic_graph_v2_terminal_gap_joins"] = 3
    completion["semantic_graph_v1_mismatches_accounted"] = 28
    completion["blueprint_bound_budget_v2_verified"] = True
    completion["budget_admission_to_signed_release_join_present"] = False
    completion["budget_admission_to_runtime_admission_join_present"] = False
    completion["production_runtime_attachment_verifier_present"] = False
    completion["empirical_cross_artifact_graph_present"] = False
    completion["semantic_dispatch_allowed"] = False
    completion["training_performed"] = False
    completion["persistent_dictionary_output_created"] = False
    boundary = result["scientific_boundary"]
    boundary["cluster_aware_sae_semantic_graph_contract_v2_verified"] = True
    boundary["v1_cross_artifact_structural_graph_preserved"] = True
    boundary["v1_cross_artifact_semantic_graph_verified"] = False
    boundary["semantic_graph_v2_contract_only"] = True
    boundary["semantic_dispatch_allowed"] = False
    boundary["budget_admission_to_signed_release_join_present"] = False
    boundary["budget_admission_to_runtime_admission_join_present"] = False
    boundary["production_runtime_attachment_verifier_present"] = False
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
        raise ValueError("cheap-executor runbook V40 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v40",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "semantic_graph_contract_v2_verified": True,
        "logical_nodes": 21,
        "implemented_node_representations": 20,
        "semantic_joins": 38,
        "terminal_gap_joins": 3,
        "v1_mismatches_accounted": 28,
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
        / "artifacts/vpd_cheap_executor_runbook_v39/runbook.json",
        "protocol_v48_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v48.json",
        "contract_builder_path": project_root
        / "experiments/authz_mechinterp/build_cluster_aware_sae_semantic_graph_contract_v2.py",
        "contract_path": project_root
        / "artifacts/vpd_cluster_aware_sae_semantic_graph_contract_v2/contract.json",
        "benchmark_path": project_root
        / "artifacts/vpd_cluster_aware_sae_semantic_graph_contract_benchmark_v2/report.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V40.")
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
