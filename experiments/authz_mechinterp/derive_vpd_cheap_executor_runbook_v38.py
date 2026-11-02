from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_cross_artifact_graph_v1 import (
    verify_benchmark as verify_graph_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v46 import (
    canonical_inputs as protocol_v46_inputs,
    verify_protocol as verify_protocol_v46,
)
from experiments.authz_mechinterp.verify_cluster_aware_sae_cross_artifact_graph_v1 import (
    canonical_inputs as graph_inputs,
    verify_graph_contract,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v38"
EXPECTED_V37_RUNBOOK_ID = (
    "0d5d1dfde0559e4040b40c12d655c95a9f7cf53f102c2350a87b0370ddd7fac2"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V38 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V37_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V37 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v37":
        raise ValueError("pinned predecessor is not runbook V37")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V37 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v46_path: Path,
    graph_verifier_path: Path,
    graph_contract_path: Path,
    graph_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v46(
        protocol_v46_path, **protocol_v46_inputs(project_root)
    )
    contract_verification = verify_graph_contract(
        graph_contract_path, **graph_inputs(project_root)
    )
    benchmark_verification = verify_graph_benchmark(
        graph_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    contract = _load(graph_contract_path)
    benchmark = _load(graph_benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "cross_artifact_graph_structure_ready_empirical_graph_semantics_"
        "authority_training_and_execution_absent"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v46"] = {
        "path": _relative(protocol_v46_path, project_root),
        "sha256": sha256_file(protocol_v46_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_aware_sae_cross_artifact_graph_verifier_v1"] = {
        "path": _relative(graph_verifier_path, project_root),
        "sha256": sha256_file(graph_verifier_path),
    }
    result["source"]["cluster_aware_sae_cross_artifact_graph_contract_v1"] = {
        "path": _relative(graph_contract_path, project_root),
        "sha256": sha256_file(graph_contract_path),
        "graph_contract_id": contract["graph_contract_id"],
        "verification": contract_verification,
    }
    result["source"]["cluster_aware_sae_cross_artifact_graph_benchmark_v1"] = {
        "path": _relative(graph_benchmark_path, project_root),
        "sha256": sha256_file(graph_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    current = result["current_result"]
    current["cluster_aware_sae_cross_artifact_graph_v1"] = (
        "pass_34_of_34_exact_node_edge_hash_address_path_reparse_"
        "binding_boundary_and_cleanup_controls"
    )
    current["cross_artifact_graph_required_nodes"] = 20
    current["cross_artifact_graph_required_edges"] = 37
    current["cross_artifact_graph_required_execution_identity_bindings"] = 10
    current["temporary_synthetic_cross_artifact_graph_verified"] = True
    current["temporary_synthetic_cross_artifact_graph_persisted"] = False
    current["empirical_cross_artifact_graph_present"] = False
    current["node_specific_semantic_verifiers_invoked"] = False
    current["signed_authority_verified"] = False
    current["durable_runtime_transition_verified"] = False
    current["training_performed"] = False
    current["persistent_dictionary_output_created"] = False

    stage = result["stages"][0]
    stage["commands"].extend(
        [
            {
                "id": "verify_cluster_aware_sae_cross_artifact_graph_contract_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.verify_cluster_aware_sae_cross_artifact_graph_v1",
                    "--verify-contract",
                    "artifacts/vpd_cluster_aware_sae_cross_artifact_graph_contract_v1/contract.json",
                ],
            },
            {
                "id": "verify_cluster_aware_sae_cross_artifact_graph_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_cross_artifact_graph_v1",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_cross_artifact_graph_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v46",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v46",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v46.json",
                ],
            },
        ]
    )
    stage["expected"]["cluster_aware_sae_cross_artifact_graph_v1"] = (
        "34 controls pass; temporary synthetic fixture covers exactly 20 nodes, "
        "37 edges, and 10 identity bindings; cleanup passes; no authority transfers"
    )
    stage["stop_if"].extend(
        [
            "protocol V46, graph contract, or graph benchmark differs",
            "any node path crosses a traversal symlink junction or graph-root boundary",
            "a file hash content address edge or execution identity binding differs",
            "structural graph integrity is described as semantic signed or execution authority",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "nonexecuting exact 20-node 37-edge graph structure passes synthetic controls; "
        "empirical graph, node semantics, signed authority, durable transition, training, "
        "and execution remain absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "a temporary synthetic graph is persisted or presented as empirical",
        "a structurally addressed node bypasses its node-specific semantic verifier",
        "graph integrity is used to infer signature authority nonce freshness or output exclusivity",
    ]
    completion = result["completion_definition"]
    completion["cluster_aware_sae_cross_artifact_graph_contract_ready"] = True
    completion["cross_artifact_graph_required_nodes"] = 20
    completion["cross_artifact_graph_required_edges"] = 37
    completion["cross_artifact_graph_required_execution_identity_bindings"] = 10
    completion["temporary_synthetic_cross_artifact_graph_verified"] = True
    completion["temporary_synthetic_cross_artifact_graph_persisted"] = False
    completion["empirical_cross_artifact_graph_present"] = False
    completion["node_specific_semantic_verifiers_invoked"] = False
    completion["signed_authority_verified"] = False
    completion["durable_runtime_transition_verified"] = False
    completion["training_performed"] = False
    completion["persistent_dictionary_output_created"] = False
    boundary = result["scientific_boundary"]
    boundary["cluster_aware_sae_cross_artifact_graph_contract_ready"] = True
    boundary["temporary_synthetic_cross_artifact_graph_verified"] = True
    boundary["temporary_synthetic_cross_artifact_graph_persisted"] = False
    boundary["empirical_cross_artifact_graph_present"] = False
    boundary["node_specific_semantic_verifiers_invoked"] = False
    boundary["receipt_signatures_verified"] = False
    boundary["authority_status_verified"] = False
    boundary["nonce_freshness_verified"] = False
    boundary["output_reservation_exclusivity_verified"] = False
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
        raise ValueError("cheap-executor runbook V38 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v38",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "cross_artifact_graph_contract_ready": True,
        "required_nodes": 20,
        "required_edges": 37,
        "required_execution_identity_bindings": 10,
        "temporary_synthetic_graph_verified": True,
        "temporary_synthetic_graph_persisted": False,
        "empirical_graph_present": False,
        "node_specific_semantic_verifiers_invoked": False,
        "signed_authority_verified": False,
        "durable_runtime_transition_verified": False,
        "training_performed": False,
        "persistent_dictionary_output_created": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v37/runbook.json"
        ),
        "protocol_v46_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v46.json"
        ),
        "graph_verifier_path": (
            project_root
            / "experiments/authz_mechinterp/verify_cluster_aware_sae_cross_artifact_graph_v1.py"
        ),
        "graph_contract_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_cross_artifact_graph_contract_v1/contract.json"
        ),
        "graph_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_cross_artifact_graph_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V38.")
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
