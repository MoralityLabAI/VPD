from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_cross_artifact_graph_v1 import (
    verify_benchmark as verify_graph_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.verify_cluster_aware_sae_cross_artifact_graph_v1 import (
    canonical_inputs as graph_inputs,
    verify_graph_contract,
)


SCHEMA = "sch.vpd-activation-category-protocol.v46"
EXPECTED_V45_SHA256 = (
    "678684a7a898488e3a3c0521ffb8b1de5b6b51912cc6ea0891ec9d8856b18dea"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V46 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _verify_predecessor(path: Path, *, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V45_SHA256:
        raise ValueError("protocol V45 differs from its pinned SHA-256")
    observed = _load(path)
    if observed.get("schema") != "sch.vpd-activation-category-protocol.v45":
        raise ValueError("pinned predecessor is not protocol V45")
    if observed.get("scientific_boundary", {}).get("execution_authorized") is not False:
        raise ValueError("protocol V45 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-protocol-predecessor-verification.v1",
        "status": "pass",
        "protocol_sha256": EXPECTED_V45_SHA256,
        "schema_verified": True,
        "non_authorizing_boundary_verified": True,
        "deep_predecessor_replay_performed": False,
    }


def build_protocol(
    *,
    predecessor_path: Path,
    graph_verifier_path: Path,
    graph_contract_path: Path,
    graph_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(
        predecessor_path, project_root=project_root
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
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["cluster_aware_sae_cross_artifact_graph_v1"] = {
        "schema": "sch.vpd-cluster-aware-sae-cross-artifact-graph-gate.v1",
        "status": (
            "temporary_synthetic_graph_structurally_verified_"
            "empirical_graph_semantics_authority_training_and_execution_absent"
        ),
        "verifier": {
            "path": _relative(graph_verifier_path, project_root),
            "sha256": sha256_file(graph_verifier_path),
        },
        "contract": {
            "path": _relative(graph_contract_path, project_root),
            "sha256": sha256_file(graph_contract_path),
            "graph_contract_id": contract["graph_contract_id"],
            "verification": contract_verification,
        },
        "benchmark": {
            "path": _relative(graph_benchmark_path, project_root),
            "sha256": sha256_file(graph_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
        },
        "required_nodes": 20,
        "required_edges": 37,
        "required_execution_identity_bindings": 10,
        "temporary_synthetic_graph_persisted": False,
        "empirical_graph_present": False,
        "node_specific_semantic_verifiers_invoked": False,
        "receipt_signatures_verified": False,
        "authority_status_verified": False,
        "nonce_freshness_verified": False,
        "output_reservation_exclusivity_verified": False,
        "training_performed": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    lineage = result["unified_lineage"]
    lineage["cluster_aware_sae_cross_artifact_graph_contract_ready"] = True
    lineage["cross_artifact_graph_required_nodes"] = 20
    lineage["cross_artifact_graph_required_edges"] = 37
    lineage["cross_artifact_graph_required_execution_identity_bindings"] = 10
    lineage["temporary_synthetic_cross_artifact_graph_verified"] = True
    lineage["temporary_synthetic_cross_artifact_graph_persisted"] = False
    lineage["empirical_cross_artifact_graph_present"] = False
    lineage["cross_artifact_node_semantics_verified"] = False
    lineage["signed_authority_verified"] = False
    lineage["durable_runtime_transition_verified"] = False
    lineage["production_trainer_callback_invoked"] = False
    lineage["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v45_verification": predecessor_verification,
        "cross_artifact_graph_contract_verification": contract_verification,
        "cross_artifact_graph_benchmark_verification": benchmark_verification,
        "cross_artifact_graph_derivation_sha256": sha256_file(Path(__file__)),
        "pinned_predecessor_hash_schema_and_boundary_verified": True,
        "deep_predecessor_replay_performed": False,
    }
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
    boundary["weights_loaded_or_mutated"] = False
    boundary["sae_dictionary_training_authorized"] = False
    boundary["execution_authorized"] = False
    boundary["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    if observed != build_protocol(**kwargs):
        raise ValueError("research protocol V46 is not reproducible")
    component = observed["component_compartments"][
        "cluster_aware_sae_cross_artifact_graph_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v46",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "graph_contract_id": component["contract"]["graph_contract_id"],
        "graph_benchmark_id": component["benchmark"]["benchmark_id"],
        "graph_controls": component["benchmark"]["controls"]["controls"],
        "required_nodes": component["required_nodes"],
        "required_edges": component["required_edges"],
        "required_execution_identity_bindings": component[
            "required_execution_identity_bindings"
        ],
        "temporary_synthetic_graph_persisted": False,
        "empirical_graph_present": False,
        "node_specific_semantic_verifiers_invoked": False,
        "signed_authority_verified": False,
        "durable_runtime_transition_verified": False,
        "training_performed": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v45.json"
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
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V46.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.output is not None:
        result = build_protocol(**inputs)
        write_json(args.output, result)
        protocol_sha256 = sha256_file(args.output)
    else:
        result = verify_protocol(args.verify, **inputs)
        protocol_sha256 = result["protocol_sha256"]
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result.get("status", "derived"),
                "protocol_sha256": protocol_sha256,
                "execution_authorized": result.get("execution_authorized", False),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
