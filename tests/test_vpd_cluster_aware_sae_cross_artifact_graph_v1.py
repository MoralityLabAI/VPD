from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_cross_artifact_graph_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.verify_cluster_aware_sae_cross_artifact_graph_v1 import (
    canonical_inputs,
    verify_graph_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "artifacts/vpd_cluster_aware_sae_cross_artifact_graph_contract_v1/contract.json"
)
REPORT = (
    ROOT
    / "artifacts/vpd_cluster_aware_sae_cross_artifact_graph_benchmark_v1/report.json"
)
CONTRACT_ID = "a01ac25024b52e64c9b23ad1e3f3203b17403a8164becac168bd1fda7a4d754e"
BENCHMARK_ID = "178c9cd1124a4b93c36b1c40746a1776ea09ef820e56f8ad2aa7f0fab3c354fb"


def test_cross_artifact_graph_contract_reproduces() -> None:
    verification = verify_graph_contract(CONTRACT, **canonical_inputs(ROOT))
    assert verification == {
        "schema": "sch.vpd-cluster-aware-sae-cross-artifact-graph-contract-verification.v1",
        "status": "pass",
        "graph_contract_id": CONTRACT_ID,
        "required_nodes": 20,
        "required_edges": 37,
        "execution_identity_bindings": 10,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def test_cross_artifact_graph_benchmark_reproduces() -> None:
    verification = verify_benchmark(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == BENCHMARK_ID
    assert verification["graph_contract_id"] == CONTRACT_ID
    assert verification["controls"] == 34
    assert verification["required_nodes"] == 20
    assert verification["required_edges"] == 37
    assert verification["required_execution_identity_bindings"] == 10
    assert verification["synthetic_graph_persisted"] is False
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False


def test_cross_artifact_graph_tamper_and_boundary_controls() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    controls = observed["controls"]
    for control in (
        "clean_synthetic_graph_structurally_verifies",
        "missing_node_rejected",
        "duplicate_node_rejected",
        "unknown_node_rejected",
        "wrong_node_file_hash_rejected",
        "wrong_descriptor_content_id_rejected",
        "readdressed_schema_tamper_rejected",
        "unaddressed_node_content_tamper_rejected",
        "readdressed_cross_artifact_edge_tamper_rejected",
        "readdressed_budget_policy_substitution_rejected",
        "preflight_execution_binding_mismatch_rejected",
        "missing_execution_binding_rejected",
        "absolute_node_path_rejected",
        "node_path_traversal_rejected",
        "graph_root_escape_rejected",
        "missing_node_file_rejected",
        "symbolic_link_or_junction_node_rejected",
        "mixed_synthetic_empirical_nodes_rejected",
        "graph_kind_node_boundary_mismatch_rejected",
        "temporary_synthetic_graph_removed",
        "benchmark_imports_no_model_frameworks",
    ):
        assert controls[control] is True
    boundary = observed["scientific_boundary"]
    assert boundary["temporary_synthetic_fixture_only"] is True
    assert boundary["empirical_graph_present"] is False
    assert boundary["node_specific_semantic_verifiers_invoked"] is False
    assert boundary["receipt_signatures_verified"] is False
    assert boundary["authority_status_verified"] is False
    assert boundary["model_loaded"] is False
    assert boundary["model_inference_performed"] is False
    assert boundary["training_performed"] is False
    assert boundary["weights_loaded_or_mutated"] is False
    assert boundary["persistent_dictionary_output_created"] is False
    assert boundary["execution_authorized"] is False
    assert boundary["sae_training_authorized"] is False
