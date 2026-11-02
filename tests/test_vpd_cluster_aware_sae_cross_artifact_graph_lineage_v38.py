from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v46 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v38 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v46.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v38/runbook.json"


def test_protocol_v46_reproduces_cross_artifact_graph_gate() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "02d046536a330c56da723080738277ba22c786bdb3737c6c30bde82aac6dd0a1"
    )
    assert verification["graph_contract_id"] == (
        "a01ac25024b52e64c9b23ad1e3f3203b17403a8164becac168bd1fda7a4d754e"
    )
    assert verification["graph_benchmark_id"] == (
        "178c9cd1124a4b93c36b1c40746a1776ea09ef820e56f8ad2aa7f0fab3c354fb"
    )
    assert verification["graph_controls"] == 34
    assert verification["required_nodes"] == 20
    assert verification["required_edges"] == 37
    assert verification["required_execution_identity_bindings"] == 10
    assert verification["temporary_synthetic_graph_persisted"] is False
    assert verification["empirical_graph_present"] is False
    assert verification["node_specific_semantic_verifiers_invoked"] is False
    assert verification["signed_authority_verified"] is False
    assert verification["durable_runtime_transition_verified"] is False
    assert verification["training_performed"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v46_preserves_structural_only_boundary() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    component = observed["component_compartments"][
        "cluster_aware_sae_cross_artifact_graph_v1"
    ]
    assert component["temporary_synthetic_graph_persisted"] is False
    assert component["empirical_graph_present"] is False
    assert component["node_specific_semantic_verifiers_invoked"] is False
    assert component["receipt_signatures_verified"] is False
    assert component["authority_status_verified"] is False
    assert component["nonce_freshness_verified"] is False
    assert component["output_reservation_exclusivity_verified"] is False
    assert component["training_performed"] is False
    assert component["execution_authorized"] is False
    predecessor = observed["protocol_derivation"]["predecessor_v45_verification"]
    assert predecessor["protocol_sha256"] == (
        "678684a7a898488e3a3c0521ffb8b1de5b6b51912cc6ea0891ec9d8856b18dea"
    )
    assert predecessor["deep_predecessor_replay_performed"] is False


def test_runbook_v38_reproduces_and_stops_before_empirical_graph() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "ab7b2c5658ef05d4ab9ea25b7f14f577c60b6c740a31d227fc45080a69ca5c33"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["cross_artifact_graph_contract_ready"] is True
    assert verification["required_nodes"] == 20
    assert verification["required_edges"] == 37
    assert verification["required_execution_identity_bindings"] == 10
    assert verification["temporary_synthetic_graph_verified"] is True
    assert verification["temporary_synthetic_graph_persisted"] is False
    assert verification["empirical_graph_present"] is False
    assert verification["node_specific_semantic_verifiers_invoked"] is False
    assert verification["signed_authority_verified"] is False
    assert verification["durable_runtime_transition_verified"] is False
    assert verification["training_performed"] is False
    assert verification["persistent_dictionary_output_created"] is False
    assert verification["sae_training_authorized"] is False
    assert verification["execution_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_cluster_aware_sae_cross_artifact_graph_contract_v1",
        "verify_cluster_aware_sae_cross_artifact_graph_benchmark_v1",
        "verify_research_protocol_v46",
    } <= command_ids
