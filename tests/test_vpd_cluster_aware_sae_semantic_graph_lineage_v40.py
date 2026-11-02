from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v48 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v40 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v48.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v40/runbook.json"
PROTOCOL_SHA256 = "bc526292ca5e84a65124129215239aa96d568331716e7abf5ec2ccbd49e536d4"
RUNBOOK_ID = "08444884365069ea91a362f0fc4be7ca4f7552947780c6a016f739553c0c4681"
CONTRACT_ID = "6db000c29e0619d6bec26fb85f8d1739bd5fae16e46cccc03516a2171d79adb0"
BENCHMARK_ID = "ade0e1f19765c8dc1accf66ee86590c21b4a904a499094441f21edb8fa40fd37"


def test_protocol_v48_reproduces_corrected_semantic_graph_contract() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == PROTOCOL_SHA256
    assert verification["semantic_graph_contract_id"] == CONTRACT_ID
    assert verification["semantic_graph_benchmark_id"] == BENCHMARK_ID
    assert verification["controls"] == 36
    assert verification["logical_nodes"] == 21
    assert verification["implemented_node_representations"] == 20
    assert verification["semantic_joins"] == 38
    assert verification["terminal_gap_joins"] == 3
    assert verification["semantic_dispatch_allowed"] is False
    assert verification["empirical_graph_present"] is False
    assert verification["training_performed"] is False
    assert verification["sae_training_authorized"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v48_preserves_v47_and_terminal_gaps() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    component = observed["component_compartments"][
        "cluster_aware_sae_semantic_graph_contract_v2"
    ]
    assert component["v1_structural_graph_preserved"] is True
    assert component["corrected_v2_contract_verified"] is True
    assert component["v1_mismatches_accounted"] == 28
    assert component["terminal_gap_joins"] == 3
    assert component["production_runtime_attachment_verifier_present"] is False
    assert component["end_to_end_semantic_dispatch_allowed"] is False
    predecessor = observed["protocol_derivation"]["predecessor_v47_verification"]
    assert predecessor["protocol_sha256"] == (
        "a7bd3d23ad2ecce3b5933dae59b70a7de72212fec103228a149931048b6dc15c"
    )
    assert predecessor["deep_predecessor_replay_performed"] is False


def test_runbook_v40_reproduces_and_stops_at_three_terminal_joins() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == RUNBOOK_ID
    assert verification["safe_stage_now"] == 0
    assert verification["semantic_graph_contract_v2_verified"] is True
    assert verification["logical_nodes"] == 21
    assert verification["implemented_node_representations"] == 20
    assert verification["semantic_joins"] == 38
    assert verification["terminal_gap_joins"] == 3
    assert verification["v1_mismatches_accounted"] == 28
    assert verification["production_runtime_attachment_verifier_present"] is False
    assert verification["empirical_graph_present"] is False
    assert verification["semantic_dispatch_allowed"] is False
    assert verification["training_performed"] is False
    assert verification["persistent_dictionary_output_created"] is False
    assert verification["sae_training_authorized"] is False
    assert verification["execution_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_cluster_aware_sae_semantic_graph_contract_v2",
        "verify_cluster_aware_sae_semantic_graph_contract_benchmark_v2",
        "verify_research_protocol_v48",
    } <= command_ids
