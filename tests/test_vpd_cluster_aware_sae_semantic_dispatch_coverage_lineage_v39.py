from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v47 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v39 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v47.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v39/runbook.json"


def test_protocol_v47_reproduces_semantic_dispatch_gap_audit() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "a7bd3d23ad2ecce3b5933dae59b70a7de72212fec103228a149931048b6dc15c"
    )
    assert verification["semantic_dispatch_coverage_audit_id"] == (
        "bd3cc043f7b658029fd20c99492c43edbace23550a7e237d36d692e46297eaa7"
    )
    assert verification["semantic_dispatch_coverage_benchmark_id"] == (
        "adc1d00bf3be2ab9d790252534bc084add403842b26ae9002e2d6216f0095732"
    )
    assert verification["controls"] == 23
    assert verification["schema_and_id_matching_nodes"] == 12
    assert verification["dispatch_ready_nodes_under_v1_graph"] == 4
    assert verification["exact_implemented_field_edges"] == 9
    assert verification["mismatched_edges"] == 28
    assert verification["semantic_dispatch_allowed"] is False
    assert verification["empirical_graph_present"] is False
    assert verification["training_performed"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v47_preserves_v46_as_structural_gap_evidence() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    component = observed["component_compartments"][
        "cluster_aware_sae_semantic_dispatch_coverage_v1"
    ]
    assert component["v1_structural_graph_preserved"] is True
    assert component["v1_semantic_graph_verified"] is False
    assert component["production_runtime_attachment_verifier_present"] is False
    assert component["semantic_dispatch_allowed"] is False
    assert component["empirical_graph_present"] is False
    predecessor = observed["protocol_derivation"]["predecessor_v46_verification"]
    assert predecessor["protocol_sha256"] == (
        "02d046536a330c56da723080738277ba22c786bdb3737c6c30bde82aac6dd0a1"
    )
    assert predecessor["deep_predecessor_replay_performed"] is False


def test_runbook_v39_reproduces_and_requires_corrected_successor() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "0533d2b242ccb8a566a8dee69146728214584abed4ff6f46a4251529d41fb68c"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["semantic_dispatch_coverage_audited"] is True
    assert verification["schema_and_id_matching_nodes"] == 12
    assert verification["dispatch_ready_nodes_under_v1_graph"] == 4
    assert verification["exact_implemented_field_edges"] == 9
    assert verification["mismatched_edges"] == 28
    assert verification["corrected_semantic_graph_successor_required"] is True
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
        "verify_cluster_aware_sae_semantic_dispatch_coverage_audit_v1",
        "verify_cluster_aware_sae_semantic_dispatch_coverage_benchmark_v1",
        "verify_research_protocol_v47",
    } <= command_ids
