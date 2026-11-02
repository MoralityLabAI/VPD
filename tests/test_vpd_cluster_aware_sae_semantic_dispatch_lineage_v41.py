from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v49 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v41 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v49.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v41/runbook.json"
PROTOCOL_SHA256 = "0c3798bdca4dd501e8f674afdfb368319ebb6010a4e8ea4f82aff92afec48fbb"
RUNBOOK_ID = "733555f2967840041f005380985305b61b265c8c15615543284258646f2019ad"
BENCHMARK_ID = "60139e5dd07adc7bd79b5028bd99d068bbb32a81ec92afe1e2c77091325f04d3"
SLICE_ID = "0efe29534a177e3e4bc242dc091ebcd79c867dfff8c0a80dfe6f153498e68449"


def test_protocol_v49_reproduces_semantic_dispatch_slice() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == PROTOCOL_SHA256
    assert verification["semantic_dispatch_benchmark_id"] == BENCHMARK_ID
    assert verification["semantic_dispatch_slice_id"] == SLICE_ID
    assert verification["controls"] == 22
    assert verification["dispatched_nodes"] == 4
    assert verification["full_reproduction_comparisons"] == 4
    assert verification["execution_jobs"] == 34
    assert verification["terminal_gap_count"] == 3
    assert verification["semantic_dispatch_allowed"] is False
    assert verification["empirical_graph_present"] is False
    assert verification["training_performed"] is False
    assert verification["sae_training_authorized"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v49_preserves_v48_and_nonauthorizing_slice_boundary() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    component = observed["component_compartments"][
        "cluster_aware_sae_semantic_dispatch_slice_v1"
    ]
    assert component["dispatched_nodes"] == 4
    assert component["full_reproduction_comparisons"] == 4
    assert component["synthetic_fixture_persisted"] is False
    assert component["empirical_graph_present"] is False
    assert component["end_to_end_semantic_dispatch_complete"] is False
    assert component["signed_authority_dispatched"] is False
    assert component["durable_transition_dispatched"] is False
    assert component["production_runtime_attachment_dispatched"] is False
    predecessor = observed["protocol_derivation"]["predecessor_v48_verification"]
    assert predecessor["protocol_sha256"] == (
        "bc526292ca5e84a65124129215239aa96d568331716e7abf5ec2ccbd49e536d4"
    )
    assert predecessor["deep_predecessor_replay_performed"] is False


def test_runbook_v41_reproduces_and_stops_at_terminal_joins() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == RUNBOOK_ID
    assert verification["safe_stage_now"] == 0
    assert verification["semantic_dispatch_slice_v1_verified"] is True
    assert verification["dispatched_nodes"] == 4
    assert verification["full_reproduction_comparisons"] == 4
    assert verification["execution_jobs"] == 34
    assert verification["terminal_gap_count"] == 3
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
        "verify_cluster_aware_sae_semantic_dispatch_slice_benchmark_v1",
        "verify_research_protocol_v49",
    } <= command_ids
