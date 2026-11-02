from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v50 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v42 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v50.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v42/runbook.json"
PROTOCOL_SHA256 = "55dc12c6f948c6fc8ecaeb1d9d06d037fbc5a72ab2fd0a5990b453bd7271a951"
RUNBOOK_ID = "917c9582a1e4d71015eaef89507b335ad54bbebd5c2930482c1aab81d6842589"
BENCHMARK_ID = "681abfc57a4103c72cd7aa7567f23ead61a3f3d9569155c14709902053c26451"
SLICE_ID = "891fbecf114deb595d984e252979301aae7a8fd5c92c305651b12178e854402b"


def test_protocol_v50_reproduces_budget_bound_semantic_dispatch_slice() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == PROTOCOL_SHA256
    assert verification["semantic_dispatch_benchmark_id"] == BENCHMARK_ID
    assert verification["semantic_dispatch_slice_id"] == SLICE_ID
    assert verification["controls"] == 25
    assert verification["dispatched_nodes"] == 7
    assert verification["full_reproduction_comparisons"] == 5
    assert verification["cryptographic_receipt_verifications"] == 2
    assert verification["execution_jobs"] == 34
    assert verification["closed_terminal_joins"] == 2
    assert verification["terminal_gap_count"] == 2
    assert verification["semantic_dispatch_allowed"] is False
    assert verification["empirical_graph_present"] is False
    assert verification["training_performed"] is False
    assert verification["sae_training_authorized"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v50_preserves_v49_and_synthetic_authority_boundary() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    component = observed["component_compartments"][
        "cluster_aware_sae_semantic_dispatch_slice_v2"
    ]
    assert component["dispatched_nodes"] == 7
    assert component["full_reproduction_comparisons"] == 5
    assert component["cryptographic_receipt_verifications"] == 2
    assert component["closed_terminal_joins"] == [
        "budget_admission_to_signed_release",
        "budget_admission_to_runtime_admission",
    ]
    assert component["remaining_terminal_gaps"] == [
        "runtime_admission_to_durable_transition",
        "production_runtime_attachment",
    ]
    assert component["synthetic_fixture_persisted"] is False
    assert component["synthetic_signed_receipts_verified"] is True
    assert component["real_signed_authority_dispatched"] is False
    assert component["end_to_end_semantic_dispatch_complete"] is False
    assert component["nonce_consumed"] is False
    assert component["output_reserved"] is False
    predecessor = observed["protocol_derivation"]["predecessor_v49_verification"]
    assert predecessor["protocol_sha256"] == (
        "0c3798bdca4dd501e8f674afdfb368319ebb6010a4e8ea4f82aff92afec48fbb"
    )
    assert predecessor["deep_predecessor_replay_performed"] is False


def test_runbook_v42_reproduces_and_stops_before_transition() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == RUNBOOK_ID
    assert verification["safe_stage_now"] == 0
    assert verification["semantic_dispatch_slice_v2_verified"] is True
    assert verification["dispatched_nodes"] == 7
    assert verification["full_reproduction_comparisons"] == 5
    assert verification["cryptographic_receipt_verifications"] == 2
    assert verification["execution_jobs"] == 34
    assert verification["closed_terminal_joins"] == 2
    assert verification["terminal_gap_count"] == 2
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
        "verify_cluster_aware_sae_semantic_dispatch_slice_benchmark_v2",
        "verify_research_protocol_v50",
    } <= command_ids
