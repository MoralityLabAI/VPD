from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_semantic_graph_contract_v2 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.build_cluster_aware_sae_semantic_graph_contract_v2 import (
    canonical_inputs,
    verify_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "artifacts/vpd_cluster_aware_sae_semantic_graph_contract_v2/contract.json"
)
REPORT = (
    ROOT
    / "artifacts/vpd_cluster_aware_sae_semantic_graph_contract_benchmark_v2/report.json"
)
CONTRACT_ID = "6db000c29e0619d6bec26fb85f8d1739bd5fae16e46cccc03516a2171d79adb0"
BENCHMARK_ID = "ade0e1f19765c8dc1accf66ee86590c21b4a904a499094441f21edb8fa40fd37"


def test_corrected_semantic_graph_contract_reproduces() -> None:
    verification = verify_contract(CONTRACT, **canonical_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["semantic_graph_contract_id"] == CONTRACT_ID
    assert verification["logical_nodes"] == 21
    assert verification["implemented_node_representations"] == 20
    assert verification["terminal_gap_nodes"] == 1
    assert verification["semantic_joins"] == 38
    assert verification["terminal_gap_joins"] == 3
    assert verification["v1_mismatches_accounted"] == 28
    assert verification["blueprint_budget_v2_controls_passed"] == 27
    assert verification["end_to_end_semantic_dispatch_allowed"] is False
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False


def test_corrected_semantic_graph_contract_benchmark_reproduces() -> None:
    verification = verify_benchmark(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == BENCHMARK_ID
    assert verification["semantic_graph_contract_id"] == CONTRACT_ID
    assert verification["controls"] == 36
    assert verification["logical_nodes"] == 21
    assert verification["semantic_joins"] == 38
    assert verification["terminal_gap_joins"] == 3
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False


def test_corrected_representations_and_remaining_gaps_are_explicit() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    nodes = {row["node"]: row for row in contract["node_specs"]}
    joins = {row["join"]: row for row in contract["join_specs"]}
    assert nodes["capture_authorization_receipt"]["id_field"] == (
        "authorization_receipt_id"
    )
    assert nodes["guarded_v3_capture_receipt"]["id_field"] == (
        "capture_execution_v3_id"
    )
    assert nodes["capture_execution_v2_receipt"]["id_field"] == (
        "capture_execution_id"
    )
    assert joins["v3_capture_to_predecessor_v2"]["target_node"] == (
        "capture_execution_v2_receipt"
    )
    assert nodes["routing_evidence_bundle"]["id_field"] == (
        "routing_evidence_bundle_id"
    )
    assert nodes["cluster_aware_sae_plan_with_embedded_split"][
        "required_companions"
    ] == ["embedded_global_pair_split"]
    assert nodes["blueprint_bound_budget_proposal_v2"]["schema"].endswith(
        "blueprint-bound-budget-proposal.v2"
    )
    assert nodes["blueprint_bound_budget_admission_v2"]["schema"].endswith(
        "blueprint-bound-budget-admission.v2"
    )
    assert nodes["compute_authorization_receipt"]["id_field"] == "receipt_id"
    assert nodes["durable_job_transition"]["id_field"] == (
        "transition_reservation_id"
    )
    assert nodes["production_runtime_attachment"]["implemented"] is False
    assert joins["budget_admission_to_release_receipt"]["join_kind"] == (
        "terminal_missing_signed_join"
    )
    assert joins["budget_admission_to_runtime_admission"]["join_kind"] == (
        "terminal_missing_runtime_join"
    )
    assert joins["transition_to_production_attachment"]["join_kind"] == (
        "terminal_prerequisite_gap"
    )
    assert contract["scientific_boundary"]["semantic_dispatch_performed"] is False
    assert contract["scientific_boundary"]["empirical_graph_present"] is False
    assert contract["scientific_boundary"]["execution_authorized"] is False
    assert contract["scientific_boundary"]["sae_training_authorized"] is False
