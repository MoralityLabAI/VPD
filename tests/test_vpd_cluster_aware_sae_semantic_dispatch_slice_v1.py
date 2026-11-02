from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_semantic_dispatch_slice_v1 import (
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "artifacts/vpd_cluster_aware_sae_semantic_dispatch_slice_benchmark_v1/report.json"
)
BENCHMARK_ID = "60139e5dd07adc7bd79b5028bd99d068bbb32a81ec92afe1e2c77091325f04d3"
SLICE_ID = "0efe29534a177e3e4bc242dc091ebcd79c867dfff8c0a80dfe6f153498e68449"


def test_semantic_dispatch_slice_benchmark_reproduces() -> None:
    verification = verify_benchmark(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == BENCHMARK_ID
    assert verification["semantic_dispatch_slice_id"] == SLICE_ID
    assert verification["controls"] == 22
    assert verification["dispatched_nodes"] == 4
    assert verification["full_reproduction_comparisons"] == 4
    assert verification["execution_jobs"] == 34
    assert verification["terminal_gap_count"] == 3
    assert verification["synthetic_fixture_persisted"] is False
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False


def test_semantic_dispatch_slice_attacks_and_boundaries_remain_explicit() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert observed["benchmark_id"] == BENCHMARK_ID
    assert observed["results"]["semantic_dispatch_slice_id"] == SLICE_ID
    assert set(observed["results"]["dispatched_node_ids"]) == {
        "materialization",
        "execution_blueprint",
        "blueprint_bound_budget_proposal_v2",
        "blueprint_bound_budget_admission_v2",
    }
    for control in (
        "four_real_semantic_verifiers_dispatch",
        "all_four_nodes_compare_full_reproduction",
        "path_traversal_rejected",
        "absolute_path_rejected",
        "symbolic_link_or_junction_rejected",
        "readdressed_predecessor_companion_substitution_rejected",
        "readdressed_materialization_mutation_rejected_by_reproduction",
        "readdressed_blueprint_mutation_rejected_by_reproduction",
        "blueprint_byte_substitution_rejected_by_exact_hash_binding",
        "readdressed_proposal_authority_broadening_rejected",
        "readdressed_admission_mutation_rejected_by_reproduction",
        "readdressed_policy_substitution_rejected",
        "three_terminal_gaps_are_returned",
        "slice_transfers_no_authority",
        "benchmark_imports_no_model_frameworks",
        "temporary_fixture_root_removed",
    ):
        assert observed["controls"][control] is True
    boundary = observed["scientific_boundary"]
    assert boundary["semantic_slice_only"] is True
    assert boundary["end_to_end_semantic_dispatch_complete"] is False
    assert boundary["empirical_graph_present"] is False
    assert boundary["signed_authority_dispatched"] is False
    assert boundary["durable_transition_dispatched"] is False
    assert boundary["production_runtime_attachment_dispatched"] is False
    assert boundary["model_loaded"] is False
    assert boundary["model_inference_performed"] is False
    assert boundary["training_performed"] is False
    assert boundary["execution_authorized"] is False
    assert boundary["sae_training_authorized"] is False
