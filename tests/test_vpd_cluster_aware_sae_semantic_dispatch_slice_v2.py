from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_semantic_dispatch_slice_v2 import (
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "artifacts/vpd_cluster_aware_sae_semantic_dispatch_slice_benchmark_v2/report.json"
)
BENCHMARK_ID = "681abfc57a4103c72cd7aa7567f23ead61a3f3d9569155c14709902053c26451"
SLICE_ID = "891fbecf114deb595d984e252979301aae7a8fd5c92c305651b12178e854402b"
RELEASE_ID = "f86921ad0b94a557cde269b1b56e9fd04c940e12035233bbc4b3815664ef37df"
RUNTIME_ADMISSION_ID = (
    "8b4947f36fd50fa162aee42a9ae7b591791d27e3ed03644d996a10b031e2b130"
)


def test_semantic_dispatch_slice_v2_benchmark_reproduces() -> None:
    verification = verify_benchmark(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == BENCHMARK_ID
    assert verification["semantic_dispatch_slice_id"] == SLICE_ID
    assert verification["controls"] == 25
    assert verification["dispatched_nodes"] == 7
    assert verification["full_reproduction_comparisons"] == 5
    assert verification["cryptographic_receipt_verifications"] == 2
    assert verification["execution_jobs"] == 34
    assert verification["terminal_gap_count"] == 2
    assert verification["synthetic_fixture_persisted"] is False
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False


def test_semantic_dispatch_slice_v2_attacks_and_boundaries_are_explicit() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert observed["benchmark_id"] == BENCHMARK_ID
    assert observed["results"]["semantic_dispatch_slice_id"] == SLICE_ID
    assert observed["results"]["scientific_release_receipt_id"] == RELEASE_ID
    assert observed["results"]["runtime_admission_id"] == RUNTIME_ADMISSION_ID
    assert set(observed["results"]["dispatched_node_ids"]) == {
        "materialization",
        "execution_blueprint",
        "blueprint_bound_budget_proposal_v2",
        "blueprint_bound_budget_admission_v2",
        "budget_bound_scientific_release_receipt_v2",
        "compute_authorization_receipt_v1",
        "budget_bound_runtime_admission_v2",
    }
    for control in (
        "seven_real_semantic_verifiers_dispatch",
        "five_nodes_compare_full_reproduction",
        "two_ed25519_receipts_verified",
        "exact_34_job_budget_shape_preserved",
        "release_binds_exact_budget_ids_and_file_hashes",
        "release_reproduces_proposal_and_admission",
        "runtime_consumes_budget_bound_release",
        "two_budget_terminal_joins_closed",
        "durable_transition_and_production_attachment_remain_gaps",
        "invalid_release_signature_rejected",
        "signed_wrong_budget_admission_id_rejected",
        "signed_wrong_budget_admission_hash_rejected",
        "expired_release_rejected",
        "disabled_release_authority_rejected",
        "revoked_release_rejected",
        "consumed_release_nonce_rejected",
        "readdressed_proposal_substitution_rejected",
        "readdressed_admission_substitution_rejected",
        "compute_receipt_wrong_release_rejected",
        "readdressed_runtime_admission_mutation_rejected",
        "empirical_or_real_authority_kind_rejected",
        "path_traversal_rejected",
        "slice_transfers_no_authority",
        "benchmark_imports_no_model_frameworks",
        "temporary_fixture_root_removed",
    ):
        assert observed["controls"][control] is True
    boundary = observed["scientific_boundary"]
    assert boundary["synthetic_signed_receipts_verified"] is True
    assert boundary["end_to_end_semantic_dispatch_complete"] is False
    for field in (
        "real_signed_authority_dispatched",
        "nonce_consumed",
        "output_reserved",
        "durable_transition_dispatched",
        "production_runtime_attachment_dispatched",
        "model_framework_imported",
        "model_loaded",
        "model_inference_performed",
        "training_performed",
        "weights_loaded_or_mutated",
        "persistent_dictionary_output_created",
        "execution_authorized",
        "sae_training_authorized",
        "paid_compute_authorized",
    ):
        assert boundary[field] is False
