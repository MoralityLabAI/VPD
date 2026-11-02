from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.audit_cluster_aware_sae_semantic_dispatch_coverage_v1 import (
    canonical_inputs,
    verify_audit,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_semantic_dispatch_coverage_v1 import (
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT = (
    ROOT
    / "artifacts/vpd_cluster_aware_sae_semantic_dispatch_coverage_audit_v1/audit.json"
)
REPORT = (
    ROOT
    / "artifacts/vpd_cluster_aware_sae_semantic_dispatch_coverage_benchmark_v1/report.json"
)
AUDIT_ID = "bd3cc043f7b658029fd20c99492c43edbace23550a7e237d36d692e46297eaa7"
BENCHMARK_ID = "adc1d00bf3be2ab9d790252534bc084add403842b26ae9002e2d6216f0095732"


def test_semantic_dispatch_coverage_audit_reproduces() -> None:
    verification = verify_audit(AUDIT, **canonical_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["semantic_dispatch_coverage_audit_id"] == AUDIT_ID
    assert verification["graph_nodes"] == 20
    assert verification["schema_and_id_matching_nodes"] == 12
    assert verification["dispatch_ready_nodes_under_v1_graph"] == 4
    assert verification["graph_edges"] == 37
    assert verification["exact_implemented_field_edges"] == 9
    assert verification["mismatched_edges"] == 28
    assert verification["production_runtime_attachment_verifier_present"] is False
    assert verification["semantic_dispatch_allowed"] is False
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False


def test_semantic_dispatch_coverage_benchmark_reproduces() -> None:
    verification = verify_benchmark(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == BENCHMARK_ID
    assert verification["semantic_dispatch_coverage_audit_id"] == AUDIT_ID
    assert verification["controls"] == 23
    assert verification["semantic_dispatch_allowed"] is False
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False


def test_semantic_dispatch_gaps_remain_explicit_and_nonauthorizing() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    nodes = {row["node"]: row for row in audit["node_rows"]}
    assert nodes["capture_authorization_receipt"]["implemented_id_field"] == (
        "authorization_receipt_id"
    )
    assert nodes["empirical_capture_manifest"]["implemented_id_field"] == "capture_id"
    assert nodes["empirical_routing_bundle"]["implemented_id_field"] == (
        "routing_evidence_bundle_id"
    )
    assert nodes["global_pair_split"]["verification_mode"] == (
        "embedded_inside_cluster_aware_plan"
    )
    assert nodes["compute_authorization_manifest"]["implemented_schema"] == (
        "sch.vpd-cluster-aware-sae-compute-authorization-receipt.v1"
    )
    assert nodes["nonce_consumption_manifest"]["implemented_id_field"] == (
        "transition_reservation_id"
    )
    assert nodes["output_reservation_manifest"]["implemented_id_field"] == (
        "transition_reservation_id"
    )
    assert nodes["worker_ledger_claim"]["implemented_id_field"] == (
        "transition_reservation_id"
    )
    assert nodes["runtime_attachment_claim"]["implemented_schema"] == (
        "sch.vpd-cluster-aware-sae-inert-runtime-attachment-claim.v1"
    )
    boundary = audit["scientific_boundary"]
    assert boundary["v1_structural_graph_verified_previously"] is True
    assert boundary["v1_semantic_graph_verified"] is False
    assert boundary["empirical_graph_present"] is False
    assert boundary["semantic_verifiers_invoked_on_empirical_inputs"] is False
    assert boundary["model_framework_imported"] is False
    assert boundary["model_loaded"] is False
    assert boundary["model_inference_performed"] is False
    assert boundary["training_performed"] is False
    assert boundary["execution_authorized"] is False
    assert boundary["sae_training_authorized"] is False
