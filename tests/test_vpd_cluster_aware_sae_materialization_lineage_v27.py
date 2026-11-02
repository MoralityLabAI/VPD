from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v35 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v27 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v35.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v27/runbook.json"


def test_protocol_v35_reproduces_guarded_cluster_plan_materialization() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "6b9650239cb69cb51a0d72dd003c9b407086007ffc5dada7a14f616b0d4f3b9c"
    )
    assert verification["materialization_benchmark_id"] == (
        "a8dfb72bd021847f2511258c1d66f08d8bf7ce9c8e1a822f41d4b1af437d3601"
    )
    assert verification["materialization_controls"] == 18
    assert verification["activation_categories"] == 8
    assert verification["cluster_bound_training_jobs"] == 34
    assert verification["persistent_empirical_bundle_present"] is False
    assert verification["empirical_cluster_aware_plan_present"] is False
    assert verification["sae_training_authorized"] is False


def test_protocol_v35_keeps_empirical_and_training_boundaries_explicit() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    component = observed["component_compartments"][
        "guarded_cluster_aware_sae_materialization_v1"
    ]
    assert component["exact_activation_categories"] == 8
    assert component["roles_per_category"] == 3
    assert component["adapter_receipts"] == 24
    assert component["routing_receipts"] == 24
    assert component["independent_stability_receipts"] == 8
    assert component["global_pair_split_required"] is True
    assert component["synthetic_evidence_as_empirical_rejected"] is True
    assert component["persistent_empirical_bundle_present"] is False
    assert component["sae_training_authorized"] is False
    assert observed["scientific_boundary"]["paid_compute_authorized"] is False


def test_runbook_v27_reproduces_and_stops_before_empirical_materialization() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "d9de56b5b51ebfd48dc7625e853c78264da640700d1a1623bf82385358285c53"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["cluster_aware_sae_materializer_ready"] is True
    assert verification["activation_categories"] == 8
    assert verification["category_role_routing_receipts_required"] == 24
    assert verification["persistent_empirical_routing_bundle_present"] is False
    assert verification["empirical_cluster_aware_plan_present"] is False
    assert verification["identity_bindings_complete"] is False
    assert verification["capture_started"] is False
    assert verification["sae_training_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_cluster_aware_sae_materialization_benchmark_v1",
        "verify_research_protocol_v35",
    } <= command_ids
