from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v45 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v37 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v45.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v37/runbook.json"


def test_protocol_v45_reproduces_empirical_budget_binding_readiness() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "678684a7a898488e3a3c0521ffb8b1de5b6b51912cc6ea0891ec9d8856b18dea"
    )
    assert verification["binding_readiness_id"] == (
        "08bdaaf4733afeae33d847e25bba0ea5657a2723ed49b58137f64c9904084c7a"
    )
    assert verification["binding_benchmark_id"] == (
        "b95ab40c46c621534b0cf2aed26c342848180a9c5bc7fd48b9674498292159a3"
    )
    assert verification["binding_controls"] == 29
    assert verification["binding_phases"] == 6
    assert verification["required_bindings"] == 50
    assert verification["missing_bindings"] == 50
    assert verification["syntax_complete"] is False
    assert verification["budget_binding_verified"] is False
    assert verification["signed_authority_verified"] is False
    assert verification["durable_runtime_transition_verified"] is False
    assert verification["training_performed"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v45_preserves_exact_blocked_boundary() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    component = observed["component_compartments"][
        "cluster_aware_sae_empirical_budget_binding_readiness_v1"
    ]
    assert component["capture_execution_identity_bindings"] == 10
    assert component["artifact_graph_bindings"] == 40
    assert component["supplied_bindings"] == 0
    assert component["missing_bindings"] == 50
    assert component["synthetic_complete_fixture_persisted"] is False
    assert component["empirical_blueprint_present"] is False
    assert component["training_performed"] is False
    assert component["execution_authorized"] is False
    boundary = observed["scientific_boundary"]
    assert boundary["empirical_budget_binding_verified"] is False
    assert boundary["signed_authority_verified"] is False
    assert boundary["durable_runtime_transition_verified"] is False
    assert boundary["paid_compute_authorized"] is False


def test_runbook_v37_reproduces_and_stops_at_missing_bindings() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "0d5d1dfde0559e4040b40c12d655c95a9f7cf53f102c2350a87b0370ddd7fac2"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["empirical_budget_binding_readiness_ready"] is True
    assert verification["binding_phases"] == 6
    assert verification["required_bindings"] == 50
    assert verification["missing_bindings"] == 50
    assert verification["syntax_complete"] is False
    assert verification["budget_binding_verified"] is False
    assert verification["signed_authority_verified"] is False
    assert verification["durable_runtime_transition_verified"] is False
    assert verification["training_performed"] is False
    assert verification["persistent_dictionary_output_created"] is False
    assert verification["empirical_cluster_aware_blueprint_present"] is False
    assert verification["real_scientific_release_receipt_present"] is False
    assert verification["real_compute_authorization_present"] is False
    assert verification["sae_training_authorized"] is False
    assert verification["execution_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_cluster_aware_sae_empirical_budget_binding_readiness_v1",
        "verify_cluster_aware_sae_empirical_budget_binding_benchmark_v1",
        "verify_research_protocol_v45",
    } <= command_ids
