from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v43 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v35 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v43.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v35/runbook.json"


def test_protocol_v43_reproduces_exact_resource_reconciliation() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "b7d2f07b19295176c5afcd06349c07dcb2f5a4aff5679d2c74f493744f5626e5"
    )
    assert verification["resource_reconciliation_id"] == (
        "c9759d5e1f89cd3f2e6d3a9fa264e27f0ae7e1eee933c3413de5ac3de54bf0ff"
    )
    assert verification["resource_benchmark_id"] == (
        "0c9800d968a49df90b7b1ab942ed5f59017da4f0ee491eb8440b6257aad0f642"
    )
    assert verification["resource_controls"] == 24
    assert verification["categories"] == 8
    assert verification["compartments"] == 17
    assert verification["jobs"] == 34
    assert verification["planned_steps"] == 68000
    assert verification["planned_activation_rows"] == 20480
    assert verification["raw_campaign_receipts_recovered"] is False
    assert verification["shared_fallback_double_counted"] is False
    assert verification["retry_authorized"] is False
    assert verification["training_performed"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v43_preserves_reconstructed_nonexecuting_boundary() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    component = observed["component_compartments"][
        "cluster_aware_sae_compartment_resource_reconciliation_v1"
    ]
    assert component["reconstructed_from_verified_campaign_summary"] is True
    assert component["raw_campaign_receipts_recovered"] is False
    assert component["specific_dictionary_features"] == 4096
    assert component["shared_fallback_dictionary_features"] == 8192
    assert component["shared_fallback_double_counted"] is False
    assert component["failure_charged_callbacks"] == 8
    assert component["failure_uninvoked_callbacks"] == 26
    assert component["fresh_authority_required_for_every_future_real_job"] is True
    assert component["empirical_blueprint_present"] is False
    assert component["retry_authorized"] is False
    assert component["training_performed"] is False
    assert component["execution_authorized"] is False
    assert observed["scientific_boundary"]["paid_compute_authorized"] is False


def test_runbook_v35_reproduces_and_stops_before_empirical_execution() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "a2d6a5c4902830ccb18658820db14a4431c7d9bb01d2b5ba29a1b7dc39610917"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["cluster_aware_resource_reconciliation_ready"] is True
    assert verification["categories"] == 8
    assert verification["compartments"] == 17
    assert verification["jobs"] == 34
    assert verification["planned_steps"] == 68000
    assert verification["planned_activation_rows"] == 20480
    assert verification["raw_campaign_receipts_recovered"] is False
    assert verification["shared_fallback_double_counted"] is False
    assert verification["retry_authorized"] is False
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
        "verify_cluster_aware_sae_resource_reconciliation_v1",
        "verify_cluster_aware_sae_resource_reconciliation_benchmark_v1",
        "verify_research_protocol_v43",
    } <= command_ids
