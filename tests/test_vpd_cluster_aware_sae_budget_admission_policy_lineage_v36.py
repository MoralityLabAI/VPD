from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v44 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v36 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v44.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v36/runbook.json"


def test_protocol_v44_reproduces_budget_admission_policy() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "d7771004061a7521fd743da293af5c20d16bd1b8667970e54f7d1f61496e8828"
    )
    assert verification["budget_policy_id"] == (
        "5e9f4fc0841d0e71a2eef6f9ac91f5e27d0702c0e0aadfa5a6f511dad50719f2"
    )
    assert verification["budget_benchmark_id"] == (
        "5a2bb38b94c43856bcc9d7e7932db98e3cb89192af60e3418706d33713b6db64"
    )
    assert verification["budget_controls"] == 32
    assert verification["categories"] == 8
    assert verification["compartments"] == 17
    assert verification["jobs_ceiling"] == 34
    assert verification["planned_steps_ceiling"] == 68000
    assert verification["planned_activation_rows_ceiling"] == 20480
    assert verification["shared_fallback_charged_to_categories"] is False
    assert verification["budget_admission_transfers_execution_authority"] is False
    assert verification["training_performed"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v44_preserves_nonexecuting_budget_boundary() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    component = observed["component_compartments"][
        "cluster_aware_sae_budget_admission_policy_v1"
    ]
    assert component["specific_dictionary_features"] == 4096
    assert component["shared_fallback_dictionary_features"] == 8192
    assert component["shared_fallback_charged_to_categories"] is False
    assert component["budget_admission_transfers_execution_authority"] is False
    assert component["fresh_empirical_blueprint_required"] is True
    assert component["fresh_active_signed_receipts_required"] is True
    assert component["fresh_nonces_and_output_reservations_required"] is True
    assert component["empirical_blueprint_present"] is False
    assert component["training_performed"] is False
    assert component["execution_authorized"] is False
    boundary = observed["scientific_boundary"]
    assert boundary["budget_admission_is_empirical_evidence"] is False
    assert boundary["paid_compute_authorized"] is False


def test_runbook_v36_reproduces_and_stops_before_empirical_execution() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "69ed3ae8195993372a58bc02231daaa223181383a472ef110c893ebf08804de5"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["cluster_aware_budget_admission_policy_ready"] is True
    assert verification["categories"] == 8
    assert verification["compartments"] == 17
    assert verification["jobs_ceiling"] == 34
    assert verification["planned_steps_ceiling"] == 68000
    assert verification["planned_activation_rows_ceiling"] == 20480
    assert verification["shared_fallback_charged_to_categories"] is False
    assert verification["budget_admission_transfers_execution_authority"] is False
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
        "verify_cluster_aware_sae_budget_admission_policy_v1",
        "verify_cluster_aware_sae_budget_admission_policy_benchmark_v1",
        "verify_research_protocol_v44",
    } <= command_ids
