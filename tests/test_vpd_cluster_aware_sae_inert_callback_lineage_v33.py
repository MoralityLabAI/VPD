from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v41 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v33 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v41.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v33/runbook.json"


def test_protocol_v41_reproduces_inert_callback_receipts() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "87c7b79aac37bc094ddfe06501f020eda9857c5ef08767eff492f5c0cb21476b"
    )
    assert verification["inert_callback_benchmark_id"] == (
        "e6ca589f69c120db4a5a9c71d34ba541b9f909673bb6963d0374a632fe326397"
    )
    assert verification["inert_callback_controls"] == 21
    assert verification["durable_callback_execution_claim_implemented"] is True
    assert verification["synthetic_inert_callback_invoked"] is True
    assert verification["failed_callback_charge_persisted"] is True
    assert verification["production_trainer_callback_invoked"] is False
    assert verification["training_performed"] is False
    assert verification["persistent_dictionary_output_created"] is False
    assert verification["empirical_blueprint_present"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v41_preserves_synthetic_vs_production_boundary() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    component = observed["component_compartments"][
        "cluster_aware_sae_inert_callback_v1"
    ]
    assert component["durable_execution_claim_before_callback"] is True
    assert component["successful_callback_charge_persisted"] is True
    assert component["failed_callback_charge_persisted"] is True
    assert component["boundary_weakening_result_is_terminal_failure"] is True
    assert component["synthetic_inert_callback_invoked"] is True
    assert component["production_trainer_callback_invoked"] is False
    assert component["training_performed"] is False
    assert component["persistent_dictionary_output_created"] is False
    assert component["execution_authorized"] is False
    assert observed["scientific_boundary"]["paid_compute_authorized"] is False


def test_runbook_v33_reproduces_and_stops_before_production_training() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "23e7c03df8c0b73f71e85cfaea4977b230e129d82211fa7e944e71920eaeecc8"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["cluster_aware_inert_callback_ready"] is True
    assert verification["durable_callback_execution_claim_implemented"] is True
    assert verification["synthetic_inert_callback_invoked"] is True
    assert verification["synthetic_callback_failure_charge_persisted"] is True
    assert verification["production_trainer_callback_invoked"] is False
    assert verification["training_performed"] is False
    assert verification["persistent_dictionary_output_created"] is False
    assert verification["empirical_cluster_aware_blueprint_present"] is False
    assert verification["real_scientific_release_receipt_present"] is False
    assert verification["real_compute_authorization_present"] is False
    assert verification["sae_training_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_cluster_aware_sae_inert_callback_benchmark_v1",
        "verify_research_protocol_v41",
    } <= command_ids
