from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v42 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v34 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v42.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v34/runbook.json"


def test_protocol_v42_reproduces_exact_inert_campaign() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "fc612123f8e293c25e57c6b470722ce2a6c5876f1e9a43c477b31945bedab383"
    )
    assert verification["inert_campaign_benchmark_id"] == (
        "34204198cc2cd605e74fff0f24da7b5d09337613f16da74872f354c53f8b32f5"
    )
    assert verification["inert_campaign_controls"] == 28
    assert verification["categories"] == 8
    assert verification["compartments"] == 17
    assert verification["jobs"] == 34
    assert verification["unique_output_directories"] == 34
    assert verification["success_callbacks_charged"] == 34
    assert verification["controlled_failure_callbacks_charged"] == 8
    assert verification["synthetic_inert_campaign_executed"] is True
    assert verification["production_campaign_executed"] is False
    assert verification["training_performed"] is False
    assert verification["persistent_dictionary_output_created"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v42_preserves_compartment_and_training_boundary() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    component = observed["component_compartments"][
        "cluster_aware_sae_inert_campaign_v1"
    ]
    assert component["activation_categories"] == 8
    assert component["compartments"] == 17
    assert component["specific_compartments"] == 16
    assert component["shared_fallback_compartments"] == 1
    assert component["execution_jobs"] == 34
    assert component["unique_output_directories"] == 34
    assert component["one_global_pair_split"] is True
    assert component["synthetic_inert_campaign_executed"] is True
    assert component["production_campaign_executed"] is False
    assert component["production_trainer_callback_invoked"] is False
    assert component["training_performed"] is False
    assert component["persistent_dictionary_output_created"] is False
    assert component["execution_authorized"] is False
    assert observed["scientific_boundary"]["paid_compute_authorized"] is False


def test_runbook_v34_reproduces_and_stops_before_production_campaign() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "9e93f62742b81b7080ba6d75fa12f58a20bf882b5f8ae3190194c6e03d7d9b06"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["cluster_aware_inert_campaign_ready"] is True
    assert verification["categories"] == 8
    assert verification["compartments"] == 17
    assert verification["jobs"] == 34
    assert verification["unique_output_directories"] == 34
    assert verification["synthetic_inert_campaign_executed"] is True
    assert verification["production_campaign_executed"] is False
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
        "verify_cluster_aware_sae_inert_campaign_benchmark_v1",
        "verify_research_protocol_v42",
    } <= command_ids
