from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v40 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v32 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v40.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v32/runbook.json"


def test_protocol_v40_reproduces_inert_attachment() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "197f7f02ffbeb5f9a530a5880391daa0951611bd42b4202236c78e33a151a9c8"
    )
    assert verification["inert_attachment_benchmark_id"] == (
        "128a32c051b18d32be0af877929f7c5344562b13d57993c6fd9c40bedfb8e1e8"
    )
    assert verification["inert_attachment_controls"] == 20
    assert verification["durable_attachment_claim_implemented"] is True
    assert verification["worker_ledger_reconstruction_implemented"] is True
    assert verification["synthetic_inert_runtime_attached"] is True
    assert verification["production_runtime_attach_implemented"] is False
    assert verification["trainer_callback_invoked"] is False
    assert verification["empirical_blueprint_present"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v40_preserves_inert_attachment_boundary() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    component = observed["component_compartments"][
        "cluster_aware_sae_inert_attachment_v1"
    ]
    assert component["durable_attachment_claim_before_runtime_factory"] is True
    assert component["transition_and_signed_admission_reconstructed_exactly"] is True
    assert component["worker_ledger_reconstructed_exactly"] is True
    assert component[
        "attachment_claim_single_use_across_process_reconstruction"
    ] is True
    assert component[
        "factory_and_active_runtime_failures_leave_terminal_claims"
    ] is True
    assert component["synthetic_inert_runtime_attached"] is True
    assert component["trainer_callback_invoked"] is False
    assert component["production_runtime_attach_implemented"] is False
    assert component["execution_authorized"] is False
    assert observed["scientific_boundary"]["paid_compute_authorized"] is False


def test_runbook_v32_reproduces_and_stops_before_production_runtime() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "30bcec255ef18f654134794c14a52d88599264d3c8f387fdca2a991f75c9bacf"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["cluster_aware_inert_attachment_ready"] is True
    assert verification["durable_attachment_claim_implemented"] is True
    assert verification["worker_ledger_reconstruction_implemented"] is True
    assert verification["synthetic_inert_runtime_attached"] is True
    assert verification["production_runtime_attach_implemented"] is False
    assert verification["trainer_callback_invoked"] is False
    assert verification["empirical_cluster_aware_blueprint_present"] is False
    assert verification["real_scientific_release_receipt_present"] is False
    assert verification["real_compute_authorization_present"] is False
    assert verification["sae_training_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_cluster_aware_sae_inert_attachment_benchmark_v1",
        "verify_research_protocol_v40",
    } <= command_ids
