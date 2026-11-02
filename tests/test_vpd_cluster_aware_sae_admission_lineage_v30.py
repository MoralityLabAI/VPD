from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v38 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v30 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v38.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v30/runbook.json"


def test_protocol_v38_reproduces_signed_admission() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "90660a5aaacaffeb33d22e6c49cb02a2e160a2b2a5a2946f9f31990c3d1fbd06"
    )
    assert verification["signed_admission_benchmark_id"] == (
        "98c5613d4fe7615ca217d51e80460811e1901d05d177d7839a4aaf8d6f8d79b2"
    )
    assert verification["signed_admission_controls"] == 24
    assert verification["dry_run_cli_implemented"] is True
    assert verification["durable_nonce_consumption_implemented"] is False
    assert verification["atomic_output_reservation_implemented"] is False
    assert verification["production_trainer_invocation_implemented"] is False
    assert verification["empirical_blueprint_present"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v38_preserves_signed_nonexecution_boundary() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    component = observed["component_compartments"][
        "cluster_aware_sae_signed_admission_v1"
    ]
    assert component["ed25519_scientific_release_receipt_verified"] is True
    assert component["ed25519_job_compute_receipt_verified"] is True
    assert component[
        "release_binds_exact_blueprint_plan_materialization_and_global_split"
    ] is True
    assert component[
        "compute_binds_exact_job_samples_resources_device_and_output"
    ] is True
    assert component["synthetic_blueprint_rejected_for_real_release"] is True
    assert component["dry_run_cli_implemented"] is True
    assert component["durable_nonce_consumption_implemented"] is False
    assert component["atomic_output_reservation_implemented"] is False
    assert component["execution_authorized"] is False
    assert observed["scientific_boundary"]["paid_compute_authorized"] is False


def test_runbook_v30_reproduces_and_stops_before_transition() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "8f3cbc0bff18b1cdc8f85e71b220f28269f2f3db44fac32fdf8883b1689d1d61"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["cluster_aware_signed_admission_ready"] is True
    assert verification["cluster_aware_dry_run_cli_implemented"] is True
    assert verification["durable_nonce_consumption_implemented"] is False
    assert verification["atomic_output_reservation_implemented"] is False
    assert verification["production_trainer_invocation_implemented"] is False
    assert verification["empirical_cluster_aware_blueprint_present"] is False
    assert verification["scientific_release_receipt_present"] is False
    assert verification["explicit_compute_authorization_present"] is False
    assert verification["sae_training_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_cluster_aware_sae_admission_benchmark_v1",
        "verify_research_protocol_v38",
    } <= command_ids
