from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v39 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v31 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v39.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v31/runbook.json"


def test_protocol_v39_reproduces_durable_transition() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["protocol_sha256"] == (
        "b167797dc91c93292ae651d05c96fcb433995cedf62a7a340e85c4e305a92ab7"
    )
    assert verification["transition_benchmark_id"] == (
        "04952c9f6f8648c0a90f9cd3426e2ef546dfc6377987639ce2733479deed1861"
    )
    assert verification["transition_controls"] == 20
    assert verification["release_nonce_campaign_reservation_implemented"] is True
    assert verification[
        "atomic_compute_nonce_and_job_root_reservation_implemented"
    ] is True
    assert verification["worker_ledger_reservation_implemented"] is True
    assert verification["runtime_attach_implemented"] is False
    assert verification["trainer_invocation_implemented"] is False
    assert verification["empirical_blueprint_present"] is False
    assert verification["execution_authorized"] is False


def test_protocol_v39_preserves_transition_boundary() -> None:
    observed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    component = observed["component_compartments"][
        "cluster_aware_sae_transition_v1"
    ]
    assert component["release_nonce_campaign_reservation_implemented"] is True
    assert component["one_release_campaign_supports_multiple_unique_jobs"] is True
    assert component[
        "atomic_compute_nonce_and_job_root_reservation_implemented"
    ] is True
    assert component["worker_ledger_reserved_before_runtime"] is True
    assert component["partial_failures_leave_terminal_tombstones"] is True
    assert component["rollback_or_resume_implemented"] is False
    assert component["runtime_attach_implemented"] is False
    assert component["trainer_invocation_implemented"] is False
    assert component["execution_authorized"] is False
    assert observed["scientific_boundary"]["paid_compute_authorized"] is False


def test_runbook_v31_reproduces_and_stops_before_runtime_attach() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["runbook_id"] == (
        "91108681b156f3ad3e6ac7449d32fa8e052147dd04ea0ba5109fe39fb69e476e"
    )
    assert verification["safe_stage_now"] == 0
    assert verification["cluster_aware_durable_transition_ready"] is True
    assert verification["release_campaign_nonce_reservation_implemented"] is True
    assert verification[
        "atomic_compute_nonce_and_job_root_reservation_implemented"
    ] is True
    assert verification["worker_ledger_reservation_implemented"] is True
    assert verification["runtime_attach_implemented"] is False
    assert verification["trainer_invocation_implemented"] is False
    assert verification["empirical_cluster_aware_blueprint_present"] is False
    assert verification["real_scientific_release_receipt_present"] is False
    assert verification["real_compute_authorization_present"] is False
    assert verification["sae_training_authorized"] is False
    observed = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    command_ids = {row["id"] for row in observed["stages"][0]["commands"]}
    assert {
        "verify_cluster_aware_sae_transition_benchmark_v1",
        "verify_research_protocol_v39",
    } <= command_ids
