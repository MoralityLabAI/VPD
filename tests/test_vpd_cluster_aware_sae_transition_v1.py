from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_transition_v1 import (
    build_report,
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_cluster_aware_sae_transition_benchmark_v1/report.json"


def test_cluster_aware_sae_transition_benchmark_reproduces() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    verification = verify_benchmark(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == (
        "04952c9f6f8648c0a90f9cd3426e2ef546dfc6377987639ce2733479deed1861"
    )
    assert verification["controls"] == 20
    assert verification["release_nonce_consumption_implemented"] is True
    assert verification[
        "atomic_compute_nonce_and_job_root_reservation_implemented"
    ] is True
    assert verification["worker_ledger_reservation_implemented"] is True
    assert verification["runtime_attached"] is False
    assert verification["trainer_callback_invoked"] is False
    assert verification["execution_authorized"] is False


def test_durable_transition_is_replay_safe_and_nonexecuting() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    controls = observed["controls"]
    assert controls["release_nonce_creates_durable_campaign"] is True
    assert controls["compute_nonce_atomically_reserves_job_root"] is True
    assert controls["one_release_campaign_can_bind_multiple_unique_jobs"] is True
    assert controls["same_compute_nonce_replay_rejected"] is True
    assert controls["same_compute_nonce_cannot_switch_jobs"] is True
    assert controls["reused_release_nonce_cannot_switch_receipts"] is True
    assert controls["incomplete_campaign_tombstone_is_terminal"] is True
    assert controls["partial_failure_policy_is_fail_closed_no_rollback"] is True
    assert controls["worker_ledger_reserved_before_runtime"] is True
    assert controls[
        "exact_dictionary_output_path_is_reserved_but_not_created"
    ] is True
    assert controls["transition_never_attaches_or_invokes_or_trains"] is True
    boundary = observed["scientific_boundary"]
    assert boundary["persistent_workspace_output_created"] is False
    assert boundary["runtime_attached"] is False
    assert boundary["trainer_callback_invoked"] is False
    assert boundary["weights_loaded_or_mutated"] is False
    assert boundary["execution_authorized"] is False
