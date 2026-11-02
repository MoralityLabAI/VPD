from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_inert_attachment_v1 import (
    build_report,
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT / "artifacts/vpd_cluster_aware_sae_inert_attachment_benchmark_v1/report.json"
)


def test_cluster_aware_sae_inert_attachment_benchmark_reproduces() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    verification = verify_benchmark(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == (
        "128a32c051b18d32be0af877929f7c5344562b13d57993c6fd9c40bedfb8e1e8"
    )
    assert verification["controls"] == 20
    assert verification["durable_attachment_claim_implemented"] is True
    assert verification["ledger_reconstruction_implemented"] is True
    assert verification["inert_runtime_attached"] is True
    assert verification["production_runtime_attached"] is False
    assert verification["trainer_callback_invoked"] is False
    assert verification["execution_authorized"] is False


def test_inert_attachment_is_ordered_replay_safe_and_nonexecuting() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    controls = observed["controls"]
    assert controls["durable_claim_precedes_runtime_factory"] is True
    assert controls["transition_and_signed_admission_reconstruct_exactly"] is True
    assert controls["inert_runtime_attaches_after_claim"] is True
    assert controls["production_runtime_remains_absent"] is True
    assert controls["trainer_callback_never_invoked"] is True
    assert controls["attachment_claim_is_single_use_across_reconstruction"] is True
    assert controls["factory_failure_leaves_terminal_claim_tombstone"] is True
    assert controls["factory_failure_retry_rejected_before_factory"] is True
    assert controls["active_runtime_rejected_and_closed"] is True
    assert controls["active_runtime_rejection_leaves_terminal_claim"] is True
    assert controls["ledger_tamper_rejected_before_claim"] is True
    assert controls["inert_attachment_creates_no_dictionary_output"] is True
    boundary = observed["scientific_boundary"]
    assert boundary["persistent_workspace_claim_created"] is False
    assert boundary["model_loaded"] is False
    assert boundary["production_runtime_attached"] is False
    assert boundary["trainer_callback_invoked"] is False
    assert boundary["execution_authorized"] is False
