from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_inert_callback_v1 import (
    build_report,
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT / "artifacts/vpd_cluster_aware_sae_inert_callback_benchmark_v1/report.json"
)


def test_cluster_aware_sae_inert_callback_benchmark_reproduces() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    verification = verify_benchmark(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == (
        "e6ca589f69c120db4a5a9c71d34ba541b9f909673bb6963d0374a632fe326397"
    )
    assert verification["controls"] == 21
    assert verification["durable_execution_claim_implemented"] is True
    assert verification["synthetic_callback_invoked"] is True
    assert verification["failure_charge_persisted"] is True
    assert verification["production_callback_invoked"] is False
    assert verification["model_loaded"] is False
    assert verification["training_performed"] is False
    assert verification["persistent_output_written"] is False
    assert verification["execution_authorized"] is False


def test_inert_callback_is_ordered_charged_replay_safe_and_nontraining() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    controls = observed["controls"]
    assert controls["execution_claim_precedes_success_callback"] is True
    assert controls["execution_claim_precedes_failed_callback"] is True
    assert controls["success_callback_invoked_exactly_once"] is True
    assert controls["success_charge_is_completed_once"] is True
    assert controls["failed_callback_is_charged"] is True
    assert controls["failure_replay_rejected_before_second_callback"] is True
    assert controls["boundary_weakening_result_becomes_charged_failure"] is True
    assert controls["attachment_claim_tamper_rejected_transitively"] is True
    assert controls["callback_execution_imports_no_model_frameworks"] is True
    assert controls["callbacks_create_no_dictionary_output"] is True
    boundary = observed["scientific_boundary"]
    assert boundary["synthetic_callback_invoked"] is True
    assert boundary["production_callback_invoked"] is False
    assert boundary["model_loaded"] is False
    assert boundary["training_performed"] is False
    assert boundary["weights_loaded_or_mutated"] is False
    assert boundary["execution_authorized"] is False
