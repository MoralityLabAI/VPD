from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_runtime_worker_v1 import (
    build_report,
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT / "artifacts/vpd_cluster_aware_sae_runtime_worker_benchmark_v1/report.json"
)


def test_cluster_aware_sae_runtime_worker_benchmark_reproduces() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    verification = verify_benchmark(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["benchmark_id"] == (
        "dbab569a095cd9c01449886d84f7a292b75edcf919ae65ba5cfc29af0a059188"
    )
    assert verification["controls"] == 22
    assert verification["execution_jobs"] == 34
    assert verification["worker_core_implemented"] is True
    assert verification["production_cli_implemented"] is False
    assert verification["sae_training_authorized"] is False
    assert verification["execution_authorized"] is False


def test_runtime_worker_is_fail_closed_and_non_authorizing() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    controls = observed["controls"]
    assert controls["job_reserved_before_runtime_loader"] is True
    assert controls["planned_steps_and_rows_charged_at_reservation"] is True
    assert controls["runtime_attachment_is_single_use"] is True
    assert controls["trainer_callback_is_single_use"] is True
    assert controls["failed_callback_remains_charged"] is True
    assert controls["replay_is_rejected"] is True
    assert controls["wrong_job_authority_rejected_before_reservation"] is True
    assert controls["synthetic_blueprint_rejected_for_production"] is True
    assert controls["callback_receives_exact_job_contract"] is True
    assert controls["worker_imports_no_model_frameworks"] is True
    boundary = observed["scientific_boundary"]
    assert boundary["model_loaded"] is False
    assert boundary["persistent_output_created"] is False
    assert boundary["weights_mutated"] is False
    assert boundary["sae_training_authorized"] is False
    assert boundary["execution_authorized"] is False
