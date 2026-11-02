from __future__ import annotations

from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_runtime_v1 import (
    build_report,
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = (
    ROOT / "artifacts/vpd_canonical_capture_job_runtime_benchmark_v1/report.json"
)


@pytest.fixture(scope="module")
def report() -> dict:
    return build_report()


def test_guarded_capture_runtime_completes_category_preserving_shard(
    report: dict,
) -> None:
    scenario = report["scenario"]
    assert report["status"] == "pass"
    assert scenario["activation_categories"] == 8
    assert scenario["components"] == 2
    assert scenario["forward_passes"] == 32
    assert scenario["factor_restored_after_success"] is True
    assert scenario["runtime_closed_after_success"] is True
    assert scenario["execution_verification"]["release_authorized"] is False


def test_guarded_capture_runtime_failures_restore_and_leave_no_output(
    report: dict,
) -> None:
    assert all(report["negative_controls"].values())
    assert report["scientific_boundary"]["torch_absent_before_and_after"] is True
    assert report["scientific_boundary"]["peft_absent_before_and_after"] is True
    assert report["scientific_boundary"]["actual_model_loaded"] is False
    assert report["scientific_boundary"]["actual_model_inference_performed"] is False


def test_stored_guarded_capture_runtime_benchmark_verifies() -> None:
    verification = verify_benchmark(REPORT_PATH)
    assert verification["status"] == "pass"
    assert verification["negative_controls"] == 8
    assert verification["actual_model_loaded"] is False
    assert verification["execution_authorized"] is False
