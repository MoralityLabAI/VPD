from __future__ import annotations

from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_plan_v1 import (
    build_report,
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = (
    ROOT / "artifacts/vpd_canonical_capture_job_plan_benchmark_v1/report.json"
)


@pytest.fixture(scope="module")
def report() -> dict:
    return build_report()


def test_capture_job_plan_benchmark_covers_exact_production_boundaries(
    report: dict,
) -> None:
    assert report["status"] == "pass"
    assert report["scenario"]["activation_categories"] == 8
    assert report["scenario"]["components"] == 2
    assert report["scenario"]["plan_reproduced"] is True
    assert report["scenario"]["plan_verification"]["execution_authorized"] is False
    assert all(report["negative_controls"].values())


def test_synthetic_authorization_is_not_execution_authority(report: dict) -> None:
    boundary = report["scientific_boundary"]
    assert boundary["synthetic_authorization_is_not_user_authorization"] is True
    assert boundary["model_loaded"] is False
    assert boundary["model_inference_performed"] is False
    assert boundary["weights_mutated"] is False
    assert boundary["execution_authorized"] is False


def test_stored_capture_job_plan_benchmark_verifies() -> None:
    verification = verify_benchmark(REPORT_PATH)
    assert verification["status"] == "pass"
    assert verification["negative_controls"] == 9
    assert verification["execution_authorized"] is False
