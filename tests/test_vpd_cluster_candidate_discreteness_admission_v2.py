from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_cluster_candidate_discreteness_admission_v2 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = (
    ROOT
    / "artifacts/vpd_cluster_candidate_discreteness_admission_benchmark_v2/report.json"
)


def _report() -> dict:
    return json.loads(BENCHMARK.read_text(encoding="utf-8"))


def test_admission_benchmark_is_content_addressed_and_reproducible() -> None:
    report = _report()
    body = dict(report)
    identifier = body.pop("benchmark_id")
    assert identifier == content_sha256(body)
    verification = verify_benchmark(BENCHMARK, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["scenarios"] == 3


def test_discrete_candidate_is_admitted_after_base_verification() -> None:
    scenario = next(
        row
        for row in _report()["scenarios"]
        if row["scenario_id"] == "two_discrete_mechanisms_admitted"
    )
    assert scenario["base_bundle_verification"]["status"] == "pass"
    assert scenario["base_bundle_verification"]["exact_partition"] is True
    assert scenario["candidate_allocation"] == scenario["v2_allocation"]
    assert scenario["mismatch_count"] == 0
    assert scenario["observed_admitted"] is True


def test_continuous_candidate_is_rejected_without_empirical_claim() -> None:
    report = _report()
    scenario = next(
        row
        for row in report["scenarios"]
        if row["scenario_id"] == "continuous_strength_gradient_rejected"
    )
    assert scenario["base_bundle_verification"]["status"] == "pass"
    assert scenario["mismatch_count"] == 7
    assert set(scenario["v2_allocation"].values()) == {1}
    assert scenario["observed_admitted"] is False
    assert report["scientific_boundary"]["does_not_admit_an_empirical_bonsai_candidate"]
    assert report["scientific_boundary"]["execution_authorized"] is False


def test_native_v2_candidate_is_accepted_and_independently_reverified() -> None:
    scenario = next(
        row
        for row in _report()["scenarios"]
        if row["scenario_id"] == "native_v2_two_discrete_mechanisms_admitted"
    )
    assert scenario["observed_admitted"] is True
    assert scenario["source_selection_schema"] == (
        "sch.vpd-category-cluster-count-selection.v2"
    )
    assert scenario["native_v2_bundle_reverified"] is True
    assert scenario["candidate_allocation"] == scenario["v2_allocation"]
