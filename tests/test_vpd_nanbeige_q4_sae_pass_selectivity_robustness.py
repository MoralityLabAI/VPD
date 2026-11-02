from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_selectivity_robustness_v1 import (
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = {
    "2048": ROOT
    / "artifacts/vpd_nanbeige_q4_sae_pass_selectivity_robustness_2048_v1/report.json",
    "8192": ROOT
    / "artifacts/vpd_nanbeige_q4_sae_pass_selectivity_robustness_8192_v1/report.json",
}


@pytest.mark.parametrize("context", ["2048", "8192"])
def test_report_reproduces_without_execution_authority(context: str) -> None:
    verification = verify_report(REPORTS[context])
    assert verification["status"] == "pass"
    assert verification["execution_authorized"] is False
    assert verification["sites"] == 3
    assert verification["selectivity_configurations_per_site"] == 12
    assert verification["sites_passing_all_selectivity_configurations"] == 2


@pytest.mark.parametrize("context", ["2048", "8192"])
def test_site_level_sensitivity_is_explicit(context: str) -> None:
    report = json.loads(REPORTS[context].read_text(encoding="utf-8"))
    sites = {row["site"]: row for row in report["sites"]}
    assert sites["model.layers.11"]["selectivity_configurations_passing"] == 11
    assert sites["model.layers.21"]["selectivity_configurations_passing"] == 12
    assert sites["model.norm"]["selectivity_configurations_passing"] == 12

    if context == "2048":
        assert sites["model.layers.11"]["minimum_held_out_explained_variance"] == pytest.approx(
            0.2511399276355816
        )
        assert sites["model.layers.21"]["minimum_held_out_explained_variance"] == pytest.approx(
            0.011668844551661728
        )
        assert sites["model.norm"]["minimum_held_out_explained_variance"] == pytest.approx(
            0.2496276574486168
        )
        assert report["summary"]["quality_threshold_admitted_sites"] == {
            "0.0": 2,
            "0.1": 1,
            "0.2": 1,
            "0.25": 0,
            "0.3": 0,
        }
    else:
        assert sites["model.layers.11"]["minimum_held_out_explained_variance"] == pytest.approx(
            0.2629356381461697
        )
        assert sites["model.layers.21"]["minimum_held_out_explained_variance"] == pytest.approx(
            -0.1879651772380071
        )
        assert sites["model.norm"]["minimum_held_out_explained_variance"] == pytest.approx(
            0.2922872645304503
        )
        assert report["summary"]["quality_threshold_admitted_sites"] == {
            "0.0": 1,
            "0.1": 1,
            "0.2": 1,
            "0.25": 1,
            "0.3": 0,
        }


def test_scientific_boundary_forbids_paid_or_weight_execution() -> None:
    for path in REPORTS.values():
        boundary = json.loads(path.read_text(encoding="utf-8"))["scientific_boundary"]
        assert boundary["stored_sparse_feature_archives_used"] is True
        assert boundary["sae_weights_loaded"] is False
        assert boundary["model_weights_loaded"] is False
        assert boundary["new_model_inference_performed"] is False
        assert boundary["new_sae_training_performed"] is False
        assert boundary["does_not_authorize_separate_dictionary_training"] is True
        assert boundary["execution_authorized"] is False
        assert boundary["paid_compute_authorized"] is False
