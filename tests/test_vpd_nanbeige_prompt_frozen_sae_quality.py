from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.audit_nanbeige_prompt_frozen_sae_quality_v1 import (
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_nanbeige_prompt_frozen_sae_quality_v1/report.json"


def test_prompt_frozen_sae_quality_report_reproduces() -> None:
    verification = verify_report(REPORT)
    assert verification["status"] == "pass"
    assert verification["site_checks"] == 3
    assert verification["feature_interpretation_quality_admissions"] == 0
    assert verification["prompt_phase_pass_feature_strata_admissions"] == 0
    assert verification["execution_authorized"] is False


def test_every_prompt_site_is_rejected_on_held_out_reconstruction() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    expected_minima = {
        "model.layers.11": -0.08210552224626455,
        "model.layers.21": -1.2934838949230807,
        "model.norm": 0.07984828542967848,
    }
    for row in report["sites"]:
        assert row["minimum_held_out_explained_variance"] == pytest.approx(
            expected_minima[row["site"]]
        )
        assert row["minimum_held_out_explained_variance"] < row["quality_threshold"]
        assert row["feature_interpretation_quality_gate"] == "fail"
        assert row["pass_feature_strata_admission"] == "fail"


def test_prompt_feature_archive_is_frozen_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["sources"]["frozen_sae_transfer"]["no_new_fitting"] is True
    assert report["design"]["held_out_examples"] == 4
    assert report["design"]["minimum_held_out_explained_variance"] == 0.2
    assert report["design"]["site_must_pass_for_both_recurrent_passes"] is True
    assert (
        report["design"]["dictionary_reload_max_explained_variance_abs_error"]
        < 2e-8
    )


def test_prompt_feature_quality_boundary_is_non_authorizing() -> None:
    boundary = json.loads(REPORT.read_text(encoding="utf-8"))["scientific_boundary"]
    assert boundary["frozen_feature_archives_exist"] is True
    assert boundary["archive_existence_is_not_feature_interpretation_validity"] is True
    assert boundary["prompt_phase_feature_interpretation_quality_admitted"] is False
    assert boundary["cross_phase_pass_feature_strata_admitted"] is False
    assert boundary["new_sae_fitted"] is False
    assert boundary["new_model_inference_performed"] is False
    assert boundary["weights_loaded_or_mutated"] is False
    assert boundary["execution_authorized"] is False
