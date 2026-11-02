from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_nanbeige_activation_phase_category_cross_validation_v1 import (
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_nanbeige_activation_phase_category_cross_validation_v1/report.json"


def test_phase_cross_validation_report_reproduces() -> None:
    verification = verify_report(REPORT)
    assert verification["status"] == "pass"
    assert verification["execution_authorized"] is False
    assert verification["sites"] == 3
    assert verification["provisional_cross_validated_phase_category_supports"] == 3
    assert verification["operative_phase_category_admissions"] == 0
    assert verification["minimum_cross_validated_phase_accuracy"] == 0.9375
    assert verification["maximum_exact_upper_tail_p_value"] == 0.0078125


def test_all_sites_have_family_blocked_provisional_support() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    sites = {row["site"]: row for row in report["sites"]}
    expected = {
        "model.layers.11": (1.0, 0.9998098406859846),
        "model.layers.21": (0.9375, 0.9967020026918122),
        "model.norm": (0.9375, 0.9257355023099765),
    }
    for site, (accuracy, cosine) in expected.items():
        row = sites[site]
        assert row["leave_one_family_out_phase_accuracy"] == accuracy
        assert row[
            "leave_one_family_out_minimum_direction_cosine_to_full_direction"
        ] == pytest.approx(cosine)
        assert row["exact_example_block_phase_label_permutation"][
            "exact_upper_tail_p_value"
        ] == 0.0078125
        assert row["provisional_cross_validated_phase_category_support"] == "pass"
        assert row["operative_phase_category_admission"] == "fail"


def test_cross_validation_is_family_blocked_and_exact() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["sources"]["semantic_families"] == [
        "arc_grid",
        "dyck_language",
        "futoshiki",
        "math",
    ]
    assert report["design"]["validation"] == (
        "leave_one_semantic_family_out_cross_validation"
    )
    assert report["design"]["exact_permutations"] == 256
    assert report["design"]["fresh_untouched_holdout_still_required"] is True


def test_provisional_support_is_not_training_authority() -> None:
    boundary = json.loads(REPORT.read_text(encoding="utf-8"))["scientific_boundary"]
    assert boundary["cross_validation_reuses_all_examples_across_folds"] is True
    assert boundary["no_fully_untouched_final_holdout"] is True
    assert boundary["provisional_support_is_not_category_admission"] is True
    assert boundary["does_not_authorize_phase_compartment_sae_training"] is True
    assert boundary["new_model_inference_performed"] is False
    assert boundary["execution_authorized"] is False
