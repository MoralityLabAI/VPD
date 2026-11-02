from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_nanbeige_activation_phase_category_bridge_v1 import (
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_nanbeige_activation_phase_category_bridge_v1/report.json"


def test_activation_phase_report_reproduces() -> None:
    verification = verify_report(REPORT)
    assert verification["status"] == "pass"
    assert verification["execution_authorized"] is False
    assert verification["sites"] == 3
    assert verification["activation_phase_category_admissions"] == 0
    assert verification["screening_sites_with_held_out_accuracy_at_least_0_875"] == 3
    assert verification["operative_nontrivial_within_phase_allocations"] == 0


def test_phase_screen_is_strong_but_exact_test_is_underpowered() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["design"]["minimum_attainable_exact_p_value"] == 0.0625
    assert report["design"]["maximum_admissible_exact_p_value"] == 0.05
    assert report["design"]["pack_is_underpowered_for_statistical_admission"] is True
    sites = {row["site"]: row for row in report["sites"]}
    expected = {
        "model.layers.11": (1.0, 0.9901150299400411),
        "model.layers.21": (0.875, 0.7921286450946958),
        "model.norm": (0.875, 0.17378540849882568),
    }
    for site, (accuracy, alignment) in expected.items():
        row = sites[site]
        assert row["held_out_phase_accuracy"] == accuracy
        assert row["observed_train_to_held_out_phase_transition_alignment"] == pytest.approx(
            alignment
        )
        assert row["exact_train_example_block_label_flip_control"][
            "exact_upper_tail_p_value"
        ] == 0.0625
        assert row["activation_phase_category_admission"] == "fail"


def test_no_within_phase_child_cluster_is_operative() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    for row in report["sites"]:
        assert row["within_phase_v2_selection"]["allocation"] == {
            "generation": 1,
            "prompt": 1,
        }
        assert row["within_phase_allocation_is_operative"] is False


def test_phase_boundary_forbids_training() -> None:
    boundary = json.loads(REPORT.read_text(encoding="utf-8"))["scientific_boundary"]
    assert boundary["activation_phase_category_is_screening_only"] is True
    assert boundary[
        "statistical_admission_impossible_at_current_exact_control_resolution"
    ] is True
    assert boundary["within_phase_child_allocations_are_inoperative"] is True
    assert boundary["does_not_authorize_phase_compartment_sae_training"] is True
    assert boundary["new_model_inference_performed"] is False
    assert boundary["execution_authorized"] is False
