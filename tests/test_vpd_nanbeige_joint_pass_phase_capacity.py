from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_nanbeige_joint_pass_phase_capacity_v1 import (
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_nanbeige_joint_pass_phase_capacity_v1/report.json"


def test_joint_pass_phase_capacity_report_reproduces() -> None:
    verification = verify_report(REPORT)
    assert verification["status"] == "pass"
    assert verification["execution_authorized"] is False
    assert verification["sites"] == 3
    assert verification["joint_capacity_preferences"] == 0
    assert verification["joint_wins_vs_monolithic"] == 0
    assert verification["joint_wins_vs_pass_routed"] == 0
    assert verification["joint_wins_vs_phase_routed"] == 0
    assert verification["joint_wins_vs_random_mean"] == 3
    assert verification["joint_compartment_training_admissions"] == 0


def test_joint_routing_loses_to_every_structured_simpler_arm() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    expected_monolithic_deltas = {
        "model.layers.11": -2.3717709365489448e-05,
        "model.layers.21": -0.00044751950482047054,
        "model.norm": -0.05536543811759942,
    }
    for row in report["sites"]:
        assert row["joint_minus"]["monolithic"]["delta"] == pytest.approx(
            expected_monolithic_deltas[row["site"]]
        )
        for reference in ("monolithic", "pass_routed", "phase_routed"):
            assert row["joint_minus"][reference][
                "paired_example_block_bootstrap_95_ci"
            ][1] < 0.0
        assert row["joint_minus"]["random_partition_mean"][
            "paired_example_block_bootstrap_95_ci"
        ][0] > 0.0
        assert row["joint_capacity_preference"] == "fail"
        assert row["joint_compartment_training_admission"] == "fail"


def test_joint_capacity_comparison_is_equal_rank_and_equal_row() -> None:
    design = json.loads(REPORT.read_text(encoding="utf-8"))["design"]
    assert design["total_rank_per_arm"] == 64
    assert {name: sum(ranks) for name, ranks in design["arm_ranks"].items()} == {
        "monolithic": 64,
        "pass_routed": 64,
        "phase_routed": 64,
        "joint_pass_phase": 64,
        "random_four_way": 64,
    }
    assert design["rows_per_pass_per_phase_per_example"] == 8
    assert design["all_arms_use_identical_rows_and_denominator"] is True
    assert design["four_way_random_train_examples_per_subspace"] == 1
    assert design["four_way_random_control_is_minimal_and_screening_only"] is True


def test_joint_capacity_boundary_is_non_authorizing() -> None:
    boundary = json.loads(REPORT.read_text(encoding="utf-8"))["scientific_boundary"]
    assert boundary["linear_subspace_proxy_is_not_sae_training"] is True
    assert boundary["phase_parent_category_operatively_admitted"] is False
    assert boundary["capacity_preference_cannot_bypass_parent_category_gate"] is True
    assert boundary["does_not_authorize_joint_compartment_sae_training"] is True
    assert boundary["new_model_inference_performed"] is False
    assert boundary["weights_loaded_or_mutated"] is False
    assert boundary["execution_authorized"] is False
