from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_nanbeige_phase_compartment_linear_proxy_v1 import (
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_nanbeige_phase_compartment_linear_proxy_v1/report.json"


def test_phase_capacity_report_reproduces() -> None:
    verification = verify_report(REPORT)
    assert verification["status"] == "pass"
    assert verification["execution_authorized"] is False
    assert verification["sites"] == 3
    assert verification["phase_routing_wins_vs_monolithic"] == 0
    assert verification["phase_routing_wins_vs_random_mean"] == 3
    assert verification["screening_capacity_preferences"] == 0
    assert verification["phase_compartment_training_admissions"] == 0


def test_phase_routing_loses_to_monolithic_at_every_site() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    expected_deltas = {
        "model.layers.11": -1.3169738381946239e-05,
        "model.layers.21": -0.0002938321838636959,
        "model.norm": -0.027262343009264778,
    }
    for row in report["sites"]:
        assert row["phase_minus_monolithic"] == pytest.approx(
            expected_deltas[row["site"]]
        )
        assert row["phase_minus_monolithic_paired_bootstrap_95_ci"][1] < 0.0
        assert row[
            "phase_minus_random_partition_mean_paired_bootstrap_95_ci"
        ][0] > 0.0
        assert row["screening_capacity_preference"] == "fail"
        assert row["phase_compartment_training_admission"] == "fail"


def test_capacity_comparison_is_equal_rank_and_equal_row() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    design = report["design"]
    assert design["monolithic_total_rank"] == 64
    assert design["phase_routed_total_rank"] == 64
    assert design["phase_routed_rank_per_subspace"] == 32
    assert design["rows_per_pass_per_phase_per_example"] == 8
    assert design["screening_only_because_phase_parent_is_not_operatively_admitted"] is True


def test_phase_capacity_boundary_is_non_authorizing() -> None:
    boundary = json.loads(REPORT.read_text(encoding="utf-8"))["scientific_boundary"]
    assert boundary["linear_subspace_proxy_is_not_sae_training"] is True
    assert boundary["phase_parent_category_operatively_admitted"] is False
    assert boundary["capacity_preference_cannot_bypass_parent_category_gate"] is True
    assert boundary["does_not_authorize_phase_compartment_sae_training"] is True
    assert boundary["new_model_inference_performed"] is False
    assert boundary["execution_authorized"] is False
