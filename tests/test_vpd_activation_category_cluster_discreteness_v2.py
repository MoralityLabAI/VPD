from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.authz_mechinterp.activation_category_cluster_count_selection_v2 import (
    minimum_cluster_boundary_gap_ratio,
)
from experiments.authz_mechinterp.benchmark_activation_category_cluster_discreteness_v2 import (
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "artifacts/vpd_activation_category_cluster_discreteness_benchmark_v2/report.json"
)


def test_discreteness_benchmark_reproduces() -> None:
    verification = verify_benchmark(REPORT)
    assert verification["status"] == "pass"
    assert verification["scenarios"] == 4
    assert verification["legacy_false_splits_reproduced"] == 2
    assert verification["execution_authorized"] is False


def test_continuous_controls_fall_back_to_one_cluster() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    scenarios = {row["scenario_id"]: row for row in report["scenarios"]}
    for scenario_id in (
        "continuous_direction_arc",
        "continuous_strength_gradient",
    ):
        row = scenarios[scenario_id]
        assert row["legacy_false_split_reproduced"] is True
        assert row["v2_allocation_summary"]["cluster_count_histogram"] == {"1": 8}
    assert scenarios["two_discrete_mechanisms"]["v2_allocation_summary"][
        "cluster_count_histogram"
    ] == {"2": 8}


def test_boundary_gap_distinguishes_continuum_from_modes() -> None:
    continuum = np.asarray([[float(index), 0.0] for index in range(12)])
    continuum_labels = np.asarray([0] * 6 + [1] * 6)
    discrete = np.vstack(
        [
            np.asarray([[index * 0.1, 0.0] for index in range(6)]),
            np.asarray([[5.0 + index * 0.1, 0.0] for index in range(6)]),
        ]
    )
    discrete_labels = np.asarray([0] * 6 + [1] * 6)
    assert minimum_cluster_boundary_gap_ratio(
        continuum, continuum_labels
    ) < 1.50
    assert minimum_cluster_boundary_gap_ratio(discrete, discrete_labels) > 1.50
