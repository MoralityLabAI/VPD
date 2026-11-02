from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_nanbeige_activation_category_bridge_v1 import (
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_nanbeige_activation_category_bridge_v1/report.json"
REASONING_REPORT = (
    ROOT / "artifacts/vpd_nanbeige_reasoning_activation_category_bridge_v1/report.json"
)


def test_empirical_nanbeige_bridge_reproduces() -> None:
    for path in (REPORT, REASONING_REPORT):
        verification = verify_report(path)
        assert verification["status"] == "pass"
        assert verification["sites"] == 3
        assert verification["execution_authorized"] is False


def test_empirical_bridge_admits_pass_category_but_no_within_pass_split() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["summary"]["transition_category_admissions"] == 3
    assert all(row["held_out_pass_accuracy"] == 1.0 for row in report["sites"])
    assert all(
        row["random_control_transition_alignment"]["exact_upper_tail_p_value"]
        == 0.015625
        for row in report["sites"]
    )
    assert all(
        row["within_pass_v2_selection"]["allocation"]
        == {"pass_0": 1, "pass_1": 1}
        for row in report["sites"]
    )
    assert report["scientific_boundary"][
        "does_not_validate_bonsai_authorization_categories"
    ] is True
    assert report["scientific_boundary"][
        "does_not_authorize_sae_training_or_model_release"
    ] is True


def test_reasoning_bridge_replicates_coarse_pass_without_subclusters() -> None:
    report = json.loads(REASONING_REPORT.read_text(encoding="utf-8"))
    assert report["summary"]["transition_category_admissions"] == 3
    assert all(row["held_out_pass_accuracy"] == 1.0 for row in report["sites"])
    assert all(
        row["random_control_transition_alignment"]["exact_upper_tail_p_value"]
        == 0.00390625
        for row in report["sites"]
    )
    assert all(
        row["within_pass_v2_selection"]["allocation"]
        == {"pass_0": 1, "pass_1": 1}
        for row in report["sites"]
    )
