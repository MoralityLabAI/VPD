from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_nanbeige_layer21_sae_seed_objective_stability_v1 import (
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_nanbeige_layer21_sae_seed_objective_stability_v1/report.json"


def test_layer21_seed_objective_report_reproduces() -> None:
    verification = verify_report(REPORT)
    assert verification["status"] == "pass"
    assert verification["conditions"] == 6
    assert verification["quality_qualified_conditions"] == 0
    assert verification["cross_seed_pair_checks"] == 6
    assert verification["cross_seed_diagnostic_supports_without_quality_gate"] == 0
    assert verification["same_seed_cross_objective_pair_checks"] == 3
    assert (
        verification[
            "same_seed_cross_objective_diagnostic_supports_without_quality_gate"
        ]
        == 3
    )
    assert verification["operative_stratum_replication_admissions"] == 0
    assert verification["execution_authorized"] is False


def test_cross_seed_pairs_have_no_primary_threshold_matches() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    pairs = [
        row
        for row in report["pairs"]
        if row["comparison_class"] == "cross_seed_same_objective"
    ]
    assert len(pairs) == 6
    assert all(row["mutual_one_to_one_matches"] == 0 for row in pairs)
    assert all(row["same_category_matches"] == {"pass_0": 0, "pass_1": 0} for row in pairs)
    assert all(row["target_label_permutation_p_value"] == 1.0 for row in pairs)


def test_same_seed_objective_change_preserves_diagnostic_strata_but_not_quality() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    pairs = [
        row
        for row in report["pairs"]
        if row["comparison_class"] == "same_seed_cross_objective"
    ]
    assert [row["mutual_one_to_one_matches"] for row in pairs] == [57, 48, 48]
    assert [row["same_category_matches"] for row in pairs] == [
        {"pass_0": 24, "pass_1": 33},
        {"pass_0": 17, "pass_1": 31},
        {"pass_0": 20, "pass_1": 28},
    ]
    assert all(row["diagnostic_stratum_stability_without_quality_gate"] == "pass" for row in pairs)
    assert all(row["quality_qualified_pair"] is False for row in pairs)
    assert all(row["operative_stratum_replication_admission"] == "fail" for row in pairs)
    boundary = report["scientific_boundary"]
    assert boundary["all_conditions_fail_reconstruction_quality"] is True
    assert boundary["diagnostic_matching_cannot_override_quality_gate"] is True
    assert boundary["does_not_test_layer11_or_final_norm"] is True
    assert boundary["new_model_inference_performed"] is False
    assert boundary["new_sae_training_performed"] is False


def test_layer21_seed_objective_report_rejects_tampering(tmp_path: Path) -> None:
    value = json.loads(REPORT.read_text(encoding="utf-8"))
    value["summary"]["cross_seed_diagnostic_supports_without_quality_gate"] = 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="content address differs"):
        verify_report(path)
