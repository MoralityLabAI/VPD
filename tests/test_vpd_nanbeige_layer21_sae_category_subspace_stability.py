from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_nanbeige_layer21_sae_category_subspace_stability_v1 import (
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_nanbeige_layer21_sae_category_subspace_stability_v1/report.json"


def test_layer21_category_subspace_report_reproduces() -> None:
    verification = verify_report(REPORT)
    assert verification["status"] == "pass"
    assert verification["pair_checks"] == 9
    assert verification["cross_seed_pair_checks"] == 6
    assert verification["cross_seed_diagnostic_subspace_supports_without_quality_gate"] == 0
    assert verification["same_seed_cross_objective_pair_checks"] == 3
    assert (
        verification[
            "same_seed_cross_objective_diagnostic_subspace_supports_without_quality_gate"
        ]
        == 3
    )
    assert verification["operative_category_subspace_replication_admissions"] == 0
    assert verification["execution_authorized"] is False


def test_cross_seed_nominal_hit_fails_familywise_gate() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    pairs = [
        row
        for row in report["pairs"]
        if row["comparison_class"] == "cross_seed_same_objective"
    ]
    nominal = [row for row in pairs if row["label_permutation_p_value"] <= 0.05]
    assert len(nominal) == 1
    assert nominal[0]["label_permutation_p_value"] == 11 / 256
    assert nominal[0]["bonferroni_familywise_alpha"] == 0.05 / 6
    assert nominal[0]["same_minus_opposite"] < 0.003
    assert nominal[0]["diagnostic_category_subspace_stability_without_quality_gate"] == "fail"
    assert all(row["operative_category_subspace_replication_admission"] == "fail" for row in pairs)


def test_fixed_seed_cross_objective_subspaces_are_large_but_quality_rejected() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    pairs = [
        row
        for row in report["pairs"]
        if row["comparison_class"] == "same_seed_cross_objective"
    ]
    assert all(row["mean_same_category_subspace_similarity"] > 0.69 for row in pairs)
    assert all(row["same_minus_opposite"] > 0.67 for row in pairs)
    assert all(row["label_permutation_p_value"] == 1 / 256 for row in pairs)
    assert all(row["bonferroni_familywise_alpha"] == 0.05 / 3 for row in pairs)
    assert all(row["diagnostic_category_subspace_stability_without_quality_gate"] == "pass" for row in pairs)
    assert all(row["quality_qualified_pair"] is False for row in pairs)
    assert all(row["operative_category_subspace_replication_admission"] == "fail" for row in pairs)
    boundary = report["scientific_boundary"]
    assert boundary["individual_feature_matching_and_subspace_stability_are_distinct"] is True
    assert boundary["diagnostic_subspace_support_cannot_override_quality_gate"] is True
    assert boundary["does_not_test_layer11_or_final_norm"] is True


def test_layer21_category_subspace_report_rejects_tampering(tmp_path: Path) -> None:
    value = json.loads(REPORT.read_text(encoding="utf-8"))
    value["summary"]["cross_seed_diagnostic_subspace_supports_without_quality_gate"] = 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="content address differs"):
        verify_report(path)
