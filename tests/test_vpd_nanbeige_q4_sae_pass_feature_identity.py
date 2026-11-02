from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_feature_identity_v1 import (
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_nanbeige_q4_sae_pass_feature_identity_v1/report.json"


def test_cross_budget_feature_identity_report_reproduces() -> None:
    verification = verify_report(REPORT)
    assert verification["status"] == "pass"
    assert verification["site_checks"] == 3
    assert verification["quality_qualified_sites"] == [
        "model.layers.11",
        "model.norm",
    ]
    assert verification["provisional_same_seed_feature_identity_supports"] == 1
    assert verification["provisional_same_seed_feature_identity_sites"] == [
        "model.norm"
    ]
    assert verification["operative_cross_budget_feature_identity_admissions"] == 0
    assert verification["execution_authorized"] is False


def test_primary_identity_screen_has_expected_stable_feature_counts() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    expected = {
        "model.layers.11": ({"pass_0": 19, "pass_1": 10}, "fail"),
        "model.layers.21": ({"pass_0": 43, "pass_1": 19}, "fail"),
        "model.norm": ({"pass_0": 22, "pass_1": 28}, "pass"),
    }
    for site in report["sites"]:
        primary = next(
            row
            for row in site["threshold_sweep"]
            if row["minimum_same_index_decoder_cosine"] == 0.8
        )
        counts, provisional = expected[site["site"]]
        assert primary["stable_same_pass_features"] == counts
        assert primary["opposite_pass_features"] == 0
        assert primary["target_label_permutation_p_value"] == 1 / 4097
        assert site["provisional_same_seed_feature_identity_support"] == provisional
        assert site["operative_cross_budget_feature_identity_admission"] == "fail"


def test_threshold_sweep_is_complete_and_not_cherry_picked() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["design"]["cosine_threshold_sweep"] == [0.5, 0.7, 0.8, 0.9]
    assert report["design"]["primary_diagnostic_cosine_threshold"] == 0.8
    assert report["design"]["minimum_stable_features_per_pass"] == 16
    assert report["design"]["target_label_permutation_replicates"] == 4096
    assert report["design"]["threshold_and_same_seed_analysis_are_posthoc"] is True
    for site in report["sites"]:
        assert [
            row["minimum_same_index_decoder_cosine"]
            for row in site["threshold_sweep"]
        ] == [0.5, 0.7, 0.8, 0.9]


def test_feature_identity_boundary_is_non_authorizing() -> None:
    boundary = json.loads(REPORT.read_text(encoding="utf-8"))["scientific_boundary"]
    assert boundary["same_seed_and_training_ids_limit_independence"] is True
    assert boundary["cosine_threshold_sweep_is_posthoc"] is True
    assert boundary["decoder_alignment_is_not_feature_semantics"] is True
    assert boundary["quality_gate_applied_before_identity_support"] is True
    assert boundary["fresh_seed_and_family_disjoint_replication_required"] is True
    assert boundary["sae_decoder_weights_loaded_for_read_only_analysis"] is True
    assert boundary["model_weights_loaded"] is False
    assert boundary["weights_mutated"] is False
    assert boundary["new_sae_training_performed"] is False
    assert boundary["execution_authorized"] is False
