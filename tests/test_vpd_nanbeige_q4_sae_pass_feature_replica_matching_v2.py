from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_feature_replica_matching_v2 import (
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "artifacts/vpd_nanbeige_q4_sae_pass_feature_replica_matching_v2/report.json"
)


def test_v2_pass_feature_stratum_report_reproduces() -> None:
    verification = verify_report(REPORT)
    assert verification["status"] == "pass"
    assert verification["site_checks"] == 3
    assert verification["quality_qualified_sites"] == [
        "model.layers.11",
        "model.norm",
    ]
    assert verification["provisional_same_seed_v2_stratum_replication_supports"] == 1
    assert verification["provisional_same_seed_v2_stratum_replication_sites"] == [
        "model.norm"
    ]
    assert verification["operative_fresh_seed_stratum_replication_admissions"] == 0
    assert verification["execution_authorized"] is False


def test_primary_v2_screen_has_expected_mutual_matches_and_coverage() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    expected = {
        "model.layers.11": ({"pass_0": 19, "pass_1": 10}, "fail"),
        "model.layers.21": ({"pass_0": 43, "pass_1": 19}, "fail"),
        "model.norm": ({"pass_0": 25, "pass_1": 28}, "pass"),
    }
    for site in report["sites"]:
        primary = next(
            row
            for row in site["threshold_sweep"]
            if row["minimum_signed_decoder_cosine"] == 0.8
        )
        counts, provisional = expected[site["site"]]
        assert primary["same_category_matches"] == counts
        assert primary["profile_qualified_same_category_matches"] == counts
        assert primary["opposite_category_matches"] == 0
        assert primary["target_label_permutation_p_value"] == 1 / 4097
        assert site["provisional_same_seed_v2_stratum_replication_support"] == provisional
        assert site["operative_fresh_seed_stratum_replication_admission"] == "fail"
    norm = next(row for row in report["sites"] if row["site"] == "model.norm")
    primary_norm = next(
        row
        for row in norm["threshold_sweep"]
        if row["minimum_signed_decoder_cosine"] == 0.8
    )
    assert primary_norm["bidirectional_category_coverage"] == {
        "pass_0": 25 / 78,
        "pass_1": 28 / 236,
    }


def test_v2_empirical_design_does_not_overclaim_full_replication() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["matcher"]["signed_decoder_cosine"] is True
    assert report["matcher"]["mutual_nearest_neighbor"] is True
    assert report["matcher"]["one_to_one"] is True
    assert report["matcher"]["matching_pool"] == (
        "training_assigned_pass_selective_features_only"
    )
    assert report["design"]["minimum_bidirectional_category_coverage"] == 0.10
    assert report["design"]["thresholds_are_diagnostic_and_posthoc_for_existing_sources"] is True
    boundary = report["scientific_boundary"]
    assert boundary["full_dictionary_replication_not_claimed"] is True
    assert boundary["fresh_seed_family_disjoint_replication_required"] is True
    assert boundary["new_model_inference_performed"] is False
    assert boundary["new_sae_training_performed"] is False
    assert boundary["model_weights_loaded"] is False
    assert boundary["weights_mutated"] is False
    assert boundary["execution_authorized"] is False


def test_v2_stratum_report_rejects_tampering(tmp_path: Path) -> None:
    tampered = json.loads(REPORT.read_text(encoding="utf-8"))
    tampered["sites"][2]["threshold_sweep"][2]["same_category_matches"][
        "pass_0"
    ] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="content address differs"):
        verify_report(path)
