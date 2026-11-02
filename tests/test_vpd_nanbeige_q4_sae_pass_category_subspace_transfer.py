from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_category_subspace_transfer_v1 import (
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "artifacts/vpd_nanbeige_q4_sae_pass_category_subspace_transfer_v1/report.json"
)


def test_q4_pass_category_subspace_transfer_report_reproduces() -> None:
    verification = verify_report(REPORT)
    assert verification["status"] == "pass"
    assert verification["site_checks"] == 3
    assert verification["quality_qualified_sites"] == [
        "model.layers.11",
        "model.norm",
    ]
    assert verification[
        "diagnostic_same_seed_category_subspace_transfer_supports"
    ] == 3
    assert verification[
        "provisional_same_seed_quality_qualified_subspace_supports"
    ] == 2
    assert verification[
        "provisional_same_seed_quality_qualified_subspace_sites"
    ] == ["model.layers.11", "model.norm"]
    assert verification[
        "operative_fresh_seed_category_subspace_replication_admissions"
    ] == 0
    assert verification["execution_authorized"] is False


def test_primary_rank_subspace_effects_are_site_corrected_and_positive() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    expected_effects = {
        "model.layers.11": 0.2371109596382418,
        "model.layers.21": 0.11059298862855133,
        "model.norm": 0.24223160501078123,
    }
    for site in report["sites"]:
        primary = next(
            row
            for row in site["rank_sensitivity"]
            if row["subspace_rank_per_category"] == 16
        )
        assert primary["same_minus_opposite"] == pytest.approx(
            expected_effects[site["site"]]
        )
        assert primary["label_permutation_p_value"] == 1 / 256
        assert primary["label_permutation_p_value"] <= site[
            "bonferroni_site_familywise_alpha"
        ]
        assert site["diagnostic_same_seed_category_subspace_transfer_support"] == (
            "pass"
        )
        assert site["all_reported_ranks_same_minus_opposite_positive"] is True
        assert site[
            "operative_fresh_seed_category_subspace_replication_admission"
        ] == "fail"


def test_subspace_transfer_report_preserves_non_authorizing_boundaries() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["design"]["primary_subspace_rank_per_category"] == 16
    assert report["design"]["reported_rank_sensitivity"] == [4, 8, 16, 32]
    assert report["design"][
        "sensitivity_ranks_do_not_supply_alternate_admission_tests"
    ] is True
    boundary = report["scientific_boundary"]
    assert boundary["same_seed_and_example_ids_limit_independence"] is True
    assert boundary[
        "primary_rank_and_site_multiplicity_control_are_weight_bearing"
    ] is True
    assert boundary["rank_sensitivity_is_not_an_admission_search"] is True
    assert boundary["subspace_similarity_is_not_feature_identity"] is True
    assert boundary["fresh_seed_family_disjoint_replication_required"] is True
    assert boundary["new_model_inference_performed"] is False
    assert boundary["new_sae_training_performed"] is False
    assert boundary["model_weights_loaded"] is False
    assert boundary["weights_mutated"] is False
    assert boundary["execution_authorized"] is False
    assert boundary["paid_compute_authorized"] is False


def test_q4_pass_category_subspace_transfer_report_rejects_tampering(
    tmp_path: Path,
) -> None:
    tampered = json.loads(REPORT.read_text(encoding="utf-8"))
    tampered["sites"][0]["rank_sensitivity"][2]["same_minus_opposite"] += 0.01
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="content address differs"):
        verify_report(path)
