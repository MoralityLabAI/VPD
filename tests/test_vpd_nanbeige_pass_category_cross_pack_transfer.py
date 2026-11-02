from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_nanbeige_pass_category_cross_pack_transfer_v1 import (
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_nanbeige_pass_category_cross_pack_transfer_v1/report.json"


def test_cross_pack_transfer_report_reproduces() -> None:
    verification = verify_report(REPORT)
    assert verification["status"] == "pass"
    assert verification["execution_authorized"] is False
    assert verification["sites"] == 3
    assert verification["bidirectional_cross_pack_category_admissions"] == 2
    assert verification["minimum_cross_pack_held_out_pass_accuracy"] == 1.0


def test_cross_pack_site_outcomes_are_exact_and_not_cherry_picked() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    sites = {row["site"]: row for row in report["sites"]}
    assert {
        site: row["bidirectional_cross_pack_category_admission"]
        for site, row in sites.items()
    } == {
        "model.layers.11": "fail",
        "model.layers.21": "pass",
        "model.norm": "pass",
    }
    assert sites["model.layers.11"]["cross_pack_train_direction_cosine"] == pytest.approx(
        0.28849665639484445
    )
    assert sites["model.layers.21"]["cross_pack_train_direction_cosine"] == pytest.approx(
        0.9921854242070439
    )
    assert sites["model.norm"]["cross_pack_train_direction_cosine"] == pytest.approx(
        0.8979968898827502
    )
    for row in sites.values():
        for direction in (
            "arithmetic_train_to_reasoning_held_out",
            "reasoning_train_to_arithmetic_held_out",
        ):
            assert row[direction]["held_out_pass_accuracy"] == 1.0
            assert row[direction]["directional_gate"] == "pass"


def test_sources_are_family_disjoint_but_revision_limited() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    sources = {row["label"]: row for row in report["sources"]}
    assert set(sources["arithmetic"]["families"]).isdisjoint(
        sources["reasoning"]["families"]
    )
    assert {row["expected_revision"] for row in sources.values()} == {
        "451ed48c3273ecef7ea8faaa43c31ce529763bb1"
    }
    assert all(row["revision_verified_by_source"] is False for row in sources.values())


def test_scientific_boundary_stays_non_authorizing() -> None:
    boundary = json.loads(REPORT.read_text(encoding="utf-8"))["scientific_boundary"]
    assert boundary["new_model_inference_performed"] is False
    assert boundary["weights_loaded_or_mutated"] is False
    assert boundary["paid_compute_authorized"] is False
    assert boundary["source_checkpoint_revision_independently_verified"] is False
    assert boundary["result_is_conditional_on_matching_claimed_checkpoint"] is True
    assert boundary["does_not_authorize_separate_sae_training"] is True
    assert boundary["execution_authorized"] is False
