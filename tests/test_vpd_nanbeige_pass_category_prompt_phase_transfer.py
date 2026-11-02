from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_nanbeige_pass_category_prompt_phase_transfer_v1 import (
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_nanbeige_pass_category_prompt_phase_transfer_v1/report.json"


def test_prompt_phase_transfer_report_reproduces() -> None:
    verification = verify_report(REPORT)
    assert verification["status"] == "pass"
    assert verification["execution_authorized"] is False
    assert verification["sites"] == 3
    assert verification["prompt_phase_transfer_admissions"] == 1
    assert verification["minimum_prompt_eval_pass_accuracy"] == 1.0


def test_only_final_norm_transfers_across_activation_phase() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    sites = {row["site"]: row for row in report["sites"]}
    assert {
        site: row["prompt_phase_transfer_admission"]
        for site, row in sites.items()
    } == {
        "model.layers.11": "fail",
        "model.layers.21": "fail",
        "model.norm": "pass",
    }
    expected = {
        "model.layers.11": (0.04767526301390991, 0.13690185546875),
        "model.layers.21": (0.7966895787427319, 0.05303955078125),
        "model.norm": (0.9785101628277528, 0.0008544921875),
    }
    for site, (cosine, p_value) in expected.items():
        assert sites[site]["shared_axis_prompt_eval_pass_accuracy"] == 1.0
        assert sites[site]["shared_axis_to_prompt_eval_transition_cosine"] == pytest.approx(
            cosine
        )
        assert sites[site]["exact_joint_training_pair_flip_control"][
            "exact_upper_tail_p_value"
        ] == pytest.approx(p_value)


def test_primary_prompt_examples_are_held_out_from_axis_training() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["design"][
        "primary_target_examples_overlap_axis_training_examples"
    ] is False
    assert all(row["prompt_eval_examples"] == 4 for row in report["sites"])


def test_prompt_phase_boundary_is_non_authorizing() -> None:
    boundary = json.loads(REPORT.read_text(encoding="utf-8"))["scientific_boundary"]
    assert boundary["new_model_inference_performed"] is False
    assert boundary["weights_loaded_or_mutated"] is False
    assert boundary["prompt_phase_is_not_a_third_semantic_domain"] is True
    assert boundary["does_not_establish_third_domain_transfer"] is True
    assert boundary["does_not_authorize_sae_training"] is True
    assert boundary["execution_authorized"] is False
