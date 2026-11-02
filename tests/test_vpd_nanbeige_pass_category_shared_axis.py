from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_nanbeige_pass_category_shared_axis_v1 import (
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_nanbeige_pass_category_shared_axis_v1/report.json"


def test_shared_axis_report_reproduces() -> None:
    verification = verify_report(REPORT)
    assert verification["status"] == "pass"
    assert verification["execution_authorized"] is False
    assert verification["sites"] == 3
    assert verification["shared_axis_admissions"] == 3
    assert verification["minimum_shared_axis_held_out_pass_accuracy"] == 1.0
    assert verification["maximum_exact_upper_tail_p_value"] == pytest.approx(
        0.01312255859375
    )


def test_layer11_has_shared_and_task_specific_components() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    sites = {row["site"]: row for row in report["sites"]}
    layer11 = sites["model.layers.11"]
    assert layer11["source_train_direction_cosine"] == pytest.approx(
        0.28849665639484445
    )
    assert layer11["shared_axis_cosine_to_pack_train_directions"][
        "minimum"
    ] == pytest.approx(0.8026508133661998)
    assert layer11["shared_axis_held_out"]["arithmetic"]["pass_accuracy"] == 1.0
    assert layer11["shared_axis_held_out"]["reasoning"]["pass_accuracy"] == 1.0
    assert layer11["exact_joint_pair_block_label_flip_control"][
        "joint_configurations"
    ] == 16_384
    assert layer11["exact_joint_pair_block_label_flip_control"][
        "exact_upper_tail_p_value"
    ] == pytest.approx(0.00048828125)


def test_every_site_clears_predeclared_shared_axis_gate() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert {
        row["site"]: row["shared_axis_admission"] for row in report["sites"]
    } == {
        "model.layers.11": "pass",
        "model.layers.21": "pass",
        "model.norm": "pass",
    }


def test_shared_axis_boundary_is_local_and_non_authorizing() -> None:
    boundary = json.loads(REPORT.read_text(encoding="utf-8"))["scientific_boundary"]
    assert boundary["new_model_inference_performed"] is False
    assert boundary["weights_loaded_or_mutated"] is False
    assert boundary["paid_compute_authorized"] is False
    assert boundary["pooled_training_uses_both_task_packs"] is True
    assert boundary["held_out_evaluation_does_not_establish_third_domain_transfer"] is True
    assert boundary["does_not_authorize_sae_training"] is True
    assert boundary["execution_authorized"] is False
