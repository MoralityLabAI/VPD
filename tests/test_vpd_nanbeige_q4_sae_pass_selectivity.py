from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_selectivity_v1 import (
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = (
    ROOT / "artifacts/vpd_nanbeige_q4_sae_pass_selectivity_2048_v1/report.json",
    ROOT / "artifacts/vpd_nanbeige_q4_sae_pass_selectivity_8192_v1/report.json",
)


def test_q4_sae_pass_selectivity_reports_reproduce() -> None:
    for path in REPORTS:
        verification = verify_report(path)
        assert verification["status"] == "pass"
        assert verification["sites"] == 3
        assert verification["admissions"] == 2
        assert verification["execution_authorized"] is False


def test_shared_sae_internal_pass_specialization_replicates_at_two_sites() -> None:
    for path in REPORTS:
        report = json.loads(path.read_text(encoding="utf-8"))
        decisions = {
            row["site"]: row["monolithic_sae_internal_pass_specialization_admission"]
            for row in report["sites"]
        }
        assert decisions == {
            "model.layers.11": "pass",
            "model.layers.21": "fail",
            "model.norm": "pass",
        }
        assert all(row["exact_upper_tail_p_value"] == 1 / 256 for row in report["sites"])


def test_layer21_is_rejected_by_reconstruction_quality_not_selectivity() -> None:
    for path in REPORTS:
        report = json.loads(path.read_text(encoding="utf-8"))
        layer21 = next(row for row in report["sites"] if row["site"] == "model.layers.21")
        assert layer21["held_out_explained_variance"]["pass_1"] < 0.20
        assert layer21["pass_0"]["held_out_mass_precision"] > 0.99
        assert layer21["pass_1"]["held_out_mass_precision"] > 0.99
        assert layer21["pass_0"]["held_out_mass_coverage"] > 0.94
        assert layer21["pass_1"]["held_out_mass_coverage"] > 0.94


def test_selectivity_analysis_uses_frozen_archives_without_weight_load() -> None:
    for path in REPORTS:
        boundary = json.loads(path.read_text(encoding="utf-8"))["scientific_boundary"]
        assert boundary["stored_sparse_feature_archives_used"] is True
        assert boundary["sae_weights_loaded"] is False
        assert boundary["model_weights_loaded"] is False
        assert boundary["new_model_inference_performed"] is False
        assert boundary["new_sae_training_performed"] is False
        assert boundary["feature_assignment_is_not_semantic_interpretation"] is True
        assert boundary["does_not_authorize_separate_dictionary_training"] is True
