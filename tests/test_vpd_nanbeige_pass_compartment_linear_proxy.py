from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_nanbeige_pass_compartment_linear_proxy_v1 import (
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = (
    ROOT / "artifacts/vpd_nanbeige_arithmetic_pass_compartment_linear_proxy_v1/report.json",
    ROOT / "artifacts/vpd_nanbeige_pass_compartment_linear_proxy_v1/report.json",
)


def test_linear_proxy_reports_reproduce_without_execution_authority() -> None:
    for path in REPORTS:
        verification = verify_report(path)
        assert verification["status"] == "pass"
        assert verification["sites"] == 3
        assert verification["screening_admissions"] == 0
        assert verification["execution_authorized"] is False


def test_pass_routing_beats_random_but_not_monolithic_at_every_site() -> None:
    for path in REPORTS:
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["source"]["revision_verified"] is False
        assert report["source"]["family_disjoint"] is False
        assert report["design"]["total_rank_per_arm"] == 64
        assert report["design"]["monolithic_rank"] == 64
        assert report["design"]["pass_compartment_rank_per_dictionary"] == 32
        assert report["design"]["random_partition_rank_per_dictionary"] == 32
        assert all(
            row["pass_minus_monolithic"] < 0.0
            and row["pass_minus_monolithic_paired_bootstrap_95_ci"][1] < 0.0
            and row["pass_minus_random_partition_mean"] > 0.0
            and row["pass_minus_random_partition_mean_paired_bootstrap_95_ci"][0]
            > 0.0
            and row["screening_admission"] == "fail"
            for row in report["sites"]
        )


def test_proxy_is_explicitly_screening_only_and_does_not_load_sae_weights() -> None:
    for path in REPORTS:
        boundary = json.loads(path.read_text(encoding="utf-8"))["scientific_boundary"]
        assert boundary["stored_empirical_bf16_activations_used"] is True
        assert boundary["model_or_sae_weights_loaded_or_mutated"] is False
        assert boundary["new_model_inference_performed"] is False
        assert boundary["new_sae_training_performed"] is False
        assert boundary["linear_proxy_is_not_an_sae_quality_result"] is True
        assert boundary["screening_result_cannot_authorize_training"] is True
