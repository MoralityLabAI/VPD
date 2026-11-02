from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_nanbeige_pass_compartment_rank_sweep_v1 import (
    TOTAL_RANKS,
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]
ARITHMETIC = (
    ROOT / "artifacts/vpd_nanbeige_arithmetic_pass_compartment_rank_sweep_v1/report.json"
)
REASONING = (
    ROOT / "artifacts/vpd_nanbeige_reasoning_pass_compartment_rank_sweep_v1/report.json"
)


def test_rank_sweep_reports_reproduce() -> None:
    for path in (ARITHMETIC, REASONING):
        verification = verify_report(path)
        assert verification["status"] == "pass"
        assert verification["sites"] == 3
        assert verification["ranks_per_site"] == 6
        assert verification["rank_site_checks"] == 18
        assert verification["pass_wins_vs_monolithic"] == 0
        assert verification["pass_compartment_screening_admissions"] == 0
        assert verification["execution_authorized"] is False


def test_monolithic_wins_35_checks_with_one_declared_inconclusive_check() -> None:
    arithmetic = json.loads(ARITHMETIC.read_text(encoding="utf-8"))
    reasoning = json.loads(REASONING.read_text(encoding="utf-8"))
    assert arithmetic["design"]["total_ranks"] == list(TOTAL_RANKS)
    assert reasoning["design"]["total_ranks"] == list(TOTAL_RANKS)
    assert arithmetic["summary"]["monolithic_wins_vs_pass"] == 17
    assert arithmetic["summary"]["monolithic_vs_pass_inconclusive"] == 1
    assert reasoning["summary"]["monolithic_wins_vs_pass"] == 18
    assert reasoning["summary"]["monolithic_vs_pass_inconclusive"] == 0
    inconclusive = [
        (site["site"], row["total_rank"])
        for site in arithmetic["sites"]
        for row in site["ranks"]
        if row["pass_minus_monolithic_paired_bootstrap_95_ci"][0]
        <= 0.0
        <= row["pass_minus_monolithic_paired_bootstrap_95_ci"][1]
    ]
    assert inconclusive == [("model.layers.21", 32)]
    assert all(
        row["screening_admission"] == "fail"
        for report in (arithmetic, reasoning)
        for site in report["sites"]
        for row in site["ranks"]
    )


def test_random_control_advantage_has_one_declared_low_rank_exception() -> None:
    arithmetic = json.loads(ARITHMETIC.read_text(encoding="utf-8"))
    reasoning = json.loads(REASONING.read_text(encoding="utf-8"))
    assert arithmetic["summary"]["pass_wins_vs_random_mean"] == 18
    assert reasoning["summary"]["pass_wins_vs_random_mean"] == 17
    exceptions = [
        (site["site"], row["total_rank"])
        for site in reasoning["sites"]
        for row in site["ranks"]
        if row["pass_minus_random_partition_mean_paired_bootstrap_95_ci"][0] <= 0.0
    ]
    assert exceptions == [("model.norm", 8)]


def test_rank_sweep_preserves_non_authorizing_boundary() -> None:
    for path in (ARITHMETIC, REASONING):
        boundary = json.loads(path.read_text(encoding="utf-8"))["scientific_boundary"]
        assert boundary["model_or_sae_weights_loaded_or_mutated"] is False
        assert boundary["new_model_inference_performed"] is False
        assert boundary["new_sae_training_performed"] is False
        assert boundary["linear_rank_sweep_is_not_an_sae_quality_result"] is True
        assert boundary["screening_result_cannot_authorize_training"] is True
