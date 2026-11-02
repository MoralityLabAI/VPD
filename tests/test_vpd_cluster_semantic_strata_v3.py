from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_activation_category_cluster_model_order_v2 import (
    verify_diagnostic,
)
from experiments.authz_mechinterp.benchmark_activation_category_cluster_semantic_strata_v3 import (
    verify_benchmark as verify_matrix_benchmark,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_candidate_semantic_strata_admission_v3 import (
    verify_benchmark as verify_production_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256


ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ROOT / "artifacts/vpd_activation_category_cluster_model_order_diagnostic_v2/report.json"
MATRIX_BENCHMARK = ROOT / "artifacts/vpd_activation_category_cluster_semantic_strata_benchmark_v3/report.json"
PRODUCTION_BENCHMARK = ROOT / "artifacts/vpd_cluster_candidate_semantic_strata_admission_benchmark_v3/report.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v2_recovers_known_two_three_and_four_mode_order() -> None:
    verification = verify_diagnostic(MODEL_ORDER)
    assert verification["status"] == "pass"
    assert verification["scenarios"] == 4
    assert verification["nonexact_scenarios"] == 0


def test_semantic_strata_reject_confounds_but_preserve_four_mechanisms() -> None:
    verification = verify_matrix_benchmark(MATRIX_BENCHMARK)
    assert verification["status"] == "pass"
    assert verification["scenarios"] == 3
    assert verification["confounds_rejected"] == 2
    rows = {row["scenario_id"]: row for row in _load(MATRIX_BENCHMARK)["scenarios"]}
    assert rows["four_mechanisms_stable_across_semantic_strata"][
        "semantic_stratified"
    ]["histogram"] == {"4": 8}


def test_production_shaped_v3_rejects_condition_confound_after_v2_passes() -> None:
    verification = verify_production_benchmark(
        PRODUCTION_BENCHMARK, project_root=ROOT
    )
    assert verification["status"] == "pass"
    report = _load(PRODUCTION_BENCHMARK)
    body = dict(report)
    assert body.pop("benchmark_id") == content_sha256(body)
    confound = next(
        row
        for row in report["scenarios"]
        if row["scenario_id"] == "condition_partition_confound_rejected"
    )
    assert confound["base_bundle_verification"]["status"] == "pass"
    assert confound["observed_v2_admitted"] is True
    assert confound["observed_v3_admitted"] is False
    assert confound["v3_mismatch_count"] == 8
    assert set(confound["v3_allocation"].values()) == {1}
    assert confound["v3_semantic_replicate_count"] == 40
