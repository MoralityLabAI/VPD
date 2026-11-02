from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ROOT / "artifacts/vpd_activation_category_cluster_model_order_diagnostic_v2/report.json"
MATRIX_BENCHMARK = ROOT / "artifacts/vpd_activation_category_cluster_semantic_strata_benchmark_v3/report.json"
PRODUCTION_BENCHMARK = ROOT / "artifacts/vpd_cluster_candidate_semantic_strata_admission_benchmark_v3/report.json"
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v12.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v4/runbook.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_v12_binds_all_semantic_strata_evidence() -> None:
    protocol = _load(PROTOCOL)
    contract = protocol["component_compartments"][
        "cluster_candidate_semantic_strata_admission_v3"
    ]
    assert contract["model_order_diagnostic"]["sha256"] == sha256_file(MODEL_ORDER)
    assert contract["semantic_matrix_benchmark"]["sha256"] == sha256_file(
        MATRIX_BENCHMARK
    )
    assert contract["production_shaped_benchmark"]["sha256"] == sha256_file(
        PRODUCTION_BENCHMARK
    )
    assert contract["v2_only_admission_for_next_empirical_candidate_prohibited"]
    assert contract["empirical_admission_receipt_present"] is False


def test_runbook_v4_is_addressed_and_only_preflight_is_safe() -> None:
    runbook = _load(RUNBOOK)
    body = dict(runbook)
    assert body.pop("runbook_id") == content_sha256(body)
    assert runbook["source"]["research_protocol_v12"]["sha256"] == sha256_file(
        PROTOCOL
    )
    assert [
        stage["id"]
        for stage in runbook["stages"]
        if stage["safe_for_cheap_executor_now"]
    ] == ["local_preflight"]
    assert runbook["scientific_boundary"]["execution_authorized"] is False


def test_runbook_v4_requires_semantic_strata_v3_not_v2_only() -> None:
    runbook = _load(RUNBOOK)
    stage = runbook["stages"][1]
    assert stage["id"] == "cluster_candidate_semantic_strata_v3"
    assert stage["command_template"]["arguments"][1] == (
        "experiments.authz_mechinterp.admit_vpd_cluster_candidate_semantic_strata_v3"
    )
    assert stage["expected"]["admission_status"] == (
        "pass_candidate_semantic_strata_v3"
    )
    assert any("condition or template" in reason for reason in stage["stop_if"])
    assert stage["safe_for_cheap_executor_now"] is False
