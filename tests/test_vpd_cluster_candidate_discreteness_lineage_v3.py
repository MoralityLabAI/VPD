from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "artifacts/vpd_cluster_candidate_discreteness_admission_benchmark_v2/report.json"
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v11.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v3/runbook.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_v11_binds_production_shaped_admission() -> None:
    benchmark = _load(BENCHMARK)
    protocol = _load(PROTOCOL)
    contract = protocol["component_compartments"][
        "cluster_candidate_discreteness_admission_v2"
    ]
    assert contract["benchmark"]["benchmark_id"] == benchmark["benchmark_id"]
    assert contract["benchmark"]["sha256"] == sha256_file(BENCHMARK)
    assert contract["base_v1_bundle_must_verify_first"] is True
    assert contract["exact_v1_v2_allocation_agreement_required"] is True
    assert contract["candidate_mutation_or_relabelling_prohibited"] is True
    assert contract["empirical_admission_receipt_present"] is False


def test_runbook_v3_is_addressed_and_only_stage_zero_is_safe_now() -> None:
    runbook = _load(RUNBOOK)
    body = dict(runbook)
    identifier = body.pop("runbook_id")
    assert identifier == content_sha256(body)
    assert runbook["source"]["research_protocol_v11"]["sha256"] == sha256_file(
        PROTOCOL
    )
    assert runbook["source"]["cluster_candidate_admission_benchmark_v2"][
        "benchmark_id"
    ] == _load(BENCHMARK)["benchmark_id"]
    assert [
        stage["id"]
        for stage in runbook["stages"]
        if stage["safe_for_cheap_executor_now"]
    ] == ["local_preflight"]


def test_runbook_v3_has_exact_admission_command_and_hard_stops() -> None:
    stage = _load(RUNBOOK)["stages"][1]
    assert stage["implementation_status"].startswith("independent V2 admission implemented")
    arguments = stage["command_template"]["arguments"]
    assert "experiments.authz_mechinterp.admit_vpd_cluster_candidate_discreteness_v2" in arguments
    assert "<CLUSTER_VALIDATION_EFFECTS>" in arguments
    assert stage["expected"]["mismatch_count"] == 0
    assert any("allocation differs" in condition for condition in stage["stop_if"])
    assert stage["safe_for_cheap_executor_now"] is False
