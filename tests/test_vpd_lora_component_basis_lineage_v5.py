from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "artifacts/vpd_lora_component_basis_identifiability_benchmark_v1/report.json"
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v13.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v5/runbook.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_v13_prohibits_raw_rank_inventory() -> None:
    protocol = _load(PROTOCOL)
    contract = protocol["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    assert contract["benchmark"]["sha256"] == sha256_file(BENCHMARK)
    assert contract["benchmark"]["raw_specific_component_count_change_under_equivalent_basis"] == -16
    assert contract["raw_lora_factor_rank_inventory_prohibited"] is True
    assert contract["near_degenerate_vectors_group_into_one_unresolved_block"] is True
    assert contract["production_canonical_effect_capture_integration_present"] is False


def test_runbook_v5_is_addressed_and_only_preflight_is_safe() -> None:
    runbook = _load(RUNBOOK)
    body = dict(runbook)
    assert body.pop("runbook_id") == content_sha256(body)
    assert runbook["source"]["research_protocol_v13"]["sha256"] == sha256_file(
        PROTOCOL
    )
    assert [
        stage["id"]
        for stage in runbook["stages"]
        if stage["safe_for_cheap_executor_now"]
    ] == ["local_preflight"]
    assert runbook["scientific_boundary"]["execution_authorized"] is False


def test_runbook_v5_stops_before_basis_dependent_capture() -> None:
    runbook = _load(RUNBOOK)
    gate = runbook["stages"][1]
    assert gate["id"] == "canonical_component_inventory_and_capture_runtime"
    assert gate["safe_for_cheap_executor_now"] is False
    assert any("raw LoRA factor rank" in reason for reason in gate["stop_if"])
    assert any("production canonical effect-capture" in reason for reason in gate["stop_if"])
    semantic = runbook["stages"][2]
    assert semantic["id"] == "cluster_candidate_semantic_strata_v3"
    assert "canonical component inventory" in semantic["additional_precondition"]
    assert runbook["completion_definition"]["paid_pipeline_complete_only_after_stage_9"]
