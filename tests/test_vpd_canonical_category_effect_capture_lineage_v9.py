from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "artifacts/vpd_canonical_category_effect_capture_benchmark_v1/report.json"
PROTOCOL_V16 = ROOT / "configs/evals/vpd_activation_category_compartments_v16.json"
PROTOCOL_V17 = ROOT / "configs/evals/vpd_activation_category_compartments_v17.json"
RUNBOOK_V8 = ROOT / "artifacts/vpd_cheap_executor_runbook_v8/runbook.json"
RUNBOOK_V9 = ROOT / "artifacts/vpd_cheap_executor_runbook_v9/runbook.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_v17_binds_category_capture_and_pinned_v16() -> None:
    protocol = _load(PROTOCOL_V17)
    contract = protocol["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    evidence = contract["canonical_category_effect_capture_benchmark"]
    assert protocol["predecessor_sha256"] == sha256_file(PROTOCOL_V16)
    assert evidence["sha256"] == sha256_file(BENCHMARK)
    assert evidence["benchmark_id"] == _load(BENCHMARK)["benchmark_id"]
    assert evidence["categories"] == ["invalid_authority", "valid_authority"]
    assert evidence["negative_controls"] == 5


def test_protocol_v17_keeps_worker_and_actual_capture_closed() -> None:
    protocol = _load(PROTOCOL_V17)
    contract = protocol["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    assert contract["category_preserving_canonical_effect_capture_kernel_present"] is True
    assert contract["actual_model_category_effect_capture_verified"] is False
    assert contract["successor_effect_worker_cli_integration_present"] is False
    assert contract["worker_batch_identity_binding_present"] is False
    assert contract["worker_condition_template_countsketch_present"] is False
    assert protocol["scientific_boundary"]["execution_authorized"] is False


def test_runbook_v9_is_addressed_and_only_preflight_is_safe() -> None:
    runbook = _load(RUNBOOK_V9)
    body = dict(runbook)
    assert body.pop("runbook_id") == content_sha256(body)
    assert runbook["predecessor"]["runbook_id"] == _load(RUNBOOK_V8)["runbook_id"]
    assert runbook["source"]["research_protocol_v17"]["sha256"] == sha256_file(
        PROTOCOL_V17
    )
    assert [
        stage["id"]
        for stage in runbook["stages"]
        if stage["safe_for_cheap_executor_now"]
    ] == ["local_preflight"]
    assert runbook["scientific_boundary"]["execution_authorized"] is False


def test_runbook_v9_requires_worker_provenance_and_semantic_sketches() -> None:
    runbook = _load(RUNBOOK_V9)
    gate = runbook["stages"][1]
    reasons = gate["stop_if"]
    assert any("successor canonical effect-worker CLI" in reason for reason in reasons)
    assert any("exact prompt order" in reason for reason in reasons)
    assert any("condition and template CountSketch" in reason for reason in reasons)
    assert any("actual-Torch/PEFT" in reason for reason in reasons)
    assert runbook["completion_definition"]["canonical_category_capture_kernel_ready"] is True
    assert runbook["completion_definition"]["successor_effect_worker_cli_ready"] is False
    assert runbook["completion_definition"]["worker_semantic_strata_sketches_ready"] is False
    assert runbook["completion_definition"]["actual_model_category_capture_ready"] is False
