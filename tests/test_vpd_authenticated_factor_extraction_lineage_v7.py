from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = (
    ROOT
    / "artifacts/vpd_authenticated_lora_factor_bundle_extraction_benchmark_v1/report.json"
)
PROTOCOL_V14 = ROOT / "configs/evals/vpd_activation_category_compartments_v14.json"
PROTOCOL_V15 = ROOT / "configs/evals/vpd_activation_category_compartments_v15.json"
RUNBOOK_V6 = ROOT / "artifacts/vpd_cheap_executor_runbook_v6/runbook.json"
RUNBOOK_V7 = ROOT / "artifacts/vpd_cheap_executor_runbook_v7/runbook.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_v15_binds_extractor_benchmark_and_pinned_v14() -> None:
    protocol = _load(PROTOCOL_V15)
    contract = protocol["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    benchmark = _load(BENCHMARK)
    evidence = contract["authenticated_factor_bundle_extraction_benchmark"]
    assert protocol["predecessor_sha256"] == sha256_file(PROTOCOL_V14)
    assert evidence["sha256"] == sha256_file(BENCHMARK)
    assert evidence["benchmark_id"] == benchmark["benchmark_id"]
    assert evidence["negative_controls"] == 7
    assert evidence["synthetic_fixture_only"] is True


def test_protocol_v15_keeps_real_extraction_and_torch_receipts_closed() -> None:
    protocol = _load(PROTOCOL_V15)
    contract = protocol["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    assert contract["authenticated_adapter_factor_extractor_implementation_present"] is True
    assert contract["production_adapter_inventory_emitter_present"] is True
    assert contract["real_adapter_factor_bundle_extraction_receipt_present"] is False
    assert contract["production_canonical_effect_capture_integration_present"] is False
    assert protocol["scientific_boundary"]["real_adapter_accessed"] is False
    assert protocol["scientific_boundary"]["execution_authorized"] is False


def test_runbook_v7_is_addressed_and_only_preflight_is_safe() -> None:
    runbook = _load(RUNBOOK_V7)
    body = dict(runbook)
    assert body.pop("runbook_id") == content_sha256(body)
    assert runbook["predecessor"]["runbook_id"] == _load(RUNBOOK_V6)["runbook_id"]
    assert runbook["source"]["research_protocol_v15"]["sha256"] == sha256_file(
        PROTOCOL_V15
    )
    assert [
        stage["id"]
        for stage in runbook["stages"]
        if stage["safe_for_cheap_executor_now"]
    ] == ["local_preflight"]
    assert runbook["scientific_boundary"]["execution_authorized"] is False


def test_runbook_v7_requires_authority_real_receipt_and_torch_integration() -> None:
    runbook = _load(RUNBOOK_V7)
    gate = runbook["stages"][1]
    assert gate["id"] == "canonical_component_inventory_and_capture_runtime"
    assert gate["safe_for_cheap_executor_now"] is False
    reasons = gate["stop_if"]
    assert any("explicit user authorization" in reason for reason in reasons)
    assert any("real-adapter factor-bundle extraction receipt" in reason for reason in reasons)
    assert any("Torch/PEFT canonical effect-capture" in reason for reason in reasons)
    command = gate["available_authenticated_extraction_command_template"]["arguments"]
    assert "--synthetic-benchmark-only" not in command
    assert "<EXACT_EXPECTED_ADAPTER_SHA256>" in command
    assert "<EXACT_EXPECTED_CONFIG_SHA256>" in command
    assert runbook["completion_definition"][
        "authenticated_factor_extractor_implementation_ready"
    ] is True
    assert runbook["completion_definition"]["real_adapter_factor_bundle_receipt_ready"] is False
    assert runbook["completion_definition"]["canonical_torch_peft_capture_runtime_ready"] is False
