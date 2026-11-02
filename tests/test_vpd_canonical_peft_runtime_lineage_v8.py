from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "artifacts/vpd_canonical_lora_peft_runtime_benchmark_v1/report.json"
PROTOCOL_V15 = ROOT / "configs/evals/vpd_activation_category_compartments_v15.json"
PROTOCOL_V16 = ROOT / "configs/evals/vpd_activation_category_compartments_v16.json"
RUNBOOK_V7 = ROOT / "artifacts/vpd_cheap_executor_runbook_v7/runbook.json"
RUNBOOK_V8 = ROOT / "artifacts/vpd_cheap_executor_runbook_v8/runbook.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_v16_binds_fake_peft_runtime_and_pinned_v15() -> None:
    protocol = _load(PROTOCOL_V16)
    contract = protocol["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    evidence = contract["canonical_peft_factor_runtime_benchmark"]
    assert protocol["predecessor_sha256"] == sha256_file(PROTOCOL_V15)
    assert evidence["sha256"] == sha256_file(BENCHMARK)
    assert evidence["benchmark_id"] == _load(BENCHMARK)["benchmark_id"]
    assert evidence["multipliers"] == [-1.0, 0.0, 1.0, 2.0]
    assert evidence["negative_controls"] == 5
    assert evidence["fake_peft_objects_only"] is True


def test_protocol_v16_keeps_actual_torch_and_worker_integration_closed() -> None:
    protocol = _load(PROTOCOL_V16)
    contract = protocol["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    assert contract["production_shaped_peft_factor_runtime_implementation_present"] is True
    assert contract["actual_torch_peft_runtime_execution_verified"] is False
    assert contract["effect_capture_worker_canonical_runtime_integration_present"] is False
    assert contract["production_canonical_effect_capture_integration_present"] is False
    assert protocol["scientific_boundary"]["torch_or_peft_loaded"] is False
    assert protocol["scientific_boundary"]["execution_authorized"] is False


def test_runbook_v8_is_addressed_and_only_preflight_is_safe() -> None:
    runbook = _load(RUNBOOK_V8)
    body = dict(runbook)
    assert body.pop("runbook_id") == content_sha256(body)
    assert runbook["predecessor"]["runbook_id"] == _load(RUNBOOK_V7)["runbook_id"]
    assert runbook["source"]["research_protocol_v16"]["sha256"] == sha256_file(
        PROTOCOL_V16
    )
    assert [
        stage["id"]
        for stage in runbook["stages"]
        if stage["safe_for_cheap_executor_now"]
    ] == ["local_preflight"]
    assert runbook["scientific_boundary"]["execution_authorized"] is False


def test_runbook_v8_requires_real_torch_and_worker_receipts() -> None:
    runbook = _load(RUNBOOK_V8)
    gate = runbook["stages"][1]
    reasons = gate["stop_if"]
    assert any("explicit user authorization" in reason for reason in reasons)
    assert any("real-adapter factor-bundle extraction receipt" in reason for reason in reasons)
    assert any("actual-Torch/PEFT canonical runtime smoke receipt" in reason for reason in reasons)
    assert any("successor effect-capture worker" in reason for reason in reasons)
    assert runbook["completion_definition"]["canonical_peft_bridge_implementation_ready"] is True
    assert runbook["completion_definition"]["actual_torch_peft_runtime_smoke_ready"] is False
    assert runbook["completion_definition"]["canonical_effect_worker_integration_ready"] is False
