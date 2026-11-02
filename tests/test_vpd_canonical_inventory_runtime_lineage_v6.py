from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_BENCHMARK = (
    ROOT / "artifacts/vpd_canonical_lora_component_inventory_benchmark_v1/report.json"
)
RUNTIME_BENCHMARK = (
    ROOT / "artifacts/vpd_canonical_lora_factor_runtime_benchmark_v1/report.json"
)
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v14.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v6/runbook.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_v14_binds_canonical_inventory_and_runtime_receipts() -> None:
    protocol = _load(PROTOCOL)
    contract = protocol["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    assert contract["canonical_inventory_benchmark"]["sha256"] == sha256_file(
        INVENTORY_BENCHMARK
    )
    assert contract["canonical_factor_runtime_benchmark"]["sha256"] == sha256_file(
        RUNTIME_BENCHMARK
    )
    assert contract["canonical_inventory_benchmark"]["benchmark_id"] == _load(
        INVENTORY_BENCHMARK
    )["benchmark_id"]
    assert contract["canonical_factor_runtime_benchmark"]["benchmark_id"] == _load(
        RUNTIME_BENCHMARK
    )["benchmark_id"]


def test_protocol_v14_keeps_production_integration_boundary_explicit() -> None:
    contract = _load(PROTOCOL)["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    assert contract["factor_bundle_canonical_inventory_emitter_present"] is True
    assert contract["model_free_restoration_runtime_present"] is True
    assert contract["production_adapter_inventory_emitter_present"] is False
    assert contract["production_canonical_effect_capture_integration_present"] is False
    assert contract["canonical_factor_runtime"][
        "production_torch_peft_integration_present"
    ] is False


def test_runbook_v6_is_addressed_and_only_preflight_is_safe() -> None:
    runbook = _load(RUNBOOK)
    body = dict(runbook)
    assert body.pop("runbook_id") == content_sha256(body)
    assert runbook["source"]["research_protocol_v14"]["sha256"] == sha256_file(
        PROTOCOL
    )
    assert [
        stage["id"]
        for stage in runbook["stages"]
        if stage["safe_for_cheap_executor_now"]
    ] == ["local_preflight"]
    assert runbook["scientific_boundary"]["execution_authorized"] is False


def test_runbook_v6_stops_before_real_adapter_or_torch_capture() -> None:
    runbook = _load(RUNBOOK)
    gate = runbook["stages"][1]
    assert gate["id"] == "canonical_component_inventory_and_capture_runtime"
    assert gate["safe_for_cheap_executor_now"] is False
    assert any(
        "adapter-to-factor-bundle extraction is absent" in reason
        for reason in gate["stop_if"]
    )
    assert any(
        "Torch/PEFT canonical effect-capture integration is absent" in reason
        for reason in gate["stop_if"]
    )
    assert runbook["completion_definition"][
        "canonical_factor_bundle_inventory_runtime_ready"
    ] is True
    assert runbook["completion_definition"][
        "canonical_torch_peft_capture_runtime_ready"
    ] is False
