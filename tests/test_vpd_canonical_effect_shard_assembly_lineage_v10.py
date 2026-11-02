from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "artifacts/vpd_canonical_effect_shard_assembly_benchmark_v1/report.json"
PROTOCOL_V17 = ROOT / "configs/evals/vpd_activation_category_compartments_v17.json"
PROTOCOL_V18 = ROOT / "configs/evals/vpd_activation_category_compartments_v18.json"
RUNBOOK_V9 = ROOT / "artifacts/vpd_cheap_executor_runbook_v9/runbook.json"
RUNBOOK_V10 = ROOT / "artifacts/vpd_cheap_executor_runbook_v10/runbook.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_v18_binds_assembly_benchmark_and_pinned_v17() -> None:
    protocol = _load(PROTOCOL_V18)
    contract = protocol["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    evidence = contract["canonical_effect_shard_assembly_benchmark"]
    assert protocol["predecessor_sha256"] == sha256_file(PROTOCOL_V17)
    assert evidence["sha256"] == sha256_file(BENCHMARK)
    assert evidence["benchmark_id"] == _load(BENCHMARK)["benchmark_id"]
    assert evidence["components"] == 2
    assert evidence["negative_controls"] == 9


def test_protocol_v18_opens_assembler_but_keeps_model_loader_closed() -> None:
    protocol = _load(PROTOCOL_V18)
    contract = protocol["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    assert contract["model_free_successor_worker_archive_assembler_present"] is True
    assert contract["worker_batch_identity_binding_present"] is True
    assert contract["worker_condition_template_countsketch_present"] is True
    assert contract["successor_effect_worker_cli_integration_present"] is False
    assert contract["successor_worker_model_loader_present"] is False
    assert contract["actual_model_category_effect_capture_verified"] is False
    assert protocol["scientific_boundary"]["execution_authorized"] is False


def test_runbook_v10_is_addressed_and_only_preflight_is_safe() -> None:
    runbook = _load(RUNBOOK_V10)
    body = dict(runbook)
    assert body.pop("runbook_id") == content_sha256(body)
    assert runbook["predecessor"]["runbook_id"] == _load(RUNBOOK_V9)["runbook_id"]
    assert runbook["source"]["research_protocol_v18"]["sha256"] == sha256_file(
        PROTOCOL_V18
    )
    assert [
        stage["id"]
        for stage in runbook["stages"]
        if stage["safe_for_cheap_executor_now"]
    ] == ["local_preflight"]
    assert runbook["scientific_boundary"]["execution_authorized"] is False


def test_runbook_v10_requires_model_loader_and_real_capture_receipts() -> None:
    runbook = _load(RUNBOOK_V10)
    gate = runbook["stages"][1]
    reasons = gate["stop_if"]
    assert any("successor worker model loader" in reason for reason in reasons)
    assert any("canonical category capture kernel" in reason for reason in reasons)
    assert any("verified V2/V3 split provenance" in reason for reason in reasons)
    assert any("actual-Torch/PEFT" in reason for reason in reasons)
    assert runbook["completion_definition"]["model_free_successor_worker_assembler_ready"] is True
    assert runbook["completion_definition"]["worker_semantic_strata_sketches_ready"] is True
    assert runbook["completion_definition"]["successor_worker_model_loader_ready"] is False
    assert runbook["completion_definition"]["actual_model_category_capture_ready"] is False
