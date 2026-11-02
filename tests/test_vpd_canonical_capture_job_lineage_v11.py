from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_BENCHMARK = ROOT / "artifacts/vpd_canonical_capture_job_plan_benchmark_v1/report.json"
RUNTIME_BENCHMARK = ROOT / "artifacts/vpd_canonical_capture_job_runtime_benchmark_v1/report.json"
PROTOCOL_V18 = ROOT / "configs/evals/vpd_activation_category_compartments_v18.json"
PROTOCOL_V19 = ROOT / "configs/evals/vpd_activation_category_compartments_v19.json"
RUNBOOK_V10 = ROOT / "artifacts/vpd_cheap_executor_runbook_v10/runbook.json"
RUNBOOK_V11 = ROOT / "artifacts/vpd_cheap_executor_runbook_v11/runbook.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_v19_binds_both_capture_job_benchmarks_and_pinned_v18() -> None:
    protocol = _load(PROTOCOL_V19)
    contract = protocol["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    assert protocol["predecessor_sha256"] == sha256_file(PROTOCOL_V18)
    assert contract["canonical_capture_job_plan_benchmark"]["sha256"] == sha256_file(
        PLAN_BENCHMARK
    )
    assert contract["canonical_capture_job_runtime_benchmark"]["sha256"] == sha256_file(
        RUNTIME_BENCHMARK
    )
    assert contract["canonical_capture_job_plan_benchmark"]["negative_controls"] == 9
    assert contract["canonical_capture_job_runtime_benchmark"]["negative_controls"] == 8


def test_protocol_v19_opens_implementation_but_not_actual_capture() -> None:
    protocol = _load(PROTOCOL_V19)
    contract = protocol["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    assert contract["successor_effect_worker_cli_integration_present"] is True
    assert contract["successor_worker_model_loader_implementation_present"] is True
    assert contract["production_canonical_effect_capture_integration_present"] is True
    assert contract["actual_torch_peft_runtime_execution_verified"] is False
    assert contract["real_adapter_factor_bundle_receipt_present"] is False
    assert contract["actual_model_category_effect_capture_verified"] is False
    assert protocol["scientific_boundary"]["execution_authorized"] is False


def test_runbook_v11_is_addressed_and_only_preflight_is_safe() -> None:
    runbook = _load(RUNBOOK_V11)
    body = dict(runbook)
    assert body.pop("runbook_id") == content_sha256(body)
    assert runbook["predecessor"]["runbook_id"] == _load(RUNBOOK_V10)["runbook_id"]
    assert runbook["source"]["research_protocol_v19"]["sha256"] == sha256_file(
        PROTOCOL_V19
    )
    assert [
        stage["id"]
        for stage in runbook["stages"]
        if stage["safe_for_cheap_executor_now"]
    ] == ["local_preflight"]
    assert runbook["scientific_boundary"]["execution_authorized"] is False


def test_runbook_v11_distinguishes_implementation_from_empirical_readiness() -> None:
    runbook = _load(RUNBOOK_V11)
    completion = runbook["completion_definition"]
    assert completion["successor_effect_worker_cli_ready"] is True
    assert completion["successor_worker_model_loader_implementation_ready"] is True
    assert completion["exact_capture_authorization_verifier_ready"] is True
    assert completion["successor_worker_model_loader_ready"] is False
    assert completion["actual_torch_peft_runtime_smoke_ready"] is False
    assert completion["actual_model_category_capture_ready"] is False
    reasons = runbook["stages"][1]["stop_if"]
    assert any("real-adapter" in reason for reason in reasons)
    assert any("real-model identity" in reason for reason in reasons)
    assert any("user authorization receipt" in reason for reason in reasons)
    assert any("actual-Torch/PEFT" in reason for reason in reasons)
