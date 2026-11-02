from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_compartment_sae_runtime_v2 import (
    canonical_inputs as benchmark_inputs,
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "artifacts/vpd_compartment_sae_training_plan_v4/plan.json"
GATE_PATH = (
    ROOT / "artifacts/vpd_compartment_sae_training_release_gate_v2/plan.json"
)
BENCHMARK_PATH = (
    ROOT / "artifacts/vpd_compartment_sae_runtime_benchmark_v2/report.json"
)
PROTOCOL_PATH = ROOT / "configs/evals/vpd_activation_category_compartments_v9.json"
RECEIPT_PATH = ROOT / "artifacts/vpd_unified_compartment_pipeline_v7/receipt.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_runtime_benchmark_v2_reproduces_all_scenarios() -> None:
    verification = verify_benchmark(BENCHMARK_PATH, **benchmark_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["scenarios"] == 11
    assert verification["execution_authorized"] is False


def test_research_protocol_v9_binds_exact_runtime_artifacts() -> None:
    plan = _load(PLAN_PATH)
    gate = _load(GATE_PATH)
    benchmark = _load(BENCHMARK_PATH)
    protocol = _load(PROTOCOL_PATH)
    runtime = protocol["sae_training"]["execution_runtime_v2"]
    assert runtime["training_plan"]["sha256"] == sha256_file(PLAN_PATH)
    assert runtime["training_plan"]["training_plan_id"] == plan["training_plan_id"]
    assert runtime["release_gate"]["sha256"] == sha256_file(GATE_PATH)
    assert runtime["release_gate"]["release_gate_plan_id"] == gate[
        "release_gate_plan_id"
    ]
    assert runtime["synthetic_benchmark"]["sha256"] == sha256_file(
        BENCHMARK_PATH
    )
    assert runtime["synthetic_benchmark"]["benchmark_id"] == benchmark[
        "benchmark_id"
    ]
    assert runtime["synthetic_benchmark"]["passed"] == 11
    assert runtime["guarded_jobs"] == 22
    assert runtime["execution_authorized"] is False


def test_unified_receipt_v7_is_content_addressed_and_nonempirical() -> None:
    receipt = _load(RECEIPT_PATH)
    body = dict(receipt)
    identifier = body.pop("pipeline_receipt_id")
    assert identifier == content_sha256(body)
    assert receipt["source"]["research_protocol_v9"]["sha256"] == sha256_file(
        PROTOCOL_PATH
    )
    assert receipt["source"]["runtime_benchmark_v2"]["sha256"] == sha256_file(
        BENCHMARK_PATH
    )
    assert receipt["resources"]["guarded_jobs"] == 22
    assert receipt["resources"]["runtime_benchmark_passed"] == 11
    assert receipt["scientific_boundary"]["model_inference_performed"] is False
    assert receipt["scientific_boundary"]["sae_training_performed"] is False
    assert receipt["scientific_boundary"]["execution_authorized"] is False
