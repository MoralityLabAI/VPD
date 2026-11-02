from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_canonical_lora_component_inventory_v1 import (
    verify_benchmark as verify_inventory_benchmark,
)
from experiments.authz_mechinterp.benchmark_canonical_lora_factor_runtime_v1 import (
    verify_benchmark as verify_runtime_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_REPORT = ROOT / "artifacts/vpd_canonical_lora_component_inventory_benchmark_v1/report.json"
RUNTIME_REPORT = ROOT / "artifacts/vpd_canonical_lora_factor_runtime_benchmark_v1/report.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_canonical_inventory_benchmark_reproduces_and_rejects_tamper() -> None:
    verification = verify_inventory_benchmark(INVENTORY_REPORT)
    assert verification["status"] == "pass"
    assert verification["canonical_components"] == 5
    assert verification["unresolved_degenerate_blocks"] == 1
    assert verification["tamper_rejected"] is True


def test_canonical_inventory_is_basis_invariant_at_semantic_tolerance() -> None:
    report = _load(INVENTORY_REPORT)
    body = dict(report)
    assert body.pop("benchmark_id") == content_sha256(body)
    results = report["results"]
    assert results["semantic_inventory_rows_equal"] is True
    assert results["maximum_canonical_factor_frobenius_difference"] < 1e-10
    assert results["maximum_canonical_reconstruction_frobenius_difference"] < 1e-10
    assert results["maximum_raw_component_frobenius_difference"] > 0.5


def test_canonical_factor_runtime_reproduces_and_restores_failures() -> None:
    verification = verify_runtime_benchmark(RUNTIME_REPORT)
    assert verification["status"] == "pass"
    assert verification["multipliers"] == 4
    assert verification["callback_failure_restored"] is True


def test_canonical_factor_runtime_never_claims_model_execution() -> None:
    report = _load(RUNTIME_REPORT)
    assert report["scientific_boundary"]["torch_or_model_loaded"] is False
    assert report["scientific_boundary"]["model_inference_performed"] is False
    assert report["scientific_boundary"]["real_weights_mutated"] is False
    assert report["scientific_boundary"][
        "production_torch_adapter_integration_present"
    ] is False
    assert all(
        row["a_factor_restored_exactly"] and row["b_factor_restored_exactly"]
        for row in report["results"]["multiplier_rows"]
    )
