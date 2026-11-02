from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_canonical_vpd_category_effect_capture_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_canonical_category_effect_capture_benchmark_v1/report.json"


def _load() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_canonical_category_effect_benchmark_reproduces() -> None:
    verification = verify_benchmark(REPORT)
    assert verification["status"] == "pass"
    assert verification["categories"] == 2
    assert verification["negative_controls"] == 5
    assert verification["actual_model_execution_verified"] is False


def test_canonical_category_effect_benchmark_is_addressed_and_exact() -> None:
    report = _load()
    body = dict(report)
    assert body.pop("benchmark_id") == content_sha256(body)
    assert report["results"]["category_row_counts"] == [3, 2]
    assert report["results"]["maximum_category_frobenius_error"] < 1e-5
    assert report["results"]["restored_after_success"] is True
    assert report["results"]["capture_receipt_address_valid"] is True


def test_canonical_category_effect_failure_paths_restore_and_reject() -> None:
    results = _load()["results"]
    assert results["callback_failure_rejected"] is True
    assert results["callback_failure_restored"] is True
    assert results["lane_shape_mismatch_rejected"] is True
    assert results["lane_shape_mismatch_restored"] is True
    assert results["nonfinite_capture_rejected"] is True
    assert results["empty_category_rejected"] is True
    assert results["invalid_epsilon_rejected"] is True


def test_canonical_category_effect_retains_per_example_contract() -> None:
    design = _load()["design"]
    assert design["categories"] == ["invalid_authority", "valid_authority"]
    assert design["central_difference_epsilon"] == 0.05
    assert design["per_example_effects_retained"] is True
    assert design["authenticated_extraction_chained"] is True
    assert design["canonical_inventory_chained"] is True
    assert design["fake_peft_bridge_chained"] is True


def test_canonical_category_effect_keeps_real_execution_closed() -> None:
    boundary = _load()["scientific_boundary"]
    assert boundary["fake_peft_objects_only"] is True
    assert boundary["synthetic_batches_only"] is True
    assert boundary["torch_or_peft_imported"] is False
    assert boundary["model_loaded"] is False
    assert boundary["real_adapter_accessed"] is False
    assert boundary["model_inference_performed"] is False
    assert boundary["real_effect_archive_emitted"] is False
    assert boundary["execution_authorized"] is False
