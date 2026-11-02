from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.authz_mechinterp.benchmark_canonical_lora_peft_runtime_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.canonical_lora_peft_runtime_v1 import (
    _tensor_array,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_canonical_lora_peft_runtime_benchmark_v1/report.json"


def _load() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_canonical_peft_runtime_benchmark_reproduces() -> None:
    verification = verify_benchmark(REPORT)
    assert verification["status"] == "pass"
    assert verification["multipliers"] == 4
    assert verification["negative_controls"] == 5
    assert verification["actual_torch_peft_execution_verified"] is False


def test_canonical_peft_runtime_benchmark_is_addressed_and_restores() -> None:
    report = _load()
    body = dict(report)
    assert body.pop("benchmark_id") == content_sha256(body)
    assert report["results"]["maximum_delta_frobenius_error"] < 1e-6
    assert report["results"]["maximum_relative_probe_error"] < 2e-6
    assert all(
        row["a_factor_restored_exactly"]
        and row["b_factor_restored_exactly"]
        and row["resolved_module_suffix_matched"]
        for row in report["results"]["multiplier_rows"]
    )


def test_canonical_peft_runtime_rejects_identity_and_runtime_mismatches() -> None:
    results = _load()["results"]
    assert results["callback_failure_propagated"] is True
    assert results["callback_failure_restored"] is True
    assert results["source_hash_mismatch_rejected"] is True
    assert results["source_hash_mismatch_never_mutated"] is True
    assert results["live_scale_mismatch_rejected"] is True
    assert results["ambiguous_module_match_rejected"] is True
    assert results["canonical_artifact_tamper_rejected"] is True


def test_canonical_peft_runtime_keeps_actual_torch_boundary_closed() -> None:
    boundary = _load()["scientific_boundary"]
    assert boundary["fake_peft_objects_only"] is True
    assert boundary["torch_imported"] is False
    assert boundary["peft_imported"] is False
    assert boundary["model_loaded"] is False
    assert boundary["real_adapter_accessed"] is False
    assert boundary["model_inference_performed"] is False
    assert boundary["real_weights_mutated"] is False
    assert boundary["actual_torch_peft_execution_verified"] is False
    assert boundary["execution_authorized"] is False


class _BfloatLikeTensor:
    def __init__(self, values: np.ndarray, *, float_view: bool = False) -> None:
        self.values = values
        self.float_view = float_view

    def detach(self) -> _BfloatLikeTensor:
        return self

    def to(self, device: str) -> _BfloatLikeTensor:
        assert device == "cpu"
        return self

    def contiguous(self) -> _BfloatLikeTensor:
        return self

    def numpy(self) -> np.ndarray:
        if not self.float_view:
            raise TypeError("synthetic bfloat16 has no direct NumPy view")
        return self.values.astype(np.float32)

    def float(self) -> _BfloatLikeTensor:
        return _BfloatLikeTensor(self.values, float_view=True)


def test_tensor_inspection_falls_back_to_float_for_bfloat_like_values() -> None:
    values = np.asarray([[1.0, -0.5]], dtype=np.float32)
    observed = _tensor_array(_BfloatLikeTensor(values))
    assert observed.dtype == np.float32
    assert np.array_equal(observed, values)
