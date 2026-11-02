from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_canonical_vpd_effect_shard_assembly_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_canonical_effect_shard_assembly_benchmark_v1/report.json"


def _load() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_canonical_effect_assembly_benchmark_reproduces() -> None:
    verification = verify_benchmark(REPORT)
    assert verification["status"] == "pass"
    assert verification["components"] == 2
    assert verification["negative_controls"] == 9
    assert verification["actual_model_execution_verified"] is False


def test_canonical_effect_assembly_benchmark_is_addressed_and_deterministic() -> None:
    report = _load()
    body = dict(report)
    assert body.pop("benchmark_id") == content_sha256(body)
    results = report["results"]
    assert results["deterministic_reassembly"] is True
    assert results["written_artifact_verified"] is True
    assert results["expected_array_shapes_match"] is True
    assert results["all_support_counts_match"] is True


def test_canonical_effect_assembly_rejects_component_and_capture_tamper() -> None:
    results = _load()["results"]
    assert results["raw_rank_component_rejected"] is True
    assert results["capture_receipt_tamper_rejected"] is True
    assert results["capture_row_count_mismatch_rejected"] is True
    assert results["written_artifact_tamper_rejected"] is True


def test_canonical_effect_assembly_rejects_provenance_and_support_tamper() -> None:
    results = _load()["results"]
    assert results["split_identity_tamper_rejected"] is True
    assert results["category_pair_set_mismatch_rejected"] is True
    assert results["category_template_set_mismatch_rejected"] is True
    assert results["duplicate_episode_rejected"] is True
    assert results["small_countsketch_rejected"] is True


def test_canonical_effect_assembly_keeps_real_execution_closed() -> None:
    boundary = _load()["scientific_boundary"]
    assert boundary["synthetic_inputs_only"] is True
    assert boundary["fake_peft_objects_only"] is True
    assert boundary["model_loaded"] is False
    assert boundary["model_inference_performed"] is False
    assert boundary["real_adapter_accessed"] is False
    assert boundary["real_effect_archive_emitted"] is False
    assert boundary["execution_authorized"] is False
