from __future__ import annotations

import json
import math
from pathlib import Path

from experiments.authz_mechinterp.benchmark_authenticated_lora_factor_bundle_extraction_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256
from experiments.authz_mechinterp.extract_authenticated_lora_factor_bundle_v1 import (
    _module_scale,
    _parse_lora_key,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "artifacts/vpd_authenticated_lora_factor_bundle_extraction_benchmark_v1/report.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_authenticated_factor_bundle_benchmark_reproduces() -> None:
    verification = verify_benchmark(REPORT)
    assert verification["status"] == "pass"
    assert verification["modules"] == 2
    assert verification["negative_controls"] == 7
    assert verification["execution_authorized"] is False


def test_authenticated_factor_bundle_benchmark_is_addressed_and_strict() -> None:
    report = _load(REPORT)
    body = dict(report)
    assert body.pop("benchmark_id") == content_sha256(body)
    results = report["results"]
    assert results["factor_arrays_extracted_exactly"] is True
    assert results["module_scales"] == [2.0, 3.0]
    assert results["canonical_inventory_verified"] is True
    assert results["weights_hash_tamper_rejected"] is True
    assert results["config_hash_tamper_rejected"] is True
    assert results["orphan_factor_rejected"] is True
    assert results["unexpected_tensor_rejected"] is True
    assert results["rank_mismatch_rejected"] is True
    assert results["rank_mismatch_left_no_partial_output"] is True
    assert results["nonempty_output_rejected"] is True
    assert results["factor_artifact_tamper_rejected"] is True


def test_rs_lora_scale_uses_square_root_rank_and_pattern_alpha() -> None:
    config = {
        "r": 4,
        "lora_alpha": 8,
        "use_rslora": True,
        "alpha_pattern": {"layers.3.q_proj": 10},
    }
    alpha, scale = _module_scale(
        config,
        source_module="base_model.model.model.layers.3.q_proj",
        canonical_module="model.layers.3.q_proj",
        tensor_rank=4,
    )
    assert alpha == 10.0
    assert math.isclose(scale, 5.0)


def test_lora_key_parser_rejects_false_lane_suffixes() -> None:
    assert _parse_lora_key("model.layers.0.q_proj.lora_Afoo.weight") is None
    assert _parse_lora_key("model.layers.0.q_proj.lora_A.weight") == (
        "model.layers.0.q_proj",
        "A",
        "",
    )
    assert _parse_lora_key("model.layers.0.q_proj.lora_B.default.weight") == (
        "model.layers.0.q_proj",
        "B",
        "default",
    )


def test_benchmark_never_claims_real_adapter_or_model_execution() -> None:
    boundary = _load(REPORT)["scientific_boundary"]
    assert boundary["synthetic_adapter_fixture_only"] is True
    assert boundary["real_adapter_accessed"] is False
    assert boundary["torch_or_peft_loaded"] is False
    assert boundary["model_loaded"] is False
    assert boundary["model_inference_performed"] is False
    assert boundary["weights_mutated"] is False
    assert boundary["execution_authorized"] is False
