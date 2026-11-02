from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.benchmark_authenticated_lora_factor_bundle_extraction_v1 import (
    _fixture_config,
    _fixture_tensors,
    _write_safetensors,
)
from experiments.authz_mechinterp.benchmark_canonical_lora_peft_runtime_v1 import (
    _FakeModel,
    _FakePeftModule,
    _factor_arrays,
)
from experiments.authz_mechinterp.canonical_lora_factor_runtime_v1 import (
    load_canonical_module_artifact,
    multiplier_factors_from_artifact,
)
from experiments.authz_mechinterp.canonical_vpd_category_effect_capture_v1 import (
    capture_canonical_component_effect_by_category,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.emit_canonical_lora_component_inventory_v1 import (
    build_inventory,
)
from experiments.authz_mechinterp.extract_authenticated_lora_factor_bundle_v1 import (
    extract_factor_bundle,
)


def _attempt(function: Any) -> bool:
    try:
        function()
    except (ValueError, RuntimeError):
        return True
    return False


def build_report() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        weights = root / "adapter_model.safetensors"
        config = root / "adapter_config.json"
        _write_safetensors(weights, _fixture_tensors(), metadata={"format": "pt"})
        write_json(config, _fixture_config())
        factor_output = root / "factor-bundle"
        extract_factor_bundle(
            adapter_weights_path=weights,
            adapter_config_path=config,
            expected_adapter_sha256=sha256_file(weights),
            expected_config_sha256=sha256_file(config),
            output_dir=factor_output,
            synthetic_benchmark_only=True,
        )
        index = json.loads(
            (factor_output / "factor-index.json").read_text(encoding="utf-8")
        )
        inventory_output = root / "canonical-inventory"
        inventory = build_inventory(
            factor_index_path=factor_output / "factor-index.json",
            output_dir=inventory_output,
        )
        component_rows = [
            json.loads(line)
            for line in (inventory_output / "canonical_components.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        module_row = index["modules"][0]
        component_row = next(
            row for row in component_rows if row["module"] == module_row["module"]
        )
        canonical_module_row = inventory["artifacts"]["modules"][0]
        artifact_path = inventory_output / canonical_module_row["path"]
        artifact_sha256 = canonical_module_row["sha256"]
        artifact = load_canonical_module_artifact(
            artifact_path, expected_sha256=artifact_sha256
        )
        with np.load(factor_output / module_row["path"], allow_pickle=False) as archive:
            source_a = np.asarray(archive["a_factor"], dtype=np.float32)
            source_b = np.asarray(archive["b_factor"], dtype=np.float32)
        module_path = module_row["module"]
        resolved_path = "base_model.model." + module_path
        module = _FakePeftModule(source_a, source_b, scale=float(module_row["scale"]))
        model = _FakeModel([(resolved_path, module)])
        original_factors = _factor_arrays(module)
        category_batches = {
            "invalid_authority": [
                np.asarray([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]], dtype=np.float32),
                np.asarray([[0.25, 0.5, 1.0]], dtype=np.float32),
            ],
            "valid_authority": [
                np.asarray([[0.5, -0.25, 1.0], [1.0, 1.0, 0.0]], dtype=np.float32)
            ],
        }

        def capture_callback(category: str, batch: np.ndarray) -> np.ndarray:
            del category
            a_factor, b_factor = _factor_arrays(module)
            delta = float(module_row["scale"]) * b_factor @ a_factor
            return (np.asarray(batch) @ delta.T).astype(np.float32)

        epsilon = 0.05
        observed = capture_canonical_component_effect_by_category(
            model=model,
            module_path=module_path,
            adapter_name="default",
            canonical_artifact_path=artifact_path,
            expected_canonical_artifact_sha256=artifact_sha256,
            expected_source_a_factor_sha256=component_row["source_a_factor_sha256"],
            expected_source_b_factor_sha256=component_row["source_b_factor_sha256"],
            block_index=0,
            category_batches=category_batches,
            capture_batch_callback=capture_callback,
            epsilon=epsilon,
            no_grad_context_factory=nullcontext,
            maximum_relative_probe_error=2e-6,
        )
        a_one, b_one = multiplier_factors_from_artifact(
            artifact, block_index=0, multiplier=1.0
        )
        a_zero, b_zero = multiplier_factors_from_artifact(
            artifact, block_index=0, multiplier=0.0
        )
        derivative_delta = float(artifact["scale"]) * (
            b_one @ a_one - b_zero @ a_zero
        )
        expected_samples = {
            category: np.concatenate(
                [batch.astype(np.float64) @ derivative_delta.T for batch in batches],
                axis=0,
            ).astype(np.float32)
            for category, batches in category_batches.items()
        }
        category_errors = {
            category: float(
                np.linalg.norm(
                    observed["delta_samples_by_category"][category].astype(np.float64)
                    - expected.astype(np.float64)
                )
            )
            for category, expected in expected_samples.items()
        }
        restored_after_success = all(
            np.array_equal(current, original)
            for current, original in zip(
                _factor_arrays(module), original_factors, strict=True
            )
        )
        receipt = observed["receipt"]
        receipt_body = dict(receipt)
        receipt_id = receipt_body.pop("canonical_category_effect_capture_id")
        receipt_address_valid = receipt_id == content_sha256(receipt_body)

        failure_module = _FakePeftModule(source_a, source_b, scale=2.0)
        failure_original = _factor_arrays(failure_module)
        callback_failure_rejected = _attempt(
            lambda: capture_canonical_component_effect_by_category(
                model=_FakeModel([(resolved_path, failure_module)]),
                module_path=module_path,
                adapter_name="default",
                canonical_artifact_path=artifact_path,
                expected_canonical_artifact_sha256=artifact_sha256,
                expected_source_a_factor_sha256=component_row[
                    "source_a_factor_sha256"
                ],
                expected_source_b_factor_sha256=component_row[
                    "source_b_factor_sha256"
                ],
                block_index=0,
                category_batches=category_batches,
                capture_batch_callback=lambda category, batch: (_ for _ in ()).throw(
                    RuntimeError("synthetic capture failure")
                ),
                epsilon=epsilon,
                no_grad_context_factory=nullcontext,
            )
        )
        callback_failure_restored = all(
            np.array_equal(current, original)
            for current, original in zip(
                _factor_arrays(failure_module), failure_original, strict=True
            )
        )

        mismatch_module = _FakePeftModule(source_a, source_b, scale=2.0)
        mismatch_original = _factor_arrays(mismatch_module)
        callback_count = 0

        def mismatched_callback(category: str, batch: np.ndarray) -> np.ndarray:
            nonlocal callback_count
            callback_count += 1
            a_factor, b_factor = _factor_arrays(mismatch_module)
            values = (np.asarray(batch) @ (2.0 * b_factor @ a_factor).T).astype(
                np.float32
            )
            if callback_count == 4:
                return values[:1]
            return values

        lane_shape_mismatch_rejected = _attempt(
            lambda: capture_canonical_component_effect_by_category(
                model=_FakeModel([(resolved_path, mismatch_module)]),
                module_path=module_path,
                adapter_name="default",
                canonical_artifact_path=artifact_path,
                expected_canonical_artifact_sha256=artifact_sha256,
                expected_source_a_factor_sha256=component_row[
                    "source_a_factor_sha256"
                ],
                expected_source_b_factor_sha256=component_row[
                    "source_b_factor_sha256"
                ],
                block_index=0,
                category_batches=category_batches,
                capture_batch_callback=mismatched_callback,
                epsilon=epsilon,
                no_grad_context_factory=nullcontext,
            )
        )
        lane_shape_mismatch_restored = all(
            np.array_equal(current, original)
            for current, original in zip(
                _factor_arrays(mismatch_module), mismatch_original, strict=True
            )
        )
        nonfinite_capture_rejected = _attempt(
            lambda: capture_canonical_component_effect_by_category(
                model=_FakeModel(
                    [(resolved_path, _FakePeftModule(source_a, source_b, scale=2.0))]
                ),
                module_path=module_path,
                adapter_name="default",
                canonical_artifact_path=artifact_path,
                expected_canonical_artifact_sha256=artifact_sha256,
                expected_source_a_factor_sha256=component_row[
                    "source_a_factor_sha256"
                ],
                expected_source_b_factor_sha256=component_row[
                    "source_b_factor_sha256"
                ],
                block_index=0,
                category_batches=category_batches,
                capture_batch_callback=lambda category, batch: np.full(
                    (len(batch), 4), np.nan, dtype=np.float32
                ),
                epsilon=epsilon,
                no_grad_context_factory=nullcontext,
            )
        )
        empty_category_rejected = _attempt(
            lambda: capture_canonical_component_effect_by_category(
                model=model,
                module_path=module_path,
                adapter_name="default",
                canonical_artifact_path=artifact_path,
                expected_canonical_artifact_sha256=artifact_sha256,
                expected_source_a_factor_sha256=component_row[
                    "source_a_factor_sha256"
                ],
                expected_source_b_factor_sha256=component_row[
                    "source_b_factor_sha256"
                ],
                block_index=0,
                category_batches={"invalid_authority": []},
                capture_batch_callback=capture_callback,
                epsilon=epsilon,
                no_grad_context_factory=nullcontext,
            )
        )
        invalid_epsilon_rejected = _attempt(
            lambda: capture_canonical_component_effect_by_category(
                model=model,
                module_path=module_path,
                adapter_name="default",
                canonical_artifact_path=artifact_path,
                expected_canonical_artifact_sha256=artifact_sha256,
                expected_source_a_factor_sha256=component_row[
                    "source_a_factor_sha256"
                ],
                expected_source_b_factor_sha256=component_row[
                    "source_b_factor_sha256"
                ],
                block_index=0,
                category_batches=category_batches,
                capture_batch_callback=capture_callback,
                epsilon=0.0,
                no_grad_context_factory=nullcontext,
            )
        )
        maximum_category_error = max(category_errors.values())
        passed = (
            observed["category_names"] == sorted(category_batches)
            and observed["category_row_counts"].tolist() == [3, 2]
            and maximum_category_error < 1e-5
            and restored_after_success
            and receipt_address_valid
            and callback_failure_rejected
            and callback_failure_restored
            and lane_shape_mismatch_rejected
            and lane_shape_mismatch_restored
            and nonfinite_capture_rejected
            and empty_category_rejected
            and invalid_epsilon_rejected
        )
    body = {
        "schema": "sch.vpd-canonical-category-effect-capture-benchmark.v1",
        "status": "pass" if passed else "fail",
        "source": {
            "runner_sha256": sha256_file(Path(__file__)),
            "kernel_sha256": sha256_file(
                Path(__file__).with_name(
                    "canonical_vpd_category_effect_capture_v1.py"
                )
            ),
        },
        "design": {
            "authenticated_extraction_chained": True,
            "canonical_inventory_chained": True,
            "fake_peft_bridge_chained": True,
            "categories": ["invalid_authority", "valid_authority"],
            "central_difference_epsilon": epsilon,
            "per_example_effects_retained": True,
        },
        "results": {
            "category_row_counts": observed["category_row_counts"].tolist(),
            "category_frobenius_errors": category_errors,
            "maximum_category_frobenius_error": maximum_category_error,
            "restored_after_success": restored_after_success,
            "capture_receipt_address_valid": receipt_address_valid,
            "callback_failure_rejected": callback_failure_rejected,
            "callback_failure_restored": callback_failure_restored,
            "lane_shape_mismatch_rejected": lane_shape_mismatch_rejected,
            "lane_shape_mismatch_restored": lane_shape_mismatch_restored,
            "nonfinite_capture_rejected": nonfinite_capture_rejected,
            "empty_category_rejected": empty_category_rejected,
            "invalid_epsilon_rejected": invalid_epsilon_rejected,
        },
        "scientific_boundary": {
            "fake_peft_objects_only": True,
            "synthetic_batches_only": True,
            "torch_or_peft_imported": False,
            "model_loaded": False,
            "real_adapter_accessed": False,
            "model_inference_performed": False,
            "real_effect_archive_emitted": False,
            "execution_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = body.pop("benchmark_id", "")
    if identifier != content_sha256(body):
        raise ValueError("canonical category-effect benchmark address mismatch")
    if observed != build_report() or observed["status"] != "pass":
        raise ValueError("canonical category-effect benchmark is not reproducible")
    return {
        "schema": "sch.vpd-canonical-category-effect-capture-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "categories": len(observed["design"]["categories"]),
        "negative_controls": 5,
        "actual_model_execution_verified": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark canonical VPD category-effect capture V1."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_benchmark(args.verify)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_report()
        write_json(args.output, result)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "benchmark_id": result["benchmark_id"],
                "categories": result.get("design", {}).get(
                    "categories", result.get("categories")
                ),
                "execution_authorized": result.get("execution_authorized", False),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
