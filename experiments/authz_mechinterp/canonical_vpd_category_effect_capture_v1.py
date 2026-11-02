from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Callable, ContextManager

import numpy as np

from experiments.authz_mechinterp.canonical_lora_peft_runtime_v1 import (
    run_with_canonical_multiplier_peft,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256


def _array_sha256(array: np.ndarray) -> str:
    values = np.asarray(array)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _capture_lane(
    *,
    category_names: list[str],
    category_batches: dict[str, list[Any]],
    capture_batch_callback: Callable[[str, Any], np.ndarray],
) -> dict[str, np.ndarray]:
    captured: dict[str, np.ndarray] = {}
    hidden_size: int | None = None
    for category in category_names:
        rows: list[np.ndarray] = []
        for batch in category_batches[category]:
            values = np.asarray(capture_batch_callback(category, batch))
            if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
                raise ValueError("canonical category capture callback must return nonempty matrices")
            if values.dtype not in (np.dtype("float32"), np.dtype("float64")):
                raise ValueError("canonical category capture callback must return float32 or float64")
            if not np.all(np.isfinite(values)):
                raise ValueError("canonical category capture callback returned nonfinite values")
            if hidden_size is None:
                hidden_size = int(values.shape[1])
            elif values.shape[1] != hidden_size:
                raise ValueError("canonical category capture hidden sizes differ")
            rows.append(values.astype(np.float64, copy=False))
        captured[category] = np.concatenate(rows, axis=0)
    return captured


def capture_canonical_component_effect_by_category(
    *,
    model: Any,
    module_path: str,
    adapter_name: str,
    canonical_artifact_path: Path,
    expected_canonical_artifact_sha256: str,
    expected_source_a_factor_sha256: str,
    expected_source_b_factor_sha256: str,
    block_index: int,
    category_batches: dict[str, list[Any]],
    capture_batch_callback: Callable[[str, Any], np.ndarray],
    epsilon: float,
    no_grad_context_factory: Callable[[], ContextManager[Any]] | None = None,
    probe_seed: int = 1707,
    probe_count: int = 8,
    maximum_relative_probe_error: float = 2e-3,
) -> dict[str, Any]:
    """Estimate one canonical component's signed effect in every category."""

    if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ValueError("canonical category central-difference epsilon must be in (0, 1)")
    if not isinstance(category_batches, dict) or not category_batches:
        raise ValueError("canonical category capture requires category batches")
    category_names = sorted(category_batches)
    if any(
        not isinstance(category, str)
        or not category
        or not isinstance(category_batches[category], list)
        or not category_batches[category]
        for category in category_names
    ):
        raise ValueError("canonical category names and batch lists must be nonempty")
    lanes: dict[str, dict[str, np.ndarray]] = {}
    lane_receipts: dict[str, dict[str, Any]] = {}
    for lane, multiplier in (("plus", 1.0 + epsilon), ("minus", 1.0 - epsilon)):
        captured, receipt = run_with_canonical_multiplier_peft(
            model=model,
            module_path=module_path,
            adapter_name=adapter_name,
            canonical_artifact_path=canonical_artifact_path,
            expected_canonical_artifact_sha256=expected_canonical_artifact_sha256,
            expected_source_a_factor_sha256=expected_source_a_factor_sha256,
            expected_source_b_factor_sha256=expected_source_b_factor_sha256,
            block_index=block_index,
            multiplier=multiplier,
            callback=lambda: _capture_lane(
                category_names=category_names,
                category_batches=category_batches,
                capture_batch_callback=capture_batch_callback,
            ),
            no_grad_context_factory=no_grad_context_factory,
            probe_seed=probe_seed,
            probe_count=probe_count,
            maximum_relative_probe_error=maximum_relative_probe_error,
        )
        lanes[lane] = captured
        lane_receipts[lane] = receipt
    delta_samples: dict[str, np.ndarray] = {}
    counts: list[int] = []
    means: list[np.ndarray] = []
    for category in category_names:
        plus = lanes["plus"][category]
        minus = lanes["minus"][category]
        if plus.shape != minus.shape:
            raise ValueError(
                f"canonical category plus/minus shapes differ: {category}"
            )
        values = ((plus - minus) / (2.0 * epsilon)).astype(np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError("canonical category central difference is nonfinite")
        delta_samples[category] = values
        counts.append(int(values.shape[0]))
        means.append(values.mean(axis=0, dtype=np.float64).astype(np.float32))
    mean_by_category = np.stack(means).astype(np.float32)
    row_counts = np.asarray(counts, dtype=np.int64)
    global_mean = np.concatenate(
        [delta_samples[category].astype(np.float64) for category in category_names],
        axis=0,
    ).mean(axis=0).astype(np.float32)
    receipt_body = {
        "schema": "sch.vpd-canonical-category-effect-capture.v1",
        "status": "complete_callback_owned_capture",
        "component": {
            "module": module_path,
            "adapter_name": adapter_name,
            "canonical_artifact_sha256": expected_canonical_artifact_sha256,
            "source_a_factor_sha256": expected_source_a_factor_sha256,
            "source_b_factor_sha256": expected_source_b_factor_sha256,
            "canonical_block_index": block_index,
        },
        "estimator": {
            "name": "per_example_central_difference_canonical_block_multiplier",
            "epsilon": float(epsilon),
            "multipliers": [1.0 + epsilon, 1.0 - epsilon],
            "probe_seed": probe_seed,
            "probe_count": probe_count,
            "maximum_relative_probe_error": maximum_relative_probe_error,
        },
        "categories": {
            "names": category_names,
            "row_counts": counts,
            "delta_sample_sha256": {
                category: _array_sha256(delta_samples[category])
                for category in category_names
            },
            "delta_mean_by_category_sha256": _array_sha256(mean_by_category),
            "delta_global_mean_sha256": _array_sha256(global_mean),
        },
        "lane_runtime_receipts": lane_receipts,
        "boundary": {
            "input_batch_identity_binding_delegated_to_successor_worker": True,
            "condition_template_countsketch_delegated_to_successor_worker": True,
            "capture_callback_invoked": True,
            "model_loading_performed_by_kernel": False,
            "model_inference_performed_by_kernel": False,
            "execution_authorized": False,
        },
    }
    receipt = {
        **receipt_body,
        "canonical_category_effect_capture_id": content_sha256(receipt_body),
    }
    return {
        "category_names": category_names,
        "category_row_counts": row_counts,
        "delta_samples_by_category": delta_samples,
        "delta_mean_by_category": mean_by_category,
        "delta_global_mean": global_mean,
        "receipt": receipt,
    }
