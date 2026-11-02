from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, ContextManager

import numpy as np

from experiments.authz_mechinterp.canonical_lora_factor_runtime_v1 import (
    load_canonical_module_artifact,
    multiplier_factors_from_artifact,
)


def _tensor_array(tensor: Any) -> np.ndarray:
    try:
        cpu_tensor = tensor.detach().to("cpu").contiguous()
        try:
            array = cpu_tensor.numpy()
        except (TypeError, RuntimeError):
            array = cpu_tensor.float().numpy()
    except (AttributeError, TypeError, RuntimeError) as exc:
        raise ValueError("PEFT factor tensor cannot be copied to a CPU NumPy array") from exc
    return np.asarray(array).copy()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes(order="C")).hexdigest()


def _normalized_factor_sha256(array: np.ndarray) -> str:
    values = np.asarray(array).astype(np.float64, copy=False)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _tensor_equal(left: Any, right: Any) -> bool:
    try:
        return bool(left.detach().equal(right.detach()))
    except (AttributeError, TypeError, RuntimeError):
        return np.array_equal(_tensor_array(left), _tensor_array(right))


def resolve_peft_lora_module(
    model: Any, *, module_path: str, adapter_name: str
) -> tuple[str, Any, Any, Any, float]:
    if not module_path or not adapter_name:
        raise ValueError("PEFT module path and adapter lane are required")
    try:
        named_modules = list(model.named_modules())
    except (AttributeError, TypeError) as exc:
        raise ValueError("model does not expose named_modules") from exc
    matches = [
        (name, module)
        for name, module in named_modules
        if name == module_path or name.endswith("." + module_path)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one PEFT module for {module_path}, found {len(matches)}"
        )
    resolved_name, module = matches[0]
    try:
        a_weight = module.lora_A[adapter_name].weight
        b_weight = module.lora_B[adapter_name].weight
        scaling = float(module.scaling[adapter_name])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"PEFT adapter lane {adapter_name} is incomplete at {module_path}"
        ) from exc
    if not math.isfinite(scaling) or abs(scaling) <= 1e-15:
        raise ValueError("PEFT adapter scaling is invalid")
    return resolved_name, module, a_weight, b_weight, scaling


def _default_no_grad_factory() -> Callable[[], ContextManager[Any]]:
    try:
        import torch
    except ImportError as exc:
        raise ValueError("canonical PEFT runtime requires Torch when no context is injected") from exc
    return torch.no_grad


def run_with_canonical_multiplier_peft(
    *,
    model: Any,
    module_path: str,
    adapter_name: str,
    canonical_artifact_path: Path,
    expected_canonical_artifact_sha256: str,
    expected_source_a_factor_sha256: str,
    expected_source_b_factor_sha256: str,
    block_index: int,
    multiplier: float,
    callback: Callable[[], Any],
    no_grad_context_factory: Callable[[], ContextManager[Any]] | None = None,
    probe_seed: int = 1707,
    probe_count: int = 8,
    maximum_relative_probe_error: float = 2e-3,
) -> tuple[Any, dict[str, Any]]:
    """Install one canonical block in a live PEFT lane and restore exact factors."""

    if probe_count < 1 or maximum_relative_probe_error < 0.0:
        raise ValueError("canonical PEFT probe contract is invalid")
    artifact = load_canonical_module_artifact(
        canonical_artifact_path,
        expected_sha256=expected_canonical_artifact_sha256,
    )
    resolved_name, _, a_weight, b_weight, live_scale = resolve_peft_lora_module(
        model, module_path=module_path, adapter_name=adapter_name
    )
    if not math.isclose(
        live_scale,
        float(artifact["scale"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("live PEFT scaling differs from canonical artifact")
    original_a_array = _tensor_array(a_weight)
    original_b_array = _tensor_array(b_weight)
    if _normalized_factor_sha256(original_a_array) != expected_source_a_factor_sha256:
        raise ValueError("live PEFT A factor differs from authenticated source")
    if _normalized_factor_sha256(original_b_array) != expected_source_b_factor_sha256:
        raise ValueError("live PEFT B factor differs from authenticated source")
    installed_a, installed_b = multiplier_factors_from_artifact(
        artifact, block_index=block_index, multiplier=multiplier
    )
    if installed_a.shape != original_a_array.shape or installed_b.shape != original_b_array.shape:
        raise ValueError("canonical factors have wrong live PEFT target shape")
    if not np.isfinite(multiplier):
        raise ValueError("canonical PEFT multiplier must be finite")
    original_a_hash = _array_sha256(original_a_array)
    original_b_hash = _array_sha256(original_b_array)
    original_a_tensor = a_weight.detach().clone()
    original_b_tensor = b_weight.detach().clone()
    try:
        installed_a_tensor = a_weight.new_tensor(installed_a)
        installed_b_tensor = b_weight.new_tensor(installed_b)
    except (AttributeError, TypeError, RuntimeError) as exc:
        raise ValueError("PEFT factor tensor cannot construct same-device values") from exc
    rng = np.random.default_rng(probe_seed)
    probes = rng.normal(size=(installed_a.shape[1], probe_count))
    expected_outputs = float(artifact["scale"]) * (
        installed_b @ (installed_a @ probes)
    )
    context_factory = no_grad_context_factory or _default_no_grad_factory()
    output: Any = None
    observed_relative_probe_error = float("inf")
    callback_error: BaseException | None = None
    callback_traceback: TracebackType | None = None
    try:
        with context_factory():
            a_weight.copy_(installed_a_tensor)
            b_weight.copy_(installed_b_tensor)
        observed_a = _tensor_array(a_weight).astype(np.float64)
        observed_b = _tensor_array(b_weight).astype(np.float64)
        observed_outputs = live_scale * (observed_b @ (observed_a @ probes))
        observed_relative_probe_error = float(
            np.linalg.norm(observed_outputs - expected_outputs)
            / max(float(np.linalg.norm(expected_outputs)), 1e-12)
        )
        if observed_relative_probe_error > maximum_relative_probe_error:
            raise ValueError("installed canonical PEFT factor probe error exceeds tolerance")
        output = callback()
    except BaseException as exc:
        callback_error = exc
        callback_traceback = exc.__traceback__
    finally:
        try:
            with context_factory():
                a_weight.copy_(original_a_tensor)
                b_weight.copy_(original_b_tensor)
        except BaseException as restoration_exc:
            raise ValueError("canonical PEFT factor restoration copy failed") from restoration_exc
    restored_a = (
        _array_sha256(_tensor_array(a_weight)) == original_a_hash
        and _tensor_equal(a_weight, original_a_tensor)
    )
    restored_b = (
        _array_sha256(_tensor_array(b_weight)) == original_b_hash
        and _tensor_equal(b_weight, original_b_tensor)
    )
    if not restored_a or not restored_b:
        raise ValueError("canonical PEFT factors were not restored exactly") from callback_error
    if callback_error is not None:
        raise callback_error.with_traceback(callback_traceback)
    return output, {
        "schema": "sch.vpd-canonical-lora-peft-factor-intervention.v1",
        "resolved_module": resolved_name,
        "module_path": module_path,
        "adapter_name": adapter_name,
        "canonical_artifact_sha256": expected_canonical_artifact_sha256,
        "block_index": block_index,
        "multiplier": float(multiplier),
        "probe_seed": probe_seed,
        "probe_count": probe_count,
        "maximum_relative_probe_error": maximum_relative_probe_error,
        "observed_relative_probe_error": observed_relative_probe_error,
        "a_factor_restored_exactly": True,
        "b_factor_restored_exactly": True,
        "source_factor_hashes_matched": True,
        "live_scaling_matched": True,
        "model_inference_performed_by_runtime": False,
        "callback_execution_performed": True,
    }
