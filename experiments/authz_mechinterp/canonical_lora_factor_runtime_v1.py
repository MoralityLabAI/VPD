from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file


def array_sha256(array: np.ndarray) -> str:
    values = np.asarray(array)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def load_canonical_module_artifact(
    path: Path, *, expected_sha256: str
) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("canonical module artifact hash differs")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "canonical_left_factors",
            "canonical_right_factors",
            "singular_values",
            "spectral_blocks",
            "raw_rank",
            "scale",
        }
        if set(archive.files) != required:
            raise ValueError("canonical module artifact fields differ")
        result = {name: archive[name] for name in archive.files}
    left = np.asarray(result["canonical_left_factors"], dtype=np.float64)
    right = np.asarray(result["canonical_right_factors"], dtype=np.float64)
    blocks = np.asarray(result["spectral_blocks"], dtype=np.int64)
    raw_rank = int(np.asarray(result["raw_rank"]).item())
    scale = float(np.asarray(result["scale"]).item())
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[1] != right.shape[0]
        or blocks.ndim != 2
        or blocks.shape[1] != 2
        or raw_rank < left.shape[1]
        or not np.isfinite(scale)
        or abs(scale) <= 1e-15
    ):
        raise ValueError("canonical module artifact geometry is invalid")
    if blocks.tolist()[0][0] != 0 or blocks.tolist()[-1][1] != left.shape[1]:
        raise ValueError("canonical module spectral blocks do not cover active rank")
    if any(
        int(blocks[index, 1]) != int(blocks[index + 1, 0])
        for index in range(len(blocks) - 1)
    ):
        raise ValueError("canonical module spectral blocks are not contiguous")
    return {
        "canonical_left_factors": left,
        "canonical_right_factors": right,
        "spectral_blocks": blocks,
        "raw_rank": raw_rank,
        "scale": scale,
        "active_rank": left.shape[1],
        "canonical_component_count": len(blocks),
    }


def multiplier_factors_from_artifact(
    artifact: dict[str, Any], *, block_index: int, multiplier: float
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(multiplier):
        raise ValueError("canonical multiplier must be finite")
    blocks = np.asarray(artifact["spectral_blocks"], dtype=np.int64)
    if not 0 <= block_index < len(blocks):
        raise ValueError("canonical block index is outside module artifact")
    left = np.asarray(artifact["canonical_left_factors"], dtype=np.float64).copy()
    right = np.asarray(artifact["canonical_right_factors"], dtype=np.float64)
    start, end = [int(value) for value in blocks[block_index]]
    left[:, start:end] *= float(multiplier)
    raw_rank = int(artifact["raw_rank"])
    scale = float(artifact["scale"])
    b_factor = np.zeros((left.shape[0], raw_rank), dtype=np.float64)
    a_factor = np.zeros((raw_rank, right.shape[1]), dtype=np.float64)
    b_factor[:, : left.shape[1]] = left
    a_factor[: right.shape[0], :] = right / scale
    return a_factor, b_factor


def _probe_outputs(
    a_factor: np.ndarray,
    b_factor: np.ndarray,
    *,
    scale: float,
    probes: np.ndarray,
) -> np.ndarray:
    return float(scale) * (b_factor @ (a_factor @ probes))


def run_with_canonical_multiplier_numpy(
    *,
    a_target: np.ndarray,
    b_target: np.ndarray,
    artifact: dict[str, Any],
    block_index: int,
    multiplier: float,
    callback: Callable[[], Any],
    probe_seed: int = 1707,
    probe_count: int = 8,
    maximum_relative_probe_error: float = 1e-6,
) -> tuple[Any, dict[str, Any]]:
    """Install a canonical block multiplier, run a callback, and restore bytes."""

    a_values = np.asarray(a_target)
    b_values = np.asarray(b_target)
    if not a_values.flags.writeable or not b_values.flags.writeable:
        raise ValueError("canonical intervention targets must be writeable")
    if b_values.shape[1] != a_values.shape[0]:
        raise ValueError("canonical intervention targets do not align")
    if probe_count < 1 or maximum_relative_probe_error < 0.0:
        raise ValueError("canonical intervention probe contract is invalid")
    installed_a, installed_b = multiplier_factors_from_artifact(
        artifact, block_index=block_index, multiplier=multiplier
    )
    if installed_a.shape != a_values.shape or installed_b.shape != b_values.shape:
        raise ValueError("canonical intervention factors have wrong target shape")
    original_a = a_values.copy()
    original_b = b_values.copy()
    original_a_hash = array_sha256(original_a)
    original_b_hash = array_sha256(original_b)
    rng = np.random.default_rng(probe_seed)
    probes = rng.normal(size=(a_values.shape[1], probe_count))
    expected_outputs = _probe_outputs(
        installed_a,
        installed_b,
        scale=float(artifact["scale"]),
        probes=probes,
    )
    output: Any = None
    try:
        np.copyto(a_values, installed_a.astype(a_values.dtype, copy=False))
        np.copyto(b_values, installed_b.astype(b_values.dtype, copy=False))
        observed_outputs = _probe_outputs(
            a_values.astype(np.float64),
            b_values.astype(np.float64),
            scale=float(artifact["scale"]),
            probes=probes,
        )
        relative_error = float(
            np.linalg.norm(observed_outputs - expected_outputs)
            / max(float(np.linalg.norm(expected_outputs)), 1e-12)
        )
        if relative_error > maximum_relative_probe_error:
            raise ValueError("installed canonical factor probe error exceeds tolerance")
        output = callback()
    finally:
        np.copyto(a_values, original_a)
        np.copyto(b_values, original_b)
    restored = (
        array_sha256(a_values) == original_a_hash
        and array_sha256(b_values) == original_b_hash
    )
    if not restored:
        raise ValueError("canonical intervention factor restoration failed")
    return output, {
        "schema": "sch.vpd-canonical-lora-factor-intervention.v1",
        "block_index": block_index,
        "multiplier": float(multiplier),
        "probe_seed": probe_seed,
        "probe_count": probe_count,
        "maximum_relative_probe_error": maximum_relative_probe_error,
        "observed_relative_probe_error": relative_error,
        "a_factor_restored": True,
        "b_factor_restored": True,
        "input_factor_hashes_restored": True,
        "model_inference_performed_by_runtime": False,
    }
