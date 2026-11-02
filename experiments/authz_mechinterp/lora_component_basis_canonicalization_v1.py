from __future__ import annotations

from typing import Any

import numpy as np


def raw_rank_components(
    a_factor: np.ndarray, b_factor: np.ndarray, *, scale: float = 1.0
) -> list[np.ndarray]:
    a = np.asarray(a_factor, dtype=np.float64)
    b = np.asarray(b_factor, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
        raise ValueError("LoRA factors do not align")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.isfinite(scale):
        raise ValueError("LoRA factors and scale must be finite")
    return [
        float(scale) * np.outer(b[:, index], a[index, :])
        for index in range(a.shape[0])
    ]


def spectral_blocks(
    singular_values: np.ndarray, *, relative_gap_tolerance: float
) -> list[tuple[int, int]]:
    values = np.asarray(singular_values, dtype=np.float64)
    if values.ndim != 1 or np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("singular values are invalid")
    if relative_gap_tolerance < 0.0 or not np.isfinite(relative_gap_tolerance):
        raise ValueError("relative spectral-gap tolerance is invalid")
    if len(values) == 0:
        return []
    blocks: list[tuple[int, int]] = []
    start = 0
    reference = max(float(values[0]), 1e-12)
    for index in range(1, len(values)):
        gap = float(values[index - 1] - values[index])
        if gap > relative_gap_tolerance * reference:
            blocks.append((start, index))
            start = index
    blocks.append((start, len(values)))
    return blocks


def canonical_spectral_factorization(
    a_factor: np.ndarray,
    b_factor: np.ndarray,
    *,
    scale: float = 1.0,
    relative_gap_tolerance: float = 1e-6,
    zero_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Factor a low-rank adapter delta into identifiable spectral blocks.

    The computation uses thin QR factorizations and an SVD of only the LoRA-rank
    core, avoiding a full dense SVD for normal adapter shapes. Singular vectors
    with no resolved spectral gap are emitted as one aggregate block because
    individual vectors in a degenerate subspace are not identifiable.
    """

    a = np.asarray(a_factor, dtype=np.float64)
    b = np.asarray(b_factor, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
        raise ValueError("LoRA factors do not align")
    if a.shape[0] < 1 or min(a.shape[1], b.shape[0]) < a.shape[0]:
        raise ValueError("LoRA factors must expose a nonempty thin rank")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.isfinite(scale):
        raise ValueError("LoRA factors and scale must be finite")
    if zero_tolerance < 0.0 or not np.isfinite(zero_tolerance):
        raise ValueError("zero tolerance is invalid")
    q_b, r_b = np.linalg.qr(b, mode="reduced")
    q_a, r_a = np.linalg.qr(a.T, mode="reduced")
    core = float(scale) * (r_b @ r_a.T)
    u_core, singular_values, vh_core = np.linalg.svd(core, full_matrices=False)
    active = singular_values > zero_tolerance
    singular_values = singular_values[active]
    left = (q_b @ u_core)[:, active]
    right = (vh_core @ q_a.T)[active, :]
    blocks = spectral_blocks(
        singular_values, relative_gap_tolerance=relative_gap_tolerance
    )
    return {
        "singular_values": singular_values,
        "left_singular_vectors": left,
        "right_singular_vectors": right,
        "spectral_blocks": blocks,
        "raw_rank": int(a.shape[0]),
        "canonical_component_count": len(blocks),
        "unresolved_degenerate_blocks": sum(end - start > 1 for start, end in blocks),
    }


def canonical_spectral_components(
    a_factor: np.ndarray,
    b_factor: np.ndarray,
    *,
    scale: float = 1.0,
    relative_gap_tolerance: float = 1e-6,
    zero_tolerance: float = 1e-12,
) -> dict[str, Any]:
    factorization = canonical_spectral_factorization(
        a_factor,
        b_factor,
        scale=scale,
        relative_gap_tolerance=relative_gap_tolerance,
        zero_tolerance=zero_tolerance,
    )
    singular_values = factorization["singular_values"]
    left = factorization["left_singular_vectors"]
    right = factorization["right_singular_vectors"]
    components = [
        (left[:, start:end] * singular_values[start:end][None, :])
        @ right[start:end, :]
        for start, end in factorization["spectral_blocks"]
    ]
    return {
        **factorization,
        "components": components,
        "reconstruction": sum(
            components,
            np.zeros(
                (
                    np.asarray(b_factor).shape[0],
                    np.asarray(a_factor).shape[1],
                ),
                dtype=np.float64,
            ),
        ),
    }


def canonical_block_multiplier_factors(
    a_factor: np.ndarray,
    b_factor: np.ndarray,
    *,
    block_index: int,
    multiplier: float,
    scale: float = 1.0,
    relative_gap_tolerance: float = 1e-6,
    zero_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Return same-shape LoRA factors that scale one canonical spectral block.

    This function is model-free and does not mutate its inputs. A runtime may
    temporarily install the returned factors, verify the resulting delta, and
    restore the original factor bytes after the intervention.
    """

    if not np.isfinite(multiplier):
        raise ValueError("canonical block multiplier must be finite")
    if not np.isfinite(scale) or abs(float(scale)) <= 1e-15:
        raise ValueError("canonical factor intervention requires nonzero finite scale")
    a = np.asarray(a_factor, dtype=np.float64)
    b = np.asarray(b_factor, dtype=np.float64)
    factorization = canonical_spectral_factorization(
        a,
        b,
        scale=scale,
        relative_gap_tolerance=relative_gap_tolerance,
        zero_tolerance=zero_tolerance,
    )
    blocks = factorization["spectral_blocks"]
    if not 0 <= block_index < len(blocks):
        raise ValueError("canonical spectral block index is out of range")
    adjusted = factorization["singular_values"].copy()
    start, end = blocks[block_index]
    adjusted[start:end] *= float(multiplier)
    active_rank = len(adjusted)
    output_b = np.zeros_like(b, dtype=np.float64)
    output_a = np.zeros_like(a, dtype=np.float64)
    output_b[:, :active_rank] = (
        factorization["left_singular_vectors"]
        * (adjusted / float(scale))[None, :]
    )
    output_a[:active_rank, :] = factorization["right_singular_vectors"]
    return {
        "a_factor": output_a,
        "b_factor": output_b,
        "block_index": block_index,
        "block_range": (start, end),
        "multiplier": float(multiplier),
        "raw_rank": int(a.shape[0]),
        "active_rank": active_rank,
        "canonical_component_count": len(blocks),
    }
