from __future__ import annotations

from typing import Any

import numpy as np


UNCERTAINTY_ARRAYS = {
    "category_uncertainty_block_counts",
    "delta_h_countsketch_block_sums",
    "uncertainty_block_count",
    "uncertainty_countsketch_dimension",
    "uncertainty_countsketch_seed",
}

CONDITION_ROBUSTNESS_ARRAYS = {
    "condition_names",
    "condition_category_names",
    "condition_episode_counts",
    "delta_h_countsketch_condition_sums",
}

TEMPLATE_ROBUSTNESS_ARRAYS = {
    "template_cell_names",
    "template_cell_category_names",
    "template_cell_group_names",
    "template_cell_episode_counts",
    "delta_h_countsketch_template_cell_sums",
}


def countsketch(values: np.ndarray, *, dimension: int, seed: int) -> np.ndarray:
    """Return a deterministic signed-hash sketch of the last array dimension."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim < 1 or matrix.shape[-1] < 1:
        raise ValueError("CountSketch input must have a non-empty final dimension")
    if dimension < 2:
        raise ValueError("CountSketch dimension must be at least two")
    rng = np.random.default_rng(seed)
    buckets = rng.integers(0, dimension, size=matrix.shape[-1], dtype=np.int64)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=matrix.shape[-1])
    flat = matrix.reshape(-1, matrix.shape[-1])
    output = np.zeros((flat.shape[0], dimension), dtype=np.float64)
    for source_index, bucket in enumerate(buckets.tolist()):
        output[:, bucket] += flat[:, source_index] * signs[source_index]
    return output.reshape(*matrix.shape[:-1], dimension)


def validate_uncertainty_arrays(
    *,
    block_counts: np.ndarray,
    block_sums: np.ndarray,
    component_count: int,
    category_count: int,
    block_count: int,
    sketch_dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(block_counts)
    sums = np.asarray(block_sums)
    if counts.shape != (category_count, block_count):
        raise ValueError("category uncertainty block counts have the wrong shape")
    if counts.dtype.kind not in "iu" or np.any(counts <= 0):
        raise ValueError("category uncertainty blocks must all have positive counts")
    if sums.shape != (component_count, category_count, block_count, sketch_dimension):
        raise ValueError("category uncertainty block sums have the wrong shape")
    if sums.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("category uncertainty block sums must be float32 or float64")
    if not np.all(np.isfinite(sums)):
        raise ValueError("category uncertainty block sums contain non-finite values")
    return counts.astype(np.int64), sums.astype(np.float32)


def leave_one_block_out_strengths(
    block_sums: np.ndarray, block_counts: np.ndarray
) -> np.ndarray:
    """Compute category strengths for each leave-one-pair-block-out replicate.

    Returns component x omitted_block x category. Blocks are omitted across every
    category together, preserving the paired prompt-block design.
    """

    means, _ = leave_one_block_out_means(block_sums, block_counts)
    return np.linalg.norm(means, axis=3)


def leave_one_block_out_means(
    block_sums: np.ndarray, block_counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return component x omitted_block x category mean sketches and counts."""

    sums = np.asarray(block_sums, dtype=np.float64)
    counts = np.asarray(block_counts, dtype=np.int64)
    if sums.ndim != 4 or counts.ndim != 2:
        raise ValueError("uncertainty arrays must be rank four and rank two")
    if sums.shape[1:3] != counts.shape:
        raise ValueError("uncertainty sums and counts do not align")
    total_sums = sums.sum(axis=2)
    total_counts = counts.sum(axis=1)
    replicate_means: list[np.ndarray] = []
    replicate_counts: list[np.ndarray] = []
    for block_index in range(counts.shape[1]):
        remaining_counts = total_counts - counts[:, block_index]
        if np.any(remaining_counts <= 0):
            raise ValueError("leave-one-block-out replicate has no remaining support")
        replicate_means.append(
            (total_sums - sums[:, :, block_index, :])
            / remaining_counts[None, :, None]
        )
        replicate_counts.append(remaining_counts)
    return (
        np.stack(replicate_means, axis=1),
        np.stack(replicate_counts, axis=0),
    )


def pair_block_direction_metrics(
    *,
    block_sums: np.ndarray,
    block_counts: np.ndarray,
    minimum_direction_cosine: float,
    minimum_relative_norm: float,
) -> dict[str, np.ndarray]:
    """Measure equal-weight directional coverage of pair blocks per category."""

    sums = np.asarray(block_sums, dtype=np.float64)
    counts = np.asarray(block_counts, dtype=np.int64)
    if sums.ndim != 4 or counts.ndim != 2 or sums.shape[1:3] != counts.shape:
        raise ValueError("pair-block direction robustness inputs do not align")
    if np.any(counts <= 0):
        raise ValueError("pair-block direction robustness support is invalid")
    if not 0.0 <= minimum_direction_cosine <= 1.0:
        raise ValueError("minimum pair-block direction cosine must be in [0, 1]")
    if not 0.0 <= minimum_relative_norm <= 1.0:
        raise ValueError("minimum pair-block relative norm must be in [0, 1]")
    block_means = sums / counts[None, :, :, None]
    references = sums.sum(axis=2) / counts.sum(axis=1)[None, :, None]
    reference_norms = np.linalg.norm(references, axis=2)
    block_norms = np.linalg.norm(block_means, axis=3)
    cosines = np.sum(block_means * references[:, :, None, :], axis=3) / np.maximum(
        block_norms * reference_norms[:, :, None], 1e-12
    )
    # Roundoff can place a mathematically exact cosine infinitesimally outside
    # its closed interval (for example, 1.0000000000000002).  Keep the stored
    # diagnostic inside the metric's declared contract.
    cosines = np.clip(cosines, -1.0, 1.0)
    relative_norms = block_norms / np.maximum(reference_norms[:, :, None], 1e-12)
    passing = (
        (reference_norms[:, :, None] > 1e-12)
        & (cosines >= minimum_direction_cosine)
        & (relative_norms >= minimum_relative_norm)
    )
    return {
        "coverage": passing.mean(axis=2),
        "minimum_direction_cosine": cosines.min(axis=2),
        "minimum_relative_norm": relative_norms.min(axis=2),
    }


def validate_condition_robustness_arrays(
    *,
    condition_names: np.ndarray,
    condition_category_names: np.ndarray,
    condition_counts: np.ndarray,
    condition_sums: np.ndarray,
    component_count: int,
    category_names: list[str],
    sketch_dimension: int,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    names = [str(value) for value in np.asarray(condition_names).tolist()]
    categories = [
        str(value) for value in np.asarray(condition_category_names).tolist()
    ]
    counts = np.asarray(condition_counts)
    sums = np.asarray(condition_sums)
    if names != sorted(names) or len(names) != len(set(names)) or not names:
        raise ValueError("condition robustness names must be sorted and unique")
    if len(categories) != len(names) or any(
        category not in set(category_names) for category in categories
    ):
        raise ValueError("condition robustness categories are invalid")
    if set(categories) != set(category_names):
        raise ValueError("condition robustness omits an activation category")
    if counts.shape != (len(names),) or counts.dtype.kind not in "iu" or np.any(
        counts <= 0
    ):
        raise ValueError("condition robustness counts are invalid")
    if sums.shape != (component_count, len(names), sketch_dimension):
        raise ValueError("condition robustness sums have the wrong shape")
    if sums.dtype not in (np.dtype("float32"), np.dtype("float64")) or not np.all(
        np.isfinite(sums)
    ):
        raise ValueError("condition robustness sums are invalid")
    return names, categories, counts.astype(np.int64), sums.astype(np.float32)


def condition_robustness_metrics(
    *,
    condition_sums: np.ndarray,
    condition_counts: np.ndarray,
    condition_categories: list[str],
    category_names: list[str],
    minimum_direction_cosine: float,
    minimum_relative_norm: float,
) -> dict[str, np.ndarray]:
    sums = np.asarray(condition_sums, dtype=np.float64)
    counts = np.asarray(condition_counts, dtype=np.int64)
    if sums.ndim != 3 or counts.shape != (sums.shape[1],):
        raise ValueError("condition robustness inputs do not align")
    if len(condition_categories) != len(counts) or np.any(counts <= 0):
        raise ValueError("condition robustness support is invalid")
    if not 0.0 <= minimum_direction_cosine <= 1.0:
        raise ValueError("minimum condition direction cosine must be in [0, 1]")
    if not 0.0 <= minimum_relative_norm <= 1.0:
        raise ValueError("minimum condition relative norm must be in [0, 1]")
    means = sums / counts[None, :, None]
    coverage = np.zeros((sums.shape[0], len(category_names)), dtype=np.float64)
    minimum_cosines = np.zeros_like(coverage)
    minimum_relative_norms = np.zeros_like(coverage)
    for category_index, category in enumerate(category_names):
        indices = [
            index
            for index, value in enumerate(condition_categories)
            if value == category
        ]
        if not indices:
            raise ValueError("condition robustness category has no conditions")
        category_sum = sums[:, indices, :].sum(axis=1)
        category_count = int(counts[indices].sum())
        reference = category_sum / category_count
        reference_norm = np.linalg.norm(reference, axis=1)
        condition_means = means[:, indices, :]
        condition_norms = np.linalg.norm(condition_means, axis=2)
        cosines = np.sum(condition_means * reference[:, None, :], axis=2) / np.maximum(
            condition_norms * reference_norm[:, None], 1e-12
        )
        cosines = np.clip(cosines, -1.0, 1.0)
        relative_norms = condition_norms / np.maximum(reference_norm[:, None], 1e-12)
        passing = (
            (reference_norm[:, None] > 1e-12)
            & (cosines >= minimum_direction_cosine)
            & (relative_norms >= minimum_relative_norm)
        )
        coverage[:, category_index] = passing.mean(axis=1)
        minimum_cosines[:, category_index] = cosines.min(axis=1)
        minimum_relative_norms[:, category_index] = relative_norms.min(axis=1)
    return {
        "coverage": coverage,
        "minimum_direction_cosine": minimum_cosines,
        "minimum_relative_norm": minimum_relative_norms,
    }


def validate_template_robustness_arrays(
    *,
    cell_names: np.ndarray,
    cell_category_names: np.ndarray,
    cell_group_names: np.ndarray,
    cell_counts: np.ndarray,
    cell_sums: np.ndarray,
    component_count: int,
    category_names: list[str],
    sketch_dimension: int,
) -> tuple[list[str], list[str], list[str], np.ndarray, np.ndarray]:
    names = [str(value) for value in np.asarray(cell_names).tolist()]
    categories = [str(value) for value in np.asarray(cell_category_names).tolist()]
    groups = [str(value) for value in np.asarray(cell_group_names).tolist()]
    counts = np.asarray(cell_counts)
    sums = np.asarray(cell_sums)
    if names != sorted(names) or len(names) != len(set(names)) or not names:
        raise ValueError("template robustness cell names must be sorted and unique")
    if len(categories) != len(names) or len(groups) != len(names):
        raise ValueError("template robustness cell identities do not align")
    if any(category not in set(category_names) for category in categories) or any(
        not group for group in groups
    ):
        raise ValueError("template robustness cell identities are invalid")
    expected_groups = set(groups)
    if not expected_groups or any(
        {groups[index] for index, value in enumerate(categories) if value == category}
        != expected_groups
        for category in category_names
    ):
        raise ValueError("template robustness requires the same template groups per category")
    if len(names) != len(category_names) * len(expected_groups):
        raise ValueError("template robustness category/group cells are incomplete")
    if counts.shape != (len(names),) or counts.dtype.kind not in "iu" or np.any(
        counts <= 0
    ):
        raise ValueError("template robustness counts are invalid")
    if sums.shape != (component_count, len(names), sketch_dimension):
        raise ValueError("template robustness sums have the wrong shape")
    if sums.dtype not in (np.dtype("float32"), np.dtype("float64")) or not np.all(
        np.isfinite(sums)
    ):
        raise ValueError("template robustness sums are invalid")
    return (
        names,
        categories,
        groups,
        counts.astype(np.int64),
        sums.astype(np.float32),
    )


def template_robustness_metrics(
    *,
    cell_sums: np.ndarray,
    cell_counts: np.ndarray,
    cell_categories: list[str],
    category_names: list[str],
    minimum_direction_cosine: float,
    minimum_relative_norm: float,
) -> dict[str, np.ndarray]:
    """Score equal-weight semantic-template coverage within each category."""

    return condition_robustness_metrics(
        condition_sums=cell_sums,
        condition_counts=cell_counts,
        condition_categories=cell_categories,
        category_names=category_names,
        minimum_direction_cosine=minimum_direction_cosine,
        minimum_relative_norm=minimum_relative_norm,
    )


def uncertainty_contract(*, block_count: int, sketch_dimension: int, seed: int) -> dict[str, Any]:
    return {
        "schema": "sch.vpd-category-effect-uncertainty.v1",
        "atomic_unit": "pair_id",
        "block_assignment": "stable_hash_rank_round_robin",
        "estimator": "leave_one_pair_block_out_countsketch_mean_strength",
        "block_count": block_count,
        "countsketch_dimension": sketch_dimension,
        "countsketch_seed": seed,
        "all_categories_share_pair_blocks": True,
        "all_category_blocks_nonempty": True,
        "claim_boundary": (
            "CountSketch jackknife stability screens category assignments; it does not "
            "establish component causality or replace independent split replication."
        ),
    }


def pair_block_direction_robustness_contract(
    *,
    minimum_coverage: float,
    minimum_direction_cosine: float,
    minimum_relative_norm: float,
) -> dict[str, Any]:
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum pair-block direction coverage must be in (0, 1]")
    if not 0.0 <= minimum_direction_cosine <= 1.0:
        raise ValueError("minimum pair-block direction cosine must be in [0, 1]")
    if not 0.0 <= minimum_relative_norm <= 1.0:
        raise ValueError("minimum pair-block relative norm must be in [0, 1]")
    return {
        "schema": "sch.vpd-category-pair-block-direction-robustness.v1",
        "estimator": "pair_block_mean_countsketch_direction_and_relative_norm_coverage",
        "block_weighting": "equal_pair_block_coverage",
        "category_reference": "episode_weighted_pair_block_mean",
        "minimum_coverage": minimum_coverage,
        "minimum_direction_cosine": minimum_direction_cosine,
        "minimum_relative_norm": minimum_relative_norm,
        "failed_specific_routes_to": "shared_cross_category",
        "reuses_signed_uncertainty_arrays": True,
        "model_outcomes_used": False,
        "claim_boundary": (
            "Pair-block direction coverage screens within-split directional coherence; "
            "it does not establish component causality or replace semantic replication."
        ),
    }


def condition_robustness_contract(
    *,
    minimum_coverage: float,
    minimum_direction_cosine: float,
    minimum_relative_norm: float,
) -> dict[str, Any]:
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum condition coverage must be in (0, 1]")
    if not 0.0 <= minimum_direction_cosine <= 1.0:
        raise ValueError("minimum condition direction cosine must be in [0, 1]")
    if not 0.0 <= minimum_relative_norm <= 1.0:
        raise ValueError("minimum condition relative norm must be in [0, 1]")
    return {
        "schema": "sch.vpd-category-condition-robustness.v1",
        "estimator": "condition_mean_countsketch_direction_and_relative_norm_coverage",
        "condition_weighting": "equal_condition_coverage",
        "category_reference": "episode_weighted_condition_mean",
        "minimum_coverage": minimum_coverage,
        "minimum_direction_cosine": minimum_direction_cosine,
        "minimum_relative_norm": minimum_relative_norm,
        "failed_specific_routes_to": "shared_cross_category",
        "model_outcomes_used": False,
        "claim_boundary": (
            "Condition coverage screens category breadth; it does not establish "
            "component causality or replace independent split replication."
        ),
    }


def template_robustness_contract(
    *,
    minimum_coverage: float,
    minimum_direction_cosine: float,
    minimum_relative_norm: float,
) -> dict[str, Any]:
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum template-group coverage must be in (0, 1]")
    if not 0.0 <= minimum_direction_cosine <= 1.0:
        raise ValueError("minimum template-group direction cosine must be in [0, 1]")
    if not 0.0 <= minimum_relative_norm <= 1.0:
        raise ValueError("minimum template-group relative norm must be in [0, 1]")
    return {
        "schema": "sch.vpd-category-template-robustness.v1",
        "estimator": "template_group_mean_countsketch_direction_and_relative_norm_coverage",
        "template_weighting": "equal_template_group_coverage",
        "category_reference": "episode_weighted_template_group_mean",
        "minimum_coverage": minimum_coverage,
        "minimum_direction_cosine": minimum_direction_cosine,
        "minimum_relative_norm": minimum_relative_norm,
        "same_template_group_set_required_per_category": True,
        "failed_specific_routes_to": "shared_cross_category",
        "model_outcomes_used": False,
        "claim_boundary": (
            "Template-group coverage screens semantic breadth within one split; it does "
            "not establish component causality or replace independent split replication."
        ),
    }
