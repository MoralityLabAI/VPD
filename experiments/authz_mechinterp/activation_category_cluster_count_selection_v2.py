from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    DEFAULT_MAXIMUM_CLUSTERS_PER_CATEGORY,
    DEFAULT_MINIMUM_CATEGORY_CLUSTER_PROJECTION_STABILITY,
    DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE,
    DEFAULT_MINIMUM_CATEGORY_CLUSTER_SIZE,
    adjusted_rand_index,
    deterministic_kmeans,
    stable_hash,
)


SCHEMA = "sch.vpd-category-cluster-count-selection.v2"
DEFAULT_MINIMUM_CATEGORY_CLUSTER_BOUNDARY_GAP_RATIO = 1.50
DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE_P10 = 0.05


def silhouette_values(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    assignments = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or assignments.shape != (len(values),):
        raise ValueError("silhouette matrix and labels do not align")
    if not np.all(np.isfinite(values)):
        raise ValueError("silhouette matrix contains non-finite values")
    unique_labels = sorted(set(assignments.tolist()))
    if len(unique_labels) < 2:
        return np.zeros(len(values), dtype=np.float64)
    distances = np.sqrt(
        np.maximum(
            np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=2),
            0.0,
        )
    )
    result = np.zeros(len(values), dtype=np.float64)
    for index, label in enumerate(assignments.tolist()):
        same = np.where(assignments == label)[0]
        same = same[same != index]
        if len(same) == 0:
            continue
        within = float(np.mean(distances[index, same]))
        between = min(
            float(np.mean(distances[index, assignments == other]))
            for other in unique_labels
            if other != label
        )
        scale = max(within, between)
        result[index] = (between - within) / scale if scale > 1e-12 else 0.0
    return result


def minimum_cluster_boundary_gap_ratio(
    matrix: np.ndarray, labels: np.ndarray
) -> float:
    """Measure whether adjacent clusters have an actual empty boundary.

    For every cluster pair, the closest cross-cluster distance is divided by
    the larger cluster's median within-cluster nearest-neighbour distance. A
    continuous manifold cut by k-means has a ratio near one at the cut, even
    when its mean silhouette and replicate ARI are high. Separated modes have a
    materially larger ratio.
    """

    values = np.asarray(matrix, dtype=np.float64)
    assignments = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or assignments.shape != (len(values),):
        raise ValueError("boundary-gap matrix and labels do not align")
    if not np.all(np.isfinite(values)):
        raise ValueError("boundary-gap matrix contains non-finite values")
    unique_labels = sorted(set(assignments.tolist()))
    if len(unique_labels) < 2:
        return 0.0
    distances = np.sqrt(
        np.maximum(
            np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=2),
            0.0,
        )
    )
    within_scale: dict[int, float] = {}
    for label in unique_labels:
        indices = np.where(assignments == label)[0]
        if len(indices) < 2:
            return 0.0
        local = distances[np.ix_(indices, indices)].copy()
        np.fill_diagonal(local, np.inf)
        nearest = np.min(local, axis=1)
        within_scale[label] = float(np.median(nearest))
    ratios: list[float] = []
    for left_position, left in enumerate(unique_labels):
        left_indices = np.where(assignments == left)[0]
        for right in unique_labels[left_position + 1 :]:
            right_indices = np.where(assignments == right)[0]
            cross_gap = float(
                np.min(distances[np.ix_(left_indices, right_indices)])
            )
            scale = max(within_scale[left], within_scale[right], 1e-12)
            ratios.append(cross_gap / scale)
    return min(ratios)


def select_category_cluster_counts_v2(
    replicate_matrices: list[np.ndarray],
    component_ids: list[str],
    categories: list[str],
    *,
    seed: int,
    replicate_kinds: list[str] | None = None,
    maximum_clusters_per_category: int = DEFAULT_MAXIMUM_CLUSTERS_PER_CATEGORY,
    minimum_replicate_stability: float = DEFAULT_MINIMUM_CATEGORY_CLUSTER_PROJECTION_STABILITY,
    minimum_mean_silhouette: float = DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE,
    minimum_silhouette_p10: float = DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE_P10,
    minimum_boundary_gap_ratio: float = DEFAULT_MINIMUM_CATEGORY_CLUSTER_BOUNDARY_GAP_RATIO,
    minimum_cluster_size: int = DEFAULT_MINIMUM_CATEGORY_CLUSTER_SIZE,
    max_iterations: int = 100,
) -> dict[str, Any]:
    if len(replicate_matrices) < 2:
        raise ValueError("category cluster selection requires multiple replicates")
    matrices = [np.asarray(matrix, dtype=np.float64) for matrix in replicate_matrices]
    if any(matrix.ndim != 2 or matrix.shape[0] != len(component_ids) for matrix in matrices):
        raise ValueError("category cluster replicate matrices do not align")
    if any(not np.all(np.isfinite(matrix)) for matrix in matrices):
        raise ValueError("category cluster replicate matrix contains non-finite values")
    if len(component_ids) != len(set(component_ids)) or len(categories) != len(component_ids):
        raise ValueError("category cluster component identities do not align")
    if maximum_clusters_per_category < 1:
        raise ValueError("maximum clusters per category must be positive")
    if not 0.0 <= minimum_replicate_stability <= 1.0:
        raise ValueError("minimum replicate stability must be in [0, 1]")
    if not -1.0 <= minimum_mean_silhouette <= 1.0:
        raise ValueError("minimum mean silhouette must be in [-1, 1]")
    if not -1.0 <= minimum_silhouette_p10 <= 1.0:
        raise ValueError("minimum silhouette p10 must be in [-1, 1]")
    if minimum_boundary_gap_ratio <= 0.0:
        raise ValueError("minimum boundary gap ratio must be positive")
    if minimum_cluster_size < 2:
        raise ValueError("minimum cluster size must be at least two")
    kinds = (
        ["projection"] * len(matrices)
        if replicate_kinds is None
        else [str(value) for value in replicate_kinds]
    )
    if len(kinds) != len(matrices) or any(not value for value in kinds):
        raise ValueError("category cluster replicate kinds do not align")

    allocation: dict[str, int] = {}
    category_results: dict[str, Any] = {}
    for category in sorted(set(categories)):
        indices = [index for index, value in enumerate(categories) if value == category]
        local_ids = [component_ids[index] for index in indices]
        maximum_candidate = min(
            maximum_clusters_per_category,
            len(indices) // minimum_cluster_size,
        )
        candidates: list[dict[str, Any]] = [
            {
                "cluster_count": 1,
                "status": "pass_fallback",
                "minimum_replicate_adjusted_rand": 1.0,
                "minimum_replicate_mean_silhouette": 0.0,
                "minimum_replicate_silhouette_p10": 0.0,
                "minimum_replicate_boundary_gap_ratio": 0.0,
                "minimum_observed_cluster_size": len(indices),
            }
        ]
        for cluster_count in range(2, maximum_candidate + 1):
            local_seed = int(
                stable_hash(
                    seed,
                    "activation-category-count-selection-v2",
                    category,
                    str(cluster_count),
                )[:8],
                16,
            )
            replicate_labels: list[np.ndarray] = []
            mean_silhouettes: list[float] = []
            silhouette_p10s: list[float] = []
            boundary_gaps: list[float] = []
            cluster_sizes: list[int] = []
            for matrix in matrices:
                labels, _, _ = deterministic_kmeans(
                    matrix[indices],
                    local_ids,
                    cluster_count=cluster_count,
                    seed=local_seed,
                    max_iterations=max_iterations,
                )
                silhouettes = silhouette_values(matrix[indices], labels)
                replicate_labels.append(labels)
                mean_silhouettes.append(float(np.mean(silhouettes)))
                silhouette_p10s.append(float(np.quantile(silhouettes, 0.10)))
                boundary_gaps.append(
                    minimum_cluster_boundary_gap_ratio(matrix[indices], labels)
                )
                cluster_sizes.append(
                    min(int(np.sum(labels == label)) for label in set(labels.tolist()))
                )
            stability_values = [
                adjusted_rand_index(replicate_labels[left], replicate_labels[right])
                for left in range(len(replicate_labels))
                for right in range(left + 1, len(replicate_labels))
            ]
            candidate = {
                "cluster_count": cluster_count,
                "status": "pass",
                "minimum_replicate_adjusted_rand": min(stability_values),
                "minimum_replicate_mean_silhouette": min(mean_silhouettes),
                "minimum_replicate_silhouette_p10": min(silhouette_p10s),
                "minimum_replicate_boundary_gap_ratio": min(boundary_gaps),
                "minimum_observed_cluster_size": min(cluster_sizes),
            }
            if candidate["minimum_replicate_adjusted_rand"] < minimum_replicate_stability:
                candidate["status"] = "reject_replicate_instability"
            elif candidate["minimum_replicate_mean_silhouette"] < minimum_mean_silhouette:
                candidate["status"] = "reject_weak_mean_separation"
            elif candidate["minimum_replicate_silhouette_p10"] < minimum_silhouette_p10:
                candidate["status"] = "reject_weak_lower_tail_separation"
            elif candidate["minimum_replicate_boundary_gap_ratio"] < minimum_boundary_gap_ratio:
                candidate["status"] = "reject_continuous_boundary"
            elif candidate["minimum_observed_cluster_size"] < minimum_cluster_size:
                candidate["status"] = "reject_small_cluster"
            candidates.append(candidate)
        passing = [row for row in candidates if row["status"] == "pass"]
        selected = (
            max(
                passing,
                key=lambda row: (
                    float(row["minimum_replicate_boundary_gap_ratio"]),
                    float(row["minimum_replicate_silhouette_p10"]),
                    float(row["minimum_replicate_mean_silhouette"]),
                    float(row["minimum_replicate_adjusted_rand"]),
                    -int(row["cluster_count"]),
                ),
            )
            if passing
            else candidates[0]
        )
        allocation[category] = int(selected["cluster_count"])
        category_results[category] = {
            "component_count": len(indices),
            "candidates": candidates,
            "selection": selected,
        }
    return {
        "schema": SCHEMA,
        "status": "pass",
        "policy": {
            "replicates": len(matrices),
            "replicate_kind_counts": dict(sorted(Counter(kinds).items())),
            "maximum_clusters_per_category": maximum_clusters_per_category,
            "minimum_replicate_adjusted_rand": minimum_replicate_stability,
            "minimum_replicate_mean_silhouette": minimum_mean_silhouette,
            "minimum_replicate_silhouette_p10": minimum_silhouette_p10,
            "minimum_replicate_boundary_gap_ratio": minimum_boundary_gap_ratio,
            "minimum_cluster_size": minimum_cluster_size,
            "no_passing_nontrivial_candidate_rule": "one_cluster_for_category",
            "selection_order": [
                "maximum_minimum_replicate_boundary_gap_ratio",
                "maximum_minimum_replicate_silhouette_p10",
                "maximum_minimum_replicate_mean_silhouette",
                "maximum_minimum_replicate_adjusted_rand",
                "minimum_cluster_count",
            ],
        },
        "allocation": allocation,
        "categories": category_results,
    }
