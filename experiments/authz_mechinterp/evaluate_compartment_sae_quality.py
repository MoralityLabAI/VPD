from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
)
from experiments.authz_mechinterp.score_sae_features import encode_topk


DEFAULT_QUALITY_POLICY: dict[str, Any] = {
    "minimum_rows_per_category": 32,
    "minimum_target_explained_variance": 0.70,
    "minimum_target_cross_category_ev_advantage": 0.20,
    "maximum_dead_feature_fraction": 0.25,
    "minimum_target_selective_feature_fraction": 0.20,
    "target_selectivity_mean_activation_ratio": 2.0,
    "minimum_mean_l0": 1.0,
    "minimum_replica_decoder_cosine": 0.80,
    "minimum_replica_decoder_coverage": 0.90,
    "minimum_selective_replica_profile_cosine": 0.80,
    "minimum_selective_replication_rate": 0.50,
    "minimum_shared_category_explained_variance": 0.25,
    "maximum_shared_category_ev_range": 0.15,
    "dictionary_duplicate_cosine_reporting_threshold": 0.98,
    "minimum_replicas_per_compartment": 2,
}

DEFAULT_QUALITY_SAMPLING_POLICY: dict[str, Any] = {
    "seed": 270829,
    "episodes_per_category": 64,
    "rows_per_episode": 128,
    "selected_rows": 65536,
}
DEFAULT_SAE_DATA_SELECTION_SEED = 17


def _string_labels(categories: np.ndarray) -> np.ndarray:
    value = np.asarray(categories)
    if value.ndim != 1:
        raise ValueError("quality categories must be one-dimensional")
    if value.dtype.kind == "U":
        return value
    if value.dtype.kind == "O" and all(
        isinstance(item, str) for item in value.tolist()
    ):
        return value
    return value.astype(str)


def _explained_variance(x: np.ndarray, reconstruction: np.ndarray) -> float:
    centered = x - x.mean(axis=0, keepdims=True)
    reconstruction_centered = reconstruction - reconstruction.mean(
        axis=0, keepdims=True
    )
    residual = float(
        np.sum(
            np.square(centered - reconstruction_centered), dtype=np.float64
        )
    )
    denominator = float(np.sum(np.square(centered), dtype=np.float64))
    return 1.0 - residual / max(denominator, 1e-12)


def _normalized_rows(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    return (value / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def _decoder_duplicate_fraction(
    left_decoder: np.ndarray,
    right_decoder: np.ndarray,
    *,
    threshold: float,
    block_features: int = 256,
) -> float:
    matched = 0
    for start in range(0, len(left_decoder), block_features):
        end = min(len(left_decoder), start + block_features)
        cosine = np.abs(left_decoder[start:end] @ right_decoder.T)
        matched += int((np.max(cosine, axis=1) >= threshold).sum())
    return matched / max(1, len(left_decoder))


def streaming_quality_resource_bounds(
    *,
    hidden_size: int,
    maximum_features: int,
    categories: int,
    batch_rows: int,
    decoder_match_block_rows: int = 256,
) -> dict[str, Any]:
    if min(
        hidden_size,
        maximum_features,
        categories,
        batch_rows,
        decoder_match_block_rows,
    ) < 1:
        raise ValueError("quality resource dimensions must be positive")
    float32_bytes = 4
    float64_bytes = 8
    dictionary_array_bytes = 2 * maximum_features * hidden_size * float32_bytes
    decoder_reference_bytes = maximum_features * hidden_size * float32_bytes
    accumulator_bytes = (
        categories * maximum_features * float64_bytes
        + 2 * categories * hidden_size * float64_bytes
    )
    evaluation_peak = (
        dictionary_array_bytes
        + 2 * decoder_reference_bytes
        + 4 * batch_rows * maximum_features * float32_bytes
        + 4 * batch_rows * hidden_size * float32_bytes
        + accumulator_bytes
    )
    comparison_peak = (
        dictionary_array_bytes
        + 3 * decoder_reference_bytes
        + decoder_match_block_rows * maximum_features * float32_bytes
        + accumulator_bytes
    )
    return {
        "maximum_dictionary_features": maximum_features,
        "dictionary_arrays_bytes": dictionary_array_bytes,
        "decoder_reference_bytes_each": decoder_reference_bytes,
        "activation_batch_rows": batch_rows,
        "decoder_match_block_rows": decoder_match_block_rows,
        "estimated_numeric_peak_bytes_excluding_python_and_npz_overhead": max(
            evaluation_peak, comparison_peak
        ),
        "full_activation_matrix_materialized": False,
        "full_feature_matrix_materialized": False,
        "full_replica_cosine_matrix_materialized": False,
    }


def evaluate_dictionary(
    *,
    activations: np.ndarray,
    categories: np.ndarray,
    category_names: list[str],
    dictionary: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    decoder = np.asarray(dictionary["decoder"], dtype=np.float32)
    encoder_weight = np.asarray(dictionary["encoder_weight"], dtype=np.float32)
    encoder_bias = np.asarray(dictionary["encoder_bias"], dtype=np.float32)
    pre_bias = np.asarray(dictionary["pre_bias"], dtype=np.float32)
    top_k = int(dictionary["top_k"])
    if decoder.ndim != 2 or decoder.shape[1] != activations.shape[1]:
        raise ValueError("SAE decoder does not align with the activation space")
    if encoder_weight.shape != decoder.shape:
        raise ValueError("SAE encoder and decoder shapes differ")
    if encoder_bias.shape != (decoder.shape[0],) or pre_bias.shape != (
        decoder.shape[1],
    ):
        raise ValueError("SAE bias arrays do not align")
    if not 1 <= top_k <= decoder.shape[0]:
        raise ValueError("SAE top-k is outside the dictionary width")
    features = encode_topk(
        activations,
        pre_bias=pre_bias,
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        top_k=top_k,
    )
    reconstruction = features @ decoder + pre_bias
    ev_by_category: dict[str, float] = {}
    feature_means: list[np.ndarray] = []
    rows_by_category: dict[str, int] = {}
    for category in category_names:
        mask = categories == category
        rows = int(mask.sum())
        if rows < int(policy["minimum_rows_per_category"]):
            raise ValueError(f"quality holdout has too few rows for {category}")
        rows_by_category[category] = rows
        ev_by_category[category] = _explained_variance(
            activations[mask], reconstruction[mask]
        )
        feature_means.append(features[mask].mean(axis=0))
    category_feature_means = np.stack(feature_means, axis=1)
    dead = np.max(features, axis=0) <= 0.0
    active_l0 = np.sum(features > 0.0, axis=1)
    compartment = str(dictionary["compartment"])
    checks: dict[str, bool] = {
        "dead_feature_fraction": float(dead.mean())
        <= float(policy["maximum_dead_feature_fraction"]),
        "mean_l0": float(active_l0.mean()) >= float(policy["minimum_mean_l0"]),
    }
    target_selective = np.zeros(decoder.shape[0], dtype=bool)
    target_advantage: float | None = None
    target_ev: float | None = None
    selective_fraction: float | None = None
    if compartment == "shared_cross_category":
        ev_values = list(ev_by_category.values())
        checks.update(
            {
                "minimum_shared_category_explained_variance": min(ev_values)
                >= float(policy["minimum_shared_category_explained_variance"]),
                "shared_category_ev_range": max(ev_values) - min(ev_values)
                <= float(policy["maximum_shared_category_ev_range"]),
            }
        )
    else:
        if compartment not in category_names:
            raise ValueError(f"unknown dictionary activation compartment: {compartment}")
        target_index = category_names.index(compartment)
        target_means = category_feature_means[:, target_index]
        cross_means = np.max(
            np.delete(category_feature_means, target_index, axis=1), axis=1
        )
        ratio = float(policy["target_selectivity_mean_activation_ratio"])
        target_selective = (target_means > 0.0) & (
            target_means >= ratio * np.maximum(cross_means, 1e-12)
        )
        selective_fraction = float(target_selective.mean())
        target_ev = float(ev_by_category[compartment])
        target_advantage = target_ev - max(
            value for name, value in ev_by_category.items() if name != compartment
        )
        checks.update(
            {
                "target_explained_variance": target_ev
                >= float(policy["minimum_target_explained_variance"]),
                "target_cross_category_ev_advantage": target_advantage
                >= float(policy["minimum_target_cross_category_ev_advantage"]),
                "target_selective_feature_fraction": selective_fraction
                >= float(policy["minimum_target_selective_feature_fraction"]),
            }
        )
    checks = {name: bool(value) for name, value in checks.items()}
    public = {
        "dictionary_id": str(dictionary["dictionary_id"]),
        "compartment": compartment,
        "replica_id": str(dictionary["replica_id"]),
        "dictionary_features": decoder.shape[0],
        "top_k": top_k,
        "rows": len(activations),
        "rows_by_category": rows_by_category,
        "explained_variance_by_category": ev_by_category,
        "explained_variance_centering": "within_category_input_and_reconstruction",
        "dead_feature_fraction": float(dead.mean()),
        "mean_l0": float(active_l0.mean()),
        "target_explained_variance": target_ev,
        "target_cross_category_ev_advantage": target_advantage,
        "target_selective_feature_fraction": selective_fraction,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "reject",
    }
    private = {
        "decoder_normalized": _normalized_rows(decoder),
        "category_feature_means": category_feature_means,
        "target_selective": target_selective,
    }
    return public, private


def evaluate_dictionary_streaming(
    *,
    row_count: int,
    hidden_size: int,
    read_rows: Callable[[np.ndarray], np.ndarray],
    categories: np.ndarray,
    category_names: list[str],
    dictionary: dict[str, Any],
    policy: dict[str, Any],
    batch_rows: int = 1024,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate one SAE with bounded activation and feature working memory."""
    if row_count < 1 or hidden_size < 1 or batch_rows < 1:
        raise ValueError("streaming quality dimensions must be positive")
    labels = _string_labels(categories)
    if len(labels) != row_count:
        raise ValueError("quality activation and category rows do not align")
    decoder = np.asarray(dictionary["decoder"], dtype=np.float32)
    encoder_weight = np.asarray(dictionary["encoder_weight"], dtype=np.float32)
    encoder_bias = np.asarray(dictionary["encoder_bias"], dtype=np.float32)
    pre_bias = np.asarray(dictionary["pre_bias"], dtype=np.float32)
    top_k = int(dictionary["top_k"])
    if decoder.ndim != 2 or decoder.shape[1] != hidden_size:
        raise ValueError("SAE decoder does not align with the activation space")
    if encoder_weight.shape != decoder.shape:
        raise ValueError("SAE encoder and decoder shapes differ")
    feature_count = decoder.shape[0]
    if encoder_bias.shape != (feature_count,) or pre_bias.shape != (hidden_size,):
        raise ValueError("SAE bias arrays do not align")
    if not 1 <= top_k <= feature_count:
        raise ValueError("SAE top-k is outside the dictionary width")

    category_index = {category: index for index, category in enumerate(category_names)}
    unknown = set(labels.tolist()) - set(category_index)
    if unknown:
        raise ValueError(f"quality holdout has unknown categories: {sorted(unknown)}")
    counts = np.zeros(len(category_names), dtype=np.int64)
    input_sums = np.zeros((len(category_names), hidden_size), dtype=np.float64)
    input_square_sums = np.zeros(len(category_names), dtype=np.float64)
    residual_sums = np.zeros((len(category_names), hidden_size), dtype=np.float64)
    residual_square_sums = np.zeros(len(category_names), dtype=np.float64)
    feature_sums = np.zeros((len(category_names), feature_count), dtype=np.float64)
    feature_max = np.full(feature_count, -np.inf, dtype=np.float32)
    active_l0_sum = 0

    for start in range(0, row_count, batch_rows):
        indices = np.arange(start, min(row_count, start + batch_rows), dtype=np.int64)
        activations = np.asarray(read_rows(indices), dtype=np.float32)
        if activations.shape != (len(indices), hidden_size):
            raise ValueError("streaming activation reader returned the wrong matrix shape")
        if not np.isfinite(activations).all():
            raise ValueError("quality activations contain non-finite values")
        features = encode_topk(
            activations,
            pre_bias=pre_bias,
            encoder_weight=encoder_weight,
            encoder_bias=encoder_bias,
            top_k=top_k,
        )
        reconstruction = features @ decoder + pre_bias
        residual = activations - reconstruction
        feature_max = np.maximum(feature_max, np.max(features, axis=0))
        active_l0_sum += int((features > 0.0).sum())
        batch_labels = labels[indices]
        for category, category_position in category_index.items():
            mask = batch_labels == category
            count = int(mask.sum())
            if count == 0:
                continue
            selected = activations[mask]
            selected_residual = residual[mask]
            counts[category_position] += count
            input_sums[category_position] += selected.sum(axis=0, dtype=np.float64)
            input_square_sums[category_position] += float(
                np.square(selected, dtype=np.float64).sum(dtype=np.float64)
            )
            residual_sums[category_position] += selected_residual.sum(
                axis=0, dtype=np.float64
            )
            residual_square_sums[category_position] += float(
                np.square(selected_residual, dtype=np.float64).sum(dtype=np.float64)
            )
            feature_sums[category_position] += features[mask].sum(
                axis=0, dtype=np.float64
            )

    rows_by_category: dict[str, int] = {}
    ev_by_category: dict[str, float] = {}
    for category, position in category_index.items():
        count = int(counts[position])
        if count < int(policy["minimum_rows_per_category"]):
            raise ValueError(f"quality holdout has too few rows for {category}")
        rows_by_category[category] = count
        denominator = input_square_sums[position] - float(
            input_sums[position] @ input_sums[position]
        ) / count
        residual = residual_square_sums[position] - float(
            residual_sums[position] @ residual_sums[position]
        ) / count
        ev_by_category[category] = 1.0 - residual / max(denominator, 1e-12)

    category_feature_means = (feature_sums / counts[:, None]).T
    dead = feature_max <= 0.0
    compartment = str(dictionary["compartment"])
    checks: dict[str, bool] = {
        "dead_feature_fraction": float(dead.mean())
        <= float(policy["maximum_dead_feature_fraction"]),
        "mean_l0": active_l0_sum / row_count >= float(policy["minimum_mean_l0"]),
    }
    target_selective = np.zeros(feature_count, dtype=bool)
    target_advantage: float | None = None
    target_ev: float | None = None
    selective_fraction: float | None = None
    if compartment == "shared_cross_category":
        ev_values = list(ev_by_category.values())
        checks.update(
            {
                "minimum_shared_category_explained_variance": min(ev_values)
                >= float(policy["minimum_shared_category_explained_variance"]),
                "shared_category_ev_range": max(ev_values) - min(ev_values)
                <= float(policy["maximum_shared_category_ev_range"]),
            }
        )
    else:
        if compartment not in category_index:
            raise ValueError(f"unknown dictionary activation compartment: {compartment}")
        target_position = category_index[compartment]
        target_means = category_feature_means[:, target_position]
        cross_means = np.max(
            np.delete(category_feature_means, target_position, axis=1), axis=1
        )
        ratio = float(policy["target_selectivity_mean_activation_ratio"])
        target_selective = (target_means > 0.0) & (
            target_means >= ratio * np.maximum(cross_means, 1e-12)
        )
        selective_fraction = float(target_selective.mean())
        target_ev = float(ev_by_category[compartment])
        target_advantage = target_ev - max(
            value for name, value in ev_by_category.items() if name != compartment
        )
        checks.update(
            {
                "target_explained_variance": target_ev
                >= float(policy["minimum_target_explained_variance"]),
                "target_cross_category_ev_advantage": target_advantage
                >= float(policy["minimum_target_cross_category_ev_advantage"]),
                "target_selective_feature_fraction": selective_fraction
                >= float(policy["minimum_target_selective_feature_fraction"]),
            }
        )
    checks = {name: bool(value) for name, value in checks.items()}
    public = {
        "dictionary_id": str(dictionary["dictionary_id"]),
        "compartment": compartment,
        "replica_id": str(dictionary["replica_id"]),
        "dictionary_features": feature_count,
        "top_k": top_k,
        "rows": row_count,
        "rows_by_category": rows_by_category,
        "explained_variance_by_category": ev_by_category,
        "explained_variance_centering": "within_category_input_and_reconstruction",
        "evaluation_mode": "bounded_memory_streaming",
        "batch_rows": batch_rows,
        "dead_feature_fraction": float(dead.mean()),
        "mean_l0": active_l0_sum / row_count,
        "target_explained_variance": target_ev,
        "target_cross_category_ev_advantage": target_advantage,
        "target_selective_feature_fraction": selective_fraction,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "reject",
    }
    private = {
        "decoder_normalized": _normalized_rows(decoder),
        "category_feature_means": category_feature_means,
        "target_selective": target_selective,
    }
    return public, private


def _replica_metrics(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    policy: dict[str, Any],
    specific: bool,
) -> dict[str, Any]:
    left_decoder = left["decoder_normalized"]
    right_decoder = right["decoder_normalized"]
    left_match = np.empty(len(left_decoder), dtype=np.int64)
    left_score = np.empty(len(left_decoder), dtype=np.float64)
    right_score = np.full(len(right_decoder), -np.inf, dtype=np.float64)
    block_features = 256
    for start in range(0, len(left_decoder), block_features):
        end = min(len(left_decoder), start + block_features)
        cosine = np.abs(left_decoder[start:end] @ right_decoder.T)
        local_match = np.argmax(cosine, axis=1)
        left_match[start:end] = local_match
        left_score[start:end] = cosine[np.arange(end - start), local_match]
        right_score = np.maximum(right_score, np.max(cosine, axis=0))
    threshold = float(policy["minimum_replica_decoder_cosine"])
    left_coverage = float((left_score >= threshold).mean())
    right_coverage = float((right_score >= threshold).mean())
    decoder_coverage = min(left_coverage, right_coverage)
    selective_rate: float | None = None
    if specific:
        selective = np.flatnonzero(left["target_selective"])
        if len(selective) == 0:
            selective_rate = 0.0
        else:
            matches = left_match[selective]
            decoder_ok = left_score[selective] >= threshold
            profile_cosines = _cosine_rows(
                left["category_feature_means"][selective],
                right["category_feature_means"][matches],
            )
            selective_rate = float(
                (
                    decoder_ok
                    & (
                        profile_cosines
                        >= float(policy["minimum_selective_replica_profile_cosine"])
                    )
                ).mean()
            )
    checks = {
        "decoder_coverage": decoder_coverage
        >= float(policy["minimum_replica_decoder_coverage"]),
        **(
            {}
            if selective_rate is None
            else {
                "selective_replication": selective_rate
                >= float(policy["minimum_selective_replication_rate"])
            }
        ),
    }
    return {
        "decoder_coverage": decoder_coverage,
        "selective_replication_rate": selective_rate,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "reject",
    }


def evaluate_compartment_bundle(
    *,
    activations: np.ndarray,
    categories: np.ndarray,
    dictionaries: list[dict[str, Any]],
    required_specific_categories: list[str],
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frozen = dict(DEFAULT_QUALITY_POLICY if policy is None else policy)
    x = np.asarray(activations, dtype=np.float32)
    labels = _string_labels(categories)
    if x.ndim != 2 or len(labels) != len(x) or not np.isfinite(x).all():
        raise ValueError("quality activations are not a finite aligned matrix")
    category_names = sorted(set(labels.tolist()))
    if set(required_specific_categories) != set(category_names):
        raise ValueError("quality holdout categories differ from the required contract")
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, np.ndarray]]]] = defaultdict(list)
    dictionary_rows: list[dict[str, Any]] = []
    for dictionary in dictionaries:
        public, private = evaluate_dictionary(
            activations=x,
            categories=labels,
            category_names=category_names,
            dictionary=dictionary,
            policy=frozen,
        )
        dictionary_rows.append(public)
        grouped[public["compartment"]].append((public, private))
    required_compartments = set(category_names) | {"shared_cross_category"}
    if set(grouped) != required_compartments:
        raise ValueError("dictionary bundle is not an exact compartment partition")
    replica_rows: list[dict[str, Any]] = []
    compartment_status: dict[str, str] = {}
    for compartment in sorted(grouped):
        replicas = sorted(grouped[compartment], key=lambda row: row[0]["replica_id"])
        if len(replicas) < int(frozen["minimum_replicas_per_compartment"]):
            raise ValueError(f"too few dictionary replicas for {compartment}")
        reference = replicas[0]
        comparisons = []
        for public, private in replicas[1:]:
            metrics = _replica_metrics(
                reference[1],
                private,
                policy=frozen,
                specific=compartment != "shared_cross_category",
            )
            row = {
                "compartment": compartment,
                "reference_replica_id": reference[0]["replica_id"],
                "replica_id": public["replica_id"],
                **metrics,
            }
            replica_rows.append(row)
            comparisons.append(row)
        passes = all(public["status"] == "pass" for public, _ in replicas) and all(
            row["status"] == "pass" for row in comparisons
        )
        compartment_status[compartment] = "pass" if passes else "reject"
    shared_reference = sorted(
        grouped["shared_cross_category"], key=lambda row: row[0]["replica_id"]
    )[0][1]
    duplicate_rows = []
    duplicate_threshold = float(
        frozen["dictionary_duplicate_cosine_reporting_threshold"]
    )
    for compartment in category_names:
        specific_reference = sorted(
            grouped[compartment], key=lambda row: row[0]["replica_id"]
        )[0][1]
        duplicate_rows.append(
            {
                "compartment": compartment,
                "specific_features_with_shared_match_fraction": float(
                    _decoder_duplicate_fraction(
                        specific_reference["decoder_normalized"],
                        shared_reference["decoder_normalized"],
                        threshold=duplicate_threshold,
                    )
                ),
                "reporting_threshold": duplicate_threshold,
                "gating_metric": False,
            }
        )
    status = (
        "pass"
        if all(value == "pass" for value in compartment_status.values())
        else "reject"
    )
    return {
        "schema": "sch.vpd-compartment-sae-quality-result.v1",
        "status": status,
        "policy": frozen,
        "coverage": {
            "rows": len(x),
            "hidden_size": x.shape[1],
            "categories": category_names,
            "compartments": sorted(grouped),
            "dictionaries": len(dictionaries),
            "replicas_per_compartment": {
                name: len(rows) for name, rows in sorted(grouped.items())
            },
        },
        "dictionary_results": sorted(
            dictionary_rows,
            key=lambda row: (row["compartment"], row["replica_id"]),
        ),
        "replica_results": replica_rows,
        "shared_duplicate_diagnostics": duplicate_rows,
        "compartment_status": compartment_status,
        "scientific_boundary": {
            "quality_screen_is_not_causal_evidence": True,
            "category_specificity_requires_heldout_interventions": True,
            "final_evaluation_must_not_select_quality_thresholds": True,
        },
    }


def evaluate_compartment_bundle_streaming(
    *,
    row_count: int,
    hidden_size: int,
    read_rows: Callable[[np.ndarray], np.ndarray],
    categories: np.ndarray,
    dictionary_descriptors: list[dict[str, Any]],
    load_dictionary: Callable[[dict[str, Any]], dict[str, Any]],
    required_specific_categories: list[str],
    policy: dict[str, Any] | None = None,
    batch_rows: int = 1024,
) -> dict[str, Any]:
    """Evaluate a replica bundle without retaining every dictionary in memory."""
    frozen = dict(DEFAULT_QUALITY_POLICY if policy is None else policy)
    labels = _string_labels(categories)
    if len(labels) != row_count:
        raise ValueError("quality activation and category rows do not align")
    category_names = sorted(set(labels.tolist()))
    if set(required_specific_categories) != set(category_names):
        raise ValueError("quality holdout categories differ from the required contract")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, str]] = set()
    for descriptor in dictionary_descriptors:
        compartment = str(descriptor["compartment"])
        replica_id = str(descriptor["replica_id"])
        identity = (compartment, replica_id)
        if identity in identities:
            raise ValueError("quality bundle contains a duplicate compartment replica")
        identities.add(identity)
        grouped[compartment].append(dict(descriptor))
    required_compartments = set(category_names) | {"shared_cross_category"}
    if set(grouped) != required_compartments:
        raise ValueError("dictionary bundle is not an exact compartment partition")

    dictionary_rows: list[dict[str, Any]] = []
    replica_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    compartment_status: dict[str, str] = {}
    shared_reference: dict[str, np.ndarray] | None = None
    duplicate_threshold = float(
        frozen["dictionary_duplicate_cosine_reporting_threshold"]
    )
    evaluation_order = ["shared_cross_category", *category_names]
    for compartment in evaluation_order:
        descriptors = sorted(grouped[compartment], key=lambda row: str(row["replica_id"]))
        if len(descriptors) < int(frozen["minimum_replicas_per_compartment"]):
            raise ValueError(f"too few dictionary replicas for {compartment}")
        reference_public: dict[str, Any] | None = None
        reference_private: dict[str, np.ndarray] | None = None
        comparisons: list[dict[str, Any]] = []
        public_rows: list[dict[str, Any]] = []
        for descriptor in descriptors:
            dictionary = load_dictionary(descriptor)
            for field in ("dictionary_id", "compartment", "replica_id"):
                if str(dictionary.get(field)) != str(descriptor.get(field)):
                    raise ValueError(f"quality dictionary loader changed {field}")
            public, private = evaluate_dictionary_streaming(
                row_count=row_count,
                hidden_size=hidden_size,
                read_rows=read_rows,
                categories=labels,
                category_names=category_names,
                dictionary=dictionary,
                policy=frozen,
                batch_rows=batch_rows,
            )
            dictionary_rows.append(public)
            public_rows.append(public)
            if reference_public is None:
                reference_public = public
                reference_private = private
                continue
            assert reference_private is not None
            metrics = _replica_metrics(
                reference_private,
                private,
                policy=frozen,
                specific=compartment != "shared_cross_category",
            )
            comparison = {
                "compartment": compartment,
                "reference_replica_id": reference_public["replica_id"],
                "replica_id": public["replica_id"],
                **metrics,
            }
            comparisons.append(comparison)
            replica_rows.append(comparison)
        assert reference_private is not None
        passes = all(row["status"] == "pass" for row in public_rows) and all(
            row["status"] == "pass" for row in comparisons
        )
        compartment_status[compartment] = "pass" if passes else "reject"
        if compartment == "shared_cross_category":
            shared_reference = reference_private
        else:
            assert shared_reference is not None
            duplicate_rows.append(
                {
                    "compartment": compartment,
                    "specific_features_with_shared_match_fraction": float(
                        _decoder_duplicate_fraction(
                            reference_private["decoder_normalized"],
                            shared_reference["decoder_normalized"],
                            threshold=duplicate_threshold,
                        )
                    ),
                    "reporting_threshold": duplicate_threshold,
                    "gating_metric": False,
                }
            )
    status = (
        "pass"
        if all(value == "pass" for value in compartment_status.values())
        else "reject"
    )
    maximum_features = max(int(row["dictionary_features"]) for row in dictionary_rows)
    resource_bounds = streaming_quality_resource_bounds(
        hidden_size=hidden_size,
        maximum_features=maximum_features,
        categories=len(category_names),
        batch_rows=batch_rows,
    )
    return {
        "schema": "sch.vpd-compartment-sae-quality-result.v1",
        "status": status,
        "policy": frozen,
        "coverage": {
            "rows": row_count,
            "hidden_size": hidden_size,
            "categories": category_names,
            "compartments": sorted(grouped),
            "dictionaries": len(dictionary_descriptors),
            "replicas_per_compartment": {
                name: len(rows) for name, rows in sorted(grouped.items())
            },
            "evaluation_mode": "bounded_memory_streaming",
            "batch_rows": batch_rows,
            "dictionary_loading": "one_replica_at_a_time",
            "decoder_matching": "feature_blocks_of_256",
            "resource_bounds": resource_bounds,
        },
        "dictionary_results": sorted(
            dictionary_rows,
            key=lambda row: (row["compartment"], row["replica_id"]),
        ),
        "replica_results": sorted(
            replica_rows,
            key=lambda row: (row["compartment"], row["replica_id"]),
        ),
        "shared_duplicate_diagnostics": sorted(
            duplicate_rows, key=lambda row: row["compartment"]
        ),
        "compartment_status": dict(sorted(compartment_status.items())),
        "scientific_boundary": {
            "quality_screen_is_not_causal_evidence": True,
            "category_specificity_requires_heldout_interventions": True,
            "final_evaluation_must_not_select_quality_thresholds": True,
        },
    }


def validate_quality_result_consistency(result: dict[str, Any]) -> None:
    if result.get("schema") != "sch.vpd-compartment-sae-quality-result.v1":
        raise ValueError("unsupported compartment SAE quality result schema")
    if result.get("policy") != DEFAULT_QUALITY_POLICY:
        raise ValueError("quality result did not use the frozen quality policy")
    policy = result["policy"]
    dictionary_rows = result.get("dictionary_results")
    replica_rows = result.get("replica_results")
    if not isinstance(dictionary_rows, list) or not isinstance(replica_rows, list):
        raise ValueError("quality result rows are malformed")
    if len({str(row["dictionary_id"]) for row in dictionary_rows}) != len(
        dictionary_rows
    ):
        raise ValueError("quality result contains duplicate dictionary IDs")
    grouped_dictionaries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dictionary_rows:
        compartment = str(row["compartment"])
        grouped_dictionaries[compartment].append(row)
        expected_checks = {
            "dead_feature_fraction": float(row["dead_feature_fraction"])
            <= float(policy["maximum_dead_feature_fraction"]),
            "mean_l0": float(row["mean_l0"])
            >= float(policy["minimum_mean_l0"]),
        }
        ev = {str(name): float(value) for name, value in row[
            "explained_variance_by_category"
        ].items()}
        if compartment == "shared_cross_category":
            expected_checks.update(
                {
                    "minimum_shared_category_explained_variance": min(ev.values())
                    >= float(policy["minimum_shared_category_explained_variance"]),
                    "shared_category_ev_range": max(ev.values()) - min(ev.values())
                    <= float(policy["maximum_shared_category_ev_range"]),
                }
            )
        else:
            expected_checks.update(
                {
                    "target_explained_variance": float(
                        row["target_explained_variance"]
                    )
                    >= float(policy["minimum_target_explained_variance"]),
                    "target_cross_category_ev_advantage": float(
                        row["target_cross_category_ev_advantage"]
                    )
                    >= float(policy["minimum_target_cross_category_ev_advantage"]),
                    "target_selective_feature_fraction": float(
                        row["target_selective_feature_fraction"]
                    )
                    >= float(policy["minimum_target_selective_feature_fraction"]),
                }
            )
        if row.get("checks") != expected_checks:
            raise ValueError("quality dictionary checks contradict recorded metrics")
        expected_status = "pass" if all(expected_checks.values()) else "reject"
        if row.get("status") != expected_status:
            raise ValueError("quality dictionary status contradicts recorded metrics")
    categories = set(result.get("coverage", {}).get("categories", []))
    required_compartments = categories | {"shared_cross_category"}
    if set(grouped_dictionaries) != required_compartments:
        raise ValueError("quality result is not an exact compartment partition")
    for compartment, rows in grouped_dictionaries.items():
        if len(rows) < int(policy["minimum_replicas_per_compartment"]):
            raise ValueError(f"quality result has too few replicas for {compartment}")

    grouped_replicas: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in replica_rows:
        checks = {
            "decoder_coverage": float(row["decoder_coverage"])
            >= float(policy["minimum_replica_decoder_coverage"])
        }
        selective_rate = row.get("selective_replication_rate")
        if selective_rate is not None:
            checks["selective_replication"] = float(selective_rate) >= float(
                policy["minimum_selective_replication_rate"]
            )
        if row.get("checks") != checks:
            raise ValueError("quality replica checks contradict recorded metrics")
        expected_status = "pass" if all(checks.values()) else "reject"
        if row.get("status") != expected_status:
            raise ValueError("quality replica status contradicts recorded metrics")
        grouped_replicas[str(row["compartment"])].append(row)
    compartment_status = result.get("compartment_status", {})
    if set(compartment_status) != required_compartments:
        raise ValueError("quality result compartment statuses are incomplete")
    for compartment, rows in grouped_dictionaries.items():
        expected_replica_rows = len(rows) - 1
        if len(grouped_replicas.get(compartment, [])) != expected_replica_rows:
            raise ValueError("quality result replica comparisons are incomplete")
        passed = all(row["status"] == "pass" for row in rows) and all(
            row["status"] == "pass" for row in grouped_replicas[compartment]
        )
        if compartment_status[compartment] != ("pass" if passed else "reject"):
            raise ValueError("quality compartment status contradicts its rows")
    expected_status = (
        "pass"
        if all(value == "pass" for value in compartment_status.values())
        else "reject"
    )
    if result.get("status") != expected_status:
        raise ValueError("quality result status contradicts its compartments")
    boundary = result.get("scientific_boundary", {})
    if not all(
        boundary.get(field) is True
        for field in (
            "quality_screen_is_not_causal_evidence",
            "category_specificity_requires_heldout_interventions",
            "final_evaluation_must_not_select_quality_thresholds",
        )
    ):
        raise ValueError("quality result weakens its scientific boundary")


def validate_quality_sampling_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("schema") != "sch.vpd-compartment-sae-quality-sampling.v1":
        raise ValueError("quality report has an invalid sampling receipt")
    for field, expected in DEFAULT_QUALITY_SAMPLING_POLICY.items():
        if int(receipt.get(field, -1)) != expected:
            raise ValueError("quality report weakens the frozen sampling policy")
    if int(receipt.get("selected_episodes", -1)) != (
        DEFAULT_QUALITY_SAMPLING_POLICY["episodes_per_category"] * 8
    ):
        raise ValueError("quality report sampling has incomplete category coverage")
    if receipt.get("observable_activation_category_only") is not True or receipt.get(
        "model_outcomes_used_for_selection"
    ) is not False:
        raise ValueError("quality report sampling used a forbidden selection signal")
    if receipt.get("whole_prompt_windows_only") is not True or receipt.get(
        "final_evaluation_rows_used"
    ) is not False:
        raise ValueError("quality report sampling crossed its split/window boundary")
    for field in (
        "selected_episode_id_set_sha256_by_category",
        "selected_pair_id_set_sha256_by_category",
    ):
        values = receipt.get(field)
        if not isinstance(values, dict) or len(values) != 8 or any(
            len(str(value)) != 64 for value in values.values()
        ):
            raise ValueError("quality report sampling has incomplete identity hashes")
    if len(str(receipt.get("selected_row_index_sha256", ""))) != 64:
        raise ValueError("quality report sampling lacks its selected-row hash")


def build_quality_report(
    *,
    result: dict[str, Any],
    dictionary_sources: list[dict[str, Any]],
    activation_category_split: dict[str, Any],
    holdout_capture: dict[str, Any],
    training_provenance: dict[str, Any],
    evaluator_path: Path | None = None,
) -> dict[str, Any]:
    """Bind a quality result to calibration data and immutable dictionary artifacts."""
    validate_quality_result_consistency(result)
    if activation_category_split.get("activation_category_split_role") != "calibration":
        raise ValueError("compartment SAE quality must use the calibration split role")
    split_id = str(
        activation_category_split.get("activation_category_split_id", "")
    )
    if len(split_id) != 64:
        raise ValueError("compartment SAE quality lacks a split content address")
    for field in ("activation_capture_sha256", "capture_manifest_sha256"):
        value = str(holdout_capture.get(field, ""))
        if len(value) != 64:
            raise ValueError(f"compartment SAE quality lacks {field}")
    sampling = holdout_capture.get("quality_sampling")
    if not isinstance(sampling, dict):
        raise ValueError("compartment SAE quality lacks a sampling receipt")
    validate_quality_sampling_receipt(sampling)
    if training_provenance.get("schema") != (
        "sch.vpd-compartment-sae-dictionary-bundle-authorization.v1"
    ) or training_provenance.get("status") != "pass":
        raise ValueError("compartment SAE quality lacks bundle admission")
    for field in (
        "training_plan_id",
        "training_plan_sha256",
        "dictionary_bundle_id",
        "dictionary_bundle_sha256",
    ):
        if len(str(training_provenance.get(field, ""))) != 64:
            raise ValueError("compartment SAE quality has invalid training provenance")
    result_rows = {
        str(row["dictionary_id"]): row for row in result["dictionary_results"]
    }
    if len(result_rows) != len(result["dictionary_results"]):
        raise ValueError("quality result contains duplicate dictionary IDs")
    sources = sorted(
        (dict(row) for row in dictionary_sources),
        key=lambda row: str(row["dictionary_id"]),
    )
    if {str(row.get("dictionary_id")) for row in sources} != set(result_rows):
        raise ValueError("quality report sources do not cover the evaluated dictionaries")
    for source in sources:
        dictionary_id = str(source["dictionary_id"])
        result_row = result_rows[dictionary_id]
        for field in ("dictionary_manifest_sha256", "dictionary_sha256"):
            if len(str(source.get(field, ""))) != 64:
                raise ValueError(f"quality dictionary source lacks {field}")
        if str(source.get("compartment")) != str(result_row["compartment"]):
            raise ValueError("quality dictionary source compartment mismatch")
        if str(source.get("replica_id")) != str(result_row["replica_id"]):
            raise ValueError("quality dictionary source replica mismatch")
        initialization_seed = int(source.get("initialization_seed", -1))
        if initialization_seed < 0 or str(source["replica_id"]) != (
            f"seed-{initialization_seed}"
        ):
            raise ValueError("quality dictionary source initialization seed mismatch")
        if int(source.get("data_selection_seed", -1)) != (
            DEFAULT_SAE_DATA_SELECTION_SEED
        ):
            raise ValueError("quality dictionary source changed its data selection")
        if source.get("capacity_authorization") is None:
            raise ValueError("quality dictionary source lacks capacity authorization")
    source_path = Path(__file__) if evaluator_path is None else evaluator_path
    body = {
        "schema": "sch.vpd-compartment-sae-quality-report.v1",
        "status": str(result["status"]),
        "source": {
            "activation_category_split": dict(activation_category_split),
            "holdout_capture": dict(holdout_capture),
            "training_provenance": dict(training_provenance),
            "evaluator_sha256": sha256_file(source_path),
            "runner_sha256": sha256_file(
                Path(__file__).with_name("run_compartment_sae_quality.py")
            ),
        },
        "dictionaries": sources,
        "result": result,
        "scientific_boundary": {
            "calibration_quality_screen_only": True,
            "final_evaluation_not_used_for_selection": True,
            "heldout_interventions_still_required": True,
        },
    }
    return {**body, "quality_report_id": content_sha256(body)}


def verify_quality_report(path: Path, *, require_pass: bool = True) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "sch.vpd-compartment-sae-quality-report.v1":
        raise ValueError("unsupported compartment SAE quality report schema")
    body = dict(report)
    report_id = str(body.pop("quality_report_id", ""))
    if content_sha256(body) != report_id:
        raise ValueError("compartment SAE quality report content address mismatch")
    result = report.get("result")
    if not isinstance(result, dict) or result.get("schema") != (
        "sch.vpd-compartment-sae-quality-result.v1"
    ):
        raise ValueError("quality report has an invalid result")
    if str(report.get("status")) != str(result.get("status")):
        raise ValueError("quality report and result statuses differ")
    if result.get("policy") != DEFAULT_QUALITY_POLICY:
        raise ValueError("quality report did not use the frozen quality policy")
    if str(report.get("source", {}).get("evaluator_sha256", "")) != sha256_file(
        Path(__file__)
    ):
        raise ValueError("quality report was produced by a different evaluator")
    if str(report.get("source", {}).get("runner_sha256", "")) != sha256_file(
        Path(__file__).with_name("run_compartment_sae_quality.py")
    ):
        raise ValueError("quality report was produced by a different runner")
    validate_quality_result_consistency(result)
    split = report.get("source", {}).get("activation_category_split", {})
    if split.get("activation_category_split_role") != "calibration":
        raise ValueError("quality report did not use the calibration split role")
    if len(str(split.get("activation_category_split_id", ""))) != 64:
        raise ValueError("quality report has an invalid split content address")
    holdout = report.get("source", {}).get("holdout_capture", {})
    if any(
        len(str(holdout.get(field, ""))) != 64
        for field in ("activation_capture_sha256", "capture_manifest_sha256")
    ):
        raise ValueError("quality report has invalid holdout capture provenance")
    sampling = holdout.get("quality_sampling")
    if not isinstance(sampling, dict):
        raise ValueError("quality report lacks its sampling receipt")
    validate_quality_sampling_receipt(sampling)
    training_provenance = report.get("source", {}).get("training_provenance", {})
    if training_provenance.get("schema") != (
        "sch.vpd-compartment-sae-dictionary-bundle-authorization.v1"
    ) or training_provenance.get("status") != "pass":
        raise ValueError("quality report lacks bundle admission")
    if any(
        len(str(training_provenance.get(field, ""))) != 64
        for field in (
            "training_plan_id",
            "training_plan_sha256",
            "dictionary_bundle_id",
            "dictionary_bundle_sha256",
        )
    ):
        raise ValueError("quality report has invalid training provenance")
    result_rows = {
        str(row["dictionary_id"]): row for row in result["dictionary_results"]
    }
    sources = report.get("dictionaries")
    if not isinstance(sources, list) or {
        str(row.get("dictionary_id")) for row in sources
    } != set(result_rows):
        raise ValueError("quality report dictionary coverage mismatch")
    if len(sources) != len(result_rows):
        raise ValueError("quality report contains duplicate dictionary sources")
    for source in sources:
        row = result_rows[str(source["dictionary_id"])]
        if any(
            len(str(source.get(field, ""))) != 64
            for field in ("dictionary_manifest_sha256", "dictionary_sha256")
        ):
            raise ValueError("quality report contains an invalid dictionary digest")
        capacity = source.get("capacity_authorization")
        if not isinstance(capacity, dict) or capacity.get("status") != "pass":
            raise ValueError("quality report dictionary lacks capacity authorization")
        if str(source.get("compartment")) != str(row["compartment"]) or str(
            source.get("replica_id")
        ) != str(row["replica_id"]):
            raise ValueError("quality report dictionary identity mismatch")
        initialization_seed = int(source.get("initialization_seed", -1))
        if initialization_seed < 0 or str(source.get("replica_id")) != (
            f"seed-{initialization_seed}"
        ):
            raise ValueError("quality report dictionary initialization seed mismatch")
        if int(source.get("data_selection_seed", -1)) != (
            DEFAULT_SAE_DATA_SELECTION_SEED
        ):
            raise ValueError("quality report dictionaries changed data selection")
    source_seeds: dict[str, set[int]] = defaultdict(set)
    for source in sources:
        source_seeds[str(source["compartment"])].add(
            int(source["initialization_seed"])
        )
    for compartment, seeds in source_seeds.items():
        expected_replicas = int(
            result["coverage"]["replicas_per_compartment"][compartment]
        )
        if len(seeds) != expected_replicas:
            raise ValueError("quality report compartment initialization seeds repeat")
    boundary = report.get("scientific_boundary", {})
    if not all(
        boundary.get(field) is True
        for field in (
            "calibration_quality_screen_only",
            "final_evaluation_not_used_for_selection",
            "heldout_interventions_still_required",
        )
    ):
        raise ValueError("quality report weakens its scientific boundary")
    if require_pass and report["status"] != "pass":
        raise ValueError("compartment SAE quality report did not pass")
    return report
