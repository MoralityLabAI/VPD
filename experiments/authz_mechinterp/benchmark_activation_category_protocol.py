from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.activation_categories import category_contract
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    build_bundle,
    classify_activation_compartments,
    content_sha256,
    load_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from experiments.authz_mechinterp.effect_uncertainty import countsketch
from experiments.authz_mechinterp.benchmark_effect_sketch_fidelity import (
    verify_benchmark as verify_sketch_fidelity_benchmark,
)
from experiments.authz_mechinterp.benchmark_activation_category_clustering_geometry import (
    verify_benchmark as verify_clustering_geometry_benchmark,
)
from experiments.authz_mechinterp.benchmark_activation_category_cluster_count_selection import (
    verify_benchmark as verify_cluster_count_benchmark,
)
from experiments.authz_mechinterp.merge_vpd_effect_shards import write_deterministic_npz


SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "id": "clean_cross_template",
        "target_strength": 2.0,
        "background_strength": 0.08,
        "block_noise": 0.03,
        "calibration_direction_cosine": 0.98,
        "final_direction_cosine": 0.95,
        "calibration_label_shift": "none",
        "expectation": "recover",
        "minimum_specific_precision": 0.99,
        "minimum_specific_recall": 0.99,
        "minimum_shared_recall": 0.99,
        "minimum_final_accuracy": 0.99,
    },
    {
        "id": "moderate_noise_and_shift",
        "target_strength": 1.6,
        "background_strength": 0.14,
        "block_noise": 0.30,
        "calibration_direction_cosine": 0.80,
        "final_direction_cosine": 0.72,
        "calibration_label_shift": "none",
        "expectation": "recover",
        "minimum_specific_precision": 0.95,
        "minimum_specific_recall": 0.90,
        "minimum_shared_recall": 0.95,
        "minimum_final_accuracy": 0.90,
    },
    {
        "id": "high_pair_block_noise",
        "target_strength": 1.35,
        "background_strength": 0.16,
        "block_noise": 1.20,
        "calibration_direction_cosine": 0.75,
        "final_direction_cosine": 0.70,
        "calibration_label_shift": "none",
        "expectation": "fail_closed_or_recover",
        "minimum_specific_precision": 0.95,
        "minimum_shared_recall": 0.95,
    },
    {
        "id": "direction_reversal",
        "target_strength": 2.0,
        "background_strength": 0.08,
        "block_noise": 0.03,
        "calibration_direction_cosine": -1.0,
        "final_direction_cosine": -1.0,
        "calibration_label_shift": "none",
        "expectation": "reject_calibration",
    },
    {
        "id": "calibration_magnitude_collapse",
        "target_strength": 2.0,
        "background_strength": 0.08,
        "block_noise": 0.03,
        "calibration_direction_cosine": 0.98,
        "calibration_magnitude_scale": 0.02,
        "final_direction_cosine": 0.95,
        "calibration_label_shift": "none",
        "expectation": "reject_calibration",
    },
    {
        "id": "category_label_swap",
        "target_strength": 2.0,
        "background_strength": 0.08,
        "block_noise": 0.03,
        "calibration_direction_cosine": 0.98,
        "final_direction_cosine": 0.95,
        "calibration_label_shift": "cyclic",
        "expectation": "reject_calibration",
    },
    {
        "id": "minority_cross_archive_category_disagreement",
        "target_strength": 2.0,
        "background_strength": 0.08,
        "block_noise": 0.03,
        "calibration_direction_cosine": 0.98,
        "final_direction_cosine": 0.95,
        "calibration_label_shift": "none",
        "calibration_minority_label_shift_count": 8,
        "expectation": "cross_archive_component_reroute",
        "minimum_specific_precision": 0.99,
        "minimum_shared_recall": 0.99,
        "minimum_cross_archive_reroutes": 8,
    },
    {
        "id": "single_condition_category_artifact",
        "target_strength": 2.0,
        "background_strength": 0.08,
        "block_noise": 0.03,
        "calibration_direction_cosine": 0.98,
        "final_direction_cosine": 0.95,
        "calibration_label_shift": "none",
        "narrow_condition_category": "replayed",
        "discovery_artifact_component_limit": 8,
        "expectation": "condition_reroute",
        "minimum_specific_precision": 0.99,
        "minimum_shared_recall": 0.99,
        "minimum_condition_reroutes": 8,
    },
    {
        "id": "single_template_category_artifact",
        "target_strength": 2.0,
        "background_strength": 0.08,
        "block_noise": 0.03,
        "calibration_direction_cosine": 0.98,
        "final_direction_cosine": 0.95,
        "calibration_label_shift": "none",
        "narrow_template_category": "replayed",
        "discovery_artifact_component_limit": 8,
        "expectation": "template_reroute",
        "minimum_specific_precision": 0.99,
        "minimum_shared_recall": 0.99,
        "minimum_template_reroutes": 8,
    },
    {
        "id": "opposing_pair_block_directions",
        "target_strength": 2.0,
        "background_strength": 0.08,
        "block_noise": 0.03,
        "calibration_direction_cosine": 0.98,
        "final_direction_cosine": 0.95,
        "calibration_label_shift": "none",
        "opposing_pair_block_category": "replayed",
        "discovery_artifact_component_limit": 8,
        "expectation": "pair_block_direction_reroute",
        "minimum_specific_precision": 0.99,
        "minimum_shared_recall": 0.99,
        "minimum_pair_block_direction_reroutes": 8,
    },
    {
        "id": "calibration_single_condition_category_artifact",
        "target_strength": 2.0,
        "background_strength": 0.08,
        "block_noise": 0.03,
        "calibration_direction_cosine": 0.98,
        "final_direction_cosine": 0.95,
        "calibration_label_shift": "none",
        "calibration_narrow_condition_category": "replayed",
        "expectation": "reject_calibration",
    },
    {
        "id": "calibration_single_template_category_artifact",
        "target_strength": 2.0,
        "background_strength": 0.08,
        "block_noise": 0.03,
        "calibration_direction_cosine": 0.98,
        "final_direction_cosine": 0.95,
        "calibration_label_shift": "none",
        "calibration_narrow_template_category": "replayed",
        "expectation": "reject_calibration",
    },
    {
        "id": "calibration_opposing_pair_block_directions",
        "target_strength": 2.0,
        "background_strength": 0.08,
        "block_noise": 0.03,
        "calibration_direction_cosine": 0.98,
        "final_direction_cosine": 0.95,
        "calibration_label_shift": "none",
        "calibration_opposing_pair_block_category": "replayed",
        "expectation": "reject_calibration",
    },
)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def synthetic_inventory(
    *, specific_per_category: int = 12, shared_components: int = 32
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    categories = sorted(row["id"] for row in category_contract()["categories"])
    truths = [
        category
        for category in categories
        for _ in range(specific_per_category)
    ] + ["shared_cross_category"] * shared_components
    families = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    rows: list[dict[str, Any]] = []
    for index, truth in enumerate(truths):
        family = families[index % len(families)]
        layer = index % 36
        rows.append(
            {
                "component_id": f"model.layers.{layer}.{family}:rank{index}",
                "module": f"model.layers.{layer}.{family}",
                "layer": layer,
                "rank_index": index,
                "projection_family": family,
                "component_fro_norm": 0.05 + (index % 17) * 0.01,
                "fraction_of_module_delta_fro_sq": 0.01 + (index % 11) * 0.005,
                "synthetic_ground_truth_compartment": truth,
            }
        )
    return rows, truths, categories


def _rotated_directions(
    directions: np.ndarray, *, cosine: float, rng: np.random.Generator
) -> np.ndarray:
    if not -1.0 <= cosine <= 1.0:
        raise ValueError("direction cosine must be in [-1, 1]")
    if abs(cosine) == 1.0:
        return directions * cosine
    noise = rng.normal(size=directions.shape)
    noise -= np.sum(noise * directions, axis=-1, keepdims=True) * directions
    noise = _unit_rows(noise)
    return cosine * directions + math.sqrt(1.0 - cosine**2) * noise


def category_means(
    *,
    truths: list[str],
    categories: list[str],
    directions: np.ndarray,
    target_strength: float,
    background_strength: float,
    label_shift: str,
) -> np.ndarray:
    result = np.empty_like(directions, dtype=np.float64)
    category_index = {category: index for index, category in enumerate(categories)}
    for component_index, truth in enumerate(truths):
        for effect_index, category in enumerate(categories):
            if truth == "shared_cross_category":
                strength = 1.0
            else:
                target = category_index[truth]
                if label_shift == "cyclic":
                    target = (target + 1) % len(categories)
                elif label_shift != "none":
                    raise ValueError("unknown synthetic label shift")
                strength = (
                    target_strength if effect_index == target else background_strength
                )
            result[component_index, effect_index] = (
                directions[component_index, effect_index] * strength
            )
    return result


def write_effect_archive(
    path: Path,
    *,
    component_ids: list[str],
    category_names: list[str],
    category_counts: np.ndarray,
    category_delta: np.ndarray,
    block_noise: float,
    seed: int,
    truths: list[str],
    narrow_condition_category: str | None = None,
    narrow_template_category: str | None = None,
    opposing_pair_block_category: str | None = None,
    artifact_component_limit: int | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    block_count = 8
    sketch_dimension = 256
    hidden_size = category_delta.shape[2]
    if artifact_component_limit is not None and artifact_component_limit < 1:
        raise ValueError("synthetic artifact component limit must be positive")

    def affected_component_indices(category: str | None) -> set[int]:
        if category is None:
            return set()
        matches = [index for index, truth in enumerate(truths) if truth == category]
        return set(
            matches
            if artifact_component_limit is None
            else matches[:artifact_component_limit]
        )

    if np.any(category_counts % block_count != 0):
        raise ValueError("synthetic category counts must divide evenly into blocks")
    block_counts = np.repeat(
        (category_counts // block_count)[:, None], block_count, axis=1
    ).astype(np.int64)
    block_means = np.repeat(category_delta[:, :, None, :], block_count, axis=2)
    if opposing_pair_block_category is not None:
        target_index = category_names.index(opposing_pair_block_category)
        affected = affected_component_indices(opposing_pair_block_category)
        multipliers = np.asarray([2.0] * 5 + [-2.0 / 3.0] * 3)
        for component_index, truth in enumerate(truths):
            if component_index in affected:
                block_means[component_index, target_index] *= multipliers[:, None]
    noise = rng.normal(
        0.0,
        block_noise / math.sqrt(hidden_size),
        size=block_means.shape,
    )
    noise -= noise.mean(axis=2, keepdims=True)
    block_means = block_means + noise
    sketch_seed = 1907
    sketches = countsketch(
        block_means, dimension=sketch_dimension, seed=sketch_seed
    )
    block_sums = sketches * block_counts[None, :, :, None]
    condition_names = [
        f"{category}::condition-{suffix}"
        for category in category_names
        for suffix in ("a", "b")
    ]
    condition_categories = [
        category for category in category_names for _ in range(2)
    ]
    condition_counts = np.asarray(
        [int(category_counts[index] // 2) for index in range(len(category_names)) for _ in range(2)],
        dtype=np.int64,
    )
    if np.any(condition_counts <= 0) or any(
        int(value) % 2 for value in category_counts
    ):
        raise ValueError("synthetic category counts must split over two conditions")
    condition_means = np.repeat(category_delta, 2, axis=1)
    if narrow_condition_category is not None:
        target_index = category_names.index(narrow_condition_category)
        affected = affected_component_indices(narrow_condition_category)
        for component_index, truth in enumerate(truths):
            if component_index in affected:
                condition_means[component_index, target_index * 2] *= 2.0
                condition_means[component_index, target_index * 2 + 1] = 0.0
    condition_sums = countsketch(
        condition_means, dimension=sketch_dimension, seed=sketch_seed
    ) * condition_counts[None, :, None]
    template_groups = [f"template-{index}" for index in range(4)]
    template_cell_names = [
        f"{category}::{template_group}"
        for category in category_names
        for template_group in template_groups
    ]
    template_cell_categories = [
        category for category in category_names for _ in template_groups
    ]
    template_cell_groups = [
        template_group for _ in category_names for template_group in template_groups
    ]
    if np.any(category_counts % len(template_groups) != 0):
        raise ValueError("synthetic category counts must split over four templates")
    template_cell_counts = np.asarray(
        [
            int(category_counts[index] // len(template_groups))
            for index in range(len(category_names))
            for _ in template_groups
        ],
        dtype=np.int64,
    )
    template_means = np.repeat(category_delta, len(template_groups), axis=1)
    if narrow_template_category is not None:
        target_index = category_names.index(narrow_template_category)
        affected = affected_component_indices(narrow_template_category)
        first_cell = target_index * len(template_groups)
        for component_index, truth in enumerate(truths):
            if component_index in affected:
                template_means[component_index, first_cell] *= len(template_groups)
                template_means[
                    component_index, first_cell + 1 : first_cell + len(template_groups)
                ] = 0.0
    template_sums = countsketch(
        template_means, dimension=sketch_dimension, seed=sketch_seed
    ) * template_cell_counts[None, :, None]
    aggregate = np.average(category_delta, axis=1, weights=category_counts)
    max_component = max(len(value) for value in component_ids)
    max_category = max(len(value) for value in category_names)
    write_deterministic_npz(
        path,
        {
            "component_ids": np.asarray(component_ids, dtype=f"<U{max_component}"),
            "delta_h": aggregate.astype(np.float32),
            "category_names": np.asarray(category_names, dtype=f"<U{max_category}"),
            "category_episode_counts": category_counts.astype(np.int64),
            "delta_h_by_category": category_delta.astype(np.float32),
            "category_uncertainty_block_counts": block_counts,
            "delta_h_countsketch_block_sums": block_sums.astype(np.float32),
            "uncertainty_block_count": np.asarray(block_count, dtype=np.int64),
            "uncertainty_countsketch_dimension": np.asarray(
                sketch_dimension, dtype=np.int64
            ),
            "uncertainty_countsketch_seed": np.asarray(sketch_seed, dtype=np.int64),
            "condition_names": np.asarray(condition_names),
            "condition_category_names": np.asarray(condition_categories),
            "condition_episode_counts": condition_counts,
            "delta_h_countsketch_condition_sums": condition_sums.astype(np.float32),
            "template_cell_names": np.asarray(template_cell_names),
            "template_cell_category_names": np.asarray(template_cell_categories),
            "template_cell_group_names": np.asarray(template_cell_groups),
            "template_cell_episode_counts": template_cell_counts,
            "delta_h_countsketch_template_cell_sums": template_sums.astype(
                np.float32
            ),
        },
    )


def write_synthetic_merge_manifest(
    path: Path,
    *,
    effects_path: Path,
    split_manifest: dict[str, Any],
    split_manifest_path: Path,
    role: str,
) -> None:
    entry = split_manifest["splits"][role]
    body = {
        "schema": "sch.vpd-effect-shard-merge.v1",
        "status": "complete",
        "identity": {
            "activation_category_split_id": split_manifest["split_id"],
            "activation_category_split_manifest_sha256": sha256_file(
                split_manifest_path
            ),
            "activation_category_split_schema": split_manifest["schema"],
            "activation_category_split_role": role,
            "activation_category_split_episode_id_set_sha256": entry[
                "episode_id_set_sha256"
            ],
            "activation_category_split_pair_id_set_sha256": entry[
                "pair_id_set_sha256"
            ],
            "activation_category_split_template_group_set_sha256": entry[
                "template_group_set_sha256"
            ],
            "activation_category_split_semantic_disjointness": "pass",
            "activation_category_split_verification_status": "pass",
        },
        "artifacts": {
            "activation_effects": {
                "path": effects_path.relative_to(path.parent).as_posix(),
                "sha256": sha256_file(effects_path),
            }
        },
        "synthetic_benchmark_only": True,
    }
    write_json(path, {**body, "merge_id": content_sha256(body)})


def classification_metrics(
    truths: list[str], predictions: list[str]
) -> dict[str, Any]:
    if len(truths) != len(predictions):
        raise ValueError("synthetic truths and predictions do not align")
    true_specific = [value != "shared_cross_category" for value in truths]
    predicted_specific = [value != "shared_cross_category" for value in predictions]
    correct_specific = [
        truth == prediction and specific
        for truth, prediction, specific in zip(truths, predictions, true_specific)
    ]
    predicted_specific_count = sum(predicted_specific)
    true_specific_count = sum(true_specific)
    true_shared_count = len(truths) - true_specific_count
    shared_correct = sum(
        truth == prediction == "shared_cross_category"
        for truth, prediction in zip(truths, predictions)
    )
    return {
        "components": len(truths),
        "exact_accuracy": sum(
            truth == prediction for truth, prediction in zip(truths, predictions)
        )
        / len(truths),
        "specific_precision": (
            sum(correct_specific) / predicted_specific_count
            if predicted_specific_count
            else 1.0
        ),
        "specific_recall": (
            sum(correct_specific) / true_specific_count if true_specific_count else 1.0
        ),
        "shared_recall": shared_correct / true_shared_count if true_shared_count else 1.0,
        "predicted_specific": predicted_specific_count,
        "correct_specific": sum(correct_specific),
        "true_specific": true_specific_count,
        "true_shared": true_shared_count,
        "confusion": {
            truth: dict(sorted(counts.items()))
            for truth, counts in sorted(
                (
                    truth,
                    Counter(
                        prediction
                        for expected, prediction in zip(truths, predictions)
                        if expected == truth
                    ),
                )
                for truth in sorted(set(truths))
            )
        },
    }


def cluster_ground_truth_purity(
    assignments: list[dict[str, Any]], truth_by_id: dict[str, str]
) -> float:
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for row in assignments:
        by_cluster[str(row["cluster_id"])].append(
            truth_by_id[str(row["component_id"])]
        )
    return sum(max(Counter(values).values()) for values in by_cluster.values()) / sum(
        len(values) for values in by_cluster.values()
    )


def run_scenario(
    scenario: dict[str, Any], *, project_root: Path, seed: int
) -> dict[str, Any]:
    rows, truths, categories = synthetic_inventory()
    component_ids = [str(row["component_id"]) for row in rows]
    hidden_size = 64
    rng = np.random.default_rng(seed)
    discovery_directions = _unit_rows(
        rng.normal(size=(len(rows), len(categories), hidden_size))
    )
    calibration_directions = _rotated_directions(
        discovery_directions,
        cosine=float(scenario["calibration_direction_cosine"]),
        rng=rng,
    )
    final_directions = _rotated_directions(
        discovery_directions,
        cosine=float(scenario["final_direction_cosine"]),
        rng=rng,
    )
    discovery_delta = category_means(
        truths=truths,
        categories=categories,
        directions=discovery_directions,
        target_strength=float(scenario["target_strength"]),
        background_strength=float(scenario["background_strength"]),
        label_shift="none",
    )
    calibration_delta = category_means(
        truths=truths,
        categories=categories,
        directions=calibration_directions,
        target_strength=float(scenario["target_strength"]) * 0.94,
        background_strength=float(scenario["background_strength"]) * 1.06,
        label_shift=str(scenario["calibration_label_shift"]),
    )
    calibration_delta *= float(scenario.get("calibration_magnitude_scale", 1.0))
    minority_shift_count = int(
        scenario.get("calibration_minority_label_shift_count", 0)
    )
    if minority_shift_count < 0 or minority_shift_count > sum(
        truth != "shared_cross_category" for truth in truths
    ):
        raise ValueError("synthetic minority label shift count is invalid")
    category_index = {category: index for index, category in enumerate(categories)}
    for component_index in range(minority_shift_count):
        target = category_index[truths[component_index]]
        shifted = (target + 1) % len(categories)
        calibration_delta[component_index, target] = (
            calibration_directions[component_index, target]
            * float(scenario["background_strength"])
            * 1.06
        )
        calibration_delta[component_index, shifted] = (
            calibration_directions[component_index, shifted]
            * float(scenario["target_strength"])
            * 0.94
        )
    validation_directions = _rotated_directions(
        discovery_directions,
        cosine=0.94,
        rng=rng,
    )
    validation_delta = category_means(
        truths=truths,
        categories=categories,
        directions=validation_directions,
        target_strength=float(scenario["target_strength"]) * 0.92,
        background_strength=float(scenario["background_strength"]) * 1.08,
        label_shift="none",
    )
    final_delta = category_means(
        truths=truths,
        categories=categories,
        directions=final_directions,
        target_strength=float(scenario["target_strength"]) * 0.90,
        background_strength=float(scenario["background_strength"]) * 1.10,
        label_shift="none",
    )
    split_manifest_path = (
        project_root
        / "artifacts"
        / "vpd_activation_category_split_v3"
        / "split_manifest.json"
    )
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    category_counts = np.asarray(
        [
            split_manifest["splits"]["discovery"][
                "activation_category_counts"
            ][category]
            for category in categories
        ],
        dtype=np.int64,
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        inventory_path = root / "inventory.jsonl"
        write_jsonl(inventory_path, rows)
        discovery_path = root / "discovery.npz"
        calibration_path = root / "calibration.npz"
        validation_path = root / "cluster_validation.npz"
        write_effect_archive(
            discovery_path,
            component_ids=component_ids,
            category_names=categories,
            category_counts=category_counts,
            category_delta=discovery_delta,
            block_noise=float(scenario["block_noise"]),
            seed=seed + 1,
            truths=truths,
            narrow_condition_category=scenario.get("narrow_condition_category"),
            narrow_template_category=scenario.get("narrow_template_category"),
            opposing_pair_block_category=scenario.get(
                "opposing_pair_block_category"
            ),
            artifact_component_limit=scenario.get(
                "discovery_artifact_component_limit"
            ),
        )
        write_effect_archive(
            calibration_path,
            component_ids=component_ids,
            category_names=categories,
            category_counts=category_counts,
            category_delta=calibration_delta,
            block_noise=float(scenario["block_noise"]),
            seed=seed + 2,
            truths=truths,
            narrow_condition_category=scenario.get(
                "calibration_narrow_condition_category"
            ),
            narrow_template_category=scenario.get(
                "calibration_narrow_template_category"
            ),
            opposing_pair_block_category=scenario.get(
                "calibration_opposing_pair_block_category"
            ),
        )
        write_effect_archive(
            validation_path,
            component_ids=component_ids,
            category_names=categories,
            category_counts=category_counts,
            category_delta=validation_delta,
            block_noise=float(scenario["block_noise"]),
            seed=seed + 4,
            truths=truths,
        )
        discovery_manifest = root / "discovery_merge.json"
        calibration_manifest = root / "calibration_merge.json"
        validation_manifest = root / "cluster_validation_merge.json"
        write_synthetic_merge_manifest(
            discovery_manifest,
            effects_path=discovery_path,
            split_manifest=split_manifest,
            split_manifest_path=split_manifest_path,
            role="discovery",
        )
        write_synthetic_merge_manifest(
            calibration_manifest,
            effects_path=calibration_path,
            split_manifest=split_manifest,
            split_manifest_path=split_manifest_path,
            role="calibration",
        )
        write_synthetic_merge_manifest(
            validation_manifest,
            effects_path=validation_path,
            split_manifest=split_manifest,
            split_manifest_path=split_manifest_path,
            role="cluster_validation",
        )
        output = root / "bundle"
        try:
            manifest = build_bundle(
                rows,
                inventory_sha256=sha256_file(inventory_path),
                effects_path=discovery_path,
                calibration_effects_path=calibration_path,
                cluster_validation_effects_path=validation_path,
                effects_merge_manifest_path=discovery_manifest,
                calibration_effects_merge_manifest_path=calibration_manifest,
                cluster_validation_effects_merge_manifest_path=validation_manifest,
                output_dir=output,
                cluster_count=16,
                seed=1707,
                projection_dim=16,
                metadata_weight=0.15,
                max_iterations=100,
                anchors=["proposal_start"],
                workers=[],
                minimum_stable_specific_components=32,
                minimum_category_calibration_episodes=8,
                require_category_uncertainty=True,
                require_category_pair_block_direction_robustness=True,
                require_category_condition_robustness=True,
                require_category_template_robustness=True,
                select_category_cluster_counts_by_projection=True,
                category_cluster_projection_replicates=8,
                maximum_clusters_per_category=4,
                minimum_category_cluster_size=4,
                require_category_cluster_sampling_stability=True,
                require_category_cluster_cross_archive_stability=True,
            )
        except ValueError as exc:
            rejected = "independent replication policy" in str(exc)
            passed = scenario["expectation"] == "reject_calibration" and rejected
            return {
                "scenario_id": scenario["id"],
                "expectation": scenario["expectation"],
                "status": "pass" if passed else "fail",
                "calibration_rejected": rejected,
                "rejection_reason": str(exc),
            }
        assignments = load_jsonl(output / "assignments.jsonl")
        assignment_by_id = {
            str(row["component_id"]): row for row in assignments
        }
        predictions = [
            str(assignment_by_id[component_id]["activation_category"])
            for component_id in component_ids
        ]
        discovery_metrics = classification_metrics(truths, predictions)
        thresholds = manifest["algorithm"]["activation_category_compartments"][
            "specificity_thresholds"
        ]
        final_strengths = np.linalg.norm(final_delta, axis=2)
        final_result = classify_activation_compartments(
            final_strengths,
            categories,
            minimum_dominant_share=float(thresholds["minimum_dominant_share"]),
            minimum_dominance_margin=float(thresholds["minimum_dominance_margin"]),
            maximum_normalized_entropy=float(
                thresholds["maximum_normalized_entropy"]
            ),
        )
        final_metrics = classification_metrics(
            truths, final_result["compartment_categories"]
        )
        truth_by_id = dict(zip(component_ids, truths, strict=True))
        jackknife_reroutes = sum(
            row["point_estimate_activation_category"]
            != row["post_uncertainty_activation_category"]
            for row in assignments
        )
        condition_reroutes = sum(
            row["post_pair_block_direction_activation_category"]
            != row["post_condition_activation_category"]
            for row in assignments
        )
        pair_block_direction_reroutes = sum(
            row["post_uncertainty_activation_category"]
            != row["post_pair_block_direction_activation_category"]
            for row in assignments
        )
        template_reroutes = sum(
            row["post_condition_activation_category"]
            != row["pre_cross_archive_activation_category"]
            for row in assignments
        )
        cross_archive_reroutes = sum(
            row["pre_cross_archive_activation_category"]
            != row["activation_category"]
            for row in assignments
        )
        checks = {
            "specific_precision": discovery_metrics["specific_precision"]
            >= float(scenario.get("minimum_specific_precision", 0.0)),
            "shared_recall": discovery_metrics["shared_recall"]
            >= float(scenario.get("minimum_shared_recall", 0.0)),
        }
        if "minimum_specific_recall" in scenario:
            checks["specific_recall"] = discovery_metrics["specific_recall"] >= float(
                scenario["minimum_specific_recall"]
            )
        if "minimum_final_accuracy" in scenario:
            checks["final_accuracy"] = final_metrics["exact_accuracy"] >= float(
                scenario["minimum_final_accuracy"]
            )
        if scenario["expectation"] == "reject_calibration":
            checks["expected_rejection"] = False
        if scenario["expectation"] == "condition_reroute":
            checks["condition_narrow_specificity_rerouted"] = condition_reroutes >= int(
                scenario["minimum_condition_reroutes"]
            )
        if scenario["expectation"] == "template_reroute":
            checks["template_narrow_specificity_rerouted"] = template_reroutes >= int(
                scenario["minimum_template_reroutes"]
            )
        if scenario["expectation"] == "pair_block_direction_reroute":
            checks["opposing_pair_block_directions_rerouted"] = (
                pair_block_direction_reroutes
                >= int(scenario["minimum_pair_block_direction_reroutes"])
            )
        if scenario["expectation"] == "cross_archive_component_reroute":
            checks["minority_cross_archive_disagreement_rerouted"] = (
                cross_archive_reroutes
                >= int(scenario["minimum_cross_archive_reroutes"])
            )
        return {
            "scenario_id": scenario["id"],
            "expectation": scenario["expectation"],
            "status": "pass" if all(checks.values()) else "fail",
            "calibration_rejected": False,
            "checks": checks,
            "discovery": discovery_metrics,
            "final_unseen_template_proxy": final_metrics,
            "jackknife_specific_to_shared_reroutes": jackknife_reroutes,
            "condition_specific_to_shared_reroutes": condition_reroutes,
            "pair_block_direction_specific_to_shared_reroutes": (
                pair_block_direction_reroutes
            ),
            "template_specific_to_shared_reroutes": template_reroutes,
            "cross_archive_specific_to_shared_reroutes": cross_archive_reroutes,
            "cluster_ground_truth_purity": cluster_ground_truth_purity(
                assignments, truth_by_id
            ),
            "selected_thresholds": thresholds,
            "calibration_metrics": manifest["algorithm"][
                "activation_category_compartments"
            ]["threshold_calibration"]["selection"]["selected_metrics"],
        }


def build_report(project_root: Path, *, seed: int = 240829) -> dict[str, Any]:
    protocol_path = (
        project_root
        / "configs"
        / "evals"
        / "vpd_activation_category_compartments_v4.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") != "sch.vpd-activation-category-protocol.v4":
        raise ValueError("protocol benchmark requires V4 per-component admission")
    if float(
        protocol.get("component_compartments", {})
        .get("threshold_calibration", {})
        .get("minimum_dominant_strength_ratio", -1.0)
    ) != 0.50:
        raise ValueError("protocol cross-archive strength ratio differs")
    if not bool(
        protocol.get("component_compartments", {})
        .get("threshold_calibration", {})
        .get("same_stratified_robustness_gates_required_in_both_archives", False)
    ):
        raise ValueError("protocol cross-archive stratified robustness differs")
    component_admission = protocol.get("component_compartments", {}).get(
        "per_component_cross_archive_admission", {}
    )
    if component_admission != {
        "schema": "sch.vpd-cross-archive-component-admission.v1",
        "mode": "same_category_direction_magnitude_and_stratified_robustness",
        "applied_after_discovery_within_split_screens": True,
        "calibration_point_estimate_same_specific_category_required": True,
        "minimum_signed_direction_cosine": 0.5,
        "minimum_dominant_strength_ratio": 0.5,
        "same_stratified_robustness_gates_required_in_both_archives": True,
        "global_threshold_replication_does_not_waive_component_admission": True,
        "failed_specific_routes_to": "shared_cross_category",
        "required_for_production_category_bundle": True,
        "assignment_diagnostics_bound_into_bundle": True,
        "independent_verifier_recomputes_every_component": True,
        "model_outcomes_used": False,
    }:
        raise ValueError("protocol per-component cross-archive admission differs")
    clustering_geometry_path = (
        project_root
        / "artifacts"
        / "vpd_activation_category_clustering_geometry_benchmark_v1"
        / "report.json"
    )
    clustering_geometry = verify_clustering_geometry_benchmark(
        clustering_geometry_path
    )
    cluster_count_path = (
        project_root
        / "artifacts"
        / "vpd_activation_category_cluster_count_benchmark_v1"
        / "report.json"
    )
    cluster_count_benchmark = verify_cluster_count_benchmark(cluster_count_path)
    geometry_contract = protocol.get("component_compartments", {}).get(
        "within_category_clustering_geometry", {}
    )
    if geometry_contract != {
        "mode": "share_weighted_direction_strength_profile_kmeans",
        "direction_weighting": "sqrt_category_strength_share",
        "weighting_principle": "one_unit_sum_squared_category_direction_weights_for_nonzero_effect_component",
        "arbitrarily_small_background_directions_have_arbitrarily_small_weight": True,
        "synthetic_benchmark_path": (
            "artifacts/vpd_activation_category_clustering_geometry_benchmark_v1/report.json"
        ),
        "synthetic_benchmark_id": clustering_geometry["benchmark_id"],
        "production_shaped_hidden_size": 4096,
        "minimum_worst_seed_pairwise_f1": 0.95,
        "minimum_worst_seed_cluster_purity": 0.98,
        "does_not_select_empirical_bonsai_cluster_count": True,
    }:
        raise ValueError("protocol within-category clustering geometry differs")
    count_contract = protocol.get("component_compartments", {}).get(
        "within_category_cluster_count_selection", {}
    )
    if count_contract != {
        "mode": "cross_archive_projection_and_pair_block_ensemble_stability_and_silhouette",
        "discovery_projection_replicates": 8,
        "discovery_leave_one_pair_block_out_replicates": 8,
        "independent_cluster_validation_projection_replicates": 8,
        "independent_cluster_validation_leave_one_pair_block_out_replicates": 8,
        "total_replicates": 32,
        "maximum_clusters_per_category": 4,
        "minimum_replicate_adjusted_rand": 0.90,
        "minimum_replicate_silhouette": 0.15,
        "minimum_cluster_size": 4,
        "no_passing_nontrivial_candidate_rule": "one_cluster_for_category",
        "synthetic_benchmark_path": (
            "artifacts/vpd_activation_category_cluster_count_benchmark_v1/report.json"
        ),
        "synthetic_benchmark_id": cluster_count_benchmark["benchmark_id"],
        "required_for_production_category_bundle": True,
        "pair_block_sampling_stability_required_for_production_category_bundle": True,
        "independent_cluster_validation_archive_required_for_production_category_bundle": True,
        "cluster_validation_archive_must_be_content_pair_and_template_disjoint_from_discovery_and_calibration": True,
        "cluster_validation_split_role_status": "locally_validated_no_model_outcomes_not_captured",
        "manual_total_cluster_count_allowed_only_as_explicit_ablation": True,
        "does_not_select_empirical_bonsai_cluster_counts": True,
    }:
        raise ValueError("protocol within-category cluster count selection differs")
    sketch_fidelity_path = (
        project_root
        / "artifacts"
        / "vpd_effect_sketch_fidelity_benchmark_v1"
        / "report.json"
    )
    sketch_fidelity = verify_sketch_fidelity_benchmark(sketch_fidelity_path)
    selected_sketch = sketch_fidelity["selected_candidate"]
    if selected_sketch != {
        "dimension": 256,
        "operational_minimum_direction_cosine": 0.5,
        "operational_minimum_relative_norm": 0.25,
        "status": "pass",
        "worst_seed_direction_cosine_absolute_error_p99": 0.12250412873752246,
        "worst_seed_relative_norm_absolute_error_p99": 0.061274488613041654,
        "worst_seed_stable_false_admission_rate": 0.001953125,
        "worst_seed_stable_false_admission_wilson_95_upper": 0.007093273978132259,
        "worst_seed_stable_pass_recall": 1.0,
        "worst_seed_stable_pass_recall_wilson_95_lower": 0.9962625959599825,
    }:
        raise ValueError("effect sketch fidelity selection differs")
    condition_screen = protocol.get("component_compartments", {}).get(
        "within_category_condition_robustness_screen", {}
    )
    pair_block_direction_screen = protocol.get("component_compartments", {}).get(
        "within_category_pair_block_direction_robustness_screen", {}
    )
    if pair_block_direction_screen != {
        "mode": "pair_block_mean_countsketch_direction_and_relative_norm_coverage",
        "minimum_pair_block_coverage": 0.75,
        "minimum_pair_block_direction_cosine": 0.50,
        "minimum_pair_block_relative_norm": 0.25,
        "minimum_countsketch_dimension": 256,
        "failed_specific_routes_to": "shared_cross_category",
        "required_for_production_category_bundle": True,
        "reuses_signed_uncertainty_arrays": True,
        "model_outcomes_used": False,
        "does_not_replace_condition_template_or_independent_split_replication": True,
    }:
        raise ValueError("protocol pair-block direction screen differs")
    if condition_screen != {
        "mode": "condition_mean_countsketch_direction_and_relative_norm_coverage",
        "minimum_condition_coverage": 0.75,
        "minimum_condition_direction_cosine": 0.50,
        "minimum_condition_relative_norm": 0.25,
        "failed_specific_routes_to": "shared_cross_category",
        "required_for_production_category_bundle": True,
        "model_outcomes_used": False,
        "minimum_countsketch_dimension": 256,
        "does_not_replace_pair_block_or_independent_split_replication": True,
    }:
        raise ValueError("protocol condition robustness screen differs")
    template_screen = protocol.get("component_compartments", {}).get(
        "within_category_template_robustness_screen", {}
    )
    if template_screen != {
        "mode": "template_group_mean_countsketch_direction_and_relative_norm_coverage",
        "minimum_template_group_coverage": 0.75,
        "minimum_template_group_direction_cosine": 0.50,
        "minimum_template_group_relative_norm": 0.25,
        "same_template_group_set_required_per_category": True,
        "failed_specific_routes_to": "shared_cross_category",
        "required_for_production_category_bundle": True,
        "model_outcomes_used": False,
        "minimum_countsketch_dimension": 256,
        "does_not_replace_pair_block_condition_or_independent_split_replication": True,
    }:
        raise ValueError("protocol template robustness screen differs")
    scenario_rows = [
        run_scenario(scenario, project_root=project_root, seed=seed + index * 101)
        for index, scenario in enumerate(SCENARIOS)
    ]
    body = {
        "schema": "sch.vpd-activation-category-synthetic-benchmark.v1",
        "status": (
            "pass" if all(row["status"] == "pass" for row in scenario_rows) else "fail"
        ),
        "seed": seed,
        "source": {
            "runner_sha256": sha256_file(Path(__file__)),
            "clustering_implementation_sha256": sha256_file(
                Path(__file__).with_name("cluster_vpd_for_sae.py")
            ),
            "protocol_sha256": sha256_file(
                protocol_path
            ),
            "pack_manifest_sha256": sha256_file(
                project_root
                / "artifacts"
                / "vpd_activation_category_pack_v2"
                / "artifact_manifest.json"
            ),
            "sketch_fidelity_benchmark_sha256": sha256_file(
                sketch_fidelity_path
            ),
            "sketch_fidelity_benchmark_id": sketch_fidelity["benchmark_id"],
            "clustering_geometry_benchmark_sha256": sha256_file(
                clustering_geometry_path
            ),
            "clustering_geometry_benchmark_id": clustering_geometry[
                "benchmark_id"
            ],
            "cluster_count_benchmark_sha256": sha256_file(cluster_count_path),
            "cluster_count_benchmark_id": cluster_count_benchmark["benchmark_id"],
        },
        "design": {
            "components": 128,
            "specific_components_per_category": 12,
            "shared_components": 32,
            "categories": 8,
            "hidden_size": 64,
            "pair_blocks": 8,
            "pair_block_direction_robustness_required": True,
            "countsketch_dimension": 256,
            "sketch_fidelity_hidden_size": 4096,
            "conditions_per_category": 2,
            "condition_robustness_required": True,
            "template_groups_per_category": 4,
            "template_robustness_required": True,
            "known_ground_truth": True,
            "scenarios": len(SCENARIOS),
        },
        "scenarios": scenario_rows,
        "scientific_boundary": {
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "synthetic_effects_only": True,
            "does_not_establish_bonsai_component_categories": True,
            "purpose": (
                "Executable protocol validation and failure-boundary checks before "
                "authorized empirical effect capture."
            ),
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_report(path: Path, *, project_root: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    body = dict(report)
    benchmark_id = str(body.pop("benchmark_id", ""))
    if content_sha256(body) != benchmark_id:
        raise ValueError("synthetic benchmark content address mismatch")
    if report.get("status") != "pass":
        raise ValueError("synthetic activation-category benchmark did not pass")
    expected = build_report(project_root, seed=int(report["seed"]))
    if expected != report:
        raise ValueError("synthetic activation-category benchmark is not reproducible")
    return {
        "schema": "sch.vpd-activation-category-synthetic-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": benchmark_id,
        "scenarios": len(report["scenarios"]),
        "all_expectations_met": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or verify the model-free activation-category protocol benchmark."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--seed", type=int, default=240829)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.verify is not None:
        print(json.dumps(verify_report(args.verify, project_root=project_root), indent=2, sort_keys=True))
        return
    if args.output is None:
        parser.error("--output is required unless --verify is used")
    report = build_report(project_root, seed=args.seed)
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
