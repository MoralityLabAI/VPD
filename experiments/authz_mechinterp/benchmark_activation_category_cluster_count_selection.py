from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.activation_categories import category_contract
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    DEFAULT_MAXIMUM_CLUSTERS_PER_CATEGORY,
    DEFAULT_MINIMUM_CATEGORY_CLUSTER_PROJECTION_STABILITY,
    DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE,
    DEFAULT_MINIMUM_CATEGORY_CLUSTER_SIZE,
    _project_effects,
    activation_category_profile_features,
    content_sha256,
    select_category_cluster_counts,
    sha256_file,
    write_json,
)


HIDDEN_SIZE = 4096
PROJECTION_DIMENSION = 64
PROJECTION_SEEDS = (17, 29, 41, 53, 67, 79, 97, 109)
SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "id": "one_mechanism_fail_closed",
        "mechanisms": 1,
        "components_per_mechanism": 12,
        "target_noise": 0.20,
        "background_mode": "aligned",
        "background_strength": 0.02,
        "expected_cluster_count": 1,
    },
    {
        "id": "two_separated_mechanisms",
        "mechanisms": 2,
        "components_per_mechanism": 6,
        "target_noise": 0.06,
        "background_mode": "aligned",
        "background_strength": 0.02,
        "expected_cluster_count": 2,
    },
    {
        "id": "three_separated_mechanisms",
        "mechanisms": 3,
        "components_per_mechanism": 4,
        "target_noise": 0.06,
        "background_mode": "aligned",
        "background_strength": 0.02,
        "expected_cluster_count": 3,
    },
    {
        "id": "two_mechanisms_with_tiny_random_background",
        "mechanisms": 2,
        "components_per_mechanism": 6,
        "target_noise": 0.06,
        "background_mode": "component_random",
        "background_strength": 1e-6,
        "expected_cluster_count": 2,
    },
    {
        "id": "aggregate_only_two_mechanism_sampling_artifact",
        "mechanisms": 2,
        "components_per_mechanism": 6,
        "target_noise": 0.06,
        "background_mode": "aligned",
        "background_strength": 0.02,
        "sampling_mode": "unstable_component_directions",
        "expected_cluster_count": 1,
    },
    {
        "id": "discovery_only_two_mechanism_cross_archive_rearrangement",
        "mechanisms": 2,
        "components_per_mechanism": 6,
        "target_noise": 0.06,
        "background_mode": "aligned",
        "background_strength": 0.02,
        "cluster_validation_mode": "rearranged_clean_mechanisms",
        "expected_cluster_count": 1,
    },
)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("cannot normalize zero synthetic vector")
    return vector / norm


def _orthonormal_directions(
    rng: np.random.Generator, count: int
) -> list[np.ndarray]:
    directions: list[np.ndarray] = []
    while len(directions) < count:
        candidate = rng.normal(size=HIDDEN_SIZE)
        for direction in directions:
            candidate -= float(candidate @ direction) * direction
        directions.append(_unit(candidate))
    return directions


def synthetic_effects(
    scenario: dict[str, Any], *, seed: int
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(seed)
    categories = sorted(row["id"] for row in category_contract()["categories"])
    mechanism_count = int(scenario["mechanisms"])
    replicas = int(scenario["components_per_mechanism"])
    target_noise = float(scenario["target_noise"])
    background_strength = float(scenario["background_strength"])
    background_mode = str(scenario["background_mode"])
    centroids = {
        category: _orthonormal_directions(rng, mechanism_count)
        for category in categories
    }
    aligned_background = {
        category: _unit(rng.normal(size=HIDDEN_SIZE)) for category in categories
    }
    component_ids: list[str] = []
    component_categories: list[str] = []
    effects: list[np.ndarray] = []
    for category_index, category in enumerate(categories):
        for mechanism_index in range(mechanism_count):
            centroid = centroids[category][mechanism_index]
            for replica in range(replicas):
                component_ids.append(
                    f"component-{category_index:02d}-{mechanism_index}-{replica:02d}"
                )
                component_categories.append(category)
                row = np.empty((len(categories), HIDDEN_SIZE), dtype=np.float64)
                for effect_index, effect_category in enumerate(categories):
                    if effect_category == category:
                        noise = rng.normal(size=HIDDEN_SIZE)
                        noise -= float(noise @ centroid) * centroid
                        row[effect_index] = _unit(
                            centroid + target_noise * _unit(noise)
                        ) * 2.0
                    elif background_mode == "aligned":
                        row[effect_index] = (
                            aligned_background[effect_category] * background_strength
                        )
                    elif background_mode == "component_random":
                        row[effect_index] = (
                            _unit(rng.normal(size=HIDDEN_SIZE)) * background_strength
                        )
                    else:
                        raise ValueError("unknown synthetic background mode")
                effects.append(row)
    category_effects = np.asarray(effects, dtype=np.float64)
    return (
        category_effects.mean(axis=1),
        category_effects,
        component_ids,
        component_categories,
    )


def _matrix_from_effects(
    category_effects: np.ndarray, *, projection_seed: int
) -> np.ndarray:
    aggregate = category_effects.mean(axis=1)
    global_features = _project_effects(
        aggregate,
        seed=projection_seed,
        projection_dim=PROJECTION_DIMENSION,
    )
    category_features = activation_category_profile_features(
        category_effects,
        seed=projection_seed,
        projection_dim=PROJECTION_DIMENSION,
    )["features"]
    return np.column_stack([global_features, category_features])


def replicate_matrices(
    aggregate: np.ndarray,
    category_effects: np.ndarray,
    component_categories: list[str],
    scenario: dict[str, Any],
    *,
    seed: int,
) -> tuple[list[np.ndarray], list[str]]:
    matrices: list[np.ndarray] = []
    for projection_seed in PROJECTION_SEEDS:
        matrices.append(
            _matrix_from_effects(
                category_effects,
                projection_seed=projection_seed,
            )
        )
    kinds = ["projection"] * len(matrices)
    rng = np.random.default_rng(seed)
    category_names = sorted(set(component_categories))
    category_index = {category: index for index, category in enumerate(category_names)}
    sampling_mode = str(scenario.get("sampling_mode", "stable_small_perturbation"))
    strengths = np.linalg.norm(category_effects, axis=2, keepdims=True)
    for _ in range(8):
        sampled = category_effects.copy()
        if sampling_mode == "stable_small_perturbation":
            noise = rng.normal(size=sampled.shape)
            noise /= np.maximum(np.linalg.norm(noise, axis=2, keepdims=True), 1e-12)
            sampled = sampled + noise * strengths * 0.01
        elif sampling_mode == "unstable_component_directions":
            for component_index, category in enumerate(component_categories):
                target = category_index[category]
                direction = rng.normal(size=HIDDEN_SIZE)
                direction /= np.linalg.norm(direction)
                sampled[component_index, target] = direction * 2.0
        else:
            raise ValueError("unknown synthetic sampling mode")
        matrices.append(_matrix_from_effects(sampled, projection_seed=PROJECTION_SEEDS[0]))
        kinds.append("leave_one_pair_block_out")
    validation = category_effects.copy()
    validation_noise = rng.normal(size=validation.shape)
    validation_noise /= np.maximum(
        np.linalg.norm(validation_noise, axis=2, keepdims=True), 1e-12
    )
    validation = validation + validation_noise * strengths * 0.01
    validation_mode = str(
        scenario.get("cluster_validation_mode", "stable_small_perturbation")
    )
    if validation_mode == "rearranged_clean_mechanisms":
        mechanisms = int(scenario["mechanisms"])
        replicas = int(scenario["components_per_mechanism"])
        if mechanisms != 2 or replicas < 4:
            raise ValueError("crossed validation scenario requires two supported mechanisms")
        components_per_category = mechanisms * replicas
        for category_position, category in enumerate(category_names):
            target = category_index[category]
            offset = category_position * components_per_category
            first_direction = category_effects[offset, target].copy()
            second_direction = category_effects[offset + replicas, target].copy()
            for mechanism_index in range(mechanisms):
                for replica in range(replicas):
                    component_index = offset + mechanism_index * replicas + replica
                    validation[component_index, target] = (
                        first_direction
                        if replica < replicas // 2
                        else second_direction
                    )
    elif validation_mode != "stable_small_perturbation":
        raise ValueError("unknown synthetic cluster validation mode")
    for projection_seed in PROJECTION_SEEDS:
        matrices.append(
            _matrix_from_effects(validation, projection_seed=projection_seed)
        )
        kinds.append("independent_cluster_validation_projection")
    validation_strengths = np.linalg.norm(validation, axis=2, keepdims=True)
    for _ in range(8):
        sampled_validation = validation.copy()
        noise = rng.normal(size=sampled_validation.shape)
        noise /= np.maximum(np.linalg.norm(noise, axis=2, keepdims=True), 1e-12)
        sampled_validation = (
            sampled_validation + noise * validation_strengths * 0.01
        )
        matrices.append(
            _matrix_from_effects(
                sampled_validation,
                projection_seed=PROJECTION_SEEDS[0],
            )
        )
        kinds.append(
            "independent_cluster_validation_leave_one_pair_block_out"
        )
    return matrices, kinds


def build_report(*, seed: int = 240829) -> dict[str, Any]:
    scenario_rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        aggregate, category_effects, component_ids, categories = synthetic_effects(
            scenario,
            seed=seed + scenario_index * 1009,
        )
        matrices, kinds = replicate_matrices(
            aggregate,
            category_effects,
            categories,
            scenario,
            seed=seed + scenario_index * 1009 + 1,
        )
        selection = select_category_cluster_counts(
            matrices,
            component_ids,
            categories,
            seed=seed,
            replicate_kinds=kinds,
            maximum_clusters_per_category=DEFAULT_MAXIMUM_CLUSTERS_PER_CATEGORY,
            minimum_replicate_stability=(
                DEFAULT_MINIMUM_CATEGORY_CLUSTER_PROJECTION_STABILITY
            ),
            minimum_silhouette=DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE,
            minimum_cluster_size=DEFAULT_MINIMUM_CATEGORY_CLUSTER_SIZE,
        )
        expected = int(scenario["expected_cluster_count"])
        if set(selection["allocation"].values()) != {expected}:
            raise ValueError(
                f"cluster count selection differs for scenario {scenario['id']}"
            )
        scenario_rows.append(
            {
                "scenario_id": scenario["id"],
                "expected_cluster_count_per_category": expected,
                "status": "pass",
                "selection": selection,
            }
        )
    body = {
        "schema": "sch.vpd-activation-category-cluster-count-benchmark.v1",
        "status": "pass",
        "seed": seed,
        "source": {
            "runner_sha256": sha256_file(Path(__file__)),
            "clustering_implementation_sha256": sha256_file(
                Path(__file__).with_name("cluster_vpd_for_sae.py")
            ),
        },
        "design": {
            "hidden_size": HIDDEN_SIZE,
            "projection_dimension": PROJECTION_DIMENSION,
            "projection_seeds": list(PROJECTION_SEEDS),
            "leave_one_pair_block_out_replicates": 8,
            "independent_cluster_validation_projection_replicates": 8,
            "independent_cluster_validation_leave_one_pair_block_out_replicates": 8,
            "total_replicates": 32,
            "categories": 8,
            "components_per_category": 12,
            "known_mechanism_counts": [1, 2, 3],
            "metadata_weight": 0.0,
        },
        "policy": {
            "maximum_clusters_per_category": DEFAULT_MAXIMUM_CLUSTERS_PER_CATEGORY,
            "minimum_replicate_adjusted_rand": (
                DEFAULT_MINIMUM_CATEGORY_CLUSTER_PROJECTION_STABILITY
            ),
            "minimum_replicate_silhouette": (
                DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE
            ),
            "minimum_cluster_size": DEFAULT_MINIMUM_CATEGORY_CLUSTER_SIZE,
            "no_passing_nontrivial_candidate_rule": "one_cluster_for_category",
        },
        "scenarios": scenario_rows,
        "scientific_boundary": {
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "synthetic_effects_only": True,
            "does_not_select_empirical_bonsai_cluster_counts": True,
            "purpose": (
                "Validate conservative within-category cluster-count selection "
                "across discovery and independent validation archives before an "
                "authorized empirical capture exists."
            ),
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    body = dict(report)
    benchmark_id = str(body.pop("benchmark_id", ""))
    if content_sha256(body) != benchmark_id:
        raise ValueError("cluster count benchmark content address mismatch")
    if report != build_report(seed=int(report["seed"])) or report.get("status") != "pass":
        raise ValueError("cluster count benchmark is not reproducible")
    return {
        "schema": "sch.vpd-activation-category-cluster-count-verification.v1",
        "status": "pass",
        "benchmark_id": benchmark_id,
        "scenarios": len(report["scenarios"]),
        "projection_seeds": len(report["design"]["projection_seeds"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or verify the model-free category cluster-count benchmark."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--seed", type=int, default=240829)
    args = parser.parse_args()
    if args.verify is not None:
        print(json.dumps(verify_benchmark(args.verify), indent=2, sort_keys=True))
        return
    if args.output is None:
        parser.error("--output is required unless --verify is used")
    report = build_report(seed=args.seed)
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
