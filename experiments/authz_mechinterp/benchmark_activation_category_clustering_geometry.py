from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.activation_categories import category_contract
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    ACTIVATION_CATEGORY_DIRECTION_WEIGHTING,
    _project_effects,
    activation_category_profile_features,
    content_sha256,
    deterministic_category_kmeans,
    sha256_file,
    write_json,
)


HIDDEN_SIZE = 4096
PROJECTION_DIMENSION = 64
COMPONENTS_PER_MECHANISM = 4
PROJECTION_SEEDS = (17, 29, 41, 53, 67, 79, 97, 109)
WEIGHTING_CANDIDATES = (
    "unweighted",
    ACTIVATION_CATEGORY_DIRECTION_WEIGHTING,
    "category_strength_share",
)
MINIMUM_WORST_SEED_PAIRWISE_F1 = 0.95
MINIMUM_WORST_SEED_CLUSTER_PURITY = 0.98

SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "id": "aligned_background",
        "background_strength": 0.05,
        "background_mode": "aligned",
    },
    {
        "id": "arbitrarily_weak_random_background",
        "background_strength": 1e-6,
        "background_mode": "component_random",
    },
    {
        "id": "moderate_random_background",
        "background_strength": 0.10,
        "background_mode": "component_random",
    },
)


def _unit(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero synthetic vector")
    return values / norm


def _orthogonal_to(
    rng: np.random.Generator, reference: np.ndarray
) -> np.ndarray:
    candidate = rng.normal(size=reference.shape)
    candidate -= float(candidate @ reference) * reference
    return _unit(candidate)


def synthetic_geometry(
    scenario: dict[str, Any], *, seed: int = 240829
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    rng = np.random.default_rng(seed)
    categories = sorted(row["id"] for row in category_contract()["categories"])
    category_centroids: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    aligned_background: dict[str, np.ndarray] = {}
    for category in categories:
        first = _unit(rng.normal(size=HIDDEN_SIZE))
        orthogonal = _orthogonal_to(rng, first)
        second = _unit(0.80 * first + 0.60 * orthogonal)
        category_centroids[category] = (first, second)
        aligned_background[category] = _unit(rng.normal(size=HIDDEN_SIZE))

    component_ids: list[str] = []
    component_categories: list[str] = []
    mechanisms: list[str] = []
    category_delta: list[np.ndarray] = []
    background_strength = float(scenario["background_strength"])
    background_mode = str(scenario["background_mode"])
    for category_index, category in enumerate(categories):
        for mechanism_index in range(2):
            centroid = category_centroids[category][mechanism_index]
            for replica_index in range(COMPONENTS_PER_MECHANISM):
                component_ids.append(
                    f"component-{category_index:02d}-{mechanism_index}-{replica_index}"
                )
                component_categories.append(category)
                mechanisms.append(f"{category}::mechanism-{mechanism_index}")
                effects = np.empty((len(categories), HIDDEN_SIZE), dtype=np.float64)
                for effect_index, effect_category in enumerate(categories):
                    if effect_category == category:
                        noise = _orthogonal_to(rng, centroid)
                        effects[effect_index] = _unit(centroid + 0.04 * noise) * 2.0
                    elif background_mode == "aligned":
                        effects[effect_index] = (
                            aligned_background[effect_category] * background_strength
                        )
                    elif background_mode == "component_random":
                        effects[effect_index] = (
                            _unit(rng.normal(size=HIDDEN_SIZE)) * background_strength
                        )
                    else:
                        raise ValueError("unknown synthetic background mode")
                category_delta.append(effects)
    category_array = np.asarray(category_delta, dtype=np.float64)
    aggregate = category_array.mean(axis=1)
    return (
        aggregate,
        category_array,
        component_ids,
        component_categories,
        mechanisms,
    )


def clustering_matrix(
    aggregate: np.ndarray,
    category_delta: np.ndarray,
    *,
    projection_seed: int,
    weighting: str,
) -> np.ndarray:
    projected_aggregate = _project_effects(
        aggregate,
        seed=projection_seed,
        projection_dim=PROJECTION_DIMENSION,
    )
    category_geometry = activation_category_profile_features(
        category_delta,
        seed=projection_seed,
        projection_dim=PROJECTION_DIMENSION,
        direction_weighting=weighting,
    )
    return np.column_stack([projected_aggregate, category_geometry["features"]])


def pairwise_metrics(
    labels: np.ndarray,
    component_categories: list[str],
    mechanisms: list[str],
) -> dict[str, float]:
    true_positive = 0
    predicted_positive = 0
    actual_positive = 0
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            if component_categories[left] != component_categories[right]:
                continue
            predicted_same = int(labels[left]) == int(labels[right])
            actual_same = mechanisms[left] == mechanisms[right]
            predicted_positive += int(predicted_same)
            actual_positive += int(actual_same)
            true_positive += int(predicted_same and actual_same)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    pairwise_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    cluster_members: dict[int, list[str]] = {}
    for label, mechanism in zip(labels.tolist(), mechanisms):
        cluster_members.setdefault(int(label), []).append(mechanism)
    purity = sum(
        max(Counter(values).values()) for values in cluster_members.values()
    ) / len(labels)
    return {
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": pairwise_f1,
        "cluster_purity": purity,
    }


def build_report(*, seed: int = 240829) -> dict[str, Any]:
    scenario_rows: list[dict[str, Any]] = []
    candidate_metrics: dict[str, list[dict[str, float]]] = {
        candidate: [] for candidate in WEIGHTING_CANDIDATES
    }
    for scenario_index, scenario in enumerate(SCENARIOS):
        aggregate, category_delta, component_ids, categories, mechanisms = (
            synthetic_geometry(scenario, seed=seed + scenario_index * 1009)
        )
        candidate_rows: dict[str, Any] = {}
        for weighting in WEIGHTING_CANDIDATES:
            seed_rows: list[dict[str, Any]] = []
            for projection_seed in PROJECTION_SEEDS:
                matrix = clustering_matrix(
                    aggregate,
                    category_delta,
                    projection_seed=projection_seed,
                    weighting=weighting,
                )
                labels, _, _, allocation = deterministic_category_kmeans(
                    matrix,
                    component_ids,
                    categories,
                    cluster_count=len(set(categories)) * 2,
                    seed=projection_seed,
                    max_iterations=100,
                )
                if set(allocation.values()) != {2}:
                    raise ValueError("synthetic category cluster allocation differs")
                metrics = pairwise_metrics(labels, categories, mechanisms)
                seed_rows.append({"projection_seed": projection_seed, **metrics})
                candidate_metrics[weighting].append(metrics)
            candidate_rows[weighting] = {
                "seeds": seed_rows,
                "worst_seed_pairwise_f1": min(
                    row["pairwise_f1"] for row in seed_rows
                ),
                "worst_seed_cluster_purity": min(
                    row["cluster_purity"] for row in seed_rows
                ),
            }
        scenario_rows.append(
            {
                "scenario_id": scenario["id"],
                "background_strength": scenario["background_strength"],
                "background_mode": scenario["background_mode"],
                "candidates": candidate_rows,
            }
        )

    candidates: list[dict[str, Any]] = []
    for weighting in WEIGHTING_CANDIDATES:
        metrics = candidate_metrics[weighting]
        worst_f1 = min(row["pairwise_f1"] for row in metrics)
        worst_purity = min(row["cluster_purity"] for row in metrics)
        status = (
            "pass"
            if worst_f1 >= MINIMUM_WORST_SEED_PAIRWISE_F1
            and worst_purity >= MINIMUM_WORST_SEED_CLUSTER_PURITY
            else "fail"
        )
        candidates.append(
            {
                "direction_weighting": weighting,
                "status": status,
                "worst_scenario_seed_pairwise_f1": worst_f1,
                "worst_scenario_seed_cluster_purity": worst_purity,
            }
        )
    passing = [row for row in candidates if row["status"] == "pass"]
    if not passing:
        raise ValueError("no activation-category direction weighting candidate passed")
    selected = passing[0]
    if selected["direction_weighting"] != ACTIVATION_CATEGORY_DIRECTION_WEIGHTING:
        raise ValueError("activation-category direction weighting selection differs")
    unweighted = next(
        row for row in candidates if row["direction_weighting"] == "unweighted"
    )
    if unweighted["status"] != "fail":
        raise ValueError("unweighted background-direction confound was not reproduced")

    runner_path = Path(__file__)
    body = {
        "schema": "sch.vpd-activation-category-clustering-geometry-benchmark.v1",
        "status": "pass",
        "seed": seed,
        "source": {
            "runner_sha256": sha256_file(runner_path),
            "clustering_implementation_sha256": sha256_file(
                runner_path.with_name("cluster_vpd_for_sae.py")
            ),
        },
        "design": {
            "hidden_size": HIDDEN_SIZE,
            "projection_dimension": PROJECTION_DIMENSION,
            "categories": 8,
            "mechanisms_per_category": 2,
            "components_per_mechanism": COMPONENTS_PER_MECHANISM,
            "components": 8 * 2 * COMPONENTS_PER_MECHANISM,
            "projection_seeds": list(PROJECTION_SEEDS),
            "known_within_category_mechanism_truth": True,
            "metadata_weight": 0.0,
        },
        "policy": {
            "minimum_worst_seed_pairwise_f1": MINIMUM_WORST_SEED_PAIRWISE_F1,
            "minimum_worst_seed_cluster_purity": MINIMUM_WORST_SEED_CLUSTER_PURITY,
            "selection_order": list(WEIGHTING_CANDIDATES),
            "selection_principle": (
                "least_aggressive_weighting_that_bounds_nondominant_direction_"
                "influence_and_passes_all_scenarios"
            ),
        },
        "candidates": candidates,
        "selected_candidate": selected,
        "scenarios": scenario_rows,
        "scientific_boundary": {
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "synthetic_effects_only": True,
            "does_not_establish_bonsai_within_category_clusters": True,
            "purpose": (
                "Prevent negligible non-dominant category directions from "
                "determining within-category SAE study clusters."
            ),
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    body = dict(report)
    benchmark_id = str(body.pop("benchmark_id", ""))
    if content_sha256(body) != benchmark_id:
        raise ValueError("clustering geometry benchmark content address mismatch")
    expected = build_report(seed=int(report["seed"]))
    if expected != report or report.get("status") != "pass":
        raise ValueError("clustering geometry benchmark is not reproducible")
    return {
        "schema": "sch.vpd-activation-category-clustering-geometry-verification.v1",
        "status": "pass",
        "benchmark_id": benchmark_id,
        "selected_direction_weighting": report["selected_candidate"][
            "direction_weighting"
        ],
        "scenarios": len(report["scenarios"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or verify the model-free category clustering geometry benchmark."
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
