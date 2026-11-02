from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.activation_categories import category_contract
from experiments.authz_mechinterp.activation_category_cluster_count_selection_v2 import (
    DEFAULT_MINIMUM_CATEGORY_CLUSTER_BOUNDARY_GAP_RATIO,
    DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE_P10,
    select_category_cluster_counts_v2,
)
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


HIDDEN_SIZE = 512
PROJECTION_DIMENSION = 64
COMPONENTS_PER_CATEGORY = 12
PROJECTION_SEEDS = (17, 29, 41, 53, 67, 79, 97, 109)
SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "id": "single_compact_mechanism",
        "mode": "single_compact",
        "expected_v2_cluster_count": 1,
        "legacy_false_split_required": False,
    },
    {
        "id": "two_discrete_mechanisms",
        "mode": "two_discrete",
        "expected_v2_cluster_count": 2,
        "legacy_false_split_required": False,
    },
    {
        "id": "continuous_direction_arc",
        "mode": "continuous_direction_arc",
        "expected_v2_cluster_count": 1,
        "legacy_false_split_required": True,
    },
    {
        "id": "continuous_strength_gradient",
        "mode": "continuous_strength_gradient",
        "expected_v2_cluster_count": 1,
        "legacy_false_split_required": True,
    },
)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("cannot normalize zero synthetic vector")
    return vector / norm


def _orthogonal_to(
    rng: np.random.Generator, references: list[np.ndarray]
) -> np.ndarray:
    vector = rng.normal(size=HIDDEN_SIZE)
    for reference in references:
        vector -= float(vector @ reference) * reference
    return _unit(vector)


def synthetic_effects(
    scenario: dict[str, Any], *, seed: int
) -> tuple[np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(seed)
    categories = sorted(row["id"] for row in category_contract()["categories"])
    effects: list[np.ndarray] = []
    component_ids: list[str] = []
    component_categories: list[str] = []
    mode = str(scenario["mode"])
    for category_index, category in enumerate(categories):
        first = _unit(rng.normal(size=HIDDEN_SIZE))
        second = _orthogonal_to(rng, [first])
        background = {
            name: _unit(rng.normal(size=HIDDEN_SIZE)) for name in categories
        }
        for component_index in range(COMPONENTS_PER_CATEGORY):
            if mode == "single_compact":
                noise = _orthogonal_to(rng, [first])
                target = _unit(first + 0.16 * noise) * 2.0
            elif mode == "two_discrete":
                centroid = first if component_index < 6 else second
                noise = _orthogonal_to(rng, [centroid])
                target = _unit(centroid + 0.04 * noise) * 2.0
            elif mode == "continuous_direction_arc":
                position = component_index / (COMPONENTS_PER_CATEGORY - 1)
                angle = -0.90 + 1.80 * position
                target = (np.cos(angle) * first + np.sin(angle) * second) * 2.0
            elif mode == "continuous_strength_gradient":
                position = component_index / (COMPONENTS_PER_CATEGORY - 1)
                target = first * (0.70 + 2.10 * position)
            else:
                raise ValueError("unknown cluster discreteness scenario")
            row = np.empty((len(categories), HIDDEN_SIZE), dtype=np.float64)
            for effect_index, effect_category in enumerate(categories):
                row[effect_index] = (
                    target
                    if effect_category == category
                    else background[effect_category] * 0.02
                )
            effects.append(row)
            component_ids.append(
                f"component-{category_index:02d}-{component_index:02d}"
            )
            component_categories.append(category)
    return np.asarray(effects), component_ids, component_categories


def _matrix(category_effects: np.ndarray, *, projection_seed: int) -> np.ndarray:
    aggregate = category_effects.mean(axis=1)
    aggregate_features = _project_effects(
        aggregate,
        seed=projection_seed,
        projection_dim=PROJECTION_DIMENSION,
    )
    category_features = activation_category_profile_features(
        category_effects,
        seed=projection_seed,
        projection_dim=PROJECTION_DIMENSION,
    )["features"]
    return np.column_stack([aggregate_features, category_features])


def replicate_matrices(
    category_effects: np.ndarray, *, seed: int
) -> tuple[list[np.ndarray], list[str]]:
    matrices: list[np.ndarray] = []
    kinds: list[str] = []
    for projection_seed in PROJECTION_SEEDS:
        matrices.append(_matrix(category_effects, projection_seed=projection_seed))
        kinds.append("discovery_projection")
    rng = np.random.default_rng(seed)
    strengths = np.linalg.norm(category_effects, axis=2, keepdims=True)
    validation = category_effects.copy()
    noise = rng.normal(size=validation.shape)
    noise /= np.maximum(np.linalg.norm(noise, axis=2, keepdims=True), 1e-12)
    validation += noise * strengths * 0.002
    for projection_seed in PROJECTION_SEEDS:
        matrices.append(_matrix(validation, projection_seed=projection_seed))
        kinds.append("independent_validation_projection")
    return matrices, kinds


def _allocation_summary(selection: dict[str, Any]) -> dict[str, Any]:
    allocation = {
        str(category): int(count)
        for category, count in sorted(selection["allocation"].items())
    }
    histogram: dict[str, int] = {}
    for count in allocation.values():
        key = str(count)
        histogram[key] = histogram.get(key, 0) + 1
    return {
        "allocation": allocation,
        "cluster_count_histogram": dict(sorted(histogram.items())),
        "nontrivial_categories": sum(count > 1 for count in allocation.values()),
    }


def build_report(*, seed: int = 250829) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    legacy_false_splits = 0
    for index, scenario in enumerate(SCENARIOS):
        effects, component_ids, categories = synthetic_effects(
            scenario, seed=seed + index * 1009
        )
        matrices, kinds = replicate_matrices(
            effects, seed=seed + index * 1009 + 1
        )
        legacy = select_category_cluster_counts(
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
        v2 = select_category_cluster_counts_v2(
            matrices,
            component_ids,
            categories,
            seed=seed,
            replicate_kinds=kinds,
        )
        legacy_summary = _allocation_summary(legacy)
        v2_summary = _allocation_summary(v2)
        expected = int(scenario["expected_v2_cluster_count"])
        v2_expected = all(
            count == expected for count in v2_summary["allocation"].values()
        )
        false_split = bool(scenario["legacy_false_split_required"]) and int(
            legacy_summary["nontrivial_categories"]
        ) > 0
        legacy_false_splits += int(false_split)
        passed = v2_expected and (
            not scenario["legacy_false_split_required"] or false_split
        )
        rows.append(
            {
                "scenario_id": scenario["id"],
                "status": "pass" if passed else "fail",
                "expected_v2_cluster_count_per_category": expected,
                "legacy_allocation_summary": legacy_summary,
                "v2_allocation_summary": v2_summary,
                "legacy_false_split_reproduced": false_split,
                "legacy_selection": legacy,
                "v2_selection": v2,
            }
        )
    required_false_splits = sum(
        bool(scenario["legacy_false_split_required"]) for scenario in SCENARIOS
    )
    body = {
        "schema": "sch.vpd-activation-category-cluster-discreteness-benchmark.v2",
        "status": "pass" if all(row["status"] == "pass" for row in rows) else "fail",
        "seed": seed,
        "source": {
            "runner_sha256": sha256_file(Path(__file__)),
            "v1_selector_sha256": sha256_file(
                Path(__file__).with_name("cluster_vpd_for_sae.py")
            ),
            "v2_selector_sha256": sha256_file(
                Path(__file__).with_name(
                    "activation_category_cluster_count_selection_v2.py"
                )
            ),
        },
        "design": {
            "hidden_size": HIDDEN_SIZE,
            "projection_dimension": PROJECTION_DIMENSION,
            "projection_seeds": list(PROJECTION_SEEDS),
            "replicates": len(PROJECTION_SEEDS) * 2,
            "categories": 8,
            "components_per_category": COMPONENTS_PER_CATEGORY,
            "continuous_negative_controls": required_false_splits,
        },
        "policy_v2": {
            "minimum_replicate_adjusted_rand": (
                DEFAULT_MINIMUM_CATEGORY_CLUSTER_PROJECTION_STABILITY
            ),
            "minimum_replicate_mean_silhouette": (
                DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE
            ),
            "minimum_replicate_silhouette_p10": (
                DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE_P10
            ),
            "minimum_replicate_boundary_gap_ratio": (
                DEFAULT_MINIMUM_CATEGORY_CLUSTER_BOUNDARY_GAP_RATIO
            ),
            "minimum_cluster_size": DEFAULT_MINIMUM_CATEGORY_CLUSTER_SIZE,
            "continuous_geometry_rule": "fall_back_to_one_cluster_per_category",
        },
        "summary": {
            "scenarios": len(rows),
            "passed": sum(row["status"] == "pass" for row in rows),
            "required_legacy_false_splits": required_false_splits,
            "legacy_false_splits_reproduced": legacy_false_splits,
        },
        "scenarios": rows,
        "scientific_boundary": {
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "synthetic_effects_only": True,
            "does_not_select_empirical_bonsai_cluster_counts": True,
            "purpose": (
                "Prevent stable continuous activation geometry from being "
                "misreported as multiple discrete within-category mechanisms."
            ),
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if content_sha256(body) != identifier:
        raise ValueError("cluster discreteness benchmark content address mismatch")
    expected = build_report(seed=int(observed["seed"]))
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("cluster discreteness benchmark is not reproducible")
    return {
        "schema": "sch.vpd-activation-category-cluster-discreteness-verification.v2",
        "status": "pass",
        "benchmark_id": identifier,
        "scenarios": int(observed["summary"]["scenarios"]),
        "legacy_false_splits_reproduced": int(
            observed["summary"]["legacy_false_splits_reproduced"]
        ),
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the V2 within-category cluster discreteness gate."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--seed", type=int, default=250829)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_benchmark(args.verify)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_report(seed=args.seed)
        write_json(args.output, result)
    print(
        json.dumps(
            {
                "schema": str(result["schema"]),
                "status": str(result["status"]),
                "benchmark_id": str(result["benchmark_id"]),
                "scenarios": int(
                    result.get("summary", {}).get(
                        "scenarios", result.get("scenarios", 0)
                    )
                ),
                "execution_authorized": bool(
                    result.get("execution_authorized", False)
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
