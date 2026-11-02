from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.activation_categories import category_contract
from experiments.authz_mechinterp.activation_category_cluster_count_selection_v2 import (
    select_category_cluster_counts_v2,
)
from experiments.authz_mechinterp.benchmark_activation_category_cluster_discreteness_v2 import (
    HIDDEN_SIZE,
    _matrix,
    replicate_matrices,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCENARIOS: tuple[dict[str, Any], ...] = (
    {"id": "two_discrete_balanced", "cluster_sizes": (6, 6)},
    {"id": "three_discrete_balanced", "cluster_sizes": (4, 4, 4)},
    {"id": "three_discrete_imbalanced", "cluster_sizes": (4, 4, 8)},
    {"id": "four_discrete_balanced", "cluster_sizes": (4, 4, 4, 4)},
)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def synthetic_discrete_effects(
    cluster_sizes: tuple[int, ...], *, seed: int
) -> tuple[np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(seed)
    category_names = sorted(row["id"] for row in category_contract()["categories"])
    effects: list[np.ndarray] = []
    component_ids: list[str] = []
    component_categories: list[str] = []
    for category_index, category in enumerate(category_names):
        basis, _ = np.linalg.qr(
            rng.normal(size=(HIDDEN_SIZE, len(cluster_sizes)))
        )
        centroids = basis.T
        backgrounds = {
            name: _unit(rng.normal(size=HIDDEN_SIZE)) for name in category_names
        }
        local_index = 0
        for mechanism_index, mechanism_size in enumerate(cluster_sizes):
            centroid = centroids[mechanism_index]
            for _ in range(mechanism_size):
                noise = rng.normal(size=HIDDEN_SIZE)
                noise -= float(noise @ centroid) * centroid
                target = _unit(centroid + 0.025 * _unit(noise)) * 2.0
                row = np.empty(
                    (len(category_names), HIDDEN_SIZE), dtype=np.float64
                )
                for effect_index, effect_category in enumerate(category_names):
                    row[effect_index] = (
                        target
                        if effect_category == category
                        else backgrounds[effect_category] * 0.02
                    )
                effects.append(row)
                component_ids.append(
                    f"component-{category_index:02d}-{local_index:02d}"
                )
                component_categories.append(category)
                local_index += 1
    return np.asarray(effects), component_ids, component_categories


def _histogram(allocation: dict[str, int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in allocation.values():
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def build_report(*, seed: int = 310829) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    for index, scenario in enumerate(SCENARIOS):
        effects, component_ids, categories = synthetic_discrete_effects(
            tuple(int(value) for value in scenario["cluster_sizes"]),
            seed=seed + index * 1009,
        )
        matrices, kinds = replicate_matrices(
            effects, seed=seed + index * 1009 + 1
        )
        selection = select_category_cluster_counts_v2(
            matrices,
            component_ids,
            categories,
            seed=seed,
            replicate_kinds=kinds,
        )
        expected = len(scenario["cluster_sizes"])
        exact_categories = sum(
            int(value) == expected for value in selection["allocation"].values()
        )
        scenarios.append(
            {
                "scenario_id": scenario["id"],
                "known_mechanism_cluster_sizes": list(scenario["cluster_sizes"]),
                "expected_cluster_count_per_category": expected,
                "exact_categories": exact_categories,
                "allocation_histogram": _histogram(selection["allocation"]),
                "selection": selection,
            }
        )
    failing = [
        row for row in scenarios if row["exact_categories"] != 8
    ]
    body = {
        "schema": "sch.vpd-activation-category-cluster-model-order-diagnostic.v2",
        "status": "diagnostic_complete",
        "seed": seed,
        "source": {
            "runner_sha256": sha256_file(Path(__file__)),
            "selector_v2_sha256": sha256_file(
                Path(__file__).with_name(
                    "activation_category_cluster_count_selection_v2.py"
                )
            ),
        },
        "design": {
            "categories": 8,
            "known_ground_truth": True,
            "independent_validation_replicates": True,
            "minimum_cluster_size": 4,
            "maximum_cluster_count": 4,
        },
        "summary": {
            "scenarios": len(scenarios),
            "exact_scenarios": len(scenarios) - len(failing),
            "nonexact_scenarios": len(failing),
            "nonexact_scenario_ids": [row["scenario_id"] for row in failing],
        },
        "scenarios": scenarios,
        "scientific_boundary": {
            "synthetic_effects_only": True,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "does_not_select_empirical_bonsai_cluster_counts": True,
            "execution_authorized": False,
        },
    }
    return {**body, "diagnostic_id": content_sha256(body)}


def verify_diagnostic(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = body.pop("diagnostic_id", "")
    if identifier != content_sha256(body):
        raise ValueError("cluster model-order diagnostic address mismatch")
    if observed != build_report(seed=int(observed["seed"])):
        raise ValueError("cluster model-order diagnostic is not reproducible")
    return {
        "schema": "sch.vpd-activation-category-cluster-model-order-diagnostic-verification.v2",
        "status": "pass",
        "diagnostic_id": identifier,
        "scenarios": observed["summary"]["scenarios"],
        "nonexact_scenarios": observed["summary"]["nonexact_scenarios"],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose V2 cluster model-order recovery.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--seed", type=int, default=310829)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_diagnostic(args.verify)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_report(seed=args.seed)
        write_json(args.output, result)
    print(json.dumps({
        "schema": result["schema"],
        "status": result["status"],
        "diagnostic_id": result["diagnostic_id"],
        "scenarios": result.get("summary", {}).get("scenarios", result.get("scenarios")),
        "nonexact_scenarios": result.get("summary", {}).get("nonexact_scenarios", result.get("nonexact_scenarios")),
        "execution_authorized": result.get("execution_authorized", False),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
