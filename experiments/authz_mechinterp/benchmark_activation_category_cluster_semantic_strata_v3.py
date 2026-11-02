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
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "id": "four_mechanisms_stable_across_semantic_strata",
        "mode": "stable_four",
        "expected_unstratified": 4,
        "expected_stratified": 4,
    },
    {
        "id": "condition_partition_confound",
        "mode": "condition_confound",
        "expected_unstratified": 4,
        "expected_stratified": 1,
    },
    {
        "id": "template_partition_confound",
        "mode": "template_confound",
        "expected_unstratified": 4,
        "expected_stratified": 1,
    },
)


def _base_rows() -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    category_names = sorted(row["id"] for row in category_contract()["categories"])
    component_ids: list[str] = []
    categories: list[str] = []
    first_factor: list[float] = []
    second_factor: list[float] = []
    for category_index, category in enumerate(category_names):
        for component_index in range(16):
            component_ids.append(f"component-{category_index:02d}-{component_index:02d}")
            categories.append(category)
            first_factor.append(-1.0 if component_index < 8 else 1.0)
            second_factor.append(-1.0 if component_index % 8 < 4 else 1.0)
    return (
        component_ids,
        categories,
        np.asarray(first_factor),
        np.asarray(second_factor),
    )


def _matrix(
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_scale: float,
    second_scale: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.column_stack(
        [first * first_scale, second * second_scale, np.zeros(len(first))]
    )
    return values + rng.normal(0.0, 0.015, size=values.shape)


def scenario_matrices(
    mode: str, *, seed: int
) -> tuple[list[np.ndarray], list[str], list[np.ndarray], list[str], list[str], list[str]]:
    component_ids, categories, first, second = _base_rows()
    aggregate = [
        _matrix(first, second, first_scale=3.0, second_scale=3.0, seed=seed + index)
        for index in range(8)
    ]
    aggregate_kinds = ["aggregate_projection"] * len(aggregate)
    if mode == "stable_four":
        semantic = [
            _matrix(first, second, first_scale=2.8, second_scale=2.8, seed=seed + 100 + index)
            for index in range(4)
        ]
        semantic_kinds = [
            "condition_stratum",
            "condition_stratum",
            "template_group_stratum",
            "template_group_stratum",
        ]
    elif mode == "condition_confound":
        semantic = [
            _matrix(first, second, first_scale=3.0, second_scale=0.02, seed=seed + 100),
            _matrix(first, second, first_scale=0.02, second_scale=3.0, seed=seed + 101),
        ]
        semantic_kinds = ["condition_stratum", "condition_stratum"]
    elif mode == "template_confound":
        semantic = [
            _matrix(first, second, first_scale=3.0, second_scale=0.02, seed=seed + 100),
            _matrix(first, second, first_scale=0.02, second_scale=3.0, seed=seed + 101),
        ]
        semantic_kinds = ["template_group_stratum", "template_group_stratum"]
    else:
        raise ValueError("unknown semantic-stratum scenario")
    return (
        aggregate,
        aggregate_kinds,
        semantic,
        semantic_kinds,
        component_ids,
        categories,
    )


def _summary(selection: dict[str, Any]) -> dict[str, Any]:
    allocation = selection["allocation"]
    histogram: dict[str, int] = {}
    for count in allocation.values():
        key = str(count)
        histogram[key] = histogram.get(key, 0) + 1
    return {
        "allocation": allocation,
        "histogram": dict(sorted(histogram.items())),
        "selection": selection,
    }


def build_report(*, seed: int = 10923) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, scenario in enumerate(SCENARIOS):
        aggregate, aggregate_kinds, semantic, semantic_kinds, ids, categories = (
            scenario_matrices(str(scenario["mode"]), seed=seed + index * 1009)
        )
        unstratified = select_category_cluster_counts_v2(
            aggregate, ids, categories, seed=seed, replicate_kinds=aggregate_kinds
        )
        stratified = select_category_cluster_counts_v2(
            aggregate + semantic,
            ids,
            categories,
            seed=seed,
            replicate_kinds=aggregate_kinds + semantic_kinds,
        )
        expected_unstratified = int(scenario["expected_unstratified"])
        expected_stratified = int(scenario["expected_stratified"])
        passed = all(
            int(value) == expected_unstratified
            for value in unstratified["allocation"].values()
        ) and all(
            int(value) == expected_stratified
            for value in stratified["allocation"].values()
        )
        rows.append(
            {
                "scenario_id": scenario["id"],
                "status": "pass" if passed else "fail",
                "expected_unstratified_cluster_count": expected_unstratified,
                "expected_semantic_stratified_cluster_count": expected_stratified,
                "unstratified": _summary(unstratified),
                "semantic_stratified": _summary(stratified),
            }
        )
    body = {
        "schema": "sch.vpd-activation-category-cluster-semantic-strata-benchmark.v3",
        "status": "pass" if all(row["status"] == "pass" for row in rows) else "fail",
        "seed": seed,
        "source": {
            "runner_sha256": sha256_file(Path(__file__)),
            "selector_v2_sha256": sha256_file(
                Path(__file__).with_name("activation_category_cluster_count_selection_v2.py")
            ),
        },
        "design": {
            "categories": 8,
            "components_per_category": 16,
            "known_ground_truth": True,
            "aggregate_replicates": 8,
            "semantic_strata_required_for_admission": True,
        },
        "summary": {
            "scenarios": len(rows),
            "passed": sum(row["status"] == "pass" for row in rows),
            "confounds_rejected": sum(
                row["semantic_stratified"]["histogram"] == {"1": 8}
                for row in rows
                if "confound" in row["scenario_id"]
            ),
        },
        "scenarios": rows,
        "scientific_boundary": {
            "synthetic_feature_matrices_only": True,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "does_not_select_empirical_bonsai_cluster_counts": True,
            "execution_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = body.pop("benchmark_id", "")
    if identifier != content_sha256(body):
        raise ValueError("semantic-strata benchmark address mismatch")
    if observed != build_report(seed=int(observed["seed"])) or observed["status"] != "pass":
        raise ValueError("semantic-strata benchmark is not reproducible")
    return {
        "schema": "sch.vpd-activation-category-cluster-semantic-strata-verification.v3",
        "status": "pass",
        "benchmark_id": identifier,
        "scenarios": observed["summary"]["scenarios"],
        "confounds_rejected": observed["summary"]["confounds_rejected"],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark semantic-stratum cluster stability.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--seed", type=int, default=10923)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_benchmark(args.verify)
    else:
        if args.output is None: parser.error("--output is required unless verifying")
        result = build_report(seed=args.seed); write_json(args.output, result)
    print(json.dumps({
        "schema": result["schema"], "status": result["status"],
        "benchmark_id": result["benchmark_id"],
        "scenarios": result.get("summary", {}).get("scenarios", result.get("scenarios")),
        "confounds_rejected": result.get("summary", {}).get("confounds_rejected", result.get("confounds_rejected")),
        "execution_authorized": result.get("execution_authorized", False),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
