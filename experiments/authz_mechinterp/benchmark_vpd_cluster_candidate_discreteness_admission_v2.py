from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.admit_vpd_cluster_candidate_discreteness_v2 import (
    build_admission_report,
)
from experiments.authz_mechinterp.benchmark_activation_category_protocol import (
    synthetic_inventory,
    write_effect_archive,
    write_synthetic_merge_manifest,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    build_bundle,
    content_sha256,
    sha256_file,
    write_json,
    write_jsonl,
)


SCENARIOS = (
    {
        "id": "two_discrete_mechanisms_admitted",
        "mode": "two_discrete",
        "selector_version": "v1",
        "expected_admitted": True,
    },
    {
        "id": "continuous_strength_gradient_rejected",
        "mode": "continuous_strength_gradient",
        "selector_version": "v1",
        "expected_admitted": False,
    },
    {
        "id": "native_v2_two_discrete_mechanisms_admitted",
        "mode": "two_discrete",
        "selector_version": "v2",
        "expected_admitted": True,
    },
)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _category_delta(
    *,
    rows: list[dict[str, Any]],
    truths: list[str],
    categories: list[str],
    mode: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hidden_size = 64
    directions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for category in categories:
        first = _unit(rng.normal(size=hidden_size))
        second = rng.normal(size=hidden_size)
        second -= float(second @ first) * first
        directions[category] = (first, _unit(second))
    indices_by_category = {
        category: [index for index, truth in enumerate(truths) if truth == category]
        for category in categories
    }
    local_positions = {
        component_index: position
        for category in categories
        for position, component_index in enumerate(indices_by_category[category])
    }
    delta = np.empty((len(rows), len(categories), hidden_size), dtype=np.float64)
    category_index = {category: index for index, category in enumerate(categories)}
    for component_index, truth in enumerate(truths):
        if truth not in category_index:
            raise ValueError("admission benchmark excludes shared synthetic components")
        position = local_positions[component_index]
        first, second = directions[truth]
        if mode == "two_discrete":
            target = first if position < 6 else second
            target = target * 2.0
        elif mode == "continuous_strength_gradient":
            target = first * (0.70 + 2.10 * position / 11.0)
        else:
            raise ValueError("unknown admission benchmark mode")
        for effect_index, effect_category in enumerate(categories):
            if effect_category == truth:
                delta[component_index, effect_index] = target
            else:
                delta[component_index, effect_index] = (
                    directions[effect_category][0] * 0.04
                )
    return delta


def run_scenario(
    scenario: dict[str, Any], *, project_root: Path, seed: int
) -> dict[str, Any]:
    rows, truths, categories = synthetic_inventory(
        specific_per_category=12,
        shared_components=0,
    )
    component_ids = [str(row["component_id"]) for row in rows]
    discovery = _category_delta(
        rows=rows,
        truths=truths,
        categories=categories,
        mode=str(scenario["mode"]),
        seed=seed,
    )
    calibration = discovery * 0.96
    validation = discovery * 0.92
    split_manifest_path = (
        project_root
        / "artifacts/vpd_activation_category_split_v3/split_manifest.json"
    )
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    category_counts = np.asarray(
        [
            split_manifest["splits"]["discovery"]["activation_category_counts"][
                category
            ]
            for category in categories
        ],
        dtype=np.int64,
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        inventory = root / "inventory.jsonl"
        write_jsonl(inventory, rows)
        effects: dict[str, Path] = {}
        merges: dict[str, Path] = {}
        for offset, (role, values) in enumerate(
            (
                ("discovery", discovery),
                ("calibration", calibration),
                ("cluster_validation", validation),
            )
        ):
            effects[role] = root / f"{role}.npz"
            write_effect_archive(
                effects[role],
                component_ids=component_ids,
                category_names=categories,
                category_counts=category_counts,
                category_delta=values,
                block_noise=0.01,
                seed=seed + offset + 1,
                truths=truths,
            )
            merges[role] = root / f"{role}_merge.json"
            write_synthetic_merge_manifest(
                merges[role],
                effects_path=effects[role],
                split_manifest=split_manifest,
                split_manifest_path=split_manifest_path,
                role=role,
            )
        bundle = root / "bundle"
        manifest = build_bundle(
            rows,
            inventory_sha256=sha256_file(inventory),
            effects_path=effects["discovery"],
            calibration_effects_path=effects["calibration"],
            cluster_validation_effects_path=effects["cluster_validation"],
            effects_merge_manifest_path=merges["discovery"],
            calibration_effects_merge_manifest_path=merges["calibration"],
            cluster_validation_effects_merge_manifest_path=(
                merges["cluster_validation"]
            ),
            output_dir=bundle,
            cluster_count=16,
            seed=1707,
            projection_dim=16,
            metadata_weight=0.0,
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
            category_cluster_count_selector_version=str(
                scenario["selector_version"]
            ),
            category_cluster_projection_replicates=8,
            maximum_clusters_per_category=4,
            minimum_category_cluster_size=4,
            require_category_cluster_sampling_stability=True,
            require_category_cluster_cross_archive_stability=True,
        )
        report = build_admission_report(
            bundle_dir=bundle,
            component_inventory_path=inventory,
            discovery_effects_path=effects["discovery"],
            discovery_merge_manifest_path=merges["discovery"],
            cluster_validation_effects_path=effects["cluster_validation"],
            cluster_validation_merge_manifest_path=merges["cluster_validation"],
        )
        expected = bool(scenario["expected_admitted"])
        observed = bool(report["admission"]["eligible"])
        candidate_allocation = manifest["algorithm"][
            "activation_category_compartments"
        ]["cluster_allocation"]
        return {
            "scenario_id": scenario["id"],
            "status": "pass" if expected == observed else "fail",
            "expected_admitted": expected,
            "observed_admitted": observed,
            "candidate_allocation": candidate_allocation,
            "v2_allocation": report["selection_v2"]["allocation"],
            "mismatch_count": report["admission"]["mismatch_count"],
            "admission_status": report["status"],
            "source_selection_schema": report["candidate"][
                "legacy_selection_schema"
            ],
            "native_v2_bundle_reverified": report["admission"][
                "native_v2_bundle_reverified"
            ],
            "base_bundle_verification": report["base_bundle_verification"],
        }


def build_report(project_root: Path, *, seed: int = 260829) -> dict[str, Any]:
    scenarios = [
        run_scenario(scenario, project_root=project_root, seed=seed + index * 1009)
        for index, scenario in enumerate(SCENARIOS)
    ]
    body = {
        "schema": "sch.vpd-cluster-candidate-discreteness-admission-benchmark.v2",
        "status": "pass" if all(row["status"] == "pass" for row in scenarios) else "fail",
        "seed": seed,
        "source": {
            "runner_sha256": sha256_file(Path(__file__)),
            "admission_runner_sha256": sha256_file(
                Path(__file__).with_name(
                    "admit_vpd_cluster_candidate_discreteness_v2.py"
                )
            ),
            "base_cluster_runner_sha256": sha256_file(
                Path(__file__).with_name("cluster_vpd_for_sae.py")
            ),
        },
        "design": {
            "scenarios": len(scenarios),
            "components": 96,
            "categories": 8,
            "discovery_and_independent_validation_archives": True,
            "pair_block_sufficient_statistics": True,
            "base_bundle_verified_before_v2_admission": True,
        },
        "scenarios": scenarios,
        "scientific_boundary": {
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "synthetic_effects_only": True,
            "does_not_admit_an_empirical_bonsai_candidate": True,
            "execution_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if content_sha256(body) != identifier:
        raise ValueError("candidate admission benchmark address mismatch")
    expected = build_report(project_root, seed=int(observed["seed"]))
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("candidate admission benchmark is not reproducible")
    return {
        "schema": "sch.vpd-cluster-candidate-discreteness-admission-benchmark-verification.v2",
        "status": "pass",
        "benchmark_id": identifier,
        "scenarios": len(observed["scenarios"]),
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark independent cluster-candidate discreteness admission V2."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--seed", type=int, default=260829)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.verify is not None:
        result = verify_benchmark(args.verify, project_root=root)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_report(root, seed=args.seed)
        write_json(args.output, result)
    print(
        json.dumps(
            {
                "schema": str(result["schema"]),
                "status": str(result["status"]),
                "benchmark_id": str(result["benchmark_id"]),
                "scenarios": (
                    len(result["scenarios"])
                    if isinstance(result.get("scenarios"), list)
                    else int(result["scenarios"])
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
