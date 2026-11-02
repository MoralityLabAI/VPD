from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.admit_vpd_cluster_candidate_discreteness_v2 import (
    build_admission_report as build_v2_admission_report,
)
from experiments.authz_mechinterp.admit_vpd_cluster_candidate_semantic_strata_v3 import (
    build_admission_report as build_v3_admission_report,
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
from experiments.authz_mechinterp.effect_uncertainty import countsketch
from experiments.authz_mechinterp.merge_vpd_effect_shards import (
    write_deterministic_npz,
)


SCENARIOS = (
    {
        "id": "four_mechanisms_semantically_stable_admitted",
        "mode": "stable_four",
        "expected_v2": True,
        "expected_v3": True,
    },
    {
        "id": "condition_partition_confound_rejected",
        "mode": "condition_confound",
        "expected_v2": True,
        "expected_v3": False,
    },
)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def synthetic_effects(
    *, truths: list[str], categories: list[str], mode: str, seed: int, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    hidden_size = 64
    bases: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    backgrounds: dict[str, np.ndarray] = {}
    for category in categories:
        basis, _ = np.linalg.qr(rng.normal(size=(hidden_size, 3)))
        bases[category] = (basis[:, 0], basis[:, 1], basis[:, 2])
        backgrounds[category] = _unit(rng.normal(size=hidden_size)) * 0.04 * scale
    category_index = {category: index for index, category in enumerate(categories)}
    local_position: dict[int, int] = {}
    for category in categories:
        indices = [index for index, truth in enumerate(truths) if truth == category]
        for position, component_index in enumerate(indices):
            local_position[component_index] = position
    aggregate = np.empty((len(truths), len(categories), hidden_size), dtype=np.float64)
    condition_means = np.empty(
        (len(truths), len(categories) * 2, hidden_size), dtype=np.float64
    )
    for component_index, truth in enumerate(truths):
        position = local_position[component_index]
        first_factor = -1.0 if position < 8 else 1.0
        second_factor = -1.0 if position % 8 < 4 else 1.0
        base, first, second = bases[truth]
        target = (2.0 * base + first_factor * first + second_factor * second) * scale
        if mode == "stable_four":
            target_conditions = (target, target)
        elif mode == "condition_confound":
            target_conditions = (
                (2.0 * base + 2.0 * first_factor * first) * scale,
                (2.0 * base + 2.0 * second_factor * second) * scale,
            )
        else:
            raise ValueError("unknown semantic-stratum admission benchmark mode")
        for effect_index, effect_category in enumerate(categories):
            if effect_category == truth:
                aggregate[component_index, effect_index] = target
                condition_means[component_index, effect_index * 2] = target_conditions[0]
                condition_means[component_index, effect_index * 2 + 1] = target_conditions[1]
            else:
                aggregate[component_index, effect_index] = backgrounds[effect_category]
                condition_means[component_index, effect_index * 2] = backgrounds[
                    effect_category
                ]
                condition_means[component_index, effect_index * 2 + 1] = backgrounds[
                    effect_category
                ]
    return aggregate, condition_means


def replace_condition_statistics(path: Path, condition_means: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    counts = np.asarray(arrays["condition_episode_counts"], dtype=np.int64)
    dimension = int(np.asarray(arrays["uncertainty_countsketch_dimension"]).item())
    seed = int(np.asarray(arrays["uncertainty_countsketch_seed"]).item())
    arrays["delta_h_countsketch_condition_sums"] = (
        countsketch(condition_means, dimension=dimension, seed=seed)
        * counts[None, :, None]
    ).astype(np.float32)
    write_deterministic_npz(path, arrays)


def run_scenario(
    scenario: dict[str, Any], *, project_root: Path, seed: int
) -> dict[str, Any]:
    rows, truths, categories = synthetic_inventory(
        specific_per_category=16, shared_components=0
    )
    component_ids = [str(row["component_id"]) for row in rows]
    split_manifest_path = project_root / (
        "artifacts/vpd_activation_category_split_v3/split_manifest.json"
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
        for offset, (role, scale) in enumerate(
            (("discovery", 1.0), ("calibration", 0.96), ("cluster_validation", 0.92))
        ):
            values, condition_means = synthetic_effects(
                truths=truths,
                categories=categories,
                mode=str(scenario["mode"]),
                seed=seed,
                scale=scale,
            )
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
            replace_condition_statistics(effects[role], condition_means)
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
            cluster_validation_effects_merge_manifest_path=merges["cluster_validation"],
            output_dir=bundle,
            cluster_count=32,
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
            category_cluster_projection_replicates=8,
            maximum_clusters_per_category=4,
            minimum_category_cluster_size=4,
            require_category_cluster_sampling_stability=True,
            require_category_cluster_cross_archive_stability=True,
        )
        inputs = {
            "bundle_dir": bundle,
            "component_inventory_path": inventory,
            "discovery_effects_path": effects["discovery"],
            "discovery_merge_manifest_path": merges["discovery"],
            "cluster_validation_effects_path": effects["cluster_validation"],
            "cluster_validation_merge_manifest_path": merges["cluster_validation"],
        }
        v2 = build_v2_admission_report(**inputs)
        v3 = build_v3_admission_report(**inputs)
        observed_v2 = bool(v2["admission"]["eligible"])
        observed_v3 = bool(v3["admission"]["eligible"])
        passed = (
            observed_v2 == bool(scenario["expected_v2"])
            and observed_v3 == bool(scenario["expected_v3"])
        )
        return {
            "scenario_id": scenario["id"],
            "status": "pass" if passed else "fail",
            "expected_v2_admitted": bool(scenario["expected_v2"]),
            "observed_v2_admitted": observed_v2,
            "expected_v3_admitted": bool(scenario["expected_v3"]),
            "observed_v3_admitted": observed_v3,
            "candidate_allocation": manifest["algorithm"][
                "activation_category_compartments"
            ]["cluster_allocation"],
            "v2_mismatch_count": v2["admission"]["mismatch_count"],
            "v3_mismatch_count": v3["admission"]["mismatch_count"],
            "v3_allocation": v3["semantic_strata_selection"]["allocation"],
            "v3_semantic_replicate_count": v3["replicate_evidence"][
                "semantic_stratum_matrices"
            ],
            "base_bundle_verification": v3["base_bundle_verification"],
        }


def build_report(project_root: Path, *, seed: int = 40923) -> dict[str, Any]:
    scenarios = [
        run_scenario(scenario, project_root=project_root, seed=seed + index * 1009)
        for index, scenario in enumerate(SCENARIOS)
    ]
    body = {
        "schema": "sch.vpd-cluster-candidate-semantic-strata-admission-benchmark.v3",
        "status": "pass" if all(row["status"] == "pass" for row in scenarios) else "fail",
        "seed": seed,
        "source": {
            "runner_sha256": sha256_file(Path(__file__)),
            "admission_v2_runner_sha256": sha256_file(
                Path(__file__).with_name("admit_vpd_cluster_candidate_discreteness_v2.py")
            ),
            "admission_v3_runner_sha256": sha256_file(
                Path(__file__).with_name("admit_vpd_cluster_candidate_semantic_strata_v3.py")
            ),
        },
        "design": {
            "production_shaped_v1_bundle_verified_first": True,
            "categories": 8,
            "components": 128,
            "discovery_and_cluster_validation_semantic_statistics": True,
            "scenarios": len(scenarios),
        },
        "scenarios": scenarios,
        "scientific_boundary": {
            "synthetic_effects_only": True,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "does_not_admit_an_empirical_bonsai_candidate": True,
            "execution_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = body.pop("benchmark_id", "")
    if identifier != content_sha256(body):
        raise ValueError("semantic-strata admission benchmark address mismatch")
    if observed != build_report(project_root, seed=int(observed["seed"])) or observed["status"] != "pass":
        raise ValueError("semantic-strata admission benchmark is not reproducible")
    return {
        "schema": "sch.vpd-cluster-candidate-semantic-strata-admission-benchmark-verification.v3",
        "status": "pass", "benchmark_id": identifier,
        "scenarios": len(observed["scenarios"]), "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark production-shaped semantic-strata admission V3.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--seed", type=int, default=40923)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.verify is not None:
        result = verify_benchmark(args.verify, project_root=root)
    else:
        if args.output is None: parser.error("--output is required unless verifying")
        result = build_report(root, seed=args.seed); write_json(args.output, result)
    print(json.dumps({
        "schema": result["schema"], "status": result["status"],
        "benchmark_id": result["benchmark_id"],
        "scenarios": len(result["scenarios"]) if isinstance(result.get("scenarios"), list) else result["scenarios"],
        "execution_authorized": result.get("execution_authorized", False),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
