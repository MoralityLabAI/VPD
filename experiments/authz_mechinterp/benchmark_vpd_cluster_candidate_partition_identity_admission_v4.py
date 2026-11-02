from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.admit_vpd_cluster_candidate_partition_identity_v4 import (
    build_admission_report as build_v4_admission_report,
)
from experiments.authz_mechinterp.admit_vpd_cluster_candidate_semantic_strata_v3 import (
    build_admission_report as build_v3_admission_report,
)
from experiments.authz_mechinterp.benchmark_activation_category_cluster_partition_identity_v4 import (
    verify_benchmark as verify_partition_benchmark,
)
from experiments.authz_mechinterp.benchmark_activation_category_protocol import (
    synthetic_inventory,
    write_effect_archive,
    write_synthetic_merge_manifest,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_candidate_semantic_strata_admission_v3 import (
    replace_condition_statistics,
    synthetic_effects,
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
        "id": "four_mechanisms_exact_partition_admitted",
        "mode": "stable_four",
        "expected_v3": True,
        "expected_v4": True,
    },
    {
        "id": "condition_partition_confound_rejected_before_membership",
        "mode": "condition_confound",
        "expected_v3": False,
        "expected_v4": False,
    },
)


def run_scenario(
    scenario: dict[str, Any], *, project_root: Path, seed: int
) -> dict[str, Any]:
    rows, truths, categories = synthetic_inventory(
        specific_per_category=16, shared_components=0
    )
    component_ids = [str(row["component_id"]) for row in rows]
    split_manifest_path = (
        project_root / "artifacts/vpd_activation_category_split_v3/split_manifest.json"
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
        v3 = build_v3_admission_report(**inputs)
        v4 = build_v4_admission_report(**inputs)
        observed_v3 = bool(v3["admission"]["eligible"])
        observed_v4 = bool(v4["admission"]["eligible"])
        passed = (
            observed_v3 == bool(scenario["expected_v3"])
            and observed_v4 == bool(scenario["expected_v4"])
        )
        return {
            "scenario_id": scenario["id"],
            "status": "pass" if passed else "fail",
            "expected_v3_admitted": bool(scenario["expected_v3"]),
            "observed_v3_admitted": observed_v3,
            "expected_v4_admitted": bool(scenario["expected_v4"]),
            "observed_v4_admitted": observed_v4,
            "candidate_allocation": manifest["algorithm"][
                "activation_category_compartments"
            ]["cluster_allocation"],
            "v4_allocation_mismatch_count": v4["admission"][
                "allocation_mismatch_count"
            ],
            "v4_partition_rejection_count": v4["admission"][
                "partition_rejection_count"
            ],
            "v4_exact_categories": v4["partition_identity"]["summary"][
                "exact_categories"
            ],
            "v4_total_replicates": v4["replicate_evidence"]["total_matrices"],
            "v4_semantic_replicates": v4["replicate_evidence"][
                "semantic_stratum_matrices"
            ],
            "base_bundle_verification": v4["base_bundle_verification"],
        }


def build_report(project_root: Path, *, seed: int = 130923) -> dict[str, Any]:
    partition_benchmark_path = (
        project_root
        / "artifacts/vpd_activation_category_cluster_partition_identity_benchmark_v4/report.json"
    )
    partition_verification = verify_partition_benchmark(partition_benchmark_path)
    scenarios = [
        run_scenario(scenario, project_root=project_root, seed=seed + index * 1009)
        for index, scenario in enumerate(SCENARIOS)
    ]
    body = {
        "schema": "sch.vpd-cluster-candidate-partition-identity-admission-benchmark.v4",
        "status": "pass" if all(row["status"] == "pass" for row in scenarios) else "fail",
        "seed": seed,
        "source": {
            "runner_sha256": sha256_file(Path(__file__)),
            "admission_v4_runner_sha256": sha256_file(
                Path(__file__).with_name(
                    "admit_vpd_cluster_candidate_partition_identity_v4.py"
                )
            ),
            "partition_identity_benchmark": {
                "path": partition_benchmark_path.relative_to(project_root).as_posix(),
                "sha256": sha256_file(partition_benchmark_path),
                "verification": partition_verification,
            },
        },
        "design": {
            "production_bundle_and_archive_contract": True,
            "discovery_and_independent_validation": True,
            "pair_condition_and_template_omissions": True,
            "exact_label_invariant_candidate_membership_required": True,
        },
        "summary": {
            "scenarios": len(scenarios),
            "passed": sum(row["status"] == "pass" for row in scenarios),
            "stable_candidate_admitted": scenarios[0]["observed_v4_admitted"],
            "semantic_confound_rejected": not scenarios[1]["observed_v4_admitted"],
        },
        "scenarios": scenarios,
        "scientific_boundary": {
            "synthetic_effect_archives_only": True,
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
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("partition identity admission benchmark address mismatch")
    if observed != build_report(project_root, seed=int(observed["seed"])):
        raise ValueError("partition identity admission benchmark is not reproducible")
    if observed["status"] != "pass":
        raise ValueError("partition identity admission benchmark did not pass")
    return {
        "schema": "sch.vpd-cluster-candidate-partition-identity-admission-benchmark-verification.v4",
        "status": "pass",
        "benchmark_id": identifier,
        "scenarios": observed["summary"]["scenarios"],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark production-shaped partition-identity admission V4."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--seed", type=int, default=130923)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.verify is not None:
        result = verify_benchmark(args.verify, project_root=project_root)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_report(project_root, seed=args.seed)
        write_json(args.output, result)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "benchmark_id": result["benchmark_id"],
                "scenarios": result.get("summary", {}).get("scenarios"),
                "execution_authorized": result.get("execution_authorized", False),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
