from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.activation_categories import category_contract
from experiments.authz_mechinterp.activation_category_cluster_count_selection_v2 import (
    select_category_cluster_counts_v2,
)
from experiments.authz_mechinterp.activation_category_cluster_partition_identity_v4 import (
    assess_partition_identity_v4,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


def _fixture(seed: int) -> tuple[
    list[np.ndarray], list[str], list[str], list[str], dict[str, str]
]:
    rng = np.random.default_rng(seed)
    category_names = sorted(row["id"] for row in category_contract()["categories"])
    component_ids: list[str] = []
    categories: list[str] = []
    candidate: dict[str, str] = {}
    base_rows: list[np.ndarray] = []
    for category_index, category in enumerate(category_names):
        basis, _ = np.linalg.qr(rng.normal(size=(24, 2)))
        for mechanism_index in range(2):
            for local_index in range(6):
                component_id = (
                    f"component-{category_index:02d}-{mechanism_index}-{local_index:02d}"
                )
                noise = rng.normal(size=24)
                noise -= float(noise @ basis[:, mechanism_index]) * basis[:, mechanism_index]
                row = 3.0 * basis[:, mechanism_index] + 0.025 * (
                    noise / max(float(np.linalg.norm(noise)), 1e-12)
                )
                component_ids.append(component_id)
                categories.append(category)
                candidate[component_id] = f"{category}-mechanism-{mechanism_index}"
                base_rows.append(row)
    base = np.asarray(base_rows, dtype=np.float64)
    matrices = [
        base + np.random.default_rng(seed + 100 + index).normal(
            scale=0.002, size=base.shape
        )
        for index in range(6)
    ]
    kinds = [
        "discovery_projection",
        "discovery_pair_block",
        "validation_projection",
        "validation_pair_block",
        "condition_stratum",
        "template_group_stratum",
    ]
    return matrices, kinds, component_ids, categories, candidate


def _swap_membership(
    candidate: dict[str, str], component_ids: list[str], categories: list[str]
) -> tuple[dict[str, str], str, list[str]]:
    result = copy.deepcopy(candidate)
    target_category = sorted(set(categories))[0]
    local = [
        component_id
        for component_id, category in zip(component_ids, categories, strict=True)
        if category == target_category
    ]
    first = local[0]
    second = local[6]
    result[first], result[second] = result[second], result[first]
    return result, target_category, sorted((first, second))


def _rename_labels(candidate: dict[str, str]) -> dict[str, str]:
    names = sorted(set(candidate.values()))
    mapping = {name: f"renamed-{len(names) - index:03d}" for index, name in enumerate(names)}
    return {component_id: mapping[name] for component_id, name in candidate.items()}


def build_report(*, seed: int = 120923) -> dict[str, Any]:
    matrices, kinds, component_ids, categories, candidate = _fixture(seed)
    selection = select_category_cluster_counts_v2(
        matrices,
        component_ids,
        categories,
        seed=1707,
        replicate_kinds=kinds,
    )
    exact_two = all(value == 2 for value in selection["allocation"].values())
    stable = assess_partition_identity_v4(
        replicate_matrices=matrices,
        replicate_kinds=kinds,
        component_ids=component_ids,
        categories=categories,
        selected_allocation=selection["allocation"],
        candidate_cluster_ids=candidate,
        seed=1707,
    )
    changed_candidate, target_category, changed_ids = _swap_membership(
        candidate, component_ids, categories
    )
    same_count_membership_change = assess_partition_identity_v4(
        replicate_matrices=matrices,
        replicate_kinds=kinds,
        component_ids=component_ids,
        categories=categories,
        selected_allocation=selection["allocation"],
        candidate_cluster_ids=changed_candidate,
        seed=1707,
    )
    renamed = assess_partition_identity_v4(
        replicate_matrices=matrices,
        replicate_kinds=kinds,
        component_ids=component_ids,
        categories=categories,
        selected_allocation=selection["allocation"],
        candidate_cluster_ids=_rename_labels(candidate),
        seed=1707,
    )
    drift_matrices = [matrix.copy() for matrix in matrices]
    target_indices = [
        index for index, category in enumerate(categories) if category == target_category
    ]
    first_index, second_index = target_indices[0], target_indices[6]
    drift_matrices[-1][[first_index, second_index]] = drift_matrices[-1][
        [second_index, first_index]
    ]
    replicate_drift = assess_partition_identity_v4(
        replicate_matrices=drift_matrices,
        replicate_kinds=kinds,
        component_ids=component_ids,
        categories=categories,
        selected_allocation=selection["allocation"],
        candidate_cluster_ids=candidate,
        seed=1707,
    )
    controls = {
        "stable_exact_partition_admitted": stable["status"]
        == "pass_exact_partition_identity_v4",
        "same_count_membership_change_rejected": same_count_membership_change["status"]
        == "reject_partition_identity_v4",
        "changed_category_identified": (
            set(changed_ids).issubset(
                set(
                    same_count_membership_change["categories"][target_category][
                        "membership_mismatch_component_ids"
                    ]
                )
            )
            and same_count_membership_change["summary"]["rejected_categories"] == 1
        ),
        "cluster_label_rename_invariant": renamed["status"]
        == "pass_exact_partition_identity_v4",
        "single_replicate_membership_drift_rejected": replicate_drift["status"]
        == "reject_partition_identity_v4",
        "candidate_not_mutated_or_relabelled": (
            candidate != changed_candidate
            and stable["scientific_boundary"]["candidate_mutated"] is False
            and same_count_membership_change["scientific_boundary"]["candidate_relabelled"]
            is False
        ),
    }
    if not exact_two or not all(controls.values()):
        failed = [name for name, passed in controls.items() if not passed]
        if not exact_two:
            failed.append("v2_exact_two_cluster_allocation")
        raise ValueError("partition identity benchmark failed: " + ", ".join(failed))
    body = {
        "schema": "sch.vpd-activation-category-cluster-partition-identity-benchmark.v4",
        "status": "pass",
        "seed": seed,
        "design": {
            "categories": 8,
            "components_per_category": 12,
            "known_mechanisms_per_category": 2,
            "replicates": len(matrices),
            "count_only_candidate_membership_changed_without_count_change": True,
        },
        "results": {
            "v2_allocation": selection["allocation"],
            "stable": stable,
            "same_count_membership_change": same_count_membership_change,
            "label_renamed": renamed,
            "replicate_membership_drift": replicate_drift,
        },
        "negative_controls": controls,
        "source": {
            "runner_sha256": sha256_file(Path(__file__)),
            "partition_identity_v4_sha256": sha256_file(
                Path(__file__).with_name(
                    "activation_category_cluster_partition_identity_v4.py"
                )
            ),
            "selector_v2_sha256": sha256_file(
                Path(__file__).with_name(
                    "activation_category_cluster_count_selection_v2.py"
                )
            ),
        },
        "scientific_boundary": {
            "synthetic_feature_matrices_only": True,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "does_not_admit_an_empirical_bonsai_candidate": True,
            "execution_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("partition identity benchmark address mismatch")
    if observed != build_report(seed=int(observed["seed"])):
        raise ValueError("partition identity benchmark is not reproducible")
    return {
        "schema": "sch.vpd-activation-category-cluster-partition-identity-benchmark-verification.v4",
        "status": "pass",
        "benchmark_id": identifier,
        "negative_controls": len(observed["negative_controls"]),
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark exact label-invariant VPD partition identity V4."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--seed", type=int, default=120923)
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
                "schema": result["schema"],
                "status": result["status"],
                "benchmark_id": result["benchmark_id"],
                "negative_controls": len(result.get("negative_controls", {})),
                "execution_authorized": result.get("execution_authorized", False),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
