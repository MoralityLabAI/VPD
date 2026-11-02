from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.activation_category_cluster_count_selection_v2 import (
    DEFAULT_MINIMUM_CATEGORY_CLUSTER_BOUNDARY_GAP_RATIO,
    DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE_P10,
    select_category_cluster_counts_v2,
)
from experiments.authz_mechinterp.admit_vpd_cluster_candidate_discreteness_v2 import (
    _selection_inputs,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    category_profile_feature_matrix,
    content_sha256,
    feature_matrix,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-cluster-candidate-semantic-strata-admission.v3"


def _category_totals(
    *,
    sums: np.ndarray,
    counts: np.ndarray,
    cell_categories: list[str],
    category_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(sums, dtype=np.float64)
    supports = np.asarray(counts, dtype=np.int64)
    if values.ndim != 3 or supports.shape != (values.shape[1],):
        raise ValueError("semantic-stratum sufficient statistics do not align")
    total_sums = np.zeros(
        (values.shape[0], len(category_names), values.shape[2]), dtype=np.float64
    )
    total_counts = np.zeros(len(category_names), dtype=np.int64)
    for category_index, category in enumerate(category_names):
        indices = [
            index for index, value in enumerate(cell_categories) if value == category
        ]
        if len(indices) < 2:
            raise ValueError("semantic-stratum category requires at least two cells")
        total_sums[:, category_index] = values[:, indices].sum(axis=1)
        total_counts[category_index] = int(supports[indices].sum())
    return total_sums, total_counts


def semantic_stratum_replicate_matrices(
    profile: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    seed: int,
    projection_dim: int,
    metadata_weight: float,
    role: str,
) -> tuple[list[np.ndarray], list[str]]:
    required_condition = {
        "condition_sums", "condition_counts", "condition_categories"
    }
    required_template = {
        "template_cell_sums", "template_cell_counts", "template_cell_categories",
        "template_cell_groups",
    }
    if not required_condition <= set(profile):
        raise ValueError("semantic-stratum admission requires condition statistics")
    if not required_template <= set(profile):
        raise ValueError("semantic-stratum admission requires template statistics")
    category_names = [str(value) for value in profile["category_names"]]
    matrices: list[np.ndarray] = []
    kinds: list[str] = []

    condition_sums = np.asarray(profile["condition_sums"], dtype=np.float64)
    condition_counts = np.asarray(profile["condition_counts"], dtype=np.int64)
    condition_categories = [str(value) for value in profile["condition_categories"]]
    total_sums, total_counts = _category_totals(
        sums=condition_sums,
        counts=condition_counts,
        cell_categories=condition_categories,
        category_names=category_names,
    )
    category_index = {category: index for index, category in enumerate(category_names)}
    for omitted in range(len(condition_counts)):
        local_sums = total_sums.copy()
        local_counts = total_counts.copy()
        target = category_index[condition_categories[omitted]]
        local_sums[:, target] -= condition_sums[:, omitted]
        local_counts[target] -= condition_counts[omitted]
        if np.any(local_counts <= 0):
            raise ValueError("condition omission removes all category support")
        matrices.append(
            category_profile_feature_matrix(
                local_sums / local_counts[None, :, None],
                local_counts,
                rows,
                seed=seed,
                projection_dim=projection_dim,
                metadata_weight=metadata_weight,
            )
        )
        kinds.append(f"{role}_leave_one_condition_out")

    template_sums = np.asarray(profile["template_cell_sums"], dtype=np.float64)
    template_counts = np.asarray(profile["template_cell_counts"], dtype=np.int64)
    template_categories = [str(value) for value in profile["template_cell_categories"]]
    template_groups = [str(value) for value in profile["template_cell_groups"]]
    total_sums, total_counts = _category_totals(
        sums=template_sums,
        counts=template_counts,
        cell_categories=template_categories,
        category_names=category_names,
    )
    unique_groups = sorted(set(template_groups))
    if len(unique_groups) < 2:
        raise ValueError("semantic-stratum admission requires multiple template groups")
    for omitted_group in unique_groups:
        local_sums = total_sums.copy()
        local_counts = total_counts.copy()
        omitted_indices = [
            index for index, value in enumerate(template_groups)
            if value == omitted_group
        ]
        omitted_categories = {
            template_categories[index] for index in omitted_indices
        }
        if omitted_categories != set(category_names):
            raise ValueError("template group does not cover every category")
        for omitted in omitted_indices:
            target = category_index[template_categories[omitted]]
            local_sums[:, target] -= template_sums[:, omitted]
            local_counts[target] -= template_counts[omitted]
        if np.any(local_counts <= 0):
            raise ValueError("template omission removes all category support")
        matrices.append(
            category_profile_feature_matrix(
                local_sums / local_counts[None, :, None],
                local_counts,
                rows,
                seed=seed,
                projection_dim=projection_dim,
                metadata_weight=metadata_weight,
            )
        )
        kinds.append(f"{role}_leave_one_template_group_out")
    return matrices, kinds


def build_admission_report(
    *,
    bundle_dir: Path,
    component_inventory_path: Path,
    discovery_effects_path: Path,
    discovery_merge_manifest_path: Path,
    cluster_validation_effects_path: Path,
    cluster_validation_merge_manifest_path: Path,
) -> dict[str, Any]:
    (
        manifest, base_verification, rows, component_ids, categories,
        matrices, kinds,
    ) = _selection_inputs(
        bundle_dir=bundle_dir,
        component_inventory_path=component_inventory_path,
        discovery_effects_path=discovery_effects_path,
        discovery_merge_manifest_path=discovery_merge_manifest_path,
        cluster_validation_effects_path=cluster_validation_effects_path,
        cluster_validation_merge_manifest_path=cluster_validation_merge_manifest_path,
    )
    algorithm = manifest["algorithm"]
    receipt = algorithm["activation_category_compartments"]["cluster_count_selection"]
    first_projection_seed = int(receipt["projection_seeds"][0])
    profiles: list[tuple[str, dict[str, Any]]] = []
    for role, effects_path in (
        ("discovery", discovery_effects_path),
        ("independent_cluster_validation", cluster_validation_effects_path),
    ):
        _, _, mode, profile = feature_matrix(
            rows,
            effects_path=effects_path,
            seed=first_projection_seed,
            projection_dim=int(algorithm["effect_projection_dim"]),
            metadata_weight=float(algorithm["metadata_weight"]),
        )
        if mode != algorithm["feature_mode"] or profile is None:
            raise ValueError("semantic-stratum feature profile contract differs")
        profiles.append((role, profile))
    semantic_count = 0
    for role, profile in profiles:
        additions, addition_kinds = semantic_stratum_replicate_matrices(
            profile,
            rows,
            seed=int(algorithm["seed"]),
            projection_dim=int(algorithm["effect_projection_dim"]),
            metadata_weight=float(algorithm["metadata_weight"]),
            role=role,
        )
        matrices.extend(additions)
        kinds.extend(addition_kinds)
        semantic_count += len(additions)
    policy = receipt["selection"]["policy"]
    selection = select_category_cluster_counts_v2(
        matrices,
        component_ids,
        categories,
        seed=int(algorithm["seed"]),
        replicate_kinds=kinds,
        maximum_clusters_per_category=int(policy["maximum_clusters_per_category"]),
        minimum_replicate_stability=float(policy["minimum_replicate_adjusted_rand"]),
        minimum_mean_silhouette=float(policy["minimum_replicate_silhouette"]),
        minimum_silhouette_p10=DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE_P10,
        minimum_boundary_gap_ratio=DEFAULT_MINIMUM_CATEGORY_CLUSTER_BOUNDARY_GAP_RATIO,
        minimum_cluster_size=int(policy["minimum_cluster_size"]),
    )
    candidate_allocation = {
        str(key): int(value)
        for key, value in algorithm["activation_category_compartments"][
            "cluster_allocation"
        ].items()
    }
    selected_allocation = {
        str(key): int(value) for key, value in selection["allocation"].items()
    }
    mismatches = [
        {
            "activation_category": category,
            "candidate_cluster_count": candidate_allocation.get(category),
            "semantic_strata_cluster_count": selected_allocation.get(category),
        }
        for category in sorted(set(candidate_allocation) | set(selected_allocation))
        if candidate_allocation.get(category) != selected_allocation.get(category)
    ]
    eligible = not mismatches
    body = {
        "schema": SCHEMA,
        "status": (
            "pass_candidate_semantic_strata_v3"
            if eligible else "reject_candidate_semantic_strata_v3"
        ),
        "source": {
            "bundle_manifest_sha256": sha256_file(bundle_dir / "manifest.json"),
            "component_inventory_sha256": sha256_file(component_inventory_path),
            "discovery_effects_sha256": sha256_file(discovery_effects_path),
            "discovery_merge_manifest_sha256": sha256_file(discovery_merge_manifest_path),
            "cluster_validation_effects_sha256": sha256_file(cluster_validation_effects_path),
            "cluster_validation_merge_manifest_sha256": sha256_file(
                cluster_validation_merge_manifest_path
            ),
            "runner_sha256": sha256_file(Path(__file__)),
        },
        "base_bundle_verification": base_verification,
        "replicate_evidence": {
            "total_matrices": len(matrices),
            "semantic_stratum_matrices": semantic_count,
            "kind_counts": dict(sorted(Counter(kinds).items())),
            "condition_and_template_statistics_required_in_both_archives": True,
        },
        "candidate_allocation": candidate_allocation,
        "semantic_strata_selection": selection,
        "admission": {
            "eligible": eligible,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "candidate_mutated": False,
            "candidate_relabelled": False,
        },
        "scientific_boundary": {
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "model_outcomes_used": False,
            "execution_authorized": False,
        },
    }
    return {**body, "semantic_strata_admission_id": content_sha256(body)}


def verify_admission_report(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = body.pop("semantic_strata_admission_id", "")
    if identifier != content_sha256(body) or observed != build_admission_report(**kwargs):
        raise ValueError("semantic-strata admission is not reproducible")
    return {
        "schema": "sch.vpd-cluster-candidate-semantic-strata-admission-verification.v3",
        "status": "pass",
        "semantic_strata_admission_id": identifier,
        "admission_decision": observed["status"],
        "eligible": observed["admission"]["eligible"],
        "mismatch_count": observed["admission"]["mismatch_count"],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Admit cluster candidates under semantic strata V3.")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    inventory = parser.add_mutually_exclusive_group(required=True)
    inventory.add_argument("--components", type=Path)
    inventory.add_argument("--hierarchy-plan", type=Path)
    parser.add_argument("--discovery-effects", type=Path, required=True)
    parser.add_argument("--discovery-merge-manifest", type=Path, required=True)
    parser.add_argument("--cluster-validation-effects", type=Path, required=True)
    parser.add_argument("--cluster-validation-merge-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = {
        "bundle_dir": args.bundle_dir,
        "component_inventory_path": args.components or args.hierarchy_plan,
        "discovery_effects_path": args.discovery_effects,
        "discovery_merge_manifest_path": args.discovery_merge_manifest,
        "cluster_validation_effects_path": args.cluster_validation_effects,
        "cluster_validation_merge_manifest_path": args.cluster_validation_merge_manifest,
    }
    if args.verify is not None:
        result = verify_admission_report(args.verify, **inputs)
    else:
        if args.output is None: parser.error("--output is required unless verifying")
        result = build_admission_report(**inputs); write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("eligible") is False or result.get("admission", {}).get("eligible") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
