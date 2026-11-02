from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.activation_category_cluster_count_selection_v2 import (
    DEFAULT_MINIMUM_CATEGORY_CLUSTER_BOUNDARY_GAP_RATIO,
    DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE_P10,
    select_category_cluster_counts_v2,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    category_profile_feature_matrix,
    content_sha256,
    feature_matrix,
    load_effect_merge_split_provenance,
    load_hierarchy_plan,
    load_jsonl,
    sha256_file,
    verify_bundle,
    write_json,
)
from experiments.authz_mechinterp.effect_uncertainty import (
    leave_one_block_out_means,
)


SCHEMA = "sch.vpd-cluster-candidate-discreteness-admission.v2"


def _selection_inputs(
    *,
    bundle_dir: Path,
    component_inventory_path: Path,
    discovery_effects_path: Path,
    discovery_merge_manifest_path: Path,
    cluster_validation_effects_path: Path,
    cluster_validation_merge_manifest_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[Any],
    list[str],
    list[str],
    list[Any],
    list[str],
]:
    base_verification = verify_bundle(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source", {})
    if sha256_file(component_inventory_path) != source.get("component_inventory_sha256"):
        raise ValueError("cluster candidate component inventory hash differs")
    if sha256_file(discovery_effects_path) != source.get("effects_sha256"):
        raise ValueError("cluster candidate discovery effects hash differs")
    if sha256_file(cluster_validation_effects_path) != source.get(
        "cluster_validation_effects_sha256"
    ):
        raise ValueError("cluster candidate validation effects hash differs")
    discovery_provenance = load_effect_merge_split_provenance(
        discovery_effects_path,
        discovery_merge_manifest_path,
        expected_role="discovery",
    )
    validation_provenance = load_effect_merge_split_provenance(
        cluster_validation_effects_path,
        cluster_validation_merge_manifest_path,
        expected_role="cluster_validation",
    )
    if discovery_provenance["effect_merge_manifest_sha256"] != source.get(
        "effects_merge_manifest_sha256"
    ):
        raise ValueError("cluster candidate discovery merge provenance differs")
    if validation_provenance["effect_merge_manifest_sha256"] != source.get(
        "cluster_validation_effects_merge_manifest_sha256"
    ):
        raise ValueError("cluster candidate validation merge provenance differs")
    if discovery_provenance["activation_category_split_id"] != validation_provenance[
        "activation_category_split_id"
    ]:
        raise ValueError("cluster candidate discovery and validation split IDs differ")

    compartment = manifest.get("algorithm", {}).get(
        "activation_category_compartments", {}
    )
    receipt = compartment.get("cluster_count_selection")
    if not isinstance(receipt, dict):
        raise ValueError("cluster candidate count-selection receipt is absent")
    receipt_body = dict(receipt)
    identifier = str(receipt_body.pop("selection_id", ""))
    if content_sha256(receipt_body) != identifier:
        raise ValueError("cluster candidate count-selection receipt address differs")
    allowed_modes = {
        "cross_archive_projection_and_pair_block_ensemble_stability_and_silhouette",
        "cross_archive_projection_and_pair_block_ensemble_stability_and_silhouette_lower_tail_and_boundary_gap_v2",
    }
    if receipt.get("status") != "pass" or receipt.get("mode") not in allowed_modes:
        raise ValueError("cluster candidate lacks a complete stability ensemble")
    selection_schema = str(receipt.get("selection", {}).get("schema"))
    if selection_schema not in {
        "sch.vpd-category-cluster-count-selection.v1",
        "sch.vpd-category-cluster-count-selection.v2",
    }:
        raise ValueError("cluster candidate selection schema is unsupported")

    source_kind = str(source.get("source_kind"))
    if source_kind == "hierarchy_plan_exact_partition":
        rows = load_hierarchy_plan(component_inventory_path)
    elif source_kind == "component_inventory_jsonl":
        rows = load_jsonl(component_inventory_path)
    else:
        raise ValueError("cluster candidate component inventory kind is unsupported")
    component_ids = [str(row["component_id"]) for row in rows]
    assignments = load_jsonl(bundle_dir / manifest["artifacts"]["assignments"]["path"])
    assignment_by_id = {str(row["component_id"]): row for row in assignments}
    if set(assignment_by_id) != set(component_ids):
        raise ValueError("cluster candidate assignments and hierarchy differ")
    categories = [
        str(assignment_by_id[component_id]["activation_category"])
        for component_id in component_ids
    ]

    algorithm = manifest["algorithm"]
    seed = int(algorithm["seed"])
    projection_dim = int(algorithm["effect_projection_dim"])
    metadata_weight = float(algorithm["metadata_weight"])
    projection_seeds = [int(value) for value in receipt["projection_seeds"]]
    matrices: list[Any] = []
    kinds: list[str] = []
    discovery_profile: dict[str, Any] | None = None
    validation_profile: dict[str, Any] | None = None
    for projection_seed in projection_seeds:
        matrix, _, mode, profile = feature_matrix(
            rows,
            effects_path=discovery_effects_path,
            seed=projection_seed,
            projection_dim=projection_dim,
            metadata_weight=metadata_weight,
        )
        if mode != algorithm["feature_mode"] or profile is None:
            raise ValueError("cluster candidate discovery feature contract differs")
        matrices.append(matrix)
        kinds.append("projection")
        if projection_seed == projection_seeds[0]:
            discovery_profile = profile
    for projection_seed in projection_seeds:
        matrix, _, mode, profile = feature_matrix(
            rows,
            effects_path=cluster_validation_effects_path,
            seed=projection_seed,
            projection_dim=projection_dim,
            metadata_weight=metadata_weight,
        )
        if mode != algorithm["feature_mode"] or profile is None:
            raise ValueError("cluster candidate validation feature contract differs")
        matrices.append(matrix)
        kinds.append("independent_cluster_validation_projection")
        if projection_seed == projection_seeds[0]:
            validation_profile = profile
    if discovery_profile is None or validation_profile is None:
        raise ValueError("cluster candidate profiles are absent")
    if discovery_profile["category_names"] != validation_profile["category_names"]:
        raise ValueError("cluster candidate discovery and validation categories differ")

    discovery_omissions = [int(value) for value in receipt["pair_block_omissions"]]
    validation_omissions = [
        int(value)
        for value in receipt["cluster_validation_pair_block_omissions"]
    ]
    discovery_means, discovery_counts = leave_one_block_out_means(
        discovery_profile["uncertainty_block_sums"],
        discovery_profile["uncertainty_block_counts"],
    )
    validation_means, validation_counts = leave_one_block_out_means(
        validation_profile["uncertainty_block_sums"],
        validation_profile["uncertainty_block_counts"],
    )
    if discovery_omissions != list(range(discovery_means.shape[1])):
        raise ValueError("cluster candidate discovery pair-block omissions differ")
    if validation_omissions != list(range(validation_means.shape[1])):
        raise ValueError("cluster candidate validation pair-block omissions differ")
    for block_index in discovery_omissions:
        matrices.append(
            category_profile_feature_matrix(
                discovery_means[:, block_index, :, :],
                discovery_counts[block_index],
                rows,
                seed=seed,
                projection_dim=projection_dim,
                metadata_weight=metadata_weight,
            )
        )
        kinds.append("leave_one_pair_block_out")
    for block_index in validation_omissions:
        matrices.append(
            category_profile_feature_matrix(
                validation_means[:, block_index, :, :],
                validation_counts[block_index],
                rows,
                seed=seed,
                projection_dim=projection_dim,
                metadata_weight=metadata_weight,
            )
        )
        kinds.append("independent_cluster_validation_leave_one_pair_block_out")
    return (
        manifest,
        base_verification,
        rows,
        component_ids,
        categories,
        matrices,
        kinds,
    )


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
        manifest,
        base_verification,
        _rows,
        component_ids,
        categories,
        matrices,
        kinds,
    ) = _selection_inputs(
        bundle_dir=bundle_dir,
        component_inventory_path=component_inventory_path,
        discovery_effects_path=discovery_effects_path,
        discovery_merge_manifest_path=discovery_merge_manifest_path,
        cluster_validation_effects_path=cluster_validation_effects_path,
        cluster_validation_merge_manifest_path=(
            cluster_validation_merge_manifest_path
        ),
    )
    compartment = manifest["algorithm"]["activation_category_compartments"]
    legacy_selection = compartment["cluster_count_selection"]["selection"]
    policy = legacy_selection["policy"]
    native_v2 = (
        legacy_selection.get("schema")
        == "sch.vpd-category-cluster-count-selection.v2"
    )
    selection_v2 = select_category_cluster_counts_v2(
        matrices,
        component_ids,
        categories,
        seed=int(manifest["algorithm"]["seed"]),
        replicate_kinds=kinds,
        maximum_clusters_per_category=int(policy["maximum_clusters_per_category"]),
        minimum_replicate_stability=float(
            policy["minimum_replicate_adjusted_rand"]
        ),
        minimum_mean_silhouette=float(
            policy[
                "minimum_replicate_mean_silhouette"
                if native_v2
                else "minimum_replicate_silhouette"
            ]
        ),
        minimum_silhouette_p10=float(
            policy.get(
                "minimum_replicate_silhouette_p10",
                DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE_P10,
            )
        ),
        minimum_boundary_gap_ratio=float(
            policy.get(
                "minimum_replicate_boundary_gap_ratio",
                DEFAULT_MINIMUM_CATEGORY_CLUSTER_BOUNDARY_GAP_RATIO,
            )
        ),
        minimum_cluster_size=int(policy["minimum_cluster_size"]),
        max_iterations=100,
    )
    candidate_allocation = {
        str(key): int(value)
        for key, value in sorted(compartment["cluster_allocation"].items())
    }
    admitted_allocation = {
        str(key): int(value)
        for key, value in sorted(selection_v2["allocation"].items())
    }
    mismatches = [
        {
            "activation_category": category,
            "candidate_cluster_count": candidate_allocation[category],
            "v2_cluster_count": admitted_allocation[category],
        }
        for category in sorted(candidate_allocation)
        if candidate_allocation[category] != admitted_allocation[category]
    ]
    admitted = not mismatches
    manifest_path = bundle_dir / "manifest.json"
    body = {
        "schema": SCHEMA,
        "status": (
            "pass_candidate_cluster_discreteness_v2"
            if admitted
            else "reject_candidate_cluster_discreteness_v2"
        ),
        "source": {
            "bundle_manifest_path": manifest_path.as_posix(),
            "bundle_manifest_sha256": sha256_file(manifest_path),
            "component_inventory_path": component_inventory_path.as_posix(),
            "component_inventory_sha256": sha256_file(component_inventory_path),
            "discovery_effects_path": discovery_effects_path.as_posix(),
            "discovery_effects_sha256": sha256_file(discovery_effects_path),
            "discovery_merge_manifest_path": (
                discovery_merge_manifest_path.as_posix()
            ),
            "discovery_merge_manifest_sha256": sha256_file(
                discovery_merge_manifest_path
            ),
            "cluster_validation_effects_path": (
                cluster_validation_effects_path.as_posix()
            ),
            "cluster_validation_effects_sha256": sha256_file(
                cluster_validation_effects_path
            ),
            "cluster_validation_merge_manifest_path": (
                cluster_validation_merge_manifest_path.as_posix()
            ),
            "cluster_validation_merge_manifest_sha256": sha256_file(
                cluster_validation_merge_manifest_path
            ),
            "admission_runner_sha256": sha256_file(Path(__file__)),
        },
        "base_bundle_verification": base_verification,
        "candidate": {
            "component_id_set_sha256": manifest["invariants"][
                "component_id_set_sha256"
            ],
            "legacy_selection_id": compartment["cluster_count_selection"][
                "selection_id"
            ],
            "legacy_selection_schema": legacy_selection["schema"],
            "candidate_allocation": candidate_allocation,
        },
        "selection_v2": selection_v2,
        "admission": {
            "eligible": admitted,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "candidate_mutated": False,
            "candidate_relabelled": False,
            "v1_bundle_preserved_as_predecessor_format": not native_v2,
            "native_v2_bundle_reverified": native_v2,
        },
        "scientific_boundary": {
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "model_outcomes_used": False,
            "scientific_release_authorized": False,
            "compute_authorized": False,
            "execution_authorized": False,
        },
    }
    return {**body, "cluster_discreteness_admission_id": content_sha256(body)}


def verify_admission_report(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("cluster_discreteness_admission_id", ""))
    if content_sha256(body) != identifier:
        raise ValueError("cluster discreteness admission address mismatch")
    expected = build_admission_report(**kwargs)
    if observed != expected:
        raise ValueError("cluster discreteness admission is not reproducible")
    return {
        "schema": "sch.vpd-cluster-candidate-discreteness-admission-verification.v2",
        "status": "pass",
        "cluster_discreteness_admission_id": identifier,
        "admission_decision": observed["status"],
        "eligible": bool(observed["admission"]["eligible"]),
        "mismatch_count": int(observed["admission"]["mismatch_count"]),
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently admit a V1 or native-V2 category bundle under V2 discreteness."
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    inventory = parser.add_mutually_exclusive_group(required=True)
    inventory.add_argument("--components", type=Path)
    inventory.add_argument("--hierarchy-plan", type=Path)
    parser.add_argument("--discovery-effects", type=Path, required=True)
    parser.add_argument("--discovery-merge-manifest", type=Path, required=True)
    parser.add_argument("--cluster-validation-effects", type=Path, required=True)
    parser.add_argument(
        "--cluster-validation-merge-manifest", type=Path, required=True
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = {
        "bundle_dir": args.bundle_dir,
        "component_inventory_path": args.components or args.hierarchy_plan,
        "discovery_effects_path": args.discovery_effects,
        "discovery_merge_manifest_path": args.discovery_merge_manifest,
        "cluster_validation_effects_path": args.cluster_validation_effects,
        "cluster_validation_merge_manifest_path": (
            args.cluster_validation_merge_manifest
        ),
    }
    if args.verify is not None:
        result = verify_admission_report(args.verify, **inputs)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_admission_report(**inputs)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("eligible") is False or result.get("admission", {}).get(
        "eligible"
    ) is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
