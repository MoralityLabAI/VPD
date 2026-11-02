from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.activation_category_cluster_count_selection_v2 import (
    DEFAULT_MINIMUM_CATEGORY_CLUSTER_BOUNDARY_GAP_RATIO,
    DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE_P10,
    select_category_cluster_counts_v2,
)
from experiments.authz_mechinterp.activation_category_cluster_partition_identity_v4 import (
    assess_partition_identity_v4,
)
from experiments.authz_mechinterp.admit_vpd_cluster_candidate_discreteness_v2 import (
    _selection_inputs,
)
from experiments.authz_mechinterp.admit_vpd_cluster_candidate_semantic_strata_v3 import (
    semantic_stratum_replicate_matrices,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    feature_matrix,
    load_jsonl,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-cluster-candidate-partition-identity-admission.v4"


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
        rows,
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
        cluster_validation_merge_manifest_path=cluster_validation_merge_manifest_path,
    )
    algorithm = manifest["algorithm"]
    receipt = algorithm["activation_category_compartments"][
        "cluster_count_selection"
    ]
    first_projection_seed = int(receipt["projection_seeds"][0])
    semantic_count = 0
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
            raise ValueError("partition identity feature profile contract differs")
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
    assignments = load_jsonl(
        bundle_dir / manifest["artifacts"]["assignments"]["path"]
    )
    assignment_by_id = {str(row["component_id"]): row for row in assignments}
    if set(assignment_by_id) != set(component_ids):
        raise ValueError("partition identity assignments differ from components")
    candidate_cluster_ids = {
        component_id: str(assignment_by_id[component_id]["cluster_id"])
        for component_id in component_ids
    }
    partition_identity = assess_partition_identity_v4(
        replicate_matrices=matrices,
        replicate_kinds=kinds,
        component_ids=component_ids,
        categories=categories,
        selected_allocation={
            str(key): int(value) for key, value in selection["allocation"].items()
        },
        candidate_cluster_ids=candidate_cluster_ids,
        seed=int(algorithm["seed"]),
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
    allocation_mismatches = [
        {
            "activation_category": category,
            "candidate_cluster_count": candidate_allocation.get(category),
            "selected_cluster_count": selected_allocation.get(category),
        }
        for category in sorted(set(candidate_allocation) | set(selected_allocation))
        if candidate_allocation.get(category) != selected_allocation.get(category)
    ]
    partition_rejections = [
        {
            "activation_category": category,
            "candidate_count_matches": evidence["candidate_count_matches"],
            "candidate_membership_matches": evidence[
                "candidate_membership_matches"
            ],
            "exact_replicate_partition_agreement": evidence[
                "exact_replicate_partition_agreement"
            ],
            "membership_mismatch_component_ids": evidence[
                "membership_mismatch_component_ids"
            ],
        }
        for category, evidence in sorted(partition_identity["categories"].items())
        if evidence["status"] != "pass_exact_partition_identity"
    ]
    eligible = not allocation_mismatches and not partition_rejections
    body = {
        "schema": SCHEMA,
        "status": (
            "pass_candidate_partition_identity_v4"
            if eligible
            else "reject_candidate_partition_identity_v4"
        ),
        "source": {
            "bundle_manifest_sha256": sha256_file(bundle_dir / "manifest.json"),
            "component_inventory_sha256": sha256_file(component_inventory_path),
            "discovery_effects_sha256": sha256_file(discovery_effects_path),
            "discovery_merge_manifest_sha256": sha256_file(
                discovery_merge_manifest_path
            ),
            "cluster_validation_effects_sha256": sha256_file(
                cluster_validation_effects_path
            ),
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
            "discovery_and_independent_validation_required": True,
            "pair_condition_and_template_omissions_required": True,
        },
        "candidate_allocation": candidate_allocation,
        "semantic_strata_selection": selection,
        "partition_identity": partition_identity,
        "admission": {
            "eligible": eligible,
            "allocation_mismatch_count": len(allocation_mismatches),
            "allocation_mismatches": allocation_mismatches,
            "partition_rejection_count": len(partition_rejections),
            "partition_rejections": partition_rejections,
            "candidate_mutated": False,
            "candidate_relabelled": False,
            "cluster_count_agreement_alone_is_insufficient": True,
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
    return {**body, "partition_identity_admission_id": content_sha256(body)}


def verify_admission_report(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("partition_identity_admission_id", ""))
    if identifier != content_sha256(body) or observed != build_admission_report(**kwargs):
        raise ValueError("partition identity admission report is not reproducible")
    return {
        "schema": "sch.vpd-cluster-candidate-partition-identity-admission-verification.v4",
        "status": "pass",
        "partition_identity_admission_id": identifier,
        "admission_decision": observed["status"],
        "eligible": observed["admission"]["eligible"],
        "allocation_mismatch_count": observed["admission"][
            "allocation_mismatch_count"
        ],
        "partition_rejection_count": observed["admission"][
            "partition_rejection_count"
        ],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Admit a VPD cluster candidate under exact partition identity V4."
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
