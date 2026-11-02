from __future__ import annotations

from typing import Any

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    deterministic_kmeans,
    stable_hash,
)


SCHEMA = "sch.vpd-activation-category-cluster-partition-identity.v4"


def canonical_partition(
    component_ids: list[str], labels: np.ndarray
) -> list[list[str]]:
    assignments = np.asarray(labels, dtype=np.int64)
    if assignments.shape != (len(component_ids),):
        raise ValueError("cluster partition labels do not align with components")
    if len(component_ids) != len(set(component_ids)) or any(not value for value in component_ids):
        raise ValueError("cluster partition component identities are invalid")
    groups = [
        sorted(
            component_ids[index]
            for index in range(len(component_ids))
            if assignments[index] == label
        )
        for label in sorted(set(assignments.tolist()))
    ]
    if any(not group for group in groups):
        raise ValueError("cluster partition contains an empty group")
    return sorted(groups, key=lambda group: (group[0], len(group), group))


def candidate_partition(
    component_ids: list[str], candidate_cluster_ids: dict[str, str]
) -> list[list[str]]:
    if set(candidate_cluster_ids) != set(component_ids):
        raise ValueError("candidate cluster membership does not cover exact components")
    cluster_names = [str(candidate_cluster_ids[component_id]) for component_id in component_ids]
    if any(not value for value in cluster_names):
        raise ValueError("candidate cluster identity is empty")
    label_by_name = {
        name: index for index, name in enumerate(sorted(set(cluster_names)))
    }
    labels = np.asarray([label_by_name[name] for name in cluster_names], dtype=np.int64)
    return canonical_partition(component_ids, labels)


def _membership_mismatch_components(
    component_ids: list[str],
    left: list[list[str]],
    right: list[list[str]],
) -> list[str]:
    left_group = {
        component_id: group_index
        for group_index, group in enumerate(left)
        for component_id in group
    }
    right_group = {
        component_id: group_index
        for group_index, group in enumerate(right)
        for component_id in group
    }
    mismatches: set[str] = set()
    for left_index, first in enumerate(component_ids):
        for second in component_ids[left_index + 1 :]:
            if (left_group[first] == left_group[second]) != (
                right_group[first] == right_group[second]
            ):
                mismatches.update((first, second))
    return sorted(mismatches)


def assess_partition_identity_v4(
    *,
    replicate_matrices: list[np.ndarray],
    replicate_kinds: list[str],
    component_ids: list[str],
    categories: list[str],
    selected_allocation: dict[str, int],
    candidate_cluster_ids: dict[str, str],
    seed: int,
    max_iterations: int = 100,
) -> dict[str, Any]:
    if len(replicate_matrices) < 2 or len(replicate_matrices) != len(replicate_kinds):
        raise ValueError("partition identity requires aligned multiple replicates")
    matrices = [np.asarray(matrix, dtype=np.float64) for matrix in replicate_matrices]
    if any(matrix.ndim != 2 or matrix.shape[0] != len(component_ids) for matrix in matrices):
        raise ValueError("partition identity replicate matrices do not align")
    if any(not np.all(np.isfinite(matrix)) for matrix in matrices):
        raise ValueError("partition identity matrix contains nonfinite values")
    if len(categories) != len(component_ids) or any(not value for value in categories):
        raise ValueError("partition identity component categories do not align")
    category_names = sorted(set(categories))
    if set(selected_allocation) != set(category_names):
        raise ValueError("partition identity allocation categories differ")
    if set(candidate_cluster_ids) != set(component_ids):
        raise ValueError("partition identity candidate membership differs")
    category_results: dict[str, Any] = {}
    all_exact = True
    for category in category_names:
        indices = [index for index, value in enumerate(categories) if value == category]
        local_ids = [component_ids[index] for index in indices]
        cluster_count = int(selected_allocation[category])
        if cluster_count < 1 or cluster_count > len(local_ids):
            raise ValueError("partition identity selected cluster count is invalid")
        local_seed = int(
            stable_hash(
                seed,
                "activation-category-count-selection-v2",
                category,
                str(cluster_count),
            )[:8],
            16,
        )
        partitions: list[list[list[str]]] = []
        for matrix in matrices:
            if cluster_count == 1:
                labels = np.zeros(len(local_ids), dtype=np.int64)
            else:
                labels, _, _ = deterministic_kmeans(
                    matrix[indices],
                    local_ids,
                    cluster_count=cluster_count,
                    seed=local_seed,
                    max_iterations=max_iterations,
                )
            partitions.append(canonical_partition(local_ids, labels))
        consensus = partitions[0]
        exact_replicate_agreement = all(
            partition == consensus for partition in partitions[1:]
        )
        candidate = candidate_partition(
            local_ids,
            {component_id: candidate_cluster_ids[component_id] for component_id in local_ids},
        )
        candidate_count_matches = len(candidate) == cluster_count
        candidate_membership_matches = candidate == consensus
        mismatch_components = _membership_mismatch_components(
            local_ids, candidate, consensus
        )
        exact = (
            exact_replicate_agreement
            and candidate_count_matches
            and candidate_membership_matches
        )
        all_exact &= exact
        category_results[category] = {
            "status": "pass_exact_partition_identity" if exact else "reject_partition_identity",
            "component_count": len(local_ids),
            "selected_cluster_count": cluster_count,
            "candidate_cluster_count": len(candidate),
            "replicates": len(partitions),
            "replicate_kind_counts": {
                kind: replicate_kinds.count(kind) for kind in sorted(set(replicate_kinds))
            },
            "replicate_partition_sha256": [content_sha256(value) for value in partitions],
            "unique_replicate_partition_count": len(
                {content_sha256(value) for value in partitions}
            ),
            "exact_replicate_partition_agreement": exact_replicate_agreement,
            "consensus_partition_sha256": content_sha256(consensus),
            "candidate_partition_sha256": content_sha256(candidate),
            "candidate_count_matches": candidate_count_matches,
            "candidate_membership_matches": candidate_membership_matches,
            "membership_mismatch_component_ids": mismatch_components,
            "cluster_label_names_ignored": True,
        }
    body = {
        "schema": SCHEMA,
        "status": "pass_exact_partition_identity_v4" if all_exact else "reject_partition_identity_v4",
        "policy": {
            "comparison": "pairwise_coassignment_label_invariant",
            "exact_replicate_partition_agreement_required": True,
            "exact_candidate_consensus_membership_required": True,
            "cluster_count_agreement_alone_is_insufficient": True,
            "unstable_or_mismatched_candidate_mutated": False,
            "unstable_or_mismatched_candidate_relabelled": False,
        },
        "summary": {
            "categories": len(category_names),
            "exact_categories": sum(
                value["status"] == "pass_exact_partition_identity"
                for value in category_results.values()
            ),
            "rejected_categories": sum(
                value["status"] != "pass_exact_partition_identity"
                for value in category_results.values()
            ),
        },
        "categories": category_results,
        "scientific_boundary": {
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "candidate_mutated": False,
            "candidate_relabelled": False,
            "execution_authorized": False,
        },
    }
    return {**body, "partition_identity_id": content_sha256(body)}
