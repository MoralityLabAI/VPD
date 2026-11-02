from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    stable_hash,
)
from experiments.authz_mechinterp.route_vpd_cluster_episodes_v1 import (
    SHARED_ROUTE,
    assess_routing_stability_v1,
    verify_routing_receipt_v1,
    verify_routing_stability_v1,
)


PLAN_SCHEMA = "sch.vpd-cluster-aware-sae-training-plan.v1"
JOB_SCHEMA = "sch.vpd-cluster-aware-sae-training-job.v1"
SHARED_COMPARTMENT = "shared_cross_category"


def _require_content_id(value: str, *, field: str) -> str:
    observed = str(value)
    if len(observed) != 64 or any(char not in "0123456789abcdef" for char in observed):
        raise ValueError(f"{field} must be a lowercase SHA-256 content ID")
    return observed


def _pair_atomic_split(
    rows: list[dict[str, Any]],
    *,
    compartment_id: str,
    validation_fraction: float,
    split_seed: int,
) -> dict[str, Any]:
    pair_ids = sorted({str(row["pair_id"]) for row in rows})
    if len(pair_ids) < 2:
        raise ValueError("cluster-aware SAE compartment requires at least two pairs")
    ordered = sorted(
        pair_ids,
        key=lambda pair_id: (stable_hash(split_seed, compartment_id, pair_id), pair_id),
    )
    validation_pairs_count = min(
        len(pair_ids) - 1,
        max(1, int(round(len(pair_ids) * validation_fraction))),
    )
    validation_pairs = set(ordered[:validation_pairs_count])
    training_pairs = set(pair_ids) - validation_pairs
    training_episode_ids = sorted(
        str(row["episode_id"])
        for row in rows
        if str(row["pair_id"]) in training_pairs
    )
    validation_episode_ids = sorted(
        str(row["episode_id"])
        for row in rows
        if str(row["pair_id"]) in validation_pairs
    )
    if set(training_episode_ids) & set(validation_episode_ids):
        raise AssertionError("pair-atomic SAE split overlaps episodes")
    return {
        "schema": "sch.vpd-cluster-aware-sae-pair-atomic-split.v1",
        "split_seed": split_seed,
        "validation_fraction": validation_fraction,
        "training_pairs": len(training_pairs),
        "validation_pairs": len(validation_pairs),
        "training_pair_id_set_sha256": content_sha256(sorted(training_pairs)),
        "validation_pair_id_set_sha256": content_sha256(sorted(validation_pairs)),
        "training_episodes": len(training_episode_ids),
        "validation_episodes": len(validation_episode_ids),
        "training_episode_id_set_sha256": content_sha256(training_episode_ids),
        "validation_episode_id_set_sha256": content_sha256(validation_episode_ids),
        "pair_sets_disjoint": True,
        "episode_sets_disjoint": True,
    }


def _validate_category_evidence(
    category: str, evidence: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    required = {"discovery", "cluster_validation", "stability", "sae_training"}
    if set(evidence) != required:
        raise ValueError("cluster-aware SAE category evidence fields differ")
    discovery = evidence["discovery"]
    validation = evidence["cluster_validation"]
    training = evidence["sae_training"]
    stability = evidence["stability"]
    for receipt in (discovery, validation, training):
        verify_routing_receipt_v1(receipt)
        if receipt["source"]["activation_category"] != category:
            raise ValueError("cluster-aware SAE routing category differs")
    stability_verification = verify_routing_stability_v1(stability)
    if not stability_verification["eligible_for_cluster_aware_sae_planning"]:
        raise ValueError("cluster-aware SAE routing stability was rejected")
    if stability["source"]["discovery_routing_id"] != discovery[
        "cluster_episode_routing_id"
    ] or stability["source"]["cluster_validation_routing_id"] != validation[
        "cluster_episode_routing_id"
    ]:
        raise ValueError("cluster-aware SAE stability receipt binds different routes")
    reproduced_stability = assess_routing_stability_v1(
        discovery,
        validation,
        maximum_total_variation=float(
            stability["policy"]["maximum_total_variation"]
        ),
        maximum_condition_total_variation=float(
            stability["policy"]["maximum_condition_total_variation"]
        ),
        maximum_shared_rate_difference=float(
            stability["policy"]["maximum_shared_rate_difference"]
        ),
        minimum_condition_support_per_role=int(
            stability["policy"]["minimum_condition_support_per_role"]
        ),
    )
    if reproduced_stability != stability:
        raise ValueError("cluster-aware SAE stability receipt is not reproducible")
    if training["source"]["split_role"] != "sae_training":
        raise ValueError("cluster-aware SAE samples must come from sae_training role")
    if discovery["source"]["split_role"] != "discovery" or validation["source"][
        "split_role"
    ] != "cluster_validation":
        raise ValueError("cluster-aware SAE admission roles differ")
    for candidate in (validation, training):
        if candidate["partition"] != discovery["partition"]:
            raise ValueError("cluster-aware SAE component partitions differ")
        if candidate["policy"] != discovery["policy"]:
            raise ValueError("cluster-aware SAE routing policies differ")
        if candidate["source"]["partition_identity_admission_id"] != discovery[
            "source"
        ]["partition_identity_admission_id"]:
            raise ValueError("cluster-aware SAE partition admission identities differ")
    if len(
        {
            discovery["source"]["split_id"],
            validation["source"]["split_id"],
            training["source"]["split_id"],
        }
    ) != 3:
        raise ValueError("cluster-aware SAE split identities are not distinct")
    discovery_ids = {row["episode_id"] for row in discovery["routing"]["rows"]}
    validation_ids = {row["episode_id"] for row in validation["routing"]["rows"]}
    training_ids = {row["episode_id"] for row in training["routing"]["rows"]}
    if discovery_ids & validation_ids or discovery_ids & training_ids or validation_ids & training_ids:
        raise ValueError("cluster-aware SAE discovery, validation, and training episodes overlap")
    return {
        "discovery": discovery,
        "cluster_validation": validation,
        "stability": stability,
        "sae_training": training,
    }


def build_cluster_aware_sae_plan_v1(
    *,
    category_evidence: dict[str, dict[str, Any]],
    predecessor_training_plan_id: str,
    predecessor_training_plan_sha256: str,
    profile: dict[str, Any],
    initialization_seeds: list[int],
    split_seed: int,
    validation_fraction: float,
    minimum_specific_episodes: int,
    minimum_specific_pairs: int,
    minimum_shared_episodes: int,
    minimum_shared_pairs: int,
    specific_dictionary_features: int,
    shared_dictionary_features: int,
    sae_activation_rows_per_episode: int,
) -> dict[str, Any]:
    if not category_evidence:
        raise ValueError("cluster-aware SAE planning requires category evidence")
    predecessor_id = _require_content_id(
        predecessor_training_plan_id, field="predecessor training plan ID"
    )
    predecessor_sha = _require_content_id(
        predecessor_training_plan_sha256, field="predecessor training plan SHA-256"
    )
    if not isinstance(profile, dict) or not profile:
        raise ValueError("cluster-aware SAE profile is empty")
    if (
        not isinstance(initialization_seeds, list)
        or len(initialization_seeds) < 2
        or len(initialization_seeds) != len(set(initialization_seeds))
        or any(not isinstance(value, int) or value < 0 for value in initialization_seeds)
    ):
        raise ValueError("cluster-aware SAE requires at least two unique nonnegative seeds")
    if not math.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        raise ValueError("cluster-aware SAE validation fraction must be in (0, 1)")
    for value, name in (
        (minimum_specific_episodes, "minimum specific episodes"),
        (minimum_specific_pairs, "minimum specific pairs"),
        (minimum_shared_episodes, "minimum shared episodes"),
        (minimum_shared_pairs, "minimum shared pairs"),
        (specific_dictionary_features, "specific dictionary features"),
        (shared_dictionary_features, "shared dictionary features"),
        (sae_activation_rows_per_episode, "SAE activation rows per episode"),
    ):
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"cluster-aware SAE {name} must be positive")
    if minimum_specific_pairs < 2 or minimum_shared_pairs < 2:
        raise ValueError("cluster-aware SAE pair support must permit pair-atomic splitting")

    validated = {
        category: _validate_category_evidence(category, evidence)
        for category, evidence in sorted(category_evidence.items())
    }
    raw_rows: list[dict[str, Any]] = []
    route_metadata: dict[str, dict[str, Any]] = {}
    for category, evidence in validated.items():
        training = evidence["sae_training"]
        for cluster_id, members in training["partition"]["cluster_members"].items():
            route_key = f"{category}::{cluster_id}"
            route_metadata[route_key] = {
                "activation_category": category,
                "cluster_id": cluster_id,
                "component_members": list(members),
                "component_membership_sha256": content_sha256(list(members)),
                "routing_receipt_id": training["cluster_episode_routing_id"],
                "routing_stability_id": evidence["stability"]["routing_stability_id"],
            }
        for row in training["routing"]["rows"]:
            primary_route = str(row["primary_route"])
            route_key = (
                SHARED_COMPARTMENT
                if primary_route == SHARED_ROUTE
                else f"{category}::{primary_route}"
            )
            if route_key != SHARED_COMPARTMENT and route_key not in route_metadata:
                raise ValueError("cluster-aware SAE route lacks component membership")
            raw_rows.append(
                {
                    **row,
                    "source_routing_receipt_id": training[
                        "cluster_episode_routing_id"
                    ],
                    "routing_stability_id": evidence["stability"][
                        "routing_stability_id"
                    ],
                    "planned_route_key": route_key,
                    "planning_reason": (
                        "router_shared_fallback"
                        if route_key == SHARED_COMPARTMENT
                        else "stable_specific_route"
                    ),
                }
            )
    episode_ids = [str(row["episode_id"]) for row in raw_rows]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("cluster-aware SAE training episode identities overlap categories")

    rows_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        rows_by_pair[str(row["pair_id"])].append(row)
    cross_route_pair_conflicts = 0
    for pair_rows in rows_by_pair.values():
        route_keys = {str(row["planned_route_key"]) for row in pair_rows}
        if len(route_keys) > 1:
            cross_route_pair_conflicts += 1
            for row in pair_rows:
                row["planned_route_key"] = SHARED_COMPARTMENT
                row["planning_reason"] = "cross_route_pair_conflict"

    support_by_specific_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        if row["planned_route_key"] != SHARED_COMPARTMENT:
            support_by_specific_route[str(row["planned_route_key"])].append(row)
    under_supported_routes: list[dict[str, Any]] = []
    for route_key, rows in sorted(support_by_specific_route.items()):
        pairs = {str(row["pair_id"]) for row in rows}
        if len(rows) < minimum_specific_episodes or len(pairs) < minimum_specific_pairs:
            under_supported_routes.append(
                {
                    "route_key": route_key,
                    "episodes": len(rows),
                    "pairs": len(pairs),
                    "reason": "below_frozen_specific_support_gate",
                }
            )
            for row in rows:
                row["planned_route_key"] = SHARED_COMPARTMENT
                row["planning_reason"] = "under_supported_route_merged_to_shared"

    compartment_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        compartment_rows[str(row["planned_route_key"])].append(row)
    shared_rows = compartment_rows.get(SHARED_COMPARTMENT, [])
    if (
        len(shared_rows) < minimum_shared_episodes
        or len({str(row["pair_id"]) for row in shared_rows}) < minimum_shared_pairs
    ):
        raise ValueError("cluster-aware SAE shared fallback lacks frozen support")

    specific_keys = sorted(
        key for key in compartment_rows if key != SHARED_COMPARTMENT
    )
    compartment_id_by_key = {
        key: f"vpd-sae-compartment-{content_sha256(key)}" for key in specific_keys
    }
    compartment_id_by_key[SHARED_COMPARTMENT] = SHARED_COMPARTMENT
    contracts: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    all_pair_sets: list[set[str]] = []
    for route_key in specific_keys + [SHARED_COMPARTMENT]:
        rows = sorted(compartment_rows[route_key], key=lambda row: str(row["episode_id"]))
        compartment_id = compartment_id_by_key[route_key]
        pair_ids = sorted({str(row["pair_id"]) for row in rows})
        pair_set = set(pair_ids)
        if any(pair_set & prior for prior in all_pair_sets):
            raise AssertionError("cluster-aware SAE compartment pair sets overlap")
        all_pair_sets.append(pair_set)
        effect_rows_per_episode_values = {
            int(
                validated[str(row["activation_category"])]["sae_training"]["policy"][
                    "activation_rows_per_episode"
                ]
            )
            for row in rows
        }
        if len(effect_rows_per_episode_values) != 1:
            raise ValueError("cluster-aware routing effect rows per episode differ")
        effect_rows_per_episode = next(iter(effect_rows_per_episode_values))
        split = _pair_atomic_split(
            rows,
            compartment_id=compartment_id,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
        )
        metadata = route_metadata.get(route_key)
        contract_body = {
            "schema": "sch.vpd-cluster-aware-sae-compartment.v1",
            "compartment_id": compartment_id,
            "kind": "shared_fallback" if metadata is None else "specific_cluster",
            "activation_categories": sorted(
                {str(row["activation_category"]) for row in rows}
            ),
            "cluster_id": None if metadata is None else metadata["cluster_id"],
            "component_members": [] if metadata is None else metadata["component_members"],
            "component_membership_sha256": (
                None if metadata is None else metadata["component_membership_sha256"]
            ),
            "routing_receipt_ids": sorted(
                {str(row["source_routing_receipt_id"]) for row in rows}
            ),
            "routing_stability_ids": sorted(
                {str(row["routing_stability_id"]) for row in rows}
            ),
            "support": {
                "episodes": len(rows),
                "pairs": len(pair_ids),
                "routing_effect_rows": len(rows) * effect_rows_per_episode,
                "routing_effect_rows_per_episode": effect_rows_per_episode,
                "sae_activation_rows": len(rows) * sae_activation_rows_per_episode,
                "sae_activation_rows_per_episode": sae_activation_rows_per_episode,
                "episode_id_set_sha256": content_sha256(
                    [str(row["episode_id"]) for row in rows]
                ),
                "pair_id_set_sha256": content_sha256(pair_ids),
                "planning_reason_counts": dict(
                    sorted(Counter(str(row["planning_reason"]) for row in rows).items())
                ),
            },
            "pair_atomic_split": split,
            "dictionary_features": (
                shared_dictionary_features
                if metadata is None
                else specific_dictionary_features
            ),
        }
        contract = {
            **contract_body,
            "compartment_contract_id": content_sha256(contract_body),
        }
        contracts.append(contract)
        for initialization_seed in sorted(initialization_seeds):
            resolved_profile = dict(profile)
            resolved_profile["dictionary_features"] = contract["dictionary_features"]
            job_body = {
                "schema": JOB_SCHEMA,
                "compartment_id": compartment_id,
                "compartment_contract_id": contract["compartment_contract_id"],
                "initialization_seed": initialization_seed,
                "data_selection_seed": split_seed,
                "profile": resolved_profile,
                "dictionary_features": contract["dictionary_features"],
                "required_sample_contract": {
                    "selection": "exact_cluster_routed_episode_set",
                    "episode_id_set_sha256": contract["support"][
                        "episode_id_set_sha256"
                    ],
                    "pair_id_set_sha256": contract["support"]["pair_id_set_sha256"],
                    "pair_atomic_split": split,
                    "component_membership_sha256": contract[
                        "component_membership_sha256"
                    ],
                    "routing_receipt_ids": contract["routing_receipt_ids"],
                },
                "boundary": {
                    "manifest_only": True,
                    "model_loaded": False,
                    "weights_written": False,
                    "sae_training_authorized": False,
                },
            }
            jobs.append({**job_body, "job_id": content_sha256(job_body)})

    body = {
        "schema": PLAN_SCHEMA,
        "status": "complete_manifest_only_cluster_aware_sae_plan",
        "source": {
            "predecessor_training_plan_id": predecessor_id,
            "predecessor_training_plan_sha256": predecessor_sha,
            "category_routing": {
                category: {
                    "discovery_routing_id": evidence["discovery"][
                        "cluster_episode_routing_id"
                    ],
                    "cluster_validation_routing_id": evidence[
                        "cluster_validation"
                    ]["cluster_episode_routing_id"],
                    "routing_stability_id": evidence["stability"][
                        "routing_stability_id"
                    ],
                    "sae_training_routing_id": evidence["sae_training"][
                        "cluster_episode_routing_id"
                    ],
                }
                for category, evidence in validated.items()
            },
        },
        "profile": dict(profile),
        "policy": {
            "initialization_seeds": sorted(initialization_seeds),
            "split_seed": split_seed,
            "validation_fraction": validation_fraction,
            "minimum_specific_episodes": minimum_specific_episodes,
            "minimum_specific_pairs": minimum_specific_pairs,
            "minimum_shared_episodes": minimum_shared_episodes,
            "minimum_shared_pairs": minimum_shared_pairs,
            "specific_dictionary_features": specific_dictionary_features,
            "shared_dictionary_features": shared_dictionary_features,
            "sae_activation_rows_per_episode": sae_activation_rows_per_episode,
            "routing_effect_rows_are_not_sae_activation_rows": True,
            "under_supported_routes_merge_to_shared": True,
            "cross_route_pair_conflicts_merge_to_shared": True,
            "category_archive_duplication_prohibited": True,
        },
        "compartments": contracts,
        "training_jobs": jobs,
        "routing_adjustments": {
            "cross_route_pair_conflicts": cross_route_pair_conflicts,
            "under_supported_routes": under_supported_routes,
        },
        "summary": {
            "categories": len(validated),
            "specific_compartments": len(specific_keys),
            "shared_compartments": 1,
            "training_jobs": len(jobs),
            "routed_training_episodes": len(raw_rows),
            "routed_training_pairs": len(rows_by_pair),
            "primary_compartment_episode_sets_disjoint": True,
            "primary_compartment_pair_sets_disjoint": True,
        },
        "boundary": {
            "model_loaded": False,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "manifest_only": True,
            "sae_training_authorized": False,
            "execution_authorized": False,
        },
    }
    return {**body, "cluster_aware_sae_plan_id": content_sha256(body)}


def verify_cluster_aware_sae_plan_v1(plan: dict[str, Any]) -> dict[str, Any]:
    body = dict(plan)
    identifier = str(body.pop("cluster_aware_sae_plan_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("cluster-aware SAE plan address differs")
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != (
        "complete_manifest_only_cluster_aware_sae_plan"
    ):
        raise ValueError("cluster-aware SAE plan status differs")
    if plan.get("boundary", {}).get("sae_training_authorized") is not False:
        raise ValueError("cluster-aware SAE plan grants training authority")
    job_ids = [str(row.get("job_id", "")) for row in plan.get("training_jobs", [])]
    if not job_ids or len(job_ids) != len(set(job_ids)):
        raise ValueError("cluster-aware SAE plan job identities are invalid")
    return {
        "schema": "sch.vpd-cluster-aware-sae-training-plan-verification.v1",
        "status": "pass",
        "cluster_aware_sae_plan_id": identifier,
        "compartments": len(plan["compartments"]),
        "training_jobs": len(job_ids),
        "sae_training_authorized": False,
    }
