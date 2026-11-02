from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    stable_hash,
)
from experiments.authz_mechinterp.plan_cluster_aware_sae_training_v1 import (
    SHARED_COMPARTMENT,
    _require_content_id,
    _validate_category_evidence,
)
from experiments.authz_mechinterp.route_vpd_cluster_episodes_v1 import SHARED_ROUTE


PLAN_SCHEMA = "sch.vpd-cluster-aware-sae-training-plan.v2"
JOB_SCHEMA = "sch.vpd-cluster-aware-sae-training-job.v2"


def _global_pair_split(
    compartment_rows: dict[str, list[dict[str, Any]]],
    *,
    validation_fraction: float,
    split_seed: int,
) -> dict[str, Any]:
    pair_sets = {
        key: {str(row["pair_id"]) for row in rows}
        for key, rows in sorted(compartment_rows.items())
    }
    if any(len(values) < 2 for values in pair_sets.values()):
        raise ValueError("cluster-aware SAE V2 compartment requires at least two pairs")
    all_pairs = sorted(set().union(*pair_sets.values()))
    target = min(
        len(all_pairs) - 1,
        max(1, int(round(len(all_pairs) * validation_fraction))),
    )
    ordered = sorted(
        all_pairs,
        key=lambda pair_id: (stable_hash(split_seed, "global-pair-split-v2", pair_id), pair_id),
    )
    selected: set[str] = set()

    def can_select(pair_id: str) -> bool:
        return all(
            len((selected | {pair_id}) & pairs) < len(pairs)
            for pairs in pair_sets.values()
            if pair_id in pairs
        )

    for compartment_id, pairs in sorted(pair_sets.items()):
        if selected & pairs:
            continue
        candidates = [pair_id for pair_id in ordered if pair_id in pairs and can_select(pair_id)]
        if not candidates:
            raise ValueError(
                f"cluster-aware SAE V2 cannot give {compartment_id} validation support"
            )
        selected.add(candidates[0])
    for pair_id in ordered:
        if len(selected) >= target:
            break
        if pair_id not in selected and can_select(pair_id):
            selected.add(pair_id)
    if any(not (selected & pairs) or not (pairs - selected) for pairs in pair_sets.values()):
        raise ValueError("cluster-aware SAE V2 global split lacks compartment coverage")
    validation_pairs = sorted(selected)
    training_pairs = sorted(set(all_pairs) - selected)
    body = {
        "schema": "sch.vpd-cluster-aware-sae-global-pair-split.v2",
        "split_seed": split_seed,
        "requested_validation_fraction": validation_fraction,
        "observed_validation_fraction": len(validation_pairs) / len(all_pairs),
        "training_pair_ids": training_pairs,
        "validation_pair_ids": validation_pairs,
        "training_pair_id_set_sha256": content_sha256(training_pairs),
        "validation_pair_id_set_sha256": content_sha256(validation_pairs),
        "all_pair_id_set_sha256": content_sha256(all_pairs),
        "pair_sets_disjoint": True,
        "one_global_assignment_across_all_category_compartments": True,
        "every_compartment_has_training_and_validation_pairs": True,
    }
    return {**body, "global_pair_split_id": content_sha256(body)}


def _compartment_split(
    rows: list[dict[str, Any]], *, global_split: dict[str, Any]
) -> dict[str, Any]:
    validation_global = set(global_split["validation_pair_ids"])
    pair_ids = sorted({str(row["pair_id"]) for row in rows})
    validation_pairs = sorted(set(pair_ids) & validation_global)
    training_pairs = sorted(set(pair_ids) - validation_global)
    if not validation_pairs or not training_pairs:
        raise ValueError("cluster-aware SAE V2 compartment split lacks both sides")
    training_episode_ids = sorted(
        str(row["episode_id"])
        for row in rows
        if str(row["pair_id"]) in set(training_pairs)
    )
    validation_episode_ids = sorted(
        str(row["episode_id"])
        for row in rows
        if str(row["pair_id"]) in set(validation_pairs)
    )
    return {
        "schema": "sch.vpd-cluster-aware-sae-pair-atomic-split.v2",
        "global_pair_split_id": global_split["global_pair_split_id"],
        "training_pair_ids": training_pairs,
        "validation_pair_ids": validation_pairs,
        "training_pair_id_set_sha256": content_sha256(training_pairs),
        "validation_pair_id_set_sha256": content_sha256(validation_pairs),
        "training_episode_ids": training_episode_ids,
        "validation_episode_ids": validation_episode_ids,
        "training_episode_id_set_sha256": content_sha256(training_episode_ids),
        "validation_episode_id_set_sha256": content_sha256(validation_episode_ids),
        "pair_sets_disjoint": True,
        "episode_sets_disjoint": True,
        "cross_compartment_assignment_globally_consistent": True,
    }


def build_cluster_aware_sae_plan_v2(
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
        raise ValueError("cluster-aware SAE V2 planning requires category evidence")
    predecessor_id = _require_content_id(
        predecessor_training_plan_id, field="predecessor training plan ID"
    )
    predecessor_sha = _require_content_id(
        predecessor_training_plan_sha256, field="predecessor training plan SHA-256"
    )
    if not isinstance(profile, dict) or not profile:
        raise ValueError("cluster-aware SAE V2 profile is empty")
    if (
        not isinstance(initialization_seeds, list)
        or len(initialization_seeds) < 2
        or len(initialization_seeds) != len(set(initialization_seeds))
        or any(not isinstance(value, int) or value < 0 for value in initialization_seeds)
    ):
        raise ValueError("cluster-aware SAE V2 requires unique nonnegative seeds")
    if not math.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        raise ValueError("cluster-aware SAE V2 validation fraction must be in (0, 1)")
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
            raise ValueError(f"cluster-aware SAE V2 {name} must be positive")
    if minimum_specific_pairs < 2 or minimum_shared_pairs < 2:
        raise ValueError("cluster-aware SAE V2 pair support must permit splitting")

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
            }
        for row in training["routing"]["rows"]:
            route = str(row["primary_route"])
            route_key = (
                SHARED_COMPARTMENT if route == SHARED_ROUTE else f"{category}::{route}"
            )
            if route_key != SHARED_COMPARTMENT and route_key not in route_metadata:
                raise ValueError("cluster-aware SAE V2 route lacks component membership")
            raw_rows.append(
                {
                    **row,
                    "source_routing_receipt_id": training["cluster_episode_routing_id"],
                    "routing_stability_id": evidence["stability"]["routing_stability_id"],
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
        raise ValueError("cluster-aware SAE V2 training episodes overlap categories")

    support_by_specific: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        if row["planned_route_key"] != SHARED_COMPARTMENT:
            support_by_specific[str(row["planned_route_key"])].append(row)
    under_supported: list[dict[str, Any]] = []
    for route_key, rows in sorted(support_by_specific.items()):
        pairs = {str(row["pair_id"]) for row in rows}
        if len(rows) < minimum_specific_episodes or len(pairs) < minimum_specific_pairs:
            under_supported.append(
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
        raise ValueError("cluster-aware SAE V2 shared fallback lacks frozen support")

    specific_keys = sorted(key for key in compartment_rows if key != SHARED_COMPARTMENT)
    compartment_id_by_key = {
        key: f"vpd-sae-compartment-{content_sha256(key)}" for key in specific_keys
    }
    compartment_id_by_key[SHARED_COMPARTMENT] = SHARED_COMPARTMENT
    rows_by_compartment = {
        compartment_id_by_key[key]: sorted(
            rows, key=lambda row: str(row["episode_id"])
        )
        for key, rows in compartment_rows.items()
    }
    global_split = _global_pair_split(
        rows_by_compartment,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )
    contracts: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for route_key in specific_keys + [SHARED_COMPARTMENT]:
        compartment_id = compartment_id_by_key[route_key]
        rows = rows_by_compartment[compartment_id]
        pair_ids = sorted({str(row["pair_id"]) for row in rows})
        row_ids = [str(row["episode_id"]) for row in rows]
        effect_rows_values = {
            int(
                validated[str(row["activation_category"])]["sae_training"]["policy"][
                    "activation_rows_per_episode"
                ]
            )
            for row in rows
        }
        if len(effect_rows_values) != 1:
            raise ValueError("cluster-aware SAE V2 routing effect row units differ")
        effect_rows_per_episode = next(iter(effect_rows_values))
        split = _compartment_split(rows, global_split=global_split)
        metadata = route_metadata.get(route_key)
        body = {
            "schema": "sch.vpd-cluster-aware-sae-compartment.v2",
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
                "episodes": len(row_ids),
                "pairs": len(pair_ids),
                "episode_ids": row_ids,
                "pair_ids": pair_ids,
                "episode_id_set_sha256": content_sha256(row_ids),
                "pair_id_set_sha256": content_sha256(pair_ids),
                "routing_effect_rows": len(row_ids) * effect_rows_per_episode,
                "routing_effect_rows_per_episode": effect_rows_per_episode,
                "sae_activation_rows": len(row_ids) * sae_activation_rows_per_episode,
                "sae_activation_rows_per_episode": sae_activation_rows_per_episode,
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
        contract = {**body, "compartment_contract_id": content_sha256(body)}
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
                    "global_pair_split_id": global_split["global_pair_split_id"],
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
        "status": "complete_manifest_only_shared_pair_safe_cluster_aware_sae_plan_v2",
        "source": {
            "predecessor_training_plan_id": predecessor_id,
            "predecessor_training_plan_sha256": predecessor_sha,
            "category_routing": {
                category: {
                    "discovery_routing_id": evidence["discovery"][
                        "cluster_episode_routing_id"
                    ],
                    "cluster_validation_routing_id": evidence["cluster_validation"][
                        "cluster_episode_routing_id"
                    ],
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
        "global_pair_split": global_split,
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
            "same_pair_may_span_activation_category_compartments": True,
            "one_global_pair_split_prevents_cross_compartment_leakage": True,
            "under_supported_routes_merge_to_shared": True,
            "category_archive_duplication_prohibited": True,
        },
        "compartments": contracts,
        "training_jobs": jobs,
        "routing_adjustments": {"under_supported_routes": under_supported},
        "summary": {
            "categories": len(validated),
            "specific_compartments": len(specific_keys),
            "shared_compartments": 1,
            "training_jobs": len(jobs),
            "routed_training_episodes": len(raw_rows),
            "routed_training_pairs": len({str(row["pair_id"]) for row in raw_rows}),
            "primary_compartment_episode_sets_disjoint": True,
            "primary_compartment_pair_sets_disjoint": False,
            "cross_compartment_pair_overlap_managed_by_global_split": True,
            "global_pair_split_consistent": True,
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


def verify_cluster_aware_sae_plan_v2(plan: dict[str, Any]) -> dict[str, Any]:
    body = dict(plan)
    identifier = str(body.pop("cluster_aware_sae_plan_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("cluster-aware SAE V2 plan address differs")
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != (
        "complete_manifest_only_shared_pair_safe_cluster_aware_sae_plan_v2"
    ):
        raise ValueError("cluster-aware SAE V2 plan status differs")
    if plan.get("boundary", {}).get("sae_training_authorized") is not False:
        raise ValueError("cluster-aware SAE V2 plan grants training authority")
    global_split = plan["global_pair_split"]
    global_body = dict(global_split)
    global_id = str(global_body.pop("global_pair_split_id", ""))
    if global_id != content_sha256(global_body):
        raise ValueError("cluster-aware SAE V2 global pair split address differs")
    training_global = set(global_split["training_pair_ids"])
    validation_global = set(global_split["validation_pair_ids"])
    if not training_global or not validation_global or training_global & validation_global:
        raise ValueError("cluster-aware SAE V2 global pair split differs")
    all_episodes: set[str] = set()
    compartment_ids: set[str] = set()
    contract_ids: set[str] = set()
    for contract in plan["compartments"]:
        contract_body = dict(contract)
        contract_id = str(contract_body.pop("compartment_contract_id", ""))
        if contract_id != content_sha256(contract_body):
            raise ValueError("cluster-aware SAE V2 compartment address differs")
        compartment_id = str(contract["compartment_id"])
        if compartment_id in compartment_ids:
            raise ValueError("cluster-aware SAE V2 compartment identity repeats")
        compartment_ids.add(compartment_id)
        contract_ids.add(contract_id)
        support = contract["support"]
        episode_ids = list(support["episode_ids"])
        pair_ids = list(support["pair_ids"])
        if (
            episode_ids != sorted(episode_ids)
            or pair_ids != sorted(pair_ids)
            or len(episode_ids) != len(set(episode_ids))
            or content_sha256(episode_ids) != support["episode_id_set_sha256"]
            or content_sha256(pair_ids) != support["pair_id_set_sha256"]
        ):
            raise ValueError("cluster-aware SAE V2 compartment support differs")
        if all_episodes & set(episode_ids):
            raise ValueError("cluster-aware SAE V2 compartment episodes overlap")
        all_episodes.update(episode_ids)
        split = contract["pair_atomic_split"]
        if split["global_pair_split_id"] != global_id:
            raise ValueError("cluster-aware SAE V2 compartment uses another global split")
        if not set(split["training_pair_ids"]) <= training_global or not set(
            split["validation_pair_ids"]
        ) <= validation_global:
            raise ValueError("cluster-aware SAE V2 compartment pair side differs")
        if set(split["training_pair_ids"]) | set(split["validation_pair_ids"]) != set(
            pair_ids
        ):
            raise ValueError("cluster-aware SAE V2 compartment pair coverage differs")
    job_ids: set[str] = set()
    for job in plan["training_jobs"]:
        job_body = dict(job)
        job_id = str(job_body.pop("job_id", ""))
        if job_id != content_sha256(job_body) or job_id in job_ids:
            raise ValueError("cluster-aware SAE V2 job address differs")
        job_ids.add(job_id)
        if job["compartment_id"] not in compartment_ids or job[
            "compartment_contract_id"
        ] not in contract_ids:
            raise ValueError("cluster-aware SAE V2 job binds unknown compartment")
        if job["boundary"].get("sae_training_authorized") is not False:
            raise ValueError("cluster-aware SAE V2 job grants training authority")
    if len(all_episodes) != int(plan["summary"]["routed_training_episodes"]):
        raise ValueError("cluster-aware SAE V2 routed episode total differs")
    return {
        "schema": "sch.vpd-cluster-aware-sae-training-plan-verification.v2",
        "status": "pass",
        "cluster_aware_sae_plan_id": identifier,
        "global_pair_split_id": global_id,
        "compartments": len(compartment_ids),
        "training_jobs": len(job_ids),
        "shared_pair_safe": True,
        "sae_training_authorized": False,
    }
