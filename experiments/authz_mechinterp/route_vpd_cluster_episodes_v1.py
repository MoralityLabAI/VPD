from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from experiments.authz_mechinterp.activation_category_cluster_partition_identity_v4 import (
    candidate_partition,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256


ROUTING_SCHEMA = "sch.vpd-cluster-episode-routing.v1"
STABILITY_SCHEMA = "sch.vpd-cluster-episode-routing-stability.v1"
ROUTING_CANDIDATE = "route-energy-margin-v1"
SHARED_ROUTE = "shared_ambiguous"


def _array_sha256(array: np.ndarray) -> str:
    values = np.asarray(array)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _require_content_id(value: str, *, field: str) -> str:
    observed = str(value)
    if len(observed) != 64 or any(char not in "0123456789abcdef" for char in observed):
        raise ValueError(f"{field} must be a lowercase SHA-256 content ID")
    return observed


def _canonical_groups(
    component_membership: dict[str, str],
) -> tuple[list[list[str]], dict[str, str]]:
    component_ids = sorted(str(value) for value in component_membership)
    if not component_ids or any(not value for value in component_ids):
        raise ValueError("cluster routing requires nonempty component identities")
    groups = candidate_partition(component_ids, component_membership)
    cluster_id_by_component: dict[str, str] = {}
    for group in groups:
        cluster_id = f"vpd-cluster-{content_sha256(group)}"
        for component_id in group:
            cluster_id_by_component[component_id] = cluster_id
    return groups, cluster_id_by_component


def _validate_episode_rows(
    episodes: list[dict[str, Any]], *, activation_category: str
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in episodes:
        if not isinstance(row, dict):
            raise ValueError("cluster routing episode row must be an object")
        normalized_row = {
            "episode_id": str(row.get("episode_id", "")),
            "pair_id": str(row.get("pair_id", "")),
            "activation_category": str(row.get("activation_category", "")),
            "condition_id": str(row.get("condition_id", "")),
            "template_group_id": str(row.get("template_group_id", "")),
        }
        if any(not value for value in normalized_row.values()):
            raise ValueError("cluster routing episode identity fields must be nonempty")
        if normalized_row["activation_category"] != activation_category:
            raise ValueError("cluster routing episode category differs")
        normalized.append(normalized_row)
    if not normalized:
        raise ValueError("cluster routing requires episodes")
    episode_ids = [row["episode_id"] for row in normalized]
    if episode_ids != sorted(episode_ids):
        raise ValueError("cluster routing episodes must use exact sorted capture order")
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("cluster routing episode identities must be unique")
    return normalized


def _support_record(rows: list[dict[str, Any]]) -> dict[str, Any]:
    episode_ids = sorted(str(row["episode_id"]) for row in rows)
    pair_ids = sorted({str(row["pair_id"]) for row in rows})
    return {
        "episodes": len(episode_ids),
        "pairs": len(pair_ids),
        "episode_id_set_sha256": content_sha256(episode_ids),
        "pair_id_set_sha256": content_sha256(pair_ids),
    }


def route_cluster_episodes_v1(
    *,
    activation_category: str,
    split_role: str,
    split_id: str,
    partition_identity_admission_id: str,
    partition_admission_status: str,
    category_partition_status: str,
    category_partition_sha256: str,
    capture_execution_id: str,
    episodes: list[dict[str, Any]],
    capture_episode_order_sha256: str,
    component_membership: dict[str, str],
    component_effects: dict[str, np.ndarray],
    expected_component_effect_sha256: dict[str, str],
    component_artifact_sha256: dict[str, str],
    activation_rows_per_episode: int,
    minimum_response_rms: float,
    minimum_margin_absolute: float,
    minimum_margin_ratio: float,
    routing_candidate: str = ROUTING_CANDIDATE,
) -> dict[str, Any]:
    """Route each episode to one canonical component cluster or shared fallback."""

    if not activation_category:
        raise ValueError("cluster routing activation category is empty")
    if split_role not in {"discovery", "cluster_validation", "sae_training"}:
        raise ValueError(
            "cluster routing split role is not discovery, cluster_validation, or sae_training"
        )
    if not split_id:
        raise ValueError("cluster routing split identity is empty")
    admission_id = _require_content_id(
        partition_identity_admission_id, field="partition identity admission ID"
    )
    capture_id = _require_content_id(capture_execution_id, field="capture execution ID")
    if partition_admission_status != "pass_candidate_partition_identity_v4":
        raise ValueError("cluster routing requires passed V4 admission")
    if category_partition_status != "pass_exact_partition_identity":
        raise ValueError("cluster routing category lacks exact V4 partition identity")
    if routing_candidate != ROUTING_CANDIDATE:
        raise ValueError("unsupported cluster routing candidate")
    if not isinstance(activation_rows_per_episode, int) or activation_rows_per_episode < 1:
        raise ValueError("activation rows per episode must be positive")
    for value, name in (
        (minimum_response_rms, "minimum response RMS"),
        (minimum_margin_absolute, "minimum absolute margin"),
        (minimum_margin_ratio, "minimum margin ratio"),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"cluster routing {name} must be finite and nonnegative")
    if minimum_margin_ratio > 1.0:
        raise ValueError("cluster routing minimum margin ratio must be at most one")

    normalized_episodes = _validate_episode_rows(
        episodes, activation_category=activation_category
    )
    episode_ids = [row["episode_id"] for row in normalized_episodes]
    expected_order = content_sha256(episode_ids)
    if capture_episode_order_sha256 != expected_order:
        raise ValueError("cluster routing episode order differs from capture")

    groups, cluster_id_by_component = _canonical_groups(component_membership)
    membership_sha256 = content_sha256(groups)
    if category_partition_sha256 != membership_sha256:
        raise ValueError("cluster routing component membership differs from V4 partition")
    component_ids = sorted(cluster_id_by_component)
    if (
        set(component_effects) != set(component_ids)
        or set(expected_component_effect_sha256) != set(component_ids)
        or set(component_artifact_sha256) != set(component_ids)
    ):
        raise ValueError("cluster routing component artifacts do not cover exact partition")

    expected_rows = len(normalized_episodes) * activation_rows_per_episode
    hidden_size: int | None = None
    per_component_energy: dict[str, np.ndarray] = {}
    observed_effect_hashes: dict[str, str] = {}
    for component_id in component_ids:
        values = np.asarray(component_effects[component_id])
        if values.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("cluster routing component effects must be float32 or float64")
        if values.ndim != 2 or values.shape[0] != expected_rows or values.shape[1] < 1:
            raise ValueError("cluster routing component effect shape differs")
        if hidden_size is None:
            hidden_size = int(values.shape[1])
        elif values.shape[1] != hidden_size:
            raise ValueError("cluster routing component hidden sizes differ")
        if not np.all(np.isfinite(values)):
            raise ValueError("cluster routing component effects contain nonfinite values")
        observed_hash = _array_sha256(values)
        observed_effect_hashes[component_id] = observed_hash
        if expected_component_effect_sha256[component_id] != observed_hash:
            raise ValueError("cluster routing component effect hash differs")
        _require_content_id(
            str(component_artifact_sha256[component_id]),
            field="component artifact SHA-256",
        )
        reshaped = values.astype(np.float64, copy=False).reshape(
            len(normalized_episodes), activation_rows_per_episode, int(hidden_size)
        )
        per_component_energy[component_id] = np.sqrt(
            np.mean(np.square(reshaped), axis=(1, 2), dtype=np.float64)
        )

    cluster_members: dict[str, list[str]] = {}
    cluster_scores: dict[str, np.ndarray] = {}
    for group in groups:
        cluster_id = cluster_id_by_component[group[0]]
        cluster_members[cluster_id] = group
        member_energy = np.stack([per_component_energy[value] for value in group])
        cluster_scores[cluster_id] = np.sqrt(
            np.mean(np.square(member_energy), axis=0, dtype=np.float64)
        )

    initial_rows: list[dict[str, Any]] = []
    cluster_ids = sorted(cluster_members)
    for index, episode in enumerate(normalized_episodes):
        scores = {
            cluster_id: float(cluster_scores[cluster_id][index])
            for cluster_id in cluster_ids
        }
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        top_cluster, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_score - second_score
        margin_ratio = margin / top_score if top_score > 0.0 else 0.0
        if top_score < minimum_response_rms:
            route = SHARED_ROUTE
            reason = "below_minimum_response"
        elif margin < minimum_margin_absolute or margin_ratio < minimum_margin_ratio:
            route = SHARED_ROUTE
            reason = "ambiguous_cluster_margin"
        else:
            route = top_cluster
            reason = "unique_cluster_response"
        initial_rows.append(
            {
                **episode,
                "primary_route": route,
                "route_reason": reason,
                "top_cluster_id": top_cluster,
                "top_score": top_score,
                "second_score": float(second_score),
                "margin_absolute": float(margin),
                "margin_ratio": float(margin_ratio),
                "cluster_scores": dict(sorted(scores.items())),
            }
        )

    rows_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in initial_rows:
        rows_by_pair[str(row["pair_id"])].append(row)
    routed_rows: list[dict[str, Any]] = []
    pair_conflicts = 0
    for pair_id in sorted(rows_by_pair):
        pair_rows = rows_by_pair[pair_id]
        routes = {str(row["primary_route"]) for row in pair_rows}
        if len(routes) != 1:
            pair_conflicts += 1
            for row in pair_rows:
                routed_rows.append(
                    {**row, "primary_route": SHARED_ROUTE, "route_reason": "pair_route_conflict"}
                )
        else:
            routed_rows.extend(pair_rows)
    routed_rows.sort(key=lambda row: str(row["episode_id"]))

    route_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    condition_groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    template_groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in routed_rows:
        route = str(row["primary_route"])
        route_groups[route].append(row)
        condition_groups[str(row["condition_id"])][route].append(row)
        template_groups[str(row["template_group_id"])][route].append(row)

    all_route_ids = sorted(set(cluster_ids) | {SHARED_ROUTE})
    support_by_route = {
        route: _support_record(route_groups.get(route, [])) for route in all_route_ids
    }
    support_by_condition = {
        condition: {
            route: len(routes.get(route, [])) for route in all_route_ids
        }
        for condition, routes in sorted(condition_groups.items())
    }
    support_by_template_group = {
        template: {
            route: len(routes.get(route, [])) for route in all_route_ids
        }
        for template, routes in sorted(template_groups.items())
    }
    pair_routes: dict[str, str] = {}
    for pair_id, pair_rows in rows_by_pair.items():
        final_routes = {
            str(row["primary_route"])
            for row in routed_rows
            if str(row["pair_id"]) == pair_id
        }
        if len(final_routes) != 1:
            raise AssertionError("pair-atomic cluster routing failed")
        pair_routes[pair_id] = next(iter(final_routes))

    body = {
        "schema": ROUTING_SCHEMA,
        "status": "complete_model_free_cluster_episode_routing",
        "source": {
            "partition_identity_admission_id": admission_id,
            "partition_admission_status": partition_admission_status,
            "category_partition_status": category_partition_status,
            "category_partition_sha256": membership_sha256,
            "capture_execution_id": capture_id,
            "split_role": split_role,
            "split_id": split_id,
            "activation_category": activation_category,
            "capture_episode_order_sha256": expected_order,
            "episode_id_set_sha256": content_sha256(sorted(episode_ids)),
            "pair_id_set_sha256": content_sha256(
                sorted({row["pair_id"] for row in normalized_episodes})
            ),
            "component_artifact_sha256": dict(sorted(component_artifact_sha256.items())),
            "component_effect_sha256": dict(sorted(observed_effect_hashes.items())),
        },
        "partition": {
            "groups": groups,
            "cluster_members": dict(sorted(cluster_members.items())),
            "cluster_count": len(groups),
            "component_count": len(component_ids),
            "label_names_ignored": True,
            "cluster_ids_derived_from_membership": True,
        },
        "policy": {
            "candidate": routing_candidate,
            "score": "rms_over_component_members_episode_rows_and_hidden_coordinates",
            "cluster_size_normalized": True,
            "activation_rows_per_episode": activation_rows_per_episode,
            "minimum_response_rms": float(minimum_response_rms),
            "minimum_margin_absolute": float(minimum_margin_absolute),
            "minimum_margin_ratio": float(minimum_margin_ratio),
            "ambiguous_route": SHARED_ROUTE,
            "pair_atomic_routing": True,
            "all_category_duplication_prohibited": True,
        },
        "routing": {
            "rows": routed_rows,
            "support_by_route": support_by_route,
            "support_by_condition": support_by_condition,
            "support_by_template_group": support_by_template_group,
            "pair_routes": dict(sorted(pair_routes.items())),
            "pair_route_conflicts_sent_to_shared": pair_conflicts,
            "episodes_routed_exactly_once": len(routed_rows) == len(normalized_episodes),
            "primary_route_episode_sets_disjoint": True,
            "primary_route_pair_sets_disjoint": True,
        },
        "boundary": {
            "model_loaded_by_router": False,
            "model_inference_performed_by_router": False,
            "weights_loaded_or_mutated_by_router": False,
            "sae_training_authorized": False,
            "execution_authorized": False,
        },
    }
    return {**body, "cluster_episode_routing_id": content_sha256(body)}


def verify_routing_receipt_v1(receipt: dict[str, Any]) -> dict[str, Any]:
    body = dict(receipt)
    identifier = str(body.pop("cluster_episode_routing_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("cluster episode routing receipt address differs")
    if receipt.get("schema") != ROUTING_SCHEMA or receipt.get("status") != (
        "complete_model_free_cluster_episode_routing"
    ):
        raise ValueError("cluster episode routing receipt status differs")
    rows = receipt.get("routing", {}).get("rows", [])
    episode_ids = [str(row.get("episode_id", "")) for row in rows]
    if len(episode_ids) != len(set(episode_ids)) or episode_ids != sorted(episode_ids):
        raise ValueError("cluster episode routing rows are not exact and ordered")
    source = receipt.get("source", {})
    pair_ids = sorted({str(row.get("pair_id", "")) for row in rows})
    if (
        source.get("capture_episode_order_sha256") != content_sha256(episode_ids)
        or source.get("episode_id_set_sha256") != content_sha256(sorted(episode_ids))
        or source.get("pair_id_set_sha256") != content_sha256(pair_ids)
    ):
        raise ValueError("cluster episode routing source support hashes differ")
    groups = receipt.get("partition", {}).get("groups", [])
    expected_cluster_members = {
        f"vpd-cluster-{content_sha256(group)}": group for group in groups
    }
    if receipt.get("partition", {}).get("cluster_members") != dict(
        sorted(expected_cluster_members.items())
    ) or source.get("category_partition_sha256") != content_sha256(groups):
        raise ValueError("cluster episode routing partition binding differs")
    cluster_ids = sorted(expected_cluster_members)
    allowed_routes = set(cluster_ids) | {SHARED_ROUTE}
    if any(
        str(row.get("primary_route", "")) not in allowed_routes
        or str(row.get("top_cluster_id", "")) not in set(cluster_ids)
        or set(row.get("cluster_scores", {})) != set(cluster_ids)
        or str(row.get("activation_category", ""))
        != str(source.get("activation_category", ""))
        for row in rows
    ):
        raise ValueError("cluster episode routing row contract differs")
    routes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    conditions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    templates: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pairs: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        route = str(row["primary_route"])
        routes[route].append(row)
        conditions[str(row["condition_id"])][route] += 1
        templates[str(row["template_group_id"])][route] += 1
        pairs[str(row["pair_id"])].add(route)
    if any(len(value) != 1 for value in pairs.values()):
        raise ValueError("cluster episode routing pairs are not atomic")
    all_route_ids = sorted(allowed_routes)
    expected_support = {
        route: _support_record(routes.get(route, [])) for route in all_route_ids
    }
    expected_conditions = {
        condition: {route: counts.get(route, 0) for route in all_route_ids}
        for condition, counts in sorted(conditions.items())
    }
    expected_templates = {
        template: {route: counts.get(route, 0) for route in all_route_ids}
        for template, counts in sorted(templates.items())
    }
    expected_pair_routes = {
        pair_id: next(iter(route_set)) for pair_id, route_set in sorted(pairs.items())
    }
    routing = receipt["routing"]
    if (
        routing.get("support_by_route") != expected_support
        or routing.get("support_by_condition") != expected_conditions
        or routing.get("support_by_template_group") != expected_templates
        or routing.get("pair_routes") != expected_pair_routes
    ):
        raise ValueError("cluster episode routing derived support differs")
    if (
        routing.get("episodes_routed_exactly_once") is not True
        or routing.get("primary_route_episode_sets_disjoint") is not True
        or routing.get("primary_route_pair_sets_disjoint") is not True
    ):
        raise ValueError("cluster episode routing route sets are not disjoint")
    if receipt.get("boundary", {}).get("sae_training_authorized") is not False:
        raise ValueError("cluster episode routing receipt grants SAE training")
    return {
        "schema": "sch.vpd-cluster-episode-routing-verification.v1",
        "status": "pass",
        "cluster_episode_routing_id": identifier,
        "episodes": len(rows),
        "routes": len(receipt["routing"]["support_by_route"]),
        "sae_training_authorized": False,
    }


def _route_distribution(receipt: dict[str, Any]) -> dict[str, float]:
    support = receipt["routing"]["support_by_route"]
    total = sum(int(value["episodes"]) for value in support.values())
    if total < 1:
        raise ValueError("routing stability requires nonempty receipts")
    return {route: int(value["episodes"]) / total for route, value in support.items()}


def _total_variation(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def assess_routing_stability_v1(
    discovery: dict[str, Any],
    cluster_validation: dict[str, Any],
    *,
    maximum_total_variation: float,
    maximum_condition_total_variation: float,
    maximum_shared_rate_difference: float,
    minimum_condition_support_per_role: int,
) -> dict[str, Any]:
    for receipt in (discovery, cluster_validation):
        verify_routing_receipt_v1(receipt)
    if discovery["source"]["split_role"] != "discovery" or cluster_validation[
        "source"
    ]["split_role"] != "cluster_validation":
        raise ValueError("routing stability requires discovery and cluster_validation roles")
    if discovery["source"]["split_id"] == cluster_validation["source"]["split_id"]:
        raise ValueError("routing stability split identities must differ")
    discovery_ids = {str(row["episode_id"]) for row in discovery["routing"]["rows"]}
    validation_ids = {
        str(row["episode_id"]) for row in cluster_validation["routing"]["rows"]
    }
    if discovery_ids & validation_ids:
        raise ValueError("routing stability episode identities overlap")
    for field in ("activation_category", "category_partition_sha256"):
        if discovery["source"][field] != cluster_validation["source"][field]:
            raise ValueError(f"routing stability {field} differs")
    if discovery["partition"] != cluster_validation["partition"]:
        raise ValueError("routing stability component partition differs")
    if discovery["policy"] != cluster_validation["policy"]:
        raise ValueError("routing stability policy differs")
    for value, name in (
        (maximum_total_variation, "maximum total variation"),
        (maximum_condition_total_variation, "maximum condition total variation"),
        (maximum_shared_rate_difference, "maximum shared-rate difference"),
    ):
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"routing stability {name} must be in [0, 1]")
    if minimum_condition_support_per_role < 1:
        raise ValueError("routing stability minimum condition support must be positive")

    discovery_distribution = _route_distribution(discovery)
    validation_distribution = _route_distribution(cluster_validation)
    total_variation = _total_variation(discovery_distribution, validation_distribution)
    shared_rate_difference = abs(
        discovery_distribution.get(SHARED_ROUTE, 0.0)
        - validation_distribution.get(SHARED_ROUTE, 0.0)
    )
    discovery_conditions = discovery["routing"]["support_by_condition"]
    validation_conditions = cluster_validation["routing"]["support_by_condition"]
    if set(discovery_conditions) != set(validation_conditions):
        raise ValueError("routing stability condition strata differ")
    condition_evidence: dict[str, Any] = {}
    conditions_pass = True
    for condition in sorted(discovery_conditions):
        left_counts = discovery_conditions[condition]
        right_counts = validation_conditions[condition]
        left_total = sum(int(value) for value in left_counts.values())
        right_total = sum(int(value) for value in right_counts.values())
        enough_support = (
            left_total >= minimum_condition_support_per_role
            and right_total >= minimum_condition_support_per_role
        )
        left_distribution = {
            route: int(value) / left_total for route, value in left_counts.items()
        }
        right_distribution = {
            route: int(value) / right_total for route, value in right_counts.items()
        }
        condition_tv = _total_variation(left_distribution, right_distribution)
        passed = enough_support and condition_tv <= maximum_condition_total_variation
        conditions_pass &= passed
        condition_evidence[condition] = {
            "status": "pass" if passed else "reject",
            "discovery_episodes": left_total,
            "cluster_validation_episodes": right_total,
            "total_variation": condition_tv,
            "minimum_support_met": enough_support,
        }
    eligible = (
        total_variation <= maximum_total_variation
        and shared_rate_difference <= maximum_shared_rate_difference
        and conditions_pass
    )
    body = {
        "schema": STABILITY_SCHEMA,
        "status": "pass_routing_stability_v1" if eligible else "reject_routing_stability_v1",
        "source": {
            "discovery_routing_id": discovery["cluster_episode_routing_id"],
            "cluster_validation_routing_id": cluster_validation[
                "cluster_episode_routing_id"
            ],
            "activation_category": discovery["source"]["activation_category"],
            "category_partition_sha256": discovery["source"][
                "category_partition_sha256"
            ],
        },
        "policy": {
            "maximum_total_variation": maximum_total_variation,
            "maximum_condition_total_variation": maximum_condition_total_variation,
            "maximum_shared_rate_difference": maximum_shared_rate_difference,
            "minimum_condition_support_per_role": minimum_condition_support_per_role,
            "aggregate_agreement_alone_is_insufficient": True,
        },
        "evidence": {
            "overall_total_variation": total_variation,
            "shared_rate_difference": shared_rate_difference,
            "condition_strata": condition_evidence,
            "all_condition_strata_pass": conditions_pass,
            "episode_sets_disjoint": True,
        },
        "admission": {
            "eligible_for_cluster_aware_sae_planning": eligible,
            "sae_training_authorized": False,
        },
        "boundary": {
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "execution_authorized": False,
        },
    }
    return {**body, "routing_stability_id": content_sha256(body)}


def verify_routing_stability_v1(receipt: dict[str, Any]) -> dict[str, Any]:
    body = dict(receipt)
    identifier = str(body.pop("routing_stability_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("cluster episode routing stability address differs")
    if receipt.get("schema") != STABILITY_SCHEMA:
        raise ValueError("cluster episode routing stability schema differs")
    eligible = bool(
        receipt.get("admission", {}).get(
            "eligible_for_cluster_aware_sae_planning", False
        )
    )
    expected_status = (
        "pass_routing_stability_v1" if eligible else "reject_routing_stability_v1"
    )
    if receipt.get("status") != expected_status:
        raise ValueError("cluster episode routing stability status differs")
    if receipt.get("admission", {}).get("sae_training_authorized") is not False:
        raise ValueError("cluster episode routing stability grants SAE training")
    return {
        "schema": "sch.vpd-cluster-episode-routing-stability-verification.v1",
        "status": "pass",
        "routing_stability_id": identifier,
        "eligible_for_cluster_aware_sae_planning": eligible,
        "sae_training_authorized": False,
    }
