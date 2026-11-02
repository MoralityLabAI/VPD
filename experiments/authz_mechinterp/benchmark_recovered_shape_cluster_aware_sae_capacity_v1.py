from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.authz_mechinterp.activation_categories import (
    CATEGORY_PROTOCOLS,
    activation_category,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.load_vpd_capture_router_inputs_v1 import (
    split_role_identity_v1,
)
from experiments.authz_mechinterp.plan_cluster_aware_sae_training_v1 import (
    build_cluster_aware_sae_plan_v1,
)
from experiments.authz_mechinterp.plan_cluster_aware_sae_training_v2 import (
    _global_pair_split,
    build_cluster_aware_sae_plan_v2,
    verify_cluster_aware_sae_plan_v2,
)
from experiments.authz_mechinterp.route_vpd_cluster_episodes_v1 import (
    _array_sha256,
    _canonical_groups,
    assess_routing_stability_v1,
    route_cluster_episodes_v1,
    verify_routing_receipt_v1,
    verify_routing_stability_v1,
)


ROLES = ("discovery", "cluster_validation", "sae_training")
COMPONENT_MEMBERSHIP = {
    "simulated-component-a0": "mechanism-a",
    "simulated-component-a1": "mechanism-a",
    "simulated-component-b0": "mechanism-b",
    "simulated-component-b1": "mechanism-b",
}
SCENARIOS = {
    "balanced": (0.45, 0.45, 0.10),
    "skewed": (0.80, 0.15, 0.05),
    "ambiguous_heavy": (0.30, 0.30, 0.40),
    "sparse_minority": (0.96, 0.01, 0.03),
    "no_shared": (0.50, 0.50, 0.00),
}


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (ValueError, FileNotFoundError, RuntimeError, KeyError):
        return True
    return False


def _normalize_row(row: dict[str, Any], *, category: str) -> dict[str, str]:
    template = str(
        row.get("template_group")
        or row.get("activation_category_pack", {}).get(
            "semantic_template_group", ""
        )
    )
    normalized = {
        "episode_id": str(row.get("episode_id", "")),
        "pair_id": str(row.get("pair_id", "")),
        "activation_category": category,
        "condition_id": str(row.get("condition", "")),
        "template_group_id": template,
    }
    if any(not value for value in normalized.values()):
        raise ValueError("recovered-shape incidence row identity is incomplete")
    return normalized


def _verify_role_summary(summary: dict[str, Any], record: dict[str, Any]) -> None:
    checks = {
        "rows": summary["rows"] == int(record["rows"]),
        "pairs": summary["pairs"] == int(record["pairs"]),
        "activation_category_counts": summary["activation_category_counts"]
        == dict(sorted(record["activation_category_counts"].items())),
        "episode_id_set_sha256": summary["episode_id_set_sha256"]
        == record["episode_id_set_sha256"],
        "pair_id_set_sha256": summary["pair_id_set_sha256"]
        == record["pair_id_set_sha256"],
        "template_group_set_sha256": summary["template_group_set_sha256"]
        == record["template_group_set_sha256"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("recovered-shape role summary differs: " + ", ".join(failed))


def _load_role(
    *, split_root: Path, manifest: dict[str, Any], role: str
) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any], str]:
    record = manifest["splits"][role]
    artifact = record["artifact"]
    path = split_root / artifact["path"]
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError("recovered-shape role artifact hash differs")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    episode_ids: list[str] = []
    pair_ids: set[str] = set()
    template_groups: set[str] = set()
    category_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (
                row.get("actual_execution") is not False
                or row.get("execution_path_exposed") is not False
                or row.get("labels", {}).get("candidate_effect") != "none"
            ):
                raise ValueError("recovered-shape role crosses no-action boundary")
            category = activation_category(row)
            normalized = _normalize_row(row, category=category)
            grouped[category].append(normalized)
            episode_ids.append(normalized["episode_id"])
            pair_ids.add(normalized["pair_id"])
            template_groups.add(normalized["template_group_id"])
            category_counts[category] += 1
    grouped = {
        category: sorted(rows, key=lambda row: row["episode_id"])
        for category, rows in sorted(grouped.items())
    }
    if set(grouped) != set(CATEGORY_PROTOCOLS):
        raise ValueError("recovered-shape role categories differ")
    summary = {
        "rows": len(episode_ids),
        "pairs": len(pair_ids),
        "activation_category_counts": dict(sorted(category_counts.items())),
        "episode_id_set_sha256": content_sha256(sorted(episode_ids)),
        "pair_id_set_sha256": content_sha256(sorted(pair_ids)),
        "template_group_set_sha256": content_sha256(sorted(template_groups)),
    }
    _verify_role_summary(summary, record)
    role_id, role_identity = split_role_identity_v1(
        {
            "split_id": manifest["split_id"],
            "split_role": role,
            "dataset_sha256": artifact["sha256"],
            "episode_id_set_sha256": record["episode_id_set_sha256"],
            "pair_id_set_sha256": record["pair_id_set_sha256"],
            "template_group_set_sha256": record["template_group_set_sha256"],
        }
    )
    summary["split_role_identity"] = role_identity
    return grouped, summary, role_id


def _pair_routes(
    pair_ids: set[str], *, weights: tuple[float, float, float]
) -> dict[str, str]:
    ordered = sorted(pair_ids)
    count_a = int(round(len(ordered) * weights[0]))
    count_b = int(round(len(ordered) * weights[1]))
    if count_a + count_b > len(ordered):
        count_b = len(ordered) - count_a
    result: dict[str, str] = {}
    for index, pair_id in enumerate(ordered):
        if index < count_a:
            route = "a"
        elif index < count_a + count_b:
            route = "b"
        else:
            route = "shared"
        result[pair_id] = route
    return result


def _route_receipt(
    *,
    role: str,
    role_id: str,
    category: str,
    episodes: list[dict[str, str]],
    pair_routes: dict[str, str],
    scenario: str,
    partition_sha256: str,
    admission_id: str,
) -> dict[str, Any]:
    arrays = {
        component_id: np.zeros((len(episodes), 2), dtype=np.float32)
        for component_id in COMPONENT_MEMBERSHIP
    }
    for index, episode in enumerate(episodes):
        route = pair_routes[episode["pair_id"]]
        if route == "a":
            values = {"a": 2.0, "b": 0.1}
        elif route == "b":
            values = {"a": 0.1, "b": 2.0}
        else:
            values = {"a": 1.0, "b": 1.0}
        for component_id, membership in COMPONENT_MEMBERSHIP.items():
            arrays[component_id][index, 0] = values[
                "a" if membership == "mechanism-a" else "b"
            ]
    effect_hashes = {
        component_id: _array_sha256(values)
        for component_id, values in arrays.items()
    }
    return route_cluster_episodes_v1(
        activation_category=category,
        split_role=role,
        split_id=role_id,
        partition_identity_admission_id=admission_id,
        partition_admission_status="pass_candidate_partition_identity_v4",
        category_partition_status="pass_exact_partition_identity",
        category_partition_sha256=partition_sha256,
        capture_execution_id=content_sha256(
            {
                "synthetic_simulation_only": True,
                "scenario": scenario,
                "role": role,
                "category": category,
            }
        ),
        episodes=episodes,
        capture_episode_order_sha256=content_sha256(
            [row["episode_id"] for row in episodes]
        ),
        component_membership=COMPONENT_MEMBERSHIP,
        component_effects=arrays,
        expected_component_effect_sha256=effect_hashes,
        component_artifact_sha256={
            component_id: content_sha256(
                {
                    "synthetic_simulation_only": True,
                    "scenario": scenario,
                    "role": role,
                    "component_id": component_id,
                }
            )
            for component_id in COMPONENT_MEMBERSHIP
        },
        activation_rows_per_episode=1,
        minimum_response_rms=0.0,
        minimum_margin_absolute=0.0,
        minimum_margin_ratio=0.5,
    )


def _stable_reference(
    *,
    role_rows: dict[str, dict[str, list[dict[str, str]]]],
    role_ids: dict[str, str],
    partition_sha256: str,
    admission_id: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for category in sorted(CATEGORY_PROTOCOLS):
        routes: dict[str, dict[str, Any]] = {}
        for role in ("discovery", "cluster_validation"):
            episodes = role_rows[role][category]
            routes[role] = _route_receipt(
                role=role,
                role_id=role_ids[role],
                category=category,
                episodes=episodes,
                pair_routes=_pair_routes(
                    {row["pair_id"] for row in episodes},
                    weights=SCENARIOS["balanced"],
                ),
                scenario="stable_reference",
                partition_sha256=partition_sha256,
                admission_id=admission_id,
            )
            verify_routing_receipt_v1(routes[role])
        stability = assess_routing_stability_v1(
            routes["discovery"],
            routes["cluster_validation"],
            maximum_total_variation=1.0,
            maximum_condition_total_variation=1.0,
            maximum_shared_rate_difference=1.0,
            minimum_condition_support_per_role=1,
        )
        if not verify_routing_stability_v1(stability)[
            "eligible_for_cluster_aware_sae_planning"
        ]:
            raise ValueError("recovered-shape reference stability rejected")
        result[category] = {
            "discovery": routes["discovery"],
            "cluster_validation": routes["cluster_validation"],
            "stability": stability,
        }
    return result


def _scenario_evidence(
    *,
    scenario: str,
    weights: tuple[float, float, float],
    reference: dict[str, dict[str, Any]],
    training_rows: dict[str, list[dict[str, str]]],
    training_role_id: str,
    partition_sha256: str,
    admission_id: str,
    sparse_category: str | None = None,
) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for category in sorted(CATEGORY_PROTOCOLS):
        episodes = training_rows[category]
        effective_weights = (
            SCENARIOS["sparse_minority"] if category == sparse_category else weights
        )
        training = _route_receipt(
            role="sae_training",
            role_id=training_role_id,
            category=category,
            episodes=episodes,
            pair_routes=_pair_routes(
                {row["pair_id"] for row in episodes}, weights=effective_weights
            ),
            scenario=scenario,
            partition_sha256=partition_sha256,
            admission_id=admission_id,
        )
        verify_routing_receipt_v1(training)
        evidence[category] = {
            **reference[category],
            "sae_training": training,
        }
    return evidence


def _planner_kwargs(
    *, project_root: Path, evidence: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    predecessor_path = (
        project_root / "artifacts/vpd_compartment_sae_training_plan_v4/plan.json"
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    return {
        "category_evidence": evidence,
        "predecessor_training_plan_id": predecessor["training_plan_id"],
        "predecessor_training_plan_sha256": sha256_file(predecessor_path),
        "profile": predecessor["profile"],
        "initialization_seeds": [17, 29],
        "split_seed": 1707,
        "validation_fraction": 0.20,
        "minimum_specific_episodes": 32,
        "minimum_specific_pairs": 16,
        "minimum_shared_episodes": 64,
        "minimum_shared_pairs": 16,
        "specific_dictionary_features": 4096,
        "shared_dictionary_features": 8192,
        "sae_activation_rows_per_episode": int(
            predecessor["profile"]["activation_rows_per_episode"]
        ),
    }


def build_report(project_root: Path) -> dict[str, Any]:
    split_root = project_root / "artifacts/vpd_activation_category_capacity_split_v4"
    manifest_path = split_root / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    role_rows: dict[str, dict[str, list[dict[str, str]]]] = {}
    role_summaries: dict[str, dict[str, Any]] = {}
    role_ids: dict[str, str] = {}
    for role in ROLES:
        role_rows[role], role_summaries[role], role_ids[role] = _load_role(
            split_root=split_root, manifest=manifest, role=role
        )
    groups, _ = _canonical_groups(COMPONENT_MEMBERSHIP)
    partition_sha256 = content_sha256(groups)
    admission_id = content_sha256(
        {"synthetic_simulation_only": True, "fixture": "recovered-shape-capacity"}
    )
    reference = _stable_reference(
        role_rows=role_rows,
        role_ids=role_ids,
        partition_sha256=partition_sha256,
        admission_id=admission_id,
    )
    scenario_results: dict[str, Any] = {}
    scenario_plans: dict[str, dict[str, Any]] = {}
    for scenario in ("balanced", "skewed", "ambiguous_heavy", "sparse_minority"):
        evidence = _scenario_evidence(
            scenario=scenario,
            weights=SCENARIOS[scenario],
            reference=reference,
            training_rows=role_rows["sae_training"],
            training_role_id=role_ids["sae_training"],
            partition_sha256=partition_sha256,
            admission_id=admission_id,
        )
        plan = build_cluster_aware_sae_plan_v2(
            **_planner_kwargs(project_root=project_root, evidence=evidence)
        )
        verification = verify_cluster_aware_sae_plan_v2(plan)
        scenario_plans[scenario] = plan
        scenario_results[scenario] = {
            "cluster_aware_sae_plan_id": plan["cluster_aware_sae_plan_id"],
            "global_pair_split_id": plan["global_pair_split"][
                "global_pair_split_id"
            ],
            "summary": plan["summary"],
            "under_supported_routes": len(
                plan["routing_adjustments"]["under_supported_routes"]
            ),
            "verification": verification,
        }
    category_sparse_evidence = _scenario_evidence(
        scenario="category_sparse",
        weights=SCENARIOS["balanced"],
        reference=reference,
        training_rows=role_rows["sae_training"],
        training_role_id=role_ids["sae_training"],
        partition_sha256=partition_sha256,
        admission_id=admission_id,
        sparse_category="valid_authority",
    )
    category_sparse_plan = build_cluster_aware_sae_plan_v2(
        **_planner_kwargs(project_root=project_root, evidence=category_sparse_evidence)
    )
    verify_cluster_aware_sae_plan_v2(category_sparse_plan)
    scenario_plans["category_sparse"] = category_sparse_plan
    scenario_results["category_sparse"] = {
        "cluster_aware_sae_plan_id": category_sparse_plan[
            "cluster_aware_sae_plan_id"
        ],
        "global_pair_split_id": category_sparse_plan["global_pair_split"][
            "global_pair_split_id"
        ],
        "summary": category_sparse_plan["summary"],
        "under_supported_routes": len(
            category_sparse_plan["routing_adjustments"]["under_supported_routes"]
        ),
    }

    no_shared_evidence = _scenario_evidence(
        scenario="no_shared",
        weights=SCENARIOS["no_shared"],
        reference=reference,
        training_rows=role_rows["sae_training"],
        training_role_id=role_ids["sae_training"],
        partition_sha256=partition_sha256,
        admission_id=admission_id,
    )
    no_shared_rejected = _rejects(
        lambda: build_cluster_aware_sae_plan_v2(
            **_planner_kwargs(project_root=project_root, evidence=no_shared_evidence)
        )
    )
    v1_balanced = build_cluster_aware_sae_plan_v1(
        **_planner_kwargs(
            project_root=project_root,
            evidence=_scenario_evidence(
                scenario="v1_balanced",
                weights=SCENARIOS["balanced"],
                reference=reference,
                training_rows=role_rows["sae_training"],
                training_role_id=role_ids["sae_training"],
                partition_sha256=partition_sha256,
                admission_id=admission_id,
            ),
        )
    )
    triangle_rejected = _rejects(
        lambda: _global_pair_split(
            {
                "triangle-a": [{"pair_id": "a"}, {"pair_id": "b"}],
                "triangle-b": [{"pair_id": "b"}, {"pair_id": "c"}],
                "triangle-c": [{"pair_id": "a"}, {"pair_id": "c"}],
            },
            validation_fraction=0.5,
            split_seed=1707,
        )
    )
    tampered_summary = copy.deepcopy(role_summaries["sae_training"])
    tampered_summary["rows"] -= 1
    recovered_row_tamper_rejected = _rejects(
        lambda: _verify_role_summary(
            tampered_summary, manifest["splits"]["sae_training"]
        )
    )
    balanced = scenario_plans["balanced"]
    sparse = scenario_plans["sparse_minority"]
    category_sparse = scenario_plans["category_sparse"]
    role_identities_distinct = len(set(role_ids.values())) == len(ROLES)
    all_roles_share_parent = len(
        {
            summary["split_role_identity"]["semantic_split_id"]
            for summary in role_summaries.values()
        }
    ) == 1
    controls = {
        "exact_recovered_manifest_and_role_hashes_pass": all(
            summary["rows"] == manifest["splits"][role]["rows"]
            for role, summary in role_summaries.items()
        ),
        "exact_sae_training_shape_6292_rows_572_pairs": role_summaries[
            "sae_training"
        ]["rows"]
        == 6292
        and role_summaries["sae_training"]["pairs"] == 572,
        "all_roles_share_parent_split_but_role_ids_are_distinct": all_roles_share_parent
        and role_identities_distinct,
        "balanced_preserves_16_specific_plus_shared": balanced["summary"][
            "specific_compartments"
        ]
        == 16
        and balanced["summary"]["shared_compartments"] == 1,
        "balanced_assigns_all_6292_episodes_once": sum(
            row["support"]["episodes"] for row in balanced["compartments"]
        )
        == 6292
        == balanced["summary"]["routed_training_episodes"],
        "balanced_preserves_572_global_pairs": balanced["summary"][
            "routed_training_pairs"
        ]
        == 572,
        "skewed_and_ambiguous_heavy_shapes_verify": all(
            scenario_results[name]["verification"]["status"] == "pass"
            for name in ("skewed", "ambiguous_heavy")
        ),
        "sparse_minority_merges_eight_routes_without_oversampling": sparse[
            "summary"
        ]["specific_compartments"]
        == 8
        and len(sparse["routing_adjustments"]["under_supported_routes"]) == 8,
        "category_sparse_merges_only_one_route": category_sparse["summary"][
            "specific_compartments"
        ]
        == 15
        and len(category_sparse["routing_adjustments"]["under_supported_routes"])
        == 1,
        "no_shared_fallback_rejected": no_shared_rejected,
        "v1_exact_recovered_shape_collapses_specific_compartments": v1_balanced[
            "summary"
        ]["specific_compartments"]
        == 0,
        "global_pair_split_rejects_unsplittable_triangle": triangle_rejected,
        "recovered_incidence_tamper_rejected": recovered_row_tamper_rejected,
        "routing_effect_and_sae_activation_units_separate": sum(
            row["support"]["routing_effect_rows"]
            for row in balanced["compartments"]
        )
        == 6292
        and sum(
            row["support"]["sae_activation_rows"]
            for row in balanced["compartments"]
        )
        == 6292 * 128,
        "all_outputs_explicitly_simulated_not_empirical": True,
    }
    if not all(controls.values()):
        raise ValueError(
            "recovered-shape capacity benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-recovered-shape-cluster-aware-sae-capacity-benchmark.v1",
        "status": "pass",
        "recovered_incidence": {
            "manifest_path": manifest_path.relative_to(project_root).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "semantic_split_id": manifest["split_id"],
            "role_summaries": role_summaries,
            "role_ids": role_ids,
        },
        "simulation_design": {
            "simulated_component_membership": COMPONENT_MEMBERSHIP,
            "simulated_partition_sha256": partition_sha256,
            "scenarios": {
                **{name: list(weights) for name, weights in SCENARIOS.items()},
                "category_sparse": {
                    "default": list(SCENARIOS["balanced"]),
                    "valid_authority": list(SCENARIOS["sparse_minority"]),
                },
            },
            "simulation_only_not_activation_evidence": True,
        },
        "results": {
            "scenarios": scenario_results,
            "v1_balanced_specific_compartments": v1_balanced["summary"][
                "specific_compartments"
            ],
            "balanced_compartment_support": {
                row["compartment_id"]: {
                    key: value
                    for key, value in row["support"].items()
                    if key not in {"episode_ids", "pair_ids"}
                }
                for row in balanced["compartments"]
            },
        },
        "controls": controls,
        "summary": {
            "controls": len(controls),
            "passed": sum(bool(value) for value in controls.values()),
        },
        "source": {
            "benchmark_sha256": sha256_file(Path(__file__)),
            "planner_v2_sha256": sha256_file(
                Path(__file__).with_name("plan_cluster_aware_sae_training_v2.py")
            ),
            "split_role_identity_sha256": sha256_file(
                Path(__file__).with_name("load_vpd_capture_router_inputs_v1.py")
            ),
            "router_v1_sha256": sha256_file(
                Path(__file__).with_name("route_vpd_cluster_episodes_v1.py")
            ),
        },
        "scientific_boundary": {
            "recovered_episode_metadata_read_only": True,
            "route_allocations_simulated": True,
            "simulated_routes_are_not_empirical_activation_evidence": True,
            "model_loaded": False,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "sae_dictionary_trained": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
            "release_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("recovered-shape capacity benchmark address differs")
    if observed != build_report(project_root):
        raise ValueError("recovered-shape capacity benchmark is not reproducible")
    return {
        "schema": "sch.vpd-recovered-shape-cluster-aware-sae-capacity-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "controls": observed["summary"]["controls"],
        "sae_training_rows": observed["recovered_incidence"]["role_summaries"][
            "sae_training"
        ]["rows"],
        "sae_training_pairs": observed["recovered_incidence"]["role_summaries"][
            "sae_training"
        ]["pairs"],
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark shared-pair-safe SAE planning on recovered incidence."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.output is not None:
        result = build_report(project_root)
        write_json(args.output, result)
    else:
        result = verify_benchmark(args.verify, project_root=project_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
