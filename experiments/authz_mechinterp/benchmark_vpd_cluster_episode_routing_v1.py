from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.authz_mechinterp.activation_category_cluster_partition_identity_v4 import (
    candidate_partition,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.route_vpd_cluster_episodes_v1 import (
    SHARED_ROUTE,
    assess_routing_stability_v1,
    route_cluster_episodes_v1,
    verify_routing_receipt_v1,
)


CATEGORY = "missing_or_untrusted_authority"
ROWS_PER_EPISODE = 2
HIDDEN_SIZE = 3
MEMBERSHIP = {
    "canonical-a0": "raw-alpha",
    "canonical-b0": "raw-beta",
    "canonical-b1": "raw-beta",
    "canonical-b2": "raw-beta",
}


def _effect_sha256(values: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _groups(membership: dict[str, str]) -> list[list[str]]:
    return candidate_partition(sorted(membership), membership)


def _route_id_for_group(group: list[str]) -> str:
    return f"vpd-cluster-{content_sha256(group)}"


def _make_inputs(
    *,
    role: str,
    prefix: str,
    kinds_by_condition: dict[str, list[str]],
    membership: dict[str, str] | None = None,
) -> dict[str, Any]:
    membership = dict(membership or MEMBERSHIP)
    episode_rows: list[dict[str, str]] = []
    kinds: list[str] = []
    index = 0
    for condition in sorted(kinds_by_condition):
        condition_kinds = kinds_by_condition[condition]
        if len(condition_kinds) % 2:
            raise ValueError("synthetic routing fixture requires two episodes per pair")
        for local_index, kind in enumerate(condition_kinds):
            pair_index = local_index // 2
            episode_rows.append(
                {
                    "episode_id": f"{prefix}-episode-{index:03d}",
                    "pair_id": f"{prefix}-pair-{condition}-{pair_index:02d}",
                    "activation_category": CATEGORY,
                    "condition_id": condition,
                    "template_group_id": f"{prefix}-template-{condition}-{pair_index:02d}",
                }
            )
            kinds.append(kind)
            index += 1
    order = np.argsort([row["episode_id"] for row in episode_rows])
    episode_rows = [episode_rows[int(value)] for value in order]
    kinds = [kinds[int(value)] for value in order]

    amplitudes = {
        "a": {
            "canonical-a0": 2.0,
            "canonical-b0": 0.2,
            "canonical-b1": 0.2,
            "canonical-b2": 0.2,
        },
        "b": {
            "canonical-a0": 0.2,
            "canonical-b0": 2.0,
            "canonical-b1": 2.0,
            "canonical-b2": 2.0,
        },
        "ambiguous": {
            "canonical-a0": 1.0,
            "canonical-b0": 1.0,
            "canonical-b1": 1.0,
            "canonical-b2": 1.0,
        },
        "low": {
            "canonical-a0": 0.05,
            "canonical-b0": 0.05,
            "canonical-b1": 0.05,
            "canonical-b2": 0.05,
        },
    }
    effects: dict[str, np.ndarray] = {}
    for component_id in sorted(membership):
        rows: list[np.ndarray] = []
        for kind in kinds:
            amplitude = amplitudes[kind][component_id]
            rows.append(
                np.full(
                    (ROWS_PER_EPISODE, HIDDEN_SIZE), amplitude, dtype=np.float32
                )
            )
        effects[component_id] = np.concatenate(rows, axis=0)
    return {
        "activation_category": CATEGORY,
        "split_role": role,
        "split_id": f"synthetic-{role}-{prefix}",
        "partition_identity_admission_id": content_sha256(
            {"synthetic_partition_admission": True}
        ),
        "partition_admission_status": "pass_candidate_partition_identity_v4",
        "category_partition_status": "pass_exact_partition_identity",
        "category_partition_sha256": content_sha256(_groups(membership)),
        "capture_execution_id": content_sha256(
            {"synthetic_capture": prefix, "role": role}
        ),
        "episodes": episode_rows,
        "capture_episode_order_sha256": content_sha256(
            [row["episode_id"] for row in episode_rows]
        ),
        "component_membership": membership,
        "component_effects": effects,
        "expected_component_effect_sha256": {
            component_id: _effect_sha256(values)
            for component_id, values in effects.items()
        },
        "component_artifact_sha256": {
            component_id: content_sha256(
                {"synthetic_component_artifact": prefix, "component": component_id}
            )
            for component_id in effects
        },
        "activation_rows_per_episode": ROWS_PER_EPISODE,
        "minimum_response_rms": 0.2,
        "minimum_margin_absolute": 0.3,
        "minimum_margin_ratio": 0.2,
    }


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except ValueError:
        return True
    return False


def _core_fixture(role: str, prefix: str) -> dict[str, Any]:
    return _make_inputs(
        role=role,
        prefix=prefix,
        kinds_by_condition={
            "condition-a": ["a", "a", "b", "b", "ambiguous", "ambiguous"],
            "condition-b": ["a", "a", "b", "b", "low", "low"],
            "condition-c": ["a", "b"],
        },
    )


def _condition_fixture(
    role: str, prefix: str, *, swapped: bool
) -> dict[str, Any]:
    return _make_inputs(
        role=role,
        prefix=prefix,
        kinds_by_condition=(
            {
                "condition-a": ["b", "b", "b", "b"],
                "condition-b": ["a", "a", "a", "a"],
            }
            if swapped
            else {
                "condition-a": ["a", "a", "a", "a"],
                "condition-b": ["b", "b", "b", "b"],
            }
        ),
    )


def build_report(*, seed: int = 140923) -> dict[str, Any]:
    del seed  # The fixture is exact and deliberately seed-free.
    discovery_inputs = _core_fixture("discovery", "d")
    validation_inputs = _core_fixture("cluster_validation", "v")
    before_effect_bytes = {
        key: value.tobytes(order="C")
        for key, value in discovery_inputs["component_effects"].items()
    }
    before_episodes = copy.deepcopy(discovery_inputs["episodes"])
    discovery = route_cluster_episodes_v1(**discovery_inputs)
    validation = route_cluster_episodes_v1(**validation_inputs)
    discovery_verification = verify_routing_receipt_v1(discovery)

    stable = assess_routing_stability_v1(
        discovery,
        validation,
        maximum_total_variation=0.05,
        maximum_condition_total_variation=0.05,
        maximum_shared_rate_difference=0.05,
        minimum_condition_support_per_role=2,
    )

    renamed_inputs = copy.deepcopy(discovery_inputs)
    renamed_inputs["component_membership"] = {
        component_id: ("renamed-one" if label == "raw-alpha" else "renamed-two")
        for component_id, label in MEMBERSHIP.items()
    }
    renamed = route_cluster_episodes_v1(**renamed_inputs)

    reordered_inputs = copy.deepcopy(discovery_inputs)
    reordered_inputs["component_membership"] = dict(
        reversed(list(reordered_inputs["component_membership"].items()))
    )
    reordered_inputs["component_effects"] = dict(
        reversed(list(reordered_inputs["component_effects"].items()))
    )
    reordered = route_cluster_episodes_v1(**reordered_inputs)

    condition_discovery = route_cluster_episodes_v1(
        **_condition_fixture("discovery", "cd", swapped=False)
    )
    condition_validation_stable = route_cluster_episodes_v1(
        **_condition_fixture("cluster_validation", "cv", swapped=False)
    )
    condition_validation_swapped = route_cluster_episodes_v1(
        **_condition_fixture("cluster_validation", "cs", swapped=True)
    )
    condition_stable = assess_routing_stability_v1(
        condition_discovery,
        condition_validation_stable,
        maximum_total_variation=0.05,
        maximum_condition_total_variation=0.05,
        maximum_shared_rate_difference=0.05,
        minimum_condition_support_per_role=4,
    )
    condition_swapped = assess_routing_stability_v1(
        condition_discovery,
        condition_validation_swapped,
        maximum_total_variation=0.05,
        maximum_condition_total_variation=0.05,
        maximum_shared_rate_difference=0.05,
        minimum_condition_support_per_role=4,
    )

    groups = _groups(MEMBERSHIP)
    route_a = _route_id_for_group(groups[0])
    route_b = _route_id_for_group(groups[1])
    rows = discovery["routing"]["rows"]
    route_episode_ids = {
        route: {
            row["episode_id"] for row in rows if row["primary_route"] == route
        }
        for route in (route_a, route_b, SHARED_ROUTE)
    }
    no_route_overlap = all(
        not route_episode_ids[left] & route_episode_ids[right]
        for index, left in enumerate(route_episode_ids)
        for right in list(route_episode_ids)[index + 1 :]
    )
    conflict_rows = [
        row for row in rows if row["route_reason"] == "pair_route_conflict"
    ]
    b_rows = [row for row in rows if row["primary_route"] == route_b]
    unequal_size_score_exact = bool(
        b_rows
        and all(abs(row["cluster_scores"][route_b] - 2.0) < 1e-12 for row in b_rows)
    )

    membership_tamper = copy.deepcopy(discovery_inputs)
    membership_tamper["category_partition_sha256"] = "0" * 64
    effect_tamper = copy.deepcopy(discovery_inputs)
    first_component = sorted(MEMBERSHIP)[0]
    effect_tamper["expected_component_effect_sha256"][first_component] = "0" * 64
    order_tamper = copy.deepcopy(discovery_inputs)
    order_tamper["capture_episode_order_sha256"] = "0" * 64
    row_tamper = copy.deepcopy(discovery_inputs)
    row_tamper["component_effects"][first_component] = row_tamper[
        "component_effects"
    ][first_component][:-1]
    nonfinite_tamper = copy.deepcopy(discovery_inputs)
    nonfinite_tamper["component_effects"][first_component][0, 0] = np.nan

    controls = {
        "known_two_cluster_routes_recovered": bool(
            discovery["routing"]["support_by_route"][route_a]["episodes"] == 4
            and discovery["routing"]["support_by_route"][route_b]["episodes"] == 4
        ),
        "label_renaming_invariant": renamed == discovery,
        "component_order_invariant": reordered == discovery,
        "unequal_cluster_size_normalized": unequal_size_score_exact,
        "ambiguous_and_low_response_use_shared": all(
            row["primary_route"] == SHARED_ROUTE
            for row in rows
            if row["route_reason"]
            in {"ambiguous_cluster_margin", "below_minimum_response"}
        ),
        "pair_conflict_is_atomic_shared": len(conflict_rows) == 2
        and all(row["primary_route"] == SHARED_ROUTE for row in conflict_rows),
        "primary_routes_disjoint_without_duplication": no_route_overlap
        and sum(len(value) for value in route_episode_ids.values()) == len(rows),
        "stable_discovery_validation_admitted": stable["admission"][
            "eligible_for_cluster_aware_sae_planning"
        ],
        "condition_stable_roles_admitted": condition_stable["admission"][
            "eligible_for_cluster_aware_sae_planning"
        ],
        "aggregate_preserving_condition_swap_rejected": (
            condition_swapped["evidence"]["overall_total_variation"] == 0.0
            and not condition_swapped["admission"][
                "eligible_for_cluster_aware_sae_planning"
            ]
        ),
        "partition_membership_tamper_rejected": _rejects(
            lambda: route_cluster_episodes_v1(**membership_tamper)
        ),
        "component_effect_tamper_rejected": _rejects(
            lambda: route_cluster_episodes_v1(**effect_tamper)
        ),
        "episode_order_tamper_rejected": _rejects(
            lambda: route_cluster_episodes_v1(**order_tamper)
        ),
        "row_count_mismatch_rejected": _rejects(
            lambda: route_cluster_episodes_v1(**row_tamper)
        ),
        "nonfinite_effect_rejected": _rejects(
            lambda: route_cluster_episodes_v1(**nonfinite_tamper)
        ),
        "inputs_byte_identical": before_episodes == discovery_inputs["episodes"]
        and all(
            before_effect_bytes[key]
            == discovery_inputs["component_effects"][key].tobytes(order="C")
            for key in before_effect_bytes
        ),
    }
    body = {
        "schema": "sch.vpd-cluster-episode-routing-benchmark.v1",
        "status": "pass" if all(controls.values()) else "fail",
        "source": {
            "runner_sha256": sha256_file(Path(__file__)),
            "router_sha256": sha256_file(
                Path(__file__).with_name("route_vpd_cluster_episodes_v1.py")
            ),
        },
        "design": {
            "model_free": True,
            "known_clusters": 2,
            "unequal_component_counts": [1, 3],
            "activation_rows_per_episode": ROWS_PER_EPISODE,
            "hidden_size": HIDDEN_SIZE,
            "discovery_and_independent_validation": True,
            "condition_stratified_stability_required": True,
            "pair_atomic_primary_routing": True,
        },
        "results": {
            "discovery_routing_id": discovery["cluster_episode_routing_id"],
            "discovery_verification": discovery_verification,
            "stable_routing_stability_id": stable["routing_stability_id"],
            "condition_swap_stability_id": condition_swapped["routing_stability_id"],
            "condition_swap_overall_total_variation": condition_swapped["evidence"][
                "overall_total_variation"
            ],
            "condition_swap_maximum_condition_total_variation": max(
                row["total_variation"]
                for row in condition_swapped["evidence"]["condition_strata"].values()
            ),
            "support_by_route": discovery["routing"]["support_by_route"],
            "pair_route_conflicts_sent_to_shared": discovery["routing"][
                "pair_route_conflicts_sent_to_shared"
            ],
        },
        "controls": controls,
        "summary": {
            "controls": len(controls),
            "passed": sum(bool(value) for value in controls.values()),
        },
        "scientific_boundary": {
            "synthetic_effect_arrays_only": True,
            "model_loaded": False,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "empirical_routing_receipt_produced": False,
            "sae_training_authorized": False,
            "execution_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("cluster episode routing benchmark address differs")
    if observed != build_report():
        raise ValueError("cluster episode routing benchmark is not reproducible")
    if observed.get("status") != "pass":
        raise ValueError("cluster episode routing benchmark did not pass")
    return {
        "schema": "sch.vpd-cluster-episode-routing-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "controls": observed["summary"]["controls"],
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark model-free VPD cluster-to-episode routing V1."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        result = build_report()
        write_json(args.output, result)
    else:
        result = verify_benchmark(args.verify)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
