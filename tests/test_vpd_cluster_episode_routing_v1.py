from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_vpd_cluster_episode_routing_v1 import (
    _condition_fixture,
    _core_fixture,
    build_report,
    verify_benchmark,
)
from experiments.authz_mechinterp.route_vpd_cluster_episodes_v1 import (
    SHARED_ROUTE,
    assess_routing_stability_v1,
    route_cluster_episodes_v1,
    verify_routing_receipt_v1,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_cluster_episode_routing_benchmark_v1/report.json"


def test_routing_benchmark_reproduces() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert observed == build_report()
    assert observed["summary"] == {"controls": 16, "passed": 16}
    assert verify_benchmark(REPORT)["status"] == "pass"


def test_routes_are_pair_atomic_disjoint_and_content_addressed() -> None:
    receipt = route_cluster_episodes_v1(**_core_fixture("discovery", "test"))
    verification = verify_routing_receipt_v1(receipt)
    assert verification["episodes"] == 14
    rows = receipt["routing"]["rows"]
    routes_by_pair: dict[str, set[str]] = {}
    for row in rows:
        routes_by_pair.setdefault(row["pair_id"], set()).add(row["primary_route"])
    assert all(len(routes) == 1 for routes in routes_by_pair.values())
    conflict = [row for row in rows if row["route_reason"] == "pair_route_conflict"]
    assert len(conflict) == 2
    assert all(row["primary_route"] == SHARED_ROUTE for row in conflict)

    tampered = copy.deepcopy(receipt)
    tampered["routing"]["rows"][0]["primary_route"] = SHARED_ROUTE
    with pytest.raises(ValueError, match="address differs"):
        verify_routing_receipt_v1(tampered)

    internally_inconsistent = copy.deepcopy(receipt)
    internally_inconsistent["routing"]["support_by_route"][SHARED_ROUTE][
        "episodes"
    ] += 1
    body = dict(internally_inconsistent)
    body.pop("cluster_episode_routing_id")
    from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256

    internally_inconsistent["cluster_episode_routing_id"] = content_sha256(body)
    with pytest.raises(ValueError, match="derived support differs"):
        verify_routing_receipt_v1(internally_inconsistent)


def test_cluster_label_renaming_does_not_change_receipt() -> None:
    inputs = _core_fixture("discovery", "rename")
    original = route_cluster_episodes_v1(**inputs)
    renamed_inputs = copy.deepcopy(inputs)
    renamed_inputs["component_membership"] = {
        component_id: "x" if label == "raw-alpha" else "y"
        for component_id, label in inputs["component_membership"].items()
    }
    assert route_cluster_episodes_v1(**renamed_inputs) == original


def test_condition_strata_reject_hidden_routing_swap() -> None:
    discovery = route_cluster_episodes_v1(
        **_condition_fixture("discovery", "condition-d", swapped=False)
    )
    validation = route_cluster_episodes_v1(
        **_condition_fixture("cluster_validation", "condition-v", swapped=True)
    )
    stability = assess_routing_stability_v1(
        discovery,
        validation,
        maximum_total_variation=0.05,
        maximum_condition_total_variation=0.05,
        maximum_shared_rate_difference=0.05,
        minimum_condition_support_per_role=4,
    )
    assert stability["evidence"]["overall_total_variation"] == 0.0
    assert not stability["admission"]["eligible_for_cluster_aware_sae_planning"]
    assert all(
        row["total_variation"] == 1.0
        for row in stability["evidence"]["condition_strata"].values()
    )


def test_router_rejects_capture_order_and_effect_hash_mismatch() -> None:
    order_inputs = _core_fixture("discovery", "bad-order")
    order_inputs["capture_episode_order_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="episode order differs"):
        route_cluster_episodes_v1(**order_inputs)

    effect_inputs = _core_fixture("discovery", "bad-effect")
    component_id = sorted(effect_inputs["component_effects"])[0]
    effect_inputs["expected_component_effect_sha256"][component_id] = "0" * 64
    with pytest.raises(ValueError, match="effect hash differs"):
        route_cluster_episodes_v1(**effect_inputs)


def test_router_accepts_separate_sae_training_role_without_granting_training() -> None:
    inputs = _core_fixture("discovery", "train-role")
    inputs["split_role"] = "sae_training"
    inputs["split_id"] = "synthetic-sae-training"
    receipt = route_cluster_episodes_v1(**inputs)
    assert receipt["source"]["split_role"] == "sae_training"
    assert receipt["boundary"]["sae_training_authorized"] is False
