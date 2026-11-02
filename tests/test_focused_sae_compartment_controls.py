from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_focused_sae_compartment_controls import (
    _fixtures,
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256
from experiments.authz_mechinterp.run_focused_vpd_sae_v2 import (
    CONTROL_POLICY,
    apply_category_matched_control_contract,
    recompute_v2_candidate_controls,
    validate_v2_plan_control_contract,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = (
    ROOT / "artifacts/vpd_focused_sae_compartment_control_benchmark_v1/report.json"
)


def test_v2_replaces_generic_cross_category_control() -> None:
    inputs = _fixtures()
    plan = apply_category_matched_control_contract(**inputs)
    candidate = plan["candidates"][0]
    assert candidate["matched_control_component_id"] == "same-category-control-2"
    assert candidate["matched_control_activation_category"] == "replayed"
    assert candidate["matched_control_policy"] == CONTROL_POLICY
    assert plan["control_compartment_contract"][
        "generic_cross_compartment_pool_used"
    ] is False
    verification = validate_v2_plan_control_contract(
        plan=plan,
        packet=inputs["packet"],
        assignment_categories=inputs["assignment_categories"],
    )
    assert verification["status"] == "pass"
    assert verification["control_pool_size"] == 2


def test_empty_truncated_packet_pool_does_not_hide_complete_assignment_pool() -> None:
    inputs = _fixtures()
    inputs["packet"]["sae_study"]["activation_compartment"][
        "category_matched_control_component_ids"
    ] = []
    plan = apply_category_matched_control_contract(**inputs)
    assert plan["control_compartment_contract"]["control_pool_size"] == 2
    assert plan["control_compartment_contract"][
        "predecessor_packet_control_pool_size"
    ] == 0


def test_truly_empty_assignment_derived_pool_rejects() -> None:
    inputs = _fixtures()
    inputs["packet"]["sae_study"]["activation_compartment"][
        "category_matched_control_component_ids"
    ] = []
    inputs["assignment_categories"]["same-category-control"] = "valid_authority"
    inputs["assignment_categories"]["same-category-control-2"] = "valid_authority"
    with pytest.raises(ValueError, match="no complete out-of-packet"):
        apply_category_matched_control_contract(**inputs)


def test_cross_category_control_pool_rejects() -> None:
    inputs = _fixtures()
    inputs["packet"]["sae_study"]["activation_compartment"][
        "category_matched_control_component_ids"
    ] = ["cross-category-control"]
    with pytest.raises(ValueError, match="predecessor control pool crosses"):
        apply_category_matched_control_contract(**inputs)


def test_candidate_control_substitution_breaks_plan_address() -> None:
    inputs = _fixtures()
    plan = apply_category_matched_control_contract(**inputs)
    tampered = copy.deepcopy(plan)
    tampered["candidates"][0]["matched_control_component_id"] = (
        "cross-category-control"
    )
    with pytest.raises(ValueError, match="plan content address mismatch"):
        validate_v2_plan_control_contract(
            plan=tampered,
            packet=inputs["packet"],
            assignment_categories=inputs["assignment_categories"],
        )


def test_readdressed_candidate_still_cannot_cross_compartments() -> None:
    inputs = _fixtures()
    plan = apply_category_matched_control_contract(**inputs)
    tampered = copy.deepcopy(plan)
    candidate = tampered["candidates"][0]
    candidate.pop("candidate_id")
    candidate["matched_control_component_id"] = "cross-category-control"
    candidate["candidate_id"] = content_sha256(candidate)
    body = copy.deepcopy(tampered)
    body.pop("plan_id")
    tampered["plan_id"] = content_sha256(body)
    with pytest.raises(ValueError, match="candidate control compartment differs"):
        validate_v2_plan_control_contract(
            plan=tampered,
            packet=inputs["packet"],
            assignment_categories=inputs["assignment_categories"],
        )


def test_readdressed_control_metrics_are_independently_recomputed() -> None:
    inputs = _fixtures()
    plan = apply_category_matched_control_contract(**inputs)
    tampered = copy.deepcopy(plan)
    candidate = tampered["candidates"][0]
    candidate.pop("candidate_id")
    candidate["matched_control_effect_norm_relative_gap"] = 0.5
    candidate["candidate_id"] = content_sha256(candidate)
    body = copy.deepcopy(tampered)
    body.pop("plan_id")
    tampered["plan_id"] = content_sha256(body)
    validate_v2_plan_control_contract(
        plan=tampered,
        packet=inputs["packet"],
        assignment_categories=inputs["assignment_categories"],
    )
    with pytest.raises(ValueError, match="control selection is not reproducible"):
        recompute_v2_candidate_controls(
            plan=tampered,
            packet=inputs["packet"],
            effect_component_ids=inputs["effect_component_ids"],
            delta_h=inputs["delta_h"],
            decoder=inputs["decoder"],
        )


def test_compartment_control_benchmark_is_reproducible() -> None:
    assert verify_benchmark(BENCHMARK) == {
        "schema": "sch.vpd-focused-sae-compartment-control-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": "57f3f9a612883b04bb9651ca8de3fee7f17a6b0996b7eb88ee7d85d22a84ed25",
        "scenarios": 8,
        "passed": 8,
        "same_compartment_plans": 2,
        "rejected_before_plan": 6,
    }
