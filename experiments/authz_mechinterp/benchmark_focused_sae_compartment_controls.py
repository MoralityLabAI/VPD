from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.run_focused_vpd_sae_v2 import (
    CONTROL_POLICY,
    apply_category_matched_control_contract,
    validate_v2_plan_control_contract,
)


def _fixtures() -> dict[str, Any]:
    candidate_body = {
        "component_id": "target",
        "feature_index": 0,
        "matched_control_component_id": "cross-category-control",
        "matched_control_effect_norm_relative_gap": 0.0,
        "matched_control_absolute_cosine": 0.0,
    }
    candidate = {
        **candidate_body,
        "candidate_id": content_sha256(candidate_body),
    }
    plan_body = {
        "schema": "sch.vpd-focused-sae-intervention-plan.v1",
        "packet_id": "1" * 64,
        "cluster_id": "cluster-replayed",
        "worker_id": "worker-a",
        "study_inputs": {
            "activation_category": "replayed",
            "alignment_activation_key": "layer_35__proposal_start",
            "executor_sha256": "2" * 64,
        },
        "intervention_contract": {
            "parameter_effect_control": "subtract_matched_norm_component_slope",
        },
        "candidates": [candidate],
    }
    predecessor_plan = {**plan_body, "plan_id": content_sha256(plan_body)}
    packet = {
        "packet_id": "1" * 64,
        "cluster_id": "cluster-replayed",
        "component_ids": ["target"],
        "sae_study": {
            "matched_control_component_ids": ["cross-category-control"],
            "activation_compartment": {
                "category": "replayed",
                "category_matched_control_component_ids": [
                    "same-category-control"
                ],
            },
        },
    }
    return {
        "predecessor_plan": predecessor_plan,
        "packet": packet,
        "assignment_categories": {
            "target": "replayed",
            "same-category-control": "replayed",
            "same-category-control-2": "replayed",
            "cross-category-control": "valid_authority",
        },
        "effect_component_ids": [
            "target",
            "same-category-control",
            "same-category-control-2",
            "cross-category-control",
        ],
        "delta_h": np.asarray(
            [
                [1.0, 0.0],
                [0.0, 0.9],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        "decoder": np.asarray([[1.0, 0.0]], dtype=np.float64),
        "category_bundle_manifest_sha256": "3" * 64,
        "category_bundle_assignments_sha256": "4" * 64,
        "predecessor_runner_sha256": "5" * 64,
        "runner_v2_sha256": "6" * 64,
    }


def build_benchmark_report() -> dict[str, Any]:
    def empty_pool(inputs: dict[str, Any]) -> None:
        inputs["packet"]["sae_study"]["activation_compartment"][
            "category_matched_control_component_ids"
        ] = []
        inputs["assignment_categories"]["same-category-control"] = (
            "valid_authority"
        )
        inputs["assignment_categories"]["same-category-control-2"] = (
            "valid_authority"
        )

    def cross_category_pool(inputs: dict[str, Any]) -> None:
        inputs["packet"]["sae_study"]["activation_compartment"][
            "category_matched_control_component_ids"
        ] = ["cross-category-control"]

    def target_category_mismatch(inputs: dict[str, Any]) -> None:
        inputs["assignment_categories"]["target"] = "valid_authority"

    def overlapping_pool(inputs: dict[str, Any]) -> None:
        inputs["packet"]["sae_study"]["activation_compartment"][
            "category_matched_control_component_ids"
        ] = ["target"]

    def missing_control_effect(inputs: dict[str, Any]) -> None:
        inputs["effect_component_ids"].remove("same-category-control")
        inputs["delta_h"] = inputs["delta_h"][[0, 2, 3]]

    def tampered_predecessor(inputs: dict[str, Any]) -> None:
        inputs["predecessor_plan"]["worker_id"] = "worker-b"

    def shared_compartment(inputs: dict[str, Any]) -> None:
        inputs["packet"]["cluster_id"] = "cluster-shared"
        inputs["packet"]["sae_study"]["activation_compartment"][
            "category"
        ] = "shared_cross_category"
        inputs["predecessor_plan"]["cluster_id"] = "cluster-shared"
        inputs["predecessor_plan"]["study_inputs"][
            "activation_category"
        ] = "shared_cross_category"
        inputs["assignment_categories"]["target"] = "shared_cross_category"
        inputs["assignment_categories"][
            "same-category-control"
        ] = "shared_cross_category"
        inputs["assignment_categories"][
            "same-category-control-2"
        ] = "shared_cross_category"
        body = copy.deepcopy(inputs["predecessor_plan"])
        body.pop("plan_id", None)
        inputs["predecessor_plan"]["plan_id"] = content_sha256(body)

    scenarios: list[
        tuple[str, Callable[[dict[str, Any]], None] | None, str | None]
    ] = [
        ("truncated_packet_pool_expands_to_complete_specific_pool", None, None),
        ("shared_same_compartment_replacement", shared_compartment, None),
        (
            "empty_same_compartment_pool",
            empty_pool,
            "no complete out-of-packet same-compartment pool",
        ),
        (
            "cross_category_pool",
            cross_category_pool,
            "predecessor control pool crosses activation compartments",
        ),
        (
            "target_category_mismatch",
            target_category_mismatch,
            "packet membership crosses activation compartments",
        ),
        (
            "control_overlaps_target_packet",
            overlapping_pool,
            "predecessor controls overlap the target packet",
        ),
        (
            "missing_control_effect",
            missing_control_effect,
            "effect archive omits same-compartment controls",
        ),
        (
            "tampered_predecessor_plan",
            tampered_predecessor,
            "predecessor plan content address mismatch",
        ),
    ]
    results: list[dict[str, Any]] = []
    for name, mutation, expected_error in scenarios:
        inputs = copy.deepcopy(_fixtures())
        if mutation is not None:
            mutation(inputs)
        try:
            plan = apply_category_matched_control_contract(**inputs)
            verification = validate_v2_plan_control_contract(
                plan=plan,
                packet=inputs["packet"],
                assignment_categories=inputs["assignment_categories"],
            )
            if expected_error is not None:
                raise AssertionError(f"scenario unexpectedly passed: {name}")
            candidate = plan["candidates"][0]
            if candidate["matched_control_component_id"] != "same-category-control-2":
                raise AssertionError("V2 did not select from the complete control pool")
            if candidate["matched_control_policy"] != CONTROL_POLICY:
                raise AssertionError("V2 did not bind the control policy")
            results.append(
                {
                    "name": name,
                    "status": "pass",
                    "observed_stage": "same_compartment_plan",
                    "activation_category": verification["activation_category"],
                    "selected_control": candidate[
                        "matched_control_component_id"
                    ],
                    "complete_control_pool_size": plan[
                        "control_compartment_contract"
                    ]["control_pool_size"],
                    "predecessor_packet_pool_size": plan[
                        "control_compartment_contract"
                    ]["predecessor_packet_control_pool_size"],
                    "plan_id": plan["plan_id"],
                }
            )
        except ValueError as exc:
            if expected_error is None or expected_error not in str(exc):
                raise
            results.append(
                {
                    "name": name,
                    "status": "pass",
                    "observed_stage": "pre_plan_rejection",
                    "error": str(exc),
                }
            )
    project_root = Path(__file__).resolve().parents[2]
    body = {
        "schema": "sch.vpd-focused-sae-compartment-control-benchmark.v1",
        "status": "pass",
        "source": {
            "predecessor_runner": {
                "path": "experiments/authz_mechinterp/run_focused_vpd_sae.py",
                "sha256": sha256_file(
                    project_root
                    / "experiments/authz_mechinterp/run_focused_vpd_sae.py"
                ),
                "preserved_unmodified": True,
            },
            "runner_v2": {
                "path": "experiments/authz_mechinterp/run_focused_vpd_sae_v2.py",
                "sha256": sha256_file(
                    project_root
                    / "experiments/authz_mechinterp/run_focused_vpd_sae_v2.py"
                ),
            },
            "executor_v2": {
                "path": "experiments/authz_mechinterp/compute_focused_sae_interventions_v2.py",
                "sha256": sha256_file(
                    project_root
                    / "experiments/authz_mechinterp/compute_focused_sae_interventions_v2.py"
                ),
            },
            "benchmark_runner_sha256": sha256_file(Path(__file__)),
        },
        "scenarios": results,
        "summary": {
            "scenarios": len(results),
            "passed": sum(row["status"] == "pass" for row in results),
            "same_compartment_plans": sum(
                row["observed_stage"] == "same_compartment_plan"
                for row in results
            ),
            "rejected_before_plan": sum(
                row["observed_stage"] == "pre_plan_rejection"
                for row in results
            ),
        },
        "scientific_boundary": {
            "synthetic_only": True,
            "model_inference_performed": False,
            "interventions_executed": False,
            "sae_training_performed": False,
            "weights_loaded_or_mutated": False,
            "execution_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = copy.deepcopy(observed)
    benchmark_id = str(body.pop("benchmark_id", ""))
    if content_sha256(body) != benchmark_id:
        raise ValueError("focused SAE compartment-control benchmark content address mismatch")
    expected = build_benchmark_report()
    if observed != expected:
        raise ValueError("focused SAE compartment-control benchmark is not reproducible")
    return {
        "schema": "sch.vpd-focused-sae-compartment-control-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": benchmark_id,
        **observed["summary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark same-compartment VPD controls for focused SAE studies."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify_benchmark(args.output)
    else:
        result = build_benchmark_report()
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
