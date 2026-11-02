from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.audit_frozen_vpd_category_partition import (
    evaluate_frozen_partition,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    write_json,
)


SCHEMA = "sch.vpd-frozen-category-partition-audit-benchmark.v1"
THRESHOLDS = {
    "minimum_dominant_share": 0.35,
    "minimum_dominance_margin": 0.10,
    "maximum_normalized_entropy": 0.80,
    "zero_total_effect_routes_to_shared": True,
}


def _base_inputs() -> dict[str, Any]:
    component_ids = [f"synthetic-component-{index}" for index in range(6)]
    category_names = ["category_a", "category_b", "category_c"]
    discovery = np.zeros((6, 3, 3), dtype=np.float64)
    audit = np.zeros_like(discovery)
    for index in range(5):
        discovery[index, 0] = [10.0, 0.0, 0.0]
        discovery[index, 1] = [0.5, 0.0, 0.0]
        discovery[index, 2] = [0.25, 0.0, 0.0]
        audit[index] = discovery[index]
    discovery[5, 2] = [0.0, 10.0, 0.0]
    audit[5] = discovery[5]
    return {
        "component_ids": component_ids,
        "frozen_categories": [
            "category_a",
            "category_a",
            "category_a",
            "category_a",
            "category_a",
            "shared_cross_category",
        ],
        "category_names": category_names,
        "discovery_effects": discovery,
        "audit_effects": audit,
        "robustness_eligibility": np.ones((6, 3), dtype=bool),
        "specificity_thresholds": THRESHOLDS,
        "source": {"kind": "synthetic_model_free_benchmark"},
    }


def build_benchmark() -> dict[str, Any]:
    passing_inputs = _base_inputs()
    passing_inputs["frozen_categories"] = [
        "category_a",
        "shared_cross_category",
        "shared_cross_category",
        "shared_cross_category",
        "shared_cross_category",
        "shared_cross_category",
    ]
    passing = evaluate_frozen_partition(**passing_inputs)

    adversarial_inputs = _base_inputs()
    audit = adversarial_inputs["audit_effects"]
    audit[1, 0] = [0.25, 0.0, 0.0]
    audit[1, 1] = [10.0, 0.0, 0.0]
    audit[2, 0] = [-10.0, 0.0, 0.0]
    audit[3, 0] = [2.0, 0.0, 0.0]
    audit[3, 1] = [0.05, 0.0, 0.0]
    audit[3, 2] = [0.05, 0.0, 0.0]
    adversarial_inputs["robustness_eligibility"][4, 0] = False
    adversarial = evaluate_frozen_partition(**adversarial_inputs)
    observed_reasons = {
        row["component_id"]: row["rejection_reason"]
        for row in adversarial["decisions"]
        if row["specific_assignment_audited"]
    }
    expected_reasons = {
        "synthetic-component-0": None,
        "synthetic-component-1": "audit_category_mismatch",
        "synthetic-component-2": "audit_direction_instability",
        "synthetic-component-3": "audit_magnitude_instability",
        "synthetic-component-4": "audit_stratified_robustness_failure",
    }
    if observed_reasons != expected_reasons:
        raise ValueError("partition-audit benchmark failure routing differs")
    shared_decision = adversarial["decisions"][5]
    if (
        shared_decision["audit_point_estimate_category"] != "category_c"
        or shared_decision["specific_assignment_audited"] is not False
        or shared_decision["shared_to_specific_promotion_allowed"] is not False
    ):
        raise ValueError("partition-audit benchmark promoted a shared component")
    if passing["status"] != "pass_partition_audit":
        raise ValueError("partition-audit passing control did not pass")
    if adversarial["status"] != "reject_partition_audit":
        raise ValueError("partition-audit adversarial control did not reject")

    body = {
        "schema": SCHEMA,
        "status": "pass",
        "scenarios": {
            "all_frozen_specific_assignments_replicate": {
                "partition_audit_id": passing["partition_audit_id"],
                "decision": passing["status"],
                "summary": passing["summary"],
            },
            "orthogonal_failure_modes_and_shared_nonpromotion": {
                "partition_audit_id": adversarial["partition_audit_id"],
                "decision": adversarial["status"],
                "summary": adversarial["summary"],
                "expected_rejection_reasons": expected_reasons,
                "shared_audit_point_estimate": shared_decision[
                    "audit_point_estimate_category"
                ],
                "shared_promoted": False,
            },
        },
        "invariants": {
            "post_freeze_only": True,
            "zero_specific_failures_required": True,
            "category_mismatch_detected": True,
            "direction_failure_detected": True,
            "magnitude_failure_detected": True,
            "stratified_robustness_failure_detected": True,
            "shared_to_specific_promotion_forbidden": True,
            "model_inference_performed": False,
            "execution_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = copy.deepcopy(observed)
    benchmark_id = str(body.pop("benchmark_id", ""))
    if content_sha256(body) != benchmark_id:
        raise ValueError("partition-audit benchmark content address mismatch")
    if build_benchmark() != observed:
        raise ValueError("partition-audit benchmark is not reproducible")
    return {
        "schema": "sch.vpd-frozen-category-partition-audit-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": benchmark_id,
        "scenarios": len(observed["scenarios"]),
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the model-free frozen-partition audit benchmark."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_benchmark(args.verify)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_benchmark()
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
