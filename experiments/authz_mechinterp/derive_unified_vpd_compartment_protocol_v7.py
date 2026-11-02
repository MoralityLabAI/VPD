from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_focused_sae_compartment_controls import (
    verify_benchmark as verify_control_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v6 import (
    canonical_inputs as v6_canonical_inputs,
    verify_protocol as verify_v6_protocol,
)


SCHEMA = "sch.vpd-activation-category-protocol.v7"


def build_protocol(
    *,
    predecessor_path: Path,
    control_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = verify_v6_protocol(
        predecessor_path, **v6_canonical_inputs(project_root)
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if predecessor.get("schema") != "sch.vpd-activation-category-protocol.v6":
        raise ValueError("focused-control protocol predecessor is not V6")
    benchmark_verification = verify_control_benchmark(control_benchmark_path)
    runner_v2 = (
        project_root / "experiments/authz_mechinterp/run_focused_vpd_sae_v2.py"
    )
    executor_v2 = (
        project_root
        / "experiments/authz_mechinterp/compute_focused_sae_interventions_v2.py"
    )
    predecessor_runner = (
        project_root / "experiments/authz_mechinterp/run_focused_vpd_sae.py"
    )
    predecessor_executor = (
        project_root
        / "experiments/authz_mechinterp/compute_focused_sae_interventions.py"
    )
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = predecessor_path.resolve().relative_to(
        project_root.resolve()
    ).as_posix()
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["focused_evaluation"]["matched_parameter_control"] = {
        "schema": "sch.vpd-focused-sae-control-compartment.v1",
        "status": "implemented_model_free_not_run",
        "policy": "complete_same_frozen_activation_compartment_out_of_packet_norm_matched_low_alignment",
        "specific_target_and_control_same_compartment_required": True,
        "shared_target_and_control_same_compartment_required": True,
        "control_outside_target_packet_required": True,
        "complete_assignment_derived_control_pool_required": True,
        "predecessor_packet_first_32_pool_used_for_selection": False,
        "generic_cross_compartment_control_pool_forbidden": True,
        "empty_same_compartment_control_pool_rule": "reject_focused_candidate_packet",
        "bundle_assignments_reverified_before_planning_and_execution": True,
        "numeric_control_selection_recomputed_before_execution": True,
        "predecessor_runner": {
            "path": predecessor_runner.resolve().relative_to(
                project_root.resolve()
            ).as_posix(),
            "sha256": sha256_file(predecessor_runner),
            "preserved_unmodified": True,
            "status": "historical_generic_control_predecessor",
        },
        "predecessor_executor": {
            "path": predecessor_executor.resolve().relative_to(
                project_root.resolve()
            ).as_posix(),
            "sha256": sha256_file(predecessor_executor),
            "preserved_unmodified": True,
            "direct_category_plan_execution_allowed": False,
        },
        "runner_v2": {
            "path": runner_v2.resolve().relative_to(
                project_root.resolve()
            ).as_posix(),
            "sha256": sha256_file(runner_v2),
        },
        "executor_v2": {
            "path": executor_v2.resolve().relative_to(
                project_root.resolve()
            ).as_posix(),
            "sha256": sha256_file(executor_v2),
        },
        "synthetic_benchmark": {
            "path": control_benchmark_path.resolve().relative_to(
                project_root.resolve()
            ).as_posix(),
            "sha256": sha256_file(control_benchmark_path),
            "benchmark_id": benchmark_verification["benchmark_id"],
            "scenarios": benchmark_verification["scenarios"],
            "status": benchmark_verification["status"],
        },
        "execution_authorized": False,
    }
    result["focused_evaluation"]["matched_parameter_control_required"] = True
    result["unified_lineage"][
        "focused_sae_same_compartment_control_required"
    ] = True
    result["unified_lineage"][
        "focused_sae_numeric_control_selection_recomputed"
    ] = True
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        "predecessor_verification": predecessor_verification,
        "focused_sae_compartment_control_benchmark_verification": (
            benchmark_verification
        ),
        "implementation_hashes": {
            "runner_v2": sha256_file(runner_v2),
            "executor_v2": sha256_file(executor_v2),
            "predecessor_runner": sha256_file(predecessor_runner),
            "predecessor_executor": sha256_file(predecessor_executor),
        },
        "research_protocol_v7_is_distinct_from_historical_experiment_v7": True,
    }
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["sae_training_performed"] = False
    result["scientific_boundary"]["weights_loaded_or_mutated"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = build_protocol(**kwargs)
    if observed != expected:
        raise ValueError("V7 focused-control research protocol is not reproducible")
    control = observed["focused_evaluation"]["matched_parameter_control"]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v7",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "predecessor_sha256": sha256_file(kwargs["predecessor_path"]),
        "control_benchmark_id": control["synthetic_benchmark"]["benchmark_id"],
        "same_compartment_control_required": True,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v6.json",
        "control_benchmark_path": project_root
        / "artifacts/vpd_focused_sae_compartment_control_benchmark_v1/report.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive or verify focused-control research protocol V7."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None:
        result = verify_protocol(args.verify, **inputs)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_protocol(**inputs)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
