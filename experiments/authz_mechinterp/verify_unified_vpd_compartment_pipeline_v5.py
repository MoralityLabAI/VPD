from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_focused_sae_compartment_controls import (
    verify_benchmark as verify_control_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v7 import (
    canonical_inputs as protocol_v7_inputs,
    verify_protocol as verify_protocol_v7,
)
from experiments.authz_mechinterp.plan_compartment_sae_training_release_gate import (
    verify_release_gate_plan,
)
from experiments.authz_mechinterp.verify_unified_vpd_compartment_pipeline_v4 import (
    canonical_inputs as receipt_v4_inputs,
    verify_receipt as verify_receipt_v4,
)


SCHEMA = "sch.vpd-unified-compartment-pipeline-receipt.v5"


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("V7 unified lineage source escapes the project")
    return resolved.relative_to(root.resolve()).as_posix()


def build_receipt(
    *,
    predecessor_receipt_path: Path,
    protocol_path: Path,
    release_gate_plan_path: Path,
    control_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = verify_receipt_v4(
        predecessor_receipt_path, **receipt_v4_inputs(project_root)
    )
    protocol_verification = verify_protocol_v7(
        protocol_path, **protocol_v7_inputs(project_root)
    )
    release_gate_verification = verify_release_gate_plan(
        release_gate_plan_path,
        training_plan_path=project_root
        / "artifacts/vpd_compartment_sae_training_plan_v3/plan.json",
        protocol_config_path=project_root
        / "configs/evals/vpd_activation_category_compartments_v6.json",
        partition_audit_gate_plan_path=project_root
        / "artifacts/vpd_partition_audit_gate_v2/plan.json",
        partition_audit_pack_dir=project_root
        / "artifacts/vpd_partition_audit_pack_v1",
    )
    benchmark_verification = verify_control_benchmark(control_benchmark_path)
    predecessor = json.loads(predecessor_receipt_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    release_gate = json.loads(release_gate_plan_path.read_text(encoding="utf-8"))
    control = protocol["focused_evaluation"]["matched_parameter_control"]
    if (
        protocol["predecessor_sha256"]
        != release_gate["source"]["protocol"]["sha256"]
    ):
        raise ValueError("focused-control protocol and SAE release gate lineage differ")
    if control["synthetic_benchmark"]["benchmark_id"] != benchmark_verification[
        "benchmark_id"
    ]:
        raise ValueError("focused-control protocol and benchmark differ")
    if predecessor["pipeline_receipt_id"] != predecessor_verification[
        "pipeline_receipt_id"
    ]:
        raise ValueError("unified lineage V4 identity differs")
    body = {
        "schema": SCHEMA,
        "status": "pass_local_category_control_path_implemented_not_run",
        "source": {
            "predecessor_unified_receipt_v4": {
                "path": _relative(predecessor_receipt_path, project_root),
                "sha256": sha256_file(predecessor_receipt_path),
                "pipeline_receipt_id": predecessor_verification[
                    "pipeline_receipt_id"
                ],
                "verification": predecessor_verification,
            },
            "research_protocol_v7": {
                "path": _relative(protocol_path, project_root),
                "sha256": sha256_file(protocol_path),
                "verification": protocol_verification,
            },
            "sae_training_release_gate_v1": {
                "path": _relative(release_gate_plan_path, project_root),
                "sha256": sha256_file(release_gate_plan_path),
                "release_gate_plan_id": release_gate_verification[
                    "release_gate_plan_id"
                ],
                "verification": release_gate_verification,
            },
            "focused_sae_compartment_control_benchmark": {
                "path": _relative(control_benchmark_path, project_root),
                "sha256": sha256_file(control_benchmark_path),
                "verification": benchmark_verification,
            },
            "focused_control_implementation": {
                "runner_v2": copy.deepcopy(control["runner_v2"]),
                "executor_v2": copy.deepcopy(control["executor_v2"]),
                "predecessor_runner": copy.deepcopy(control["predecessor_runner"]),
                "predecessor_executor": copy.deepcopy(
                    control["predecessor_executor"]
                ),
            },
        },
        "lineage_amendment": {
            "predecessor_receipt_v4_preserved": True,
            "research_protocol_v6_preserved": True,
            "research_protocol_v7_is_not_historical_experiment_v7": True,
            "sae_training_release_gate_still_required": True,
            "focused_category_packet_requires_v2_planner": True,
            "focused_category_plan_requires_v2_executor": True,
            "specific_controls_share_frozen_activation_compartment": True,
            "shared_controls_share_frozen_shared_compartment": True,
            "controls_are_outside_target_packet": True,
            "complete_assignment_derived_control_pool_required": True,
            "predecessor_packet_first_32_pool_not_used_for_selection": True,
            "generic_cross_compartment_control_pool_forbidden": True,
            "numeric_control_selection_recomputed_before_execution": True,
            "empty_same_compartment_pool_rejects_packet": True,
            "final_evaluation_remains_untouched": True,
        },
        "resources": {
            "guarded_sae_blueprint_jobs": release_gate_verification[
                "guarded_jobs"
            ],
            "focused_control_benchmark_scenarios": benchmark_verification[
                "scenarios"
            ],
            "focused_control_benchmark_rejections": benchmark_verification[
                "rejected_before_plan"
            ],
        },
        "invariants": {
            "predecessor_receipt_v4_reverified": True,
            "research_protocol_v7_reverified": True,
            "sae_training_release_gate_reverified": True,
            "focused_control_benchmark_reverified": True,
            "v7_protocol_predecessor_equals_release_gate_protocol": True,
            "category_bundle_assignments_checked_at_planning_and_execution": True,
            "complete_control_pool_rederived_from_bundle_assignments": True,
            "control_numeric_selection_recomputed_from_effects_and_dictionary": True,
            "historical_focused_runner_and_executor_unmodified": True,
            "no_cross_compartment_parameter_control": True,
            "execution_not_authorized": True,
        },
        "scientific_boundary": {
            "model_inference_performed": False,
            "sae_training_performed": False,
            "interventions_executed": False,
            "weights_loaded_or_mutated": False,
            "empirical_category_bundle_present": False,
            "focused_v2_plan_created": False,
            "execution_authorized": False,
            "historical_v7_negative_release_decision_unchanged": True,
        },
    }
    return {**body, "pipeline_receipt_id": content_sha256(body)}


def verify_receipt(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = copy.deepcopy(observed)
    receipt_id = str(body.pop("pipeline_receipt_id", ""))
    if content_sha256(body) != receipt_id:
        raise ValueError("V7 unified pipeline receipt content address mismatch")
    if build_receipt(**kwargs) != observed:
        raise ValueError("V7 unified pipeline receipt is not reproducible")
    return {
        "schema": "sch.vpd-unified-compartment-pipeline-verification.v5",
        "status": "pass",
        "pipeline_receipt_id": receipt_id,
        "focused_control_scenarios": observed["resources"][
            "focused_control_benchmark_scenarios"
        ],
        "guarded_sae_blueprint_jobs": observed["resources"][
            "guarded_sae_blueprint_jobs"
        ],
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_receipt_path": project_root
        / "artifacts/vpd_unified_compartment_pipeline_v4/receipt.json",
        "protocol_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v7.json",
        "release_gate_plan_path": project_root
        / "artifacts/vpd_compartment_sae_training_release_gate_v1/plan.json",
        "control_benchmark_path": project_root
        / "artifacts/vpd_focused_sae_compartment_control_benchmark_v1/report.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the research-protocol V7 lineage amendment."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None:
        result = verify_receipt(args.verify, **inputs)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_receipt(**inputs)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
