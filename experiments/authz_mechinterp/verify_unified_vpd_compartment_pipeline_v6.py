from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_compartment_sae_training_execution_contract import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_compartment_sae_training_execution_blueprint_v2 import (
    validate_blueprint,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v8 import (
    canonical_inputs as protocol_v8_inputs,
    verify_protocol as verify_protocol_v8,
)
from experiments.authz_mechinterp.verify_unified_vpd_compartment_pipeline_v5 import (
    canonical_inputs as receipt_v5_inputs,
    verify_receipt as verify_receipt_v5,
)


SCHEMA = "sch.vpd-unified-compartment-pipeline-receipt.v6"


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("V8 unified lineage source escapes the project")
    return resolved.relative_to(root.resolve()).as_posix()


def build_receipt(
    *,
    predecessor_receipt_path: Path,
    protocol_path: Path,
    training_plan_path: Path,
    execution_blueprint_path: Path,
    execution_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = verify_receipt_v5(
        predecessor_receipt_path, **receipt_v5_inputs(project_root)
    )
    protocol_verification = verify_protocol_v8(
        protocol_path, **protocol_v8_inputs(project_root)
    )
    blueprint = json.loads(execution_blueprint_path.read_text(encoding="utf-8"))
    validate_blueprint(
        blueprint,
        training_plan_path=training_plan_path,
        project_root=project_root,
    )
    benchmark_verification = verify_benchmark(
        execution_benchmark_path,
        training_plan_path=training_plan_path,
        project_root=project_root,
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    admission = protocol["sae_training"]["execution_blueprint_v2"]
    if (
        admission["blueprint"]["execution_blueprint_id"]
        != blueprint["execution_blueprint_id"]
        or admission["synthetic_benchmark"]["benchmark_id"]
        != benchmark_verification["benchmark_id"]
    ):
        raise ValueError("research protocol V8 training execution lineage differs")
    body = {
        "schema": SCHEMA,
        "status": "pass_local_training_execution_design_implemented_not_run",
        "source": {
            "predecessor_unified_receipt_v5": {
                "path": _relative(predecessor_receipt_path, project_root),
                "sha256": sha256_file(predecessor_receipt_path),
                "pipeline_receipt_id": predecessor_verification[
                    "pipeline_receipt_id"
                ],
                "verification": predecessor_verification,
            },
            "research_protocol_v8": {
                "path": _relative(protocol_path, project_root),
                "sha256": sha256_file(protocol_path),
                "verification": protocol_verification,
            },
            "training_execution_blueprint_v2": {
                "path": _relative(execution_blueprint_path, project_root),
                "sha256": sha256_file(execution_blueprint_path),
                "execution_blueprint_id": blueprint["execution_blueprint_id"],
            },
            "training_execution_contract_benchmark_v1": {
                "path": _relative(execution_benchmark_path, project_root),
                "sha256": sha256_file(execution_benchmark_path),
                "verification": benchmark_verification,
            },
            "preserved_training_plan_v3": {
                "path": _relative(training_plan_path, project_root),
                "sha256": sha256_file(training_plan_path),
                "training_plan_id": blueprint["source"]["training_plan_id"],
            },
        },
        "lineage_amendment": {
            "predecessor_receipt_v5_preserved_and_reverified": True,
            "research_protocol_v7_preserved": True,
            "research_protocol_v8_is_not_historical_experiment_v8": True,
            "training_plan_v3_preserved_as_predecessor_design": True,
            "all_18_replica_sample_contracts_recomputed_from_frozen_split": True,
            "specific_training_cross_compartment_rows_equal_zero": True,
            "shared_training_equal_whole_episode_balance_required": True,
            "batch_size_and_learning_rate_manifest_binding_required": True,
            "predecessor_v1_dictionary_manifest_empirically_inadmissible": True,
            "current_v1_release_gate_not_sufficient_by_itself": True,
            "versioned_empirical_runner_and_admission_still_required": True,
            "final_evaluation_remains_untouched": True,
        },
        "resources": {
            "training_jobs": blueprint["coverage"]["training_jobs"],
            "compartments": blueprint["coverage"]["compartments"],
            "specific_compartments": blueprint["coverage"][
                "specific_compartments"
            ],
            "shared_compartments": blueprint["coverage"]["shared_compartments"],
            "execution_contract_benchmark_scenarios": benchmark_verification[
                "scenarios"
            ],
        },
        "invariants": {
            "predecessor_receipt_v5_reverified": True,
            "research_protocol_v8_reverified": True,
            "execution_blueprint_v2_recomputed": True,
            "execution_contract_benchmark_reverified": True,
            "v1_unbound_hyperparameter_gap_reproduced": True,
            "exact_category_pair_and_prompt_window_hashes_bound": True,
            "execution_not_authorized": True,
        },
        "scientific_boundary": {
            "model_inference_performed": False,
            "sae_training_performed": False,
            "interventions_executed": False,
            "weights_loaded_or_mutated": False,
            "empirical_category_bundle_present": False,
            "empirical_dictionary_manifest_v2_present": False,
            "execution_authorized": False,
            "historical_v7_negative_release_decision_unchanged": True,
        },
    }
    return {**body, "pipeline_receipt_id": content_sha256(body)}


def verify_receipt(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("pipeline_receipt_id", ""))
    if content_sha256(body) != identifier:
        raise ValueError("V8 unified pipeline receipt content address mismatch")
    expected = build_receipt(**kwargs)
    if observed != expected:
        raise ValueError("V8 unified pipeline receipt is not reproducible")
    return {
        "schema": "sch.vpd-unified-compartment-pipeline-verification.v6",
        "status": "pass",
        "pipeline_receipt_id": identifier,
        "training_jobs": observed["resources"]["training_jobs"],
        "execution_contract_scenarios": observed["resources"][
            "execution_contract_benchmark_scenarios"
        ],
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_receipt_path": project_root
        / "artifacts/vpd_unified_compartment_pipeline_v5/receipt.json",
        "protocol_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v8.json",
        "training_plan_path": project_root
        / "artifacts/vpd_compartment_sae_training_plan_v3/plan.json",
        "execution_blueprint_path": project_root
        / "artifacts/vpd_compartment_sae_training_execution_blueprint_v2/blueprint.json",
        "execution_benchmark_path": project_root
        / "artifacts/vpd_compartment_sae_training_execution_contract_benchmark_v1/report.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the research-protocol V8 lineage amendment."
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
