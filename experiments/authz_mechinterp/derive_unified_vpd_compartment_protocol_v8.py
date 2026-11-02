from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_compartment_sae_training_execution_contract import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_compartment_sae_training_execution_blueprint_v2 import (
    validate_blueprint,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v7 import (
    canonical_inputs as v7_canonical_inputs,
    verify_protocol as verify_v7_protocol,
)


SCHEMA = "sch.vpd-activation-category-protocol.v8"


def _relative(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(project_root.resolve()):
        raise ValueError("V8 training execution source escapes the project")
    return resolved.relative_to(project_root.resolve()).as_posix()


def build_protocol(
    *,
    predecessor_path: Path,
    training_plan_path: Path,
    execution_blueprint_path: Path,
    execution_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = verify_v7_protocol(
        predecessor_path, **v7_canonical_inputs(project_root)
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if predecessor.get("schema") != "sch.vpd-activation-category-protocol.v7":
        raise ValueError("training execution protocol predecessor is not V7")
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
    derivation_runner = project_root / (
        "experiments/authz_mechinterp/"
        "derive_compartment_sae_training_execution_blueprint_v2.py"
    )
    benchmark_runner = project_root / (
        "experiments/authz_mechinterp/"
        "benchmark_compartment_sae_training_execution_contract.py"
    )
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["sae_training"]["execution_blueprint_v2"] = {
        "schema": "sch.vpd-compartment-sae-training-execution-admission.v2",
        "status": "design_only_implemented_model_free_not_run",
        "training_plan_v3": {
            "path": _relative(training_plan_path, project_root),
            "sha256": sha256_file(training_plan_path),
            "training_plan_id": blueprint["source"]["training_plan_id"],
            "preserved_as_predecessor_evidence": True,
        },
        "blueprint": {
            "path": _relative(execution_blueprint_path, project_root),
            "sha256": sha256_file(execution_blueprint_path),
            "execution_blueprint_id": blueprint["execution_blueprint_id"],
            "training_jobs": blueprint["coverage"]["training_jobs"],
            "compartments": blueprint["coverage"]["compartments"],
        },
        "synthetic_benchmark": {
            "path": _relative(execution_benchmark_path, project_root),
            "sha256": sha256_file(execution_benchmark_path),
            **benchmark_verification,
        },
        "implementation": {
            "blueprint_derivation": {
                "path": _relative(derivation_runner, project_root),
                "sha256": sha256_file(derivation_runner),
            },
            "synthetic_benchmark": {
                "path": _relative(benchmark_runner, project_root),
                "sha256": sha256_file(benchmark_runner),
            },
        },
        "required_dictionary_manifest_schema": (
            "sch.vpd-fixed-sae-dictionary-manifest.v2"
        ),
        "batch_size_and_learning_rate_manifest_binding_required": True,
        "exact_sample_contract_manifest_binding_required": True,
        "specific_cross_compartment_rows_must_equal": 0,
        "shared_equal_whole_episode_balance_required": True,
        "predecessor_v1_dictionary_manifest_admissible": False,
        "current_v1_training_release_gate_sufficient_by_itself": False,
        "versioned_empirical_runner_and_admission_still_required": True,
        "execution_authorized": False,
    }
    result["unified_lineage"][
        "sae_training_execution_blueprint_v2_required"
    ] = True
    result["unified_lineage"][
        "sae_training_v1_manifest_forbidden_for_empirical_admission"
    ] = True
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        "predecessor_verification": predecessor_verification,
        "training_execution_blueprint_id": blueprint["execution_blueprint_id"],
        "training_execution_benchmark_verification": benchmark_verification,
        "implementation_hashes": {
            "blueprint_derivation": sha256_file(derivation_runner),
            "synthetic_benchmark": sha256_file(benchmark_runner),
        },
        "research_protocol_v8_is_distinct_from_historical_experiment_v8": True,
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
        raise ValueError("V8 training execution research protocol is not reproducible")
    admission = observed["sae_training"]["execution_blueprint_v2"]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v8",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "predecessor_sha256": sha256_file(kwargs["predecessor_path"]),
        "execution_blueprint_id": admission["blueprint"][
            "execution_blueprint_id"
        ],
        "execution_benchmark_id": admission["synthetic_benchmark"][
            "benchmark_id"
        ],
        "training_jobs": admission["blueprint"]["training_jobs"],
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v7.json",
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
        description="Derive or verify training-execution research protocol V8."
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
