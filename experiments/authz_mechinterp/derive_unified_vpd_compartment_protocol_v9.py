from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_compartment_sae_training_plan_v4 import (
    canonical_inputs as plan_v4_inputs,
    verify_training_plan_v4,
)
from experiments.authz_mechinterp.derive_compartment_sae_training_release_gate_v2 import (
    canonical_inputs as gate_v2_inputs,
    verify_release_gate_v2,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v8 import (
    canonical_inputs as protocol_v8_inputs,
    verify_protocol as verify_protocol_v8,
)
from experiments.authz_mechinterp.benchmark_compartment_sae_runtime_v2 import (
    canonical_inputs as runtime_benchmark_inputs,
    verify_benchmark as verify_runtime_benchmark_v2,
)


SCHEMA = "sch.vpd-activation-category-protocol.v9"


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("research protocol V9 source escapes the project")
    return resolved.relative_to(root.resolve()).as_posix()


def build_protocol(
    *,
    predecessor_path: Path,
    training_plan_path: Path,
    release_gate_path: Path,
    runtime_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = verify_protocol_v8(
        predecessor_path, **protocol_v8_inputs(project_root)
    )
    plan_verification = verify_training_plan_v4(
        training_plan_path, **plan_v4_inputs(project_root)
    )
    gate_verification = verify_release_gate_v2(
        release_gate_path, **gate_v2_inputs(project_root)
    )
    benchmark_verification = verify_runtime_benchmark_v2(
        runtime_benchmark_path, **runtime_benchmark_inputs(project_root)
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    plan = json.loads(training_plan_path.read_text(encoding="utf-8"))
    gate = json.loads(release_gate_path.read_text(encoding="utf-8"))
    benchmark = json.loads(runtime_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    implementation = plan["source"]["implementation"]
    result["sae_training"]["execution_runtime_v2"] = {
        "schema": "sch.vpd-compartment-sae-execution-runtime.v2",
        "status": "implemented_model_free_not_run",
        "training_plan": {
            "path": _relative(training_plan_path, project_root),
            "sha256": sha256_file(training_plan_path),
            "training_plan_id": str(plan["training_plan_id"]),
            "verification": plan_verification,
        },
        "release_gate": {
            "path": _relative(release_gate_path, project_root),
            "sha256": sha256_file(release_gate_path),
            "release_gate_plan_id": str(gate["release_gate_plan_id"]),
            "verification": gate_verification,
        },
        "synthetic_benchmark": {
            "path": _relative(runtime_benchmark_path, project_root),
            "sha256": sha256_file(runtime_benchmark_path),
            "benchmark_id": str(benchmark["benchmark_id"]),
            "status": str(benchmark["status"]),
            "scenarios": int(benchmark["summary"]["scenarios"]),
            "passed": int(benchmark["summary"]["passed"]),
            "verification": benchmark_verification,
        },
        "implementation": {
            name: json.loads(json.dumps(implementation[name]))
            for name in (
                "capture_runner_v2",
                "sae_training_runner_v2",
                "dictionary_bundle_admission_v2",
                "quality_runner_v2",
                "release_admission_v2",
            )
        },
        "guarded_jobs": 22,
        "capture_jobs": 2,
        "training_jobs": 18,
        "bundle_admission_jobs": 1,
        "quality_jobs": 1,
        "required_dictionary_manifest_schema": (
            "sch.vpd-fixed-sae-dictionary-manifest.v2"
        ),
        "scientific_release_receipt_required": True,
        "scientific_release_receipt_compute_authorized": False,
        "external_explicit_compute_authorization_required": True,
        "external_authorization_must_cover_exact_v2_job_ids": True,
        "repository_generates_explicit_compute_authorization": False,
        "base_model_weights_mutable": False,
        "real_actions_authorized": False,
        "execution_authorized": False,
    }
    result["sae_training"]["replica_training_plan"] = {
        **result["sae_training"]["replica_training_plan"],
        "path": _relative(training_plan_path, project_root),
        "schema": "sch.vpd-compartment-sae-training-plan.v2",
        "status": "locally_verified_requires_scientific_release_and_explicit_compute_authorization",
    }
    result["unified_lineage"]["sae_training_manifest_v2_runtime_required"] = True
    result["unified_lineage"][
        "scientific_release_and_explicit_compute_authorization_both_required"
    ] = True
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        "predecessor_verification": predecessor_verification,
        "training_plan_v4_verification": plan_verification,
        "release_gate_v2_verification": gate_verification,
        "runtime_benchmark_v2_verification": benchmark_verification,
        "implementation_hashes": {
            name: str(implementation[name]["sha256"])
            for name in (
                "capture_runner_v2",
                "sae_training_runner_v2",
                "dictionary_bundle_admission_v2",
                "quality_runner_v2",
                "release_admission_v2",
            )
        },
        "research_protocol_v9_is_distinct_from_historical_experiment_labels": True,
    }
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["sae_training_performed"] = False
    result["scientific_boundary"]["weights_loaded_or_mutated"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = build_protocol(**kwargs)
    if observed != expected:
        raise ValueError("research protocol V9 is not reproducible")
    runtime = observed["sae_training"]["execution_runtime_v2"]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v9",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "predecessor_sha256": sha256_file(kwargs["predecessor_path"]),
        "training_plan_id": runtime["training_plan"]["training_plan_id"],
        "release_gate_plan_id": runtime["release_gate"]["release_gate_plan_id"],
        "guarded_jobs": runtime["guarded_jobs"],
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v8.json",
        "training_plan_path": project_root
        / "artifacts/vpd_compartment_sae_training_plan_v4/plan.json",
        "release_gate_path": project_root
        / "artifacts/vpd_compartment_sae_training_release_gate_v2/plan.json",
        "runtime_benchmark_path": project_root
        / "artifacts/vpd_compartment_sae_runtime_benchmark_v2/report.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive or verify executable-but-not-run research protocol V9."
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
    print(
        json.dumps(
            {
                "schema": str(result["schema"]),
                "status": str(result.get("status", "derived")),
                "protocol_sha256": str(
                    result.get("protocol_sha256", sha256_file(args.output))
                )
                if args.output is not None
                else str(result["protocol_sha256"]),
                "execution_authorized": bool(
                    result.get("execution_authorized", False)
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
