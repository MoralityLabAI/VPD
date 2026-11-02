from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v9 import (
    canonical_inputs as protocol_v9_inputs,
    verify_protocol as verify_protocol_v9,
)
from experiments.authz_mechinterp.verify_unified_vpd_compartment_pipeline_v6 import (
    canonical_inputs as receipt_v6_inputs,
    verify_receipt as verify_receipt_v6,
)


SCHEMA = "sch.vpd-unified-compartment-pipeline-receipt.v7"


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("V9 unified lineage source escapes the project")
    return resolved.relative_to(root.resolve()).as_posix()


def build_receipt(
    *,
    predecessor_receipt_path: Path,
    protocol_path: Path,
    training_plan_path: Path,
    release_gate_path: Path,
    runtime_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = verify_receipt_v6(
        predecessor_receipt_path, **receipt_v6_inputs(project_root)
    )
    protocol_verification = verify_protocol_v9(
        protocol_path, **protocol_v9_inputs(project_root)
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    plan = json.loads(training_plan_path.read_text(encoding="utf-8"))
    gate = json.loads(release_gate_path.read_text(encoding="utf-8"))
    benchmark = json.loads(runtime_benchmark_path.read_text(encoding="utf-8"))
    runtime = protocol["sae_training"]["execution_runtime_v2"]
    plan_verification = runtime["training_plan"]["verification"]
    gate_verification = runtime["release_gate"]["verification"]
    benchmark_verification = runtime["synthetic_benchmark"]["verification"]
    if runtime["training_plan"]["training_plan_id"] != plan["training_plan_id"]:
        raise ValueError("research protocol V9 training plan lineage differs")
    if runtime["release_gate"]["release_gate_plan_id"] != gate["release_gate_plan_id"]:
        raise ValueError("research protocol V9 release gate lineage differs")
    if runtime["synthetic_benchmark"]["benchmark_id"] != benchmark["benchmark_id"]:
        raise ValueError("research protocol V9 runtime benchmark lineage differs")
    guarded_jobs = gate["guarded_execution"]["guarded_jobs"]
    if int(runtime["guarded_jobs"]) != len(guarded_jobs):
        raise ValueError("research protocol V9 guarded-job count differs")
    body = {
        "schema": SCHEMA,
        "status": "pass_local_runtime_v2_implemented_and_benchmarked_not_run",
        "source": {
            "predecessor_unified_receipt_v6": {
                "path": _relative(predecessor_receipt_path, project_root),
                "sha256": sha256_file(predecessor_receipt_path),
                "pipeline_receipt_id": predecessor_verification["pipeline_receipt_id"],
                "verification": predecessor_verification,
            },
            "research_protocol_v9": {
                "path": _relative(protocol_path, project_root),
                "sha256": sha256_file(protocol_path),
                "verification": protocol_verification,
            },
            "training_plan_v4": {
                "path": _relative(training_plan_path, project_root),
                "sha256": sha256_file(training_plan_path),
                "training_plan_id": plan["training_plan_id"],
                "verification": plan_verification,
            },
            "release_gate_v2": {
                "path": _relative(release_gate_path, project_root),
                "sha256": sha256_file(release_gate_path),
                "release_gate_plan_id": gate["release_gate_plan_id"],
                "verification": gate_verification,
            },
            "runtime_benchmark_v2": {
                "path": _relative(runtime_benchmark_path, project_root),
                "sha256": sha256_file(runtime_benchmark_path),
                "benchmark_id": benchmark["benchmark_id"],
                "verification": benchmark_verification,
            },
        },
        "lineage_amendment": {
            "predecessor_receipt_v6_preserved_and_reverified": True,
            "research_protocol_v9_is_not_historical_experiment_v8": True,
            "recovered_artifacts_not_relabelled_as_reconstructed": True,
            "runtime_v2_is_new_reconstructed_implementation": True,
            "all_22_jobs_require_scientific_release_receipt": True,
            "all_22_jobs_require_external_exact_job_compute_authorization": True,
            "scientific_release_is_not_compute_authorization": True,
            "repository_does_not_generate_external_authorization": True,
            "final_evaluation_remains_untouched": True,
        },
        "resources": {
            "guarded_jobs": len(guarded_jobs),
            "capture_jobs": int(runtime["capture_jobs"]),
            "training_jobs": int(runtime["training_jobs"]),
            "bundle_admission_jobs": int(runtime["bundle_admission_jobs"]),
            "quality_jobs": int(runtime["quality_jobs"]),
            "runtime_benchmark_scenarios": int(benchmark["summary"]["scenarios"]),
            "runtime_benchmark_passed": int(benchmark["summary"]["passed"]),
        },
        "invariants": {
            "predecessor_receipt_v6_reverified": True,
            "research_protocol_v9_reverified": True,
            "training_plan_v4_recomputed": True,
            "release_gate_v2_recomputed": True,
            "runtime_benchmark_v2_recomputed": True,
            "manifest_v2_hyperparameters_and_exact_samples_bound": True,
            "capture_training_bundle_and_quality_handoffs_bound": True,
            "execution_not_authorized": True,
        },
        "scientific_boundary": {
            "model_inference_performed": False,
            "activation_capture_performed": False,
            "sae_training_performed": False,
            "interventions_executed": False,
            "weights_loaded_or_mutated": False,
            "empirical_category_bundle_present": False,
            "empirical_dictionary_manifest_v2_present": False,
            "scientific_release_receipt_present": False,
            "external_compute_authorization_present": False,
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
        raise ValueError("V9 unified pipeline receipt content address mismatch")
    expected = build_receipt(**kwargs)
    if observed != expected:
        raise ValueError("V9 unified pipeline receipt is not reproducible")
    return {
        "schema": "sch.vpd-unified-compartment-pipeline-verification.v7",
        "status": "pass",
        "pipeline_receipt_id": identifier,
        "guarded_jobs": observed["resources"]["guarded_jobs"],
        "runtime_benchmark_scenarios": observed["resources"][
            "runtime_benchmark_scenarios"
        ],
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_receipt_path": project_root
        / "artifacts/vpd_unified_compartment_pipeline_v6/receipt.json",
        "protocol_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v9.json",
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
        description="Build or verify the research-protocol V9 runtime lineage amendment."
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
    print(
        json.dumps(
            {
                "schema": str(result["schema"]),
                "status": str(result["status"]),
                "pipeline_receipt_id": str(result["pipeline_receipt_id"]),
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
