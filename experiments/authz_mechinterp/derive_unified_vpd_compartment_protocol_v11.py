from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_candidate_discreteness_admission_v2 import (
    verify_benchmark as verify_admission_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v10 import (
    canonical_inputs as protocol_v10_inputs,
    verify_protocol as verify_protocol_v10,
)


SCHEMA = "sch.vpd-activation-category-protocol.v11"


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("research protocol V11 source escapes the project")
    return resolved.relative_to(root.resolve()).as_posix()


def build_protocol(
    *,
    predecessor_path: Path,
    admission_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = verify_protocol_v10(
        predecessor_path, **protocol_v10_inputs(project_root)
    )
    benchmark_verification = verify_admission_benchmark(
        admission_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    benchmark = json.loads(admission_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    admission_runner = project_root / (
        "experiments/authz_mechinterp/"
        "admit_vpd_cluster_candidate_discreteness_v2.py"
    )
    benchmark_runner = project_root / (
        "experiments/authz_mechinterp/"
        "benchmark_vpd_cluster_candidate_discreteness_admission_v2.py"
    )
    result["component_compartments"]["cluster_candidate_discreteness_admission_v2"] = {
        "schema": "sch.vpd-cluster-candidate-discreteness-admission.v2",
        "status": "implemented_model_free_benchmarked_not_empirically_run",
        "runner": {
            "path": _relative(admission_runner, project_root),
            "sha256": sha256_file(admission_runner),
        },
        "benchmark": {
            "path": _relative(admission_benchmark_path, project_root),
            "sha256": sha256_file(admission_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "scenarios": len(benchmark["scenarios"]),
        },
        "base_v1_bundle_must_verify_first": True,
        "independent_raw_discovery_and_cluster_validation_reconstruction": True,
        "exact_v1_v2_allocation_agreement_required": True,
        "candidate_mutation_or_relabelling_prohibited": True,
        "allocation_mismatch_action": "reject_candidate_without_mutation",
        "empirical_admission_receipt_present": False,
        "execution_authorized": False,
    }
    result["unified_lineage"]["cluster_candidate_discreteness_admission_v2_required"] = True
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v10_verification": predecessor_verification,
        "cluster_candidate_admission_benchmark_verification": benchmark_verification,
        "cluster_candidate_admission_runner_sha256": sha256_file(admission_runner),
        "cluster_candidate_admission_benchmark_runner_sha256": sha256_file(
            benchmark_runner
        ),
        "research_protocol_v11_is_distinct_from_historical_experiment_labels": True,
    }
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["sae_training_performed"] = False
    result["scientific_boundary"]["weights_loaded_or_mutated"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != build_protocol(**kwargs):
        raise ValueError("research protocol V11 is not reproducible")
    contract = observed["component_compartments"][
        "cluster_candidate_discreteness_admission_v2"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v11",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "predecessor_sha256": sha256_file(kwargs["predecessor_path"]),
        "admission_benchmark_id": contract["benchmark"]["benchmark_id"],
        "production_admission_runtime_present": True,
        "empirical_admission_receipt_present": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v10.json",
        "admission_benchmark_path": project_root
        / "artifacts/vpd_cluster_candidate_discreteness_admission_benchmark_v2/report.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V11.")
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
    print(json.dumps({
        "schema": result["schema"],
        "status": result.get("status", "derived"),
        "protocol_sha256": result.get("protocol_sha256", sha256_file(args.output)),
        "execution_authorized": result.get("execution_authorized", False),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
