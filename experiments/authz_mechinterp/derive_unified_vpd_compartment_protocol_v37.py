from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_runtime_worker_v1 import (
    verify_benchmark as verify_worker_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v36 import (
    canonical_inputs as protocol_v36_inputs,
    verify_protocol as verify_protocol_v36,
)


SCHEMA = "sch.vpd-activation-category-protocol.v37"
EXPECTED_V36_SHA256 = (
    "734ec49cd8c4b691333e4936e25fa9f691b3f5a4f6f84a6c9aa13b0da076ecc2"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V37 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, *, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V36_SHA256:
        raise ValueError("protocol V36 differs from its pinned SHA-256")
    verification = verify_protocol_v36(path, **protocol_v36_inputs(project_root))
    if verification["execution_authorized"] is not False:
        raise ValueError("protocol V36 authorization boundary differs")
    return verification


def build_protocol(
    *,
    predecessor_path: Path,
    runtime_worker_path: Path,
    runtime_worker_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(
        predecessor_path, project_root=project_root
    )
    benchmark_verification = verify_worker_benchmark(
        runtime_worker_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    benchmark = json.loads(runtime_worker_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["cluster_aware_sae_runtime_worker_v1"] = {
        "schema": "sch.vpd-cluster-aware-sae-runtime-worker-contract.v1",
        "status": (
            "injected_worker_core_ready_synthetic_controls_pass_"
            "production_cli_and_real_authority_verifiers_absent"
        ),
        "worker": {
            "path": _relative(runtime_worker_path, project_root),
            "sha256": sha256_file(runtime_worker_path),
            "receipt_schema": "sch.vpd-cluster-aware-sae-runtime-worker-receipt.v1",
        },
        "benchmark": {
            "path": _relative(runtime_worker_benchmark_path, project_root),
            "sha256": sha256_file(runtime_worker_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
        },
        "execution_jobs_in_fixture": 34,
        "exact_blueprint_and_job_content_validation": True,
        "job_and_planned_resources_reserved_before_runtime_attach": True,
        "runtime_attachment_single_use": True,
        "trainer_callback_single_use": True,
        "failed_callbacks_remain_charged": True,
        "wrong_job_authority_rejected_before_reservation": True,
        "synthetic_blueprint_rejected_for_production": True,
        "worker_core_implemented": True,
        "production_cli_implemented": False,
        "real_scientific_release_verifier_implemented": False,
        "real_compute_authorization_verifier_implemented": False,
        "empirical_blueprint_present": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    result["unified_lineage"]["cluster_aware_sae_runtime_worker_core_implemented"] = (
        True
    )
    result["unified_lineage"]["cluster_aware_sae_production_cli_implemented"] = False
    result["unified_lineage"][
        "cluster_aware_sae_real_authority_verifiers_implemented"
    ] = False
    result["unified_lineage"]["empirical_cluster_aware_blueprint_present"] = False
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v36_verification": predecessor_verification,
        "runtime_worker_benchmark_verification": benchmark_verification,
        "runtime_worker_derivation_sha256": sha256_file(Path(__file__)),
        "bounded_predecessor_replay": True,
    }
    result["scientific_boundary"]["synthetic_runtime_worker_callbacks_only"] = True
    result["scientific_boundary"]["cluster_aware_runtime_worker_core_implemented"] = (
        True
    )
    result["scientific_boundary"]["cluster_aware_production_cli_implemented"] = False
    result["scientific_boundary"]["real_release_verifier_implemented"] = False
    result["scientific_boundary"]["real_compute_authorization_verifier_implemented"] = (
        False
    )
    result["scientific_boundary"]["empirical_cluster_aware_blueprint_present"] = (
        False
    )
    result["scientific_boundary"]["model_framework_imported"] = False
    result["scientific_boundary"]["model_or_adapter_weights_opened"] = False
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["weights_loaded_or_mutated"] = False
    result["scientific_boundary"]["sae_dictionary_training_authorized"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != build_protocol(**kwargs):
        raise ValueError("research protocol V37 is not reproducible")
    component = observed["component_compartments"][
        "cluster_aware_sae_runtime_worker_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v37",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "runtime_worker_benchmark_id": component["benchmark"]["benchmark_id"],
        "runtime_worker_controls": component["benchmark"]["controls"]["controls"],
        "execution_jobs": component["execution_jobs_in_fixture"],
        "worker_core_implemented": True,
        "production_cli_implemented": False,
        "real_authority_verifiers_implemented": False,
        "empirical_blueprint_present": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v36.json"
        ),
        "runtime_worker_path": (
            project_root
            / "experiments/authz_mechinterp/run_cluster_aware_sae_training_job_v1.py"
        ),
        "runtime_worker_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_runtime_worker_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V37.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.output is not None:
        result = build_protocol(**inputs)
        write_json(args.output, result)
        protocol_sha256 = sha256_file(args.output)
    else:
        result = verify_protocol(args.verify, **inputs)
        protocol_sha256 = result["protocol_sha256"]
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result.get("status", "derived"),
                "protocol_sha256": protocol_sha256,
                "execution_authorized": result.get("execution_authorized", False),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
