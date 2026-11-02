from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_admission_v1 import (
    verify_benchmark as verify_admission_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v37 import (
    canonical_inputs as protocol_v37_inputs,
    verify_protocol as verify_protocol_v37,
)


SCHEMA = "sch.vpd-activation-category-protocol.v38"
EXPECTED_V37_SHA256 = (
    "48462a79bd32534c2472b64dfd8badd13c3d5246cd7bd7ff6742bc41fcde04e7"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V38 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, *, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V37_SHA256:
        raise ValueError("protocol V37 differs from its pinned SHA-256")
    verification = verify_protocol_v37(path, **protocol_v37_inputs(project_root))
    if verification["execution_authorized"] is not False:
        raise ValueError("protocol V37 authorization boundary differs")
    return verification


def build_protocol(
    *,
    predecessor_path: Path,
    admission_verifier_path: Path,
    admission_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(
        predecessor_path, project_root=project_root
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
    result["component_compartments"]["cluster_aware_sae_signed_admission_v1"] = {
        "schema": "sch.vpd-cluster-aware-sae-signed-admission-contract.v1",
        "status": (
            "ed25519_release_and_job_compute_verifiers_ready_dry_run_only_"
            "durable_nonce_output_reservation_and_trainer_invocation_absent"
        ),
        "verifier": {
            "path": _relative(admission_verifier_path, project_root),
            "sha256": sha256_file(admission_verifier_path),
            "authority_registry_schema": (
                "sch.vpd-cluster-aware-sae-authority-registry.v1"
            ),
            "scientific_release_receipt_schema": (
                "sch.vpd-cluster-aware-sae-scientific-release-receipt.v1"
            ),
            "compute_authorization_receipt_schema": (
                "sch.vpd-cluster-aware-sae-compute-authorization-receipt.v1"
            ),
            "dry_run_admission_schema": (
                "sch.vpd-cluster-aware-sae-runtime-admission.v1"
            ),
        },
        "benchmark": {
            "path": _relative(admission_benchmark_path, project_root),
            "sha256": sha256_file(admission_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
        },
        "ed25519_scientific_release_receipt_verified": True,
        "ed25519_job_compute_receipt_verified": True,
        "release_binds_exact_blueprint_plan_materialization_and_global_split": True,
        "compute_binds_exact_job_samples_resources_device_and_output": True,
        "time_revocation_and_nonce_replay_checked": True,
        "synthetic_blueprint_rejected_for_real_release": True,
        "dry_run_cli_implemented": True,
        "durable_nonce_consumption_implemented": False,
        "atomic_output_reservation_implemented": False,
        "production_trainer_invocation_implemented": False,
        "empirical_blueprint_present": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    result["unified_lineage"]["cluster_aware_sae_signed_admission_ready"] = True
    result["unified_lineage"]["cluster_aware_sae_dry_run_cli_implemented"] = True
    result["unified_lineage"]["cluster_aware_sae_durable_nonce_consumption"] = False
    result["unified_lineage"]["cluster_aware_sae_atomic_output_reservation"] = False
    result["unified_lineage"]["cluster_aware_sae_production_trainer_invocation"] = (
        False
    )
    result["unified_lineage"]["empirical_cluster_aware_blueprint_present"] = False
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v37_verification": predecessor_verification,
        "signed_admission_benchmark_verification": benchmark_verification,
        "signed_admission_derivation_sha256": sha256_file(Path(__file__)),
        "bounded_predecessor_replay": True,
    }
    result["scientific_boundary"]["synthetic_signed_admission_fixture_only"] = True
    result["scientific_boundary"]["cluster_aware_signed_admission_ready"] = True
    result["scientific_boundary"]["cluster_aware_dry_run_cli_implemented"] = True
    result["scientific_boundary"]["durable_nonce_consumption_implemented"] = False
    result["scientific_boundary"]["atomic_output_reservation_implemented"] = False
    result["scientific_boundary"]["production_trainer_invocation_implemented"] = False
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
        raise ValueError("research protocol V38 is not reproducible")
    component = observed["component_compartments"][
        "cluster_aware_sae_signed_admission_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v38",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "signed_admission_benchmark_id": component["benchmark"]["benchmark_id"],
        "signed_admission_controls": component["benchmark"]["controls"]["controls"],
        "dry_run_cli_implemented": True,
        "durable_nonce_consumption_implemented": False,
        "atomic_output_reservation_implemented": False,
        "production_trainer_invocation_implemented": False,
        "empirical_blueprint_present": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v37.json"
        ),
        "admission_verifier_path": (
            project_root
            / "experiments/authz_mechinterp/admit_cluster_aware_sae_training_v1.py"
        ),
        "admission_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_admission_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V38.")
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
