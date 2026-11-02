from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_transition_v1 import (
    verify_benchmark as verify_transition_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v38 import (
    canonical_inputs as protocol_v38_inputs,
    verify_protocol as verify_protocol_v38,
)


SCHEMA = "sch.vpd-activation-category-protocol.v39"
EXPECTED_V38_SHA256 = (
    "90660a5aaacaffeb33d22e6c49cb02a2e160a2b2a5a2946f9f31990c3d1fbd06"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V39 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, *, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V38_SHA256:
        raise ValueError("protocol V38 differs from its pinned SHA-256")
    verification = verify_protocol_v38(path, **protocol_v38_inputs(project_root))
    if verification["execution_authorized"] is not False:
        raise ValueError("protocol V38 authorization boundary differs")
    return verification


def build_protocol(
    *,
    predecessor_path: Path,
    transition_path: Path,
    transition_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(
        predecessor_path, project_root=project_root
    )
    benchmark_verification = verify_transition_benchmark(
        transition_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    benchmark = json.loads(transition_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["cluster_aware_sae_transition_v1"] = {
        "schema": "sch.vpd-cluster-aware-sae-durable-transition-contract.v1",
        "status": (
            "release_campaign_and_atomic_compute_job_reservation_ready_"
            "runtime_attach_trainer_invocation_and_empirical_inputs_absent"
        ),
        "transition": {
            "path": _relative(transition_path, project_root),
            "sha256": sha256_file(transition_path),
            "campaign_receipt_schema": (
                "sch.vpd-cluster-aware-sae-release-campaign-reservation.v1"
            ),
            "job_transition_schema": (
                "sch.vpd-cluster-aware-sae-job-transition-reservation.v1"
            ),
        },
        "benchmark": {
            "path": _relative(transition_benchmark_path, project_root),
            "sha256": sha256_file(transition_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
        },
        "release_nonce_campaign_reservation_implemented": True,
        "one_release_campaign_supports_multiple_unique_jobs": True,
        "atomic_compute_nonce_and_job_root_reservation_implemented": True,
        "exact_output_namespace_reserved_without_dictionary_creation": True,
        "worker_ledger_reserved_before_runtime": True,
        "replay_and_cross_job_nonce_reuse_rejected": True,
        "partial_failures_leave_terminal_tombstones": True,
        "rollback_or_resume_implemented": False,
        "runtime_attach_implemented": False,
        "trainer_invocation_implemented": False,
        "empirical_blueprint_present": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    result["unified_lineage"]["cluster_aware_sae_durable_transition_ready"] = True
    result["unified_lineage"]["release_campaign_nonce_reservation_implemented"] = True
    result["unified_lineage"][
        "atomic_compute_nonce_and_job_root_reservation_implemented"
    ] = True
    result["unified_lineage"]["cluster_aware_sae_worker_ledger_reserved"] = True
    result["unified_lineage"]["cluster_aware_sae_runtime_attach_implemented"] = False
    result["unified_lineage"]["cluster_aware_sae_trainer_invocation_implemented"] = (
        False
    )
    result["unified_lineage"]["empirical_cluster_aware_blueprint_present"] = False
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v38_verification": predecessor_verification,
        "durable_transition_benchmark_verification": benchmark_verification,
        "durable_transition_derivation_sha256": sha256_file(Path(__file__)),
        "bounded_predecessor_replay": True,
    }
    result["scientific_boundary"]["temporary_transition_fixture_only"] = True
    result["scientific_boundary"]["cluster_aware_durable_transition_ready"] = True
    result["scientific_boundary"]["release_campaign_nonce_reservation_implemented"] = (
        True
    )
    result["scientific_boundary"][
        "atomic_compute_nonce_and_job_root_reservation_implemented"
    ] = True
    result["scientific_boundary"]["worker_ledger_reservation_implemented"] = True
    result["scientific_boundary"]["runtime_attach_implemented"] = False
    result["scientific_boundary"]["trainer_invocation_implemented"] = False
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
        raise ValueError("research protocol V39 is not reproducible")
    component = observed["component_compartments"]["cluster_aware_sae_transition_v1"]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v39",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "transition_benchmark_id": component["benchmark"]["benchmark_id"],
        "transition_controls": component["benchmark"]["controls"]["controls"],
        "release_nonce_campaign_reservation_implemented": True,
        "atomic_compute_nonce_and_job_root_reservation_implemented": True,
        "worker_ledger_reservation_implemented": True,
        "runtime_attach_implemented": False,
        "trainer_invocation_implemented": False,
        "empirical_blueprint_present": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v38.json"
        ),
        "transition_path": (
            project_root
            / "experiments/authz_mechinterp/reserve_cluster_aware_sae_training_transition_v1.py"
        ),
        "transition_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_transition_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V39.")
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
