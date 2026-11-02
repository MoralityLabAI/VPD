from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_inert_attachment_v1 import (
    verify_benchmark as verify_attachment_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v39 import (
    canonical_inputs as protocol_v39_inputs,
    verify_protocol as verify_protocol_v39,
)


SCHEMA = "sch.vpd-activation-category-protocol.v40"
EXPECTED_V39_SHA256 = (
    "b167797dc91c93292ae651d05c96fcb433995cedf62a7a340e85c4e305a92ab7"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V40 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, *, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V39_SHA256:
        raise ValueError("protocol V39 differs from its pinned SHA-256")
    verification = verify_protocol_v39(path, **protocol_v39_inputs(project_root))
    if verification["execution_authorized"] is not False:
        raise ValueError("protocol V39 authorization boundary differs")
    return verification


def build_protocol(
    *,
    predecessor_path: Path,
    attachment_path: Path,
    attachment_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(
        predecessor_path, project_root=project_root
    )
    benchmark_verification = verify_attachment_benchmark(
        attachment_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    benchmark = json.loads(attachment_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["cluster_aware_sae_inert_attachment_v1"] = {
        "schema": "sch.vpd-cluster-aware-sae-inert-attachment-contract.v1",
        "status": (
            "durable_claim_and_exact_ledger_reconstruction_ready_inert_runtime_only_"
            "production_runtime_trainer_and_empirical_inputs_absent"
        ),
        "attachment": {
            "path": _relative(attachment_path, project_root),
            "sha256": sha256_file(attachment_path),
            "claim_schema": (
                "sch.vpd-cluster-aware-sae-inert-runtime-attachment-claim.v1"
            ),
        },
        "benchmark": {
            "path": _relative(attachment_benchmark_path, project_root),
            "sha256": sha256_file(attachment_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
        },
        "durable_attachment_claim_before_runtime_factory": True,
        "transition_and_signed_admission_reconstructed_exactly": True,
        "worker_ledger_reconstructed_exactly": True,
        "attachment_claim_single_use_across_process_reconstruction": True,
        "factory_and_active_runtime_failures_leave_terminal_claims": True,
        "synthetic_inert_runtime_attached": True,
        "trainer_callback_invoked": False,
        "production_runtime_attach_implemented": False,
        "trainer_invocation_implemented": False,
        "empirical_blueprint_present": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    result["unified_lineage"]["cluster_aware_sae_inert_attachment_ready"] = True
    result["unified_lineage"]["durable_runtime_attachment_claim_implemented"] = True
    result["unified_lineage"]["worker_ledger_reconstruction_implemented"] = True
    result["unified_lineage"]["synthetic_inert_runtime_attached"] = True
    result["unified_lineage"]["cluster_aware_sae_production_runtime_attach"] = False
    result["unified_lineage"]["cluster_aware_sae_trainer_invocation_implemented"] = (
        False
    )
    result["unified_lineage"]["empirical_cluster_aware_blueprint_present"] = False
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v39_verification": predecessor_verification,
        "inert_attachment_benchmark_verification": benchmark_verification,
        "inert_attachment_derivation_sha256": sha256_file(Path(__file__)),
        "bounded_predecessor_replay": True,
    }
    result["scientific_boundary"]["temporary_inert_attachment_fixture_only"] = True
    result["scientific_boundary"]["cluster_aware_inert_attachment_ready"] = True
    result["scientific_boundary"]["durable_attachment_claim_implemented"] = True
    result["scientific_boundary"]["worker_ledger_reconstruction_implemented"] = True
    result["scientific_boundary"]["synthetic_inert_runtime_attached"] = True
    result["scientific_boundary"]["production_runtime_attach_implemented"] = False
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
        raise ValueError("research protocol V40 is not reproducible")
    component = observed["component_compartments"][
        "cluster_aware_sae_inert_attachment_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v40",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "inert_attachment_benchmark_id": component["benchmark"]["benchmark_id"],
        "inert_attachment_controls": component["benchmark"]["controls"]["controls"],
        "durable_attachment_claim_implemented": True,
        "worker_ledger_reconstruction_implemented": True,
        "synthetic_inert_runtime_attached": True,
        "production_runtime_attach_implemented": False,
        "trainer_callback_invoked": False,
        "empirical_blueprint_present": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v39.json"
        ),
        "attachment_path": (
            project_root
            / "experiments/authz_mechinterp/attach_cluster_aware_sae_inert_runtime_v1.py"
        ),
        "attachment_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_inert_attachment_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V40.")
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
