from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_inert_callback_v1 import (
    verify_benchmark as verify_callback_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v40 import (
    canonical_inputs as protocol_v40_inputs,
    verify_protocol as verify_protocol_v40,
)


SCHEMA = "sch.vpd-activation-category-protocol.v41"
EXPECTED_V40_SHA256 = (
    "197f7f02ffbeb5f9a530a5880391daa0951611bd42b4202236c78e33a151a9c8"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V41 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, *, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V40_SHA256:
        raise ValueError("protocol V40 differs from its pinned SHA-256")
    verification = verify_protocol_v40(path, **protocol_v40_inputs(project_root))
    if verification["execution_authorized"] is not False:
        raise ValueError("protocol V40 authorization boundary differs")
    return verification


def build_protocol(
    *,
    predecessor_path: Path,
    callback_execution_path: Path,
    callback_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(
        predecessor_path, project_root=project_root
    )
    benchmark_verification = verify_callback_benchmark(
        callback_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    benchmark = json.loads(callback_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["cluster_aware_sae_inert_callback_v1"] = {
        "schema": "sch.vpd-cluster-aware-sae-inert-callback-contract.v1",
        "status": (
            "durable_synthetic_callback_claim_success_and_charged_failure_"
            "receipts_ready_production_trainer_and_empirical_inputs_absent"
        ),
        "execution": {
            "path": _relative(callback_execution_path, project_root),
            "sha256": sha256_file(callback_execution_path),
            "success_schema": (
                "sch.vpd-cluster-aware-sae-synthetic-callback-execution.v1"
            ),
            "failure_schema": (
                "sch.vpd-cluster-aware-sae-synthetic-callback-failure.v1"
            ),
        },
        "benchmark": {
            "path": _relative(callback_benchmark_path, project_root),
            "sha256": sha256_file(callback_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
        },
        "durable_execution_claim_before_callback": True,
        "successful_callback_charge_persisted": True,
        "failed_callback_charge_persisted": True,
        "callback_single_use": True,
        "boundary_weakening_result_is_terminal_failure": True,
        "synthetic_inert_callback_invoked": True,
        "production_trainer_callback_invoked": False,
        "training_performed": False,
        "persistent_dictionary_output_created": False,
        "empirical_blueprint_present": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    result["unified_lineage"]["cluster_aware_sae_inert_callback_ready"] = True
    result["unified_lineage"]["durable_callback_execution_claim_implemented"] = True
    result["unified_lineage"]["synthetic_inert_callback_invoked"] = True
    result["unified_lineage"]["synthetic_callback_failure_charge_persisted"] = True
    result["unified_lineage"]["cluster_aware_sae_trainer_invocation_implemented"] = False
    result["unified_lineage"]["production_trainer_callback_invoked"] = False
    result["unified_lineage"]["empirical_cluster_aware_blueprint_present"] = False
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v40_verification": predecessor_verification,
        "inert_callback_benchmark_verification": benchmark_verification,
        "inert_callback_derivation_sha256": sha256_file(Path(__file__)),
        "bounded_predecessor_replay": True,
    }
    boundary = result["scientific_boundary"]
    boundary["temporary_inert_callback_fixture_only"] = True
    boundary["cluster_aware_inert_callback_ready"] = True
    boundary["durable_callback_execution_claim_implemented"] = True
    boundary["successful_callback_charge_persisted"] = True
    boundary["failed_callback_charge_persisted"] = True
    boundary["synthetic_inert_callback_invoked"] = True
    boundary["production_trainer_callback_invoked"] = False
    boundary["training_performed"] = False
    boundary["persistent_dictionary_output_created"] = False
    boundary["empirical_cluster_aware_blueprint_present"] = False
    boundary["model_framework_imported"] = False
    boundary["model_or_adapter_weights_opened"] = False
    boundary["model_inference_performed"] = False
    boundary["weights_loaded_or_mutated"] = False
    boundary["sae_dictionary_training_authorized"] = False
    boundary["execution_authorized"] = False
    boundary["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != build_protocol(**kwargs):
        raise ValueError("research protocol V41 is not reproducible")
    component = observed["component_compartments"][
        "cluster_aware_sae_inert_callback_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v41",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "inert_callback_benchmark_id": component["benchmark"]["benchmark_id"],
        "inert_callback_controls": component["benchmark"]["controls"]["controls"],
        "durable_callback_execution_claim_implemented": True,
        "synthetic_inert_callback_invoked": True,
        "failed_callback_charge_persisted": True,
        "production_trainer_callback_invoked": False,
        "training_performed": False,
        "persistent_dictionary_output_created": False,
        "empirical_blueprint_present": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v40.json"
        ),
        "callback_execution_path": (
            project_root
            / "experiments/authz_mechinterp/execute_cluster_aware_sae_inert_callback_v1.py"
        ),
        "callback_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_inert_callback_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V41.")
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
