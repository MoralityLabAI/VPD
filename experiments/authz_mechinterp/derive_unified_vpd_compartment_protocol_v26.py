from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_capture_authorization_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_empirical_preflight_v2 import (
    verify_preflight,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json


SCHEMA = "sch.vpd-activation-category-protocol.v26"
EXPECTED_V25_SHA256 = (
    "e1175601443bf4162df3400ad3e0740eb65c74b5e6eeeb30c92657998ddf7626"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V26 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V25_SHA256:
        raise ValueError("protocol V25 differs from its pinned SHA-256")
    predecessor = json.loads(path.read_text(encoding="utf-8"))
    if predecessor.get("schema") != "sch.vpd-activation-category-protocol.v25":
        raise ValueError("pinned predecessor is not protocol V25")
    if predecessor["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("protocol V25 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-predecessor-verification.v1",
        "status": "pass",
        "method": "exact_sha256_and_boundary_fields",
        "sha256": EXPECTED_V25_SHA256,
    }


def build_protocol(
    *,
    predecessor_path: Path,
    preflight_v2_path: Path,
    authorization_validator_path: Path,
    authorization_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    preflight_verification = verify_preflight(
        preflight_v2_path, project_root=project_root
    )
    benchmark_verification = verify_benchmark(
        authorization_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_v2_path.read_text(encoding="utf-8"))
    benchmark = json.loads(authorization_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["cluster_to_sae_empirical_preflight_v1"][
        "status"
    ] = "preserved_predecessor_superseded_due_to_authorization_content_address_cycle"
    result["component_compartments"]["cluster_to_sae_capture_authorization_v1"] = {
        "schema": "sch.vpd-cluster-to-sae-capture-authorization-contract.v1",
        "status": "cycle_free_preflight_and_signed_receipt_validator_ready_real_receipt_absent",
        "preflight_v2": {
            "path": _relative(preflight_v2_path, project_root),
            "sha256": sha256_file(preflight_v2_path),
            "preflight_id": preflight["preflight_id"],
            "verification": preflight_verification,
            "missing_execution_identity_bindings": preflight[
                "missing_required_bindings"
            ],
            "authorization_receipt_is_deferred_one_way_handoff": True,
            "preflight_v1_content_address_cycle_preserved_as_gap_evidence": True,
        },
        "authorization_validator": {
            "path": _relative(authorization_validator_path, project_root),
            "sha256": sha256_file(authorization_validator_path),
            "receipt_schema": (
                "sch.vpd-cluster-to-sae-capture-authorization-receipt.v1"
            ),
            "registry_schema": (
                "sch.vpd-cluster-to-sae-capture-authority-registry.v1"
            ),
            "ed25519_required": True,
            "exact_scope_and_identity_binding_required": True,
            "active_time_window_required": True,
            "revocation_and_nonce_replay_checks_required": True,
            "signature_alone_does_not_authorize_runtime": True,
        },
        "authorization_benchmark": {
            "path": _relative(authorization_benchmark_path, project_root),
            "sha256": sha256_file(authorization_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"]["controls"],
            "synthetic_keys_and_identities_only": True,
            "private_key_persisted": False,
        },
        "real_identity_complete_preflight_present": False,
        "real_authority_registry_present": False,
        "real_signed_authorization_receipt_present": False,
        "empirical_capture_ready": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
        "final_evaluation_authorized": False,
    }
    result["unified_lineage"]["cluster_to_sae_preflight_v1_cycle_detected"] = True
    result["unified_lineage"]["cluster_to_sae_preflight_v2_cycle_free"] = True
    result["unified_lineage"]["capture_authorization_validator_ready"] = True
    result["unified_lineage"]["capture_authorization_controls"] = 16
    result["unified_lineage"]["cluster_to_sae_missing_execution_bindings"] = 10
    result["unified_lineage"]["real_signed_authorization_receipt_present"] = False
    result["unified_lineage"]["empirical_capture_ready"] = False
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v25_verification": predecessor_verification,
        "cycle_free_preflight_v2_verification": preflight_verification,
        "capture_authorization_benchmark_verification": benchmark_verification,
        "authorization_validator_sha256": sha256_file(authorization_validator_path),
        "bounded_predecessor_replay": True,
        "research_protocol_v26_is_distinct_from_historical_experiment_labels": True,
    }
    result["scientific_boundary"]["synthetic_authorization_fixture_only"] = True
    result["scientific_boundary"]["real_authorization_receipt_present"] = False
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
        raise ValueError("research protocol V26 is not reproducible")
    contract = observed["component_compartments"][
        "cluster_to_sae_capture_authorization_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v26",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "preflight_v2_id": contract["preflight_v2"]["preflight_id"],
        "authorization_benchmark_id": contract["authorization_benchmark"][
            "benchmark_id"
        ],
        "missing_execution_identity_bindings": len(
            contract["preflight_v2"]["missing_execution_identity_bindings"]
        ),
        "real_signed_authorization_receipt_present": False,
        "empirical_capture_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v25.json"
        ),
        "preflight_v2_path": (
            project_root
            / "artifacts/vpd_cluster_to_sae_empirical_preflight_v2/preflight.json"
        ),
        "authorization_validator_path": (
            project_root
            / "experiments/authz_mechinterp/verify_vpd_cluster_to_sae_capture_authorization_v1.py"
        ),
        "authorization_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_to_sae_capture_authorization_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V26.")
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
