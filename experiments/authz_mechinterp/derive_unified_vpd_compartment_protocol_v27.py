from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_binding_readiness_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    verify_readiness,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v26 import (
    canonical_inputs as protocol_v26_inputs,
    verify_protocol as verify_protocol_v26,
)


SCHEMA = "sch.vpd-activation-category-protocol.v27"
EXPECTED_V26_SHA256 = (
    "3372cf627793588a9e5c2078f44f7955e028c8f43fb071d37d07735ced02eaba"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V27 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, *, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V26_SHA256:
        raise ValueError("protocol V26 differs from its pinned SHA-256")
    verification = verify_protocol_v26(path, **protocol_v26_inputs(project_root))
    if verification["execution_authorized"] is not False:
        raise ValueError("protocol V26 authorization boundary differs")
    return verification


def build_protocol(
    *,
    predecessor_path: Path,
    readiness_path: Path,
    readiness_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(
        predecessor_path, project_root=project_root
    )
    readiness_verification = verify_readiness(readiness_path, project_root=project_root)
    benchmark_verification = verify_benchmark(
        readiness_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    benchmark = json.loads(readiness_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["cluster_to_sae_binding_readiness_v1"] = {
        "schema": "sch.vpd-cluster-to-sae-binding-readiness-contract.v1",
        "status": "worksheet_ready_real_bindings_and_signed_authorization_absent",
        "readiness": {
            "path": _relative(readiness_path, project_root),
            "sha256": sha256_file(readiness_path),
            "readiness_id": readiness["readiness_id"],
            "verification": readiness_verification,
            "required_bindings": readiness["summary"]["required_bindings"],
            "missing_bindings": readiness["summary"]["missing_bindings"],
        },
        "benchmark": {
            "path": _relative(readiness_benchmark_path, project_root),
            "sha256": sha256_file(readiness_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
        },
        "runtime_isolation_contract": readiness["runtime_isolation_contract"],
        "resource_ceiling": readiness["resource_ceiling"],
        "real_bindings_present": False,
        "model_or_adapter_content_opened": False,
        "real_signed_authorization_receipt_present": False,
        "empirical_capture_ready": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    result["unified_lineage"]["binding_readiness_worksheet_ready"] = True
    result["unified_lineage"]["binding_readiness_controls"] = benchmark["summary"][
        "controls"
    ]
    result["unified_lineage"]["canonical_runtime_bindings_present"] = False
    result["unified_lineage"]["output_isolation_verified_for_real_run"] = False
    result["unified_lineage"]["real_signed_authorization_receipt_present"] = False
    result["unified_lineage"]["empirical_capture_ready"] = False
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v26_verification": predecessor_verification,
        "binding_readiness_verification": readiness_verification,
        "binding_readiness_benchmark_verification": benchmark_verification,
        "binding_readiness_derivation_sha256": sha256_file(Path(__file__)),
        "runtime_binding_checks_are_distinct_from_signed_authorization": True,
        "bounded_predecessor_replay": True,
    }
    result["scientific_boundary"]["binding_worksheet_only"] = True
    result["scientific_boundary"]["real_runtime_bindings_present"] = False
    result["scientific_boundary"]["output_root_created"] = False
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
        raise ValueError("research protocol V27 is not reproducible")
    contract = observed["component_compartments"][
        "cluster_to_sae_binding_readiness_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v27",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "readiness_id": contract["readiness"]["readiness_id"],
        "readiness_benchmark_id": contract["benchmark"]["benchmark_id"],
        "readiness_controls": contract["benchmark"]["controls"]["controls"],
        "missing_execution_identity_bindings": len(
            contract["readiness"]["missing_bindings"]
        ),
        "output_isolation_verified_for_real_run": False,
        "empirical_capture_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v26.json"
        ),
        "readiness_path": (
            project_root
            / "artifacts/vpd_cluster_to_sae_binding_readiness_v1/readiness.json"
        ),
        "readiness_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_to_sae_binding_readiness_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V27.")
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
