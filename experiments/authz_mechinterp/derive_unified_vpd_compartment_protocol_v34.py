from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_capture_router_adapter_v2 import (
    verify_benchmark as verify_adapter_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v33 import (
    canonical_inputs as protocol_v33_inputs,
    verify_protocol as verify_protocol_v33,
)


SCHEMA = "sch.vpd-activation-category-protocol.v34"
EXPECTED_V33_SHA256 = (
    "cc25fac546c628e4618511e569485faf9a0a9984000affe85dfdbabaf0be2559"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V34 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, *, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V33_SHA256:
        raise ValueError("protocol V33 differs from its pinned SHA-256")
    verification = verify_protocol_v33(path, **protocol_v33_inputs(project_root))
    if verification["execution_authorized"] is not False:
        raise ValueError("protocol V33 authorization boundary differs")
    return verification


def build_protocol(
    *,
    predecessor_path: Path,
    adapter_v2_path: Path,
    adapter_v2_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(
        predecessor_path, project_root=project_root
    )
    adapter_verification = verify_adapter_benchmark(
        adapter_v2_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    benchmark = json.loads(adapter_v2_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["guarded_capture_router_adapter_v2"] = {
        "schema": "sch.vpd-capture-receipt-router-input-adapter-contract.v2",
        "status": "guarded_v3_lineage_bridge_ready_synthetic_controls_only",
        "adapter": {
            "path": _relative(adapter_v2_path, project_root),
            "sha256": sha256_file(adapter_v2_path),
            "receipt_schema": "sch.vpd-capture-receipt-router-input-adapter.v2",
        },
        "benchmark": {
            "path": _relative(adapter_v2_benchmark_path, project_root),
            "sha256": sha256_file(adapter_v2_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": adapter_verification,
            "controls": benchmark["summary"],
        },
        "v3_execution_identity_becomes_router_identity": True,
        "v2_payload_identity_preserved": True,
        "guarded_job_receipt_identity_preserved": True,
        "production_allocation_and_reservation_replay_required": True,
        "capture_artifacts_read_only": True,
        "production_path_empirically_exercised": False,
        "persistent_empirical_router_receipt_present": False,
        "model_loaded": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    result["unified_lineage"]["guarded_capture_router_adapter_v2_ready"] = True
    result["unified_lineage"]["v3_capture_identity_preserved_into_router"] = True
    result["unified_lineage"]["production_router_adapter_empirically_exercised"] = False
    result["unified_lineage"]["empirical_router_receipts_present"] = False
    result["unified_lineage"]["empirical_capture_ready"] = False
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v33_verification": predecessor_verification,
        "capture_router_adapter_v2_benchmark_verification": adapter_verification,
        "capture_router_adapter_v2_derivation_sha256": sha256_file(Path(__file__)),
        "bounded_predecessor_replay": True,
    }
    result["scientific_boundary"]["synthetic_v3_to_router_fixture_only"] = True
    result["scientific_boundary"]["v3_capture_identity_preserved_into_router"] = True
    result["scientific_boundary"]["production_router_adapter_empirically_exercised"] = False
    result["scientific_boundary"]["persistent_empirical_router_receipt_present"] = False
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
        raise ValueError("research protocol V34 is not reproducible")
    adapter = observed["component_compartments"][
        "guarded_capture_router_adapter_v2"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v34",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "capture_router_adapter_v2_benchmark_id": adapter["benchmark"][
            "benchmark_id"
        ],
        "capture_router_adapter_v2_controls": adapter["benchmark"]["controls"][
            "controls"
        ],
        "v3_capture_identity_preserved_into_router": True,
        "production_router_adapter_empirically_exercised": False,
        "persistent_empirical_router_receipt_present": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v33.json"
        ),
        "adapter_v2_path": (
            project_root
            / "experiments/authz_mechinterp/load_vpd_capture_router_inputs_v2.py"
        ),
        "adapter_v2_benchmark_path": (
            project_root / "artifacts/vpd_capture_router_adapter_benchmark_v2/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V34.")
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
