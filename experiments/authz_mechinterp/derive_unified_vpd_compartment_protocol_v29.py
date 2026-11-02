from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_runtime_admission_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v28 import (
    canonical_inputs as protocol_v28_inputs,
    verify_protocol as verify_protocol_v28,
)


SCHEMA = "sch.vpd-activation-category-protocol.v29"
EXPECTED_V28_SHA256 = (
    "128d8ad5a7fc34b56b24e46b5221b919e035c74b4b5faa7d2d6abcb15dd316b5"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V29 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, *, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V28_SHA256:
        raise ValueError("protocol V28 differs from its pinned SHA-256")
    verification = verify_protocol_v28(path, **protocol_v28_inputs(project_root))
    if verification["execution_authorized"] is not False:
        raise ValueError("protocol V28 authorization boundary differs")
    return verification


def build_protocol(
    *,
    predecessor_path: Path,
    runtime_admission_builder_path: Path,
    runtime_admission_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(
        predecessor_path, project_root=project_root
    )
    benchmark_verification = verify_benchmark(
        runtime_admission_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    benchmark = json.loads(runtime_admission_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["cluster_to_sae_runtime_admission_v1"] = {
        "schema": "sch.vpd-cluster-to-sae-runtime-admission-contract.v1",
        "status": "readonly_join_ready_real_inputs_and_reservation_absent",
        "builder": {
            "path": _relative(runtime_admission_builder_path, project_root),
            "sha256": sha256_file(runtime_admission_builder_path),
            "admission_schema": "sch.vpd-cluster-to-sae-runtime-admission.v1",
            "reservation_schema": (
                "sch.vpd-cluster-to-sae-output-root-reservation.v1"
            ),
        },
        "benchmark": {
            "path": _relative(runtime_admission_benchmark_path, project_root),
            "sha256": sha256_file(runtime_admission_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
        },
        "joined_proofs": [
            "cycle_free_identity_complete_preflight_v2",
            "canonical_binding_and_nonexistent_output_audit",
            "exact_model_tree_and_adapter_byte_verification",
            "active_scope_exact_signed_authorization_verification",
        ],
        "all_proofs_must_join_same_preflight_id": True,
        "synthetic_authorization_cannot_reserve_output": True,
        "atomic_output_creation_is_exact_receipt_replay_barrier": True,
        "existing_output_root_is_terminal_fail_closed": True,
        "reservation_never_starts_capture": True,
        "real_runtime_admission_present": False,
        "real_output_reservation_present": False,
        "capture_started": False,
        "empirical_capture_ready": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    result["unified_lineage"]["runtime_admission_join_ready"] = True
    result["unified_lineage"]["runtime_admission_controls"] = benchmark["summary"][
        "controls"
    ]
    result["unified_lineage"]["exact_receipt_replay_barrier_pinned"] = True
    result["unified_lineage"]["real_runtime_admission_present"] = False
    result["unified_lineage"]["real_output_reservation_present"] = False
    result["unified_lineage"]["empirical_capture_ready"] = False
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v28_verification": predecessor_verification,
        "runtime_admission_benchmark_verification": benchmark_verification,
        "runtime_admission_builder_sha256": sha256_file(
            runtime_admission_builder_path
        ),
        "runtime_admission_derivation_sha256": sha256_file(Path(__file__)),
        "readonly_join_is_distinct_from_atomic_reservation": True,
        "bounded_predecessor_replay": True,
    }
    result["scientific_boundary"]["synthetic_runtime_admission_fixture_only"] = True
    result["scientific_boundary"]["real_runtime_admission_present"] = False
    result["scientific_boundary"]["persistent_output_root_created"] = False
    result["scientific_boundary"]["capture_started"] = False
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
        raise ValueError("research protocol V29 is not reproducible")
    contract = observed["component_compartments"][
        "cluster_to_sae_runtime_admission_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v29",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "runtime_admission_benchmark_id": contract["benchmark"]["benchmark_id"],
        "runtime_admission_controls": contract["benchmark"]["controls"]["controls"],
        "real_runtime_admission_present": False,
        "persistent_output_root_created": False,
        "capture_started": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v28.json"
        ),
        "runtime_admission_builder_path": (
            project_root
            / "experiments/authz_mechinterp/build_vpd_cluster_to_sae_runtime_admission_v1.py"
        ),
        "runtime_admission_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_to_sae_runtime_admission_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V29.")
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
