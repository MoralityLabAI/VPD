from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_runtime_v3 import (
    verify_benchmark as verify_runner_benchmark,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_output_reservation_v1 import (
    verify_benchmark as verify_reservation_benchmark,
)
from experiments.authz_mechinterp.benchmark_vpd_guarded_capture_runtime_adapter_v2 import (
    verify_benchmark as verify_adapter_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v32 import (
    canonical_inputs as protocol_v32_inputs,
    verify_protocol as verify_protocol_v32,
)


SCHEMA = "sch.vpd-activation-category-protocol.v33"
EXPECTED_V32_SHA256 = (
    "59749a0512ddbb0d4c7ea7fc3b91b9269da7cee8857e1e9a9f9bf11d20e13011"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V33 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, *, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V32_SHA256:
        raise ValueError("protocol V32 differs from its pinned SHA-256")
    verification = verify_protocol_v32(path, **protocol_v32_inputs(project_root))
    if verification["execution_authorized"] is not False:
        raise ValueError("protocol V32 authorization boundary differs")
    return verification


def build_protocol(
    *,
    predecessor_path: Path,
    adapter_v2_path: Path,
    adapter_v2_benchmark_path: Path,
    reservation_path: Path,
    reservation_benchmark_path: Path,
    runner_v3_path: Path,
    runner_v3_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(
        predecessor_path, project_root=project_root
    )
    adapter_verification = verify_adapter_benchmark(
        adapter_v2_benchmark_path, project_root=project_root
    )
    reservation_verification = verify_reservation_benchmark(
        reservation_benchmark_path, project_root=project_root
    )
    runner_verification = verify_runner_benchmark(
        runner_v3_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    adapter_benchmark = json.loads(
        adapter_v2_benchmark_path.read_text(encoding="utf-8")
    )
    reservation_benchmark = json.loads(
        reservation_benchmark_path.read_text(encoding="utf-8")
    )
    runner_benchmark = json.loads(
        runner_v3_benchmark_path.read_text(encoding="utf-8")
    )
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["guarded_capture_runtime_adapter_v2"] = {
        "schema": "sch.vpd-guarded-capture-runtime-adapter-contract.v2",
        "status": "reservation_before_runtime_attachment_ready_synthetic_controls_only",
        "adapter": {
            "path": _relative(adapter_v2_path, project_root),
            "sha256": sha256_file(adapter_v2_path),
        },
        "benchmark": {
            "path": _relative(adapter_v2_benchmark_path, project_root),
            "sha256": sha256_file(adapter_v2_benchmark_path),
            "benchmark_id": adapter_benchmark["benchmark_id"],
            "verification": adapter_verification,
            "controls": adapter_benchmark["summary"],
        },
        "job_and_payload_reservation_precedes_runtime_load": True,
        "runtime_attachment_is_single_use": True,
        "failed_callbacks_remain_charged": True,
        "model_loaded": False,
        "execution_authorized": False,
    }
    result["component_compartments"]["output_reservation_transition_v1"] = {
        "schema": "sch.vpd-cluster-to-sae-output-reservation-transition-contract.v1",
        "status": "atomic_transition_ready_real_transition_absent",
        "transition": {
            "path": _relative(reservation_path, project_root),
            "sha256": sha256_file(reservation_path),
        },
        "benchmark": {
            "path": _relative(reservation_benchmark_path, project_root),
            "sha256": sha256_file(reservation_benchmark_path),
            "benchmark_id": reservation_benchmark["benchmark_id"],
            "verification": reservation_verification,
            "controls": reservation_benchmark["summary"],
        },
        "real_admission_must_reverify_before_atomic_create": True,
        "synthetic_admission_rejected_in_real_mode": True,
        "double_reservation_is_terminal": True,
        "real_reservation_receipt_present": False,
        "persistent_output_root_created": False,
        "capture_started": False,
        "execution_authorized": False,
    }
    result["component_compartments"]["canonical_capture_runner_v3"] = {
        "schema": "sch.vpd-canonical-capture-job-runner-contract.v3",
        "status": "guarded_runner_code_ready_synthetic_integration_only_real_inputs_absent",
        "runner": {
            "path": _relative(runner_v3_path, project_root),
            "sha256": sha256_file(runner_v3_path),
            "execution_schema": "sch.vpd-canonical-category-capture-job-execution.v3",
        },
        "benchmark": {
            "path": _relative(runner_v3_benchmark_path, project_root),
            "sha256": sha256_file(runner_v3_benchmark_path),
            "benchmark_id": runner_benchmark["benchmark_id"],
            "verification": runner_verification,
            "controls": runner_benchmark["summary"],
        },
        "reservation_precedes_runtime_load": True,
        "v3_and_predecessor_v2_receipts_required": True,
        "production_receipt_replays_allocation_and_reservation": True,
        "production_runner_code_path_present": True,
        "production_runner_empirically_exercised": False,
        "actual_recovered_shape_capture_plans_present": False,
        "real_reservation_receipt_present": False,
        "capture_started": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    result["unified_lineage"]["guarded_runtime_adapter_v2_ready"] = True
    result["unified_lineage"]["output_reservation_transition_ready"] = True
    result["unified_lineage"]["canonical_capture_runner_v3_code_ready"] = True
    result["unified_lineage"]["canonical_capture_runner_v3_controls"] = runner_benchmark[
        "summary"
    ]["controls"]
    result["unified_lineage"]["production_runner_empirically_exercised"] = False
    result["unified_lineage"]["identity_bindings_complete"] = False
    result["unified_lineage"]["capture_started"] = False
    result["unified_lineage"]["empirical_capture_ready"] = False
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v32_verification": predecessor_verification,
        "adapter_v2_benchmark_verification": adapter_verification,
        "reservation_benchmark_verification": reservation_verification,
        "runner_v3_benchmark_verification": runner_verification,
        "runner_v3_derivation_sha256": sha256_file(Path(__file__)),
        "bounded_predecessor_replay": True,
    }
    result["scientific_boundary"]["synthetic_guarded_runner_fixture_only"] = True
    result["scientific_boundary"]["production_runner_code_path_present"] = True
    result["scientific_boundary"]["production_runner_empirically_exercised"] = False
    result["scientific_boundary"]["actual_capture_plans_present"] = False
    result["scientific_boundary"]["identity_bindings_complete"] = False
    result["scientific_boundary"]["real_reservation_receipt_present"] = False
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
        raise ValueError("research protocol V33 is not reproducible")
    adapter = observed["component_compartments"]["guarded_capture_runtime_adapter_v2"]
    reservation = observed["component_compartments"]["output_reservation_transition_v1"]
    runner = observed["component_compartments"]["canonical_capture_runner_v3"]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v33",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "adapter_v2_benchmark_id": adapter["benchmark"]["benchmark_id"],
        "reservation_benchmark_id": reservation["benchmark"]["benchmark_id"],
        "runner_v3_benchmark_id": runner["benchmark"]["benchmark_id"],
        "runner_v3_controls": runner["benchmark"]["controls"]["controls"],
        "production_runner_code_path_present": True,
        "production_runner_empirically_exercised": False,
        "capture_started": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v32.json"
        ),
        "adapter_v2_path": (
            project_root
            / "experiments/authz_mechinterp/vpd_guarded_capture_runtime_adapter_v2.py"
        ),
        "adapter_v2_benchmark_path": (
            project_root
            / "artifacts/vpd_guarded_capture_runtime_adapter_benchmark_v2/report.json"
        ),
        "reservation_path": (
            project_root
            / "experiments/authz_mechinterp/vpd_cluster_to_sae_output_reservation_v1.py"
        ),
        "reservation_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_to_sae_output_reservation_benchmark_v1/report.json"
        ),
        "runner_v3_path": (
            project_root
            / "experiments/authz_mechinterp/run_canonical_vpd_capture_job_v3.py"
        ),
        "runner_v3_benchmark_path": (
            project_root
            / "artifacts/vpd_canonical_capture_job_runtime_benchmark_v3/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V33.")
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
