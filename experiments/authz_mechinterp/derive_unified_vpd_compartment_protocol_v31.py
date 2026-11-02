from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_guarded_capture_runtime_adapter_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v30 import (
    canonical_inputs as protocol_v30_inputs,
    verify_protocol as verify_protocol_v30,
)


SCHEMA = "sch.vpd-activation-category-protocol.v31"
EXPECTED_V30_SHA256 = (
    "6e231ace77b43b154e6622da1ee62017a02fbfebc44fbd2ad6b80720a9f47925"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V31 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, *, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V30_SHA256:
        raise ValueError("protocol V30 differs from its pinned SHA-256")
    verification = verify_protocol_v30(path, **protocol_v30_inputs(project_root))
    if verification["execution_authorized"] is not False:
        raise ValueError("protocol V30 authorization boundary differs")
    return verification


def build_protocol(
    *,
    predecessor_path: Path,
    adapter_path: Path,
    adapter_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(
        predecessor_path, project_root=project_root
    )
    benchmark_verification = verify_benchmark(
        adapter_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    benchmark = json.loads(adapter_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["guarded_capture_runtime_adapter_v1"] = {
        "schema": "sch.vpd-guarded-capture-runtime-adapter-contract.v1",
        "status": "synthetic_callback_adapter_ready_production_runner_integration_absent",
        "adapter": {
            "path": _relative(adapter_path, project_root),
            "sha256": sha256_file(adapter_path),
            "job_receipt_schema": "sch.vpd-guarded-capture-runtime-job-receipt.v1",
        },
        "benchmark": {
            "path": _relative(adapter_benchmark_path, project_root),
            "sha256": sha256_file(adapter_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
        },
        "job_and_payload_reserved_before_first_callback": True,
        "forward_and_actual_episode_counters_increment_before_callback": True,
        "failed_callbacks_remain_charged": True,
        "per_job_overage_stops_before_callback": True,
        "exact_job_completion_receipt_required": True,
        "job_receipt_does_not_finalize_role_guard": True,
        "canonical_capture_worker_v2_modified": False,
        "production_runner_integration_present": False,
        "real_guard_receipt_present": False,
        "capture_started": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    result["unified_lineage"]["guarded_runtime_adapter_ready"] = True
    result["unified_lineage"]["guarded_runtime_adapter_controls"] = benchmark[
        "summary"
    ]["controls"]
    result["unified_lineage"]["production_runner_integration_present"] = False
    result["unified_lineage"]["real_guard_receipt_present"] = False
    result["unified_lineage"]["capture_started"] = False
    result["unified_lineage"]["empirical_capture_ready"] = False
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v30_verification": predecessor_verification,
        "guarded_runtime_adapter_benchmark_verification": benchmark_verification,
        "guarded_runtime_adapter_sha256": sha256_file(adapter_path),
        "guarded_runtime_adapter_derivation_sha256": sha256_file(Path(__file__)),
        "historical_capture_worker_left_unchanged": True,
        "bounded_predecessor_replay": True,
    }
    result["scientific_boundary"]["synthetic_guarded_callbacks_only"] = True
    result["scientific_boundary"]["production_runner_integration_present"] = False
    result["scientific_boundary"]["real_guard_receipt_present"] = False
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
        raise ValueError("research protocol V31 is not reproducible")
    contract = observed["component_compartments"]["guarded_capture_runtime_adapter_v1"]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v31",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "adapter_benchmark_id": contract["benchmark"]["benchmark_id"],
        "adapter_controls": contract["benchmark"]["controls"]["controls"],
        "production_runner_integration_present": False,
        "capture_started": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v30.json"
        ),
        "adapter_path": (
            project_root
            / "experiments/authz_mechinterp/vpd_guarded_capture_runtime_adapter_v1.py"
        ),
        "adapter_benchmark_path": (
            project_root
            / "artifacts/vpd_guarded_capture_runtime_adapter_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V31.")
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
