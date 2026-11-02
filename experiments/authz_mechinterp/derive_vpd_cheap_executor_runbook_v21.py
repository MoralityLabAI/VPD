from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_runtime_admission_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v29 import (
    canonical_inputs as protocol_v29_inputs,
    verify_protocol as verify_protocol_v29,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v21"
EXPECTED_V20_RUNBOOK_ID = (
    "0586ec6de4f355743e2f3372c4642140de23ee6329deb6739cb156436d68def2"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V21 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V20_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V20 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v20":
        raise ValueError("pinned predecessor is not runbook V20")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V20 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v29_path: Path,
    runtime_admission_builder_path: Path,
    runtime_admission_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v29(
        protocol_v29_path, **protocol_v29_inputs(project_root)
    )
    benchmark_verification = verify_benchmark(
        runtime_admission_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    benchmark = _load(runtime_admission_benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "runtime_admission_join_ready_empirical_capture_blocked_on_real_inputs_"
        "atomic_reservation_and_capture_authority"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v29"] = {
        "path": _relative(protocol_v29_path, project_root),
        "sha256": sha256_file(protocol_v29_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_to_sae_runtime_admission_builder_v1"] = {
        "path": _relative(runtime_admission_builder_path, project_root),
        "sha256": sha256_file(runtime_admission_builder_path),
    }
    result["source"]["cluster_to_sae_runtime_admission_benchmark_v1"] = {
        "path": _relative(runtime_admission_benchmark_path, project_root),
        "sha256": sha256_file(runtime_admission_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["runtime_admission_join"] = (
        "ready_four_proofs_join_same_preflight"
    )
    result["current_result"]["runtime_admission_benchmark"] = (
        "pass_12_of_12_join_tamper_nonce_expiry_and_atomic_reservation_controls"
    )
    result["current_result"]["real_runtime_admission_present"] = False
    result["current_result"]["persistent_output_root_created"] = False
    result["current_result"]["capture_started"] = False
    result["current_result"]["empirical_capture_ready"] = False

    preflight_stage = result["stages"][0]
    preflight_stage["commands"].extend(
        [
            {
                "id": "verify_cluster_to_sae_runtime_admission_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_runtime_admission_v1",
                    "--verify",
                    "artifacts/vpd_cluster_to_sae_runtime_admission_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v29",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v29",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v29.json",
                ],
            },
        ]
    )
    preflight_stage["expected"]["runtime_admission_benchmark"] = (
        "12 controls pass; synthetic receipts cannot reserve; no capture starts"
    )
    preflight_stage["stop_if"].extend(
        [
            "protocol V29, runtime admission builder, or admission benchmark differs",
            "binding, byte, authorization, and output checks do not join the same preflight",
            "synthetic authorization is permitted to reserve an output root",
            "existing output root is treated as resumable rather than terminal fail-closed",
            "reservation is treated as capture start or SAE-training authority",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "read-only four-proof admission join and atomic reservation primitive pass "
        "synthetic controls; real inputs, real receipt, explicit reservation, and "
        "capture start remain absent"
    )
    gate["required_one_way_commands"].append(
        {
            "id": "build_readonly_real_runtime_admission",
            "command_template": (
                ".venv/Scripts/python.exe -m "
                "experiments.authz_mechinterp.build_vpd_cluster_to_sae_runtime_admission_v1 "
                "--output <RUNTIME_ADMISSION_JSON> --preflight <COMPLETED_PREFLIGHT_V2> "
                "--model-manifest <BOUND_MODEL_MANIFEST> "
                "--authorization-receipt <REAL_SIGNED_RECEIPT> "
                "--authority-registry <REAL_REGISTRY> --as-of <UTC_TIMESTAMP>"
            ),
            "authorized_now": False,
        }
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "runtime admission receipt is synthetic or does not join all checks",
        "output root appeared between read-only admission and atomic reservation",
        "atomic reservation is retried after a crash or existing-root failure",
    ]
    result["completion_definition"]["runtime_admission_join_ready"] = True
    result["completion_definition"]["runtime_admission_benchmark_ready"] = True
    result["completion_definition"]["real_runtime_admission_present"] = False
    result["completion_definition"]["persistent_output_root_created"] = False
    result["completion_definition"]["capture_started"] = False
    result["completion_definition"]["empirical_capture_ready"] = False
    result["scientific_boundary"]["synthetic_runtime_admission_fixture_only"] = True
    result["scientific_boundary"]["real_runtime_admission_present"] = False
    result["scientific_boundary"]["persistent_output_root_created"] = False
    result["scientific_boundary"]["capture_started"] = False
    result["scientific_boundary"]["model_framework_imported"] = False
    result["scientific_boundary"]["model_or_adapter_weights_opened"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["sae_training_authorized"] = False
    result["scientific_boundary"]["final_evaluation_consumed"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V21 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v21",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "real_runtime_admission_present": False,
        "persistent_output_root_created": False,
        "capture_started": False,
        "empirical_capture_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v20/runbook.json"
        ),
        "protocol_v29_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v29.json"
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
    parser = argparse.ArgumentParser(
        description="Derive or verify cheap-executor runbook V21."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.output is not None:
        result = build_runbook(**inputs)
        write_json(args.output, result)
    else:
        result = verify_runbook(args.verify, **inputs)
    print(
        json.dumps(
            {key: result[key] for key in ("schema", "status", "runbook_id")},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
