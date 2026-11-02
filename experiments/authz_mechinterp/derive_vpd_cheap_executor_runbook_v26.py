from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_capture_router_adapter_v2 import (
    verify_benchmark as verify_adapter_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v34 import (
    canonical_inputs as protocol_v34_inputs,
    verify_protocol as verify_protocol_v34,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v26"
EXPECTED_V25_RUNBOOK_ID = (
    "a6ba3497fc808e65305a096b113c0e141cdb03094296c42d0f58ba099da3f08a"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V26 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V25_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V25 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v25":
        raise ValueError("pinned predecessor is not runbook V25")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V25 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v34_path: Path,
    adapter_v2_path: Path,
    adapter_v2_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v34(
        protocol_v34_path, **protocol_v34_inputs(project_root)
    )
    adapter_verification = verify_adapter_benchmark(
        adapter_v2_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    benchmark = _load(adapter_v2_benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "guarded_v3_capture_to_router_lineage_ready_synthetic_controls_pass_"
        "empirical_capture_and_routing_absent"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v34"] = {
        "path": _relative(protocol_v34_path, project_root),
        "sha256": sha256_file(protocol_v34_path),
        "verification": protocol_verification,
    }
    result["source"]["guarded_capture_router_adapter_v2"] = {
        "path": _relative(adapter_v2_path, project_root),
        "sha256": sha256_file(adapter_v2_path),
    }
    result["source"]["guarded_capture_router_adapter_benchmark_v2"] = {
        "path": _relative(adapter_v2_benchmark_path, project_root),
        "sha256": sha256_file(adapter_v2_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": adapter_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["guarded_capture_router_adapter_v2"] = (
        "pass_18_of_18_v3_guard_and_payload_identity_preserved_into_router"
    )
    result["current_result"]["v3_capture_identity_preserved_into_router"] = True
    result["current_result"]["production_router_adapter_empirically_exercised"] = False
    result["current_result"]["persistent_empirical_router_receipts_present"] = False
    result["current_result"]["identity_bindings_complete"] = False
    result["current_result"]["actual_recovered_shape_capture_plans_present"] = False
    result["current_result"]["capture_started"] = False
    result["current_result"]["empirical_capture_ready"] = False

    preflight_stage = result["stages"][0]
    preflight_stage["commands"].extend(
        [
            {
                "id": "verify_guarded_capture_router_adapter_benchmark_v2",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_capture_router_adapter_v2",
                    "--verify",
                    "artifacts/vpd_capture_router_adapter_benchmark_v2/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v34",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v34",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v34.json",
                ],
            },
        ]
    )
    preflight_stage["expected"]["guarded_capture_router_adapter_v2"] = (
        "18 controls pass; router-facing capture identity is guarded V3, while V2 payload and guard IDs remain bound"
    )
    preflight_stage["stop_if"].extend(
        [
            "protocol V34 or guarded capture-to-router benchmark differs",
            "a router receipt binds only the predecessor V2 execution identity",
            "V3 allocation, reservation, or guard lineage is omitted before routing",
            "a synthetic adapter receipt is described as empirical routing evidence",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "guarded V3 capture-to-router lineage bridge passes synthetic controls; "
        "identity-complete plans, authorized empirical captures, and persistent router "
        "receipts remain absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "post-capture routing begins from an unverified V3 wrapper",
        "a V2 payload receipt is substituted for the router-facing V3 identity",
        "adapter readiness is treated as authority to capture, route, or train",
    ]
    result["completion_definition"]["guarded_capture_router_adapter_v2_ready"] = True
    result["completion_definition"]["v3_capture_identity_preserved_into_router"] = True
    result["completion_definition"]["production_router_adapter_empirically_exercised"] = False
    result["completion_definition"]["persistent_empirical_router_receipts_present"] = False
    result["completion_definition"]["identity_bindings_complete"] = False
    result["completion_definition"]["actual_capture_plans_present"] = False
    result["completion_definition"]["capture_started"] = False
    result["completion_definition"]["empirical_capture_ready"] = False
    result["scientific_boundary"]["synthetic_v3_to_router_fixture_only"] = True
    result["scientific_boundary"]["v3_capture_identity_preserved_into_router"] = True
    result["scientific_boundary"]["production_router_adapter_empirically_exercised"] = False
    result["scientific_boundary"]["persistent_empirical_router_receipts_present"] = False
    result["scientific_boundary"]["model_framework_imported"] = False
    result["scientific_boundary"]["model_or_adapter_weights_opened"] = False
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["sae_training_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V26 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v26",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "guarded_capture_router_adapter_v2_ready": True,
        "v3_capture_identity_preserved_into_router": True,
        "production_router_adapter_empirically_exercised": False,
        "persistent_empirical_router_receipts_present": False,
        "identity_bindings_complete": False,
        "capture_started": False,
        "empirical_capture_ready": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v25/runbook.json"
        ),
        "protocol_v34_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v34.json"
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
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V26.")
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
