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
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v27 import (
    canonical_inputs as protocol_v27_inputs,
    verify_protocol as verify_protocol_v27,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v19"
EXPECTED_V18_RUNBOOK_ID = (
    "c96df7676f62501531c33dbb259b8c5e846f8683b3d0e11b756d4d48ca8b140f"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V19 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V18_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V18 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v18":
        raise ValueError("pinned predecessor is not runbook V18")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V18 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v27_path: Path,
    readiness_path: Path,
    readiness_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v27(
        protocol_v27_path, **protocol_v27_inputs(project_root)
    )
    readiness_verification = verify_readiness(readiness_path, project_root=project_root)
    benchmark_verification = verify_benchmark(
        readiness_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    readiness = _load(readiness_path)
    benchmark = _load(readiness_benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "binding_readiness_verified_empirical_capture_blocked_on_10_real_bindings_"
        "runtime_content_verification_and_real_signed_receipt"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v27"] = {
        "path": _relative(protocol_v27_path, project_root),
        "sha256": sha256_file(protocol_v27_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_to_sae_binding_readiness_v1"] = {
        "path": _relative(readiness_path, project_root),
        "sha256": sha256_file(readiness_path),
        "readiness_id": readiness["readiness_id"],
        "verification": readiness_verification,
    }
    result["source"]["cluster_to_sae_binding_readiness_benchmark_v1"] = {
        "path": _relative(readiness_benchmark_path, project_root),
        "sha256": sha256_file(readiness_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["binding_readiness_template"] = (
        "pass_10_fields_missing_no_model_or_adapter_content_opened"
    )
    result["current_result"]["binding_readiness_benchmark"] = (
        "pass_17_of_17_path_device_digest_and_non_authorization_controls"
    )
    result["current_result"]["real_runtime_bindings_present"] = False
    result["current_result"]["output_isolation_verified_for_real_run"] = False
    result["current_result"]["real_signed_authorization_receipt_present"] = False
    result["current_result"]["empirical_capture_ready"] = False

    preflight_stage = result["stages"][0]
    preflight_stage["commands"].extend(
        [
            {
                "id": "verify_cluster_to_sae_binding_readiness_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1",
                    "--verify",
                    "artifacts/vpd_cluster_to_sae_binding_readiness_v1/readiness.json",
                ],
            },
            {
                "id": "verify_cluster_to_sae_binding_readiness_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_binding_readiness_v1",
                    "--verify",
                    "artifacts/vpd_cluster_to_sae_binding_readiness_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v27",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v27",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v27.json",
                ],
            },
        ]
    )
    preflight_stage["expected"]["binding_readiness"] = (
        "10 fields are explicit; real values absent; output-root policy pinned"
    )
    preflight_stage["expected"]["binding_readiness_benchmark"] = (
        "17 controls pass; complete synthetic values remain non-authorizing"
    )
    preflight_stage["stop_if"].extend(
        [
            "protocol V27, binding worksheet, or binding benchmark differs",
            "output root is relative, noncanonical, pre-existing, nested, escaped, or crosses a reparse point",
            "adapter path is relative or noncanonical",
            "device is not cpu, cuda, or cuda:N",
            "binding syntax or isolation success is treated as model-content verification or runtime permission",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "cycle-free identity, signed-receipt, and canonical binding-isolation "
        "validators pass synthetic controls; real bindings, independent content "
        "verification, and a real authenticated receipt remain absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "model manifest or adapter digests are not independently checked before loading",
        "runtime output root is not atomically created as a new direct child",
        "any write is attempted outside the authorized output root",
    ]
    result["completion_definition"]["binding_readiness_worksheet_ready"] = True
    result["completion_definition"]["binding_readiness_benchmark_ready"] = True
    result["completion_definition"]["canonical_runtime_bindings_present"] = False
    result["completion_definition"]["real_output_isolation_verified"] = False
    result["completion_definition"]["real_signed_authorization_receipt_present"] = False
    result["completion_definition"]["empirical_capture_ready"] = False
    result["scientific_boundary"]["synthetic_binding_fixture_only"] = True
    result["scientific_boundary"]["real_runtime_bindings_present"] = False
    result["scientific_boundary"]["output_root_created"] = False
    result["scientific_boundary"]["real_authorization_receipt_present"] = False
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
        raise ValueError("cheap-executor runbook V19 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v19",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "missing_execution_identity_bindings": 10,
        "output_isolation_verified_for_real_run": False,
        "real_signed_authorization_receipt_present": False,
        "empirical_capture_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v18/runbook.json"
        ),
        "protocol_v27_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v27.json"
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
    parser = argparse.ArgumentParser(
        description="Derive or verify cheap-executor runbook V19."
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
