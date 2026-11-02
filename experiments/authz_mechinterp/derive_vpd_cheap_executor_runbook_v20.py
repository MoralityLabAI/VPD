from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_bound_content_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v28 import (
    canonical_inputs as protocol_v28_inputs,
    verify_protocol as verify_protocol_v28,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v20"
EXPECTED_V19_RUNBOOK_ID = (
    "74e48f8b1aee138f69135845f31d4ff2a8eda34eba189ff7506f2de0084fff44"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V20 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V19_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V19 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v19":
        raise ValueError("pinned predecessor is not runbook V19")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V19 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v28_path: Path,
    content_verifier_path: Path,
    content_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v28(
        protocol_v28_path, **protocol_v28_inputs(project_root)
    )
    benchmark_verification = verify_benchmark(
        content_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    benchmark = _load(content_benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "streaming_content_verifier_ready_empirical_capture_blocked_on_real_manifest_"
        "bindings_content_receipt_and_runtime_output_gate"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v28"] = {
        "path": _relative(protocol_v28_path, project_root),
        "sha256": sha256_file(protocol_v28_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_to_sae_bound_content_verifier_v1"] = {
        "path": _relative(content_verifier_path, project_root),
        "sha256": sha256_file(content_verifier_path),
    }
    result["source"]["cluster_to_sae_bound_content_benchmark_v1"] = {
        "path": _relative(content_benchmark_path, project_root),
        "sha256": sha256_file(content_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["bound_content_verifier"] = (
        "ready_streaming_exact_tree_and_adapter_digest_checks"
    )
    result["current_result"]["bound_content_benchmark"] = (
        "pass_19_of_19_cycle_tree_mutation_reparse_adapter_and_nonimport_controls"
    )
    result["current_result"]["real_model_manifest_present"] = False
    result["current_result"]["real_bound_content_verification_present"] = False
    result["current_result"]["empirical_capture_ready"] = False

    preflight_stage = result["stages"][0]
    preflight_stage["commands"].extend(
        [
            {
                "id": "verify_cluster_to_sae_bound_content_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_bound_content_v1",
                    "--verify",
                    "artifacts/vpd_cluster_to_sae_bound_content_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v28",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v28",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v28.json",
                ],
            },
        ]
    )
    preflight_stage["expected"]["bound_content_benchmark"] = (
        "19 controls pass using temporary synthetic bytes; no real weight content opens"
    )
    preflight_stage["stop_if"].extend(
        [
            "protocol V28, content verifier, or content benchmark differs",
            "model manifest includes a preflight back-reference or resides inside the addressed model tree",
            "model tree has unlisted, missing, mutated, indirect, duplicate, unsorted, or escaping content",
            "adapter weight or configuration digest differs or adapter weight filename is ambiguous",
            "streaming digest success is treated as model loading or runtime permission",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "model-free content verifier passes synthetic controls; a real external model "
        "manifest must be built first, bound into preflight V2, and reverified before "
        "signed authorization and atomic output-root admission"
    )
    gate["required_one_way_commands"] = [
        {
            "id": "build_real_model_files_manifest_outside_model_root",
            "command_template": (
                ".venv/Scripts/python.exe -m "
                "experiments.authz_mechinterp.verify_vpd_cluster_to_sae_bound_content_v1 "
                "build-model-manifest --model-root <ABSOLUTE_MODEL_ROOT> "
                "--model-id <MODEL_ID> --model-revision <MODEL_REVISION> "
                "--output <MANIFEST_OUTSIDE_MODEL_ROOT>"
            ),
            "authorized_now": False,
        },
        {
            "id": "verify_real_content_against_completed_preflight",
            "command_template": (
                ".venv/Scripts/python.exe -m "
                "experiments.authz_mechinterp.verify_vpd_cluster_to_sae_bound_content_v1 "
                "verify-bound-content --preflight <COMPLETED_PREFLIGHT_V2> "
                "--model-manifest <BOUND_MODEL_MANIFEST>"
            ),
            "authorized_now": False,
        },
    ]
    gate["stop_if"] = list(gate["stop_if"]) + [
        "real model manifest was built after rather than before the preflight binding",
        "real content verification does not exactly reproduce all model and adapter digests",
        "any model framework import or deserialization occurs during byte verification",
    ]
    result["completion_definition"]["streaming_bound_content_verifier_ready"] = True
    result["completion_definition"]["bound_content_benchmark_ready"] = True
    result["completion_definition"]["real_model_manifest_present"] = False
    result["completion_definition"]["real_bound_content_verification_present"] = False
    result["completion_definition"]["empirical_capture_ready"] = False
    result["scientific_boundary"]["synthetic_content_fixture_only"] = True
    result["scientific_boundary"]["real_model_manifest_present"] = False
    result["scientific_boundary"]["real_model_or_adapter_content_opened"] = False
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
        raise ValueError("cheap-executor runbook V20 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v20",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "real_model_manifest_present": False,
        "real_bound_content_verification_present": False,
        "real_signed_authorization_receipt_present": False,
        "empirical_capture_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v19/runbook.json"
        ),
        "protocol_v28_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v28.json"
        ),
        "content_verifier_path": (
            project_root
            / "experiments/authz_mechinterp/verify_vpd_cluster_to_sae_bound_content_v1.py"
        ),
        "content_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_to_sae_bound_content_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive or verify cheap-executor runbook V20."
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
