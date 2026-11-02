from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_canonical_vpd_category_effect_capture_v1 import (
    verify_benchmark as verify_category_capture_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v17 import (
    canonical_inputs as protocol_v17_inputs,
    verify_protocol as verify_protocol_v17,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v9"
EXPECTED_V8_RUNBOOK_ID = (
    "2dd26b4a7ceab0c96e9ac8530180fe0ff81cff9be7d6e39a79db655ad8222f6e"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V9 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_pinned_runbook_v8(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = body.pop("runbook_id", "")
    if identifier != EXPECTED_V8_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V8 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v8":
        raise ValueError("pinned predecessor is not runbook V8")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V8 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "method": "content_address_and_boundary_fields",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v17_path: Path,
    category_capture_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_pinned_runbook_v8(predecessor_path)
    protocol_verification = verify_protocol_v17(
        protocol_v17_path, **protocol_v17_inputs(project_root)
    )
    benchmark_verification = verify_category_capture_benchmark(
        category_capture_benchmark_path
    )
    predecessor = _load(predecessor_path)
    protocol = _load(protocol_v17_path)
    benchmark = _load(category_capture_benchmark_path)
    contract = protocol["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "ready_for_local_preflight_successor_effect_worker_and_real_receipts_pending"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v17"] = {
        "path": _relative(protocol_v17_path, project_root),
        "sha256": sha256_file(protocol_v17_path),
        "verification": protocol_verification,
    }
    result["source"]["canonical_category_effect_capture_benchmark_v1"] = {
        "path": _relative(category_capture_benchmark_path, project_root),
        "sha256": sha256_file(category_capture_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["canonical_category_effect_capture_v1"] = contract[
        "canonical_category_effect_capture_kernel"
    ]
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["canonical_category_effect_capture_benchmark"] = "pass"
    result["current_result"]["successor_effect_worker_cli_integration_present"] = False
    result["current_result"]["actual_model_category_effect_capture_verified"] = False
    preflight = result["stages"][0]
    preflight["commands"].append(
        {
            "id": "verify_canonical_category_effect_capture_benchmark_v1",
            "executable": ".venv/Scripts/python.exe",
            "arguments": [
                "-m",
                "experiments.authz_mechinterp.benchmark_canonical_vpd_category_effect_capture_v1",
                "--verify",
                "artifacts/vpd_canonical_category_effect_capture_benchmark_v1/report.json",
            ],
        }
    )
    preflight["expected"]["canonical_category_effect_capture_benchmark"] = (
        "two category effects exact, per-example rows retained, five negative controls pass"
    )
    preflight["stop_if"].append(
        "protocol V17 or canonical category-effect capture evidence differs"
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "authenticated extractor, canonical inventory, PEFT bridge, and category-preserving "
        "central-difference capture kernel implemented with fake objects; successor worker "
        "CLI, prompt/split identity binding, condition/template sketches, and actual model "
        "receipts absent"
    )
    gate["stop_if"] = [
        "raw LoRA factor rank is used as the component identity",
        "explicit user authorization to access real adapter or model files is absent",
        "authenticated real-adapter factor-bundle extraction receipt is absent",
        "authorized actual-Torch/PEFT canonical runtime smoke receipt is absent",
        "new successor canonical effect-worker CLI is absent",
        "successor worker does not bind exact prompt order, dataset hash, split manifest, and category row identities",
        "successor worker does not emit condition and template CountSketch sufficient statistics for semantic-strata V3",
        "any source hash, module identity, factor scale, lane shape, probe, callback, or restoration check fails",
    ]
    result["completion_definition"]["canonical_category_capture_kernel_ready"] = True
    result["completion_definition"]["successor_effect_worker_cli_ready"] = False
    result["completion_definition"]["worker_batch_and_split_identity_ready"] = False
    result["completion_definition"]["worker_semantic_strata_sketches_ready"] = False
    result["completion_definition"]["actual_model_category_capture_ready"] = False
    result["scientific_boundary"]["real_adapter_access_authorized"] = False
    result["scientific_boundary"]["model_or_torch_execution_authorized"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = body.pop("runbook_id", "")
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V9 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v9",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "canonical_category_capture_kernel_ready": True,
        "successor_effect_worker_cli_ready": False,
        "worker_semantic_strata_sketches_ready": False,
        "actual_model_category_capture_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v8/runbook.json"
        ),
        "protocol_v17_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v17.json"
        ),
        "category_capture_benchmark_path": (
            project_root
            / "artifacts/vpd_canonical_category_effect_capture_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V9.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None:
        result = verify_runbook(args.verify, **inputs)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_runbook(**inputs)
        write_json(args.output, result)
    print(
        json.dumps(
            {key: result[key] for key in ("schema", "status", "runbook_id")},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
