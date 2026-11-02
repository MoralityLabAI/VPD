from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_canonical_vpd_effect_shard_assembly_v1 import (
    verify_benchmark as verify_assembly_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v18 import (
    canonical_inputs as protocol_v18_inputs,
    verify_protocol as verify_protocol_v18,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v10"
EXPECTED_V9_RUNBOOK_ID = (
    "f6fb6fec15440bdc5c307801ec361afa0b40f50b136516c0c3c94b18410d310b"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V10 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_pinned_runbook_v9(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = body.pop("runbook_id", "")
    if identifier != EXPECTED_V9_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V9 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v9":
        raise ValueError("pinned predecessor is not runbook V9")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V9 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "method": "content_address_and_boundary_fields",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v18_path: Path,
    assembly_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_pinned_runbook_v9(predecessor_path)
    protocol_verification = verify_protocol_v18(
        protocol_v18_path, **protocol_v18_inputs(project_root)
    )
    benchmark_verification = verify_assembly_benchmark(assembly_benchmark_path)
    predecessor = _load(predecessor_path)
    protocol = _load(protocol_v18_path)
    benchmark = _load(assembly_benchmark_path)
    contract = protocol["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "ready_for_local_preflight_successor_model_loader_and_real_capture_pending"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v18"] = {
        "path": _relative(protocol_v18_path, project_root),
        "sha256": sha256_file(protocol_v18_path),
        "verification": protocol_verification,
    }
    result["source"]["canonical_effect_shard_assembly_benchmark_v1"] = {
        "path": _relative(assembly_benchmark_path, project_root),
        "sha256": sha256_file(assembly_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["canonical_effect_shard_assembler_v1"] = contract[
        "canonical_effect_shard_assembler"
    ]
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["canonical_effect_shard_assembly_benchmark"] = "pass"
    result["current_result"]["model_free_successor_worker_assembler_present"] = True
    result["current_result"]["successor_worker_model_loader_present"] = False
    result["current_result"]["actual_model_category_effect_capture_verified"] = False
    preflight = result["stages"][0]
    preflight["commands"].append(
        {
            "id": "verify_canonical_effect_shard_assembly_benchmark_v1",
            "executable": ".venv/Scripts/python.exe",
            "arguments": [
                "-m",
                "experiments.authz_mechinterp.benchmark_canonical_vpd_effect_shard_assembly_v1",
                "--verify",
                "artifacts/vpd_canonical_effect_shard_assembly_benchmark_v1/report.json",
            ],
        }
    )
    preflight["expected"]["canonical_effect_shard_assembly_benchmark"] = (
        "V5-compatible category/uncertainty/condition/template arrays; nine controls pass"
    )
    preflight["stop_if"].append(
        "protocol V18 or canonical effect-shard assembly evidence differs"
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "authenticated extractor, canonical inventory, PEFT bridge, category capture kernel, "
        "and provenance-complete model-free archive assembler implemented; successor model "
        "loader/capture CLI and actual model receipts absent"
    )
    gate["stop_if"] = [
        "raw LoRA factor rank is used as the component identity",
        "explicit user authorization to access real adapter or model files is absent",
        "authenticated real-adapter factor-bundle extraction receipt is absent",
        "authorized actual-Torch/PEFT canonical runtime smoke receipt is absent",
        "successor worker model loader and capture CLI is absent",
        "actual worker does not call the canonical category capture kernel for every inventory component",
        "actual worker does not pass verified V2/V3 split provenance and exact sorted row metadata to the assembler",
        "any dataset, split, capture receipt, component, category, pair, condition, template, sketch, artifact, or restoration check fails",
    ]
    result["completion_definition"]["model_free_successor_worker_assembler_ready"] = True
    result["completion_definition"]["worker_batch_and_split_identity_ready"] = True
    result["completion_definition"]["worker_semantic_strata_sketches_ready"] = True
    result["completion_definition"]["successor_worker_model_loader_ready"] = False
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
        raise ValueError("cheap-executor runbook V10 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v10",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "model_free_successor_worker_assembler_ready": True,
        "worker_semantic_strata_sketches_ready": True,
        "successor_worker_model_loader_ready": False,
        "actual_model_category_capture_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v9/runbook.json"
        ),
        "protocol_v18_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v18.json"
        ),
        "assembly_benchmark_path": (
            project_root
            / "artifacts/vpd_canonical_effect_shard_assembly_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V10.")
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
