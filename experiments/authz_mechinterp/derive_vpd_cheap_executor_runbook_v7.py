from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_authenticated_lora_factor_bundle_extraction_v1 import (
    verify_benchmark as verify_extraction_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v15 import (
    canonical_inputs as protocol_v15_inputs,
    verify_protocol as verify_protocol_v15,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v7"
EXPECTED_V6_RUNBOOK_ID = (
    "8b887d5dc583ed85b83cf597bbdaef0a13f1fde3ecbafe1045b2477f4226da76"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V7 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_pinned_runbook_v6(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = body.pop("runbook_id", "")
    if identifier != EXPECTED_V6_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V6 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v6":
        raise ValueError("pinned predecessor is not runbook V6")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V6 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "method": "content_address_and_boundary_fields",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v15_path: Path,
    extraction_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_pinned_runbook_v6(predecessor_path)
    protocol_verification = verify_protocol_v15(
        protocol_v15_path, **protocol_v15_inputs(project_root)
    )
    benchmark_verification = verify_extraction_benchmark(extraction_benchmark_path)
    predecessor = _load(predecessor_path)
    protocol = _load(protocol_v15_path)
    benchmark = _load(extraction_benchmark_path)
    contract = protocol["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "ready_for_local_preflight_real_factor_receipt_and_torch_integration_pending"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v15"] = {
        "path": _relative(protocol_v15_path, project_root),
        "sha256": sha256_file(protocol_v15_path),
        "verification": protocol_verification,
    }
    result["source"]["authenticated_lora_factor_extraction_benchmark_v1"] = {
        "path": _relative(extraction_benchmark_path, project_root),
        "sha256": sha256_file(extraction_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["authenticated_lora_factor_extractor_v1"] = contract[
        "authenticated_factor_bundle_extractor"
    ]
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["authenticated_factor_extractor_benchmark"] = "pass"
    result["current_result"]["real_adapter_factor_bundle_receipt_present"] = False
    result["current_result"]["production_torch_peft_integration_present"] = False
    preflight = result["stages"][0]
    preflight["commands"].append(
        {
            "id": "verify_authenticated_lora_factor_extraction_benchmark_v1",
            "executable": ".venv/Scripts/python.exe",
            "arguments": [
                "-m",
                "experiments.authz_mechinterp.benchmark_authenticated_lora_factor_bundle_extraction_v1",
                "--verify",
                "artifacts/vpd_authenticated_lora_factor_bundle_extraction_benchmark_v1/report.json",
            ],
        }
    )
    preflight["expected"]["authenticated_factor_extraction_benchmark"] = (
        "exact factors/scales, canonical inventory chain, seven negative controls"
    )
    preflight["stop_if"].append(
        "protocol V15 or authenticated factor-extraction evidence differs"
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "authenticated plain-LoRA safetensors extractor, factor-bundle canonical "
        "inventory, and model-free NumPy restoration runtime implemented; real-adapter "
        "extraction receipt and Torch/PEFT canonical effect-capture integration absent"
    )
    gate["available_authenticated_extraction_command_template"] = {
        "executable": ".venv/Scripts/python.exe",
        "arguments": [
            "-m",
            "experiments.authz_mechinterp.extract_authenticated_lora_factor_bundle_v1",
            "--adapter-weights",
            "<AUTHORIZED_ADAPTER_MODEL_SAFETENSORS>",
            "--adapter-config",
            "<AUTHORIZED_ADAPTER_CONFIG_JSON>",
            "--expected-adapter-sha256",
            "<EXACT_EXPECTED_ADAPTER_SHA256>",
            "--expected-config-sha256",
            "<EXACT_EXPECTED_CONFIG_SHA256>",
            "--output-dir",
            "<AUTHENTICATED_FACTOR_BUNDLE_DIR>",
        ],
    }
    gate["available_model_free_command_template"] = {
        "executable": ".venv/Scripts/python.exe",
        "arguments": [
            "-m",
            "experiments.authz_mechinterp.emit_canonical_lora_component_inventory_v1",
            "--factor-index",
            "<AUTHENTICATED_FACTOR_BUNDLE_DIR>/factor-index.json",
            "--output-dir",
            "<CANONICAL_COMPONENT_INVENTORY_DIR>",
        ],
    }
    gate["stop_if"] = [
        "raw LoRA factor rank is used as the component identity",
        "explicit user authorization to access the real adapter files is absent",
        "adapter or config expected SHA-256 is absent or differs",
        "plain-LoRA extraction encounters DoRA, bias, modules_to_save, fan-in/fan-out, unexpected tensors, or ambiguous lanes",
        "authenticated real-adapter factor-bundle extraction receipt is absent",
        "production Torch/PEFT canonical effect-capture integration is absent",
        "spectral-gap tolerance is not frozen and benchmarked",
        "any source hash, factor pairing, scale, baseline reconstruction, probe, or restoration check fails",
    ]
    result["completion_definition"][
        "authenticated_factor_extractor_implementation_ready"
    ] = True
    result["completion_definition"]["real_adapter_factor_bundle_receipt_ready"] = False
    result["completion_definition"]["canonical_torch_peft_capture_runtime_ready"] = False
    result["scientific_boundary"]["real_adapter_access_authorized"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = body.pop("runbook_id", "")
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V7 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v7",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "authenticated_factor_extractor_implementation_ready": True,
        "real_adapter_factor_bundle_receipt_ready": False,
        "canonical_torch_peft_capture_runtime_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v6/runbook.json"
        ),
        "protocol_v15_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v15.json"
        ),
        "extraction_benchmark_path": (
            project_root
            / "artifacts/vpd_authenticated_lora_factor_bundle_extraction_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V7.")
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
