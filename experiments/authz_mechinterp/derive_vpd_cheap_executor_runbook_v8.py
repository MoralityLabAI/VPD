from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_canonical_lora_peft_runtime_v1 import (
    verify_benchmark as verify_peft_runtime_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v16 import (
    canonical_inputs as protocol_v16_inputs,
    verify_protocol as verify_protocol_v16,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v8"
EXPECTED_V7_RUNBOOK_ID = (
    "b7beb4b1ad9a88465d9f38333a229da0c25792005c30de7da907c6d8d769627f"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V8 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_pinned_runbook_v7(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = body.pop("runbook_id", "")
    if identifier != EXPECTED_V7_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V7 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v7":
        raise ValueError("pinned predecessor is not runbook V7")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V7 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "method": "content_address_and_boundary_fields",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v16_path: Path,
    peft_runtime_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_pinned_runbook_v7(predecessor_path)
    protocol_verification = verify_protocol_v16(
        protocol_v16_path, **protocol_v16_inputs(project_root)
    )
    benchmark_verification = verify_peft_runtime_benchmark(
        peft_runtime_benchmark_path
    )
    predecessor = _load(predecessor_path)
    protocol = _load(protocol_v16_path)
    benchmark = _load(peft_runtime_benchmark_path)
    contract = protocol["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "ready_for_local_preflight_real_and_torch_receipts_and_worker_integration_pending"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v16"] = {
        "path": _relative(protocol_v16_path, project_root),
        "sha256": sha256_file(protocol_v16_path),
        "verification": protocol_verification,
    }
    result["source"]["canonical_lora_peft_runtime_benchmark_v1"] = {
        "path": _relative(peft_runtime_benchmark_path, project_root),
        "sha256": sha256_file(peft_runtime_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["canonical_lora_peft_runtime_v1"] = contract[
        "canonical_peft_factor_runtime"
    ]
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["canonical_fake_peft_runtime_benchmark"] = "pass"
    result["current_result"]["actual_torch_peft_runtime_execution_verified"] = False
    result["current_result"][
        "effect_capture_worker_canonical_runtime_integration_present"
    ] = False
    preflight = result["stages"][0]
    preflight["commands"].append(
        {
            "id": "verify_canonical_lora_peft_runtime_benchmark_v1",
            "executable": ".venv/Scripts/python.exe",
            "arguments": [
                "-m",
                "experiments.authz_mechinterp.benchmark_canonical_lora_peft_runtime_v1",
                "--verify",
                "artifacts/vpd_canonical_lora_peft_runtime_benchmark_v1/report.json",
            ],
        }
    )
    preflight["expected"]["canonical_lora_peft_runtime_benchmark"] = (
        "four fake-PEFT multipliers restore exactly; five negative controls pass"
    )
    preflight["stop_if"].append(
        "protocol V16 or canonical PEFT runtime evidence differs"
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "authenticated plain-LoRA extractor, canonical inventory, NumPy runtime, "
        "and lazy-Torch PEFT factor bridge implemented; only fake PEFT execution "
        "verified, with real-adapter receipt, authorized Torch smoke receipt, and "
        "effect-worker integration absent"
    )
    gate["stop_if"] = [
        "raw LoRA factor rank is used as the component identity",
        "explicit user authorization to access real adapter or model files is absent",
        "adapter or config expected SHA-256 is absent or differs",
        "authenticated real-adapter factor-bundle extraction receipt is absent",
        "authorized actual-Torch/PEFT canonical runtime smoke receipt is absent",
        "canonical PEFT runtime is not integrated into a new successor effect-capture worker",
        "spectral-gap tolerance is not frozen and benchmarked",
        "any source hash, module identity, factor pairing, scale, probe, callback, or restoration check fails",
    ]
    result["completion_definition"]["canonical_peft_bridge_implementation_ready"] = True
    result["completion_definition"]["actual_torch_peft_runtime_smoke_ready"] = False
    result["completion_definition"]["canonical_effect_worker_integration_ready"] = False
    result["completion_definition"]["canonical_torch_peft_capture_runtime_ready"] = False
    result["scientific_boundary"]["real_adapter_access_authorized"] = False
    result["scientific_boundary"]["model_or_torch_execution_authorized"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = body.pop("runbook_id", "")
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V8 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v8",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "canonical_peft_bridge_implementation_ready": True,
        "actual_torch_peft_runtime_smoke_ready": False,
        "canonical_effect_worker_integration_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v7/runbook.json"
        ),
        "protocol_v16_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v16.json"
        ),
        "peft_runtime_benchmark_path": (
            project_root
            / "artifacts/vpd_canonical_lora_peft_runtime_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V8.")
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
