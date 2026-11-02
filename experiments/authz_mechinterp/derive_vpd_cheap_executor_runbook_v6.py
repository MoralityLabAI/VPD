from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_canonical_lora_component_inventory_v1 import verify_benchmark as verify_inventory_benchmark
from experiments.authz_mechinterp.benchmark_canonical_lora_factor_runtime_v1 import verify_benchmark as verify_runtime_benchmark
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v14 import canonical_inputs as protocol_v14_inputs, verify_protocol as verify_protocol_v14
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v5 import canonical_inputs as runbook_v5_inputs, verify_runbook as verify_runbook_v5


SCHEMA = "sch.vpd-cheap-executor-runbook.v6"


def _load(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()): raise ValueError("runbook V6 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def build_runbook(
    *, predecessor_path: Path, protocol_v14_path: Path,
    inventory_benchmark_path: Path, runtime_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = verify_runbook_v5(predecessor_path, **runbook_v5_inputs(project_root))
    protocol_verification = verify_protocol_v14(protocol_v14_path, **protocol_v14_inputs(project_root))
    inventory_verification = verify_inventory_benchmark(inventory_benchmark_path)
    runtime_verification = verify_runtime_benchmark(runtime_benchmark_path)
    predecessor = _load(predecessor_path); protocol = _load(protocol_v14_path)
    inventory_benchmark = _load(inventory_benchmark_path); runtime_benchmark = _load(runtime_benchmark_path)
    contract = protocol["component_compartments"]["lora_component_basis_identifiability_v1"]
    result = json.loads(json.dumps(predecessor)); result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = "ready_for_local_preflight_canonical_torch_integration_pending"
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root), "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"], "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v14"] = {
        "path": _relative(protocol_v14_path, project_root), "sha256": sha256_file(protocol_v14_path),
        "verification": protocol_verification,
    }
    result["source"]["canonical_lora_inventory_benchmark_v1"] = {
        "path": _relative(inventory_benchmark_path, project_root), "sha256": sha256_file(inventory_benchmark_path),
        "benchmark_id": inventory_benchmark["benchmark_id"], "verification": inventory_verification,
    }
    result["source"]["canonical_lora_factor_runtime_benchmark_v1"] = {
        "path": _relative(runtime_benchmark_path, project_root), "sha256": sha256_file(runtime_benchmark_path),
        "benchmark_id": runtime_benchmark["benchmark_id"], "verification": runtime_verification,
    }
    result["source"]["canonical_lora_inventory_emitter_v1"] = contract["canonical_inventory_emitter"]
    result["source"]["canonical_lora_factor_runtime_v1"] = contract["canonical_factor_runtime"]
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["canonical_factor_bundle_inventory_benchmark"] = "pass"
    result["current_result"]["canonical_numpy_restoration_runtime_benchmark"] = "pass"
    result["current_result"]["production_torch_peft_integration_present"] = False
    preflight = result["stages"][0]
    for command_id, module, artifact in (
        ("verify_canonical_lora_inventory_benchmark_v1", "experiments.authz_mechinterp.benchmark_canonical_lora_component_inventory_v1", "artifacts/vpd_canonical_lora_component_inventory_benchmark_v1/report.json"),
        ("verify_canonical_lora_factor_runtime_benchmark_v1", "experiments.authz_mechinterp.benchmark_canonical_lora_factor_runtime_v1", "artifacts/vpd_canonical_lora_factor_runtime_benchmark_v1/report.json"),
    ):
        preflight["commands"].append({"id": command_id, "executable": ".venv/Scripts/python.exe", "arguments": ["-m", module, "--verify", artifact]})
    preflight["expected"]["canonical_inventory_benchmark"] = "basis-invariant semantic inventory; tamper rejected"
    preflight["expected"]["canonical_factor_runtime_benchmark"] = "four multipliers and injected callback failure restore factors"
    preflight["stop_if"].append("protocol V14 or canonical inventory/runtime evidence differs")
    gate = result["stages"][1]
    gate["implementation_status"] = "factor-bundle canonical inventory emitter and model-free NumPy restoration runtime implemented; authenticated real-adapter extraction and Torch/PEFT capture integration absent"
    gate["available_model_free_command_template"] = {
        "executable": ".venv/Scripts/python.exe",
        "arguments": [
            "-m", "experiments.authz_mechinterp.emit_canonical_lora_component_inventory_v1",
            "--factor-index", "<AUTHENTICATED_FACTOR_BUNDLE_INDEX>",
            "--output-dir", "<CANONICAL_COMPONENT_INVENTORY_DIR>",
        ],
    }
    gate["stop_if"] = [
        "raw LoRA factor rank is used as the component identity",
        "authenticated adapter-to-factor-bundle extraction is absent",
        "production Torch/PEFT canonical effect-capture integration is absent",
        "spectral-gap tolerance is not frozen and benchmarked",
        "any source hash, baseline delta reconstruction, probe, or restoration check fails",
    ]
    result["completion_definition"]["canonical_factor_bundle_inventory_runtime_ready"] = True
    result["completion_definition"]["canonical_torch_peft_capture_runtime_ready"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path); body = dict(observed); identifier = body.pop("runbook_id", "")
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V6 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v6", "status": "pass",
        "runbook_id": identifier, "stages": len(observed["stages"]), "safe_stage_now": 0,
        "canonical_factor_bundle_inventory_runtime_ready": True,
        "canonical_torch_peft_capture_runtime_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root / "artifacts/vpd_cheap_executor_runbook_v5/runbook.json",
        "protocol_v14_path": project_root / "configs/evals/vpd_activation_category_compartments_v14.json",
        "inventory_benchmark_path": project_root / "artifacts/vpd_canonical_lora_component_inventory_benchmark_v1/report.json",
        "runtime_benchmark_path": project_root / "artifacts/vpd_canonical_lora_factor_runtime_benchmark_v1/report.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V6.")
    parser.add_argument("--output", type=Path); parser.add_argument("--verify", type=Path)
    args = parser.parse_args(); inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None: result = verify_runbook(args.verify, **inputs)
    else:
        if args.output is None: parser.error("--output is required unless verifying")
        result = build_runbook(**inputs); write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("schema", "status", "runbook_id")}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
