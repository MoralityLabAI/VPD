from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_canonical_lora_component_inventory_v1 import verify_benchmark as verify_inventory_benchmark
from experiments.authz_mechinterp.benchmark_canonical_lora_factor_runtime_v1 import verify_benchmark as verify_runtime_benchmark
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v13 import canonical_inputs as protocol_v13_inputs, verify_protocol as verify_protocol_v13


SCHEMA = "sch.vpd-activation-category-protocol.v14"


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()): raise ValueError("protocol V14 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def build_protocol(
    *, predecessor_path: Path, inventory_benchmark_path: Path,
    runtime_benchmark_path: Path, project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = verify_protocol_v13(predecessor_path, **protocol_v13_inputs(project_root))
    inventory_verification = verify_inventory_benchmark(inventory_benchmark_path)
    runtime_verification = verify_runtime_benchmark(runtime_benchmark_path)
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    inventory_benchmark = json.loads(inventory_benchmark_path.read_text(encoding="utf-8"))
    runtime_benchmark = json.loads(runtime_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor)); result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    emitter = project_root / "experiments/authz_mechinterp/emit_canonical_lora_component_inventory_v1.py"
    runtime = project_root / "experiments/authz_mechinterp/canonical_lora_factor_runtime_v1.py"
    contract = result["component_compartments"]["lora_component_basis_identifiability_v1"]
    contract["status"] = "canonical_factor_bundle_inventory_and_model_free_restoration_runtime_benchmarked_torch_integration_pending"
    contract["canonical_inventory_emitter"] = {
        "path": _relative(emitter, project_root), "sha256": sha256_file(emitter),
        "input_contract": "authenticated_lora_factor_bundle_index_v1",
        "production_real_adapter_extraction_performed": False,
    }
    contract["canonical_inventory_benchmark"] = {
        "path": _relative(inventory_benchmark_path, project_root),
        "sha256": sha256_file(inventory_benchmark_path),
        "benchmark_id": inventory_benchmark["benchmark_id"],
        "verification": inventory_verification,
        "equivalent_factorization_semantic_rows_equal": True,
        "tampered_module_artifact_rejected": True,
    }
    contract["canonical_factor_runtime"] = {
        "path": _relative(runtime, project_root), "sha256": sha256_file(runtime),
        "mode": "numpy_model_free_install_probe_callback_restore",
        "production_torch_peft_integration_present": False,
    }
    contract["canonical_factor_runtime_benchmark"] = {
        "path": _relative(runtime_benchmark_path, project_root),
        "sha256": sha256_file(runtime_benchmark_path),
        "benchmark_id": runtime_benchmark["benchmark_id"],
        "verification": runtime_verification,
        "multipliers": [-1.0, 0.0, 1.0, 2.0],
        "callback_failure_restoration_verified": True,
    }
    contract["production_adapter_inventory_emitter_present"] = False
    contract["factor_bundle_canonical_inventory_emitter_present"] = True
    contract["model_free_restoration_runtime_present"] = True
    contract["production_canonical_effect_capture_integration_present"] = False
    result["unified_lineage"]["canonical_factor_bundle_inventory_benchmarked"] = True
    result["unified_lineage"]["canonical_numpy_restoration_runtime_benchmarked"] = True
    result["unified_lineage"]["production_torch_peft_integration_required"] = True
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v13_verification": predecessor_verification,
        "canonical_inventory_benchmark_verification": inventory_verification,
        "canonical_factor_runtime_benchmark_verification": runtime_verification,
        "canonical_inventory_emitter_sha256": sha256_file(emitter),
        "canonical_factor_runtime_sha256": sha256_file(runtime),
        "research_protocol_v14_is_distinct_from_historical_experiment_labels": True,
    }
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["weights_loaded_or_mutated"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != build_protocol(**kwargs): raise ValueError("research protocol V14 is not reproducible")
    contract = observed["component_compartments"]["lora_component_basis_identifiability_v1"]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v14", "status": "pass",
        "protocol_sha256": sha256_file(path), "predecessor_sha256": sha256_file(kwargs["predecessor_path"]),
        "inventory_benchmark_id": contract["canonical_inventory_benchmark"]["benchmark_id"],
        "runtime_benchmark_id": contract["canonical_factor_runtime_benchmark"]["benchmark_id"],
        "production_torch_peft_integration_present": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root / "configs/evals/vpd_activation_category_compartments_v13.json",
        "inventory_benchmark_path": project_root / "artifacts/vpd_canonical_lora_component_inventory_benchmark_v1/report.json",
        "runtime_benchmark_path": project_root / "artifacts/vpd_canonical_lora_factor_runtime_benchmark_v1/report.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V14.")
    parser.add_argument("--output", type=Path); parser.add_argument("--verify", type=Path)
    args = parser.parse_args(); inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None: result = verify_protocol(args.verify, **inputs)
    else:
        if args.output is None: parser.error("--output is required unless verifying")
        result = build_protocol(**inputs); write_json(args.output, result)
    protocol_sha256 = result.get("protocol_sha256")
    if protocol_sha256 is None:
        if args.output is None:
            raise ValueError("derived protocol output path is unavailable")
        protocol_sha256 = sha256_file(args.output)
    print(json.dumps({
        "schema": result["schema"], "status": result.get("status", "derived"),
        "protocol_sha256": protocol_sha256,
        "execution_authorized": result.get("execution_authorized", False),
    }, indent=2, sort_keys=True))


if __name__ == "__main__": main()
