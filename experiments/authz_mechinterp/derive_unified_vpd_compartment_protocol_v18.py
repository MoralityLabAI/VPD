from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_canonical_vpd_effect_shard_assembly_v1 import (
    verify_benchmark as verify_assembly_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json


SCHEMA = "sch.vpd-activation-category-protocol.v18"
EXPECTED_V17_SHA256 = (
    "7740bdb5f7d47d5f4247f4feedc754887fd26d702006a469a9fd38bd14691feb"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V18 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_pinned_predecessor(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V17_SHA256:
        raise ValueError("protocol V17 differs from its verified pinned SHA-256")
    predecessor = json.loads(path.read_text(encoding="utf-8"))
    if predecessor.get("schema") != "sch.vpd-activation-category-protocol.v17":
        raise ValueError("pinned predecessor is not protocol V17")
    if predecessor["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("protocol V17 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-predecessor-verification.v1",
        "status": "pass",
        "method": "exact_sha256_and_boundary_fields",
        "sha256": EXPECTED_V17_SHA256,
    }


def build_protocol(
    *,
    predecessor_path: Path,
    assembly_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_pinned_predecessor(predecessor_path)
    benchmark_verification = verify_assembly_benchmark(assembly_benchmark_path)
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    benchmark = json.loads(assembly_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    assembler = (
        project_root
        / "experiments/authz_mechinterp/assemble_canonical_vpd_effect_shard_v1.py"
    )
    contract = result["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    contract["status"] = (
        "provenance_complete_model_free_effect_shard_assembler_benchmarked_"
        "model_loading_worker_cli_and_actual_capture_pending"
    )
    contract["canonical_effect_shard_assembler"] = {
        "path": _relative(assembler, project_root),
        "sha256": sha256_file(assembler),
        "output_array_contract": "v5_compatible_category_uncertainty_condition_template",
        "semantic_split_v2_or_v3_required": True,
        "minimum_countsketch_dimension": 256,
        "model_execution_performed": False,
    }
    contract["canonical_effect_shard_assembly_benchmark"] = {
        "path": _relative(assembly_benchmark_path, project_root),
        "sha256": sha256_file(assembly_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
        "components": benchmark["results"]["components"],
        "negative_controls": 9,
        "synthetic_inputs_only": True,
    }
    contract["model_free_successor_worker_archive_assembler_present"] = True
    contract["worker_batch_identity_binding_present"] = True
    contract["worker_condition_template_countsketch_present"] = True
    contract["successor_effect_worker_cli_integration_present"] = False
    contract["successor_worker_model_loader_present"] = False
    contract["actual_model_category_effect_capture_verified"] = False
    contract["production_canonical_effect_capture_integration_present"] = False
    result["unified_lineage"]["canonical_effect_shard_assembler_benchmarked"] = True
    result["unified_lineage"]["successor_effect_worker_model_loader_required"] = True
    result["unified_lineage"]["actual_model_category_capture_receipt_required"] = True
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v17_verification": predecessor_verification,
        "canonical_effect_shard_assembly_benchmark_verification": benchmark_verification,
        "canonical_effect_shard_assembler_sha256": sha256_file(assembler),
        "bounded_predecessor_replay": True,
        "research_protocol_v18_is_distinct_from_historical_experiment_labels": True,
    }
    result["scientific_boundary"]["real_adapter_accessed"] = False
    result["scientific_boundary"]["torch_or_peft_loaded"] = False
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["real_effect_archive_emitted"] = False
    result["scientific_boundary"]["weights_loaded_or_mutated"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != build_protocol(**kwargs):
        raise ValueError("research protocol V18 is not reproducible")
    contract = observed["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v18",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "predecessor_sha256": sha256_file(kwargs["predecessor_path"]),
        "assembly_benchmark_id": contract[
            "canonical_effect_shard_assembly_benchmark"
        ]["benchmark_id"],
        "model_free_successor_worker_archive_assembler_present": True,
        "successor_effect_worker_cli_integration_present": False,
        "actual_model_category_effect_capture_verified": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v17.json"
        ),
        "assembly_benchmark_path": (
            project_root
            / "artifacts/vpd_canonical_effect_shard_assembly_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V18.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None:
        result = verify_protocol(args.verify, **inputs)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_protocol(**inputs)
        write_json(args.output, result)
    protocol_sha256 = result.get("protocol_sha256")
    if protocol_sha256 is None:
        if args.output is None:
            raise ValueError("derived protocol output path is unavailable")
        protocol_sha256 = sha256_file(args.output)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result.get("status", "derived"),
                "protocol_sha256": protocol_sha256,
                "execution_authorized": result.get("execution_authorized", False),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
