from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_lora_component_basis_identifiability_v1 import verify_benchmark
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v12 import canonical_inputs as protocol_v12_inputs, verify_protocol as verify_protocol_v12


SCHEMA = "sch.vpd-activation-category-protocol.v13"


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()): raise ValueError("research protocol V13 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def build_protocol(
    *, predecessor_path: Path, basis_benchmark_path: Path, project_root: Path
) -> dict[str, Any]:
    predecessor_verification = verify_protocol_v12(predecessor_path, **protocol_v12_inputs(project_root))
    benchmark_verification = verify_benchmark(basis_benchmark_path)
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    benchmark = json.loads(basis_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    canonicalizer = project_root / "experiments/authz_mechinterp/lora_component_basis_canonicalization_v1.py"
    result["component_compartments"]["lora_component_basis_identifiability_v1"] = {
        "schema": "sch.vpd-lora-component-basis-identifiability.v1",
        "status": "raw_rank_failure_reproduced_canonical_math_benchmarked_production_integration_pending",
        "canonicalizer": {"path": _relative(canonicalizer, project_root), "sha256": sha256_file(canonicalizer)},
        "benchmark": {
            "path": _relative(basis_benchmark_path, project_root),
            "sha256": sha256_file(basis_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "raw_specific_component_count_change_under_equivalent_basis": -16,
            "canonical_basis_invariant": True,
        },
        "component_definition_for_next_empirical_candidate": "thin_svd_spectral_component_or_unresolved_degenerate_block",
        "raw_lora_factor_rank_inventory_prohibited": True,
        "relative_spectral_gap_tolerance_default": 1e-6,
        "near_degenerate_vectors_group_into_one_unresolved_block": True,
        "same_shape_canonical_block_multiplier_factors_implemented_model_free": True,
        "production_adapter_inventory_emitter_present": False,
        "production_canonical_effect_capture_integration_present": False,
        "historical_raw_rank_evidence_preserved_without_relabelling": True,
        "required_before_new_empirical_activation_category_capture": True,
        "execution_authorized": False,
    }
    result["component_compartments"]["cluster_candidate_semantic_strata_admission_v3"][
        "additional_precondition"
    ] = "basis-identifiable canonical spectral component inventory"
    result["unified_lineage"]["canonical_component_basis_required"] = True
    result["unified_lineage"]["raw_lora_rank_inventory_prohibited_for_next_candidate"] = True
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v12_verification": predecessor_verification,
        "lora_basis_identifiability_verification": benchmark_verification,
        "canonicalizer_sha256": sha256_file(canonicalizer),
        "research_protocol_v13_is_distinct_from_historical_experiment_labels": True,
    }
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["sae_training_performed"] = False
    result["scientific_boundary"]["weights_loaded_or_mutated"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != build_protocol(**kwargs): raise ValueError("research protocol V13 is not reproducible")
    contract = observed["component_compartments"]["lora_component_basis_identifiability_v1"]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v13", "status": "pass",
        "protocol_sha256": sha256_file(path), "predecessor_sha256": sha256_file(kwargs["predecessor_path"]),
        "basis_benchmark_id": contract["benchmark"]["benchmark_id"],
        "raw_rank_inventory_prohibited": True,
        "production_canonical_capture_integration_present": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root / "configs/evals/vpd_activation_category_compartments_v12.json",
        "basis_benchmark_path": project_root / "artifacts/vpd_lora_component_basis_identifiability_benchmark_v1/report.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V13.")
    parser.add_argument("--output", type=Path); parser.add_argument("--verify", type=Path)
    args = parser.parse_args(); inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None: result = verify_protocol(args.verify, **inputs)
    else:
        if args.output is None: parser.error("--output is required unless verifying")
        result = build_protocol(**inputs); write_json(args.output, result)
    print(json.dumps({
        "schema": result["schema"], "status": result.get("status", "derived"),
        "protocol_sha256": result.get("protocol_sha256", sha256_file(args.output)),
        "execution_authorized": result.get("execution_authorized", False),
    }, indent=2, sort_keys=True))


if __name__ == "__main__": main()
