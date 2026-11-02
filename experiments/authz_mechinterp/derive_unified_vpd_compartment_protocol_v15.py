from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_authenticated_lora_factor_bundle_extraction_v1 import (
    verify_benchmark as verify_extraction_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json


SCHEMA = "sch.vpd-activation-category-protocol.v15"
EXPECTED_V14_SHA256 = (
    "4ca2e1c6c9a1491424ce63d6717c841c72328c903077ca98d44e0aaaf5eb60b8"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V15 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_pinned_predecessor(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V14_SHA256:
        raise ValueError("protocol V14 differs from its verified pinned SHA-256")
    predecessor = json.loads(path.read_text(encoding="utf-8"))
    if predecessor.get("schema") != "sch.vpd-activation-category-protocol.v14":
        raise ValueError("pinned predecessor is not protocol V14")
    if predecessor["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("protocol V14 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-predecessor-verification.v1",
        "status": "pass",
        "method": "exact_sha256_and_boundary_fields",
        "sha256": EXPECTED_V14_SHA256,
    }


def build_protocol(
    *,
    predecessor_path: Path,
    extraction_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_pinned_predecessor(predecessor_path)
    benchmark_verification = verify_extraction_benchmark(extraction_benchmark_path)
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    benchmark = json.loads(extraction_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    extractor = (
        project_root
        / "experiments/authz_mechinterp/extract_authenticated_lora_factor_bundle_v1.py"
    )
    contract = result["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    contract["status"] = (
        "authenticated_safetensors_factor_extractor_benchmarked_"
        "real_extraction_receipt_and_torch_integration_pending"
    )
    contract["authenticated_factor_bundle_extractor"] = {
        "path": _relative(extractor, project_root),
        "sha256": sha256_file(extractor),
        "input_contract": (
            "plain_lora_safetensors_plus_peft_config_plus_exact_expected_hashes"
        ),
        "output_contract": "sch.vpd-lora-factor-bundle-index.v1",
        "all_modules_validated_before_output_write": True,
        "real_adapter_extraction_performed": False,
    }
    contract["authenticated_factor_bundle_extraction_benchmark"] = {
        "path": _relative(extraction_benchmark_path, project_root),
        "sha256": sha256_file(extraction_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
        "negative_controls": 7,
        "canonical_inventory_chained": True,
        "synthetic_fixture_only": True,
    }
    contract["authenticated_adapter_factor_extractor_implementation_present"] = True
    contract["production_adapter_inventory_emitter_present"] = True
    contract["real_adapter_factor_bundle_extraction_receipt_present"] = False
    contract["production_canonical_effect_capture_integration_present"] = False
    result["unified_lineage"]["authenticated_factor_extractor_benchmarked"] = True
    result["unified_lineage"]["real_adapter_factor_bundle_receipt_required"] = True
    result["unified_lineage"]["production_torch_peft_integration_required"] = True
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v14_verification": predecessor_verification,
        "authenticated_factor_extraction_benchmark_verification": (
            benchmark_verification
        ),
        "authenticated_factor_extractor_sha256": sha256_file(extractor),
        "bounded_predecessor_replay": True,
        "research_protocol_v15_is_distinct_from_historical_experiment_labels": True,
    }
    result["scientific_boundary"]["real_adapter_accessed"] = False
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["weights_loaded_or_mutated"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != build_protocol(**kwargs):
        raise ValueError("research protocol V15 is not reproducible")
    contract = observed["component_compartments"][
        "lora_component_basis_identifiability_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v15",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "predecessor_sha256": sha256_file(kwargs["predecessor_path"]),
        "extraction_benchmark_id": contract[
            "authenticated_factor_bundle_extraction_benchmark"
        ]["benchmark_id"],
        "real_adapter_factor_bundle_extraction_receipt_present": False,
        "production_torch_peft_integration_present": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v14.json"
        ),
        "extraction_benchmark_path": (
            project_root
            / "artifacts/vpd_authenticated_lora_factor_bundle_extraction_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V15.")
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
