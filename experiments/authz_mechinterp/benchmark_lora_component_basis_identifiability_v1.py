from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    classify_activation_compartments,
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.lora_component_basis_canonicalization_v1 import (
    canonical_block_multiplier_factors,
    canonical_spectral_components,
    raw_rank_components,
)


RANK = 16
CATEGORY_NAMES = [f"category_{index}" for index in range(8)]


def _hadamard(order: int) -> np.ndarray:
    if order < 1 or order & (order - 1):
        raise ValueError("Hadamard order must be a positive power of two")
    result = np.ones((1, 1), dtype=np.float64)
    while len(result) < order:
        result = np.block([[result, result], [result, -result]])
    return result / np.sqrt(order)


def _category_strengths(components: list[np.ndarray]) -> np.ndarray:
    strengths = np.zeros((len(components), len(CATEGORY_NAMES)), dtype=np.float64)
    for component_index, component in enumerate(components):
        diagonal = np.abs(np.diag(component))
        for category_index in range(len(CATEGORY_NAMES)):
            strengths[component_index, category_index] = float(
                diagonal[category_index * 2 : category_index * 2 + 2].sum()
            )
    return strengths


def _classification(components: list[np.ndarray]) -> dict[str, Any]:
    result = classify_activation_compartments(
        _category_strengths(components),
        CATEGORY_NAMES,
        minimum_dominant_share=0.35,
        minimum_dominance_margin=0.10,
        maximum_normalized_entropy=0.80,
    )
    categories = [str(value) for value in result["compartment_categories"]]
    return {
        "specific_components": sum(value != "shared_cross_category" for value in categories),
        "shared_components": sum(value == "shared_cross_category" for value in categories),
        "category_histogram": {
            category: categories.count(category)
            for category in sorted(set(categories))
        },
        "minimum_dominant_share": float(np.min(result["dominant_shares"])),
        "maximum_normalized_entropy": float(np.max(result["normalized_entropy"])),
    }


def _maximum_component_difference(
    left: list[np.ndarray], right: list[np.ndarray]
) -> float:
    if len(left) != len(right):
        return float("inf")
    return max(
        float(np.linalg.norm(left_item - right_item))
        for left_item, right_item in zip(left, right)
    )


def _same_compartment_classification(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        left[key] == right[key]
        for key in (
            "specific_components",
            "shared_components",
            "category_histogram",
        )
    )


def build_report() -> dict[str, Any]:
    singular_values = np.linspace(1.0, 1.15, RANK)
    a_original = np.diag(singular_values)
    b_original = np.eye(RANK)
    rotation = _hadamard(RANK)
    b_rotated = b_original @ rotation
    a_rotated = rotation.T @ a_original
    delta_original = b_original @ a_original
    delta_rotated = b_rotated @ a_rotated
    raw_original = raw_rank_components(a_original, b_original)
    raw_rotated = raw_rank_components(a_rotated, b_rotated)
    canonical_original = canonical_spectral_components(
        a_original, b_original, relative_gap_tolerance=1e-8
    )
    canonical_rotated = canonical_spectral_components(
        a_rotated, b_rotated, relative_gap_tolerance=1e-8
    )

    equal_a = np.eye(RANK)
    equal_b = np.eye(RANK)
    equal_original = canonical_spectral_components(
        equal_a, equal_b, relative_gap_tolerance=1e-6
    )
    equal_rotated = canonical_spectral_components(
        rotation.T @ equal_a,
        equal_b @ rotation,
        relative_gap_tolerance=1e-6,
    )
    reconstruction_error = float(np.linalg.norm(delta_original - delta_rotated))
    canonical_difference = _maximum_component_difference(
        canonical_original["components"], canonical_rotated["components"]
    )
    degenerate_difference = _maximum_component_difference(
        equal_original["components"], equal_rotated["components"]
    )
    raw_original_classification = _classification(raw_original)
    raw_rotated_classification = _classification(raw_rotated)
    canonical_original_classification = _classification(
        canonical_original["components"]
    )
    canonical_rotated_classification = _classification(
        canonical_rotated["components"]
    )
    intervention_errors: dict[str, float] = {}
    target_component = canonical_original["components"][3]
    for multiplier in (-1.0, 0.0, 1.0):
        factors = canonical_block_multiplier_factors(
            a_original,
            b_original,
            block_index=3,
            multiplier=multiplier,
            relative_gap_tolerance=1e-8,
        )
        observed_delta = factors["b_factor"] @ factors["a_factor"]
        expected_delta = delta_original + (multiplier - 1.0) * target_component
        intervention_errors[str(multiplier)] = float(
            np.linalg.norm(observed_delta - expected_delta)
        )
    maximum_intervention_error = max(intervention_errors.values())
    passed = (
        reconstruction_error < 1e-12
        and raw_original_classification["specific_components"] == RANK
        and raw_rotated_classification["specific_components"] == 0
        and _same_compartment_classification(
            canonical_original_classification,
            canonical_rotated_classification,
        )
        and canonical_difference < 1e-10
        and equal_original["canonical_component_count"] == 1
        and equal_rotated["canonical_component_count"] == 1
        and degenerate_difference < 1e-10
        and maximum_intervention_error < 1e-10
    )
    body = {
        "schema": "sch.vpd-lora-component-basis-identifiability-benchmark.v1",
        "status": "pass" if passed else "fail",
        "source": {
            "runner_sha256": sha256_file(Path(__file__)),
            "canonicalizer_sha256": sha256_file(
                Path(__file__).with_name("lora_component_basis_canonicalization_v1.py")
            ),
        },
        "design": {
            "rank": RANK,
            "categories": len(CATEGORY_NAMES),
            "equivalent_basis_change": "normalized_hadamard_orthogonal_rotation",
            "same_delta_weight_required": True,
            "raw_component_formula": "B[:,r] outer A[r,:]",
            "canonical_formula": "spectrally_gapped_thin_svd_blocks_of_B_at_A",
        },
        "raw_factor_coordinate_result": {
            "delta_reconstruction_frobenius_error": reconstruction_error,
            "original_classification": raw_original_classification,
            "rotated_classification": raw_rotated_classification,
            "specific_component_count_change": (
                raw_rotated_classification["specific_components"]
                - raw_original_classification["specific_components"]
            ),
            "basis_identifiable": False,
        },
        "canonical_spectral_result": {
            "distinct_spectrum_component_count": canonical_original[
                "canonical_component_count"
            ],
            "maximum_corresponding_component_frobenius_difference": canonical_difference,
            "original_classification": canonical_original_classification,
            "rotated_classification": canonical_rotated_classification,
            "basis_invariant": True,
        },
        "degenerate_spectrum_result": {
            "raw_rank": RANK,
            "canonical_block_count": equal_original["canonical_component_count"],
            "unresolved_degenerate_blocks": equal_original[
                "unresolved_degenerate_blocks"
            ],
            "maximum_block_frobenius_difference": degenerate_difference,
            "individual_singular_vectors_claimed": False,
            "basis_invariant": True,
        },
        "canonical_intervention_factorization": {
            "block_index": 3,
            "multipliers": [-1.0, 0.0, 1.0],
            "frobenius_errors_by_multiplier": intervention_errors,
            "maximum_frobenius_error": maximum_intervention_error,
            "same_shape_factor_reconstruction": True,
            "input_factors_mutated": False,
        },
        "required_protocol_action": {
            "raw_lora_factor_rank_may_not_define_next_empirical_cluster_inventory": True,
            "canonical_spectral_component_or_unresolved_block_inventory_required": True,
            "canonical_intervention_runtime_required_before_empirical_capture": True,
            "historical_raw_rank_evidence_must_not_be_relabelled": True,
        },
        "scientific_boundary": {
            "synthetic_factors_only": True,
            "model_loaded": False,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "does_not_canonicalize_the_recovered_bonsai_adapter": True,
            "execution_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = body.pop("benchmark_id", "")
    if identifier != content_sha256(body):
        raise ValueError("basis-identifiability benchmark address mismatch")
    if observed != build_report() or observed["status"] != "pass":
        raise ValueError("basis-identifiability benchmark is not reproducible")
    return {
        "schema": "sch.vpd-lora-component-basis-identifiability-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "raw_specific_component_count_change": observed[
            "raw_factor_coordinate_result"
        ]["specific_component_count_change"],
        "canonical_basis_invariant": observed["canonical_spectral_result"][
            "basis_invariant"
        ],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LoRA component basis identifiability.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_benchmark(args.verify)
    else:
        if args.output is None: parser.error("--output is required unless verifying")
        result = build_report(); write_json(args.output, result)
    print(json.dumps({
        "schema": result["schema"], "status": result["status"],
        "benchmark_id": result["benchmark_id"],
        "raw_specific_component_count_change": result.get("raw_factor_coordinate_result", {}).get(
            "specific_component_count_change", result.get("raw_specific_component_count_change")
        ),
        "canonical_basis_invariant": result.get("canonical_spectral_result", {}).get(
            "basis_invariant", result.get("canonical_basis_invariant")
        ),
        "execution_authorized": result.get("execution_authorized", False),
    }, indent=2, sort_keys=True))


if __name__ == "__main__": main()
