from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-sae-feature-replica-matching-benchmark.v2"
MINIMUM_DECODER_COSINE = 0.80
MINIMUM_DECODER_COVERAGE = 0.90
MINIMUM_PROFILE_COSINE = 0.80
MINIMUM_SELECTIVE_REPLICATION_RATE = 0.50


def _normalized_rows(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("replica matching requires a finite matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("replica matching rejects zero vectors")
    return matrix / norms


def mutual_signed_decoder_matches(
    left_decoder: np.ndarray,
    right_decoder: np.ndarray,
    *,
    minimum_cosine: float = MINIMUM_DECODER_COSINE,
    block_rows: int = 256,
) -> dict[str, Any]:
    """Match SAE features with signed, mutual nearest-neighbor decoder cosine."""
    if not -1.0 <= minimum_cosine <= 1.0:
        raise ValueError("minimum cosine is outside [-1, 1]")
    if block_rows < 1:
        raise ValueError("block_rows must be positive")
    left = _normalized_rows(left_decoder)
    right = _normalized_rows(right_decoder)
    if left.shape[1] != right.shape[1]:
        raise ValueError("replica decoder hidden dimensions differ")

    left_best_index = np.empty(len(left), dtype=np.int64)
    left_best_score = np.empty(len(left), dtype=np.float64)
    right_best_index = np.full(len(right), -1, dtype=np.int64)
    right_best_score = np.full(len(right), -np.inf, dtype=np.float64)
    for start in range(0, len(left), block_rows):
        end = min(len(left), start + block_rows)
        cosine = left[start:end] @ right.T
        local_left_index = np.argmax(cosine, axis=1)
        left_best_index[start:end] = local_left_index
        left_best_score[start:end] = cosine[
            np.arange(end - start), local_left_index
        ]
        local_right_row = np.argmax(cosine, axis=0)
        local_right_score = cosine[local_right_row, np.arange(len(right))]
        improve = local_right_score > right_best_score
        right_best_score[improve] = local_right_score[improve]
        right_best_index[improve] = start + local_right_row[improve]

    left_indices = np.arange(len(left), dtype=np.int64)
    mutual = (
        right_best_index[left_best_index] == left_indices
    ) & (left_best_score >= minimum_cosine)
    matched_left = left_indices[mutual]
    matched_right = left_best_index[mutual]
    matched_cosine = left_best_score[mutual]
    if len(np.unique(matched_right)) != len(matched_right):
        raise ValueError("mutual matching produced a right-side collision")
    return {
        "left_features": len(left),
        "right_features": len(right),
        "minimum_signed_decoder_cosine": minimum_cosine,
        "mutual_matches": int(len(matched_left)),
        "left_coverage": float(len(matched_left) / max(1, len(left))),
        "right_coverage": float(len(matched_right) / max(1, len(right))),
        "decoder_coverage": float(
            min(
                len(matched_left) / max(1, len(left)),
                len(matched_right) / max(1, len(right)),
            )
        ),
        "matched_left_indices": matched_left.tolist(),
        "matched_right_indices": matched_right.tolist(),
        "matched_decoder_cosines": matched_cosine.tolist(),
        "signed_cosine": True,
        "mutual_nearest_neighbor": True,
        "one_to_one": True,
    }


def evaluate_replica_match_v2(
    left_decoder: np.ndarray,
    right_decoder: np.ndarray,
    *,
    left_category_profiles: np.ndarray | None = None,
    right_category_profiles: np.ndarray | None = None,
    left_selective: np.ndarray | None = None,
    minimum_decoder_cosine: float = MINIMUM_DECODER_COSINE,
    minimum_decoder_coverage: float = MINIMUM_DECODER_COVERAGE,
    minimum_profile_cosine: float = MINIMUM_PROFILE_COSINE,
    minimum_selective_replication_rate: float = MINIMUM_SELECTIVE_REPLICATION_RATE,
    block_rows: int = 256,
) -> dict[str, Any]:
    match = mutual_signed_decoder_matches(
        left_decoder,
        right_decoder,
        minimum_cosine=minimum_decoder_cosine,
        block_rows=block_rows,
    )
    checks = {"decoder_coverage": match["decoder_coverage"] >= minimum_decoder_coverage}
    selective_rate: float | None = None
    if any(value is not None for value in (left_category_profiles, right_category_profiles, left_selective)):
        if any(value is None for value in (left_category_profiles, right_category_profiles, left_selective)):
            raise ValueError("profiles and selective mask must be supplied together")
        left_profiles = _normalized_rows(np.asarray(left_category_profiles))
        right_profiles = _normalized_rows(np.asarray(right_category_profiles))
        selective = np.asarray(left_selective, dtype=bool)
        if len(left_profiles) != len(left_decoder) or len(right_profiles) != len(right_decoder):
            raise ValueError("category profiles do not align with decoder features")
        if selective.shape != (len(left_decoder),):
            raise ValueError("selective mask does not align with left decoder")
        paired = {
            left: right
            for left, right in zip(
                match["matched_left_indices"], match["matched_right_indices"]
            )
        }
        selective_indices = np.flatnonzero(selective)
        replicated = 0
        for left_index in selective_indices:
            right_index = paired.get(int(left_index))
            if right_index is None:
                continue
            profile_cosine = float(
                left_profiles[left_index] @ right_profiles[right_index]
            )
            replicated += profile_cosine >= minimum_profile_cosine
        selective_rate = replicated / max(1, len(selective_indices))
        checks["selective_replication"] = (
            selective_rate >= minimum_selective_replication_rate
        )
    return {
        **match,
        "minimum_decoder_coverage": minimum_decoder_coverage,
        "minimum_profile_cosine": minimum_profile_cosine,
        "minimum_selective_replication_rate": minimum_selective_replication_rate,
        "selective_replication_rate": selective_rate,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "reject",
    }


def _synthetic_inputs(seed: int = 260911) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    decoder = _normalized_rows(generator.normal(size=(64, 32)))
    category = np.arange(64) % 4
    profiles = np.full((64, 4), 0.01, dtype=np.float64)
    profiles[np.arange(64), category] = 1.0
    profiles = _normalized_rows(profiles)
    selective = category < 2
    return decoder, profiles, selective


def build_benchmark() -> dict[str, Any]:
    decoder, profiles, selective = _synthetic_inputs()
    generator = np.random.default_rng(260912)
    permutation = generator.permutation(len(decoder))
    inverse = np.argsort(permutation)
    clean_right = _normalized_rows(
        decoder[permutation] + generator.normal(scale=0.005, size=decoder.shape)
    )
    clean_profiles = _normalized_rows(
        profiles[permutation] + generator.normal(scale=0.005, size=profiles.shape)
    )
    collapsed = clean_right.copy()
    collapsed[32:] = collapsed[:32]
    swapped_profiles = clean_profiles.copy()
    swapped_profiles[:, [0, 1]] = swapped_profiles[:, [1, 0]]

    scenarios = []
    for scenario_id, right_decoder, right_profiles, expected in (
        ("permuted_noisy_replica", clean_right, clean_profiles, "pass"),
        ("sign_flipped_replica", -clean_right, clean_profiles, "reject"),
        ("duplicate_collapsed_replica", collapsed, clean_profiles, "reject"),
        ("category_profile_swap", clean_right, swapped_profiles, "reject"),
    ):
        result = evaluate_replica_match_v2(
            decoder,
            right_decoder,
            left_category_profiles=profiles,
            right_category_profiles=right_profiles,
            left_selective=selective,
            block_rows=11,
        )
        if result["status"] != expected:
            raise ValueError(f"synthetic V2 replica scenario differs: {scenario_id}")
        recovered_permutation = None
        if scenario_id == "permuted_noisy_replica":
            observed = np.full(len(decoder), -1, dtype=np.int64)
            observed[result["matched_left_indices"]] = result["matched_right_indices"]
            recovered_permutation = bool(np.array_equal(observed, inverse))
            if not recovered_permutation:
                raise ValueError("clean replica permutation was not recovered")
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "expected_status": expected,
                "observed_status": result["status"],
                "decoder_coverage": result["decoder_coverage"],
                "selective_replication_rate": result["selective_replication_rate"],
                "mutual_matches": result["mutual_matches"],
                "recovered_exact_permutation": recovered_permutation,
                "checks": result["checks"],
            }
        )
    body = {
        "schema": SCHEMA,
        "status": "pass",
        "matcher": {
            "version": "v2",
            "signed_decoder_cosine": True,
            "mutual_nearest_neighbor": True,
            "one_to_one": True,
            "minimum_decoder_cosine": MINIMUM_DECODER_COSINE,
            "minimum_decoder_coverage": MINIMUM_DECODER_COVERAGE,
            "minimum_profile_cosine": MINIMUM_PROFILE_COSINE,
            "minimum_selective_replication_rate": MINIMUM_SELECTIVE_REPLICATION_RATE,
            "bounded_cosine_block_rows": 11,
            "full_cosine_matrix_retained": False,
        },
        "scenarios": scenarios,
        "summary": {
            "scenarios": len(scenarios),
            "expected_outcomes_reproduced": sum(
                row["expected_status"] == row["observed_status"] for row in scenarios
            ),
            "clean_permutation_recovered": True,
            "sign_flip_rejected": True,
            "duplicate_collapse_rejected": True,
            "category_profile_swap_rejected": True,
        },
        "scientific_boundary": {
            "synthetic_benchmark_only": True,
            "no_empirical_feature_replication_claim": True,
            "v1_matcher_behavior_unchanged": True,
            "decoder_matching_is_not_semantic_interpretation": True,
            "new_model_inference_performed": False,
            "new_sae_training_performed": False,
            "weights_loaded_or_mutated": False,
            "execution_authorized": False,
            "paid_compute_authorized": False,
        },
        "provenance": {"runner_sha256": sha256_file(Path(__file__))},
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    benchmark_id = str(body.pop("benchmark_id", ""))
    if content_sha256(body) != benchmark_id:
        raise ValueError("V2 feature-replica benchmark content address differs")
    expected = build_benchmark()
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("V2 feature-replica benchmark does not reproduce")
    return {
        "schema": "sch.vpd-sae-feature-replica-matching-benchmark-verification.v2",
        "status": "pass",
        "benchmark_id": benchmark_id,
        **observed["summary"],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_benchmark(args.verify)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_benchmark()
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
