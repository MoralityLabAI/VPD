from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.authz_mechinterp.evaluate_sae_feature_replica_matching_v2 import (
    evaluate_replica_match_v2,
    mutual_signed_decoder_matches,
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_sae_feature_replica_matching_benchmark_v2/report.json"


def test_v2_replica_matching_benchmark_reproduces() -> None:
    verification = verify_benchmark(REPORT)
    assert verification["status"] == "pass"
    assert verification["scenarios"] == 4
    assert verification["expected_outcomes_reproduced"] == 4
    assert verification["clean_permutation_recovered"] is True
    assert verification["sign_flip_rejected"] is True
    assert verification["duplicate_collapse_rejected"] is True
    assert verification["category_profile_swap_rejected"] is True
    assert verification["execution_authorized"] is False


def test_signed_mutual_match_recovers_a_permutation_one_to_one() -> None:
    generator = np.random.default_rng(11)
    left = generator.normal(size=(24, 12))
    permutation = generator.permutation(24)
    right = left[permutation]
    result = mutual_signed_decoder_matches(
        left, right, minimum_cosine=0.99, block_rows=5
    )
    inverse = np.argsort(permutation)
    assert result["mutual_matches"] == 24
    assert result["decoder_coverage"] == 1.0
    assert result["matched_left_indices"] == list(range(24))
    assert result["matched_right_indices"] == inverse.tolist()
    assert len(set(result["matched_right_indices"])) == 24
    assert result["signed_cosine"] is True
    assert result["mutual_nearest_neighbor"] is True
    assert result["one_to_one"] is True


def test_sign_flip_cannot_pass_as_an_sae_feature_match() -> None:
    left = np.eye(8, dtype=np.float64)
    result = mutual_signed_decoder_matches(
        left, -left, minimum_cosine=0.8, block_rows=3
    )
    assert result["mutual_matches"] == 0
    assert result["decoder_coverage"] == 0.0


def test_duplicate_collision_reduces_mutual_coverage() -> None:
    left = np.eye(8, dtype=np.float64)
    right = left.copy()
    right[4:] = right[:4]
    result = mutual_signed_decoder_matches(
        left, right, minimum_cosine=0.8, block_rows=3
    )
    assert result["mutual_matches"] == 4
    assert result["decoder_coverage"] == 0.5
    assert len(set(result["matched_right_indices"])) == 4


def test_profile_mismatch_rejects_decoder_aligned_replica() -> None:
    decoder = np.eye(8, dtype=np.float64)
    profiles = np.tile(np.eye(2), (4, 1))
    swapped = profiles[:, ::-1]
    result = evaluate_replica_match_v2(
        decoder,
        decoder,
        left_category_profiles=profiles,
        right_category_profiles=swapped,
        left_selective=np.ones(8, dtype=bool),
        minimum_decoder_cosine=0.8,
        minimum_decoder_coverage=0.9,
        minimum_profile_cosine=0.8,
        minimum_selective_replication_rate=0.5,
        block_rows=3,
    )
    assert result["checks"]["decoder_coverage"] is True
    assert result["checks"]["selective_replication"] is False
    assert result["selective_replication_rate"] == 0.0
    assert result["status"] == "reject"


def test_benchmark_boundary_preserves_v1_and_authority() -> None:
    boundary = json.loads(REPORT.read_text(encoding="utf-8"))["scientific_boundary"]
    assert boundary["synthetic_benchmark_only"] is True
    assert boundary["no_empirical_feature_replication_claim"] is True
    assert boundary["v1_matcher_behavior_unchanged"] is True
    assert boundary["decoder_matching_is_not_semantic_interpretation"] is True
    assert boundary["new_model_inference_performed"] is False
    assert boundary["new_sae_training_performed"] is False
    assert boundary["execution_authorized"] is False
