"""Local mathematical controls; no empirical SAE or model weights are read."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.authz_mechinterp.benchmark_nanbeige_layer21_sae_category_subspace_stability_v1 import _basis, _similarity
from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_category_subspace_transfer_v1 import _projected_subset_basis
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json
from experiments.authz_mechinterp.evaluate_sae_feature_replica_matching_v2 import mutual_signed_decoder_matches


def orthonormal_nonnegative_error(samples: np.ndarray, decoder: np.ndarray) -> dict:
    """Exact NNLS for orthonormal decoder rows, not a general sparse encoder."""
    if not np.allclose(decoder @ decoder.T, np.eye(len(decoder)), atol=1e-12):
        raise ValueError("analytic control requires orthonormal decoder rows")
    coefficients = np.maximum(samples @ decoder.T, 0.0)
    residual = samples - coefficients @ decoder
    return {
        "relative_squared_error": float(np.sum(residual**2) / np.sum(samples**2)),
        "mean_positive_coefficients": float(np.mean(np.sum(coefficients > 1e-12, axis=1))),
    }


def gram_reference_checks() -> list[dict]:
    rng = np.random.default_rng(260918)
    checks = []
    for rows, width, subset_size in ((24, 48, 16), (48, 24, 32)):
        decoder = rng.normal(size=(rows, width))
        decoder /= np.linalg.norm(decoder, axis=1, keepdims=True)
        left = _basis(rng.normal(size=(20, width)), rank=8)
        for draw in range(4):
            indices = rng.choice(rows, subset_size, replace=False)
            direct = _basis(decoder[indices], rank=8)
            projected = _projected_subset_basis(left.T @ decoder.T, decoder @ decoder.T, indices, 8)
            errors = []
            for rank in (2, 4, 8):
                reference = _similarity(left[:, :rank], direct[:, :rank])
                optimized = float(np.sum(projected[:rank, :rank] ** 2) / rank)
                errors.append(abs(reference - optimized))
            checks.append({"rows": rows, "width": width, "subset_size": subset_size,
                           "draw": draw, "maximum_absolute_similarity_error": max(errors)})
    return checks


def build_report() -> dict:
    left = np.eye(8)[:4]
    hadamard = np.array([[1, 1, 1, 1], [1, -1, 1, -1],
                         [1, 1, -1, -1], [1, -1, -1, 1]], dtype=float) / 2
    candidates = {
        "positive_permutation": left[[2, 0, 3, 1]],
        "sign_reversal": -left,
        "orthogonal_feature_mixing": hadamard @ left,
        "unrelated_span": np.eye(8)[4:],
    }
    scenarios = []
    for name, right in candidates.items():
        scenarios.append({
            "name": name,
            "span_similarity": _similarity(_basis(left, rank=4), _basis(right, rank=4)),
            "signed_mutual_matches_at_0_8": mutual_signed_decoder_matches(left, right)["mutual_matches"],
            "nonnegative_reconstruction_of_positive_source_atoms": orthonormal_nonnegative_error(left, right),
            "signed_least_squares_relative_error": float(np.sum((left - (left @ right.T) @ right)**2) / np.sum(left**2)),
        })
    dependencies = [Path(__file__), Path(__import__(_basis.__module__, fromlist=["__file__"]).__file__),
                    Path(__import__(_projected_subset_basis.__module__, fromlist=["__file__"]).__file__),
                    Path(__import__(mutual_signed_decoder_matches.__module__, fromlist=["__file__"]).__file__)]
    body = {
        "schema": "sch.vpd-sae-subspace-semantics-audit.v1",
        "provenance": {"kind": "newly_constructed_synthetic_controls",
                       "source_files": [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in dependencies]},
        "scenarios": scenarios,
        "gram_vs_direct_svd": gram_reference_checks(),
        "interpretation": {
            "span_overlap_does_not_establish_nonnegative_reconstruction": True,
            "span_overlap_does_not_establish_sparse_feature_identity": True,
            "orthonormal_nnls_is_only_a_counterexample_not_an_empirical_encoder": True,
            "near_degenerate_rank_cutoffs_not_covered_by_random_reference_checks": True,
            "empirical_reports_modified": False,
            "new_training_performed": False,
            "empirical_weights_loaded": False,
            "execution_authorized": False,
        },
    }
    return {**body, "report_id": content_sha256(body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    expected = build_report()
    if args.verify:
        if json.loads(args.verify.read_text(encoding="utf-8")) != expected:
            raise ValueError("synthetic semantics audit does not reproduce")
    else:
        write_json(args.output, expected)
    print(json.dumps({"report_id": expected["report_id"], "scenarios": expected["scenarios"],
                      "maximum_gram_error": max(r["maximum_absolute_similarity_error"] for r in expected["gram_vs_direct_svd"])}, indent=2))


if __name__ == "__main__":
    main()
