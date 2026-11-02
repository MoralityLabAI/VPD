from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.authz_mechinterp.benchmark_lora_component_basis_identifiability_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256
from experiments.authz_mechinterp.lora_component_basis_canonicalization_v1 import (
    canonical_block_multiplier_factors,
    canonical_spectral_components,
    raw_rank_components,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_lora_component_basis_identifiability_benchmark_v1/report.json"


def test_basis_identifiability_benchmark_reproduces() -> None:
    verification = verify_benchmark(REPORT)
    assert verification["status"] == "pass"
    assert verification["raw_specific_component_count_change"] == -16
    assert verification["canonical_basis_invariant"] is True
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    body = dict(report)
    assert body.pop("benchmark_id") == content_sha256(body)


def test_equivalent_factor_rotation_changes_raw_components_not_delta() -> None:
    rotation = np.asarray([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    a = np.diag([1.0, 1.2])
    b = np.eye(2)
    rotated_a = rotation.T @ a
    rotated_b = b @ rotation
    np.testing.assert_allclose(b @ a, rotated_b @ rotated_a, atol=1e-14)
    raw = raw_rank_components(a, b)
    rotated_raw = raw_rank_components(rotated_a, rotated_b)
    assert max(
        np.linalg.norm(left - right)
        for left, right in zip(raw, rotated_raw)
    ) > 0.5
    canonical = canonical_spectral_components(
        a, b, relative_gap_tolerance=1e-8
    )
    rotated_canonical = canonical_spectral_components(
        rotated_a, rotated_b, relative_gap_tolerance=1e-8
    )
    assert canonical["canonical_component_count"] == 2
    for left, right in zip(
        canonical["components"], rotated_canonical["components"]
    ):
        np.testing.assert_allclose(left, right, atol=1e-12)


def test_degenerate_spectrum_is_one_unresolved_block() -> None:
    result = canonical_spectral_components(
        np.eye(4), np.eye(4), relative_gap_tolerance=1e-6
    )
    assert result["canonical_component_count"] == 1
    assert result["unresolved_degenerate_blocks"] == 1
    assert result["spectral_blocks"] == [(0, 4)]
    np.testing.assert_allclose(result["reconstruction"], np.eye(4), atol=1e-14)


def test_canonical_block_multiplier_reconstructs_exact_target_delta() -> None:
    a = np.diag([1.0, 1.2, 1.4])
    b = np.eye(3)
    baseline = b @ a
    canonical = canonical_spectral_components(a, b, relative_gap_tolerance=1e-8)
    target = canonical["components"][1]
    original_a = a.copy()
    original_b = b.copy()
    for multiplier in (-1.0, 0.0, 1.0, 2.0):
        factors = canonical_block_multiplier_factors(
            a,
            b,
            block_index=1,
            multiplier=multiplier,
            relative_gap_tolerance=1e-8,
        )
        np.testing.assert_allclose(
            factors["b_factor"] @ factors["a_factor"],
            baseline + (multiplier - 1.0) * target,
            atol=1e-12,
        )
    np.testing.assert_array_equal(a, original_a)
    np.testing.assert_array_equal(b, original_b)
