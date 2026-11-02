import numpy as np
import pytest

from experiments.authz_mechinterp.audit_sae_subspace_semantics_v1 import (
    build_report, gram_reference_checks, orthonormal_nonnegative_error,
)


def test_gram_optimization_agrees_with_independent_direct_svd():
    checks = gram_reference_checks()
    assert len(checks) == 8
    assert max(row["maximum_absolute_similarity_error"] for row in checks) < 1e-12


def test_identical_spans_can_have_incompatible_nonnegative_codes():
    rows = {row["name"]: row for row in build_report()["scenarios"]}
    expected = {"positive_permutation": (1, 4, 0, 1),
                "sign_reversal": (1, 0, 1, 0),
                "orthogonal_feature_mixing": (1, 0, 0.375, 2.5),
                "unrelated_span": (0, 0, 1, 0)}
    for name, (span, matches, error, l0) in expected.items():
        result = rows[name]
        assert result["span_similarity"] == pytest.approx(span)
        assert result["signed_mutual_matches_at_0_8"] == matches
        reconstruction = result["nonnegative_reconstruction_of_positive_source_atoms"]
        assert reconstruction["relative_squared_error"] == pytest.approx(error)
        assert reconstruction["mean_positive_coefficients"] == pytest.approx(l0)
    for name in ("positive_permutation", "sign_reversal", "orthogonal_feature_mixing"):
        assert rows[name]["signed_least_squares_relative_error"] < 1e-25


def test_analytic_nnls_rejects_nonorthonormal_dictionary():
    with pytest.raises(ValueError, match="orthonormal"):
        orthonormal_nonnegative_error(np.eye(2), np.ones((2, 2)))
