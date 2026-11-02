import numpy as np
import pytest

from experiments.authz_mechinterp.audit_q4_sae_training_coverage_v1 import (
    coverage_scores, descriptive_correlation, summarize_phase,
)


def test_loto_excludes_every_row_of_same_trajectory():
    train = np.array([[0.], [1.], [10.], [11.]])
    result = coverage_scores(train, np.array([[5.]]), ["a", "a", "b", "b"])
    scale = result["training_scale"]
    assert result["training_loto"] * scale == pytest.approx([100, 81, 81, 100])
    assert result["evaluation_full_pool"] * scale == pytest.approx([16])
    assert result["evaluation_matched"] * scale == pytest.approx([20.5])
    assert result["reference_trajectories"] == 1


def test_reference_pool_matching_uses_mean_of_all_omissions():
    result = coverage_scores(np.array([[0.], [10.], [20.]]), np.array([[1.]]), ["a", "b", "c"])
    assert result["evaluation_matched"] * result["training_scale"] == pytest.approx([(81+1+1)/3])


def test_common_translation_and_positive_scaling_preserve_scores():
    train, evaluation = np.array([[0., 2.], [3., 5.], [8., 1.]]), np.array([[7., 3.]])
    original = coverage_scores(train, evaluation, ["a", "b", "c"])
    shifted = coverage_scores(3*train+17, 3*evaluation+17, ["a", "b", "c"])
    for field in ("training_loto", "evaluation_matched", "evaluation_full_pool"):
        np.testing.assert_allclose(original[field], shifted[field])


@pytest.mark.parametrize("train,evaluation,ids", [
    ([[0.], [1.]], [[float("nan")]], ["a", "b"]),
    ([[0.], [1.]], [[2., 3.]], ["a", "b"]),
    ([[0.], [1.]], [[2.]], ["a", "a"]),
    ([[0.], [1.]], [[2.]], ["a"]),
    ([[1.], [1.]], [[2.]], ["a", "b"]),
    ([[0.], [1.], [2.]], [[2.]], ["a", "a", "b"]),
])
def test_invalid_or_unmatched_references_rejected(train, evaluation, ids):
    with pytest.raises(ValueError):
        coverage_scores(train, evaluation, ids)


def test_summary_uses_examples_not_independent_rows():
    result = summarize_phase(np.array([[0.], [2.]]), np.array([[1.], [3.], [5.]]),
                             np.array([[1.], [2.], [3.]]), ["a", "b"], ["c", "c", "d"], ["f", "f", "g"])
    assert len(result["evaluation_examples"]) == 2
    assert result["evaluation_examples"][0]["rows"] == 2
    assert result["evaluation_examples"][0]["mean_reconstruction_error_squared_over_training_variance"] == 2.5
    assert result["descriptive_evaluation_example_pearson_r"] is None


def test_split_overlap_and_family_conflicts_rejected():
    args = (np.array([[0.], [2.]]), np.array([[1.], [3.]]), np.array([[1.], [2.]]), ["a", "b"])
    with pytest.raises(ValueError, match="disjoint"):
        summarize_phase(*args, ["a", "c"], ["f", "f"])
    with pytest.raises(ValueError, match="one family"):
        summarize_phase(*args, ["c", "c"], ["f", "g"])


def test_correlation_has_no_fake_zero_for_undefined_values():
    assert descriptive_correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1)
    assert descriptive_correlation([1, 1, 1], [2, 4, 6]) is None
    assert descriptive_correlation([1, 2], [2, 4]) is None
    with pytest.raises(ValueError):
        descriptive_correlation([1, 2, float("nan")], [2, 4, 6])
