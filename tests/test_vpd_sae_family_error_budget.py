import numpy as np
import pytest
from experiments.authz_mechinterp.audit_q4_sae_family_error_budget_v1 import error_budget


def test_pure_offset_is_oracle_correctable_not_a_raw_quality_pass():
    x = np.array([[-1., 0.], [1., 0.]])
    result = error_budget(x, x+3)
    assert result["raw_ev"] == -17
    assert result["oracle_mean_corrected_ev"] == 1
    assert result["mean_offset_fraction_of_sse"] == 1


def test_missing_variation_is_not_fixed_by_oracle_mean():
    result = error_budget([[-1], [1]], [[0], [0]])
    assert result["raw_ev"] == result["oracle_mean_corrected_ev"] == 0
    assert result["mean_offset_fraction_of_sse"] == 0


def test_mixed_error_decomposition_matches_explicit_mean_correction():
    rng = np.random.default_rng(921)
    x = rng.normal(size=(16, 3))
    reconstructed = .7*x+2
    result = error_budget(x, reconstructed)
    corrected = reconstructed+(x-reconstructed).mean(0)
    assert result["sse"] == pytest.approx(result["mean_offset_sse"]+result["within_family_residual_sse"])
    assert result["oracle_mean_corrected_ev"] == pytest.approx(1-np.sum((x-corrected)**2)/np.sum((x-x.mean(0))**2))
    assert result["oracle_mean_corrected_ev"] >= result["raw_ev"]


def test_perfect_reconstruction_has_no_defined_error_share():
    result = error_budget([[-1], [1]], [[-1], [1]])
    assert result["raw_ev"] == 1
    assert result["mean_offset_fraction_of_sse"] is None


def test_constant_family_does_not_silently_pass_ev():
    result = error_budget([[1], [1]], [[1], [1]])
    assert result["raw_ev"] is result["oracle_mean_corrected_ev"] is None


@pytest.mark.parametrize("x,y", [([], []), ([[1]], [[1], [2]]), ([[float("nan")]], [[1]]), ([[1]], [[float("inf")]]), ([1, 2], [1, 2])])
def test_bad_matrices_rejected(x, y):
    with pytest.raises(ValueError):
        error_budget(x, y)
