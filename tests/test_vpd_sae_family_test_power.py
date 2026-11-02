from itertools import product

import numpy as np
import pytest

from experiments.authz_mechinterp.simulate_sae_family_test_power_v1 import exact_upper_p_values, family_means, wilson_interval


def test_exact_probabilities_agree_with_scalar_enumeration():
    for vector in ([1, 2, 3, 5], [0, 0, 0, 0], [1, -1, 2, -2], [-3, -2, -1, -4]):
        scores = [sum(s*v for s, v in zip(signs, vector)) for signs in product([-1, 1], repeat=4)]
        expected = sum(v >= sum(vector) for v in scores) / len(scores)
        assert exact_upper_p_values([vector])[0] == expected


def test_resolution_and_direction():
    assert exact_upper_p_values([[1]*10])[0] == 1/1024
    assert exact_upper_p_values([[-1]*10])[0] == 1
    assert exact_upper_p_values([[0]*10])[0] == 1


def test_repeated_examples_do_not_create_independent_families():
    values = [1, 2, 3, 4]
    labels = ["a", "b", "c", "d"]
    a = family_means(values, labels)
    b = family_means(np.repeat(values, 8), np.repeat(labels, 8))
    np.testing.assert_array_equal(a, b)
    assert exact_upper_p_values([b])[0] == 1/16


def test_wilson_interval_remains_nonzero_at_no_observed_successes():
    low, high = wilson_interval(0, 512)
    assert low == pytest.approx(0, abs=1e-15)
    assert 0 < high < 0.01


@pytest.mark.parametrize("bad", [[[np.nan, 1]], [[np.inf, 1]], []])
def test_nonfinite_or_empty_values_are_rejected(bad):
    with pytest.raises(ValueError):
        exact_upper_p_values(bad)
