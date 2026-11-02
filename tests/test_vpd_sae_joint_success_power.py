from itertools import product

import pytest

from experiments.authz_mechinterp.audit_sae_joint_success_power_v1 import equal_marginal_requirement, joint_bounds


def test_six_eighty_percent_events():
    result = joint_bounds([0.8]*6)
    assert result["dependence_unrestricted_lower"] == 0
    assert result["dependence_unrestricted_upper"] == 0.8
    assert result["independent_events_only"] == pytest.approx(0.262144)


def test_bounds_cover_enumerated_joint_distributions():
    outcomes = list(product((0, 1), repeat=3))
    for masses in ([1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1]):
        total = sum(masses)
        ps = [sum(w*state[i] for w, state in zip(masses, outcomes))/total for i in range(3)]
        joint = masses[-1]/total
        bounds = joint_bounds(ps)
        assert bounds["dependence_unrestricted_lower"]-1e-12 <= joint <= bounds["dependence_unrestricted_upper"]+1e-12


def test_conjunction_cannot_improve_by_adding_gate():
    assert joint_bounds([0.95]*6+[0.5])["dependence_unrestricted_upper"] == 0.5


def test_union_bound_requirement():
    result = equal_marginal_requirement(6, 0.8)
    p = result["sufficient_equal_marginal_for_arbitrary_dependence"]
    assert joint_bounds([p]*6)["dependence_unrestricted_lower"] == pytest.approx(0.8)


@pytest.mark.parametrize("values", [[], [True], [-0.1], [1.1], [float("nan")], [float("inf")]])
def test_invalid_probabilities(values):
    with pytest.raises(ValueError):
        joint_bounds(values)
