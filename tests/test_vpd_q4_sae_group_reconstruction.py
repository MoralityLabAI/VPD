import numpy as np
import pytest

from experiments.authz_mechinterp.audit_q4_sae_group_reconstruction_v1 import matched_groups, removal_delta_sse, removal_components


def test_removal_formula_matches_explicit_reconstruction():
    rng = np.random.default_rng(77)
    x, reconstruction, c = rng.normal(size=(3, 5, 7))
    residual = x - reconstruction
    expected = np.sum((x - (reconstruction - c))**2) - np.sum(residual**2)
    assert removal_delta_sse(residual, c) == pytest.approx(expected)


def test_removing_harmful_feature_can_improve_reconstruction():
    x = np.ones((4, 2))
    assert removal_delta_sse(np.zeros_like(x), x) == pytest.approx(8)
    assert removal_delta_sse(-x, x) == pytest.approx(-8)


def test_constant_contribution_has_no_within_pass_variation():
    r = np.arange(12, dtype=float).reshape(4, 3)
    c = np.ones_like(r)
    components = removal_components(r, c, 10)
    assert components["within_pass_variation"] == 0
    assert components["pass_mean_component"] == pytest.approx(removal_delta_sse(r, c) / 10)


def test_mean_component_matches_direct_mean_algebra():
    rng = np.random.default_rng(2)
    r, c = rng.normal(size=(2, 20, 8)) + 2
    result = removal_components(r, c, 17)
    direct_mean = len(r) * (np.sum(c.mean(0)**2) + 2 * np.sum(r.mean(0) * c.mean(0))) / 17
    assert result["pass_mean_component"] == pytest.approx(direct_mean)


def test_random_groups_preserve_training_bins_and_are_reproducible():
    rng = np.random.default_rng(91)
    mass = rng.random((8, 2, 100)) * (rng.random((8, 2, 100)) > 0.5)
    selected = np.arange(100) % 4 == 0
    controls, bins = matched_groups(mass, selected, 1)
    repeated, _ = matched_groups(mass, selected, 1)
    for group, again in zip(controls, repeated):
        np.testing.assert_array_equal(group, again)
        assert group.sum() == selected.sum()
        for b in np.unique(bins):
            assert np.sum(group & (bins == b)) == np.sum(selected & (bins == b))
