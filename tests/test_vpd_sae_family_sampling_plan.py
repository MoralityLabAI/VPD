import copy
import numpy as np
import pytest
from experiments.authz_mechinterp.plan_sae_family_sampling_v1 import sampling_distribution, sample_indices, compare_sampling


def row(id, family, positions, split="train"):
    return dict(id=id, family=family, positions=positions, split=split)


def test_equal_family_example_row_probability_with_unequal_rows():
    records = [row("a1", "a", [0]), row("a2", "a", [0, 1, 2]), row("b1", "b", [0, 1])]
    roster, probabilities = sampling_distribution(records, "equal_family_example_row")
    np.testing.assert_allclose(probabilities, [.25, 1/12, 1/12, 1/12, .25, .25])
    assert len(roster) == 6


def test_uniform_rows_is_not_uniform_examples_with_variable_lengths():
    _, probabilities = sampling_distribution([row("a", "a", [0]), row("b", "b", [0, 1])], "uniform_rows")
    np.testing.assert_allclose(probabilities, [1/3]*3)


def test_equal_roster_has_no_distinct_experimental_arm():
    result = compare_sampling([row(str(i), str(i//2), [0, 1]) for i in range(4)], steps=3)
    assert not result["distinct_sampling_distribution"]
    assert result["identical_dry_run_schedule"]


def test_evaluation_rows_never_enter_sampling_distribution():
    records = [row("a", "a", [0, 1]), row("heldout", "missing", [999], "eval")]
    before = compare_sampling(records, steps=2)
    records[1]["positions"] = list(range(9999))
    after = compare_sampling(records, steps=2)
    assert before == after
    assert before["evaluation_families_absent_from_training"] == ["missing"]
    roster, _ = sampling_distribution(records, "uniform_rows")
    assert {r["id"] for r in roster} == {"a"}


def test_same_random_variates_couple_identical_distributions():
    a = sample_indices([.1, .9], steps=10, rows_per_pass=64, seed=42)
    b = sample_indices([.1, .9], steps=10, rows_per_pass=64, seed=42)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (10, 64)
    assert not np.array_equal(a, sample_indices([.1, .9], steps=10, rows_per_pass=64, seed=43))


@pytest.mark.parametrize("bad", [[], [row("a", "a", [], "eval")], [row("a", "a", [])], [row("a", "a", [0, 0])], [row("a", "a", [-1])], [row("a", "", [0])], [row("a", "a", [0], "unknown")], [row("a", "a", [0])]*2])
def test_invalid_training_rosters_fail(bad):
    with pytest.raises(ValueError):
        sampling_distribution(copy.deepcopy(bad), "uniform_rows")


@pytest.mark.parametrize("bad", [[], [0, 1], [-1, 2], [float("nan")], [.2, .2]])
def test_invalid_distributions_fail(bad):
    with pytest.raises(ValueError):
        sample_indices(bad, steps=2, rows_per_pass=64, seed=42)


def test_both_arms_keep_matched_row_budget():
    result = compare_sampling([row("a", "a", [0, 1]), row("b", "b", [0])], steps=10, rows_per_pass=64)
    assert result["distinct_sampling_distribution"]
    for arm in result["arms"]:
        assert arm["sampled_rows_per_pass"] == 640
        assert sum(arm["family_sampled_rows_per_pass"].values()) == 640
