import math

import pytest

from experiments.authz_mechinterp.audit_q4_sae_coverage_family_controls_v1 import family_controls, DISTANCE, ERROR


def rows(xs, ys, labels):
    return [{"id": str(i), "family": f, DISTANCE: x, ERROR: y} for i, (x, y, f) in enumerate(zip(xs, ys, labels))]


def test_family_only_association_disappears_after_centering():
    result = family_controls(rows([1, 1, 10, 10], [2, 2, 20, 20], ["a", "a", "b", "b"]))
    assert result["pooled_pearson_r"] == pytest.approx(1)
    assert result["family_centered_pearson_r"] is None
    assert result["posthoc_conditional_upper_tail_p"] is None
    assert result["within_family_cross_product"] == 0


def test_perfect_within_family_order_has_exact_factorial_null():
    result = family_controls(rows([1, 2, 3, 8, 9, 10], [2, 4, 6, 16, 18, 20], ["a"]*3+["b"]*3))
    assert result["family_centered_pearson_r"] == pytest.approx(1)
    assert result["exact_within_family_permutations"] == 36
    assert result["posthoc_conditional_upper_tail_p"] == pytest.approx(1/36)
    assert result["within_family_cross_product"]+result["between_family_cross_product"] == pytest.approx(result["total_centered_cross_product"])


def test_simpson_reversal_is_retained_not_hidden():
    result = family_controls(rows([1, 2, 3, 11, 12, 13], [3, 2, 1, 13, 12, 11], ["a"]*3+["b"]*3))
    assert result["pooled_pearson_r"] > 0
    assert result["family_centered_pearson_r"] == pytest.approx(-1)
    assert result["within_family_fraction_of_total_cross_product"] < 0
    assert result["posthoc_conditional_upper_tail_p"] == 1


def test_singletons_do_not_create_conditional_information():
    result = family_controls(rows([1, 2, 8], [3, 2, 20], ["a", "a", "b"]))
    assert result["examples_in_nonsingleton_families"] == 2
    assert result["families_with_within_family_variation_possible"] == 1
    assert result["exact_within_family_permutations"] == 2
    assert all(r["descriptive_pearson_r"] is None for r in result["family_sizes"])


def test_input_order_does_not_change_control_report():
    examples = rows([1, 2, 4, 5], [3, 7, 5, 8], ["a", "a", "b", "b"])
    assert family_controls(examples) == family_controls(examples[::-1])


@pytest.mark.parametrize("bad", [[], rows([1, 2, 3], [1, float("nan"), 3], ["a"]*3), rows([-1, 2, 3], [1, 2, 3], ["a"]*3)])
def test_invalid_inputs_rejected(bad):
    with pytest.raises(ValueError):
        family_controls(bad)


def test_duplicate_ids_and_unbounded_permutations_rejected():
    examples = rows([1, 2, 3], [1, 2, 3], ["a"]*3)
    examples[1]["id"] = examples[0]["id"]
    with pytest.raises(ValueError, match="unique"):
        family_controls(examples)
    assert math.factorial(9) > 100_000
    with pytest.raises(ValueError, match="bounded"):
        family_controls(rows(range(9), range(9), ["a"]*9))
