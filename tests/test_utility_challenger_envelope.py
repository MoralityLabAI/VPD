import math

import pytest

from experiments.authz_mechinterp.utility_challenger_envelope import evaluate_selection


def evaluate(scores, selected="q", baseline="b", n=10000, rho=0):
    columns = {key: [value] * n if value is not None else None for key, value in scores.items()}
    return evaluate_selection(
        block_ids=[str(i) for i in range(n)], utility=columns, benign=columns,
        selected=selected, baseline=baseline, error_radius=rho,
    )


def test_positive_baseline_improvement_can_have_large_regret():
    result = evaluate({"b": 0.2, "q": 0.5, "missed": 0.9})
    assert result["baseline_improvement"]["interval"][0] > 0
    assert result["regret"]["interval"][0] > 0.3
    assert result["regret"]["point"] == pytest.approx(0.4)
    assert result["regret"]["upper_bound_attainers"] == ["missed"]
    assert result["hoeffding_radius"] == pytest.approx(math.sqrt(math.log(12 / 0.05) / 20000))


def test_missing_challenger_is_not_removed_or_given_a_point_estimate():
    result = evaluate({"b": 0.3, "q": 0.6, "unmeasured": None})
    assert result["intervals"]["unmeasured"]["utility"]["interval"] == [0, 1]
    assert result["regret"]["point"] is None
    assert result["regret"]["interval"][1] > 0.4
    assert "unmeasured" in result["regret"]["possible_missed_challengers"]


def test_self_comparisons_cancel_even_with_score_error():
    result = evaluate({"b": 0.5}, selected="b", rho=0.05)
    assert result["baseline_improvement"]["interval"] == [0, 0]
    assert result["regret"]["interval"] == [0, 0]
    assert result["benign_loss_interval"] == [0, 0]


def test_declared_error_monotonically_widens_regret():
    clean = evaluate({"b": 0.3, "q": 0.6, "p": 0.6})
    stressed = evaluate({"b": 0.3, "q": 0.6, "p": 0.6}, rho=0.05)
    assert stressed["regret"]["interval"][1] == pytest.approx(clean["regret"]["interval"][1] + 0.1)


def test_missing_cell_keeps_full_mean_range():
    result = evaluate_selection(
        block_ids=["a", "b"], utility={"q": [0.8, None]}, benign={"q": [1, 1]},
        selected="q", baseline="q",
    )
    assert result["intervals"]["q"]["utility"]["missing_blocks"] == 1
    assert result["intervals"]["q"]["utility"]["interval"] == [0, 1]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_scores_fail_closed(bad):
    with pytest.raises(ValueError, match="scores"):
        evaluate({"b": 0.3, "q": bad}, n=2)


def test_duplicate_blocks_and_mismatched_catalogs_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_selection(block_ids=["x", "x"], utility={"b": [1, 1]},
                           benign={"b": [1, 1]}, selected="b", baseline="b")
    with pytest.raises(ValueError, match="entire catalog"):
        evaluate_selection(block_ids=["x"], utility={"b": [1], "p": None},
                           benign={"b": [1]}, selected="b", baseline="b")
