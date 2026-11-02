import numpy as np
import pytest
from experiments.authz_mechinterp.audit_q4_sae_family_contributions_v1 import summarize_family_advantages


def test_family_summary_exposes_single_family_dependence():
    result = summarize_family_advantages([10, -1, -1, -1])
    assert result["equal_family_mean_advantage"] == pytest.approx(1.75)
    assert result["minimum_leave_one_family_out_mean"] == -1
    assert result["families_with_positive_advantage"] == 1
    assert result["family_advantage_sample_sd"] == pytest.approx(5.5)


def test_constant_family_advantage_has_no_infinite_standardized_effect():
    result = summarize_family_advantages([1, 1, 1, 1])
    assert result["mean_over_sample_sd_descriptive"] is None
    assert result["minimum_leave_one_family_out_mean"] == 1


@pytest.mark.parametrize("bad", [[1], [], [np.nan, 1], [[1, 2], [3, 4]]])
def test_invalid_family_scores_fail(bad):
    with pytest.raises(ValueError):
        summarize_family_advantages(bad)
