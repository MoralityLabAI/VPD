from __future__ import annotations

import math

import pytest

from experiments.authz_mechinterp.logit_margin_metric import (
    candidate_argmax_valid,
    decision_logit_margin,
    token_logprob,
)


def test_margin_is_common_offset_invariant() -> None:
    assert decision_logit_margin(3.0, 1.0) == decision_logit_margin(13.0, 11.0)


def test_candidate_argmax_guard_rejects_unrelated_token() -> None:
    assert candidate_argmax_valid([2.0, 1.0, 0.0], 0, 1)
    assert not candidate_argmax_valid([2.0, 1.0, 3.0], 0, 1)


def test_token_logprob_is_stable_and_monotonic() -> None:
    baseline = token_logprob([-5.0, 0.0, 0.0], 0)
    shifted = token_logprob([0.0, 5.0, 5.0], 0)
    increased = token_logprob([-4.5, 0.0, 0.0], 0)
    assert shifted == pytest.approx(baseline, abs=1e-12)
    assert increased > baseline
    assert math.isfinite(baseline)
