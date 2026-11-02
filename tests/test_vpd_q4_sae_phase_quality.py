import numpy as np
import pytest
from experiments.authz_mechinterp.audit_q4_sae_phase_quality_v1 import phase_labels, explained_variance, phase_quality


def test_prompt_boundary_is_not_inferred_from_row_number():
    assert phase_labels([8, 9, 10, 20], 10, 21).tolist() == ["prompt", "prompt", "generated", "generated"]


def test_mixed_quality_does_not_establish_phase_quality():
    x = np.array([[-10.], [-9.], [9.], [10.]])
    reconstruction = x+1
    assert explained_variance(x, reconstruction) > .9
    results = phase_quality([x, x], [reconstruction, reconstruction], ["a"]*4,
                            ["prompt"]*2+["generated"]*2, ["example"]*4, ["a"])
    for result in results:
        assert result["pooled_ev"] == [-3., -3.]
        assert not result["quality_decision"]["quality_prerequisite_pass"]


def test_absent_phase_and_family_remain_unevaluable():
    x = np.array([[-1.], [1.]])
    results = phase_quality([x, x], [x, x], ["a"]*2, ["prompt"]*2, ["one"]*2, ["a", "b"])
    assert results[0]["quality_decision"]["unevaluable_families"] == ["b"]
    assert results[1]["quality_decision"]["unevaluable_families"] == ["a", "b"]
    assert results[1]["pooled_ev"] == [None, None]


def test_examples_not_token_count():
    x = np.array([[-1.], [1.], [-1.], [1.]])
    results = phase_quality([x, x], [x, x], ["a"]*4, ["generated"]*4, ["one"]*2+["two"]*2, ["a"])
    assert results[1]["family_reconstruction_quality"][0]["examples"] == 2


@pytest.mark.parametrize("positions,prompt,total", [([0, 0], 1, 2), ([-1], 1, 2), ([2], 1, 2), ([0.], 1, 2), ([0], 0, 2), ([0], 3, 2)])
def test_bad_phase_metadata_rejected(positions, prompt, total):
    with pytest.raises(ValueError):
        phase_labels(positions, prompt, total)


def test_undefined_variance_not_a_pass():
    assert explained_variance([[1], [1]], [[1], [1]]) is None
