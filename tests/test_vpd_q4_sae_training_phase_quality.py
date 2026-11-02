import pytest
from experiments.authz_mechinterp.audit_q4_sae_training_phase_quality_v1 import compare_common_families


def phases(families):
    return [{"phase": phase, "family_reconstruction_quality": [
        {"family": name, "examples": count, "per_pass_ev": ev} for name, count, ev in families]}
        for phase in ("prompt", "generated")]


def test_common_labels_do_not_pair_different_families():
    train = phases([("shared", 4, [.9, .8]), ("train_only", 1, [.99, .99])])
    evaluation = phases([("eval_only", 1, [.1, .1]), ("shared", 6, [.2, .3])])
    result = compare_common_families(train, evaluation)
    for phase in result:
        assert phase["training_only_families"] == ["train_only"]
        assert phase["evaluation_only_families"] == ["eval_only"]
        assert len(phase["common_family_labels"]) == 1
        shared = phase["common_family_labels"][0]
        assert shared["training_examples"] == 4
        assert shared["evaluation_examples"] == 6
        assert shared["descriptive_training_minus_evaluation_ev"] == pytest.approx([.7, .5])


def test_undefined_ev_does_not_become_numerical_gap():
    result = compare_common_families(phases([("a", 1, [None, .7])]), phases([("a", 1, [.2, None])]))
    assert result[0]["common_family_labels"][0]["descriptive_training_minus_evaluation_ev"] == [None, None]


def test_no_shared_families_means_no_comparison():
    result = compare_common_families(phases([("a", 1, [.9, .9])]), phases([("b", 1, [.1, .1])]))
    assert all(not phase["common_family_labels"] for phase in result)
