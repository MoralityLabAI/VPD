import pytest
from experiments.authz_mechinterp.audit_q4_common_capture_rows_v1 import common_prefix_rows


def test_same_current_token_after_divergence_is_not_same_context():
    result = common_prefix_rows([1, 2, 3, 4], [0, 2, 3], [1, 9, 3, 4], [0, 2, 3])
    assert result["position_intersection"] == [0, 2, 3]
    assert result["prefix_verified_positions"] == [0]


def test_window_offsets_do_not_pair_same_row_number():
    result = common_prefix_rows([1, 2, 3, 4, 5], [0, 2, 4], [1, 2, 3, 4, 5], [0, 1, 2])
    assert result["prefix_verified_positions"] == [0, 2]
    assert result["left_row_indices"] == [0, 1]
    assert result["right_row_indices"] == [0, 2]


def test_shorter_identical_prefix_stays_eligible():
    result = common_prefix_rows([1, 2], [0, 1], [1, 2, 3], [0, 1, 2])
    assert result["prefix_verified_positions"] == [0, 1]


def test_changed_current_token_is_excluded():
    assert common_prefix_rows([1, 2], [0, 1], [1, 3], [0, 1])["prefix_verified_positions"] == [0]


@pytest.mark.parametrize("tokens,positions", [([], [0]), ([1], [1]), ([1], [-1]), ([1], [0, 0]), ([1.], [0]), ([1], [0.]), ([-1], [0])])
def test_bad_alignment_inputs_fail(tokens, positions):
    with pytest.raises(ValueError):
        common_prefix_rows(tokens, positions, [1], [0])
