import pytest

from experiments.authz_mechinterp.audit_bf16_q2_capture_coverage_v1 import position_coverage
from experiments.nanbeige_reasoning.positions import capture_positions


def test_eight_tail_rows_are_not_four_window_capture():
    result = position_coverage(100, 228, list(range(220, 228)))
    assert result["q2_window_layout_possible"]
    assert not result["q2_windows_present"]
    assert len(result["missing_positions"]) == 24
    assert result["region_coverage"] == {"prompt": {"required": 8, "available": 0}, "early": {"required": 8, "available": 0},
                                         "middle": {"required": 8, "available": 0}, "late": {"required": 8, "available": 8}}


def test_complete_window_positions_are_compatible_regardless_order():
    positions, _ = capture_positions(100, 228)
    result = position_coverage(100, 228, positions[::-1])
    assert result["q2_windows_present"]
    assert result["missing_positions"] == []


@pytest.mark.parametrize("positions", [[220, 220], [-1], [228], [1.0], [True]])
def test_invalid_absolute_positions_rejected(positions):
    with pytest.raises(ValueError):
        position_coverage(100, 228, positions)


def test_short_generation_is_not_admitted_even_if_every_token_saved():
    result = position_coverage(100, 104, list(range(104)))
    assert not result["q2_window_layout_possible"]
    assert not result["q2_windows_present"]
