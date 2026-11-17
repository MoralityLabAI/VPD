"""CPU source-lock and paired inference regression for B4 reserved64."""
from __future__ import annotations

import pytest

from mechinterp.bpd.scripts.audit_b4_online_svd_reserved64 import (
    BOARD_INDICES,
    BOARD_INDICES_SHA256,
    _collect_rows,
    analyze_rows,
    validate_registration,
)


def _rows(*, online_shift: float) -> list[dict]:
    return [
        {
            "board_index": BOARD_INDICES[i],
            "static_kl": 1.0,
            "oracle_kl": 0.4,
            "online_kl": 0.4 + online_shift,
            "target_repeat_kl": 0.0,
        }
        for i in range(64)
    ]


def test_source_index_lock() -> None:
    validate_registration(183)
    assert len(BOARD_INDICES) == len(set(BOARD_INDICES)) == 64
    assert min(BOARD_INDICES) > 15 and max(BOARD_INDICES) < 183
    assert BOARD_INDICES_SHA256.startswith("0f474925")
    with pytest.raises(AssertionError, match="source mismatch"):
        validate_registration(184)


def test_noninferiority_within_static_10pct_margin() -> None:
    stats = analyze_rows(_rows(online_shift=0.05))
    assert stats["n_boards"] == 64
    assert stats["noninferior_per_checkpoint"] is True
    assert stats["mean_delta_minus_margin"] == pytest.approx(-0.05)
    assert stats["upper_97_5pct_bootstrap_mean_delta_minus_margin"] < 0
    stats_repeat = analyze_rows(_rows(online_shift=0.05))
    assert stats == stats_repeat


def test_noninferiority_fails_outside_margin() -> None:
    stats = analyze_rows(_rows(online_shift=0.25))
    assert stats["noninferior_per_checkpoint"] is False
    assert stats["mean_delta_minus_margin"] > 0
    assert stats["upper_97_5pct_bootstrap_mean_delta_minus_margin"] > 0


def test_identity_control_failure_rejected_before_inference() -> None:
    rows = _rows(online_shift=0.0)
    rows[20]["target_repeat_kl"] = 1e-3
    with pytest.raises(ValueError, match="precision floor"):
        analyze_rows(rows)


def test_chunk_collection_pairs_and_mask_telemetry() -> None:
    n=8
    chunk={
        "readouts":[20],
        "board_sha256_prefix16":[f"{i:016x}" for i in range(n)],
        "same_dtype_target_rerun":{"pass_20":{"per_board":[0.0]*n}},
        "static_rank_k":{"pass_20":{"per_board":[0.6]*n}},
        "clean_oracle_dynamic_topk":{"pass_20":{"per_board":[0.2]*n}},
        "online_dynamic_topk":{"pass_20":{"per_board":[0.25]*n}},
        "online_vs_oracle_rollout":{"pass_20":{"per_board":[0.01]*n}},
        "paired_online_minus_oracle_KL_by_board":{"20":[0.05]*n},
        "oracle_mask_disagreement":{"test.site":{"by_pass_by_board":[[0.125]*n for _ in range(21)]}},
    }
    rows=_collect_rows(chunk, BOARD_INDICES[:n])
    assert len(rows)==8
    assert rows[0]["board_index"]==BOARD_INDICES[0]
    assert rows[-1]["online_mask_disagreement"]["test.site"] == pytest.approx(0.125)
    chunk["paired_online_minus_oracle_KL_by_board"]["20"][0]=0.5
    with pytest.raises(ValueError, match="paired-KL"):
        _collect_rows(chunk, BOARD_INDICES[:n])
