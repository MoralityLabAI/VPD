"""Online SVD selector regressions: causal inputs and oracle-mask isolation."""
from __future__ import annotations

import torch
import pytest
from torch import nn

from mechinterp.bpd.components import PassMaskSchedule, TiedComponentLinear


def _identity_wrapper() -> TiedComponentLinear:
    linear = nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.eye(3))
    wrapper = TiedComponentLinear(linear, c=3, max_passes=3, init_svd=True)
    with torch.no_grad():
        wrapper.V.copy_(torch.eye(3))
        wrapper.U.copy_(torch.eye(3))
    wrapper.mode = "component"
    return wrapper


def _run(wrapper: TiedComponentLinear, x: torch.Tensor, mask: torch.Tensor, *, online: bool):
    wrapper.mode = "component"
    wrapper.schedule = PassMaskSchedule(
        comp_mask=mask,
        delta_mask=torch.zeros(x.shape[0], 1, x.shape[1]),
        online_topk=1 if online else None,
    )
    wrapper.reset_pass_state()
    out = wrapper(x)
    disagreement = wrapper.online_mask_disagreement_by_pass
    return out, disagreement


def test_online_uses_live_input_without_oracle_mask_feedback() -> None:
    wrapper = _identity_wrapper()
    x = torch.tensor([[[0.1, 0.2, 7.0], [9.0, 0.3, 0.1]]])
    oracle_0 = torch.zeros(1, 1, 2, 3)
    oracle_0[..., 0] = 1.0
    oracle_1 = torch.zeros_like(oracle_0)
    oracle_1[..., 1] = 1.0

    online_0, dis_0 = _run(wrapper, x, oracle_0, online=True)
    online_1, dis_1 = _run(wrapper, x, oracle_1, online=True)
    torch.testing.assert_close(online_0, online_1, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        online_0, torch.tensor([[[0.0, 0.0, 7.0], [9.0, 0.0, 0.0]]])
    )
    assert len(dis_0) == len(dis_1) == 1
    assert dis_0[0].shape == (1,)
    # Oracle reference swaps, so mismatch telemetry must change while output stays the same.
    assert abs(float(dis_0[0][0]) - float(dis_1[0][0])) > 0.0


def test_online_gating_tracks_perturbed_input() -> None:
    wrapper = _identity_wrapper()
    oracle = torch.zeros(1, 1, 1, 3)
    oracle[..., 0] = 1.0
    first, _ = _run(wrapper, torch.tensor([[[8.0, 1.0, 0.0]]]), oracle, online=True)
    second, _ = _run(wrapper, torch.tensor([[[0.0, 1.0, 8.0]]]), oracle, online=True)
    assert first[0, 0, 0] == 8.0
    assert second[0, 0, 2] == 8.0
    assert first[0, 0, 2] == 0.0
    assert second[0, 0, 0] == 0.0


def test_default_offline_schedule_unchanged() -> None:
    wrapper = _identity_wrapper()
    x = torch.tensor([[[0.0, 0.0, 10.0]]])
    mask = torch.zeros(1, 1, 1, 3)
    mask[..., 0] = 1.0
    off, report = _run(wrapper, x, mask, online=False)
    assert torch.count_nonzero(off) == 0
    assert report == []


def test_online_topk_validation() -> None:
    wrapper = _identity_wrapper()
    x = torch.ones(1, 1, 3)
    mask = torch.ones(1, 1, 1, 3)
    wrapper.schedule = PassMaskSchedule(
        comp_mask=mask, delta_mask=torch.zeros(1, 1, 1), online_topk=4
    )
    wrapper.reset_pass_state()
    with pytest.raises(ValueError, match="online_topk"):
        wrapper(x)
