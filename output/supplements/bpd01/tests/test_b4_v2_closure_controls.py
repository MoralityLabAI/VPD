"""Fast CPU-only regression tests for B4 precision/ablation audit."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mechinterp.bpd.components import PassMaskSchedule, TiedComponentLinear
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.scripts.audit_b4_v2_controls_and_ablation_split import (
    per_board_kl,
    summarize_per_board,
    zero_component_and_delta_means,
)


def test_per_board_kl_and_full_precision_receipt() -> None:
    torch.manual_seed(17)
    clean = torch.randn(2, 5, 9)
    pred = clean.clone()
    pred[1, :, 2] += 1.3
    values = per_board_kl(pred, clean)
    assert len(values) == 2
    assert abs(values[0]) < 1e-6
    assert values[1] > 0.0
    stats = summarize_per_board(values)
    assert stats["per_board"] == values
    assert abs(stats["mean"] - sum(values) / 2.0) < 1e-8
    assert stats["top3_boards"][0]["board_idx"] == 1


def test_both_means_zeroed_and_restored_after_exception() -> None:
    wrapper = SimpleNamespace(
        mean_comp_acts=torch.tensor([[1.0, 2.0]]),
        mean_delta_out=torch.tensor([[4.0, -3.0]]),
    )
    orig_comp, orig_delta = wrapper.mean_comp_acts.clone(), wrapper.mean_delta_out.clone()
    with pytest.raises(RuntimeError, match="intentional"):
        with zero_component_and_delta_means({"module": wrapper}):
            assert torch.count_nonzero(wrapper.mean_comp_acts) == 0
            assert torch.count_nonzero(wrapper.mean_delta_out) == 0
            raise RuntimeError("intentional")
    torch.testing.assert_close(wrapper.mean_comp_acts, orig_comp)
    torch.testing.assert_close(wrapper.mean_delta_out, orig_delta)


def test_zero_plus_sphere_matches_explicit_write_formula() -> None:
    torch.manual_seed(91)
    layer = nn.Linear(4, 4, bias=False)
    wrapper = TiedComponentLinear(
        layer, c=4, max_passes=1,
        ablation=AblationConfig(mode="pass_mean", sphere_rescale=True),
        init_svd=True,
    )
    x = torch.randn(2, 5, 4)
    wrapper.observe_clean_pass(x, 0)
    # Artificial nonzero calibrated baselines ensure both must be suppressed.
    wrapper.mean_comp_acts[0].fill_(4.0)
    wrapper.mean_delta_out[0].fill_(7.0)
    component_mask = torch.zeros((2, 1, 5, 4))
    component_mask[..., 0] = 1.0
    delta_mask = torch.zeros((2, 1, 5))
    wrapper.mode = "component"
    wrapper.schedule = PassMaskSchedule(comp_mask=component_mask, delta_mask=delta_mask)

    with zero_component_and_delta_means({"module": wrapper}):
        wrapper.reset_pass_state()
        actual = wrapper(x)
        with torch.no_grad():
            active_acts = (x.float() @ wrapper.V) * component_mask[:, 0]
            raw = active_acts @ wrapper.U
            eps = wrapper.ablation.eps
            cur = (raw.pow(2).mean(-1, keepdim=True) + eps).sqrt()
            ablation_frac = 1.0 - component_mask[:, 0].mean(-1, keepdim=True)
            desired = (1.0 - ablation_frac) * cur + ablation_frac * wrapper.mean_write_rms[0]
            scale = torch.where(
                raw.pow(2).mean(-1, keepdim=True) > eps,
                desired / cur,
                torch.ones_like(cur),
            )
            expected = raw * scale
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert torch.all(wrapper.mean_comp_acts[0] == 4.0)
    assert torch.all(wrapper.mean_delta_out[0] == 7.0)
