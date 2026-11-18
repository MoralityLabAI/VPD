"""Closed-loop CI training (B4-CAUSAL-CI-02) on the mini TRM."""
from __future__ import annotations

import torch

from mechinterp.bpd.adapters.trm import TRMBPDAdapter, build_mini_trm, default_trm_c_per_module
from mechinterp.bpd.causal_ci import causal_ci_rollout, clean_informed_ci_masks
from mechinterp.bpd.causal_ci_train import (
    causal_topk_rollout_st,
    clean_reference,
    closed_loop_loss,
    straight_through_topk,
    teacher_schedule,
    train_closed_loop_ci,
    validate_causal_topk,
)
from mechinterp.bpd.ci import TiedCINetwork
from mechinterp.bpd.components import BPDModel, PassMaskSchedule
from mechinterp.bpd.config import AblationConfig, CIConfig

N_PASSES = 6
K = 9


def _setup(c: int = 16, seed: int = 3):
    torch.manual_seed(seed)
    model = build_mini_trm(hidden_size=32, num_heads=4, l_layers=2, h_cycles=1,
                           l_cycles=2, puzzle_emb_len=4, seq_len=16, seed=113).float().eval()
    bpd = BPDModel(model, default_trm_c_per_module(model, c=c), max_passes=N_PASSES,
                   ablation=AblationConfig(mode="zero", sphere_rescale=False), init_svd=True)
    adapter = TRMBPDAdapter(bpd.target_model, readout_at="outer_step")
    dims = {m: bpd.wrappers[m].d_in for m in bpd.module_order}
    ci = TiedCINetwork(dims, bpd.c_per_module,
                       CIConfig(variant="state_only", d_model=32, n_blocks=1, n_heads=4,
                                mlp_hidden=64, max_passes=16, max_seq_len=32)).eval()
    with torch.no_grad():
        ci.out_proj.weight.normal_(0.0, 0.5)
    inputs = torch.randint(1, 11, (3, 16), dtype=torch.long)
    return bpd, adapter, ci, inputs


def test_straight_through_topk_forward_is_hard_and_backward_is_sigmoid() -> None:
    logits = torch.randn(2, 1, 5, 12, requires_grad=True)
    m = straight_through_topk(logits, 4)
    assert torch.equal(m.detach().sum(-1), torch.full((2, 1, 5), 4.0))
    assert set(m.detach().unique().tolist()) <= {0.0, 1.0}
    m.sum().backward()
    s = torch.sigmoid(logits.detach())
    assert torch.allclose(logits.grad, s * (1 - s))


def test_st_rollout_forward_equals_scored_causal_rollout() -> None:
    bpd, adapter, ci, inputs = _setup()
    out = causal_topk_rollout_st(bpd, adapter, ci, inputs, N_PASSES, K)
    ref = causal_ci_rollout(bpd, adapter, ci, inputs, N_PASSES, observation="preflight_target",
                            selection="topk", k=K)
    assert set(out["readouts"]) == set(ref["readouts"])
    for t, v in ref["readouts"].items():
        assert torch.equal(out["readouts"][t].detach(), v), t


def test_full_teacher_forcing_equals_clean_informed_masks() -> None:
    bpd, adapter, ci, inputs = _setup()
    _, _, clean_in = clean_reference(bpd, adapter, inputs, N_PASSES)
    out = causal_topk_rollout_st(bpd, adapter, ci, inputs, N_PASSES, K, clean_inputs=clean_in,
                                 teacher_prob=1.0, generator=torch.Generator().manual_seed(0))
    t2 = clean_informed_ci_masks(bpd, adapter, ci, inputs, N_PASSES, selection="topk", k=K)
    sched = {m: PassMaskSchedule(comp_mask=v, delta_mask=torch.zeros(inputs.shape[0], N_PASSES, 1))
             for m, v in t2.items()}
    with torch.no_grad(), bpd.masked_components(sched, pass_offset=0):
        ro, _, _, _ = adapter.rollout(inputs, N_PASSES, 0, None)
    for t, v in ro.items():
        assert torch.allclose(out["readouts"][t].detach(), v, atol=1e-6), t
    assert out["teacher_fraction"] == [1.0] * N_PASSES


def test_loss_backpropagates_to_ci_only() -> None:
    bpd, adapter, ci, inputs = _setup()
    for p in bpd.parameters():
        p.requires_grad_(False)
    clean_ro, clean_h, _ = clean_reference(bpd, adapter, inputs, N_PASSES)
    assert len(clean_h) >= 1
    out = causal_topk_rollout_st(bpd, adapter, ci, inputs, N_PASSES, K)
    loss, parts = closed_loop_loss(out, clean_ro, clean_h)
    loss.backward()
    assert parts["kl_mean"] > 0.0
    assert ci.out_proj.weight.grad is not None and float(ci.out_proj.weight.grad.abs().sum()) > 0.0
    assert ci.proj_in.weight.grad is not None and float(ci.proj_in.weight.grad.abs().sum()) > 0.0
    assert all(p.grad is None for p in bpd.parameters())


def test_teacher_schedule_and_short_training_keeps_best_validation() -> None:
    assert teacher_schedule(0, 10, 0.5) == 1.0
    assert teacher_schedule(5, 10, 0.5) == 0.0
    assert teacher_schedule(9, 10, 0.5) == 0.0
    bpd, adapter, ci, inputs = _setup()
    val = torch.randint(1, 11, (2, 16), dtype=torch.long)
    v0 = validate_causal_topk(bpd, adapter, ci, val, N_PASSES, K)
    net, info = train_closed_loop_ci(bpd, adapter, ci, lambda s: inputs, val, n_passes=N_PASSES, k=K,
                                     n_steps=6, lr=3e-3, val_every=3, seed=0)
    assert len(info["history"]) == 6
    assert [v["step"] for v in info["validation"]] == [0, 3, 6]
    assert info["best_validation"]["kl_mean"] <= v0["kl_mean"] + 1e-12
    v_best = validate_causal_topk(bpd, adapter, net, val, N_PASSES, K)
    assert abs(v_best["kl_mean"] - info["best_validation"]["kl_mean"]) < 1e-9
