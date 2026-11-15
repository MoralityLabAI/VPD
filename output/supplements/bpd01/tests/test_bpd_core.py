"""Unit tests for BPD core algebra, exact T=1 VPD reduction, sphere-rescaled
pass-mean ablation, temporal switch minimality, and time-shared PersistentPGD.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from mechinterp.bpd.adversary import PassSharedPersistentPGD, evaluate_pgd_stability_curve
from mechinterp.bpd.components import (
    BPDModel,
    PassMaskSchedule,
    RolloutTrajectory,
    lower_leaky_sigmoid,
    upper_leaky_sigmoid,
)
from mechinterp.bpd.config import (
    AblationConfig,
    AdversaryConfig,
    BPDConfig,
    MinimalityConfig,
)
from mechinterp.bpd.losses import (
    bpd_minimality_loss,
    faithfulness_loss,
    kl_divergence_logits,
    trajectory_recon_loss,
)


class _SinglePassMLP(nn.Module):
    def __init__(self, d_in: int = 16, d_hid: int = 24, d_out: int = 12) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid, bias=True)
        self.fc2 = nn.Linear(d_hid, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def test_leaky_sigmoids_match_goodfire_vpd_semantics() -> None:
    x = torch.tensor([-2.0, 0.4, 1.5], requires_grad=True)
    y_low = lower_leaky_sigmoid(x, alpha=0.01)
    assert torch.allclose(y_low.detach(), torch.tensor([0.0, 0.4, 1.0]))

    # Negative upstream gradient on x <= 0 triggers resurrection leak alpha * grad
    grad_neg = torch.autograd.grad(
        y_low, x, grad_outputs=torch.tensor([-3.0, 2.0, -5.0]), retain_graph=True
    )[0]
    assert torch.allclose(grad_neg, torch.tensor([-0.03, 2.0, 0.0]), atol=1e-7)

    # Positive upstream gradient on x <= 0 stays blocked at 0
    grad_pos = torch.autograd.grad(y_low, x, grad_outputs=torch.tensor([3.0, 2.0, 5.0]))[0]
    assert torch.allclose(grad_pos, torch.tensor([0.0, 2.0, 0.0]), atol=1e-7)

    y_up = upper_leaky_sigmoid(torch.tensor([-1.0, 0.6, 3.0]), alpha=0.01)
    assert torch.allclose(y_up, torch.tensor([0.0, 0.6, 1.02]), atol=1e-7)


def test_single_pass_reduces_identically_to_vpd() -> None:
    torch.manual_seed(42)
    net = _SinglePassMLP()
    cfg = BPDConfig.vpd_single_pass_equivalent({"fc1": 6, "fc2": 5}, seq_len=8, d_model=32)
    bpd = BPDModel(net, cfg.c_per_module, max_passes=1, ablation=cfg.ablation, init_svd=False)

    # 1. Faithfulness loss exact match to upstream VPD formula
    w1, w2 = bpd.wrappers["fc1"], bpd.wrappers["fc2"]
    ref_d1 = w1.W_target - (w1.V @ w1.U).T
    ref_d2 = w2.W_target - (w2.V @ w2.U).T
    ref_faith = (ref_d1.pow(2).sum() + ref_d2.pow(2).sum()) / (ref_d1.numel() + ref_d2.numel())
    assert torch.allclose(faithfulness_loss(bpd.wrappers), ref_faith, atol=1e-7)

    # 2. Minimality loss at T=1 exact match to Goodfire importance_minimality_loss
    g1 = torch.rand(4, 1, 8, 6)
    g2 = torch.rand(4, 1, 8, 5)
    min_loss, stats = bpd_minimality_loss(
        {"fc1": g1, "fc2": g2}, step=250, total_steps=1000, cfg=cfg.minimality
    )
    assert stats["switch_weighted"] == 0.0

    p = cfg.minimality.p_start + (cfg.minimality.p_end - cfg.minimality.p_start) * 0.25
    cat_lp = torch.cat(
        [(g1[:, 0] + cfg.minimality.eps).pow(p), (g2[:, 0] + cfg.minimality.eps).pow(p)], dim=-1
    )
    sum_c = cat_lp.sum(dim=-1)
    mean_c = cat_lp.mean(dim=-1)
    vpd_ref_min = cfg.minimality.coeff_level * (
        mean_c + cfg.minimality.beta * mean_c * torch.log2(1.0 + sum_c)
    ).mean()
    assert torch.allclose(min_loss, vpd_ref_min, atol=1e-7)

    # 3. Masked forward + trajectory loss exact match to 1-pass VPD KL
    x = torch.randn(4, 8, 16)
    with bpd.target_capture(pass_offset=0):
        clean_logits = net(x)
    clean_inputs = bpd.collect_recorded_inputs(expected_passes=1)

    m1 = torch.rand(4, 1, 8, 6)
    d1 = torch.rand(4, 1, 8)
    r_mask = torch.tensor([[True, False, True, True, False, True, False, True]] * 4).unsqueeze(1)
    m2 = torch.rand(4, 1, 8, 5)
    d2 = torch.rand(4, 1, 8)

    sched = {
        "fc1": PassMaskSchedule(comp_mask=m1, delta_mask=d1, routing_mask=r_mask),
        "fc2": PassMaskSchedule(comp_mask=m2, delta_mask=d2, routing_mask=r_mask),
    }
    with bpd.masked_components(sched):
        pred_logits = net(x)

    # Manual single-pass VPD algebra
    h1_comp = ((x @ w1.V) * m1[:, 0]) @ w1.U + w1.bias + d1[:, 0].unsqueeze(-1) * F.linear(x, ref_d1)
    h1_targ = F.linear(x, w1.W_target, w1.bias)
    h1 = F.gelu(torch.where(r_mask[:, 0].unsqueeze(-1), h1_comp, h1_targ))
    h2_comp = ((h1 @ w2.V) * m2[:, 0]) @ w2.U + d2[:, 0].unsqueeze(-1) * F.linear(h1, ref_d2)
    h2_targ = F.linear(h1, w2.W_target)
    ref_logits = torch.where(r_mask[:, 0].unsqueeze(-1), h2_comp, h2_targ)
    assert torch.allclose(pred_logits, ref_logits, atol=1e-6)

    clean_traj = RolloutTrajectory(
        readouts={0: clean_logits}, boundary_states={}, module_inputs=clean_inputs
    )
    pred_traj = RolloutTrajectory(readouts={0: pred_logits}, boundary_states={}, module_inputs={})
    rec_loss, _ = trajectory_recon_loss(pred_traj, clean_traj)
    assert torch.allclose(rec_loss, kl_divergence_logits(pred_logits, clean_logits), atol=1e-7)


def test_pass_mean_sphere_rescale_prevents_recurrent_norm_collapse() -> None:
    torch.manual_seed(7)
    d = 24
    lin_base = nn.Linear(d, d, bias=False)
    nn.init.orthogonal_(lin_base.weight)

    class _LoopMod(nn.Module):
        def __init__(self, proj: nn.Linear) -> None:
            super().__init__()
            self.proj = proj

        def forward(self, x0: torch.Tensor, n_passes: int) -> list[torch.Tensor]:
            h = x0
            outs = []
            for _ in range(n_passes):
                w = self.proj(h)
                outs.append(w)
                h = h + 0.3 * w
            return outs

    # Shifted non-zero-mean inputs so pass-conditional mean write is nonzero
    x = torch.randn(16, 6, d) + 0.5
    n_passes = 6

    m_sphere = _LoopMod(nn.Linear(d, d, bias=False))
    m_sphere.proj.weight.data.copy_(lin_base.weight.data)
    bpd_sphere = BPDModel(
        m_sphere,
        {"proj": 8},
        max_passes=n_passes,
        ablation=AblationConfig(mode="pass_mean", sphere_rescale=True),
        init_svd=True,
    )

    with bpd_sphere.target_capture(update_means=True):
        clean_writes = m_sphere(x, n_passes)

    # 1. Open mask (mask=1) matches clean target exactly
    ones_sched = {
        "proj": PassMaskSchedule(
            comp_mask=torch.ones(16, n_passes, 6, 8),
            delta_mask=torch.ones(16, n_passes, 6),
        )
    }
    with bpd_sphere.masked_components(ones_sched):
        open_writes = m_sphere(x, n_passes)
    for cw, ow in zip(clean_writes, open_writes):
        assert torch.allclose(cw, ow, atol=1e-5)

    # 2. Full ablation (mask=0) with pass_mean + sphere_rescale keeps write RMS on the pass sphere
    zero_sched = {
        "proj": PassMaskSchedule(
            comp_mask=torch.zeros(16, n_passes, 6, 8),
            delta_mask=torch.zeros(16, n_passes, 6),
        )
    }
    with bpd_sphere.masked_components(zero_sched):
        ablated_writes = m_sphere(x, n_passes)

    w_wrap = bpd_sphere.wrappers["proj"]
    for tau in range(n_passes):
        emp_rms = ablated_writes[tau].pow(2).mean(dim=-1).sqrt().mean()
        target_rms = w_wrap.mean_write_rms[tau]
        assert torch.allclose(emp_rms, target_rms, rtol=1e-3, atol=1e-4)

    # 3. Counterfactual single-component ablation preserves per-token write RMS on pass 0
    with bpd_sphere.intervene("proj", [0, 1], passes={0}, mode="ablate"):
        iv_writes = m_sphere(x, n_passes)
    clean_rms_0 = clean_writes[0].pow(2).mean(dim=-1).sqrt()
    iv_rms_0 = iv_writes[0].pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(clean_rms_0, iv_rms_0, rtol=1e-4, atol=1e-5)


def test_minimality_pass_invariance_and_switch_penalty() -> None:
    cfg = MinimalityConfig(pool_mode="mean", coeff_level=1e-3, coeff_switch=1e-3)

    # Stationary pattern [1, 0, 1, 0] across C=4 has identical cost at T=1, T=4, and T=16
    pattern = torch.tensor([1.0, 0.0, 1.0, 0.0]).view(1, 1, 1, 4)
    loss_t1, s1 = bpd_minimality_loss(
        {"m": pattern.expand(2, 1, 3, 4)}, step=50, total_steps=100, cfg=cfg
    )
    loss_t4, s4 = bpd_minimality_loss(
        {"m": pattern.expand(2, 4, 3, 4)}, step=50, total_steps=100, cfg=cfg
    )
    loss_t16, s16 = bpd_minimality_loss(
        {"m": pattern.expand(2, 16, 3, 4)}, step=50, total_steps=100, cfg=cfg
    )
    assert s1["switch_weighted"] == 0.0
    assert s4["switch_weighted"] == 0.0
    assert s16["switch_weighted"] == 0.0
    assert torch.allclose(loss_t1, loss_t4, atol=1e-7)
    assert torch.allclose(loss_t1, loss_t16, atol=1e-7)

    # Splitting 1 stationary mechanism across 2 alternating components incurs strictly higher cost
    stationary_1 = torch.zeros(2, 4, 3, 2)
    stationary_1[..., 0] = 1.0  # c=0 active all passes, c=1 off

    split_2 = torch.zeros(2, 4, 3, 2)
    split_2[:, 0::2, :, 0] = 1.0  # c=0 on even passes
    split_2[:, 1::2, :, 1] = 1.0  # c=1 on odd passes

    l_stat, st_stat = bpd_minimality_loss({"m": stationary_1}, step=80, total_steps=100, cfg=cfg)
    l_split, st_split = bpd_minimality_loss({"m": split_2}, step=80, total_steps=100, cfg=cfg)
    assert st_stat["switch_weighted"] == 0.0
    assert st_split["switch_weighted"] > 0.0
    assert float(l_split.item()) > float(l_stat.item())


def test_persistent_pgd_time_shared_vs_per_pass() -> None:
    torch.manual_seed(9)
    ci = {"m": torch.full((3, 4, 5, 6), 0.25)}
    target_dir = torch.tensor([1.0, -1.0, 1.0, -1.0]).view(1, 4, 1, 1)

    # Synthetic resonant objective: alternating pass signs reward per_pass adversary over time-shared
    def resonant_loss_fn(sched: dict[str, PassMaskSchedule]) -> torch.Tensor:
        cm = sched["m"].comp_mask  # [3, 4, 5, 6]
        # Quadratic + resonant cross-pass term
        base = (cm - 0.8).pow(2).mean()
        resonant = (cm * target_dir).mean() * 0.5
        return base + resonant

    pgd_shared = PassSharedPersistentPGD(
        {"m": 6}, AdversaryConfig(scope="shared_batch_and_pass", inner_steps=3, lr=0.1)
    )
    sched0, _ = pgd_shared.build_schedules(ci)
    assert pgd_shared.sources["m"].shape == (1, 1, 1, 7)
    loss_before = float(resonant_loss_fn(sched0).item())
    pgd_shared.warmup_inner_steps(ci, resonant_loss_fn)
    sched1, _ = pgd_shared.build_schedules(ci)
    loss_after = float(resonant_loss_fn(sched1).item())
    assert loss_after > loss_before

    curve_shared = evaluate_pgd_stability_curve(
        ci, {"m": 6}, resonant_loss_fn, scope="shared_batch_and_pass", steps_grid=(0, 5, 15)
    )
    curve_resonant = evaluate_pgd_stability_curve(
        ci, {"m": 6}, resonant_loss_fn, scope="per_pass", steps_grid=(0, 5, 15)
    )
    # Unconstrained per-pass adversary exploits pass-alternating resonance more strongly
    assert curve_resonant.loss_by_steps[15] >= curve_shared.loss_by_steps[15] - 1e-5
