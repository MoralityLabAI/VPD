"""Unit tests for BPD core algebra, numeric T=1 parity against the four-term
PyTorch VPD port (vpd_math.py @ 69509c39), sphere-rescaled pass-mean ablation,
max-over-tau frequency minimality, fragility hinge, and time-shared PersistentPGD.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from mechinterp.bpd.adversary import PassSharedPersistentPGD, evaluate_pgd_stability_curve
from mechinterp.bpd.ci import TiedCINetwork
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
    boundary_state_loss,
    compute_pass_loss_shares,
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


def _load_vpd_math_module():
    """Load MoralityLabAI/VPD experiments/nanbeige_looped/vpd_math.py (@ 69509c39)."""
    here = Path(__file__).resolve()
    candidates = [
        # When running inside MoralityLabAI/VPD/output/supplements/bpd01/tests/
        here.parents[4] / "experiments" / "nanbeige_looped" / "vpd_math.py",
        # When running inside C:/Users/patri/LoopedTransformers/mechinterp/bpd/tests/
        Path(r"C:\Users\patri\Documents\Codex\VPD\experiments\nanbeige_looped\vpd_math.py"),
    ]
    for cand in candidates:
        if cand.exists():
            spec = importlib.util.spec_from_file_location("vpd_math_ref_69509c39", cand)
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            sys.modules["vpd_math_ref_69509c39"] = mod
            spec.loader.exec_module(mod)
            return mod
    pytest.skip("MoralityLabAI/VPD experiments/nanbeige_looped/vpd_math.py not present on disk")


def test_single_pass_matches_four_term_vpd_port_algebra() -> None:
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

    # 2. Minimality loss at T=1 exact match to vpd_math.importance_minimality_loss
    g1 = torch.rand(4, 1, 8, 6)
    g2 = torch.rand(4, 1, 8, 5)
    min_loss, stats = bpd_minimality_loss(
        {"fc1": g1, "fc2": g2}, step=250, total_steps=1000, cfg=cfg.minimality
    )
    assert stats["switch_weighted"] == 0.0

    p = cfg.minimality.p_start + (cfg.minimality.p_end - cfg.minimality.p_start) * 0.25
    ref_terms = []
    for g_mod in (g1[:, 0], g2[:, 0]):
        vals = (g_mod + cfg.minimality.eps).pow(p)  # [B=4, P=8, C]
        sum_c = vals.sum(dim=(0, 1))
        mean_c = sum_c / float(vals.shape[0] * vals.shape[1])
        ref_terms.append((mean_c + cfg.minimality.beta * mean_c * torch.log2(1.0 + sum_c)).sum())
    vpd_ref_min = cfg.minimality.coeff_level * (ref_terms[0] + ref_terms[1])
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


def test_t1_numeric_parity_against_vpd_math_torch_port() -> None:
    """Numeric T=1 parity against MoralityLabAI/VPD experiments/nanbeige_looped/vpd_math.py
    at commit 69509c39e96a912f99b1a83640250ac01797c571 (the four-term PyTorch VPD port:
    faithfulness + minimality + uniform-k-subset stochastic routing KL + persistent PGD KL;
    Goodfire's paper adds a fifth deterministic CI-masked reconstruction term).
    """
    vpd_math = _load_vpd_math_module()
    torch.manual_seed(123)
    c_map = {"fc1": 6, "fc2": 5}
    bsz, seq_len, d_in = 4, 8, 16

    net_bpd = _SinglePassMLP(d_in=d_in, d_hid=24, d_out=12)
    net_vpd = _SinglePassMLP(d_in=d_in, d_hid=24, d_out=12)
    net_vpd.load_state_dict(net_bpd.state_dict())

    bpd_cfg = BPDConfig.vpd_single_pass_equivalent(c_map, seq_len=seq_len, d_model=32, n_blocks=2, n_heads=4)
    bpd_model = BPDModel(net_bpd, c_map, max_passes=1, ablation=bpd_cfg.ablation, init_svd=False)

    vpd_cfg = vpd_math.Config(
        C_per_module=dict(c_map),
        n_steps=1000,
        batch_size=bsz,
        seq_len=seq_len,
        coeff_faith=bpd_cfg.coeff_faith,
        coeff_imp=bpd_cfg.minimality.coeff_level,
        coeff_stoch=bpd_cfg.coeff_stoch,
        coeff_ppgd=bpd_cfg.coeff_adv,
        p_start=bpd_cfg.minimality.p_start,
        p_end=bpd_cfg.minimality.p_end,
        imp_eps=bpd_cfg.minimality.eps,
        imp_beta=bpd_cfg.minimality.beta,
        ci_d_model=32,
        ci_n_blocks=2,
        ci_n_heads=4,
        ci_mlp_hidden=64,
    )
    vpd_wrappers = vpd_math.install_components(net_vpd, dict(c_map))
    with torch.no_grad():
        for name in c_map:
            vpd_wrappers[name].V.copy_(bpd_model.wrappers[name].V)
            vpd_wrappers[name].U.copy_(bpd_model.wrappers[name].U)

    # Term 1: Faithfulness loss parity
    f_bpd = faithfulness_loss(bpd_model.wrappers)
    f_vpd = vpd_math.faithfulness_loss(vpd_wrappers)
    assert torch.allclose(f_bpd, f_vpd, atol=1e-7)

    # CI Transformer parity (weight-for-weight copy)
    d_in_map = {"fc1": 16, "fc2": 24}
    ci_bpd = TiedCINetwork(d_in_map, c_map, bpd_cfg.ci)
    ci_vpd = vpd_math.CITransformer(d_in_map, c_map, vpd_cfg)
    with torch.no_grad():
        ci_vpd.proj_in.weight.copy_(ci_bpd.proj_in.weight)
        ci_vpd.proj_in.bias.copy_(ci_bpd.proj_in.bias)
        ci_vpd.proj_out.weight.copy_(ci_bpd.out_proj.weight)
        ci_vpd.proj_out.bias.copy_(ci_bpd.out_proj.bias)
        for blk_b, blk_v in zip(ci_bpd.blocks, ci_vpd.blocks):
            q_w, k_w, v_w = blk_b.qkv.weight.chunk(3, dim=0)
            blk_v.attn.q_proj.weight.copy_(q_w)
            blk_v.attn.k_proj.weight.copy_(k_w)
            blk_v.attn.v_proj.weight.copy_(v_w)
            blk_v.attn.o_proj.weight.copy_(blk_b.o_proj.weight)
            blk_v.mlp[0].weight.copy_(blk_b.fc1.weight)
            blk_v.mlp[0].bias.copy_(blk_b.fc1.bias)
            blk_v.mlp[2].weight.copy_(blk_b.fc2.weight)
            blk_v.mlp[2].bias.copy_(blk_b.fc2.bias)

    x = torch.randn(bsz, seq_len, d_in)
    with bpd_model.target_capture(pass_offset=0):
        clean_bpd = net_bpd(x)
    bpd_inputs = bpd_model.collect_recorded_inputs(expected_passes=1)
    clean_vpd = net_vpd(x)
    assert torch.allclose(clean_bpd, clean_vpd, atol=1e-6)

    vpd_acts = {n: vpd_wrappers[n].last_input for n in c_map}
    ci_low_v, ci_up_v, pre_v = ci_vpd(vpd_acts)
    ci_out_b = ci_bpd(bpd_inputs)
    for name in c_map:
        assert torch.allclose(ci_out_b.pre_sigmoid[name][:, 0], pre_v[name], atol=1e-5)
        assert torch.allclose(ci_out_b.lower_leaky[name][:, 0], ci_low_v[name], atol=1e-5)
        assert torch.allclose(ci_out_b.upper_leaky[name][:, 0], ci_up_v[name], atol=1e-5)

    # Term 2: Importance minimality loss parity
    step_i = 250
    p_val = vpd_math.anneal_p(step_i, vpd_cfg.n_steps, vpd_cfg.p_start, vpd_cfg.p_end)
    min_vpd = vpd_cfg.coeff_imp * vpd_math.importance_minimality_loss(
        ci_up_v, p_val, vpd_cfg.imp_eps, vpd_cfg.imp_beta, world_size=1
    )
    min_bpd, min_stats = bpd_minimality_loss(
        ci_out_b.upper_leaky, step=step_i, total_steps=vpd_cfg.n_steps, cfg=bpd_cfg.minimality
    )
    assert min_stats["switch_weighted"] == 0.0
    assert torch.allclose(min_bpd, min_vpd, rtol=1e-5, atol=1e-6)

    # Term 3 & Term 4: Masked forward + KL parity under shared stochastic/adversarial masks
    masks_v, dmasks_v = vpd_math.sample_continuous_masks(ci_low_v)
    routing_v = vpd_math.sample_uniform_k_subset_routing(list(c_map), (bsz, seq_len), x.device)
    vpd_math.set_wrapper_masks(vpd_wrappers, masks_v, dmasks_v, routing_v)
    try:
        pred_stoch_v = net_vpd(x)
    finally:
        vpd_math.clear_wrapper_masks(vpd_wrappers)
    kl_stoch_v = vpd_math.kl_logits(pred_stoch_v, clean_vpd)

    sched_stoch_b = {
        n: PassMaskSchedule(
            comp_mask=masks_v[n].unsqueeze(1),
            delta_mask=dmasks_v[n].unsqueeze(1),
            routing_mask=routing_v[n].unsqueeze(1),
        )
        for n in c_map
    }
    with bpd_model.masked_components(sched_stoch_b):
        pred_stoch_b = net_bpd(x)
    kl_stoch_b, _ = trajectory_recon_loss(
        RolloutTrajectory(readouts={0: pred_stoch_b}, boundary_states={}, module_inputs={}),
        RolloutTrajectory(readouts={0: clean_bpd}, boundary_states={}, module_inputs=bpd_inputs),
    )
    assert torch.allclose(pred_stoch_b, pred_stoch_v, atol=1e-6)
    assert torch.allclose(kl_stoch_b, kl_stoch_v, atol=1e-6)

    # Term 4: Adversarial source formula mask = ci + (1 - ci) * source parity
    ppgd_b = PassSharedPersistentPGD(c_map, bpd_cfg.adversary)
    sched_adv_b, raw_src_b = ppgd_b.build_schedules(ci_out_b.lower_leaky)
    adv_masks_v = {
        n: ci_low_v[n] + (1.0 - ci_low_v[n]) * raw_src_b[n][:, 0, :, : c_map[n]]
        for n in c_map
    }
    adv_dmasks_v = {n: raw_src_b[n][:, 0, :, c_map[n]] for n in c_map}
    vpd_math.set_wrapper_masks(vpd_wrappers, adv_masks_v, adv_dmasks_v, None)
    try:
        pred_adv_v = net_vpd(x)
    finally:
        vpd_math.clear_wrapper_masks(vpd_wrappers)
    with bpd_model.masked_components(sched_adv_b):
        pred_adv_b = net_bpd(x)
    assert torch.allclose(
        trajectory_recon_loss(
            RolloutTrajectory(readouts={0: pred_adv_b}, boundary_states={}, module_inputs={}),
            RolloutTrajectory(readouts={0: clean_bpd}, boundary_states={}, module_inputs=bpd_inputs),
        )[0],
        vpd_math.kl_logits(pred_adv_v, clean_vpd),
        atol=1e-6,
    )


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


def test_stationary_mechanism_not_cheaper_when_split_into_half_trajectories() -> None:
    """Part 1.4: With f_c computed via max over tau, splitting an all-pass stationary
    mechanism into two half-trajectory copies is strictly MORE expensive even at coeff_switch=0.
    """
    cfg_no_sw = MinimalityConfig(pool_mode="mean", coeff_level=1e-3, coeff_switch=0.0)
    bsz, n_passes, seq_len = 4, 8, 6

    stationary = torch.zeros(bsz, n_passes, seq_len, 2)
    stationary[:, :, :, 0] = 1.0  # Component 0 active on all 8 passes; Component 1 inactive

    half_split = torch.zeros(bsz, n_passes, seq_len, 2)
    half_split[:, : n_passes // 2, :, 0] = 1.0  # Copy 0 active on first half [0..3]
    half_split[:, n_passes // 2 :, :, 1] = 1.0  # Copy 1 active on second half [4..7]

    loss_stat, st_stat = bpd_minimality_loss(
        {"m": stationary}, step=60, total_steps=100, cfg=cfg_no_sw
    )
    loss_half, st_half = bpd_minimality_loss(
        {"m": half_split}, step=60, total_steps=100, cfg=cfg_no_sw
    )
    assert st_stat["switch_weighted"] == 0.0
    assert st_half["switch_weighted"] == 0.0
    assert float(loss_half.item()) > float(loss_stat.item())


def test_fragility_hinge_pass_loss_shares_and_inner_state_switch() -> None:
    """Part 1.2 & Part 1.3: Verify match_inner_boundary_states switch and fragility hinge."""
    torch.manual_seed(31)
    clean_ro3 = torch.randn(4, 5, 10)
    pred_ro3 = clean_ro3 + 0.8 * torch.randn(4, 5, 10)
    clean_bs0 = torch.randn(4, 5, 16)
    pred_bs0 = clean_bs0 + 0.9 * torch.randn(4, 5, 16)

    clean_traj = RolloutTrajectory(
        readouts={3: clean_ro3},
        boundary_states={0: clean_bs0, 3: clean_bs0},
        module_inputs={},
    )
    pred_traj = RolloutTrajectory(
        readouts={3: pred_ro3},
        boundary_states={0: pred_bs0, 3: pred_bs0},
        module_inputs={},
    )

    # 1. Default match_inner_boundary_states=False supervises ONLY readout pass 3
    loss_ro_only, per_pass_ro = trajectory_recon_loss(
        pred_traj, clean_traj, match_inner_boundary_states=False
    )
    assert list(per_pass_ro.keys()) == [3]
    assert torch.allclose(loss_ro_only, kl_divergence_logits(pred_ro3, clean_ro3), atol=1e-6)

    # 2. match_inner_boundary_states=True supervises BOTH inner pass 0 and readout pass 3
    loss_both, per_pass_both = trajectory_recon_loss(
        pred_traj, clean_traj, match_inner_boundary_states=True, state_boundary_weight=1.0
    )
    assert sorted(per_pass_both.keys()) == [0, 3]
    expected_p0 = float(boundary_state_loss(pred_bs0, clean_bs0).item())
    assert math.isclose(per_pass_both[0], expected_p0, rel_tol=1e-5, abs_tol=1e-6)

    # 3. Fragility hinge: loss_tau = max(0, raw_tau - kappa * null_tau) + pass loss shares
    raw0 = per_pass_both[0]
    raw3 = per_pass_both[3]
    nulls = {0: raw0 * 10.0, 3: raw3 * 0.5}  #kappa=0.2 clamps pass 0 to 0.0 and shrinks pass 3 to 0.9 * raw3
    loss_hinged, hinged_map = trajectory_recon_loss(
        pred_traj,
        clean_traj,
        match_inner_boundary_states=True,
        pass_null_losses=nulls,
        hinge_kappa=0.20,
    )
    assert hinged_map[0] == 0.0
    assert math.isclose(hinged_map[3], raw3 - 0.20 * nulls[3], rel_tol=1e-5, abs_tol=1e-6)
    assert math.isclose(float(loss_hinged.item()), 0.5 * hinged_map[3], rel_tol=1e-5, abs_tol=1e-6)

    shares = compute_pass_loss_shares(hinged_map)
    assert shares[0] == 0.0
    assert math.isclose(shares[3], 1.0, rel_tol=1e-6, abs_tol=1e-6)
