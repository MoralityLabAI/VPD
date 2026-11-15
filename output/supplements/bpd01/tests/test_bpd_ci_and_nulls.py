"""Unit tests for BPD's three pass-tied CI variants (Clock Gap & Visible Gap),
zero-shot OOD pass generalization, and RMS-matched Haar rotation nulls (Gate B2).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mechinterp.bpd.adapters.huginn import SettlingCoreLooper, evaluate_hug_cal02_eight_factors
from mechinterp.bpd.adapters.trm import compute_head_projector_f64
from mechinterp.bpd.ci import TiedCINetwork
from mechinterp.bpd.components import BPDModel, RolloutTrajectory
from mechinterp.bpd.config import (
    AblationConfig,
    BPDConfig,
    CIConfig,
    MinimalityConfig,
)
from mechinterp.bpd.losses import trajectory_recon_loss
from mechinterp.bpd.nulls import (
    calibrate_pass_tolerances,
    evaluate_component_null_report,
    sample_rms_matched_read_rotations,
)
from mechinterp.bpd.trainer import train_ci_heads_on_frozen_components


def test_ci_variants_generalize_to_ood_pass_counts() -> None:
    torch.manual_seed(3)
    mod_dims = {"m1": 16, "m2": 12}
    c_map = {"m1": 4, "m2": 3}

    for variant in ("state_only", "state_plus_clock", "visible_only"):
        cfg = CIConfig(
            variant=variant,  # type: ignore[arg-type]
            d_model=32,
            n_blocks=1,
            n_heads=4,
            mlp_hidden=64,
            max_passes=4,  # Trained clock table size = 4
        )
        net = TiedCINetwork(
            mod_dims,
            c_map,
            cfg,
            visible_dim=6 if variant == "visible_only" else None,
            prompt_dim=10 if variant == "visible_only" else None,
        )

        # Evaluate zero-shot at T = 11 > max_passes = 4
        t_ood = 11
        mod_in = {
            "m1": torch.randn(2, t_ood, 5, 16),
            "m2": torch.randn(2, t_ood, 5, 12),
        }
        vis = {tau: torch.randn(2, 5, 6) for tau in range(t_ood)}
        prm = torch.randn(2, 5, 10)

        out = net(mod_in, visible_states=vis, prompt_features=prm)
        assert out.lower_leaky["m1"].shape == (2, t_ood, 5, 4)
        assert out.lower_leaky["m2"].shape == (2, t_ood, 5, 3)
        loss = out.upper_leaky["m1"].sum() + out.lower_leaky["m2"].sum()
        loss.backward()
        assert net.out_proj.weight.grad is not None
        assert torch.isfinite(net.out_proj.weight.grad).all()


def test_visible_only_ci_strictly_invariant_to_head_nullspace() -> None:
    torch.manual_seed(12)
    d_state = 24
    head = nn.Linear(d_state, 6, bias=False)
    v_head, _, p_perp = compute_head_projector_f64(head.weight)
    vis_dim = v_head.shape[1]

    state_ci = TiedCINetwork(
        {"core": d_state}, {"core": 5}, CIConfig(variant="state_only", d_model=32, n_blocks=1)
    )
    vis_ci = TiedCINetwork(
        {"core": d_state},
        {"core": 5},
        CIConfig(variant="visible_only", d_model=32, n_blocks=1),
        visible_dim=vis_dim,
        prompt_dim=8,
    )

    h_clean = torch.randn(3, 4, 7, d_state)
    prompt = torch.randn(3, 7, 8)
    # Perturb hidden states strictly inside ker(head) (dark fiber R_t)
    dark_noise = (torch.randn(3, 4, 7, d_state) * 5.0) @ p_perp
    h_dark = h_clean + dark_noise

    vis_clean = {tau: h_clean[:, tau] @ v_head for tau in range(4)}
    vis_dark = {tau: h_dark[:, tau] @ v_head for tau in range(4)}

    out_vis_clean = vis_ci({"core": h_clean}, visible_states=vis_clean, prompt_features=prompt)
    out_vis_dark = vis_ci({"core": h_dark}, visible_states=vis_dark, prompt_features=prompt)
    assert torch.allclose(
        out_vis_clean.pre_sigmoid["core"], out_vis_dark.pre_sigmoid["core"], atol=1e-5
    )

    out_state_clean = state_ci({"core": h_clean})
    out_state_dark = state_ci({"core": h_dark})
    diff_state = (out_state_clean.pre_sigmoid["core"] - out_state_dark.pre_sigmoid["core"]).abs().max()
    assert float(diff_state.item()) > 1e-2


def test_visible_gap_detects_dark_fiber_addressed_mechanism() -> None:
    """Construct a 2-component loop where component choice is addressed by a ker(head) coordinate."""
    torch.manual_seed(21)
    d = 8
    # Head reads ONLY dimensions 0..1; dimensions 2..7 lie in ker(head)
    head_w = torch.zeros(4, d)
    head_w[0, 0] = 2.0
    head_w[1, 1] = -2.0
    head_w[2, 0] = -1.5
    head_w[3, 1] = 1.5
    v_head, _, _ = compute_head_projector_f64(head_w)

    class _DarkAddressedLooper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = nn.Linear(d, d, bias=False)
            with torch.no_grad():
                self.core.weight.zero_()
                # Component A: reads dark coord 2, writes to visible coord 0
                self.core.weight[0, 2] = 3.0
                # Component B: reads dark coord 3, writes to visible coord 1
                self.core.weight[1, 3] = 3.0

        def rollout(
            self,
            dark_switch: Tensor,  # [B] in {0, 1}, invisible to prompt & V_0!
            n_passes: int,
            pass_offset: int = 0,
            init_state: Tensor | None = None,
        ):
            bsz = dark_switch.shape[0]
            h = torch.zeros(bsz, 1, d)
            # Encode switch strictly in ker(head) dims 2 vs 3 at pass 0
            h[:, 0, 2] = (dark_switch == 0).float() * 2.0
            h[:, 0, 3] = (dark_switch == 1).float() * 2.0
            prompt = torch.ones(bsz, 1, 2)  # Constant prompt -> gives no clue about dark_switch

            readouts = {}
            boundaries = {}
            vis = {}
            for k in range(n_passes):
                tau = pass_offset + k
                vis[tau] = (h @ v_head).float()
                h = h + self.core(h)
                readouts[tau] = F.linear(h, head_w)
                boundaries[tau] = h
            return readouts, boundaries, vis, prompt

    looper = _DarkAddressedLooper()
    bpd = BPDModel(
        looper,
        {"core": 2},
        max_passes=2,
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        init_svd=True,
    )

    cfg = BPDConfig(
        c_per_module={"core": 2},
        n_passes=1,
        n_steps=90,
        batch_size=32,
        seq_len=1,
        seed=5,
        main_lr=1e-2,
        ci=CIConfig(variant="state_only", d_model=16, n_blocks=1, n_heads=2, mlp_hidden=32),
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        minimality=MinimalityConfig(pool_mode="mean", coeff_level=0.15, coeff_switch=0.0),
    )

    gen = torch.Generator().manual_seed(99)
    eval_sw = torch.randint(0, 2, (64,), generator=gen)

    def _batch_fn(step: int) -> Tensor:
        g = torch.Generator().manual_seed(200 + step)
        return torch.randint(0, 2, (32,), generator=g)

    _, var_metrics, gaps = train_ci_heads_on_frozen_components(
        bpd,
        cfg,
        looper.rollout,
        _batch_fn,
        eval_sw,
        visible_dim=v_head.shape[1],
        prompt_dim=2,
    )
    # state_only sees dims 2 & 3 so it activates ONLY the 1 needed component (~1.0 active) with ~0 KL,
    # whereas visible_only is blind to dims 2 & 3 so it suffers higher recon_loss or must keep both on!
    total_cost_state = var_metrics["state_only"]["recon_loss"] + 0.15 * var_metrics["state_only"]["mean_active_per_pass"]
    total_cost_vis = var_metrics["visible_only"]["recon_loss"] + 0.15 * var_metrics["visible_only"]["mean_active_per_pass"]
    assert total_cost_vis > total_cost_state + 0.03
    assert gaps["visible_gap_recon"] > 0.01 or gaps["visible_gap_active"] > 0.25


def test_rotation_null_cancels_1808x_fragility_and_passes_hug_cal02() -> None:
    torch.manual_seed(17)
    d = 32
    x_pass = torch.randn(20, 4, d) * torch.linspace(0.2, 3.0, d)
    v_vec = torch.randn(d)
    rots = sample_rms_matched_read_rotations(x_pass, v_vec, n_rotations=8)
    target_rms = (x_pass @ v_vec).pow(2).mean().sqrt()
    for k in range(8):
        rk_rms = (x_pass @ rots[:, k]).pow(2).mean().sqrt()
        assert torch.allclose(rk_rms, target_rms, rtol=1e-4, atol=1e-5)

    # 1. Artificial 1,808.4x cross-pass sensitivity swing (matching LOOPSCOPE-02 TRM F_1/F_6)
    amp = math.sqrt(1808.4)

    class _FragileTwoPass(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(d, d, bias=False)
            nn.init.normal_(self.shared.weight, 0.0, 0.2)

        def rollout(self, x: Tensor, n_passes: int = 2):
            # Pass 0 write is amplified by sqrt(1808.4); pass 1 write has unit gain
            y0 = self.shared(x)
            y1 = self.shared(x * 0.9 + 0.1)
            final_state = amp * y0 + 1.0 * y1
            return {}, {0: y0, 1: final_state}

    frag = _FragileTwoPass()
    bpd_frag = BPDModel(
        frag,
        {"shared": 4},
        max_passes=2,
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        init_svd=True,
    )
    x_in = torch.randn(24, 4, d)
    with bpd_frag.target_capture():
        ro, bs = frag.rollout(x_in)
    mod_in = bpd_frag.collect_recorded_inputs(expected_passes=2)
    clean_traj = RolloutTrajectory(readouts=ro, boundary_states=bs, module_inputs=mod_in)

    def _run_frag() -> RolloutTrajectory:
        p_ro, p_bs = frag.rollout(x_in)
        return RolloutTrajectory(readouts=p_ro, boundary_states=p_bs, module_inputs=mod_in)

    tols = calibrate_pass_tolerances(
        bpd_frag, clean_traj, _run_frag, n_rotations=8, eval_at_final_readout=False, seed=3
    )
    # Note: with eval_at_final_readout=False, trajectory average still weights pass 0 strongly;
    # let's also check final-pass sensitivity explicitly by adding a linear readout at pass 1:
    clean_with_ro = RolloutTrajectory(
        readouts={1: bs[1]}, boundary_states={0: bs[0]}, module_inputs=mod_in
    )

    def _run_frag_ro() -> RolloutTrajectory:
        _, p_bs = frag.rollout(x_in)
        return RolloutTrajectory(
            readouts={1: p_bs[1]}, boundary_states={0: p_bs[0]}, module_inputs=mod_in
        )

    tols_ro = calibrate_pass_tolerances(
        bpd_frag, clean_with_ro, _run_frag_ro, n_rotations=8, eval_at_final_readout=True, seed=3
    )
    assert tols_ro[0] / tols_ro[1] > 100.0
    # Verify tolerance-weighted trajectory_recon_loss runs cleanly
    w_loss, _ = trajectory_recon_loss(clean_with_ro, clean_with_ro, pass_tolerances=tols_ro)
    assert float(w_loss.item()) < 1e-6

    # 2. Gate B2: HUG-CAL-02 8-factor negative control -> all 8 isotropic settling factors ~0 net
    settler = SettlingCoreLooper(vocab_size=32, d_model=48, n_heads=4, n_core_blocks=4, seed=19)
    toks = torch.randint(0, 32, (20, 8), generator=torch.Generator().manual_seed(29))
    audit = evaluate_hug_cal02_eight_factors(
        settler, toks, n_passes=5, n_rotations=16, z_threshold=2.8, seed=19
    )
    assert len(audit.factor_entries) == 8
    assert audit.max_raw_importance > 0.0
    assert audit.passed_negative_control is True

    # 3. Positive-control check: downstream-selective alignment (X @ v0 correlated with A @ u0)
    # beats the RMS-matched rotation null (net_importance > 0, z_rot > 2.0), while both raw
    # and net importance are always reported together.
    class _SelectiveTwoPass(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(d, d, bias=False)
            with torch.no_grad():
                self.shared.weight.normal_(0.0, 0.05)
                self.shared.weight[0, 0] = 2.5

        def rollout(self, batch_pair: tuple[Tensor, Tensor], n_passes: int = 2):
            x_tok, active_gate = batch_pair  # x_tok: [B, P, d], active_gate: [B, P] in {0.0, 1.0}
            y0 = self.shared(x_tok)
            y1 = self.shared(x_tok * 0.5)
            # Downstream sparse attention/MLP gate passes u0=[1,0,...] only when active_gate == 1
            gated_logit = active_gate * y0[..., 0] + 0.05 * y1[..., 0]
            logits = torch.stack([gated_logit, -gated_logit], dim=-1)
            return {1: logits}, {0: y0}

    sel_mod = _SelectiveTwoPass()
    bpd_pos = BPDModel(
        sel_mod,
        {"shared": 4},
        max_passes=2,
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        init_svd=True,
    )
    # Coord 0 fires specifically on the 1-in-8 gated positions; coords 1..d-1 are dense background
    x_iso = torch.randn(32, 8, d)
    active_gate = torch.zeros(32, 8)
    active_gate[:, 0] = 1.0  # Only position 0 is routed downstream
    x_iso[:, :, 0] = active_gate * math.sqrt(8.0)  # Exact same pass RMS = 1.0 as coords 1..d-1!

    with bpd_pos.target_capture():
        ro_pos, bs_pos = sel_mod.rollout((x_iso, active_gate))
    in_pos = bpd_pos.collect_recorded_inputs(expected_passes=2)
    clean_pos = RolloutTrajectory(
        readouts=ro_pos, boundary_states=bs_pos, module_inputs=in_pos
    )

    def _run_pos() -> RolloutTrajectory:
        p_ro, p_bs = sel_mod.rollout((x_iso, active_gate))
        return RolloutTrajectory(readouts=p_ro, boundary_states=p_bs, module_inputs=in_pos)

    rep_pos = evaluate_component_null_report(
        bpd_pos,
        clean_pos,
        _run_pos,
        components_by_module={"shared": [0]},
        passes=[0],
        include_all_pass=False,
        n_rotations=12,
        seed=5,
    )
    e0 = rep_pos.entries[0]
    assert e0.raw_importance > e0.rotation_null_mean
    assert e0.net_importance > 0.05
    assert e0.z_rot > 2.0
