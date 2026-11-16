"""BPD adapter for ``Huginn-0125`` and the ``HUG-CAL-02`` dual negative/positive control.

Supports:
  1. ``HuginnBPDAdapter``: wraps a loaded ``tomg-group-umd/huginn-0125`` recurrent
     transformer (prelude -> ``T`` shared ``core_block`` iterations -> coda + ``lm_head``
     readout at every pass), discovering the 4 recurrent core blocks' shared linear
     projections automatically.
  2. ``SettlingCoreLooper``, ``evaluate_hug_cal02_eight_factors``, and
     ``evaluate_hug_cal02_with_positive_arm``: self-contained 4-core-block settling
     model (plus real-model entrypoint) for Gate B2, auditing the 8 top singular
     factors across ``core.{0..3}.down_proj`` alongside a planted rank-1 positive
     control arm while reporting raw and net-of-rotation-null importance side by side.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mechinterp.bpd.adapters.trm import compute_head_projector_f64
from mechinterp.bpd.components import BPDModel, RolloutTrajectory
from mechinterp.bpd.config import AblationConfig, BPDConfig, CIConfig, MinimalityConfig, NullCalibrationConfig
from mechinterp.bpd.losses import kl_divergence_logits
from mechinterp.bpd.nulls import ComponentNullEntry, evaluate_component_null_report
from mechinterp.bpd.trainer import BPDTrainer


def discover_huginn_core_linears(model: nn.Module, c: int = 32) -> dict[str, int]:
    """Discover ``nn.Linear`` layers inside ``core_block`` (or ``core``) of a Huginn model."""
    found: dict[str, int] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and ("core_block" in name or name.startswith("core.")):
            found[name] = c
    if not found:
        raise ValueError("No recurrent core_block nn.Linear modules found on model")
    return found


class HuginnBPDAdapter:
    """Adapter for HuggingFace ``tomg-group-umd/huginn-0125`` recurrent depth models."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        lm_head = getattr(model, "lm_head", None)
        if not isinstance(lm_head, nn.Linear):
            raise TypeError("Expected model.lm_head to be an nn.Linear")
        v_head, p_head, _ = compute_head_projector_f64(lm_head.weight[: min(64, lm_head.out_features)])
        self.v_head = v_head
        self.p_head = p_head
        self.visible_dim = int(v_head.shape[1])
        self.prompt_dim = int(lm_head.in_features)

    def rollout(
        self,
        input_ids: Tensor,
        n_passes: int,
        pass_offset: int = 0,
        init_state: Tensor | None = None,
    ) -> tuple[dict[int, Tensor], dict[int, Tensor], dict[int, Tensor], Tensor]:
        """Run prelude -> ``n_passes`` recurrent steps -> per-pass coda/lm_head readout."""
        transformer = getattr(self.model, "transformer")
        lm_head = getattr(self.model, "lm_head")
        dev = lm_head.weight.device
        toks = input_ids.to(dev)

        x_emb = transformer.wte(toks)
        for blk in transformer.prelude:
            x_emb = blk(x_emb)

        state = init_state.to(dev) if init_state is not None else torch.zeros_like(x_emb)
        adapter = getattr(transformer, "adapter", None)
        ln_f = getattr(transformer, "ln_f", nn.Identity())
        coda = getattr(transformer, "coda", ())
        vh = self.v_head.to(device=dev, dtype=torch.float32)

        readouts: dict[int, Tensor] = {}
        boundary_states: dict[int, Tensor] = {}
        visible_states: dict[int, Tensor] = {}

        for local_k in range(n_passes):
            tau = pass_offset + local_k
            visible_states[tau] = state.float() @ vh
            if adapter is not None:
                step_in = adapter(torch.cat([state, x_emb], dim=-1))
            else:
                step_in = state + x_emb
            h = step_in
            for blk in transformer.core_block:
                h = blk(h)
            state = h
            coda_h = state
            for blk in coda:
                coda_h = blk(coda_h)
            logits = lm_head(ln_f(coda_h))
            readouts[tau] = logits
            boundary_states[tau] = state

        return readouts, boundary_states, visible_states, x_emb.float()


class _SettlingBlock(nn.Module):
    """Recurrent Pre-RMSNorm block with ``o_proj`` and ``down_proj`` matching Huginn's 4-block core."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_up_proj = nn.Linear(d_model, 4 * d_model, bias=False)
        self.down_proj = nn.Linear(2 * d_model, d_model, bias=False)

    @staticmethod
    def _rms(x: Tensor, eps: float = 1e-6) -> Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)

    def forward(self, h: Tensor) -> Tensor:
        b, p, d = h.shape
        xn = self._rms(h)
        q, k, v = (
            self.qkv_proj(xn)
            .view(b, p, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, p, d)
        h = h + self.o_proj(attn)
        g, u = self.gate_up_proj(self._rms(h)).chunk(2, dim=-1)
        h = h + self.down_proj(F.silu(g) * u)
        return h


class SettlingCoreLooper(nn.Module):
    """4-block recurrent settling loop reproducing ``HUG-CAL-02`` fragility physics.

    Each recurrent pass injects prompt embeddings ``e`` into latent state ``s`` via
    ``adapter([s, e])`` and runs ``core.{0..3}``. Early passes experience high
    non-normal amplification while later passes contract toward an equilibrium,
    creating steep cross-pass fragility without requiring factor-specific semantics
    on the isotropic bulk weights.
    """

    def __init__(
        self,
        vocab_size: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_core_blocks: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.wte = nn.Embedding(vocab_size, d_model)
        self.adapter = nn.Linear(2 * d_model, d_model, bias=False)
        self.core = nn.ModuleList([_SettlingBlock(d_model, n_heads) for _ in range(n_core_blocks)])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        with torch.no_grad():
            self.wte.weight.normal_(0.0, 0.5, generator=gen)
            # Contractive state carry + prompt re-injection -> classic settling dynamics
            eye = torch.eye(d_model)
            self.adapter.weight[:, :d_model].copy_(0.55 * eye)
            self.adapter.weight[:, d_model:].copy_(0.45 * eye)
            for blk in self.core:
                for lin in (blk.qkv_proj, blk.o_proj, blk.gate_up_proj, blk.down_proj):
                    # Isotropic random weights -> raw sensitivity without privileged read alignment
                    lin.weight.normal_(0.0, 0.85 / (lin.weight.shape[1] ** 0.5), generator=gen)
            self.lm_head.weight.normal_(0.0, 1.5 / (d_model ** 0.5), generator=gen)

    def rollout(
        self,
        tokens: Tensor,
        n_passes: int,
        pass_offset: int = 0,
        init_state: Tensor | None = None,
    ) -> tuple[dict[int, Tensor], dict[int, Tensor], dict[int, Tensor], Tensor]:
        dev = self.lm_head.weight.device
        tok = tokens.to(dev)
        x_emb = self.wte(tok)
        state = init_state.to(dev) if init_state is not None else torch.zeros_like(x_emb)

        readouts: dict[int, Tensor] = {}
        boundary_states: dict[int, Tensor] = {}
        visible_states: dict[int, Tensor] = {}

        for local_k in range(n_passes):
            tau = pass_offset + local_k
            visible_states[tau] = self.lm_head( _SettlingBlock._rms(state) ).float()
            h = self.adapter(torch.cat([state, x_emb], dim=-1))
            for blk in self.core:
                h = blk(h)
            state = h
            logits = self.lm_head(_SettlingBlock._rms(state))
            readouts[tau] = logits
            boundary_states[tau] = state

        return readouts, boundary_states, visible_states, x_emb.float()


@dataclass
class HugCal02AuditResult:
    """Gate B2 negative-control summary over the 8 ``HUG-CAL-02`` settling factors."""

    factor_entries: list[ComponentNullEntry]
    max_raw_importance: float
    mean_raw_importance: float
    mean_abs_net_importance: float
    max_abs_z_rot: float
    passed_negative_control: bool
    min_p_rank: float = 1.0
    min_p_rank_two_sided: float = 1.0
    bonferroni_alpha: float = 0.00625


def evaluate_hug_cal02_eight_factors(
    model: SettlingCoreLooper,
    tokens: Tensor,
    *,
    n_passes: int = 6,
    n_rotations: int = 16,
    z_threshold: float = 2.5,
    fwer_alpha: float = 0.05,
    include_o_proj: bool = False,
    seed: int = 0,
) -> HugCal02AuditResult:
    """Audit the 8 leading core-block factors across ``core.{0..3}.{o_proj, down_proj}``.

    When ``include_o_proj=True`` (default, restoring the initial ``e6baa744`` scope),
    evaluates leading factor ``c = 0`` on both ``o_proj`` and ``down_proj`` across all 4
    recurrent core blocks (``8`` factors total). For ``n_rotations >= 50`` (e.g. ``K = 199``),
    gates strictly on exact Monte Carlo rank p-values ``min_p_rank > fwer_alpha / 8``.
    """
    work_model = copy.deepcopy(model)
    if include_o_proj:
        modules_map = {}
        for i in range(4):
            modules_map[f"core.{i}.o_proj"] = 4
            modules_map[f"core.{i}.down_proj"] = 4
        comp_sel = {m: [0] for m in modules_map}
    else:
        modules_map = {f"core.{i}.down_proj": 4 for i in range(4)}
        comp_sel = {m: [0, 1] for m in modules_map}

    bpd_model = BPDModel(work_model, modules_map, max_passes=n_passes, init_svd=True)

    with torch.no_grad():
        with bpd_model.target_capture(pass_offset=0, update_means=True, detach_inputs=True):
            ro, bs, vis, prm = work_model.rollout(tokens, n_passes, 0, None)
        mod_in = bpd_model.collect_recorded_inputs(expected_passes=n_passes)
    clean_traj = RolloutTrajectory(
        readouts=ro,
        boundary_states=bs,
        module_inputs=mod_in,
        visible_states=vis,
        prompt_features=prm,
    )

    def _run() -> RolloutTrajectory:
        p_ro, p_bs, p_vis, p_prm = work_model.rollout(tokens, n_passes, 0, None)
        return RolloutTrajectory(
            readouts=p_ro,
            boundary_states=p_bs,
            module_inputs=mod_in,
            visible_states=p_vis,
            prompt_features=p_prm,
        )

    report = evaluate_component_null_report(
        bpd_model,
        clean_traj,
        _run,
        components_by_module=comp_sel,
        passes=[],  # evaluate multi-pass tied factor ablation (pass_idx = -1)
        include_all_pass=True,
        n_rotations=n_rotations,
        seed=seed,
    )
    entries = report.entries
    raws = [e.raw_importance for e in entries]
    abs_nets = [abs(e.net_importance) for e in entries]
    abs_zs = [abs(e.z_rot) for e in entries]
    p_ranks = [e.p_rank for e in entries]
    p_twos = [e.p_rank_two_sided for e in entries]

    max_raw = max(raws) if raws else 0.0
    mean_raw = sum(raws) / max(len(raws), 1)
    mean_abs_net = sum(abs_nets) / max(len(abs_nets), 1)
    max_z = max(abs_zs) if abs_zs else 0.0
    min_p = min(p_ranks) if p_ranks else 1.0
    min_p2 = min(p_twos) if p_twos else 1.0
    bonf = float(fwer_alpha) / float(max(len(entries), 1))

    if n_rotations >= 50:
        passed_neg = bool(max_raw > 0.0 and min_p > bonf)
    else:
        passed_neg = bool(max_raw > 0.0 and max_z <= z_threshold)

    return HugCal02AuditResult(
        factor_entries=entries,
        max_raw_importance=max_raw,
        mean_raw_importance=mean_raw,
        mean_abs_net_importance=mean_abs_net,
        max_abs_z_rot=max_z,
        passed_negative_control=passed_neg,
        min_p_rank=min_p,
        min_p_rank_two_sided=min_p2,
        bonferroni_alpha=bonf,
    )


@dataclass
class HugCal02PositiveArmResult:
    """Gate B2 planted rank-1 positive-control arm evaluated beside the 8-factor negative control."""

    target_module: str
    component: int
    sigma: float
    raw_importance: float
    rotation_null_mean: float
    rotation_null_std: float
    net_importance: float
    observed_z_rot: float
    registered_z_threshold: float
    margin: float
    passed_positive_control: bool


@dataclass
class HugCal02DualAuditResult:
    """Combined Gate B2 result: 8-factor settling negative control + planted positive arm."""

    negative_arm: HugCal02AuditResult
    negative_margin: float
    positive_arm: HugCal02PositiveArmResult
    passed_gate_b2: bool


@torch.no_grad()
def evaluate_hug_cal02_with_positive_arm(
    model: SettlingCoreLooper,
    tokens: Tensor,
    *,
    n_passes: int = 6,
    n_rotations: int = 16,
    z_threshold: float = 2.5,
    cal_seed: int = 777,
    cal_batch_size: int = 128,
    cancel_fraction: float = 1.25,
    target_token: int = 0,
    seed: int = 0,
) -> HugCal02DualAuditResult:
    """Run the 8-factor settling negative control alongside a planted rank-1 positive arm.

    The positive arm clones ``model``, calibrates a rank-1 read vector ``v_plant`` on an
    independent token batch (``cal_seed``) to fire on ``tok == target_token`` across passes
    paired with an equilibrium-reversing write ``u_plant``, adds ``u_plant v_plant^T`` to
    ``core.{last}.down_proj.weight``, decomposes via ``BPDModel(..., init_svd=True)``, and
    evaluates factor ``0`` out-of-sample on ``tokens`` under the identical ``n_rotations``
    RMS-matched Haar rotation null.
    """
    clean_base = copy.deepcopy(model)
    neg_res = evaluate_hug_cal02_eight_factors(
        copy.deepcopy(clean_base),
        tokens,
        n_passes=n_passes,
        n_rotations=n_rotations,
        z_threshold=z_threshold,
        seed=seed,
    )
    neg_margin = float(z_threshold - neg_res.max_abs_z_rot)

    dev = clean_base.lm_head.weight.device
    seq_len = int(tokens.shape[1])
    gen_cal = torch.Generator().manual_seed(cal_seed)
    cal_tok = torch.randint(
        0, clean_base.vocab_size, (cal_batch_size, seq_len), generator=gen_cal
    ).to(dev)

    last_idx = len(clean_base.core) - 1
    mod_name = f"core.{last_idx}.down_proj"
    down_ins: list[Tensor] = []
    hook = clean_base.core[last_idx].down_proj.register_forward_hook(
        lambda _m, inp, _out: down_ins.append(inp[0].detach())
    )
    try:
        _, bs_cal, _, _ = clean_base.rollout(cal_tok, n_passes, 0, None)
    finally:
        hook.remove()

    mask_tok = cal_tok == target_token
    s_final = bs_cal[n_passes - 1]
    s0_mean = s_final[mask_tok].float().mean(dim=0)

    x_all = torch.stack(down_ins, dim=0)  # [T, B_cal, P, d_in]
    d_in = x_all.shape[-1]
    y_ind = mask_tok.unsqueeze(0).expand(n_passes, -1, -1).to(dtype=torch.float64)
    x_flat = x_all.reshape(-1, d_in).double()
    reg = 1e-2 * torch.eye(d_in, device=dev, dtype=torch.float64)
    v64 = torch.linalg.solve(x_flat.T @ x_flat + reg, x_flat.T @ y_ind.reshape(-1))
    v_unit = (v64 / v64.norm().clamp_min(1e-12)).float()

    alpha_target = float((down_ins[-1][mask_tok].float() @ v_unit).mean().item())
    u_vec = (-float(cancel_fraction) * s0_mean / max(abs(alpha_target), 1e-6)).float()
    if alpha_target < 0.0:
        u_vec = -u_vec

    planted_model = copy.deepcopy(clean_base).to(dev).eval()
    planted_model.core[last_idx].down_proj.weight.add_(
        torch.outer(u_vec, v_unit).to(planted_model.lm_head.weight.dtype)
    )

    bpd_pos = BPDModel(planted_model, {mod_name: 4}, max_passes=n_passes, init_svd=True)
    tok_eval = tokens.to(dev)
    with bpd_pos.target_capture(pass_offset=0, update_means=True, detach_inputs=True):
        ro, bs, vis, prm = planted_model.rollout(tok_eval, n_passes, 0, None)
    mod_in = bpd_pos.collect_recorded_inputs(expected_passes=n_passes)
    clean_traj = RolloutTrajectory(
        readouts=ro,
        boundary_states=bs,
        module_inputs=mod_in,
        visible_states=vis,
        prompt_features=prm,
    )

    def _run_pos() -> RolloutTrajectory:
        p_ro, p_bs, p_vis, p_prm = planted_model.rollout(tok_eval, n_passes, 0, None)
        return RolloutTrajectory(
            readouts=p_ro,
            boundary_states=p_bs,
            module_inputs=mod_in,
            visible_states=p_vis,
            prompt_features=p_prm,
        )

    pos_rep = evaluate_component_null_report(
        bpd_pos,
        clean_traj,
        _run_pos,
        components_by_module={mod_name: [0]},
        passes=[],
        include_all_pass=True,
        n_rotations=n_rotations,
        seed=seed,
    )
    e0 = pos_rep.entries[0]
    pos_margin = float(e0.z_rot - z_threshold)
    pos_arm = HugCal02PositiveArmResult(
        target_module=mod_name,
        component=0,
        sigma=float(u_vec.norm().item()),
        raw_importance=float(e0.raw_importance),
        rotation_null_mean=float(e0.rotation_null_mean),
        rotation_null_std=float(e0.rotation_null_std),
        net_importance=float(e0.net_importance),
        observed_z_rot=float(e0.z_rot),
        registered_z_threshold=float(z_threshold),
        margin=pos_margin,
        passed_positive_control=bool(e0.net_importance > 0.0 and e0.z_rot >= z_threshold),
    )

    return HugCal02DualAuditResult(
        negative_arm=neg_res,
        negative_margin=neg_margin,
        positive_arm=pos_arm,
        passed_gate_b2=bool(neg_res.passed_negative_control and pos_arm.passed_positive_control),
    )


def _calibrate_sigma_for_target_raw_kl(
    clean_base: SettlingCoreLooper,
    cal_tok: Tensor,
    *,
    n_passes: int,
    blk_idx: int,
    u_unit: Tensor,
    v_unit: Tensor,
    target_raw_kl: float,
) -> tuple[float, float]:
    """Binary-search ``sigma > 0`` on ``cal_tok`` so planted raw final KL matches ``target_raw_kl``."""
    dev = clean_base.lm_head.weight.device
    with torch.no_grad():
        ro_clean, _, _, _ = clean_base.rollout(cal_tok, n_passes, 0, None)
        clean_final = ro_clean[n_passes - 1]

        def _kl_at(sig: float) -> float:
            probe = copy.deepcopy(clean_base).to(dev).eval()
            probe.core[blk_idx].down_proj.weight.add_(
                torch.outer(float(sig) * u_unit, v_unit).to(probe.lm_head.weight.dtype)
            )
            ro_p, _, _, _ = probe.rollout(cal_tok, n_passes, 0, None)
            return float(kl_divergence_logits(ro_p[n_passes - 1], clean_final).item())

        lo = 0.0
        hi = 1.0
        while _kl_at(hi) < target_raw_kl and hi < 1e4:
            hi *= 2.0
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            if _kl_at(mid) < target_raw_kl:
                lo = mid
            else:
                hi = mid
        sigma_star = 0.5 * (lo + hi)
        achieved = _kl_at(sigma_star)
    return sigma_star, achieved


def evaluate_hug_cal02_sigma_sweep(
    model: SettlingCoreLooper,
    eval_tokens: Tensor,
    *,
    n_passes: int = 6,
    n_rotations: int = 199,
    fwer_alpha: float = 0.05,
    pos_alpha: float = 0.01,
    effect_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    bpd_steps: int = 80,
    cal_seed: int = 777,
    cal_batch_size: int = 128,
    target_token: int = 0,
    seed: int = 11,
) -> dict[str, Any]:
    """Amendment v2 Gate B2: 8-factor ``o_proj + down_proj`` negative control + natural-scale ``sigma`` sweep.

    1. Audits the 8 leading ``core.{0..3}.{o_proj, down_proj}`` factors under ``n_rotations``
       RMS-matched Haar read rotations using exact rank p-values (``min_p_rank > fwer_alpha / 8``).
    2. Calibrates planted rank-1 amplitudes ``sigma`` on disjoint ``cal_tok`` so induced raw KL
       matches ``m * max_natural_raw_cal`` for ``m in effect_multipliers`` (down to ``0.5x`` and
       ``1.0x`` of the largest natural settling factor).
    3. Reports out-of-sample ``n_rotations`` exact rank p-values on ``eval_tokens`` for both:
       - ``instrument_power``: known planted rank-1 component under the ``HUG-CAL-02`` rotation null,
       - ``blind_bpd_recovery``: full-rank BPD trained strictly on ``cal_seed`` batches, single
         component ``c*`` selected on ``cal_tok`` by quick net calibration shift, audited once
         out-of-sample on ``eval_tokens`` under ``n_rotations`` Haar rotations.
    """
    dev = model.lm_head.weight.device
    clean_base = copy.deepcopy(model).to(dev).eval()
    eval_tok = eval_tokens.to(dev)

    neg_eval = evaluate_hug_cal02_eight_factors(
        copy.deepcopy(clean_base),
        eval_tok,
        n_passes=n_passes,
        n_rotations=n_rotations,
        fwer_alpha=fwer_alpha,
        include_o_proj=True,
        seed=seed,
    )

    seq_len = int(eval_tok.shape[1])
    gen_cal = torch.Generator().manual_seed(cal_seed)
    cal_tok = torch.randint(
        0, clean_base.vocab_size, (cal_batch_size, seq_len), generator=gen_cal
    ).to(dev)

    # Measure max natural factor raw effect on cal_tok (zero peeking at eval_tok)
    neg_cal = evaluate_hug_cal02_eight_factors(
        copy.deepcopy(clean_base),
        cal_tok[: int(eval_tok.shape[0])],
        n_passes=n_passes,
        n_rotations=4,
        include_o_proj=True,
        seed=cal_seed,
    )
    ref_natural_raw_cal = max(neg_cal.max_raw_importance, 1e-4)

    last_idx = len(clean_base.core) - 1
    mod_name = f"core.{last_idx}.down_proj"
    down_ins: list[Tensor] = []
    hook = clean_base.core[last_idx].down_proj.register_forward_hook(
        lambda _m, inp, _out: down_ins.append(inp[0].detach())
    )
    with torch.no_grad():
        try:
            _, bs_cal, _, _ = clean_base.rollout(cal_tok, n_passes, 0, None)
        finally:
            hook.remove()

    mask_tok = cal_tok == target_token
    s_final = bs_cal[n_passes - 1]
    s0_mean = s_final[mask_tok].float().mean(dim=0)
    x_all = torch.stack(down_ins, dim=0)
    d_in = x_all.shape[-1]
    y_ind = mask_tok.unsqueeze(0).expand(n_passes, -1, -1).to(dtype=torch.float64)
    x_flat = x_all.reshape(-1, d_in).double()
    reg = 1e-2 * torch.eye(d_in, device=dev, dtype=torch.float64)
    v64 = torch.linalg.solve(x_flat.T @ x_flat + reg, x_flat.T @ y_ind.reshape(-1))
    v_unit = (v64 / v64.norm().clamp_min(1e-12)).float()

    alpha_target = float((down_ins[-1][mask_tok].float() @ v_unit).mean().item())
    raw_u = -s0_mean * (1.0 if alpha_target >= 0.0 else -1.0)
    u_unit = F.normalize(raw_u.float(), dim=0)

    sweep_rows: list[dict[str, Any]] = []
    for mult_idx, mult in enumerate(effect_multipliers):
        target_kl = float(mult) * ref_natural_raw_cal
        sigma_val, cal_kl = _calibrate_sigma_for_target_raw_kl(
            clean_base,
            cal_tok[: int(eval_tok.shape[0])],
            n_passes=n_passes,
            blk_idx=last_idx,
            u_unit=u_unit,
            v_unit=v_unit,
            target_raw_kl=target_kl,
        )
        u_vec = float(sigma_val) * u_unit

        # A. Null-instrument sensitivity on exact planted rank-1 factor (1 + rank(W) exact representation)
        inst_model = copy.deepcopy(clean_base).to(dev).eval()
        inst_model.core[last_idx].down_proj.weight.data.add_(
            torch.outer(u_vec, v_unit).to(inst_model.lm_head.weight.dtype)
        )
        c_full = int(clean_base.d_model)
        bpd_inst = BPDModel(inst_model, {mod_name: c_full + 1}, max_passes=n_passes, init_svd=False)
        with torch.no_grad():
            # Decompose W_planted = u_vec v_unit^T + W_clean (component 0 is planted, 1..d are exact clean SVD)
            w_clean_64 = clean_base.core[last_idx].down_proj.weight.detach().double().cpu()
            u_s, s_s, vh_s = torch.linalg.svd(w_clean_64, full_matrices=False)
            k_s = min(c_full, int(s_s.numel()))
            sq = s_s[:k_s].sqrt()
            v_mat = torch.zeros((d_in, c_full + 1), dtype=torch.float32)
            u_mat = torch.zeros((c_full + 1, clean_base.d_model), dtype=torch.float32)
            v_mat[:, 0] = v_unit.cpu()
            u_mat[0, :] = u_vec.cpu()
            v_mat[:, 1 : 1 + k_s] = (vh_s[:k_s].T * sq.unsqueeze(0)).float()
            u_mat[1 : 1 + k_s, :] = (sq.unsqueeze(1) * u_s[:, :k_s].T).float()
            bpd_inst.wrappers[mod_name].V.copy_(v_mat.to(dev))
            bpd_inst.wrappers[mod_name].U.copy_(u_mat.to(dev))

            with bpd_inst.target_capture(pass_offset=0, update_means=True, detach_inputs=True):
                ro_i, bs_i, vis_i, prm_i = inst_model.rollout(eval_tok, n_passes, 0, None)
            mod_in_i = bpd_inst.collect_recorded_inputs(expected_passes=n_passes)
            clean_traj_i = RolloutTrajectory(
                readouts=ro_i,
                boundary_states=bs_i,
                module_inputs=mod_in_i,
                visible_states=vis_i,
                prompt_features=prm_i,
            )

            def _run_inst() -> RolloutTrajectory:
                r2, b2, v2, p2 = inst_model.rollout(eval_tok, n_passes, 0, None)
                return RolloutTrajectory(
                    readouts=r2,
                    boundary_states=b2,
                    module_inputs=mod_in_i,
                    visible_states=v2,
                    prompt_features=p2,
                )

            inst_rep = evaluate_component_null_report(
                bpd_inst,
                clean_traj_i,
                _run_inst,
                components_by_module={mod_name: [0]},
                passes=[],
                include_all_pass=True,
                n_rotations=n_rotations,
                seed=seed + 100 * mult_idx,
            )
            e_inst = inst_rep.entries[0]

        # B. Blind BPD recovery trained strictly on cal_seed stream (no access to u_vec, v_unit, or eval_tok)
        blind_model = copy.deepcopy(clean_base).to(dev).eval()
        blind_model.core[last_idx].down_proj.weight.data.add_(
            torch.outer(u_vec, v_unit).to(blind_model.lm_head.weight.dtype)
        )
        c_blind = int(clean_base.d_model)
        cfg_b = BPDConfig(
            c_per_module={mod_name: c_blind},
            n_passes=n_passes,
            n_steps=bpd_steps,
            batch_size=int(eval_tok.shape[0]),
            seq_len=seq_len,
            seed=cal_seed + mult_idx,
            faithfulness_warmup_steps=10,
            main_lr=4e-3,
            coeff_faith=1e4,
            coeff_det=0.5,
            coeff_stoch=0.5,
            coeff_adv=0.5,
            rollout_mode="closed_loop",
            ci=CIConfig(variant="state_only", d_model=48, n_blocks=1, n_heads=4, mlp_hidden=96),
            ablation=AblationConfig(mode="pass_mean", sphere_rescale=True),
            minimality=MinimalityConfig(pool_mode="mean", coeff_level=5e-4, coeff_switch=2e-4),
            null_calibration=NullCalibrationConfig(use_fragility_hinge=False, hinge_kappa=0.0),
        )
        bpd_blind = BPDModel(
            blind_model, {mod_name: c_blind}, max_passes=n_passes, ablation=cfg_b.ablation, init_svd=True
        )
        tr_blind = BPDTrainer(bpd_blind, cfg_b, blind_model.rollout, device=dev)
        tr_blind.run_faithfulness_warmup()
        cal_sel_tok = cal_tok[: int(eval_tok.shape[0])]
        tr_blind.calibrate_baselines(cal_sel_tok)
        gen_tr = torch.Generator().manual_seed(cal_seed + 5000 + mult_idx)
        for st in range(bpd_steps):
            bt = torch.randint(
                0, clean_base.vocab_size, (int(eval_tok.shape[0]), seq_len), generator=gen_tr
            ).to(dev)
            tr_blind.train_step(bt, st)

        # Select single best candidate component c* strictly on cal_sel_tok (zero peeking at eval_tok)
        with torch.no_grad():
            with bpd_blind.target_capture(pass_offset=0, update_means=True, detach_inputs=True):
                ro_c, bs_c, vis_c, prm_c = blind_model.rollout(cal_sel_tok, n_passes, 0, None)
            mod_in_c = bpd_blind.collect_recorded_inputs(expected_passes=n_passes)
            clean_c = RolloutTrajectory(
                readouts=ro_c, boundary_states=bs_c, module_inputs=mod_in_c, visible_states=vis_c, prompt_features=prm_c
            )

            def _run_cal() -> RolloutTrajectory:
                rc2, bc2, vc2, pc2 = blind_model.rollout(cal_sel_tok, n_passes, 0, None)
                return RolloutTrajectory(
                    readouts=rc2, boundary_states=bc2, module_inputs=mod_in_c, visible_states=vc2, prompt_features=pc2
                )

            cal_screen = evaluate_component_null_report(
                bpd_blind,
                clean_c,
                _run_cal,
                components_by_module={mod_name: list(range(c_blind))},
                passes=[],
                include_all_pass=True,
                n_rotations=6,
                seed=cal_seed + 900 + mult_idx,
            )
            best_cal_entry = max(cal_screen.entries, key=lambda e: e.z_rot)
            c_star = int(best_cal_entry.component)

            # Audit ONLY pre-selected c* out-of-sample on eval_tok with n_rotations
            with bpd_blind.target_capture(pass_offset=0, update_means=True, detach_inputs=True):
                ro_e, bs_e, vis_e, prm_e = blind_model.rollout(eval_tok, n_passes, 0, None)
            mod_in_e = bpd_blind.collect_recorded_inputs(expected_passes=n_passes)
            clean_e = RolloutTrajectory(
                readouts=ro_e, boundary_states=bs_e, module_inputs=mod_in_e, visible_states=vis_e, prompt_features=prm_e
            )

            def _run_eval_b() -> RolloutTrajectory:
                re2, be2, ve2, pe2 = blind_model.rollout(eval_tok, n_passes, 0, None)
                return RolloutTrajectory(
                    readouts=re2, boundary_states=be2, module_inputs=mod_in_e, visible_states=ve2, prompt_features=pe2
                )

            eval_b_rep = evaluate_component_null_report(
                bpd_blind,
                clean_e,
                _run_eval_b,
                components_by_module={mod_name: [c_star]},
                passes=[],
                include_all_pass=True,
                n_rotations=n_rotations,
                seed=seed + 300 + 100 * mult_idx,
            )
            e_blind = eval_b_rep.entries[0]
            v_cos = float(
                F.cosine_similarity(
                    bpd_blind.wrappers[mod_name].V[:, c_star].detach().float(),
                    v_unit,
                    dim=0,
                ).abs().item()
            )

        sweep_rows.append(
            {
                "effect_multiplier_vs_max_natural": float(mult),
                "calibrated_sigma": sigma_val,
                "calibration_raw_kl": cal_kl,
                "instrument_power": {
                    "raw_importance": e_inst.raw_importance,
                    "rotation_null_mean": e_inst.rotation_null_mean,
                    "rotation_null_std": e_inst.rotation_null_std,
                    "net_importance": e_inst.net_importance,
                    "z_rot": e_inst.z_rot,
                    "exceed_count": e_inst.exceed_count,
                    "n_rotations": e_inst.n_rotations,
                    "p_rank": e_inst.p_rank,
                    "detected_at_alpha": bool(e_inst.net_importance > 0.0 and e_inst.p_rank <= pos_alpha),
                },
                "blind_bpd_recovery": {
                    "selected_component_on_cal": c_star,
                    "read_alignment_abs_cos": v_cos,
                    "raw_importance": e_blind.raw_importance,
                    "rotation_null_mean": e_blind.rotation_null_mean,
                    "rotation_null_std": e_blind.rotation_null_std,
                    "net_importance": e_blind.net_importance,
                    "z_rot": e_blind.z_rot,
                    "exceed_count": e_blind.exceed_count,
                    "n_rotations": e_blind.n_rotations,
                    "p_rank": e_blind.p_rank,
                    "detected_at_alpha": bool(e_blind.net_importance > 0.0 and e_blind.p_rank <= pos_alpha),
                },
            }
        )

    # Registered v2 pass rule: negative arm passes Bonferroni FWER=0.05 across all 8 o_proj+down_proj
    # factors AND positive sweep detects at >= 1.0x natural scale (1.0x, 2.0x, 4.0x) at p_rank <= pos_alpha!
    pos_ge_1x_inst = [
        r["instrument_power"]["detected_at_alpha"]
        for r in sweep_rows
        if float(r["effect_multiplier_vs_max_natural"]) >= 1.0
    ]
    pos_ge_1x_blind = [
        r["blind_bpd_recovery"]["detected_at_alpha"]
        for r in sweep_rows
        if float(r["effect_multiplier_vs_max_natural"]) >= 1.0
    ]
    passed_v2 = bool(
        neg_eval.passed_negative_control
        and all(pos_ge_1x_inst)
        and all(pos_ge_1x_blind)
    )

    return {
        "negative_arm": {
            "scope": "core.{0..3}.{o_proj, down_proj} leading factor c=0 (8 factors)",
            "n_rotations": n_rotations,
            "fwer_alpha": fwer_alpha,
            "bonferroni_alpha": neg_eval.bonferroni_alpha,
            "max_raw_importance": neg_eval.max_raw_importance,
            "mean_raw_importance": neg_eval.mean_raw_importance,
            "mean_abs_net_importance": neg_eval.mean_abs_net_importance,
            "max_abs_z_rot": neg_eval.max_abs_z_rot,
            "min_p_rank": neg_eval.min_p_rank,
            "min_p_rank_two_sided": neg_eval.min_p_rank_two_sided,
            "passed_negative_control": neg_eval.passed_negative_control,
            "factors": [
                {
                    "module": e.module,
                    "component": e.component,
                    "raw_importance": e.raw_importance,
                    "rotation_null_mean": e.rotation_null_mean,
                    "rotation_null_std": e.rotation_null_std,
                    "net_importance": e.net_importance,
                    "z_rot": e.z_rot,
                    "exceed_count": e.exceed_count,
                    "n_rotations": e.n_rotations,
                    "p_rank": e.p_rank,
                    "p_rank_two_sided": e.p_rank_two_sided,
                }
                for e in neg_eval.factor_entries
            ],
        },
        "positive_sigma_sweep": {
            "reference_max_natural_raw_cal": ref_natural_raw_cal,
            "pos_alpha": pos_alpha,
            "n_rotations": n_rotations,
            "bpd_steps": bpd_steps,
            "rows": sweep_rows,
            "all_ge_1x_instrument_detected": all(pos_ge_1x_inst),
            "all_ge_1x_blind_bpd_detected": all(pos_ge_1x_blind),
        },
        "gate_b2_passed": passed_v2,
    }
