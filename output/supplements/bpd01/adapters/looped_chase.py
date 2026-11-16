"""BPD adapter for ``LoopedChase`` (LOOPSCOPE-01 Q4 pointer-chase ground truth).

Exposes per-loop answer-cursor readouts ``tau = 0..T-1``, loop-boundary states
``h_tau``, and strict head-rowspace visible projections ``V_tau`` at the cursor
position so ``visible_only`` CI tests whether hop scheduling lives inside
``rowspace(head)`` (``lam = 0.1`` and ``lam = 1.0`` seed 3) or inside dark-fiber
``ker(head)`` phase coordinates (``lam = 1.0`` seeds 1 & 2; Gate B3).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F

from mechinterp.bpd.adapters.trm import compute_head_projector_f64
from mechinterp.bpd.components import BPDModel, RolloutTrajectory
from mechinterp.bpd.config import AblationConfig, BPDConfig, CIConfig, MinimalityConfig, NullCalibrationConfig
from mechinterp.bpd.losses import kl_divergence_logits
from mechinterp.bpd.nulls import evaluate_component_null_report
from mechinterp.bpd.trainer import BPDTrainer
from mechinterp.loopscope01.scripts.loopscope_toy import LoopedChase, make_batch, model_args_from

DEFAULT_LOOPED_CHASE_CKPT_DIR = Path(
    r"C:\Users\patri\Documents\Codex\LoopedTransformers-rmp"
    r"\experiments\recurrence_monitorability_v0\loopscope_toy\ckpt"
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_looped_chase_checkpoint(
    name_or_path: str | Path,
    *,
    ckpt_dir: str | Path | None = None,
) -> tuple[Path, str]:
    """Resolve a ``LoopedChase`` ``.final.pt`` path (or stem like ``lam0.1_s1``) and its SHA-256."""
    cand = Path(name_or_path)
    if not cand.exists():
        base = Path(ckpt_dir) if ckpt_dir is not None else DEFAULT_LOOPED_CHASE_CKPT_DIR
        stem = str(name_or_path)
        if not stem.endswith(".final.pt"):
            stem = f"{stem}.final.pt"
        cand = base / stem
    if not cand.exists():
        raise FileNotFoundError(f"LoopedChase checkpoint not found: {cand}")
    return cand, _sha256_file(cand)


def default_looped_chase_c_per_module(model: LoopedChase, c: int = 24) -> dict[str, int]:
    """Shared linear projections inside ``LoopedChase.core``."""
    out: dict[str, int] = {}
    for idx in range(len(model.core)):
        out[f"core.{idx}.qkv"] = c
        out[f"core.{idx}.proj"] = c
        out[f"core.{idx}.fc1"] = c
        out[f"core.{idx}.fc2"] = c
    return out


def load_looped_chase_checkpoint(path: str | Path, *, device: str = "cpu") -> LoopedChase:
    """Load a trained ``LoopedChase`` ``.final.pt`` checkpoint."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    args = model_args_from(payload["cfg"])
    model = LoopedChase(args.n_nodes, args.d, args.layers, args.heads).to(device)
    model.load_state_dict(payload["model"])
    return model.eval()


@dataclass
class LoopedChaseBatch:
    """Pointer-chase token batch with optional cached pre-loop states ``h_{tau}``."""

    tokens: Tensor  # [B, seq_len] long
    hops: Tensor | None = None  # [B, max_depth] long
    clean_states: list[Tensor] | None = None


class LoopedChaseBPDAdapter:
    """Connects ``LoopedChase`` to ``BPDTrainer`` and ``train_ci_heads_on_frozen_components``."""

    def __init__(self, model: LoopedChase) -> None:
        self.model = model
        eff_head = (
            model.head.weight.detach().float()
            * model.ln_f.weight.detach().float().unsqueeze(0)
        )
        v_head, p_head, p_perp = compute_head_projector_f64(eff_head)
        self.v_head = v_head
        self.p_head = p_head
        self.p_perp = p_perp
        self.visible_dim = int(v_head.shape[1])
        self.prompt_dim = int(model.head.in_features)

    def _embed_prompt(self, tokens: Tensor) -> Tensor:
        n = self.model.n_nodes
        edges = tokens[:, : 2 * n].reshape(tokens.shape[0], n, 2)
        h = torch.cat(
            [self.model.tok(edges[..., 0]) + self.model.dst(edges[..., 1]), self.model.tok(tokens[:, 2 * n :])],
            dim=1,
        )
        return h + self.model.pos(torch.arange(h.shape[1], device=tokens.device))

    def _visible_cursor(self, h: Tensor) -> Tensor:
        """Project all token states ``h`` onto ``V_head`` -> ``[B, P, r]`` (strictly blind to ``ker(head)``)."""
        vh = self.v_head.to(device=h.device, dtype=torch.float32)
        return h.float() @ vh

    @torch.no_grad()
    def make_batch_with_clean_cache(
        self,
        tokens: Tensor,
        n_loops: int,
        *,
        hops: Tensor | None = None,
    ) -> LoopedChaseBatch:
        dev = self.model.head.weight.device
        tok = tokens.to(dev)
        e = self._embed_prompt(tok)
        h = torch.zeros_like(e)
        states: list[Tensor] = []
        for _ in range(n_loops):
            states.append(h.detach().clone())
            h = self.model.inject(h) + e
            for blk in self.model.core:
                h = blk(h)
        return LoopedChaseBatch(
            tokens=tok,
            hops=hops.to(dev) if hops is not None else None,
            clean_states=states,
        )

    def sample_clean_window_state(self, batch: LoopedChaseBatch, tau0: int) -> Tensor:
        if batch.clean_states is None:
            raise ValueError("LoopedChaseBatch has no cached clean_states")
        return batch.clean_states[tau0].clone()

    def rollout(
        self,
        batch: LoopedChaseBatch | Tensor,
        n_passes: int,
        pass_offset: int = 0,
        init_state: Tensor | None = None,
    ) -> tuple[dict[int, Tensor], dict[int, Tensor], dict[int, Tensor], Tensor]:
        dev = self.model.head.weight.device
        tok = batch.tokens.to(dev) if isinstance(batch, LoopedChaseBatch) else batch.to(dev)
        e = self._embed_prompt(tok)
        h = init_state.to(dev) if init_state is not None else torch.zeros_like(e)

        readouts: dict[int, Tensor] = {}
        boundary_states: dict[int, Tensor] = {}
        visible_states: dict[int, Tensor] = {}

        for local_k in range(n_passes):
            tau = pass_offset + local_k
            visible_states[tau] = self._visible_cursor(h)
            h = self.model.inject(h) + e
            for blk in self.model.core:
                h = blk(h)
            readouts[tau] = self.model.readout(h[:, -1:, :])  # [B, 1, n_nodes]
            boundary_states[tau] = h

        return readouts, boundary_states, visible_states, e.float()


def sample_pointer_chase_batch(
    *,
    n_nodes: int = 16,
    batch_size: int = 32,
    max_depth: int = 8,
    fixed_depth_prompt: int | None = None,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Generate a pointer-chase batch via ``loopscope_toy.make_batch``."""
    depth_t = (
        torch.full((batch_size,), fixed_depth_prompt, dtype=torch.long)
        if fixed_depth_prompt is not None
        else None
    )
    return make_batch(n_nodes, batch_size, seed, max_depth, depth=depth_t)


@dataclass
class PlantedRank1ControlResult:
    """Positive-control audit receipt for a planted rank-1 component on ``LoopedChase``."""

    checkpoint_label: str
    checkpoint_sha256: str
    target_module: str
    component: int
    sigma: float
    registered_z_threshold: float
    raw_importance: float
    rotation_null_mean: float
    rotation_null_std: float
    net_importance: float
    observed_z_rot: float
    margin: float
    single_pass_z_rot: float
    passed_positive_control: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_label": self.checkpoint_label,
            "checkpoint_sha256": self.checkpoint_sha256,
            "target_module": self.target_module,
            "component": self.component,
            "sigma": self.sigma,
            "registered_z_threshold": self.registered_z_threshold,
            "raw_importance": self.raw_importance,
            "rotation_null_mean": self.rotation_null_mean,
            "rotation_null_std": self.rotation_null_std,
            "net_importance": self.net_importance,
            "observed_z_rot": self.observed_z_rot,
            "margin": self.margin,
            "single_pass_z_rot": self.single_pass_z_rot,
            "passed_positive_control": self.passed_positive_control,
        }


@torch.no_grad()
def evaluate_planted_rank1_looped_chase(
    model: LoopedChase,
    *,
    n_passes: int = 8,
    cal_batch_size: int = 256,
    eval_batch_size: int = 64,
    cal_seed: int = 777,
    eval_seed: int = 9000,
    null_seed: int = 42,
    n_rotations: int = 16,
    sigma: float = 600.0,
    z_threshold: float = 2.5,
    target_module: str | None = None,
    checkpoint_label: str = "lam0.1_s1",
    checkpoint_sha256: str = "",
) -> PlantedRank1ControlResult:
    """Plant a rank-1 component ``u_plant v_plant^T`` on ``LoopedChase`` and audit recovery margin.

    Fits ``v_plant`` by ridge regression on calibration seed ``cal_seed`` to fire selectively
    at the answer cursor ``p = -1`` when ``start_node == 0``, pairs it with head-rowspace
    direction ``u_plant = normalize(W_head[0] - W_head[1])`` scaled by ``sigma``, adds the
    pure rank-1 perturbation ``W_m <- W_m + u_plant v_plant^T`` to a cloned checkpoint,
    decomposes ``W_m`` blindly via ``BPDModel(..., init_svd=True)``, and evaluates factor ``0``
    out-of-sample on ``eval_seed`` against ``n_rotations`` RMS-matched Haar read rotations.
    """
    dev = model.head.weight.device
    last_blk_idx = len(model.core) - 1
    mod_name = target_module if target_module is not None else f"core.{last_blk_idx}.fc2"
    if mod_name not in dict(model.named_modules()):
        mod_name = f"core.{last_blk_idx}.fc2"
    blk_idx = int(mod_name.split(".")[1])

    cal_adapter = LoopedChaseBPDAdapter(model)
    cal_tok, _ = sample_pointer_chase_batch(
        n_nodes=model.n_nodes, batch_size=cal_batch_size, max_depth=n_passes, seed=cal_seed
    )
    cal_tok = cal_tok.to(dev)

    fc2_ins: list[Tensor] = []
    hook = model.core[blk_idx].fc2.register_forward_hook(
        lambda _m, inp, _out: fc2_ins.append(inp[0].detach())
    )
    try:
        cal_adapter.rollout(cal_tok, n_passes, 0, None)
    finally:
        hook.remove()

    start_node = cal_tok[:, 2 * model.n_nodes]
    x_all = torch.stack(fc2_ins, dim=0)  # [T, B_cal, P, d_in]
    d_in = x_all.shape[-1]
    y_ind = torch.zeros(x_all.shape[:3], device=dev, dtype=torch.float64)
    y_ind[:, start_node == 0, -1] = 1.0

    x_flat = x_all.reshape(-1, d_in).double()
    y_flat = y_ind.reshape(-1)
    reg = 1e-2 * torch.eye(d_in, device=dev, dtype=torch.float64)
    v64 = torch.linalg.solve(x_flat.T @ x_flat + reg, x_flat.T @ y_flat)
    v_unit = (v64 / v64.norm().clamp_min(1e-12)).float()

    head_diff = (model.head.weight[0] - model.head.weight[1]).detach().float()
    u_unit = F.normalize(head_diff, dim=0)
    u_vec = float(sigma) * u_unit

    planted_model = copy.deepcopy(model).to(dev).eval()
    planted_model.core[blk_idx].fc2.weight.add_(torch.outer(u_vec, v_unit).to(planted_model.head.weight.dtype))

    bpd_planted = BPDModel(planted_model, {mod_name: 8}, max_passes=n_passes, init_svd=True)
    eval_adapter = LoopedChaseBPDAdapter(planted_model)
    eval_tok, _ = sample_pointer_chase_batch(
        n_nodes=planted_model.n_nodes,
        batch_size=eval_batch_size,
        max_depth=n_passes,
        seed=eval_seed,
    )
    eval_tok = eval_tok.to(dev)

    with bpd_planted.target_capture(pass_offset=0, update_means=True, detach_inputs=True):
        ro, bs, vis, prm = eval_adapter.rollout(eval_tok, n_passes, 0, None)
    mod_in = bpd_planted.collect_recorded_inputs(expected_passes=n_passes)
    clean_traj = RolloutTrajectory(
        readouts=ro,
        boundary_states=bs,
        module_inputs=mod_in,
        visible_states=vis,
        prompt_features=prm,
    )

    def _run_planted() -> RolloutTrajectory:
        r2, b2, v2, p2 = eval_adapter.rollout(eval_tok, n_passes, 0, None)
        return RolloutTrajectory(
            readouts=r2,
            boundary_states=b2,
            module_inputs=mod_in,
            visible_states=v2,
            prompt_features=p2,
        )

    rep = evaluate_component_null_report(
        bpd_planted,
        clean_traj,
        _run_planted,
        components_by_module={mod_name: [0]},
        passes=[n_passes - 1],
        include_all_pass=True,
        n_rotations=n_rotations,
        seed=null_seed,
    )
    by_pass = {e.pass_idx: e for e in rep.entries}
    all_pass_entry = by_pass[-1]
    single_pass_entry = by_pass[n_passes - 1]
    margin = float(all_pass_entry.z_rot - z_threshold)

    return PlantedRank1ControlResult(
        checkpoint_label=checkpoint_label,
        checkpoint_sha256=checkpoint_sha256,
        target_module=mod_name,
        component=0,
        sigma=float(sigma),
        registered_z_threshold=float(z_threshold),
        raw_importance=float(all_pass_entry.raw_importance),
        rotation_null_mean=float(all_pass_entry.rotation_null_mean),
        rotation_null_std=float(all_pass_entry.rotation_null_std),
        net_importance=float(all_pass_entry.net_importance),
        observed_z_rot=float(all_pass_entry.z_rot),
        margin=margin,
        single_pass_z_rot=float(single_pass_entry.z_rot),
        passed_positive_control=bool(
            all_pass_entry.net_importance > 0.0 and all_pass_entry.z_rot >= z_threshold
        ),
    )


def evaluate_looped_chase_sigma_sweep(
    model: LoopedChase,
    *,
    n_passes: int = 8,
    n_rotations: int = 199,
    pos_alpha: float = 0.01,
    effect_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    bpd_steps: int = 100,
    cal_batch_size: int = 256,
    eval_batch_size: int = 64,
    cal_seed: int = 777,
    eval_seed: int = 9000,
    null_seed: int = 42,
    checkpoint_label: str = "lam0.1_s1",
    checkpoint_sha256: str = "",
) -> dict[str, object]:
    """Amendment v2 Gate B3 positive control: natural-scale ``sigma`` sweep at ``K = 199`` rotations.

    Measures the largest natural leading SVD factor's raw KL shift ``R_max`` across
    ``core.*.{proj, fc2}`` on ``cal_seed``, calibrates planted rank-1 amplitudes ``sigma``
    for ``m * R_max`` across ``effect_multipliers``, and evaluates both null-instrument
    sensitivity and blind BPD recovery out-of-sample on ``eval_seed`` via exact rank p-values.
    """
    dev = model.head.weight.device
    clean_base = copy.deepcopy(model).to(dev).eval()
    last_blk_idx = len(clean_base.core) - 1
    mod_name = f"core.{last_blk_idx}.fc2"

    cal_tok, _ = sample_pointer_chase_batch(
        n_nodes=clean_base.n_nodes, batch_size=cal_batch_size, max_depth=n_passes, seed=cal_seed
    )
    cal_tok = cal_tok.to(dev)
    cal_sel_tok = cal_tok[:eval_batch_size]

    eval_tok, _ = sample_pointer_chase_batch(
        n_nodes=clean_base.n_nodes, batch_size=eval_batch_size, max_depth=n_passes, seed=eval_seed
    )
    eval_tok = eval_tok.to(dev)

    # 1. Measure max natural leading factor raw effect on cal_sel_tok across output projections
    nat_mods = {}
    for bi in range(len(clean_base.core)):
        nat_mods[f"core.{bi}.proj"] = 4
        nat_mods[f"core.{bi}.fc2"] = 4
    nat_probe_model = copy.deepcopy(clean_base).to(dev).eval()
    bpd_nat = BPDModel(nat_probe_model, nat_mods, max_passes=n_passes, init_svd=True)
    nat_adapter = LoopedChaseBPDAdapter(nat_probe_model)
    with torch.no_grad():
        with bpd_nat.target_capture(pass_offset=0, update_means=True, detach_inputs=True):
            ro_n, bs_n, vis_n, prm_n = nat_adapter.rollout(cal_sel_tok, n_passes, 0, None)
        mod_in_n = bpd_nat.collect_recorded_inputs(expected_passes=n_passes)
        clean_n = RolloutTrajectory(
            readouts=ro_n, boundary_states=bs_n, module_inputs=mod_in_n, visible_states=vis_n, prompt_features=prm_n
        )

        def _run_nat() -> RolloutTrajectory:
            rn2, bn2, vn2, pn2 = nat_adapter.rollout(cal_sel_tok, n_passes, 0, None)
            return RolloutTrajectory(
                readouts=rn2, boundary_states=bn2, module_inputs=mod_in_n, visible_states=vn2, prompt_features=pn2
            )

        nat_rep = evaluate_component_null_report(
            bpd_nat,
            clean_n,
            _run_nat,
            components_by_module={m: [0] for m in nat_mods},
            passes=[],
            include_all_pass=True,
            n_rotations=4,
            seed=cal_seed,
        )
    ref_natural_raw_cal = max(max((e.raw_importance for e in nat_rep.entries), default=0.0), 1e-3)

    # 2. Fit selective read vector v_unit on cal_tok (start_node == 0 at cursor p = -1)
    cal_base_adapter = LoopedChaseBPDAdapter(clean_base)
    fc2_ins: list[Tensor] = []
    hook = clean_base.core[last_blk_idx].fc2.register_forward_hook(
        lambda _m, inp, _out: fc2_ins.append(inp[0].detach())
    )
    with torch.no_grad():
        try:
            ro_base_cal, _, _, _ = cal_base_adapter.rollout(cal_tok, n_passes, 0, None)
        finally:
            hook.remove()
        clean_cal_sel_final = ro_base_cal[n_passes - 1][:eval_batch_size]

    start_node = cal_tok[:, 2 * clean_base.n_nodes]
    x_all = torch.stack(fc2_ins, dim=0)
    d_in = x_all.shape[-1]
    d_out = int(clean_base.head.in_features)
    y_ind = torch.zeros(x_all.shape[:3], device=dev, dtype=torch.float64)
    y_ind[:, start_node == 0, -1] = 1.0
    x_flat = x_all.reshape(-1, d_in).double()
    reg = 1e-2 * torch.eye(d_in, device=dev, dtype=torch.float64)
    v64 = torch.linalg.solve(x_flat.T @ x_flat + reg, x_flat.T @ y_ind.reshape(-1))
    v_unit = (v64 / v64.norm().clamp_min(1e-12)).float()
    head_diff = (clean_base.head.weight[0] - clean_base.head.weight[1]).detach().float()
    u_unit = F.normalize(head_diff, dim=0)

    def _cal_kl_at_sigma(sig: float) -> float:
        with torch.no_grad():
            pm = copy.deepcopy(clean_base).to(dev).eval()
            pm.core[last_blk_idx].fc2.weight.add_(
                torch.outer(float(sig) * u_unit, v_unit).to(pm.head.weight.dtype)
            )
            pa = LoopedChaseBPDAdapter(pm)
            rop, _, _, _ = pa.rollout(cal_sel_tok, n_passes, 0, None)
            return float(kl_divergence_logits(rop[n_passes - 1], clean_cal_sel_final).item())

    rows: list[dict[str, object]] = []
    for m_idx, mult in enumerate(effect_multipliers):
        target_kl = float(mult) * ref_natural_raw_cal
        lo, hi = 0.0, 5.0
        while _cal_kl_at_sigma(hi) < target_kl and hi < 1e5:
            hi *= 2.0
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            if _cal_kl_at_sigma(mid) < target_kl:
                lo = mid
            else:
                hi = mid
        sigma_val = 0.5 * (lo + hi)
        cal_kl = _cal_kl_at_sigma(sigma_val)
        u_vec = float(sigma_val) * u_unit

        # A. Instrument sensitivity on exact planted factor under n_rotations Haar rotations
        inst_model = copy.deepcopy(clean_base).to(dev).eval()
        inst_model.core[last_blk_idx].fc2.weight.data.add_(
            torch.outer(u_vec, v_unit).to(inst_model.head.weight.dtype)
        )
        bpd_inst = BPDModel(inst_model, {mod_name: d_out + 1}, max_passes=n_passes, init_svd=False)
        with torch.no_grad():
            w_clean_64 = clean_base.core[last_blk_idx].fc2.weight.detach().double().cpu()
            u_s, s_s, vh_s = torch.linalg.svd(w_clean_64, full_matrices=False)
            k_s = min(d_out, int(s_s.numel()))
            sq = s_s[:k_s].sqrt()
            v_mat = torch.zeros((d_in, d_out + 1), dtype=torch.float32)
            u_mat = torch.zeros((d_out + 1, d_out), dtype=torch.float32)
            v_mat[:, 0] = v_unit.cpu()
            u_mat[0, :] = u_vec.cpu()
            v_mat[:, 1 : 1 + k_s] = (vh_s[:k_s].T * sq.unsqueeze(0)).float()
            u_mat[1 : 1 + k_s, :] = (sq.unsqueeze(1) * u_s[:, :k_s].T).float()
            bpd_inst.wrappers[mod_name].V.copy_(v_mat.to(dev))
            bpd_inst.wrappers[mod_name].U.copy_(u_mat.to(dev))

            ia = LoopedChaseBPDAdapter(inst_model)
            with bpd_inst.target_capture(pass_offset=0, update_means=True, detach_inputs=True):
                ro_i, bs_i, vis_i, prm_i = ia.rollout(eval_tok, n_passes, 0, None)
            mod_in_i = bpd_inst.collect_recorded_inputs(expected_passes=n_passes)
            clean_i = RolloutTrajectory(
                readouts=ro_i, boundary_states=bs_i, module_inputs=mod_in_i, visible_states=vis_i, prompt_features=prm_i
            )

            def _run_inst() -> RolloutTrajectory:
                ri2, bi2, vi2, pi2 = ia.rollout(eval_tok, n_passes, 0, None)
                return RolloutTrajectory(
                    readouts=ri2, boundary_states=bi2, module_inputs=mod_in_i, visible_states=vi2, prompt_features=pi2
                )

            inst_rep = evaluate_component_null_report(
                bpd_inst,
                clean_i,
                _run_inst,
                components_by_module={mod_name: [0]},
                passes=[],
                include_all_pass=True,
                n_rotations=n_rotations,
                seed=null_seed + 100 * m_idx,
            )
            e_inst = inst_rep.entries[0]

        # B. Blind BPD recovery trained strictly on cal_seed stream
        blind_model = copy.deepcopy(clean_base).to(dev).eval()
        blind_model.core[last_blk_idx].fc2.weight.data.add_(
            torch.outer(u_vec, v_unit).to(blind_model.head.weight.dtype)
        )
        ba = LoopedChaseBPDAdapter(blind_model)
        c_blind = d_out
        cfg_b = BPDConfig(
            c_per_module={mod_name: c_blind},
            n_passes=n_passes,
            n_steps=bpd_steps,
            batch_size=eval_batch_size,
            seq_len=clean_base.n_nodes + 1,
            seed=cal_seed + m_idx,
            faithfulness_warmup_steps=10,
            main_lr=3e-3,
            coeff_faith=1e4,
            coeff_det=0.5,
            coeff_stoch=0.5,
            coeff_adv=0.5,
            rollout_mode="closed_loop",
            ci=CIConfig(variant="state_only", d_model=64, n_blocks=1, n_heads=4, mlp_hidden=128),
            ablation=AblationConfig(mode="pass_mean", sphere_rescale=True),
            minimality=MinimalityConfig(pool_mode="mean", coeff_level=3e-4, coeff_switch=2e-4),
            null_calibration=NullCalibrationConfig(use_fragility_hinge=False, hinge_kappa=0.0),
        )
        bpd_blind = BPDModel(
            blind_model, {mod_name: c_blind}, max_passes=n_passes, ablation=cfg_b.ablation, init_svd=True
        )
        tr_blind = BPDTrainer(bpd_blind, cfg_b, ba.rollout, device=dev)
        tr_blind.run_faithfulness_warmup()
        tr_blind.calibrate_baselines(cal_sel_tok)
        for st in range(bpd_steps):
            bt, _ = sample_pointer_chase_batch(
                n_nodes=clean_base.n_nodes,
                batch_size=eval_batch_size,
                max_depth=n_passes,
                seed=cal_seed + 4000 + 100 * m_idx + st,
            )
            tr_blind.train_step(bt.to(dev), st)

        with torch.no_grad():
            with bpd_blind.target_capture(pass_offset=0, update_means=True, detach_inputs=True):
                ro_c, bs_c, vis_c, prm_c = ba.rollout(cal_sel_tok, n_passes, 0, None)
            mod_in_c = bpd_blind.collect_recorded_inputs(expected_passes=n_passes)
            clean_c = RolloutTrajectory(
                readouts=ro_c, boundary_states=bs_c, module_inputs=mod_in_c, visible_states=vis_c, prompt_features=prm_c
            )

            def _run_cal_b() -> RolloutTrajectory:
                rc2, bc2, vc2, pc2 = ba.rollout(cal_sel_tok, n_passes, 0, None)
                return RolloutTrajectory(
                    readouts=rc2, boundary_states=bc2, module_inputs=mod_in_c, visible_states=vc2, prompt_features=pc2
                )

            cal_screen = evaluate_component_null_report(
                bpd_blind,
                clean_c,
                _run_cal_b,
                components_by_module={mod_name: list(range(c_blind))},
                passes=[],
                include_all_pass=True,
                n_rotations=6,
                seed=cal_seed + 900 + m_idx,
            )
            c_star = int(max(cal_screen.entries, key=lambda e: e.z_rot).component)

            with bpd_blind.target_capture(pass_offset=0, update_means=True, detach_inputs=True):
                ro_e, bs_e, vis_e, prm_e = ba.rollout(eval_tok, n_passes, 0, None)
            mod_in_e = bpd_blind.collect_recorded_inputs(expected_passes=n_passes)
            clean_e = RolloutTrajectory(
                readouts=ro_e, boundary_states=bs_e, module_inputs=mod_in_e, visible_states=vis_e, prompt_features=prm_e
            )

            def _run_eval_b() -> RolloutTrajectory:
                re2, be2, ve2, pe2 = ba.rollout(eval_tok, n_passes, 0, None)
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
                seed=null_seed + 500 + 100 * m_idx,
            )
            e_blind = eval_b_rep.entries[0]
            v_cos = float(
                F.cosine_similarity(
                    bpd_blind.wrappers[mod_name].V[:, c_star].detach().float(),
                    v_unit,
                    dim=0,
                ).abs().item()
            )

        rows.append(
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

    ge_1x_inst = [
        bool(r["instrument_power"]["detected_at_alpha"])  # type: ignore[index]
        for r in rows
        if float(r["effect_multiplier_vs_max_natural"]) >= 1.0  # type: ignore[arg-type]
    ]
    ge_1x_blind = [
        bool(r["blind_bpd_recovery"]["detected_at_alpha"])  # type: ignore[index]
        for r in rows
        if float(r["effect_multiplier_vs_max_natural"]) >= 1.0  # type: ignore[arg-type]
    ]
    return {
        "checkpoint_label": checkpoint_label,
        "checkpoint_sha256": checkpoint_sha256,
        "target_module": mod_name,
        "reference_max_natural_raw_cal": ref_natural_raw_cal,
        "n_rotations": n_rotations,
        "pos_alpha": pos_alpha,
        "rows": rows,
        "all_ge_1x_instrument_detected": all(ge_1x_inst),
        "all_ge_1x_blind_bpd_detected": all(ge_1x_blind),
        "passed_positive_control": bool(all(ge_1x_inst) and all(ge_1x_blind)),
    }
