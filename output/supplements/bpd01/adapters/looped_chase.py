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
from mechinterp.bpd.nulls import evaluate_component_null_report
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
        v_head, p_head, p_perp = compute_head_projector_f64(model.head.weight)
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
        """Project ``ln_f(h[:, -1:])`` onto ``V_head`` -> ``[B, 1, r]`` (broadcastable across seq)."""
        vh = self.v_head.to(device=h.device, dtype=torch.float32)
        cursor_normed = self.model.ln_f(h[:, -1:, :]).float()
        return cursor_normed @ vh

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
