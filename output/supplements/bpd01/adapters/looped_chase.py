"""BPD adapter for ``LoopedChase`` (LOOPSCOPE-01 Q4 pointer-chase ground truth).

Exposes per-loop answer-cursor readouts ``tau = 0..T-1``, loop-boundary states
``h_tau``, and strict head-rowspace visible projections ``V_tau`` at the cursor
position so ``visible_only`` CI tests whether hop scheduling lives inside
``rowspace(head)`` (``lam = 0.1`` and ``lam = 1.0`` seed 3) or inside dark-fiber
``ker(head)`` phase coordinates (``lam = 1.0`` seeds 1 & 2; Gate B3).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from mechinterp.bpd.adapters.trm import compute_head_projector_f64
from mechinterp.loopscope01.scripts.loopscope_toy import LoopedChase, make_batch, model_args_from


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
