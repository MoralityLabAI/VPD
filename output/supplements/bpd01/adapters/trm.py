"""BPD adapter for 7M Attention-TRM (``PureSudokuTRM``) on 9x9 Sudoku.

Handles TRM's two-level recurrence (``S`` outer ACT commits x ``h_cycles`` H-cycles
x ``(l_cycles + 1)`` inner ``L_level`` invocations per H-cycle):
  - Inner ``z_L`` updates have **no** behavioral readout -> matched via loop-boundary
    state discrepancy on ``z_L``.
  - H-cycle ``z_H`` commits expose the 81-cell digit readout -> matched via exact
    digit-logit KL at every commit pass.
  - ``visible_only`` CI receives strictly ``(x_emb, V_tau)`` where ``V_tau`` projects
    board cells ``z_H[:, puzzle_emb_len:]`` onto ``rowspace(lm_head)``, masking out
    both ``z_L`` and ``ker(lm_head)`` (exact parity with LOOPSCOPE-03).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from mechinterp.loopscope02.scripts.trm_sudoku import (
    TRM_CHECKPOINTS,
    PureSudokuTRM,
    encode_questions,
)


def compute_head_projector_f64(weight: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return float32 ``(V_head [D, r], P_head [D, D], P_perp [D, D])`` via float64 SVD."""
    w64 = weight.detach().cpu().double()
    _u, s, vh = torch.linalg.svd(w64, full_matrices=False)
    rank = int((s > 1e-8 * s[0]).sum().item())
    v64 = vh[:rank].T.contiguous()
    p64 = v64 @ v64.T
    p_perp64 = torch.eye(w64.shape[1], dtype=torch.float64) - p64
    return v64.float(), p64.float(), p_perp64.float()


def default_trm_c_per_module(model: PureSudokuTRM, c: int = 32, *, include_in_projs: bool = False) -> dict[str, int]:
    """Standard shared linear sites inside ``PureSudokuTRM.layers``."""
    out: dict[str, int] = {}
    for idx in range(len(model.layers)):
        out[f"layers.{idx}.o_proj"] = c
        out[f"layers.{idx}.down_proj"] = c
        if include_in_projs:
            out[f"layers.{idx}.qkv_proj"] = c
            out[f"layers.{idx}.gate_up_proj"] = c
    return out


def build_mini_trm(
    *,
    hidden_size: int = 64,
    num_heads: int = 4,
    l_layers: int = 2,
    h_cycles: int = 1,
    l_cycles: int = 2,
    puzzle_emb_len: int = 4,
    seq_len: int = 16,
    seed: int = 0,
) -> PureSudokuTRM:
    """Build a compact deterministic ``PureSudokuTRM`` for fast CPU tests and smoke runs."""
    gen = torch.Generator().manual_seed(seed)
    m = PureSudokuTRM(
        hidden_size=hidden_size,
        num_heads=num_heads,
        l_layers=l_layers,
        h_cycles=h_cycles,
        l_cycles=l_cycles,
        puzzle_emb_len=puzzle_emb_len,
        seq_len=seq_len,
    )
    with torch.no_grad():
        m.H_init.normal_(0.0, 0.2, generator=gen)
        m.L_init.normal_(0.0, 0.2, generator=gen)
        m.embed_tokens.weight.normal_(0.0, 0.05, generator=gen)
        m.puzzle_weight.normal_(0.0, 0.05, generator=gen)
        m.lm_head.weight.normal_(0.0, 0.1, generator=gen)
        for blk in m.layers:
            for lin in (blk.qkv_proj, blk.o_proj, blk.gate_up_proj, blk.down_proj):
                lin.weight.normal_(0.0, 1.0 / (lin.weight.shape[1] ** 0.5), generator=gen)
    return m.eval()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pretrained_trm(
    key_or_path: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
    verify_sha256: bool = True,
) -> PureSudokuTRM:
    """Load ``gaoxin492`` or ``yagizdevre`` pretrained 7M TRM (with SHA-256 verification)."""
    s_key = str(key_or_path)
    if s_key in TRM_CHECKPOINTS:
        meta = TRM_CHECKPOINTS[s_key]
        from huggingface_hub import hf_hub_download

        ckpt_path = Path(hf_hub_download(repo_id=meta["repo_id"], filename=meta["filename"]))
        if verify_sha256:
            digest = _sha256_file(ckpt_path)
            if digest != meta["sha256"]:
                raise RuntimeError(
                    f"SHA-256 mismatch for {s_key}: got {digest}, expected {meta['sha256']}"
                )
    else:
        ckpt_path = Path(key_or_path)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    return PureSudokuTRM.from_state_dict(sd, dtype=dtype)


@dataclass
class TRMBatch:
    """Tokenized Sudoku batch with optional cached clean states ``(z_h, z_l)`` per invocation."""

    inputs: Tensor  # [B, seq_len] long
    clean_states: list[tuple[Tensor, Tensor]] | None = None


class TRMBPDAdapter:
    """Connects ``PureSudokuTRM`` to ``BPDTrainer`` (closed-loop + clean-transplant windows)."""

    def __init__(self, model: PureSudokuTRM) -> None:
        self.model = model
        # Compute head rowspace projector strictly over digit vocabulary columns 2..10
        digit_w = model.lm_head.weight[2:11, :]  # [9, hidden_size]
        v_head, p_head, p_perp = compute_head_projector_f64(digit_w)
        self.v_head = v_head
        self.p_head = p_head
        self.p_perp = p_perp
        self.visible_dim = int(v_head.shape[1])
        self.prompt_dim = int(model.hidden_size)

    @property
    def calls_per_h_cycle(self) -> int:
        return int(self.model.l_cycles + 1)

    @property
    def calls_per_outer_step(self) -> int:
        return int(self.model.h_cycles * (self.model.l_cycles + 1))

    def invocations_for_outer_steps(self, outer_steps: int) -> int:
        return int(outer_steps * self.calls_per_outer_step)

    def _visible_projection(self, z_h: Tensor) -> Tensor:
        """Project board cells ``z_h[:, puz:]`` onto ``V_head``; zero puzzle prefix ``0:puz``."""
        vh = self.v_head.to(device=z_h.device, dtype=torch.float32)
        puz = self.model.puzzle_emb_len
        board_vis = z_h[:, puz:, :].float() @ vh  # [B, seq_len, r]
        if puz == 0:
            return board_vis
        prefix = torch.zeros(
            (z_h.shape[0], puz, self.visible_dim), device=z_h.device, dtype=torch.float32
        )
        return torch.cat([prefix, board_vis], dim=1)  # [B, total_len, r]

    @torch.no_grad()
    def make_batch_with_clean_cache(self, inputs: Tensor, total_invocations: int) -> TRMBatch:
        """Run clean target once to cache ``(z_h, z_l)`` prior to every invocation ``0..T-1``."""
        dev = self.model.embed_tokens.weight.device
        inp = inputs.to(dev)
        x_emb = self.model.encode_inputs(inp)
        z_h, z_l = self.model.init_states(inp.shape[0])
        cyc = self.calls_per_h_cycle

        states: list[tuple[Tensor, Tensor]] = []
        for tau in range(total_invocations):
            states.append((z_h.detach().clone(), z_l.detach().clone()))
            if (tau % cyc) < self.model.l_cycles:
                z_l = self.model.run_l_level(z_l, z_h + x_emb)
            else:
                z_h = self.model.run_l_level(z_h, z_l)
        return TRMBatch(inputs=inp, clean_states=states)

    def sample_clean_window_state(self, batch: TRMBatch, tau0: int) -> tuple[Tensor, Tensor]:
        if batch.clean_states is None:
            raise ValueError("TRMBatch has no cached clean_states for clean_window sampling")
        zh, zl = batch.clean_states[tau0]
        return zh.clone(), zl.clone()

    def rollout(
        self,
        batch: TRMBatch | Tensor,
        n_passes: int,
        pass_offset: int = 0,
        init_state: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[dict[int, Tensor], dict[int, Tensor], dict[int, Tensor], Tensor]:
        """Execute ``n_passes`` ``run_l_level`` invocations starting from ``pass_offset``."""
        dev = self.model.embed_tokens.weight.device
        inp = batch.inputs.to(dev) if isinstance(batch, TRMBatch) else batch.to(dev)
        x_emb = self.model.encode_inputs(inp)

        if init_state is not None:
            z_h, z_l = init_state[0].to(dev), init_state[1].to(dev)
        else:
            z_h, z_l = self.model.init_states(inp.shape[0])

        cyc = self.calls_per_h_cycle
        readouts: dict[int, Tensor] = {}
        boundary_states: dict[int, Tensor] = {}
        visible_states: dict[int, Tensor] = {}

        for local_k in range(n_passes):
            tau = pass_offset + local_k
            visible_states[tau] = self._visible_projection(z_h)
            if (tau % cyc) < self.model.l_cycles:
                z_l = self.model.run_l_level(z_l, z_h + x_emb)
                boundary_states[tau] = z_l
            else:
                z_h = self.model.run_l_level(z_h, z_l)
                readouts[tau] = self.model.digit_logits(z_h)
                boundary_states[tau] = z_h

        return readouts, boundary_states, visible_states, x_emb.float()


@dataclass
class SudokuHorizonMetrics:
    """Exact board accuracy, cell accuracy, and visible oscillation rate across outer steps."""

    board_exact_by_step: dict[int, float]
    cell_acc_by_step: dict[int, float]
    period2_oscillation_rate: float
    step_flip_rate_tail: float


@torch.no_grad()
def evaluate_sudoku_horizon(
    model: PureSudokuTRM,
    inputs: Tensor,
    targets: Tensor,
    *,
    eval_outer_steps: Sequence[int] = (16, 32, 64),
    batch_size: int = 64,
) -> SudokuHorizonMetrics:
    """Evaluate exact Sudoku solve rate and OOD limit-cycle oscillation up to ``max(eval_outer_steps)``.

    ``inputs`` has shape ``[N, seq_len]`` (token IDs), ``targets`` has shape ``[N, seq_len]``
    with ground-truth digit indices ``0..8`` (corresponding to digits ``1..9``).
    """
    dev = model.embed_tokens.weight.device
    max_s = max(eval_outer_steps)
    milestones = set(eval_outer_steps)
    n_total = inputs.shape[0]

    exact_sums = {s: 0 for s in eval_outer_steps}
    cell_sums = {s: 0.0 for s in eval_outer_steps}
    osc_boards = 0
    tail_flips = 0.0

    for start in range(0, n_total, batch_size):
        inp = inputs[start : start + batch_size].to(dev)
        tgt = targets[start : start + batch_size].to(dev)
        bsz = inp.shape[0]
        x_emb = model.encode_inputs(inp)
        z_h, z_l = model.init_states(bsz)

        preds_tail: list[Tensor] = []  # last 3 steps to detect period-2 oscillation
        for s in range(1, max_s + 1):
            z_h, z_l = model.act_step(z_h, z_l, x_emb)
            if s in milestones or s >= max_s - 2:
                pred = model.digit_logits(z_h).argmax(dim=-1)  # [B, seq_len] in 0..8
                if s in milestones:
                    match = pred == tgt
                    exact_sums[s] += int(match.all(dim=-1).sum().item())
                    cell_sums[s] += float(match.float().mean(dim=-1).sum().item())
                if s >= max_s - 2:
                    preds_tail.append(pred)

        if len(preds_tail) == 3:
            p_m2, p_m1, p_last = preds_tail
            flips_last = (p_last != p_m1).any(dim=-1)
            period2 = flips_last & (p_last == p_m2).all(dim=-1)
            osc_boards += int(period2.sum().item())
            tail_flips += float((p_last != p_m1).float().mean(dim=-1).sum().item())

    board_exact = {s: exact_sums[s] / float(max(n_total, 1)) for s in eval_outer_steps}
    cell_acc = {s: cell_sums[s] / float(max(n_total, 1)) for s in eval_outer_steps}
    return SudokuHorizonMetrics(
        board_exact_by_step=board_exact,
        cell_acc_by_step=cell_acc,
        period2_oscillation_rate=osc_boards / float(max(n_total, 1)),
        step_flip_rate_tail=tail_flips / float(max(n_total, 1)),
    )


def encode_sudoku_pairs(
    questions: list[str], solutions: list[str], device: torch.device | str = "cpu"
) -> tuple[Tensor, Tensor]:
    """Convert 81-char question/solution strings into ``(inputs [N, 81], targets_0_to_8 [N, 81])``."""
    inp = encode_questions(questions, device=device)
    tgt_arr = [[int(ch) - 1 for ch in sol] for sol in solutions]
    tgt = torch.tensor(tgt_arr, dtype=torch.long, device=device)
    return inp, tgt
