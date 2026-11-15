"""BPD adapter for 7M Attention-TRM (``PureSudokuTRM``) on 9x9 Sudoku.

Handles TRM's two-level recurrence (``S`` outer ACT commits x ``h_cycles`` H-cycles
x ``(l_cycles + 1)`` inner ``L_level`` invocations per H-cycle):
  - Outer-step ``z_H`` commits expose the 81-cell digit readout -> matched via exact
    digit-logit KL at every outer step by default (``readout_at="outer_step"``).
  - Inner ``z_L`` updates (and intermediate H-cycle commits when ``readout_at="outer_step"``)
    have no outer-step readout -> matched via loop-boundary state discrepancy **only** when
    ``BPDConfig.match_inner_boundary_states=True`` (off by default for ``TRM``).
  - ``visible_only`` CI receives strictly ``(x_emb, V_tau)`` where ``V_tau`` projects
    board cells ``z_H[:, puzzle_emb_len:]`` onto ``rowspace(lm_head)``, masking out
    both ``z_L`` and ``ker(lm_head)`` (exact parity with LOOPSCOPE-03).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from torch import Tensor

from mechinterp.loopscope02.scripts import sudoku_prop as SP
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
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return PureSudokuTRM.from_state_dict(sd, dtype=dtype)


@dataclass
class TRMBatch:
    """Tokenized Sudoku batch with optional cached clean states ``(z_h, z_l)`` per invocation."""

    inputs: Tensor  # [B, seq_len] long
    clean_states: list[tuple[Tensor, Tensor]] | None = None


class TRMBPDAdapter:
    """Connects ``PureSudokuTRM`` to ``BPDTrainer`` (closed-loop + clean-transplant windows)."""

    def __init__(
        self,
        model: PureSudokuTRM,
        *,
        readout_at: Literal["outer_step", "h_cycle"] = "outer_step",
    ) -> None:
        self.model = model
        self.readout_at = readout_at
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

    def is_readout_pass(self, tau: int) -> bool:
        """Return ``True`` when invocation ``tau`` emits a behavioral digit readout."""
        if self.readout_at == "outer_step":
            return ((tau + 1) % self.calls_per_outer_step) == 0
        return (tau % self.calls_per_h_cycle) == int(self.model.l_cycles)

    def valid_window_starts(
        self,
        total_invocations: int,
        window_steps: int,
        match_inner_boundary_states: bool = False,
    ) -> list[int]:
        """Return valid ``tau0`` starts so every window contains a supervised target pass."""
        w = max(1, min(int(window_steps), int(total_invocations)))
        max_start = max(int(total_invocations) - w, 0)
        if match_inner_boundary_states:
            return list(range(max_start + 1))
        starts = [
            tau0
            for tau0 in range(max_start + 1)
            if any(self.is_readout_pass(tau0 + k) for k in range(w))
        ]
        return starts if starts else [max_start]

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
                if self.is_readout_pass(tau):
                    readouts[tau] = self.model.digit_logits(z_h)
                boundary_states[tau] = z_h

        return readouts, boundary_states, visible_states, x_emb.float()


@dataclass
class SudokuHorizonMetrics:
    """Exact board accuracy (final & ever-solved), cell accuracy, and oscillation across outer steps."""

    board_exact_by_step: dict[int, float]
    cell_acc_by_step: dict[int, float]
    final_solve_rate: float
    ever_solved_rate: float
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
    """Evaluate exact Sudoku solve rate (final & ever-solved) and oscillation up to ``max(eval_outer_steps)``.

    ``inputs`` has shape ``[N, seq_len]`` (token IDs), ``targets`` has shape ``[N, seq_len]``
    with ground-truth digit indices ``0..8`` (corresponding to digits ``1..9``).
    """
    dev = model.embed_tokens.weight.device
    max_s = max(eval_outer_steps)
    milestones = set(eval_outer_steps)
    n_total = inputs.shape[0]

    exact_sums = {s: 0 for s in eval_outer_steps}
    cell_sums = {s: 0.0 for s in eval_outer_steps}
    ever_solved_total = 0
    osc_boards = 0
    tail_flips = 0.0

    for start in range(0, n_total, batch_size):
        inp = inputs[start : start + batch_size].to(dev)
        tgt = targets[start : start + batch_size].to(dev)
        bsz = inp.shape[0]
        x_emb = model.encode_inputs(inp)
        z_h, z_l = model.init_states(bsz)

        ever_solved_batch = torch.zeros(bsz, dtype=torch.bool, device=dev)
        preds_tail: list[Tensor] = []  # last 3 steps to detect period-2 oscillation
        for s in range(1, max_s + 1):
            z_h, z_l = model.act_step(z_h, z_l, x_emb)
            pred = model.digit_logits(z_h).argmax(dim=-1)  # [B, seq_len] in 0..8
            match = pred == tgt
            board_exact_s = match.all(dim=-1)
            ever_solved_batch |= board_exact_s
            if s in milestones:
                exact_sums[s] += int(board_exact_s.sum().item())
                cell_sums[s] += float(match.float().mean(dim=-1).sum().item())
            if s >= max_s - 2:
                preds_tail.append(pred)

        ever_solved_total += int(ever_solved_batch.sum().item())
        if len(preds_tail) == 3:
            p_m2, p_m1, p_last = preds_tail
            flips_last = (p_last != p_m1).any(dim=-1)
            period2 = flips_last & (p_last == p_m2).all(dim=-1)
            osc_boards += int(period2.sum().item())
            tail_flips += float((p_last != p_m1).float().mean(dim=-1).sum().item())

    denom = float(max(n_total, 1))
    board_exact = {s: exact_sums[s] / denom for s in eval_outer_steps}
    cell_acc = {s: cell_sums[s] / denom for s in eval_outer_steps}
    return SudokuHorizonMetrics(
        board_exact_by_step=board_exact,
        cell_acc_by_step=cell_acc,
        final_solve_rate=board_exact[max_s],
        ever_solved_rate=ever_solved_total / denom,
        period2_oscillation_rate=osc_boards / denom,
        step_flip_rate_tail=tail_flips / denom,
    )


def encode_sudoku_pairs(
    questions: list[str], solutions: list[str], device: torch.device | str = "cpu"
) -> tuple[Tensor, Tensor]:
    """Convert 81-char question/solution strings into ``(inputs [N, 81], targets_0_to_8 [N, 81])``."""
    inp = encode_questions(questions, device=device)
    tgt_arr = [[int(ch) - 1 for ch in sol] for sol in solutions]
    tgt = torch.tensor(tgt_arr, dtype=torch.long, device=device)
    return inp, tgt


def load_wb_stall_and_train_sudoku(
    bundle_path: str | Path | None = None,
    *,
    n_train: int = 32,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load the 183 ``WB-Stall`` (``prop_solved == False``) boards and disjoint ``train.csv`` boards."""
    if bundle_path is None:
        bundle_path = (
            Path(__file__).resolve().parents[2] / "loopscope02" / "results" / "benchmark_bundle.json"
        )
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    wb_stall_items = [it for it in bundle["items"] if not bool(it.get("prop_solved", True))]
    eval_questions = [it["question"] for it in wb_stall_items]
    eval_answers = [it["answer"] for it in wb_stall_items]
    eval_q_set = set(eval_questions)

    ds_paths = SP.resolve_dataset_files()
    train_questions: list[str] = []
    train_answers: list[str] = []
    with open(ds_paths["train.csv"], "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = row["question"]
            if q in eval_q_set:
                continue
            train_questions.append(q)
            train_answers.append(row["answer"])
            if len(train_questions) >= n_train:
                break

    train_inputs, train_targets = encode_sudoku_pairs(train_questions, train_answers, device=device)
    eval_inputs, eval_targets = encode_sudoku_pairs(eval_questions, eval_answers, device=device)
    return {
        "train_inputs": train_inputs,
        "train_targets": train_targets,
        "eval_inputs": eval_inputs,
        "eval_targets": eval_targets,
        "n_train": len(train_questions),
        "n_wb_stall": len(wb_stall_items),
        "bundle_path": str(bundle_path),
    }
