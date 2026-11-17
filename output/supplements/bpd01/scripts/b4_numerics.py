"""Shared numerics for the BPD-01 B4 closure and online-gating audits.

All divergence measures here are computed in float64 so that values near the
numerical floor are not themselves dominated by float32 log-softmax rounding
(the historical ``per_board_kl`` helper evaluates in float32 and can return
small negative KL values).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F


def readout_logits64(model: Any, z_h: Tensor) -> Tensor:
    """Digit logits ``[B, 81, 9]`` from ``z_h`` in the model dtype, promoted to float64.

    ``PureSudokuTRM.digit_logits`` casts to float32, which truncates FP64 runs.
    For BF16/FP32 models the values are identical to ``digit_logits`` (exact promotion).
    """
    raw = model.lm_head(z_h)[:, model.puzzle_emb_len:, 2:11]
    return raw.double()


def kl64_per_board(pred_logits: Tensor, target_logits: Tensor) -> list[float]:
    """Token-mean KL(softmax(target) || softmax(pred)) per board, evaluated in float64."""
    p_lp = F.log_softmax(pred_logits.double(), dim=-1)
    t_lp = F.log_softmax(target_logits.double(), dim=-1)
    kl = (t_lp.exp() * (t_lp - p_lp)).sum(dim=-1).mean(dim=-1)
    return [float(v) for v in kl.cpu().tolist()]


def argmax_flips_per_board(pred_logits: Tensor, target_logits: Tensor) -> list[int]:
    """Number of the 81 cells whose predicted digit differs from the reference."""
    flips = (pred_logits.argmax(dim=-1) != target_logits.argmax(dim=-1)).sum(dim=-1)
    return [int(v) for v in flips.cpu().tolist()]


def relative_state_divergence(z: Tensor, z_ref: Tensor) -> list[float]:
    """Per-board ``||z - z_ref||_F / ||z_ref||_F`` over (positions, hidden) in float64."""
    diff = (z.double() - z_ref.double()).flatten(1).norm(dim=-1)
    ref = z_ref.double().flatten(1).norm(dim=-1).clamp_min(1e-300)
    return [float(v) for v in (diff / ref).cpu().tolist()]


def bf16_ulp(y: Tensor) -> Tensor:
    """Spacing of the bfloat16 grid at ``|y|`` (8 significand bits); tiny value at zero."""
    yf = y.float().abs()
    _, e = torch.frexp(yf)
    ulp = torch.ldexp(torch.ones_like(yf), e - 8)
    return torch.where(yf > 0, ulp, torch.full_like(yf, 2.0 ** -133))


def summarize(vals: Sequence[float]) -> dict[str, Any]:
    t = torch.tensor(list(vals), dtype=torch.float64)
    order = torch.argsort(t, descending=True)
    return {
        "mean": float(t.mean()),
        "median": float(t.median()),
        "min": float(t.min()),
        "max": float(t.max()),
        "top3_boards": [
            {"board_idx": int(i), "value": float(t[i])} for i in order[: min(3, t.numel())]
        ],
        "per_board": [float(v) for v in t.tolist()],
    }


def compare_readouts(
    pred: Mapping[int, Tensor],
    ref: Mapping[int, Tensor],
    readouts: Sequence[int],
) -> dict[str, Any]:
    """Per-board float64 KL and argmax flips for each readout pass."""
    out: dict[str, Any] = {}
    for tau in readouts:
        out[f"pass_{tau}"] = {
            "kl64": summarize(kl64_per_board(pred[tau], ref[tau])),
            "argmax_flips_per_board": argmax_flips_per_board(pred[tau], ref[tau]),
        }
    return out


def divergence_curve(
    states: Mapping[int, Tensor],
    ref_states: Mapping[int, Tensor],
    n_passes: int,
) -> dict[str, Any]:
    """Relative boundary-state divergence at every invocation: ``by_pass_by_board [T][B]``."""
    rows = [relative_state_divergence(states[t], ref_states[t]) for t in range(n_passes)]
    t = torch.tensor(rows, dtype=torch.float64)
    return {
        "by_pass_by_board": rows,
        "by_pass_median": [float(v) for v in t.median(dim=1).values.tolist()],
        "by_pass_max": [float(v) for v in t.amax(dim=1).tolist()],
    }


def _rms_norm_dtype_preserving(x: Tensor, eps: float = 1e-5) -> Tensor:
    """RMSNorm that keeps float64 inputs in float64; other dtypes follow the TRM reference."""
    if x.dtype == torch.float64:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    orig = x.dtype
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(orig)


@contextmanager
def exact_fp64_rmsnorm() -> Iterator[None]:
    """Patch the TRM ``rms_norm`` so FP64 models normalize in FP64 instead of FP32.

    The reference implementation always normalizes in float32, which caps an
    "FP64" rollout at float32 precision inside every block. Non-FP64 inputs keep
    the reference behaviour bit-for-bit.
    """
    from mechinterp.loopscope02.scripts import trm_sudoku as trm_mod

    saved: Callable[..., Tensor] = trm_mod.rms_norm
    trm_mod.rms_norm = _rms_norm_dtype_preserving
    try:
        yield
    finally:
        trm_mod.rms_norm = saved


class OutputPerturbation:
    """Forward hook adding a fixed-relative-size random perturbation to one invocation.

    At invocation ``at_call`` (0-based count of calls to the hooked module; ``-1``
    perturbs every invocation with an independent draw) the output ``y`` becomes ``y + eps * ||y_token|| * g / ||g_token||`` per token,
    with ``g`` drawn from a seeded Gaussian on CPU in float64.
    """

    def __init__(self, eps: float, at_call: int, seed: int) -> None:
        self.eps = float(eps)
        self.at_call = int(at_call)
        self.seed = int(seed)
        self.calls = 0

    def __call__(self, _module: Any, _inputs: Any, output: Tensor) -> Tensor:
        idx = self.calls
        self.calls += 1
        if self.eps == 0.0 or (self.at_call >= 0 and idx != self.at_call):
            return output
        gen = torch.Generator().manual_seed(self.seed * 1000 + idx)
        g = torch.randn(output.shape, generator=gen, dtype=torch.float64)
        g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-300)
        y = output.double()
        scale = self.eps * y.norm(dim=-1, keepdim=True)
        return (y + scale * g.to(y.device)).to(output.dtype)


def paired_bootstrap(
    diffs: Sequence[float],
    *,
    n_boot: int = 20000,
    seed: int = 20261117,
    n_signflip: int = 100000,
) -> dict[str, Any]:
    """Board-paired bootstrap CI for the mean difference plus a two-sided sign-flip p-value."""
    import numpy as np

    d = np.asarray(list(diffs), dtype=np.float64)
    n = d.size
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    observed = float(d.mean())
    count = 0
    left = n_signflip
    while left:
        b = min(4096, left)
        signs = rng.integers(0, 2, size=(b, n)) * 2 - 1
        null = (signs * d[None, :]).mean(axis=1)
        count += int((np.abs(null) >= abs(observed) - 1e-300).sum())
        left -= b
    return {
        "n": int(n),
        "mean": observed,
        "median": float(np.median(d)),
        "bootstrap_ci95": [float(lo), float(hi)],
        "signflip_two_sided_p": (1 + count) / (n_signflip + 1),
        "n_negative": int((d < 0).sum()),
        "n_positive": int((d > 0).sum()),
        "n_zero": int((d == 0).sum()),
        "bootstrap_resamples": n_boot,
        "signflip_draws": n_signflip,
        "seed": seed,
    }
