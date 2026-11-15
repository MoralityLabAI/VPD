"""Gate B1: Unrolled VPD baseline vs. Weight-Tied BPD on a stationary loop.

Tests the falsifiable claim:
  On a stationary weight-tied loop of depth ``T``, weight-tied BPD with ``C``
  shared components matches the trajectory reconstruction KL of unrolled VPD
  with ``T * C`` components (``~1/T`` component allocation), while outperforming
  parameter-matched unrolled VPD (``max(C // T, 1)`` components per pass).
  If tied BPD fails to match ``T * C`` unrolled VPD at ``~1/T`` components,
  parameter tying in the decomposition buys nothing.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch
from torch import Tensor, nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import (
    AblationConfig,
    AdversaryConfig,
    BPDConfig,
    CIConfig,
    MinimalityConfig,
    NullCalibrationConfig,
)
from mechinterp.bpd.trainer import BPDTrainer, UnrolledVPDTrainer


class StationaryResidualLooper(nn.Module):
    """Stationary weight-tied loop whose shared core matrix has low intrinsic rank ``r_true``."""

    def __init__(
        self,
        vocab_size: int = 24,
        d_model: int = 32,
        r_true: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.core_proj = nn.Linear(d_model, d_model, bias=False)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        with torch.no_grad():
            self.embed.weight.normal_(0.0, 0.4, generator=gen)
            a = torch.randn((d_model, r_true), generator=gen) / (d_model ** 0.5)
            b = torch.randn((r_true, d_model), generator=gen) / (r_true ** 0.5)
            self.core_proj.weight.copy_((a @ b).T)
            self.head.weight.normal_(0.0, 1.0 / (d_model ** 0.5), generator=gen)

    @staticmethod
    def _rms(x: Tensor, eps: float = 1e-6) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    def rollout(
        self,
        tokens: Tensor,
        n_passes: int,
        pass_offset: int = 0,
        init_state: Tensor | None = None,
    ) -> tuple[dict[int, Tensor], dict[int, Tensor], dict[int, Tensor], Tensor]:
        x_emb = self.embed(tokens)
        h = init_state if init_state is not None else x_emb.clone()
        readouts: dict[int, Tensor] = {}
        boundaries: dict[int, Tensor] = {}
        vis: dict[int, Tensor] = {}
        for k in range(n_passes):
            tau = pass_offset + k
            vis[tau] = self.head(h).float()
            h = self._rms(h + F.gelu(self.core_proj(h)) + 0.25 * x_emb)
            readouts[tau] = self.head(h)
            boundaries[tau] = h
        return readouts, boundaries, vis, x_emb.float()


def run_benchmark(
    *,
    n_passes: int = 4,
    c_shared: int = 4,
    n_steps: int = 80,
    batch_size: int = 24,
    seq_len: int = 8,
    device: str = "cpu",
    seed: int = 7,
) -> dict[str, object]:
    torch.manual_seed(seed)
    base_model = StationaryResidualLooper(vocab_size=24, d_model=32, r_true=c_shared, seed=seed).to(
        device
    )

    cfg = BPDConfig(
        c_per_module={"core_proj": c_shared},
        n_passes=n_passes,
        n_steps=n_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        seed=seed,
        faithfulness_warmup_steps=20,
        main_lr=3e-3,
        coeff_faith=1e3,
        coeff_stoch=1.0,
        coeff_adv=0.0,
        rollout_mode="closed_loop",
        ci=CIConfig(variant="state_only", d_model=32, n_blocks=1, n_heads=4, mlp_hidden=64),
        ablation=AblationConfig(mode="pass_mean", sphere_rescale=True),
        minimality=MinimalityConfig(pool_mode="mean", coeff_level=1e-4, coeff_switch=2e-4),
        adversary=AdversaryConfig(scope="shared_batch_and_pass"),
        null_calibration=NullCalibrationConfig(use_tolerance_weights=False),
    )

    gen = torch.Generator().manual_seed(seed + 100)
    eval_tokens = torch.randint(0, 24, (batch_size, seq_len), generator=gen).to(device)

    # 1. Weight-tied BPD (C total components)
    tied_target = copy.deepcopy(base_model)
    bpd_model = BPDModel(
        tied_target,
        {"core_proj": c_shared},
        max_passes=n_passes,
        ablation=cfg.ablation,
        init_svd=True,
    )
    tied_trainer = BPDTrainer(bpd_model, cfg, tied_target.rollout, device=device)
    tied_trainer.run_faithfulness_warmup()
    tied_trainer.calibrate_baselines(eval_tokens)
    for step in range(n_steps):
        tok = torch.randint(0, 24, (batch_size, seq_len), generator=gen).to(device)
        tied_trainer.train_step(tok, step)
    tied_eval, _, _ = tied_trainer.evaluate(eval_tokens, delta_value=0.0)

    # 2. Unrolled VPD with T * C components (each pass gets C components)
    unrolled_full_target = copy.deepcopy(base_model)
    unrolled_full = UnrolledVPDTrainer(
        unrolled_full_target,
        {"core_proj": c_shared},
        cfg,
        unrolled_full_target.rollout,
        device=device,
    )
    for step in range(n_steps):
        tok = torch.randint(0, 24, (batch_size, seq_len), generator=gen).to(device)
        unrolled_full.train_step(tok, step)
    unrolled_full_eval = unrolled_full.evaluate(eval_tokens, delta_value=0.0)

    # 3. Budget-matched Unrolled VPD with max(C // T, 1) components per pass (~C total)
    c_split = max(c_shared // n_passes, 1)
    unrolled_matched_target = copy.deepcopy(base_model)
    unrolled_matched = UnrolledVPDTrainer(
        unrolled_matched_target,
        {"core_proj": c_split},
        cfg,
        unrolled_matched_target.rollout,
        device=device,
    )
    for step in range(n_steps):
        tok = torch.randint(0, 24, (batch_size, seq_len), generator=gen).to(device)
        unrolled_matched.train_step(tok, step)
    unrolled_matched_eval = unrolled_matched.evaluate(eval_tokens, delta_value=0.0)

    tied_c = bpd_model.total_components()
    unrolled_full_c = unrolled_full.total_unrolled_components()
    gate_b1_pass = (
        tied_c * n_passes == unrolled_full_c
        and tied_eval["recon_loss"] <= max(unrolled_full_eval["recon_loss"] * 1.5, 5e-3)
        and tied_eval["recon_loss"] < unrolled_matched_eval["recon_loss"]
    )

    return {
        "gate": "B1_unrolled_vs_tied",
        "n_passes": n_passes,
        "c_shared": c_shared,
        "tied_bpd": {
            "allocated_components": tied_c,
            **tied_eval,
        },
        "unrolled_vpd_TxC": {
            "allocated_components": unrolled_full_c,
            **unrolled_full_eval,
        },
        "unrolled_vpd_budget_matched": {
            "allocated_components": unrolled_matched.total_unrolled_components(),
            **unrolled_matched_eval,
        },
        "compression_ratio": float(unrolled_full_c) / float(tied_c),
        "gate_b1_passed": bool(gate_b1_pass),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--c-shared", type=int, default=4)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--smoke", action="store_true", help="Fast CPU smoke test")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    steps = 25 if args.smoke else args.steps
    res = run_benchmark(
        n_passes=args.passes,
        c_shared=args.c_shared,
        n_steps=steps,
        batch_size=16 if args.smoke else 24,
        device=args.device,
    )
    text = json.dumps(res, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
