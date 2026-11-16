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
import math
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
    resolve_registration_commit,
)
from mechinterp.bpd.trainer import BPDTrainer, UnrolledVPDTrainer


class StationaryRoutedLooper(nn.Module):
    """Stationary weight-tied loop with ``n_mechanisms`` routed rank-1 mechanisms (``k_active`` per item).

    Constructs ``core_proj`` of rank ``n_mechanisms`` over orthogonal input/output subspaces,
    while each input item embeds non-zero signal in only ``k_active <= n_mechanisms`` subspaces
    stationarily across all ``T`` passes. Thus every single item executes a rank-``k_active``
    update per pass (representable by ``k_active = C // T`` components if mechanisms were fixed),
    while covering the full input distribution requires all ``n_mechanisms = C`` mechanisms.
    """

    mech_v: Tensor

    def __init__(
        self,
        vocab_size: int = 32,
        d_model: int = 32,
        n_mechanisms: int = 8,
        k_active: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_mechanisms = min(n_mechanisms, d_model)
        self.k_active = max(1, min(k_active, self.n_mechanisms))

        q_in, _ = torch.linalg.qr(torch.randn((d_model, d_model), generator=gen))
        q_out, _ = torch.linalg.qr(torch.randn((d_model, d_model), generator=gen))
        v_cols = q_in[:, : self.n_mechanisms]  # [d_model, M] orthonormal
        u_cols = q_out[:, : self.n_mechanisms]  # [d_model, M] orthonormal
        scales = torch.linspace(1.1, 1.6, self.n_mechanisms)

        self.register_buffer("mech_v", v_cols.contiguous())
        self.token_coeffs = nn.Embedding(vocab_size, self.n_mechanisms)
        self.core_proj = nn.Linear(d_model, d_model, bias=False)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        with torch.no_grad():
            # Each token activates exactly k_active mechanisms
            coeffs = torch.zeros((vocab_size, self.n_mechanisms))
            for tok_i in range(vocab_size):
                chosen = torch.randperm(self.n_mechanisms, generator=gen)[: self.k_active]
                signs = torch.where(
                    torch.rand((self.k_active,), generator=gen) > 0.5,
                    torch.ones(self.k_active),
                    -torch.ones(self.k_active),
                )
                coeffs[tok_i, chosen] = signs * (1.2 / math.sqrt(float(self.k_active)))
            self.token_coeffs.weight.copy_(coeffs)
            w_mat = (u_cols * scales.unsqueeze(0)) @ v_cols.T
            self.core_proj.weight.copy_(w_mat)
            self.head.weight.normal_(0.0, 3.5 / math.sqrt(float(d_model)), generator=gen)

    def _embed(self, tokens: Tensor) -> Tensor:
        c = self.token_coeffs(tokens)  # [B, P, M]
        return c @ self.mech_v.T  # [B, P, d_model], strictly inside the k_active active subspaces

    def rollout(
        self,
        tokens: Tensor,
        n_passes: int,
        pass_offset: int = 0,
        init_state: Tensor | None = None,
    ) -> tuple[dict[int, Tensor], dict[int, Tensor], dict[int, Tensor], Tensor]:
        dev = self.head.weight.device
        tok = tokens.to(dev)
        x_emb = self._embed(tok)
        h = init_state.to(dev) if init_state is not None else x_emb.clone()
        readouts: dict[int, Tensor] = {}
        boundaries: dict[int, Tensor] = {}
        vis: dict[int, Tensor] = {}
        # Project carrying state back onto active token subspaces so each item stays rank k_active
        c_mask = (self.token_coeffs(tok).abs() > 1e-6).float()  # [B, P, M]
        for k in range(n_passes):
            tau = pass_offset + k
            vis[tau] = self.head(h).float()
            raw_step = 0.40 * h + 1.25 * torch.tanh(self.core_proj(h)) + 0.35 * x_emb
            # Keep trajectory stationary within the item's k_active routed subspaces
            proj_coords = (raw_step @ self.mech_v) * c_mask
            h = proj_coords @ self.mech_v.T
            readouts[tau] = self.head(h)
            boundaries[tau] = h
        return readouts, boundaries, vis, x_emb.float()


# Backward-compatible alias
StationaryResidualLooper = StationaryRoutedLooper


def _run_single_seed_benchmark(
    *,
    n_passes: int,
    c_shared: int,
    n_steps: int,
    batch_size: int,
    seq_len: int,
    device: str,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    k_act = max(c_shared // n_passes, 1)
    vocab_sz = 32
    base_model = StationaryRoutedLooper(
        vocab_size=vocab_sz,
        d_model=max(32, c_shared * 2),
        n_mechanisms=c_shared,
        k_active=k_act,
        seed=seed,
    ).to(device)

    # Strictly like-for-like objective across Tied BPD and Unrolled VPD:
    # unhinged trajectory KL (use_fragility_hinge=False), identical zero ablation, identical warmup & LR decay
    cfg = BPDConfig(
        c_per_module={"core_proj": c_shared},
        n_passes=n_passes,
        n_steps=n_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        seed=seed,
        faithfulness_warmup_steps=20,
        main_lr=2e-3,
        coeff_faith=1e4,
        coeff_det=0.5,
        coeff_stoch=0.5,
        coeff_adv=0.0,
        rollout_mode="closed_loop",
        ci=CIConfig(variant="state_only", d_model=48, n_blocks=1, n_heads=4, mlp_hidden=96),
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        minimality=MinimalityConfig(pool_mode="mean", coeff_level=2e-4, coeff_switch=2e-4),
        adversary=AdversaryConfig(scope="shared_batch_and_pass"),
        null_calibration=NullCalibrationConfig(use_fragility_hinge=False, hinge_kappa=0.0),
    )

    gen_eval = torch.Generator().manual_seed(seed + 100)
    eval_tokens = torch.randint(0, vocab_sz, (batch_size * 2, seq_len), generator=gen_eval).to(device)

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
    gen_t1 = torch.Generator().manual_seed(seed + 500)
    last_train_shares: dict[int, float] = {}
    for step in range(n_steps):
        tok = torch.randint(0, vocab_sz, (batch_size, seq_len), generator=gen_t1).to(device)
        sm = tied_trainer.train_step(tok, step)
        last_train_shares = sm.pass_loss_shares
    tied_eval, _, _ = tied_trainer.evaluate(eval_tokens, delta_value=0.0)

    # 2. Unrolled VPD with T * C components (each pass gets C components, identical data stream)
    unrolled_full_target = copy.deepcopy(base_model)
    unrolled_full = UnrolledVPDTrainer(
        unrolled_full_target,
        {"core_proj": c_shared},
        cfg,
        unrolled_full_target.rollout,
        device=device,
    )
    unrolled_full.run_faithfulness_warmup()
    gen_t2 = torch.Generator().manual_seed(seed + 500)
    for step in range(n_steps):
        tok = torch.randint(0, vocab_sz, (batch_size, seq_len), generator=gen_t2).to(device)
        unrolled_full.train_step(tok, step)
    unrolled_full_eval = unrolled_full.evaluate(eval_tokens, delta_value=0.0)

    # 3. Budget-matched Unrolled VPD with C // T = k_active components per pass (C total components)
    c_split = max(c_shared // n_passes, 1)
    unrolled_matched_target = copy.deepcopy(base_model)
    unrolled_matched = UnrolledVPDTrainer(
        unrolled_matched_target,
        {"core_proj": c_split},
        cfg,
        unrolled_matched_target.rollout,
        device=device,
    )
    unrolled_matched.run_faithfulness_warmup()
    gen_t3 = torch.Generator().manual_seed(seed + 500)
    for step in range(n_steps):
        tok = torch.randint(0, vocab_sz, (batch_size, seq_len), generator=gen_t3).to(device)
        unrolled_matched.train_step(tok, step)
    unrolled_matched_eval = unrolled_matched.evaluate(eval_tokens, delta_value=0.0)

    return {
        "seed": seed,
        "k_active_per_item": k_act,
        "last_train_pass_loss_shares": {str(k): v for k, v in last_train_shares.items()},
        "tied_bpd": {
            "allocated_components": bpd_model.total_components(),
            **tied_eval,
        },
        "unrolled_vpd_TxC": {
            "allocated_components": unrolled_full.total_unrolled_components(),
            **unrolled_full_eval,
        },
        "unrolled_vpd_budget_matched": {
            "allocated_components": unrolled_matched.total_unrolled_components(),
            **unrolled_matched_eval,
        },
    }


def run_benchmark(
    *,
    n_passes: int = 4,
    c_shared: int = 8,
    n_steps: int = 150,
    batch_size: int = 32,
    seq_len: int = 8,
    n_seeds: int = 1,
    hinge_kappa: float = 0.0,
    registration_commit: str | None = None,
    device: str = "cpu",
    seed: int = 7,
) -> dict[str, object]:
    seed_runs = [
        _run_single_seed_benchmark(
            n_passes=n_passes,
            c_shared=c_shared,
            n_steps=n_steps,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            seed=seed + i,
        )
        for i in range(max(n_seeds, 1))
    ]

    def _mean(vals: list[float]) -> float:
        return sum(vals) / float(max(len(vals), 1))

    tied_losses = [float(r["tied_bpd"]["recon_loss"]) for r in seed_runs]  # type: ignore[index]
    full_losses = [float(r["unrolled_vpd_TxC"]["recon_loss"]) for r in seed_runs]  # type: ignore[index]
    match_losses = [float(r["unrolled_vpd_budget_matched"]["recon_loss"]) for r in seed_runs]  # type: ignore[index]

    mean_tied = _mean(tied_losses)
    mean_full = _mean(full_losses)
    mean_match = _mean(match_losses)

    r0 = seed_runs[0]
    tied_c = int(r0["tied_bpd"]["allocated_components"])  # type: ignore[index]
    unrolled_full_c = int(r0["unrolled_vpd_TxC"]["allocated_components"])  # type: ignore[index]

    # Genuine relative parity without artificial 5e-3 floor (only 1e-5 float32 precision guard)
    parity_bound = max(mean_full * 2.0, 1e-5)
    gate_b1_pass = (
        tied_c * n_passes == unrolled_full_c
        and mean_tied <= parity_bound
        and mean_tied <= 0.25 * mean_match
    )

    return {
        "gate": "B1_v2_like_for_like_unrolled_vs_tied",
        "registration_commit": resolve_registration_commit(registration_commit),
        "n_passes": n_passes,
        "c_shared": c_shared,
        "k_active_per_item": max(c_shared // n_passes, 1),
        "use_fragility_hinge": False,
        "hinge_kappa": 0.0,
        "n_seeds": len(seed_runs),
        "mean_tied_recon_loss": mean_tied,
        "mean_unrolled_TxC_recon_loss": mean_full,
        "mean_unrolled_budget_matched_recon_loss": mean_match,
        "parity_upper_bound": parity_bound,
        "tied_to_TxC_ratio": mean_tied / max(mean_full, 1e-12),
        "tied_to_budget_matched_ratio": mean_tied / max(mean_match, 1e-12),
        "tied_bpd": r0["tied_bpd"],
        "unrolled_vpd_TxC": r0["unrolled_vpd_TxC"],
        "unrolled_vpd_budget_matched": r0["unrolled_vpd_budget_matched"],
        "seed_runs": seed_runs,
        "compression_ratio": float(unrolled_full_c) / float(tied_c),
        "gate_b1_passed": bool(gate_b1_pass),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--c-shared", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--registration-commit", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--smoke", action="store_true", help="Fast CPU smoke test")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    steps = 30 if args.smoke else args.steps
    res = run_benchmark(
        n_passes=args.passes,
        c_shared=4 if args.smoke else args.c_shared,
        n_steps=steps,
        batch_size=16 if args.smoke else 32,
        n_seeds=1 if args.smoke else args.seeds,
        registration_commit=args.registration_commit,
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
