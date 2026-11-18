"""B4-CAUSAL-CI-02: train the state_only CI closed-loop for causal preflight top-k gating.

Starts from the B4-CAUSAL-CI-01 refit CI (``ci_refit/{ckpt}_seed0_state_only_ci.pt``), keeps
the stage-4 (V, U) frozen, and trains on the scored policy. Every gate comes from a
preflight observation of the CI's own masked rollout and keeps the top k = 307 logits,
with a straight-through gradient. Regime: FP32 compute with BF16-rounded checkpoint
weights, TF32 off, the same as the scoring. Board roles match the refit: the first 128
train.csv boards for training, the 32 hash-ranked WB-Stall validation boards for
best-state selection, and the reserved 64 never read.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechinterp.bpd.adapters.trm import (
    TRMBPDAdapter,
    default_trm_c_per_module,
    load_pretrained_trm,
    load_wb_stall_and_train_sudoku,
)
from mechinterp.bpd.causal_ci_train import train_closed_loop_ci
from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.scripts.audit_b4_causal_ci import C, FACTOR_KEY, K, N_PASSES, ZERO, load_ci
from mechinterp.bpd.scripts.audit_b4_v2_controls_and_ablation_split import EXPECTED_WEIGHTS_SHA256
from mechinterp.bpd.scripts.refit_b4_ci_frozen_factors import validation_indices


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, choices=sorted(FACTOR_KEY))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--anneal-fraction", type=float, default=0.5)
    ap.add_argument("--state-weight", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init-ci-dir", type=Path, default=Path("mechinterp/bpd/results/v2/ci_refit"))
    ap.add_argument("--weights", type=Path, default=Path("mechinterp/bpd/results/v2/bpd_b4_v2_stage4_weights.pt"))
    ap.add_argument("--out-dir", type=Path, default=Path("mechinterp/bpd/results/v2/ci_closed_loop"))
    ap.add_argument("--pilot", action="store_true", help="timing/memory pilot: no files written")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sha = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if sha != EXPECTED_WEIGHTS_SHA256:
        ap.error(f"Factor weights SHA-256 mismatch: {sha}")
    t0 = time.time()
    torch.manual_seed(args.seed)
    dev, ck = args.device, args.checkpoint
    factors = torch.load(args.weights, map_location="cpu", weights_only=True)[FACTOR_KEY[ck]]
    init_rec = json.loads((args.init_ci_dir / f"{ck}_seed0_state_only_ci.json").read_text(encoding="utf-8"))
    init_path = args.init_ci_dir / f"{ck}_seed0_state_only_ci.pt"
    ci = load_ci(init_path, init_rec["ci_weights_sha256"], dev)

    base = load_pretrained_trm(ck, dtype=torch.bfloat16, verify_sha256=True).float()
    bpd = BPDModel(base.to(dev).eval(), default_trm_c_per_module(base, c=C), max_passes=N_PASSES,
                   ablation=ZERO, init_svd=False).to(dev)
    for m, w in bpd.wrappers.items():
        w.V.data.copy_(factors[m]["V"].to(device=dev, dtype=torch.float32))
        w.U.data.copy_(factors[m]["U"].to(device=dev, dtype=torch.float32))
    adapter = TRMBPDAdapter(bpd.target_model, readout_at="outer_step")

    wb = load_wb_stall_and_train_sudoku(None, n_train=128, device="cpu")
    train = wb["train_inputs"][:128]
    val_idx = validation_indices()
    val = wb["eval_inputs"][val_idx].to(dev)
    gen = torch.Generator().manual_seed(args.seed)
    n_per_epoch = train.shape[0] // args.batch_size
    perms: dict[int, torch.Tensor] = {}

    def get_batch(step: int) -> torch.Tensor:
        epoch, j = divmod(step, n_per_epoch)
        if epoch not in perms:
            perms[epoch] = torch.randperm(train.shape[0], generator=gen)
        idx = perms[epoch][j * args.batch_size:(j + 1) * args.batch_size]
        return train[idx].to(dev)

    lines: list[str] = []

    def log(s: str) -> None:
        lines.append(s)
        print(s, flush=True)

    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t_train = time.time()
    net, info = train_closed_loop_ci(
        bpd, adapter, ci, get_batch, val, n_passes=N_PASSES, k=K, n_steps=args.steps, lr=args.lr,
        anneal_fraction=args.anneal_fraction, state_weight=args.state_weight,
        val_every=args.val_every if not args.pilot else max(args.steps, 1), seed=args.seed, log=log,
    )
    train_s = time.time() - t_train
    peak = torch.cuda.max_memory_allocated() / 2**30 if dev.startswith("cuda") else None
    receipt: dict[str, Any] = {
        "experiment": "B4-CAUSAL-CI-02_closed_loop_CI",
        "registration": "mechinterp/bpd/docs/B4_CAUSAL_CI_CLOSED_LOOP_REGISTRATION_20261118.md",
        "replaces_historical_trained_gate_receipt": False,
        "checkpoint": ck,
        "factor_key": FACTOR_KEY[ck],
        "factor_weights_sha256": sha,
        "init_ci_weights_sha256": init_rec["ci_weights_sha256"],
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "anneal_fraction": args.anneal_fraction,
        "state_weight": args.state_weight,
        "k": K,
        "regime": "FP32 compute, BF16-rounded checkpoint weights, TF32 off",
        "device": dev,
        "cuda_device": torch.cuda.get_device_name() if dev.startswith("cuda") else None,
        "torch_version": torch.__version__,
        "train_boards": "train.csv first 128 non-WB-Stall boards (B4 factor-training set)",
        "validation_wb_stall_indices": val_idx,
        "evaluation_boards_read": False,
        "ci_config": asdict(ci.cfg),
        "validation_metrics_best_state": {**info["best_validation"],
                                          "recon_loss": info["best_validation"]["kl_mean"]},
        "validation_history": info["validation"],
        "train_history": info["history"],
        "train_seconds": train_s,
        "seconds_per_step": train_s / max(args.steps, 1),
        "peak_cuda_gib": peak,
        "total_seconds": time.time() - t0,
    }
    if args.pilot:
        print(json.dumps({k: receipt[k] for k in ("seconds_per_step", "peak_cuda_gib", "validation_metrics_best_state")},
                         indent=1))
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{ck}_seed{args.seed}_closed_loop_ci"
    wpath = args.out_dir / f"{stem}.pt"
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "ci_config": asdict(net.cfg),
                "module_dims": {m: bpd.wrappers[m].d_in for m in bpd.module_order},
                "c_per_module": dict(bpd.c_per_module)}, wpath)
    receipt["ci_weights_file"] = wpath.as_posix()
    receipt["ci_weights_sha256"] = hashlib.sha256(wpath.read_bytes()).hexdigest()
    (args.out_dir / f"{stem}.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("checkpoint", "ci_weights_sha256", "seconds_per_step", "peak_cuda_gib",
                                               "validation_metrics_best_state")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
