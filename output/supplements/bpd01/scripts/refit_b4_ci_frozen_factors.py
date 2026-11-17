"""BPD-01 Experiment C: refit the B4 state_only CI network on the frozen stage-4 factors.

The historical B4 v2 run deleted its trained CI networks before the stage-4 export
(``gate_b4_v2_stage4_receipt.json``: ``includes_ci_networks: false``), so the
0.0103159 trained-gate receipt cannot be re-evaluated. This script fits a NEW
``state_only`` CI on the SHA-verified stage-4 (V, U) with V and U frozen, using
the B4 stage-1a CI architecture and objective. It never replaces the historical
receipt and never reads evaluation boards.

Board roles (WB-Stall indices refer to ``load_wb_stall_and_train_sudoku`` order):
- calibration/training: the first 128 ``train.csv`` boards (the B4 factor-training set;
  disjoint from WB-Stall by construction of the loader);
- validation (best-state selection only): 32 WB-Stall boards ranked by
  ``sha256("BPD01-CI-REFIT-VAL-20261117|{idx:03d}")`` among indices 16..182 that are
  not in the reserved-64 set;
- evaluation: the reserved-64 set, used only by ``audit_b4_causal_ci.py``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
from mechinterp.bpd.components import BPDModel, RolloutTrajectory
from mechinterp.bpd.config import (
    AblationConfig,
    AdversaryConfig,
    BPDConfig,
    CIConfig,
    MinimalityConfig,
    NullCalibrationConfig,
)
from mechinterp.bpd.nulls import calibrate_pass_null_losses
from mechinterp.bpd.scripts.audit_b4_online_svd_reserved64 import BOARD_INDICES as RESERVED64
from mechinterp.bpd.scripts.audit_b4_v2_controls_and_ablation_split import EXPECTED_WEIGHTS_SHA256
from mechinterp.bpd.trainer import train_ci_heads_on_frozen_components

N_PASSES = 63
VAL_SALT = "BPD01-CI-REFIT-VAL-20261117"
N_VAL = 32
FACTOR_KEY = {"gaoxin492": "bpd_ref", "yagizdevre": "bpd_tgt"}


def validation_indices() -> list[int]:
    """32 WB-Stall indices outside the diagnostic 16 and the reserved 64, hash-ranked."""
    reserved = set(RESERVED64)
    pool = [i for i in range(16, 183) if i not in reserved]
    pool.sort(key=lambda i: hashlib.sha256(f"{VAL_SALT}|{i:03d}".encode()).digest())
    return pool[:N_VAL]


def b4_ci_config(steps: int, seed: int, batch_size: int, total_len: int, hidden: int) -> BPDConfig:
    """Stage-1a configuration from ``run_trm_bpd.py`` (C=512, 3 outer steps)."""
    ood_invocations = 6 * 21
    return BPDConfig(
        c_per_module={},
        n_passes=N_PASSES,
        n_steps=steps,
        batch_size=batch_size,
        seq_len=total_len,
        seed=seed,
        main_lr=1e-3,
        coeff_det=0.5,
        coeff_stoch=0.5,
        match_inner_boundary_states=False,
        ci=CIConfig(
            variant="state_only",
            d_model=min(128, max(32, hidden)),
            n_blocks=1,
            n_heads=4,
            mlp_hidden=min(256, max(64, 2 * hidden)),
            max_passes=max(ood_invocations + 8, N_PASSES * 2, 16),
            max_seq_len=max(total_len, 16),
        ),
        ablation=AblationConfig(mode="pass_mean", sphere_rescale=True),
        minimality=MinimalityConfig(pool_mode="mean", coeff_level=2e-4, coeff_switch=2e-4),
        adversary=AdversaryConfig(scope="shared_batch_and_pass", lr=0.03),
        null_calibration=NullCalibrationConfig(use_fragility_hinge=True, hinge_kappa=0.1,
                                               n_rotations_train=2),
    )


def build_frozen_bpd(ckpt: str, factors: dict[str, dict[str, torch.Tensor]], dtype: torch.dtype,
                     device: str) -> tuple[BPDModel, TRMBPDAdapter]:
    base = load_pretrained_trm(ckpt, dtype=torch.bfloat16, verify_sha256=True)
    if dtype != torch.bfloat16:
        base = base.to(dtype)  # BF16-rounded checkpoint weights promoted
    base = base.to(device).eval()
    bpd = BPDModel(base, default_trm_c_per_module(base, c=512), max_passes=N_PASSES * 2 + 8,
                   ablation=AblationConfig(mode="pass_mean", sphere_rescale=True),
                   init_svd=False).to(device)
    for m, w in bpd.wrappers.items():
        w.V.data.copy_(factors[m]["V"].to(device=device, dtype=torch.float32))
        w.U.data.copy_(factors[m]["U"].to(device=device, dtype=torch.float32))
        w.V.requires_grad_(False)
        w.U.requires_grad_(False)
    return bpd, TRMBPDAdapter(bpd.target_model, readout_at="outer_step")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, choices=sorted(FACTOR_KEY))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", type=Path,
                    default=Path("mechinterp/bpd/results/v2/bpd_b4_v2_stage4_weights.pt"))
    ap.add_argument("--out-dir", type=Path, default=Path("mechinterp/bpd/results/v2/ci_refit"))
    ap.add_argument("--pilot", action="store_true", help="timing/memory pilot: no files written")
    args = ap.parse_args()

    sha = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if sha != EXPECTED_WEIGHTS_SHA256:
        ap.error(f"Factor weights SHA-256 mismatch: {sha}")
    t0 = time.time()
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    factors = torch.load(args.weights, map_location="cpu", weights_only=True)[FACTOR_KEY[args.checkpoint]]
    bpd, adapter = build_frozen_bpd(args.checkpoint, factors, dtype, args.device)

    wb = load_wb_stall_and_train_sudoku(None, n_train=128, device="cpu")
    train = wb["train_inputs"][:128]
    val_idx = validation_indices()
    val = wb["eval_inputs"][val_idx]
    cfg = b4_ci_config(args.steps, args.seed, args.batch_size, int(bpd.target_model.total_len),
                       int(bpd.target_model.hidden_size))
    cfg = replace(cfg, c_per_module=dict(bpd.c_per_module))

    batches = [
        adapter.make_batch_with_clean_cache(train[i:i + args.batch_size].to(args.device), N_PASSES)
        for i in range(0, train.shape[0], args.batch_size)
    ]
    val_batch = adapter.make_batch_with_clean_cache(val.to(args.device), N_PASSES)

    # Pass-mean calibration and per-pass rotation-null hinge baselines on the first batch,
    # as BPDTrainer.calibrate_baselines does for stage 1a.
    with torch.no_grad():
        with bpd.target_capture(pass_offset=0, update_means=True, detach_inputs=True):
            ro, bs, vis, prm = adapter.rollout(batches[0], N_PASSES, 0, None)
        clean = RolloutTrajectory(readouts=ro, boundary_states=bs,
                                  module_inputs=bpd.collect_recorded_inputs(expected_passes=N_PASSES),
                                  visible_states=vis, prompt_features=prm)

        def _plain() -> RolloutTrajectory:
            r2, b2, v2, p2 = adapter.rollout(batches[0], N_PASSES, 0, None)
            return RolloutTrajectory(readouts=r2, boundary_states=b2, module_inputs=clean.module_inputs,
                                     visible_states=v2, prompt_features=p2)

        null_losses = calibrate_pass_null_losses(
            bpd, clean, _plain,
            n_rotations=cfg.null_calibration.n_rotations_train,
            perturbation_scale=cfg.null_calibration.perturbation_scale,
            match_inner_boundary_states=False,
            state_boundary_weight=cfg.state_boundary_weight,
            eps=cfg.null_calibration.eps,
            seed=cfg.seed,
        )
    del clean
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t_train = time.time()
    nets, metrics, _gaps = train_ci_heads_on_frozen_components(
        bpd, cfg, adapter.rollout, lambda step: batches[step % len(batches)], val_batch,
        val_batch=val_batch, visible_dim=adapter.visible_dim, prompt_dim=adapter.prompt_dim,
        pass_null_losses=null_losses, hinge_kappa=cfg.null_calibration.hinge_kappa,
        det_recon_weight=0.5, device=args.device, variants=("state_only",),
    )
    train_s = time.time() - t_train
    peak = torch.cuda.max_memory_allocated() / 2**30 if args.device.startswith("cuda") else None
    net = nets["state_only"]
    receipt: dict[str, Any] = {
        "experiment": "BPD-01_C_frozen_factor_CI_refit",
        "replaces_historical_trained_gate_receipt": False,
        "checkpoint": args.checkpoint,
        "factor_key": FACTOR_KEY[args.checkpoint],
        "factor_weights_sha256": sha,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "model_dtype": args.dtype,
        "device": args.device,
        "cuda_device": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "torch_version": torch.__version__,
        "train_boards": "train.csv first 128 non-WB-Stall boards (B4 factor-training set)",
        "validation_wb_stall_indices": val_idx,
        "evaluation_boards_read": False,
        "pass_null_losses": {str(k): float(v) for k, v in null_losses.items()},
        "ci_config": asdict(cfg.ci),
        "validation_metrics_best_state": metrics["state_only"],
        "train_seconds": train_s,
        "seconds_per_step": train_s / max(args.steps, 1),
        "peak_cuda_gib": peak,
        "total_seconds": time.time() - t0,
    }
    if args.pilot:
        print(json.dumps(receipt, indent=1))
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.checkpoint}_seed{args.seed}_state_only_ci"
    wpath = args.out_dir / f"{stem}.pt"
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "ci_config": asdict(cfg.ci),
                "module_dims": {m: bpd.wrappers[m].d_in for m in bpd.module_order},
                "c_per_module": dict(bpd.c_per_module)}, wpath)
    receipt["ci_weights_file"] = wpath.as_posix()
    receipt["ci_weights_sha256"] = hashlib.sha256(wpath.read_bytes()).hexdigest()
    (args.out_dir / f"{stem}.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("checkpoint", "seed", "ci_weights_sha256",
                                               "seconds_per_step", "peak_cuda_gib")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
