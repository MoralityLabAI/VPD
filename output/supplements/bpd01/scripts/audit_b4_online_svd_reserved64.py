"""Source-locked B4-ONLINE-SVD-02 CPU FP32 reserved-64 prospective audit.

The 64 board indices and the statistical gate were registered in commit
89c265176e7b1a97e9f14f727278ab458963bb2e before model inference.
This benchmark is separate from the 16-board CUDA BF16 closure audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechinterp.bpd.adapters.trm import (
    load_pretrained_trm,
    load_wb_stall_and_train_sudoku,
)
from mechinterp.bpd.scripts.audit_b4_v2_online_svd import audit_one
from mechinterp.loopscope02.scripts.trm_sudoku import TRM_CHECKPOINTS

REGISTRATION_COMMIT = "89c265176e7b1a97e9f14f727278ab458963bb2e"
SALT = "BPD-ONLINE-SVD-01-RESERVED64-20261117"
BOARD_INDICES = (
    31, 57, 67, 154, 68, 138, 83, 50,
    92, 176, 124, 151, 89, 163, 76, 137,
    23, 146, 115, 132, 94, 78, 21, 74,
    135, 86, 181, 112, 90, 44, 113, 87,
    131, 136, 46, 97, 166, 109, 60, 147,
    110, 134, 168, 174, 179, 101, 150, 48,
    62, 72, 157, 116, 145, 130, 107, 140,
    171, 26, 38, 165, 170, 25, 34, 77,
)
BOARD_INDICES_SHA256 = "0f474925d83f86839b411c2f1286c5beec5b3630b40275f299e66a267ce97ab0"
ARMS = (
    "same_dtype_target_rerun",
    "static_rank_k",
    "clean_oracle_dynamic_topk",
    "online_dynamic_topk",
)
N_BOOTSTRAP = 20000
N_SIGNFLIP = 100000
STATS_SEED = 20261117117
NONINFERIOR_MARGIN = 0.10
FAMILYWISE_UPPER_QUANTILE = 0.975


def validate_registration(n_available: int) -> None:
    """Reject board-selection drift before loading checkpoints."""
    selected = sorted(
        range(16, 183),
        key=lambda i: hashlib.sha256(f"{SALT}|{i:03d}".encode("ascii")).digest(),
    )[:64]
    if n_available != 183 or tuple(selected) != BOARD_INDICES:
        raise AssertionError(
            f"Reserved board source mismatch: expected 183 and frozen {BOARD_INDICES}"
        )
    compact = json.dumps(selected, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256(compact).hexdigest()
    if digest != BOARD_INDICES_SHA256:
        raise AssertionError(f"Reserved board index SHA256 mismatch: {digest}")


def _fraction_online_better(deltas: np.ndarray) -> tuple[int, int]:
    return int((deltas < 0).sum()), int((deltas > 0).sum())


def analyze_rows(rows: list[dict[str, Any]], *, seed: int = STATS_SEED) -> dict[str, Any]:
    """Locked paired noninferiority gate with board-level bootstrap and sign flips."""
    if len(rows) != 64:
        raise ValueError(f"Exactly 64 board rows required, got {len(rows)}")
    S = np.array([r["static_kl"] for r in rows], dtype=np.float64)
    C = np.array([r["oracle_kl"] for r in rows], dtype=np.float64)
    O = np.array([r["online_kl"] for r in rows], dtype=np.float64)
    identity = np.array([r["target_repeat_kl"] for r in rows], dtype=np.float64)
    for name, data in (("S", S), ("C", C), ("O", O), ("identity", identity)):
        if not np.isfinite(data).all():
            raise ValueError(f"Nonfinite {name} values")
    if np.max(np.abs(identity)) > 1e-5:
        raise ValueError("Same-dtype target rerun exceeds per-board precision floor")
    # The running estimator measures online excess KL against the clean-informed
    # source. M<0 means online stays within 10% of static KL on average.
    D = O - C
    M = D - NONINFERIOR_MARGIN * S
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(N_BOOTSTRAP, len(rows)))
    boot_D = D[indices].mean(axis=1)
    boot_M = M[indices].mean(axis=1)
    ci_D = np.quantile(boot_D, [0.025, 0.975])
    upper_M = float(np.quantile(boot_M, FAMILYWISE_UPPER_QUANTILE))

    # Monte Carlo paired sign-flip under the symmetric zero-effect null.
    # The test is one-sided for online superiority (mean D < 0).
    # Adding one to count and denominator avoids a zero reported p-value.
    count = 0
    left = N_SIGNFLIP
    observed = float(D.mean())
    while left:
        batch = min(2048, left)
        signs = rng.integers(0, 2, size=(batch, len(rows))) * 2 - 1
        null_mean = (signs * D[None, :]).mean(axis=1)
        count += int((null_mean <= observed).sum())
        left -= batch
    signflip_p = (1 + count) / (N_SIGNFLIP + 1)

    wins, losses = _fraction_online_better(D)
    abs_effect = np.abs(D)
    eff_count = float(abs_effect.sum() ** 2 / max(np.dot(D, D), 1e-30))
    dominant = float(abs_effect.max() / max(abs_effect.sum(), 1e-30))
    return {
        "n_boards": len(rows),
        "target_repeat_max_abs_kl": float(np.abs(identity).max()),
        "mean_static_kl": float(S.mean()),
        "mean_oracle_kl": float(C.mean()),
        "mean_online_kl": float(O.mean()),
        "mean_online_minus_oracle_kl": observed,
        "mean_margin_kl": float(NONINFERIOR_MARGIN * S.mean()),
        "mean_delta_minus_margin": float(M.mean()),
        "upper_97_5pct_bootstrap_mean_delta_minus_margin": upper_M,
        "noninferior_per_checkpoint": bool(upper_M < 0),
        "paired_mean_delta_bootstrap_ci95": [float(x) for x in ci_D],
        "signflip_one_sided_superiority_p": signflip_p,
        "online_better_boards": wins,
        "online_worse_boards": losses,
        "exact_ties": int((D == 0).sum()),
        "median_online_minus_oracle_kl": float(np.median(D)),
        "effect_abs_ess": eff_count,
        "top_board_abs_effect_share": dominant,
        "bootstrap_resamples": N_BOOTSTRAP,
        "signflip_draws": N_SIGNFLIP,
        "rng_seed": seed,
        "bootstrap_quantile_noninferiority": FAMILYWISE_UPPER_QUANTILE,
        "relative_margin_static_kl": NONINFERIOR_MARGIN,
    }


def _collect_rows(
    chunk: dict[str, Any], board_indices: tuple[int, ...],
) -> list[dict[str, Any]]:
    if chunk["readouts"] != [20]:
        raise ValueError(f"Unexpected readouts {chunk['readouts']}")
    rows: list[dict[str, Any]] = []
    for local_idx, index in enumerate(board_indices):
        entry = {
            "board_index": index,
            "board_sha256_prefix16": chunk["board_sha256_prefix16"][local_idx],
            "target_repeat_kl": float(
                chunk["same_dtype_target_rerun"]["pass_20"]["per_board"][local_idx]
            ),
            "static_kl": float(
                chunk["static_rank_k"]["pass_20"]["per_board"][local_idx]
            ),
            "oracle_kl": float(
                chunk["clean_oracle_dynamic_topk"]["pass_20"]["per_board"][local_idx]
            ),
            "online_kl": float(
                chunk["online_dynamic_topk"]["pass_20"]["per_board"][local_idx]
            ),
            "online_vs_oracle_pair_kl": float(
                chunk["online_vs_oracle_rollout"]["pass_20"]["per_board"][local_idx]
            ),
            "online_mask_disagreement": {},
        }
        for site, diagnostic in chunk["oracle_mask_disagreement"].items():
            # 21 invocations x B boards, genuine per-board cross-time mismatch.
            per_pass = np.asarray(diagnostic["by_pass_by_board"], dtype=np.float64)
            if per_pass.shape != (21, len(board_indices)):
                raise ValueError(
                    f"Expected [21,{len(board_indices)}] masks for {site}; {per_pass.shape}"
                )
            entry["online_mask_disagreement"][site] = float(per_pass[:, local_idx].mean())
        if abs(entry["target_repeat_kl"]) > 1e-5:
            raise ValueError(f"Nonclosing identity control for board index {index}")
        if abs(
            (entry["online_kl"] - entry["oracle_kl"])
            - chunk["paired_online_minus_oracle_KL_by_board"]["20"][local_idx]
        ) > 2e-7:
            raise ValueError(f"Inconsistent paired-KL receipt for board {index}")
        rows.append(entry)
    return rows


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", choices=("gaoxin492", "yagizdevre"), required=True)
    ap.add_argument("--device", choices=("cpu",), default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-chunks", type=int, default=8,
                    help="For source/debug testing only; partial reports are invalid for gate")
    args = ap.parse_args()
    if not 1 <= args.max_chunks <= 8:
        ap.error("--max-chunks must be 1..8")
    torch.set_num_threads(2)
    wb = load_wb_stall_and_train_sudoku(None, n_train=128, device="cpu")
    inputs = wb["eval_inputs"]
    validate_registration(int(inputs.shape[0]))
    model = load_pretrained_trm(args.checkpoint, dtype=torch.float32, verify_sha256=True)
    metadata = TRM_CHECKPOINTS[args.checkpoint]

    receipt: dict[str, Any] = {
        "audit": "B4_online_SVD_reserved64_CPU_FP32",
        "status": "running",
        "registration_commit": REGISTRATION_COMMIT,
        "checkpoint": args.checkpoint,
        "model_sha256": metadata["sha256"],
        "model_repo": metadata["repo_id"],
        "model_filename": metadata["filename"],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": "cpu",
        "platform": platform.platform(),
        "git_sha": os.getenv("GITHUB_SHA"),
        "workflow_run_id": os.getenv("GITHUB_RUN_ID"),
        "selection_salt": SALT,
        "selected_board_indices": list(BOARD_INDICES),
        "selected_board_indices_sha256": BOARD_INDICES_SHA256,
        "inference_n_passes": 21,
        "readout_pass": 20,
        "C": 512,
        "k": 307,
        "ablation": "zero_without_rescale_delta_off",
        "shard_size": 8,
        "per_board": [],
        "completed_chunks": 0,
        "result": None,
    }
    for chunk_index in range(args.max_chunks):
        sub_indices = BOARD_INDICES[chunk_index * 8 : (chunk_index + 1) * 8]
        subset = inputs[torch.tensor(sub_indices, dtype=torch.long)]
        part = audit_one(
            args.checkpoint, model, subset,
            n_passes=21, c=512, keep_frac=0.60, device="cpu",
        )
        rows = _collect_rows(part, sub_indices)
        receipt["per_board"].extend(rows)
        receipt["completed_chunks"] = chunk_index + 1
        _write_receipt(args.out, receipt)
        print(
            f"{args.checkpoint} source-locked chunk {chunk_index + 1}/8 "
            f"complete: {len(receipt['per_board'])} boards", flush=True
        )

    if args.max_chunks == 8:
        if [r["board_index"] for r in receipt["per_board"]] != list(BOARD_INDICES):
            raise AssertionError("Reserved board ordering changed")
        receipt["result"] = analyze_rows(receipt["per_board"])
        receipt["status"] = "complete"
        _write_receipt(args.out, receipt)
        print(
            json.dumps({
                "checkpoint": args.checkpoint,
                "receipt": str(args.out),
                "status": receipt["status"],
                "result": receipt["result"],
            }, indent=2),
            flush=True,
        )
    else:
        receipt["status"] = "partial_debug_only"
        _write_receipt(args.out, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
