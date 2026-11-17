"""B4: compare clean-informed SVD top-k masks with true current-state online top-k.

The online arm computes its top-k selector *inside each TiedComponentLinear*
from its own perturbed pre-weight input. Clean-target masks are passed into
PassMaskSchedule only to record disagreement; they never drive online writes.

This is a zero-fit selector baseline, not a trained CI-network evaluation.
Run with --mini to test the actual recurrent execution path without downloads.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from pathlib import Path
import sys
from typing import Any, Mapping

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechinterp.bpd.adapters.trm import (
    TRMBPDAdapter,
    build_mini_trm,
    default_trm_c_per_module,
    load_pretrained_trm,
    load_wb_stall_and_train_sudoku,
)
from mechinterp.bpd.components import BPDModel, PassMaskSchedule
from mechinterp.bpd.config import AblationConfig
from mechinterp.bpd.scripts.audit_b4_v2_controls_and_ablation_split import (
    eval_rollout_per_board,
    per_board_kl,
)
from mechinterp.bpd.scripts.b4_numerics import compare_readouts


def _capture_clean(
    bpd: BPDModel,
    adapter: TRMBPDAdapter,
    batch: Any,
    n_passes: int,
) -> tuple[dict[int, torch.Tensor], dict[str, torch.Tensor]]:
    with torch.no_grad():
        with bpd.target_capture(pass_offset=0, update_means=False, detach_inputs=True):
            ro, _, _, _ = adapter.rollout(batch, n_passes, 0, None)
        module_inputs = {
            m: v.detach().cpu() for m, v in
            bpd.collect_recorded_inputs(expected_passes=n_passes).items()
        }
    return {tau: val.detach().clone() for tau, val in ro.items()}, module_inputs


@torch.no_grad()
def clean_oracle_topk_schedules(
    bpd: BPDModel,
    module_inputs: Mapping[str, torch.Tensor],
    *,
    k: int,
    device: str,
    delta_mask: torch.Tensor,
) -> dict[str, PassMaskSchedule]:
    """Calculate deterministic clean-informed masks using frozen SVD factors."""
    schedules = {}
    for m, w in bpd.wrappers.items():
        x = module_inputs[m].to(device=device, dtype=torch.float32)
        scores = torch.einsum("btpd,dc->btpc", x, w.V.detach().float()).abs()
        scores = scores * w.U.detach().float().norm(dim=-1)
        selected = scores.topk(k, dim=-1).indices
        mask = torch.zeros_like(scores).scatter_(-1, selected, 1.0)
        schedules[m] = PassMaskSchedule(comp_mask=mask, delta_mask=delta_mask)
        del x, scores, selected
    return schedules


@torch.no_grad()
def _rollout(
    bpd: BPDModel,
    adapter: TRMBPDAdapter,
    batch: Any,
    schedules: dict[str, PassMaskSchedule],
    n_passes: int,
) -> dict[int, torch.Tensor]:
    with bpd.masked_components(schedules, pass_offset=0, record_inputs=False):
        ro, _, _, _ = adapter.rollout(batch, n_passes, 0, None)
    return ro


def _disagreement_report(bpd: BPDModel, n_passes: int) -> dict[str, Any]:
    report = {}
    for m, w in bpd.wrappers.items():
        records = w.online_mask_disagreement_by_pass
        if len(records) != n_passes:
            raise AssertionError(
                f"{m}: online selector invoked {len(records)} times, expected {n_passes}"
            )
        stacked = torch.stack(records).float()  # [T, B]
        report[m] = {
            "by_pass_mean": [float(v) for v in stacked.mean(dim=1).tolist()],
            "by_pass_by_board": [[float(v) for v in row] for row in stacked.tolist()],
            "overall_mean": float(stacked.mean().item()),
            "max_by_board_over_passes": [float(v) for v in stacked.amax(dim=0).tolist()],
        }
    return report


def _pairwise_per_board(
    a: Mapping[int, torch.Tensor],
    b: Mapping[int, torch.Tensor],
) -> dict[str, Any]:
    results = {}
    for tau in sorted(a):
        if tau not in b:
            raise RuntimeError(f"Readout {tau} absent in reference")
        results[str(tau)] = per_board_kl(a[tau], b[tau])
    return results


@torch.no_grad()
def audit_one(
    name: str,
    model: torch.nn.Module,
    eval_inputs: torch.Tensor,
    *,
    n_passes: int,
    c: int,
    keep_frac: float,
    device: str,
) -> dict[str, Any]:
    base = copy.deepcopy(model).to(device).eval()
    c_map = default_trm_c_per_module(base, c=c)
    bpd = BPDModel(
        base, c_map, max_passes=n_passes,
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        init_svd=True,
    ).to(device)
    adapter = TRMBPDAdapter(bpd.target_model, readout_at="outer_step")
    batch = adapter.make_batch_with_clean_cache(eval_inputs.to(device), n_passes)
    clean_ro, clean_inputs = _capture_clean(bpd, adapter, batch, n_passes)
    with bpd.target_capture(pass_offset=0, update_means=False, detach_inputs=True):
        repeat_ro, _, _, _ = adapter.rollout(batch, n_passes, 0, None)

    k = min(c, max(1, int(round(c * keep_frac))))
    bsz = int(eval_inputs.shape[0])
    seq_len = int(bpd.target_model.total_len)
    delta_zero = torch.zeros(
        bsz, n_passes, seq_len, device=device, dtype=torch.float32
    )
    oracle_sched = clean_oracle_topk_schedules(
        bpd, clean_inputs, k=k, device=device, delta_mask=delta_zero
    )

    oracle_ro = _rollout(bpd, adapter, batch, oracle_sched, n_passes)
    online_sched = {
        m: PassMaskSchedule(
            comp_mask=sched.comp_mask,
            delta_mask=sched.delta_mask,
            online_topk=k,
        )
        for m, sched in oracle_sched.items()
    }
    online_ro = _rollout(bpd, adapter, batch, online_sched, n_passes)
    disagreement = _disagreement_report(bpd, n_passes)

    static_mask = torch.zeros(1, n_passes, 1, c, device=device)
    static_mask[..., :k] = 1.0
    static_schedule = {
        m: PassMaskSchedule(comp_mask=static_mask, delta_mask=delta_zero)
        for m in bpd.wrappers
    }
    static_ro = _rollout(bpd, adapter, batch, static_schedule, n_passes)

    # Component-path identity floor inside this evaluator: every component on,
    # residual off, same zero ablation. Its KL is the floor for the three arms.
    all_on = torch.ones(1, n_passes, 1, c, device=device)
    identity_ro = _rollout(
        bpd, adapter, batch,
        {m: PassMaskSchedule(comp_mask=all_on, delta_mask=delta_zero) for m in bpd.wrappers},
        n_passes,
    )
    readout_taus = sorted(clean_ro)
    density = {
        "static_rank_k": float(k / c),
        "online_dynamic_topk": float(k / c),
        "clean_oracle_dynamic_topk_measured": {
            m: float(sched.comp_mask.mean().item()) for m, sched in oracle_sched.items()
        },
        "svd_all_on_identity_floor": 1.0,
    }

    result = {
        "checkpoint": name,
        "eval_boards": bsz,
        "board_sha256_prefix16": [
            hashlib.sha256(bytes(map(int, row.tolist()))).hexdigest()[:16]
            for row in eval_inputs.cpu()
        ],
        "n_passes": n_passes,
        "readouts": sorted(clean_ro),
        "components_per_module": c,
        "k_keep": k,
        "reference_mode": "native target dtype",
        "oracle": "clean target pre-weight states select top-k at every module and pass",
        "online": "selector uses current perturbed pre-weight input only",
        "ablation": "zero, residual off, no sphere",
        "same_dtype_target_rerun": eval_rollout_per_board(repeat_ro, clean_ro),
        "static_rank_k": eval_rollout_per_board(static_ro, clean_ro),
        "clean_oracle_dynamic_topk": eval_rollout_per_board(oracle_ro, clean_ro),
        "online_dynamic_topk": eval_rollout_per_board(online_ro, clean_ro),
        "online_vs_oracle_rollout": eval_rollout_per_board(online_ro, oracle_ro),
        "oracle_mask_disagreement": disagreement,
        "svd_all_on_identity_floor": eval_rollout_per_board(identity_ro, clean_ro),
        "component_density": density,
        "float64_kl_and_argmax_flips": {
            "same_dtype_target_rerun": compare_readouts(repeat_ro, clean_ro, readout_taus),
            "svd_all_on_identity_floor": compare_readouts(identity_ro, clean_ro, readout_taus),
            "static_rank_k": compare_readouts(static_ro, clean_ro, readout_taus),
            "clean_oracle_dynamic_topk": compare_readouts(oracle_ro, clean_ro, readout_taus),
            "online_dynamic_topk": compare_readouts(online_ro, clean_ro, readout_taus),
            "online_vs_oracle_rollout": compare_readouts(online_ro, oracle_ro, readout_taus),
        },
        "paired_online_minus_oracle_KL_by_board": {},
        "precision_warning": "Late-pass KL may exceed identity-control precision floor.",
    }
    for tau in sorted(clean_ro):
        a = per_board_kl(online_ro[tau], clean_ro[tau])
        b = per_board_kl(oracle_ro[tau], clean_ro[tau])
        result["paired_online_minus_oracle_KL_by_board"][str(tau)] = [
            float(v - u) for v, u in zip(a, b)
        ]
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mini", action="store_true",
                    help="small synthetic recurrent-path sanity test, no pretrained data")
    ap.add_argument("--n-passes", type=int, default=63)
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--eval-boards", type=int, default=16)
    ap.add_argument("--checkpoints", nargs="+", default=["gaoxin492", "yagizdevre"],
                    choices=["gaoxin492", "yagizdevre"],
                    help="Audit only selected pretrained checkpoints (ignored in --mini).")
    ap.add_argument("--precision", choices=["native", "fp32"], default="native",
                    help="native: BF16 on CUDA, FP32 on CPU (registered). fp32: original FP32 "
                         "checkpoint weights and FP32 compute on any device (separate regime).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if not (0 < args.keep_frac <= 1):
        ap.error("--keep-frac must be in (0, 1]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        ap.error("CUDA unavailable. --device cpu is an exploratory FP32 regime.")
    if not 1 <= args.eval_boards <= 16:
        ap.error("--eval-boards must be 1..16")
    if args.n_passes < 1:
        ap.error("--n-passes must be positive")

    torch.manual_seed(113)
    results = {
        "audit": "B4_v2_current_state_online_svd_topk",
        "mode": "mini_cpu_software_test" if args.mini else "pretrained_16_board_descriptive",
        "torch_version": torch.__version__,
        "device": args.device,
        "cuda_device": (torch.cuda.get_device_name() if args.device.startswith("cuda") else None),
        "precision": args.precision,
        "tf32_matmul_allowed": (
            bool(torch.backends.cuda.matmul.allow_tf32) if args.device.startswith("cuda") else None
        ),
        "metrics": {},
    }
    if args.mini:
        n_passes = 6
        inputs = torch.randint(1, 11, (min(3, args.eval_boards), 16), dtype=torch.long)
        model = build_mini_trm(
            hidden_size=32, num_heads=4, l_layers=2, h_cycles=1,
            l_cycles=2, puzzle_emb_len=4, seq_len=16, seed=113,
        ).to(device=args.device, dtype=torch.float32).eval()
        results["metrics"]["mini"] = audit_one(
            "mini", model, inputs, n_passes=n_passes,
            c=8, keep_frac=args.keep_frac, device=args.device,
        )
    else:
        wb = load_wb_stall_and_train_sudoku(None, n_train=128, device="cpu")
        inputs = wb["eval_inputs"][:args.eval_boards]
        for ckpt in args.checkpoints:
            dtype = (
                torch.bfloat16
                if args.device.startswith("cuda") and args.precision == "native"
                else torch.float32
            )
            model = load_pretrained_trm(ckpt, dtype=dtype, verify_sha256=True)
            results["metrics"][ckpt] = audit_one(
                ckpt, model, inputs, n_passes=args.n_passes,
                c=512, keep_frac=args.keep_frac, device=args.device,
            )
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            # Persist completed checkpoints even if a later model hits
            # host memory/time limits. The final write below also handles --mini.
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    text = json.dumps(results, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text if args.mini else json.dumps(
        {"audit": results["audit"], "mode": results["mode"],
         "checkpoints": list(results["metrics"])}, indent=2
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
