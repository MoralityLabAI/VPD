"""Fast (~10s) zero-fit controls requested before the v3 toy spec:

  1. Control A: Target float32 matmul (F.linear(x.float(), W.float()).to(bf16)) vs native
     bfloat16 target across the 16 eval boards, reporting full per-board KL at passes 20, 41, 62.
  2. Control B: All 512 SVD-start components ON (comp_mask=1, delta_mask=0 and delta_mask=1)
     plus trained (comp=1, delta=1), with full per-board KL at passes 20, 41, 62.
  3. 2x2 factorial split of (zero vs pass_mean) x (sphere_rescale False vs True) on the
     60% dynamic SVD basis at passes 20, 41, and 62 (mean + median + per-board at pass 20 & 62).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechinterp.bpd.adapters.trm import (
    TRMBPDAdapter,
    default_trm_c_per_module,
    load_pretrained_trm,
    load_wb_stall_and_train_sudoku,
)
from mechinterp.bpd.components import (
    AblationConfig,
    BPDModel,
    PassMaskSchedule,
    RolloutTrajectory,
)


def per_board_kl(pred_logits: torch.Tensor, target_logits: torch.Tensor) -> list[float]:
    """Return per-board token-averaged KL(softmax(target) || softmax(pred)) as a list of floats."""
    pred_lp = F.log_softmax(pred_logits.float(), dim=-1)
    targ_lp = F.log_softmax(target_logits.float(), dim=-1)
    targ_p = targ_lp.exp()
    # [B, P] -> mean over P -> [B]
    kl_b = (targ_p * (targ_lp - pred_lp)).sum(dim=-1).mean(dim=-1)
    return [float(x) for x in kl_b.detach().cpu().tolist()]


def summarize_per_board(vals: list[float]) -> dict[str, Any]:
    t = torch.tensor(vals, dtype=torch.float64)
    s_vals, s_idx = torch.sort(t, descending=True)
    return {
        "mean": float(t.mean().item()),
        "median": float(t.median().item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "top3_boards": [
            {"board_idx": int(s_idx[i].item()), "kl": float(s_vals[i].item())}
            for i in range(min(3, len(vals)))
        ],
        "per_board": [round(v, 6) for v in vals],
    }


class _Float32AccumLinear(nn.Module):
    """Exact target matrix W multiplied in float32 and cast back to input dtype."""

    def __init__(self, lin: nn.Linear) -> None:
        super().__init__()
        self.w_f32 = lin.weight.detach().float()
        self.b_f32 = lin.bias.detach().float() if lin.bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x.float(), self.w_f32, self.b_f32)
        return out.to(x.dtype)


@torch.no_grad()
def eval_rollout_per_board(
    readouts: dict[int, torch.Tensor],
    clean_readouts: dict[int, torch.Tensor],
    *,
    include_per_board_passes: tuple[int, ...] = (20, 41, 62),
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pass_means: list[float] = []
    for tau in sorted(clean_readouts.keys()):
        pb = per_board_kl(readouts[tau], clean_readouts[tau])
        summary = summarize_per_board(pb)
        pass_means.append(summary["mean"])
        if tau in include_per_board_passes:
            out[f"pass_{tau}"] = summary
        else:
            out[f"pass_{tau}_mean"] = summary["mean"]
    out["mean_across_readout_passes"] = sum(pass_means) / max(len(pass_means), 1)
    return out


@torch.no_grad()
def run_controls_for_checkpoint(
    ckpt_name: str,
    saved_factors: dict[str, dict[str, torch.Tensor]],
    train_cal_inputs: torch.Tensor,
    eval_inputs: torch.Tensor,
    *,
    n_passes: int = 63,
    keep_frac: float = 0.60,
    device: str = "cuda",
) -> dict[str, Any]:
    dt = torch.bfloat16 if device.startswith("cuda") else torch.float32
    base_model = load_pretrained_trm(ckpt_name, dtype=dt, verify_sha256=True)

    # Clean native bfloat16 rollout + captured module inputs
    clean_work = copy.deepcopy(base_model).to(device).eval()
    clean_adapter = TRMBPDAdapter(clean_work, readout_at="outer_step")
    c_map = default_trm_c_per_module(clean_work, c=512)
    bpd_capture = BPDModel(
        clean_work,
        c_map,
        max_passes=n_passes,
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        init_svd=True,
    ).to(device)

    eval_batch = clean_adapter.make_batch_with_clean_cache(eval_inputs.to(device), n_passes)
    with bpd_capture.target_capture(pass_offset=0, update_means=False, detach_inputs=True):
        clean_ro, _, _, _ = clean_adapter.rollout(eval_batch, n_passes, 0, None)
    c_mod_inputs = {
        k: v.detach().cpu()
        for k, v in bpd_capture.collect_recorded_inputs(expected_passes=n_passes).items()
    }
    clean_readouts = {k: v.detach().clone() for k, v in clean_ro.items()}
    del bpd_capture, clean_work
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Control 1a: Target vs Target under float32 matmul accumulation -> cast back to bf16
    f32_matmul_model = copy.deepcopy(base_model).to(device).eval()
    for blk in f32_matmul_model.layers:
        blk.o_proj = _Float32AccumLinear(blk.o_proj)  # type: ignore[assignment]
        blk.down_proj = _Float32AccumLinear(blk.down_proj)  # type: ignore[assignment]
    f32_matmul_adapter = TRMBPDAdapter(f32_matmul_model, readout_at="outer_step")
    ro_f32_mm, _, _, _ = f32_matmul_adapter.rollout(eval_batch, n_passes, 0, None)
    ctrl_target_f32_matmul = eval_rollout_per_board(ro_f32_mm, clean_readouts)
    del f32_matmul_model, ro_f32_mm
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Control 1b: Full model in float32 vs native bfloat16
    full_f32_model = load_pretrained_trm(ckpt_name, dtype=torch.float32, verify_sha256=False).to(device).eval()
    full_f32_adapter = TRMBPDAdapter(full_f32_model, readout_at="outer_step")
    eval_batch_f32 = full_f32_adapter.make_batch_with_clean_cache(eval_inputs.to(device), n_passes)
    ro_full_f32, _, _, _ = full_f32_adapter.rollout(eval_batch_f32, n_passes, 0, None)
    ctrl_full_float32_vs_bf16 = eval_rollout_per_board(ro_full_f32, clean_readouts)
    del full_f32_model, eval_batch_f32, ro_full_f32
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    bsz = int(eval_inputs.shape[0])
    seq_len = int(base_model.total_len)
    ones_comp = torch.ones((1, 1, 1, 512), device=device, dtype=torch.float32).expand(
        bsz, n_passes, seq_len, 512
    )
    ones_delta = torch.ones((1, 1, 1), device=device, dtype=torch.float32).expand(
        bsz, n_passes, seq_len
    )
    zeros_delta = torch.zeros((1, 1, 1), device=device, dtype=torch.float32).expand(
        bsz, n_passes, seq_len
    )

    # Control 2a: All 512 SVD-start components ON (no remainder, exact SVD of W_target)
    bpd_svd = BPDModel(
        copy.deepcopy(base_model).to(device).eval(),
        c_map,
        max_passes=n_passes,
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        init_svd=True,
    ).to(device)
    svd_adapter = TRMBPDAdapter(bpd_svd.target_model, readout_at="outer_step")
    svd_residuals = {
        m: float((w.weight_delta().norm() / w.W_target.float().norm()).item())
        for m, w in bpd_svd.wrappers.items()
    }
    sched_svd_all_on = {
        m: PassMaskSchedule(comp_mask=ones_comp, delta_mask=zeros_delta)
        for m in bpd_svd.module_order
    }
    with bpd_svd.masked_components(sched_svd_all_on, pass_offset=0, record_inputs=False):
        ro_svd_all, _, _, _ = svd_adapter.rollout(eval_batch, n_passes, 0, None)
    ctrl_svd_all_on = eval_rollout_per_board(ro_svd_all, clean_readouts)
    del ro_svd_all

    # Control 2b: Trained factors with comp=1, delta=1 (per-board breakdown of the 0.37 / 1.53 drift)
    bpd_tr = BPDModel(
        copy.deepcopy(base_model).to(device).eval(),
        c_map,
        max_passes=n_passes,
        ablation=AblationConfig(mode="pass_mean", sphere_rescale=True),
        init_svd=False,
    ).to(device)
    for m, w in bpd_tr.wrappers.items():
        w.V.data.copy_(saved_factors[m]["V"].to(device=device, dtype=torch.float32))
        w.U.data.copy_(saved_factors[m]["U"].to(device=device, dtype=torch.float32))
    tr_adapter = TRMBPDAdapter(bpd_tr.target_model, readout_at="outer_step")
    cal_b = tr_adapter.make_batch_with_clean_cache(train_cal_inputs.to(device), n_passes)
    with bpd_tr.target_capture(pass_offset=0, update_means=True, detach_inputs=True):
        tr_adapter.rollout(cal_b, n_passes, 0, None)
    for w in bpd_tr.wrappers.values():
        w.recorded_inputs.clear()
    del cal_b
    sched_tr_c1_d1 = {
        m: PassMaskSchedule(comp_mask=ones_comp, delta_mask=ones_delta)
        for m in bpd_tr.module_order
    }
    with bpd_tr.masked_components(sched_tr_c1_d1, pass_offset=0, record_inputs=False):
        ro_tr_c1_d1, _, _, _ = tr_adapter.rollout(eval_batch, n_passes, 0, None)
    ctrl_trained_c1_d1 = eval_rollout_per_board(ro_tr_c1_d1, clean_readouts)
    del bpd_tr, ro_tr_c1_d1
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # 3. 2x2 split of (zero vs pass_mean) x (sphere_rescale False vs True) on dynamic top-60% SVD
    k_keep = max(1, min(512, int(round(keep_frac * 512))))
    # Calibrate pass_mean buffers on bpd_svd once upfront
    bpd_svd_cal_b = svd_adapter.make_batch_with_clean_cache(train_cal_inputs.to(device), n_passes)
    with bpd_svd.target_capture(pass_offset=0, update_means=True, detach_inputs=True):
        svd_adapter.rollout(bpd_svd_cal_b, n_passes, 0, None)
    for w in bpd_svd.wrappers.values():
        w.recorded_inputs.clear()
    del bpd_svd_cal_b
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    svd_topk_schedules: dict[str, PassMaskSchedule] = {}
    for m, w in bpd_svd.wrappers.items():
        x_btp = c_mod_inputs[m].float().to(device)
        u_norms = w.U.detach().float().norm(dim=-1)
        scores = torch.einsum("btpd,dc->btpc", x_btp, w.V.detach().float()).abs() * u_norms
        del x_btp
        topk_idx = scores.topk(k_keep, dim=-1).indices
        del scores
        dyn_mask = torch.zeros((bsz, n_passes, seq_len, 512), device=device, dtype=torch.float32)
        dyn_mask.scatter_(-1, topk_idx, 1.0)
        del topk_idx
        svd_topk_schedules[m] = PassMaskSchedule(comp_mask=dyn_mask, delta_mask=zeros_delta)

    ablation_factorial: dict[str, Any] = {}
    for mode_val in ("zero", "pass_mean"):
        for rescale_val in (False, True):
            key = f"mode_{mode_val}__sphere_rescale_{str(rescale_val).lower()}"
            if mode_val == "zero" and rescale_val:
                # components.py line 369 gates sphere_rescale on mode=="pass_mean", so evaluate
                # genuine zero-ablation + sphere_rescale by setting mode="pass_mean" with zeroed mean_comp_acts!
                saved_means = {m: w.mean_comp_acts.clone() for m, w in bpd_svd.wrappers.items()}
                for w in bpd_svd.wrappers.values():
                    w.ablation = AblationConfig(mode="pass_mean", sphere_rescale=True)
                    w.mean_comp_acts.zero_()
                with bpd_svd.masked_components(svd_topk_schedules, pass_offset=0, record_inputs=False):
                    ro_ab, _, _, _ = svd_adapter.rollout(eval_batch, n_passes, 0, None)
                for m, w in bpd_svd.wrappers.items():
                    w.mean_comp_acts.copy_(saved_means[m])
            else:
                for w in bpd_svd.wrappers.values():
                    w.ablation = AblationConfig(mode=mode_val, sphere_rescale=rescale_val)  # type: ignore[arg-type]
                with bpd_svd.masked_components(svd_topk_schedules, pass_offset=0, record_inputs=False):
                    ro_ab, _, _, _ = svd_adapter.rollout(eval_batch, n_passes, 0, None)
            ablation_factorial[key] = eval_rollout_per_board(ro_ab, clean_readouts)
            del ro_ab

    return {
        "checkpoint": ckpt_name,
        "svd_start_relative_residual_fro": svd_residuals,
        "control_1a_target_f32_linear_vs_bf16_target": ctrl_target_f32_matmul,
        "control_1b_full_f32_model_vs_bf16_target": ctrl_full_float32_vs_bf16,
        "control_2a_svd_start_all_512_on": ctrl_svd_all_on,
        "control_2b_trained_all_512_plus_delta_on": ctrl_trained_c1_d1,
        "svd_top60_ablation_2x2_factorial": ablation_factorial,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--weights",
        type=Path,
        default=Path("mechinterp/bpd/results/v2/bpd_b4_v2_stage4_weights.pt"),
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.45)

    saved = torch.load(args.weights, map_location="cpu", weights_only=True)
    wb = load_wb_stall_and_train_sudoku(None, n_train=128, device="cpu")
    train_cal = wb["train_inputs"][:32]
    eval_slice = wb["eval_inputs"][:16]

    ref_res = run_controls_for_checkpoint(
        "gaoxin492", saved["bpd_ref"], train_cal, eval_slice, device=args.device
    )
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    tgt_res = run_controls_for_checkpoint(
        "yagizdevre", saved["bpd_tgt"], train_cal, eval_slice, device=args.device
    )

    report = {
        "audit": "B4_v2_numerical_floor_controls_and_ablation_2x2_split",
        "eval_boards": int(eval_slice.shape[0]),
        "n_passes": 63,
        "code_confirmed_oracle_gate_steering": {
            "file": "mechinterp/bpd/trainer.py",
            "lines": "571-576",
            "mechanism": "BPDTrainer.evaluate() computes ci_out = self.ci_net.forward_trajectory(clean_traj) from the clean target trajectory at all 63 passes rather than evaluating ci_net closed-loop on the masked rollout's own hidden states.",
        },
        "gaoxin492_ref": ref_res,
        "yagizdevre_tgt": tgt_res,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
