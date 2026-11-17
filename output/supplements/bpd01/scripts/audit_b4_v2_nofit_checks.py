"""Zero-fit, zero-CI forward-only diagnostics on saved B4 v2 (V, U) factors.

Runs in ~5 seconds under torch.no_grad() and exits cleanly:
  1. Exact relative Frobenius remainder ||W_target - (V U)^T||_F / ||W_target||_F against the
     pretrained gaoxin492 & yagizdevre matrices.
  2. All-components-on separation test (comp_mask = 1) comparing:
       (a) delta_mask = 1 (exact target sanity check),
       (b) delta_mask = 0 with pass-mean mu_d (Fable's cancellation vs inert-remainder test),
       (c) pure (V U)^T with delta_contrib = 0 (no mu_d pass-mean bias).
  3. Empirical input variance along the shared read direction vs typical isotropic variance
     tr(Sigma_x) / d_in, alignment of the shared read direction with mean input E[x], and
     clean-rollout Var(x @ v_c) / E[(x @ v_c)^2] inside vs outside the shared read cluster.
  4. Gate-free SVD baseline at 60% active density (top 307/512 components per token & pass
     by write norm |x @ v_c| * ||u_c||_2, evaluated both raw/zero-masked and pass-mean masked).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import torch
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
from mechinterp.bpd.losses import kl_divergence_logits, trajectory_recon_loss


@torch.no_grad()
def evaluate_schedule_kl(
    bpd_mod: BPDModel,
    adapter: TRMBPDAdapter,
    eval_batch: Any,
    clean_traj: RolloutTrajectory,
    schedules: dict[str, PassMaskSchedule],
    n_passes: int,
) -> dict[str, float]:
    with bpd_mod.masked_components(schedules, pass_offset=0, record_inputs=False):
        ro, bs, vis, prm = adapter.rollout(eval_batch, n_passes, 0, None)
    pred_traj = RolloutTrajectory(
        readouts=ro,
        boundary_states=bs,
        module_inputs={},
        visible_states=vis,
        prompt_features=prm,
    )
    recon, per_pass = trajectory_recon_loss(
        pred_traj,
        clean_traj,
        match_inner_boundary_states=False,
        pass_null_losses=None,
        hinge_kappa=0.0,
    )
    f_tau = max(clean_traj.readouts.keys())
    final_kl = float(kl_divergence_logits(pred_traj.readouts[f_tau], clean_traj.readouts[f_tau]).item())
    out = {"recon_loss": float(recon.item()), "final_kl": final_kl}
    for tau, val in sorted(per_pass.items()):
        out[f"pass_{tau}_kl"] = float(val)
    return out


@torch.no_grad()
def audit_checkpoint(
    ckpt_name: str,
    factors_key: str,
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
    work_model = copy.deepcopy(base_model).to(device).eval()
    adapter = TRMBPDAdapter(work_model, readout_at="outer_step")
    c_map = default_trm_c_per_module(work_model, c=512)

    bpd_mod = BPDModel(
        work_model,
        c_map,
        max_passes=n_passes,
        ablation=AblationConfig(mode="pass_mean", sphere_rescale=True),
        init_svd=True,
    ).to(device)

    # Record exact SVD-start directions before loading trained (V, U)
    svd_init_dirs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for m, w in bpd_mod.wrappers.items():
        v0 = F.normalize(w.V.detach().float(), dim=0)
        u0 = F.normalize(w.U.detach().float(), dim=1)
        svd_init_dirs[m] = (v0, u0)
        w.V.data.copy_(saved_factors[m]["V"].to(device=device, dtype=torch.float32))
        w.U.data.copy_(saved_factors[m]["U"].to(device=device, dtype=torch.float32))

    # 1. Exact relative residual norms vs true W_target
    weight_residuals: dict[str, Any] = {}
    for m, w in bpd_mod.wrappers.items():
        w_true = w.W_target.detach().float()
        w_recon = (w.V.detach().float() @ w.U.detach().float()).T
        delta_w = w_true - w_recon
        v_normed = F.normalize(w.V.detach().float(), dim=0)
        v0, _ = svd_init_dirs[m]
        best_svd_cos = (v_normed.T @ v0).abs().max(dim=1).values
        weight_residuals[m] = {
            "w_target_fro": float(w_true.norm().item()),
            "w_recon_fro": float(w_recon.norm().item()),
            "delta_w_fro": float(delta_w.norm().item()),
            "delta_over_target_fro": float((delta_w.norm() / w_true.norm()).item()),
            "delta_over_recon_fro": float((delta_w.norm() / w_recon.norm()).item()),
            "entrywise_mse": float(delta_w.pow(2).mean().item()),
            "median_best_abs_cos_to_any_svd_read": float(best_svd_cos.median().item()),
        }

    # Calibrate pass-conditional means on train_inputs[:32], matching run_trm_bpd.py line 239
    cal_batch = adapter.make_batch_with_clean_cache(train_cal_inputs.to(device), n_passes)
    with bpd_mod.target_capture(pass_offset=0, update_means=True, detach_inputs=True):
        adapter.rollout(cal_batch, n_passes, 0, None)
    for w in bpd_mod.wrappers.values():
        w.recorded_inputs.clear()
    del cal_batch
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Clean trajectory on eval_inputs[:16], matching run_trm_bpd.py line 261
    eval_batch = adapter.make_batch_with_clean_cache(eval_inputs.to(device), n_passes)
    with bpd_mod.target_capture(pass_offset=0, update_means=False, detach_inputs=True):
        c_ro, c_bs, c_vis, c_prm = adapter.rollout(eval_batch, n_passes, 0, None)
    c_mod_inputs = {
        k: v.detach().cpu()
        for k, v in bpd_mod.collect_recorded_inputs(expected_passes=n_passes).items()
    }
    clean_traj = RolloutTrajectory(
        readouts=c_ro,
        boundary_states={k: v.cpu() for k, v in c_bs.items()},
        module_inputs={},
        visible_states={k: v.cpu() for k, v in c_vis.items()},
        prompt_features=c_prm.cpu() if c_prm is not None else None,
    )
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    bsz = int(eval_inputs.shape[0])
    seq_len = int(work_model.total_len)

    # 2. All-components-on separation test (no CI needed)
    ones_comp_1 = torch.ones((1, 1, 1, 512), device=device, dtype=torch.float32).expand(
        bsz, n_passes, seq_len, 512
    )
    ones_delta = torch.ones((1, 1, 1), device=device, dtype=torch.float32).expand(
        bsz, n_passes, seq_len
    )
    zeros_delta = torch.zeros((1, 1, 1), device=device, dtype=torch.float32).expand(
        bsz, n_passes, seq_len
    )

    sched_all_on_delta_on = {
        m: PassMaskSchedule(comp_mask=ones_comp_1, delta_mask=ones_delta)
        for m in bpd_mod.module_order
    }
    sched_all_on_delta_mean = {
        m: PassMaskSchedule(comp_mask=ones_comp_1, delta_mask=zeros_delta)
        for m in bpd_mod.module_order
    }
    kl_all_on_delta_on = evaluate_schedule_kl(
        bpd_mod, adapter, eval_batch, clean_traj, sched_all_on_delta_on, n_passes
    )
    kl_all_on_delta_mean = evaluate_schedule_kl(
        bpd_mod, adapter, eval_batch, clean_traj, sched_all_on_delta_mean, n_passes
    )

    # Also test pure (V U)^T with delta_contrib = 0 (uncalibrated / zero ablation for delta)
    saved_ablation_modes = {}
    for m, w in bpd_mod.wrappers.items():
        saved_ablation_modes[m] = w.ablation
        w.ablation = AblationConfig(mode="zero", sphere_rescale=False)
    kl_pure_vu_no_delta_bias = evaluate_schedule_kl(
        bpd_mod, adapter, eval_batch, clean_traj, sched_all_on_delta_mean, n_passes
    )
    for m, w in bpd_mod.wrappers.items():
        w.ablation = saved_ablation_modes[m]

    # 3. Empirical input variance along shared read direction vs typical direction + Var(x @ v_c)
    variance_diagnostics: dict[str, Any] = {}
    for m, w in bpd_mod.wrappers.items():
        # Compute on CPU from already-staged c_mod_inputs[m] [B, T, P, d_in]
        x_btp = c_mod_inputs[m].float()
        x_all = x_btp.reshape(-1, w.d_in)
        mu_x = x_all.mean(dim=0)
        x_centered = x_all - mu_x
        total_var = float(x_centered.pow(2).sum(dim=-1).mean().item())
        typical_dir_var = total_var / float(w.d_in)

        v_mat = w.V.detach().cpu().float()  # [d_in, C]
        v_unit = F.normalize(v_mat, dim=0)
        # Principal direction of normalized read columns
        u_top, s_top, _ = torch.linalg.svd(v_unit, full_matrices=False)
        d_shared = u_top[:, 0]  # unit vector in R^{d_in}
        cos_to_shared = (v_unit.T @ d_shared).abs()  # [C]
        shared_mask = cos_to_shared >= 0.5

        proj_shared_centered = x_centered @ d_shared
        var_along_shared = float(proj_shared_centered.pow(2).mean().item())
        mean_align_cos = float(
            F.cosine_similarity(d_shared.unsqueeze(0), mu_x.unsqueeze(0)).abs().item()
        )

        # Per-component activation variance vs second moment across all (B, T, P)
        acts_btp = torch.einsum("btpd,dc->btpc", x_btp, v_mat)  # [B, T, P, C]
        acts = acts_btp.reshape(-1, 512)
        act_var = acts.var(dim=0, unbiased=False)
        act_m2 = acts.pow(2).mean(dim=0).clamp_min(1e-12)
        var_to_m2 = act_var / act_m2  # 0 = purely constant activation, 1 = zero-mean activation

        # Pass-conditional variance (pass_mean subtracts E[x_tau @ v_c] per pass tau)
        acts_pass_centered = acts_btp - acts_btp.mean(dim=(0, 2), keepdim=True)
        pass_cond_var = acts_pass_centered.pow(2).mean(dim=(0, 1, 2))
        pass_cond_var_ratio = pass_cond_var / act_m2

        variance_diagnostics[m] = {
            "frac_reads_within_cos_0_5_of_shared": float(shared_mask.float().mean().item()),
            "shared_singular_energy_frac": float((s_top[0].pow(2) / s_top.pow(2).sum()).item()),
            "typical_isotropic_dir_var": typical_dir_var,
            "input_var_along_shared_dir": var_along_shared,
            "shared_over_typical_var_ratio": var_along_shared / max(typical_dir_var, 1e-12),
            "abs_cos_shared_dir_with_global_mean_input": mean_align_cos,
            "mean_input_norm_over_rms_x": float(
                (mu_x.norm() / x_all.pow(2).sum(dim=-1).mean().sqrt()).item()
            ),
            "component_var_over_m2_median_all": float(var_to_m2.median().item()),
            "component_var_over_m2_median_shared_cluster": (
                float(var_to_m2[shared_mask].median().item()) if bool(shared_mask.any()) else None
            ),
            "component_var_over_m2_median_outside_shared": (
                float(var_to_m2[~shared_mask].median().item()) if bool((~shared_mask).any()) else None
            ),
            "pass_cond_var_over_m2_median_shared_cluster": (
                float(pass_cond_var_ratio[shared_mask].median().item())
                if bool(shared_mask.any())
                else None
            ),
            "pass_cond_var_over_m2_median_outside_shared": (
                float(pass_cond_var_ratio[~shared_mask].median().item())
                if bool((~shared_mask).any())
                else None
            ),
        }

        del x_all, x_centered, acts, x_btp, acts_btp, acts_pass_centered
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    del bpd_mod, work_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # 4. Gate-free SVD baseline at 60% density (top-k per (b, t, p) by write norm |x @ v_c| * ||u_c||_2)
    bpd_svd = BPDModel(
        copy.deepcopy(base_model).to(device).eval(),
        c_map,
        max_passes=n_passes,
        ablation=AblationConfig(mode="zero", sphere_rescale=False),
        init_svd=True,
    ).to(device)
    svd_adapter = TRMBPDAdapter(bpd_svd.target_model, readout_at="outer_step")

    k_keep = max(1, min(512, int(round(keep_frac * 512))))
    svd_topk_schedules: dict[str, PassMaskSchedule] = {}
    stat_1 = torch.zeros((1, 1, 1, 512), device=device, dtype=torch.float32)
    stat_1[..., :k_keep] = 1.0
    stat_exp = stat_1.expand(bsz, n_passes, seq_len, 512)
    svd_static_schedules: dict[str, PassMaskSchedule] = {
        m: PassMaskSchedule(comp_mask=stat_exp, delta_mask=zeros_delta)
        for m in bpd_svd.wrappers
    }
    for m, w in bpd_svd.wrappers.items():
        x_btp = c_mod_inputs[m].float().to(device)  # [B, T, P, d_in]
        u_norms = w.U.detach().float().norm(dim=-1)  # [C]
        scores = torch.einsum("btpd,dc->btpc", x_btp, w.V.detach().float()).abs() * u_norms
        del x_btp
        topk_idx = scores.topk(k_keep, dim=-1).indices
        del scores
        dyn_mask = torch.zeros((bsz, n_passes, seq_len, 512), device=device, dtype=torch.float32)
        dyn_mask.scatter_(-1, topk_idx, 1.0)
        del topk_idx
        svd_topk_schedules[m] = PassMaskSchedule(comp_mask=dyn_mask, delta_mask=zeros_delta)

    kl_svd_dynamic_top60_zero = evaluate_schedule_kl(
        bpd_svd, svd_adapter, eval_batch, clean_traj, svd_topk_schedules, n_passes
    )
    kl_svd_static_top60_zero = evaluate_schedule_kl(
        bpd_svd, svd_adapter, eval_batch, clean_traj, svd_static_schedules, n_passes
    )

    # Also evaluate dynamic top-60% SVD with pass_mean + sphere_rescale (same ablation mode as BPD)
    for w in bpd_svd.wrappers.values():
        w.ablation = AblationConfig(mode="pass_mean", sphere_rescale=True)
    cal_batch_svd = svd_adapter.make_batch_with_clean_cache(train_cal_inputs.to(device), n_passes)
    with bpd_svd.target_capture(pass_offset=0, update_means=True, detach_inputs=True):
        svd_adapter.rollout(cal_batch_svd, n_passes, 0, None)
    for w in bpd_svd.wrappers.values():
        w.recorded_inputs.clear()
    del cal_batch_svd
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    kl_svd_dynamic_top60_pass_mean = evaluate_schedule_kl(
        bpd_svd, svd_adapter, eval_batch, clean_traj, svd_topk_schedules, n_passes
    )

    return {
        "checkpoint": ckpt_name,
        "factors_key": factors_key,
        "exact_weight_residuals": weight_residuals,
        "all_components_on_separation_test": {
            "comp_1_delta_1_sanity": kl_all_on_delta_on,
            "comp_1_delta_0_pass_mean_fable_test": kl_all_on_delta_mean,
            "comp_1_pure_vu_no_delta_bias": kl_pure_vu_no_delta_bias,
        },
        "shared_direction_and_activation_variance": variance_diagnostics,
        "gate_free_svd_60pct_baselines": {
            "k_keep_of_512": k_keep,
            "dynamic_top60_zero_ablation": kl_svd_dynamic_top60_zero,
            "dynamic_top60_pass_mean_ablation": kl_svd_dynamic_top60_pass_mean,
            "static_rank307_svd_truncation": kl_svd_static_top60_zero,
        },
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

    ref_res = audit_checkpoint(
        "gaoxin492",
        "bpd_ref",
        saved["bpd_ref"],
        train_cal,
        eval_slice,
        device=args.device,
    )
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    tgt_res = audit_checkpoint(
        "yagizdevre",
        "bpd_tgt",
        saved["bpd_tgt"],
        train_cal,
        eval_slice,
        device=args.device,
    )

    report = {
        "audit": "B4_v2_zero_fit_zero_ci_diagnostics",
        "weights_path": args.weights.as_posix(),
        "eval_boards": int(eval_slice.shape[0]),
        "n_passes": 63,
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
