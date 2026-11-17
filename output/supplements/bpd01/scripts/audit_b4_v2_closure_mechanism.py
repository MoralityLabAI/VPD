"""BPD-01 Experiment A supplement: locate the source of the late-pass identity failure.

The registered closure controls (``audit_b4_v2_controls_and_ablation_split.py``)
show that every algebraic identity path diverges from the native CUDA BF16
target on a few boards by pass 62. Three explanations are separated here on the
same 16 WB-Stall evaluation boards and 63 invocations:

1. Implementation mismatch. Teacher-forced, per-invocation output error of each
   identity path on the clean target's own module inputs, in BF16 ulps and as a
   bit-mismatch rate, next to the arithmetically exact FP32-linear path.
2. Arithmetic precision. Identity controls repeated in an FP32 regime (BF16-rounded
   and original checkpoint weights) and against an FP64 reference whose RMSNorm
   also runs in FP64.
3. Recurrent instability. Seeded relative perturbations of size eps injected into one
   ``layers.0.o_proj`` invocation (or every invocation) of the FP64 model, with
   per-invocation state divergence and readout KL; FP64 rollouts that round only the
   four decomposed sites' outputs, or only the recurrent state, to BF16; and the
   cuBLAS BF16 reduced-precision-reduction flag.

Decision rules (fixed before the 16-board execution; the implementation rule was
revised after a 2-board smoke run showed that raw per-element ulp error is unbounded
at near-zero outputs, where even the correctly rounded FP64 result differs from the
native BF16 matmul by hundreds of ulps):
- implementation mismatch is flagged if any component identity path, on any module,
  has teacher-forced error above 2 effective ulps (ulp of the element, floored at the
  ulp of the token's output RMS) or a mean relative error versus the FP64 product
  more than 10% above the native BF16 matmul's own;
- a regime "closes" at a readout only if every identity arm has per-board float64
  KL <= 1e-5 on all 16 boards (the registered reserved-64 identity tolerance);
- the FP64 perturbation sweep reports, per eps, how many boards exceed KL 1e-3 at
  each readout; a board is rounding-sensitive if a BF16-unit-roundoff perturbation
  (eps = 2^-9) moves it above 1e-3.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechinterp.bpd.adapters.trm import (
    TRMBPDAdapter,
    default_trm_c_per_module,
    load_pretrained_trm,
    load_wb_stall_and_train_sudoku,
)
from mechinterp.bpd.components import AblationConfig, BPDModel, PassMaskSchedule
from mechinterp.bpd.scripts.audit_b4_v2_controls_and_ablation_split import (
    EXPECTED_WEIGHTS_SHA256,
    _Float32AccumLinear,
)
from mechinterp.bpd.scripts.b4_numerics import (
    OutputPerturbation,
    bf16_ulp,
    compare_readouts,
    divergence_curve,
    exact_fp64_rmsnorm,
    readout_logits64,
)

READOUTS = (20, 41, 62)
N_PASSES = 63
CLOSURE_KL_TOL = 1e-5
SENSITIVE_KL = 1e-3
PERTURB_EPS = (1e-12, 1e-9, 2.0 ** -24, 1e-5, 2.0 ** -9)
PERTURB_AT_CALL = (0, 21, -1)  # -1: every invocation
PERTURB_SEEDS = (0, 1)
IDENTITY_ARMS = ("svd_all_on_delta_off", "svd_all_on_delta_on", "trained_all_on_delta_on")


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def run_rollout(
    adapter: TRMBPDAdapter,
    inputs: torch.Tensor,
    n_passes: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Rollout returning float64 readout logits and every boundary state."""
    _ro, states, _vis, _x = adapter.rollout(inputs, n_passes, 0, None)
    readouts = {
        tau: readout_logits64(adapter.model, states[tau])
        for tau in range(n_passes)
        if adapter.is_readout_pass(tau)
    }
    return readouts, states


def _all_on(bpd: BPDModel, delta: float, device: str) -> dict[str, PassMaskSchedule]:
    return {
        m: PassMaskSchedule(
            comp_mask=torch.ones(1, N_PASSES, 1, bpd.c_per_module[m], device=device),
            delta_mask=torch.full((1, N_PASSES, 1), float(delta), device=device),
        )
        for m in bpd.module_order
    }


@torch.no_grad()
def run_component_rollout(
    bpd: BPDModel,
    adapter: TRMBPDAdapter,
    inputs: torch.Tensor,
    delta: float,
    device: str,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    with bpd.masked_components(_all_on(bpd, delta, device), pass_offset=0, record_inputs=False):
        return run_rollout(adapter, inputs, N_PASSES)


def _arm_record(
    readouts: Mapping[int, torch.Tensor],
    states: Mapping[int, torch.Tensor],
    ref_readouts: Mapping[int, torch.Tensor],
    ref_states: Mapping[int, torch.Tensor],
) -> dict[str, Any]:
    rec = compare_readouts(readouts, ref_readouts, READOUTS)
    rec["state_divergence"] = divergence_curve(states, ref_states, N_PASSES)
    return rec


def _build_bpd(
    model: torch.nn.Module,
    device: str,
    *,
    trained: Mapping[str, Mapping[str, torch.Tensor]] | None,
    cal_inputs: torch.Tensor,
) -> tuple[BPDModel, TRMBPDAdapter]:
    """SVD-start (zero, no sphere) or trained-factor (pass-mean + sphere, calibrated) wrapper."""
    work = copy.deepcopy(model).to(device).eval()
    c_map = default_trm_c_per_module(work, c=512)
    if trained is None:
        bpd = BPDModel(work, c_map, max_passes=N_PASSES,
                       ablation=AblationConfig(mode="zero", sphere_rescale=False),
                       init_svd=True).to(device)
    else:
        bpd = BPDModel(work, c_map, max_passes=N_PASSES,
                       ablation=AblationConfig(mode="pass_mean", sphere_rescale=True),
                       init_svd=False).to(device)
        for m, w in bpd.wrappers.items():
            w.V.data.copy_(trained[m]["V"].to(device=device, dtype=torch.float32))
            w.U.data.copy_(trained[m]["U"].to(device=device, dtype=torch.float32))
    adapter = TRMBPDAdapter(bpd.target_model, readout_at="outer_step")
    if trained is not None:
        with bpd.target_capture(pass_offset=0, update_means=True, detach_inputs=True):
            adapter.rollout(cal_inputs.to(device), N_PASSES, 0, None)
        for w in bpd.wrappers.values():
            w.recorded_inputs.clear()
    return bpd, adapter


@torch.no_grad()
def _wrapper_identity_output(w: Any, x: torch.Tensor, tau: int, delta: float) -> torch.Tensor:
    """Call the real ``TiedComponentLinear.forward`` at pass ``tau`` with every component on."""
    dev = x.device
    w.mode = "component"
    w.schedule = PassMaskSchedule(
        comp_mask=torch.ones(1, 1, 1, w.C, device=dev),
        delta_mask=torch.full((1, 1, 1), float(delta), device=dev),
    )
    w.reset_pass_state(pass_offset=tau)
    try:
        return w(x)
    finally:
        w.mode = "target"
        w.schedule = None
        w.reset_pass_state(pass_offset=0)


@torch.no_grad()
def local_identity_errors(
    bpd_svd: BPDModel,
    bpd_tr: BPDModel,
    clean_inputs: Mapping[str, torch.Tensor],
    device: str,
) -> dict[str, Any]:
    """Teacher-forced per-invocation error of each identity path vs the native BF16 matmul."""
    report: dict[str, Any] = {}
    arm_names = (
        "correctly_rounded_fp64",
        "fp32_linear",
        "svd_all_on_delta_off",
        "svd_all_on_delta_on",
        "trained_all_on_delta_on",
    )
    for m in bpd_svd.module_order:
        w_svd, w_tr = bpd_svd.wrappers[m], bpd_tr.wrappers[m]
        W = w_svd.W_target
        x_all = clean_inputs[m]
        n_t = int(x_all.shape[1])
        per: dict[str, dict[str, list[float]]] = {
            a: {"mismatch_frac_by_pass": [], "max_eff_ulps_by_pass": [], "rel_err_vs_fp64_by_pass": []}
            for a in arm_names
        }
        native_rel_err: list[float] = []
        for t in range(n_t):
            x = x_all[:, t].to(device)
            nat = F.linear(x, W.to(x.dtype))
            exact = F.linear(x.double(), W.double())
            enorm = exact.norm().clamp_min(1e-300)
            native_rel_err.append(float((nat.double() - exact).norm() / enorm))
            outs = {
                "correctly_rounded_fp64": exact.float().to(x.dtype),
                "fp32_linear": F.linear(x.float(), W.float()).to(x.dtype),
                "svd_all_on_delta_off": _wrapper_identity_output(w_svd, x, t, 0.0),
                "svd_all_on_delta_on": _wrapper_identity_output(w_svd, x, t, 1.0),
                "trained_all_on_delta_on": _wrapper_identity_output(w_tr, x, t, 1.0),
            }
            tok_rms = nat.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
            ulp = torch.maximum(bf16_ulp(nat), bf16_ulp(tok_rms))
            for a, y in outs.items():
                per[a]["mismatch_frac_by_pass"].append(float((y != nat).float().mean()))
                per[a]["max_eff_ulps_by_pass"].append(
                    float(((y.float() - nat.float()).abs() / ulp).max())
                )
                per[a]["rel_err_vs_fp64_by_pass"].append(
                    float((y.double() - exact).norm() / enorm)
                )
        report[m] = {
            "native_bf16_rel_err_vs_fp64_by_pass": native_rel_err,
            "native_bf16_rel_err_vs_fp64_mean": sum(native_rel_err) / n_t,
            "arms": {
                a: {
                    **v,
                    "mismatch_frac_mean": sum(v["mismatch_frac_by_pass"]) / n_t,
                    "max_eff_ulps": max(v["max_eff_ulps_by_pass"]),
                    "rel_err_vs_fp64_mean": sum(v["rel_err_vs_fp64_by_pass"]) / n_t,
                }
                for a, v in per.items()
            },
        }
    return report


def _max_board_kl(rec: Mapping[str, Any], tau: int) -> float:
    return float(rec[f"pass_{tau}"]["kl64"]["max"])


def _n_boards_above(rec: Mapping[str, Any], tau: int, thr: float) -> int:
    return int(sum(v > thr for v in rec[f"pass_{tau}"]["kl64"]["per_board"]))


@torch.no_grad()
def audit_checkpoint(
    ckpt: str,
    trained: Mapping[str, Mapping[str, torch.Tensor]],
    cal_inputs: torch.Tensor,
    eval_inputs: torch.Tensor,
    eval_targets: torch.Tensor,
    device: str,
    *,
    skip_perturbation: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    base_bf16 = load_pretrained_trm(ckpt, dtype=torch.bfloat16, verify_sha256=True)
    base_orig32 = load_pretrained_trm(ckpt, dtype=torch.float32, verify_sha256=False)
    inp = eval_inputs.to(device)
    out: dict[str, Any] = {"checkpoint": ckpt}

    # ---------------- BF16 regime: reference = native CUDA BF16 target ----------------
    bf16_model = copy.deepcopy(base_bf16).to(device).eval()
    bf16_adapter = TRMBPDAdapter(bf16_model, readout_at="outer_step")
    ref_ro, ref_st = run_rollout(bf16_adapter, inp, N_PASSES)
    bf16_arms: dict[str, Any] = {}
    ro, st = run_rollout(bf16_adapter, inp, N_PASSES)
    bf16_arms["target_rerun"] = _arm_record(ro, st, ref_ro, ref_st)

    prev_flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        ro, st = run_rollout(bf16_adapter, inp, N_PASSES)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = prev_flag
    bf16_arms["target_bf16_reduced_precision_reduction_off"] = _arm_record(ro, st, ref_ro, ref_st)

    f32_lin = copy.deepcopy(base_bf16).to(device).eval()
    for blk in f32_lin.layers:
        blk.o_proj = _Float32AccumLinear(blk.o_proj)  # type: ignore[assignment]
        blk.down_proj = _Float32AccumLinear(blk.down_proj)  # type: ignore[assignment]
    ro, st = run_rollout(TRMBPDAdapter(f32_lin, readout_at="outer_step"), inp, N_PASSES)
    bf16_arms["fp32_linear_at_4_sites"] = _arm_record(ro, st, ref_ro, ref_st)
    del f32_lin

    bpd_svd, svd_adapter = _build_bpd(base_bf16, device, trained=None, cal_inputs=cal_inputs)
    bpd_tr, tr_adapter = _build_bpd(base_bf16, device, trained=trained, cal_inputs=cal_inputs)
    for name, bpd, adapter, delta in (
        ("svd_all_on_delta_off", bpd_svd, svd_adapter, 0.0),
        ("svd_all_on_delta_on", bpd_svd, svd_adapter, 1.0),
        ("trained_all_on_delta_on", bpd_tr, tr_adapter, 1.0),
    ):
        ro, st = run_component_rollout(bpd, adapter, inp, delta, device)
        bf16_arms[name] = _arm_record(ro, st, ref_ro, ref_st)

    # Teacher-forced local error on the clean target's own inputs.
    with bpd_svd.target_capture(pass_offset=0, update_means=False, detach_inputs=True):
        svd_adapter.rollout(inp, N_PASSES, 0, None)
    clean_inputs = {k: v.cpu() for k, v in bpd_svd.collect_recorded_inputs(N_PASSES).items()}
    for w in bpd_svd.wrappers.values():
        w.recorded_inputs.clear()
    out["local_identity_error_bf16"] = local_identity_errors(bpd_svd, bpd_tr, clean_inputs, device)
    out["svd_start_relative_residual_fro"] = {
        m: float((w.weight_delta().norm() / w.W_target.float().norm()).item())
        for m, w in bpd_svd.wrappers.items()
    }
    out["trained_relative_residual_fro"] = {
        m: float((w.weight_delta().norm() / w.W_target.float().norm()).item())
        for m, w in bpd_tr.wrappers.items()
    }
    del clean_inputs, bpd_svd, bpd_tr, svd_adapter, tr_adapter

    orig32 = copy.deepcopy(base_orig32).to(device).eval()
    ro, st = run_rollout(TRMBPDAdapter(orig32, readout_at="outer_step"), inp, N_PASSES)
    bf16_arms["full_fp32_original_weights"] = _arm_record(ro, st, ref_ro, ref_st)
    del orig32

    q32 = copy.deepcopy(base_bf16).float().to(device).eval()
    q32_adapter = TRMBPDAdapter(q32, readout_at="outer_step")
    q32_ro, q32_st = run_rollout(q32_adapter, inp, N_PASSES)
    bf16_arms["full_fp32_bf16_weights"] = _arm_record(q32_ro, q32_st, ref_ro, ref_st)

    q64 = copy.deepcopy(base_bf16).double().to(device).eval()
    q64_adapter = TRMBPDAdapter(q64, readout_at="outer_step")
    with exact_fp64_rmsnorm():
        x64_ro, x64_st = run_rollout(q64_adapter, inp, N_PASSES)
        x64_rerun_ro, x64_rerun_st = run_rollout(q64_adapter, inp, N_PASSES)
    bf16_arms["full_fp64_bf16_weights"] = _arm_record(x64_ro, x64_st, ref_ro, ref_st)
    out["bf16_regime"] = bf16_arms
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---------------- FP32 regimes: same-precision identity closure ----------------
    fp32_regimes: dict[str, Any] = {}
    # nn.Module.float() is in place, so promote a copy and keep base_bf16 intact.
    for regime, base32 in (("bf16_rounded_weights", copy.deepcopy(base_bf16).float()),
                           ("original_fp32_weights", base_orig32)):
        m32 = copy.deepcopy(base32).to(device).eval()
        a32 = TRMBPDAdapter(m32, readout_at="outer_step")
        r_ro, r_st = run_rollout(a32, inp, N_PASSES)
        arms: dict[str, Any] = {}
        ro, st = run_rollout(a32, inp, N_PASSES)
        arms["target_rerun"] = _arm_record(ro, st, r_ro, r_st)
        b_svd, a_svd = _build_bpd(base32, device, trained=None, cal_inputs=cal_inputs)
        b_tr, a_tr = _build_bpd(base32, device, trained=trained, cal_inputs=cal_inputs)
        for name, bpd, adapter, delta in (
            ("svd_all_on_delta_off", b_svd, a_svd, 0.0),
            ("svd_all_on_delta_on", b_svd, a_svd, 1.0),
            ("trained_all_on_delta_on", b_tr, a_tr, 1.0),
        ):
            ro, st = run_component_rollout(bpd, adapter, inp, delta, device)
            arms[name] = _arm_record(ro, st, r_ro, r_st)
        fp32_regimes[regime] = arms
        del m32, a32, b_svd, a_svd, b_tr, a_tr, r_ro, r_st
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    out["fp32_regimes"] = fp32_regimes

    # ---------------- FP64 reference: distance of each target from exact arithmetic ----
    out["fp64_reference"] = {
        "fp64_rerun": _arm_record(x64_rerun_ro, x64_rerun_st, x64_ro, x64_st),
        "fp32_target_bf16_weights": _arm_record(q32_ro, q32_st, x64_ro, x64_st),
        "native_bf16_target": _arm_record(ref_ro, ref_st, x64_ro, x64_st),
    }
    del q32, q32_adapter, q32_ro, q32_st, x64_rerun_ro, x64_rerun_st

    # ---------------- Board profile (clean targets) ----------------
    tgt = eval_targets.to(device)
    profile: dict[str, Any] = {}
    for label, ro in (("native_bf16", ref_ro), ("fp64_exact", x64_ro)):
        rows: dict[str, Any] = {}
        prev_pred = None
        for tau in READOUTS:
            logits = ro[tau]
            pred = logits.argmax(dim=-1)
            top2 = logits.topk(2, dim=-1).values
            margin = (top2[..., 0] - top2[..., 1]).amin(dim=-1)
            rows[f"pass_{tau}"] = {
                "cell_acc_per_board": [float(v) for v in (pred == tgt).float().mean(dim=-1).tolist()],
                "solved_per_board": [bool(v) for v in (pred == tgt).all(dim=-1).tolist()],
                "min_cell_logit_margin_per_board": [float(v) for v in margin.tolist()],
                "argmax_changes_from_previous_readout": (
                    None if prev_pred is None
                    else [int(v) for v in (pred != prev_pred).sum(dim=-1).tolist()]
                ),
            }
            prev_pred = pred
        profile[label] = rows
    out["board_profile"] = profile

    # ---------------- FP64 with BF16 rounding confined to one place ----------------
    def _round_bf16(_m: Any, _i: Any, y: torch.Tensor) -> torch.Tensor:
        return y.to(torch.bfloat16).to(y.dtype)

    rounding: dict[str, Any] = {}
    sites = [blk.o_proj for blk in q64.layers] + [blk.down_proj for blk in q64.layers]
    for label, mods in (("bf16_round_4_site_outputs", sites),
                        ("bf16_round_state_each_invocation", [q64.layers[-1]])):
        handles = [mod.register_forward_hook(_round_bf16) for mod in mods]
        try:
            with exact_fp64_rmsnorm():
                ro, st = run_rollout(q64_adapter, inp, N_PASSES)
        finally:
            for h in handles:
                h.remove()
        rounding[label] = _arm_record(ro, st, x64_ro, x64_st)
    out["fp64_reference"].update(rounding)

    # ---------------- FP64 perturbation growth ----------------
    if not skip_perturbation:
        sweep: list[dict[str, Any]] = []
        site = q64.layers[0].o_proj
        with exact_fp64_rmsnorm():
            for eps in PERTURB_EPS:
                for at_call in PERTURB_AT_CALL:
                    for seed in PERTURB_SEEDS:
                        hook = OutputPerturbation(eps, at_call, seed)
                        handle = site.register_forward_hook(hook)
                        try:
                            ro, st = run_rollout(q64_adapter, inp, N_PASSES)
                        finally:
                            handle.remove()
                        if hook.calls != N_PASSES:
                            raise AssertionError(f"hook saw {hook.calls} calls, expected {N_PASSES}")
                        rec = _arm_record(ro, st, x64_ro, x64_st)
                        rec.update({"eps": eps, "at_invocation": at_call, "seed": seed})
                        sweep.append(rec)
        out["fp64_perturbation_sweep"] = {
            "site": "layers.0.o_proj output, one invocation",
            "rmsnorm": "float64 (patched)",
            "runs": sweep,
        }
    del q64, q64_adapter
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    out["decisions"] = decide(out)
    out["elapsed_s"] = time.time() - t0
    return out


def decide(res: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the fixed decision rules from the module docstring."""
    local = res["local_identity_error_bf16"]
    impl_flags: dict[str, Any] = {}
    for m, rec in local.items():
        err_cap = 1.10 * rec["native_bf16_rel_err_vs_fp64_mean"]
        impl_flags[m] = {
            a: bool(rec["arms"][a]["max_eff_ulps"] > 2.0
                    or rec["arms"][a]["rel_err_vs_fp64_mean"] > err_cap)
            for a in IDENTITY_ARMS
        }
    implementation_mismatch = any(any(v.values()) for v in impl_flags.values())

    def closure(arms: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for tau in READOUTS:
            worst = {a: _max_board_kl(arms[a], tau) for a in IDENTITY_ARMS}
            out[f"pass_{tau}"] = {
                "closed": bool(all(v <= CLOSURE_KL_TOL for v in worst.values())),
                "worst_board_kl_by_arm": worst,
            }
        return out

    closures = {"native_bf16": closure(res["bf16_regime"])}
    for regime, arms in res["fp32_regimes"].items():
        closures[f"fp32_{regime}"] = closure(arms)

    sensitivity: dict[str, Any] = {}
    sens_boards: set[int] = set()
    for run in res.get("fp64_perturbation_sweep", {}).get("runs", []):
        key = f"eps_{run['eps']:.3g}"
        row = sensitivity.setdefault(key, {f"pass_{t}": [] for t in READOUTS})
        for t in READOUTS:
            row[f"pass_{t}"].append(_n_boards_above(run, t, SENSITIVE_KL))
        if run["eps"] == 2.0 ** -9:
            for t in READOUTS:
                for b, v in enumerate(run[f"pass_{t}"]["kl64"]["per_board"]):
                    if v > SENSITIVE_KL:
                        sens_boards.add(b)
    bf16_div: set[int] = set()
    for a in ("fp32_linear_at_4_sites",) + IDENTITY_ARMS:
        for b, v in enumerate(res["bf16_regime"][a]["pass_62"]["kl64"]["per_board"]):
            if v > SENSITIVE_KL:
                bf16_div.add(b)
    return {
        "implementation_mismatch_detected": implementation_mismatch,
        "implementation_flags_by_module": impl_flags,
        "identity_closure_by_regime": closures,
        "fp64_boards_above_1e-3_by_eps": sensitivity,
        "rounding_sensitive_boards_fp64_eps_2^-9": sorted(sens_boards),
        "bf16_identity_divergent_boards_pass62": sorted(bf16_div),
        "closure_tolerance_kl": CLOSURE_KL_TOL,
        "sensitive_kl": SENSITIVE_KL,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path,
                    default=Path("mechinterp/bpd/results/v2/bpd_b4_v2_stage4_weights.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoints", nargs="+", default=["gaoxin492", "yagizdevre"],
                    choices=["gaoxin492", "yagizdevre"])
    ap.add_argument("--eval-boards", type=int, default=16)
    ap.add_argument("--skip-perturbation", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        ap.error("CUDA unavailable")

    actual_sha = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if actual_sha != EXPECTED_WEIGHTS_SHA256:
        ap.error(f"Factor weights SHA-256 mismatch: {actual_sha}")
    torch.manual_seed(0)
    saved = torch.load(args.weights, map_location="cpu", weights_only=True)
    wb = load_wb_stall_and_train_sudoku(None, n_train=128, device="cpu")
    cal = wb["train_inputs"][:32]
    ev = wb["eval_inputs"][: args.eval_boards]
    ev_t = wb["eval_targets"][: args.eval_boards]
    key = {"gaoxin492": "bpd_ref", "yagizdevre": "bpd_tgt"}

    report: dict[str, Any] = {
        "audit": "BPD-01_B4_v2_closure_mechanism",
        "weights_sha256": actual_sha,
        "torch_version": torch.__version__,
        "device": args.device,
        "cuda_device": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "bf16_reduced_precision_reduction_default": bool(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "eval_boards": int(ev.shape[0]),
        "n_passes": N_PASSES,
        "readouts": list(READOUTS),
        "kl": "float64 token-mean KL(reference || arm) over 81 cells, 9 digits",
        "board_hashes_sha256_prefix16": [
            hashlib.sha256(bytes(map(int, row.tolist()))).hexdigest()[:16] for row in ev
        ],
        "perturbation_eps": list(PERTURB_EPS),
        "checkpoints": {},
    }
    for ckpt in args.checkpoints:
        report["checkpoints"][ckpt] = audit_checkpoint(
            ckpt, saved[key[ckpt]], cal, ev, ev_t, args.device,
            skip_perturbation=args.skip_perturbation,
        )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({ckpt: report["checkpoints"][ckpt]["decisions"]}, indent=1), flush=True)
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
