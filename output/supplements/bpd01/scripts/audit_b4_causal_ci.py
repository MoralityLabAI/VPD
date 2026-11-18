"""B4-CAUSAL-CI-01 scoring: refit CI (clean-informed and strictly causal) vs SVD/magnitude top-k.

Implements ``docs/B4_CAUSAL_CI_REFIT_REGISTRATION_20261117.md`` exactly: CUDA FP32 with
BF16-rounded checkpoint weights, reserved-64 WB-Stall boards in chunks of 16, 63
invocations, readouts 20/41/62 (pass 20 primary), k = 307 of 512 per module/token/
invocation, zero ablation, residual off, no sphere for the budget-matched arms.
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

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechinterp.bpd.adapters.trm import (
    TRMBPDAdapter,
    default_trm_c_per_module,
    load_pretrained_trm,
    load_wb_stall_and_train_sudoku,
)
from mechinterp.bpd.causal_ci import (
    causal_ci_rollout,
    clean_informed_ci_masks,
    module_execution_order,
    within_pass_dependency,
)
from mechinterp.bpd.ci import TiedCINetwork
from mechinterp.bpd.components import BPDModel, PassMaskSchedule
from mechinterp.bpd.config import AblationConfig, CIConfig
from mechinterp.bpd.scripts.audit_b4_online_svd_reserved64 import (
    BOARD_INDICES as RESERVED64,
    validate_registration,
)
from mechinterp.bpd.scripts.audit_b4_v2_controls_and_ablation_split import EXPECTED_WEIGHTS_SHA256
from mechinterp.bpd.scripts.b4_numerics import argmax_flips_per_board, kl64_per_board

N_PASSES = 63
READOUTS = (20, 41, 62)
K = 307
C = 512
FLOOR_TOL = 1e-5
BOOT_SEED = 20261117301
N_BOOT = 20000
N_FLIP = 100000
FACTOR_KEY = {"gaoxin492": "bpd_ref", "yagizdevre": "bpd_tgt"}
ZERO = AblationConfig(mode="zero", sphere_rescale=False)
NATIVE = AblationConfig(mode="pass_mean", sphere_rescale=True)
BUDGET_ARMS = ("S0", "S1", "S2", "T1", "T2", "T3", "T4")
NATIVE_ARMS = ("N1", "N2")
FLOOR_ARMS = ("target_repeat", "svd_all_on_residual_off", "trained_all_on_residual_on")


def _set_ablation(bpd: BPDModel, ab: AblationConfig) -> None:
    for w in bpd.wrappers.values():
        w.ablation = ab


@torch.no_grad()
def _calibrate_means(bpd: BPDModel, adapter: TRMBPDAdapter, inputs: torch.Tensor) -> None:
    """Plain pass-conditional means over all calibration boards in one batch (no input recording)."""
    bpd.clear_schedules()
    bpd.reset_wrappers(pass_offset=0)
    for w in bpd.wrappers.values():
        w.calibrated_passes.zero_()
        w.update_running_means = True
    try:
        adapter.rollout(inputs, N_PASSES, 0, None)
    finally:
        for w in bpd.wrappers.values():
            w.update_running_means = False


@torch.no_grad()
def _masked(bpd: BPDModel, adapter: TRMBPDAdapter, inputs: torch.Tensor,
            sched: Mapping[str, PassMaskSchedule]) -> dict[int, torch.Tensor]:
    with bpd.masked_components(dict(sched), pass_offset=0, record_inputs=False):
        ro, _, _, _ = adapter.rollout(inputs, N_PASSES, 0, None)
    return ro


def _const_sched(bpd: BPDModel, comp: float, delta: float, dev: str) -> dict[str, PassMaskSchedule]:
    return {
        m: PassMaskSchedule(comp_mask=torch.full((1, N_PASSES, 1, C), comp, device=dev),
                            delta_mask=torch.full((1, N_PASSES, 1), delta, device=dev))
        for m in bpd.module_order
    }


def _magnitude_masks(bpd: BPDModel, clean_inputs: Mapping[str, torch.Tensor], dev: str) -> dict[str, torch.Tensor]:
    masks = {}
    for m, w in bpd.wrappers.items():
        x = clean_inputs[m].to(device=dev, dtype=torch.float32)
        score = torch.einsum("btpd,dc->btpc", x, w.V.detach().float()).abs() * w.U.detach().float().norm(dim=-1)
        masks[m] = torch.zeros_like(score).scatter_(-1, score.topk(K, dim=-1).indices, 1.0)
        del x, score
    return masks


def _sched_from_masks(masks: Mapping[str, torch.Tensor], bsz: int, dev: str, *,
                      online: bool = False) -> dict[str, PassMaskSchedule]:
    zero_delta = torch.zeros(bsz, N_PASSES, 1, device=dev)
    return {m: PassMaskSchedule(comp_mask=v, delta_mask=zero_delta, online_topk=K if online else None)
            for m, v in masks.items()}


def load_ci(path: Path, expected_sha: str, dev: str) -> TiedCINetwork:
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != expected_sha:
        raise RuntimeError(f"CI weights SHA mismatch for {path}: {sha} != {expected_sha}")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    net = TiedCINetwork(blob["module_dims"], blob["c_per_module"], CIConfig(**blob["ci_config"]))
    net.load_state_dict(blob["state_dict"])
    return net.to(dev).eval()


@torch.no_grad()
def score_chunk(
    bpd_svd: BPDModel, ad_svd: TRMBPDAdapter, bpd_tr: BPDModel, ad_tr: TRMBPDAdapter,
    ci: TiedCINetwork, inputs: torch.Tensor, dev: str,
) -> dict[str, Any]:
    bsz = int(inputs.shape[0])
    _set_ablation(bpd_svd, ZERO)
    _set_ablation(bpd_tr, ZERO)
    with bpd_svd.target_capture(pass_offset=0, update_means=False, detach_inputs=True):
        clean_ro, _, _, _ = ad_svd.rollout(inputs, N_PASSES, 0, None)
    clean_in = bpd_svd.collect_recorded_inputs(expected_passes=N_PASSES)
    for w in bpd_svd.wrappers.values():
        w.recorded_inputs.clear()
    with bpd_tr.target_capture(pass_offset=0, update_means=False, detach_inputs=True):
        clean_in_tr_ro, _, _, _ = ad_tr.rollout(inputs, N_PASSES, 0, None)
    clean_in_tr = bpd_tr.collect_recorded_inputs(expected_passes=N_PASSES)
    for w in bpd_tr.wrappers.values():
        w.recorded_inputs.clear()
    ro: dict[str, dict[int, torch.Tensor]] = {}
    repeat, _, _, _ = ad_svd.rollout(inputs, N_PASSES, 0, None)
    ro["target_repeat"] = repeat
    ro["target_repeat_trained_wrapper"] = clean_in_tr_ro

    ro["svd_all_on_residual_off"] = _masked(bpd_svd, ad_svd, inputs, _const_sched(bpd_svd, 1.0, 0.0, dev))
    ro["trained_all_on_residual_on"] = _masked(bpd_tr, ad_tr, inputs, _const_sched(bpd_tr, 1.0, 1.0, dev))

    static = torch.zeros(1, N_PASSES, 1, C, device=dev)
    static[..., :K] = 1.0
    ro["S0"] = _masked(bpd_svd, ad_svd, inputs, _sched_from_masks({m: static for m in bpd_svd.module_order}, 1, dev))
    s1 = _magnitude_masks(bpd_svd, clean_in, dev)
    ro["S1"] = _masked(bpd_svd, ad_svd, inputs, _sched_from_masks(s1, bsz, dev))
    ro["S2"] = _masked(bpd_svd, ad_svd, inputs, _sched_from_masks(s1, bsz, dev, online=True))
    del s1
    t1_ref = _magnitude_masks(bpd_tr, clean_in_tr, dev)
    ro["T1"] = _masked(bpd_tr, ad_tr, inputs, _sched_from_masks(t1_ref, bsz, dev, online=True))
    del t1_ref

    t2 = clean_informed_ci_masks(bpd_tr, ad_tr, ci, inputs, N_PASSES, selection="topk", k=K)
    ro["T2"] = _masked(bpd_tr, ad_tr, inputs, _sched_from_masks(t2, bsz, dev))
    t3 = causal_ci_rollout(bpd_tr, ad_tr, ci, inputs, N_PASSES, observation="preflight_target",
                           selection="topk", k=K, reference_masks=t2)
    ro["T3"] = t3["readouts"]
    t4 = causal_ci_rollout(bpd_tr, ad_tr, ci, inputs, N_PASSES, observation="lagged",
                           selection="topk", k=K, reference_masks=t2)
    ro["T4"] = t4["readouts"]
    del t2

    _set_ablation(bpd_tr, NATIVE)
    n1 = clean_informed_ci_masks(bpd_tr, ad_tr, ci, inputs, N_PASSES, selection="continuous")
    n1_density = {m: float(v.mean()) for m, v in n1.items()}
    n1_active = {m: float((v > 0.5).float().mean()) for m, v in n1.items()}
    ro["N1"] = _masked(bpd_tr, ad_tr, inputs, _sched_from_masks(n1, bsz, dev))
    del n1
    n2 = causal_ci_rollout(bpd_tr, ad_tr, ci, inputs, N_PASSES, observation="preflight_target",
                           selection="continuous")
    ro["N2"] = n2["readouts"]
    _set_ablation(bpd_tr, ZERO)

    arms: dict[str, Any] = {}
    for name, r in ro.items():
        arms[name] = {
            f"pass_{t}": {"kl64": kl64_per_board(r[t], clean_ro[t]),
                          "argmax_flips": argmax_flips_per_board(r[t], clean_ro[t])}
            for t in READOUTS
        }
    return {
        "arms": arms,
        "T3_vs_T2_disagreement": t3["disagreement_by_module_by_pass_by_board"],
        "T4_vs_T2_disagreement": t4["disagreement_by_module_by_pass_by_board"],
        "N1_mean_mask": n1_density,
        "N1_fraction_above_0p5": n1_active,
        "N2_mean_mask_by_module_by_pass": n2["density_by_module_by_pass"],
    }


def _boot_mean(d: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return d[rng.integers(0, d.size, size=(N_BOOT, d.size))].mean(axis=1)


def _signflip_one_sided(d: np.ndarray, rng: np.random.Generator) -> float:
    obs = d.mean()
    cnt, left = 0, N_FLIP
    while left:
        b = min(4096, left)
        s = rng.integers(0, 2, size=(b, d.size)) * 2 - 1
        cnt += int(((s * d[None, :]).mean(axis=1) <= obs).sum())
        left -= b
    return (1 + cnt) / (N_FLIP + 1)


def decide(per_ckpt: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Registered pass-20 rules (H1, H2, H3) plus descriptive late-pass statistics."""
    out: dict[str, Any] = {"by_checkpoint": {}}
    for ck, res in per_ckpt.items():
        rng = np.random.default_rng(BOOT_SEED)
        a = res["arms"]
        rows: dict[str, Any] = {}
        for t in READOUTS:
            kl = {n: np.array(a[n][f"pass_{t}"]["kl64"]) for n in a}
            floor_max = max(float(kl[f].max()) for f in FLOOR_ARMS)
            stats: dict[str, Any] = {"floor_max_kl": floor_max, "interpretable": floor_max <= FLOOR_TOL,
                                     "mean_kl": {n: float(v.mean()) for n, v in kl.items()}}
            for label, diff in (("H1_T3_minus_T1", kl["T3"] - kl["T1"]),
                                ("H2_T3_minus_S2", kl["T3"] - kl["S2"]),
                                ("H3_M_T3_minus_T2_minus_0.10_S0", kl["T3"] - kl["T2"] - 0.10 * kl["S0"]),
                                ("T3_minus_T2", kl["T3"] - kl["T2"]),
                                ("T4_minus_T3", kl["T4"] - kl["T3"]),
                                ("N2_minus_N1", kl["N2"] - kl["N1"]),
                                ("T2_minus_S1", kl["T2"] - kl["S1"]),
                                ("T1_minus_S2", kl["T1"] - kl["S2"])):
                boots = _boot_mean(diff, rng)
                stats[label] = {
                    "mean": float(diff.mean()), "median": float(np.median(diff)),
                    "lower_2_5": float(np.quantile(boots, 0.025)),
                    "upper_97_5": float(np.quantile(boots, 0.975)),
                    "signflip_one_sided_p_negative": _signflip_one_sided(diff, rng),
                    "n_negative": int((diff < 0).sum()), "n_positive": int((diff > 0).sum()),
                    "abs_effect_top_board_share": float(np.abs(diff).max() / max(np.abs(diff).sum(), 1e-300)),
                }
            rows[f"pass_{t}"] = stats
        out["by_checkpoint"][ck] = rows

    def both(label: str, test: str) -> bool:
        vals = [out["by_checkpoint"][ck]["pass_20"][label] for ck in out["by_checkpoint"]]
        ok = all(out["by_checkpoint"][ck]["pass_20"]["interpretable"] for ck in out["by_checkpoint"])
        if test == "upper<0":
            return ok and all(v["upper_97_5"] < 0 for v in vals)
        return ok and all(v["lower_2_5"] > 0 for v in vals)

    def verdict(label: str) -> str:
        if both(label, "upper<0"):
            return "T3 better (upper bound < 0 on both checkpoints)"
        if both(label, "lower>0"):
            return "T3 worse (lower bound > 0 on both checkpoints)"
        return "inconclusive"

    out["registered_pass20_decisions"] = {
        "H1_gate_value_beyond_magnitude_selection": verdict("H1_T3_minus_T1"),
        "H2_causal_BPD_vs_zero_fit_SVD_online": verdict("H2_T3_minus_S2"),
        "H3_causal_CI_noninferior_to_clean_informed_CI": (
            "non-inferior" if both("H3_M_T3_minus_T2_minus_0.10_S0", "upper<0") else "not shown"
        ),
    }
    if len(per_ckpt) == 2:
        holm: dict[str, Any] = {}
        for label in ("H1_T3_minus_T1", "H2_T3_minus_S2"):
            ps = sorted((out["by_checkpoint"][ck]["pass_20"][label]["signflip_one_sided_p_negative"], ck)
                        for ck in out["by_checkpoint"])
            holm[label] = {ck: min(1.0, p * (len(ps) - i)) for i, (p, ck) in enumerate(ps)}
        out["holm_adjusted_signflip_p_pass20"] = holm
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoints", nargs="+", default=["gaoxin492", "yagizdevre"], choices=sorted(FACTOR_KEY))
    ap.add_argument("--ci-dir", type=Path, default=Path("mechinterp/bpd/results/v2/ci_refit"))
    ap.add_argument("--ci-stem", default="seed0_state_only_ci",
                    help="CI files are {ci-dir}/{checkpoint}_{ci-stem}.pt/.json")
    ap.add_argument("--experiment", default="B4-CAUSAL-CI-01")
    ap.add_argument("--registration", default="mechinterp/bpd/docs/B4_CAUSAL_CI_REFIT_REGISTRATION_20261117.md")
    ap.add_argument("--weights", type=Path, default=Path("mechinterp/bpd/results/v2/bpd_b4_v2_stage4_weights.pt"))
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("mechinterp/bpd/results/v2/gate_b4_causal_ci.json"))
    args = ap.parse_args()
    dev = args.device
    sha = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if sha != EXPECTED_WEIGHTS_SHA256:
        ap.error(f"factor SHA mismatch {sha}")
    torch.manual_seed(0)
    factors = torch.load(args.weights, map_location="cpu", weights_only=True)
    wb = load_wb_stall_and_train_sudoku(None, n_train=128, device="cpu")
    validate_registration(int(wb["eval_inputs"].shape[0]))
    ev = wb["eval_inputs"][list(RESERVED64)]
    train = wb["train_inputs"][:128]

    report: dict[str, Any] = {
        "experiment": args.experiment,
        "registration": args.registration,
        "ci_stem": args.ci_stem,
        "replaces_historical_trained_gate_receipt": False,
        "factor_weights_sha256": sha,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name() if dev.startswith("cuda") else None,
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "regime": "FP32 compute, BF16-rounded checkpoint weights",
        "eval_board_indices": list(RESERVED64),
        "board_sha256_prefix16": [hashlib.sha256(bytes(map(int, r.tolist()))).hexdigest()[:16] for r in ev],
        "k": K, "C": C, "n_passes": N_PASSES, "readouts": list(READOUTS),
        "checkpoints": {},
    }
    t0 = time.time()
    for ck in args.checkpoints:
        rec = json.loads((args.ci_dir / f"{ck}_{args.ci_stem}.json").read_text(encoding="utf-8"))
        ci = load_ci(args.ci_dir / f"{ck}_{args.ci_stem}.pt", rec["ci_weights_sha256"], dev)
        rec = {k: v for k, v in rec.items() if k not in ("train_history", "validation_history")}
        base = load_pretrained_trm(ck, dtype=torch.bfloat16, verify_sha256=True).float()
        c_map = default_trm_c_per_module(base, c=C)
        bpd_svd = BPDModel(copy.deepcopy(base).to(dev).eval(), c_map, max_passes=N_PASSES,
                           ablation=ZERO, init_svd=True).to(dev)
        bpd_tr = BPDModel(copy.deepcopy(base).to(dev).eval(), c_map, max_passes=N_PASSES,
                          ablation=NATIVE, init_svd=False).to(dev)
        for m, w in bpd_tr.wrappers.items():
            w.V.data.copy_(factors[FACTOR_KEY[ck]][m]["V"].to(device=dev, dtype=torch.float32))
            w.U.data.copy_(factors[FACTOR_KEY[ck]][m]["U"].to(device=dev, dtype=torch.float32))
        ad_svd = TRMBPDAdapter(bpd_svd.target_model, readout_at="outer_step")
        ad_tr = TRMBPDAdapter(bpd_tr.target_model, readout_at="outer_step")
        _calibrate_means(bpd_tr, ad_tr, train.to(dev))

        order = module_execution_order(bpd_tr, lambda: ad_tr.rollout(train[:2].to(dev), 1, 0, None))
        with bpd_tr.target_capture(pass_offset=0):
            ad_tr.rollout(train[:4].to(dev), 1, 0, None)
        x0 = bpd_tr.collect_recorded_inputs(expected_passes=1)
        with torch.enable_grad():
            dep = within_pass_dependency(ci, x0)
        for w in bpd_tr.wrappers.values():
            w.recorded_inputs.clear()

        chunks = []
        for i in range(0, ev.shape[0], args.chunk):
            chunks.append(score_chunk(bpd_svd, ad_svd, bpd_tr, ad_tr, ci, ev[i:i + args.chunk].to(dev), dev))
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()
        merged_arms: dict[str, Any] = {}
        for name in chunks[0]["arms"]:
            merged_arms[name] = {
                f"pass_{t}": {
                    "kl64": sum((c["arms"][name][f"pass_{t}"]["kl64"] for c in chunks), []),
                    "argmax_flips": sum((c["arms"][name][f"pass_{t}"]["argmax_flips"] for c in chunks), []),
                }
                for t in READOUTS
            }

        def _merge_dis(key: str) -> dict[str, list[float]]:
            # mean over all 64 boards, per module, per invocation
            mods = chunks[0][key].keys()
            return {m: [float(np.mean(sum((c[key][m][t] for c in chunks), []))) for t in range(N_PASSES)]
                    for m in mods}

        report["checkpoints"][ck] = {
            "ci_refit_receipt": rec,
            "module_execution_order": order,
            "ci_module_order": list(ci.module_order),
            "within_pass_dependency_grad_norm_tau0": dep,
            "arms": merged_arms,
            "T3_vs_T2_disagreement_by_module_by_pass": _merge_dis("T3_vs_T2_disagreement"),
            "T4_vs_T2_disagreement_by_module_by_pass": _merge_dis("T4_vs_T2_disagreement"),
            "N1_mean_mask_by_chunk": [c["N1_mean_mask"] for c in chunks],
            "N1_fraction_above_0p5_by_chunk": [c["N1_fraction_above_0p5"] for c in chunks],
            "N2_mean_mask_by_module_by_pass": {
                m: [float(np.mean([c["N2_mean_mask_by_module_by_pass"][m][t] for c in chunks])) for t in range(N_PASSES)]
                for m in chunks[0]["N2_mean_mask_by_module_by_pass"]
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        del bpd_svd, bpd_tr, ad_svd, ad_tr, ci
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
    report["decisions"] = decide(report["checkpoints"])
    report["elapsed_s"] = time.time() - t0
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report["decisions"]["registered_pass20_decisions"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
