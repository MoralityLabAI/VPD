"""BPD-01 C pre-scoring diagnostic on TRAINING boards (never the reserved 64).

Checks, for one refit CI:
1. replay identity: the causal loop with the clean-informed top-k masks replayed
   reproduces the open-loop clean-informed (T2) rollout exactly;
2. KL(clean || T2) and KL(clean || T3 causal preflight) per board at the readouts;
3. T3 vs T2 top-k mask disagreement at every invocation;
4. how close to tied the CI logits are around rank k on clean inputs.

Run on CPU FP32 with BF16-rounded checkpoint weights, the registered evaluation regime
except for the device.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechinterp.bpd.adapters.trm import (
    TRMBPDAdapter,
    default_trm_c_per_module,
    load_pretrained_trm,
    load_wb_stall_and_train_sudoku,
)
from mechinterp.bpd.causal_ci import causal_ci_rollout, clean_informed_ci_masks
from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.scripts import audit_b4_causal_ci as S
from mechinterp.bpd.scripts.b4_numerics import kl64_per_board


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, choices=sorted(S.FACTOR_KEY))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--train-boards", type=int, nargs=2, default=(100, 102),
                    help="half-open range of train.csv board indices")
    ap.add_argument("--ci-dir", type=Path, default=Path("mechinterp/bpd/results/v2/ci_refit"))
    ap.add_argument("--weights", type=Path, default=Path("mechinterp/bpd/results/v2/bpd_b4_v2_stage4_weights.pt"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev, ck = args.device, args.checkpoint
    torch.manual_seed(0)
    factors = torch.load(args.weights, map_location="cpu", weights_only=True)[S.FACTOR_KEY[ck]]
    train = load_wb_stall_and_train_sudoku(None, n_train=128, device="cpu")["train_inputs"][:128]
    x = train[args.train_boards[0]:args.train_boards[1]].to(dev)
    bsz = int(x.shape[0])
    rec = json.loads((args.ci_dir / f"{ck}_seed0_state_only_ci.json").read_text(encoding="utf-8"))
    ci = S.load_ci(args.ci_dir / f"{ck}_seed0_state_only_ci.pt", rec["ci_weights_sha256"], dev)
    base = load_pretrained_trm(ck, dtype=torch.bfloat16, verify_sha256=True).float()
    bpd = BPDModel(copy.deepcopy(base).to(dev).eval(), default_trm_c_per_module(base, c=S.C),
                   max_passes=S.N_PASSES, ablation=S.ZERO, init_svd=False).to(dev)
    for m, w in bpd.wrappers.items():
        w.V.data.copy_(factors[m]["V"].to(device=dev, dtype=torch.float32))
        w.U.data.copy_(factors[m]["U"].to(device=dev, dtype=torch.float32))
    ad = TRMBPDAdapter(bpd.target_model, readout_at="outer_step")

    with bpd.target_capture(pass_offset=0, update_means=False, detach_inputs=True):
        clean, _, _, _ = ad.rollout(x, S.N_PASSES, 0, None)
    clean_in = bpd.collect_recorded_inputs(expected_passes=S.N_PASSES)
    for w in bpd.wrappers.values():
        w.recorded_inputs.clear()
    t2 = clean_informed_ci_masks(bpd, ad, ci, x, S.N_PASSES, selection="topk", k=S.K)
    ro_t2 = S._masked(bpd, ad, x, S._sched_from_masks(t2, bsz, dev))
    replay = causal_ci_rollout(bpd, ad, ci, x, S.N_PASSES, observation="preflight_target",
                               selection="topk", k=S.K, replay_masks=t2)
    t3 = causal_ci_rollout(bpd, ad, ci, x, S.N_PASSES, observation="preflight_target",
                           selection="topk", k=S.K, reference_masks=t2)

    logit_stats: dict[str, dict[str, float]] = {}
    for t0 in range(0, S.N_PASSES, 7):
        out = ci({m: clean_in[m][:, t0:t0 + 7] for m in bpd.module_order}, pass_offset=t0)
        for m in bpd.module_order:
            p = out.pre_sigmoid[m].float()
            srt = p.sort(dim=-1, descending=True).values
            st = logit_stats.setdefault(m, {"kth_logit": 0.0, "gap_k_k1": 0.0, "n_positive": 0.0,
                                            "logit_std_over_components": 0.0, "lower_leaky_mean": 0.0})
            st["kth_logit"] += float(srt[..., S.K - 1].mean()) / 9
            st["gap_k_k1"] += float((srt[..., S.K - 1] - srt[..., S.K]).mean()) / 9
            st["n_positive"] += float((p > 0).float().sum(-1).mean()) / 9
            st["logit_std_over_components"] += float(p.std(dim=-1).mean()) / 9
            st["lower_leaky_mean"] += float(out.lower_leaky[m].mean()) / 9

    report = {
        "experiment": "BPD-01_C_prescore_diagnostic",
        "checkpoint": ck,
        "boards": f"train.csv indices {args.train_boards[0]}..{args.train_boards[1] - 1} (not WB-Stall, not reserved-64)",
        "device": dev,
        "regime": "FP32 compute, BF16-rounded checkpoint weights",
        "ci_weights_sha256": rec["ci_weights_sha256"],
        "k": S.K, "C": S.C,
        "replay_max_abs_logit_diff_vs_T2": {
            f"pass_{t}": float((replay["readouts"][t] - ro_t2[t]).abs().max()) for t in S.READOUTS},
        "kl_T2_clean_informed": {f"pass_{t}": kl64_per_board(ro_t2[t], clean[t]) for t in S.READOUTS},
        "kl_T3_causal_preflight": {f"pass_{t}": kl64_per_board(t3["readouts"][t], clean[t]) for t in S.READOUTS},
        "T3_vs_T2_disagreement_mean_over_boards": {
            m: [sum(r) / len(r) for r in rows] for m, rows in t3["disagreement_by_module_by_pass_by_board"].items()},
        "ci_logits_on_clean_inputs_mean_over_invocations": logit_stats,
    }
    out_path = args.out or Path(f"mechinterp/bpd/results/v2/ci_refit/{ck}_prescore_diagnostic_train_boards.json")
    out_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("replay_max_abs_logit_diff_vs_T2", "kl_T2_clean_informed",
                                             "kl_T3_causal_preflight")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
