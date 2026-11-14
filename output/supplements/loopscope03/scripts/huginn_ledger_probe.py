#!/usr/bin/env python3
"""LOOPSCOPE-03 Supplementary Model Check: Huginn-0125 Path Independence & Cache Ledger Audit.

Tests the pre-registered prediction for `tomg-group-umd/huginn-0125` (Geiping et al. 2025, 3.5B):
  - Given native truncated-normal random latent starts r_0^(m) and per-step input re-injection
    adapter([r_t, e]), Huginn-0125 rapidly contracts independent forks onto the same input-
    determined trajectory r*(e), making its hidden state overwhelmingly path-independent Cache C_t
    rather than path-dependent Memory R_t.

Measures across 32 diverse reasoning/deduction prompts (16 Sudoku S1/S4 prompts + 16 multi-step
arithmetic/algebraic reasoning prompts), K = 4 independent native random initializations per prompt,
and T = 32 recurrent steps:
  1. Stationarity admission gate: exact state-transplant invariance ||F_e^(4)(r_4) - r_8||_inf.
  2. Law-of-Total-Variance upper bound on R_t:
       frac_path_var(t) = (0.5 * E_{A!=B}[||r_t^A - r_t^B||^2]) / Var_total(r_t)
     At t = 0 (random starts), frac_path_var = 1.000; contraction toward 0 bounds E[||R_t||^2]
     without any parametric refill approximation.
  3. Cross-fork cosine similarity, top-1 Coda+lm_head prediction agreement, and KL divergence
     across t in {1, 2, 4, 8, 16, 32}.
  4. Causal Fork Swap & Resume Test at t = 4 and t = 8:
     - Replacing r_t^(A) with cross-fork cache mean C_hat_t(x) = mean_{m!=A} r_t^(m) vs swapping
       donor residual R_t^(B) = r_t^(B) - C_hat_t(x), scoring next-step top-1 flip rate and
       iterations to reach the step-32 steady-state prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mechinterp.loopscope02.scripts import trm_sudoku as TS  # noqa: E402

RESULTS_DIR = REPO_ROOT / "mechinterp" / "loopscope03" / "results"
HUGINN_SNAP = Path(
    r"C:\Users\patri\.cache\huggingface\hub\models--tomg-group-umd--huginn-0125\snapshots\bb6621b65e90b6a4b9b29ef88dc83866d450470c"
)


def build_eval_prompts() -> list[dict]:
    bundle = json.loads((REPO_ROOT / "mechinterp" / "loopscope02" / "results" / "benchmark_bundle.json").read_text("utf-8"))
    items = bundle["items"]
    s1 = [it for it in items if it["stratum"] == "S1_naked1"][:8]
    s4 = [it for it in items if it["stratum"] == "S4_stalled"][:8]
    prompts = []
    for it in s1 + s4:
        q = it["question"]
        r, c = divmod(int(it["target_idx"]), 9)
        rows = [" ".join(q[k * 9 : (k + 1) * 9]) for k in range(9)]
        txt = "Sudoku 9x9 (. is empty):\n" + "\n".join(rows) + f"\nDigit at row {r+1} col {c+1} is:"
        prompts.append({"category": it["stratum"], "prompt": txt})

    math_logic = [
        "If x + 7 = 19 and y = 3 * x - 5, then y =",
        "Solve for n: 4 * n - 9 = 27. Answer: n =",
        "A train travels 180 km in 3 hours. At the same speed, in 5 hours it travels",
        "Let a_1 = 2 and a_{k+1} = 2 * a_k + 1. Then a_4 =",
        "All bloops are razzies and all razzies are lazzies. Therefore all bloops are",
        "What is (17 * 6) - (13 * 5)? Answer:",
        "If p is prime and 20 < p < 28, then p =",
        "Suppose f(x) = x^2 - 4x + 7. Then f(5) =",
        "Three consecutive integers sum to 72. The largest of the three integers is",
        "In mod 11 arithmetic, (8 * 7) mod 11 =",
        "A rectangle has perimeter 30 and width 4. Its area is",
        "If 2^k = 128, then k + 5 =",
        "Alice is taller than Bob, and Bob is taller than Cara. The shortest of the three is",
        "The GCD of 84 and 126 is",
        "If 3a + 2b = 16 and b = 5, then a =",
        "Next term in arithmetic sequence 7, 13, 19, 25, 31 is",
    ]
    for mp in math_logic:
        prompts.append({"category": "math_logic", "prompt": mp})
    return prompts


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "huginn_cache_and_fork.json")
    parser.add_argument("--mem-fraction", type=float, default=0.95)
    parser.add_argument("--min-watts", type=float, default=40.0)
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        watts = TS.power_limit_watts()
        if watts is not None and watts < args.min_watts:
            print(f"[power-guard] GPU limit {watts}W < {args.min_watts}W; exiting 75.")
            return TS.EXIT_POWER_CAP
        if 0.0 < args.mem_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
        torch.backends.cuda.matmul.allow_tf32 = False

    t0 = time.time()
    print(f"Loading cached Huginn-0125 from {HUGINN_SNAP}...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(HUGINN_SNAP), local_files_only=True, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(str(HUGINN_SNAP), local_files_only=True, trust_remote_code=True)
    raven_cls = get_class_from_dynamic_module("raven_modeling_minimal.RavenForCausalLM", str(HUGINN_SNAP))
    for base in raven_cls.__mro__:
        if isinstance(getattr(base, "_tied_weights_keys", None), list):
            base._tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}
    model = raven_cls.from_pretrained(
        str(HUGINN_SNAP),
        config=cfg,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(dev).eval()

    prompts = build_eval_prompts()
    n_p = len(prompts)
    k_forks = 4
    n_steps = 32
    audit_steps = (0, 1, 2, 4, 8, 16, 32)  # 0 = random init before step 1

    # Collect final-token latent vectors [N_p, K, len(audit_steps), D] and step-by-step top-1 token IDs [N_p, K, 32]
    d_emb = int(model.config.n_embd)
    lat_snaps = torch.empty((n_p, k_forks, len(audit_steps), d_emb), dtype=torch.float32)
    top1_steps = np.empty((n_p, k_forks, n_steps), dtype=np.int64)
    kl_pairs_by_step = {s: [] for s in (1, 2, 4, 8, 16, 32)}

    # Also test transplant exactness + Fork R_t swap at t = 4 and t = 8
    transplant_max_diff = 0.0
    swap_flip_rates = {4: [], 8: []}
    cache_only_match_rates = {4: [], 8: []}
    resume_steps_to_steady = {4: {"Fresh": [], "CacheMean": [], "Full": []}}

    for pi, item in enumerate(prompts):
        enc = tok(item["prompt"], return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(dev)
        seq_len = input_ids.shape[1]
        # Expand batch = k_forks for exact parallel execution on the same prompt!
        ids_k = input_ids.expand(k_forks, -1).contiguous()
        input_embeds, block_idx = model.embed_inputs(ids_k)

        torch.manual_seed(20261114 + 101 * pi)
        lat = model.initialize_state(input_embeds, scale=1.0)
        lat_snaps[pi, :, 0, :] = lat[:, -1, :].float().cpu()

        probs_at_audit = {}
        saved_full_lat = {}

        for s in range(1, n_steps + 1):
            lat, block_idx, _ = model.iterate_one_step(input_embeds, lat, block_idx=block_idx, current_step=s - 1)
            out = model.predict_from_latents(lat)
            last_logits = out.logits[:, -1, :].float()  # [K, V]
            top1_steps[pi, :, s - 1] = last_logits.argmax(dim=-1).cpu().numpy()

            if s in audit_steps:
                a_idx = audit_steps.index(s)
                lat_snaps[pi, :, a_idx, :] = lat[:, -1, :].float().cpu()
                probs_at_audit[s] = torch.softmax(last_logits, dim=-1).cpu()
            if s in (4, 8):
                saved_full_lat[s] = lat.clone()

        # Verify exact time-invariance transplant on pi == 0: F_e^(4)(lat_4) vs lat_8
        if pi == 0:
            lat_test = saved_full_lat[4].clone()
            b_idx_test = torch.tensor(1, dtype=torch.long)
            for u in range(4):
                lat_test, b_idx_test, _ = model.iterate_one_step(
                    input_embeds, lat_test, block_idx=b_idx_test, current_step=4 + u
                )
            transplant_max_diff = float((lat_test.float() - saved_full_lat[8].float()).abs().max().item())

        # Compute symmetric KL between fork pairs at audit steps
        for s, pr in probs_at_audit.items():
            lp = (pr + 1e-12).log()
            kls = []
            for m1 in range(k_forks):
                for m2 in range(m1 + 1, k_forks):
                    kl12 = float((pr[m1] * (lp[m1] - lp[m2])).sum().item())
                    kl21 = float((pr[m2] * (lp[m2] - lp[m1])).sum().item())
                    kls.append(0.5 * (kl12 + kl21))
            kl_pairs_by_step[s].append(float(np.mean(kls)))

        # Causal Fork Swap & Leave-One-Out Cache Mean at t in {4, 8}
        for t_sw in (4, 8):
            lat_t = saved_full_lat[t_sw]  # [K, L, D]
            # Leave-one-out cache estimate for fork 0: mean of forks 1..K-1
            c_hat_0 = lat_t[1:].float().mean(dim=0, keepdim=True)  # [1, L, D]
            # Donor fork 1 residual relative to forks {2, 3} mean:
            c_hat_1 = lat_t[2:].float().mean(dim=0, keepdim=True)
            r_donor_1 = lat_t[1:2].float() - c_hat_1

            lat_clean0 = lat_t[0:1]
            lat_cache0 = c_hat_0.to(lat_t.dtype)
            lat_swap0 = (c_hat_0 + r_donor_1).to(lat_t.dtype)
            e1 = input_embeds[0:1]

            nxt_clean, _, _ = model.iterate_one_step(e1, lat_clean0, current_step=t_sw)
            nxt_cache, _, _ = model.iterate_one_step(e1, lat_cache0, current_step=t_sw)
            nxt_swap, _, _ = model.iterate_one_step(e1, lat_swap0, current_step=t_sw)

            p_cl = int(model.predict_from_latents(nxt_clean).logits[0, -1].argmax().item())
            p_ca = int(model.predict_from_latents(nxt_cache).logits[0, -1].argmax().item())
            p_sw = int(model.predict_from_latents(nxt_swap).logits[0, -1].argmax().item())
            cache_only_match_rates[t_sw].append(float(p_ca == p_cl))
            swap_flip_rates[t_sw].append(float(p_sw != p_cl))

            if t_sw == 4:
                steady_tok = int(top1_steps[pi, 0, -1])
                for rname, l_init in [
                    ("Fresh", model.initialize_state(e1, scale=1.0)),
                    ("CacheMean", lat_cache0),
                    ("Full", lat_clean0),
                ]:
                    cur = l_init.clone()
                    preds_r = [int(model.predict_from_latents(cur).logits[0, -1].argmax().item())]
                    for u in range(1, 29):
                        cur, _, _ = model.iterate_one_step(e1, cur, current_step=u - 1)
                        preds_r.append(int(model.predict_from_latents(cur).logits[0, -1].argmax().item()))
                    ex = np.array([p == steady_tok for p in preds_r], dtype=bool)[:, None]
                    # Lock-in step to steady-state token
                    k = len(preds_r) - 1
                    if not ex[k, 0]:
                        tau = len(preds_r)
                    else:
                        while k > 0 and ex[k - 1, 0]:
                            k -= 1
                        tau = k
                    resume_steps_to_steady[4][rname].append(int(tau))

    # Compute Law-of-Total-Variance path-dependent fraction across prompts at each audit step
    # Total variance across (prompt, fork) vs within-prompt cross-fork variance 0.5 * E[||r^A - r^B||^2]
    path_var_share_by_step = {}
    fork_cosine_by_step = {}
    top1_agree_by_step = {}

    for ai, s in enumerate(audit_steps):
        vecs = lat_snaps[:, :, ai, :]  # [N_p, K, D]
        grand_mean = vecs.mean(dim=(0, 1), keepdim=True)
        total_var = float((vecs - grand_mean).pow(2).mean().item())

        pair_sq = []
        pair_cos = []
        agree = []
        for m1 in range(k_forks):
            for m2 in range(m1 + 1, k_forks):
                v1, v2 = vecs[:, m1, :], vecs[:, m2, :]
                pair_sq.append(0.5 * float((v1 - v2).pow(2).mean().item()))
                pair_cos.append(float(F.cosine_similarity(v1, v2, dim=-1).mean().item()))
                if s >= 1:
                    agree.append(float((top1_steps[:, m1, s - 1] == top1_steps[:, m2, s - 1]).mean()))

        path_var_share_by_step[str(s)] = float(np.mean(pair_sq) / max(total_var, 1e-12))
        fork_cosine_by_step[str(s)] = float(np.mean(pair_cos))
        if s >= 1:
            top1_agree_by_step[str(s)] = float(np.mean(agree))

    t0_h = np.array(resume_steps_to_steady[4]["Fresh"], dtype=np.float64)
    tc_h = np.array(resume_steps_to_steady[4]["CacheMean"], dtype=np.float64)
    tf_h = np.array(resume_steps_to_steady[4]["Full"], dtype=np.float64)

    payload = {
        "model": "tomg-group-umd/huginn-0125",
        "snapshot_commit": "bb6621b65e90b6a4b9b29ef88dc83866d450470c",
        "elapsed_sec": float(time.time() - t0),
        "n_prompts": n_p,
        "k_forks": k_forks,
        "n_steps": n_steps,
        "stationarity_transplant_linf_diff": transplant_max_diff,
        "path_dependent_variance_share_upper_bound_by_step": path_var_share_by_step,
        "cross_fork_latent_cosine_by_step": fork_cosine_by_step,
        "cross_fork_top1_token_agreement_by_step": top1_agree_by_step,
        "cross_fork_symmetric_kl_by_step": {str(k): float(np.mean(v)) for k, v in kl_pairs_by_step.items()},
        "fork_swap_and_cache_mean": {
            "step_4_cache_mean_next_top1_match_rate": float(np.mean(cache_only_match_rates[4])),
            "step_4_donor_R_swap_top1_flip_rate": float(np.mean(swap_flip_rates[4])),
            "step_8_cache_mean_next_top1_match_rate": float(np.mean(cache_only_match_rates[8])),
            "step_8_donor_R_swap_top1_flip_rate": float(np.mean(swap_flip_rates[8])),
        },
        "resume_to_step32_steady_token_at_t4": {
            "mean_tau_fresh": float(t0_h.mean()),
            "mean_tau_cache_mean": float(tc_h.mean()),
            "mean_tau_full": float(tf_h.mean()),
            "accrued_cache_loops": float((t0_h - tc_h).mean()),
            "price_R_loops": float((tc_h - tf_h).mean()),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
