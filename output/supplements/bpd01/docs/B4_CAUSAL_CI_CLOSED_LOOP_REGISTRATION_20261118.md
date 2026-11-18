# B4-CAUSAL-CI-02: closed-loop-trained CI for strictly causal top-k gating | 2026-11-18

This protocol was registered before any full training run and before any evaluation-board
inference with a closed-loop CI. Three pilots preceded it, all on gaoxin492:
- a GPU run of 4 steps at batch 16, which peaked at 5.9 GiB with activation checkpointing;
- a GPU step-timing profile on 16 training boards;
- a CPU run of 2 steps at batch 2, stopped before it finished.

None of them wrote files, and none of their numbers are used.

The profile measured 27–45 s per step at batch 16. The workload is bound by kernel launches,
and other processes held the CPU at about 97% load during the profile. That fixes 240
steps, with the two checkpoints trained at the same time under separate 7 GB leases.

## Motivation

In B4-CAUSAL-CI-01 (report `BPD01_CUDA_BF16_CLOSURE_20261117.md`, receipt
`gate_b4_causal_ci.json`), the refit CI gating causally at top-307 (T3) destroyed the output:
KL 5.4 on gaoxin492 and 6.5 on yagizdevre at pass 20. It lost to zero-fit online SVD on every
board.

The pre-scoring diagnostics located the cause:
- That CI was trained only on clean target trajectories.
- Its logits are nearly tied at rank 307: the gap between the 307th and 308th logit is 0.002.
- Replaying its clean-informed masks through the causal loop is exact, so the rollout is
  not at fault.
- Its gates on perturbed inputs reshuffle, about 5% of masks after one masked invocation,
  and the error compounds.

The fix tested here trains the CI on the policy it is scored with.

## Training, fixed

Code: `mechinterp/bpd/causal_ci_train.py`. Script: `scripts/train_b4_ci_closed_loop.py`.

- **Factors:** stage-4 (V, U), SHA-256
  `65ecb2ab6c4a144b3563cecf49d6a55b157201fa98ef0b6dc652ff3effc33cbc`, frozen. All target
  model parameters are frozen too.
- **Initialisation:** the B4-CAUSAL-CI-01 refit CI for the same checkpoint:
  - gaoxin492 `5a3bfd99cee9d3615c176c88b8fd7cd3fc21c88fed6a8b018ff339c70cbb6410`;
  - yagizdevre `bcb6c8c7d872544d868df252e6db92d7abe87ea8de0a37632a6bb907fdee074f`.

  The architecture is unchanged: `state_only`, d_model 128, 1 block, 4 heads, MLP 256.
- **Regime:** CUDA FP32 with the BF16-rounded checkpoint weights, TF32 off. This matches the
  scoring.
- **Policy:**
  - Before invocation tau, a preflight with target weights from the current masked state
    records the four module inputs.
  - The CI keeps the top k = 307 logits per module, token and invocation, with zero
    ablation and residual off.
  - The forward mask is the hard top-k, so it equals the scored T3 mask bit for bit.
  - The backward gradient is straight-through via `sigmoid(logit)`.
  - Observation features are detached. Gradients run through the masked recurrence over all
    63 invocations, and each invocation is activation-checkpointed.
- **Scheduled sampling:** each board's observation at tau is the clean target input with
  probability p, otherwise the preflight input. p falls linearly from 1 to 0 over the first
  50% of steps and stays at 0 after that.
- **Loss:** the mean over readouts 20, 41 and 62 of the token-mean KL(clean || closed-loop),
  plus 1.0 × the mean over the 9 H invocations of the relative squared error of z_H
  (‖z_H − z_H^clean‖² / ‖z_H^clean‖², token-averaged).
- **Optimiser:** AdamW, lr 3e-4, weight decay 0, cosine decay to 0.1×, gradient clip 1.0,
  **240 steps at batch 16**, seed 0, one run per checkpoint.
- **Boards:** batches are drawn from a seeded per-epoch permutation of the first 128
  train.csv boards.
- **Validation:** best-state selection only, on the same 32 hash-ranked WB-Stall boards as
  B4-CAUSAL-CI-01. It runs the exact scored policy (`causal_ci_rollout`, preflight, top-307)
  at step 0 and every 48 steps, and keeps the state with the lowest mean KL over passes 20,
  41 and 62.
- **Evaluation boards** (the reserved 64): not read by any training or validation step.
- **Command:** `python mechinterp/bpd/scripts/train_b4_ci_closed_loop.py --checkpoint <ckpt>`
  (defaults as above). It writes `results/v2/ci_closed_loop/{ckpt}_seed0_closed_loop_ci.pt`
  and a JSON receipt with the full train and validation history. The weights are committed.

## Evaluation, fixed

The B4-CAUSAL-CI-01 scorer is unchanged, pointed at the new weights:

`python mechinterp/bpd/scripts/audit_b4_causal_ci.py --device cuda --ci-dir mechinterp/bpd/results/v2/ci_closed_loop --ci-stem seed0_closed_loop_ci --experiment B4-CAUSAL-CI-02 --registration mechinterp/bpd/docs/B4_CAUSAL_CI_CLOSED_LOOP_REGISTRATION_20261118.md --out mechinterp/bpd/results/v2/gate_b4_causal_ci_closed_loop.json`

Everything else matches B4-CAUSAL-CI-01:
- regime, boards (the reserved 64, chunks of 16), readouts and k;
- arms S0, S1, S2, T1, T2, T3, T4, N1, N2;
- floors, and the 1e-5 interpretability rule.

T2, T3, T4, N1 and N2 now use the closed-loop CI. S0, S1, S2 and T1 do not depend on the CI.

## Hypotheses and decision rules (pass 20 primary)

Per checkpoint, bootstrap the board mean with 20,000 PCG64 resamples. A two-checkpoint
claim needs both checkpoints.

- **H4, does closed-loop training fix causal gating (primary).**
  D4_i = KL(T3 closed-loop)_i − KL(T3 refit, `gate_b4_causal_ci.json`)_i, on the same 64
  boards, bootstrap seed 20261118401.
  - "Closed-loop CI better" if the 97.5% upper bound of mean D4 < 0 on both checkpoints.
  - "Worse" if the 2.5% lower bound > 0 on both.
  - Otherwise inconclusive.
  - Computed by `reduce_bpd01.py`.
- **H2 (re-tested): causal BPD vs zero-fit SVD online.** D2_i = KL(T3)_i − KL(S2)_i. The
  rule and seed (20261117301) are those of B4-CAUSAL-CI-01, computed by the scorer.
- **H1 and H3:** the B4-CAUSAL-CI-01 rules, unchanged, on the new CI.

Secondary, all reported:
- passes 41 and 62 where the floors close;
- the lagged schedule (T4);
- the clean-informed use of the closed-loop CI (T2);
- N1 and N2 (continuous gates, which this CI was not trained for);
- T3 vs T2 mask disagreement;
- training and validation curves.

"Fixed" in the report means H4 better on both checkpoints. Beating untrained SVD selection
is the separate claim, H2. One training seed per checkpoint limits conclusions to this
run.
