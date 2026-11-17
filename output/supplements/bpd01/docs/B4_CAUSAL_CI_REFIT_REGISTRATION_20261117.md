# B4-CAUSAL-CI-01: frozen-factor CI refit and strictly causal gate evaluation | 2026-11-17

Registered before any CI refit step beyond a timing and memory pilot, and before any
evaluation-board inference with a refit CI. The pilot ran gaoxin492 only, 2 steps at
batch 32 and then 5 steps at batch 16. It wrote no files. Its validation numbers are not
used.

## Why a refit

The historical trained-gate receipt (final KL 0.0103159, `gate_b4_v2_stage4_receipt.json`)
cannot be re-evaluated. Its CI networks were deleted inside `run_trm_bpd.py` before the
stage-4 export. The receipt records `includes_ci_networks: false`, and no CI weights exist
on disk or in the VPD mirror. The receipt also came from
`BPDTrainer.evaluate()`, which computes `ci_net.forward_trajectory(clean_traj)` before the
masked rollout. Its gates were read from the clean target trajectory, so it is a
clean-informed oracle.

This protocol fits a NEW CI on the frozen stage-4 factors. The refit **does not replace or
re-score the historical receipt**. Its numbers are reported under their own name.

## Within-pass dependency (measured)

Inside one `run_l_level` invocation the wrapped modules execute in the order
`layers.0.o_proj -> layers.0.down_proj -> layers.1.o_proj -> layers.1.down_proj`, measured
with forward pre-hooks (`causal_ci.module_execution_order`). The CI network reads the
modules in sorted order, which differs. The `state_only` CI concatenates the pre-weight
inputs of all four modules at the same invocation. The gate for `layers.0.o_proj`
therefore depends on the inputs of three modules that execute after it and that depend on
its own masked write. `test_causal_ci.py` checks this with a nonzero autograd dependence.
Those features do not exist at the moment the first gated module runs, so
`BPDTrainer.evaluate()`'s gating cannot run closed-loop as written.

Causal observation schedules (`mechinterp/bpd/causal_ci.py`):

- **preflight_target** (primary). Before invocation tau runs, execute that invocation once
  with target weights from the current perturbed state and record the four module inputs.
  Compute gates from them, discard the preflight output, then run the masked invocation.
  This uses only `(z_main, z_inj)` at tau. At tau = 0 its gates equal the clean-informed
  gates.
- **lagged** (secondary). Gate invocation tau with the inputs the masked modules actually
  received at tau − 1. It needs no preflight and is one invocation stale.

Tests check three properties: no lookahead (a truncated rollout reproduces the prefix
exactly), reference masks used as telemetry only, and all-open causal gates reproducing the
target rollout.

## Refit (calibration), fixed

- Factors: `bpd_b4_v2_stage4_weights.pt`, SHA-256
  `65ecb2ab6c4a144b3563cecf49d6a55b157201fa98ef0b6dc652ff3effc33cbc`, keys `bpd_ref`
  (gaoxin492) and `bpd_tgt` (yagizdevre). V and U are frozen (`requires_grad=False`).
- Models: SHA-256-verified pretrained gaoxin492 and yagizdevre, loaded in BF16 on CUDA as in
  B4 stage 1a.
- CI: `state_only`, d_model 128, 1 block, 4 heads, MLP 256, max_passes 134. This is the
  stage-1a architecture.
- Objective: `train_ci_heads_on_frozen_components` restricted to `state_only`:
  - level/switch minimality, pool mean, coefficients 2e-4 / 2e-4;
  - 0.5 deterministic plus 0.5 stochastic trajectory reconstruction;
  - per-pass fragility hinge, kappa 0.1;
  - rotation-null baselines from 2 rotations at scale 0.10;
  - pass-mean plus sphere ablation, residual 0.
  The means and null baselines are calibrated on the first training batch.
- Optimizer: AdamW at lr 1e-3, cosine decay to 0.1x, **480 steps at batch 16**.
  B4 used 240 × 32. A 32-board step peaks at 17.1 GiB on the 16 GB card, so the batch is
  halved and the steps doubled to keep 7,680 board-rollouts. The best state is kept by the
  validation objective, checked every 96 steps.
- Seed 0, one fit per checkpoint.
- Board roles:
  - training: first 128 `train.csv` boards, the B4 factor-training set, disjoint from
    WB-Stall;
  - validation, for best-state selection only: 32 WB-Stall boards ranked by
    `sha256("BPD01-CI-REFIT-VAL-20261117|{idx:03d}")` among indices 16..182 outside the
    reserved 64;
  - evaluation: the reserved 64 (registration
    `B4_ONLINE_SVD_RESERVED64_REGISTRATION_20261117.md`, list SHA-256
    `0f474925d83f86839b411c2f1286c5beec5b3630b40275f299e66a267ce97ab0`). No refit step reads
    them.
- Command, per checkpoint:
  `python mechinterp/bpd/scripts/refit_b4_ci_frozen_factors.py --checkpoint <ckpt> --steps 480 --batch-size 16 --seed 0`.
  It writes the CI weights, their SHA-256 and a JSON receipt to
  `mechinterp/bpd/results/v2/ci_refit/`. The weights are committed.

## Evaluation, fixed

- Regime: CUDA FP32 with the BF16-rounded checkpoint weights, TF32 off. These are the
  weights the factors were fitted to. BPD-01 Experiment A showed this regime closes the
  identity controls at passes 20, 41 and 62.
- Boards: reserved 64, in four chunks of 16. Invocations 0..62, readouts 20, 41, 62.
  **Pass 20 is primary.**
- Budget-matched arms. Each keeps k = 307 of 512 components per module, token and
  invocation, with zero ablation, residual off and no sphere:

  | Arm | Factors | Selection |
  |---|---|---|
  | S0 | SVD | static first 307 |
  | S1 | SVD | clean-informed top-307 by abs(x V) · norm(U) on clean inputs |
  | S2 | SVD | online top-307 by the same score on the current perturbed input |
  | T1 | trained | online top-307 by abs(x V) · norm(U) on the current input |
  | T2 | trained | clean-informed CI top-307 by CI logit, CI read from the clean trajectory |
  | T3 | trained | causal CI top-307, preflight_target |
  | T4 | trained | causal CI top-307, lagged (secondary) |

- CI-native arms (secondary): pass-mean plus sphere, calibrated on the 128 training boards,
  continuous lower-leaky CI masks, residual off. N1 is clean-informed CI, which is
  `BPDTrainer.evaluate` semantics. N2 is causal preflight CI.
- Validity floors:
  - the target repeat;
  - SVD with all components on and residual off;
  - trained factors with all components on and residual on.
  A readout is interpretable only if every floor is <= 1e-5 KL on every board in that
  chunk set.

## Hypotheses and decision rules (pass 20)

KL_i is the token-mean KL(clean FP32 target || arm) on board i. Bootstrap the board mean
with 20,000 PCG64 resamples, seed 20261117301, separately per checkpoint. A two-checkpoint
claim needs both checkpoints, which keeps the Bonferroni level at 0.025 each.

- **H1, gate value beyond untrained selection on the same factors (primary).**
  D1_i = KL(T3)_i − KL(T1)_i. "Trained causal CI improves on magnitude selection" if the
  97.5% upper bound of mean D1 < 0 on both checkpoints. "Worse" if the 2.5% lower bound > 0
  on both. Otherwise inconclusive.
- **H2, full causal BPD vs zero-fit SVD online.** D2_i = KL(T3)_i − KL(S2)_i, same rule.
- **H3, cost of causality for the trained CI.**
  M3_i = (KL(T3)_i − KL(T2)_i) − 0.10 · KL(S0)_i. Non-inferior if the 97.5% upper bound of
  mean M3 < 0 on both checkpoints.

Secondary, all reported:
- passes 41 and 62 where the floors close;
- lagged vs preflight;
- N2 vs N1;
- T2 vs S1, the clean-informed CI vs SVD selector;
- one-sided sign-flip p-values (100,000 draws) with Holm adjustment across checkpoints;
- wins and losses, effect concentration, argmax flips;
- T3 vs T2 mask disagreement per module and invocation;
- realized CI density under N1 and N2.

Report every per-board value. A secondary result cannot upgrade a primary decision. A single
refit seed limits conclusions to this refit. It is not a statement about CI training in
general.
