# BPD-01: CUDA BF16 numerical closure, native online-SVD comparison and causal CI refit | 2026-11-17

Branch `research/bpd01-cuda-bf16-closure-20261117`, cut from `feat/mechinterp-bpd` at
`92f93219` (PRs #156, #158, #159, #160 merged).

Hardware and software: NVIDIA GeForce RTX 5080 Laptop GPU (16 GB, 150 W limit), torch
2.11.0+cu128, TF32 matmul off, `allow_bf16_reduced_precision_reduction` True (default).
Jobs ran under vramorama leases of 8 GB.

Checkpoints verified by SHA-256 before every run:

| Artifact | SHA-256 |
|---|---|
| Stage-4 factors `bpd_b4_v2_stage4_weights.pt` | `65ecb2ab6c4a144b3563cecf49d6a55b157201fa98ef0b6dc652ff3effc33cbc` |
| gaoxin492 `step_21700` | `932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3` |
| yagizdevre `model.pt` | `d67723394a7fe439f1fa498b58196399ee89cd82a794c7da9ce0a8cc3df7fa39` |
| `benchmark_bundle.json` | `5f9e262e63623e11ee0e1a431224e007497eca140c0b02bd0b85ff54b871f70f` |

The 16 boards are the first 16 WB-Stall evaluation boards and 63 invocations, read out at 20,
41 and 62. KL is the token-mean KL(reference || arm) over 81 cells and 9 digits. Full
per-board values for every arm are in the JSON receipts and in
[`results/v2/bpd01_tables.md`](../results/v2/bpd01_tables.md).

## Commands

```powershell
python mechinterp/bpd/scripts/audit_b4_v2_controls_and_ablation_split.py --device cuda --out mechinterp/bpd/results/v2/gate_b4_v2_closure_controls_corrected.json
python mechinterp/bpd/scripts/audit_b4_v2_closure_mechanism.py --device cuda --out mechinterp/bpd/results/v2/gate_b4_v2_closure_mechanism.json
python mechinterp/bpd/scripts/audit_b4_v2_online_svd.py --device cuda --out mechinterp/bpd/results/v2/gate_b4_v2_online_svd.json
python mechinterp/bpd/scripts/audit_b4_v2_online_svd.py --device cuda --precision fp32 --out mechinterp/bpd/results/v2/gate_b4_v2_online_svd_cuda_fp32.json
python mechinterp/bpd/scripts/refit_b4_ci_frozen_factors.py --checkpoint gaoxin492 --steps 480 --batch-size 16 --seed 0
python mechinterp/bpd/scripts/refit_b4_ci_frozen_factors.py --checkpoint yagizdevre --steps 480 --batch-size 16 --seed 0
python mechinterp/bpd/scripts/diagnose_causal_ci_train_boards.py --checkpoint gaoxin492   # CPU
python mechinterp/bpd/scripts/diagnose_causal_ci_train_boards.py --checkpoint yagizdevre  # CPU
python mechinterp/bpd/scripts/audit_b4_causal_ci.py --device cuda
python mechinterp/bpd/scripts/train_b4_ci_closed_loop.py --checkpoint gaoxin492
python mechinterp/bpd/scripts/train_b4_ci_closed_loop.py --checkpoint yagizdevre
python mechinterp/bpd/scripts/audit_b4_causal_ci.py --device cuda --ci-dir mechinterp/bpd/results/v2/ci_closed_loop --ci-stem seed0_closed_loop_ci --experiment B4-CAUSAL-CI-02 --registration mechinterp/bpd/docs/B4_CAUSAL_CI_CLOSED_LOOP_REGISTRATION_20261118.md --out mechinterp/bpd/results/v2/gate_b4_causal_ci_closed_loop.json
python mechinterp/bpd/scripts/reduce_bpd01.py
```

Experiments A and B ran through `vramorama run --vram 8G` with `HF_HUB_OFFLINE=1`. The
refits used 14 GB leases, the closed-loop trainings 7 GB each, and each causal-CI scoring
run a 10 GB lease.

## Experiment A: the identity controls under native CUDA BF16

The registered controls script reproduced the historical receipt exactly where the arms
overlap. For example, the FP32-linear arm gives 0.309060 and 0.263795 at pass 62, and the
trained identity arm gives 0.365314 and 1.534241.

| Control vs native BF16 target | gaoxin492 p20 mean / max | gaoxin492 p62 mean / max (boards > 1e-3) | yagizdevre p20 mean / max | yagizdevre p62 mean / max (boards > 1e-3) |
|---|---|---|---|---|
| 1 same-dtype repeat | 0 / 0 | 0 / 0 (0) | 0 / 0 | 0 / 0 (0) |
| 2 target FP32 linear | 0.0031 / 0.0158 | 0.3091 / 4.0286 (4) | 0.0029 / 0.0095 | 0.2638 / 2.5376 (6) |
| 3 full FP32, original weights | 0.0038 / 0.0279 | 0.3159 / 4.3801 (4) | 0.0082 / 0.0409 | 0.7184 / 11.0780 (5) |
| 4 FP32, BF16-quantized weights | 0.0037 / 0.0278 | 0.0530 / 0.6914 (2) | 0.0048 / 0.0160 | 0.7306 / 11.1868 (5) |
| 5 SVD all 512, residual off | 0.0072 / 0.0514 | 0.0541 / 0.3787 (3) | 0.0080 / 0.0440 | 0.8186 / 6.7567 (5) |
| 6 SVD all 512, residual on | 0.0050 / 0.0386 | 1.0015 / 13.1288 (4) | 0.0036 / 0.0131 | 0.5024 / 6.7987 (5) |
| 7 trained all, residual on | 0.0059 / 0.0315 | 0.3653 / 5.3172 (4) | 0.0047 / 0.0337 | 1.5342 / 11.0486 (6) |

SVD weight reconstruction, measured independently: the relative Frobenius residual of the
full-rank SVD start is 3.19e-7 to 3.47e-7 across the four sites on both checkpoints.

### Identified failure mechanism: BF16 rounding accumulated through the recurrence on unconverged boards

`audit_b4_v2_closure_mechanism.py` separates the three candidate causes. Its decision rules
are in the script docstring. One rule changed after a 2-board smoke run, before the 16-board
execution. Raw per-element ulp error is unbounded at near-zero outputs: there, even the
correctly rounded FP64 product differs from cuBLAS BF16 by hundreds of ulps. The rule now
uses an effective ulp floored at the token RMS, plus relative error against FP64.

1. **No implementation mismatch.** Every identity path was teacher-forced on the clean
   target's own inputs, at every invocation and site. All of them, including the correctly
   rounded FP64 product, stay within 1.0 effective BF16 ulp of the native output. Their mean
   relative error against the FP64 product equals the native BF16 matmul's own,
   1.657e-3 to 1.663e-3, identical to four significant figures. The component paths
   change the BF16 output bits slightly more often than FP32 arithmetic does: 0.032–0.073%
   of elements against 0.021–0.054%. No path is less accurate than native.
2. **Arithmetic precision is sufficient to close the controls in FP32.** The FP32 rerun of the
   three identity arms (SVD residual off, SVD residual on, trained residual on) stays below
   the 1e-5 per-board tolerance at every readout, on all 16 boards, for both checkpoints and
   both weight sources:

   | Worst board identity KL | p20 | p41 | p62 |
   |---|---:|---:|---:|
   | gaoxin492, FP32, BF16-rounded weights | 3.4e-10 | 1.4e-9 | 3.2e-6 |
   | gaoxin492, FP32, original weights | 6.3e-10 | 1.5e-9 | 1.1e-6 |
   | yagizdevre, FP32, BF16-rounded weights | 7.2e-10 | 3.6e-8 | 2.3e-8 |
   | yagizdevre, FP32, original weights | 2.5e-10 | 5.1e-7 | 1.8e-7 |

   The FP32 target is within 3.7e-7 (gaoxin492) and 1.2e-8 (yagizdevre) worst-board KL
   at pass 62 of an FP64 rollout, with RMSNorm patched to run in FP64 too. The native BF16
   target is not. Its KL from the FP64 rollout of the same weights is 0.0044 / 0.0061 mean
   at pass 20 and 0.050 / 0.241 mean (max 0.46 / 3.05) at pass 62.
3. **No chaotic amplification of isolated perturbations.** A single seeded relative kick of
   size 1e-12, 1e-9, 2^-24 or 1e-5 to one `layers.0.o_proj` invocation (0 or 21) of the FP64
   model moves no board above KL 1e-3 at any readout. A single BF16-sized kick (2^-9) also
   moves none. The median state divergence decays, for example from 4.1e-4 to 8e-6 by
   invocation 62. Injecting 2^-9 noise at every invocation of that one site moves 2–4
   boards above 1e-3 at pass 62. Rounding only the four decomposed sites' outputs to BF16,
   in an otherwise FP64 rollout, does the same: up to 0.85 KL at pass 62 on gaoxin492.
4. **The affected boards are the unconverged ones.** In the exact FP64 trajectory, gaoxin492
   leaves four boards unsolved at pass 41: {0, 2, 7, 11}. Those are exactly its
   BF16-divergent boards, and exactly the boards that the persistent-noise FP64 run moves.
   Their minimum cell logit margins at pass 41 are 0.08–0.93, against 55–67 on the twelve
   solved boards. For yagizdevre the divergent boards {2, 3, 7, 10, 11, 13, 14} are all unsolved at
   pass 41. Solved boards stay at KL ~0 under every arm.
5. The cuBLAS reduced-precision-reduction flag has no effect: the BF16 target is
   bit-identical with it off.

Conclusion: the late-pass identity failure is arithmetic-precision noise. BF16 rounding
(~2^-9 relative) enters at every operation of every invocation. On boards whose search has
not converged, the recurrence accumulates it until near-tied digit commitments flip.
The component path and the native path are two equally accurate BF16 discretizations of
the same exact computation, and on those boards they take different branches. The
decomposition is not mis-implemented. The dynamics are not chaotic for isolated
perturbations.

### Gate consequences

- Under the 1e-5 per-board identity tolerance, CUDA BF16 does **not** close at any readout.
  Even pass 20 has identity-arm board KL up to 0.051, at mean 0.003–0.008. All BF16
  measurements at passes 41 and 62 are **uninterpretable for mechanism classification**. At
  pass 20, BF16 effects smaller than about 0.01 mean KL cannot be separated from rounding.
- **CUDA FP32 closes** the identity controls at passes 20, 41 and 62 on all 16 boards. It is
  the regime in which late-pass measurements become interpretable.
- The registered B4 final-KL criterion of 0.15 lies below the BF16 identity divergence, so
  it cannot be evaluated in BF16. Gate B4 stays INCOMPLETE: stages 5a–8c have not run.

### Corrected zero+sphere arm and the ablation factorial

Both `mean_comp_acts` and `mean_delta_out` are now zeroed for the zero+sphere arm. All four
cells use the same frozen clean-informed top-307 masks (CUDA BF16, mean KL; paired
contrasts over 16 boards with bootstrap 95% CI):

| Cell | gaoxin492 p20 | yagizdevre p20 |
|---|---:|---:|
| zero, no sphere | 0.0201 | 0.0062 |
| zero, sphere (corrected) | 0.2683 | 0.3965 |
| zero, sphere (historical, confounded) | 0.2748 | 0.4187 |
| pass-mean, no sphere | 0.4436 | 0.1623 |
| pass-mean, sphere | 0.6216 | 0.4372 |

| Pass-20 contrast | gaoxin492 | yagizdevre |
|---|---|---|
| sphere effect, zero fill | +0.248 [0.153, 0.353], 16/16 boards up | +0.390 [0.227, 0.594], 16/16 |
| mean fill, no sphere | +0.424 [0.209, 0.722], 16/16 | +0.156 [0.076, 0.267], 15/16 |
| sphere effect, mean fill | +0.178 [0.049, 0.316] | +0.275 [0.020, 0.572] |
| interaction | −0.070 [−0.178, 0.034] | −0.116 [−0.313, 0.082] |

The confound in the historical zero+sphere arm was small at pass 20, about 0.007 and 0.022.
Two corrections follow for the previous reading that "mean replacement drives the main
damage":
- Sphere rescaling on its own damages every board on both checkpoints.
- On yagizdevre its effect is larger than mean fill's.

The rescale pulls each token's write norm 40% of the way toward the population-mean write
RMS for that pass. That change is large even when the kept components carry most of the
write. These are BF16 values. The no-sphere zero arm is close to the BF16 identity floor on
yagizdevre (0.0062 vs 0.0080). The sphere and mean-fill contrasts are 20–60x that floor.

## Experiment B: online vs clean-informed SVD top-307

Static rank-307, clean-informed dynamic top-307 and online top-307 were compared with zero
ablation, residual off and no sphere. The online selector reads only the current perturbed
input of each wrapped module. Clean masks feed the Hamming telemetry only. A new arm records
the in-evaluator identity floor: all 512 components on, residual off.

| Pass 20 | floor mean / max | static S | clean-informed C | online O | O − C [95% CI] | online better / worse | M upper 97.5% |
|---|---|---:|---:|---:|---|---|---:|
| gaoxin492, CUDA BF16 | 0.0072 / 0.0514 | 0.0329 | 0.0201 | 0.0095 | −0.0106 [−0.0305, 0.0014] | 10 / 6 | −8.3e-4 |
| yagizdevre, CUDA BF16 | 0.0080 / 0.0440 | 0.0060 | 0.0062 | 0.0032 | −0.0030 [−0.0070, 1.5e-4] | 10 / 6 | −4.9e-4 |
| gaoxin492, CUDA FP32 | 7.5e-11 / 3.5e-10 | 0.0285 | 0.0092 | 0.0074 | −0.0017 [−0.0052, 6.0e-4] | 9 / 7 | −1.3e-3 |
| yagizdevre, CUDA FP32 | 5.7e-11 / 2.4e-10 | 7.2e-4 | 1.55e-4 | 1.68e-4 | +1.3e-5 [−8.9e-6, 3.8e-5] | 6 / 10 | −4.9e-6 |
| gaoxin492, CPU FP32, reserved 64 (PR #160) | | | 0.005518 | 0.005986 | | | |
| yagizdevre, CPU FP32, reserved 64 (PR #160) | | | 0.000162 | 0.000147 | | | |

| Pass 62 | floor mean / max | static S | clean-informed C | online O | O − C [95% CI] |
|---|---|---:|---:|---:|---|
| gaoxin492, CUDA BF16 | 0.0541 / 0.3787 | 0.0996 | 0.0629 | 0.0245 | −0.0384 [−0.1011, −1e-28] |
| yagizdevre, CUDA BF16 | 0.8186 / 6.7567 | 0.0990 | 0.4603 | 0.5074 | +0.0471 [−0.0375, 0.1820] |
| gaoxin492, CUDA FP32 | 5.0e-9 / 5.0e-8 | 0.9273 | 0.8861 | 0.9249 | +0.0387 [0.0073, 0.0767] |
| yagizdevre, CUDA FP32 | 3.6e-9 / 4.5e-8 | 0.4730 | 0.4265 | 0.2950 | −0.1315 [−0.4719, 0.0781] |

Mask disagreement between online and clean-informed selection is 0 at pass 0 by
construction. The mean Hamming fraction per module rises to 0.0015–0.016 at pass 20 and to
0.019–0.057 at pass 62. Component density is exactly 307/512 = 0.5996 for all three
selectors (measured for the clean-informed masks). Argmax cell flips against the clean
target at pass 20, summed over 16 × 81 cells:
- gaoxin492: BF16 floor 8, static 18, clean-informed 12, online 8; FP32 floor 0, static 19,
  clean-informed 11, online 10.
- yagizdevre: BF16 floor 10, static 5, clean-informed 11, online 6; FP32 floor 0, static 2,
  clean-informed 2, online 2.

Reading:
- **CUDA BF16.** Not interpretable at any readout, because the identity floor is not closed.
  At pass 20, yagizdevre's static, clean-informed and online KLs all lie below its own
  floor mean. Online is numerically lower than clean-informed on both checkpoints, and
  neither paired interval excludes zero.
- **CUDA FP32 (closed).** The pass-20 result reproduces the CPU FP32 reserved-64 finding.
  Online selection has no detectable cost against clean-informed selection on 16 boards.
  The reserved-64 margin statistic has a 97.5% upper bound below zero on both
  checkpoints. That is descriptive at n = 16, not a registered gate.
- **FP32 late passes.** These are now interpretable. They are dominated by one or two
  unconverged boards: gaoxin492 board 11 at KL 13.6–14.3 for every selector, and yagizdevre
  boards 10 and 14. Static rank-307 is no better than dynamic selection at pass 62. A 60%
  zero-fit selector fails on the boards where the search is still open, whichever
  selection rule it uses.

## Experiment C: refit CI gates, clean-informed vs strictly causal

### What could and could not be evaluated

The historical trained-gate receipt (final KL 0.0103159) cannot be re-scored. Its CI
networks were deleted inside `run_trm_bpd.py` before the stage-4 export, and nothing on disk
or in the VPD mirror holds them. That receipt also came from `BPDTrainer.evaluate()`, which
reads the gates from the clean target trajectory, so it is a clean-informed oracle.

In its place, a new `state_only` CI was fitted on the frozen stage-4 (V, U) under the
protocol registered at `0b6ec3f6`
([`B4_CAUSAL_CI_REFIT_REGISTRATION_20261117.md`](B4_CAUSAL_CI_REFIT_REGISTRATION_20261117.md))
before any refit step. The refit numbers are reported under their own name and do not
replace the historical receipt.

| Checkpoint | Steps x batch | s/step | Peak | Validation recon loss | CI weights SHA-256 |
|---|---|---:|---:|---:|---|
| gaoxin492 | 480 x 16, seed 0 | 14.8 | 12.98 GiB | 0.0636 | `5a3bfd99cee9d3615c176c88b8fd7cd3fc21c88fed6a8b018ff339c70cbb6410` |
| yagizdevre | 480 x 16, seed 0 | 12.8 | 12.98 GiB | 0.0442 | `bcb6c8c7d872544d868df252e6db92d7abe87ea8de0a37632a6bb907fdee074f` |

Strict causality needs a schedule, because the `state_only` gate for the first-executed
module (`layers.0.o_proj`) reads the inputs of the three modules that run after it in the
same invocation. The measured autograd gradient norm of its gate logits with respect to those
downstream inputs is 11.9–61.2 at tau = 0.
`causal_ci.py` provides two schedules:
- `preflight_target`: run invocation tau once with target weights from the current
  perturbed state, gate from those inputs, then run the masked invocation.
- `lagged`: gate from the inputs the masked modules received at tau − 1.

### Pre-scoring checks on training boards

These ran after the refits and before the evaluation-board run, on CPU, using train.csv
boards 100–101. They are receipted as `ci_refit/*_prescore_diagnostic_train_boards.json`.

- Replaying the clean-informed top-307 masks through the full causal loop (preflight, CI and
  telemetry) reproduces the open-loop T2 rollout exactly: max |Δlogit| 0.0 at passes 20, 41
  and 62 on both checkpoints. `test_causal_ci.py` locks this in on the mini TRM. The gap
  between causal and clean-informed gating is therefore not an implementation difference
  in the rollout.
- The refit CI logits are nearly tied around rank 307:
  - the gap between the 307th and 308th logit averages 0.002–0.003;
  - the spread across the 512 components is 0.36–0.46;
  - 460–503 of 512 logits are positive;
  - the mean lower-leaky gate is 0.59–0.77, around the 0.75 init bias.

  The objective trained the CI only on clean target trajectories, with minimality
  coefficients of 2e-4.

The protocol was not changed after these checks.

### Registered scoring: reserved 64, CUDA FP32 with BF16-rounded weights, TF32 off

The run took 93 s. Validity floors (maximum board KL over target repeat, SVD all-on with
residual off, and trained all-on with residual on):

| Checkpoint | pass 20 | pass 41 | pass 62 |
|---|---:|---:|---:|
| gaoxin492 | 1.6e-10 | 1.9e-6 | 2.4e-6 |
| yagizdevre | 5.7e-10 | 3.1e-7 | **7.5e-5 (> 1e-5: uninterpretable)** |

Mean KL(clean FP32 target || arm) over 64 boards. The parenthesised numbers are summed
argmax cell flips out of 5,184:

| Arm | gaoxin p20 | gaoxin p41 | gaoxin p62 | yagiz p20 | yagiz p41 | yagiz p62 (uninterp.) |
|---|---:|---:|---:|---:|---:|---:|
| S0 SVD static first 307 | 0.0291 (89) | 0.358 | 0.436 | 4.5e-4 (11) | 0.0308 | 0.122 |
| S1 SVD clean-informed magnitude | 0.0057 (43) | 0.100 | 0.703 | 1.4e-4 (11) | 0.0221 | 0.337 |
| S2 SVD online magnitude | 0.0056 (42) | 0.111 | 0.731 | 1.3e-4 (8) | 0.0164 | 0.120 |
| T1 trained factors, online magnitude | 6.570 (4599) | 6.595 | 6.586 | 4.420 (4611) | 4.323 | 4.328 |
| T2 trained, clean-informed CI top-307 | 0.134 (278) | 0.098 | 0.077 | 0.197 (342) | 0.153 | 0.143 |
| **T3 trained, causal CI top-307 (preflight)** | **5.428 (4599)** | 5.464 | 5.467 | **6.454 (4608)** | 6.202 | 6.178 |
| T4 trained, causal CI top-307 (lagged) | 5.648 (4600) | 5.705 | 5.705 | 3.903 (4602) | 4.018 | 4.029 |
| N1 continuous clean-informed CI, mean + sphere (`evaluate()` semantics) | 0.0627 (186) | 0.0584 | 0.0430 | 0.0588 (206) | 0.0426 | 0.0356 |
| N2 continuous causal CI, mean + sphere | 3.055 (4194) | 3.736 | 3.954 | 1.462 (2862) | 2.012 | 2.126 |

Registered pass-20 contrasts. Each cell gives the mean board difference, the bootstrap 95%
CI (20,000 resamples, seed 20261117301) and the board counts (T3 lower / T3 higher):

| Hypothesis | gaoxin492 | yagizdevre | Decision |
|---|---|---|---|
| H1: T3 − T1 | −1.142 [−1.204, −1.079], 64 / 0 | +2.034 [+2.001, +2.067], 0 / 64 | **inconclusive** (opposite signs) |
| H2: T3 − S2 | +5.422 [+5.384, +5.461], 0 / 64 | +6.454 [+6.414, +6.494], 0 / 64 | **T3 worse on both** |
| H3: (T3 − T2) − 0.10 · S0 | +5.291 [+5.246, +5.335] | +6.257 [+6.208, +6.304] | **non-inferiority not shown** |

Holm-adjusted one-sided sign-flip p-values (100,000 draws):
- H1: 2.0e-5 on gaoxin492, 1.0 on yagizdevre;
- H2: 1.0 on both.

The same signs hold at pass 41 on both checkpoints and at pass 62 on gaoxin492.

Secondary results (none upgrades a primary decision):

- **Clean-informed CI vs clean-informed SVD selection (T2 − S1).**
  - At pass 20, the refit CI selector is worse on both checkpoints, and on all 64 boards
    each: +0.128 [+0.111, +0.146] on gaoxin492 and +0.197 [+0.167, +0.228] on yagizdevre.
  - At gaoxin492 pass 62, T2 is lower on average: −0.626 [−1.245, −0.157]. The difference
    is concentrated: 16 boards favour T2 and 48 favour S1, and the top board carries 31% of
    the absolute effect. These are boards where SVD selection fails late, the same
    unconverged-board pattern as in Experiment B.
- **Trained factors with magnitude selection (T1).** Keeping the 307 largest
  |x V| · ‖U‖ terms of the trained factors destroys the output on every board, flipping
  about 89% of cells. The same rule on the SVD factors is near-lossless (S2). For the
  trained factors, which components are kept matters far more than each component's write
  norm, so magnitude is not a usable selector for them.
- **Mask drift under causal gating.**
  - T3 and T2 masks agree at invocation 0, as built.
  - At invocation 1 they differ on 4.2–6.1% of components (gaoxin492) and 2.9–3.7%
    (yagizdevre).
  - From invocation 20 on they differ on 22–28% (gaoxin492) and 18–35% (yagizdevre) per
    module and invocation.
  - The lagged schedule behaves the same way. T4 − T3 at pass 20 is +0.221 on gaoxin492
    and −2.551 on yagizdevre; both stay in the destroyed regime.
- **Native continuous semantics.** N1 matches `BPDTrainer.evaluate()` semantics and stays
  low: 0.06 at pass 20. Its regime, boards and CI weights all differ from the historical
  0.0103 receipt, so the two are not comparable. The causal version, N2, is 1.5–3.1 at
  pass 20. Its mean gate density per module and invocation is 0.61–0.79, the same range
  as the clean-informed N1 (0.60–0.77). The failure is in which components are gated, not
  how many.

### Answer to priority 3

The refit trained CI gates do not beat untrained SVD selection:
- Run strictly causally, they lose to zero-fit online SVD top-307 by 5.4–6.5 nats at pass
  20 on both checkpoints and on every board.
- Even with oracle access to the clean trajectory, the budget-matched CI top-307 (T2) is
  worse than SVD magnitude top-307 (S1) at pass 20.

The CI separates good from bad subsets only on the clean trajectory it was trained on:
- Its logits are nearly tied around the budget.
- About 5% of the masks flip after one masked invocation.
- The resulting state change compounds.

Limits:
- One refit seed per checkpoint, at batch 16 × 480 steps where B4 used 32 × 240.
- A clean-trajectory training objective.

The results do not cover a CI trained on its own closed-loop inputs. Experiment C2 tests
that next.

## Experiment C2: closed-loop CI training (B4-CAUSAL-CI-02)

### Protocol and training

The protocol was registered at `e1bfdf05`
([`B4_CAUSAL_CI_CLOSED_LOOP_REGISTRATION_20261118.md`](B4_CAUSAL_CI_CLOSED_LOOP_REGISTRATION_20261118.md))
before any full training. It trains the CI on the policy it is scored with:
- preflight observations of its own masked rollout;
- a hard top-307 forward mask with a straight-through sigmoid gradient;
- backpropagation through all 63 invocations;
- scheduled sampling from clean observations, with p annealed from 1 to 0 over the first
  120 steps.

Each run is warm-started from the C refit CI and trained for 240 steps at batch 16, lr 3e-4,
in FP32 with BF16-rounded weights and TF32 off. Both checkpoints trained at once under two
7 GB leases, at 13.6 s/step with a 5.93 GiB peak. The reserved 64 boards were not read during
training.

Validation, on the 32 WB-Stall boards under the exact scored policy (mean KL over passes 20,
41 and 62):

| Checkpoint | step 0 | 48 | 96 | 144 | 192 (best) | 240 | CI weights SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| gaoxin492 | 5.473 | 2.608 | 1.945 | 1.419 | 1.415 | 1.467 | `e5675fe056d1f8d50a1b4f3a6a6b52377d69ed8d01bce2c942faa7a201c15f99` |
| yagizdevre | 6.277 | 3.085 | 1.403 | 0.952 | 0.942 | 0.954 | `254e4fca5c1a3707d8af8f003d9bf8cf441a41f539ed630a4628f202c0d321a1` |

### Registered scoring

The scorer and settings match Experiment C: reserved 64, CUDA FP32. The floors are
identical, so passes 20 and 41 are interpretable on both checkpoints and pass 62 on
gaoxin492 only.

Mean KL; the parenthesised numbers are summed argmax cell flips out of 5,184. S0–S2 and T1
do not use the CI and repeat Experiment C exactly.

| Arm | gaoxin p20 | gaoxin p41 | gaoxin p62 | yagiz p20 | yagiz p41 | yagiz p62 (uninterp.) |
|---|---:|---:|---:|---:|---:|---:|
| S2 SVD online magnitude | 0.0056 (42) | 0.111 | 0.731 | 1.3e-4 (8) | 0.0164 | 0.120 |
| T3 refit CI, causal (Experiment C) | 5.428 (4599) | 5.464 | 5.467 | 6.454 (4608) | 6.202 | 6.178 |
| **T3 closed-loop CI, causal** | **1.484 (2750)** | 1.411 | 1.410 | **0.866 (1860)** | 0.970 | 1.016 |
| T2 closed-loop CI, clean-informed | 0.236 (230) | 0.207 | 0.127 | 0.149 (353) | 0.105 | 0.311 |
| T4 closed-loop CI, causal lagged | 25.66 (3212) | 26.54 | 27.37 | 2.820 (2200) | 1.984 | 1.938 |
| N1 closed-loop CI, continuous, clean-informed | 6.582 | 25.03 | 33.46 | 2.023 | 1.085 | 0.851 |
| N2 closed-loop CI, continuous, causal | 5.525 | 8.797 | 9.494 | 2.607 | 6.290 | 5.878 |

Registered pass-20 decisions (bootstrap 95% CI; board counts are new arm lower / higher):

| Hypothesis | gaoxin492 | yagizdevre | Decision |
|---|---|---|---|
| **H4: T3 closed-loop − T3 refit** | −3.944 [−3.991, −3.898], 64 / 0 | −5.588 [−5.641, −5.535], 64 / 0 | **closed-loop CI better on both** |
| H1: T3 − T1 | −5.086 [−5.136, −5.034], 64 / 0 | −3.554 [−3.600, −3.508], 64 / 0 | **T3 better on both** |
| H2: T3 − S2 | +1.478 [+1.453, +1.504], 0 / 64 | +0.866 [+0.841, +0.892], 0 / 64 | **T3 worse on both** |
| H3: (T3 − T2) − 0.10 · S0 | +1.245 [+1.192, +1.295] | +0.717 [+0.683, +0.752] | non-inferiority not shown |

H4 holds at passes 41 and 62 as well, on every board.

What changed and what did not:

- **Mask drift.** Training the CI on its own rollouts reduces the drift after one masked
  invocation. T3 vs T2 disagreement at invocation 1 falls to 1.2–2.8% on gaoxin492 and
  1.1–1.8% on yagizdevre, from 4.2–6.1% and 2.9–3.7%. The rollout no longer collapses.
- **Remaining damage.** Causal KL falls 3.7× (gaoxin492) and 7.5× (yagizdevre) but stays
  0.87–1.48. That is 36–53% of cells
  with a changed argmax at pass 20, against 0.2–0.8% for zero-fit online SVD.
- **Gap to oracle gating.** The clean-informed use of the same CI (T2) remains 0.7–1.2 nats
  better than causal use (T3 − T2 = +1.25 and +0.72 at pass 20).
- **Specialisation.** The CI is now tuned to preflight top-307. Its lagged (T4) and
  continuous (N1, N2) uses became much worse than in Experiment C: up to 27 nats for T4 on
  gaoxin492. A CI trained for one observation and selection rule does not transfer to the
  others.

### Answer to priority 3, updated

Closed-loop training is the right fix for the causal failure, which is confirmed by H4 on
every board. Even so, trained gates on the stage-4 factors still do not beat untrained SVD
selection. Causal BPD at top-307 is worse than zero-fit online SVD by 0.87–1.48 nats at
pass 20 on both checkpoints.

Part of the gap belongs to the factors, not the gate. Magnitude selection on the trained
factors (T1) is destroyed (4.4–6.6) while the same rule on SVD factors is near-lossless. The
stage-4 components behave as if they rely on cancellation between kept and dropped terms.

Limits: one seed per checkpoint, 240 steps, and the validation curve flattens after 144
steps.

## Registered-gate decisions

| Item | Decision |
|---|---|
| BF16 same-dtype identity (repeat) | Exactly 0 on every board and readout |
| BF16 algebraic identity controls, 1e-5 per board | FAIL at passes 20, 41, 62 on both checkpoints |
| FP32 algebraic identity controls, 1e-5 per board | PASS at passes 20, 41, 62 on both checkpoints and both weight sources |
| BF16 passes 41 and 62 | Uninterpretable for mechanism classification |
| Online-SVD protocol (16-board CUDA BF16, descriptive) | Executed; no interpretable difference; identity floor dominates |
| B4-CAUSAL-CI-01 floors (CUDA FP32, reserved 64) | Closed at passes 20 and 41 on both checkpoints and at pass 62 on gaoxin492; yagizdevre pass 62 uninterpretable (7.5e-5) |
| B4-CAUSAL-CI-01 H1, causal CI vs magnitude top-307 on the same factors | Inconclusive (gaoxin492 T3 better, yagizdevre T3 worse; both arms KL 4.4–6.6) |
| B4-CAUSAL-CI-01 H2, causal BPD vs zero-fit online SVD | T3 worse on both checkpoints |
| B4-CAUSAL-CI-01 H3, causal CI non-inferior to clean-informed CI | Not shown |
| Historical trained-gate receipt 0.0103159 | Not re-evaluable (CI weights deleted before export); clean-informed by construction |
| B4-CAUSAL-CI-02 H4, closed-loop vs refit CI under causal top-307 | Closed-loop CI better on both checkpoints (−3.94, −5.59 KL at pass 20; 64/64 boards each) |
| B4-CAUSAL-CI-02 H2, causal BPD vs zero-fit online SVD | T3 worse on both checkpoints (+1.48, +0.87) |
| B4-CAUSAL-CI-02 H1 / H3 | H1 T3 better than magnitude top-307 on both; H3 not shown |
| Gate B4 | INCOMPLETE (stages 5a–8c not run; 0.15 final-KL criterion untestable in BF16) |
