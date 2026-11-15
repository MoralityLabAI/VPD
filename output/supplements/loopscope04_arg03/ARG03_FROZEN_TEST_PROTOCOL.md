# Frozen Held-Out Test Protocol: `ARG-03-N` (`Nanbeige/Nanbeige4.2-3B`)

**Status:** FROZEN PRE-TEST SPECIFICATION — **HELD-OUT `test.jsonl` ($N = 168$) REMAINS STRICTLY SEALED (`0` RUNS).**  
**Date Locked:** 15 November 2026  
**Branch:** `research/causal-dark-fibers-arg03-20261115` (Draft PR #47)  
**Companion Documents:** [`CAUSAL_QUOTIENT_ARG03.md`](CAUSAL_QUOTIENT_ARG03.md) (theory & design), [`DARK_FIBERS_CAMPAIGN_ADDENDUM.md`](DARK_FIBERS_CAMPAIGN_ADDENDUM.md) (Phase 1–3 campaign synthesis), [`campaign_20261115/arg03_dev_calibration.json`](campaign_20261115/arg03_dev_calibration.json) (frozen development calibration lock).  
**Public Mirror (`HTTP 200`, `MoralityLabAI/VPD`):** [https://github.com/MoralityLabAI/VPD/blob/main/output/supplements/loopscope04_arg03/ARG03_FROZEN_TEST_PROTOCOL.md](https://github.com/MoralityLabAI/VPD/blob/main/output/supplements/loopscope04_arg03/ARG03_FROZEN_TEST_PROTOCOL.md)

> **Authorization Gate:** Do **not** execute `arg03_nanbeige --permit-heldout-test` on `test.jsonl` until a human reviewer has inspected [`DARK_FIBERS_CAMPAIGN_ADDENDUM.md`](DARK_FIBERS_CAMPAIGN_ADDENDUM.md) and this frozen protocol, and has explicitly authorized unsealing `test.jsonl` for a single confirmatory pass.

---

## 1. Provenance & Cryptographic Locks

All code, model weights, split digests, and calibration parameters below were locked using **only** `development.jsonl` ($N = 96$). As verified in [`campaign_20261115/campaign_summary.json`](campaign_20261115/campaign_summary.json) (`arg03_nanbeige.heldout_test_seal_verified = true`), zero forward passes or grading evaluations have touched `test.jsonl`.

| Artifact / Parameter | Locked Value / SHA-256 |
| --- | --- |
| Base model & revision | `Nanbeige/Nanbeige4.2-3B` @ `451ed48c3273ecef7ea8faaa43c31ce529763bb1` (`bfloat16`, 22 layers) |
| `ARG-02` `checkpoint_lock.json` SHA-256 | `87e338f112fd7b304aeece22f7da268ee0f7b3b45d932553d45cc7258cef78d5` |
| `mechinterp/dark_fiber_gyms/arg03_nanbeige.py` SHA-256 | `b1de797bb7ca285db682f34409844ec55ae899963a8ea9ab89c69c19c1e9547f` |
| `EvalMath` `algebraic_reasoning_oracle.py` SHA-256 | `c0df77faae6351e522f35d36fa6031d6270ecced2fd92d0477cb5e08992ccb0b` |
| Excluded `ARG-02` `splits.json` SHA-256 | `c4d1f8afd77da814ad146b1ab6e5fdea36dfbe6338589be602007e452f87e128` |
| Public `development.jsonl` ($N = 96$, seed `2026111503`) SHA-256 | `7dec3fc1cc1d78f5a133410d53f484ea9d7f908ed38232ace9ff4c2e60518e79` |
| Public **SEALED** `test.jsonl` ($N = 168$, seed `2026111504`) SHA-256 | `ddd580212da5a5ea4f5fea740b515dd191b39e7a501d3e31b45944a666e7c8b5` |
| Contamination overlaps (`EvalMath`, `ARG-02` dev/shots, dev-vs-test) | `0` across all 8 signature & source-graph checks (`audit.json`) |
| Structural program split rule | `development`: final operator $\neq \texttt{R}$ (`signed`, `product`, `sparse`); `test`: final operator $= \texttt{R}$ (`signed`, `product`, `sparse`, `rational`, `squared`, `twisted_product`, `lacunary`) |
| Recurrent loop depths evaluated | $t \in \{1, 2, 4, 8\}$ (trained native anchor: $t = 2$) |
| Context-fiber intervention window & scale | `width = 8` prefix-context tokens before candidate continuation; $\sigma = 0.02$; RNG `seed = 20261115` |
| Pre-continuation readout tolerance | Hard runtime assertion `max_precontinuation_*_logprob_error <= 1e-6` across full vocabulary (`0.00e+00` achieved across all $96/96$ dev cases) |

---

## 2. Frozen Development-Only Calibrations

Locked in [`campaign_20261115/arg03_dev_calibration.json`](campaign_20261115/arg03_dev_calibration.json) (`schema: darkfibers.arg03.dev_calibration_lock.v1`, fitted on `development.jsonl`, $N = 96$):

1. **Frozen scalar full-vocabulary temperature (`selected_full_vocab_scalar_temp`):** `"T0.35"`  
   - Selected from the 10-point grid $T \in \{0.25, 0.35, 0.50, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0\}$.  
   - On `development.jsonl`, `"T0.35"` (tied at `27/96 = 28.1%` `future-correct@12` with `T0.5` and `T0.75`, vs. `26/96` at `T1`) achieved the highest `accepted@12` (`33/96`).
2. **Frozen per-position operator-site temperatures (`selected_per_position_op_temps`):** `[0.35, 0.5, 2.0]`  
   - Selected from $6^3 = 216$ tuples $(T_1, T_2, T_3) \in \{0.35, 0.5, 0.75, 1.0, 1.5, 2.0\}^3$ applied solely to the 4 operator tokens $\{\texttt{I}, \texttt{N}, \texttt{S}, \texttt{R}\}$ at each of the 3 program positions on single-pass `loop 1` operator-site logits.  
   - **Critical development discovery motivating the held-out test:** In-sample on `development.jsonl`, `[0.35, 0.5, 2.0]` lifted single-pass `loop 1` `future-correct@12` from `26/96` (`27.1%`) to **`43/96` (`44.8%`, McNemar $p = 0.00151$)**, matching `baseline@2` (`42/96 = 43.8%`, $p = 1.00$) and `baseline@4` (`43/96 = 44.8%`), while `top1_future_correct` dropped from `7/96` to `1/96`.  
   - Because `development.jsonl` holds out final operator `R` (all true programs end in $\{\texttt{I}, \texttt{N}, \texttt{S}\}$) whereas `test.jsonl` ($N = 168$) consists exclusively of programs whose final operator **is** `R` across seven algebraic families, applying the **frozen** `[0.35, 0.5, 2.0]` vector without refitting on `test.jsonl` cleanly separates:
     - **(a) Genuine closed-loop iterative computation** (`baseline@2` / `baseline@4` generalizes to final-operator-`R` programs while frozen `[0.35, 0.5, 2.0]` degrades out-of-sample), versus
     - **(b) Position-wise readout sharpening / prior flattening** (`dev_calibrated_loop1@per_pos_op` retains parity with `baseline@2` out-of-sample).
3. **Reference development candidate margins (`dev_median_top1_top2_margin_by_depth`):**  
   `{"1": 0.2704, "2": 1.1590, "4": 0.3750, "8": 0.4381}` nats (recorded as descriptive diagnostics; not tuned on test).

---

## 3. Preregistered Hypotheses & Decision Rules on `test.jsonl` ($N = 168$)

Multiplicity across the **three primary confirmatory accuracy comparisons (H1a, H1b, H1c)** is controlled at family-wise $\alpha = 0.05$ via two-sided exact McNemar tests with **Holm–Bonferroni** step-down adjustment.

### Primary Family H1 — Closed-Loop Recurrence at Native Depth ($t = 2$) vs. Single-Pass & No-Feedback Controls

Anchored at native trained depth $t = 2$ (`baseline@2`) on `future-correct@12` ($N = 168$):

- **H1a (Recurrence vs. uncalibrated single pass):** `baseline@2` vs. `baseline@1` (exact two-sided McNemar).
- **H1b (Closed-loop feedback vs. equal-compute / open-loop):** `baseline@2` vs. the stronger of `open_loop@2` and `equal_compute@2` (conservative min-gap comparison; both individual McNemar $p$-values reported).
- **H1c (Closed-loop recurrence vs. frozen per-position single-pass calibration):** `baseline@2` vs. frozen `dev_calibrated_loop1@per_pos_op` (`[0.35, 0.5, 2.0]`) **and** frozen `temperature_grid@T0.35` (conservative min-gap comparison; both reported).

**Decision Rule for H1:**
- **Strong closed-loop computation confirmed** iff H1a, H1b, and H1c all reject in favor of `baseline@2` under Holm–Bonferroni ($\alpha = 0.05$).
- **Closed-loop feedback confirmed, readout-reweighting equivalence unrefuted** iff H1a and H1b pass ($\alpha = 0.05$) while H1c (`baseline@2` vs. frozen `dev_calibrated_loop1@per_pos_op`) fails to reach significance; in that case, retain the virtual-depth / position-wise readout-sharpening caveat from [`CAUSAL_QUOTIENT_ARG03.md`](CAUSAL_QUOTIENT_ARG03.md) §6.
- Depth $t = 4$ (`baseline@4` vs. `baseline@2`, `open_loop@4`, `equal_compute@4`, and `dev_calibrated_loop1@per_pos_op`) is pre-specified as **secondary confirmatory** (evaluated only if H1a passes, or reported with explicit secondary label).

### Primary Continuous & Secondary Task-Level Family H2 — Exact Context-Fiber Observability vs. Operator-Subspace Stability

Evaluated at $t = 4$ (where development continuous separation peaked at $+9.72$ log-prob $L_2$, $p = 2.15 \times 10^{-5}$) and $t = 2$, subject to hard pre-continuation parity (`max_precontinuation_donor_logprob_error <= 1e-6` at $t = 1$):

- **H2a (Continuous delayed fiber observability — confirmatory):** Paired tail full-vocabulary log-probability $L_2$ divergence $\|q(F_x^t(s_{\text{swap}})) - q(F_x^t(s_{\text{clean}}))\|_2 > \|q(F_x^t(s_{\text{rand}})) - q(F_x^t(s_{\text{clean}}))\|_2$ at $t = 4$ (two-sided exact sign test and Wilcoxon signed-rank at $\alpha = 0.01$).
- **H2b (Task-level `future-correct@12` causal necessity under local $\sigma = 0.02$, `width = 8` swaps — preregistered expected null):** Paired exact McNemar test between `context_swap@4` and `random_context_norm_matched@4` (and at $t = 2$). Based on `development.jsonl` (`46/96` vs. `44/96`, $p = 0.625$; 64-program Spearman $\rho = 0.9950$), **we preregister the expectation that H2a passes while H2b is null**, confirming that local 8-token context fibers are *continuously observable across loops* ($q \circ F_x \neq T_x \circ q$ on full-vocabulary logits) yet *attractor-buffered inside the 4-token operator subspace* $\{\texttt{I}, \texttt{N}, \texttt{S}, \texttt{R}\}$ at $\sigma = 0.02$. A claim of task-level dark-fiber causal necessity on `ARG-03-N` is allowed **only** if H2b rejects at two-sided exact McNemar $p < 0.025$.

---

## 4. Exact One-Shot Execution Commands (For Post-Approval Use Only)

Once authorized by the user, execute **exactly once** in two isolated OS processes (model inference first without `EvalMath` in `sys.modules`; isolated `EvalMath` grading second with `--apply-dev-calibration` and **without** `--fit-dev-calibration`):

```powershell
$C = "C:/path/to/Compactification-Theory"
$E = "C:/path/to/EvalMath"

# Step 1 (RTX 5080, isolated from EvalMath oracle): Run locked Nanbeige4.2-3B on test.jsonl (N = 168)
& "$HOME/.venvs/bvg-v03/Scripts/python.exe" -m mechinterp.dark_fiber_gyms.arg03_nanbeige `
  --compactification-root $C `
  --public-jsonl "runs/dark-fibers/arg03/public/test.jsonl" `
  --out-dir "runs/dark-fibers/arg03/test_full168" `
  --full `
  --permit-heldout-test `
  --sigma 0.02 `
  --width 8 `
  --seed 20261115 `
  --device cuda:0

# Step 2 (Isolated CPU grading process): Apply frozen development calibration without refitting
& "$HOME/.venvs/bluebeam/Scripts/python.exe" -m mechinterp.dark_fiber_gyms.arg03_roster grade `
  --evalmath-root $E `
  --compactification-root $C `
  --split-dir "runs/dark-fibers/arg03/public" `
  --split test `
  --full `
  --tape "runs/dark-fibers/arg03/test_full168/answer_tape.jsonl" `
  --apply-dev-calibration "papers/shared_weights_dark_fibers/campaign_20261115/arg03_dev_calibration.json" `
  --out "runs/dark-fibers/arg03/test_full168/grade_test168.json"
```
