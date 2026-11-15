# Public Supplements for `LOOPSCOPE-04` / `ARG-03-N`, `LOOPSCOPE-03`, `LOOPSCOPE-02`, and `BPD-01`

Because the development repository `MoralityLabAI/LoopedTransformers` is private, this directory mirrors the pre-registrations, verbatim git commit patches, evaluation reports, experimental scripts, unit tests, and JSON result receipts for **`LOOPSCOPE-04` & `ARG-03-N`**, **`LOOPSCOPE-03`**, **`LOOPSCOPE-02`**, and **`BPD-01`** in the public `MoralityLabAI/VPD` repository so every citation and empirical audit is anonymously verifiable (`HTTP 200`).

## 0A. `LOOPSCOPE-04` & `ARG-03-N` — Causal Quotients, History-Conditioned Caches, Delayed Observability, and Frozen `Nanbeige4.2-3B` Test Protocol

- **Mirrored from `MoralityLabAI/LoopedTransformers` (`research/causal-dark-fibers-arg03-20261115`, Draft PR #47, commits `938fd72` / `bf8feca`):**
  - Supplement overview & receipt index: [`loopscope04_arg03/README.md`](loopscope04_arg03/README.md)
  - Empirical campaign addendum (with inline `fig6` & `fig7`): [`loopscope04_arg03/DARK_FIBERS_CAMPAIGN_ADDENDUM.md`](loopscope04_arg03/DARK_FIBERS_CAMPAIGN_ADDENDUM.md)
  - Frozen preregistered held-out test protocol (`ARG-03-N` `test.jsonl`, $N = 168$ **strictly sealed**, `0` runs): [`loopscope04_arg03/ARG03_FROZEN_TEST_PROTOCOL.md`](loopscope04_arg03/ARG03_FROZEN_TEST_PROTOCOL.md)
  - Theoretical rationale & design note (Propositions 1–4): [`loopscope04_arg03/CAUSAL_QUOTIENT_ARG03.md`](loopscope04_arg03/CAUSAL_QUOTIENT_ARG03.md)
  - Unified statistical summary & frozen dev calibration: [`loopscope04_arg03/campaign_20261115/campaign_summary.json`](loopscope04_arg03/campaign_20261115/campaign_summary.json), [`loopscope04_arg03/campaign_20261115/arg03_dev_calibration.json`](loopscope04_arg03/campaign_20261115/arg03_dev_calibration.json)
  - Per-board `LOOPSCOPE-04` receipts (24 JSONs + `rep_eval_strata.json`) & per-case `ARG-03-N` `dev_full96` diagnostics (`grade_dev96.json` + `per_case_diagnostics_dev96.json`): [`loopscope04_arg03/results/`](loopscope04_arg03/results/)
  - Figures, gym runners, 18-test suite, & verbatim campaign diff patch: [`loopscope04_arg03/figures/`](loopscope04_arg03/figures/), [`loopscope04_arg03/gyms/`](loopscope04_arg03/gyms/), [`loopscope04_arg03/campaign_diff_26ae0cc_to_bf8feca.patch`](loopscope04_arg03/campaign_diff_26ae0cc_to_bf8feca.patch)

## 0B. `BPD-01` — Behavioral Parameter Decomposition (`mechinterp/bpd` Mirror)

- **Mirrored from `MoralityLabAI/LoopedTransformers` (`feat/mechinterp-bpd`, initial commit `e6baa744b82d91cd2812a73799c5d1aea5de2ce8`):**
  - Package overview: [`bpd01/README.md`](bpd01/README.md)
  - Pre-registration document: [`bpd01/docs/registration.md`](bpd01/docs/registration.md)
  - Design note: [`bpd01/docs/design_and_positioning.md`](bpd01/docs/design_and_positioning.md)
  - Positive-control test run history: [`bpd01/docs/positive_control_history.md`](bpd01/docs/positive_control_history.md)
  - Verbatim initial commit patch: [`bpd01/initial_commit_e6baa744.patch`](bpd01/initial_commit_e6baa744.patch)
  - Source modules, adapters, scripts, and tests: [`bpd01/`](bpd01/)

## 1. `LOOPSCOPE-03` — Three-Ledger Iteration Pricing, Conditional Cache Refill, Enforced Path Independence, and Data Scaling

- **Pre-registration (`b1ec77550997cbc5ae2cd84618af74e8f34e38e7`, committed `2026-11-14T08:28:03+08:00` prior to held-out inference):**
  - Pre-registration document: [`loopscope03/docs/registration.md`](loopscope03/docs/registration.md)
  - Verbatim git commit patch: [`loopscope03/preregistration_commit_b1ec7755.patch`](loopscope03/preregistration_commit_b1ec7755.patch)
- **Evaluation report (Stages 1–4, including `v014` post-review data-size scaling, covariate-shift / periodic-refill audit, uncapped `Resume` estimators, and unconditional $K=8$ enforced path-independence analysis):**
  - Evaluation report: [`loopscope03/docs/report.md`](loopscope03/docs/report.md)
- **JSON receipts (`loopscope03/results/`):**
  - [`loopscope03/results/stage1_conditional_refill.json`](loopscope03/results/stage1_conditional_refill.json)
  - [`loopscope03/results/stage2_3_resume_and_fork.json`](loopscope03/results/stage2_3_resume_and_fork.json)
  - [`loopscope03/results/huginn_cache_and_fork.json`](loopscope03/results/huginn_cache_and_fork.json)
  - [`loopscope03/results/stage4_path_independence_and_scaling.json`](loopscope03/results/stage4_path_independence_and_scaling.json)
- **Scripts & invariant tests:** [`loopscope03/scripts/`](loopscope03/scripts/), [`loopscope03/tests/test_loopscope03.py`](loopscope03/tests/test_loopscope03.py)

## 2. `LOOPSCOPE-02` — Deductively Stratified 9×9 Sudoku Triangulation, Manifold Controls, and History Probes

- **Pre-registration (`1f205d67f940ec198dcbd7f7854ab101e932e283`, committed `2026-11-13T17:46:36+08:00` prior to evaluation commit `1c479cc109e2896b05dc4cfc978d99a8dd3e79ce` at `2026-11-13T19:23:15+08:00`):**
  - Pre-registration document: [`loopscope02/docs/registration.md`](loopscope02/docs/registration.md)
  - Verbatim git commit patch: [`loopscope02/preregistration_commit_1f205d67.patch`](loopscope02/preregistration_commit_1f205d67.patch)
- **Evaluation report:** [`loopscope02/docs/report.md`](loopscope02/docs/report.md)
- **JSON receipts, scripts, and tests:** [`loopscope02/results/`](loopscope02/results/), [`loopscope02/scripts/`](loopscope02/scripts/), [`loopscope02/tests/test_loopscope02.py`](loopscope02/tests/test_loopscope02.py)
