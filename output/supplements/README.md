# Public Supplements for `LOOPSCOPE-02`, `LOOPSCOPE-03`, and `BPD-01`

Because the development repository `MoralityLabAI/LoopedTransformers` is private, this directory mirrors the pre-registrations, verbatim git commit patches, evaluation reports, experimental scripts, unit tests, and JSON result receipts for **`LOOPSCOPE-02`**, **`LOOPSCOPE-03`**, and **`BPD-01`** in the public `MoralityLabAI/VPD` repository so every citation and empirical audit is anonymously verifiable (`HTTP 200`).

## 0. `BPD-01` — Behavioral Parameter Decomposition (`mechinterp/bpd` Mirror)

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
