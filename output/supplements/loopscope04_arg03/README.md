# Public Supplement: `LOOPSCOPE-04` & `ARG-03-N` Causal Dark Fibers Campaign

Public (`HTTP 200`) mirror of the post-`v014` local mechanistic-interpretability campaign from `MoralityLabAI/LoopedTransformers` (`research/causal-dark-fibers-arg03-20261115`, Draft PR #47, commit `938fd72`).

- **Companion `v014` manuscript (`PAPER.md`, unchanged):** [`../../paper_versions/v014_2026-11-14_enforced_path_independence_and_scaling/PAPER.md`](../../paper_versions/v014_2026-11-14_enforced_path_independence_and_scaling/PAPER.md)
- **Empirical Campaign Addendum (with inline Figures 6 & 7):** [`DARK_FIBERS_CAMPAIGN_ADDENDUM.md`](DARK_FIBERS_CAMPAIGN_ADDENDUM.md)
- **Frozen Preregistered Held-Out Test Protocol (`ARG-03-N` `test.jsonl`, $N = 168$ SEALED):** [`ARG03_FROZEN_TEST_PROTOCOL.md`](ARG03_FROZEN_TEST_PROTOCOL.md)
- **Theoretical Rationale & Protocol Design Note (Propositions 1–4):** [`CAUSAL_QUOTIENT_ARG03.md`](CAUSAL_QUOTIENT_ARG03.md)

## Provenance-Locked Receipts for Independent Recomputation

- **Unified Statistical Summary (`15,933` lines):** [`campaign_20261115/campaign_summary.json`](campaign_20261115/campaign_summary.json)
- **Frozen `ARG-03-N` Development Calibration Lock:** [`campaign_20261115/arg03_dev_calibration.json`](campaign_20261115/arg03_dev_calibration.json)
- **`ARG-03-N` (`Nanbeige4.2-3B`, `development.jsonl` $N = 96$; `test.jsonl` $N = 168$ strictly sealed with `0` runs):**
  - Aggregate graded table: [`results/arg03_dev96/grade_dev96.json`](results/arg03_dev96/grade_dev96.json)
  - Per-case binary & continuous paired diagnostics ($N = 96$ rows across all arms, depths, distances, and rankings): [`results/arg03_dev96/per_case_diagnostics_dev96.json`](results/arg03_dev96/per_case_diagnostics_dev96.json)
  - Run & split audit manifests: [`results/arg03_dev96/manifest.json`](results/arg03_dev96/manifest.json), [`results/arg03_dev96/public_audit.json`](results/arg03_dev96/public_audit.json)
- **`LOOPSCOPE-04` (`gaoxin492` primary & `yagizdevre` replication 7M Sudoku `TRM`, 24 GPU receipts):**
  - Representative stratum lookup (`40 WB-Prop + 40 WB-Stall_partial + 40 S4_stalled = 120`): [`results/loopscope04/rep_eval_strata.json`](results/loopscope04/rep_eval_strata.json)
  - 3A Exact-readout fiber swaps ($t \in \{2, 4, 8\}$): [`gaoxin492_rep_swap_t2_h32.json`](results/loopscope04/gaoxin492_rep_swap_t2_h32.json), [`gaoxin492_rep_swap_t4_h32.json`](results/loopscope04/gaoxin492_rep_swap_t4_h32.json), [`gaoxin492_rep_swap_t8_h32.json`](results/loopscope04/gaoxin492_rep_swap_t8_h32.json), [`yagizdevre_rep_swap_t2_h32.json`](results/loopscope04/yagizdevre_rep_swap_t2_h32.json), [`yagizdevre_rep_swap_t4_h32.json`](results/loopscope04/yagizdevre_rep_swap_t4_h32.json), [`yagizdevre_rep_swap_t8_h32.json`](results/loopscope04/yagizdevre_rep_swap_t8_h32.json)
  - 3B Snapshot vs. trace caches (`SeqRefill97` $N=360$ + cellwise ridge $N \in \{90, 180, 360\}$): [`gaoxin492_rep_cache_full97_n360.json`](results/loopscope04/gaoxin492_rep_cache_full97_n360.json), [`yagizdevre_rep_cache_full97_n360.json`](results/loopscope04/yagizdevre_rep_cache_full97_n360.json), [`gaoxin492_rep_cache_ridge_n90.json`](results/loopscope04/gaoxin492_rep_cache_ridge_n90.json), [`gaoxin492_rep_cache_ridge_n180.json`](results/loopscope04/gaoxin492_rep_cache_ridge_n180.json), [`yagizdevre_rep_cache_ridge_n90.json`](results/loopscope04/yagizdevre_rep_cache_ridge_n90.json), [`yagizdevre_rep_cache_ridge_n180.json`](results/loopscope04/yagizdevre_rep_cache_ridge_n180.json)
  - 3C Delayed observability Gramians $G_{4,4}$ & finite-angle sweeps ($\beta \in \{0.04, 0.15, 0.35, 0.75\}$): [`gaoxin492_rep_gramian_t4_p8_k4_h12.json`](results/loopscope04/gaoxin492_rep_gramian_t4_p8_k4_h12.json), [`yagizdevre_rep_gramian_t4_p8_k4_h12.json`](results/loopscope04/yagizdevre_rep_gramian_t4_p8_k4_h12.json) (plus pilot & dev Gramians in [`results/loopscope04/`](results/loopscope04/))
- **Figures, Gyms, Tests, & Verbatim Patch:**
  - Figures 6 & 7 (`.png` and `.pdf`): [`figures/`](figures/)
  - Gym runners, statistical aggregator, figure builder, and 18-test suite: [`gyms/`](gyms/)
  - Verbatim `LoopedTransformers` campaign diff patch (`26ae0cc..bf8feca`): [`campaign_diff_26ae0cc_to_bf8feca.patch`](campaign_diff_26ae0cc_to_bf8feca.patch)
