# BPD v2 Execution Status Summary

All `v2` gate runs are tied to pre-registration commit `180f1bcbbc633a16bfd887a9593252cefd7a6a65` (`docs/registration.md`).

| Gate | Verdict | Receipt | Notes |
| :--- | :--- | :--- | :--- |
| **Gate B1 v2** | **PASS** | `gate_b1_v2_receipt.json` | Like-for-like unhinged (`hinge_kappa = 0.0`, 3 seeds) on `StationaryRoutedLooper` (`C = 8`, `2` active/item); tied KL `2.36e-5` vs unrolled `TxC` `5.11e-5` (`0.462x`) and budget-matched `0.0307` (`0.00077x`). |
| **Gate B2 v2** | **FAIL** | `gate_b2_v2_receipt.json` | 8-factor `o_proj + down_proj` negative arm passed (`min_p_rank = 0.030 > 0.00625`, `K = 199`), while `0.5x–4.0x` natural-scale `sigma` sweep failed `p_rank <= 0.01` under the Haar read-rotation null. |
| **Gate B3 v2** | **FAIL** | `gate_b3_v2_receipt.json` | Nested `state_only -> state_plus_clock` refits eliminated negative clock inversions across all 4 checkpoints (`3` CI seeds, `C = 256`), while `lam1_s1, s2 > lam1_s3` (`2 SD`) and the `0.5x–4.0x` planted `sigma` sweep failed. |
| **Gate B4 v2** | **INCOMPLETE** | `gate_b4_v2_stage4_receipt.json` | Completed Stages `1a`, `1b`, `2`, `3a`, `3b`, and `4` (`41,255.7s` elapsed) and stopped at the start of Stage `5a` without running Stages `5a–8c`. Live Stage `1a–4` metrics and `(V, U)` factor matrices (`bpd_b4_v2_stage4_weights.pt`, SHA-256 `65ecb2ab6c4a144b3563cecf49d6a55b157201fa98ef0b6dc652ff3effc33cbc`; `ci_net` weights not present) were exported via `scripts/extract_stage4_from_pid.py`. |

## BPD-01 follow-up (2026-11-17, branch `research/bpd01-cuda-bf16-closure-20261117`)

Report: `docs/BPD01_CUDA_BF16_CLOSURE_20261117.md`. Tables: `bpd01_tables.md`, `bpd01_summary.json`.

| Item | Result | Receipt |
| :--- | :--- | :--- |
| Corrected B4 closure controls, CUDA BF16, 16 boards | Same-dtype repeat exactly 0. All seven identity controls fail the 1e-5 per-board tolerance from pass 20, with max board KL 0.051 at pass 20 and up to 13.1 at pass 62 on 3-7 boards. | `gate_b4_v2_closure_controls_corrected.json` |
| Failure mechanism | BF16 arithmetic noise, accumulated through the recurrence on boards still unsolved at pass 41. There is no implementation mismatch: teacher-forced paths are within 1 effective ulp and match native error vs FP64. FP32 identity closes, with worst-board KL <= 3.2e-6 at pass 62. Isolated FP64 perturbations do not grow. | `gate_b4_v2_closure_mechanism.json` |
| BF16 passes 41 and 62 | Uninterpretable for mechanism classification | |
| Online vs clean-informed SVD top-307, CUDA BF16 | Differences lie inside the identity floor at every readout | `gate_b4_v2_online_svd.json` |
| Online vs clean-informed SVD top-307, CUDA FP32 | Identity floor <= 5e-8. Online is non-inferior at pass 20 on both checkpoints (descriptive, n = 16). | `gate_b4_v2_online_svd_cuda_fp32.json` |
| Trained CI (historical) | CI weights deleted before the stage-4 export. The 0.0103159 receipt is clean-informed and cannot be re-scored. | `gate_b4_v2_stage4_receipt.json` |
| B4-CAUSAL-CI-01 refit CI, CUDA FP32, reserved 64 | Floors closed except yagizdevre pass 62 (7.5e-5). Pass-20 decisions: H1 inconclusive (opposite signs); H2 causal CI top-307 worse than zero-fit online SVD on both checkpoints (+5.42, +6.45 KL); H3 non-inferiority to clean-informed CI not shown. The clean-informed CI top-307 also trails SVD magnitude top-307 at pass 20 (+0.13, +0.20). | `ci_refit/`, `gate_b4_causal_ci.json` |

Gate B4 v2 remains **INCOMPLETE**.
