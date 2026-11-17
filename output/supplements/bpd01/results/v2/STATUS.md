# BPD v2 Execution Status Summary

All `v2` gate runs are tied to pre-registration commit `180f1bcbbc633a16bfd887a9593252cefd7a6a65` (`docs/registration.md`).

| Gate | Verdict | Receipt | Notes |
| :--- | :--- | :--- | :--- |
| **Gate B1 v2** | **PASS** | `gate_b1_v2_receipt.json` | Like-for-like unhinged (`hinge_kappa = 0.0`, 3 seeds) on `StationaryRoutedLooper` (`C = 8`, `2` active/item); tied KL `2.36e-5` vs unrolled `TxC` `5.11e-5` (`0.462x`) and budget-matched `0.0307` (`0.00077x`). |
| **Gate B2 v2** | **FAIL** | `gate_b2_v2_receipt.json` | 8-factor `o_proj + down_proj` negative arm passed (`min_p_rank = 0.030 > 0.00625`, `K = 199`), while `0.5x–4.0x` natural-scale `sigma` sweep failed `p_rank <= 0.01` under the Haar read-rotation null. |
| **Gate B3 v2** | **FAIL** | `gate_b3_v2_receipt.json` | Nested `state_only -> state_plus_clock` refits eliminated negative clock inversions across all 4 checkpoints (`3` CI seeds, `C = 256`), while `lam1_s1, s2 > lam1_s3` (`2 SD`) and the `0.5x–4.0x` planted `sigma` sweep failed. |
| **Gate B4 v2** | **INCOMPLETE** | `gate_b4_v2_stage4_receipt.json` | Completed Stages `1a`, `1b`, `2`, `3a`, `3b`, and `4` (`41,255.7s` elapsed) and stopped at the start of Stage `5a` without running Stages `5a–8c`. Live Stage `1a–4` metrics and `(V, U)` factor matrices (`bpd_b4_v2_stage4_weights.pt`, SHA-256 `65ecb2ab6c4a144b3563cecf49d6a55b157201fa98ef0b6dc652ff3effc33cbc`; `ci_net` weights not present) were exported via `scripts/extract_stage4_from_pid.py`. |
