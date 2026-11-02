# HRM-Text-1B PrimeLab Cheap-Executor Handoff V1

**Purpose:** execute the next experiment faithfully with a cheaper model after the user authorizes a remote stage.  
**Current state:** local forensics and protocol drafting are complete. No Prime Lab job has been launched.

## Read first

1. C:\projects\VPD\HRM_TEXT_1B_FULL_VPD_RECOVERY_AUDIT_V1.md
2. C:\projects\VPD\HRM_TEXT_1B_VPD_RL_POLICY_PROTOCOL_V1.md
3. C:\projects\VPD\HRM_TEXT_1B_PRIMELAB_EXECUTION_PACKET_V1.json
4. D:\Research_Engine\prime_lab\AGENTS.md
5. The applicable instructions under D:\Research_Engine\prime_lab\.prime\skills and D:\Research_Engine\prime_lab\environments

The JSON packet is normative if this handoff and the packet differ.

## Do not make these mistakes

- Do not call the recovered overnight_v2 model_20 files valid VPD. Their learned state is non-finite.
- Do not call the low_rank_svd feature map trained VPD.
- Do not infer that status completed means numerical validation passed.
- Do not reuse the old split-module loader without proving that it preserves the fused trained HRM weights.
- Do not run the RL policy against proxy attention features and report it as the full-VPD experiment.
- Do not let VPD select LoRA modules or update coordinates. VPD is an observation channel only.
- Do not launch, spend, mutate weights, or reveal the final pack without the required explicit authorization.

## Exact current finding

The May 19 full run covered paired L/H layer indices 0–15. It created 16 ComponentModel checkpoints totaling 12,434,920,496 bytes. All 16 final metric rows are step 20 and all 16 have train/loss/total equal to NaN. All 16 load logs report both unused original weights and newly initialized replacement weights.

The inspected layer-14 terminal checkpoint has:

- 262,669,824 finite target-model elements;
- 4,128,768 component elements, all non-finite;
- 121,766,912 CI-function elements, all non-finite;
- SHA-256 20D87DE733A447EC88C124DDDFAE608CA86DE361076AB29F0E963F24EBD5B1C3 on both D and E.

The older unpruned, one-step, layer-0 smoke checkpoint is finite. Use it only to verify parsers and receipt generation.

## Run ladder

| Gate | Run ID | Work | Proceed only if |
|---|---|---|---|
| 0 | LOCAL-AUDIT-V1 | Inventory and finiteness audit | Completed; historical full artifact quarantined |
| 1 | PL-HRM-BASE-CONFORMANCE-V1 | Native fused load and fixed-logit equivalence | No missing/unexpected/new weights; exact tokens; fp32 max logit error at most 1e-5 |
| 2 | PL-VPD-NUMERICS-PILOT-V1 | Layer-14, 100-step A/B/C optimizer pilot | Finite throughout; positive control and reconstruction gates pass |
| 3 | PL-VPD-FULL-FUSED-V1 | Sixteen paired-layer ComponentModel jobs | Complete finite manifest, stability, reconstruction, and ablation receipts |
| 4 | PL-VPD-OBS-CACHE-V1 | Freeze true, shuffled, and matched-random observations | Same schema and no outcome/final-pack leakage |
| 5 | PL-RL-ONE-STEP-V1 | Four-arm, four-seed, one-action policy test | True VPD beats all matched controls without regression |
| 6 | PL-RL-SEQUENTIAL-V1 | Six macro-steps, at least eight seed blocks | Registered grouped acceptance gate passes |
| 7 | PL-HELDOUT-REVEAL-V1 | One sealed reveal | Frozen artifacts and complete arm-by-seed table exist |

Stop at the first failed gate. Do not “rescue” a failed run by changing parameters after seeing its outcome; issue a new versioned protocol and new sealed pack where applicable.

## Prime Lab implementation shape

Use a custom Prime Lab GPU environment for Gates 1–4 because those gates need arbitrary internal HRM tensors. Package the environment and receipts using the existing Prime workspace conventions. The historical vdp_int3_logic environment is useful as a structural example for environment packaging, multiple arms, and evaluation receipts, but its Qwen late-attention proxy is scientifically ineligible here.

After Gate 4, either:

- keep the controller and HRM update environment on the custom pod; or
- pass only the sealed, compact observation summaries into hosted RL if all arms receive structurally identical inputs and the environment can still enforce deterministic reset, adapter equality, rollback, and full receipts.

Never assume hosted RL exposes the internal tensors required to compute VPD.

## Required first deliverable after authorization

For PL-HRM-BASE-CONFORMANCE-V1, return only:

1. source and environment hashes;
2. missing, unexpected, and newly initialized tensor counts;
3. native fused-module inventory;
4. per-prompt token equality and maximum logit error;
5. peak memory, runtime, exit status, and cost receipt;
6. an explicit PASS or FAIL.

If FAIL, stop without training or modifying a model. If PASS, wait for authorization before launching the numerical pilot unless the user's authorization explicitly covered both Gates 1 and 2.
