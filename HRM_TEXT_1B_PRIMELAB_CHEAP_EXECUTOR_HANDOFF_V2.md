# HRM-Text-1B PrimeLab Cheap-Executor Handoff V2

**Current state:** local base-conformance Gate 1 passed; two Prime attempts ended operationally; no fresh VPD or RL policy has been trained.  
**Normative machine packet:** `C:\projects\VPD\HRM_TEXT_1B_PRIMELAB_EXECUTION_PACKET_V2.json`  
**Evidence report:** `C:\projects\VPD\HRM_TEXT_1B_LOADER_CONFORMANCE_AND_PRIMELAB_RUN_REPORT_V1.md`

This handoff supersedes `HRM_TEXT_1B_PRIMELAB_CHEAP_EXECUTOR_HANDOFF_V1.md` for current status. Preserve V1 as historical evidence.

## Required reading order

1. `C:\projects\VPD\HRM_TEXT_1B_LOADER_CONFORMANCE_AND_PRIMELAB_RUN_REPORT_V1.md`
2. `C:\projects\VPD\HRM_TEXT_1B_FULL_VPD_RECOVERY_AUDIT_V1.md`
3. `C:\projects\VPD\HRM_TEXT_1B_VPD_RL_POLICY_PROTOCOL_V1.md`
4. `C:\projects\VPD\HRM_TEXT_1B_PRIMELAB_EXECUTION_PACKET_V2.json`
5. `D:\Research_Engine\prime_lab\AGENTS.md`
6. Applicable Prime skill files, especially `.prime\skills\create-environments\SKILL.md` and `.prime\skills\evaluate-environments\SKILL.md`

The V2 JSON packet is normative if prose differs.

## Immediate task after explicit authorization

Run only `PL-HRM-BASE-CONFORMANCE-V1-R3` with the reconstructed fused-loader overlay and the limits in the V2 packet. Return the conformance receipt, source/environment hashes, billing receipt, and termination receipt. Do not start the VPD numerical pilot in the same run.

## Non-negotiable provenance

- The recovered May 16-chunk ComponentModel run is real but invalid: its terminal learned state is non-finite.
- The pinned May loader is a split-module implementation incompatible with the recovered fused checkpoint.
- The repaired fused loader is explicitly reconstructed and locally validated; it is not claimed to be the lost historical loader.
- The later `low_rank_svd` map is a sensitivity artifact, not trained VPD.
- A fresh finite trained VPD does not yet exist, so the VPD-informed RL claim remains untested.

## Authorization boundary

No future paid launch or weight-producing run is authorized merely by this handoff. Require an explicit user instruction naming the next run ID. Never touch unrelated Prime pods.
