# HRM-Text-1B Loader Conformance and PrimeLab Run Report V1

**Date:** 2026-09-08  
**Status:** local Gate 1 PASS; PrimeLab Gate 1 not yet completed; no VPD or policy weights trained  
**Purpose:** durable evidence and an execution boundary for a cheaper follow-on model

## Executive result

The recovered May 2026 HRM-Text-1B ComponentModel endpoints remain invalid. The investigation now identifies two independent causes that prevent their scientific reuse:

1. all 16 terminal chunk metrics report non-finite total loss, and the inspected learned component and CI-function tensors are entirely non-finite; and
2. the pinned May model code expects split attention and MLP matrices, while the recovered trained checkpoint contains native fused matrices. Loading the fused checkpoint through that code produces missing, unexpected, and newly initialized tensors.

A separately labeled **reconstructed fused loader** was made from the recovered June fused architecture plus the mask-call contract required by Transformers 5.16.1. It does not claim to reproduce a lost historical file or validate the old VPD. It loads the recovered base checkpoint exactly and passes the full 32-prompt local no-op instrumentation gate.

No optimizer was created, no weight was changed, and no real action was executed by the conformance work.

## Authoritative local Gate 1 receipt

Run ID: `LOCAL-HRM-FUSED-REPAIRED-CONFORMANCE-V1`

Receipt directory:

`C:\projects\VPD\primelab\hrm_base_conformance_v1\fused_repaired_full32_20260908`

| Check | Result |
|---|---:|
| Recovered base SHA-256 | `F8FE2B2BF6948414E8E8D6538659198726D98F967C55B533B7AABE8A1FA9A584` |
| Weight hash matches declared base | PASS |
| Missing keys | 0 |
| Unexpected keys | 0 |
| Mismatched keys | 0 |
| Load errors | 0 |
| Newly initialized keys | 0 |
| Floating/complex tensor entries | 131 |
| Model elements checked | 1,182,795,264 |
| Non-finite model elements | 0 |
| Expected fused hook targets | 128 |
| Discovered fused hook targets | 128 |
| Missing or extra hook targets | 0 |
| Fixed prompts | 32 distinct prompts |
| Token mismatches | 0 |
| Non-finite logit values | 0 |
| Maximum fp32 no-op logit error | 0.0 |
| Runtime | 76.88101935386658 seconds |
| Receipt status | **PASS** |

Receipt hashes:

| File | SHA-256 |
|---|---|
| `base_conformance.json` | `681B09A9B36ABEC735F63A3F7AD4EE5E50A9D0B3EC24FA55DA9CE98ED231A84E` |
| `environment_lock.json` | `54859EFD81B48EDCAA676E44F7DA22337CD5A46AC091A992E8F2FB7A03C0FC8D` |
| `probe_results.jsonl` | `026A3E70F6CABE7F46DC8B8751CB6589F316F20EA0899F3673A7589862198229` |
| `source_manifest.json` | `9B0002AC90570C7CD4D4967EC0006FFB6B690C908E9F5846254D4AC78AF9F0E7` |
| `termination_receipt.json` | `930C8CCCF9955716A0B82783B48DD2AFFB5FF275E19CD9D55AE1EC71C8E905ED` |

The one-probe preflight also produced finite logits, identical tokens, and a 0.0 no-op logit error. Its overall `FAIL` is expected because `probe_count_32` was deliberately false; it is not a scientific failure.

## Loader provenance

### Historical May loader

Pinned Hugging Face revision:

`46e495977e4aff73b4b8bbb7fed6ff8e3f92d9f0`

The pinned implementation uses split `q_proj`, `k_proj`, `v_proj`, `gate_proj`, and `up_proj` tensors. Direct loading against the recovered base reproduces the unused fused weights and newly initialized split tensors recorded in all 16 historical VPD logs. This is an exact loader incompatibility, not merely an untested suspicion.

### Reconstructed fused loader

Directory:

`C:\projects\VPD\primelab\hrm_base_conformance_v1\fused_repaired_composite`

The reconstruction retains the recovered June native-fused module architecture and changes only the incompatible mask arguments:

- `input_embeds` was corrected to `inputs_embeds`;
- unsupported `cache_position` was removed; and
- `position_ids` remains the declared position input.

The resulting hashes are:

| File | SHA-256 |
|---|---|
| `modeling_hrm_text.py` | `59BA5E6FDEC1F7D6EDA7DF38F259255857B8F9CA2A62253156E0BC20C0516240` |
| `configuration_hrm_text.py` | `337A2FA6B8ABA0579E8D631D97CD2BE36143A12B48608012DFB8FE4BB6A75301` |

This loader is **reconstructed**, not recovered byte-for-byte. Its validation is behavioral and structural. It must never be cited as the code used by the failed May VPD run.

## PrimeLab attempts and spend

Three A10 24 GB pods were used only for the authorized conformance stage.

| Attempt | Pod ID | Result | Final attributable billing |
|---|---|---|---:|
| 1 | `212e2a5574944aebb9acdd622f7577d9` | Packaging failure: incompatible `huggingface-hub` pin | $0.0055 |
| 2 | `d308c00eab3244c282d5670a41cfbf98` | Operational timeout before a scientific receipt | $1.15 |
| 3 | `2969c22a8d1e414ba328ae69c0ec0634` | Launcher session was lost; pod was orphaned overnight; SSH receipt recovery failed; manually terminated and verified absent | $15.6905 |

Total attributable spend: **$16.8460**.

The two-row billing extract is sealed at `C:\projects\VPD\primelab\hrm_base_conformance_v1\prime_billing_receipt_20260908.json` with SHA-256 `2F1C126AD9C71BFB83C2ADB13773CFDA6A50BE57C211F817A32740350DC00005`.

All three pods are terminated. The third termination is sealed in `C:\projects\VPD\primelab\hrm_base_conformance_v1\r3_orphan_termination_and_billing_receipt_20260909.json`. Unrelated Prime pods were not modified or charged to this work.

Attempts 2 and 3 are operational failures, not evidence that the model failed conformance. Attempt 3 exposed a billing-safety defect: a client-side shell trap was the sole termination mechanism, so losing the local launcher session orphaned the paid pod. No further paid launch is eligible until termination is enforced by a provider-side TTL/watchdog or a separately durable monitor.

## Prepared remote Gate 1 bundle

Bundle source:

`C:\projects\VPD\primelab\hrm_base_conformance_v1`

The launcher now:

1. pins the Hugging Face revision and package versions;
2. downloads the immutable checkpoint/tokenizer snapshot;
3. builds a separate symlink overlay and copies in the reconstructed fused Python loader;
4. hashes the overlay and exact base weights;
5. runs all 32 no-op probes with line-buffered logs;
6. retrieves receipts; and
7. terminates the pod in normal and error paths.

Prepared script hashes:

| File | SHA-256 |
|---|---|
| `remote_conformance.py` | `BCF906A01987F89848583956507F498CDDFC5F6DE0E90539E30A9CF5E35C4F05` |
| `run_remote.sh` | `999242C9126CCACCB45CDE95D457D8747BCB2164F94B9A1995B1E1C9ABDC0A78` |
| `launch_prime.sh` | `E8E3699EA1E7C0064F13153C6D480AD1C0781FCA3642BA13191DD223654E6533` |
| `probe_pack.json` | `534B474B51F76EADAA5479B32BEC1CF0E5AC89A1DAD71E2FCD3503588ACF8801` |
| `requirements.txt` | `4A485D4741336079737EA01B58C519D67C4646DD606ED380023CAD7463C04113` |

Python compilation passed. Both shell scripts passed `bash -n` under installed Git Bash. WSL validation could not be run from the local sandbox because WSL returned `E_ACCESSDENIED`; do not describe WSL syntax validation as completed.

## Exact scientific state

| Item | State | Consequence |
|---|---|---|
| Historical 16-chunk VPD | Recovered but invalid | Quarantine; never use as observation source |
| Historical May loader | Recovered/pinned and incompatible with fused weights | Evidence of a second historical failure mode |
| Later low-rank SVD map | Valid as a sensitivity artifact | May prioritize pilot layer, but is not trained VPD |
| Reconstructed fused loader | Local Gate 1 PASS | Eligible for a remote conformance replay and fresh VPD training after authorization |
| PrimeLab Gate 1 | Two operationally failed attempts; no scientific receipt | Must be rerun with the corrected bundle before Prime-only claims |
| Fresh finite ComponentModel VPD | Does not exist | RL observation experiment remains blocked |
| VPD-informed RL policy result | Does not exist | Paper must call it a prospective experiment |

The broader Bonsai authorization-control V1-V8 recovery map remains authoritative at:

`C:\projects\VPD\VPD_STATE_MAP_V1_V8_2026-08-29.md`

This HRM report is a later research branch and does not replace that map.

## Proposed run ladder for a cheaper executor

Execute in order and stop on the first failure. Do not tune a failed candidate after viewing task outcomes.

### Run 1: `PL-HRM-BASE-CONFORMANCE-V1-R3`

Purpose: reproduce the local PASS on PrimeLab using the reconstructed overlay.

- hardware: one A10 24 GB or cheaper compatible 24 GB GPU;
- hard price cap: $1.30/hour;
- hard wall-clock cap: 20 minutes after SSH becomes available;
- hard spend cap: $0.45;
- mutations: none;
- output: the six Gate 1 receipt types plus Prime billing and termination receipts.

Pass only if every field in the local Gate 1 table passes. If download or packaging does not reach model load within ten minutes, terminate and report an operational failure. Do not launch VPD training from the same shell invocation.

### Run 2: `PL-VPD-NUMERICS-PILOT-V1`

Purpose: choose a stable ComponentModel optimizer without task-outcome access.

- prerequisite: remote Gate 1 PASS and separate authorization for weight-producing VPD training;
- scope: paired L/H layer index 14; native fused `gqkv_proj`, `o_proj`, `gate_up_proj`, and `down_proj` only;
- candidates: learning rates `3e-4`, `1e-4`, and `3e-5`;
- component and CI dtype: fp32;
- gradient clip: 1.0 for both learned component and CI parameters;
- steps: 100 each;
- save/check: 0, 1, 2, 5, 10, 20, 50, 100;
- fail closed on any non-finite loss, gradient, parameter, or optimizer state;
- no final task pack and no policy training.

Select the lowest reconstruction-loss candidate among those that remain finite, avoid degenerate activation, and pass the planted-component and matched-random ablation controls. Break a tie toward the lower learning rate. If none pass, stop and version a diagnosis before any new run.

### Run 3: `PL-VPD-FULL-FUSED-V1`

Purpose: generate the first scientifically eligible fresh HRM VPD.

- prerequisite: Run 2 PASS and separate material-spend authorization;
- 16 paired-layer jobs, each covering both L and H stacks and the same four fused module families;
- 1,000 steps with checkpoints at 0, 1, 2, 5, 10, 20, 50, 100, 250, 500, 750, and 1,000;
- sequence length 1,024;
- parallelism determined by an explicit budget, not by executor convenience;
- resume only from independently finite checkpoints;
- require reconstruction, learned-ablation, zero-ablation, matched-random, planted-component, cross-half stability, and complete hash-manifest receipts.

### Run 4: `PL-VPD-OBS-CACHE-V1`

Freeze equal-width observation channels for true VPD, behavior-only, within-stratum shuffled VPD, and energy/rank/module-matched random maps. Fit normalization on controller-training data only. The held-out pack and action outcomes must be absent.

### Run 5: `PL-RL-ONE-STEP-V1`

Run four arms (`behavior_only`, `true_vpd`, `shuffled_vpd`, `matched_random_map`) over four common seed blocks. Each policy chooses exactly one fixed-size training tranche or `ABSTAIN`, then the common update mechanism is evaluated and retained or rolled back. Stop unless true VPD beats every matched control without a regression breach.

### Run 6: `PL-RL-SEQUENTIAL-V1`

Only after Run 5 passes, execute six bounded macro-steps for at least eight common seed blocks across uniform, random-action, behavior-only, true-VPD, shuffled-VPD, and matched-random arms. VPD may change observations only; it may not choose modules, ranks, norms, learning rates, or edit coordinates.

### Run 7: `PL-HELDOUT-REVEAL-V1`

Reveal the fresh hash-sealed pack once after all policies and checkpoints are frozen. Report every arm and seed. Do not tune or rerun on that pack.

## Cheap-executor stop rules

- Do not launch any paid stage without an authorization receipt naming its run ID.
- Do not overwrite base weights; all model updates must be reversible copies or adapters.
- Do not call the May endpoints, the SVD map, or attention proxies “full VPD.”
- Do not treat a completed process as a passed numerical gate.
- Do not proceed after non-finiteness, loader mismatch, missing receipt, leakage, or control failure.
- Do not reinterpret a negative matched-control result as positive RSI.
- Do not touch unrelated Prime pods.

## Paper implication

The repaired loader strengthens the experimental opportunity but does not change the existing conclusion. The historical full VPD attempt failed numerically and structurally. The paper can now say that the base instrumentation obstacle has a locally validated reconstruction, while fresh VPD training and the VPD-informed RL comparison remain prospective. A future positive result would support bounded policy-guided training on the declared distribution, not open-ended recursive self-improvement.
