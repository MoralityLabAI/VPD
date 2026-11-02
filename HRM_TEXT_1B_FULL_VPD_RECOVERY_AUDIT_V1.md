# HRM-Text-1B Full VPD Recovery Audit V1

**Date:** 2026-09-08  
**Status:** forensic audit; no paid run launched; no model weights modified  
**Claim boundary:** the May run is a genuine attempt at full ComponentModel VPD coverage. Its recovered terminal learned state is not a valid VPD artifact for downstream science.

## Executive finding

The user's recollection is correct: a full, 16-chunk HRM-Text-1B ComponentModel decomposition was run on 2026-05-19. Each chunk paired the same indexed layer from the L and H recurrent stacks, so chunks 0–15 covered the full depth. This was not the later low_rank_svd feature-map refinement.

The recovered run cannot be supplied directly to a VPD-informed RL policy. Every chunk's last metric row has a non-finite total loss, and direct inspection of a representative terminal checkpoint finds that every learned component and every learned component-importance function value is non-finite. The frozen target-model tensors in that checkpoint remain finite. The D and E copies of the inspected layer-14 endpoint are byte-for-byte identical, so E is a recovery replica rather than an independent earlier checkpoint.

The original logs also record a base-loader conformance hazard: fused HRM checkpoint tensors such as gqkv_proj and gate_up_proj were reported unused while split q_proj/k_proj/v_proj, gate_proj, and up_proj tensors were reported newly initialized. The recovered loader does not contain an explicit fused-to-split transfer. Until an exact-logit base-equivalence test proves otherwise, the historical run must not be assumed to have decomposed the intended trained HRM weights.

## Authoritative locations

- Primary recovered run: D:\Research_Engine\runs\hrm_text_vpd_chunked_overnight_v2
- Disk-rescue mirror: E:\PC_Rescue_20260723\Research\Research_Engine\runs\hrm_text_vpd_chunked_overnight_v2
- Finite one-step plumbing fixture: D:\Research_Engine\runs\hrm_text_vpd_smoke\decompositions\hrm-text-vpd-smoke-20260519\model_1.pth
- Finiteness inspector: C:\projects\VPD\scripts\inspect_vpd_checkpoint_finiteness.py
- Prime Lab workspace: D:\Research_Engine\prime_lab

## What was actually run

The 16 terminal checkpoints have identical sizes of 777,182,531 bytes, for 12,434,920,496 bytes total (about 11.58 GiB). Their timestamps run from 20:52 through 21:17 America/New_York on 2026-05-19. Each chunk's final configuration records:

- ComponentModel continuous sampling with leaky-hard sigmoid;
- one indexed L-stack layer and the same indexed H-stack layer;
- eight split matrix families per stack, 16 module patterns per chunk;
- 64 components per module pattern, 1,024 components per paired-layer chunk;
- ImportanceMinimalityLoss and StochasticReconLayerwiseLoss;
- HrmTextRemoteCausalLM loading D:\Research_Engine\models\HRM-Text-1B;
- SimpleStories, maximum sequence length 32, batch size 1;
- 20 optimizer steps, learning rate starting at 3e-4;
- no component or CI-function gradient clipping;
- no periodic save frequency, leaving only model_20.pth.

The runner marked all 16 chunks completed because each child process exited with code 0. That process status is not a numerical-validity receipt.

## Numerical audit

All 16 metrics files end at step 20 with train/loss/total equal to NaN and non-finite component and CI-function gradient norms.

Direct inspection of chunk 14 model_20.pth:

| State group | Floating tensors | Elements | Non-finite elements |
|---|---:|---:|---:|
| target_model | 19 | 262,669,824 | 0 |
| components | 32 | 4,128,768 | 4,128,768 |
| ci_fn | 64 | 121,766,912 | 121,766,912 |

There are 96 affected learned tensors. The inspection command correctly exits nonzero.

The later chunked smoke/fixcheck model_1.pth has the same complete non-finiteness in its learned components and CI functions. By contrast, the earlier unpruned layer-0 smoke model_1.pth is finite:

| State group | Floating tensors | Elements | Non-finite elements |
|---|---:|---:|---:|
| target_model | 259 | 1,182,795,264 | 0 |
| components | 16 | 278,528 | 0 |
| ci_fn | 32 | 1,115,200 | 0 |

That earlier file covers only four matrix families in layer 0 of each stack, with eight components per pattern and one optimizer step. It is a valuable positive fixture, but it is not a full trained VPD.

## D/E recovery comparison

Both recovered trees contain exactly the same 16 model_20.pth relative paths, sizes, and timestamps. Neither tree contains model_1.pth through model_19.pth for overnight_v2. All non-checkpoint files in the D tree that have E counterparts matched by SHA-256 in the audit comparison.

The representative layer-14 checkpoint matches exactly:

SHA-256: 20D87DE733A447EC88C124DDDFAE608CA86DE361076AB29F0E963F24EBD5B1C3

This confirms that the E rescue preserved the endpoint, but it provides no finite earlier state for that chunk.

## Base-loader conformance risk

The overnight logs say many fused original HRM weights were not used and many split replacement matrices were newly initialized. The currently recovered local HRM implementation uses fused gqkv_proj and gate_up_proj tensors. The later repository configuration also targets those fused modules directly, which is the correct direction for a repair.

The historical run therefore establishes intent, coverage, and an important failure mode. It does not currently establish a faithful full decomposition of the trained base model. A repaired run must:

1. load the sealed original checkpoint with zero missing and zero unexpected tensors;
2. avoid all newly-initialized weight warnings;
3. prove exact or tolerance-bounded logit equivalence before ComponentModel wrapping;
4. target the native fused modules;
5. stop immediately on any non-finite loss, gradient, parameter, or optimizer state;
6. save early and periodic checkpoints so a terminal collapse cannot erase the last finite state.

## Scientific disposition

| Artifact | Provenance | Scientific use |
|---|---|---|
| overnight_v2 configs, logs, and chunk plan | recovered historical | Evidence that a full 16-chunk ComponentModel VPD was attempted |
| overnight_v2 model_20.pth files | recovered historical | Quarantined; terminal learned state is non-finite |
| E overnight_v2 copy | disk-recovery mirror | Independent preservation path, not an independent result |
| early layer-0 model_1.pth | recovered historical | Finite plumbing and parser fixture only |
| later low_rank_svd feature map | recovered historical | Sensitivity-map evidence; never label trained VPD |
| future fused-module VPD | proposed reconstruction/rerun | Eligible only after all conformance and validity gates pass |

Nothing in this audit erases or downgrades the fact that the full experiment was run. It separates “full coverage was attempted” from “the recovered learned artifact is valid enough to drive a new scientific claim.”

## Next authorized boundary

Local auditing, protocol preparation, and non-mutating validation are authorized. A Prime Lab GPU launch, paid inference, decomposition retraining, adapter training, or model-weight mutation is not authorized by this document and requires an explicit user go-ahead.
