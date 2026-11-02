# Q2 recovered-capture compatibility

Date: 2026-09-11. New read-only audit of existing real arrays; no inference, capture replay, training, research-weight change or paid compute.

## Actual blocker

The recovered BF16 reasoning calibration cannot directly supply Q2's required input views. Its 18 examples each retain eight tail rows; Q2 requires 32 rows spanning prompt, early, middle and late windows. All 18 lack the other 24 positions. An importer cannot recover those missing observed activations by interpolation, duplication or relabeling.

The audit checked every one of 72 arrays (18 examples × two sites × two passes), their containing archive hashes, shapes, finite values, absolute positions and saved token tails against the full token sequences. It reproduced exactly.

| Requirement | Observed recovered BF16 reasoning pack |
|---|---|
| Training inventory | 8 examples: ARC 4, math 2, Futoshiki 1, goods exchange 1; Q2 requires 40 across ten training families |
| Four eight-row windows | 0/18 examples complete; only late/tail rows retained |
| Missing positions | 192 training positions per site/pass; 432 across train and evaluation per site/pass |
| Model execution dtype | Source declares `torch.bfloat16`; saved arrays are float32 storage |
| Checkpoint revision verified | Source explicitly says false |
| Family-disjoint train/evaluation | Source explicitly says false |
| Fresh pack | Historical calibration, not a new Q2 holdout |

Float32 storage does not establish FP32 inference or BF16 provenance. The model-precision statement comes from the recovered manifest, and its revision caveat must remain attached. Likewise, hashes and token alignment do not independently certify the original producer's model-hook or pass labeling.

This narrower row scope does **not** invalidate the earlier pass-category bridge, which analyzed the saved tail rows. It prevents treating that archive as if the new four-window experiment had already been captured.

## Source and reproduction

Pinned source: `artifacts/nanbeige_reasoning/prime-20260906T204058Z-561/retrieved/artifacts/nanbeige_looped/bf16_multipass_calibration`.

Manifest SHA-256: `7c5205eeb3c1e7c025ac1aababc6e0526bbe82048f633a52e73bfefa07c8a546`.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_bf16_q2_capture_coverage.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.audit_bf16_q2_capture_coverage_v1 --verify artifacts/vpd_bf16_q2_capture_coverage_v1/report.json
```

Eight focused tests pass. The receipt reproduces from real saved arrays; process success means the negative compatibility finding was verified, not that import is eligible.

- Receipt: `artifacts/vpd_bf16_q2_capture_coverage_v1/report.json`
- Report ID: `91e264bd5cc82885cbe9c62796fc6523ad459fc620755a3157f3de7fef06b96d`
- File SHA-256: `c59d1a70c301549939f9c6a0439a393014218a2fb04884562f9ee41c330ae36c`

## Importer implications

The current Q4 capture manifest does retain four windows, but explicitly identifies quantized Q4_K_M weights and unvalidated BF16 parity. That source is historical development evidence, not a replacement for the proposed BF16 capture. The BF16 calibration uses `examples.json` plus `baseline_archives`; the native-reasoning capture producer instead writes `captures.jsonl`, `generations.jsonl` and `calibration_tokens.jsonl`. An importer must recognize each producer's schema and preserve precision, position-window and revision provenance; file extensions or float32 storage cannot select the scientific protocol.

The inspected native BF16 reasoning producer source is `experiments/nanbeige_reasoning/run_capture.py`; it was **not executed**. Its intended four-window replay is not evidence that such an eligible saved pack exists. A bounded filename inventory found no `captures.jsonl` under the inspected reasoning/looped artifact roots, but this is not an exhaustive search of E: or other repositories.

A successful real Q2 import still requires an actual reviewed pack and captures with the declared windows, source identity and roles. No real Q2 importer or research training condition was completed by this audit. Preserve the existing arrays and their historical use; do not try to repair the missing observations by editing their manifests. The synthetic loader/initialization fixtures and the sealed canonical protocol remain unchanged.

## Recovery search and byte-level triage checkpoint

The bounded E: search receipt is `artifacts/vpd_q2_recovery_capture_search_v1/report.json` (SHA-256 `b12e7514fb24a45eff46055a05fc6165c8a5a892634146d0817dda50d1d75051`). This is a new observation of recovery directories, not an original research receipt. Its UTC timestamp is September 12, corresponding to September 11 in the workspace timezone.

Four of five filename scans completed: three HRM-re roots had no matching filenames, and the TRM observability root had two. SmallControlHarness timed out after 25 seconds with 50 partial matches. The earlier broader search was interrupted and is not absence evidence. No full-disk, Git-object or archive-content search was completed.

All 52 returned candidate files were then read and hashed. The separate `artifacts/vpd_q2_recovery_capture_search_v1/triage.json` records every path, byte length, SHA-256, parse status, selected schema/model declarations and the observation time:

| Candidate state | Count | Interpretation |
|---|---:|---|
| Readable JSON | 36 | Manifest-level leads; referenced payloads have not been certified |
| Entirely zero-filled, nonempty files | 16 | These copies contain no usable manifest JSON |

Both TRM manifests are zero-filled: `data/qwen27b_intellect3_logic_flow_policy_decomp/manifest.json` is 42,722 bytes, SHA-256 `14eb989cf4b1083519a2b509df606292d1c92e903c0eff2596ac85edf7018f77`; `data/constitutional_compare/moral_quandaries_same_endpoint_smoke/manifest.json` is 1,314 bytes, SHA-256 `42bd00674f81293c888f766b2d27945385337beaad0ef49f9e408cdb26933242`. This establishes damage to these copies, **not** that the original research or paper was deleted or that every other copy is lost.

Readable model-bearing leads declare a Qwen Bluebeam bridge, deterministic hash embeddings, or DistilGPT2 embeddings/projections. Other readable schemas describe control-harness evaluations, datasets or Bonsai contrasts/clustering. None has been admitted as a fresh Nanbeige BF16 Q2 capture. Absence of the text marker `nanbeige` in these manifests is only a search aid, not a provenance test. In particular, an HRM-containing directory name does not establish an HRM model activation capture.

The recovered and local Bonsai V8 cluster manifests have identical SHA-256 `6b4baf7491887c23bb34b0e5dd6db121d3ccc4ab2e891086ad0e6bfb547c5445`. Their declared feature mode is `metadata_and_joint_evidence`, not a new Q2 activation pack. Likewise, local `artifacts/nanbeige_reasoning/vpd_precision_inputs_v1/fit_manifest.json` has the same `7c5205...a546` hash as the already audited eight-tail-row BF16 reasoning manifest; it is not an independent capture.

No recovery files were modified or repaired. No recovery code, inference or training was executed. The historical filename-search receipt remains unchanged; the triage receipt adds byte-level observations without upgrading the incomplete search into an exhaustive one.

### Consequence for the next comparison

This triage yields zero **newly verified eligible** Q2 packs, not proof that none exists anywhere. Do not build an importer on the assumption that these candidates resolve the missing observations. Further recovery should target a specific unsearched directory, archive or producer receipt; do not repeat the same bounded scan without a new lead. In parallel, local work can specify the actual four-window capture producer contract and the paired analysis/decision rule. A new capture remains contingent on real pack feasibility, producer verification, explicit compute approval and independent provider shutdown. Neither broadening an eligibility claim nor duplicating saved rows repairs the evidence shortage.
