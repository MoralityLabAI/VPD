# Nanbeige pass-resolved mechanistic dataset

This local index connects the paper's saved measurements by public input ID, execution variant, physical layer, pass and dictionary. It indexes existing files without copying weights or rerunning a model. Reading the examples below does not start or alter inference.

There are 18 fixed inputs: five ARC-AGI-1 test inputs, five ARC-AGI-2 test inputs, four INTELLECT-3 logic questions and four INTELLECT-3 math questions. Eight inputs are used for calibration and ten are held out from fitting. An ARC input is one test grid, so input counts are not complete-task scores. The historical Campsite slice remains identified separately in the `split` field.

The larger source corpus contains 34,886 question records. This index describes the measured 18-input pilot; it does not imply that activations were collected for the full corpus or that the full benchmark/TRM comparison was completed.

## Files

| File | Purpose |
|---|---|
| `examples.jsonl` | One record per input, with links to paired trajectories, activations, parameter coordinates, sparse features and intervention effects. |
| `assets.json` | Project-relative paths, SHA256 checksums, sizes and roles for every indexed asset. |
| `manifest.json` | Counts, source-index checksums, builder checksum and limitations. |

All asset paths resolve from the project root, three directories above this folder. Paths within an archive are NumPy keys, not filesystem paths. Read archives using `numpy.load(..., allow_pickle=False)`.

The index links 36 one-pass/two-pass trajectories, 18 activation archives, 162 SAE feature archives, 72 VPD coordinate archives, 120 frozen-SAE intervention archives, their dictionaries/bases, raw BF16 intervention records and source receipts. Each extended activation archive contains 50 arrays: the two invocations of 22 physical layers, both loop-normalization outputs, and four selected MLP-input arrays.

## Read a matched pair

From the project root, with the existing NumPy environment:

```powershell
.venv/Scripts/python.exe -m scripts.nanbeige.inspect_mechinterp_pair
```

The default example is the first calibration input. The command verifies index and selected-file hashes, checks identical token positions, confirms the stored feature difference equals pass 1 minus pass 0, and reports descriptive differences. It does not train, generate text or load a model. To select an input explicitly:

```powershell
.venv/Scripts/python.exe -m scripts.nanbeige.inspect_mechinterp_pair --example arc_agi_1/12422b43/0 --site model.layers.21 --optimizer muon_hybrid_paired_cuda
```

The largest feature changes are numerical measurements, not semantic explanations or evidence of an accuracy gain. Inspect reconstruction quality in the linked SAE condition manifests and transfer-quality file before interpreting features.

## Coordinates and alignment

- `model.layers.21` is physical layer 22; `model.layers.11` is physical layer 12. Layer indices in archive keys are zero-based.
- `__pass_0` and `__pass_1` denote the first and second invocations of the same physical weights. They are not separate independently trained layers. This checkpoint uses fixed two-pass recurrence, without an observed adaptive router.
- Positions are absolute zero-based indices in the saved prompt-plus-generation token sequence. Region intervals are half-open. Extended captures retain 32 positions per input: eight each from the prompt suffix and early, middle and late generation.
- Q4 activation archives store float32 tensors produced using Q4_K_M model weights. Float32 storage does not make this a full-precision model run.
- Every feature difference uses one dictionary shared across passes. The index labels the original CPU AdamW dictionaries separately from the paired CUDA AdamW/Muon dictionaries. Feature IDs are not aligned between independently fitted dictionaries.
- Public `source_split`, evaluation `split`, and SAE `calibration_split` are preserved separately. A held-out SAE input is not necessarily an uncontaminated external benchmark example. INTELLECT sources include a public RL training corpus.

## Keep execution variants separate

| Variant | Measurements and alignment |
|---|---|
| `extended_8192_native_two_pass` | Native two-pass trajectories with an 8,192-token output cap, 32 retained positions, original eight-component BF16 VPD basis transfer, and the original/paired SAE features. |
| `initial_2048_budget_q4` | Earlier Q4 trajectories projected into a separately fitted 64-component BF16 basis. These coordinates belong to the earlier capture, not the extended capture's rows. |
| `initial_2048_budget_bf16_replay_last8` | BF16 replay/interventions at the final eight positions of the initial-budget trajectories, re-encoded using frozen dictionaries trained on the extended Q4 captures. |

Matching question IDs alone does not establish matching token positions, precision or context. Before any row-wise comparison, inspect the exact token IDs and positions from both archives. The BF16-to-Q4 feature transfer changes execution precision/runtime and the position distribution together. Its quality measurements do not isolate those causes.

The VPD coordinate files omit residual weights from diagnostic basis outputs. Eight-component and 64-component decompositions are different bases; component IDs and coefficient magnitudes cannot be equated across them. Frozen-SAE effect files link the underlying raw BF16 intervention archive. Their conditions include baseline and signed first-pass, second-pass and both-pass interventions, measured in the same frozen dictionary for each comparison.

## Evidence and limits

The extended native run has 13/18 token-capped outputs; the one-pass run has 18/18. Saved activations and feature changes do not prove improved reasoning. Reconstruction quality varies by layer/pass; one Muon seed improves four of six paired measurements and worsens layer 12 slightly. The paper reports the actual values and comparison limitations.

The separate cache controls are diagnostic artifacts, not benchmark examples in this index. The batched numerical gate failed despite greedy-token agreement. All eight serial conditions now pass with bitwise-identical cached and fresh-prefix logits on one 383-token calibration prompt and eight decoding steps. Three controls were carried from earlier runs and five completed with guarded checkpoint continuation; their saved arrays and provenance were independently verified. See `../q4_cached_steering_burst_v1/verification.json` and `../cached_steering_completion_verification.json`.

The completed 70-continuation Muon-feature answer extension has its own supplement, `../answer_steering_index_v1`, linking immutable token IDs/native text, final grades and provenance. Its fresh eight-condition gate passes at the full 10,153-token context allocation, but still evaluates only a short calibration prefix and eight decoding steps. That does not validate full long trajectories. The `native_decode_audit_v3` checks the pinned native punctuation/leading-space cleanup against independent raw token decoding. Final grades in `grading_v2` fix a Windows input-transport limit while preserving the published math verifier; raw and native rendering agree on all 70 grades.

The extension contains 561,231 generated tokens, with 62 outputs reaching the cap. Three conditions solve the same individual ARC test input, but all seven conditions score zero on eight complete evaluation units. It adds answer trajectories, not hidden-state captures. The older activation/VPD/SAE arrays must not be interpreted as captures of these fresh generations merely because question IDs match. The completed run and receipts are in `../prime-20260908T200850Z-601`.

`build_mechinterp_index.py` checks referenced file hashes, per-input coverage, split identity and cohort grouping. It does not replace the numerical re-encoding and model-execution verifiers linked from the source artifacts. Rebuild the index only when intentionally updating the measured dataset:

```powershell
.venv/Scripts/python.exe -m scripts.nanbeige.build_mechinterp_index
```

This is a local, unpublished research index. Upstream source revisions and available licenses remain in `artifacts/nanbeige_reasoning/sources.lock.json` and the referenced source directories; VPD implementation attribution is retained in `experiments/nanbeige_looped/VPD_BACKEND_LICENSE`. The index grants no additional redistribution rights. The working paper is `output/pdf/nanbeige_hermes_research_draft.pdf`.
