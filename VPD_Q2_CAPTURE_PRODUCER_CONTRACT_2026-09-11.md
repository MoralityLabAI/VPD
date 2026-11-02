# Q2 capture producer: source audit and implemented replay component

New local engineering work, not recovered research and not a capture result. No checkpoint was loaded, no language-model inference or generation was run, no weights were created or modified, and no paid compute was launched. CPU tests use parameter-free Identity hooks and position/pass/site sentinel tensors.

## Source finding

The historical `experiments/nanbeige_reasoning/run_capture.py` already replays native two-pass continuations at four position windows. Its `main` also generates both two-pass and one-pass conditions, and `replay` records all 22 physical layers plus norm. Those broader measurements belong to that historical experiment; they are not required to compare Q2-N versus Q2-B at layer 11 and norm. The older runner remains unchanged.

The native runner records a constant revision label alongside hashes of local checkpoint files, but does not itself prove that those local bytes correspond to that revision. It also permits short/overlapping windows through the generic position selector; Q2 requires four distinct eight-row windows. A completed native-runner manifest alone would therefore not certify Q2 eligibility. Cache parity, model identity, semantic family separation and censoring remain independent obligations.

Source hashes inspected:

| File | SHA-256 |
|---|---|
| `experiments/nanbeige_reasoning/run_capture.py` | `3a2c2e24ed2095e37391a15f2038915feb051cd3f2531acac85f5bf851aaaf6d` |
| `experiments/nanbeige_looped/capture.py` | `e6105969cfc554b643b718c804e91e0383b9214bd888da1e69f65a9fb480231f` |
| `experiments/nanbeige_reasoning/positions.py` | `83e3127a79788949b430082201f0308a62125595a5a37e44f3719fecfcc35864` |

## Implemented component

`experiments/nanbeige_reasoning/q2_replay.py::replay_q2` is a callable capture-only component with no CLI, model loader, downloader, generator, optimizer or file writer. Calling it on a research model performs inference and therefore requires a separately authorized capture job. Merely importing it does not run a model.

Inputs are an already loaded model, a frozen complete token-ID list, one record with identity/role/length/window metadata and canonical token-list SHA-256, and an explicit sequence bound no greater than 8192. It checks these before forwarding:

- Exact four-window recipe, 32 distinct positions, full token length and token-list hash; no filling missing rows.
- Explicit supported record role and identity, integer token IDs in the model vocabulary, and the explicit sequence bound.
- The 22-layer, two-pass, width-3072 architecture; no alternative loop-loss/split configuration; evaluation mode, BF16 model dtype and no declared quantization.
- Distinct physical layer-11 and norm modules.

It then performs exactly one no-cache backbone forward inside inference mode, with no configuration changes. At each selected physical module output it requires shape `[1, full_sequence_length, 3072]` and BF16 dtype, extracts the frozen positions as float32 copies, and rejects nonfinite selected values. Both sites must fire exactly twice. Fewer or extra invocations fail; hooks are removed on success or forward/extraction failure. Returned arrays use `model.layers.11__pass_0`, `model.layers.11__pass_1`, `model.norm__pass_0`, and `model.norm__pass_1`.

The return record carries positions, regions, calls, dtype and physical-output semantics. It does **not** assert checkpoint identity, freshness, generation provenance, provider shutdown or authorization. Site names and matching invocation counts cannot alone prove that a remote model implementation follows the intended recurrence. Model-source review and a real-model hook/noninterference check remain necessary.

## Capture payload calculation

For 32 selected rows and width 3072, float32 activation values alone occupy:

| Capture sites | Bytes per trajectory, both passes | MiB per trajectory | MiB for 40 training trajectories |
|---|---:|---:|---:|
| Historical 23-site replay | 18,087,936 | 17.25 | 690 |
| Q2 two-site replay | 1,572,864 | 1.5 | 60 |

These are exact uncompressed array-value counts, not measured file sizes, GPU memory requirements, runtime or dollar estimates. The backbone still executes all layers on both passes. Reducing recorded sites does not remove those model computations. Q2's proposed capture job needs native two-pass generation only; eliminating the historical one-pass comparison is a change to a future capture job, not a claim that any work or money has already been saved.

## Verification completed

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_q2_capture_replay.py tests/test_nanbeige_looped_capture.py tests/test_vpd_sae_trajectory_breadth_plan.py
```

Result: **35 passed**, including 16 new replay tests. This is a focused suite, not a full-project result. Exact row/site/pass values are checked against independently constructed tensor sentinels. Tests cover one-forward behavior, no configuration/record mutation, malformed input rejection before forwarding, missing/extra physical invocations, incorrect output dtype, nonfinite output, and hook cleanup.

- New component SHA-256: `87753704d3d7a66dc33c2d8de1f7d764b4fb798af3a08976bac011c22553aefb`.
- New test SHA-256: `cce2ca26f855a1a3d11d9f67be94cdc00a1b6175d89625f9838a6ece1ad36895`.

This establishes a tested implementation component, not real BF16 activation provenance or an SAE quality improvement. The immutable synthetic Q2 receipts and canonical experiment protocol were not rewritten.

## Integration still required before capture or training

1. Resolve the real family/role/exposure inventory and freeze the actual pack. Existing tail-only BF16 captures remain incompatible; see the recovered-capture compatibility report.
2. Freeze a two-pass native-generation configuration, tokenizer and reviewed checkpoint/code hashes, context/token limits, EOS handling and censoring policy. A short or otherwise ineligible trajectory must remain a reported failure, not be silently replaced or extended until it qualifies. Do not select on answer correctness or reconstruction quality.
3. Verify cache/full-prefix parity and hooked/unhooked noninterference on the actual reviewed runtime. The synthetic tests here do neither.
4. Integrate an executor that persists full sequences and generation provenance separately, calls this replay component, writes immutable array archives/hashes, and translates the training-only results into the existing Q2 array-binding contract. Preserve role separation for evaluation outputs. This executor/importer is not implemented here.
5. Freeze paired training/analysis rules and obtain explicit authorization for the exact capture/fit conditions, spend cap and independently enforced provider termination deadline. A local timeout does not establish that a pod stopped billing.

Do not launch the historical broad runner as an undocumented substitute. Do not run the new helper against a real model merely because the local tests pass. The next implementation step is the generation/provenance-to-archive integration, not another rerun of the synthetic replay tests reported as scientific progress.
