# Nanbeige looped-model pilot

Future utility comparisons follow the Nanbeige track in the [cross-research addendum](../../VPD_CROSS_RESEARCH_UTILITY_ADDENDUM_V1.md): match total rank and fit cost, retain random and label-only controls, and evaluate behavioral improvement and catalog regret separately from VPD/SAE reconstruction.

Started 2026-09-04. Preflight checks model/runtime compatibility and retains both
passes' activations. The multipass runner now adds bounded VPD optimization,
common-dictionary SAE training, and signed component-effect capture. CPU canaries
have completed. The pretrained BF16 experiment also completed on an A100 on
2026-09-06; see `artifacts/nanbeige_looped/PRETRAINED_RESULTS.md`.

## Pinned assets

- Official BF16 model: [Nanbeige/Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B),
  revision `451ed48c3273ecef7ea8faaa43c31ce529763bb1`.
- Local evaluation quant: [loqy/Nanbeige4.2-3B-GGUF](https://huggingface.co/loqy/Nanbeige4.2-3B-GGUF),
  revision `5ae468e507f641599c25f34f85772212f7e00dc7`, file `Nanbeige4.2-3B-Q4_K_M.gguf`,
  2,579,880,160 bytes; SHA-256
  `01739fad13be318c5c8620f8de52731608bb750ae887e9445cc81cb4a189e45d`.
- [llama.cpp b10731](https://github.com/ggml-org/llama.cpp/releases/tag/b10731),
  commit `0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0`, Windows CUDA 12.4 runtime.
  Archive checksums and original URLs are saved beside the archives.

Model, tokenizer, and runtime are under `C:\projects\VPD\models` and
`C:\projects\VPD\tools\nanbeige`. D: had only about 80 MB free at startup.
The official tokenizer files have individual SHA-256 entries in their
`provenance.json`. Both official BF16 shards are now also stored in the legacy
`models/Nanbeige4.2-3B-tokenizer` directory; `checkpoint_provenance.json` records
their verified publisher hashes. The GPU run independently downloaded the same
pinned checkpoint, and all 11 recorded checkpoint/source hashes matched locally.

## Local startup

The detected GPU is an RTX 3050 Laptop with **4 GB VRAM**. The Q4 smoke runner uses
a 1,024-token context, one slot, 128-token batches, and at most 128 generated tokens
per example. It stops its own localhost server after the check, including on error.

```powershell
.\.venv\Scripts\python.exe -m experiments.nanbeige_looped.q4_smoke --output-dir artifacts\nanbeige_looped\q4_smoke
```

The runner rechecks the Q4 checksum, renders the official HF chat template, and
passes **HF token IDs** to llama.cpp's `/completion` endpoint. It records native
tokenizer differences separately; this quant has a reported leading-space caveat.
It runs two arithmetic prompts with thinking both off and on. A token limit can
stop the thinking case before an answer; these are smoke checks, not accuracy
measurements. Passing this check does not establish BF16/Q4 behavioral parity.

Startup checks already passed: CUDA runtime detects the GPU; three repeated-hook
unit tests; Python compilation; HF fast/slow tokenizers agree on 32 pilot inputs
(maximum 82 prompt tokens). The Q4 smoke test has now passed; see
`artifacts/nanbeige_looped/q4_smoke/manifest.json` and `completions.jsonl`.
All four tested inputs also matched llama.cpp's native tokenizer. Generation was
about 34 tokens/second on the two 128-token thinking samples, which stopped at the
configured token cap. The no-thinking cases returned A (correct) and B (incorrect).
No accuracy benchmark or BF16 parity claim follows from these four smoke samples.
The local server was stopped after completion.

## Prime and BF16 preflight

After disk cleanup, WSL starts successfully. The first A6000 attempt,
`f5f2d1a932a649cba44d8f0d75e542e7`, was terminated without BF16 results; its receipt
is under `artifacts/nanbeige_looped/prime-20260906T015238Z-479/`.
After a successful Firefox browser authorization, a multipass retry created
`8fa276016fa342e0b018b20c40661273` on an A6000 48GB at $1.09/hour. It became ACTIVE
but both the pod-details endpoint and raw status response returned null IP/SSH
fields, with installation PENDING. The run was stopped through its cleanup
handler and Prime confirmed termination. No model upload, BF16 preflight, or
pretrained VPD/SAE capture began. See the report and termination receipt in
`artifacts/nanbeige_looped/prime-20260906T024406Z-767/`.
A subsequent A100 attempt eventually supplied an address after roughly 16 minutes,
but rejected the WSL SSH key because a different key was primary during creation.
That instance was terminated. The corrected launcher temporarily selects the
registered local key, restores the previous primary setting after connection,
and also attempts restoration during cleanup.

The successful run is `artifacts/nanbeige_looped/prime-20260906T034844Z-719/`.
Preflight, VPD fitting, SAE fitting, 16 intervention records, retrieval, and pod
termination all completed. The 74 original artifact files passed hash and numeric
checks. The previous primary SSH key setting was independently verified restored.
The current launcher selects a US A100 at no more than $2/hour and uses a one-hour
outer timeout; allow up to two extra minutes for cleanup.

Existing authentication lives in `D:\research_engine\prime_lab` and uses the WSL
Ubuntu Prime CLI. The user refreshed `PRIME_API_KEY` in that workspace's `.env`.
The latest browser challenge succeeded, and cached CLI authentication was verified
by a successful pod listing and creation/termination of the retry. The wrapper's
default mode still uses the file key; `--cached` selects the refreshed CLI login.

```powershell
.\.venv\Scripts\python.exe scripts\nanbeige\prime_cli.py --cached pods list --output json
```

`scripts/nanbeige/run_preflight_remote.sh` is ready for a provisioned CUDA PyTorch
pod. It installs the pinned Transformers version and runs 32 BF16 forwards twice:
once normally and once with capture hooks. It requires matching output logits and
exactly two calls per selected physical module. Outputs include separate pass
activations at physical layers 11/21 and the final RMSNorm, prompt IDs, reference
next-token logits, and a manifest. The small capture is not an SAE training corpus.
The process timeout does not stop cloud billing; pod termination and artifact
retrieval must be handled by the launch controller. Do not leave a pod running.

## Invocation-aware VPD and SAE comparison

Nanbeige has 22 physical layers used in two passes. Shared matrices must have one
set of decomposition parameters; pass-indexed activation caches and causal masks
must distinguish both invocations. A physical weight intervention normally acts
on both passes. A pass-specific activation intervention is a separate diagnostic.
The existing single-call VPD effect capture cannot be used unchanged.

`recurrent.py` adapts the math in
`D:\research_engine\param-decomp-rust\nano_param_decomp\run.py`. Sections A-F are
vendored in `vpd_math.py` with their MIT license and source SHA-256. The original
external repository is unchanged. V/U and faithfulness are counted once per
physical matrix; CI inputs, stochastic masks, PPGD sources, and activation caches
are indexed by physical module and invocation. The causal-importance network
is an analysis/training instrument, not a claim that Nanbeige has a router.

The bounded optimizer retains faithfulness, importance minimality, stochastic
reconstruction, and persistent adversarial reconstruction. It uses constant
learning rates for calibration. Small runs are explicitly not accepted as
validated mechanistic decompositions. The low component counts are compatibility
budgets and are not expected to faithfully reconstruct full Nanbeige matrices.

For cross-pass differences, use one common 2x dictionary (6,144 features) per
activation site, fitted on balanced rows from both passes of training examples.
Do not subtract indices from independently trained dictionaries. The runner
trains one dictionary at a time on CPU and records evaluation reconstruction
quality separately by pass. It does not pool training and evaluation examples.
The current FP32/Adam design requires roughly 576 MiB for parameters, gradients,
and optimizer state, plus batches, temporary tensors, and the CUDA runtime; this
is a sizing estimate for the full-size SAE, not a measured training run. Keep train/eval splits
grouped by problem family/pair. Validate quantization effects against BF16 before
using Q4 changes to draw mechanistic conclusions.

### Local verification

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_nanbeige_looped_capture tests.test_nanbeige_multipass -v
.\.venv\Scripts\python.exe -m experiments.nanbeige_looped.run_multipass --canary --output-dir artifacts\nanbeige_looped\multipass_canary --vpd-steps 3 --sae-steps 40 --components 4 --measure-components 2
.\.venv\Scripts\python.exe -m experiments.nanbeige_looped.run_multipass --architecture-canary-dir models\Nanbeige4.2-3B-tokenizer --output-dir artifacts\nanbeige_looped\architecture_canary --vpd-steps 3 --sae-steps 40 --components 4 --measure-components 2
```

Choose a fresh output directory for each run. The architecture canary verifies
publisher-source hashes and exercises 22 layers x 2 passes with 32-wide random
weights. It has no pretrained language capability. Both canaries actually train
VPD and SAEs, but their feature effects have no established semantic meaning.

### Pretrained capture

After the existing BF16 preflight succeeds on a GPU, run from the project root:

```bash
python -m experiments.nanbeige_looped.run_multipass \
  --model-dir /path/to/pinned/Nanbeige4.2-3B --device cuda \
  --output-dir artifacts/nanbeige_looped/bf16_multipass_calibration \
  --layers 11,21 --components 8 --measure-components 2 \
  --vpd-steps 50 --warmup-steps 50 --sae-steps 1000 --rows 16 --seq-len 64
```

This selects zero-based layers 11 and 21, decomposes their MLP down projections,
and observes both layer outputs plus the loop-boundary norm. `--layers` can select
other physical layers. Gradients from all invocations accumulate into the same
physical factors. Gradient checkpoint recomputation and cached generation are
unsupported. All captures use a single unpadded teacher-forced sequence.

Default examples are an arithmetic **prompt-prefill calibration**, without
generated CoT. `--dataset` accepts JSONL records containing `id`, `pair_id`,
`family`, `split` (`train` or `eval`), and nonempty model-native `token_ids`.
Use full teacher-forced reasoning prefixes there to inspect CoT positions.
The calibration has a 512-token input limit. Pair IDs and exact token sequences
may not cross splits; family overlap is disclosed in the manifest. Defaults
hold out pairs within addition, not an entire task family. By default VPD trains
on equal-length unpadded prefixes; activation analysis captures aligned tail positions.
For generated trajectories, pass `--training-window generated_prefixes` and
provide `prompt_tokens` on every training record. This fits prefixes ending at
25%, 50%, and 100% of the continuation, preserving the entire prompt and preceding
generation. `--max-example-tokens` bounds these contexts; `--seq-len` controls
only the default fixed-prefix mode. Factors and the calibration network are
shared; persistent adversarial state is separate for each sequence length.
The manifest records each selected prefix. This mode passed small-architecture
optimizer controls but still requires pretrained GPU memory and quality checks.

Each effect archive contains baseline and +/-alpha perturbations at first pass,
second pass, and both passes. Both-pass intervention adds the same rank-one
weight component at both invocations. Single-pass interventions temporarily
break sharing for diagnosis. Saved arrays include:

- Original token IDs and absolute sequence positions.
- Hidden states and common-dictionary SAE activations for every condition/pass.
- Baseline pass-1 minus pass-0 differences in hidden and SAE coordinates.
- Per-scope positive effects and central finite differences for each site/pass.
- Interaction: both(+alpha) - first(+alpha) - second(+alpha) + baseline.
- Last-token logits; JSON summaries include KL and reference-top-token logit
  changes as output proxies, not answer accuracy.

Second-pass-only perturbations are checked to leave first-pass captures exactly
unchanged. VPD factors are stored once per physical matrix in `vpd_factors.npz`;
`ci_state.pt` and `vpd_training.json` retain the invocation-aware CI model and
configuration. SAE dictionaries use the project's NPZ dictionary field names.
`sae_quality.json`, `effects.json`, and the hashed manifest bind the outputs.
Positive/negative SAE effects are computed by re-encoding actual perturbed
activations, not by projecting hidden-state differences onto decoder vectors.

No semantic acceptance threshold is claimed. Low reconstruction quality,
dictionary instability, or large finite-dose nonlinear interactions require
further work before interpretation. A task-specific behavioral endpoint and
held-out causal validation are still needed for a mechanism claim.
