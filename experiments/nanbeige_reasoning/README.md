# Nanbeige reasoning and pass-resolved interpretation

The prospective [cross-research utility addendum](../../VPD_CROSS_RESEARCH_UTILITY_ADDENDUM_V1.md) applies to future intervention selection: a nine-action catalog, global/geometric/interval selectors, complete-task utility, missed-challenger regret and separately charged costs. Existing cohorts remain exploratory and execution gates remain in force.

Run commands from `C:/projects/VPD` with the project virtual environment.

```
python -m experiments.nanbeige_reasoning.fetch_sources
python -m experiments.nanbeige_reasoning.build_dataset
python -m experiments.nanbeige_reasoning.prepare_pilot
python -m experiments.nanbeige_reasoning.audit_hermes
python -m experiments.nanbeige_reasoning.audit_graders
python -m experiments.nanbeige_reasoning.audit_futoshiki
python -m pytest tests/test_nanbeige_reasoning.py
python -m scripts.nanbeige.prepare_reasoning_bundle
```

The GPU launcher is `scripts/nanbeige/launch_reasoning.sh`, accessed through the existing Prime CLI wrapper. It owns one temporary A100, retrieves results, restores the prior SSH selection, and terminates the pod within its eight-hour outer bound. The prepared full-context VPD stage has a six-hour bound; the earlier one-hour launcher was insufficient for this expanded fit. No new rental is implied by preparing these scripts. Do not start a duplicate while a run is active.

After retrieval:

```
python -m scripts.nanbeige.collect_reasoning artifacts/nanbeige_reasoning/prime-RUN
python -m experiments.nanbeige_reasoning.scoring artifacts/nanbeige_reasoning/prime-RUN/retrieved/artifacts/nanbeige_reasoning/reasoning_capture
python -m scripts.nanbeige.summarize_multipass artifacts/nanbeige_reasoning/prime-RUN/retrieved/artifacts/nanbeige_looped/bf16_multipass_calibration
```

The dataset retains all four source lanes; the 18-input subset is a feasibility run. Logic answer formats come from each task's own prompt. ARC is graded per test input and per complete task; INTELLECT splits are local holdouts of public RL training data. Gold answers and grader metadata stay in a separate sidecar and are not uploaded with the inference bundle.

Each native continuation is replayed at exact token positions through both invocations of all 22 physical layers. Selected sites have a common frozen SAE encoder and VPD interventions at first, second, and both passes. The initial eight-component calibration is documented separately; the new reasoning run requests 64 components. Neither component count nor SAE reconstruction alone establishes a reasoning mechanism.

The paper and evidence ledger are under `artifacts/nanbeige_reasoning`. Historical 27B runs used substantially smaller output budgets and cannot be treated as matched controls. The reproduced 109/109 Campsite control is an enumerator with an official verifier, not a neural TRM-only result. See the audit's separate canonical, official-valid, and prompt-literal metrics.

Logic grading replaces upstream `eval` lookups with the bounded parser in `safe_expression.py`; score records identify that adapter. Grader health probes reconstruct known answers only for verification, never for model inference. The full Futoshiki audit found 697/714 supplied grids violate recorded inequalities; keep family-level results and this reference-quality diagnostic visible.

The completed local fallback is analyzed with `analyze_prompt_transfer` and plotted with `plot_prompt_transfer`. It records prompt positions from both passes at all 22 layers plus normalization and transfers three existing arithmetic dictionaries without refitting. Check its manifest for completion and its explained variance before interpreting SAE differences. It does not generate reasoning or benchmark accuracy.

The separate local Q4 experiment uses `run_q4_reasoning`. It compares both loop counts with identical Q4_K_M weights and q8_0 KV caches. Its current 2,048-token cap is a feasibility setting; retain censored traces and do not present them as a full reasoning benchmark. On a completed run, `replay_q4_reasoning` captures activations directly through the pinned llama.cpp C API, and `fit_q4_reasoning_saes` fits shared dictionaries on training trajectories only. The finite `finish_q4_pipeline` can wait for a verified generator PID and run these stages sequentially. It does not mark the broader research objective complete.

Direct callback controls are in `q4_callback_preflight_v1`: pretrained logit parity, logical layer counts, and first-pass invariance. GPU mode explicitly loads the pinned CUDA dependencies and requires a CUDA device; it must not silently fall back to CPU. See `STATE.md` for current live handles before starting another workload.

The finite pipeline completes grading, activation/VPD capture, and SAE fitting for the initial budget before its optional `--extended-run` follow-up. That follow-up uses 8,192 output tokens and 10,240 context tokens, preserving the same questions and weight/KV precision in a separate directory. Its longer budget was motivated by initial-run censoring and is exploratory. GPU allocation is still subject to available memory; failure must stay visible rather than trigger an implicit precision change. Prompt-cache RAM is disabled. Experimental cooling pauses did not produce sustained speed gains and are disabled by default; no device power, fan, or clock settings are modified.

Grading also writes `paired_scores.json`: ARC comparisons use complete tasks, and INTELLECT comparisons use questions. Missing conditions, wholly unobserved tasks, and ungradable pairs remain visible separately from success/failure counts. `generation_health.json` records continuous token-repetition diagnostics independently of scoring and explicitly labels partial runs. Neither token diversity nor an ablation's general generation damage establishes reasoning compression.

`cache_diagnostics` compares full-prefix and cached logits in BF16 and FP32 without relaxing the generation gate. `launch_cache_diagnostics_india.sh` selects that short diagnostic mode on the regional A100; it does not launch benchmark generation or VPD fitting. Preflight numerical and greedy checks are now recorded before rejection, including cache-layer counts. Each future regional launcher freezes its upload bundle and receipt in the run directory before provisioning.

The local finite pipeline allows one four-hour continuation of the initial generator after a successful time-budget pause. It preserves the original inference arguments and requires the runner's hash and saved-prefix checks. Crashes, contract changes, or a second incomplete result stop the pipeline for inspection.
