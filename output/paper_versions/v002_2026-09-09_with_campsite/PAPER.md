# Interpreting looped layers: pass-resolved activations, parameter components and sparse features in Nanbeige

Working methods draft, 9 September 2026. This study demonstrates how to instrument and audit looped layers in a small open model. Tables report completed measurements only; broader performance comparisons remain uncompleted extensions.

## Abstract

Looped language models reuse the same parameters at different points in a computation. We present a reproducible workflow for interpreting these repeated invocations in the open Nanbeige4.2-3B checkpoint: index activations by physical layer and pass, decompose shared parameters with pass-specific intervention masks, compare sparse features in a common dictionary, and independently verify saved measurements. A versioned corpus contains 34,886 question records from ARC-AGI-1, ARC-AGI-2 and INTELLECT-3-RL, with grading metadata separated from inference inputs. The worked example uses 18 fixed inputs and captures both invocations of all 22 physical layers, with focused parameter and sparse-feature analysis at layers 12 and 22 and sparse-feature analysis at loop normalization.

Both 2,048-token and 8,192-token Q4 pilots are complete. The longer pilot produces 36 outputs and 280,325 generated tokens; its native trajectories yield 900 activation arrays, 36 frozen-basis VPD coordinate archives and 54 SAE feature-difference archives, all independently verified. Five native two-pass traces finish before the cap, while all one-pass traces reach it. Both conditions score zero on eight evaluation units. A separate 64-component BF16 VPD fit on initial Q4 trajectories captures about 2.2% of selected weight energy and supports signed interventions on individual passes. Held-out reconstruction exposes important SAE limitations, including negative second-pass explained variance at layer 22 in the longer Q4 run. A subsequent paired CUDA control improves that pass from -0.180 with AdamW to 0.453 with Muon-hybrid, while slightly worsening reconstruction at layer 12. The contribution is an executable method with explicit measurement checks and failure modes. This small, censored pilot neither establishes nor excludes a recurrence benefit at larger scales; capability thresholds and answer-level mechanisms remain prospective questions. A completed 70-continuation sparse-feature steering test yields three correct condition outputs on one ARC test input, but zero complete-task gains; 62 outputs reach the token cap. A fresh 109-puzzle Qwen3.5-27B Campsite control scores 0/109 with the vanilla prompt, 13/109 with the plain skill and 35/109 after routed correction. The frozen gate, synthetic support and different decoding budgets limit cross-model interpretation.

## 1. Research question

The primary question is how to make repeated uses of shared layers separately observable and experimentally addressable. A small open checkpoint suffices to demonstrate pass-indexed activation capture, a shared parameter decomposition with pass-specific interventions, and sparse feature differences measured in one common dictionary. The contribution is a reproducible measurement method and an audit of its failure modes. Demonstrating that recurrence improves reasoning outcomes would additionally require interventions tied to scored behavioral endpoints; a larger-model comparison is not required to demonstrate the method. The Hermes controls provide historical context and are not a prerequisite for this demonstration.

The workflow has five parts. First, identify physical shared weights and their invocation order; index captures by input, token position, physical layer and pass. Second, replay identical token sequences and check that capture callbacks preserve outputs, reporting any generation/replay disagreement. Third, fit one parameter basis per selected physical matrix, retain its residual, and perturb a component in the first pass, second pass or both. Fourth, fit a common SAE on training examples from both passes and compare feature activations at matched held-out positions; report reconstruction quality separately for each pass before interpreting differences. Fifth, verify saved factors, feature encodings, intervention arithmetic and provenance independently. This study implements those steps across all 22 physical layers for activation capture, with focused VPD measurements at layers 12 and 22 and SAE measurements at those sites and loop normalization. Its fixed two-pass execution supplies invocation labels without an adaptive router.

Nanbeige is an inspectable model for these experiments. Its publisher describes parameter reuse through a looped transformer architecture. Our pinned configuration and instrumented execution use 22 physical transformer layers twice, with normalization at each loop boundary. This is fixed recurrence. Neither this implementation nor its measurements establish facts about Astra's undisclosed architecture. [Nanbeige model and source](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/tree/451ed48c3273ecef7ea8faaa43c31ce529763bb1).

The term “tuck” is treated as a hypothesis requiring an operational definition. One prospective test is whether a second-pass feature change carries task-relevant information that causally improves a later answer while reducing the need for additional generated reasoning. A shorter trace, a smaller activation norm, or a second-pass router count alone would not satisfy this definition. Recursive architectures and external recurrent controllers must be evaluated with their own training and inference budgets. [Recursive reasoning with tiny networks](https://arxiv.org/abs/2510.04871).

The Nanbeige report describes diminishing additional gains beyond two passes, weaker results with cross-pass KV sharing, and better performance when recurrence is trained from scratch. It also uses difficulty-aware length penalties during reasoning RL, alternating constrained and free-expansion phases. Thus post-training creates a separate potential cause of shorter reasoning. Our checkpoint ablation cannot separate this training effect from the architecture's contribution; intermediate training checkpoints would be needed. These are publisher-reported findings, not replications in our experiment. [Nanbeige technical report, Sections 2.1 and 3.2.3](https://arxiv.org/html/2607.22083v1).

## 2. Sources and dataset

All inputs are pinned by revision and file hash in `sources.lock.json`. The four source lanes are retained separately. INTELLECT examples come from the public RL **training** corpus; their local splits are not an official held-out test and do not establish freedom from model pretraining exposure. ARC evaluation here is public, not private. [ARC-AGI-1](https://github.com/fchollet/ARC-AGI), [ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2), [INTELLECT-3-RL](https://huggingface.co/datasets/PrimeIntellect/INTELLECT-3-RL/tree/a9eb9183224cec63fe9a71e7a3a80cff14868624).

| Lane | Calibration/train | Local dev | Evaluation | Historical evaluation | Quarantine |
|---|---:|---:|---:|---:|---:|
| ARC-AGI-1 | 415 | 0 | 419 | 0 | 1 |
| ARC-AGI-2 | 681 | 0 | 167 | 0 | 395 |
| INTELLECT logic | 8,952 | 1,145 | 1,050 | 109 | 391 |
| INTELLECT math | 16,931 | 2,068 | 2,162 | 0 | 0 |

Counts are question records: ARC contributes one record per test input, so these numbers are not counts of independent ARC tasks. Every test input of a task receives the same partition. ARC groups join both task identifiers and hashes of public examples/test inputs across releases; any alias of an evaluation task is excluded from calibration. INTELLECT's 391 embedded ARC examples are quarantined pending a complete mapping. The 109 Campsite examples are reserved for comparison with historical Hermes receipts. Exact question duplication is also checked across partitions; semantic near-duplicate decontamination remains future work.

The model-facing JSONL contains questions, provenance, and grouping identifiers. A separate grader sidecar contains ARC test outputs, INTELLECT math answers, or logic metadata. Neither the sidecar nor historical predictions are included in the GPU inference payload. Demonstration outputs supplied as part of an ARC task are legitimate inputs.

The feasibility subset selects two groups per lane and calibration/evaluation partition by a fixed hash order, with one historical Campsite evaluation case. All test inputs of the selected ARC groups are retained, producing 18 inputs. Selection uses prompt length and family, never gold outcomes; 13 candidate groups exceeded the 2,048-token prompt cap. Consequently this subset cannot estimate full-benchmark performance without selection bias. The full corpus remains available for subsequent runs.

The subsequent benchmark cohort has also been materialized in `benchmark_v1`. It includes every public ARC-AGI-1 evaluation task (400 tasks, 419 test inputs), every public ARC-AGI-2 evaluation task (120 tasks, 167 inputs), 200 fixed logic holdout examples across 27 eligible families, 200 fixed math holdout examples, all 109 historical Campsite cases, and 128 calibration records. Its 1,223 records contain 2,528,898 prompt tokens, with median 929 and maximum 17,660; none is excluded by prompt length. Logic sampling is family-stratified and must be reported as such, while the Campsite historical slice remains separate. Materializing this cohort does not mean it has been evaluated.

## 3. Measurement methods

### 3.1 Generation and pass ablation

The planned BF16 experiment compares native two-pass execution with a one-pass ablation of the same checkpoint. Both use greedy decoding, thinking-enabled prompts, and a 2,048-new-token cap. This is a test of the trained model's dependence on its second pass; it is **not** a comparison between independently trained one-pass and two-pass architectures. Equal token caps are not equal compute. Elapsed time, prompt/generated token counts, and executed physical-layer token counts are recorded; the latter are a proxy rather than a FLOP estimate. Unfinished reasoning and token-cap censoring are retained in the denominator and reported explicitly.

Cached decoding is checked against full-prefix execution for both loop counts before generation. For each native continuation, the exact prompt and generated token sequence are replayed without caching. Hooks retain both invocations of all 22 physical layers and loop-boundary normalization. Up to eight positions are retained in each of four regions: prompt suffix and the early, middle, and late portions of generation. Token IDs and absolute positions accompany every activation archive. These regions are positional, not presumed semantic reasoning stages.

The larger-cohort runner supports left-padded batches and durable resume records, with a separately recorded default cap of 8,192 new tokens. It avoids allocating vocabulary logits for unused prefill positions during generation; that optimization is disabled for losses and parameter decomposition. Batched decoding, final-position logit parity, interruption recovery, and complete-task ARC scoring are checked. Every resume contract includes question, checkpoint, and decoding hashes. Completed batches are skipped; an interrupted batch is regenerated at its original composition, and already-saved trajectories must match before missing records are appended. Replayed work is included in batch-attempt receipts. A timed pause is recorded as incomplete, never as a successful benchmark completion.

The local behavioral fallback uses the separately pinned Q4_K_M conversion of Nanbeige, with q8_0 key/value caches and a 4,096-token context. It renders the same official HF input token IDs and compares the metadata loop-count settings 2 and 1 using the same quantized physical weights. Greedy decoding, a 2,048-token continuation cap, repeat penalty 1, and disabled context shifting are fixed across conditions. This is a quantized-model experiment; BF16 behavioral parity has not been established. Runtime source confirms physical weight sharing and separate logical KV indices across passes. The full command, DLL/executable hashes, settings returned by the server, stop reasons, and exact output tokens are retained. Laptop thermal throttling affects elapsed time, which is not a controlled compute comparison. [Pinned runtime implementation](https://github.com/ggml-org/llama.cpp/blob/0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0/src/models/nanbeige.cpp).

Direct C API callbacks also support activation capture from this same Q4 model. A completed pretrained CPU preflight observes 23 arrays under one pass and 46 under two passes on 32 tokens. Capturing leaves the full next-token logit vector exactly unchanged in both cases. All 23 first-pass sites are also exactly invariant to enabling the second pass in this control. These checks establish callback and pass-index behavior, not reasoning accuracy. Completed generated-trajectory replay records conditional greedy agreement because chunked replay and single-token generation can differ numerically; Section 4.7 reports the observed mismatches.

### 3.2 Parameter components and shared sparse dictionaries

We adapt Goodfire's adVersarial Parameter Decomposition (VPD), using the core mathematics from its compact `nano_param_decomp` implementation. VPD decomposes parameters; a sparse autoencoder (SAE) instead learns a dictionary of activation features. Our adapter keeps one physical set of parameter factors shared across recurrent invocations while assigning separate invocation caches and intervention masks. The retained upstream source sections, source SHA256 and MIT license are recorded in `experiments/nanbeige_looped/vpd_backend_provenance.json` and `VPD_BACKEND_LICENSE` in the workspace. This is an adaptation to selected matrices in a recurrent model, not a replication of the original whole-model decomposition. [VPD method](https://www.goodfire.com/research/interpreting-lm-parameters), [upstream implementation](https://github.com/goodfire-ai/param-decomp).

For each selected linear map, the fitted product `V @ U` has the orientation of the transposed PyTorch weight. The residual is `W - (V @ U).T`; retaining it preserves the original map algebraically when all components and the residual are included. Reported component coverage concerns the fitted product alone. Joint calibration optimizes four terms: parameter faithfulness, importance minimality, stochastic reconstruction and adversarial reconstruction. The importance predictor is auxiliary calibration machinery; it does not make Nanbeige's inference loop adaptive. Our short optimization budgets and low fitted coverage limit what can be inferred from an individual component.

At the MLP down projections of physical layers 12 and 22, the decomposition assigns one V/U component basis to each shared weight matrix. Invocation-specific calibration state and intervention masks distinguish the passes. The initial calibration used eight components, 50 warm-up updates, and 50 updates with the four VPD objectives. The completed generated-prefix run uses 64 components with the same update counts. A 64-component fit should not be presumed adequate merely because it is larger; reconstruction and intervention stability must be measured.

For a component contribution \(D_c\), we compare additions \(\alpha D_c\) at the first invocation, the second invocation, and both. Applying the same contribution twice preserves sharing in real arithmetic; materializing a BF16 matrix edit can round differently. Single-pass interventions intentionally break sharing to localize propagation. We retain positive and negative doses, central differences, and the vector interaction

\[
I_c(\alpha)=h_{\mathrm{both}}(\alpha)-h_{\mathrm{first}}(\alpha)-h_{\mathrm{second}}(\alpha)+h_0.
\]

The first-pass effect has more downstream transformations before the final endpoint. Different input states, downstream depth, finite-dose nonlinearity, and BF16 rounding must therefore be separated before interpreting an asymmetry as a specialized reasoning role.

At each selected capture site, one top-k SAE is fitted on balanced training rows from both passes and frozen across all conditions. Feature changes are actual nonlinear re-encodings, \(\Delta z=E(h+\Delta h)-E(h)\), not decoder projections of activation differences. Reconstruction is evaluated separately by pass. Feature IDs have no validated semantic interpretation at this stage. The prepared BF16 run retains full generated contexts for capture and effect evaluation. Its VPD fit now selects unpadded prefixes ending at 25%, 50%, and 100% of each training continuation, retaining every preceding prompt token. It shares physical factors and the invocation-aware calibration network across lengths, with separate persistent adversarial state for each length. A small random-weight fixture using the publisher's recurrent architecture passed actual alternating-length optimization and mask-restoration checks. The first generated-prefix attempt exhausted GPU memory before completion and saved no fitted decomposition; the corrected completed run is reported in Section 4.9. Earlier completed arithmetic results used their original fixed prefix and are unchanged.

The Q4 replay additionally retains inputs to the physical layer-12 and layer-22 MLP down projections at both invocations. A completed 16-token CPU control captured these four extra input arrays with unchanged logits. The completed transfer analysis computes coordinates \(c_p=x_p V\) and basis outputs \(c_p U\) using the earlier frozen BF16 VPD factors. This is a transfer diagnostic across weight precision, not a newly fitted Q4 decomposition, a causal intervention, or coverage of the omitted residual. The low coverage of the original eight-component bases remains a limitation. New Q4 SAEs were fitted on training trajectories from the completed initial run, with held-out reconstruction reported separately.

A separate cross-precision calibration has now completed and passed independent verification: BF16 Nanbeige receives the exact native two-pass Q4 trajectories from the completed initial pilot, comprising eight training and ten held-out inputs with 49,987 total tokens. All 18 continuations are budget-censored. This VPD uses 64 components at physical layers 12 and 22, 50 faithfulness warm-up updates and 50 joint updates, followed by shared-SAE fitting and signed pass-specific interventions. Full conditioning is retained at the three training-prefix fractions above. Source tokens, split assignments and generation hashes are checked before bundling; remote checkpoint assets must match all 12 pinned file hashes. This is teacher-forced BF16 analysis of Q4-generated trajectories, not BF16 self-generation, Q4 parameter fitting, or an answer-level accuracy experiment. It uses no KV cache and does not relax the separate cached-generation equivalence checks. The original specification is in `cross_precision_vpd_inputs_v1/contract.json`. That attempt logged its first update, then failed while allocating another 1.51 GiB for the adversarial KL computation on a 39.49 GiB GPU, with 38.51 GiB in use by the process. Exact returned input trajectories and cleanup were verified; no completed fitted factors were saved. The revised `cross_precision_vpd_inputs_v2/contract.json` preserves the data, objectives, prefix lengths and update counts while offloading tensors saved for backward to CPU RAM. It introduces no forward recomputation. Small CPU and actual CUDA controls reproduced all losses and 28 parameter/gradient arrays exactly over three varying-length updates. The pretrained retry completed all 50 joint updates, with peak allocated CUDA memory of 23,893,116,928 bytes. The first full-pretrained adversarial loss nevertheless differs between the separate original and retry pods (0.26720446 versus 0.26673719, about 0.175%); the other three first-step objective values agree. We have not isolated the cause and do not claim full-model bitwise equivalence. Completed decomposition and held-out measurements are reported in Section 4.9.

### 3.3 Behavioral grading

ARC predictions require a rectangular JSON grid of integers 0–9 and exact equality with the target. Both per-input accuracy and all-test-input task success are reported. This feasibility run has one greedy attempt and does not estimate pass@2. Logic uses pinned task-specific verifiers, including explicitly retained task families rather than the environment's default skip list. A documented adapter replaces their Python `eval` lookups with bounded literal/arithmetic parsing: function calls, attribute access, and comprehensions are rejected. This preserves the tested expected answer forms without executing generated Python; it is an adapter, not byte-for-byte official execution. Math uses the official boxed-answer extraction and `math-verify` equivalence with pinned dependencies. On Windows, an outer process timeout replaces the library's incompatible internal multiprocessing timeout; answer extraction and equivalence logic remain unchanged. Infrastructure grading errors are recorded separately from incorrect answers. [Official environment source](https://github.com/PrimeIntellect-ai/research-environments/tree/48cb24baecee7dc76bb55be86bf4a181ffcb60e8).

## 4. Completed results

### 4.1 Pretrained implementation calibration

The completed initial experiment used 12 training and four evaluation arithmetic prompts, with thinking disabled and the final 16 prefill positions retained. The retained token IDs were identical across examples, although preceding questions differed. On this narrow suffix, SAE explained variance ranged from 91.87% to 99.12% across the measured sites and passes. These figures are not evidence of broad semantic feature quality.

| Physical layer | Component | First-pass effect norm | Second-pass effect norm | Both-pass effect norm | Interaction norm |
|---|---:|---:|---:|---:|---:|
| 12 | 0 | 32.882 | 17.140 | 32.946 | 23.964 |
| 12 | 1 | 33.830 | 17.604 | 33.928 | 24.623 |
| 22 | 0 | 23.764 | 0.615 | 23.764 | 0.769 |
| 22 | 1 | 25.324 | 1.005 | 25.331 | 1.257 |

These are pooled final-normalization activation norms at dose 0.05, not task scores. For the two layer-22 components, the first-pass effect is approximately 25–39 times the second-pass effect. The selected eight-component bases reconstruct only 0.268% and 0.260% of squared weight energy. The remainder stays in the residual component. The calibration passed exact adapter-logit parity and second-pass intervention causality checks; 74 original artifact files were hash-verified. See `../nanbeige_looped/PRETRAINED_RESULTS.md` and its linked manifests.

A subsequent independent check reloaded the dictionaries and verified all 16 saved intervention records, including 1,632 pass-difference, finite-difference, interaction and causal-order calculations and 672 nonlinear SAE re-encodings. It also reproduced all six held-out SAE quality measurements. This strengthens saved-artifact verification without rerunning the model or changing the narrow prefill scope. The receipt is `../nanbeige_looped/prime-20260906T034844Z-719/retrieved/artifacts/nanbeige_looped/bf16_multipass_calibration/independent_verification.json`.

### 4.2 Hermes solver and historical TRM audit

We pinned the supplied Hermes-Skills repository and recovered all 109 corresponding historical Campsite rows by matching target grids. Targets were used for this retrospective matching and final exact scoring, never supplied to candidate generation. We reran its public enumerator with only the original grid and row/column constraints. It reproduced 92 unique, 16 two-solution, and one four-solution case under its relaxed rule contract. [Pinned Hermes source](https://github.com/MoralityLabAI/Hermes-Skills/tree/07a177f2751c2adb1cc16dc819c1173c0cd57ad5).

| Arm on the same 109 cases | Canonical target match | Official verifier accepted |
|---|---:|---:|
| Historical vanilla 27B predictions | 0/109 | 0/109 |
| Historical generic skill | 1/109 | 1/109 |
| Historical logic skill | 0/109 | 0/109 |
| Historical logic skill + TRM | 33/109 | 38/109 |
| Replayed public solver, first candidate | 96/109 | 106/109 |
| Public solver, first candidate passing official verifier | 99/109 | 109/109 |

The last arm is a new deterministic control: it filters the enumerated candidates using an official verifier object containing only public puzzle inputs. It does not use the canonical target to select a candidate. It establishes complete verifier acceptance for this 109-row slice. It does not establish that a neural TRM alone solved all INTELLECT logic, all math, or ARC. Historical model receipts are not a fresh, compute-matched 27B comparison. Recovered configurations identify Qwen3.5-27B Q4_K_M with 96 output tokens for logic and 32 for numeric-only math, whereas the new Nanbeige run allows 2,048. Thinking settings, retrieval exposure, and controller costs must also be reconciled before attributing a scale advantage.

For the separate 200-row historical math slice, independent string-exact recomputation agrees with the receipt: vanilla 17/200, generic skill 14/200, math skill 12/200, and routed math-skill/TRM final 14/200. The first three arms record 200 model calls each; the TRM arm records 400. The available receipt therefore supports a low-scoring, tightly constrained 27B baseline, but does not itself demonstrate a complete math-benchmark clear. It should not be compared directly with boxed-answer symbolic equivalence scores without regrading and aligning task identities.

A subsequent task-identity audit finds that all 200 ordered historical labels match the first 200 rows of the pinned public math source. The 25 historical observations recoverable from support banks also match those source prompts exactly, with no contradictions and no overlap with the larger cohort's planned calibration records. The other 175 historical prompt identities remain provisional because the original trajectory file is missing. This evidence strengthens the candidate mapping but does not make the output budgets or task exposure comparable.

An archived support-bank audit matches public Campsite grids and row/column constraints for 100 historical questions. Of these, 77 pass the retriever's bucket and supervision-weight filters; all 100 enter the archived critic's source bank. Historical logic/TRM outputs pass the official verifier on 21/77 retriever-member cases and 17/32 remaining cases, versus zero vanilla successes in either partition. The latter 32 cases are not a clean support holdout: 23 still appear in the critic's source bank. These counts describe overlap, not which examples reached a particular model call.

The six eligible math support entries match six evaluated prompts and their reference answers. All six historical TRM receipts contain a retrieved answer equal to the reference, although exemplar IDs are absent. On these six rows, vanilla, generic skill, math skill, and routed TRM score 3, 1, 4, and 6 exact matches respectively. On the remaining 194 rows, they score 14, 13, 8, and 8. Thus the aggregate two-answer advantage of routed TRM over the math skill occurs entirely within this support-member partition. Support selection was not randomized, so these partitions do not estimate a causal retrieval benefit. Archived runner code contains normalized-observation exclusion filters, but execution of that exact code for these receipts has not been established. The bank files and audit source are hashed; the audit does not claim that all historical runtime provenance is recovered.

Inspection of the archived hybrid runners further constrains the meaning of the TRM label in these receipts. The logic loader constructs a retrieval bank, a nearest-neighbor critic, and a constraint/exact-search corrector; the math loader constructs a retrieval bank and nearest-neighbor critic. These auxiliary paths do not load a neural recurrent checkpoint. This statement concerns the inspected historical paths, not every system called TRM in the broader Hermes research. Source hashes and entry points are recorded in `hermes_audit/controller_implementation_audit.json`.

The larger benchmark remains fixed. `hermes_audit/benchmark_support_exclusions.json` identifies its 100 public inputs that overlap archived critic/source support, for exclusion from support banks in a fresh comparison. It does not alter either the benchmark or historical receipts. `retrieval_exposure_summary.json` and its row records preserve the overlap partitions, and their totals reproduce every previously audited historical arm.

![Canonical agreement and official acceptance on the 109 Campsite cases](figures/hermes_campsite_audit.png)

### 4.3 A scoring-contract discrepancy

The Campsite prompt literally requires each tent to touch exactly one tree. Only 9/109 supplied targets satisfy that condition, while all 109 pass the official verifier's one-to-one matching test. The Hermes enumerator instead admits any tent adjacent to at least one tree; its first candidates pass that relaxed check on 109/109 cases. Enumerating and filtering all candidates finds a literal-contract solution for only nine cases, so the discrepancy is not simply a poor first-candidate choice. Under the official verifier, candidate filtering resolves all 109 cases without canonical-answer access.

These are three different contracts. Reporting only exact target agreement would penalize some valid alternatives; reporting only the relaxed checker would admit three first candidates rejected by the official verifier. The primary comparison in this study uses official verifier acceptance, with canonical agreement and prompt-literal validity retained as diagnostic columns.

### 4.4 Logic reference and formatting audit

Positive-path probes use the first three source records in each of 29 logic families. They serialize supplied answers or solution metadata into task-specific output formats solely to check the graders; these reconstructions are never passed to the model. All three probes pass in 28 families. All 87 empty-answer probes fail. Futoshiki is the exception: after using its specified space-separated row format, the reference grids still fail the inequality checks.

A complete audit of all 714 Futoshiki source records finds that 697 supplied grids violate at least one recorded inequality; only 17 supplied grids pass the verifier. Nine of the ten Futoshiki records selected anywhere in the larger cohort have this issue. This demonstrates inconsistent reference solutions, not that the underlying puzzles are unsatisfiable or that every generated solution will fail. The cohort and scorer are retained, and this family must be reported separately with reference-quality diagnostics. Records and violated coordinates are preserved in `grader_audit/futoshiki_rows.json`. No reference answers have been repaired.

### 4.5 Completed 2,048-token Q4 pilot outcomes

The paired Q4 pilot completed 36 continuations on 18 public inputs: 73,728 generated tokens in total. All 36 continuations reached the 2,048-token cap, none closed the thinking delimiter, and none supplied an eligible completed final answer. Both conditions therefore scored zero on every evaluation unit, with no grader errors or missing pairs:

| Evaluation slice | Units per condition | One pass correct | Two passes correct |
|---|---:|---:|---:|
| ARC-AGI-1 | 2 complete tasks / 3 test inputs | 0 | 0 |
| ARC-AGI-2 | 2 complete tasks / 3 test inputs | 0 | 0 |
| INTELLECT logic, local evaluation | 1 question | 0 | 0 |
| INTELLECT logic, historical Campsite | 1 question | 0 | 0 |
| INTELLECT math | 2 questions | 0 | 0 |

This is an output-censored feasibility result, not evidence of equal underlying reasoning ability or an accuracy benefit from recurrence. ARC is paired at the complete-task level, requiring every test input to succeed; historical evaluation stays separate from the local holdout. Source and score hashes are preserved in `q4_reasoning_pilot_v1/pilot_outcome_summary.json`.

The output-token distributions nevertheless differ. Across all 18 paired inputs, median unigram entropy is 1.95 bits for one pass and 5.96 bits for two passes; the longest identical-token run has medians 20.5 and 2 tokens. Two-pass entropy is higher on all 18 inputs. Among the 10 evaluation input records, entropy medians are 2.34 and 5.70 bits. These are descriptive input-level statistics, not independent task-level significance tests. Structured answers can naturally have low token diversity. Ablating a model trained for two passes can damage general generation; these differences do not identify reasoning compression or a reasoning-specific mechanism.

![Paired token entropy and single-token repetition on the completed Q4 pilot](figures/q4_token_health.png)

Each point is one input; filled points are evaluation inputs and open points are calibration inputs. Dashed lines indicate equality between pass conditions. Both panels use the same actual 2,048-token continuations. The right panel uses logarithmic axes. No condition produced a completed eligible final answer at this cap.

The exploratory 8,192-token follow-up is now complete and reported separately in Section 4.12. It uses the same 18 inputs with a 10,240-token context and unchanged weight/KV precisions. Its first attempt stopped after saving 17 native outputs when a request exceeded the 4,096-second HTTP response timeout. A transport-timeout adjustment and successful saved-prefix check allowed completion without changing model or sampling settings. The original failure receipt remains in `runtime_observations/extended_generation_timeout.json`; failed-attempt logs and durable outputs were retained. Local elapsed time includes uncontrolled thermal, memory and interruption effects and is not a controlled compute comparison.

One one-pass ARC request exposed a response-transport failure: the server completed 2,048 tokens, but its content parser rejected malformed UTF-8 and returned HTTP 500. That request contributed no recorded data. An actual-runtime control reproduced the failure with a forced byte token and verified that an empty response parser preserves returned token IDs. Both parser settings reproduced a saved 16-token greedy prefix. The failed input was rerun successfully, preserving all earlier raw records and unchanged sampling. Forced token bias was confined to the control; raw token IDs remain authoritative, while decoded text may contain replacement characters. See `parser_failure_receipt.json` and `raw_token_transport_test/verification.json`. The pinned UTF-8 validation is visible in the [llama.cpp parser](https://github.com/ggml-org/llama.cpp/blob/0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0/common/peg-parser.cpp).

BF16 generated-reasoning results remain pending. An A100 attempt stopped before generation when one of 166,144 logits violated the original cache-equivalence tolerance (absolute difference 0.2734375; relative tolerance 0.02 and absolute tolerance 0.25). A subsequent diagnostic on an RTX 6000 Ada completed and was independently verified against both local checkpoint shards, the frozen source bundle, exact input tokens, and all saved logits. Both BF16 loop counts pass the unchanged full-versus-cached test, with maximum absolute differences 0.046875 and 0.25. FP32 arithmetic on the upcast model reduces those maxima to 0.00000882 and 0.0000610; both pass the predeclared 0.0001 absolute/relative checks, with correct 22/44 cache cardinalities.

Full-versus-cached agreement does not imply cross-precision greedy agreement. In the one-pass condition, BF16 has a zero top-two margin and chooses token 1628, whereas FP32 chooses token 2753 with margin 0.00461. In the native two-pass condition, both precisions choose token 2753. This single 32-token prefix diagnoses numerical behavior; it does not establish benchmark parity. No generation threshold was relaxed. Results and termination are verified in `prime-20260906T191119Z-509/cache_verification.json`. The successful diagnostic pod was terminated; a subsequent BF16-generation launch found no available matching GPU and produced no inference.

### 4.6 Completed local prompt capture and frozen SAE transfer

The local fallback completed all eight selected prompts: one calibration and one evaluation example per source lane. It executed the original BF16 checkpoint over 4,208 prompt tokens in 1,757.5 seconds of recorded forward/capture time. Each archive retains the first eight and last 24 positions, yielding 256 retained token positions and 368 finite activation arrays across 23 sites and two invocations. All archive hashes, invocation counts, and array shapes were independently verified. These are the shortest available inputs within each pilot lane/split, not a representative accuracy sample.

The three previously fitted arithmetic SAEs were frozen and applied without new fitting. This produced 24 archives containing both encodings and their actual difference, alongside hidden-state differences for all layers. Reloading each dictionary reproduced its original held-out explained variance within 1.1e-8, ruling out a dictionary-loading error in this control. Across the eight new examples, the reconstruction results were:

| Frozen dictionary site | Pass 1 explained variance range | Pass 2 explained variance range |
|---|---:|---:|
| Physical layer 12 | -0.0821 to -0.0819 | -0.0758 to -0.0737 |
| Physical layer 22 | -0.0584 to -0.0404 | -1.2935 to -0.9310 |
| Loop-boundary normalization | 0.0775 to 0.0897 | 0.1902 to 0.3120 |

Explained variance here is 1 minus reconstruction squared error divided by the within-example, within-pass sum of squared deviations from the mean activation. Negative values mean the reconstruction is worse than that mean baseline. These pooled-position results demonstrate poor transfer from the earlier fixed arithmetic suffix to the new combination of beginning and ending prompt positions. Task, token-position, thinking-template, and CPU/GPU execution differences are not separated by this test. The saved feature differences therefore cannot be treated as validated semantic reasoning features. Broader training-only calibration and generated reasoning remain necessary.

![Pass similarities and frozen SAE transfer quality across eight prompts](figures/prompt_pass_transfer.png)

### 4.7 Generated-trajectory Q4 captures

Native two-pass trajectories from all 18 pilot inputs have now been replayed through the same pinned Q4 checkpoint: 49,987 full-context tokens in 243.7 seconds of measured replay time. The archives retain 576 token positions across prompt, early, middle and late continuation regions. They contain 900 activation arrays: both invocations of all 22 transformer layers and loop-boundary normalization, plus the down-projection inputs at physical layers 12 and 22.

The 64-token GPU callback control produced exactly identical full-vocabulary logits with capture enabled and disabled. Conditional greedy checks during chunked trajectory replay matched 428/432 sampled next-token choices. The four mismatches had original-token logit gaps of approximately0.10–0.15. These are teacher-forced activations of the saved token sequences; they are not a claim that chunked replay reproduces every autoregressive choice. An initial process-level OpenMP conflict was resolved by separating a pure token-position helper from PyTorch imports, without a duplicate-runtime override.

The replay also produced 36 coordinate-transfer archives using the prior frozen BF16 VPD bases at physical 12/22. These store actual input projections, pass differences and basis-output differences, while omitting the residual outside the sparse basis. They do not constitute a newly fitted reasoning VPD or causal answer effects. Three shared SAEs have completed fitting on 256 retained training positions per pass from eight calibration inputs. Independent verification reproduced all 36 VPD coordinate archives, all 54 SAE feature-difference archives, and six held-out reconstruction measurements after reloading the dictionaries. All 900 activation arrays passed source-hash, shape, finite-value and position checks. The arithmetic verification does not validate semantic feature interpretations. See `q4_reasoning_pilot_v1/mechanistic_verification.json`.

A descriptive region analysis contains 1,656 hidden-state comparisons (18 inputs, 23 sites, four regions) and 144 frozen-VPD region comparisons (18 inputs, two sites, four regions). Each region contains eight retained token positions. Among the ten evaluation input records, median pass cosine at physical layer 12 ranges from 0.819 to 0.857 across regions. At physical layer 22 it ranges from 0.178 to 0.333, and median second/first activation norm ratios range from 0.162 to 0.253. The sharp layer-22 change is present in the prompt suffix as well as the generated continuation. These are medians over input records, including multiple inputs from the same ARC task; they are not independent task-level estimates.

The callback sites are the residual outputs before loop-boundary normalization. At the separately recorded normalization site, median norm ratios instead range from 5.20 to 6.14. This site dependence cautions against equating a smaller residual norm with information compression. The pinned graph places both physical-layer output callbacks before their corresponding normalization operations. [Pinned Nanbeige graph implementation](https://github.com/ggml-org/llama.cpp/blob/0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0/src/models/nanbeige.cpp).

![Pass cosine and norm ratio across all physical layers and four trajectory regions](q4_reasoning_pilot_v1/pass_region_analysis/pass_regions.png)

The generated-trajectory SAE at physical layer 12 yields held-out explained variance 0.2961 and 0.2511 for the two passes, versus training values 0.9832 and 0.9840. At physical layer 22, held-out values are 0.8662 and 0.0117, versus training values 0.9840 and 0.9381. These generalization gaps limit feature interpretation. Equal numbers of training rows do not equalize relative reconstruction loss: first-pass training variance at layer 22 is 5.62 times second-pass variance. This is a possible objective-scaling confound, not a demonstrated cause of the held-out gap; no objective has been changed in these fits. At loop-boundary normalization, held-out explained variance is 0.2496 and 0.4648, versus training values 0.8438 and 0.9939. All six held-out values were reproduced after independently reloading the saved dictionaries.

| Shared SAE site | Training EV, pass 1 | Training EV, pass 2 | Held-out EV, pass 1 | Held-out EV, pass 2 |
|---|---:|---:|---:|---:|
| Physical layer 12 | 0.9832 | 0.9840 | 0.2961 | 0.2511 |
| Physical layer 22 | 0.9840 | 0.9381 | 0.8662 | 0.0117 |
| Loop-boundary normalization | 0.8438 | 0.9939 | 0.2496 | 0.4648 |

The dictionaries remain shared across passes, with no evaluation fitting or pass-specific rescaling. The layer-22 second-pass reconstruction in particular is inadequate for strong feature-level claims.

### 4.8 Fresh external-controller controls

The empty archived logic bank after public-input exclusion was replaced, for a new controller-only experiment, by 128 separately constructed synthetic support puzzles. Another 32 synthetic puzzles were reserved for development and never entered retrieval. Generation uses seed 20260906 and eight fixed rectangular shapes from 4x4 to 7x6. Every constructed solution passes the official verifier and the literal per-tent adjacency check. All exact, rotated, and reflected aliases of the 109 public source puzzles are excluded, as are aliases within the synthetic set. The public set contains 108 distinct symmetry orbits. This construction favors puzzles satisfying both adjacency contracts and is not the publisher's generator or a reconstruction of historical training.

We ran the unmodified archived retriever and constraint corrector on the 32 synthetic development cases and all 109 public Campsite cases. These controls make zero language-model calls and do not constitute a full 27B+skill/TRM run. Retrieval supplies the nearest support solution as an initial candidate. Local repair first preserves the query's trees and dimensions, then performs up to the configured number of improving cell-sweep passes. Thus a zero-pass normalized initial grid differs from returning the raw retrieved answer. Search-enabled repair first attempts the archived exact routine and can fall back to local repair with an eight-pass cap and its original 200,000-node search bound.

| Controller-only arm | Public official acceptance | Synthetic development acceptance |
|---|---:|---:|
| Raw retrieved solution | 0/109 | 0/32 |
| Normalize retrieved grid, zero repair passes | 3/109 | 1/32 |
| Local repair, cap 1 | 25/109 | 8/32 |
| Local repair, cap 2 | 37/109 | 14/32 |
| Local repair, cap 4 | 38/109 | 15/32 |
| Local repair, cap 8 | 38/109 | 15/32 |
| Search-enabled repair, empty start | 40/109 | 32/32 |
| Search-enabled repair, retrieved start | 44/109 | 32/32 |

A separate validator using bit-mask dynamic programming for tent/tree matching reproduced all 1,128 saved official, literal and canonical outcome records. Literal adjacency and official matching are evaluated separately: two tents can each touch exactly one shared tree while failing a perfect matching. Four focused tests cover rectangular symmetry transformations, reproducible valid construction, and matching counterexamples. Original controller code and outputs remain unchanged.

Python call tracing then reproduced all 987 repair outputs exactly while observing actual execution. Search returned a solution on nine public cases and all 32 synthetic development cases, regardless of initial candidate. The remaining 100 public cases used local fallback. With search disabled and an eight-pass cap, public cases used one to five repair sweeps: counts 5, 57, 39, 7, and 1 respectively. The four- and eight-pass arms have identical final outputs on all 109 public cases and all 32 development cases. This is an observed early-stopping property of the external symbolic controller, not evidence about transformer routing.

Aggregate improvements can conceal individual regressions. Raising the local cap from two to four passes repairs two previously incorrect public cases but breaks one previously correct case. Changing the search-enabled initial candidate from empty to retrieved produces 13 gains and nine losses, for a net four-case increase. These paired controls identify effects within this particular controller and synthetic support bank. They do not isolate the historical language model's contribution or establish a neural recurrence mechanism. The perfect development acceptance of search-enabled repair also does not transfer to the public distribution.

![Verified external-controller performance across configured repair caps](hermes_controller_control_v1/controller_passes.png)

Artifacts are in `hermes_synthetic_campsite_v1` and `hermes_controller_control_v1`, including source copies, every answer, independent verification, paired transitions, and output-preserving execution traces. No neural controller was trained, and no new 27B inference was run in these controls.

### 4.9 BF16 generated-prefix VPD and pass-specific effects

The revised calibration completed on the exact 18 native Q4 trajectories, with eight training and ten held-out inputs. Its 64-component bases reconstruct 2.1919% and 2.1514% of squared weight energy at physical layers 12 and 22. Most weight energy remains in the residual. The increase from the earlier eight-component calibration is descriptive: both component count and calibration data changed, so this is not a controlled capacity scaling experiment.

All 20 held-out component-0 intervention records were retrieved and independently checked: 2,040 saved difference, interaction and causal-order calculations; 840 SAE re-encodings; and six held-out reconstruction measurements. A separate provenance check matches checkpoint assets, executed source, exact trajectories, 24 training prefixes and 50 joint updates to the frozen input contract. These checks verify saved arithmetic and lineage; they do not rerun the model interventions or validate feature semantics.

| Site | SAE held-out EV, pass 1 | SAE held-out EV, pass 2 |
|---|---:|---:|
| Physical layer 12 | 0.0035 | -0.0087 |
| Physical layer 22 | 0.6667 | -2.6589 |
| Loop normalization | -0.0774 | 0.1381 |

These dictionaries were fitted to 64 training rows per pass, retaining the final eight positions of each training trajectory. They use the same 6,144-feature, top-32 architecture and 1,000-update budget. Several held-out reconstructions are worse than the held-out mean, particularly layer 22's second pass. The saved sparse-feature differences therefore do not support semantic mechanism claims. This fit also differs from Section 4.7 in precision and retained position coverage, preventing attribution of the quality change to one factor.

At coefficient +0.05, pooled final-normalization, second-pass hidden-state effect norms are:

| Perturbed matrix | First only | Second only | Both passes | Interaction |
|---|---:|---:|---:|---:|
| Physical layer 12 | 75.7012 | 15.7032 | 75.6625 | 22.2407 |
| Physical layer 22 | 31.7039 | 2.7813 | 31.7484 | 2.9706 |

Each row pools the same 80 held-out token positions. Negative doses and central differences are also retained in the raw archives. These are finite-dose hidden-state effects; intervention position changes downstream depth, and BF16 rounding can contribute to apparent interaction. They are not answer-success effects or evidence of adaptive routing. Baseline pooled pass cosine is 0.8530 at layer 12 and 0.2771 at layer 22; the second/first norm ratio is 0.9069 and 0.2096 respectively, then 5.4779 after normalization. The dependence on capture site again prevents a norm-only compression interpretation.

The new basis was also transferred to the previously verified Q4 down-projection inputs at both passes, yielding 36 coordinate and basis-output difference archives over 18 trajectories and both physical matrices. Exact full-token identity, split alignment and source hashes were checked. Float64 recomputation gives maximum relative coordinate error 3.86e-7. This is a projection of saved Q4 activations into the frozen BF16 basis, not direct BF16 input capture or a Q4 parameter fit; the residual contribution is omitted from these diagnostic outputs.

Run: `prime-20260906T204058Z-561`; manifest SHA256 `7c5205eeb3c1e7c025ac1aababc6e0526bbe82048f633a52e73bfefa07c8a546`; factors SHA256 `a1acd3ffa1fe0c4f4d2566b5e95670112bb152b31a149c05777b46ee6b4a502c`. The archive passed retrieval checks and the paid pod terminated successfully. `q4_reasoning_vpd64_transfer_v1` contains the coordinate transfer. The associated PNG/PDF/SVG figure records its source hashes and was visually reviewed.

### 4.10 Exploratory reconstruction diagnostics

Following the poor held-out SAE reconstruction, we fitted dense linear projection controls using only the original BF16 training activations. Each site supplies 64 rows per pass, giving 128 pooled rows and a centered training rank of 127. Shared-mean PCA controls retain ranks 0, 8, 32, 64 and 127; all ranks are reported in the diagnostic archive, without choosing a rank on held-out performance. Separate per-pass training means are also retained as controls. The table shows the saved SAE and two of the recorded projection ranks.

| Site and pass | Saved SAE EV | Training PCA rank 8 EV | Full training-span EV |
|---|---:|---:|---:|
| Layer 12, pass 1 | 0.0035 | 0.0008 | 0.1636 |
| Layer 12, pass 2 | -0.0087 | 0.0003 | 0.1705 |
| Layer 22, pass 1 | 0.6667 | 0.9360 | 0.9470 |
| Layer 22, pass 2 | -2.6589 | 0.2044 | 0.3593 |
| Loop norm, pass 1 | -0.0774 | -0.0976 | 0.1423 |
| Loop norm, pass 2 | 0.1381 | 0.2692 | 0.4092 |

At layer 22's second pass, the training span supports substantially better reconstruction than the saved SAE achieves. This shows that the observed SAE failure is not explained solely by the portion of held-out activations outside that span. Dense PCA does not match the SAE's sparsity, feature count or semantic aims, and it is not a replacement mechanistic interpretation. Full-span projection still leaves substantial unexplained variance at most sites. An independent QR construction from training rows reproduces all 12 train/evaluation pass-level full-span reconstruction errors; all 12 saved SAE reconstruction errors were also recomputed. The main diagnostic additionally checks 72 projection/residual energy identities and stores all bases and reconstruction arrays.

First-pass training variance at layer 22 is 7.3216 times second-pass variance. Equal sampled row counts therefore do not equalize relative reconstruction error in the raw-MSE objective. A completed exploratory CPU control compares raw MSE with inverse-training-variance weighted MSE across three fixed seeds (42, 43, 44). Each fit retains 6,144 features, top-32 sparsity, 1,000 updates, batch size 128, the same pooled training-mean initialization, and AdamW at learning rate 0.001. Each seed uses identical initial parameters and sampled batches across the two arms; the two pass weights are 0.2403384 and 1.7596616, normalized to mean one. All six conditions are reported on the same training and held-out inputs:

| Seed / objective | Training EV, pass 1 | Training EV, pass 2 | Held-out EV, pass 1 | Held-out EV, pass 2 |
|---|---:|---:|---:|---:|
| 42 / raw MSE | 0.9827 | 0.9505 | 0.6959 | -2.6407 |
| 42 / weighted MSE | 0.9729 | 0.9861 | 0.7068 | -2.9388 |
| 43 / raw MSE | 0.9957 | 0.9574 | 0.7192 | -2.5289 |
| 43 / weighted MSE | 0.9886 | 0.9851 | 0.7586 | -3.0489 |
| 44 / raw MSE | 0.9950 | 0.9639 | 0.7653 | -2.7166 |
| 44 / weighted MSE | 0.9860 | 0.9851 | 0.7730 | -3.0866 |

Weighting improves training second-pass reconstruction in all three pairs but worsens held-out second-pass reconstruction in every pair. The paired held-out EV changes are -0.2981, -0.5199 and -0.3699 for seeds 42, 43 and 44, respectively. All six second-pass held-out EVs remain negative. Thus this particular variance-weighted objective does not repair the observed generalization failure; the control does not exclude other objective choices, optimization budgets or sources of mismatch. The three seeds do not supply a new independent evaluation cohort, and no feature semantics or answer-level benefit is established.

A separate NumPy float64 implementation recomputes encoder activations, top-k selection, decoder reconstructions and saved pass differences for all six dictionaries. It verifies all 108 feature archives and 24 pooled train/evaluation reconstruction measurements, together with source hashes, split identities, regenerated batch indices, paired initialization receipts, and all 1,000 recorded loss reductions per condition. These checks validate saved arithmetic and provenance without rerunning training. The complete receipt is `sae_loss_balance_verification/complete.json`.

We also align BF16 and Q4 captures at exactly the same final eight token positions on all 18 trajectories. On the ten held-out inputs, relative activation difference norms (Q4 minus BF16, divided by the BF16 norm) are 0.1363/0.1676 at layer 12, 0.0851/0.2623 at layer 22, and 0.1645/0.2287 at loop normalization for passes 1/2. These are pooled activation differences, not answer divergence rates. Quantization, numerical precision and implementation/runtime differences are not isolated by this comparison. The 108 per-input/site/pass records and split-level summaries are retained.

Artifacts: `sae_generalization_diagnostic_v2`, including `qr_verification.json`; completed paired control: `sae_loss_balance_control_v1`. These analyses were motivated by observed failures and are explicitly post-hoc. They do not establish new held-out generalization claims after model selection, feature semantics, or an answer-level recurrence benefit.

### 4.11 Signed response and output-proxy resolution

For each saved hidden-state pair at coefficients ±0.05, define the odd response as `(h_plus - h_minus) / 2` and the even response as `(h_plus + h_minus - 2*h_baseline) / 2`. These exactly reconstruct both signed changes in 720 arithmetic checks across 360 example/matrix/site/pass/scope records. A linear sign-reversing response would have zero even component. At final normalization after the second pass, pooled even/odd norm ratios are:

| Perturbed matrix | First-pass-only intervention | Second-pass-only intervention | Both-pass intervention |
|---|---:|---:|---:|
| Physical layer 12 | 1.5083 | 1.6505 | 1.4906 |
| Physical layer 22 | 2.5569 | 0.7619 | 2.5268 |

The even component is comparable to or larger than the odd component in most of these conditions. Thus the saved coefficient is not demonstrated to lie in a linear-response regime. Nonlinearity and numerical execution effects are not isolated by this single-magnitude experiment. All sampled hidden-state values are exactly representable in BF16; at layer 22's second-only intervention, 47.58% of changed final-state coordinates move by no more than one larger adjacent BF16 step. This is a representational scale, not a measured noise floor. Smaller-dose, repeated-zero-dose and higher-precision model controls remain required before assigning an observed response to a stable component mechanism.

Every effect archive also contains full next-token vocabulary logits for baseline and all six signed interventions. Repeated baseline logits exactly match the separate baseline capture, and stored reference-token IDs/logit changes agree with those arrays. The original float32 KL proxies contain 34 negative values among 140 entries (including 20 baselines), with a minimum of -8.94e-8. Those negative values are numerical error. Float64 recomputation uses an equivalent generalized-KL expression to avoid cancellation near equality and agrees with direct normalized log-probability KL within 1e-12 absolute plus 1e-8 relative tolerance; original proxies remain unchanged.

Across the 120 non-baseline interventions, corrected KL ranges from 2.86e-16 to 0.02634, with median 1.54e-7; 68 values are below 1e-6 and 18 exceed 1e-3. None changes the highest-probability next-token ID. These distributions condition on one saved, budget-censored trajectory endpoint per input. They neither prove a lack of later behavioral effects nor demonstrate a final-answer improvement. Artifacts and full per-record values are in `signed_effect_resolution_v2`.

### 4.12 Completed longer-budget local methods demonstration

The separate `q4_reasoning_pilot_8192_v1` run completed all 36 outputs on the original 18 inputs, followed by grading, native-trajectory replay, shared-SAE fitting and independent verification. Its two-pass and one-pass conditions use the same pinned Q4 weights, greedy decoding, q8_0 KV caches, 10,240-token context and 8,192-new-token cap. The resumed attempt retained all 17 previously saved outputs and passed the 16-token greedy resume control. Increasing the HTTP wait changed transport tolerance, not the continuation budget or sampling parameters. A complete comparison also confirms that all 36 longer outputs preserve their corresponding initial 2,048-token continuation prefixes.

| Completed condition | Outputs | Generated tokens | Token-capped outputs | EOS / thinking closed |
|---|---:|---:|---:|---:|
| One pass | 18 | 147,456 | 18 | 0 |
| Native two passes | 18 | 132,869 | 13 | 5 |

The five uncensored native outputs comprise three calibration inputs and two evaluation inputs. Both conditions score 0/8 evaluation units: two complete ARC-AGI-1 tasks, two complete ARC-AGI-2 tasks, one local logic question, one separate historical Campsite question and two math questions. There are no missing conditions or ungradable pairs. Closing the thinking delimiter and reaching EOS is an observed stopping behavior, not evidence that an answer is correct. These results describe this small Q4 pilot under its fixed budgets; they do not estimate a model-size threshold or establish equal underlying reasoning ability.

The 18 native trajectories contain 145,992 prompt-plus-generation tokens. Replay retains 32 positions per input, covering the prompt suffix and early, middle and late generation: 576 positions in total. It captures 900 arrays across both invocations of all 22 physical layers, loop normalization and selected MLP inputs. Capture-on versus capture-off full-vocabulary logits agree exactly in the 64-token control. Conditional greedy replay agrees at 429/432 sampled positions, so exact equivalence to every autoregressive decoding step is not claimed.

At layers 12 and 22, the 36 VPD coordinate archives project these new Q4 activations into the original frozen eight-component BF16 bases. This transfer step exercises pass-indexed coordinate measurement; it is distinct from the separately completed 64-component BF16 fitting and interventions in Section 4.9. No new parameter decomposition or causal answer intervention was fitted or executed by this local follow-up.

Three shared SAEs were newly fitted on these longer trajectories, using eight calibration inputs and ten held-out input records, 256 training positions per pass, 6,144 features, top-k 32, seed 42 and 1,000 updates. Each dictionary is shared between the two passes, so feature IDs are directly comparable within that fit. Independently reloaded held-out explained variance is:

| Shared SAE site | Pass 1 held-out EV | Pass 2 held-out EV |
|---|---:|---:|
| Physical layer 12 | 0.2965 | 0.2629 |
| Physical layer 22 | 0.8311 | -0.1880 |
| Loop-boundary normalization | 0.2923 | 0.4280 |

All 54 saved feature-difference archives were re-encoded and all six held-out EV values reproduced after dictionary reload. The negative layer-22 second-pass EV means reconstruction is worse than that held-out pass's mean baseline. Its feature deltas are reproducible numerical measurements with inadequate reconstruction for strong semantic interpretation. SAE coordinates from separately fitted dictionaries across token budgets are not automatically aligned; this study does not equate feature IDs between the initial and longer runs.

The worked example therefore completes the local capture-to-verification workflow even though behavior remains heavily censored and feature reconstruction is uneven. The verification receipt is `q4_reasoning_pilot_8192_v1/mechanistic_verification.json`; the source-linked numerical summary is `manuscript_summary.json` in the same run. Initial-budget artifacts remain separate. The generation payload SHA256 is `5a15e6165664882e7991889d56b8ba82524a9eacb29fcfe5e725215365ab7684`.

### 4.13 Muon-hybrid control inspired by the constitutional-adapter SAE study

Dugan's earlier constitutional-adapter study reports improved SAE quality with Windows CUDA Muon-hybrid and recommends comparing activations in a fixed dictionary. It motivates an optimizer control here, particularly given the poor layer-22 second-pass reconstruction. Its PDF does not specify the original optimizer code or hyperparameters; this run is a documented adaptation using PyTorch Muon, not a replication of its numerical benchmark or causal steering results. The downloaded paper and local copy match SHA256 `036cc6b51fcb5d7307cf2772f4f967aea1adf0d6cd30ac4f2d8d03f9ee40da4b`. [Prior SAE paper](https://www.moralitylab.xyz/papers/Late_Layer_Feature_Reweighting_by_a_Constitutional_Adapter_An_Industrial_Sparse_Autoencoder_Analysis.pdf).

We train six SAEs: AdamW and Muon-hybrid at physical layers 12 and 22 and loop normalization, using the verified longer-run Q4 activations from Section 4.12. Each pair shares the exact initialization and stored batch indices, seed 42, eight training and ten held-out input records, 256 training positions per pass, 6,144 features, top-k 32, 1,000 updates, batch size 128 and raw pooled MSE. Decoder columns are normalized after each update. Held-out data are not used for fitting or learning-rate selection.

Muon updates the encoder and decoder matrices; AdamW updates the encoder bias and pre-bias. The nominal learning rate is 0.001 with zero weight decay. Muon uses momentum 0.95, Nesterov updates, five Newton-Schulz iterations with coefficients (3.4445, -4.775, 2.0315), epsilon 1e-7 and the `match_rms_adamw` learning-rate adjustment. Both arms run on Windows CUDA with PyTorch 2.10.0+cu126, float32 parameters and TF32 disabled; Muon's native orthogonalization calculation uses bfloat16. A fresh AdamW arm on that same runtime supplies the paired baseline; the earlier CPU results are not used as its substitute. The nominal LR match is not a claim of identical update dynamics or optimizer tuning. [PyTorch Muon](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html).

| Site and pass | AdamW held-out EV | Muon-hybrid held-out EV | Muon minus AdamW |
|---|---:|---:|---:|
| Physical layer 12, pass 1 | 0.2953 | 0.2867 | -0.0086 |
| Physical layer 12, pass 2 | 0.2641 | 0.2455 | -0.0186 |
| Physical layer 22, pass 1 | 0.8300 | 0.9179 | +0.0879 |
| Physical layer 22, pass 2 | -0.1800 | 0.4533 | +0.6333 |
| Loop normalization, pass 1 | 0.2945 | 0.3022 | +0.0077 |
| Loop normalization, pass 2 | 0.4283 | 0.4446 | +0.0163 |

Muon-hybrid improves held-out EV in 4/6 site/pass comparisons in this single-seed pilot. At physical layer 22, the paired changes are +0.0879 on pass 1 and +0.6333 on pass 2. These are separate pass measurements, not six independent replications. The optimizer recipe was fixed before the full run; the two-update CUDA preflight is excluded from this table.

![Paired held-out reconstruction changes under Muon-hybrid; each point is one site and pass, with no uncertainty interval estimated from this single seed.](figures/sae_muon_comparison.png)

For feature usage, the following counts refer to features inactive across the union of both passes on the observed training split, out of 6,144 features:

| Shared SAE site | AdamW inactive | Muon-hybrid inactive | Muon minus AdamW |
|---|---:|---:|---:|
| Physical layer 12 | 3,931 | 1,579 | -2,352 |
| Physical layer 22 | 4,705 | 4,634 | -71 |
| Loop normalization | 4,892 | 3,620 | -1,272 |

An inactive feature on this finite split is not necessarily permanently dead. More active features do not by themselves establish semantic clarity, transportability or better causal interventions. Each dictionary supports pass-to-pass feature differences within that fit; independently trained AdamW and Muon dictionaries do not supply aligned feature IDs.

All six dictionaries and 108 feature-difference archives passed independent NumPy float64 checks of re-encoding, reconstruction, pass differences, source hashes, split identity, paired initialization/batch receipts and decoder norms. All 24 pooled train/evaluation reconstruction measurements and feature-usage counts were reproduced. Three implementation tests cover baseline-recipe agreement, complete matrix/bias optimizer assignment and rejection of corrupted feature differences. Training itself was not independently rerun.

This is a post-hoc, single-seed control on an already inspected cohort. It does not establish a general optimizer ranking, feature semantics, model reasoning improvement or a capability threshold. Local training times and peak CUDA allocation are retained in the condition manifests; uncontrolled laptop clocks and thermal state prevent a hardware-efficiency claim. No language model was reloaded, no VPD was refitted and no answer intervention was performed. Artifacts are in `muon_sae_control_v1`, with the independent receipt in `muon_sae_verification/complete.json` and the reproducible protocol in `muon_sources/PROTOCOL.md`.

### 4.14 Frozen Muon dictionaries on saved VPD interventions

To connect the optimizer control to the parameter interventions, we freeze all six dictionaries from Section 4.13 and apply them to the verified BF16 baseline and intervention activations from Section 4.9. Those activations retain the final eight positions of the initial-budget trajectories, while the dictionaries were fitted on Q4 activations across prompt and generation regions of the longer run. Precision, execution runtime and position distribution therefore differ. We verify that all 18 BF16 source token sequences are exact prefixes of the corresponding longer Q4 sequences and that the original eight training and ten evaluation input assignments match. No dictionary is refitted, no new model invocation is run and no held-out input is used for fitting.

Before interpreting feature differences, we measure frozen-dictionary reconstruction on these BF16 baselines. Held-out explained variance pools 80 positions per pass across the ten evaluation input records:

| Site and pass | Frozen AdamW SAE EV | Frozen Muon-hybrid SAE EV |
|---|---:|---:|
| Physical layer 12, pass 1 | 0.0578 | 0.0366 |
| Physical layer 12, pass 2 | 0.0477 | 0.0315 |
| Physical layer 22, pass 1 | 0.7627 | 0.9419 |
| Physical layer 22, pass 2 | -0.5350 | 0.3120 |
| Loop normalization, pass 1 | 0.0738 | 0.0902 |
| Loop normalization, pass 2 | 0.2245 | 0.2923 |

At layer 22's second pass, the optimizer advantage persists in this particular transfer: EV is 0.3120 with Muon-hybrid versus -0.5350 with AdamW. It is lower than the Muon dictionary's 0.4533 on its original longer-run Q4 evaluation positions, and most variance remains unreconstructed. The weaker layer-12 and normalization reconstructions similarly constrain interpretation. This is a transfer-quality measurement; it does not isolate precision from position/context effects or validate feature semantics.

We then re-encode all 20 previously executed VPD intervention records: component 0 at each of the layer-12 and layer-22 MLP down projections, on ten held-out inputs. Each record contains its baseline and perturbations at coefficients +0.05 and -0.05 applied in the first pass, second pass or both. For every site and optimizer dictionary, archives retain both pass encodings under all seven conditions, their pass-to-pass differences, and each perturbed encoding minus its baseline encoding. These latter differences are obtained by re-encoding each full hidden state, not by encoding the hidden-state difference. Each comparison uses the same frozen dictionary; feature IDs are not aligned across the AdamW and Muon dictionaries.

The resulting 120 archives passed 1,680 independent NumPy float64 re-encoding checks, 24 pooled baseline reconstruction checks and 2,520 checks of saved feature arithmetic and causal ordering. In particular, a second-pass-only intervention leaves every captured first-pass feature vector unchanged. Saved token IDs/positions, source intervention hashes and dictionary provenance were checked. A corruption control verifies rejection of altered intervention-feature differences. These checks connect shared parameter components to observed changes in a fixed feature basis, while leaving the original model intervention results unchanged.

The analysis is post-hoc and reuses the existing signed VPD responses. It adds no dose sweep, independent training seed, feature-semantic validation or answer-level causal result. The original component coverage, finite-dose nonlinearity and BF16 numerical-resolution limits still apply. Artifacts and verified source links are in `muon_vpd_transfer_v1/{contract,quality,records,manifest}.json`, with the full feature arrays alongside them.

### 4.15 Local Muon-feature steering implementation control

We additionally test pass-specific residual steering in the pretrained Q4 runtime, using the frozen physical-layer-22 Muon dictionary. The direction is the unit decoder vector with the largest mean absolute pass-to-pass feature difference across the eight training inputs only. This selects a numerically changing feature without attributing a semantic meaning. The test uses only the first 96 prompt tokens of one calibration input, perturbing position 95 at layer 22 before loop normalization. It does not use a held-out task or a generated reasoning trace.

The pinned ggml activation-write API applies signed additive doses of +0.1 and -0.1 on the first pass, second pass or both. A zero-dose control writes identical activation values back to both passes. Full-vocabulary logits match exactly between capture-disabled and capture-enabled baselines, zero writes preserve captured activations and logits exactly, and clearing steering restores the baseline exactly. All earlier retained positions remain unchanged; second-pass-only steering preserves all captured first-pass activations. Backend readback confirms each requested write, and the post-intervention capture agrees with it.

Across the six nonzero conditions, maximum absolute next-token logit changes range from 0.0143 to 0.8051; none changes the greedy next token. These values demonstrate that the runtime applies the intervention and propagates it to the output. They are neither an answer-accuracy result nor evidence that this selected feature controls reasoning. The truncated calibration prompt, one feature and one dose magnitude do not support comparisons of causal importance between passes. Cached generation parity, semantic controls and task-level steering remain untested by this preflight.

The seven condition archives and their source/code hashes are retained in `q4_steering_preflight_v1`. A separate reload verifies saved intervention arithmetic, causal ordering, token identity and full-vocabulary logit differences; runtime checks and cleanup are recorded in its manifest and cleanup receipt. This residual-direction intervention complements the parameter-component interventions in Section 4.14; its coefficients use a different basis and are not directly comparable.

A subsequent calibration-only cache test uses the full 383-token training prompt and eight greedy decoding steps under a baseline, identity writes and six signed pass-specific conditions. Interventions apply at every processed position. For each step, cached logits are compared with a fresh evaluation of the same complete token prefix. With 64-token prefill batches and single-token cached decoding, all eight conditions fail the fixed numerical gate (absolute and relative tolerances both 1e-4), despite agreement on all 64 greedy decisions. Maximum absolute logit discrepancy is 3.6051 for the baseline and reaches 3.7099 across conditions. Identity writes still reproduce the baseline exactly. This failure prevents using that execution path for the planned answer-intervention experiment; greedy agreement alone is insufficient for its gate.

Using single-token processing throughout prefill, cached decoding and fresh-prefix replay gives bitwise-identical logits in all eight conditions and agreement on all 64 greedy decisions. Maximum absolute discrepancy is zero; identity writes reproduce the baseline exactly. The original absolute and relative tolerances of 1e-4 are unchanged. Three controls were carried from earlier serial runs and five completed through guarded, checksummed checkpoint continuation. Re-executing the three carried cached trajectories under the added guard also reproduces their logits bitwise. Independent reloads verify all eight archives, mutation counts, checkpoint hashes and source lineage. This passes the serial gate for this calibration prompt and eight decoding steps; it does not explain the batched discrepancy, validate long-context generation or show answer-level benefit. The failed batched run remains in `q4_cached_steering_v1`; completed serial evidence and verification are in `q4_cached_steering_burst_v1`.

### 4.16 Completed answer steering and decoding audit

The extension completed 70 fresh Q4 continuations on ten held-out input records, using the selected layer-22 Muon decoder direction (feature 574). Seven conditions comprise a baseline and coefficients +0.1 and -0.1 on the first pass, second pass or both. All prefill and decode positions are eligible; the context allocation is 10,153 tokens and the output cap is 8,192. The A100 run independently repeats all eight serial cache controls at this allocation, obtaining zero full-vocabulary logit discrepancies over 64 comparisons. This short-prefix gate does not establish long-trajectory parity. All generated outputs were retrieved and checksummed before pod termination.

The continuations contain 561,231 generated tokens. Eight stop on an end token and 62 reach the output cap. After independent token decoding and grading, every condition scores zero on eight complete evaluation units: two ARC-AGI-1 tasks, two ARC-AGI-2 tasks, one local logic question, one historical Campsite question and two math questions. There are no missing records or remaining grader errors.

| Condition | Complete units correct | Individual inputs correct | Capped inputs | Generated tokens |
|---|---:|---:|---:|---:|
| Baseline | 0/8 | 0/10 | 9/10 | 81,731 |
| Pass 2 +0.1 | 0/8 | 1/10 | 9/10 | 80,303 |
| Pass 2 -0.1 | 0/8 | 1/10 | 8/10 | 79,951 |
| Pass 1 +0.1 | 0/8 | 0/10 | 10/10 | 81,920 |
| Pass 1 -0.1 | 0/8 | 0/10 | 9/10 | 78,266 |
| Both +0.1 | 0/8 | 1/10 | 8/10 | 80,794 |
| Both -0.1 | 0/8 | 0/10 | 9/10 | 78,266 |

The three individual-input successes concern the same second test input of ARC-AGI-1 task `c074846d`: pass-2 +0.1, pass-2 -0.1 and both-pass +0.1 finish correctly after 6,575, 7,453 and 8,078 generated tokens, respectively. The baseline is incorrect on this input. Each successful condition still fails the task's other input, so none produces a complete-task improvement. These are three condition outcomes on one input, not three independent solved tasks. Success with both signs of the pass-2 perturbation also does not support a monotonic feature interpretation.

![Complete-task outcomes under the seven feature-steering conditions. An asterisk denotes at least one token-capped input within that unit. Individual-input successes are reported separately in the table.](figures/feature_answer_outcomes.png)

Independent decoding of long checkpoints exposed a presentation difference missed by the eight-step gate. This GGUF enables a leading-space rule, and the pinned llama.cpp detokenizer performs three in-place punctuation/apostrophe cleanup passes. For example, raw token text `6 . 6` becomes `6. 6`, and `column0 != 0` becomes `column0!= 0`. A vocabulary-only control reproduces native rendering and verifies that independent HF text equals concatenated native token pieces. An exact reproduction of the pinned cleanup matches 18 checkpoint snapshots and ten synthetic boundary cases; the completed cohort subsequently passes the same rendering check on all 70 records. The original token IDs, native text, inference code and short-prefix verification remain unchanged. [Pinned detokenizer source](https://github.com/ggml-org/llama.cpp/blob/0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0/src/llama-vocab.cpp).

Primary scores use raw HF decoding, with native rendering graded separately: all 70 grading records agree between renderings. The first full grading attempt encountered a Windows command-line-length failure on one long math output. Passing the unchanged text through standard input into the unchanged published verifier resolves that transport error; regrading all 70 outputs leaves the other 69 grades unchanged. The affected response scores zero. Short-input parity controls and long correct, incorrect and unfinished-thinking controls verify the transport change. Original and revised grading receipts remain in `grading` and `grading_v2`.

This experiment demonstrates an auditable path from a frozen sparse feature to pass-specific intervention and scored output. It finds a narrow individual-input change but no complete-task gain at this feature, dose and cap. Heavy censoring, one selected feature, two signed doses and a previously inspected small cohort preclude a general reasoning-benefit claim. It does not establish feature semantics, an adaptive router, model-scale effects or reasoning compression. This two-pass zero-intervention baseline is distinct from the earlier one-pass ablation. All source receipts, raw outputs, decoding audits and completed tables are retained in `prime-20260908T200850Z-601`.

### 4.17 Additional historical math and ARC-AGI receipts

A further audit of the same pinned Hermes-Skills revision recovers a four-shard math ledger with 40 unique row IDs and seven arms, totaling 280 recorded calls. Independent string-exact recomputation reproduces every reported arm score, recorded error count, and paired fix/regression count:

| Historical arm | String-exact answers | Recorded errors |
|---|---:|---:|
| Raw baseline, no skill | 3/40 | 0 |
| Incumbent skill | 7/40 | 0 |
| Qwen27B auditor patch | 4/40 | 0 |
| Domain router | 3/40 | 0 |
| Answer-shape verifier | 2/40 | 0 |
| Slow-path trigger | 4/40 | 1 |
| Patch-commit controller | 3/40 | 1 |

The outcome-aware oracle across all seven arms reaches 8/40; it is not an executable selection policy. These saved rows lack original prompts, preventing a verified join to the Nanbeige cohort by answer value or row order alone. The counts are historical string-exact scores, not a new 27B run, symbolic-equivalence regrading or evidence of a complete math clear. This slice is separate from the 200-row historical math audit in Section 4.2. Source bytes, shard identities and recomputation are retained in `hermes_extended_audit_v1`. [Committed math receipts](https://github.com/MoralityLabAI/Hermes-Skills/tree/07a177f2751c2adb1cc16dc819c1173c0cd57ad5/research/studies/2026-04-22-metta-trm-hermes-pipeline/artifacts).

Two inspected 27B traces labeled `arc_agi` and `arc_agi_tool` report adapter-reset failures before receipt of a valid puzzle. Their logged zero scores therefore describe infrastructure failures and are excluded from reasoning-accuracy denominators. Each records a 64-token response. These two traces establish neither an ARC-AGI-1/2 clear nor a valid-task 27B failure rate; the conclusion applies only to the inspected episodes. [Committed ARC-AGI traces](https://github.com/MoralityLabAI/Hermes-Skills/tree/07a177f2751c2adb1cc16dc819c1173c0cd57ad5/data/primehub_overnight_all/qwen35_27b).

### 4.18 Fresh model-assisted Campsite comparison

A fresh Qwen3.5-27B Q4_K_M run evaluates the same 109 public Campsite puzzles across a vanilla prompt, the selected `intellect3-logic-hermes` prompt, and a routed TRM workspace. The skill and its latest prompt imprint are frozen at Hermes-Skills revision `07a177f2751c2adb1cc16dc819c1173c0cd57ad5`. The packaged gate routes 41 constraint signatures to the augmented path and leaves 68 on the plain skill. There are 259 unique model calls: 109 vanilla, 109 plain skill and 41 augmented calls. Unrouted candidates are reused exactly across routed arms.

All calls use the publisher template with thinking disabled, temperature zero, seed 42, a 1,024-token output cap, 8,192-token context and Q8 KV cache. The longest prompt is 792 tokens. The pinned llama.cpp runtime verifies full CUDA offload and GGUF/Hugging Face tokenizer agreement for all 259 inputs. Native detokenization and independently decoded token IDs are both retained. The run generates 123,184 tokens, with 109 capped calls.

The auxiliary implementation is the previously audited, hash-pinned retrieval/nearest-neighbor critic/constraint-search path. It uses the existing 128 synthetic training puzzles instead of the exposed historical bank; none matches an evaluation puzzle under the checked rotations or reflections. The positive-only support labels make this a test of the integrated workspace, not a test of critic discrimination. The external corrector retains an eight-pass local-search limit, exact search enabled and a 200,000-node cap. These are controller search settings, not measured neural transformer invocations. No evaluation target enters a retrieval query or correction call.

| Arm | Official correct / 109 | Canonical exact | Literal rule valid | Capped candidates |
|---|---:|---:|---:|---:|
| Vanilla | 0 | 0 | 0 | 109 |
| Plain skill | 13 | 11 | 4 | 0 |
| Routed model candidate | 11 | 9 | 2 | 0 |
| Routed final with correction | 35 | 31 | 6 | 0 |
| Plain candidate with routed correction | 34 | 32 | 6 | 0 |

Within the routed pipeline, correction fixes 24 previously incorrect candidates and regresses 0 previously correct candidates under the official verifier. The plain-candidate correction control separates the effect of commit-stage search from the additional retrieval workspace. All corrections were produced before loading grading sidecars. The model candidates remain immutable.

Official verifier success and the stricter literal adjacency rule are reported separately because of the documented dataset-contract discrepancy in Section 4.3. Candidate grades disagree between raw-token and native text views in 0 unique model calls. Arm-level token totals share calls and must not be added as independent compute.

This is a fresh integration, not a historical reproduction. The signature policy was selected using historical outcomes on this public cohort, so its result is not a clean generalization estimate. Synthetic support changes the helper distribution. The matched comparisons here are within Qwen on identical puzzles and decoding budgets. Nanbeige used different thinking and output budgets; this experiment does not isolate model scale, recurrence, or an architectural advantage. Disabling the template thinking channel does not prevent visible explanations. Vanilla explanations can exhaust the fixed cap, while the skill explicitly requests a grid only. Consequently, differences can reflect formatting and budget use as well as problem solving; a separate generic grid-only instruction control would be needed to isolate the specialized skill content. It also does not evaluate ARC-AGI or the other INTELLECT logic families. The completed source, routing, token, helper and grading receipts are retained in `logic_skill_comparison_v1` and `logic-prime-20260909T042949Z-897/grading_v2`.

## 5. Prospective tests and limits

A capability gain could emerge at a different model size, training regime or recurrent-compute budget. This remains a hypothesis: one small checkpoint, with a one-pass inference ablation and substantial output censoring, cannot locate such a threshold or rule it out. A future test would vary trained model size and recurrence jointly, control data and training/inference compute, and retain continuous outcome measures alongside discrete benchmark scores. Demonstrating the present instrumentation does not require a large comparator. Its transferable elements are the distinction between physical weights and invocations, matched-token capture, pass-specific parameter interventions, common-dictionary feature differences and independent verification. An adaptive model would additionally require recording actual route and invocation decisions; fixed two-pass Nanbeige does not exercise that extension.

A future reasoning-performance study could evaluate the full public ARC evaluation tasks and a fixed, family-stratified INTELLECT holdout; preserve all prompt-length exclusions and censoring; and compare paired outcomes under both equal-token and equal-compute budgets. Beyond the completed Campsite control, a broader comparison could run the same ARC and INTELLECT questions through applicable pinned helpers with an auditable retrieval corpus and full tool-call costs. These extensions are outside the current methods demonstration. A solver-assisted arm must be labeled distinctly from a language-model-only arm.

For a fresh 27B comparator, we have pinned official Qwen3.5-27B metadata and a community Q4_K_M file by revision and SHA256. The weights are now evaluated on the 109-puzzle Campsite comparison in Section 4.18; the full prepared benchmark remains unrun. Its 64-layer hybrid architecture combines Gated DeltaNet and gated attention, so Nanbeige's layer-token execution proxy is not a portable compute measure across these models. Equivalence to the historical quantized checkpoint is unverified. See `comparator_sources/qwen35_27b/model_lock.json`. [Official model description](https://huggingface.co/Qwen/Qwen3.5-27B/tree/fc05daec18b0a78c049392ed2e771dde82bdf654), [pinned quantized checkpoint](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/tree/3221f178a6b842d04f1fb42f1c413534adcc0a6a).

The fresh comparator now has verified model-facing inputs for both the 18-record pilot and all 1,223 fixed-cohort records. The full Qwen input set contains 2,503,714 prompt tokens, with a maximum of 17,609. Every original question field is retained; the publisher's native template trims only boundary whitespace on 31 model-facing prompts, and every rendered prompt round-trips through the pinned tokenizer. The inference bundle excludes grading sidecars. The prepared runner requires actual GGUF tokenizer agreement for every input, full GPU offload, exact weight/runtime hashes, and a saved-prefix check on resume. The separate 109-puzzle answer-only experiment in Section 4.18 exercises weight/runtime identity, full GPU offload and per-input tokenizer checks. The original 18-record and 1,223-record thinking-mode runs and resume-prefix checks remain unexecuted. Independent input-identity checks are in `comparator_sources/qwen35_27b/input_identity_verification.json`.

Mechanism claims require stable component coverage, multiple SAE and decomposition seeds, a dose sweep, higher-precision controls, and causal interventions tied to final-answer success. Activation or feature norms alone do not prove reasoning compression. Larger or smaller trained architectures are required to estimate non-linear scale effects; changing inference loop count on one checkpoint is insufficient. Neither fixed Nanbeige recurrence nor an external Hermes router is evidence of an internal Astra router.

For prospective retrieval controls, the archived banks have now been filtered against the union of the fixed pilot and benchmark public inputs, preserving the evaluation cohorts. This removes all 100 logic-family support rows, leaving zero retriever or critic examples. Six math support rows remain, with no exact public-input match to these fresh cohorts; their overlap with the separate historical math evaluation remains documented. `hermes_support_fresh_v1/manifest.json` records source and output hashes and exclusion IDs. A separate synthetic logic bank and controller-only experiment now address this preparation gap as described in Section 4.8; they do not reproduce historical training. Section 4.18 now adds a fresh model-assisted Campsite comparison using the synthetic bank.

The prepared answer-intervention stage starts fresh generations from each original held-out public prompt, rather than continuing the saved Q4 reasoning. It compares a zero-intervention baseline with positive and negative component-0 updates at each of physical layers 12 and 22, applied to the first pass, second pass or both. At the initial coefficient magnitude 0.05 this gives 13 conditions and 130 generations on ten held-out inputs. Interventions cover all prefill and decode calls, so this stage does not isolate changes made during generated reasoning alone. Invocation counters reset at every model call, including cached decode steps; each condition uses a fresh KV cache. Seven controls on the publisher's small random recurrent architecture cover agreement with a physical shared-weight edit, full-prefix versus cached decoding, exact greedy-token agreement, causal ordering and restoration after an exception. These are implementation controls, not pretrained answer results. The pretrained stage requires completed VPD artifact and source-lineage verification, applies the unchanged cache-parity gate to all conditions, and grades re-decoded final answers locally with complete ARC-task aggregation. Dose and seed stability remain subsequent tests.

## Reproducibility

Source lock: `sources.lock.json`. Dataset: `dataset_v1/{questions,graders}.jsonl` and `manifest.json`. Subset: `pilot_v1`. Solver audit: `hermes_audit/{summary.json,rows.jsonl}` with committed source copies and hashes. Experiment implementation: `../../experiments/nanbeige_reasoning/`; component and SAE implementation: `../../experiments/nanbeige_looped/`. The dependency files and tests are part of the experiment. Completed longer local run: `q4_reasoning_pilot_8192_v1`, including generation and capture manifests, SAE dictionaries, per-input differences, grading records and independent verification. This draft has not been submitted or published.
