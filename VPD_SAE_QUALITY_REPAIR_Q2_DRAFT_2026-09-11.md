# Q2 v1: trajectory breadth versus within-trajectory density

Status: proposed alternative quality-repair pilot, not an addition automatically scheduled alongside F1. No training, capture, fresh pack, or compute authorization exists for Q2. This is a new design, not a reconstruction of a lost experiment.

Executor entry point: `VPD_Q2_CHEAP_EXECUTOR_START_HERE_2026-09-11.md` provides one bounded local preflight command, explicit exit-code semantics and the remaining research requirements. Its successful fixture/evidence replay still reports `research_readiness: not_ready`, `launch_authorized: false` and native exit code 3. It neither replaces a real-data launch review nor verifies existing pod shutdown.

Source feasibility update: the audited local question corpus provides at most 29 eligible family labels, fewer than F1-A's 32-role-family layout. Q2's ten-training-family row construction is tested, but its proposed dependence on a complete F1-A pack cannot be fulfilled from that corpus as written. Do not run Q2 by silently omitting roles or relabeling examples as families. Additional audited sources or an explicitly versioned quality-only role design remain necessary; neither has been supplied here.

Recovered-capture feasibility update: all 72 checked site/pass arrays in the historical BF16 reasoning calibration retain only eight tail rows per example. None of its 18 examples has all four Q2 windows; 24 positions per example are missing, including 192 training positions per site/pass. Its training inventory is eight examples/four families, and checkpoint revision/family-disjointness remain unverified/false in the source. This is not an importer-only gap. See `VPD_Q2_RECOVERED_CAPTURE_COMPATIBILITY_2026-09-11.md` for the real-array audit and exact source scope; no new capture was run.

A subsequent bounded E: search found seven nominal procedural task-type leads in a readable recovered clone; see `VPD_SAE_RECOVERY_SOURCE_LEADS_2026-09-11.md`. None has completed the required source, semantic-independence and freshness checks; they add no admitted families to Q2. The older directory's two checked zero-filled files are not evidence that every recovered copy is unusable.

One source-integration component is now implemented: a graph-coloring question-only adapter and independent JSON constraint checker pass 24 small-fixture tests. It preserves an at-most-color-budget task rather than claiming an optimal chromatic number, and separates public questions from reference assignments. A bounded CPU canary of two reviewed, hash-pinned source modules now also reproduces four medium-difficulty development puzzles and verifies valid/permuted answers and conflicting-answer rejection. The canary adds two source-integrity tests. This clears a development step only, not a fresh holdout or semantic-family admission. See the source-lead report for exact scope, receipt and commands.

## Why this comparison

Saved-code evidence now shows generated-row training EV of 0.844–0.994, while every archived fit fails pooled both-pass generated-row evaluation EV 0.20. ARC and math also show large descriptive train/evaluation differences within shared family labels. This favors investigating transfer and training coverage over assuming insufficient ability to fit the existing rows. It does not prove a memorization mechanism or rule out optimization, distribution shift, or other explanations.

Q2 asks: at a fixed retained-activation-row and optimization budget, does distributing training across more trajectories improve generated-row reconstruction and preserve useful pass groups? It deliberately trades within-trajectory density for trajectory breadth. It cannot identify a pure causal effect of breadth separately from density.

New saved-array coverage diagnostic: generated evaluation-example error correlates with nearest-training-trajectory distance (descriptive r 0.949–0.994), but median evaluation distance is only 0.972–1.101 times training leave-one-trajectory-out distance across the eight related site/pass/archive checks. Both comparisons use seven reference trajectories, with entire query trajectories excluded from training references. This keeps breadth repair a hypothesis, not an established explanation or cure; family/content/scale confounding and the small reused evaluation sample remain. See `VPD_SAE_TRAINING_COVERAGE_AUDIT_2026-09-11.md` and its reproducible receipt. Do not use this development diagnostic to select favorable final evaluation examples or change the proposed quality threshold.

Subsequent post-hoc controls leave family-centered generated correlations at 0.957–0.998, with minimum leave-one-example-out pooled r 0.925 and minimum leave-one-family-out pooled r 0.943. Thus the association is not solely a family-mean or single-example effect in these archives. Only ARC and math supply within-family variation; conditional permutation checks are exploratory and do not remove scale/content confounding. This strengthens the rationale for measuring coverage in a reviewed experiment but provides no basis to expand Q2's run count, bypass its fresh-pack requirement or assume a repair will succeed.

## Proposed data construction

Use the proposed F1-A training role only after an actual pack is frozen: ten training families, four fresh examples per family. Neither the pack nor these examples currently exists. Final evaluation families remain disjoint from training; ARC/Dyck/etc. archival failures are development evidence, not the new final holdout.

Capture each of the 40 training examples once at the same 32 positions for both recurrent passes: eight prompt-tail, eight early-generation, eight middle-generation, eight late-generation positions. Freeze absolute-position selection, generation/censoring rules, and region definitions before capture. All four windows must have eight distinct rows, with no overlapping positions. Insufficient rows invalidate the proposed layout; do not duplicate positions or replace difficult examples based on outcomes.

| Arm | Trajectories/family | Retained rows/trajectory/pass | Total rows/pass | Prompt/generated rows/pass |
|---|---:|---:|---:|---:|
| Q2-N: narrower, denser | 1 | 32 | 320 | 80 / 240 |
| Q2-B: broader, sparser | 4 | 8 | 320 | 80 / 240 |

For Q2-N, choose one of the four training examples per family by a frozen independent data-selection seed, before inspecting its activations or quality. The implemented draft seed is `20260911`; choose the minimum SHA-256 of the canonical `{seed, family, id}` object within each family (ID breaks a hash tie). This is a deterministic seeded ordering, not selection using activations. Keep that choice identical across all initialization seeds and sites. For Q2-B, retain the first and last position of each of the four declared eight-row windows. Both arms therefore have exactly 32 retained positions per family/pass and the same phase mixture. B includes the N-selected trajectory but fewer of its positions; document this nested overlap.

This is not the redundant equal-family sampler Q1: the probability mass across selected families stays balanced, but the actual retained row/trajectory roster changes. Do not synthesize additional reasoning traces, perturb evaluation labels, or route dictionaries by task family.

## Matched training controls

One shared SAE per arm/site/seed. Proposed inherited architecture: width 3072, 6144 features, top-k 32, raw MSE, AdamW learning rate 0.001 and zero weight decay, 1000 updates, 64 aligned rows per pass per update, decoder normalization after each update. These are review-draft choices from the current archived-fit code, not proven optimal settings or a verified BF16 training entry point.

Across N/B, start from exactly identical parameter tensors. In particular, use a common training-only pre-bias computed from the complete 40-example capture pool for both arms, and record its hash. This shared preprocessing intentionally gives both arms access to the same pooled first moment; the intervention concerns optimization-row exposure, not isolation from all unselected data. Do not recompute arm-specific biases and then attribute the result solely to exposure.

Implementation qualification: the historical `experiments/nanbeige_looped/sae.py::fit_shared_sae` recomputes its pre-bias from the rows passed to it. Do not invoke it independently on N/B retained arrays for Q2; that permits arm-specific initial biases. The new initialization-only helper described below uses the existing `SharedSAE` architecture but installs one complete-pool vector before independently cloning the common state. The historical helper and its artifacts are unchanged.

Uniformly sample each arm's 320 retained row IDs. Use identical schedule indices on both aligned passes. Couple random variates across arms; record their distinct row rosters and sampled IDs. Each arm gets 64,000 draws per pass, or 200 expected draws per retained row. These repeated draws are not 64,000 independent observations. No extra epochs, early stopping, best-checkpoint selection, loss reweighting, optimizer switch, or larger dictionary is allowed in this comparison.

Implemented schedule refinement: do not reuse identical **native roster indices** across arms. The original rosters order rows differently; 120/320 same-index slots mismatch phase and 240/320 mismatch region. Instead, map each arm bijectively to common slots sorted by family, region (`prompt/early/middle/late`), example ID and position. Uniform slot draws remain uniform over each arm's rows, while matching family/region/phase on every paired draw. Keep each arm's mapped native row indices identical on both passes. This does not require identical trajectories or absolute token positions between arms.

The new draft fixes optimization-schedule seed `20260911`, domain-separated from data selection by SHA-256 of `{domain: "Q2-paired-optimization-schedule-v1", seed: 20260911}`. One `random.Random` schedule is reused across arms, passes, sites and all three initialization seeds. This isolates initialization changes conditional on a fixed draw schedule; it is not replication across independent schedules. The runtime version and exported slots are recorded for exact verification. This refines an unspecified scheduling choice without changing the twelve-fit upper design count or authorizing any fit.

Use one pinned model revision/precision/capture stack across all conditions. BF16 is the proposed fresh target; these Q4 development results do not establish BF16 parity. Validate actual dimensions before applying the inherited width/feature settings.

## Run matrix and stop boundaries

Proposed complete matrix: two arms × sites layer 11/norm × initialization seeds 1707/250829/260829 = 12 fits. This is an upper design count, not an authorized run list or cost estimate. Do not combine these with F1's 12 fits without a separately reviewed combined budget and purpose.

First-stage candidate: N and B at layer 11, seed 1707 (two fits). Finish both to the same fixed budget and include failures. If B fails generated pooled or any predeclared generated-family quality prerequisite, stop Q2 expansion and report a failed quality pilot; no seed replacement or family deletion. A pass merely permits review of further runs—it does not automatically authorize them or establish replication. Any stopped matrix must show all unrun conditions, not count them as passes or completed trials.

## Endpoint and information firewall

Use exactly the same frozen untouched evaluation activation arrays for every arm and seed. Score full uncorrected reconstruction, with fitted bias, on generated rows as primary and prompt rows as a separate retained outcome. Report both passes, every planned family, pooled EV, mean L0, and equal-family summaries. Do not select a favorable common-position subset after seeing reconstruction.

Before interpreting pass-feature groups, B must meet generated pooled and every planned generated-family both-pass EV ≥ 0.20. Prompt outcomes must also be reported, with a reviewed non-degradation rule frozen before execution. Useful improvement over N, uncertainty, and a joint success rule remain unset: derive a minimum-useful margin and adequate family count using calibration-only evidence. Do not substitute nominal significance or training EV for those missing criteria.

The subsequent `VPD_Q2_DECISION_AND_JOINT_POWER_REVIEW_2026-09-11.md` establishes that the saved marginal functional-test simulation cannot certify all-required-condition success. It also distinguishes this fixed-benchmark point-quality gate from population confidence and proposes a family-balanced paired EV estimand for review, without freezing a margin or replacing this draft's unresolved hypothesis set. Shared families/fits/schedules make independent-event power products unverified; quality, missing data and prompt-retention gates must enter the eventual joint analysis.

If the quality prerequisite passes, freeze training-only pass-group assignments separately for N/B and assess reconstruction contribution against matched random groups on the same evaluation rows. This is secondary until its numerical useful-effect rule is fixed. Cross-seed group replication remains necessary for a stable-compartment claim. This experiment cannot establish a separate-dictionary capacity advantage because both arms use shared dictionaries.

## Remaining launch requirements

- Actual family definitions, fresh pack contents, role separation, IDs and hashes; apply the tested row planner to real metadata and verify it against arrays.
- Resolve the verified source-family shortfall: only 29 eligible labels are available in the currently audited corpus, versus the inherited 32-family F1-A layout. Semantic independence/exposure checks may reduce availability further.
- Pin capture configuration, censoring policy, runtime and common evaluation roster. The draft selection seed and window recipe are now implemented; they still require inclusion in a reviewed real-data contract.
- Verify the paired training entry point, initialization equality, schedule/roster receipts, and quality gates end to end without a paid run.
- Freeze minimum-useful gain, prompt non-degradation rule, uncertainty/power analysis and joint decision rule without final-holdout inspection.
- Obtain explicit authorization for any weight fitting or paid compute, with dollar cap, an independent termination deadline and a verified provider termination receipt. These provider safeguards are requirements, not installed controls.

## Implemented local row planner

`experiments/authz_mechinterp/plan_sae_trajectory_breadth_v1.py` implements the two row rosters without reading activations or importing a training framework. It requires exactly ten declared training families and exactly four examples per family; rejects duplicate IDs and family overlap across declared roles; and validates all training examples before selection, so an invalid example cannot be silently replaced by a favorable one.

The window recipe is pinned by the hash of `experiments/nanbeige_reasoning/positions.py`: final eight prompt positions and eight-position windows ending at 25%, 50%, and 100% of generated length. Rounded boundary behavior comes from that function. The planner rejects short/overlapping windows or metadata that differs from the recipe. It does not certify token provenance, capture settings, semantic family independence, censoring acceptability or freshness; those require real-data checks.

Input metadata consists of `expected_training_families`, `data_selection_seed`, and `records`. Every record needs `id`, `family` and `split`; training records additionally require `prompt_tokens`, `generated_tokens`, `positions`, and the four half-open `regions` bounds. Supported roles are `train`, `eval`, `discovery`, `calibration`, and `validation`. Non-training positions and optional quality fields do not affect selection. Real metadata can be supplied with the CLI's `--metadata` option when such a file exists; none has been prepared yet.

The synthetic receipt at `artifacts/vpd_sae_trajectory_breadth_plan_v1/report.json` proves these structural properties:

- N retains 320 rows/pass across ten trajectories; B retains 320 across forty.
- Both retain 80 prompt + 240 generated positions/pass and 32 positions/family/pass.
- The two rosters overlap at 80 positions/pass; the common preprocessing pool contains 1280 positions/pass.
- Input ordering and evaluation-position/quality fields do not change the selected training rosters.
- Boundary flags explicitly record no fresh-pack verification, training, activation loading, initialization verification or launch authorization.

Reproduce the local checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_sae_trajectory_breadth_plan.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.plan_sae_trajectory_breadth_v1 --verify artifacts/vpd_sae_trajectory_breadth_plan_v1/report.json
```

Sixteen focused tests and receipt verification pass. Report ID: `c5e71e54cad2f756757f7ac9286b0f7ce492bcdcf51bbc75936953be197e7887`; SHA-256: `9d3296e86d4ba42008362f1065d85750aa6601619ce7fc3e337882f4052b1114`. This receipt contains synthetic identifiers and row metadata only, not research examples or a fresh activation pack. The paired training entry point, schedule integration and provider termination controls remain unimplemented for Q2.

## Implemented paired schedule dry run

`experiments/authz_mechinterp/plan_sae_paired_schedule_v1.py` reproduces the source row plan from metadata, constructs the bijective arm maps, and exports 1,000 batches of 64 common slot indices. `paired_batch_indices` resolves each batch into each arm's original roster indices for pass 0 and pass 1. It does not load activation arrays or create an optimizer.

The completed synthetic dry run has exactly 64,000 draws per arm/pass: 15,988 prompt and 48,012 generated. Region totals are 16,136 early, 16,083 middle and 15,793 late. All family/region/phase labels match on every paired draw; all 320 retained rows are drawn, 162–240 times each. These realized counts fluctuate around uniform expectations but are identical between arms; no resampling was performed to obtain a favorable schedule.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_sae_paired_schedule.py tests/test_vpd_sae_trajectory_breadth_plan.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.plan_sae_paired_schedule_v1 --verify artifacts/vpd_sae_paired_schedule_v1/report.json
```

Eighteen schedule tests and sixteen row-plan tests pass. The report and exported `slot_indices.json` reproduce exactly. Tests include the naive-index mismatch regression, slot bijections, matching of all 64,000 draws, paired pass indices, input-order invariance, deterministic seeding and malformed-budget/index rejection.

- Report: `artifacts/vpd_sae_paired_schedule_v1/report.json`
- Report ID: `63eecd979e8438df1c2fce012b8aad7a9825abac865ba4e0ea22a2b9a95828fb`
- Report SHA-256: `1e06836e09bd27b2a365b861a163166195bd2d1dae2172ef44fb5b62dea73d8d`
- Exported slot-file SHA-256: `706e031742175ca3d30f9e7a1b74cb5349762a33828c891fcb84a72de0394ecd`

This receipt binds the **synthetic demo**, not a fresh pack. A real-data executor must first produce and verify a real metadata row plan, then export its schedule to a new directory. Before fitting, it must additionally prove that the actual captured arrays match those native roster indices on both passes and sites, that initial parameters/common pre-bias are equal, and that the trainer consumes exactly these batches for exactly 1,000 updates. None of those array/initialization/trainer checks is supplied by this metadata dry run. The source-family shortage, endpoint review and independent provider shutdown requirements remain unchanged.

## Activation-row binding: implemented and synthetic-tested

`experiments/authz_mechinterp/bind_sae_paired_activation_rows_v1.py` connects the retained-row plan and paired schedule to actual NumPy row indexing. Its reusable `bind_rows` function takes plan metadata, a separately frozen per-array manifest, in-memory arrays, and an explicit expected width. The manifest identifies each training example, family, site, pass, storage-order position list and array-content hash. It requires the exact training-only inventory, both declared sites/passes, all 32 unique frozen positions per example, finite float32/float64 arrays and the declared dimension. Evaluation arrays are not accepted into this training binding.

Lookup uses example ID and absolute position, not an assumption that physical storage order already matches the roster. Each pass's positions are checked independently. Changed array values, shuffled values without corresponding frozen metadata, and site/pass swaps are rejected against the frozen per-key hashes. Legitimate storage permutations with correctly corresponding position metadata preserve the bound rows. Hashes establish consistency with a manifest, not that the original capture producer attached the correct model hook or pass label; producer/capture-stack verification remains necessary.

The complete training pool is reconstructed and checked against its 1,280-row-per-pass roster hash. One common pre-bias vector per site is computed from both passes of that complete pool, using float64 accumulation and float32 storage. It is not computed separately from the N/B retained subsets. Bound views and the common vector are read-only; minibatches are copies. No model parameter is initialized or changed by this code.

`iter_paired_batches` independently checks that supplied maps equal the bound plan's factor-aligned maps, then resolves each exported draw to its arm/site/pass array row. The completed benchmark consumed all 1,000 batches, 64,000 draws per site/arm/pass. Every consumed vector matched an independent identity oracle encoding family, example, site, pass and absolute position. This oracle uses **five-dimensional synthetic sentinels**, not model activations, SAE features or a new fresh pack. It verifies row-selection mechanics, not training quality or a real capture.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_sae_paired_activation_binding.py tests/test_vpd_sae_paired_schedule.py tests/test_vpd_sae_trajectory_breadth_plan.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.bind_sae_paired_activation_rows_v1 --verify artifacts/vpd_sae_paired_activation_binding_v1/report.json
```

Sixteen binding tests, eighteen schedule tests and sixteen row-plan tests pass (50 total). The combined fifteen-file focused suite passes 190 tests in 1.83 seconds; this is not the entire project test suite. The binding benchmark receipt reproduces exactly, including schedule/plan replay and consumption trace hashes.

- Receipt: `artifacts/vpd_sae_paired_activation_binding_v1/report.json`
- Report ID: `5a5beb7fd739c413aeba58d299e5ab413ff36d6bf3e382f258618d0eb56a0ff4`
- File SHA-256: `39d920a11f223d78b4e54d5f28d59df0a4175955892ccf03ad96835e4e2463a7`

Still unverified: actual 3072-wide fresh captures and their producer, shared initialized SAE parameter tensors, assignment of the common vector to those parameters, loss/optimizer consumption and exactly 1,000 optimizer updates. No encoder, decoder, model or optimizer runs in this benchmark. No training/compute authority or compartment admission follows from it. A real capture importer must create this binding contract from independently verified capture evidence; the synthetic fixture cannot substitute for that importer or pack.

## Paired initialization: implemented and tiny-fixture verified

`experiments/authz_mechinterp/check_sae_paired_initialization_v1.py` adds an initialization-only helper using the unchanged `SharedSAE` class. It requires explicit dimensions, seed, and a finite float32 complete-pool bias. It constructs a CPU float32 template inside a restoring CPU-RNG context, installs the common vector once, freezes the template, and independently deep-copies it for N/B. It does not call the historical fitting helper, enable training, evaluate a forward pass or create an optimizer.

The checker compares every named parameter and shape exactly between arms, verifies disjoint parameter storage, checks the common pre-bias and zero encoder bias, and checks the initial normalized decoder orientation relative to the encoder. CPU RNG state is restored; CUDA RNGs are not seeded. Different initialization seeds change the encoder matrix. Returned instances are frozen in evaluation mode; a later authorized trainer must explicitly enable training and check this contract on its actual runtime.

The completed benchmark covers both synthetic site labels × initialization seeds 1707/250829/260829, using width 5, ten features and top-k 2. All six tiny pairs pass and the receipt reproduces exactly. This is a verification of the initialization method, **not** creation or verification of the planned width-3072/features-6144 research conditions. No checkpoint is exported and no research weights are modified. Unit tests deliberately mutate only disposable tiny fixtures to test rejection.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_sae_paired_initialization.py tests/test_vpd_sae_paired_activation_binding.py tests/test_vpd_sae_paired_schedule.py tests/test_vpd_sae_trajectory_breadth_plan.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.check_sae_paired_initialization_v1 --verify artifacts/vpd_sae_paired_initialization_v1/report.json
```

Fourteen new tests pass; the initialization/binding/schedule/row-plan subset passes 64 tests. Tests cover exact state equality, storage independence, input-bias isolation, CPU RNG restoration, deterministic/different seeds, explicit dtype/dimensions, and rejection of altered bias, decoder, nonfinite values or unexpectedly trainable parameters.

- Receipt: `artifacts/vpd_sae_paired_initialization_v1/report.json`
- Report ID: `60fc5386875c9cf2d877c75c4eb8a72ae683aafb4600f81859e4a272ab6b871a`
- File SHA-256: `7fb7a48a2f3efd26874c28b4c138e414662c07e02bb48b347eac2213d323765f`
- Verified runtime: PyTorch `2.14.0+cpu`; no GPU runtime or numerical parity claim.

The executable method now covers synthetic row selection, array binding, shared preprocessing and equal initialization. Research-size initialization, a real capture importer/fresh pack, loss/optimizer integration and provider cost/termination controls are still outstanding. None of these fixture checks establishes an SAE reconstruction improvement, a useful compartment, launch authority or completion of Q2.
