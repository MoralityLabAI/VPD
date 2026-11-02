# Fresh-seed SAE replication: review draft F1

Status: local design only. Training and capture have not run. This is a new candidate plan, not a modification of the sealed 25-report specification or an execution authorization.

Source feasibility correction: the hash-verified local `artifacts/nanbeige_reasoning/dataset_v1` corpus has only 29 eligible family labels. F1 needs 32 pairwise-disjoint families per pack and 64 across A/B. **This corpus cannot supply F1 as written.** The counts below remain the original proposed design, not an executable allocation. Additional audited sources or a separately versioned design are required; changing example/group IDs into family labels would not resolve this scientific constraint.

Recovery-source follow-up: `VPD_SAE_RECOVERY_SOURCE_LEADS_2026-09-11.md` records a readable E: clone with seven nominal additional procedural task types after excluding two obvious label aliases. Fifteen source hashes were checked. These are generator leads, not new admitted families or ready examples: the documented task uses a Prolog/container interface and requires code, validator, prompt-adaptation, independence and exposure checks. The 29-label corpus count remains unchanged, and even the hypothetical 29+7 label count would not fill the 64-family design.

## Decision this experiment must resolve

New pre-launch evidence: **zero of four archived Q4 shared SAE fits pass** pooled both-pass EV 0.20 on generated rows alone, despite all four passing mixed-phase EV. Prompt-tail rows pass pooled quality in 4/4, but both phases still fail the all-family prerequisite. The archived mixed-phase evidence does not validate generated-reasoning compartments. Prioritize a bounded quality prerequisite before expanding replication; F1 remains unexecuted and this finding does not authorize a repair or extra training arms.

The new training-side audit sharpens that priority: generated training EV is 0.844–0.994 and all training-family phase checks pass, while held-out generated quality fails. ARC/math also show large descriptive gaps within shared family labels, so this is not solely a held-out-family-composition effect. See `VPD_SAE_QUALITY_REPAIR_Q2_DRAFT_2026-09-11.md` for a concrete alternative quality pilot: fixed-row-budget trajectory breadth versus density, with matched shared-SAE optimization and common initialization. Q2 is not an automatically added run matrix. Its row planner is now implemented and verified on synthetic metadata (16 tests), but fresh-pack construction, real-array alignment, paired training integration, endpoint calibration and execution authorization remain unresolved.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_q4_sae_training_phase_quality.py tests/test_vpd_q4_sae_phase_quality.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.audit_q4_sae_training_phase_quality_v1 --verify artifacts/vpd_q4_sae_training_phase_quality_v1/report.json
```

These checks use saved codes and activations only. The verifier first reproduces the evaluation-side receipt, then checks training reconstruction; it performs no new SAE fitting. High training EV is not a substitute for held-out quality or a proof of a specific overfitting mechanism.

Can recurrent-pass feature groups be reproduced across independently initialized shared SAEs and independently sampled task families, with useful nonnegative reconstructions? The archived 2K/8K Q4 comparison supports span overlap at all three sites, but uses the same seed and example IDs. The layer-21 independent-seed experiment failed reconstruction quality in every condition. Neither establishes independent replication at layer 11 or final norm.

The new synthetic audit gives a counterexample to using span overlap as sufficient evidence: a sign reversal has overlap 1 and nonnegative reconstruction error 1; orthogonal feature mixing has overlap 1 and error 0.375. These are exact orthonormal toy calculations, not measurements of the empirical model.

## Local executor steps available now

From `C:\projects\VPD`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_sae_subspace_semantics_audit.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.audit_sae_subspace_semantics_v1 --verify artifacts/vpd_sae_subspace_semantics_audit_v1/report.json
```

These checks use synthetic arrays. The audit binds the implementation files it uses and compares Gram-space results with a direct-SVD reference. It does not replay the entire empirical lineage. The existing V44 preflight remains the empirical checkpoint.

A supplementary saved-code reconstruction audit is also available:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_q4_sae_group_reconstruction.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.audit_q4_sae_group_reconstruction_v1 --verify artifacts/vpd_q4_sae_group_reconstruction_v1/report.json
```

This reads existing SAE decoders, sparse codes and captures, and checks their alignment and reconstruction quality. It performs no encoder fitting or model inference. Across eight archived groups, target-minus-other removal effects exceed matched-random means even after decomposing out the pass mean. The large norm pass-0 effect is approximately 95–97% mean-related. These results inform endpoint design but supply no significance or fresh-seed replication claim.

## Candidate run matrix

Use shared monolithic SAEs at `model.layers.11` and `model.norm`. Keep layer 21 as archived negative evidence; no new layer-21 training is requested by F1.

| Candidate | Data | Seeds | Sites | New SAE fits | Purpose |
|---|---|---|---|---:|---|
| F1-A | Fresh pack A | 1707, 250829, 260829 | layer 11, norm | 6 | Seed replication on one fixed corpus |
| F1-B | Fresh pack B, families disjoint from A | same three seeds | same two sites | 6 | Separate corpus effects from seed effects |

Total proposed fits: 12. Seeds are crossed with packs: do not assign a different seed to every pack and confound seed with corpus. Use identical verified model revision, BF16 capture settings, decoder width, optimizer, loss, steps, centering rule and row sampling across A and B. Preserve the sealed shared-arm parameter budget unless a separately reviewed revision changes it. The archived Q4 results motivate this experiment; Q4-to-BF16 differences cannot be attributed solely to seed or corpus.

First run all six A fits to a fixed budget, including quality failures. At each site, every A seed must pass both pooled and all-declared-family reconstruction prerequisites below before that site's B fits are eligible. Any failed or unevaluable prerequisite defers that site's B fits pending a revised design. Do not retune an individual failed seed and keep only the successful replacement, or drop failing families to qualify a site. This is a proposed stop rule, not yet connected to an executable paid runner.

## Pack and inference requirements to resolve before launch

Implemented inventory check: `experiments/authz_mechinterp/audit_sae_source_family_inventory_v1.py` verifies the corpus manifest/question-file hash and counts all 34,886 rows by family, source split, and unique declared group. There are 31 source labels in total; `arc_agi` occurs only in quarantine and `campsite` only in historical evaluation, leaving 29 labels with at least four distinct declared groups in an allowed split. Source `train` and `eval` each supply 29 such labels; `dev` supplies 28. Counting labels is only an upper bound: correlated task variants, historical exposure, model pretraining, group near-duplicates, and semantic family independence have not been cleared. No prompt, role or grader was selected by this audit.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_sae_source_family_inventory.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.audit_sae_source_family_inventory_v1 --verify artifacts/vpd_sae_source_family_inventory_v1/report.json
```

Seven tests and receipt verification pass. This is an inventory of the named corpus, not a claim that no other local/recovered sources exist. Q2's ten-training-family row planner remains structurally valid, but inheriting all 32 F1-A role families from this corpus is also infeasible. A revised quality-only role layout would need a new explicit design and cannot stand in for F1's full independent-replication requirements.

Each candidate pack has 32 semantic families with four examples each (128 examples): four discovery families, four threshold-calibration families, four cluster-validation families, ten SAE-training families, and ten final-evaluation families. All role sets and both packs are pairwise family-disjoint. Total proposed capture: 256 examples. Pack contents, family definitions, IDs and hashes are still absent; this document does not claim that a fresh pack exists.

This is a deliberate proposed departure from the sealed 20-family/80-example layout. Four independent training families yield only 16 family-level label flips, giving a minimum exact p-value of 0.0625. Treating related examples as independent would not fix that. Ten training families yield 1,024 flips, allowing the multiplicity plan below to resolve significance. The four-family discovery/validation roles do not authorize new fine clusters or a new activation-phase category under F1; use the existing coarse pass labels, with no child subdivision.

Family counts establish attainable p-value resolution, not statistical power. The prospective functional-test simulation below now shows that ten evaluation families have low power for moderate standardized effects. The 256-example layout is therefore a candidate cost envelope, not a justified final sample size. Before capture, calibrate a useful effect and family-level variability without inspecting the final holdout, then revise the family count if needed. Dollar cap, TTL, provider quote, memory requirements and actual pack content must be filled in before paid execution. No cost estimate or executable training CLI has been verified for F1 yet.

## Measurements and controls

### Phase scope and identical-input prerequisite

The phase-quality audit is now implemented, not just a requirement. Generated-row minimum pooled EV is 0.094015/0.120083 for layer 11 at 2K/8K and 0.114869/0.175037 for norm. All are below 0.20. Prompt family checks pass 12/16, generated checks 1/16; no site/archive/phase meets the all-family prerequisite. The sole passing generated-family cell is ARC/norm/8K, which is not a site-wide or family-general admission. Retain all families and report both phases; do not select that cell after the fact as a successful experiment.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_q4_sae_phase_quality.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.audit_q4_sae_phase_quality_v1 --verify artifacts/vpd_q4_sae_phase_quality_v1/report.json
```

Eleven new tests and the full saved-code receipt verifier pass. The diagnostic uses the actual saved sparse codes and decoder, not a newly fitted SAE or oracle mean correction. The result concerns retained prompt tails and generated windows in mixed-trained Q4 archives; it does not establish quality across full trajectories or contradict the separate frozen prompt-transfer negative.

The archived Q4 fits were not generation-only: each example contributes eight prompt and 24 generated positions, all used in training and aggregate EV. F1 must freeze its training phase mixture and report quality/contributions by phase as well as pooled. A generation-specific claim requires generated-row quality within every planned evaluation family; a mixed-phase pass cannot substitute for that. Do not silently remove prompt rows from training across only one arm. The planned 32-row count alone does not specify a phase mixture or justify a generation-only interpretation.

Saved 2K/8K codes cannot simply be paired by example ID and row number. A new local feasibility audit finds 144 common token-prefix positions per pass; 135 have exactly equal activations at all compared sites/passes. Only 55 of those are generated rows, and none belongs to Dyck. Nine prefix-matched generated positions differ numerically. The cause is unestablished; do not choose an after-the-fact tolerance or drop missing families to declare success. Any archive-subset functional analysis would be descriptive and must remeasure quality on that subset before interpretation.

For fresh cross-fit tests, freeze a common untouched activation roster for each comparison and run each independently fitted SAE encoder on those same arrays. Bind example, family, role, full token-prefix hash, absolute token position, site, recurrent pass, phase, and activation-array hash. Category labels or identical current tokens after prefix divergence are insufficient. Save the full eligible/failed family denominator. Choosing the actual common cross-pack evaluation roster remains part of the unresolved pack design; this diagnostic does not supply it.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_q4_common_capture_rows.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.audit_q4_common_capture_rows_v1 --verify artifacts/vpd_q4_common_capture_rows_v1/report.json
```

Eleven tests and receipt verification passed. The command reads saved captures only, not SAE encoders or model weights. It establishes alignment limits, not functional agreement.

### Q1 sampling-only candidate: dry-run disposition, not an added training run

`Q1-sampling-only-draft-v1` tests a specific candidate change: uniform sampling over training rows versus equal family probability, then equal example probability within family, then uniform row probability within example. It is not inverse-variance loss weighting and does not create family-specific dictionaries. The metadata-only implementation is `experiments/authz_mechinterp/plan_sae_family_sampling_v1.py`.

Inspection of the current archived-fit implementation (`experiments/nanbeige_reasoning/fit_q4_reasoning_saes.py` and `experiments/nanbeige_looped/sae.py`) shows shared top-k SAEs with width 3072, 6144 features, top-k 32, 1000 updates, 64 aligned rows per pass per update, raw MSE, AdamW learning rate 0.001, weight decay zero, decoder normalization after each update, and training-pooled initial pre-bias. The archived manifests independently record 1000 steps and seed 42; the other settings come from current source inspection, not a claim of recovered historical source-version hashes. Any executed candidate must pin the actual implementation and environment anew.

| Training family in archived 2K roster | Examples | Uniform-row probability | Family-balanced probability |
|---|---:|---:|---:|
| ARC grid | 4 | 0.50 | 0.25 |
| Math | 2 | 0.25 | 0.25 |
| Futoshiki | 1 | 0.125 | 0.25 |
| Goods exchange | 1 | 0.125 | 0.25 |

The two distributions differ on the archived roster. But campsite and Dyck have no training examples; balancing cannot introduce them. Transfer improvement is possible but untested. Their absence alone does not invalidate an intentionally family-held-out experiment, and evaluation examples must not be moved into training to make coverage look complete.

**Do not schedule Q1 as an extra arm on a balanced F1 pack.** A synthetic roster with F1's ten training families and four examples per family, assuming 32 retained rows per example, gives exactly identical distributions and batch-index hashes for both arms. It is a shape/count demonstration, not the missing fresh pack. Recheck equivalence against actual retained counts before any launch; variable lengths or censoring can break equivalence. Do not manufacture imbalance merely to create a contrast. Equal weighting of family EV and inverse-variance optimization are different changes and cannot be smuggled into this sampling-only arm.

For any later nonredundant sampling control, hold initialized parameters (including pre-bias), architecture, objective, step count, rows per step, optimizer, and evaluation fixed across arms. Couple sampling randomness by applying each arm's inverse CDF to the same uniform draws, and use the same selected row indices on both passes. This is matched distributional sampling, not replay of the historical `rng.integers` batches. Record all failures. Before authorizing training, require a distinct sampling distribution, a frozen training-only rationale and fresh evaluation plan, reviewed cost/TTL, and a verified execution entry point. Q1 has not met those launch requirements and adds zero authorized fits to F1.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_sae_family_sampling_plan.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.plan_sae_family_sampling_v1 --verify artifacts/vpd_sae_family_sampling_plan_v1/report.json
```

Nineteen focused tests passed and the receipt reproduced. The module samples indices only, imports no training framework, and loads no activation arrays. Report ID: `56b87b315e041f8c3a0a14caeb1c223adc2bc45f21fb9a9c9bd09bf6b47ba663`; SHA-256: `b72e2d45de7c8058494e7ca742d6dbe80068ade88f24632044ebf531439eec46`.

### Reconstruction and replication endpoints

The subsequent local error-budget diagnostic narrows what could repair the pilot. Mean offsets contribute 4.51–17.50% of family reconstruction SSE. Even an oracle output correction using each held-out family/pass residual mean leaves Dyck below paired EV 0.20 at both sites and budgets; oracle paired passes are 11/16 versus the actual 4/16. This only bounds constant output-offset corrections with saved codes fixed, not retraining or changed encoders. Keep actual raw reconstruction EV as the prerequisite; oracle-corrected EV cannot replace it. Between-family means explain just 3.20–5.34% of pooled activation TSS, so do not attribute the entire pooled-quality discrepancy to those means. Task families remain evaluation strata, not admitted activation categories or routing labels.

Reproduce this supplementary diagnostic without training:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_sae_family_error_budget.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.audit_q4_sae_family_error_budget_v1 --verify artifacts/vpd_q4_sae_family_error_budget_v1/report.json
```

Ten tests and receipt verification passed. The analysis motivates testing within-family quality repair before scaling, but does not establish a repair. Any added objective, family-balanced sampler, input-centering change, or dictionary allocation must be a separately frozen candidate with matched-budget shared controls; do not silently change F1 across seeds or optimize on the final evaluation families.

1. **Reconstruction prerequisite.** Use the fitted encoder and decoder on untouched evaluation rows. Require pooled per-pass EV at least 0.20 in every compared fit; report sensitivity at 0.25. Also report within-family per-pass EV for every predeclared evaluation family. A claim covering all planned families requires every family to meet both-pass EV 0.20; do not drop failing families after evaluation. Undefined within-family variance yields an unevaluable prerequisite, not a pass. Retain the fitted training-derived SAE bias; define the variance denominator explicitly for pooled and family EV. Include all failed fits and families in the denominator. Freeze evaluation rows and generation censoring rules before comparing seeds.
2. **Individual feature identity.** Use the existing signed, mutual, one-to-one V2 matcher at cosine 0.80, with training-only pass assignments. Report counts, bidirectional coverage and category profiles. The archived selective-stratum thresholds (16 matches and 0.10 coverage per pass) define a limited stratum endpoint; they cannot establish full-dictionary equivalence.
3. **Subspace geometry.** Rank 16 is primary; ranks 4/8/32 are descriptive. Record singular-value spectra and the gap at the rank cutoff. A near-tied cutoff requires a prespecified sensitivity analysis and cannot silently select a favorable basis. Report each pass separately as well as the mean, because a mean can hide one failing category.
4. **Functional contribution.** For dictionary i and pass group p, form `c_i,p(x) = z_i,p(x) D_i,p` using that dictionary's actual nonnegative encoder outputs on the same untouched rows. Compare these contributions across seeds and packs, with row norms, per-pass EV and L0 reported alongside agreement. Independently fit each encoder; do not transfer signed least-squares coefficients and call them SAE features.
5. **Matched random groups.** Build ten training-derived random feature-group assignments per fit, preserving group sizes and bins of training activation frequency and mass. Freeze these assignments before final evaluation. Compare functional contribution agreement and loss under within-dictionary group removal against these controls. Group removal is an offline reconstruction diagnostic and makes no model-behavior or causal claim.

The functional endpoint and minimum useful effect still need a frozen numerical acceptance rule. Specify the denominator, treatment of near-zero contributions, family-level uncertainty, and calibration-only threshold selection before launch. Pure span significance cannot substitute for this missing rule. Reconstruction after unrestricted NNLS is a secondary oracle diagnostic only; it does not validate the learned sparse encoder or matched L0.

The saved-code pilot now provides an implemented removal statistic: for residual `r = x - full_reconstruction`, group removal changes SSE by `||c||² + 2<r,c>`. Report the full effect and its exact decomposition into pass-mean and within-pass variation components, using the same held-out variance denominator for the selected and random groups. Neither component is automatically a causal behavioral effect. Freeze which component is primary on the calibration role; keep both in the final report. Save per-example contributions and family membership in the future experiment so family-block uncertainty is possible. Do not infer cross-fit functional agreement merely because two within-fit removal effects have the same sign.

## Multiplicity and uncertainty

The saved-archive family audit (`artifacts/vpd_q4_sae_family_contributions_v1/report.json`) sharpens this quality prerequisite: only ARC clears both-pass EV 0.20 at either proposed site, while campsite, Dyck-language and math fail. All eight archived pass groups have positive selected-minus-random contribution advantages in all four families, so contribution usefulness and sufficient reconstruction quality are separate conditions. The historical pooled quality admissions do not establish family-general quality. Six ARC examples versus one campsite and one Dyck example also make a pooled token/example weighting unsuitable as the sole generalization summary.

Reproduce the supplementary audit locally with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_q4_sae_family_contributions.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.audit_q4_sae_family_contributions_v1 --verify artifacts/vpd_q4_sae_family_contributions_v1/report.json
```

This uses saved codes and decoders and no new fitting or model inference. It reports descriptive family SD, family scores and leave-one-family-out means, but four broad family labels cannot calibrate a reliable population variance or confirm the symmetry assumption used in the power simulation. ARC and math overlap between training and evaluation. Fresh pack family definitions and adequate examples per family remain necessary.

The all-family quality prerequisite also has a local tested implementation:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_sae_family_quality_gate.py tests/test_vpd_q4_sae_family_contributions.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.evaluate_sae_family_quality_v1 --source artifacts/vpd_q4_sae_family_contributions_v1/report.json --expected-family arc_grid --expected-family campsite --expected-family dyck_language --expected-family math
```

Verified result: 32 tests pass; four pooled fits pass, only four of sixteen family checks pass, and zero of four fits pass the all-family prerequisite. The family scope is explicit and must not be inferred by dropping absent rows. The CLI checks receipt content integrity only; run the source audit verifier above to verify the arrays. Its exit code zero means the diagnostic completed, not that quality passed. Read each fit's `quality_prerequisite_pass`. For F1, call the reusable `evaluate_quality` function with the frozen pack's complete evaluation-family inventory and measured per-pass EV; the current CLI consumes the archived Q4 schema only. Passing this prerequisite alone establishes neither reproducible groups nor a case for split dictionaries. These checks do not launch training or enforce provider termination.

At each site there are three seed pairs within A, three within B, and three same-seed A/B comparisons: nine comparisons per site, eighteen overall. Report all eighteen. For the primary geometry endpoint, use Bonferroni alpha `0.05 / 18 = 0.002777...`; ten training-family flips give a minimum exact p-value `1/1024 = 0.0009765625`. Recompute training-derived feature assignments under family-level pass-label flips. Do not use decoder feature rows as independent experimental replicates. The existing feature-label permutation remains a descriptive geometry null and must be labeled as such.

Geometry, feature identity and functional reconstruction are distinct endpoints. Freeze their joint decision rule before capture; do not take the best endpoint or rank after observing results. Use family-block uncertainty for evaluation differences, and report seed-pair dependence explicitly. Three training seeds are not enough to make a precise population claim about arbitrary initialization seeds.

## Interpretation and handoff

### Prospective family-test sensitivity (implemented locally)

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_sae_family_test_power.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.simulate_sae_family_test_power_v1 --verify artifacts/vpd_sae_family_test_power_v1/report.json
```

The simulation tests the one-sided mean of paired family effects after subtracting a candidate minimum-useful margin. Examples receive equal weight within family and families receive equal weight. It enumerates every family sign flip, using alpha `0.05/18`, 512 synthetic experiments per cell, and Gaussian and unit-variance Student-t(3) family errors. Effects below are excess over the useful margin in units of the standard deviation of **family means**, not individual tokens or examples.

| Independent evaluation families | Excess 0.5 SD | Excess 1.0 SD | Excess 1.5 SD | Excess 2.0 SD |
|---|---:|---:|---:|---:|
| 10, Gaussian | 4.9% | 30.1% | 71.9% | 95.7% |
| 12, Gaussian | 9.6% | 56.6% | 93.0% | 99.0% |

Four and eight families cannot pass this alpha at any effect size because their exact p-value floors exceed it. For ten families, the smallest tested Gaussian effect with estimated power at least 80% is 2 SD; for twelve, it is 1.5 SD. These are coarse-grid results, not exact minimum detectable effects. The report contains Monte Carlo Wilson intervals: for example, ten families at 1 SD have estimated power 30.1%, interval [26.3%,34.2%]. Student-t cases are sensitivity checks and not evidence that empirical family errors follow that distribution.

This analysis is conditional on independent, symmetric family errors and frozen dictionaries, groups and normalizers. Zero expected effect alone does not justify a sign-flip test. It does not establish power for the geometry test's training-label permutations, for jointly passing all eighteen comparisons, or for passing reconstruction prerequisites. Those are separate simulations still required by the final joint decision rule. Three seeds and four archived held-out families do not supply a reliable empirical variance estimate for this calculation.

For a future functional test, compute a paired score per example on common evaluation inputs, aggregate within family, and subtract the numerical useful margin before the family-level test. Freeze per-pass normalizers using calibration data; reject undefined or effectively zero normalizers under a fixed numerical rule. If targeting within-family variation, center residual and contribution within that family and pass, rather than centering across the entire evaluation population and silently coupling family scores. Record the full mean contribution separately. Group selection and any matching remain training-only. The implementation's family aggregation test confirms that replicating examples within a family does not increase the number of independent units.

The useful margin in original reconstruction units is intentionally unset; the script reports standardized excess effects. A nominally small p-value cannot fill in this missing scientific choice. Set the joint functional/geometry rule and useful margin using calibration evidence, then require a suitably powered design before approving compute. Do not upgrade this power sensitivity analysis into an empirical model result.

Receipt: `artifacts/vpd_sae_family_test_power_v1/report.json`; report ID `650e0ee6b2a76b7a15a5388f1e89c44ef62bf9b69d39334dadfcf42f8953a5aa`; SHA-256 `ddede6ed18abeb712e853c176f8261c362e43f19bc133425fc628720c2d92164`. Seven focused tests passed, including independent scalar enumeration, direction/tie handling, finite input checks and family duplication invariance.

If geometry passes while functional contributions fail, report geometric overlap only. If a reproducible functional group passes, it supports grouping features within a shared SAE. It does not establish that separate dictionaries use capacity better. Any later split-SAE trial still requires a fresh equal-budget comparison against both monolithic and matched random routing.

Completion before requesting paid execution means: freeze two actual packs; validate role separation and model/capture dimensions; freeze functional endpoint and power analysis; verify the training entry point and full run matrix; set a reviewed compute budget and independent termination controls. F1 is a concrete review draft with these remaining requirements exposed, not a launch-ready job.
