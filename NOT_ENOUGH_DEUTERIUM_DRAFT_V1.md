# Not Enough Deuterium to Detonate the Ocean

## Limits to Mechanistic-Interpretability-Fueled Recursive Self-Improvement in DRAM-Trained Tiny Models

**Draft 0.1 — reconstructed synthesis, 2026-09-07**  
**Evidence window:** 2026-05-12 through 2026-07-12  
**Author:** [name to be added]

> **Provenance note.** This is a new paper assembled from intact experiment reports, receipts, and the earlier manuscript *VPD-Abetted Feature Steering in Tiny Recursive Models*. It is not represented as a recovered copy of a lost May–July paper. Quantitative claims below come from the durable local sources listed in Appendix A. The title's spelling has been normalized from “Dueterium” to “Deuterium.”

## Abstract

Mechanistic interpretability is sometimes imagined as an accelerant for recursive self-improvement: decompose a model into meaningful mechanisms, identify the mechanisms responsible for a desired capability, edit or train those mechanisms, retain the improvement, and repeat. Small recursive reasoning models appear to be a favorable test bed for this idea. They are compact, can be inspected and modified on commodity CPU and DRAM, and expose repeated internal computation through a small number of weights. If an interpretation-guided improvement loop can ignite anywhere cheaply, this regime should offer unusually favorable conditions.

Across two months of experiments, we tested successive links in that loop using low-rank sensitivity maps, adVersarial Parameter Decomposition (VPD), ablation rankings, runtime grafts, basis-constrained edits, decomposed LoRA patches, targeted adapters, symbolic training objectives, Jacobian-based persistence measures, and preregistered cross-context identity tests. The program found reproducible local structure. On a small Hermes critic, independently sampled row windows produced an ablation-damage Spearman correlation of 0.991866 and a top-ten overlap of 9/10. A six-seed tiny-model graft study also produced 16 accepted VPD grafts among 1,305,171 records. These results show that the models are not homogeneous and that decomposition can identify locally consequential directions.

They did not, however, close an improvement loop. Proxy-positive edits yielded zero accepted downstream improvements after matched-random controls. VPD-targeted LoRA failed to outperform raw-base, zero-scaffold, vanilla, or random-target controls on compact symbolic reasoning. An apparent read-write advantage disappeared under attribution-shuffled controls. A regression-cleaning patch retained target movement in one cached case but did not beat the same-diff/random acceptance threshold. Finally, a registered Qwen3-1.7B calibration found stable scalar rankings but zero certifiable cross-layer identities among 192 sites, stopping before behavioral intervention.

The result is a negative one with safety relevance. In this tested regime, interpretability supplied useful diagnostics, candidates, and regression controls, but not the stable identities, causal transfer, held-out advantage, and recurrence required for self-sustaining mechanistic-interpretability-fueled improvement. In the paper's metaphor, there was detectable fuel, but not enough effective fuel under the required confinement and loss conditions to propagate a detonation. This does not establish that recursive self-improvement is impossible, nor that interpretability-guided editing is safe. It identifies concrete empirical gates that the tested approach did not cross and argues that local interpretability artifacts should not be granted autonomous authority over weight updates merely because they are reproducible or legible. Crucially, most completed interventions were micro-edits, direct VPD-targeted adapters, or reinforcement-like arbitration rather than a grounded learned policy. The stronger hypothesis—that VPD observations can improve an RL controller which allocates training experience and authorizes broad updates—remains untested. We specify that decisive experiment here rather than treating its outcome as known.

## 1. Introduction

Early nuclear physicists considered whether a sufficiently energetic explosion could ignite a self-propagating reaction in the atmosphere or ocean. The formal 1946 Los Alamos report LA-602 analyzed atmospheric ignition; the paper's title deliberately borrows the broader ocean-ignition version of the historical worry as a metaphor, not as a summary of that report [1]. The important question was not whether a hot region or an individual reaction could be produced. It was whether the released energy, available reactants, reaction rates, and losses jointly permitted propagation. A dramatic initiating event is not a chain reaction when each stage fails to reproduce the conditions required for the next.

This distinction is useful for evaluating claims about recursive self-improvement (RSI). A model that helps improve one artifact is not necessarily recursively self-improving. A mechanistic interpretation that predicts one local ablation is not necessarily an edit coordinate. An edit that moves a proxy is not necessarily a capability gain. A capability gain on one evaluation is not necessarily transferable, retained, or repeatable. The loop closes only when improvements reliably create the conditions for further improvements.

This paper studies one proposed ignition path:

1. decompose a model into simple parameter-level components;
2. identify components associated with a target behavior;
3. use those components to choose edits, adapters, training data, or control policies;
4. verify a net improvement on held-out behavior;
5. preserve the improvement without unacceptable regression;
6. repeat the process with equal or greater effectiveness.

The experiments were unusually favorable to the cheap-compute version of this hypothesis. They focused on tiny recursive models, small critics, cached heads, and locally runnable language models. Much of the mapping, decomposition, control generation, and evaluation could be conducted using commodity CPU and DRAM, with tightly capped GPU work reserved for model execution. Recursive architectures also reuse a small set of weights, making the hope of finding compact, repeatedly influential mechanisms more plausible than in a large feed-forward stack.

Yet the full program produced a consistent separation between **local structure** and **portable control**. Sensitive directions existed. Ablation rankings could be highly reproducible. Some runtime grafts survived random controls. But the candidate mechanisms generally failed one or more of four requirements: stable identity across contexts, downstream causal efficacy, superiority to matched controls, and recurrence across seeds or iterations.

The contribution of this paper is therefore not a new method for self-improvement. It is a controlled negative result and a set of empirical thresholds for deciding when interpretability has earned the right to steer an improvement process.

## 2. Scope and terminology

### 2.1 VPD and parameter components

AdVersarial Parameter Decomposition (VPD) decomposes a model's weight matrices into simple parameter subcomponents while training those components to remain faithful under adversarially selected ablations [2]. It is a parameter-level method: the proposed objects live in the transformations that produce activations rather than solely in activation space. The originating work shows that such components can be interpretable, can span conventional attention-head boundaries, and can support targeted manual edits in a small language model.

The present study asks a different question. It does not test only whether VPD can find interpretable or causally active pieces. It asks whether VPD-style pieces can act as reliable coordinates for repeated model improvement.

Some early experiments used singular-value or low-rank atoms as VPD-style proxies because full ComponentModel training was not feasible in every local configuration. The evidence ledger distinguishes those spectral decompositions from genuine trained VPD components. In particular, a 248-matrix rank-64 Qwen3.5-4B artifact was an SVD map, not a full VPD map.

For HRM-Text-1B specifically, recovery confirms that a genuine full-coverage ComponentModel run was attempted across all 16 paired L/H layer indices on May 19. This corrects any implication that the HRM program consisted only of SVD. The recovered terminal learned state is not scientifically usable: every chunk ends with a non-finite loss, and direct inspection of a representative checkpoint finds every learned component and CI-function element non-finite. Loader reconstruction also resolves the historical ambiguity adversely: the May-pinned model code expects split q/k/v and gate/up matrices, whereas the recovered trained checkpoint contains native fused `gqkv_proj` and `gate_up_proj` weights, producing missing, unexpected, and newly initialized tensors. A separately labeled reconstructed fused loader now loads all recovered weights exactly and passes a 32-prompt local no-op instrumentation test, but that repair does not retroactively validate the May decomposition. Thus the historical record establishes a full VPD attempt and two concrete failure modes, not a validated full VPD artifact. The later low-rank map remains a separate SVD-derived sensitivity artifact.

### 2.2 Tiny recursive models

The program was motivated by Hierarchical Reasoning Models (HRMs) and Tiny Recursive Models (TRMs): small architectures that obtain computational depth by recurrently applying compact networks [3,4]. “Tiny” is relative to contemporary language models. The tested family ranged from several-million-parameter recursive models and small task heads to local language models in the roughly one-to-four-billion-parameter range.

Recurrence matters because a component that participates in a repeated update may exert influence many times. That creates a plausible amplification mechanism. It also creates a stricter stability problem: an edit must remain beneficial over repeated state transitions rather than merely moving a single-step output in a desired direction.

### 2.3 “DRAM-trained” is an operational constraint

In this paper, **DRAM-trained** means that the relevant decomposition, candidate generation, adapter training, or experiment orchestration can be performed primarily within commodity CPU and system-memory constraints. It does not name a new learning algorithm, and it does not imply that every forward pass was GPU-free. The phrase marks the threat model: if useful model modifications can be found and validated with abundant consumer DRAM rather than scarce accelerator clusters, the cost threshold for automated experimentation falls sharply.

This definition should be read conservatively. Existing model compression already moves inference toward cheaper hardware without mechanistic interpretability. The relevant comparison is therefore not “interpretability versus no compression,” but whether interpretation-guided search delivers more reliable improvement per unit of compute than ordinary low-rank, random, gradient-based, or data-selection baselines.

### 2.4 What counts as recursive self-improvement

We use RSI narrowly enough to be testable. A self-improvement cycle must produce a verified change to a system involved in generating, selecting, or evaluating subsequent changes. Open-ended autonomy is not required for a single experimental cycle, but a credible RSI mechanism must show that cycles can recur without the signal collapsing into noise, overfitting, or evaluator exploitation.

For this study, an interpretation-guided loop is not considered closed unless it satisfies all of the following:

- **Identity:** the selected component or subspace retains a measurable identity across the contexts in which it will be used.
- **Causality:** intervention on the object changes the intended behavior, not merely an internal proxy.
- **Controlled advantage:** the intervention beats raw-base, scaffold-only, ordinary-training, shuffled, and matched-random controls appropriate to the claim.
- **Generalization:** the gain survives a fresh held-out evaluation and is not concentrated in one family, norm, seed, or cached surface.
- **Retention:** the gain remains after checkpointing or deployment without unacceptable off-target regression.
- **Recurrence:** the procedure remains effective over more than one improvement cycle.

These conditions are conjunctive. A strong result on one axis cannot compensate for a failed gate on another.

## 3. Experimental program

The research proceeded by escalating both the intervention strength and the quality of the controls.

First, low-rank perturbations and decompositions mapped where model behavior was sensitive. Second, component ablations and runtime grafts tested whether those maps contained repeatable causal structure. Third, basis edits and downstream hill climbs tested whether proxy movement transferred to held-out task performance. Fourth, LoRA and patch-extraction experiments asked whether decomposition could guide training or clean a learned weight diff. Fifth, symbolic operators and storyworlds attempted to supply more semantically explicit objectives. Finally, Jacobian, sheaf, and identity-calibration work tested whether components preserved enough cross-context identity to justify persistent edits.

The control hierarchy also became stricter over time. Apparent successes were re-evaluated against:

- raw checkpoints;
- zero-adapter scaffolds;
- ordinary or naive LoRA;
- matched-random modules or directions;
- attribution-shuffled read-write pairings;
- same-diff random patch extraction;
- grouped family- and seed-level inference;
- preregistered stopping thresholds before behavioral outcomes.

This progression is central to the result. Several promising intermediate effects survived simple controls and disappeared only when the experiment tested the exact causal claim being made.

| Required link | Strongest supporting observation | Strongest limiting observation | Verdict |
|---|---|---|---|
| Local sensitivity | HRM depth sensitivity was sharply nonuniform | Sensitivity did not predict portable task gain | Supported as diagnosis |
| Reproducible components | Hermes damage ranks: Spearman 0.991866; top-ten overlap 9/10 | Rank stability did not establish coordinate identity | Supported locally |
| Portability | 16 accepted grafts across four of six seeds | Sparse among 1,305,171 records; seed-concentrated | Provisional and narrow |
| Proxy-to-task transfer | Targeted search produced apparent accepts before full controls | Zero accepted Intellect edits after matched-random controls | Not supported |
| Interpretation-specific causal advantage | Identified K=1 read direction beat random on a 4D surface | Shuffled attribution tied or beat the pairing; random recovered at K=2 | Read signal only |
| Training advantage | Working capped VPD-targeted adapter path | VPD LoRA failed to beat raw, zero, vanilla, or random controls | Not supported |
| Regression-safe retention | One extracted patch preserved target delta and removed measured regression | Did not beat same-diff/random p95; no aggregate advantage | Governance lead only |
| Cross-context identity | Several scalar rankings were split-half reliable | Zero of 192 Qwen sites certified | Not supported |
| Multi-cycle recurrence | No qualifying observation | No controlled sequence of retained improvements | Not demonstrated |

### 3.1 The stronger untested hypothesis

The completed work does not justify the claim that mechanistic interpretability is irrelevant to reinforcement learning. It rejects a narrower route: treating interpreted components or symbolic labels as direct edit coordinates and expecting local movement to become downstream gain.

A stronger use of interpretability would leave the update mechanism fixed and use mechanistic measurements as information for a learned outer policy. Such a policy could decide which training examples or failure families deserve the next update, when to abstain, and whether a candidate checkpoint should be retained or rolled back. Reward would come from post-update behavioral outcomes and regression costs, not from the VPD score itself. This creates a real action-to-outcome credit path while preventing the interpretation method from receiving a privileged edit norm or module set.

That distinction matters for the paper's thesis. If a VPD-informed training policy beats behavior-only, shuffled-VPD, and matched-random-map policies under equal compute, then the local components may be useful fuel even though micro-editing failed. If it cannot, the negative boundary becomes substantially stronger.

The preregistered design is specified in `HRM_TEXT_1B_VPD_RL_POLICY_PROTOCOL_V1.md`. It uses HRM-Text-1B, requires a newly trained, validated, and frozen fused-module VPD, and then tests a common broad adapter scaffold, a discrete macro-action RL controller, and a fresh sealed held-out pack. The recovered May endpoints are explicitly quarantined rather than silently reused. The reconstructed fused loader has passed the local 32-prompt base-conformance and no-op instrumentation gate; Prime execution and all VPD training remain separate prospective stages. The experiment is staged so that remote conformance or numerical failure stops before policy training, and a one-step policy-information failure stops the more expensive sequential run.

## 4. Results

### 4.1 Local sensitivity was real and highly nonuniform

The validated downstream HRM-Text-1B sensitivity pass processed 16 layer chunks with four singular-value slices per chunk. The mean of chunk-level absolute logit movements was 0.01797. Layer 14 reached a mean movement of 0.06989 and a maximum of 0.26440, while other layers were far less sensitive. These numbers belong to the low-rank SVD refinement and must not be presented as metrics from the recovered ComponentModel checkpoints.

This rejects a homogeneous-editability picture. Model depth and parameter direction matter. But sensitivity alone says only that a perturbation has leverage. It does not show that the direction has a stable semantic identity, that its sign is portable, or that using it improves a task.

A more important reproducibility result came from genuine SPD/VPD extraction on a small Hermes critic. Across disjoint row windows, component ablation damage had Spearman correlation 0.991866. Nine of the ten highest-damage components overlapped, for a top-ten Jaccard index of 0.818182. This is strong evidence that the decomposition captured reproducible local causal ranking rather than pure sampling noise.

The result also sets up the paper's central negative. Even near-perfect rank reproducibility did not establish a coordinate stable enough for downstream improvement.

### 4.2 Runtime portability existed, but was sparse and uneven

The strongest positive result in the tiny-TRM branch came from runtime grafts among related gates. A six-seed replication generated 1,305,171 graft records and 16 accepted VPD grafts under increased random-control pressure. Accepted grafts appeared in four of six seeds, but seven were concentrated in a single seed.

The correct interpretation is **provisional narrow portability**. Related tiny models sometimes shared a runtime control motif that could be transferred under the tested gate. The evidence does not support uniform recurrence, permanent checkpoint transfer, or transfer into a different downstream benchmark. The concentration by seed also warns against treating a large candidate count as a large number of independent successes.

### 4.3 Proxy improvements failed at the downstream evaluation cliff

The Intellect-3-Logic program explicitly tested whether editable proxy margins predicted accepted task improvements.

An initial gate examined 7,764 candidates over 12 aligned rows and accepted none. A strict six-round hill climb evaluated 1,450 frontier rows, found 24 near misses, and again accepted none. A targeted format-commit run then appeared to produce 36 accepts. Once the same search was compared with 976 matched-random controls, the accepted count returned to zero across 1,056 targeted probes.

This sequence is more informative than an immediate null. It shows how a plausible improvement story can arise from a target-aligned search and disappear when the null distribution matches the search opportunity. The harness improved by learning to reject its own apparent successes; the model did not improve.

### 4.4 Low-dimensional surfaces created false confidence

On a cached four-dimensional ARC head, basis-constrained edits initially appeared meaningful. At rank K=1, identified directions yielded 193 accepts while matched-random directions yielded zero. At K=2, however, matched random recovered to 175 accepts as random subspaces approached saturation of the small surface. Attribution-shuffled controls tied or beat the identified basis at every tested rank, with counts of 194, 315, and 180.

The result separates two propositions. The selected **read direction** contained information. The proposed **read-write pairing** did not demonstrate added causal value. In a low-dimensional space, a random subspace can rapidly become a competitive intervention basis; attribution can then look special because the evaluation geometry is nearly spanned, not because the recovered mechanism is correctly paired.

Two richer surfaces failed for different reasons. A D=72 organelle experiment produced zero accepts in every arm because all 2,400 evaluations failed touch-specificity. An Intellect cliff experiment found a best guarded delta of 0.087 at norm 4, but no arm produced an accepted edit and all 2,400 cross-seed recurrence tests failed.

### 4.5 Decomposition cleaned one dirty patch but did not become an optimizer

The meso-patch branch tested a more modest use of interpretability: train a short LoRA episode, decompose the resulting dirty diff, and retain a smaller reversible patch with less regression.

The strongest cached case was `B->A:A:letter`. The raw LoRA diff and extracted patch each produced a target delta of 0.5. The raw diff's measured regression score was 2.015873, while the patch's score was 0.0. This is a genuine governance-positive example: decomposition preserved the measured target movement while removing the measured collateral effect.

It did not satisfy the capability gate. The patch tied rather than exceeded the matched-random p95 threshold, and its same-diff support was median-level rather than p95-level. Across 15 cells, the extractor won eight, lost six, and tied one against best-of-N; the two-sided sign test was p=0.790527. A separate cached verifier lane was a metric plateau in which every adaptation arm had zero target delta.

Patch extraction therefore remains promising for forensics, rollback, and regression control. These experiments do not show that it is a superior search algorithm or a generator of new capability.

### 4.6 VPD-targeted adapters did not beat their controls

A capped Windows training path successfully loaded quantized models, performed one-step LoRA updates, checkpointed results, monitored resources, and cleaned up exactly. This established that a commodity-compute experimental loop was technically feasible.

The behavioral outcomes were negative. In a live HRM-Text-1B smoke comparison, both VPD-targeted and matched-random adapters moved the true ARC target probability downward. In storyworld experiments, soft-only VPD-targeted LoRA scored 0.6875, below random-module LoRA at 0.765625 and soft-majority arbitration at 0.78125. On a ten-card evidence-ratchet test, VPD's margin advantage over random was +0.003065 after eight steps and reversed to -0.006077 after 16 stronger steps, with no accuracy gain.

The compact symbolic-operator branch produced the clearest controlled comparison:

| Arm | Accuracy | Mean target margin |
|---|---:|---:|
| Raw base | 0.50 | -0.172656 |
| Zero VPD LoRA | 0.45 | -0.200787 |
| Vanilla pairwise VPD LoRA | 0.45 | -0.200906 |
| Symbolic pairwise VPD LoRA | 0.45 | -0.200749 |
| Symbolic pairwise random-target LoRA | 0.45 | -0.196669 |

The VPD-targeted symbolic adapter was 0.05 less accurate than raw base, only 0.000038 better in margin than the zero scaffold, and 0.000157 better than vanilla VPD LoRA. The random-target adapter beat it by 0.004080 in margin.

This experiment exposed an additional confound: the zero PEFT scaffold was itself behaviorally non-neutral. On a larger contrastive set, the raw checkpoint scored 0.50 accuracy with mean margin -0.156738; the zero scaffold scored 0.45 and -0.196924. Any analysis that compared trained adapters only with one another would have attributed scaffold damage to training or overlooked it entirely.

The conclusion is not that symbolic operators are useless. They improved evaluation design and made failure modes easier to state. They did not become effective direct targets for VPD-selected LoRA in this regime.

### 4.7 Persistent identity failed before behavioral intervention

The final branch asked whether a component could be tracked as the “same” object across contexts or adjacent layers. Synthetic sheaf experiments first demonstrated the intended distinction: a locally selected edit had near-zero immediate error but 0.93872 persistent error, while a planted coordinated section had 0.0126462 error both immediately and after consistency dynamics. The construction worked on data built to contain the relevant geometry.

The real tiny-model bridge did not establish incremental predictive value. Raw Jacobian visibility predicted persistence better than section overlap, and a topology control showed that the scalar score followed the embedded global-section projector rather than path-versus-cycle topology itself. Cluster-aware reanalysis also showed that apparently tiny candidate-level p-values came from one or two family/norm cells; family-level sign-flip tests were not significant.

A preregistered Qwen3-1.7B identity calibration then evaluated 192 sites across layers 16–27, including 96 attribution-selected sites, 96 full-calibration selection checkpoints, 384 split-half estimator checkpoints, 24,576 candidate rows, and 12,288 contrast pairs.

Several scalar rankings were repeatable. Minimum-lineage reliability was 0.9891 and holonomy-angle reliability was 0.8873. But repeatability of a scalar ordering was not identity. Both split halves assigned certification level 0 to all 192 sites. The best bundle lineage reached only 23–27% of the random-pair p95 requirement. Eleven of twelve modules had a negative holonomy determinant in each half, and median same-module worst-direction retention was approximately 7×10^-6. The preregistered design required 16 level-2 sites; it observed zero and therefore stopped before null interventions and behavioral outcomes.

This is a particularly important negative result. A measurement can be reliable in rank while the purported object it ranks has no usable absolute identity. Reproducibility is necessary, but it is not sufficient for intervention authority.

## 5. The missing conditions for ignition

The ocean-detonation analogy should not be taken literally. It is a way to distinguish isolated energetic events from a self-propagating process. The experiments identify at least five loss mechanisms in the proposed interpretability-to-RSI chain.

### 5.1 Sensitivity is not fuel quality

Large local effects show that a direction can alter computation. They do not show that the direction corresponds to the desired computation. A perturbation can be powerful because it disrupts a shared bottleneck, changes normalization, or moves the model off-distribution. The HRM depth map and Hermes stability result locate leverage; they do not identify a portable semantic control.

### 5.2 Reproducibility is not identity

A component score can be stable while the underlying coordinate rotates, reflects, or collapses across contexts. The Qwen calibration made this separation explicit: highly repeatable lineage and angle rankings coexisted with zero certified sites and nearly vanishing worst-direction retention. An RSI loop cannot safely retain or reapply an edit if the object named by the edit does not persist.

### 5.3 Proxy gain is not released energy

Many searches optimized quantities close to the decomposition: gate margins, cached heads, local verifier movement, or task-adjacent format commitments. These are useful instruments, but they are not the downstream objective. The Intellect evaluation cliff showed that apparent accepts vanished when tested against aligned downstream behavior and matched search controls.

### 5.4 Matched controls absorb apparent excess

An interpretation-guided edit has to beat alternatives with the same norm, rank, module count, scaffold, and search budget. Otherwise the effect may arise from generic perturbation, low-dimensional coverage, or optimizer opportunity. Random subspaces, attribution-shuffled pairings, zero adapters, and same-diff patch controls repeatedly explained effects that weaker baselines would have credited to interpretation.

### 5.5 Sparse successes do not sustain recurrence

The runtime-graft result shows that nonzero portability can exist. Its uneven distribution across seeds shows why that is not enough. A self-sustaining loop needs a sufficiently high yield of independent improvements after regression costs and failed trials. Sixteen accepts in more than 1.3 million records, clustered by seed, are scientifically interesting but not evidence of an accelerating process.

Taken together, the program found local reactions but not propagation. The effective fuel fraction was reduced by coordinate instability; losses were imposed by regression and scaffold drift; confinement failed at cross-context transfer; and apparent excess disappeared into matched nulls.

## 6. Safety interpretation

The negative result is safety-relevant in two directions.

First, it weakens a specific cheap-RSI hypothesis. In this study, commodity CPU/DRAM made broad mechanistic search and tightly governed small-model experimentation feasible, but cheap search did not make reliable improvement cheap. Evaluation, identity, and causal transfer remained the bottlenecks. The results provide no evidence that the tested VPD-driven loop can autonomously bootstrap increasing capability.

Second, the same failures make naive automation dangerous. A system that treats high ablation damage, stable rank, or semantic legibility as permission to edit weights will act on false identities and proxy gains. It may repeatedly select interventions that look mechanistically justified while accumulating drift, regressions, or evaluator-specific behavior. The absence of demonstrated beneficial RSI is not the absence of harmful self-modification.

Accordingly, mechanistic artifacts in this regime are better treated as **advisory evidence** than as **authorization tokens**. They may prioritize tests, identify fragile regions, propose candidate data, or help decompose a dirty diff. They should not independently authorize persistent weight changes unless they pass fresh behavioral, matched-control, regression, and identity gates.

This paper also does not support a broad claim that interpretability reduces AI risk by failing to improve models. A different decomposition, architecture, objective, evaluator, or compute scale may behave differently. Negative evidence about one route is valuable precisely when its boundary is stated.

## 7. Implications for experiment design

The most productive output of the program was an evidence discipline.

### 7.1 Separate discovery from authorization

Interpretability can be used to generate hypotheses with permissive thresholds. Authorization to retain an intervention should occur in a separate stage using frozen metrics, fresh held-out data, and controls not visible to the discovery process.

### 7.2 Treat scaffold effects as a first-class baseline

Quantization, adapter insertion, checkpoint wrapping, and evaluation plumbing can change model behavior before learning occurs. Every adapter experiment should include a raw checkpoint and a zero-initialized scaffold evaluated through the same path.

### 7.3 Match the null to the search opportunity

Random controls must receive the same number of proposals, norms, ranks, module choices, and evaluation opportunities as the interpretation-guided arm. A single random direction is not an adequate control for an interpretation method allowed to search thousands of candidates.

### 7.4 Count independent units, not rows

Millions of candidate records can reduce to a handful of independent families, seeds, or norm cells. Claims should be powered and tested at the highest level at which results could share a cause. Candidate-level p-values should not be presented as independent evidence when outcomes cluster by construction.

### 7.5 Require identity before persistent editing

If an edit is intended to transfer across prompts, layers, seeds, or checkpoints, identity should be measured before behavioral outcomes are revealed. Split-half reliability is not enough; absolute retention and null-separated certification thresholds are required.

## 8. Minimal decisive follow-up experiments

The next experiments should be small enough for a cheaper execution model to run from a frozen specification. They should aim to falsify a missing link, not launch a broad optimization sweep.

### Run A: fixed-object identity positive control

**Question:** Can the identity pipeline certify an object whose coordinates are known not to change?

- Freeze one model, one weight matrix, one checkpoint, and one deterministic data slice.
- Run the estimator twice with independent sampling noise but no transport across layers or checkpoints.
- Include an explicit identical-object positive control, an orthogonally rotated negative control, and a random-pair null.
- Reveal behavioral outcomes only if both halves certify the positive control at the preregistered level and reject the negative controls.
- **Stop condition:** if the fixed-object positive control fails, repair the estimator; do not run a larger lineage experiment.

### Run B: fresh held-out symbolic adapter comparison

**Question:** Does VPD-guided data or module selection beat ordinary and random selection after accounting for scaffold damage?

- Use the prepared balanced 400-card evaluation, with 80 cards per operator family and balanced A/B labels.
- Freeze six arms: raw base, zero scaffold, naive adapter, VPD-policy adapter, shuffled-policy adapter, and matched-random adapter.
- Keep rank, train steps, examples, optimizer, quantization, and module count identical across adapter arms.
- Analyze operator families and seeds as grouped units.
- **Acceptance:** VPD must beat raw base and every adapter control on the frozen primary metric, with no preregistered regression breach.
- **Stop condition:** no hyperparameter rescue on the same held-out pack.

### Run C: HRM-Text-1B VPD-informed RL training policy

**Question:** Can VPD improve a learned policy over training experience and update retention when it is denied direct control over edit coordinates?

- First reproduce the trained HRM base with native fused modules and pass a zero-missing-key, zero-new-initialization, fixed-logit conformance test.
- Train and freeze a fresh finite full-coverage ComponentModel VPD. Preserve the recovered May run as historical evidence, but quarantine its non-finite terminal checkpoints and do not relabel the existing `low_rank_svd` map as trained VPD.
- Hold adapter architecture, target modules, rank, optimizer, update size, tokens, and evaluation budget fixed across arms.
- Train equal-capacity policies over macro-actions: choose a curriculum tranche, train or abstain, then retain or roll back.
- Compare true-VPD observations with behavior-only, within-matrix shuffled VPD, and energy/rank/module-matched random component maps. A random-action policy and uniform curriculum are weaker secondary baselines.
- Derive reward from post-update development behavior minus regression, KL, and compute penalties. Never reward high VPD score directly.
- Gate the sequential six-step experiment on a cheaper one-step action-selection test.
- Evaluate frozen policies once on a newly generated, hash-sealed held-out pack.
- **Acceptance:** true-VPD must beat the best matched policy control with a positive 95% grouped confidence bound, exceed raw base, avoid every regression gate, and show gains across at least four of five task families and 75% of seed blocks.
- **RSI boundary:** a final gain demonstrates bounded policy-guided training. Evidence for recursive improvement additionally requires controlled gains not to collapse over successive retained macro-steps.

### Run D: patch governance replication

**Question:** Can decomposition reliably remove regression from dirty diffs without claiming capability generation?

- Preselect multiple dirty LoRA diffs before decomposition.
- Compare VPD extraction with magnitude pruning, low-rank SVD, and same-diff random extraction at matched sparsity and norm.
- Use separate target-retention and off-target-regression packs.
- Treat each dirty diff as the independent unit.
- **Acceptance:** consistent regression reduction at matched target retention across diffs, not a win on one cached family.

These runs deliberately exclude autonomous checkpoint mutation. Candidate artifacts should remain reversible, and no result should be promoted without an immutable raw checkpoint, a complete receipt, and an explicit human release decision.

## 9. Limitations

The evidence comes from heterogeneous models, proxies, and evaluation surfaces accumulated during a fast-moving research program. Some branches used SVD or rank-one atoms as decomposition proxies; those results cannot be silently upgraded into claims about trained VPD. Several behavioral tests were small or cached. The strongest positive graft signal was sparse and clustered. The Qwen identity experiment stopped before intervention, so it establishes a coordinate-identity failure under its registered construction rather than a behavioral null.

Most importantly, the completed HRM-Text-1B program did not run the policy-level experiment specified in Section 3.1. It did attempt a full 16-chunk ComponentModel decomposition, but the only recovered production endpoints are non-finite; moreover, the historical loader demonstrably mismatched split modules to a fused checkpoint. A reconstructed native-fused loader passes local base conformance, but no fresh ComponentModel VPD has yet been trained with it. The earlier feedback loop was reinforcement-like rather than a grounded learned RL policy, and later training experiments primarily used decomposition-derived labels to target LoRA/SFT. Until a fresh finite VPD and the proposed matched-policy experiment are executed on a fresh held-out pack, this paper can claim limits to VPD-guided micro-editing and direct adapter targeting, but not a decisive limit to all mechanistic-interpretability-informed training policy.

The program also did not test every plausible path from interpretability to improvement. Better feature dictionaries, nonlinear components, activation interventions, data curation, verifier design, or joint representation alignment could cross gates that these methods failed. Larger hardware could make a numerically guarded, native-fused VPD rerun practical, but hardware alone does not repair checkpoint-loader or identifiability failures. Conversely, greater scale could worsen identifiability and make matched controls more important.

Finally, the RSI framing extrapolates from component experiments to a multi-cycle process. No open-ended autonomous RSI system was built or run. The empirical conclusion concerns necessary precursors: when identity, controlled causal gain, generalization, and recurrence fail, these experiments do not provide a credible ignition mechanism.

## 10. Conclusion

The two-month program found something more useful than either a sweeping success or an empty null. It found a boundary.

Mechanistic decomposition can reveal stable local structure in tiny recursive models. It can identify sensitive regions, produce reproducible ablation rankings, generate candidate interventions, and in at least one case remove measured regression from a learned diff while preserving target movement. Those are substantive capabilities for diagnosis and governance.

But local interpretability did not become reliable recursive improvement through the interventions actually tested. The selected directions failed to transfer from proxy to task, failed to beat appropriately matched random and shuffled controls, failed to overcome adapter-scaffold confounds, or failed to retain identity across the contexts needed for persistent editing. The few positive transfers were too sparse and uneven to establish recurrence.

There was not enough effective deuterium to detonate the ocean. More precisely: the tested system lacked the combination of stable fuel, confinement, net energy, and repeatability needed for propagation. That negative result should reduce confidence in this particular cheap-RSI pathway while increasing caution about granting automated weight-edit authority to mechanistic signals that have not passed the same gates.

The safest and best-supported use of VPD in the completed evidence is not autonomous self-modification. It is disciplined hypothesis generation, diagnosis, candidate triage, regression analysis, and the construction of experiments capable of proving the method wrong. The next decisive test is no longer another micro-edit: it is whether the same mechanistic evidence adds information to a bounded RL training policy after behavior-only, shuffled-mechanism, and matched-random policies receive equal budgets.

## Appendix A. Evidence provenance

### A.1 Intact synthesis sources

- `D:\projects\VDP\papers\VPD_TRM_Feature_Steering\vpd_trm_feature_steering_paper.md`
  - Title: *VPD-Abetted Feature Steering in Tiny Recursive Models*
  - Size: 132,336 bytes
  - SHA-256: `0BCFA35BF6F4599CE2B521EE5655BDA6370A86CEE8529CEF9701F7F9DD01DC6B`
- `D:\projects\VDP\silico_reports\plans\reports\2026-07-12_two_month_research_summary.md`
  - Research window: 2026-05-12 through 2026-07-12
  - Size: 18,704 bytes
  - SHA-256: `22E199A3787195ECD60B28D891F020A4F2DFBC58C1E736356B1E654B1A6E3AD5`

### A.2 Principal result reports

- `D:\projects\VDP\reports\hermes_logic_critic_vpd_stability_report.md`
- `D:\projects\VDP\docs\hrm_vpd_rank_stability_report.md`
- `D:\projects\VDP\reports\meso_patch_negative_result_note.md`
- `D:\projects\VDP\reports\hrm_symbolic_operator_negative_roundout.md`
- `D:\projects\VDP\reports\2026-07-02_research_note_B_what_doesnt_work.md`
- `D:\projects\VDP\JSpace\reports\tiny_vpd_j_experiment_report.md`
- `D:\projects\VDP\JSpace\reports\topology_degeneracy_control.md`
- `D:\projects\VDP\JSpace\reports\vpd_identity_calibration_run_v1_noise_floor.md`

### A.3 Independent recovery evidence

An independent disk-recovery set exists at `E:\PC_Rescue_20260723\GitPreservation\VPD\`. Its `VPD.bundle` and `untracked.tar` preserve repository and untracked-file evidence separately. This draft uses the intact `D:\projects\VDP` files as its direct sources; the E-drive artifacts establish an independent recovery path rather than a second experimental result.

## Appendix B. External references

1. E. J. Konopinski, C. Marvin, and E. Teller. *Ignition of the Atmosphere with Nuclear Bombs*. Los Alamos report LA-602, 1946. Archived report: https://sgp.fas.org/othergov/doe/lanl/docs1/00329010.pdf
2. L. Bushnaq et al. *Interpreting Language Model Parameters*. Goodfire, 2026. https://www.goodfire.com/research/interpreting-lm-parameters
3. G. Wang et al. *Hierarchical Reasoning Model*. arXiv:2506.21734, 2025. https://arxiv.org/abs/2506.21734
4. A. Jolicoeur-Martineau. *Less is More: Recursive Reasoning with Tiny Networks*. arXiv:2510.04871, 2025. https://arxiv.org/abs/2510.04871
