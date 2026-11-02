# HRM-Text-1B VPD-Informed RL Training-Policy Protocol

Prospective companion: [cross-research utility addendum](VPD_CROSS_RESEARCH_UTILITY_ADDENDUM_V1.md). It specifies a bounded Stage-2 tranche-selection comparison and distinguishes regret to a fixed reachable candidate from this protocol's per-seed hindsight maximum. Existing authenticity, training, and staged execution gates remain binding.

**Protocol version:** 1.1-draft (provenance/status clarification; scientific gates unchanged)  
**Status:** proposed; local base conformance passed; no fresh VPD or policy training launched  
**Primary purpose:** distinguish VPD-informed training policy from VPD-targeted micro-editing  
**Model:** HRM-Text-1B (exact checkpoint hash required before execution)

## 1. Research question

Can a reinforcement-learning controller that observes frozen VPD measurements choose **which training experience to present and when to accept or roll back a broad model update** better than equally resourced controllers using behavioral observations, shuffled VPD measurements, or matched-random component maps?

This is the missing experiment behind the paper *Not Enough Deuterium to Detonate the Ocean*. Previous HRM-Text-1B work primarily used mechanistic labels as direct adapter targets or used local edits and routing rules. The earlier “Boltzmann-ish” loop was not grounded RL: it lacked a learned policy with a reliable action-to-outcome credit path. This protocol tests a stronger and more defensible hypothesis.

## 2. Claim boundary

A positive result would support only this claim:

> On the declared HRM-Text-1B task distribution, a frozen VPD observation channel improves the sample efficiency or final held-out performance of a bounded training-policy controller beyond matched non-mechanistic policies.

It would not establish open-ended recursive self-improvement, general transfer, semantic component identity, or safe autonomous weight mutation. A negative result would reject this versioned policy design, not all possible mechanistic-interpretability-informed RL.

## 3. Essential separation from micro-editing

The experiment must hold the model-update mechanism constant across policy arms.

- VPD does **not** choose individual weight vectors, signs, norms, LoRA target modules, or rank-one edits.
- Every learned arm uses the same adapter architecture, target-module set, rank, optimizer, learning rate, quantization path, and maximum optimizer steps.
- The policy chooses only macro-actions: the next training-data tranche, train/abstain, and retain/rollback.
- VPD is therefore an observation channel for allocating learning, not an edit coordinate.

If any arm receives a different module set or update norm because of VPD, that is a different experiment and must not be reported as this protocol.

## 4. Gate 0: artifact authenticity and positive controls

A genuine full-coverage HRM ComponentModel VPD attempt exists at `D:\Research_Engine\runs\hrm_text_vpd_chunked_overnight_v2`. It processed all 16 paired L/H layer indices and is distinct from the later `low_rank_svd` refinement. The recovered terminal checkpoints are nevertheless ineligible for this protocol: every chunk's step-20 loss is non-finite, and direct checkpoint inspection finds all learned component and CI-function elements non-finite. The D and E copies of the inspected layer-14 checkpoint have the same SHA-256, so the recovery mirror does not supply an earlier finite endpoint. Direct reconstruction resolves the loader issue adversely: the pinned May code expects split attention and MLP matrices, while the recovered trained checkpoint contains native fused matrices, producing unused fused weights and newly initialized split replacements. A separately labeled reconstructed fused loader passes the local base-conformance gate but does not retroactively validate the historical decomposition. See `HRM_TEXT_1B_FULL_VPD_RECOVERY_AUDIT_V1.md` and `HRM_TEXT_1B_LOADER_CONFORMANCE_AND_PRIMELAB_RUN_REPORT_V1.md`.

The earlier finite one-step layer-0 smoke checkpoint may be used only as a parser and plumbing fixture. It is not the full trained VPD. The later HRM low-rank refinement remains recorded as `low_rank_svd`; it may select candidate matrices for decomposition, but it must not be relabeled as trained VPD.

Before policy training:

1. Hash and freeze the HRM-Text-1B checkpoint, tokenizer, model code, calibration data, training pool, development reward pack, regression pack, and final held-out pack.
2. Prove native-base conformance before wrapping: zero missing, unexpected, or newly initialized tensors; exact tokenization; and deterministic fp32 logit maximum absolute error no greater than 1e-5 across a fixed 32-prompt pack.
3. Target the HRM's native fused `gqkv_proj`, `o_proj`, `gate_up_proj`, and `down_proj` modules. Do not split fused weights unless a separately tested conversion is algebraically and behaviorally equivalent.
4. Produce fresh, finite trained VPD components using the declared ComponentModel implementation. Save steps 0, 1, 2, 5, 10, 20, 50, 100, and regular later checkpoints; fail closed on any non-finite loss, gradient, parameter, or optimizer state.
5. Verify reconstruction, ablation faithfulness, determinism tolerance, cross-half importance stability, and component-count metadata.
6. Run a planted-component positive control and an energy/rank/module-matched random negative control on the observation pipeline.

**Gate 0 stop:** do not train a policy if base conformance fails, if any learned VPD tensor is non-finite, if the VPD artifact is missing its training receipt, if reconstruction/ablation checks fail, or if the pipeline cannot distinguish the fixed positive control from matched random.

## 5. Environment

One episode starts from the same immutable base checkpoint and zero-initialized adapter scaffold. It contains at most six macro-steps.

The proposed execution surface is the existing Prime Lab workspace at `D:\Research_Engine\prime_lab`, using the machine-readable handoff in `HRM_TEXT_1B_PRIMELAB_EXECUTION_PACKET_V2.json`. Local base conformance has passed; Prime replay, VPD training, and observation-cache generation require a custom GPU environment that can inspect arbitrary HRM tensors. A hosted RL interface may be used only after frozen observation summaries exist and only if it preserves identical arm schemas and the full receipt contract. The existing `vdp_int3_logic` environment is a packaging reference; its late-attention proxy is not genuine VPD and cannot score this experiment.

At macro-step `t`:

1. The controller receives its permitted state representation.
2. It selects one training tranche or `ABSTAIN`.
3. The trainer performs a fixed number of optimizer steps using the common adapter recipe.
4. The candidate checkpoint is scored on the rotating development-reward slice and the fixed regression slice.
5. The controller selects `RETAIN` or `ROLLBACK`.
6. The resulting checkpoint and observation summary become the next state.

The final held-out pack remains unavailable to the controller, reward function, checkpoint selector, and human analyst until the preregistered run completes.

## 6. State representations

All numeric inputs are normalized using training/calibration data only. Dimensionality and policy parameter count must be matched across learned arms.

### 6.1 Shared behavioral state

- accuracy and target margin by development family;
- calibration error and abstention rate;
- per-family loss slope over prior macro-steps;
- adapter norm, gradient norm, and KL divergence from the raw base;
- regression-pack loss and accuracy deltas;
- remaining optimizer-step and evaluation budgets;
- prior action and retain/rollback decision.

### 6.2 VPD observation channel

For each candidate training tranche, aggregate without exposing labels from the reward or final packs:

- causal-importance mass on frozen VPD components;
- concentration/entropy of component use;
- signed ablation effect summaries;
- overlap with component profiles of current failure families;
- component stability and reconstruction-confidence flags;
- distribution shift from the VPD calibration set;
- high-sensitivity depth-region mass, reported separately rather than collapsed.

The controller consumes summaries, not component identities as direct edit instructions.

### 6.3 Matched control channels

- **Shuffled VPD:** permute component-to-example associations within matrix and difficulty strata while preserving marginal distributions. Freeze each permutation before learning.
- **Matched random map:** random orthonormal directions matched by matrix, rank, component count, Frobenius energy, and calibration exposure; run the same ablations and aggregation code.
- **Behavior-only padding:** replace VPD dimensions with noise sampled from the VPD channel's training-set covariance so the policy network has identical input width.

## 7. Macro-action space

Training examples are partitioned before policy learning by task family, difficulty, and failure signature. A macro-action chooses one of the following fixed-size tranches:

- uniform balanced tranche;
- one tranche for each of the five symbolic operator families: consent, repair, deception, proportionality, and replay/memory;
- near-boundary/high-disagreement tranche;
- high-loss tranche;
- high-VPD-concentration tranche;
- low-VPD-concentration counterfactual tranche;
- distribution-shift recovery tranche;
- `ABSTAIN`.

VPD-derived tranche names are present in every arm. Non-VPD arms receive the same rows and action IDs; they simply lack the true VPD observation channel. This prevents the VPD arm from receiving a larger action space.

Every non-abstain action trains on the same number of tokens for the same number of optimizer steps. The controller cannot change learning rate, rank, module set, batch size, or update norm.

## 8. Reward and credit assignment

The policy receives reward from model outcomes after each macro-update, never from VPD scores themselves.

Let:

`R_t = ΔTask_t - λ_reg * Regression_t - λ_kl * KL_t - λ_cost * Cost_t + λ_eff * Efficiency_t`

where:

- `ΔTask_t` is the grouped change in frozen development target margin, clipped at preregistered bounds;
- `Regression_t` is the worst preregistered family-level regression beyond tolerance;
- `KL_t` is divergence from the raw base on a neutral corpus;
- `Cost_t` is normalized optimizer-token and evaluation cost;
- `Efficiency_t` rewards equal task gain with fewer training examples or abstentions.

The coefficients are selected on a synthetic positive-control environment, then frozen. No coefficient is tuned on the final held-out pack.

Hard overrides:

- Any integrity failure gives terminal reward `-1` and invalidates the checkpoint.
- Any regression beyond the hard safety threshold forces `ROLLBACK`.
- `ABSTAIN` receives a small cost but is preferable to a damaging update.

The primary controller is a small discrete-action PPO policy with generalized advantage estimation. The controller, value model, optimizer, horizon, entropy coefficient, and parameter count are identical across learned arms. If PPO is unstable in the synthetic positive control, use REINFORCE with a learned baseline and version the protocol before any real-model outcomes are inspected.

## 9. Experimental arms

Run all arms from identical checkpoint/scaffold/seed blocks.

| Arm | Policy input | Purpose |
|---|---|---|
| Raw base | none; no adapter | Absolute behavioral baseline |
| Zero scaffold | no training | Measures PEFT/quantization path damage |
| Uniform curriculum | fixed schedule | Non-RL training baseline |
| Random-action RL | shared state; random legal action | Search-opportunity baseline |
| Behavior-only RL | shared behavioral state | Tests whether ordinary feedback is sufficient |
| Shuffled-VPD RL | behavioral + frozen shuffled VPD | Controls marginal statistics and policy capacity |
| Random-map RL | behavioral + matched random components | Controls decomposition/search machinery |
| True-VPD RL | behavioral + frozen genuine VPD | Experimental arm |
| Oracle-family policy | true failure family only; development | Positive control for environment learnability; excluded from primary comparison |

The main comparison is True-VPD RL against the **best** of behavior-only, shuffled-VPD, and random-map RL under the same seed blocks and compute budget. Beating only random-action RL is insufficient.

## 10. Staged execution

### Stage 1: plumbing and falsification smoke

- One seed; two macro-steps; synthetic reward only.
- Verify deterministic resets, data isolation, reward signs, rollback, raw/zero distinction, receipts, and control-channel matching.
- No scientific claim.

### Stage 2: one-step policy-information test

- Each policy selects exactly one tranche from the base state.
- Minimum four independent seed blocks.
- Tests whether the VPD channel improves action selection before sequential credit assignment is introduced.
- **Stop:** if True-VPD does not beat behavior-only and both matched mechanistic nulls on development reward, do not run sequential training.

### Stage 3: bounded recursive policy test

- Six macro-steps per episode.
- Minimum eight independent seed blocks, adjusted upward by a preregistered power calculation based only on Stage 2 variance.
- Each arm receives equal maximum tokens, optimizer steps, evaluation calls, and wall-clock cap.
- Policies are frozen before final held-out evaluation.

### Stage 4: fresh held-out confirmation

- Evaluate every retained final checkpoint once on the sealed pack.
- Report every arm and seed, including failures and rollbacks.
- Do not tune or rerun after reveal. Any new attempt requires a new pack and protocol version.

## 11. Data splits

Use disjoint generation seeds and deduplicate normalized content across:

1. VPD calibration data;
2. controller training pool;
3. development-reward pack;
4. regression pack;
5. final held-out pack.

The previously prepared balanced 400-card symbolic evaluation may be used as development evidence only if its rows have already been inspected during prior research. The final claim requires a newly generated, hash-sealed pack with balanced labels and equal representation of the five operator families.

## 12. Primary estimand and acceptance gate

For seed block `i`, define:

`D_i = HeldoutScore(TrueVPD_i) - max[HeldoutScore(BehaviorOnly_i), HeldoutScore(ShuffledVPD_i), HeldoutScore(RandomMap_i)]`

The primary endpoint is the grouped mean of `D_i` over seed blocks, with a hierarchical bootstrap over seeds and task families.

All conditions must hold:

1. lower bound of the 95% confidence interval for mean `D_i` is greater than zero;
2. True-VPD exceeds raw base on the frozen primary behavioral score;
3. True-VPD does not breach any family-level regression threshold;
4. gain is present in at least four of five operator families and at least 75% of seed blocks;
5. True-VPD's advantage is not explained by more retained updates, training tokens, evaluation calls, or wall-clock time;
6. a leave-one-family-out analysis preserves the sign of the advantage;
7. the result survives comparison with zero scaffold, not merely other trained adapters.

Failure of any condition means the policy-level claim is unsupported.

## 13. Secondary RSI-relevance measures

These are descriptive unless separately powered:

- reward per macro-step;
- probability of beneficial update selection;
- rollback rate;
- cumulative gain curve and whether increments decay with step number;
- policy entropy and collapse to one curriculum family;
- component-observation mutual information with chosen actions;
- gain per training token and per evaluation call;
- transfer to one frozen out-of-family reasoning suite.

The strongest RSI-relevant signature would be non-declining controlled gain over successive retained steps. A single final improvement is policy-guided training, not evidence of recursive acceleration.

## 14. Receipts and artifacts

Each run must write:

- protocol and code hashes;
- checkpoint, tokenizer, and dataset hashes;
- a base-conformance receipt with missing, unexpected, and newly initialized key counts plus fixed-probe logit differences;
- exact VPD artifact provenance and proof it is trained VPD rather than SVD;
- full finiteness coverage for model, components, CI functions, gradients, and optimizer state at every saved checkpoint;
- arm, seed, episode, macro-step, state, action probabilities, sampled action, reward terms, retain/rollback decision;
- model and adapter hashes before and after every step;
- resource use and termination reason;
- raw per-example development, regression, and final scores;
- control-map generation seeds and matching diagnostics;
- immutable reveal receipt for the final pack.

No result should be summarized without the complete arm-by-seed table.

## 15. Safety and authorization boundary

- No additional external paid inference or remote GPU launch is authorized by this protocol itself. Each run requires a separate authorization receipt naming its run ID.
- No existing base weights are overwritten. All model changes are reversible adapters or copied checkpoints.
- The controller cannot execute external actions, change its evaluator, generate its own final test, increase its budget, or promote a checkpoint.
- Final held-out reveal and any checkpoint release require a human decision.
- A successful bounded policy experiment is not permission to extend the horizon or add autonomous tool access.

## 16. Paper decision table

| Outcome | Paper interpretation |
|---|---|
| Gate 0 fails | The experiment cannot test VPD-informed policy; report artifact/authenticity failure |
| Stage 2 fails | VPD observations do not improve even one-step policy selection under this design; stop cheaply |
| Stage 2 passes, Stage 3 fails | VPD contains local allocation signal but does not survive sequential learning |
| Stage 3 passes, held-out fails | Controller overfit the development reward; no policy-level claim |
| Held-out passes, increments decay | VPD improves bounded training policy, but not recursive acceleration |
| Held-out passes with stable multi-step increments | Evidence for bounded mechanistic-interpretability-fueled recursive improvement; replication still required |

## 17. Existing evidence used to motivate, not score, this protocol

- `D:\projects\VDP\docs\hrm_vpd_trm_organization.md`
- `D:\projects\VDP\docs\hrm_vpd_rank_stability_report.md`
- `D:\projects\VDP\reports\2026-07-02_research_note_B_what_doesnt_work.md`
- `D:\projects\VDP\reports\hrm_symbolic_operator_negative_roundout.md`
- `D:\projects\VDP\silico_reports\plans\reports\2026-07-12_two_month_research_summary.md`
