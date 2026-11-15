# Causal Dark Fibers: Observable Quotients, History-Conditioned Caches, and a Cross-Regime ARG-03 Protocol

**Status:** Theoretical rationale and pre-run protocol design note, 15 November 2026. The main `v014` manuscript ([`PAPER.md`](PAPER.md)) is unchanged. Verified local GPU campaign results executing §§6–7 (`LOOPSCOPE-04` across both frozen `TRM` checkpoints and `ARG-03-N` on `Nanbeige4.2-3B` `development.jsonl` $N=96$, with `test.jsonl` $N=168$ kept strictly sealed) are reported separately in [`DARK_FIBERS_CAMPAIGN_ADDENDUM.md`](DARK_FIBERS_CAMPAIGN_ADDENDUM.md), and the frozen held-out evaluation protocol is locked in [`ARG03_FROZEN_TEST_PROTOCOL.md`](ARG03_FROZEN_TEST_PROTOCOL.md).

**Companion:** From Shared Weights to Head-Orthogonal Fibers, papers/shared_weights_dark_fibers/PAPER.md (v014). This note extends its Regimes I–III using the already reported ARG-02 Nanbeige experiment and LOOPSCOPE-03. References to reported measurements are to that manuscript and its committed receipts, not to an independent re-execution.

## 1. Empirical constraints that a deeper theory must preserve

| Evidence | Recorded result | Interpretation allowed |
| --- | --- | --- |
| ARG-02 Nanbeige, held-out future@12 on 168 positive programs | 1/2/4/8 loops: 44/65/76/71; dev-selected depth 8 | Native trained recurrence improves over a single pass. Improvement from loop 2 to loop 8 is +6, McNemar p=0.61. |
| ARG-02 Nanbeige, causal controls at loop 8 | Open-loop 55 (vs. 71, p=0.097); 8-pass single-loop ensemble 51; donor-shuffle 62 | The registered strong feedback-mechanism gate H2a is unmet. The shuffle comparison supports some case-specific recurrence by loop 8. |
| ARG-02 Nanbeige, loop 2 donor displacement | Clean 65 vs. shuffled 69, six differing outcomes | Much of the early displacement is transferable between cases; this is a result about the measured arm, not proof of a universal case-independent update. |
| LOOPSCOPE-01 LoopChase lambda=0.1 | 3/3 seeds maintain 1.000 OOD depth 9–16; measured R-delay 0 | A task admits a visible-plus-cache solution. |
| LOOPSCOPE-01 LoopChase lambda=1.0 | 2/3 seeds show residual delays +3.06, +5.46 loops | Dense intermediate supervision can coexist with causally relevant readout-invisible state. |
| LOOPSCOPE-02 TRM | Raw 501-d head-orthogonal projection destroys propagation solves too; norm-calibrated replacements preserve propagation solves | Uncalibrated erasure mixes manifold damage with causal dependence on orthogonal state. |
| LOOPSCOPE-03 TRM, stalled boards | Data-scaled C5 R² 0.915 primary, 0.837 replication; m=2 refill restores primary; on replication C5 improves path independence and s=64 solve | Refill error and useful/harmful trajectory effects must be distinguished. The effect of a residual is checkpoint-dependent. |
| LOOPSCOPE-03 TRM, causal matched-step swaps | Residual donor swaps steer next-step commitments above Zero_R; strength varies by checkpoint | A causal transition-history effect is present under the tested interventions, especially on stalled search; exact hidden-state sufficiency is not established. |

The original three regimes describe observed behavior. They should not be equated with three mutually exclusive mathematical classes. Shared-parameter stationarity, nonlinear functional role switching, quotient closure, finite-horizon observability, and attractor stability can vary independently.

## 2. Exact observable quotients

Let \(\mathcal S_x\) be an invariant reachable state set for fixed input \(x\), with recurrent transition \(F_x:\mathcal S_x\to\mathcal S_x\). Let \(q:\mathcal S_x\to\mathcal V\) be the *actual* per-step readout, including normalization, head, and the relevant token/candidate contexts. Denote \(v_t=q(s_t)\).

**Definition 1 (observational fiber).** At visible state \(v\), the fiber is
\[
\mathcal F_{x,v}=\{s\in\mathcal S_x:q(s)=v\}.
\]
This is a level set of the full readout. The linear space \(\ker W_{\rm head}\) is a special case only when \(q(s)=W_{\rm head}s\) is linear on the intervened coordinates.

**Proposition 1 (exact quotient / lumpability).** There exists \(T_x:q(\mathcal S_x)\to q(\mathcal S_x)\) such that
\[
q\circ F_x=T_x\circ q\quad\text{on }\mathcal S_x
\]
if and only if \(q(F_x(s))=q(F_x(s'))\) whenever \(q(s)=q(s')\) for \(s,s'\in\mathcal S_x\).

*Proof.* Under the constant-on-fibers condition, define \(T_x(v)=q(F_x(s))\) using any \(s\) with \(q(s)=v\). This is well-defined. Conversely, the displayed factorization gives \(q(F_x(s))=T_x(v)=q(F_x(s'))\) for states in the same fiber. QED.

**Proposition 2 (visible-only prediction obstruction).** If a matched-fiber pair obeys \(q(s)=q(s')=v\) but
\[
\Delta=\|q(F_x(s))-q(F_x(s'))\|>0,
\]
then every deterministic predictor \(g(x,v)\) of the next visible state incurs error at least \(\Delta/2\) for one member of that pair, measured in the same norm.

*Proof.* The triangle inequality bounds the distance between the two actual successor readouts by the sum of their distances to the shared prediction. QED.

This provides an interventionally falsifiable definition of visible-state insufficiency. A hidden coordinate may have large norm and remain in \(\ker W_{\rm head}\) while failing to create any such matched-fiber successor difference. A small hidden displacement can be sufficient when it changes the successor.

**Proposition 3 (approximate quotient bound).** Assume an invariant reachable set \(\mathcal S_x\), an \(L\)-Lipschitz candidate quotient \(T_x\), and a one-step commutation bound
\[
\sup_{s\in\mathcal S_x}\|q(F_x(s))-T_x(q(s))\|\le\epsilon.
\]
Then every reachable trajectory satisfies
\[
\|q(F_x^k(s))-T_x^k(q(s))\|
 \le\epsilon\sum_{j=0}^{k-1}L^j.
\]

*Proof.* Apply the one-step bound at \(F_x^{k-1}(s)\), then use \(L\)-Lipschitzness to obtain \(e_k\le\epsilon+Le_{k-1}\), starting from \(e_0=0\). Induction gives the geometric sum. QED.

At \(L<1\), visible prediction error remains bounded by \(\epsilon/(1-L)\). At \(L=1\), it can accumulate linearly in depth. For \(L>1\), an apparently small one-step discrepancy can become important at later loops. This supplies a mathematical bridge from the manuscript's Q11 commuting-square diagnostics to held-out depth extrapolation. Its assumptions must be checked on a reachable tube that contains the continued trajectory.

## 3. Readout geometry and delayed visibility

For a differentiable readout \(q\), the instantaneous invisible *tangent directions* at state \(s\) form
\[
K_s=\ker Dq(s).
\]
A nonlinear normalization makes these state-dependent. For \(q(s)=W\,\mathrm{RMSNorm}(s)\), positive radial rescaling is an exact readout symmetry wherever the normalization is scale-invariant; consequently \(s\in K_s\). A fixed projection onto \(\ker W\) at the pre-normalization state is generally insufficient to preserve readout equality. The TRM experiments act at a post-RMSNorm carrier and already provide sphere-calibrated controls, while a new Nanbeige experiment must impose the actual normalized-logit equality.

Let \(A_j(s)=D(q\circ F_x^j)(s)\) and \(P_{K_s}\) be an orthonormal basis projector onto \(K_s\). A finite-horizon local dark-to-visible observability Gramian is
\[
G_{t,k}(s)=P_{K_s}^{\top}\left(\sum_{j=1}^{k}A_j(s)^{\top}A_j(s)\right)P_{K_s}.
\]
Its spectrum measures **first-order** visibility of directions invisible at step \(t\) after up to \(k\) further recurrent updates. The delayed-visible rank is \(\operatorname{rank}_{\epsilon}G_{t,k}\), with threshold and norm convention preregistered. A nonzero eigenvalue predicts a local direction capable of changing a future readout; the exact finite-displacement matched-fiber experiment remains the causal test because first-order derivatives can miss nonlinear effects or overstate effects outside the local tube.

This Gramian gives Transported Component Attribution (TCA) a state-space counterpart. TCA diagnoses parameter-direction specificity after correcting for transported gradients. The restricted Gramian diagnoses readout-blind *state* directions that later become visible. Both should use equal-norm and equal-logit controls, with finite perturbation-radius sweeps, to separate tangent amplification from nonlinearly contracting basins.

## 4. Conditional cache is relative to an observer

LOOPSCOPE-03 fits \(C_t=\hat H_\theta(x,V_t)\) and defines the measured residual \(R_t=H_t-\hat H_\theta(x,V_t)\). This residual combines trajectory information and prediction error. It may also contain information already available in earlier visible outputs, which a monitor of the *full trace* could recover.

Define observer information sets:
\[
\mathcal O_t^{\rm now}=\sigma(x,t,V_t),\qquad
\mathcal O_t^{\rm trace}=\sigma(x,t,V_0,\ldots,V_t).
\]
For an actual randomized initial-state/fork distribution, the ideal squared-error conditional caches are
\[
C_t^{\rm now}=\mathbb E[H_t\mid\mathcal O_t^{\rm now}],\qquad
C_t^{\rm trace}=\mathbb E[H_t\mid\mathcal O_t^{\rm trace}].
\]
The residual \(H_t-C_t^{\rm trace}\) is the innovation inaccessible even from the visible trace under that specified distribution. Real neural refill models approximate these expectations within a bounded hypothesis class. Their test-set errors cannot by themselves establish irreducible darkness.

**Proposition 4 (information monotonicity).** Since \(\mathcal O_t^{\rm now}\subseteq\mathcal O_t^{\rm trace}\),
\[
\mathbb E\|H_t-C_t^{\rm trace}\|^2
\le
\mathbb E\|H_t-C_t^{\rm now}\|^2.
\]
*Proof.* Conditional expectation is orthogonal projection in \(L^2\); projection onto the larger closed measurable subspace cannot increase squared error. QED.

**Important deterministic limit.** For a deterministic model with a fixed initializer and complete input \(x\), \(H_t\) is a deterministic function of \((x,t)\). An unrestricted reconstructor could recompute all \(t\) loops from the input. Accordingly, irreducible conditional uncertainty vanishes in this limit despite substantial internal computation. Claims of *inaccessible reasoning* therefore require an explicit observer budget, partial input access, randomized forks, or an intervention-defined notion of causal dependence. The intended quantity for compact-model research is often **resource-bounded reconstruction cost** rather than unconditional information-theoretic entropy.

For Sudoku with a unique target board, \(I(H_t;Y^*\mid x)=0\) when the target is a deterministic function of \(x\); it cannot measure latent reasoning. Future visible transitions, time-to-lock, bifurcation of fork outcomes, or responses to controlled perturbations are appropriate targets.

A practical four-part accounting refines the paper's three ledgers:
1. Current visible information \(V_t\).
2. Cache reconstructible from \((x,t,V_t)\) within a fixed auditor budget.
3. Additional history recoverable from \(V_{0:t}\).
4. The remaining trace-inaccessible or budget-inaccessible innovation, measured with held-out fork distributions and paired causal interventions.

The report should preserve both the paper's existing \(V\oplus C\oplus R\) empirical convention and this more precise *observer-relative* interpretation.

## 5. Time-homogeneity and functional multiplexing

A shared block may implement a time-homogeneous map \(F\) even if its activation distribution changes radically from pass 1 to pass 2. The mean SVD input-usage correlation \(\bar r=0.355\) for Nanbeige is valuable evidence of *functional role switching* at different input states. It is not a mathematical test that the update map depends explicitly on loop index.

Before comparing stationarity across model families, define the complete carried state, including any attention cache, position information, normalization carrier, injected embeddings, or explicit iteration embeddings. Test deterministic replay from the same complete state at distinct times. If the same \(F\) is applied, the system is time-homogeneous on that state space; if externally changing inputs are omitted from the state, include them or state the forcing explicitly.

Use separate descriptive axes rather than exclusive class membership:

| Axis | Instrument | What it tests |
| --- | --- | --- |
| Functional role switching | SVD usage / activation-matched TCA | Whether the shared weights implement different effective operations on visited states |
| Quotient closure | One-step commuting-square error; matched-fiber successor divergence | Whether the visible state suffices to predict future visible transitions |
| Trajectory dependence | Equal-readout fork perturbations; single-transplant Resume; unconditional fork agreement | Whether different histories affect continuations and at what cost |
| Basin geometry | Local Gramian, finite kicks, cycle tests, depth curves | Stability, transient growth, and asymptotic behavior |

Nanbeige currently has strong role-switching evidence and a weak registered beyond-native-depth benefit; its exact quotient closure and matched-fiber causal observability remain unmeasured. TRM has extensive manifold-calibrated causal probes and a two-checkpoint contrast. These facts need no reclassification of existing runs.

## 6. Prospective ARG-03-N: Nanbeige dark-fiber experiment

**Scope:** A new registered campaign. Do not reuse ARG-02 test rows for confirmatory significance. Preserve the ARG-02 protocol, receipts and negative verdict. Fit all controls on a new development partition and score once on a fresh generator-/program-family-disjoint held-out test.

- **Locked model and depth:** Nanbeige4.2-3B at ARG-02 revision 451ed48c, same implementation and tokenizer, with candidate depths 1/2/4/8; native depth 2 is the principal anchor. Compare 2 vs. 4/8 as exploratory unless a newly registered split selects one.
- **Actual readout:** Score the complete 64-program candidate ranking, including all model contexts/prefix positions needed for joint log probabilities. Monitor all relevant token positions. At each proposed donor patch assert equality of the recipient's *complete pre-intervention readout* under the adapter's real final norm and head; stop if it fails tolerance.
- **No-feedback calibration:** On development only, fit scalar temperature, per-position temperature, affine logit biases and matched logit margins for the one-loop and open-loop controls. Reserve a fresh held-out test for comparing recurrence with these controls. Account for changing token normalizers in sequence-level joint log-probability scores.
- **Same-readout intervention:** Construct different states for the *same prompt* with the same full readout but different head-invisible tangent/fiber components, using constrained norm-preserving optimization and independent seeded perturbations. Test one-step successor-logit divergence, 2/4/8-step delayed divergence, rank-64 candidate changes, future@12, and continuation repair. Perform full-sequence and answer-site-only interventions as separately labeled arms.
- **Donor and manifold controls:** Same-prompt different-fork donors, different-prompt donors with readout-matching, exact-reconstruction self-donors, shuffled donors, random tangent equal-norm donors, no-op normalization, and global readout-temperature matched controls. Report per-arm pre/post readout leakage and state norm.
- **Causal head-fiber metric:** \(D_k=\mathbb E[\|q(F_x^k(s))-q(F_x^k(s'))\|]\) on explicitly matched fibers, plus the paired rate of correct-to-incorrect and incorrect-to-correct verified programs. Report the first delay \(k\) at which divergence exceeds a fixed numerical and statistical gate. Use exact source program checks and held-out future grading.
- **Attribution:** Fit development-only linearized restricted observability Gramians and compare their high-eigenvalue directions with first-order TCA directions, then test finite interventions. Distinguish local magnitude from verified task improvement.

**Decision rules to preregister before model evaluations:** An observable-fiber causal claim needs (i) exact readout preservation before resuming, (ii) an effect above matched norm/margin/manifold controls, (iii) paired task-level evidence after multiplicity correction, and (iv) held-out replication. A spectrum or a 2-vs-8 accuracy gain alone is insufficient. If all causal effects vanish under margin-matched controls, retain the existing virtual-depth interpretation and report the negative result.

## 7. Prospective LOOPSCOPE-04: history-conditioned cache and exact matched-fiber forks

**Scope:** two frozen TRM checkpoints, primary gaoxin492 and replication yagizdevre, using new disjoint Sudoku boards (including canonical digit-relabelling exclusions); existing items256 and bench100 become descriptive calibration only.

1. Train capacity- and compute-matched \(C^{\rm now}(x,t,V_t)\) and \(C^{\rm trace}(x,t,V_{0:t})\) reconstructors. Equalize train size, parameter/FLOP budgets, and manifold calibration; include the existing C5 as a direct baseline.
2. Compare held-out state \(R^2\), exact logit conservation, single-transplant continuation outcome, lock-time and never-lock separately, and extrapolation at 16/32/64 steps. Estimate how much currently unexplained variance is observable in the *visible history*.
3. Construct same-board same-step forks from independent seeds, then replace only an exactly head-invisible sphere-normalized component while holding the entire 81-cell visible logit field fixed. Preserve the 16 prefix tokens and \(z_L\) in independently audited controls before jointly testing all hidden buffers.
4. Define \(z_H^{\rm board}=V+U\), \(\|U_i\|=\sqrt{512-\|V_i\|^2}\), \(W_{\rm head}U_i=0\). Any donor \(U_i'\) must be rescaled into this radius, with numerical certification of norm and readout equality before resumption. Capture recipient and donor clean/changed successor readouts to measure \(\Delta_x\) directly.
5. Test residual erasure, matched-fiber residual swaps, randomized equal-radius fiber replacements, positive and negative Resume price, unit-conflict revisions, and unconditional fork agreement. Report paired effects separately for WB-Prop, WB-Stall and round-0 S4.
6. Measure \(G_{t,k}\) on a restricted sampled tangent basis and validate high/low predicted directions with actual finite swaps. Use an independent random-basis control and a fixed perturbation-radius ladder.
7. Distinguish a harmful path-dependent basin bias from a beneficial search history, using signed effects on exact solve, lock time and late-cycle frequency. Require the result to reproduce on both checkpoints or explicitly report checkpoint heterogeneity.

## 8. Proposed placement in v015

- After the present Section 6, introduce **Observable quotients and delayed fiber visibility** with Propositions 1–3 and a readout-geometry clarification.
- In Section 6.1, narrow the statement "only \(R_t\) is genuinely dark" to "measured residual relative to the specified snapshot auditor and training distribution". Add the trace-observer refinement (Proposition 4).
- Replace "stationarity admission" terminology for the \(\bar r\) gate with "stationarity-compatible activation-use admission". Reserve *time-homogeneous map* for verified equal-state replay. Retain the existing measured \(\bar r\) values.
- In the limitations, separate *dark state necessity* from *cache approximation error*, and explain why deterministic \(x\)-conditional hidden entropy is vacuous without resource constraints or random forks.
- Label the ARG-03-N and LOOPSCOPE-04 experiments as **prospective** and keep all empirical results in v014 unchanged until preregistered runs produce receipts.

## Source anchors

- Manuscript v014: https://github.com/MoralityLabAI/LoopedTransformers/blob/main/papers/shared_weights_dark_fibers/PAPER.md
- LOOPSCOPE-03 registered report: https://github.com/MoralityLabAI/LoopedTransformers/blob/main/mechinterp/loopscope03/docs/report.md
- LOOPSCOPE-02 report and manifold controls: https://github.com/MoralityLabAI/LoopedTransformers/blob/main/mechinterp/loopscope02/docs/report.md
- ARG-02 Nanbeige report: https://github.com/MoralityLabAI/Compactification-Theory/blob/research/arg02-recurrent-qwen-20261110/research/algernon/algebraic_reasoning_gym/arg02/REPORT_NANBEIGE.md
- ARG-02 branch through commit 1ac36353b36581b6a634626bcd55542e53dae456; PR #141: https://github.com/MoralityLabAI/Compactification-Theory/pull/141
- Sharma & Vu, 2026: https://arxiv.org/abs/2606.24898
- Anil et al., 2022: https://arxiv.org/abs/2211.09961
- Kubo et al., 2026: https://arxiv.org/abs/2604.13521
