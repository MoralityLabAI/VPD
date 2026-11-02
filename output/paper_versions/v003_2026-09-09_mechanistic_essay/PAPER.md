# Shared Weights, Different Roles: Mechanistic Interpretation Across Transformer Passes

Version 3 | 9 September 2026 | Research manuscript

## Abstract

Reusing a transformer layer separates the identity of a parameter from its role in a computation. The same weights encounter states produced by different computational histories, so a mechanism must be located by invocation as well as physical layer. We develop this perspective through an instrumented study of Nanbeige4.2-3B, connecting a shared parameter decomposition to activation features measured separately across its two passes. The central methodological claim is that an interpretation of recurrence requires three connected descriptions: the parameter operation, the state on which it acts, and the downstream consequences of that operation at a particular invocation. Our measurements establish a path between these descriptions while exposing a consequential limitation in the feature representation. A paired SAE optimizer control raises held-out explained variance at layer 22's second pass from -0.180 to 0.453; the improved dictionary also gives better reconstruction when transferred to saved parameter interventions. This makes feature differences more usable, without assigning them semantic identities. We argue for interpreting recurrent computation through state-conditioned relations among features, tested by selective intervention and mediation. The resulting contribution is a method for investigating mechanisms in shared layers and a precise account of the evidence still needed to turn numerical differences into semantic explanations.

## 1. A layer's address does not specify its role

A physical layer provides an address at which to inspect a model. In a recurrent transformer, that address names several events. A layer encountered on the first pass receives a state with one computational history; on the second pass it receives a state that has already traversed the shared stack. Parameter identity survives this transition. The conditions under which those parameters operate change. Mechanistic interpretation therefore needs to explain how an operation acquires a role from its place in an unfolding computation.

This distinction is useful even before any feature has been given a name. Suppose a component writes a direction that is subsequently preserved, transformed, or suppressed. Its contribution depends on what was present before the write and on how later operations read it. Repeating the component does not necessarily repeat its functional contribution. The second invocation might reinforce an earlier representation, act on a relation constructed during the first pass, or alter information that was previously irrelevant. These are alternative mechanisms to investigate, rather than descriptions that follow from the existence of a loop.

The question for this paper is consequently concrete: how can we connect the repeated use of a physical parameter to changes in represented information? A useful answer should let an investigator identify a shared operation, distinguish its invocations, observe its effects in a common feature vocabulary, and test a proposed account of what those effects do. This question remains meaningful in an unsuccessful solution trajectory. Errors, incomplete transformations, and ineffective operations also have mechanisms. Aggregate task performance cannot substitute for an explanation of any of them.

We use the open Nanbeige4.2-3B checkpoint as an accessible system in which to construct this analysis. Its pinned implementation executes all 22 physical transformer layers twice, with normalization at the loop boundary. Activation capture covers both invocations of all 22 layers. Focused parameter decomposition targets the MLP down projections at physical layers 12 and 22; sparse activation analysis includes those layers and loop normalization. These are selected measurement sites within a repeated stack. The execution is fixed, with no adaptive router selecting additional passes. [Nanbeige checkpoint and source](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/tree/451ed48c3273ecef7ea8faaa43c31ce529763bb1).

The unit of analysis is an invocation in context: an input, token position, physical layer, pass, and intervention condition. Omitting the pass merges distinct computational events. Omitting the token context can make unrelated states appear comparable. Holding the token sequence fixed lets us compare the two invocations at matched positions and localize the consequences of a perturbation. Allowing generation to continue answers a different question about the trajectory that follows. Both are useful, but their evidence must remain distinguishable.

## 2. Parameters describe an operation; features describe its effects

Parameter decomposition and activation decomposition approach a mechanism from different sides. A parameter decomposition identifies components of a transformation available whenever the shared matrix is invoked. A sparse autoencoder supplies coordinates for the states observed during those invocations. Combining them creates a way to ask which state changes follow from a particular parameter component, and whether those changes differ by pass.

Our VPD implementation adapts Goodfire's parameter decomposition work to shared matrices with invocation-specific intervention masks. Each selected matrix has one fitted set of components. The remaining matrix contribution is retained as a residual, so the intervention baseline preserves the original transformation rather than replacing it with an incomplete approximation. A component can then be perturbed on the first pass, second pass, or both. The physical component remains the same object across these conditions. [Goodfire parameter interpretation research](https://www.goodfire.com/research/interpreting-lm-parameters), [implementation source](https://github.com/goodfire-ai/param-decomp).

This arrangement preserves a distinction that is easy to lose when studying recurrence. Different component outputs across passes need not mean that the model has switched to a different operation. They may arise because the same operation receives different inputs. Conversely, identical physical weights do not imply identical effects on later representations. A mechanistic explanation must account for the relationship between the operation and the state, rather than attributing everything either to weight identity or to activation difference.

The completed 64-component fit captures approximately 2.2% of the selected matrices' squared weight energy. Retaining the residual makes the baseline faithful, but it does not make those components an exhaustive account of the matrices. Nor does weight energy directly measure semantic importance. A component with low energy might matter to a particular relation, while a large component might support a broadly shared transformation. In this study, the decomposition supplies addressable parameter contributions whose coverage is explicitly limited.

For activation comparisons, we fit a common top-k SAE at each selected site using training rows from both passes. We freeze that dictionary before comparing conditions. This supplies a consistent coordinate system: feature 574, for example, refers to the same encoder and decoder within a given dictionary wherever it is measured. It does not imply that feature 574 in a separately trained dictionary has the same identity. Comparisons between optimizers concern reconstruction quality unless feature correspondence has been independently established.

Two differences then answer two distinct questions. A pass difference compares the encodings of two baseline states at matched positions: what changes between invocations in this dictionary? An intervention difference compares a perturbed state with its baseline at the same invocation: what changes when this component is altered here? The latter is computed by re-encoding the complete states:

\[
\Delta z = E(h + \Delta h) - E(h).
\]

The nonlinear encoder matters. Encoding only the activation difference, or projecting it onto decoder directions, need not reproduce the change in the actual sparse code. An intervention can change which features enter the top-k support. Re-encoding captures that change; a linear projection can miss it. Even so, a feature entering the support is initially a fact about the representation supplied by the SAE. Its meaning in the model requires further evidence.

The combined analysis thus provides a chain of experimentally addressable relations: a parameter component is altered at a specified invocation, a hidden state changes, and the same dictionary reports how its coordinates change. This is more informative than an unlabeled activation distance, because it preserves both the origin of the intervention and a reusable description of its effects. It is the beginning of a mechanistic account, with the semantic claims still to be earned.

## 3. The dictionary is part of the measurement

The clearest lesson from the measurements concerns the instrument used to describe the second pass. A shared dictionary prevents arbitrary changes of coordinate system between invocations. It does not guarantee that either invocation is adequately represented. If the SAE reconstructs one pass well and the other poorly, their apparent feature differences mix changes in the model with unequal failures of the measurement.

This problem occurred at layer 22. In the paired optimizer control, the AdamW dictionary achieved held-out explained variance of 0.830 on the first pass and -0.180 on the second. The second figure means that, under the recorded evaluation convention, its reconstruction error exceeded that of the held-out mean predictor. Treating its second-pass code as an adequate inventory of represented information would therefore be difficult to justify. An apparent disappearance of features could reflect the dictionary's failure to express the state.

Following the optimizer comparison developed in Patrick Dugan's earlier SAE study, we fitted paired dictionaries with identical initial parameters and sampled batches. The Muon-hybrid condition applies Muon to matrix parameters and AdamW to biases. Each pair uses the same 6,144-feature, top-32 architecture and 1,000-update budget. Eight inputs supply training trajectories and ten supply held-out evaluation trajectories. This is a single-seed, post-hoc control on an already inspected cohort. [Late-Layer Feature Reweighting by an Alignment Adapter: Improving SAE Quality with Muon](https://www.moralitylab.xyz/papers/Late_Layer_Feature_Reweighting_by_a_Constitutional_Adapter_An_Industrial_Sparse_Autoencoder_Analysis.pdf).

| Held-out reconstruction | AdamW SAE | Muon-hybrid SAE |
|---|---:|---:|
| Layer 22, pass 1 | 0.830 | 0.918 |
| Layer 22, pass 2 | -0.180 | 0.453 |
| Layer 22, pass 2, transferred to BF16 intervention baselines | -0.535 | 0.312 |

The second-pass improvement is substantial enough to change the status of subsequent analysis. We now have a dictionary that captures a meaningful fraction of held-out variance where the original one failed. The transfer result also connects the improved measurement to the saved parameter interventions: freezing the dictionaries and applying them to those baselines retains an advantage for Muon. Because both precision and sampled positions differ between the original evaluation and the transfer, the lower transferred value cannot be assigned to either factor alone.

The effect is local to the measured setting. Muon improves four of six site-and-pass reconstruction comparisons in the original control; it slightly worsens both layer-12 comparisons. Those six comparisons are not six independent replications. At layer 22's second pass, considerable variance remains unexplained even after improvement. The conclusion is that dictionary optimization materially affects the accessibility of this state's structure, rather than that one optimizer has solved feature discovery.

Earlier diagnostics sharpen this interpretation. In a separate BF16 dictionary fit, layer 22's second-pass reconstruction was much worse than a dense projection onto the training activation span. That observation showed that the SAE's failure could not be attributed solely to held-out directions absent from the training span. Reweighting the loss to account for unequal pass variance improved training reconstruction but worsened held-out second-pass reconstruction in all three tested seeds. Sampling both passes equally and balancing their losses therefore did not, by themselves, produce a dependable common vocabulary.

These findings have a direct semantic consequence. A model may preserve or transform a relation that the chosen dictionary fails to reconstruct. The absence of a readable feature change is then weak evidence for the absence of the underlying computation. Conversely, improving reconstruction increases what can be inspected without automatically telling us what the newly expressed dimensions mean. Reconstruction quality and semantic validation address different parts of the inference, and both belong in a mechanistic account.

## 4. A mechanism is a relation across invocations

Invocation-specific interventions make the causal structure of a loop experimentally accessible. A perturbation on the first pass can alter what the second pass receives. A perturbation confined to the second pass cannot change an earlier first-pass state on the same fixed token sequence. This ordering is a useful implementation check and a foundation for a more substantive question: which first-pass changes are carried forward, transformed, or cancelled when the shared computation runs again?

The completed parameter interventions show strong dependence on invocation. For component 0 at layer 22 and a positive coefficient of 0.05, the pooled final-normalization effect norm is 31.70 for first-pass intervention, 2.78 for second-pass intervention, and 31.75 when both are perturbed. These measurements pool the same 80 held-out token positions. They show that applying the same component perturbation at different points in the computation has different consequences. They do not identify the information responsible for that difference.

In particular, the first-pass perturbation traverses more downstream transformations before reaching the measured endpoint. Input state, propagation depth, finite-dose nonlinearity, and numerical rounding all contribute to the observed response. A larger norm therefore cannot, by itself, assign a more important semantic role to the first invocation. Layer-level cosine similarity and norm ratios are similarly descriptive: they locate changes in geometry while leaving the represented relations unspecified.

We also retain the interaction between first-pass and second-pass interventions, defined as the both-pass endpoint minus the two single-pass endpoints plus the baseline. A nonzero interaction records departure from additive endpoint effects at the tested dose. It is a candidate location for further analysis of composition. Establishing a particular interaction between represented concepts would additionally require identifying those concepts and separating their effects from generic nonlinear propagation and finite-precision behavior. Signed-response diagnostics in the archive show that numerical resolution is a material concern for some small effects.

The improved dictionaries let us revisit these interventions in feature coordinates. All saved parameter-intervention states were re-encoded using frozen AdamW and Muon dictionaries. The resulting archives retain baseline pass differences and intervention-induced changes separately, with source tokens, positions, and dictionary identities preserved. Independent recomputation verifies the nonlinear re-encodings and the expected causal ordering. This connects an addressable parameter contribution to a feature response at each measured invocation; the feature coordinates remain provisional descriptions of that response.

What would make such a relation semantic? Consider a proposed grid-transformation mechanism in which one representation marks an object's membership and another marks its destination under a spatial rule. A first-pass intervention might alter membership information; the second pass might then construct a different destination representation. This is an illustrative hypothesis, not a finding about the saved features. Its value is that it specifies what should covary, what should remain fixed, and which downstream change should disappear if the proposed intermediate representation is restored.

This way of formulating a mechanism moves beyond labeling a feature as an object or a rule. The relevant claim concerns how represented relations participate in a transformation. A feature could have similar activation at both passes while its downstream use changes. A large pass difference could instead track formatting or context without participating in the proposed transformation. Semantic interpretation requires evidence about these uses, including cases where the proposed relation is absent or deliberately changed.

## 5. From a feature difference to a semantic explanation

The current study makes components and feature responses separately addressable. It does not yet establish a named semantic circuit. The next analytical step is to formulate claims that the available intervention machinery can distinguish, rather than assign broad labels to whichever dimensions change most.

A useful candidate explanation should specify a relation in the input or intermediate solution, the conditions under which it is represented, and the downstream operation that uses it. Controlled contrasts can then vary that relation while retaining irrelevant surface details, or vary the surface while preserving the relation. For grid problems, these contrasts might preserve an object's identity while changing its position, or preserve a spatial transformation while changing colors. These examples describe a proposed design, not experiments already completed here.

Pass-resolved measurements would establish when the candidate relation becomes accessible in the fixed dictionary and how that pattern changes across the contrasts. Pass-specific parameter intervention would then test whether a shared component contributes to that pattern at the predicted invocation. Finally, restoring the proposed intermediate feature state under intervention could test whether it mediates the downstream effect. A successful restoration would need comparison with irrelevant directions and other candidate mediators; an SAE decoder edit alone may leave substantial representational error untouched.

The distinction between discovery and explanation is visible in the completed residual-steering extension. We selected layer-22 Muon feature 574 by its maximum mean absolute pass difference on the eight training inputs. That rule identifies a dimension worth inspecting because it changes across passes. It provides no semantic label. Perturbing its unit decoder direction at positive and negative doses of 0.1, on the first pass, second pass, or both, demonstrated that the frozen feature can be used for invocation-specific interventions during generation.

Across the resulting 70 continuations, three intervention conditions produced a correct answer on the same ARC test input where the baseline was wrong. Both signs of second-pass perturbation were among those conditions. The companion test input remained wrong, and all arms retained zero complete-task successes across eight evaluation units; 62 continuations reached the output cap. These observations establish a narrow trajectory sensitivity. They do not give feature 574 a directional semantic interpretation, and the two successful signs resist a simple account in which increasing this feature monotonically improves the relevant representation.

For mechanistic interpretation, the important missing observation is what changed between the intervention and the altered answer. The extension saved continuations but did not collect new hidden-state trajectories for those 70 outputs. Earlier activation archives therefore cannot supply that intermediate evidence. A targeted follow-up would trace a specified relation through selected states and intervene where the proposed mechanism predicts, rather than treat another change in final correctness as sufficient explanation.

Correctness can anchor such a test when the hypothesized operation concerns a particular answer relation. Other endpoints may be more discriminating: which cells are assigned to an object, whether a constraint remains represented, or which candidate relation is preserved after the second invocation. These endpoints require their own validation. Their purpose is to make the mechanistic claim precise enough that a counterexample can refute it, including on trajectories that never produce a correct final answer.

## 6. Toward a semantic account of shared computation

The semantic vision developed here is an account of reusable operations acting on evolving representations. A shared component supplies a stable experimental reference across passes. A common, adequately evaluated dictionary supplies a provisional language for describing the states it encounters and changes. Selective intervention connects the operation to those changes. Mediation and controlled semantic contrasts would establish how a particular relation participates in the computation.

This perspective makes several possible outcomes scientifically distinct. The second pass could preserve a relation while changing the form in which later operations access it. It could combine relations made available during the first pass. It could suppress an inconsistent candidate, or repeatedly apply an operation without resolving the relevant error. These possibilities concern the organization of computation. They cannot be distinguished by pass count alone, and none should be assigned to this checkpoint without the corresponding evidence.

The completed work contributes the ability to investigate that organization across shared layers. It also shows why measurement quality is part of the scientific argument: an inadequate SAE can make an invocation poorly described even when the underlying state is available for direct capture. The Muon comparison improves that description at the late layer and carries the improvement into saved parameter interventions. The intervention machinery then permits components and feature directions to be addressed at separate invocations. Together, these results establish an empirical basis for asking which represented relations are transformed by which shared operations.

A mechanistic explanation of recurrence should ultimately say what an invocation receives, what it changes, and how that change is used. Physical layers, sparse features, and endpoint measurements each provide only part of that account. Connecting them around a testable semantic relation is the work that turns a record of repeated computation into an explanation of it.

## Appendix A. Evidence and scope

This manuscript reorganizes completed measurements around their mechanistic implications. Version 2 remains the detailed technical companion, including full tables, execution controls, grading audits, and the separate Hermes study. The original sources and all archived versions are preserved. No new model run or dictionary fit was performed for this revision.

| Evidence used here | Scope and inferential limit |
|---|---|
| Invocation capture | Both passes of all 22 physical layers; focused VPD at layers 12 and 22, SAEs also at loop normalization. Fixed two-pass execution. |
| Shared parameter interventions | Completed 64-component BF16 fit; component 0 perturbed at both signs of 0.05 on first, second, or both passes. Approximately 2.2% selected weight-energy coverage; residual retained. |
| Paired SAE optimization | Three sites, two optimizers, one paired seed; 6,144 features, top-32, 1,000 updates, batch size 128. Eight training and ten held-out inputs. |
| Frozen-dictionary transfer | Previously saved BF16 interventions re-encoded with the Q4-trained dictionaries. Precision and sampled positions both change; no isolated precision-effect claim. |
| Residual steering | Ten evaluation input records, seven conditions, 70 continuations. Feature selected on training pass differences. No semantic label or new intermediate activation capture. |

The 18-input trajectory set contains eight training and ten evaluation records. The ten evaluation records correspond to eight complete evaluation units, because some ARC tasks have more than one test input. Local holdouts drawn from public training data do not establish pretraining decontamination. A much larger prepared corpus was not evaluated and contributes no sample count to the findings in this manuscript.

Teacher-forced BF16 analysis uses saved Q4-generated token sequences. It measures a BF16 computation on those sequences rather than BF16 self-generation. Pass-local causal ordering statements refer to a fixed sequence: after free generation diverges, subsequent tokens can change later first-pass states. Short-prefix cache checks and exact zero-write controls support the intervention implementation but do not establish full-logit equivalence across every long continuation.

The separate completed Hermes Campsite study is retained in the technical companion. It analyzes a model-plus-controller system involving retrieval and constraint correction; its inspected auxiliary path does not load a neural recurrent checkpoint. Its results do not identify an internal transformer mechanism and are therefore outside the evidential argument of this manuscript. This study likewise makes no inference about an undisclosed model's internal architecture from its benchmark behavior.

## Appendix B. Reproducibility and source map

The preserved technical companion is `output/paper_versions/v002_2026-09-09_with_campsite/`. The paths below are relative to `artifacts/nanbeige_reasoning/` in the accompanying VPD workspace. Version 3 retains a separate source, PDF, renderer, render receipt, and provenance manifest.

| Claim or measurement | Local source |
|---|---|
| Completed BF16 VPD and interventions | `prime-20260906T204058Z-561/` |
| Signed-effect numerical diagnostics | `signed_effect_resolution_v2/` |
| Paired optimizer reconstruction | `muon_sae_control_v1/summary.json` |
| Independent SAE verification | `muon_sae_verification/complete.json` |
| Frozen SAE intervention transfer | `muon_vpd_transfer_v1/` |
| Completed residual steering | `prime-20260908T200850Z-601/paper_results_v2/` |
| Separate Campsite study | `logic-prime-20260909T042949Z-897/grading_v2/` |

The model revision is `451ed48c3273ecef7ea8faaa43c31ce529763bb1`. The pinned implementation sources, trajectory identities, dictionary and factor hashes, and independent arithmetic checks are retained with their respective runs. Recomputed feature encodings verify the saved calculations; they do not independently replicate model execution or validate feature semantics. Numerical precision, coverage, and reconstruction limits should accompany any reuse of these archives.
