# VPD semantic-repair checkpoint and cheaper-model execution report

Date: 2026-09-02  
Project: `C:\projects\VPD`  
Companion packet: `VPD_CHEAP_MODEL_SEMANTIC_REPAIR_RUN_PACKET_2026-09-02.json`

## Decision

The Bonsai-8B base model and historical adapter remain rejected for
authorization-control release. V7 is the authoritative corrected result:
base separation `0.692347`, adapter separation `0.580743`, adapter-minus-base
`-0.111604`, paired 95% CI `[-0.122397, -0.100996]`, invalid `REQUEST` rate
`0%`, and valid `PROPOSE` rate `100%`. V7 trained no weights.

No model, adapter, GPU, inference endpoint, trainer, or real action was used in
this checkpoint. No empirical blueprint or dictionary was created.

## Intermediate result

The V1 cross-artifact graph remains useful structural-gap evidence, not a
semantic dispatcher. Its audit found:

| Check | Result |
|---|---:|
| Graph nodes | 20 |
| Schema-and-ID matches | 12/20 |
| Dispatch-ready V1 node representations | 4/20 |
| Graph edges | 37 |
| Exact implemented field edges | 9/37 |
| Mismatched edges | 28/37 |
| Production runtime-attachment verifier | absent |
| Semantic dispatch allowed | no |

Audit ID:
`bd3cc043f7b658029fd20c99492c43edbace23550a7e237d36d692e46297eaa7`.
Its fail-closed benchmark passes `23/23`, ID
`adc1d00bf3be2ab9d790252534bc084add403842b26ae9002e2d6216f0095732`.

The highest-impact join has now been repaired in a versioned successor. The V2
budget builder derives resource requests from the exact execution blueprint
and binds both proposal and admission to:

- budget policy ID;
- execution-blueprint ID;
- exact blueprint file SHA-256;
- every job ID and compartment/category grouping;
- exact jobs, steps, activation rows, episode entries, pair entries, seeds,
  batch sizes, dictionary widths, and unique output directories.

It then projects that exact derived shape through the existing V1 ceiling
evaluator. Admission is reproducible and content-addressed, but transfers no
execution, paid-compute, training, or weight-mutation authority.

The synthetic, temporary-only benchmark passes `27/27` adversarial controls:

| Derived shape | Value |
|---|---:|
| Activation categories | 8 |
| Specific compartments | 16 |
| Shared fallback compartments | 1 |
| Execution jobs | 34 |
| Planned steps | 68,000 |
| Planned activation rows | 20,480 |
| Unique output directories | 34 |
| Specific dictionary width | 4,096 |
| Shared fallback dictionary width | 8,192 |

The controls reject wrong blueprint IDs/hashes, wrong policy IDs, authority
broadening, resource substitution, readdressed resource widening, category
rebinding, output reuse, and unaddressed or readdressed tampering.

Pinned V2 evidence:

| Artifact | ID or SHA-256 |
|---|---|
| Blueprint-bound proposal | `50e09fa4b4247483bf616c9f31226bdd021d7ca609b20a6b911cd361175bb400` |
| Blueprint-bound admission | `dadb42b793f491486bcd9677a840f39ad7644f73460254f3b793c5c7c3a60f72` |
| V2 benchmark | `1c239f6416c57374a810732496c20f36a7245d882b41c2282c72b8465678da58` |
| V2 builder source SHA-256 | `31396ffcab4b5ea88f1287745c17c79a47d9b2620d38187b776db9465c0117e1` |
| V2 benchmark source SHA-256 | `f3d0abc1516e20ddd5be70d90ae905cf74045135fcb5ef34db90bf571cad3d54` |
| V2 test source SHA-256 | `88e6269b0ef222dc4bd41b37e0e5a57403b35becc0ad8384c340dbd29611bd6f` |
| V2 benchmark file SHA-256 | `b5e04370fd550667c1abdb74c3215f1e32b0dd016680e1f1dd5501fb114c7949` |

Focused verification passed `7/7` in `92.44` seconds across the new V2 gate,
the semantic-dispatch audit, and the execution-blueprint contract.

## Recovery versus reconstruction provenance

- The recovered tree and the three byte-identical restored modules remain
  governed by `VPD_CONTEXT_RECOVERY_2026-08-29.md`.
- The lost V7 validator was not recovered at its manifest hash
  `1cafe45ca4487a969226f09f9c9c8208d84a04cf5790ad1b8517e0ff6ad2f444`.
  Its successor is explicitly reconstructed and is not represented as the
  lost original.
- The semantic coverage audit, V2 blueprint-bound budget builder, benchmark,
  test, this report, and its run packet are newly derived artifacts. They are
  not recovered historical files.
- The V2 benchmark blueprint exists only inside a temporary directory and is
  deleted. Its proposal/admission IDs are fixture evidence, not empirical
  authorization artifacts.

## Exact V1–V8 state map

The detailed evidence table remains in `VPD_STATE_MAP_V1_V8_2026-08-29.md`.
This is the controlling compact map:

| Version | State | Controlling interpretation |
|---|---|---|
| V1 | partial/corrupt historical foundation | Preserve as history; not current authority. |
| V2 | partial/corrupt historical foundation | Preserve as history; not current authority. |
| V3 | partly exact; sealed task-3 pack missing | Executable sources and recovered receipts do not replace the missing pack. |
| V4 | exact negative receipt; some raw payload absent | Negative result is evidence; missing raw material stays explicit. |
| V5 | exact decomposition receipts; some raw payload absent | 4,032 rank-one components, 252 projections, selected 144-component cell; not a release. |
| V6 | recovered negative but scientifically invalid | Hidden/conflicting-label evaluation; do not use for release judgment. |
| V7 | authoritative corrected negative | Trusted observable `AUTH_STATUS`, hard invariants, release rejection; reconstructed validator provenance explicit; no training. |
| V8 | exact deterministic component packetization | Planning/decomposition evidence only; not empirical retraining or authorization. |

## Proposed cheaper-model runs

### Successor update — C1 and contract lineage completed locally

The corrected semantic graph contract is now implemented and sealed without
modifying V1. It declares 21 logical representations, 20 implemented verifier
representations, 38 typed semantic joins, one terminal node, and three terminal
joins. It accounts for all 28 V1 mismatches and distinguishes exact fields,
canonical-object digests, file hashes, embedded subdocuments, compound receipt
verification, cryptographic authority membership, full reproduction, and
missing joins.

The contract benchmark passes `36/36`. The combined contract, V2 budget,
semantic-gap, Protocol V48, and Runbook V40 suite passes `11/11` in `58.54`
seconds.

| Successor evidence | ID or SHA-256 |
|---|---|
| Semantic graph contract V2 | `6db000c29e0619d6bec26fb85f8d1739bd5fae16e46cccc03516a2171d79adb0` |
| Semantic graph benchmark V2 | `ade0e1f19765c8dc1accf66ee86590c21b4a904a499094441f21edb8fa40fd37` |
| Contract file SHA-256 | `d2d0a051fdd1dcd6656a586a2f7ac8258057c077d15de7fb07a3687128e53399` |
| Benchmark file SHA-256 | `b1e468e4c59effa5dc5b3dac55ca5add190fd954eace3475eedf4ded1e46fe39` |
| Protocol V48 SHA-256 | `bc526292ca5e84a65124129215239aa96d568331716e7abf5ec2ccbd49e536d4` |
| Runbook V40 ID | `08444884365069ea91a362f0fc4be7ca4f7552947780c6a016f739553c0c4681` |

The three explicit terminal joins are: budget admission -> signed scientific
release, budget admission -> runtime admission, and durable transition ->
production runtime attachment. No end-to-end semantic dispatch is authorized.

### Dispatcher update — C2 completed locally

The first real semantic dispatcher slice now invokes the implemented
materialization, execution-blueprint, blueprint-bound budget-proposal, and
blueprint-bound budget-admission verifiers. All four compare the complete
reproduced object. The dispatcher also independently rechecks exact IDs, the
blueprint file SHA-256, policy binding, proposal binding, graph kind, safe root,
and non-authorizing boundaries.

Its temporary synthetic benchmark passes `22/22` controls. It rejects path
traversal, absolute paths, missing files, symlink/junction inputs, graph-kind
substitution, predecessor companion substitution, unaddressed and readdressed
materialization mutation, readdressed blueprint mutation, blueprint byte-only
substitution, policy substitution, proposal authority broadening, and admission
mutation. The combined dispatcher, contract, Protocol V49, and Runbook V41
suite passes `11/11` in `99.40` seconds. The framework-import control requires
the relevant model-framework set to be empty both before and after execution.

| Dispatcher evidence | ID or SHA-256 |
|---|---|
| Semantic dispatch slice | `0efe29534a177e3e4bc242dc091ebcd79c867dfff8c0a80dfe6f153498e68449` |
| Semantic dispatch benchmark | `60139e5dd07adc7bd79b5028bd99d068bbb32a81ec92afe1e2c77091325f04d3` |
| Dispatcher source SHA-256 | `d5bb495f24f7f7c6279a59a9e030eaae63f81b2bc95aa3591ea820ea9cdc924c` |
| Benchmark source SHA-256 | `14caba328596aa2fcf43df488c7727353b4f582a4299f1e0148a6e87617f3873` |
| Benchmark file SHA-256 | `ba6ad9d16e41fb65b2d3583d0d5afab475e392ecc6811945fcc8ff4c68aab09c` |
| Protocol V49 SHA-256 | `0c3798bdca4dd501e8f674afdfb368319ebb6010a4e8ea4f82aff92afec48fbb` |
| Runbook V41 ID | `733555f2967840041f005380985305b61b265c8c15615543284258646f2019ad` |

The benchmark creates and deletes its entire synthetic artifact root. It does
not persist a synthetic blueprint, proposal, admission, or dispatcher result.
Signed authority, durable transition, and runtime attachment remain outside
the dispatched slice.

### C0 — execute now: reproduce the current checkpoint

Run only the three commands in packet run `C0`. Require the pinned benchmark,
proposal, and admission IDs, `27/27` benchmark controls, and `7/7` focused
tests. Stop on the first mismatch. This run may write only Python cache and
test cache data; it must not rewrite the pinned report.

### C1 — completed locally: corrected semantic graph V2 contract

The new V2 contract was implemented rather than editing V1. It uses the actual
implemented representations:

- capture authorization uses `authorization_receipt_id`;
- capture manifest uses `capture_id` and its required dataset/split/role
  companions;
- routing uses `routing_evidence_bundle_id`;
- pair split is embedded in the cluster-aware plan;
- materialization embeds the plan/split and binds its predecessor plan;
- blueprint-bound proposal/admission use the V2 gate in this checkpoint;
- compute authorization is the signed receipt with `receipt_id`;
- nonce, output, and worker reservation are one combined durable transition
  with `transition_reservation_id`;
- production runtime attachment is an explicit absent terminal prerequisite,
  never substituted with the synthetic inert claim.

Acceptance passed: the content-addressed contract rejects V1 fictional IDs, missing
companions, path traversal, symlinks/junctions, readdressed edge substitution,
synthetic-as-empirical evidence, and any claim that the missing production
attachment exists. Measure the new node/edge coverage; do not assume a target
count. Persist no synthetic empirical graph.

### C2 — completed locally: semantic verifier dispatcher, model-free slice

Dispatch is implemented only for representations whose real verifier interfaces are
available: materialization -> execution blueprint -> V2 budget
proposal -> V2 budget admission slice. Each dispatch step must reproduce the
exact upstream artifact and compare the full object, not merely its schema or
ID. Unsupported nodes must return a named terminal gap. Add adversarial tests
for companion-input substitution and content-address-preserving mutation.

Do not dispatch signed authority, durable transition, or runtime attachment
until their exact real inputs are supplied. A synthetic signature or inert
attachment must remain non-authorizing.

### C3 — contract and dispatcher lineage completed locally

Protocol V48 and Runbook V40 seal the contract-only successor. Protocol V49 and
Runbook V41 seal the four-node dispatcher successor. They preserve
V47/V39 and the V1 structural graph/audit unchanged and pin the new contract,
dispatcher, benchmark, test, and boundary evidence. V48/V40 remain contract-only
and must not be relabeled as dispatcher proof.

### C4 — completed locally: budget-bound release and runtime admission

New V2 successors now bind the signed scientific release to the exact
execution-blueprint, budget-policy, budget-proposal, and budget-admission IDs
and file SHA-256 values. The release verifier reproduces the full proposal and
admission and checks the existing Ed25519 authority registry, enabled status,
validity window, revocation list, and unconsumed nonce. The runtime-admission
successor consumes that verified budget-bound release plus the existing exact
job-scoped compute receipt and reproduces the complete read-only admission.

The semantic dispatcher V2 covers seven representations: materialization,
execution blueprint, budget proposal, budget admission, budget-bound signed
release, compute authorization, and budget-bound runtime admission. Five are
full-object reproductions and two are cryptographic receipt verifications. It
closes the budget-admission-to-signed-release and
budget-admission-to-runtime-admission joins. Durable transition and production
runtime attachment remain explicit gaps.

The deterministic temporary benchmark passes `25/25` adversarial controls.
The combined V1/V2 dispatcher and V41/V42 lineage suite passes `10/10` in
`181.19` seconds. Synthetic keys and receipts are deleted with the temporary
fixture and never confer authority.

| C4 evidence | ID or SHA-256 |
|---|---|
| Semantic dispatch slice V2 | `891fbecf114deb595d984e252979301aae7a8fd5c92c305651b12178e854402b` |
| Semantic dispatch benchmark V2 | `681abfc57a4103c72cd7aa7567f23ead61a3f3d9569155c14709902053c26451` |
| Synthetic release receipt | `f86921ad0b94a557cde269b1b56e9fd04c940e12035233bbc4b3815664ef37df` |
| Synthetic runtime admission | `8b4947f36fd50fa162aee42a9ae7b591791d27e3ed03644d996a10b031e2b130` |
| Release/runtime verifier source | `ecbaeb33fbfb7f2ca2db0083d20d15e82d3c5792f948ee3044dc43608f098259` |
| Dispatcher source | `3d5b22df1ac23686cabbbfccb8127c10f8921ee628461bccdde19e5febf8ac19` |
| Benchmark source | `d5e584b19a00c8fc757769f3c0d8e77c70c0431071014d0b5177835a3efb529b` |
| Benchmark file | `5a3249ec0bf0b164f6a0995761c5f893bd32fc335bdcb9b8f048d8253a44795f` |
| Protocol V50 | `55dc12c6f948c6fc8ecaeb1d9d06d037fbc5a72ab2fd0a5990b453bd7271a951` |
| Runbook V42 | `917c9582a1e4d71015eaef89507b335ad54bbebd5c2930482c1aab81d6842589` |

No real signed authority was created, no nonce was consumed, no output was
reserved, and no production runtime, model framework, model, adapter,
inference, training, or weight/dictionary mutation was used.

### E0 — completed locally, design only: fresh authorization retraining experiment

The versioned experiment specification now freezes seven mutually disjoint
roles: candidate training, candidate validation, SAE dictionary training,
cluster discovery, cluster validation, threshold calibration, and final
evaluation. Pair, semantic-template, and condition-archetype namespaces are
distinct across every role. This corrects an important limitation of split V3,
which was pair- and template-disjoint but reused condition types across roles.

All seven roles cover the same eight activation categories and four action
families using role-specific condition archetypes. The planned pack has 840
contrast pairs and 1,680 rows. Invalid or untrusted observable state maps only
to `REQUEST`; valid trusted observable state maps only to `PROPOSE`. Hidden
label state and identical visible prompts with conflicting labels are forbidden.

Four candidates are preregistered: an unchanged zero-effect LoRA wrapper
control, fresh LoRA SFT, fresh LoRA margin training as the primary candidate,
and a non-primary historical-adapter repair diagnostic. Evaluation freezes
invalid `REQUEST`, valid `PROPOSE`, signed separation, paired candidate-minus-
base delta with a pair-level 95% bootstrap CI, action-family/category/template/
condition slices, operational minimums, and zero-action hard invariants.

The schema-mutation benchmark passes `25/25`; the combined E0 and C4 suite
passes `11/11` in `68.22` seconds.

| E0 evidence | ID or SHA-256 |
|---|---|
| Experiment specification | `0c2f14cad8a05efeddda7a2e3428f2d323f49ee1a48f173dfc778a71f855dd86` |
| Specification file | `102d2548adba2d02cfb7a5f5494b4034fceaf470319ffcb3e32b5989bd453178` |
| Mutation benchmark | `c0c88eeb6cdce5fc00564a38712d6e4ac1f771d59df216985d18fd8d497611a3` |
| Benchmark file | `7dbb61e8475e48baaf321c531a0b87c8019b0fda8c62def1176f3ca85a129f06` |
| Protocol V51 | `1b49c83ed86571a9fb9c9f54b768b0ee470123d9a77d63d1bdf7ef637ad2ba2c` |
| Runbook V43 | `870a7cbd0622e804cb6c91ad675160de3acc348df4f99256b576023b99017600` |

No held-out pack was generated and no empirical capture, model load,
inference, training, GPU/external job, or weight/dictionary mutation occurred.

### E1 — authorization-gated next: pack generation, training, and evaluation

Requires new explicit user authorization, exact empirical inputs, active signed
receipts, fresh nonces/output reservations, a separately approved production
runtime, and an approved compute/cost ceiling. Do not repeat V7 merely to spend
compute. Do not train or mutate dictionaries under this report.

## Mandatory stop rules

Stop on any hash/ID mismatch, failed verifier, missing companion, network or
package request, model-framework import, model/adapter load, inference/GPU
work, persistent weight/dictionary write, fabricated empirical input, missing
signed authority, or attempt to treat synthetic evidence as empirical. Every
run receipt must include command, exit code, observed IDs/hashes, files changed,
boundary flags, and first stop reason.
