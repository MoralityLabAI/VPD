# VPD recovered-shape execution report for a cheaper model

Date: 2026-09-01  
Project root: `C:\projects\VPD`  
Authority: local, model-free verification and successor implementation only

Paid inference, model or adapter loading, external/GPU jobs, weight mutation,
SAE training, held-out generation, and real actions are not authorized.

## Outcome

The recovered authorization-control project is locally operational. The
historical release decision remains **reject both Bonsai-8B base and adapter**.
V7 measured base separation `0.692347`, adapter separation `0.580743`,
adapter-minus-base `-0.111604` with 95% CI
`[-0.122397, -0.100996]`, invalid `REQUEST` rate `0%`, and valid `PROPOSE`
rate `100%`. V7 trained no weights.

The new recovered-shape capacity benchmark passes `15/15` controls with ID
`e8021583211fde3fc1185dfb2423a7c56dd33089db66d3143ca5823692239390`.
It reads recovered episode metadata without modification and performs no model
inference. Its route allocations are explicitly simulated and are not evidence
of Bonsai activation clusters.

## Intermediate results

- The recovered `sae_training` role contains exactly `6,292` episodes and
  `572` authorization pairs. Every pair has 11 episodes: four
  `missing_or_untrusted_authority` episodes and one episode in each of the
  other seven categories.
- `discovery` and `cluster_validation` each contain `4,719` episodes and `429`
  pairs. The three roles share semantic split ID
  `a5aa3370406876c1c48ed702b431a64cb6a68c8376ab28b0aad1d852c22dd8dc`.
- A provenance defect was repaired in the receipt adapter: each role now has a
  distinct identity derived from the common semantic split, role, dataset
  hash, and episode/pair/template hashes. The semantic parent is preserved.
- The synthetic three-role capture-to-router-to-planner benchmark passes
  `18/18`, ID
  `b27a7f901d191a81a3bc4e708244edf1a3325e968b979471d064986d2c773df6`.
- Planner V1 is unsafe for this incidence shape: shared cross-category pair IDs
  collapse all specific compartments to the shared fallback.
- Planner V2 assigns each pair once to a global training/validation partition.
  Balanced, skewed, and ambiguous-heavy simulations preserve 16
  category-cluster compartments plus one shared fallback, 34 manifest-only
  jobs, all 6,292 episodes once, and all 572 pairs.
- Sparse routes fail closed. A 1% minority route merges eight under-supported
  routes; making only `valid_authority` sparse merges exactly one. The planner
  does not oversample.
- A scenario without shared fallback support is rejected. An unsplittable
  pair-incidence triangle and recovered-incidence tampering are also rejected.
- The canonical-binding readiness worksheet exposes all ten unresolved fields,
  requires canonical absolute adapter/output paths, and reserves a new direct
  child below `artifacts/vpd_cluster_to_sae_empirical_capture_runs` for each
  run. It opens no model or adapter content.
- Its adversarial benchmark passes `17/17` controls. The latest focused lineage
  and authorization regression passed `15/15` tests.
- A one-way streaming content verifier now hashes the exact model tree and
  adapter weight/configuration bytes without importing a model framework or
  deserializing weights. Its benchmark passes `19/19` controls, and the latest
  combined content/binding/authorization lineage regression passes `20/20`
  tests.
- A read-only runtime-admission join now requires the preflight, canonical
  binding audit, exact-byte verification, and signed authorization to share one
  preflight ID. Its benchmark passes `12/12` controls, and the expanded focused
  lineage passes `25/25` tests. Atomic creation of the receipt-bound output root
  is the fail-closed replay barrier; reservation does not start capture.
- A four-role campaign guard now enforces exact discovery → calibration →
  cluster-validation → SAE-routing prerequisites and meters jobs, batch
  forwards, episode-forward equivalents, effect bytes, wall time, and cost.
  Its benchmark passes `22/22` controls and its focused lineage passes `22/22`
  tests. Counter increments must occur before compute; final evaluation,
  generation, real actions, and persistent weight mutation are rejected.
  The last authoritative broad checkpoint before this source/lineage refresh was
  `334 passed, 1 deselected, 3 subtests passed`.
- A guarded runtime adapter now reserves the job and its full effect-payload
  allocation before the first callback, then increments batch-forward and
  actual episode counters before every callback. A callback failure remains
  charged. Its synthetic, no-model benchmark passes `20/20` controls with ID
  `7dca595cfc6d37709b144a12089b9407b06ca3a19c5607c271bde7e854763ae2`;
  focused adapter and successor-lineage tests pass `5/5` in 169.73 seconds.
  This is an injectable boundary, not a production capture runner and not
  capture authority.
- An exact guarded-job allocation manifest now maps all four roles across the
  16 recovered V8 packets: 64 unique content-addressed jobs whose sums exactly
  equal the preflight ceiling. Manifest ID
  `b046a591e380bc4882534f07329ba32c1ce0fdcb1735dcb99fc362196ea23a65`;
  allocation benchmark `18/18`, ID
  `bc929b7751eb09cc9b0a8d4ff258c546e59f27b5fa61e71decc4753e02e007a8`.
  The manifest records that the ten recovered execution bindings are still
  incomplete and must be regenerated from the future completed preflight.
- A plan-to-allocation binding verifier rejects packet, role, dataset,
  forward, episode, category, sketch, execution-identity, and output-root
  substitutions. Its benchmark passes `15/15`, ID
  `5eb11595d2d24b75fd8535e42625150c7257bcdeea4f44feb69f1e3f602225a9`.
  An identity-complete plan remains non-authorizing without runtime admission.
  Focused allocation, binding, and successor-lineage tests pass `7/7` in
  177.47 seconds.
- Guarded runtime adapter V2 now reserves the exact job and full effect payload
  before a runtime object can be attached; attachment is single-use and failed
  callbacks remain charged. Its fake-runtime benchmark passes `14/14`, ID
  `f6b9bef6f89f786e90e0ca1f8af4d24395c2dc54be50605d4d7cab339a5d2d36`.
- A versioned admission-to-output transition now reverifies real admission
  immediately before atomically creating a new direct-child output root. It
  rejects synthetic admission in real mode and treats duplicate reservation as
  terminal. Its model-free benchmark passes `13/13`, ID
  `cc8938f2cb5fcf8d0b0ef7922ca07a22622712ae5e8bc8a27bb6a4947bd0c06a`;
  no persistent run root was created.
- Canonical capture runner V3 is implemented with allocation, reservation,
  campaign-guard, callback-preaccounting, cleanup, and V3+V2 receipt replay.
  Its injected fake-runtime benchmark passes `18/18`, ID
  `882f038f16bddca72b9db40b8b952406e860c21eb20e21de3163b399180f7a43`.
  The production code path exists but has not been empirically exercised; no
  model framework, model, adapter, or inference was used. The runner-layer and
  V33/V25 lineage tests pass `6/6` (`3/3` in 16.20 seconds plus `3/3` in
  301.18 seconds).
- The prior SAE plan audit exposed a real lineage gap: its 18 category-level
  jobs were not yet bound to admitted within-category cluster routes. Capture-
  to-router adapter V2 now closes the first half of that gap by verifying the
  guarded V3 wrapper and preserving the V3 execution ID as the router-facing
  identity while retaining the V2 payload and guarded-job identities. Its
  synthetic benchmark passes `18/18`, ID
  `10486291b569bd2eeea91f8429702b6c668463d727013cad98baf2f676628688`.
  The direct and V34/V26 lineage tests pass `5/5` in 359.66 seconds. No
  persistent empirical routing receipt was produced.
- A durable cluster-aware SAE materializer now requires exact eight-category
  coverage across discovery, independent cluster validation, and SAE-training
  routing: 3 guarded V3 capture IDs, 24 adapter receipts, 24 router receipts,
  and 8 stability receipts. Its synthetic benchmark produces 16 specific
  compartments plus one shared fallback and 34 manifest-only jobs under one
  global pair split. It passes `18/18`, ID
  `a8dfb72bd021847f2511258c1d66f08d8bf7ce9c8e1a822f41d4b1af437d3601`;
  direct and V35/V27 lineage tests pass `5/5` in 232.37 seconds. Synthetic
  evidence cannot be relabeled as empirical, and no empirical plan was emitted.
- A cluster-aware SAE execution-blueprint builder now converts that exact
  materialization contract into 34 isolated, manifest-only execution jobs: two
  seeds for each of 16 category-cluster compartments plus one shared fallback.
  Every job binds exact episode and pair lists, its source routing and stability
  receipt IDs, the single global pair split, and a unique relative output
  directory. Its exact eight-category benchmark passes `17/17`, ID
  `7612bb0da7e388f029cbfe66b17a104ff93a32fef22763df3625a462db5daa75`;
  direct and V36/V28 lineage tests pass `5/5` in 185.57 seconds. The runtime
  worker, empirical blueprint, scientific release receipt, and compute
  authorization are explicitly absent, so these contracts do not train an SAE.
- A fail-closed, injectable cluster-aware SAE runtime worker core now validates
  all 34 exact execution contracts before accepting a job. It reserves the job,
  planned steps, and activation rows before a runtime can attach; runtime and
  trainer callbacks are single-use; replay and wrong-job authority fail before
  attachment; failed callbacks remain terminal and charged. Its synthetic,
  no-training benchmark passes `22/22`, ID
  `dbab569a095cd9c01449886d84f7a292b75edcf919ae65ba5cfc29af0a059188`;
  direct and V37/V29 lineage tests pass `5/5` in 423.96 seconds. Synthetic
  blueprints are rejected for production. A production CLI and real scientific-
  release and compute-authorization verifiers are not implemented, no empirical
  blueprint exists, and the worker core grants no execution authority.
- A signed cluster-aware SAE admission layer now verifies separate Ed25519
  scientific-release and exact-job compute receipts. Release scope binds the
  complete blueprint, plan, materialization, global pair split, routing, and
  independent stability evidence; compute scope binds one job, exact samples,
  steps, activation rows, dictionary width, seeds, device, output directory,
  and base-weight immutability. Time windows, disabled authorities, revocation,
  nonce replay, signature tampering, and scope broadening fail closed. Its
  synthetic benchmark passes `24/24`, ID
  `98c5613d4fe7615ca217d51e80460811e1901d05d177d7839a4aaf8d6f8d79b2`;
  direct and V38/V30 lineage tests pass `5/5` in 436.32 seconds. The dry-run CLI
  never consumes a nonce, reserves output, attaches the worker, or invokes the
  trainer. Durable nonce consumption and atomic output/job reservation remain
  absent, as do an empirical blueprint and real signed receipts.
- A durable cluster-aware SAE transition now atomically binds each compute
  nonce to a unique reserved job root while a release-nonce campaign can admit
  multiple distinct, exactly scoped jobs. It persists the worker-ledger
  reservation before any runtime attachment, reserves the exact dictionary
  output namespace without creating dictionary output, rejects replay and
  cross-job nonce substitution, and treats partial tombstones as terminal with
  no automatic rollback. Its temporary synthetic benchmark passes `20/20`, ID
  `04952c9f6f8648c0a90f9cd3426e2ef546dfc6377987639ce2733479deed1861`;
  direct and V39/V31 lineage tests pass `5/5` in 666.93 seconds. The first
  benchmark draft exposed temp-path-dependent transition IDs; protocol replay
  rejected it, those unstable result fields were removed, and the current ID
  reproduced before sealing. No workspace reservation root, runtime attachment,
  trainer invocation, or dictionary output was created.
- A durable inert-runtime attachment layer now reconstructs the signed
  admission and reserved worker ledger exactly, creates a single-use attachment
  claim before calling the runtime factory, and accepts only a synthetic runtime
  declaring no model load, training, or persistent output. Factory failure and
  forbidden active runtimes leave terminal claim tombstones; replay is rejected
  before the factory runs again. Its temporary benchmark passes `20/20`, ID
  `128a32c051b18d32be0af877929f7c5344562b13d57993c6fd9c40bedfb8e1e8`;
  direct and V40/V32 lineage tests pass `5/5` in 763.09 seconds. The inert
  runtime was closed exactly once, its trainer callback was never invoked, and
  no production runtime, model framework, dictionary output, or workspace claim
  was created.
- A durable synthetic callback-execution layer now creates a single-use claim
  before invoking the already-attached inert callback, persists a content-
  addressed success receipt, and persists terminal charged failure receipts for
  callback exceptions or boundary-weakening results. Replay is rejected before
  a second callback. Its temporary benchmark passes `21/21`, ID
  `e6ca589f69c120db4a5a9c71d34ba541b9f909673bb6963d0374a632fe326397`;
  direct and V41/V33 lineage tests pass `5/5` in 1,915.89 seconds. One valid
  inert success and two charged failures ran only against synthetic fixtures.
  No model framework, model, inference, training, weight mutation, dictionary
  output, production callback, or persistent workspace receipt was created. An
  initial implementation attempt correctly failed lineage replay after changing
  the pinned attachment source; that source was restored byte-for-byte before
  V41 was derived.
- A complete synthetic inert campaign layer now requires the exact blueprint
  job set before creating its campaign claim, then executes every per-job inert
  callback behind its own durable claim. The success path covers all eight
  activation categories, 17 compartments (16 specific plus one shared
  fallback), 34 jobs, and 34 unique output directories under one global pair
  split. Its controlled failure at job eight charges exactly seven successes
  plus the failing callback, leaves 26 later callbacks uninvoked, closes every
  runtime once, and emits a terminal non-retry receipt. The benchmark passes
  `28/28`, ID
  `34204198cc2cd605e74fff0f24da7b5d09337613f16da74872f354c53f8b32f5`;
  direct and V42/V34 lineage tests pass `5/5` in 2,413.61 seconds. No model,
  inference, training, dictionary output, production campaign, or persistent
  workspace receipt was created.
- A reconstructed, nonexecuting compartment-resource ledger now turns the
  verified synthetic campaign summary into reviewable per-job, per-compartment,
  and per-category accounts. Raw campaign receipts were temporary and are not
  claimed as recovered. The ledger covers eight categories, 17 compartments,
  and 34 jobs: 68,000 planned steps, 20,480 planned activation rows, widths
  4,096 and 8,192, and 34 unique output directories. The shared fallback is a
  separate two-job account and is never double-counted into category totals.
  Success reconciles 34 callback charges; the terminal-failure example
  reconciles seven successes, one charged failure, and 26 uninvoked jobs with
  no retry authority. Its benchmark passes `24/24`, ID
  `0c9800d968a49df90b7b1ab942ed5f59017da4f0ee491eb8440b6257aad0f642`.
  Direct and V43/V35 lineage tests pass `5/5` in 2,853.69 seconds.
  These are planned synthetic resources, not observed consumption or empirical
  evidence; no model, training, output, retry, or execution authority exists.
- A sealed nonexecuting budget-admission policy now converts those reconstructed
  accounts into fail-closed category, specific-compartment, shared-fallback,
  and global ceilings. The exact reference shape is 8 categories, 16 specific
  compartments plus one shared fallback, 34 jobs, 68,000 steps, and 20,480
  activation rows. The shared fallback is charged once globally and never to a
  category. The benchmark passes `32/32`, ID
  `5a2bb38b94c43856bcc9d7e7932db98e3cb89192af60e3418706d33713b6db64`;
  direct and V44/V36 lineage tests pass `5/5` in 678.55 seconds. Budget
  admission transfers no empirical, compute, retry, training, or execution
  authority.
- A nonexecuting empirical-blueprint budget-binding worksheet now makes the
  remaining graph explicit: 50 required bindings across six phases, comprising
  10 capture execution identities and 40 artifact, receipt, authority, nonce,
  reservation, worker-claim, and runtime-attachment ID/hash fields. All 50 are
  currently missing. A synthetic complete fixture is used only in memory to
  prove syntax handling; it opens no model or artifact content and is never
  persisted. The benchmark passes `29/29`, ID
  `b95ab40c46c621534b0cf2aed26c342848180a9c5bc7fd48b9674498292159a3`;
  direct and V45/V37 lineage tests pass `5/5` in 1,274.35 seconds. Syntax
  completion alone does not verify the artifact graph, budget binding, signed
  authority, durable transition, training, or execution.

## Current identifiers

| Artifact | Identifier |
|---|---|
| Recovered capacity split V4 | `a5aa3370406876c1c48ed702b431a64cb6a68c8376ab28b0aad1d852c22dd8dc` |
| Recovered-shape capacity benchmark V1 | `e8021583211fde3fc1185dfb2423a7c56dd33089db66d3143ca5823692239390` |
| Five-role recovered incidence audit V1 | `c14b80e53ea6fe39e171fbd201ab0a87b86dc39737c091fd83616bb813d2cefc` |
| Cluster-to-SAE empirical preflight V1 | `edf41948607ac8471030545ea9782c1f6ea76dd2aec72f09aa7cffac5fb45f91` |
| Empirical preflight benchmark V1 | `03f8f0e82283b82b28997403a4f68bcd84d6d9630e1a23cc825ccbb805b28e13` |
| Cycle-free empirical preflight V2 | `5465f17d583a0dd19f5c4ea4939883ac8f65b7a19e128f7ad130246eb2a816bf` |
| Capture-authorization benchmark V1 | `6d359a0c2608afe5f4d5bd8366ec4afe53a667b7be497cb521b814b22fee0f18` |
| Binding-readiness worksheet V1 | `77e2c766499f22d401ffa60166b31dbc8fe4d0ecaefa349ab3b636202fef1472` |
| Binding-readiness benchmark V1 | `987909d0fef72a41a3a50775c2330402632bb89458c5c6ba4b917634a9f0eda1` |
| Bound-content benchmark V1 | `dd035b55e6d56d941c369338123257f7bb5548326e4297f8c2d547315cb50f1a` |
| Runtime-admission benchmark V1 | `32e17a9ba015320b44cf5ae4b9a88e56aa8eabde42e45e5e67635cbe2c366a66` |
| Capture-campaign guard benchmark V1 | `68555a4dd6e87341cadd4eb3a83bd49223314a44617add980a5337f02d1e6288` |
| Guarded capture runtime adapter benchmark V1 | `7dca595cfc6d37709b144a12089b9407b06ca3a19c5607c271bde7e854763ae2` |
| Guarded capture job allocation manifest V1 | `b046a591e380bc4882534f07329ba32c1ce0fdcb1735dcb99fc362196ea23a65` |
| Guarded capture job allocation benchmark V1 | `bc929b7751eb09cc9b0a8d4ff258c546e59f27b5fa61e71decc4753e02e007a8` |
| Guarded capture job binding benchmark V1 | `5eb11595d2d24b75fd8535e42625150c7257bcdeea4f44feb69f1e3f602225a9` |
| Three-role integration benchmark V1 | `b27a7f901d191a81a3bc4e708244edf1a3325e968b979471d064986d2c773df6` |
| Capture-receipt adapter source | `071a261d2250b28a236f760f8d6a16fa0e4858cc72495725d1ee02f1c43ec9f8` |
| Shared-pair-safe planner V2 source | `b7fce50af30687d0fe4d7b75f589e1acc61863bddc73c7517caf91b1ad8d17a3` |
| Protocol V24 | `c42dc6e43296c7e4a3d6054174e015620221cf141377bff44773d8cfc0add2b2` |
| Runbook V16 | `321b0cab4fc95d0270395f39ba95d0139a3ea258e3cb9c05822b6c82f6ce5dd3` |
| Protocol V25 | `e1175601443bf4162df3400ad3e0740eb65c74b5e6eeeb30c92657998ddf7626` |
| Runbook V17 | `c8d9bf9a1ecdd327af6066a7d3f2084ddd7bdccb3782d2e321e904af8784dff4` |
| Protocol V26 | `3372cf627793588a9e5c2078f44f7955e028c8f43fb071d37d07735ced02eaba` |
| Runbook V18 | `c96df7676f62501531c33dbb259b8c5e846f8683b3d0e11b756d4d48ca8b140f` |
| Protocol V27 | `36c66bfb8591961342ea717bde6289f2511e32fbe2577959dad683ab51b94eca` |
| Runbook V19 | `74e48f8b1aee138f69135845f31d4ff2a8eda34eba189ff7506f2de0084fff44` |
| Protocol V28 | `128d8ad5a7fc34b56b24e46b5221b919e035c74b4b5faa7d2d6abcb15dd316b5` |
| Runbook V20 | `0586ec6de4f355743e2f3372c4642140de23ee6329deb6739cb156436d68def2` |
| Protocol V29 | `8dfdfe72a6fd245d4d5a9b4d5612a1463bada0cdc3d2ae61afe32a342bef4a8b` |
| Runbook V21 | `3dc3f49680e15add3a1e2346662576472ea72673bbee602b157b5714099c82a7` |
| Protocol V30 | `6e231ace77b43b154e6622da1ee62017a02fbfebc44fbd2ad6b80720a9f47925` |
| Runbook V22 | `8b39c4b66bfbe6edcd3688d0aff9842bdfd3fecc318bcf3ac2bd3c9a04e6ff1d` |
| Protocol V31 | `d36d6dc655b4879b391e604965f8a214ae4397cc9c569c1e70a0f66d35712104` |
| Runbook V23 | `79f888f86513ad58c6640f6d00c4337716566c26ace55263207300c1f0e5bb2f` |
| Protocol V32 | `59749a0512ddbb0d4c7ea7fc3b91b9269da7cee8857e1e9a9f9bf11d20e13011` |
| Runbook V24 | `7a40353f8c085840c587c0cb2fb239a27bf9eb26381a91690993f8c1557c6f6f` |
| Guarded capture runtime adapter benchmark V2 | `f6b9bef6f89f786e90e0ca1f8af4d24395c2dc54be50605d4d7cab339a5d2d36` |
| Output-reservation transition benchmark V1 | `cc8938f2cb5fcf8d0b0ef7922ca07a22622712ae5e8bc8a27bb6a4947bd0c06a` |
| Canonical capture runner benchmark V3 | `882f038f16bddca72b9db40b8b952406e860c21eb20e21de3163b399180f7a43` |
| Protocol V33 | `cc25fac546c628e4618511e569485faf9a0a9984000affe85dfdbabaf0be2559` |
| Runbook V25 | `a6ba3497fc808e65305a096b113c0e141cdb03094296c42d0f58ba099da3f08a` |
| Guarded V3 capture-to-router adapter benchmark V2 | `10486291b569bd2eeea91f8429702b6c668463d727013cad98baf2f676628688` |
| Protocol V34 | `4a1d90e56b29f0eca513874aa62d5478731e6d5c529ec2a81f56f2fb2c35b370` |
| Runbook V26 | `746a4ed9b85be93e7dee9b95e949b6a685e7c98a2bda08d91ffd7e8822a81727` |
| Guarded cluster-aware SAE materialization benchmark V1 | `a8dfb72bd021847f2511258c1d66f08d8bf7ce9c8e1a822f41d4b1af437d3601` |
| Protocol V35 | `6b9650239cb69cb51a0d72dd003c9b407086007ffc5dada7a14f616b0d4f3b9c` |
| Runbook V27 | `d9de56b5b51ebfd48dc7625e853c78264da640700d1a1623bf82385358285c53` |
| Cluster-aware SAE execution blueprint benchmark V1 | `7612bb0da7e388f029cbfe66b17a104ff93a32fef22763df3625a462db5daa75` |
| Protocol V36 | `734ec49cd8c4b691333e4936e25fa9f691b3f5a4f6f84a6c9aa13b0da076ecc2` |
| Runbook V28 | `d19d71c3dbfc304bda207fad49f300e7709769b90e3f6ecb1964b42ea3e565b7` |
| Cluster-aware SAE runtime worker benchmark V1 | `dbab569a095cd9c01449886d84f7a292b75edcf919ae65ba5cfc29af0a059188` |
| Protocol V37 | `48462a79bd32534c2472b64dfd8badd13c3d5246cd7bd7ff6742bc41fcde04e7` |
| Runbook V29 | `d4d7e3d9099b5f454569e8d752fc16081c2f1d5d8b959c5a985f88dc3d3cb3f1` |
| Cluster-aware SAE signed admission benchmark V1 | `98c5613d4fe7615ca217d51e80460811e1901d05d177d7839a4aaf8d6f8d79b2` |
| Protocol V38 | `90660a5aaacaffeb33d22e6c49cb02a2e160a2b2a5a2946f9f31990c3d1fbd06` |
| Runbook V30 | `8f3cbc0bff18b1cdc8f85e71b220f28269f2f3db44fac32fdf8883b1689d1d61` |
| Cluster-aware SAE durable transition benchmark V1 | `04952c9f6f8648c0a90f9cd3426e2ef546dfc6377987639ce2733479deed1861` |
| Protocol V39 | `b167797dc91c93292ae651d05c96fcb433995cedf62a7a340e85c4e305a92ab7` |
| Runbook V31 | `91108681b156f3ad3e6ac7449d32fa8e052147dd04ea0ba5109fe39fb69e476e` |
| Cluster-aware SAE inert attachment benchmark V1 | `128a32c051b18d32be0af877929f7c5344562b13d57993c6fd9c40bedfb8e1e8` |
| Protocol V40 | `197f7f02ffbeb5f9a530a5880391daa0951611bd42b4202236c78e33a151a9c8` |
| Runbook V32 | `30bcec255ef18f654134794c14a52d88599264d3c8f387fdca2a991f75c9bacf` |
| Cluster-aware SAE inert callback benchmark V1 | `e6ca589f69c120db4a5a9c71d34ba541b9f909673bb6963d0374a632fe326397` |
| Protocol V41 | `87c7b79aac37bc094ddfe06501f020eda9857c5ef08767eff492f5c0cb21476b` |
| Runbook V33 | `23e7c03df8c0b73f71e85cfaea4977b230e129d82211fa7e944e71920eaeecc8` |
| Cluster-aware SAE inert campaign benchmark V1 | `34204198cc2cd605e74fff0f24da7b5d09337613f16da74872f354c53f8b32f5` |
| Protocol V42 | `fc612123f8e293c25e57c6b470722ce2a6c5876f1e9a43c477b31945bedab383` |
| Runbook V34 | `9e93f62742b81b7080ba6d75fa12f58a20bf882b5f8ae3190194c6e03d7d9b06` |
| Compartment resource reconciliation V1 | `c9759d5e1f89cd3f2e6d3a9fa264e27f0ae7e1eee933c3413de5ac3de54bf0ff` |
| Compartment resource reconciliation benchmark V1 | `0c9800d968a49df90b7b1ab942ed5f59017da4f0ee491eb8440b6257aad0f642` |
| Protocol V43 | `b7d2f07b19295176c5afcd06349c07dcb2f5a4aff5679d2c74f493744f5626e5` |
| Runbook V35 | `a2d6a5c4902830ccb18658820db14a4431c7d9bb01d2b5ba29a1b7dc39610917` |
| Cluster-aware SAE budget admission policy V1 | `5e9f4fc0841d0e71a2eef6f9ac91f5e27d0702c0e0aadfa5a6f511dad50719f2` |
| Cluster-aware SAE budget admission benchmark V1 | `5a2bb38b94c43856bcc9d7e7932db98e3cb89192af60e3418706d33713b6db64` |
| Protocol V44 | `d7771004061a7521fd743da293af5c20d16bd1b8667970e54f7d1f61496e8828` |
| Runbook V36 | `69ed3ae8195993372a58bc02231daaa223181383a472ef110c893ebf08804de5` |
| Empirical-blueprint budget-binding readiness V1 | `08bdaaf4733afeae33d847e25bba0ea5657a2723ed49b58137f64c9904084c7a` |
| Empirical-blueprint budget-binding benchmark V1 | `b95ab40c46c621534b0cf2aed26c342848180a9c5bc7fd48b9674498292159a3` |
| Protocol V45 | `678684a7a898488e3a3c0521ffb8b1de5b6b51912cc6ea0891ec9d8856b18dea` |
| Runbook V37 | `0d5d1dfde0559e4040b40c12d655c95a9f7cf53f102c2350a87b0370ddd7fac2` |
| Cross-artifact graph contract V1 | `a01ac25024b52e64c9b23ad1e3f3203b17403a8164becac168bd1fda7a4d754e` |
| Cross-artifact graph benchmark V1 | `178c9cd1124a4b93c36b1c40746a1776ea09ef820e56f8ad2aa7f0fab3c354fb` |
| Protocol V46 | `02d046536a330c56da723080738277ba22c786bdb3737c6c30bde82aac6dd0a1` |
| Runbook V38 | `ab7b2c5658ef05d4ab9ea25b7f14f577c60b6c740a31d227fc45080a69ca5c33` |
| Semantic-dispatch coverage audit V1 | `bd3cc043f7b658029fd20c99492c43edbace23550a7e237d36d692e46297eaa7` |
| Semantic-dispatch coverage benchmark V1 | `adc1d00bf3be2ab9d790252534bc084add403842b26ae9002e2d6216f0095732` |
| Protocol V47 | `a7bd3d23ad2ecce3b5933dae59b70a7de72212fec103228a149931048b6dc15c` |
| Runbook V39 | `0533d2b242ccb8a566a8dee69146728214584abed4ff6f46a4251529d41fb68c` |

## Exact V1-V8 state map

The full evidence table is in `VPD_STATE_MAP_V1_V8_2026-08-29.md`. In compact
form: V1 and V2 are partly corrupt historical foundations; V3 is partly exact
with its sealed task-3 pack missing; V4 and V5 are exact negative/decomposition
receipts with some raw payloads absent; V6 is negative but scientifically
invalid; V7 is the authoritative corrected negative result with an explicitly
reconstructed validator; V8 is exact deterministic component packetization
and contains no new model evaluation or weights.

## Proposed run ladder

### R0 — allowed now: reproduce current model-free evidence

Run from `C:\projects\VPD`, stopping at the first nonzero exit or identifier
mismatch:

```powershell
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_three_role_capture_to_sae_integration_v1 --verify artifacts/vpd_three_role_capture_to_sae_integration_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_recovered_shape_cluster_aware_sae_capacity_v1 --verify artifacts/vpd_recovered_shape_cluster_aware_sae_capacity_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_five_role_recovered_incidence_audit_v1 --verify artifacts/vpd_five_role_recovered_incidence_audit_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.build_vpd_cluster_to_sae_empirical_preflight_v1 --verify artifacts/vpd_cluster_to_sae_empirical_preflight_v1/preflight.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_empirical_preflight_v1 --verify artifacts/vpd_cluster_to_sae_empirical_preflight_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v24 --verify configs/evals/vpd_activation_category_compartments_v24.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v16 --verify artifacts/vpd_cheap_executor_runbook_v16/runbook.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v25 --verify configs/evals/vpd_activation_category_compartments_v25.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v17 --verify artifacts/vpd_cheap_executor_runbook_v17/runbook.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.build_vpd_cluster_to_sae_empirical_preflight_v2 --verify artifacts/vpd_cluster_to_sae_empirical_preflight_v2/preflight.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_capture_authorization_v1 --verify artifacts/vpd_cluster_to_sae_capture_authorization_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v26 --verify configs/evals/vpd_activation_category_compartments_v26.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v18 --verify artifacts/vpd_cheap_executor_runbook_v18/runbook.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 --verify artifacts/vpd_cluster_to_sae_binding_readiness_v1/readiness.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_binding_readiness_v1 --verify artifacts/vpd_cluster_to_sae_binding_readiness_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v27 --verify configs/evals/vpd_activation_category_compartments_v27.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v19 --verify artifacts/vpd_cheap_executor_runbook_v19/runbook.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_bound_content_v1 --verify artifacts/vpd_cluster_to_sae_bound_content_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v28 --verify configs/evals/vpd_activation_category_compartments_v28.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v20 --verify artifacts/vpd_cheap_executor_runbook_v20/runbook.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_runtime_admission_v1 --verify artifacts/vpd_cluster_to_sae_runtime_admission_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v29 --verify configs/evals/vpd_activation_category_compartments_v29.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v21 --verify artifacts/vpd_cheap_executor_runbook_v21/runbook.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_capture_campaign_guard_v1 --verify artifacts/vpd_cluster_capture_campaign_guard_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v30 --verify configs/evals/vpd_activation_category_compartments_v30.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v22 --verify artifacts/vpd_cheap_executor_runbook_v22/runbook.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_guarded_capture_runtime_adapter_v1 --verify artifacts/vpd_guarded_capture_runtime_adapter_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v31 --verify configs/evals/vpd_activation_category_compartments_v31.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v23 --verify artifacts/vpd_cheap_executor_runbook_v23/runbook.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.plan_vpd_guarded_capture_job_allocations_v1 --verify artifacts/vpd_guarded_capture_job_allocations_v1/manifest.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_guarded_capture_job_allocations_v1 --verify artifacts/vpd_guarded_capture_job_allocations_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_guarded_capture_job_binding_v1 --verify artifacts/vpd_guarded_capture_job_binding_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v32 --verify configs/evals/vpd_activation_category_compartments_v32.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v24 --verify artifacts/vpd_cheap_executor_runbook_v24/runbook.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_guarded_capture_runtime_adapter_v2 --verify artifacts/vpd_guarded_capture_runtime_adapter_benchmark_v2/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_output_reservation_v1 --verify artifacts/vpd_cluster_to_sae_output_reservation_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_runtime_v3 --verify artifacts/vpd_canonical_capture_job_runtime_benchmark_v3/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v33 --verify configs/evals/vpd_activation_category_compartments_v33.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v25 --verify artifacts/vpd_cheap_executor_runbook_v25/runbook.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_capture_router_adapter_v2 --verify artifacts/vpd_capture_router_adapter_benchmark_v2/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v34 --verify configs/evals/vpd_activation_category_compartments_v34.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v26 --verify artifacts/vpd_cheap_executor_runbook_v26/runbook.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_materialization_v1 --verify artifacts/vpd_cluster_aware_sae_materialization_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v35 --verify configs/evals/vpd_activation_category_compartments_v35.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v27 --verify artifacts/vpd_cheap_executor_runbook_v27/runbook.json
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_cluster_aware_sae_materialization_v1.py tests/test_vpd_cluster_aware_sae_materialization_lineage_v27.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_execution_blueprint_v1 --verify artifacts/vpd_cluster_aware_sae_execution_blueprint_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v36 --verify configs/evals/vpd_activation_category_compartments_v36.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v28 --verify artifacts/vpd_cheap_executor_runbook_v28/runbook.json
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_cluster_aware_sae_execution_blueprint_v1.py tests/test_vpd_cluster_aware_sae_execution_blueprint_lineage_v28.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_runtime_worker_v1 --verify artifacts/vpd_cluster_aware_sae_runtime_worker_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v37 --verify configs/evals/vpd_activation_category_compartments_v37.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v29 --verify artifacts/vpd_cheap_executor_runbook_v29/runbook.json
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_cluster_aware_sae_runtime_worker_v1.py tests/test_vpd_cluster_aware_sae_runtime_worker_lineage_v29.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_admission_v1 --verify artifacts/vpd_cluster_aware_sae_admission_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v38 --verify configs/evals/vpd_activation_category_compartments_v38.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v30 --verify artifacts/vpd_cheap_executor_runbook_v30/runbook.json
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_cluster_aware_sae_admission_v1.py tests/test_vpd_cluster_aware_sae_admission_lineage_v30.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_transition_v1 --verify artifacts/vpd_cluster_aware_sae_transition_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v39 --verify configs/evals/vpd_activation_category_compartments_v39.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v31 --verify artifacts/vpd_cheap_executor_runbook_v31/runbook.json
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_cluster_aware_sae_transition_v1.py tests/test_vpd_cluster_aware_sae_transition_lineage_v31.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_inert_attachment_v1 --verify artifacts/vpd_cluster_aware_sae_inert_attachment_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v40 --verify configs/evals/vpd_activation_category_compartments_v40.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v32 --verify artifacts/vpd_cheap_executor_runbook_v32/runbook.json
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_cluster_aware_sae_inert_attachment_v1.py tests/test_vpd_cluster_aware_sae_inert_attachment_lineage_v32.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_inert_callback_v1 --verify artifacts/vpd_cluster_aware_sae_inert_callback_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v41 --verify configs/evals/vpd_activation_category_compartments_v41.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v33 --verify artifacts/vpd_cheap_executor_runbook_v33/runbook.json
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_cluster_aware_sae_inert_callback_v1.py tests/test_vpd_cluster_aware_sae_inert_callback_lineage_v33.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_inert_campaign_v1 --verify artifacts/vpd_cluster_aware_sae_inert_campaign_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v42 --verify configs/evals/vpd_activation_category_compartments_v42.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v34 --verify artifacts/vpd_cheap_executor_runbook_v34/runbook.json
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_cluster_aware_sae_inert_campaign_v1.py tests/test_vpd_cluster_aware_sae_inert_campaign_lineage_v34.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.build_cluster_aware_sae_compartment_resource_reconciliation_v1 --verify artifacts/vpd_cluster_aware_sae_compartment_resource_reconciliation_v1/ledger.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_compartment_resource_reconciliation_v1 --verify artifacts/vpd_cluster_aware_sae_compartment_resource_reconciliation_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v43 --verify configs/evals/vpd_activation_category_compartments_v43.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v35 --verify artifacts/vpd_cheap_executor_runbook_v35/runbook.json
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_cluster_aware_sae_compartment_resource_reconciliation_v1.py tests/test_vpd_cluster_aware_sae_compartment_resource_reconciliation_lineage_v35.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.build_cluster_aware_sae_budget_admission_policy_v1 --verify artifacts/vpd_cluster_aware_sae_budget_admission_policy_v1/policy.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_budget_admission_policy_v1 --verify artifacts/vpd_cluster_aware_sae_budget_admission_policy_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v44 --verify configs/evals/vpd_activation_category_compartments_v44.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v36 --verify artifacts/vpd_cheap_executor_runbook_v36/runbook.json
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_cluster_aware_sae_budget_admission_policy_v1.py tests/test_vpd_cluster_aware_sae_budget_admission_policy_lineage_v36.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.build_cluster_aware_sae_empirical_budget_binding_readiness_v1 --verify artifacts/vpd_cluster_aware_sae_empirical_budget_binding_readiness_v1/readiness.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_empirical_budget_binding_readiness_v1 --verify artifacts/vpd_cluster_aware_sae_empirical_budget_binding_readiness_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v45 --verify configs/evals/vpd_activation_category_compartments_v45.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v37 --verify artifacts/vpd_cheap_executor_runbook_v37/runbook.json
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_cluster_aware_sae_empirical_budget_binding_readiness_v1.py tests/test_vpd_cluster_aware_sae_empirical_budget_binding_readiness_lineage_v37.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.verify_cluster_aware_sae_cross_artifact_graph_v1 --verify-contract artifacts/vpd_cluster_aware_sae_cross_artifact_graph_contract_v1/contract.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_cross_artifact_graph_v1 --verify artifacts/vpd_cluster_aware_sae_cross_artifact_graph_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v46 --verify configs/evals/vpd_activation_category_compartments_v46.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v38 --verify artifacts/vpd_cheap_executor_runbook_v38/runbook.json
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_cluster_aware_sae_cross_artifact_graph_v1.py tests/test_vpd_cluster_aware_sae_cross_artifact_graph_lineage_v38.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.audit_cluster_aware_sae_semantic_dispatch_coverage_v1 --verify artifacts/vpd_cluster_aware_sae_semantic_dispatch_coverage_audit_v1/audit.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_semantic_dispatch_coverage_v1 --verify artifacts/vpd_cluster_aware_sae_semantic_dispatch_coverage_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v47 --verify configs/evals/vpd_activation_category_compartments_v47.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v39 --verify artifacts/vpd_cheap_executor_runbook_v39/runbook.json
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_cluster_aware_sae_semantic_dispatch_coverage_v1.py tests/test_vpd_cluster_aware_sae_semantic_dispatch_coverage_lineage_v39.py
```

Expected IDs are pinned above. Passing R0 grants no further authority.

### R1 — completed locally: five-role read-only incidence audit

The new content-addressed audit passes 15/15 controls with ID
`c14b80e53ea6fe39e171fbd201ab0a87b86dc39737c091fd83616bb813d2cefc`.
It exactly reconstructs 25,168 rows, 2,288 pairs, and 16 template groups from
five pairwise-disjoint roles. `calibration` and `final_evaluation` have
distinct hash-derived identities under the common semantic parent. Final
evaluation is mechanically forbidden for selection, routing, and dictionary
training. The audit performs no selection, routing, inference, or training.

### R2 — completed locally: empirical preflight packet only

The preserved V1 preflight template and its benchmark are complete. V1 is
blocked on 13 named fields and ID
`edf41948607ac8471030545ea9782c1f6ea76dd2aec72f09aa7cffac5fb45f91`.
The benchmark passes 15/15 controls with ID
`03f8f0e82283b82b28997403a4f68bcd84d6d9630e1a23cc825ccbb805b28e13`.
It binds the three partition-selection roles plus the later `sae_training`
routing extension. The full ceiling is 64 packet jobs, 41,352,192 batch
forwards at batch size four, 164,900,736 episode-forward equivalents, and
4,046,192,640 bytes of float32 effect payload before container/runtime
overhead. Wall time, model memory, and cost remain unestimated. No model
framework or weights were opened and complete synthetic bindings still do not
authorize execution. A later audit found that V1 included the future signed
receipt ID/hash inside the preflight that receipt must sign, creating a
content-address cycle. V1, Protocol V25, and Runbook V17 remain preserved as
gap evidence and are not current execution authority.

Cycle-free preflight V2 has ID
`5465f17d583a0dd19f5c4ea4939883ac8f65b7a19e128f7ad130246eb2a816bf`.
It separates ten execution-identity bindings from the later one-way signed
authorization receipt. The authorization benchmark passes 16/16 controls with
ID `6d359a0c2608afe5f4d5bd8366ec4afe53a667b7be497cb521b814b22fee0f18`.
It verifies Ed25519 signatures, exact identity/scope/ceiling binding, time
windows, authority status, revocation, and nonce replay protection. Synthetic
clean signatures remain non-authorizing, and no private key is persisted.

Protocol V47 and Runbook V39 are the current successor lineage. They preserve
V46/V38 and all earlier evidence. V45/V37 bind exact reconstructed resource and
terminal-failure reconciliation (`24/24`) to the fail-closed budget-admission
policy (`32/32`) and the explicit six-phase/50-binding readiness worksheet
(`29/29`). V46/V38 add a content-addressed 20-node, 37-edge, 10-identity-binding
cross-artifact graph gate whose temporary synthetic benchmark passes `34/34`;
the combined focused suite passes `6/6` in 115.42 seconds. The verifier rejects
file/content-address, edge, policy, binding, traversal, symlink/junction,
synthetic-boundary, and cleanup failures. It deliberately does not invoke
node-specific semantic verifiers or authenticate signatures, authority status,
nonce freshness, output exclusivity, execution, or training. `safe_stage_now`
remains 0. A subsequent semantic-dispatch coverage audit found that V46's graph
is structural-gap evidence, not a valid semantic dispatcher contract: only
12/20 node schema/ID pairs match implemented artifacts, only 4/20 nodes are
dispatch-ready under that representation, and only 9/37 field edges match;
28 edges differ. Its fail-closed benchmark passes `23/23`, and the focused audit
plus V47/V39 lineage suite passes `6/6` in 1.71 seconds. The mismatches include
capture receipt/manifest and routing bundle IDs, embedded plan/split content,
missing blueprint binding in budget V1, signed compute receipt representation,
one combined durable transition, and the absence of a production runtime-
attachment verifier. Semantic dispatch is therefore disabled pending a
corrected successor. Campaign
admission requires all eight categories, all 17 compartments, all 34 jobs, and
34 distinct output directories before its durable claim is created. Campaign
and per-job claims precede callbacks; the all-success path charges 34, and the
controlled failure path charges eight and invokes none of the remaining 26.
Only inert synthetic callbacks ran; production campaign execution and training
performed remain false. The current 64-job capture allocation remains a
rehearsal tied to the incomplete preflight and must be regenerated after all ten
bindings are filled. Actual recovered-shape capture plans, empirical V3
captures, persistent routing evidence, an empirical cluster-aware blueprint,
real signed receipts, a workspace transition/claim, production runtime
attachment, production trainer invocation, and dictionary output remain absent.
The resource ledger records 68,000 planned steps and 20,480 planned activation
rows, keeps 4,096-wide specific dictionaries separate from the 8,192-wide
shared fallback, and explicitly states that raw receipts were not recovered.
The budget policy separately caps every category, specific compartment, shared
fallback, and global aggregate. It rejects widened ceilings, rebinding,
dictionary-width substitution, output reuse, double counting, and authority
broadening. An admitted reference shape is not empirical evidence or permission
to consume resources.
The empirical binding worksheet reports 0 supplied and 50 missing bindings.
Even a syntactically complete fixture remains unopened and does not establish
cross-artifact content, budget binding, signed authority, or durable transition.
Readiness of any local control does not authorize model loading, inference,
capture, routing, materialization, or SAE training. V24/V16 through V41/V33
remain exact predecessors.

The V39 cheap-executor task is strictly local and reviewable: verify the
semantic coverage audit, its 23-control benchmark, Protocol V47, Runbook V39,
and the focused tests; report exact IDs; then stop. It must not coerce the V1
graph into the implemented interfaces, fill missing values, create a workspace
claim, load a model, or write a dictionary. The next safe implementation step
is a corrected versioned semantic graph based on guarded V3 capture receipts,
routing evidence, embedded plan/split materialization, blueprint-bound budget
admission V2, signed receipts, and the combined durable transition. A production
runtime attachment remains explicitly absent. Real trainer invocation
would still require a fresh empirical
blueprint, active signed receipts, new nonces and output reservations, explicit
user authorization, and a separately approved production runtime.

### R3 — blocked: empirical four-role staged capture

Requires a separate explicit user authorization naming all ten execution
bindings plus a separately signed, active, scope-exact authorization receipt
from a real enabled authority registry. Capture only zero-action prompt-window effects, in gate order:
`discovery`, `calibration`, `cluster_validation`, then the `sae_training`
routing extension after partition admission. Do not use `final_evaluation`;
do not train an SAE.

### R4 — blocked: empirical routing and SAE manifests

Only after authenticated R3 receipts exist: run V4 partition admission,
independent discovery/cluster-validation stability, the router, and Planner
V2. Reject any identity, overlap, category, condition-stratum, membership, or
minimum-support mismatch. Never duplicate a full category archive into each
cluster.

### R5 — blocked separately: compartment SAE candidates

Train only from an explicitly authorized, passed R4 manifest. Freeze dictionary
width, sparsity penalty, learning rate, seeds, resource ceiling, and output
paths in advance. Shared and category-cluster dictionaries remain separate.
Dictionary quality and held-out interventions must pass before any scientific
claim. SAE training is weight mutation and is not authorized by this report.

### R6 — design ready, execution blocked: V9 authorization retraining

Use a fresh, versioned, pair- and template-disjoint held-out pack. Labels must
come only from observable trusted `AUTH_STATUS`: invalid or untrusted host state
trains `REQUEST`; valid trusted state trains `PROPOSE`. Never train against
hidden/conflicting labels. Compare four frozen candidates: unchanged LoRA
control, fresh LoRA SFT, fresh LoRA margin objective (primary), and a
historical-adapter repair diagnostic. Pre-register invalid `REQUEST`, valid
`PROPOSE`, separation, paired adapter-minus-base delta with 95% CI, category
minimums, and hard zero-action invariants. Reject release unless all hard gates
pass on the untouched pack. Do not rerun V7 merely to reproduce its negative
result.

## Cheaper-model stop rules and receipt

Stop immediately on a file/hash mismatch, failed verifier, network/package
request, model-framework import, model/adapter load, inference/GPU work,
weight/dictionary write, missing authorization receipt, or need to fabricate
empirical input. Do not repair provenance by relabeling successor or
reconstructed files as recovered originals.

For every command return: command text, exit code, observed ID/hash, files
created or changed, boundary flags, and first stop reason. The machine-readable
companion is `VPD_CHEAP_MODEL_RECOVERED_SHAPE_RUN_PACKET_2026-09-01.json`.
