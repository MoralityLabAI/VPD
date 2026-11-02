# Bonsai-8B VPD authorization state map — 2026-08-29

## Status vocabulary

- **Exact:** the local file matches a recorded authoritative SHA-256.
- **Recovered exact:** restored byte-for-byte from a local recovery source or
  deterministically regenerated and accepted only after matching the recorded
  SHA-256.
- **Authoritative receipt only:** the report/manifest/hash survives, but some
  payload bytes are missing or corrupt locally.
- **Reconstructed:** behavior or content was rebuilt from surviving evidence;
  it must not be represented as the lost original.
- **External/missing:** referenced outside this rebuilt project and not present
  at the historical path.

## Version map

| Version | Scientific role and outcome | Current executable/data state | Authority boundary |
|---|---|---|---|
| V1 | Initial authorization harness and VPD dataset. The split manifest records 448 rows: 24 development pairs/336 rows and 8 task-3 pairs/112 rows. | Source, manifests, reviews, and held-out pack survive. The held-out pack is exact (`38793540...`). The combined and development gated packs do not match the split-manifest hashes and are malformed/corrupt. | Historical foundation only; not release evidence. |
| V2 calibration | Prompt/contract calibration over 448 rows with the same 24/8 pair split. | Manifests, reviews, and held-out pack survive. The held-out pack is exact (`cac957d8...`). Combined and development gated packs do not match their recorded hashes and are malformed/corrupt. | Calibration evidence only. |
| V2 confirm | Larger confirmation family: 1,680 rows, 90 development pairs/1,260 rows and 30 task-3 pairs/420 rows. | Reports/manifests survive. Combined, development, and held-out gated packs all differ from recorded hashes; the combined pack is malformed. | Historical confirmation family, not a substitute for V7. |
| V3 | Four-token authorization decision contract (`PROPOSE`, `REQUEST`, receipt acknowledgments); 90 development pairs and a sealed 30-pair task-3 pack. | The manifest and development pack were recovered exactly from `bonsai8b_authz_runner_stage.tar.gz`; development SHA-256 is `d1ccd4bd...`. The task-3 pack (`be659850...`) is absent. | Task-3 remained sealed in V4/V5. Missing held-out bytes prevent a complete local replay. |
| V4 | Grouped-logit VPD experiment over 90 development pairs. Found adapter-wide PRO bias (`+0.2145` invalid-visible margin) but 0/7 prespecified component groups passed. | Exact report and derived review material survive; the seven raw `remote/group_*.jsonl` outputs are absent. | Controlled development null; task-3 was never staged. |
| V5 | Full hierarchy decomposition of all 4,032 LoRA rank-one components; 252 projections; selected 144-component `self_attn.k_proj`, layers 18–26 cell. The cell failed discovery and internal validation. | Group plan and both summaries are exact. Internal-validation raw output is exact. The discovery raw output and 4,032-row component-evidence file are corrupt/mismatched locally. Adapter, base projection inventory, and component inventory are external/missing. | Definitive VPD decomposition receipt is the exact report/manifest; no positive authorization mechanism and task-3 remained sealed. |
| V6 | Independent-action behavioral evaluation. Negative result, but scientifically invalid for revoked/replayed conflicting labels and ambiguous expiry. | Exact report, config, validation report, analysis, visibility audit, remote logits/receipt/log survive. `run_provenance.json` is missing; `FREEZE.json` and gated pack mismatch, and the gated pack is malformed. Six transfer chunks total 174,655,536 bytes but concatenate to `035901ad...`, not the recorded adapter hash, so they are not a valid checkpoint reconstruction. | Preserve as historical flawed evaluation. Do not use its hidden-state labels as training/evaluation truth. |
| V7 | Corrected observable `AUTH_STATUS` evaluation. Definitive negative: base separation `0.692347`, adapter `0.580743`, delta `-0.111604`, CI `[-0.122397,-0.100996]`; invalid REQUEST `0%`; valid PROPOSE `100%`. | Generator, tests, manifest, reports, validation receipt, logits and receipts are exact. `episodes.jsonl` was regenerated exactly (`b3ca4500...`); preflight JSON was recovered exactly (`66df639b...`). Historical gated pack is corrupt (`e6c58554...`, expected `70b5b3...`). A semantic gated reconstruction (`f0b51964...`) passes the reconstructed validator on 960 rows/96 pairs with zero conflicts, zero execution, and 1,260 exact recovered V3 development rows checked. The original validator was not found; reconstructed source hash is recorded separately and is not the lost `1cafe45...` file. Several result-manifest payloads are missing/corrupt, but the exact `FINAL_VERIFICATION.json` records the original 21-entry, zero-mismatch verification. | Authoritative current authorization result. Reject both adapter and base for authorization-control release. V7 trained no weights. |
| V8 | Deterministic clustering/packetization of the V5 4,032-component hierarchy into 16 VPD-SAE work packets. No new model evaluation or weights. | Manifest, assignments, clusters, and all 16 packet hashes match. Status is complete. `effects_sha256` is null because the source is hierarchy metadata/joint evidence rather than a new per-component effect tensor. | Derived organizational artifact only; it does not override V7 behavior. |

## Recovery actions performed in this continuation

1. Searched accessible recovery trees, the original ZIP, staging archive, public
   indexed web results, and configured GitHub access for the V2 validator.
   No exact source was found. GitHub CLI authentication for `patrickdugan` is
   expired/invalid, so authenticated repository history could not be queried.
2. Reconstructed `validate_independent_authority_pack_v2.py` from the exact V7
   generator/tests and frozen V7 contracts. Its source explicitly states that
   it is reconstructed. It passes all three V2 regression tests.
3. Preserved the corrupt V7 episode and gated files under
   `artifacts/recovery_2026-08-29/corrupt_originals/`.
4. Regenerated V7 `episodes.jsonl`; its bytes match the recorded hash exactly.
5. Restored V7 preflight JSON, the V3 manifest/development pack, and
   `schemas/run_contract.schema.json` from exact local recovery copies.
6. Generated a separate, non-original V7 gated pack and validation receipt.
   It is never labeled as the frozen V7 pack and does not replace it.

## Verification receipts

- Repository cheap suite after recovery and activation-category protocol work:
  `208 passed, 3 subtests passed` in `1766.06s` with `python -m pytest -q`
  after adding focused-control research protocol V7, unified lineage V5, the
  complete assignment-derived same-compartment V2 planner/executor, training
  execution research protocol V8, and unified lineage V6. The preceding
  complete-control checkpoint was `202 passed, 3 subtests passed` in
  `1369.25s`. The preceding training-release
  gate checkpoint was `190 passed, 3 subtests passed` in `696.64s`. The earlier
  campaign V4/seed-corrected-worker checkpoint was
  `184 passed, 3 subtests passed` in `684.99s`; its focused campaign and
  partition-audit capture suite passed `15` tests plus `3` subtests in
  `246.49s`.
  Research protocol V6 and unified lineage V4 both reproduce. The earlier
  `169 passed` / `423.46s` V5 checkpoint remains part of the historical record.
- Model-free activation-category routing benchmark: pass, 13 scenarios,
  content ID `eb6ea4b61c17bd124a784b1efe532eb94f7fd79a3743725c284be296e48635c0`.
  The cross-archive gate is now stratified as well as aggregate: pair-block
  direction, observable-condition, and semantic-template coverage must hold in
  discovery and independent calibration before a component counts as stable.
  Three discovery-only artifacts affect 8/96 specific components and are
  contained by component rerouting while remaining above the 90% replication
  floor. Three calibration-only counterparts affect 12/96 and reject the
  bundle below that floor. Direction reversal, category swap, and a 2%
  calibration-magnitude collapse also reject as designed.
  A minority-disagreement scenario leaves global replication at 88/96
  (`0.916667`) but proves that all eight individually unstable components are
  rerouted to `shared_cross_category`; global calibration no longer admits its
  unstable minority to category-specific SAE compartments.
- Production-shaped within-category clustering-geometry benchmark: pass,
  content ID `7098172918613a6f74e1f2d4032a5d720440dc654d1465cbc73f25d416c07428`.
  At hidden size 4,096, the previous unweighted normalized directions fall to
  worst-seed pairwise F1 `0.565022` and cluster purity `0.703125` under tiny
  random non-dominant effects. Square-root category-strength-share weighting
  reaches `1.0` on both measures across all three scenarios and is now bound
  into production bundle manifests. This is synthetic protocol evidence and
  does not establish a Bonsai cluster count or mechanism assignment.
- Production-shaped per-category cluster-count benchmark: pass, content ID
  `32ced135dd8ee66af5a1fd56f592c5f407da0a7636f050c75c5a9c3febaaa996`.
  Thirty-two discovery/independent-validation projection and pair-block views
  recover known one-, two-, and three-mechanism fixtures. An aggregate-only
  sampling artifact and a clean but cross-archive membership rearrangement
  both fall back to one cluster. Categories without a stable subdivision likewise
  fail closed. Production bundles bind this selection; manual totals are marked
  unvalidated ablations. This does not choose empirical Bonsai counts without
  capture. The required independent `cluster_validation` split role is now
  generated, content-addressed, and enforced by the bundle builder, but no
  Bonsai activation archive has been captured for it.
- Five-role model-free split receipt: pass. The 1,056-row protocol split at
  `artifacts/vpd_activation_category_split_v3` has ID
  `7f8157155dc5cb6d816b09a91b4d0ec6114ba0499022633aaf4160b3ffe575d5`.
  It partitions 96 pairs and 16 whole template groups across `sae_training`,
  `discovery`, `calibration`, `cluster_validation`, and `final_evaluation`
  with exact pair, episode, and template disjointness.
- Five-role capacity split receipt: pass. The 25,168-row split at
  `artifacts/vpd_activation_category_capacity_split_v4` has ID
  `a5aa3370406876c1c48ed702b431a64cb6a68c8376ab28b0aad1d852c22dd8dc`.
  It partitions 2,288 pairs and 16 whole templates across the same five roles;
  only `sae_training` is used for dictionary fitting.
- Unified capacity-scale three-archive effect-capture campaign plan: pass,
  content ID
  `324dac30e39fa46305ede80eeb00ce5037f219133e2db66e8128a15e506ef6b8`.
  It freezes 48 exact packet/role compute jobs, 48 per-worker signing jobs,
  three signed exact-coverage merges, and one production 32-view clustering
  job. The scope is 114,162,048 episode-equivalent or 28,667,520 batch-size-four
  forwards and approximately 3.11 GB of float32 effect payload before
  container/runtime overhead. It performed and authorizes no model inference.
  Discovery and calibration are separated by two content-addressed,
  model-free continuation gates; any rejection prevents later-stage compute.
  This V3 plan is preserved as exact predecessor evidence but is not executable:
  its worker would apply integer modulo directly to a hexadecimal string while
  deriving the CountSketch seed. The current corrected candidate is campaign V4,
  ID `ffcc7351bade510166c449099f19d43bd1a28a89e83bc85bc912f883c742b528`.
  It preserves V3 and the original worker hash, routes all 48 compute jobs
  through a versioned seed-correction wrapper, and re-addresses every dependent
  job. Neither campaign performed or authorizes model inference.
- Model-free unified effect-campaign stage-gate benchmark: pass, seven scenarios,
  content ID
  `9a12ecaed95889d139c1bf0e504e7172090f17941622d5935752262bc3ec92b4`.
  Clean discovery and calibration continue; insufficient capacity, a
  condition artifact, category reassignment, direction reversal, and
  magnitude collapse reject. The campaign plan binds this receipt. It makes
  no empirical Bonsai gate or cluster claim.
- Production-shaped effect-sketch fidelity benchmark: pass, 3,584 dense and
  sparse cases at hidden size 4,096 over eight seeds, content ID
  `4ed9790051cabd86dd4b2faffd68d57bfd1a8fbead2e8ea0df0b107ab29e04dd`.
  It rejects the prior 64-dimensional setting and selects 256 dimensions under
  Wilson-bounded false-admission/recall plus p99 metric-error requirements.
- Model-free compartment-SAE screening benchmark V2: pass, three scenarios,
  content ID `8f06090f64db0a1fadcaec50e2f14bf73eff3fe8e3e372de836dd55f8a501b12`.
  It proves on paired synthetic activations that specific compartments select
  observable category membership rather than the confounded global behavior
  outcome, while shared screening keeps the behavior endpoint. Held-out
  interventions still measure the signed authorization decision margin.
- Model-free compartment-SAE quality benchmark: pass, six scenarios,
  content ID `01d7c090a7af3f2ffe2046cf9474cb849b6774128172af28fc747b3620370d38`.
  It uses the production `[dictionary_feature, hidden]` decoder convention;
  clean replicas pass, four corruption families reject, and bounded streaming
  reproduces the in-memory gates. The production-shaped sharded fixture covers
  eighteen dictionaries and the frozen 65,536-row calibration sample. This is
  protocol evidence only and makes no empirical Bonsai SAE claim.
- Model-free activation-capture semantic-admission benchmark: pass, seven
  scenarios, content ID
  `51ebf5afb2cf0e1cc109ed8a7672d9ff8f0adb7c65fb4e442128164fbc5f0964`.
  Exact source order/category/window metadata passes; reordered or duplicated
  episode windows, relabeled observable categories, shifted token coordinates,
  substituted source-episode digests, and misstated sampling identities reject.
  Training, dictionary admission, and calibration quality require the
  content-addressed semantic receipt. This benchmark is synthetic and performs
  no model inference.
- Sharded capture resource receipt: pass. `sae_training` has 805,376 rows/197
  shards and `calibration` has 604,032 rows/148 shards per layer, for 1,409,408
  rows total; content ID
  `c99dd5c6e06bc5408aedc8df468d2b5d98a5825b73ca8e52fe6969b3b13d6ee0`.
  It binds role-specific counts under the same zero-generation prompt-window
  storage contract for `sae_training` and `calibration`; `final_evaluation`
  remains forbidden.
- Unified compartment SAE training plan: pass, content ID
  `d4ef3eca0c642d22f3df415bacb61d5a2130fed969570eb6836777589d1301dd`.
  It freezes two capture jobs, eighteen training jobs over all nine
  compartments, initialization seeds 17/29 with a common data-selection seed
  17, one exact dictionary-bundle admission job, and one calibration quality
  job. Quality scoring now requires the content-addressed admission receipt;
  the admission step verifies the entire planned grid but makes no quality
  claim. It performed no capture or training and still requires a frozen model
  identity plus explicit compute authorization.
- Compartment SAE training release gate: pass, content ID
  `7109adb38749c4c0e91154df0fddfe5fb79c625972f3ff89eca06ffda39596b9`,
  file SHA-256
  `1ac02aa361dcfe40c4cb0925d127249d1e14ece47e4b0918b6fe57771d8a0d85`.
  It preserves training-plan V3 as predecessor evidence and guards all 22
  blueprint jobs behind one reproducible scientific release receipt. The
  receipt must join the exact frozen category bundle and component census, a
  passing fresh-pack partition audit, gate/pack identities, activation space,
  model/revision/adapter measurement identity, protocol, and training
  blueprint. A receipt still records `compute_authorized: false`; no runtime
  receipt currently exists and direct execution of V3 is prohibited.
- Model-free training-release benchmark: pass, nine scenarios, content ID
  `f706337b2141574a537a013c8d04ea2ec28f5c4e8fac5f4a580b75f9c00b90a3`,
  file SHA-256
  `dbc099598d9fff422e92ba0eb98344a00d33b48147135dbf0bfdeea6df0739a3`.
  One clean synthetic handoff emits a non-authorizing receipt. Missing or
  rejected audit, wrong bundle, component-census mismatch, stale gate plan,
  stale fresh pack, shared-only candidate, and wrong effect space all reject
  before release.
  It performs no model inference or training.
- Fresh post-freeze partition-audit pack: pass, pack ID
  `e65c78bb07fcf25c5a008d94a8f447b1b75f01cb8734632e7e2e567c81ca2f23`.
  It has 2,816 candidate-only rows, 256 pairs, and four new semantic templates.
  Every non-missing activation category has 256 observations; the
  missing/untrusted category has 1,024. Validation found zero episode, pair,
  exact-message, or semantic-template overlap against 25,168 capacity-pack
  rows. Gate evaluation was deterministic and recorded zero executions and
  zero exposed execution paths. No model outcome was captured.
- Frozen-partition audit benchmark and gate plan: pass. Benchmark ID
  `f817cecf0833e66b9013b311eecbcad72f8e8091d0226094bae58bfed6f16a6f`
  detects category mismatch, signed-direction reversal, magnitude collapse,
  and stratified-robustness failure while forbidding shared-to-specific
  promotion. Gate-plan ID
  `e6762af41fcce70bca47b1a6925c9e3b53b8a88e65aee946c5c7451054818358`
  binds 16 work units, all 4,032 components, and 22,708,224
  episode-equivalent forwards. It is post-freeze, reject-only, and does not use
  `final_evaluation`. Capture is not authorized, and the versioned capture
  adapter for the new role is explicitly still unimplemented in this preserved
  V1 plan.
- Partition-audit Gate-plan V2 and capture boundary: pass, content ID
  `1341c8f6371fab72cd9e8f8be92b1e740ae47f08c71a77a8bab39b04fe2a119b`.
  It preserves the 16-work-unit/4,032-component census and 22,708,224
  episode-equivalent forwards, and adds exact compute, pre-signing attestation,
  signing, dedicated signed merge, and production evaluation commands. The
  adapter SHA-256 is
  `a662e0c770a07d8e0064f0f7847e802ece57e34292a980f14d552a58f5bbd7f5`;
  the signed merge front-end is
  `e5f2c72ba9b1c4415fac7e5b36fae6df6791e6980d9d2e3ab652ad45e1d362c1`;
  and the production evaluator is
  `ae4c4bb8630d968159b6c3cbcf2a5161a0d4cae95033123d19df46d6cf47f33b`.
  The preserved effect worker and base merge/signer were not changed. Five-role
  split provenance is forbidden, fresh-pack/work-unit provenance is signed,
  and exact signed coverage is required. Capture packetization is now explicitly
  independent of category assignment: the hash-bound metadata bundle supplies
  exact capture packets, while the future frozen category-aware bundle supplies
  assignments and discovery lineage. Component sets must match; packet IDs need
  not. Packet signing is also exact: five jobs are assigned to `worker-a`, five
  to `worker-b`, and six to `worker-c`; compute commands freeze those IDs and
  signing commands use three separate worker-key placeholders tied to enabled
  registry entries. The old global worker/key placeholders are forbidden. The
  verified corrected campaign V4 is now an explicit input. Its content ID is
  `ffcc7351bade510166c449099f19d43bd1a28a89e83bc85bc912f883c742b528`;
  maximum episodes, seed, epsilon, batch size, uncertainty settings, effect
  space, and `float16` model-load dtype are derived from its content-addressed
  parameters. `DTYPE` is no longer a runtime placeholder. The plan still has
  unresolved identity, device, credential, and output placeholders and
  authorizes no execution.
- Model-free signed partition-audit pipeline benchmark: pass, seven scenarios,
  content ID
  `9efa073d053ee89551c847f09af0b56b450f134b333a88d80ae8ed5ee5d7ec89`.
  A clean fixture with distinct capture/category packetizations passes after
  attestation, Ed25519 verification, exact signed merge, and production report
  verification. Category mismatch and observable-condition robustness failure
  reject for the intended reasons; the wrong capture packet bundle rejects.
  Wrong model and adapter identities reject at the production discovery/audit
  join, while wrong effect space rejects during pre-signing campaign
  attestation. Five
  complete signed fixture receipt sets were merged and one was rejected before
  merge. The shared identity freezes model, revision, adapter, runner, effect
  space, estimator, perturbation settings, category contract, and robustness
  contracts while allowing the audit dataset and split role to differ. All
  effects are synthetic and no model inference was performed. The fixture now
  uses the production 4,096 hidden width and 256-dimensional CountSketch. Each
  signed V2 receipt binds a common campaign measurement-contract ID and its
  packet-specific command hash. Receipt-observable numeric fields must match;
  batch size, maximum episodes, and model-load dtype remain explicitly
  gate-command-bound. Wrong effect space rejects before signing.
- Activation-category protocol V5: pass, SHA-256
  `0894ac8e06ba7828bf5cb21ae79e1ae48ae96e2c27cd66e6853899765c1b9c24`.
  It preserves V4 as its predecessor and blocks category-specific SAE training
  until every frozen specific component passes the fresh partition audit.
  Failure may not tune a candidate on the same pack; a successor requires a
  fresh version and seed. This is protocol evidence, not an empirical Bonsai
  partition result.
- Unified V5 lineage amendment: pass, content ID
  `bec00550d7d523f4ece993aafdac61f203a88c4417d8cab7bafc3ef0752831e2`.
  It reverifies the V2 unified receipt, preserves the original five-role split,
  effect campaign, and SAE plan, and binds the V5 protocol plus the separate
  audit pack, benchmark, and gate plan. The receipt explicitly records that the
  capture adapter is still required and authorizes no execution.
- Activation-category research protocol V6: pass, SHA-256
  `c5d9a47f651fcf654a9b33780c0c556e681a80cf4379a1684c65dba985e2b84a`.
  It preserves research protocol V5, binds Gate-plan V2 and the adapter/merge/
  evaluator hashes, and keeps category-specific SAE training blocked until a
  signed fresh-pack audit passes. This protocol label is distinct from the
  historical Bonsai experiment V6. It performs and authorizes no compute.
- Unified research-protocol V6 lineage amendment: pass, content ID
  `e12976b0161b3c947006fb785c5cf598e63a6620f4cd1563b3690e2769af6f75`.
  It reverifies unified lineage V3, research protocol V6, and Gate-plan V2;
  records 16 prepared capture job groups; binds the seven-scenario signed
  pipeline benchmark; and marks the capture path
  `implemented_locally_not_executed`. No runtime signed partition-audit receipt,
  empirical Bonsai category partition, or audit report was created.
- Unified effect-cluster-to-SAE lineage receipt: pass, content ID
  `12dea48a1928ec3780f31419aed772f9584962216391bc10aabc5d2ab753ffd3`.
  Protocol V4 SHA-256
  `ff8e42728f77e21c5d61d6425858e16159e0cef887959052ed2de50699c41822`,
  effect campaign V3, and SAE training plan V3 share exact split
  `a5aa3370...` and five distinct semantic role-set hashes. The historical
  small-pack campaign and capacity SAE plan are intentionally not joinable.
  This is local provenance evidence and performs or authorizes no compute.
- Focused SAE packet/effect-lineage benchmark: pass, eight scenarios, content
  ID `72a45142b1817269e4b991556585ad9e139c02290c31753edcc994d87ec783fe`.
  Exact archive/manifest lineage passes; either substitution and any category
  packet without lineage reject. Legacy metadata-only packets are explicitly
  unattested and cannot support empirical category claims. The independent
  signed-result reducer also rejects substituted signed inputs and misstated
  lineage receipts.
- Focused SAE same-compartment control benchmark: pass, eight scenarios,
  content ID
  `57f3f9a612883b04bb9651ca8de3fee7f17a6b0996b7eb88ee7d85d22a84ed25`,
  file SHA-256
  `b716c819b7ba22598cf489cfc8a830a88bf8f32366657ea12a2d95e29d78b98c`.
  The clean fixture proves that V2 expands the predecessor packet's truncated
  first-32 category list to the complete assignment-derived same-compartment,
  out-of-packet pool and can select a better control absent from that packet
  list. The predecessor list is retained only as metadata. The shared fixture
  also passes. A truly empty complete pool, cross-category predecessor pool,
  target-category mismatch, target/control overlap, missing full-pool effects,
  and predecessor tampering reject. The V2 executor independently rederives the
  complete pool and recomputes the numeric norm/alignment choice from the exact
  dictionary and effect archive before delegating to the preserved worker.
- Activation-category research protocol V7: pass, SHA-256
  `3e584f718df415cf4307511da5a04ea1ccc425695dc2a5bfb0360ce9617c94f7`.
  It preserves research protocol V6 and binds the V2 focused planner
  (`459ba8e0...`), V2 executor (`d5b1f200...`), and same-compartment benchmark.
  This research label is distinct from historical Bonsai experiment V7.
- Unified research-protocol V7 lineage amendment V5: pass, content ID
  `aedb91d6a90700517005c517ef29acbc3f87594fe82695adeb6ae10d3af639a4`,
  file SHA-256
  `994032901b16cdf1062b03f15183cfa6bafff333cc5a1503693854d2ed7b4a0b`.
  It reverifies lineage V4, protocol V7, the same-compartment benchmark, and
  the 22-job SAE training-release gate. No empirical category bundle, focused
  V2 plan, model inference, or intervention was created.
- Compartment SAE training-execution blueprint V2: design-only pass, content ID
  `33bc09fac66fcec62bc0374c39d7ad4209a5d17eafd9b1003b4353073687e40e`,
  file SHA-256
  `167bd226158f47279fecd6c15364f63cd422e3ca7f73f12c9d4c7e4c4b650793`.
  It recomputes exact category, episode, prompt-pair, and 128-token-window
  identity hashes for all 18 planned replicas from the frozen `sae_training`
  split. All eight specific contracts contain zero cross-compartment rows. The
  shared contract contains 572 whole episodes per category, 4,576 episodes and
  585,728 activation rows total.
- Training-execution contract benchmark: pass, eight scenarios, content ID
  `08255dba491199e4963080c8e059c8ebd0fdddaffb3e4fbd70c781a8685b2ff4`,
  file SHA-256
  `06afd8c4e630e4e0affd9e736817684dd6bb23ae1de9e2672adbbc1d69332ed2`.
  It reproduces that the predecessor V1 bundle accepts a synthetic record whose
  reported batch size and learning rate differ from the frozen plan, because
  those fields are absent from the V1 dictionary manifest. V2 rejects changed
  batch size, learning rate, sample identity, specific-compartment purity,
  shared balance, and job coverage.
- Activation-category research protocol V8: pass, SHA-256
  `b66111e7d5e7001ddef5dadefd97eaa0f6ef3bdd658699d6ad3add72e68fab49`.
  It preserves research protocol V7, requires a future dictionary manifest V2
  to bind every planned hyperparameter and the exact sample contract, and makes
  explicit that the current V1 release gate is not sufficient by itself. This
  research label is distinct from historical Bonsai experiment V8.
- Unified research-protocol V8 lineage amendment V6: pass, content ID
  `f4391a7bed1c8047d2b4ea08247b667a53619586f3931802b4146dee0d34187f`,
  file SHA-256
  `242e0bdea2b39de5e4b00370a000f022e1209fa9f1de7d85813990b52a1d3a2b`.
  It reverifies lineage V5, protocol V8, the 18-job execution blueprint, and the
  eight-scenario benchmark. Training plan V3 and its V1 implementations remain
  predecessor evidence. A versioned empirical runner/admission is still
  required; no capture, training, dictionary, checkpoint, or weight mutation
  occurred.
- Guarded compartment SAE training plan V4 / runtime schema V2: local design
  verification pass. Plan content ID
  `a7a050abead2d60a63e16a6d07126609ac38c034f18ad3e37f98cd7c9f4d8c2b`,
  file SHA-256
  `f90dc761017b15369fbc31bf8d9efdee4f4d8f6e9c7fce585e8ecc5e94b5975c`.
  The plan has exactly 22 jobs: two capture, 18 training, one bundle admission,
  and one quality evaluation. Dictionary manifest V2 binds the frozen batch
  size, learning rate, exact sample contract, capture execution, training
  execution, scientific release, and external exact-job compute authorization.
  This runtime is reconstructed implementation, not recovered historical code.
- Compartment SAE release Gate-plan V2: local design verification pass, content
  ID `5d821b591fa46ee306d335efc9df10e5a0db3a301482d13d54caeaa0d42e575c`,
  file SHA-256
  `b6b369258a928621516e56b92c7625b58bf73279309a41c1affc4c4ca09ec417`.
  All 22 jobs require both a scientific-release receipt and a separately
  supplied authorization covering the exact V2 job ID. The release receipt is
  deliberately not compute authorization, and the repository does not create
  the external authorization.
- Model-free compartment SAE runtime benchmark V2: pass, 11/11 scenarios,
  content ID
  `f044212429c08737719ffe1f64496122d20d94f2f6b4e2333c42d00d452b185c`,
  file SHA-256
  `ce04d89c1d7f6c16177e6e02bdb76a616b125ccd50fcf970b943fc03def408c9`.
  Exact V2 manifests and capture/bundle handoffs pass. Changed hyperparameters,
  changed samples, wrong-job authorization, incomplete bundle coverage, and a
  V1 dictionary manifest in the V2 path reject. This is synthetic runtime
  evidence and contains no Bonsai activations or trained dictionary.
- Activation-category research protocol V9: pass, SHA-256
  `3d4397f3ef270f4d2b0bc00995034543f34a60dedfc9c9587eb06963e2b5bd9c`.
  It binds the V4 plan, Gate-plan V2, reconstructed runtime implementations,
  and the 11-scenario benchmark. This research label is distinct from all
  historical Bonsai experiment labels and authorizes no execution.
- Unified research-protocol V9 lineage amendment V7: pass, content ID
  `e97ea34df8a8e4c12f4b13e5bdf02fe9ed0fe985a1dfdad8237734420a7dc90c`,
  file SHA-256
  `4d54b3ec0c422dd5f494340cb2808c658fe930434c99a09e98638c33bbd80ddb`.
  It reverifies recovered predecessor lineage and reconstructed runtime lineage,
  records all 22 guarded jobs and 11 benchmark passes, and explicitly records
  that no empirical bundle, manifest V2 dictionary, release receipt, external
  authorization, capture, model inference, or SAE training exists.
- Within-category cluster-discreteness benchmark V2: pass, 4/4 scenarios,
  content ID
  `3c72d47b3662ccc78bdf2470850f0a644e597226cde988b78335b07a72609d91`,
  file SHA-256
  `b0d6e945f0af0c8bd146cc760d54341d3bcdac68b86569a9d1a1978ba4754e52`.
  It reproduces two legacy false splits: a continuous strength gradient split
  in all eight categories and a continuous direction arc split in one. V2
  falls back to one cluster for both controls and retains two clusters for the
  separated two-mechanism control. This is synthetic evidence only.
- Activation-category research protocol V10: pass, SHA-256
  `1b53f290aa05a2aefefea665202ba6c22e045c019a05b5796028e895fd062f1c`.
  It preserves V9 and requires lower-tail silhouette plus an explicit boundary
  gap for every future nontrivial within-category cluster. A V1-only empirical
  candidate is prohibited. This research label is not a historical Bonsai
  experiment version.
- Cheap-executor runbook V2: pass, content ID
  `7c4a52e50e0105d26be1badeb002cd5fca5985d8a416f9017e08f61a44ee8c41`,
  file SHA-256
  `3dd1e0479d4bf75723a562a694e45e0e6c1c5336ed371d74b1ef003ac5e8d5b4`.
  It adds the discreteness benchmark to local Stage 0 and inserts a hard Stage
  1 stop until a production V2 candidate wrapper and independent admission are
  implemented. No paid stage is authorized.
- Unified research-protocol V10 lineage amendment V8: pass, content ID
  `6315c32b329278aba56f8a29d84ff9c5e9ebaccfe32a168bb36d6d70987a26d9`,
  file SHA-256
  `9b420d191f5355f0ed044809b78918ebba234153c1901b33f156edbe6396a04f`.
  It reverifies unified V7, protocol V10, and runbook V2 while recording that
  the production cluster-candidate V2 runtime, empirical discreteness audit,
  scientific release, and external compute authorization remain absent.
- Reconstructed V7 validator: 960 rows, 96 pairs, 576 observable-status rows,
  96/96 status-only contrasts, 0 conflicting prompt-label hashes, 0 executed
  rows, 0 exposed execution paths, and 0 overlap against the 1,260 recovered
  exact V3 development rows.
- Frozen V7 validation receipt remains authoritative for the broader original
  2,640-row prior comparison.
- V8 nested manifest: 18/18 payload hashes match (assignments, clusters, and 16
  packets).

## Release and compute decision

Neither the base model nor the adapter is approved for authorization-control
release. No paid inference, external GPU work, weight mutation, signing,
broadcast, or real action execution was performed during this recovery. No
empirical category-aware bundle, partition-audit report, SAE training-release
receipt, activation capture, or SAE dictionary exists yet.
