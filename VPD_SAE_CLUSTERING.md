# VPD clusters for focused SAE studies

`cluster_vpd_for_sae.py` converts a flat VPD component inventory into a
deterministic, content-addressed partition. Each cluster becomes a small work
packet that a decentralized worker can execute without selecting components or
changing the scientific claim after seeing outcomes.

## Why this layer exists

The earlier hierarchy partitions all LoRA components by projection family and
depth. Those groups are useful for broad ablations, but a 144-component cell is
still too coarse for a focused SAE study. This pipeline can additionally use
VPD activation-effect vectors (`delta_h`) so components with similar activation
effects are studied together even when they occur in different modules.

When an effects archive is unavailable, the fallback feature space combines
layer, rank, projection family, component norm, module-delta fraction, and any
surviving joint causal evidence. Fallback clusters are screening strata, not a
substitute for effect-vector clustering.

## Build and verify a bundle

```powershell
python -m experiments.authz_mechinterp.cluster_vpd_for_sae `
  --hierarchy-plan artifacts/bonsai8b_authz_v5/hierarchy_group_plan.json `
  --output-dir artifacts/bonsai8b_authz_v8/vpd_sae_clusters `
  --clusters 16 `
  --workers worker-a,worker-b,worker-c

python -m experiments.authz_mechinterp.cluster_vpd_for_sae `
  --verify-output artifacts/bonsai8b_authz_v8/vpd_sae_clusters
```

Add `--effects activation_effects.npz` when the archive contains aligned
`component_ids` and `delta_h` arrays. Effect vectors are normalized and reduced
through a seeded random projection before deterministic farthest-seed k-means.

`--hierarchy-plan` reconstructs the exact component partition and norm receipts
from a prior hierarchy plan. Use `--components` for a clean flat JSONL
inventory. Inputs containing null bytes are rejected rather than silently
salvaged.

## Bundle contract

- `manifest.json` freezes input hashes, clustering parameters, partition
  invariants, artifact hashes, and the worker assignment algorithm.
- `assignments.jsonl` maps every component to exactly one cluster.
- `clusters.jsonl` records cluster statistics and representative components.
- `packets/*.json` contains the exact component set, recommended capture layers
  and activation keys, SAE stages, required controls, and a content-derived
  packet ID.

Workers are assigned with rendezvous hashing. A worker result must cite the
packet ID it executed; changing membership, representatives, study controls, or
the worker set changes that content identity and is detectable.

## Signed worker results and deterministic merge

Workers sign `sch.vpd-sae-worker-result.v1` receipts with Ed25519. The receipt
binds the worker to the packet and to SHA-256 identities for the packet file,
fixed SAE dictionary, activation capture, and held-out dataset. Candidate rows
may only reference components and activation keys frozen in that packet.

```powershell
python -m experiments.authz_mechinterp.merge_vpd_sae_results `
  --bundle artifacts/bonsai8b_authz_v8/vpd_sae_clusters `
  --registry worker_registry.json `
  --results-dir worker_results `
  --output-dir artifacts/bonsai8b_authz_v8/vpd_sae_merge `
  --require-complete

python -m experiments.authz_mechinterp.merge_vpd_sae_results `
  --verify-output artifacts/bonsai8b_authz_v8/vpd_sae_merge
```

The reducer verifies signatures, assigned-worker identity, packet membership,
all frozen controls, SAE quality, screening thresholds, signed causal-effect
agreement, and guardrails. Identical duplicate receipts are ignored;
conflicting receipts from one worker for one packet fail the merge. Outputs are
sorted and content-addressed, so independent reducers produce the same merge.

## Decentralized web exchange

`vpd_web_exchange.py` turns the immutable bundle and public worker registry
into a small HTTP exchange. It is an append-only object exchange, not a mutable
job scheduler: rendezvous assignment in each signed packet remains the source
of truth. A worker discovers only its assigned packets, downloads them by
SHA-256, and submits signed artifacts. Any number of independent mirrors may
admit the same artifacts.

```powershell
python -m experiments.authz_mechinterp.vpd_web_exchange `
  --bundle artifacts/bonsai8b_authz_v8/vpd_sae_clusters `
  --registry worker_registry.json `
  --state-dir artifacts/bonsai8b_authz_v8/web_exchange `
  --host 127.0.0.1 `
  --port 8787
```

The versioned endpoints are:

- `GET /v1/network`: immutable network identity and source hashes.
- `GET /v1/workers/{worker_id}/work`: assigned packet discovery and round
  status.
- `GET /v1/objects/sha256/{digest}`: immutable packet, registry, policy,
  receipt, or shard bytes with an immutable cache header.
- `GET /v1/snapshot`: a content-addressed, order-independent view of admitted
  work and exact packet coverage.
- `POST /v1/submissions/effect`: a
  `sch.vpd-effect-web-submission.v1` JSON object containing the signed receipt
  and base64 shard bytes.
- `POST /v1/submissions/sae`: a `sch.vpd-sae-web-submission.v1` JSON object
  containing the signed SAE result.

Submissions are validated before admission with the same functions used by the
offline reducers. Exact retries are idempotent. A second result ID from the same
assigned worker for the same packet is a conflict and cannot change the public
snapshot. Packet IDs and worker IDs cannot select arbitrary filesystem paths,
request bodies and decoded shards are bounded, and state objects are stored by
verified SHA-256.

Mirrors derive snapshot identity by sorting admitted content addresses, never
by receive time. Therefore mirrors that see the same signed objects produce the
same snapshot even if messages arrive in different orders. The exchange can
also reduce its admitted state directly; shard objects are resolved by content
address, so workers do not need globally unique local filenames:

```powershell
python -m experiments.authz_mechinterp.vpd_web_exchange `
  --bundle artifacts/bonsai8b_authz_v8/vpd_sae_clusters `
  --registry worker_registry.json `
  --state-dir artifacts/bonsai8b_authz_v8/web_exchange `
  --reduce-effects artifacts/bonsai8b_authz_v8/effect_merge

python -m experiments.authz_mechinterp.vpd_web_exchange `
  --bundle artifacts/bonsai8b_authz_v8/vpd_sae_clusters `
  --registry worker_registry.json `
  --state-dir artifacts/bonsai8b_authz_v8/web_exchange `
  --reduce-sae artifacts/bonsai8b_authz_v8/vpd_sae_merge
```

The exchange intentionally does not provide worker enrollment, key rotation,
or an unsigned administrative override. Those change the frozen registry or
scientific network identity and must be published as a new content-addressed
network rather than mutating an active study.

### Worker transport

An independent worker uses `vpd_web_worker.py` to synchronize its immutable
assignments. The client checks the network/view relationship, packet SHA-256,
internal packet ID, and object ETag before writing a local sync manifest.

```powershell
python -m experiments.authz_mechinterp.vpd_web_worker `
  --exchange-url https://study.example `
  --sync `
  --worker-id worker-a `
  --output-dir worker-a-work
```

After `compute_vpd_effect_shard.py` creates a shard and unsigned receipt, sign
the receipt with the worker's Ed25519 key using
`merge_vpd_effect_shards.py --sign-receipt`. Submit the exact signed receipt and
shard together:

For category calibration and cluster-count validation, the worker must also
pass `--activation-category-split-manifest` and one of
`--activation-category-split-role discovery`, `calibration`, or
`cluster_validation`. The runner
verifies that `--dataset` is the exact artifact for that role and binds the
split ID, manifest hash, episode set, and pair set into the receipt. It rejects
the `final_evaluation` role before model loading.

```powershell
python -m experiments.authz_mechinterp.vpd_web_worker `
  --exchange-url https://study.example `
  --submit-effect-receipt worker-a-work/effect-receipt.signed.json `
  --effect-shard worker-a-work/effect-shard.npz
```

The client checks the local shard hash and filename against the signed receipt
before upload. A signed focused-SAE result is submitted similarly:

```powershell
python -m experiments.authz_mechinterp.vpd_web_worker `
  --exchange-url https://study.example `
  --submit-sae-result worker-a-work/sae-result.signed.json
```

## Two-round effect-space refinement

If a global `delta_h` archive does not yet exist, use metadata clusters as
round-zero compute shards. Each assigned worker computes effect vectors for the
exact packet component set and signs a `sch.vpd-effect-shard-receipt.v5`. The
effect reducer requires one model revision, adapter hash, dataset hash, runner
hash, effect-space anchor, hidden dimension, and float32 representation across
all shards.

```powershell
python -m experiments.authz_mechinterp.merge_vpd_effect_shards `
  --bundle artifacts/bonsai8b_authz_v8/vpd_sae_clusters `
  --registry worker_registry.json `
  --receipts-dir effect_receipts `
  --shards-dir effect_shards `
  --output-dir artifacts/bonsai8b_authz_v8/effect_merge
```

The reducer creates a deterministic `activation_effects.npz` only after exact
packet and global component coverage passes. Feed that archive back to
`cluster_vpd_for_sae.py --effects ...` to create round-one effect-space
clusters. This avoids requiring any one web worker to hold the full VPD effect
matrix.

The preserved predecessor run shape remains frozen in
`artifacts/vpd_effect_capture_campaign_v3/plan.json`, content ID
`324dac30e39fa46305ede80eeb00ce5037f219133e2db66e8128a15e506ef6b8`.
It binds the exact V8 4,032-component partition to all sixteen packets in each
of the disjoint `discovery`, `calibration`, and `cluster_validation` roles: 48
effect jobs, 48 worker-key signing jobs, three exact-coverage merges, and one
production-gated cluster job. Shards and receipts use unique flat basenames so
the reducer's non-recursive input layout cannot collide. The plan estimates
114,162,048 episode-equivalent signed-perturbation forwards, 28,667,520 batched
forwards at batch size four, and 3,009,871,872 bytes of float32 effect payload
before NPZ, identifier, model-memory, and runtime overhead. These estimates are
an authorization warning, not permission to run; model identity, revision,
adapter bytes/hash, worker registry/keys, device, wall time, and cost remain
unbound runtime requirements.

Campaign V3 is not executable as written: its preserved worker computes the
CountSketch seed as a hexadecimal string modulo an integer, which would raise a
`TypeError` before effect capture. The worker remains byte-for-byte preserved at
SHA-256 `02db527aa019cc9120435bcf9960fb819568045f3ef7cc3378d901808c5eb1a4`.
The current execution candidate is the versioned amendment at
`artifacts/vpd_effect_capture_campaign_v4/plan.json`, content ID
`ffcc7351bade510166c449099f19d43bd1a28a89e83bc85bc912f883c742b528`.
All 48 compute jobs use the hash-bound V2 wrapper, SHA-256
`1cc03ee68265b52f1454f8aec93e5f236568bc99ef3f9d273e6daa7d8e5e6159`,
which converts the full SHA-256 hexadecimal digest to an integer before the
frozen modulo-`2**32` operation. All downstream job dependencies are
re-addressed. This correction was found and verified without model inference;
campaign V4 also authorizes no execution.

Compute is staged rather than launched all at once. The discovery merge is
fully verified before a content-addressed continuation gate evaluates the
loosest frozen specificity thresholds plus jackknife, pair-block, condition,
and template robustness. Fewer than 32 surviving specific components is a
sound upper-bound rejection: stricter calibration candidates cannot increase
that count. Only `pass_continue_calibration` unlocks calibration. The second
gate runs the exact independent cross-archive threshold selector and only
`pass_continue_cluster_validation` unlocks the third archive. Either
`reject_stop_campaign` terminates later compute. Gate passes are necessary
continuation conditions, not Bonsai clustering evidence.

The plan also binds the reproducible seven-scenario synthetic gate benchmark
at `artifacts/vpd_effect_campaign_stage_gate_benchmark_v3/report.json`, content
ID `9a12ecaed95889d139c1bf0e504e7172090f17941622d5935752262bc3ec92b4`.
Clean discovery and calibration fixtures continue, while insufficient
specific-component capacity, a condition artifact, category reassignment,
direction reversal, and magnitude collapse all stop. This tests fail-closed
control flow only; synthetic effects do not establish a Bonsai gate outcome.

The predecessor small-pack campaign remains preserved at
`artifacts/vpd_effect_capture_campaign_v1/plan.json`, current reproducible
content ID `cd5819dcbc4cb5d5b6e7ca19692ec74e46aa065a03a88b5a7799af956f0e1a9b`.
It is a protocol pilot, not the current end-to-end plan: its split ID differs
from the capacity-scale SAE plan and its semantic role families overlap that
pack in incompatible roles. The unified verifier rejects joining them.

All round-zero packets freeze the same downstream effect space
(`layer_35__proposal_start` by default). `proposal_start` means the final
prompt-token hidden state that produces the first REQUEST/PROPOSE token; zero
generated tokens may be fed before capture. Every packet also requires a fixed
base SAE dictionary at that exact activation key. Local component-layer keys
remain exploratory capture suggestions, but they cannot be cosine-aligned to a
different-space `delta_h`. Use `--effect-space-id` to change the common space
before distributing packets; workers cannot change it in their receipts.

### Activation-category compartments

The current machine-readable design contract is
`configs/evals/vpd_activation_category_compartments_v5.json`; V4, V3, V2, and
V1 remain preserved predecessors. Current V5 SHA-256 is
`0894ac8e06ba7828bf5cb21ae79e1ae48ae96e2c27cd66e6853899765c1b9c24`.
V4 added fail-closed per-component cross-archive admission. V5 adds a separate,
post-freeze partition audit that must pass before any category-specific SAE
training. V3 had already unified the exact capacity pack and five-role split.
Regression tests pin the category ordering,
specificity defaults, durable pack identity, and role-specific semantic
template sets to the executable Python contract. It is a locally verified
design record, not evidence that an empirical category-effect archive or
compartment SAE has already been run.

The content-addressed join receipt at
`artifacts/vpd_unified_compartment_pipeline_v2/receipt.json`, ID
`12dea48a1928ec3780f31419aed772f9584962216391bc10aabc5d2ab753ffd3`,
proves that the current effect campaign and compartment-SAE plan use the same
capacity split `a5aa3370...`, the same protocol bytes, and five pairwise-disjoint
semantic role sets. It also proves only local lineage: it performs and
authorizes no capture, training, clustering, or intervention.

The V5 lineage amendment at
`artifacts/vpd_unified_compartment_pipeline_v3/receipt.json`, ID
`bec00550d7d523f4ece993aafdac61f203a88c4417d8cab7bafc3ef0752831e2`,
reverifies that V2 receipt and binds the V5 protocol, fresh audit pack,
benchmark, and 16-work-unit gate plan. It preserves the original five-role
split and planned campaign/training artifacts while adding the new pre-training
gate; it likewise performs and authorizes no compute.

Effect shard V5 preserves both the aggregate `delta_h` and a tensor
`delta_h_by_category[component, category, hidden]`. Categories describe the
observable authorization state, never the model output:

- valid authority;
- missing or untrusted authority;
- invalid signature;
- inactive time window;
- revoked grant;
- replayed or consumed nonce;
- invalid scope or resources;
- conservative fallback for other invalid observable state.

The signed receipt binds the category contract, sorted category names, episode
counts, the exact episode-ID set in every category, and pair-block plus
within-category condition- and template-robustness contracts. In addition to the full category means, V5 retains eight compact
pair-atomic CountSketch block sums (256 dimensions by default) and their counts.
This supports leave-one-pair-block-out stability checks without retaining every
episode's full hidden-state vector. The reducer rejects partial uncertainty
arrays, empty blocks, different block identities across workers, or block sums
that do not reproduce the sketched category means. It also retains one compact
CountSketch sum and episode count per observable `category::condition` cell,
plus one per `category::template_group` cell.
The reducer verifies that those condition summaries exactly reconstitute each
category mean. V2 remains readable only as a legacy archive without within-split
uncertainty evidence; V3 has only the pair-block screen and V4 adds conditions
but lacks template-group breadth.

The compact-summary dimension is now admitted by a separate production-shaped
benchmark at
`artifacts/vpd_effect_sketch_fidelity_benchmark_v1/report.json`, content ID
`4ed9790051cabd86dd4b2faffd68d57bfd1a8fbead2e8ea0df0b107ab29e04dd`.
It evaluates 3,584 dense and sparse synthetic effect cells at hidden size 4,096
over eight sketch seeds. Sixty-four dimensions fail the false-admission gate;
128 fails the frozen p99 metric-error bounds. The selected 256-dimensional
candidate has worst-seed false-admission Wilson upper `0.007093`, stable-pass
recall Wilson lower `0.996263`, cosine-error p99 `0.122504`, and relative-norm
error p99 `0.061274`. Production workers default to 256 and production routing
rejects lower-dimensional condition/template evidence. Boundary profiles remain
diagnostic and conservatively route to shared; this synthetic receipt does not
establish Bonsai categories.

The current local model-free episode resource is
`artifacts/vpd_activation_category_pack_v2`. Its content-addressed pack ID is
`dc5224d9838d8661764f45342543a4f3ea35152cc16be60dd7d2d11b3589910d` and
its original embedded four-role split ID is
`bf36a956222f517a68a6d7596033b4d66a59f49e2be37f946d0984f85715c606`.
It contains 1,056 candidate-only rows: 96 pair groups, sixteen semantic action
templates, eleven observable conditions, and all eight declared activation
categories. V1 is preserved at `artifacts/vpd_activation_category_pack_v1`
with its original four-template, pair-disjoint split and content identities.
The `other_invalid` category is represented by an explicit model-visible
malformed-status contrast. Pair-constant audit references vary prompt bytes but
are shared across every condition in a pair, so they cannot encode the label.
The pack validator found zero episode, pair, or exact-message overlap against
the parseable recovered priors. For five corrupt historical JSONL files it
records a narrower byte-level witness: none of the 96 new case references
occurs in their preserved bytes.

The current semantic split is the separately content-addressed V3 artifact at
`artifacts/vpd_activation_category_split_v3`, ID
`7f8157155dc5cb6d816b09a91b4d0ec6114ba0499022633aaf4160b3ffe575d5`.
It assigns each whole semantic `template_group`, and every `pair_id` under it,
to exactly one of five roles: `sae_training` (132 rows/12 pairs), `discovery`
(264/24), `calibration` (264/24), `cluster_validation` (264/24), and
`final_evaluation` (132/12). The role weights are 1:2:2:2:1 over all sixteen
templates. Episode, pair, and template-group sets are pairwise disjoint.
Discovery supplies clustering features, calibration selects thresholds,
cluster validation selects cluster counts independently, and final evaluation
remains algorithmically held out rather than secret or externally
authenticated. The artifact records zero model outcomes and does not itself
establish an empirical mechanism result.

When the merged archive carries category effects, clustering defaults to
category compartments. A component is assigned to its strongest category only
when the frozen dominant-share, dominance-margin, and normalized-entropy gates
all establish sufficient specificity. Zero-effect, tied, and diffuse profiles
route to `shared_cross_category` rather than receiving an arbitrary category.
Cluster allocation is deterministic and every packet is compartment-homogeneous.
A packet freezes the corresponding SAE protocol, same-compartment controls,
and a cross-category evaluation requirement. The default specificity gates are
dominant share at least `0.35`, share margin at least `0.10`, and normalized
entropy at most `0.80`; changing them creates different packet identities.

Within a compartment, direction features are weighted by the square root of
their category strength share before K-means. Previously, every category
direction was normalized independently and concatenated at equal weight. That
allowed an arbitrarily tiny non-dominant effect to control distance merely
because seven background categories supplied seven normalized directions.
Square-root share weighting makes the sum of squared category weights exactly
one for every component with nonzero total effect, while preserving more
secondary-category geometry than linear-share weighting. The production-shaped
model-free receipt at
`artifacts/vpd_activation_category_clustering_geometry_benchmark_v1/report.json`
uses hidden size 4,096, eight categories, two known mechanisms per category,
and eight projection seeds. The unweighted predecessor fails with worst-seed
pairwise F1 `0.565022` and purity `0.703125` even when random background
strength is only `1e-6`; square-root share weighting reaches `1.0` on both
metrics in every scenario. Its content ID is
`7098172918613a6f74e1f2d4032a5d720440dc654d1465cbc73f25d416c07428`.
This validates a geometry invariant, not the empirical Bonsai cluster count or
mechanism assignments.

Production cluster counts are selected independently inside each observed
compartment across 32 views: eight deterministic projections and all eight
leave-one-pair-block-out CountSketch means from discovery, plus the same 16
views from a content-, pair-, and template-disjoint `cluster_validation`
archive. Counts `2..4` must retain
adjusted-Rand agreement of at least `0.90`, minimum replicate silhouette of at
least `0.15`, and at least four components in every resulting cluster. Among
passing candidates, selection maximizes the worst-replicate silhouette, then
replicate agreement, then prefers the smaller count. If no nontrivial candidate
passes, that compartment remains one screening cluster.
The production-shaped receipt at
`artifacts/vpd_activation_category_cluster_count_benchmark_v1/report.json`
recovers known one-, two-, and three-mechanism fixtures across all eight
authorization categories, including a two-mechanism fixture with `1e-6`
random background effects. A fifth adversarial fixture exposes two mechanisms
only in the aggregate while resampled component directions disagree; the new
sampling gate correctly falls back to one cluster. A sixth has two clean,
well-separated mechanisms in both archives but rearranges which components
belong together in the independent archive; cross-archive adjusted-Rand
agreement rejects the discovery-only subdivision. Its content ID is
`32ced135dd8ee66af5a1fd56f592c5f407da0a7636f050c75c5a9c3febaaa996`.
Manual total counts remain available only as an explicitly labeled ablation;
they are stored as `unvalidated_manual` in manifests and packets.

For a production category bundle, add `--require-category-uncertainty`. The
point-estimate category must then survive at least seven of eight
leave-one-pair-block-out replicates (`0.875`). A point-specific component below
that stability floor is conservatively rerouted to `shared_cross_category`.
Label stability alone does not prove that the omitted blocks share a direction:
opposing blocks can leave the same strongest category after every omission.
Production therefore also requires
`--require-category-pair-block-direction-robustness`. At least six of eight
blocks (`0.75`) in the dominant category must align with its aggregate at cosine
`0.50` and relative norm `0.25`; blocks receive equal coverage weight.
Production also requires `--require-category-condition-robustness`. For the
dominant category, at least `0.75` of its observable conditions must retain the
category direction (cosine at least `0.50`) and nontrivial magnitude (at least
`0.25` of the category-reference norm). Coverage is equal-weighted across
conditions so a high-support condition cannot conceal a zero or contradictory
condition. A failed specific component is rerouted to shared. These within-split
checks still permit a component supported by every condition but only one
semantic action family, so production additionally requires
`--require-category-template-robustness`. At least three of the four template
groups (`0.75`) must retain the category direction at cosine `0.50` and relative
norm `0.25`; template groups receive equal coverage weight. These within-split
screens are bound into the manifest and packets and complement, rather than
replace, the independent discovery/calibration replication test.
Use `--no-category-compartments` only for an explicitly declared ablation of
this partition rule.

Those defaults are a fail-safe manual design, not empirically mastered gates.
For an empirical bundle, provide a second activation-effect archive made from
an independent calibration prompt split:

```powershell
python -m experiments.authz_mechinterp.cluster_vpd_for_sae `
  --hierarchy-plan artifacts/bonsai8b_authz_v8/hierarchy_vpd_plan.json `
  --effects discovery/activation_effects.npz `
  --effects-merge-manifest discovery/effect_merge_manifest.json `
  --calibration-effects calibration/activation_effects.npz `
  --calibration-effects-merge-manifest calibration/effect_merge_manifest.json `
  --cluster-validation-effects cluster_validation/activation_effects.npz `
  --cluster-validation-effects-merge-manifest cluster_validation/effect_merge_manifest.json `
  --require-category-uncertainty `
  --require-category-pair-block-direction-robustness `
  --require-category-condition-robustness `
  --require-category-template-robustness `
  --select-category-cluster-counts-by-projection `
  --require-category-cluster-sampling-stability `
  --require-category-cluster-cross-archive-stability `
  --output-dir category_clusters
```

Effect workers default to the full verified split-role file
(`--maximum-episodes 0`). This is intentional: truncating individual episodes
can break the common `pair_id` support required by the paired jackknife, and the
worker fails closed if the selected categories do not share one exact pair set.

All three archives must first pass the signed effect-shard merge verifier and
must carry verified `discovery`, `calibration`, and `cluster_validation` roles
from the same split ID. The
worker receipt binds its split manifest, role, episode set, pair set, split
schema, and role-specific template-group set hash; merge manifests propagate
that identity. With the V3 split, calibrated clustering additionally rejects
identical role template sets or missing semantic-disjointness attestation.
Cluster packets bind all available discovery, calibration, and cluster-validation
template-set hashes. Focused SAE planning combines those with the dictionary's
`sae_training` hash and the capture/intervention `final_evaluation` hash, and
requires all five role hashes to be distinct. `--allow-unattested-calibration` exists only for
explicit synthetic pilots and marks their receipt `pass_unattested_pilot`.
The five-role semantic split needed for empirical cluster validation is now
generated and locally verified. Its `cluster_validation` role is independently
content-addressed and semantically disjoint, so the production 32-view gate can
be satisfied after authorized effect capture. No Bonsai activations or model
outcomes have been captured for any of these roles.
The archives must otherwise have the same component/category identity but
different content hashes. The
discovery archive alone supplies clustering features. The calibration archive
may only select specificity gates from the frozen grid in the protocol. A gate
triple passes only when discovery-specific assignments reproduce as the same
non-shared category in the independent archive with at least `0.90` discovery
and symmetric replication and a signed activation-effect direction cosine of
at least `0.50`. Direction alone is insufficient: the dominant category's
smaller archive strength must also be at least `0.50` of its larger archive
strength. This rejects a calibration effect that keeps the same normalized
profile and direction while collapsing toward zero. Production defaults also require at least `32` stable specific
components and `8` episodes per category in each archive. Thus a norm-stable
but direction-reversed effect cannot define an SAE compartment. If no candidate
passes, bundle construction stops; it cannot silently fall back to heuristic
category assignments. The selected grid, metrics, source hashes, per-component
directional diagnostics, and calibration ID are bound into the manifest and
every packet, and the bundle verifier recomputes the selection from the stored
diagnostics.

V4 adds a second, per-component admission gate after all discovery-side
uncertainty and robustness screens. A component remains in a specific SAE
compartment only if calibration assigns that same specific category and the
component individually passes the signed direction-cosine, strength-ratio, and
cross-archive stratified-robustness requirements. Failure routes that component
to `shared_cross_category`. The global `0.90` threshold-replication gate can no
longer waive admission for its unstable minority. Each assignment records its
pre-admission category, calibration point estimate, admission decision, and
rejection reason; the bundle verifier recomputes all of them and checks the
manifest totals.

V5 closes a later leakage boundary: a category partition that survives
discovery, calibration, and cluster validation must reproduce once more on a
fresh pack after every threshold, assignment, and cluster is frozen. The audit
is reject-only. It may not select a threshold or cluster, change a frozen
category, or promote `shared_cross_category` to a specific compartment. Every
frozen specific component must independently preserve the same category, a
signed target-direction cosine of at least `0.50`, a target-strength ratio of
at least `0.50`, and the same pair-block, condition, and template robustness
screens. One failure rejects the candidate for category-specific SAE training.
The rejected candidate may not be tuned on that pack; a successor requires a
new pack version and seed.

The fresh prompt-only resource is
`artifacts/vpd_partition_audit_pack_v1`, pack ID
`e65c78bb07fcf25c5a008d94a8f447b1b75f01cb8734632e7e2e567c81ca2f23`.
It contains 2,816 candidate-only rows over 256 pairs and four new semantic
families. Each non-missing category has 256 observations; missing/untrusted has
1,024. Validation found zero episode, pair, exact-message, or semantic-template
overlap against all 25,168 rows of the capacity pack. With zero observed
failures, 256 observations imply a one-sided exact 95% failure-rate upper bound
of `0.011634`; this bound describes the pack support, not a Bonsai result.

The model-free evaluator benchmark at
`artifacts/vpd_frozen_partition_audit_benchmark_v1/report.json`, ID
`f817cecf0833e66b9013b311eecbcad72f8e8091d0226094bae58bfed6f16a6f`,
passes a clean frozen partition and independently detects category mismatch,
direction reversal, magnitude collapse, and stratified-robustness failure. It
also verifies that a shared component is not promoted when the audit point
estimate is category-specific. The exact 16-work-unit census plan at
`artifacts/vpd_partition_audit_gate_v1/plan.json`, ID
`e6762af41fcce70bca47b1a6925c9e3b53b8a88e65aee946c5c7451054818358`,
binds 4,032 components and 22,708,224 episode-equivalent forwards. It authorizes
no execution. That V1 plan is preserved as the historical record of the then-
missing adapter. Gate-plan V2 at
`artifacts/vpd_partition_audit_gate_v2/plan.json`, ID
`1341c8f6371fab72cd9e8f8be92b1e740ae47f08c71a77a8bab39b04fe2a119b`,
closes that local implementation gap without modifying the preserved worker or
base merger. Its 16 reviewable job groups compute an unsigned V5 shard receipt
with no five-role split claim, attach content-addressed fresh-pack, category,
packet, and work-unit provenance before signing, require Ed25519 verification
and exact coverage in a dedicated merge front-end, then run a production
reject-only evaluator against the already frozen category-aware bundle. Capture
packetization and category assignment are deliberately separate: the historical
metadata bundle supplies a hash-bound, exact component packet census for shard
collection, while the later category-aware bundle supplies frozen assignments
and discovery lineage. Their component sets must match; their packet IDs need
not. This avoids making historical packet membership into category evidence.
The adapter, merge front-end, and evaluator are hash-bound in the plan. All
model, revision, adapter, key, registry, bundle, and output locations remain
explicit placeholders, and every step records `execution_authorized: false`.
Worker identity is not a global runtime placeholder: the canonical packet
bundle assigns five work units to `worker-a`, five to `worker-b`, and six to
`worker-c`. Every compute command now freezes that exact worker ID, and every
sign command uses a distinct worker-scoped private-key placeholder whose public
key must match an enabled registry entry. The merge already rejects an otherwise
valid receipt from a worker not assigned to its packet.
The audit measurement settings are no longer a parallel hand-maintained
contract. Gate-plan V2 verifies and binds corrected discovery campaign V4, ID
`ffcc7351bade510166c449099f19d43bd1a28a89e83bc85bc912f883c742b528`,
then derives maximum episodes, selection seed, perturbation epsilon, batch
size, uncertainty blocks, sketch dimension, and model-load dtype from its
content-addressed parameters. Every audit command therefore freezes
`--dtype float16`; `DTYPE` is forbidden as a runtime placeholder. The capture
packet effect space must also equal the campaign's
`layer_35__proposal_start`, and the audit/campaign batch sizes must match.

The model-free signed-pipeline benchmark at
`artifacts/vpd_partition_audit_capture_pipeline_benchmark_v1/report.json`, ID
`9efa073d053ee89551c847f09af0b56b450f134b333a88d80ae8ed5ee5d7ec89`,
exercises the distinction end to end. A clean six-component fixture with
different capture and category packetizations passes after pre-signing
attestation, Ed25519 verification, five exact signed merges, and production
report verification. A category swap and an observable-condition robustness
defect each reject one component for the intended reason, and substituting the
category bundle as the capture packet registry is rejected. Signed audit
archives from a different model or adapter pass their own internal merge but
are rejected by the production discovery/audit identity join; a different
effect space is rejected earlier by pre-signing campaign attestation. The required
shared identity includes model, revision, adapter, runner, effect space, phase,
estimator, perturbation epsilon, selection seed, zero-generation contract,
activation-category contract, and all three robustness contracts. Dataset and
split-role identities must differ. These are synthetic effects only, not Bonsai
category evidence.
The benchmark now uses the production 4,096 hidden width and 256-dimensional
CountSketch contract. Every V2 audit receipt binds a common, content-addressed
campaign measurement contract plus its packet-specific compute-command hash
before signing. Epsilon, seed, effect space, estimator, zero generation, hidden
size, uncertainty blocks, sketch dimension, and derived sketch seed are checked
against the campaign. Batch size, maximum episodes, and `float16` model-load
dtype are explicitly labeled as gate-command-bound rather than falsely claimed
as worker-observed receipt fields. A wrong effect space now rejects during
pre-signing campaign attestation.

Research protocol V6 at
`configs/evals/vpd_activation_category_compartments_v6.json`, SHA-256
`c5d9a47f651fcf654a9b33780c0c556e681a80cf4379a1684c65dba985e2b84a`,
binds Gate-plan V2 and preserves research protocol V5 as its predecessor. This
research-protocol version is distinct from the historical Bonsai experiment
V6. Unified lineage amendment V4 at
`artifacts/vpd_unified_compartment_pipeline_v4/receipt.json`, content ID
`e12976b0161b3c947006fb785c5cf598e63a6620f4cd1563b3690e2769af6f75`,
reverifies the entire predecessor lineage and records the capture path as
implemented locally but not run. It also binds and reverifies the model-free
signed-pipeline benchmark. No signed runtime receipt, empirical category
partition, or partition-audit report exists yet.

The downstream SAE handoff is now fail-closed rather than relying on that prose
ordering alone. The versioned orchestration plan at
`artifacts/vpd_compartment_sae_training_release_gate_v1/plan.json`, ID
`7109adb38749c4c0e91154df0fddfe5fb79c625972f3ff89eca06ffda39596b9`,
preserves training plan V3 byte-for-byte as a predecessor blueprint and guards
all 22 of its jobs: both base-capture jobs, all eighteen SAE replicas, bundle
admission, and calibration quality. Direct execution of the predecessor
blueprint is prohibited. A single content-addressed scientific release receipt
must first reproduce the production partition audit and join the exact training
plan and protocol hashes, category-aware bundle manifest and component census,
fresh audit-pack and gate-plan IDs, audit report, category set, model/revision/
adapter measurement identity, and `layer_35__proposal_start` effect space.
The receipt deliberately records `compute_authorized: false`; explicit compute
authorization is still a separate later prerequisite.

The model-free benchmark at
`artifacts/vpd_compartment_sae_training_release_gate_benchmark_v1/report.json`,
ID `f706337b2141574a537a013c8d04ea2ec28f5c4e8fac5f4a580b75f9c00b90a3`,
passes one clean synthetic release and rejects eight unsafe handoffs before a
receipt is emitted: missing or rejected audit, wrong bundle, component-census
mismatch, stale gate plan, stale fresh pack, a shared-only candidate, and the
wrong activation space. This is synthetic contract evidence only. There is no
empirical Bonsai category bundle, audit report, release receipt, capture, or
SAE training.

Research protocol V7 at
`configs/evals/vpd_activation_category_compartments_v7.json`, SHA-256
`3e584f718df415cf4307511da5a04ea1ccc425695dc2a5bfb0360ce9617c94f7`,
adds the next fail-closed handoff while preserving V6: a focused component may
only use an out-of-packet parameter control from the same frozen activation
compartment. The preserved V1 planner populated both a generic control pool and
a category-matched pool but selected only from the generic pool. That packet's
category-matched pool was also truncated to the first 32 inventory entries. The
versioned V2 planner preserves that first-32 list only as predecessor metadata:
it binds the bundle manifest and assignments, derives the complete sorted set of
same-compartment components outside the target packet, and selects from that
complete pool. It rejects a category packet with no such controls. Its V2
executor reloads the exact effect archive and dictionary, independently
rederives the complete pool, and recomputes the norm/alignment selection before
it delegates to the preserved intervention worker.

The eight-scenario synthetic receipt at
`artifacts/vpd_focused_sae_compartment_control_benchmark_v1/report.json`, ID
`57f3f9a612883b04bb9651ca8de3fee7f17a6b0996b7eb88ee7d85d22a84ed25`,
passes a clean case in which the complete assignment-derived pool expands the
truncated packet list and selects a better undeclared same-compartment control.
It also passes the shared case and rejects a truly empty complete pool,
cross-category predecessor pool, target-category mismatch, target/control
overlap, missing full-pool effects, and predecessor-plan tampering. Unified
lineage V5 at
`artifacts/vpd_unified_compartment_pipeline_v5/receipt.json`, ID
`aedb91d6a90700517005c517ef29acbc3f87594fe82695adeb6ae10d3af639a4`,
reverifies protocol V7, this benchmark, lineage V4, and the 22-job SAE release
gate. This research V7 label is distinct from the historical Bonsai experiment
V7. No focused V2 plan or empirical intervention exists.

Research protocol V8 at
`configs/evals/vpd_activation_category_compartments_v8.json`, SHA-256
`b66111e7d5e7001ddef5dadefd97eaa0f6ef3bdd658699d6ad3add72e68fab49`,
adds a design-only training-execution boundary while preserving V7. A trace of
the 22-job release blueprint confirmed that category selection is correctly
compartmentalized, but found that the predecessor V1 dictionary manifest binds
steps, seeds, expansion, and top-k while omitting the planned batch size and
learning rate. The V1 bundle builder's acceptance of a re-addressed synthetic
record that explicitly reports `batch_size=64` and `learning_rate=0.01` against
the plan's `128` and `0.0003` reproduces this gap. The predecessor training plan
V3, trainer, admission code, and release gate remain preserved evidence, but
they are not sufficient by themselves for an empirical training admission.

The V2 execution blueprint at
`artifacts/vpd_compartment_sae_training_execution_blueprint_v2/blueprint.json`,
ID `33bc09fac66fcec62bc0374c39d7ad4209a5d17eafd9b1003b4353073687e40e`,
recomputes exact episode, pair, and prompt-window identity hashes from the
frozen `sae_training` split for all 18 replica jobs. Every specific compartment
has zero cross-compartment rows. The shared compartment selects 572 whole
episodes from each of all eight observable categories: 4,576 episodes and
585,728 activation rows. Its eight-scenario benchmark at
`artifacts/vpd_compartment_sae_training_execution_contract_benchmark_v1/report.json`,
ID `08255dba491199e4963080c8e059c8ebd0fdddaffb3e4fbd70c781a8685b2ff4`,
accepts the exact contract, rejects hyperparameter, sample-identity, balance,
and job-coverage substitutions, and reproduces the V1 gap. Unified lineage V6
at `artifacts/vpd_unified_compartment_pipeline_v6/receipt.json`, ID
`f4391a7bed1c8047d2b4ea08247b667a53619586f3931802b4146dee0d34187f`,
reverifies V5, protocol V8, the exact blueprint, and the benchmark. A versioned
empirical runner and admission path that emits
`sch.vpd-fixed-sae-dictionary-manifest.v2` is still required. No capture,
training, checkpoint, dictionary, or weight mutation was performed. This
research V8 label is distinct from historical Bonsai experiment V8.

Calibration ties are resolved by the largest minimum decision slack among the
stable specific components before the final deterministic threshold tie-break.
The earlier strictest-threshold-first tie-break was removed after the
model-free protocol benchmark showed that it placed otherwise clean components
on the boundary: a small unseen-template amplitude shift could then route every
specific component to the shared compartment despite perfect discovery and
calibration agreement. Decision slack is the minimum remaining margin across
dominant share, dominance margin, entropy, and signed direction cosine in both
archives. It is stored in the selected calibration metrics and independently
recomputed by the bundle verifier.

The deterministic synthetic benchmark is recorded at
`artifacts/vpd_activation_category_protocol_benchmark_v1/report.json`. It uses
128 components with known ground truth across all eight categories plus a
shared compartment. Clean and moderate cross-template scenarios recover all
components in discovery and an unseen-template proxy; high pair-block noise
routes 95 of 96 specific components conservatively to shared while preserving
specific precision and shared recall; direction reversal and cyclic category
label swap are rejected during independent calibration. Calibration now applies
the same pair-block direction, observable-condition, and semantic-template
coverage gates as discovery before a component can count as independently
stable. Three discovery-only artifact scenarios contaminate 8 of 96 specific
components: condition concentration, template concentration, and opposing
pair-block directions are each contained by rerouting exactly those eight
components while the archive remains above the frozen 90% replication floor.
Three matching calibration-only scenarios contaminate all 12 replay-specific
components; each drops stable replication below 90% and rejects the entire
bundle. A new minority-disagreement scenario changes eight of 96 calibration
categories: aggregate replication remains `0.916667` and legitimately passes,
but all eight unstable components are individually routed to shared. A further
scenario scales every calibration effect to 2% of discovery
while preserving category labels and directions and is rejected by the
dominant-strength ratio alone. The 13-scenario benchmark content ID is
`eb6ea4b61c17bd124a784b1efe532eb94f7fd79a3743725c284be296e48635c0`.
These are executable
protocol and failure-boundary checks only—synthetic effects do not establish
any Bonsai component category or SAE mechanism.

The V2 pack is sufficient to exercise split, effect, routing, and focused-study
contracts, but it is **not** large enough to train a compartment SAE. The
content-addressed audit at
`artifacts/vpd_compartment_sae_capacity_v1/plan.json` has capacity plan ID
`f2f92a86a33ee8193bd6d8df9c3841e7710a9fe220237d5660679ecf040b0067` and
marks all three profiles insufficient. Seven specific categories have only 24
single-anchor observations (the missing/untrusted category has 96), with 24
unique pairs; the historical `16`-row implementation floor is not scientific
support for a 32,768-feature dictionary.

The smallest frozen candidate, `bounded_context_pilot`, uses an 8,192-feature
dictionary (`expansion=2`, `top_k=16`) and a prompt-only 128-token context
window. Under the conservative support floor it requires at least 143 pairs per
training template group, or about 25,168 rows across a fresh 16-template pack.
This arithmetic is a design floor, not evidence that the resulting SAE will be
trainable or mechanistically useful. Reconstruction quality, dead features,
sparsity, and held-out behavior still need empirical checks.

That capacity candidate is now materialized at
`artifacts/vpd_activation_category_capacity_pack_v3`. Its pack ID is
`ad753063ac5f83bd7197fbb1d17c4bd7bdced1d3a5140a7d765ab0f8e90f3a11`:
25,168 rows, 2,288 pairs, and 16 templates. Its independent V3 split is at
`artifacts/vpd_activation_category_capacity_split_v4`, ID
`a5aa3370406876c1c48ed702b431a64cb6a68c8376ab28b0aad1d852c22dd8dc`:
6,292 rows/572 pairs/four templates for `sae_training`, and 4,719 rows/429
pairs/three templates for each of `discovery`, `calibration`,
`cluster_validation`, and `final_evaluation`. It uses a new seed-bound V3
identity namespace after an earlier attempt was correctly rejected
for overlapping V2 pair and episode IDs. Validation found zero pair, episode,
or exact-message overlap across 4,752 parseable prior rows, and every role is
pair- and template-disjoint. Its capacity plan at
`artifacts/vpd_compartment_sae_capacity_v2/plan.json` has ID
`809deb31cba4642ee38cbaf2f7417120610b4fc9cfe6c37d0ed6cbaa885efb7d`;
only `bounded_context_pilot` clears the arithmetic support floor. No activations
or model outcomes have been collected, so this is still a design artifact.

`train_sae.py` now fails closed for category and balanced dictionaries unless a
content-addressed capacity plan and exact profile are provided, their selected
compartment is marked ready, the split IDs and manifests match, and the capture
records a prompt-only sampling contract with zero generated tokens. The current
V2-pack plan therefore intentionally rejects training. On a separately and
explicitly authorized model run, and only after the release-gate receipt above
has passed, the capacity V3 capture command would be:

```powershell
python -m experiments.authz_mechinterp.capture_activations `
  --input artifacts/vpd_activation_category_capacity_split_v4/sae_training_episodes.jsonl `
  --activation-category-split-manifest artifacts/vpd_activation_category_capacity_split_v4/split_manifest.json `
  --activation-category-split-role sae_training `
  --model <frozen-base-model> `
  --layers 35 `
  --phases pre_gate `
  --sae-training-window 128 `
  --capture-resource-plan artifacts/vpd_compartment_capture_resources_v1/plan.json `
  --activation-shard-rows 4096 `
  --output-dir training_capture
```

The worker performs one prompt-only forward pass per episode, never calls
generation in this mode, and records exactly offsets `-127..0` ending at
`proposal_start`. It rejects short prompts rather than padding or silently
shrinking a window. Each metadata row binds its episode, pair, category, token
position, and offset. Before either dictionary training or calibration quality,
`verify_activation_capture_semantics.py` now independently replays the exact
source-dataset order and recomputes every episode digest, observable category,
split identity, row index, and token-window coordinate. Its seven-scenario
model-free benchmark at
`artifacts/vpd_activation_capture_semantic_benchmark_v1/report.json`, ID
`51ebf5afb2cf0e1cc109ed8a7672d9ff8f0adb7c65fb4e442128164fbc5f0964`,
passes the exact fixture and rejects reordered, relabeled, shifted, duplicated,
source-substituted, and sampling-receipt-substituted captures. Semantic receipts
are themselves content-addressed and are rechecked during dictionary-bundle
admission. The subsequent training command is:

The exact storage receipt is
`artifacts/vpd_compartment_capture_resources_v1/plan.json`, ID
`c99dd5c6e06bc5408aedc8df468d2b5d98a5825b73ca8e52fe6969b3b13d6ee0`.
For layer 35 it projects 805,376 `sae_training` rows in 197 shards and
604,032 `calibration` rows in 148 shards: 1,409,408 rows total and respectively
6,597,640,192 and 4,948,230,144 FP16 bytes. Role-specific episode and row
counts are checked against the split before an admissible capture can be
produced. A balanced shared dictionary would represent 585,728 rows and
9,596,567,552 bytes after FP32 conversion. Monolithic NPZ capture and full
training concatenation are therefore forbidden. The worker streams metadata
and writes 197 content-hashed NPY shards (4,096 rows/32 MiB for full shards);
the trainer verifies them once, memory-maps them, and fetches only requested
rows. Training/validation partitioning is pair-atomic, so token windows or
conditions from the same prompt pair cannot leak across the validation boundary.

```powershell
python -m experiments.authz_mechinterp.train_sae `
  --activations training_capture `
  --capture-manifest training_capture/capture_manifest.json `
  --metadata training_capture/metadata.jsonl `
  --source-dataset artifacts/vpd_activation_category_capacity_split_v4/sae_training_episodes.jsonl `
  --activation-category-split-manifest artifacts/vpd_activation_category_capacity_split_v4/split_manifest.json `
  --activation-category replayed `
  --capacity-plan artifacts/vpd_compartment_sae_capacity_v2/plan.json `
  --capacity-profile bounded_context_pilot `
  --key layer_35__proposal_start `
  --expansion 2 `
  --top-k 16 `
  --seed 17 `
  --split-seed 17 `
  --output-dir fixed_sae/replayed
```

The dictionary manifest records the category contract, protocol, metadata
hash, source and selected row counts, all capture category counts, and the exact
capacity plan/profile receipt. Focused SAE planning rejects a compartment
dictionary when that receipt is absent, failed, or differs between the source
and compartment records. It also rejects a category packet paired with a shared or
different-category dictionary. Its evaluation capture must include both target
and cross-category rows, and intervention episodes are drawn only from the
target category. This keeps training compartmentalized while testing whether
learned features remain specific outside their own compartment.

Capacity is only the admission gate for training; dictionary quality has a
separate calibration-only gate. Train at least two fixed-seed replicas for each
of the eight specific compartments and for `shared_cross_category`, then score
all eighteen dictionaries on the fresh `calibration` capture. A specific
dictionary must reconstruct its target category with centered explained
variance at least `0.70`, beat its best cross-category reconstruction by at
least `0.20`, keep dead features at or below `0.25`, and allocate at least
`0.20` of features preferentially to its target at a `2x` mean-activation
ratio. The shared dictionary must achieve at least `0.25` explained variance
in every category with a range no larger than `0.15`. Replicas must reproduce
at least `0.90` of decoder directions at absolute cosine `0.80`, and at least
half of target-selective features must also reproduce their category profile.
Exact or near-exact shared/specific decoder duplication is reported at cosine
`0.98` but is not itself a rejection: shared structure is intentionally present
in specific dictionaries.

The executable model-free failure-boundary receipt is
`artifacts/vpd_compartment_sae_quality_benchmark_v1/report.json`. Its non-square
synthetic dictionaries use the production artifact orientation
`[dictionary_feature, hidden]`; this guards against the row/column convention
mismatch that square fixtures previously concealed. The clean replicated
partition passes, while collapsed-specific, cyclic-swap, dead-feature, and
unstable-replica variants reject. A sixth scenario proves that bounded
streaming produces the same dictionary and replica gate decisions as the
in-memory reference while respecting its read and allocation bounds. This
receipt validates the quality protocol,
not Bonsai SAE quality. No Bonsai activations or weights were loaded to produce
it.

The production evaluator is
`experiments/authz_mechinterp/run_compartment_sae_quality.py`. The complete
pre-compute experiment is frozen in
`artifacts/vpd_compartment_sae_training_plan_v3/plan.json`, ID
`d4ef3eca0c642d22f3df415bacb61d5a2130fed969570eb6836777589d1301dd`.
It contains two capture jobs, eighteen training jobs (seeds 17 and 29 for each
of nine compartments), one dictionary-bundle admission job, and one quality
job. Admission verifies that all eighteen artifacts and manifests match their
exact planned compartment, initialization seed, common data-selection seed,
capture, model, trainer, capacity authorization, pair-atomic split, and output
path. It emits a content-addressed receipt but makes no quality claim. Both
replicas use data-selection and pair-atomic split seed 17; only initialization
changes. This removes the earlier confound where the single `--seed` controlled
both initialization and the examples assigned to train/validation. The plan estimates 2,782,208
training activation-row visits, 1,409,408 total capture rows across the two
roles, and at least 4,832,722,944 bytes for the eighteen
numeric dictionary artifacts; checkpoints, optimizer state, framework overhead,
and GPU feasibility remain explicitly unestimated. Its status requires a
frozen model identity and separate compute authorization—it is not an execution
receipt. It is also a preserved predecessor blueprint: the release-gate plan
above, not direct V3 execution, is the canonical orchestration entry point.

The production evaluator accepts only a passed dictionary-bundle admission
receipt tied to this exact training plan, plus a
content-addressed, base-model, prompt-window capture from the `calibration`
split and exact dictionary manifests from `sae_training` under the same split
ID. It verifies every activation shard before opening memory maps. It then
selects exactly 64 whole 128-token episodes per observable category by a frozen
stable-hash rule (65,536 rows total); neither model outcomes nor final-evaluation
rows can participate in selection. The selected episode, pair, and row-index
sets are content-addressed in the quality report.

Capture the complete calibration role with the same prompt-only storage
contract used for training. Capturing the whole role keeps the immutable split
artifact authoritative; deterministic quality subsampling happens afterward:

```powershell
python -m experiments.authz_mechinterp.capture_activations `
  --input artifacts/vpd_activation_category_capacity_split_v4/calibration_episodes.jsonl `
  --activation-category-split-manifest artifacts/vpd_activation_category_capacity_split_v4/split_manifest.json `
  --activation-category-split-role calibration `
  --model <frozen-base-model> `
  --layers 35 `
  --phases pre_gate `
  --sae-training-window 128 `
  --capture-resource-plan artifacts/vpd_compartment_capture_resources_v1/plan.json `
  --activation-shard-rows 4096 `
  --output-dir calibration_capture
```

After two fixed-seed dictionaries exist in each of the nine compartments,
admit the run directory. This verifies outputs against the immutable plan; it
does not evaluate feature quality:

```powershell
python -m experiments.authz_mechinterp.admit_compartment_sae_dictionary_bundle `
  --run-dir . `
  --training-plan artifacts/vpd_compartment_sae_training_plan_v3/plan.json `
  --output dictionary_bundle/dictionary_bundle_receipt.json
```

Then run the bounded evaluator. Supply eighteen `--dictionary-dir` arguments;
each directory must contain `sae_dictionary.npz` and its manifest:

```powershell
python -m experiments.authz_mechinterp.run_compartment_sae_quality `
  --capture-dir calibration_capture `
  --capture-manifest calibration_capture/capture_manifest.json `
  --metadata calibration_capture/metadata.jsonl `
  --activation-key layer_35__proposal_start `
  --training-plan artifacts/vpd_compartment_sae_training_plan_v3/plan.json `
  --dictionary-bundle-receipt dictionary_bundle/dictionary_bundle_receipt.json `
  --dictionary-dir fixed_sae/valid_authority/seed-17 `
  --dictionary-dir fixed_sae/valid_authority/seed-29 `
  --dictionary-dir <repeat-for-the-other-eight-compartments> `
  --batch-rows 256 `
  --output quality/compartment_sae_quality_report.json
```

For the bounded 8,192-feature pilot, the evaluator reads one replica at a time,
never materializes the full activation or feature matrix, and matches decoder
directions in 256-feature blocks instead of allocating an 8,192-by-8,192 cosine
matrix. Its conservative numeric-array peak is 680,525,824 bytes, excluding
Python object and NPZ-reader overhead. The report records the observed bound and
the fact that all three forbidden full-matrix allocations remained false.

Focused feature screening uses a compartment-aware endpoint. For any of the
eight specific compartments, the selection label is membership in that
observable activation category versus all other categories; the global model
outcome label is forbidden for feature selection. This avoids selecting a
generic PROPOSE/REQUEST feature merely because invalid authorization categories
are mostly outcome-constant. `shared_cross_category` retains the behavior label
because its purpose is explicitly cross-category. In both cases, the held-out
causal endpoint remains the signed authorization decision margin; category
membership is never substituted for the intervention outcome.

The current executable model-free failure-boundary receipt is
`artifacts/vpd_compartment_sae_screening_benchmark_v2/report.json`, ID
`8f06090f64db0a1fadcaec50e2f14bf73eff3fe8e3e372de836dd55f8a501b12`.
On a paired synthetic fixture with separate known category and behavior
features, category-specific screening selects the category feature, the legacy
global endpoint selects the different outcome feature, and shared screening
preserves the behavior feature. This is protocol evidence only and does not
establish a Bonsai feature or causal mechanism.

Focused planning now also requires the supplied discovery effect archive and
merge manifest to match the exact hashes stored in its cluster packet. The
eight-scenario model-free receipt at
`artifacts/vpd_focused_sae_input_lineage_benchmark_v1/report.json`, ID
`72a45142b1817269e4b991556585ad9e139c02290c31753edcc994d87ec783fe`,
passes the exact pair and rejects archive substitution, merge-manifest
substitution, and category packets lacking effect lineage. Metadata-only
predecessor packets remain usable solely as
`unattested_legacy_metadata_pilot`; they cannot support empirical category
claims. The independent signed-result reducer repeats the archive/manifest
comparison and rejects substituted signed inputs or a worker receipt that
merely asserts an unverified lineage status.

Shared components use a separately attested dictionary with an equal number of
whole training episodes from every observed category; prompt-window rows are
never split during balancing:

```powershell
python -m experiments.authz_mechinterp.train_sae `
  --activations training_capture `
  --capture-manifest training_capture/capture_manifest.json `
  --metadata training_capture/metadata.jsonl `
  --balanced-across-activation-categories `
  --capacity-plan artifacts/vpd_compartment_sae_capacity_v2/plan.json `
  --capacity-profile bounded_context_pilot `
  --key layer_35__proposal_start `
  --expansion 2 `
  --top-k 16 `
  --seed 17 `
  --split-seed 17 `
  --output-dir fixed_sae/shared
```

Shared-component intervention plans freeze separate episodes for every
family/category cell, and reducers report activation and parameter effects per
category as well as in aggregate. This prevents a broad component from looking
category-specific only because one category dominates the evaluation mix.

`compute_vpd_effect_shard.py` is the worker-side runner. It verifies the packet
and adapter hashes, selects one deterministic stratified prompt set, captures
the final prompt state before generation, and estimates each assigned LoRA rank
component with the mean central difference between multipliers `1+epsilon` and
`1-epsilon`. The exact estimator, epsilon, episode ordering, runner, model,
adapter, dataset, and effect space are frozen in the receipt. LoRA factors are
restored and checked after every component before the unsigned receipt is
emitted for worker signing. Receipts claiming generated tokens in a
`proposal_start` effect space are rejected.

## Focused SAE worker study

The fixed dictionary and its evaluation capture must both come from the exact
common activation space and the same frozen base-model revision. They are
separately content-addressed: the dictionary is trained on an independent base
capture, while behavior is evaluated on a different held-out base capture.
`capture_activations.py` now writes deterministic NPZ bytes and a
content-addressed `capture_manifest.json`; `train_sae.py --capture-manifest`
writes `sae_dictionary.npz` plus a content-addressed dictionary manifest.

For a category-aware packet, `run_focused_vpd_sae_v2.py` first creates an
immutable intervention plan. The preserved V1 runner is historical predecessor
evidence and must not plan a category-specific intervention:

```powershell
python -m experiments.authz_mechinterp.run_focused_vpd_sae_v2 `
  --bundle artifacts/bonsai8b_authz_v8/vpd_sae_clusters_effect_round `
  --packet-id PACKET_ID `
  --dictionary fixed_sae/sae_dictionary.npz `
  --dictionary-manifest fixed_sae/sae_dictionary_manifest.json `
  --dictionary-quality-report quality/compartment_sae_quality_report.json `
  --activations heldout_capture/activations.npz `
  --activation-capture-manifest heldout_capture/capture_manifest.json `
  --metadata heldout_capture/metadata.jsonl `
  --effects artifacts/bonsai8b_authz_v8/effect_merge/activation_effects.npz `
  --effect-merge-manifest artifacts/bonsai8b_authz_v8/effect_merge/effect_merge_manifest.json `
  --intervention-dataset heldout_episodes.jsonl `
  --output-dir worker-study
```

Screening residualizes prompt length and the exact anchor position. The
permutation max statistic is recomputed over the full SAE dictionary on every
shuffle, including features not selected by the observed labels. Held-out
family scores use nested feature selection: a feature is evaluated in a family
only when selected without that family's labels. VPD/SAE cosine is computed
only after the same-space invariant passes.

The plan freezes candidate component/feature pairs, matched-norm control
components, coefficients, prompt families, exact episode IDs, and prompt
hashes. `compute_focused_sae_interventions.py` executes its full factorial grid:

```powershell
python -m experiments.authz_mechinterp.compute_focused_sae_interventions_v2 `
  --plan worker-study/intervention_plan.json `
  --bundle artifacts/bonsai8b_authz_v8/vpd_sae_clusters_effect_round `
  --dictionary fixed_sae/sae_dictionary.npz `
  --effects artifacts/bonsai8b_authz_v8/effect_merge/activation_effects.npz `
  --dataset heldout_episodes.jsonl `
  --model prism-ml/Bonsai-8B-unpacked `
  --revision d916578504398e3d38753127e1fdeafb82ae4f0f `
  --adapter adapter `
  --expected-adapter-sha256 a95433ba5ceb9b6375ac649549a098ea4e4b2ad5695e095d49be61d0c3ee38e6 `
  --worker-id WORKER_ID `
  --output worker-study/intervention_trace.json
```

For every frozen prompt it measures activation-direction steering, exact LoRA
rank scaling, and the frozen matched-control rank at coefficients `-1, 0, 1`.
The endpoint is `logsumexp(REQUEST)-logsumexp(PROPOSE)` at the first decision
token. Target and control factors are restored and byte-checked after every
cell. Finalization rejects missing or duplicate factorial cells, changed
prompts or targets, inconsistent margins, incomplete held-out families,
guardrail-label contradictions, and mismatched model/dictionary identities.
Run `run_focused_vpd_sae.py` again with `--intervention-trace` to emit the
unsigned worker result, then sign and submit it through the web worker client.

## Guarded runtime V2 (implemented, benchmarked, not run)

Research protocol V9 adds an executable V2 capture/training/admission/quality
path without changing the empirical boundary. The exact 22-job plan is
`artifacts/vpd_compartment_sae_training_plan_v4/plan.json`, content ID
`a7a050abead2d60a63e16a6d07126609ac38c034f18ad3e37f98cd7c9f4d8c2b`.
It comprises two activation captures, 18 fixed-dictionary replicas, one bundle
admission, and one calibration quality job. Every job requires both a passing
scientific-release receipt and a separate external authorization covering its
exact V2 job ID. A scientific-release receipt always keeps
`compute_authorized=false`; the repository verifies but does not generate the
external authorization.

Dictionary manifest V2 binds batch size, learning rate, seed, compartment,
exact episode/pair/window sample identities, capture execution receipt, and
both authorization receipts. Bundle admission independently reproduces all 18
planned jobs and verifies their training execution receipts. Quality evaluation
independently verifies calibration capture execution, bundle coverage, and the
same external authorization scope.

The model-free runtime benchmark at
`artifacts/vpd_compartment_sae_runtime_benchmark_v2/report.json`, content ID
`f044212429c08737719ffe1f64496122d20d94f2f6b4e2333c42d00d452b185c`,
passes 11/11 positive and negative scenarios. It includes rejection of changed
batch size, learning rate, exact sample contract, wrong-job authorization,
incomplete bundle job coverage, and a V1 manifest presented to the V2 bundle
admission path. This is synthetic contract evidence only.

Research protocol V9 has SHA-256
`3d4397f3ef270f4d2b0bc00995034543f34a60dedfc9c9587eb06963e2b5bd9c`.
Unified lineage receipt V7 has content ID
`e97ea34df8a8e4c12f4b13e5bdf02fe9ed0fe985a1dfdad8237734420a7dc90c`.
These research labels are not historical Bonsai experiment versions. The V2
runtime is explicitly reconstructed implementation; recovered predecessor
evidence remains preserved and is not relabelled as reconstructed.

No scientific-release receipt, external compute authorization, activation
capture, checkpoint, dictionary, empirical bundle, or quality report exists.
Consequently the runtime is reviewable but cannot start any model job.

## Within-category cluster discreteness V2

Cross-archive adjusted-Rand stability and mean silhouette do not by themselves
prove that a category contains discrete mechanisms. A stable continuous
direction or strength gradient can be reproducibly sliced by k-means. The
model-free V2 benchmark at
`artifacts/vpd_activation_category_cluster_discreteness_benchmark_v2/report.json`,
ID `3c72d47b3662ccc78bdf2470850f0a644e597226cde988b78335b07a72609d91`,
reproduces two such legacy false-split scenarios.

Selection V2 retains every previous projection, pair-block, independent-archive,
mean-silhouette, and minimum-size requirement. It additionally requires a
positive lower-tail silhouette and an explicit empty cluster boundary. The
continuous strength and direction controls fall back to one cluster in all
eight categories, while the separated two-mechanism control remains two
clusters in all eight categories.

Research protocol V10 has SHA-256
`1b53f290aa05a2aefefea665202ba6c22e045c019a05b5796028e895fd062f1c`.
Unified lineage V8 has content ID
`6315c32b329278aba56f8a29d84ff9c5e9ebaccfe32a168bb36d6d70987a26d9`.
The independent V2 admission wrapper is implemented and its production-shaped
benchmark passes. It preserves a genuinely discrete candidate and rejects a
continuous strength-gradient candidate without mutating or relabelling either
bundle.

## Semantic-stratum cluster admission V3

V2 recovers known two-, three-, and four-mechanism model order, including an
imbalanced three-mode control. It nevertheless leaves a VPD-specific failure:
aggregate, projection, archive, and pair-block stability can all agree on
clusters that are actually condition or template partitions. Per-component
condition/template coverage does not prove that within-category cluster
membership itself is invariant across those strata.

Admission V3 independently reconstructs leave-one-condition-out and
leave-one-template-group-out category profiles from the signed CountSketch
sufficient statistics in both discovery and cluster-validation archives. The
same V2 separation and stability policy is then rerun over the expanded
ensemble. Exact agreement with the frozen candidate allocation is required;
otherwise the candidate is rejected without mutation or relabelling.

The direct semantic-strata benchmark passes 3/3 scenarios. Its genuine
four-mechanism control remains four clusters in all eight categories, while
condition and template confounds fall back to one cluster. The
production-shaped benchmark at
`artifacts/vpd_cluster_candidate_semantic_strata_admission_benchmark_v3/report.json`,
ID `7d0528afae3369fcbfa4289be09bd2e453b2cc8b085d43dfdc585557ddb8da22`,
passes 2/2: V2 admits the condition confound, V3 rejects it in all eight
categories, and V3 preserves the genuine four-mechanism bundle.

Research protocol V12 has SHA-256
`b384e6f3d428698eccf2af6aaa41ab9c026c0f4fa77c81ad4194f53218d66a43`.
Cheap-executor runbook V4 has content ID
`56c652375cf68d638b5368e14ca3c89eaa43a3ba17007278839c740dcc0e774f`.
The next empirical category candidate must pass semantic-strata admission V3;
V1 selection or V2-only admission is prohibited. No empirical candidate or
admission receipt currently exists.

## LoRA factor-basis identifiability

The historical inventory locates exact factor-coordinate terms using
`scale * B[:,rank_index] outer A[rank_index,:]`. Those terms sum to the adapter
delta but are not individually invariant under an equivalent LoRA rank-basis
change. The model-free benchmark at
`artifacts/vpd_lora_component_basis_identifiability_benchmark_v1/report.json`,
ID `4c851a3f717a2c22694e6b77c3a149f64d0309f3e770fad9e54a135d4e3d0c27`,
keeps the full delta fixed to numerical error while raw category-specific
components change from 16 to zero.

The successor component definition uses a thin SVD of the low-rank delta.
Resolved singular directions define canonical rank-one spectral components;
near-degenerate directions are retained as one unresolved aggregate spectral
block. The computation requires only QR factorizations of the LoRA factors and
an SVD of the small rank-by-rank core. A model-free utility also constructs
same-shape LoRA factors for exact canonical block multipliers without mutating
its inputs.

Research protocol V13 has SHA-256
`7d7392a912d3bfe7174bb64b7f974d783f298afe354d3ed032052a5e5a6a820d`.
Runbook V5 has content ID
`95ca2f3503fdb0cedd804efb00fb2a9950e648814942742df9e77ffdffabee25`.
That predecessor established that raw factor-rank component identities are
prohibited for the next empirical candidate and stopped because canonical
inventory/runtime integration was absent. Recovered raw-rank artifacts remain
valid historical evidence at their stated granularity and are not relabelled
as canonical evidence.

## Canonical factor-bundle inventory and restoration runtime

The successor factor-bundle emitter writes deterministic thin-SVD spectral
block inventories without forming the full dense adapter delta. Its synthetic
benchmark is content-addressed at
`artifacts/vpd_canonical_lora_component_inventory_benchmark_v1/report.json`, ID
`f3a38b715b27be5864c5080320340d526be1f77041f2b4801c946430802f672c`.
Equivalent raw-factor rotations agree structurally and within `1e-12`
numerically, five canonical components are emitted, one exactly degenerate
block remains explicitly unresolved, and artifact tampering is rejected.

The model-free NumPy runtime installs canonical block multipliers into
same-shape factor arrays, verifies a frozen random probe, and restores exact
original bytes even when the callback raises. Its benchmark at
`artifacts/vpd_canonical_lora_factor_runtime_benchmark_v1/report.json`, ID
`007b1fb898ba7952c309393955e1b85bc72fbf1d7635fe784384b0bbaefa6297`,
passes multipliers `-1`, `0`, `1`, and `2` plus an injected failure.

These receipts establish only the factor-bundle and model-free runtime
contracts. Authenticated extraction from a real adapter and production
Torch/PEFT effect-capture integration are still absent. Protocol V14, SHA-256
`4ca2e1c6c9a1491424ce63d6717c841c72328c903077ca98d44e0aaaf5eb60b8`,
and runbook V6, content ID
`8b887d5dc583ed85b83cf597bbdaef0a13f1fde3ecbafe1045b2477f4226da76`,
bind this boundary and keep all empirical capture and SAE stages closed until
those missing integrations are independently verified.

## Authenticated plain-LoRA factor extraction

The successor extractor reads safetensors directly without importing Torch or
PEFT. It requires exact expected hashes for both the adapter weights and PEFT
configuration, rejects adapter extensions outside the frozen plain-LoRA
contract, validates all factor pairs and scales before writing, and emits the
existing deterministic factor-bundle index consumed by canonical inventory V1.

The model-free benchmark at
`artifacts/vpd_authenticated_lora_factor_bundle_extraction_benchmark_v1/report.json`,
ID `c61fbf5a051f39f9374d415dd31a609ad9517e1efc980762e7c92d720bf4c605`,
recovers two synthetic modules and their per-module scales exactly, verifies the
downstream canonical inventory, and passes seven negative controls. Those
controls cover weights/config hash tampering, orphan factors, unexpected
tensors, rank mismatch with no partial output, nonempty output, and tampered
factor artifacts.

Protocol V15 has SHA-256
`e57f3acca4774dc92c5b3a1609c9211f70edc64bde289dc37721ebe180fb271d`.
Runbook V7 has content ID
`b7beb4b1ad9a88465d9f38333a229da0c25792005c30de7da907c6d8d769627f`.
They mark the extractor implementation ready but retain two hard stops: no
authorized real-adapter factor-bundle receipt exists, and production Torch/PEFT
canonical effect capture remains unimplemented. The synthetic receipt is never
presented as empirical Bonsai evidence.

## Production-shaped canonical PEFT factor bridge

The successor runtime resolves one unique live PEFT module and adapter lane,
checks the live factor hashes and scaling against the authenticated/canonical
artifacts, installs one canonical spectral-block multiplier, verifies frozen
probe outputs, invokes a callback, and restores exact tensor values. Torch is
imported lazily only when no injected no-grad context is provided. BF16 factor
inspection has a float32 fallback, while restoration additionally requires
exact tensor equality.

The fake-object benchmark at
`artifacts/vpd_canonical_lora_peft_runtime_benchmark_v1/report.json`, ID
`9474f42a0cc59297a6e52cba1ca5a8b709e3e6c981e2693b3a70145eeea21de5`,
chains authenticated extraction and canonical inventory, passes multipliers
`-1`, `0`, `1`, and `2`, restores after callback failure, and rejects source
hash mismatch without mutation, live-scale mismatch, ambiguous module matches,
and canonical-artifact tampering.

Protocol V16 has SHA-256
`58132c36d857a8a9958adf1d02a3e65a9b1ef098518d762e03f4d08087508bdd`.
Runbook V8 has content ID
`2dd26b4a7ceab0c96e9ac8530180fe0ff81cff9be7d6e39a79db655ad8222f6e`.
They mark the production-shaped bridge implementation ready but do not claim
actual Torch/PEFT execution. A real-adapter extraction receipt, explicitly
authorized Torch smoke receipt, and integration into a new successor effect
worker remain mandatory before empirical component capture can start.

## Category-preserving canonical effect capture

The successor kernel executes two fully restored canonical block lanes at
`1+epsilon` and `1-epsilon`, retains the signed central difference for every
captured row, and emits deterministic category counts, category means, a direct
row-weighted global mean, and hashes for every category sample matrix. This is
the data granularity needed for downstream activation-category clustering and
for a worker to construct condition/template robustness sketches.

The fake-object benchmark at
`artifacts/vpd_canonical_category_effect_capture_benchmark_v1/report.json`, ID
`d33457b5799f93c3c8b319eff3de67a81e290be76e552c4b98044ba03e53205b`,
recovers exact effects for synthetic invalid- and valid-authority categories.
It passes callback-failure restoration, lane-shape mismatch restoration,
nonfinite capture rejection, empty-category rejection, and invalid-epsilon
rejection.

Protocol V17 has SHA-256
`7740bdb5f7d47d5f4247f4feedc754887fd26d702006a469a9fd38bd14691feb`.
Runbook V9 has content ID
`f6fb6fec15440bdc5c307801ec361afa0b40f50b136516c0c3c94b18410d310b`.
They mark the kernel ready while keeping the successor worker closed. That
worker must still bind exact prompt order, dataset and split hashes, per-row
category identities, and condition/template CountSketch sufficient statistics;
actual model execution also remains unauthorized and unverified.

## Canonical effect-shard archive assembly

The model-free successor-worker assembler accepts the content-addressed capture
result for every canonical spectral block and binds each retained effect row to
sorted episode, pair, activation-category, condition, and template-group
identity. It requires verified semantic split V2/V3 provenance with exact
episode, pair, and template-group set hashes. Raw LoRA rank IDs are rejected.

The output uses the existing effect-shard array contract: global and per-category
effects, pair-block CountSketch counts/sums, condition counts/sums, and
template-cell counts/sums. The production minimum sketch dimension is 256, all
category pair sets and template groups must agree, and both the assembly and
written NPZ artifact are content-addressed.

The chained benchmark at
`artifacts/vpd_canonical_effect_shard_assembly_benchmark_v1/report.json`, ID
`a722200d27ae115a92e6b77d865eba137b6fd0111d6adda11ace0937e400e30f`,
assembles two canonical components across eight synthetic episodes and passes
nine negative controls. Protocol V18 has SHA-256
`97c445d5ba9ab1ea833e1fd223e2e7d2e576c46451d88775a70488aec4f66e84`;
runbook V10 has content ID
`d2b30149591145310b25b73f36920c1916ac8a350e4676c277e7b2de0c087713`.
The model-free worker half is ready. The successor model loader/capture CLI and
authorized actual-model receipts remain absent.

## Exact capture-job planning and guarded successor worker

The V19 successor closes the implementation gap without claiming empirical
capture. The content-addressed planner binds a frozen model identity, exact
adapter/config and extraction hashes, canonical SVD inventory, semantic split
V2/V3 identity, dataset support, component assignments, activation key,
category counts, central-difference settings, semantic-sketch parameters,
output location, and a computed maximum forward-pass count. Planning always
emits `execution_authorized: false`; its verifier accepts only a pre-existing
authorization receipt scoped to the exact plan ID, output, and pass ceiling.

The lazy worker reproduces and authorizes the plan before importing Torch,
Transformers, or PEFT. It batches exact sorted rows within all eight activation
categories, calls the canonical effect kernel for each SVD block, persists the
per-example category effects, constructs condition/template and pair-block
sketches through the model-free assembler, uses atomic staging, and emits a
separately verified execution receipt. Neither the authorization nor execution
receipt grants release, real-action, or SAE-training authority.

The planner benchmark ID is
`91186f71a7b57364b1ecfd1fbe879eee811c7504db2605f505e2cc0338b14511`;
its nine controls cover adapter tampering, raw-rank identity, unsafe rows,
category omission, mutable model revision, readdressed plan tampering, exact-job
authorization, persistence scope, and synthetic provenance. The guarded worker
benchmark ID is
`df90986f90badd27512b4539093aac77198331409bc7719e1d17ad05b14d6746`;
it executes 32 planned fake callbacks for two components across eight categories
and passes eight restoration, cleanup, scope, and artifact controls with Torch
and PEFT absent.

Protocol V19 has SHA-256
`a689a19dc28ba3a2822b8f762625e467ba2515ba9f6faf665da140628da888a7`.
Runbook V11 has content ID
`ea909e4c81514150353be464e87f4f464469184d070bc72c4121bc67893af5bc`.
Implementation readiness is now true; empirical readiness remains false because
no real-adapter extraction, immutable real-model identity, genuine user-scoped
authorization, actual Torch/PEFT smoke, or empirical category-capture receipt
exists.

## Exact label-invariant cluster membership

The V20 audit found that V3 admission reproduced category cluster counts but did
not separately require the frozen candidate's component membership to equal the
independent reconstruction. Equal counts do not imply equal partitions: two
clusters can remain two clusters while components move between them. Training
separate SAE dictionaries on such unstable membership would silently change the
compartment definition.

Partition identity V4 represents a partition by pairwise component
co-assignment, so arbitrary numeric labels or cluster names have no effect. It
requires exact membership agreement across discovery projections, independent
cluster-validation projections, discovery and validation pair-block omissions,
condition omissions, and template-group omissions. The resulting consensus
partition must then exactly match the frozen candidate assignment. Any mismatch
rejects without mutating or relabelling the candidate. Count-only admission is
prohibited.

The direct benchmark at
`artifacts/vpd_activation_category_cluster_partition_identity_benchmark_v4/report.json`,
ID `3128fd74149c1fa6763d97a67269094d3c465a3d4f8f29878287c44e4b519c67`,
passes six controls: a stable two-mechanism partition is admitted in every
category, same-count membership substitution is rejected, cluster-label
renaming is invariant, one-replicate membership drift is rejected, the changed
category is identified, and the candidate remains frozen.

The production-shaped admission benchmark at
`artifacts/vpd_cluster_candidate_partition_identity_admission_benchmark_v4/report.json`,
ID `0246dde56cb758cbfb5fb42775bc69c3466b7e94003ed81884589f2c9b4f2ca3`,
admits the stable four-mechanism candidate with exact membership in all eight
categories and retains the semantic-confound rejection. Protocol V20 has
SHA-256 `b4edb971141764c6c6a4ac34c7971a4968ac93400f8c0e3c4340ae3d91df1199`;
runbook V12 has ID
`61e128644cce58c4c873ff2fd01d26b44ec6c021c70c60a6a70a7379259e20da`.
No empirical partition-identity admission receipt exists.

## Cluster-to-episode routing gap

The exact V4 component partition is not yet a cluster-specific SAE sample
partition. A reproducible audit of the current V4 SAE plan finds 18 jobs across
nine category/shared compartments and zero cluster-bound jobs. All 16 specific
jobs select by exact observable activation category; none binds a cluster ID,
component-membership hash, allocation hash, or routing receipt. Audit ID:
`8a873f01610ce17313bce07459e64dc80ded7bdb6f23d4011ae0f15e57fb2793`.

Copying the same category rows into one job per component cluster is
prohibited. A successor must first route episodes using authenticated
per-component per-example effects, normalize for cluster size, send ambiguous
episodes to an explicit shared compartment, prove discovery/validation
stability and pair-atomic disjointness, and then bind each successor SAE job to
the routing receipt and exact component membership. The model-free C0-C5 design
and stop rules are in
`VPD_CLUSTER_TO_SAE_CHEAP_EXECUTOR_HANDOFF_2026-09-01.md`.

Router V1 now implements that model-free bridge. It derives cluster IDs only
from canonical member sets, verifies per-component effect-array hashes and
episode order, aggregates RMS response without favoring larger clusters,
requires response and margin gates, and sends ambiguous, low-response, or
pair-conflicting episodes to `shared_ambiguous`. Independent routing stability
requires both overall and condition-stratified agreement. Benchmark ID
`0d5668cdbec6629f1632b3cf0f26a69695c95ec137fcaa2cae0342cde1056ac1`
passes 16 controls; an aggregate-preserving condition swap is rejected. This is
synthetic evidence only and grants no SAE-training authority.

The manifest-only successor planner then requires a third, disjoint
`sae_training` routing receipt under the exact frozen partition and routing
policy. Its benchmark ID
`d79ff71b50d0bfc0bddb649609234b6bf9fbf9be78ef985daa773ebd5687ca10`
passes 17 controls. Across two synthetic categories it creates three supported
specific compartments and one shared fallback, with two initialization jobs
per compartment. All episodes and pairs are assigned once, pair-atomic
train/validation splits are disjoint, job capacity matches the resolved
profile, and a 2-episode/1-pair route is merged to shared instead of
oversampled. No empirical plan or training authority was produced.

Protocol V21 binds the gap audit, router, planner, and both benchmark IDs with
SHA-256 `b26f5668a877923c4f45a877edd8836e65ec061d4132b9ca309abb3d9e74556a`.
Runbook V13 has content ID
`79f1c87f1cfa84fb05d575be0265cdf7e41405e1e791306e1b36c9b41b0ea757`.
It replaces the two defective legacy benchmark CLI summary commands with a
separate evidence wrapper while leaving the hash-bound V4 sources untouched.
The next missing implementation is an exact canonical capture-job successor
for a disjoint `sae_training` role; empirical routing and all SAE execution
remain unauthorized.

Capture planner V2 now closes the planning half of that gap without changing
V1. Benchmark ID
`525fbf3433a410ad883b438ee301285f31d704815a86f955751a9cb29b0f09f2`
passes 12 controls: V1 continues to reject the new role, V2 binds the exact
role, dataset, components, and forward-pass ceiling, and authorization cannot
grant dictionary training. The recovered V3 `sae_training` split validates at
6,292 episodes and 572 shared pairs across all eight categories. Protocol V22
SHA-256 is
`4a59df7c095e407be02ebfa5f01504eee6f98bc0872b4c9379c3f40ea98d016c`;
runbook V14 ID is
`bea30e4ef92a8237f44db800db816db436e2c29f9ec7233b37eff9ac8798690e`.
At the V22/V14 checkpoint the V2 execution worker and empirical
plan/authorization remained absent.

Capture worker V2 now closes the synthetic execution half of the role gap. Its
benchmark ID
`2023baf1646230680ed7d23d84fb1c6174f63d99abcc8cf629580f8f77086bf3`
passes 16/16 controls with exactly 32 fake callbacks across two canonical
components and eight activation categories. It verifies the V2 execution
receipt, exact `sae_training` role binding, factor restoration and runtime
closure on success and failure, atomic cleanup, V1/V2 schema separation, and
component/effect-shard tamper rejection. Torch, Transformers, and PEFT remain
absent; no model, inference, weights, SAE dictionary, or empirical archive was
used.

The benchmark exposed a downstream role mismatch: assembler V1 correctly
retained its historical role set and rejected `sae_training`. A new assembler
V2 extends only the role gate, leaves V1 byte-unchanged, reuses its
role-independent array kernel, and persists the exact original role in the
content-addressed receipt. Protocol V23 has SHA-256
`8eda03fd5c982af058da674feff55c223b58598a8d38422fd8bb27afc34e5c49`;
Runbook V15 has content ID
`60e3a3e8778a2efe435eeaa904d58c11621add35ed105345974fad26e5aa2699`.
They mark the synthetic worker ready but keep safe stage zero, empirical
capture absent, execution unauthorized, and SAE training unauthorized. The
next model-free gap is a receipt-to-router adapter plus a three-role synthetic
capture-to-routing-to-cluster-aware-plan integration benchmark.

That integration benchmark now passes 18/18 controls with benchmark ID
`b27a7f901d191a81a3bc4e708244edf1a3325e968b979471d064986d2c773df6`.
It reads all router arrays from verified component capture artifacts across
`discovery`, `cluster_validation`, and `sae_training`, covers all eight
activation categories, and rejects episode-order, effect-hash, role, and
adapter-receipt tampering. All three fake runtimes restore their factors and
close; model frameworks remain absent.

The integration also found a production-shape defect in cluster-aware planner
V1. The recovered split shares pair IDs across activation categories. V1 treats
any pair appearing in different category-cluster compartments as a conflict,
so the synthetic real-shape fixture produces zero specific compartments and
collapses all rows to shared. Planner V2 leaves V1 unchanged and introduces one
content-addressed global pair split. A pair may span category compartments, but
all occurrences must use the same training or validation side. Exact episode
and pair lists are persisted per compartment. Under the identical fixture V2
preserves 16 specific compartments plus one shared fallback, emits 34
manifest-only jobs, and assigns all 80 `sae_training` episodes once. Routing
effects total 80 rows while the SAE sample contract totals 10,240 activation
rows, preserving the unit distinction.

Protocol V24 has SHA-256
`c42dc6e43296c7e4a3d6054174e015620221cf141377bff44773d8cfc0add2b2`;
Runbook V16 has content ID
`321b0cab4fc95d0270395f39ba95d0139a3ea258e3cb9c05822b6c82f6ce5dd3`.
They prohibit planner V1 for empirical splits with shared cross-category pair
IDs, require the global pair split, and keep empirical capture, SAE training,
and execution unauthorized.

## Scientific boundary

Clusters prioritize where to spend SAE compute. They do not establish
individual-component causality. Promotion still requires a fixed base SAE,
held-out prompt families, length/position residualization, a max-stat screen,
matched VPD controls, guardrails, and signed agreement between activation and
parameter interventions.

For compartment dictionaries the focused planner first verifies a
content-addressed `sch.vpd-compartment-sae-quality-report.v1`. The report must
bind the exact dictionary artifact and manifest, capacity authorization,
activation-category split ID, and a calibration-role capture. Both the selected
dictionary and its whole compartment replica gate must pass. Final-evaluation
prompts may never tune these thresholds, and the quality screen is never
treated as causal evidence; held-out signed interventions remain the promotion
criterion.
