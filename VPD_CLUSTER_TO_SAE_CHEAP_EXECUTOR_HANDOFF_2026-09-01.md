# VPD cluster-to-SAE handoff for a cheaper executor

Date: 2026-09-01  
Project root: `C:\projects\VPD`  
Protocol authority: `configs/evals/vpd_activation_category_compartments_v22.json`  
Machine authority: `artifacts/vpd_cheap_executor_runbook_v14/runbook.json`  
Scope: local, model-free design and synthetic verification only  
Model inference, external compute, package installation, and weight writes authorized: **no**

## Outcome

Exact V4 partition admission now proves that independently reconstructed
within-category clusters have the same canonical-component membership as the
frozen candidate. That result is useful, but it does **not** yet flow into SAE
training.

The current SAE training plan has 18 jobs: two initialization seeds for each of
eight observable activation categories and one shared cross-category
compartment. It contains zero `cluster_id` values, zero component-membership
bindings, and zero cluster-allocation hashes. Every category-specific job uses
`selection: exact_observable_activation_category`. For example,
`missing_or_untrusted_authority` selects all 2,288 category episodes and all
292,864 activation rows for each seed.

Therefore the current plan is category-compartmentalized, not
within-category-cluster-compartmentalized. It must not be described as training
one SAE dictionary per admitted VPD cluster.

C1 and C2 are now implemented locally as successor, model-free work. Router V1
uses membership-derived cluster identities, exact capture-row hashes,
cluster-size-normalized RMS response, absolute and relative response margins,
pair-atomic primary routing, and an explicit `shared_ambiguous` fallback. Its
synthetic benchmark passes 16/16 controls. It catches a condition-level route
swap with condition total variation `1.0` even though overall route total
variation is exactly `0.0`.

## Reproducible evidence

| Evidence | Observed value |
|---|---|
| Current SAE training plan | `artifacts/vpd_compartment_sae_training_plan_v4/plan.json` |
| Training-plan ID | `a7a050abead2d60a63e16a6d07126609ac38c034f18ad3e37f98cd7c9f4d8c2b` |
| Training-plan SHA-256 | `f90dc761017b15369fbc31bf8d9efdee4f4d8f6e9c7fce585e8ecc5e94b5975c` |
| Capacity-plan ID | `809deb31cba4642ee38cbaf2f7417120610b4fc9cfe6c37d0ed6cbaa885efb7d` |
| Capacity-plan SHA-256 | `3746589ad195625d91b505cb58eb371b464e2c36ed6319ffbaa4b70c5d45d0bf` |
| Training jobs | 18 |
| Unique compartments | 9 |
| Category-specific jobs | 16 |
| Shared jobs | 2 |
| Unique cluster IDs in jobs | 0 |
| Partition-identity benchmark ID | `3128fd74149c1fa6763d97a67269094d3c465a3d4f8f29878287c44e4b519c67` |
| Production V4 admission benchmark ID | `0246dde56cb758cbfb5fb42775bc69c3466b7e94003ed81884589f2c9b4f2ca3` |
| Reproducible cluster-routing gap audit ID | `8a873f01610ce17313bce07459e64dc80ded7bdb6f23d4011ae0f15e57fb2793` |
| V4 evidence-wrapper verification ID | `deb0759cc6c1b9436ce0675d6b7d0ec7347847099c04d84b2dad504375215dcd` |
| Cluster-to-episode routing benchmark ID | `0d5668cdbec6629f1632b3cf0f26a69695c95ec137fcaa2cae0342cde1056ac1` |
| Router controls | 16/16 pass |
| Manifest-only cluster-aware SAE plan benchmark ID | `d79ff71b50d0bfc0bddb649609234b6bf9fbf9be78ef985daa773ebd5687ca10` |
| Synthetic cluster-aware plan ID | `43ecfc3bfe7356d06cdcbb03f64eb50b9aeb0d083d2207b891357067cc526973` |
| Planner controls | 17/17 pass |
| Protocol V21 SHA-256 | `b26f5668a877923c4f45a877edd8836e65ec061d4132b9ca309abb3d9e74556a` |
| Runbook V13 ID | `79f1c87f1cfa84fb05d575be0265cdf7e41405e1e791306e1b36c9b41b0ea757` |
| Router/planner/lineage regression | 22 tests pass |
| SAE-training capture-plan V2 benchmark ID | `525fbf3433a410ad883b438ee301285f31d704815a86f955751a9cb29b0f09f2` |
| Capture-plan controls | 12/12 pass |
| Recovered SAE-training split | 6,292 episodes; 572 pairs; exact V3 hashes pass |
| Protocol V22 SHA-256 | `4a59df7c095e407be02ebfa5f01504eee6f98bc0872b4c9379c3f40ea98d016c` |
| Runbook V14 ID | `bea30e4ef92a8237f44db800db816db436e2c29f9ec7233b37eff9ac8798690e` |
| SAE-training capture-runtime V2 benchmark ID | `2023baf1646230680ed7d23d84fb1c6174f63d99abcc8cf629580f8f77086bf3` |
| Capture-runtime controls | 16/16 pass; exactly 32 fake callbacks |
| Protocol V23 SHA-256 | `8eda03fd5c982af058da674feff55c223b58598a8d38422fd8bb27afc34e5c49` |
| Runbook V15 ID | `60e3a3e8778a2efe435eeaa904d58c11621add35ed105345974fad26e5aa2699` |
| Current worker/lineage regression | 21 tests pass |
| Three-role capture-to-SAE integration benchmark ID | `b27a7f901d191a81a3bc4e708244edf1a3325e968b979471d064986d2c773df6` |
| Integration controls | 18/18 pass across three roles and eight categories |
| Planner V1 under shared cross-category pairs | 0 specific compartments; all rows collapse to shared |
| Planner V2 under the same shape | 16 specific + 1 shared compartment; 34 jobs; 80 episodes assigned once |
| Protocol V24 SHA-256 | `c42dc6e43296c7e4a3d6054174e015620221cf141377bff44773d8cfc0add2b2` |
| Runbook V16 ID | `321b0cab4fc95d0270395f39ba95d0139a3ea258e3cb9c05822b6c82f6ce5dd3` |
| Current capture-plan/cluster regression | 17 tests pass |
| Focused V20/V12 regression | 99 passed, 1 named recursive reproduction test deselected |

The current SAE-plan derivation copies predecessor jobs and carries only the
old `compartment` value into the training runner. The capture worker does,
however, persist one authenticated NPZ per canonical component containing
`delta_samples__<category>` arrays plus an episode-order hash. This makes a
model-free router feasible after an authorized empirical capture exists.

The two hash-bound V4 benchmark modules have a legacy CLI-summary defect: one
calls `len()` on an already-counted integer, and the other prints a null
scenario count after successful verification. Their underlying verifier
functions pass. They were deliberately left byte-identical because the
benchmark artifacts bind their source hashes. C0 uses the new non-mutating
`verify_vpd_partition_identity_evidence_v1` wrapper instead.

## Scientific stop condition

Do not create cluster-specific SAE jobs by giving every cluster the complete
category archive. That would duplicate the same examples under different
cluster names and would not demonstrate mechanistic compartmentalization.

Before a cluster-specific SAE plan may exist, a routing receipt must prove:

1. exact binding to a passed V4 component partition and its membership hash;
2. exact binding to authenticated per-component capture artifacts;
3. exact episode order and fixed activation-rows-per-episode;
4. a versioned, cluster-size-normalized routing score;
5. disjoint primary routing, with ambiguous or low-response episodes sent to a
   shared/fallback compartment rather than copied to every cluster;
6. discovery/validation stability without using final held-out data;
7. pair-atomic train/validation partitions after routing;
8. minimum support and capacity gates for every proposed dictionary.

## Proposed runs

### C0 — evidence preflight; allowed now

Run the commands in `VPD_CLUSTER_TO_SAE_CHEAP_EXECUTION_PACKET_2026-09-01.json`
under `C0_evidence_preflight`. Stop on the first mismatch. This is read-only
apart from ordinary local test caches and loads no model.

Expected result: V4 partition benchmarks verify, the existing plan reports 18
jobs and 9 compartments, and no training job contains a cluster identity.

### C1 — router contract V1; completed model-free

Implement a model-free module and tests only. The input contract must include:

- passed V4 admission receipt and label-invariant canonical component groups;
- authenticated capture execution receipt plus every component NPZ hash;
- category episode IDs in exact order and activation rows per episode;
- discovery or cluster-validation split identity;
- routing candidate ID and frozen thresholds.

Emit a content-addressed receipt with one primary route per episode or
`shared_ambiguous`, per-route support, response margins, pair IDs, membership
hashes, source hashes, and explicit `sae_training_authorized: false`.

Versioned scoring candidates for synthetic comparison:

| Candidate | Score | Purpose |
|---|---|---|
| `route-energy-rms-v1` | RMS effect over component members, episode activation rows, and hidden coordinates | Size-normalized baseline |
| `route-energy-margin-v1` | RMS baseline plus top-vs-runner-up margin and minimum-response gates | Primary hard-routing candidate |
| `route-prototype-cosine-v1` | Discovery-only cluster prototype with validation-only assignment | Direction-sensitive diagnostic |

Do not select among candidates using a final held-out pack. Do not invent
empirical thresholds from synthetic fixtures.

### C2 — synthetic routing benchmark V1; completed model-free

Use tiny disposable arrays. Required positive and negative controls:

- recover two known clusters inside one category;
- remain invariant to cluster-label renaming and component order;
- normalize for unequal component counts;
- route low-response and tied episodes to `shared_ambiguous`;
- never duplicate every category episode across cluster routes;
- reject changed V4 membership, component-artifact tampering, episode-order
  tampering, row-count mismatch, nonfinite arrays, and split-role reuse;
- reject a router that appears stable in aggregate but moves episodes between
  discovery and validation beyond the frozen stability threshold;
- leave all inputs byte-identical.

The benchmark does not import Torch, Transformers, or PEFT. Its artifact is
`artifacts/vpd_cluster_episode_routing_benchmark_v1/report.json`, with benchmark
ID `0d5668cdbec6629f1632b3cf0f26a69695c95ec137fcaa2cae0342cde1056ac1`.

### C3 — cluster-aware SAE plan V1; completed model-free

Implement a successor plan; do not mutate or relabel the hash-bound V4 plan.
Create a job only for an admitted `(activation_category, cluster_id)` route
whose support and capacity gates pass. Every job must bind:

- component-membership hash;
- routing-receipt ID;
- exact routed episode and pair sets;
- pair-atomic train/validation hashes;
- dictionary capacity and two initialization seeds;
- no-overlap proof against other primary routes.

Keep one explicit shared/fallback dictionary. If a route lacks support, merge
it into shared/fallback and record the reason; never silently duplicate or
oversample it. A manifest-only dry run may write plans and tiny fixtures, but
must load no model and write no learned weights.

The predecessor planner V1 and its benchmark implement this contract only for
fixtures whose pair identities are disjoint across activation categories. Across two
synthetic activation categories, three supported cluster compartments plus one
shared fallback produce eight jobs (two seeds each). All 22 training episodes
and 11 pairs are assigned exactly once. A weak route with 2 episodes/1 pair is
merged to shared instead of oversampled. Benchmark ID:
`d79ff71b50d0bfc0bddb649609234b6bf9fbf9be78ef985daa773ebd5687ca10`.

The planner correctly requires a third, disjoint `sae_training` routing receipt
after discovery/cluster-validation stability. Canonical capture planner V2 now
supports that role while preserving V1 unchanged. It validates the recovered
6,292-episode/572-pair split and requires authorization scoped to the exact
role, dataset hash, output, and forward-pass ceiling. That authorization must
explicitly deny SAE dictionary training. The V2 execution worker and V2
role-extension assembler now pass 16/16 fake-runtime controls with exactly 32
callbacks, exact factor restoration, and tamper rejection. The receipt-to-router
adapter and three-role synthetic capture-to-routing-to-plan benchmark now also
exist. That benchmark exposed that V1 collapses real-shape shared pair IDs to
the fallback. Planner V2 corrects this with one global pair split: the same pair
may appear in multiple activation-category compartments, but every occurrence
is assigned to the same training or validation side. It preserves 16 specific
compartments plus one shared fallback across all eight categories. No empirical
plan or authorization exists.

### C4 — empirical routing; blocked

This requires an explicitly authorized and completed real canonical capture,
an authenticated capture execution receipt, and a passed empirical V4
partition-identity admission. None exists locally. Stop rather than fabricate
paths, hashes, or receipts.

### C5 — cluster-specific SAE training; blocked

This requires C4, exact immutable job hashes, approved device/time/cost limits,
and explicit compute and weight-mutation authorization. A router or plan
receipt is not compute authority and is not authorization-control release.

### C6 — V9 authorization retraining; separate blocked line

The observable-host-state retraining design remains in
`AUTHZ_RETRAINING_EXPERIMENT_V9.md` and
`configs/training/bonsai8b_authz_retraining_v9_design.yaml`. The cheaper model
may implement local generators, validators, samplers, loss tests, and tiny
fixtures as described in the main report, but paid candidates and the fresh
2,304-row one-shot held-out pack remain behind their existing freeze and
authorization gates.

## Required executor receipt

Return one JSON receipt containing the run ID, exact argv arrays, exit codes,
observed file hashes/content IDs, changed files, tests, first stop reason, and
these booleans: `model_loaded`, `inference_performed`, `weights_written`,
`external_compute_used`, and `execution_authorized`. All five must be false for
C0-C3.

## Copyable cheaper-model prompt

```text
Work only in C:\projects\VPD. Read
VPD_CLUSTER_TO_SAE_CHEAP_EXECUTOR_HANDOFF_2026-09-01.md and
VPD_CLUSTER_TO_SAE_CHEAP_EXECUTION_PACKET_2026-09-01.json in full. Execute C0
first and stop on any mismatch. If C0 passes, implement C1 and C2 using only
NumPy and tiny synthetic fixtures; then implement C3 as a manifest-only
successor planner. Do not modify hash-bound historical or predecessor
artifacts. Do not load Bonsai or any model, install packages, run inference,
launch external compute, mutate weights, create an empirical receipt, or
generate/open the final held-out pack. Never call duplicated category data a
cluster-specific SAE sample. Return the required receipt and the first exact
stop reason.
```

## Provenance boundary

This file is a new successor design report, not recovered historical evidence.
It does not change the V7 rejection result, claim an empirical Bonsai capture,
or authorize SAE training. Protocol V22 and runbook V14 are the current
research-lineage and machine-authority artifacts; V21/V13 remain pinned
predecessors.
