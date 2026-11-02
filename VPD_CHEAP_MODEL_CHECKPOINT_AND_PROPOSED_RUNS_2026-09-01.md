# VPD cheap-model checkpoint and proposed runs

Date: 2026-09-01  
Project root: `C:\projects\VPD`  
Default authority: local, model-free verification and implementation only  
Paid inference, external GPU jobs, real model/adapter loading, weight mutation, SAE training, and real actions: **not authorized**

## Decision summary

The historical release decision is unchanged: reject both the Bonsai-8B base
and adapter for authorization-control release. V7 measured base separation
`0.692347`, adapter separation `0.580743`, adapter-minus-base `-0.111604`, and
95% CI `[-0.122397, -0.100996]`; invalid `REQUEST` was `0%` and valid
`PROPOSE` was `100%`. V7 trained no weights.

Recovery and the successor model-free clustering work are operational. The
exact V1-V8 recovery map is `VPD_STATE_MAP_V1_V8_2026-08-29.md`. Recovered
files, reconstructed files, and new successor work must retain their existing
provenance labels.

## Intermediate results worth preserving

1. The model-free cluster-to-episode router passes 16/16 controls. It uses
   exact component membership, pair-atomic routing, an explicit
   `shared_ambiguous` fallback, and independent overall plus
   condition-stratified stability.
2. The corrected cluster-aware SAE planner passes 17/17 controls. Its synthetic
   result contains three supported cluster-specific compartments and one
   shared compartment, producing eight manifest-only jobs across 22 episodes
   and 11 pairs. One weak two-episode/one-pair route is conservatively merged
   into shared.
3. A row-unit defect was caught and repaired before empirical execution:
   routing component-effect rows and SAE activation rows are now independent
   units. The fixture has 44 routing-effect rows (`22 * 2`) and 2,816 SAE
   activation rows (`22 * 128`).
4. The V2 `sae_training` capture planner passes 12/12 controls and validates
   the recovered real split: 6,292 episodes, 572 pairs, all eight activation
   categories. The plan explicitly denies SAE dictionary training authority.
5. A distinct V2 capture worker now exists and is synthetically verified. Its preflight binds
   split role and dataset hash; its execution path has an exact forward-pass
   ceiling, atomic staging, effect-shard assembly, factor restoration, and a
   separately verifiable receipt. Its fake-runtime benchmark passes 16/16
   controls with exactly 32 callbacks and exact restoration on success and
   failure. It remains blocked from empirical use because real identities,
   receipts, and user authorization are absent.
6. The worker benchmark exposed that the V1 effect-shard assembler did not
   allow `sae_training`. A new V2 role-extension wrapper now admits that role,
   preserves the V1 source unchanged, reuses the role-independent V1 array
   kernel, and persists the exact original role in the addressed receipt.
7. Protocol V23 and Runbook V15 reproduce and mark the V2 worker ready while
   keeping `safe_stage_now: 0`. No model was loaded, no inference was performed,
   and no weights were written.
8. The three-role integration benchmark passes 18/18 controls and exposes a
   real-shape planner defect: because the recovered split reuses pair IDs across
   activation categories, planner V1 collapses every specific compartment to
   shared. Planner V2 preserves 16 category-cluster compartments plus one
   shared fallback by assigning each pair globally to exactly one of training
   or validation. It emits 34 manifest-only jobs and assigns all 80 synthetic
   training episodes exactly once.
9. The authoritative full local suite passes: 334 tests, plus 3 subtests, with
   the one known recursive reproduction test intentionally deselected as
   required by the runbook.

## Current pinned authority

| Item | Current value |
|---|---|
| Cluster-routing benchmark ID | `0d5668cdbec6629f1632b3cf0f26a69695c95ec137fcaa2cae0342cde1056ac1` |
| Corrected cluster-aware planner benchmark ID | `d79ff71b50d0bfc0bddb649609234b6bf9fbf9be78ef985daa773ebd5687ca10` |
| Corrected synthetic plan ID | `43ecfc3bfe7356d06cdcbb03f64eb50b9aeb0d083d2207b891357067cc526973` |
| V2 SAE-training capture-plan benchmark ID | `525fbf3433a410ad883b438ee301285f31d704815a86f955751a9cb29b0f09f2` |
| Protocol V21 SHA-256 | `b26f5668a877923c4f45a877edd8836e65ec061d4132b9ca309abb3d9e74556a` |
| Protocol V22 SHA-256 | `4a59df7c095e407be02ebfa5f01504eee6f98bc0872b4c9379c3f40ea98d016c` |
| Protocol V23 SHA-256 | `8eda03fd5c982af058da674feff55c223b58598a8d38422fd8bb27afc34e5c49` |
| Protocol V24 SHA-256 | `c42dc6e43296c7e4a3d6054174e015620221cf141377bff44773d8cfc0add2b2` |
| Runbook V13 ID | `79f1c87f1cfa84fb05d575be0265cdf7e41405e1e791306e1b36c9b41b0ea757` |
| Runbook V14 ID | `bea30e4ef92a8237f44db800db816db436e2c29f9ec7233b37eff9ac8798690e` |
| Runbook V15 ID | `60e3a3e8778a2efe435eeaa904d58c11621add35ed105345974fad26e5aa2699` |
| Runbook V16 ID | `321b0cab4fc95d0270395f39ba95d0139a3ea258e3cb9c05822b6c82f6ce5dd3` |
| V2 runtime benchmark ID | `2023baf1646230680ed7d23d84fb1c6174f63d99abcc8cf629580f8f77086bf3` |
| Three-role integration benchmark ID | `b27a7f901d191a81a3bc4e708244edf1a3325e968b979471d064986d2c773df6` |
| Recovered-shape capacity benchmark ID | `e8021583211fde3fc1185dfb2423a7c56dd33089db66d3143ca5823692239390` |
| V2 worker source SHA-256 | `0db5cad97effe8c06a8caf329a6b973537c4fa11b75b5433ae7d0b889d29dd87` |
| V2 assembler source SHA-256 | `f82d08497a7c0d97371bdfd8e629432d326808afcecd05ef361330adb1f38cd3` |
| Capture-receipt adapter source SHA-256 | `071a261d2250b28a236f760f8d6a16fa0e4858cc72495725d1ee02f1c43ec9f8` |
| Shared-pair-safe planner V2 SHA-256 | `b7fce50af30687d0fe4d7b75f589e1acc61863bddc73c7517caf91b1ad8d17a3` |
| Corrected planner source SHA-256 | `f6203ac7fe66f6181a5c6ecd24081c1f28d64463b8f3ef8fdf84068890b23eef` |

## Proposed run ladder

### R0 — execute now: cheap local reproduction

Run exactly these commands from `C:\projects\VPD`. Stop at the first nonzero
exit or identifier mismatch.

```powershell
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_episode_routing_v1 --verify artifacts/vpd_cluster_episode_routing_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_cluster_aware_sae_training_plan_v1 --verify artifacts/vpd_cluster_aware_sae_training_plan_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_canonical_vpd_sae_training_capture_plan_v2 --verify artifacts/vpd_canonical_sae_training_capture_plan_benchmark_v2/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v21 --verify configs/evals/vpd_activation_category_compartments_v21.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v22 --verify configs/evals/vpd_activation_category_compartments_v22.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v13 --verify artifacts/vpd_cheap_executor_runbook_v13/runbook.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v14 --verify artifacts/vpd_cheap_executor_runbook_v14/runbook.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_canonical_vpd_sae_training_capture_runtime_v2 --verify artifacts/vpd_canonical_sae_training_capture_runtime_benchmark_v2/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v23 --verify configs/evals/vpd_activation_category_compartments_v23.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v15 --verify artifacts/vpd_cheap_executor_runbook_v15/runbook.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_three_role_capture_to_sae_integration_v1 --verify artifacts/vpd_three_role_capture_to_sae_integration_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v24 --verify configs/evals/vpd_activation_category_compartments_v24.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v16 --verify artifacts/vpd_cheap_executor_runbook_v16/runbook.json
.\.venv\Scripts\python.exe -m py_compile experiments/authz_mechinterp/run_canonical_vpd_capture_job_v2.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_cluster_episode_routing_v1.py tests/test_vpd_cluster_aware_sae_training_plan_v1.py tests/test_vpd_cluster_routing_planner_lineage_v13.py tests/test_vpd_canonical_sae_training_capture_plan_v2.py tests/test_vpd_sae_training_capture_plan_lineage_v14.py tests/test_vpd_canonical_sae_training_capture_runtime_v2.py tests/test_vpd_sae_training_capture_runtime_lineage_v15.py
```

Return a receipt listing command, exit code, observed content ID/hash, and first
stop reason. A passing R0 does not authorize R1 or any empirical run.

### R1 — completed locally: recovered-shape global-pair capacity stress test

The three-role chain passes on a production-shaped synthetic fixture. The
model-free stress test now uses the exact recovered
6,292-episode/572-pair `sae_training` incidence graph. It must not fabricate
empirical activation effects or claim an empirical routing result. Generate
versioned simulated route allocations solely to test planner capacity and
global split feasibility. The benchmark must prove:

- exact recovered episode, pair, category, condition, and template identities
  remain unchanged and read-only;
- every simulated allocation assigns each episode exactly once;
- shared pair IDs may span category compartments only through one global
  train/validation assignment;
- every retained compartment has nonzero training and validation support;
- minimum-support failures merge to shared instead of oversampling;
- routing-effect and SAE-activation row units stay independent;
- skewed, sparse, ambiguous-heavy, and category-imbalanced simulations either
  produce valid manifests or reject with an exact reason;
- no simulated receipt is relabelled as empirical evidence.

The versioned artifact passes 15/15 controls with ID
`e8021583211fde3fc1185dfb2423a7c56dd33089db66d3143ca5823692239390`.
It performed no model inference and grants no execution or SAE-training authority.

### R2 — completed locally: worker and integration lineage freeze

Protocol V23 and Runbook V15 exist as new files; V22/V14 remain preserved. They
bind the V2 worker, V2 assembler, and 16/16 runtime benchmark while keeping
`safe_stage_now: 0`, `execution_authorized: false`, empirical capture absent,
and SAE training unauthorized. Protocol V24 and Runbook V16 additionally bind
the 18/18 three-role integration result, receipt adapter, and shared-pair-safe
planner V2. All predecessors remain preserved.

### R3 — blocked: empirical preflight and canonical effect capture

Do not run without explicit user authorization and exact immutable identities
for the real model, adapter weights/config, dataset, split, plan, output path,
resource ceiling, and authorization receipt. Preflight first; return the
computed pass ceiling and stop. Execution, if later authorized, captures
effects only and does not authorize SAE training.

### R4 — blocked: empirical cluster routing and manifest planning

After authenticated empirical discovery, cluster-validation, and SAE-training
effect receipts exist, run partition admission V4, then the router, then the
cluster-aware manifest planner. Reject on any identity, overlap, stability,
condition-stratum, membership, or support mismatch. Do not duplicate an entire
category archive into every cluster.

### R5 — blocked: SAE candidates and authorization retraining

SAE dictionary training and the V9 authorization candidates mutate weights and
may use paid compute. They require a separate explicit authorization with
device, cost/time cap, code hashes, data hashes, seeds, output paths, and stop
conditions. The candidate order remains frozen LoRA control, fresh LoRA SFT,
fresh LoRA margin (primary), and historical-adapter repair diagnostic. Do not
repeat V7 merely to reproduce its negative result.

## Stop rules for the cheaper model

Stop immediately if a required file/hash differs, a verifier fails, a command
would install packages or access the network, a model/adapter would be loaded,
inference or GPU work would begin, a weight/dictionary would be written, an
authorization receipt is absent, or empirical inputs would need to be
fabricated. Report the exact mismatch; do not repair provenance by relabelling
reconstructed or successor files as recovered originals.

## Copyable executor prompt

```text
Work only in C:\projects\VPD. Read
VPD_CHEAP_MODEL_CHECKPOINT_AND_PROPOSED_RUNS_2026-09-01.md and
VPD_CHEAP_MODEL_CHECKPOINT_PACKET_2026-09-01.json in full. Execute R0 first and
stop on the first mismatch. If R0 passes, you may implement and test the new
model-free recovered-shape R1 capacity stress benchmark only.
Do not load a model or adapter, run inference, install packages, use the
network, launch GPU/external jobs, mutate weights, train an SAE, generate final
held-out data, fabricate receipts, or execute real actions. Preserve all
recovery-versus-reconstruction provenance. Return exact commands, exit codes,
IDs/hashes, files changed, and the first stop reason.
```
