# VPD cheaper-executor handoff — cluster discreteness V2

Date: 2026-09-01  
Project root: `C:\projects\VPD`  
Current machine authority:
`artifacts/vpd_cheap_executor_runbook_v3/runbook.json`  
Default permission: Stage 0 local preflight only  
Model or paid execution authorized: **no**

> Superseded operationally by
> `VPD_CHEAP_EXECUTOR_REPORT_AND_PROPOSED_RUNS_2026-09-01.md`. This file is
> preserved as predecessor evidence of the earlier implementation-missing
> boundary; runbook V3 records that independent admission is now implemented
> and benchmarked, while empirical inputs remain absent.

## What changed after the 2026-08-30 handoff

The legacy within-category cluster-count selector used cross-archive adjusted
Rand stability, mean silhouette, and minimum cluster size. A new model-free
negative-control benchmark proves that this is insufficient: a stable continuous
activation geometry can be sliced into apparently reproducible discrete
clusters.

Observed synthetic failure:

- continuous strength gradient: legacy selector chose two clusters in all 8
  activation categories;
- continuous direction arc: legacy selector chose two clusters in 1 of 8
  activation categories;
- truly separated two-mechanism geometry: legacy and V2 both chose two clusters
  in all 8 categories;
- compact single-mechanism geometry: legacy and V2 both chose one cluster in all
  8 categories.

V2 adds two conservative requirements to every nontrivial candidate in every
projection/resampling/archive replicate:

1. lower 10th-percentile silhouette at least `0.05`;
2. minimum cross-cluster boundary gap at least `1.50` times the larger cluster's
   median within-cluster nearest-neighbour distance.

If no candidate passes, the category remains one cluster. V2 can therefore
remove unsupported subdivisions but cannot invent a new empirical mechanism.

## Current identities

| Artifact | Identity |
|---|---|
| Cluster discreteness benchmark V2 | ID `3c72d47b3662ccc78bdf2470850f0a644e597226cde988b78335b07a72609d91` |
| Benchmark file | SHA-256 `b0d6e945f0af0c8bd146cc760d54341d3bcdac68b86569a9d1a1978ba4754e52` |
| Research protocol V10 | SHA-256 `1b53f290aa05a2aefefea665202ba6c22e045c019a05b5796028e895fd062f1c` |
| Cheap-executor runbook V2 | ID `7c4a52e50e0105d26be1badeb002cd5fca5985d8a416f9017e08f61a44ee8c41` |
| Runbook V2 file | SHA-256 `3dd1e0479d4bf75723a562a694e45e0e6c1c5336ed371d74b1ef003ac5e8d5b4` |
| Unified lineage V8 | ID `6315c32b329278aba56f8a29d84ff9c5e9ebaccfe32a168bb36d6d70987a26d9` |
| Unified V8 file | SHA-256 `9b420d191f5355f0ed044809b78918ebba234153c1901b33f156edbe6396a04f` |

The 2026-08-30 report remains predecessor evidence for the 22-job capture,
training, admission, and quality plan. Runbook V2 supersedes runbook V1 as the
current machine authority.

## Current nine-stage schedule

| Stage | Action | Can run now? |
|---:|---|---|
| 0 | Verify runbook, runtime benchmark, fast tests, and cluster-discreteness benchmark | Yes, local/model-free |
| 1 | Build and independently admit a cluster candidate using selection schema V2 | **No — production wrapper and admission are not yet implemented** |
| 2 | Emit compute-disabled scientific-release receipt from frozen audit evidence | No — Stage 1 and empirical audit inputs required |
| 3 | Capture SAE-training activations | No — release plus exact-job external authorization required |
| 4 | Train one fail-fast canary dictionary | No — same authorization required |
| 5 | Train remaining 17 dictionaries | No — canary and resource budget required |
| 6 | Capture calibration activations | No — all training manifests must validate first |
| 7 | Admit exact 18-dictionary bundle | No |
| 8 | Run calibration quality | No |

Stage 1 is deliberately a hard stop. The selector and synthetic benchmark exist,
but the production candidate wrapper and independent V2 admission do not yet
exist. A cheaper executor must not apply the selector by monkeypatching the V1
worker or relabel a V1 candidate as V2.

## Stage 0 commands

Run from `C:\projects\VPD`:

```powershell
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v2 `
  --verify artifacts/vpd_cheap_executor_runbook_v2/runbook.json

.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_activation_category_cluster_discreteness_v2 `
  --verify artifacts/vpd_activation_category_cluster_discreteness_benchmark_v2/report.json

.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_compartment_sae_runtime_v2 `
  --verify artifacts/vpd_compartment_sae_runtime_benchmark_v2/report.json

.\.venv\Scripts\python.exe -m pytest `
  tests/test_vpd_activation_category_cluster_discreteness_v2.py `
  tests/test_vpd_cluster_discreteness_lineage_v2.py `
  tests/test_vpd_compartment_sae_runtime_v2.py `
  tests/test_vpd_compartment_sae_runtime_lineage_v2.py `
  tests/test_vpd_cheap_executor_runbook_v1.py `
  -q -k "not plan_v4_and_release_gate_v2_reproduce"
```

Stop on any nonzero exit or source-hash difference. Do not proceed to Stage 1.

## Copyable prompt for a cheaper executor

```text
Work in C:\projects\VPD. Read VPD_CHEAP_EXECUTOR_HANDOFF_2026-09-01.md,
artifacts/vpd_cheap_executor_runbook_v2/runbook.json, and its predecessor report
VPD_CHEAP_EXECUTOR_HANDOFF_2026-08-30.md in full. Verify runbook V2 and execute
Stage 0 only. Treat JSON argument arrays and stop conditions as authoritative.
Do not proceed to Stage 1: the production cluster-candidate V2 wrapper and
independent admission are explicitly absent. Never monkeypatch the V1 worker,
never relabel a V1 candidate as V2, never fabricate receipts or placeholders,
never generate external compute authorization, never mutate base/adapter
weights, and never execute real actions. Return commands, exit codes, output
hashes, and the exact stop reason.
```

## Scientific boundary

This improves the clustering protocol, not the Bonsai empirical result. No
Bonsai activation-category cluster count has been selected. Historical V7 still
rejects both base and adapter for authorization-control release. No capture,
dictionary, checkpoint, scientific-release receipt, or compute authorization
was created.
