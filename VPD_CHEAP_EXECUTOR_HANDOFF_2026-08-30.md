# VPD cheaper-executor handoff report and proposed runs

Date: 2026-08-30  
Project root: `C:\projects\VPD`  
Default permission: local model-free preflight only  
Paid/model execution authorized: **no**

## Executive result

The recovered Bonsai-8B authorization-control evidence and the reconstructed
V2 SAE runtime are locally coherent. The complete test suite passes: 217 tests
and 3 subtests. The runtime contract benchmark passes 11 of 11 scenarios.

The authoritative scientific result remains negative:

- base separation: `0.692347`
- adapter separation: `0.580743`
- adapter minus base: `-0.111604`
- 95% CI: `[-0.122397, -0.100996]`
- invalid REQUEST rate: `0%`
- valid PROPOSE rate: `100%`
- release decision: **reject both base and adapter**

No activation capture, SAE dictionary, checkpoint, empirical category bundle,
scientific-release receipt, external compute-authorization receipt, or quality
report currently exists. No model inference or weight mutation is authorized.

## Authoritative handoff assets

| Asset | Identity |
|---|---|
| Machine runbook | `artifacts/vpd_cheap_executor_runbook_v1/runbook.json` |
| Runbook ID | `02229f6d02d1310fb81a556d4d0807a9012b7ae1e403ba9af9989228411105a4` |
| Runbook file SHA-256 | `37b688b71d3ce661e072dd8cf4757d5ce72bbf7a549b33d27f4e637dbe32f088` |
| Training plan V4 | ID `a7a050abead2d60a63e16a6d07126609ac38c034f18ad3e37f98cd7c9f4d8c2b` |
| Release Gate-plan V2 | ID `5d821b591fa46ee306d335efc9df10e5a0db3a301482d13d54caeaa0d42e575c` |
| Runtime benchmark V2 | ID `f044212429c08737719ffe1f64496122d20d94f2f6b4e2333c42d00d452b185c` |
| Research protocol V9 | SHA-256 `3d4397f3ef270f4d2b0bc00995034543f34a60dedfc9c9587eb06963e2b5bd9c` |
| Unified lineage V7 | ID `e97ea34df8a8e4c12f4b13e5bdf02fe9ed0fe985a1dfdad8237734420a7dc90c` |

The JSON runbook is the command authority for a cheaper executor. It contains
argument arrays rather than shell-built strings, all 22 exact job IDs, required
placeholders, expected outputs, acceptance conditions, and stop conditions.

## Non-negotiable executor rules

1. Run Stage 0 only by default.
2. Never invent a missing path, hash, model revision, receipt, or authorization.
3. Never generate `explicit_compute_authorization.json`; the repository only
   verifies authorization supplied externally by the user.
4. A scientific-release receipt is not compute authorization and must retain
   `compute_authorized=false`.
5. Every paid job requires external authorization covering that exact V2 job
   ID and action.
6. Do not mutate base or adapter weights, sign results, execute real actions, or
   use `final_evaluation` for tuning.
7. Stop on the first nonzero exit, source-hash change, receipt failure, sample
   mismatch, or unresolved placeholder.
8. Preserve recovered files as recovered evidence and label new runtime outputs
   as reconstructed/new evidence.

## Proposed run sequence

| Stage | Proposed run | Cost boundary | Proceed condition | Expected result |
|---:|---|---|---|---|
| 0 | Reverify runbook, benchmark, and fast tests | Local CPU, model-free | Can run now | Runbook pass; benchmark 11/11; 8 tests pass, 1 deselected |
| 1 | Emit scientific-release prerequisite | Local verification, but needs empirical inputs | All six frozen audit inputs exist and validate | V2 release receipt with compute still false |
| 2 | Capture `sae_training` activations | Paid/model compute | V2 release receipt plus exact-job external authorization | 805,376 rows; about 6.144 GiB numeric arrays; zero generated tokens |
| 3 | Train one canary dictionary: `other_invalid`, seed 17 | Paid SAE training | Stage 2 capture validates and canary job is explicitly authorized | One independently valid manifest V2 and dictionary |
| 4 | Train remaining 17 dictionaries | Paid SAE training | Canary passes; resource/concurrency budget explicitly supplied | All 18 planned manifest V2 dictionaries validate |
| 5 | Capture calibration activations | Paid/model compute | All 18 training manifests validate | 604,032 rows; about 4.608 GiB numeric arrays; zero generated tokens |
| 6 | Admit dictionary bundle | Local artifact verification | Exact 18-job coverage and both authorization joins validate | `pass_ready_for_calibration_quality_v2` |
| 7 | Run calibration quality | Compute over captures/dictionaries | Bundle and calibration capture validate | Quality report status `pass` or hard rejection |

The order deliberately postpones the 4.608 GiB calibration capture until all
training manifests pass. The canary does not change the official plan; it is
the first of the 18 required jobs and provides a cheap fail-fast boundary before
launching the other 17.

GPU duration and monetary cost are not estimated in the recovered evidence.
The executor must not invent them. Before Stage 2, require a device-specific
budget containing maximum wall time, concurrency, storage, and spend.

## Stage 0: commands safe to run now

Run from `C:\projects\VPD`:

```powershell
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v1 `
  --verify artifacts/vpd_cheap_executor_runbook_v1/runbook.json

.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_compartment_sae_runtime_v2 `
  --verify artifacts/vpd_compartment_sae_runtime_benchmark_v2/report.json

.\.venv\Scripts\python.exe -m pytest `
  tests/test_vpd_compartment_sae_runtime_v2.py `
  tests/test_vpd_compartment_sae_runtime_lineage_v2.py `
  tests/test_vpd_cheap_executor_runbook_v1.py `
  -q -k "not plan_v4_and_release_gate_v2_reproduce"
```

Expected output after the new runbook tests are included:

- runbook verification: `pass`
- runtime benchmark: 11 scenarios, `pass`
- fast tests: 11 passed, 1 deselected

If source code changes, also run the slow plan/gate reproduction and then the
full suite. The last authoritative full-suite result is `217 passed, 3 subtests
passed in 2109.59s`; do not spend 35 minutes repeating it when inputs are
unchanged.

## Stage 1: scientific-release prerequisite

Do not run until all of these are resolved to frozen, existing artifacts:

- `FROZEN_CATEGORY_BUNDLE_DIR`
- `COMPONENT_INVENTORY`
- `DISCOVERY_EFFECTS`
- `DISCOVERY_MERGE_MANIFEST`
- `PARTITION_AUDIT_MERGE_DIR`
- `PARTITION_AUDIT_REPORT`

The exact argument array is stored at
`runbook.stages[1].command.arguments`. Its output must be
`training_release/training_release_receipt_v2.json` with schema
`sch.vpd-compartment-sae-training-release-receipt.v2`, status
`pass_scientific_release_prerequisite_compute_not_authorized_v2`, and
`compute_authorized=false`.

If any input is missing or the fresh held-out audit fails, report the missing or
failed join and stop. Do not proceed to model capture.

## Stages 2–7: authorized paid pipeline

These stages additionally require:

- `SAE_TRAINING_RELEASE_RECEIPT`
- `SAE_COMPUTE_AUTHORIZATION_RECEIPT`
- `FROZEN_BASE_MODEL`
- `FROZEN_BASE_REVISION`
- `SAE_TRAINING_DEVICE`

The cheaper executor should load the runbook JSON and execute each job's
`arguments` array using `.venv/Scripts/python.exe`. It must not reconstruct
commands from prose. Recommended order:

1. `stages[2].job` — training capture.
2. `stages[3].job` — canary dictionary.
3. `stages[4].jobs` — remaining dictionaries, in the stored fail-fast order.
4. `stages[5].job` — calibration capture.
5. `stages[6].job` — exact bundle admission.
6. `stages[7].job` — calibration quality.

Do not infer parallelism. The runbook explicitly requires a supplied resource
budget. Serial execution is the safe default.

## Required executor receipt

After each attempted stage, the cheaper executor should write a concise JSON
receipt containing:

```json
{
  "runbook_id": "02229f6d02d1310fb81a556d4d0807a9012b7ae1e403ba9af9989228411105a4",
  "stage_id": "local_preflight",
  "status": "pass_or_stopped",
  "commands": [
    {
      "job_id": "if_applicable",
      "exit_code": 0,
      "output_paths": [],
      "output_sha256": {}
    }
  ],
  "stop_reason": null,
  "model_inference_performed": false,
  "weights_mutated": false,
  "real_actions_executed": false
}
```

For a stopped stage, record the exact missing placeholder, failed invariant, or
nonzero command. A stop is a correct outcome; bypassing the gate is not.

## Copyable prompt for the cheaper executor

```text
Work in C:\projects\VPD. Read VPD_CHEAP_EXECUTOR_HANDOFF_2026-08-30.md and
artifacts/vpd_cheap_executor_runbook_v1/runbook.json in full. Verify the runbook
content address, then execute Stage 0 only. Treat the JSON argument arrays as
authoritative. Stop on any nonzero exit or hash mismatch. Do not proceed to
Stage 1 unless all six scientific-release inputs are explicitly supplied and
frozen. Do not proceed to Stages 2–7 without a passing V2 scientific-release
receipt and external authorization covering each exact job ID and action.
Never fabricate receipts or placeholders, never generate authorization, never
mutate base/adapter weights, and never execute real actions. Return a receipt
with commands, exit codes, output hashes, and the exact stop reason.
```

## Scientific interpretation after the proposed runs

A Stage 7 quality pass would establish only that the compartment dictionaries
meet the frozen calibration criteria. It would not reverse the historical V7
rejection and would not authorize release. Promotion would still require fresh,
held-out, signed activation and parameter interventions with matched controls
and unchanged guardrails.
