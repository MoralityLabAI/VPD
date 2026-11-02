# VPD Context Recovery — 2026-08-29

## Purpose

This file is the durable restart point for the VPD/Bonsai authorization-control work after the Codex rollout history and the original `C:\projects\VPD` registration became unusable. It separates recovered evidence from reconstructed context and prevents a new task from repeating expensive inference or overstating the result.

## Recovery provenance

- Rebuilt project location: `C:\projects\VPD`
- Copied intact from: `E:\Recovered_C_projects\Spiral_target\projects\SmallControlHarness\bitagent_small_control_authz_bonsai8b`
- Copied payload: 375 files, 388,216,256 bytes
- The recovered source was not a Git working tree; no `.git` history was present.
- Original Codex task ID: `019fe3e6-8292-7141-909f-b391dc5b6f6c`
- Original working directory: `C:\projects\SmallControlHarness`
- Surviving continuation brief: `E:\Recovered_C_projects\Sensitive_state\current_2026-08-23\Codex_history_recovery\continuation_briefs\019fe3e6-8292-7141-909f-b391dc5b6f6c.md`
- The continuation brief contained no recovered log rows. The files, reports, manifests, hashes, tests, and receipts in this tree are the primary evidence.

Recovered user prompts, in order:

1. `don't we get a VPD artifact of some weight ouf ot that`
2. Supplied `D:\Download\bitagent_small_control_authz.zip` and said a Pro task had designed an experiment leveraging the foundation.
3. `proceed`
4. `do`
5. `fix eval flaw`
6. `perform`
7. `cont`
8. `so why did this fail so badly`

## What the artifacts establish

The recovered tree contains the authorization-control harness, VPD/SAE experiment code, tests, fixtures, configs, schemas, and Bonsai experiment artifacts through V8. The authoritative corrected result is V7:

- `artifacts/bonsai8b_authz_v7/FIX_REPORT.md`
- `artifacts/bonsai8b_authz_v7/RESULT_REPORT.md`
- `artifacts/bonsai8b_authz_v7/FINAL_VERIFICATION.json`

V6 had a genuine evaluation flaw. Revoked and replayed cases could present model-visible prompts identical to valid cases while assigning conflicting labels, because the decisive gate state was hidden from the model. Expiry also lacked trusted observation time. V7 corrected this by adding a trusted `AUTH_STATUS` block containing grant ID, observation time, signature validity, active-time status, revoked status, and nonce state. It added hard invariants against prompt/label conflicts and status mismatches.

V7 verification records:

- 960 episodes and 96 independent pairs
- four action families and ten conditions
- 576 rows with visible status
- zero prompt-label conflicts
- zero executed rows
- zero overlap against the prior 2,640 rows
- 38 relevant tests passed
- artifact manifest: 21 entries, zero hash mismatches
- final verification status: pass
- pod terminated; zero active pods

## Corrected result and release boundary

The corrected V7 evaluation is a valid negative result:

- Base separation: 0.692347
- Adapter separation: 0.580743
- Adapter-minus-base effect: -0.111604
- 95% CI: [-0.122397, -0.100996]
- Invalid-case `REQUEST` rate: 0%
- Valid-case `PROPOSE` rate: 100%
- Candidate argmax coverage: 0.846875
- All four action-family effects were negative.

Decision: reject the adapter for release as an authorization-control model; the base model also fails the authorization-control requirement. The corrected experiment shows that the adapter systematically continued to choose `PROPOSE` on invalid observable host state and degraded separation relative to base.

This failure is not explained away by the V6 evaluation defect. V7 fixed the defect and confirmed the behavior. Evaluation-only VPD analysis cannot create a missing authorization decision policy.

## Weight-bearing artifacts

V7 did not train a new model. Its weight-bearing model artifact is the pre-existing Bonsai-8B BitAgent LoRA adapter historically stored at:

`C:\projects\SmallControlHarness\Control-Harness-public\reports\evals\bonsai8b_bitagent_vpd_v1\remote\adapter\adapter_model.safetensors`

Recorded SHA-256:

`a95433ba5ceb9b6375ac649549a098ea4e4b2ad5695e095d49be61d0c3ee38e6`

V7 produced a 1,920-row base/adapter logit pack plus deterministic analysis and verification receipts, not newly learned weights. The V5 work is the full decomposition artifact: 4,032 LoRA rank-one components, a 252-projection inventory, and a selected 144-component cell.

## Safe continuation plan

1. Validate this copied tree: file inventory, recorded hashes, manifests, and cheap local tests.
2. Produce an exact map of V1–V8 artifacts and distinguish executable source, recovered results, external/missing checkpoints, and derived reports.
3. Treat V7 as the definitive current authorization result. Do not rerun paid inference merely to seek a different outcome.
4. Design a retraining experiment in which invalid but observable host state explicitly trains `REQUEST`, with versioned candidates and a fresh held-out evaluation pack.
5. Do not release either base or adapter as an authorization-control model.
6. Do not start external GPU jobs, spend paid inference, mutate model weights, or execute real actions without explicit user authorization.

## Reconstruction audit performed on 2026-08-29

The primary recovered Bonsai/VPD tree was missing three core package modules that were present in the sibling recovered foundation tree `bitagent_small_control_authz`. They were restored into `small_control_harness/` without modification:

- `crypto.py` — SHA-256 `72745591bed77dcc8e8ffcc10f081d5cd3e0f6c0e4235dd2fd3f7b17fce4199`
- `environment.py` — SHA-256 `83a2279b6bab4ee5c1e992858899b305257205edb2ced213f9492bbc96c71251`
- `gate.py` — SHA-256 `9994522ced116440f5ac3ae4123d9e442e2c6a3a4bf9b4e5b9a100f07a0a7f68`

An isolated `.venv` was created and the project, `pytest`, and `cryptography` were installed. Test collection then narrowed from nine errors to one: `experiments/authz_mechinterp/validate_independent_authority_pack_v2.py` is absent. The V7 manifest proves that file existed and records its expected SHA-256 as `1cafe45ca4487a969226f09f9c9c8208d84a04cf5790ad1b8517e0ff6ad2f444`, but no recovered sibling copy was found. Do not silently invent or relabel a replacement as recovered. The next task should first seek an exact copy in remote history or other recovery media; only if unavailable should it reconstruct a new version from tests and clearly mark it as reconstructed.

## Safety facts

The recovered reports state that the evaluation had no action-execution path, executed no actual actions, and contained no secrets. Preserve that boundary in subsequent work.
