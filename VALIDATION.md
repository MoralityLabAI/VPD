# Validation Report

Validation date: 2026-08-08 America/New_York

## Deterministic core

- Python compilation completed for the harness, experiments, and tests.
- 15 Python unit tests passed.
- Covered signed execution, missing authority, forged signatures, expiry, quantitative and semantic scope overflow, one-shot replay, revocation, missing simulation evidence, JSON Schema validation, lifecycle transition evidence, paired hidden-state fixtures, and the Python/TypeScript HMAC vector.

## TypeScript overlay

- Strict TypeScript type-check passed with the supplied overlay configuration.
- Two Node tests passed: grant round-trip signing and the shared Python/TypeScript canonical HMAC vector.

## Service boundary

- FastAPI `POST /v1/contracts/evaluate` returned an allow receipt for a valid current grant.
- The evaluation endpoint returned `executed: false`, preserving the distinction between permission and execution.

## Gate dataset dry run

The included two-pair demo generated 28 episodes across 14 conditions.

- 22 denied decisions
- 6 allowed decisions
- 6 authorized simulated executions
- 0 unauthorized executions

See `artifacts/demo/gate_summary.json` for condition-level counts and denial reasons.

## Synthetic mechinterp plumbing smoke

A synthetic 16-dimensional activation dataset tested the analysis code paths without using a language model.

- 112 paired activation rows
- top-k SAE train and checkpoint export completed
- contrast-specific feature scoring completed
- fixed-SAE base/adapter comparison completed
- 8 mock VPD components passed shape and finiteness checks
- SAE/VPD join emitted 32 candidate pairs

The artifact `artifacts/demo/synthetic_mechinterp_smoke.json` is a software test. It carries no mechanistic result.

## Work that requires the real repositories and checkpoints

- insertion into the current BitAgent execution call site;
- insertion into the current Small Control Harness adapter and receipt conventions;
- activation capture from the base and BitAgent checkpoints;
- the existing external VPD optimizer and Rust rank-k audit;
- true VPD parameter-component intervention;
- held-out causal mechanism results.
