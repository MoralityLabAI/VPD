# VPD cheap-executor report: cross-artifact graph gate

Date: 2026-09-01  
Project root: `C:\projects\VPD`  
Scope: local, model-free, nonexecuting verification and implementation

## Decision and authority boundary

The authorization-control release decision remains **reject Bonsai-8B base and
adapter**. The authoritative V7 values are base separation `0.692347`, adapter
separation `0.580743`, adapter-minus-base `-0.111604`, 95% CI
`[-0.122397, -0.100996]`, invalid `REQUEST` rate `0%`, and valid `PROPOSE`
rate `100%`. V7 trained no weights. V6 remains scientifically invalid.

This packet authorizes only local source inspection, JSON construction in the
named artifact locations, syntax checks, model-free benchmarks, and focused
tests. It does not authorize model or adapter loading, inference, GPU or paid
jobs, empirical capture, routing of new activations, SAE training, weight
mutation, dictionary output, real actions, or use of `final_evaluation` for
selection.

## Verified intermediate result

The new cross-artifact contract builder/verifier compiles and its generated
contract replays exactly.

| Item | Result |
|---|---|
| Required graph nodes | 20 |
| Required cross-artifact edges | 37 |
| Required execution-identity bindings | 10 |
| Contract verification | pass |
| Contract ID | `a01ac25024b52e64c9b23ad1e3f3203b17403a8164becac168bd1fda7a4d754e` |
| Builder SHA-256 | `e1b6e8435f8a779571f77d978ec217fd967536cc2d1175e2bbc4bb5889bf3f10` |
| Contract-file SHA-256 | `45fc17c7d522b8659b4a2c7628780974edd9d86469cc32c44aa954b7d63a8e27` |
| Synthetic benchmark | 34/34 controls pass |
| Benchmark ID | `178c9cd1124a4b93c36b1c40746a1776ea09ef820e56f8ad2aa7f0fab3c354fb` |
| Protocol V46 SHA-256 | `02d046536a330c56da723080738277ba22c786bdb3737c6c30bde82aac6dd0a1` |
| Runbook V38 ID | `ab7b2c5658ef05d4ab9ea25b7f14f577c60b6c740a31d227fc45080a69ca5c33` |
| Focused tests | 6/6 pass in 115.42 seconds |
| Semantic-dispatch coverage audit | 12/20 node identities, 4/20 dispatch-ready, 9/37 exact edges |
| Semantic coverage audit ID | `bd3cc043f7b658029fd20c99492c43edbace23550a7e237d36d692e46297eaa7` |
| Semantic coverage benchmark | 23/23 controls pass |
| Semantic coverage benchmark ID | `adc1d00bf3be2ab9d790252534bc084add403842b26ae9002e2d6216f0095732` |
| Protocol V47 SHA-256 | `a7bd3d23ad2ecce3b5933dae59b70a7de72212fec103228a149931048b6dc15c` |
| Runbook V39 ID | `0533d2b242ccb8a566a8dee69146728214584abed4ff6f46a4251529d41fb68c` |
| Semantic audit and lineage tests | 6/6 pass in 1.71 seconds |
| Empirical graph supplied | no |
| Synthetic graph persisted | no |
| Execution/training authorized | no |

The contract joins capture, routing, stability, global pair split, SAE plan,
materialization, execution blueprint, budget proposal/admission, signed release
and compute authority, nonce consumption, output reservation, worker claim, and
runtime attachment. Structural verification deliberately does not authenticate
signatures, authority status, nonce freshness, output exclusivity, or execution.

The existing empirical-budget worksheet remains at 0 supplied / 50 missing
bindings. Therefore no empirical manifest can honestly pass this gate yet.

## Cheap-model run ladder

Run each stage from `C:\projects\VPD`. Stop at the first nonzero exit, observed
ID/hash mismatch, model-framework import, network request, model/adapter access,
or request for empirical data that is not already supplied.

### C0 — reproduce the sealed contract (allowed now, cheapest)

```powershell
.\.venv\Scripts\python.exe -m py_compile experiments/authz_mechinterp/verify_cluster_aware_sae_cross_artifact_graph_v1.py
.\.venv\Scripts\python.exe -c "from experiments.authz_mechinterp.verify_cluster_aware_sae_cross_artifact_graph_v1 import NODE_SPECS,EDGE_SPECS,REQUIRED_EXECUTION_BINDINGS; assert (len(NODE_SPECS),len(EDGE_SPECS),len(REQUIRED_EXECUTION_BINDINGS))==(20,37,10); print('20/37/10 pass')"
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.verify_cluster_aware_sae_cross_artifact_graph_v1 --verify-contract artifacts/vpd_cluster_aware_sae_cross_artifact_graph_contract_v1/contract.json
```

Required result: exit 0 and contract ID
`a01ac25024b52e64c9b23ad1e3f3203b17403a8164becac168bd1fda7a4d754e`.
Do not regenerate the contract unless the builder intentionally changes.

### C1 — synthetic structural benchmark (completed locally)

Created:

- `experiments/authz_mechinterp/benchmark_vpd_cluster_aware_sae_cross_artifact_graph_v1.py`
- `artifacts/vpd_cluster_aware_sae_cross_artifact_graph_benchmark_v1/report.json`

The benchmark creates its 20-node fixture only in a temporary child below
`artifacts/vpd_cluster_aware_sae_empirical_graph_inputs`, validate it, remove the
temporary child, and persist only the deterministic benchmark report. It must
not open a model, adapter, activation tensor, or dictionary.

All 34 controls pass, including:

1. Clean synthetic graph passes with 20 nodes, 37 edges, and 10 bindings.
2. Contract reproduction and report reproduction pass.
3. Missing, duplicate, and unknown nodes fail.
4. Wrong file hash, content ID, schema, and edited node content fail.
5. Edge substitution fails even when the edited source node, descriptor, and
   manifest are all re-addressed.
6. Budget-policy ID substitution fails.
7. Preflight/execution-binding mismatch and missing binding fail.
8. Absolute paths, traversal, graph-root escape, missing files, and symlinks
   fail closed.
9. Mixed synthetic/empirical nodes and graph-kind mismatch fail.
10. The result keeps signature, authority, nonce, reservation, execution, and
    training verification false.
11. No synthetic graph child remains after the benchmark.

Use content addressing exactly as declared by each node spec. For signed receipt
nodes, exclude the ID field and `signature`; for other nodes, exclude only their
declared ID field. Do not weaken the verifier merely to make a fixture pass.

Then run:

```powershell
.\.venv\Scripts\python.exe -m py_compile experiments/authz_mechinterp/benchmark_vpd_cluster_aware_sae_cross_artifact_graph_v1.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_cross_artifact_graph_v1 --output artifacts/vpd_cluster_aware_sae_cross_artifact_graph_benchmark_v1/report.json
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_cross_artifact_graph_v1 --verify artifacts/vpd_cluster_aware_sae_cross_artifact_graph_benchmark_v1/report.json
```

If the CLI convention in neighboring benchmarks is `--verify` without
`--output`, follow the neighboring convention and document the exact commands.

### C2 — focused tests (completed locally)

Direct tests cover contract reproduction, clean validation, representative
node/hash/edge/path/reparse-point tampering, synthetic cleanup, and the
non-authorizing boundary. The combined direct and lineage suite passes 6/6 in
115.42 seconds.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_cluster_aware_sae_cross_artifact_graph_v1.py
```

### C3 — successor lineage seal (completed locally)

Protocol V46 and Runbook V38 pin the contract and benchmark IDs/hashes,
preserve all older evidence, state that only a temporary synthetic graph
passed, and keep `safe_stage_now: 0`. V45/V37 are checked by immutable hash,
schema/content address, and non-authorizing boundary; the new graph artifacts
are fully replayed. This avoids recursively rerunning the entire historical
lineage on every cheap verification.

New paths:

- `experiments/authz_mechinterp/derive_unified_vpd_compartment_protocol_v46.py`
- `configs/evals/vpd_activation_category_compartments_v46.json`
- `experiments/authz_mechinterp/derive_vpd_cheap_executor_runbook_v38.py`
- `artifacts/vpd_cheap_executor_runbook_v38/runbook.json`
- `tests/test_vpd_cluster_aware_sae_cross_artifact_graph_lineage_v38.py`

### C4 — semantic-dispatch coverage audit (completed locally)

The audit found that directly implementing the planned dispatcher would have
cemented fictional interfaces. Only 12/20 graph node schema/ID pairs match the
implemented artifacts, only 4/20 nodes are dispatch-ready under the V1 graph,
and only 9/37 declared field edges match implemented fields. The 23-control
benchmark reproduces these gaps and confirms that all referenced real verifier
callables exist without importing a model framework. Protocol V47 and Runbook
V39 preserve V46/V38 as structural-gap evidence and keep semantic dispatch
disabled.

Key mismatches include `authorization_receipt_id` versus `receipt_id`,
`capture_id` versus `capture_manifest_id`, `routing_evidence_bundle_id` versus
`routing_bundle_id`, an embedded rather than standalone global split, no
blueprint binding in budget proposal/admission V1, a signed compute receipt
rather than a compute manifest, one combined durable transition rather than
three standalone nonce/output/worker documents, and only a synthetic inert
runtime attachment verifier.

### C4A — next implementation: corrected semantic graph successor

Preserve V1 and build a versioned successor around actual guarded V3 capture
execution receipts and semantic sidecars, nested routing evidence, embedded
plan/split materialization, exact blueprint binding, a budget-admission V2 that
binds the blueprint ID and file hash, signed release/compute receipts, and the
combined durable transition receipt. Production runtime attachment remains an
explicit unimplemented boundary. Do not invoke a semantic dispatcher until the
successor contract and its adversarial benchmark pass.

### C5 — blocked empirical/training stages

Empirical capture, routing/materialization, compartment SAE training, and V9
authorization retraining remain separately blocked. They require explicit user
authorization, complete real bindings and content hashes, active signed scope,
fresh nonces, isolated outputs, an admitted budget, and a production runtime.
V9 design must train invalid/untrusted observable `AUTH_STATUS` to `REQUEST`
and valid/trusted observable `AUTH_STATUS` to `PROPOSE`, with a fresh pair- and
template-disjoint held-out pack. Hidden/conflicting labels are forbidden.

## Required receipt from the cheaper executor

For every attempted stage, report:

- exact command and exit code;
- observed artifact ID and file SHA-256;
- files created or changed;
- control/test pass counts and elapsed time;
- whether any temporary directory remains;
- all scientific-boundary flags;
- the first stop reason, if any.

Never describe reconstructed or newly generated artifacts as recovered
originals. Preserve the historical task evidence and all V1–V8 receipts.
