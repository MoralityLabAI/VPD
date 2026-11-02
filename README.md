# MoralityLabAI VPD

MoralityLabAI's VPD and sparse-autoencoder research workspace, including Nanbeige
pass-resolved experiments, receipt-replay evaluation, authorization controls and
HRM experiments. This repository is a fork of
[Goodfire's parameter decomposition project](https://github.com/goodfire-ai/param-decomp).
The original upstream project is preserved in [`upstream/goodfire/`](upstream/goodfire/).

- [Latest paper: v008, with monitorability Future Work](output/paper_versions/v008_2026-09-12_monitorability_future_work/paper.pdf)
- [Paper source](output/paper_versions/v008_2026-09-12_monitorability_future_work/PAPER.md)
  and [version history](output/paper_versions/README.md)
- [Mechanistic dataset index v2](artifacts/nanbeige_reasoning/dataset_index_v2/README.md)
- [Nanbeige experiments](experiments/nanbeige_reasoning/README.md)
  and [receipt-replay harness](experiments/nanbeige_receipt_replay/README.md)
- [Repository layout, upstream provenance and data availability](REPOSITORY.md)

The completed state-carry study comprises 450 conditions on ten existing inputs.
The full model/skill benchmark and the ledger behavioral study are incomplete;
the paper does not claim recovered chain-of-thought monitorability. Raw model
weights and activation archives are local data dependencies, not included in Git.

## BitAgent Run Contracts + Small Control Harness Authorization Lab

Research follow-up: [utility-aware selection across VPD tracks](VPD_CROSS_RESEARCH_UTILITY_ADDENDUM_V1.md) defines prospective candidate catalogs, fit/access budgets, task utility and challenger-regret comparisons for Nanbeige, Receipt Replay, HRM policy and authorization retraining.

This starter repository adds explicit run contracts, host-issued authorization grants, deterministic action gating, replay and revocation state, execution receipts, and a paired VPD/SAE experiment to BitAgent and the Small Control Harness.

The core boundary is simple:

```text
user confirmation UI
        |
        v
HostAuthorizationService  -- signs --> AuthorizationGrant
        |                                    |
        |                                    v
BitAgent model -- proposes --> ActionProposal --> ContractGate
                                                  |
                                      allow/deny DecisionReceipt
                                                  |
                                    host executor or simulated ledger
```

The acting model can describe a grant, request one, and reference its identifier. It cannot mint a trusted grant, change its scope, validate a hidden signature key, or commit an action. The host owns those powers.

## What is included

- Language-neutral JSON Schemas for `RunContract`, `AuthorizationGrant`, `ActionProposal`, and `DecisionReceipt`.
- A dependency-light Python reference gate with exact scope checks, resource accounting, HMAC-signed local grants, expiry, revocation, one-shot replay protection, invariant hooks, and hashed receipts.
- A simulated TradeLayer ledger for safe experiments.
- A TypeScript BitAgent overlay that compiles strategy intents, issues grants from a host-only service, and calls the Small Control Harness gate.
- A paired counterfactual authorization dataset with model-visible and host-hidden factors crossed independently.
- Activation capture, top-k SAE training, feature scoring, held-out prompt-family checks, paired permutation tests, VPD artifact contracts, SAE/VPD joining, and causal steering scripts.
- A base-model versus BitAgent-adapter comparison using a fixed SAE dictionary.
- A paste-ready Codex integration prompt and Windows overlay installer.

## Repository layout

```text
schemas/                         shared boundary schemas
small_control_harness/           Python gate, lifecycle, ledger, service
bitagent_ts/src/control/         TypeScript integration overlay
experiments/authz_mechinterp/    dataset, capture, SAE, VPD, intervention
scripts/                         demo and overlay helpers
tests/                           deterministic unit tests
artifacts/demo/                  generated dry-run receipts
```

## Core quick start

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[test]"
python -m unittest discover -s tests -v
```

On PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[test]"
python -m unittest discover -s tests -v
```

Generate paired episodes and run them through the deterministic gate:

```bash
python -m small_control_harness.episodes \
  --output artifacts/authz_mechinterp/episodes.jsonl \
  --count-per-condition 8

python -m experiments.authz_mechinterp.run_gate_dataset \
  --input artifacts/authz_mechinterp/episodes.jsonl \
  --output artifacts/authz_mechinterp/gated_episodes.jsonl

python -m experiments.authz_mechinterp.prepare_vpd_dataset \
  --input artifacts/authz_mechinterp/gated_episodes.jsonl \
  --output artifacts/authz_mechinterp/vpd_dataset.jsonl

python -m experiments.authz_mechinterp.build_contrast_manifest \
  --input artifacts/authz_mechinterp/gated_episodes.jsonl \
  --output-dir artifacts/authz_mechinterp/contrasts
```

The gate dry run performs no network call and touches no real wallet.

## Optional local gate service

```bash
pip install -e ".[service]"
SCH_LOCAL_HMAC_KEY=local-development-key \
  uvicorn small_control_harness.service:app --host 127.0.0.1 --port 8765
```

The TypeScript `ContractGateClient` calls `POST /v1/contracts/evaluate`. The HMAC path exists for local integration tests. A deployed version should use a platform-backed asymmetric signer or key service.

## Mechinterp pilot

Install the research extras:

```bash
pip install -e ".[mechinterp]"
```

Capture the same episodes from the base checkpoint and the BitAgent post-trained checkpoint. Use separate output directories while keeping row order fixed:

```bash
python -m experiments.authz_mechinterp.capture_activations \
  --input artifacts/authz_mechinterp/gated_episodes.jsonl \
  --output-dir artifacts/authz_mechinterp/capture_base \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --model-variant base \
  --layers 0,0.5,0.75,0.875

python -m experiments.authz_mechinterp.capture_activations \
  --input artifacts/authz_mechinterp/gated_episodes.jsonl \
  --output-dir artifacts/authz_mechinterp/capture_bitagent \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --adapter /path/to/bitagent-peft-adapter \
  --model-variant bitagent \
  --layers 0,0.5,0.75,0.875
```

Train one SAE per selected layer and anchor on the base capture. The example key below should be replaced with a key listed in `capture_manifest.json`:

```bash
python -m experiments.authz_mechinterp.train_sae \
  --activations artifacts/authz_mechinterp/capture_base/activations.npz \
  --key layer_12__proposal_start \
  --output-dir artifacts/authz_mechinterp/sae/layer_12_proposal \
  --expansion 8 --top-k 32 --steps 2000
```

Score a causal contrast on the BitAgent capture with the fixed base SAE:

```bash
python -m experiments.authz_mechinterp.score_sae_features \
  --activations artifacts/authz_mechinterp/capture_bitagent/activations.npz \
  --metadata artifacts/authz_mechinterp/capture_bitagent/metadata.jsonl \
  --sae-checkpoint artifacts/authz_mechinterp/sae/layer_12_proposal/sae_checkpoint.pt \
  --phase pre_gate \
  --subset-conditions valid_visible,scope_overflow \
  --labels scope_match \
  --output artifacts/authz_mechinterp/sae/scope_features.jsonl
```

Compare post-training feature reweighting directly:

```bash
python -m experiments.authz_mechinterp.compare_model_variants \
  --base-activations artifacts/authz_mechinterp/capture_base/activations.npz \
  --base-metadata artifacts/authz_mechinterp/capture_base/metadata.jsonl \
  --adapter-activations artifacts/authz_mechinterp/capture_bitagent/activations.npz \
  --adapter-metadata artifacts/authz_mechinterp/capture_bitagent/metadata.jsonl \
  --sae-checkpoint artifacts/authz_mechinterp/sae/layer_12_proposal/sae_checkpoint.pt \
  --output artifacts/authz_mechinterp/sae/adapter_feature_deltas.jsonl
```

The VPD stage uses the existing external VPD runner. Its required interchange format is specified in `VPD_ADAPTER.md` and `config/vpd_manifest.yaml`. The included mock artifact only validates plumbing and supports no scientific claim.

## Scientific separation

The experiment isolates two different objects:

1. **Learned authorization cognition.** Pre-gate activations may encode approval language, trusted provenance, revocation, and exact scope. Comparing base and BitAgent checkpoints identifies what post-training changed.
2. **Harness authority.** Hidden grant validity, replay state, budgets, and the final commit boundary live outside the model. Matched pre-gate prompts should contain no information about hidden state. The gate still enforces it exactly.

A useful feature becomes a mechanism candidate only after it survives paired controls, length and position residualization, held-out prompt families, max-stat permutation screening, SAE steering, and a signed VPD parameter intervention on the same behavioral endpoint.

## Current validation status

The deterministic core, schemas, paired episode generator, and integration utilities are runnable locally. The included unit suite covers valid execution, missing grants, forged signatures, expiry, scope overflow, replay, revocation, missing simulation receipts, lifecycle transitions, and schema validation.

The deep-learning pass still requires a local model checkpoint, the BitAgent adapter checkpoint, and the external VPD repository. Its scripts compile here, while no SAE or VPD result is claimed by this starter package.
