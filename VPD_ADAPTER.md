# VPD Adapter Contract

The starter package treats the existing VPD repository as an external parameter-analysis engine. Small Control Harness supplies datasets and validates outputs. VPD remains independently versioned.

## Input

`vpd_dataset.jsonl` contains one row per paired episode:

```json
{
  "id": "authz-valid_visible-0000",
  "pair_id": "pair-0000",
  "group": "task-0",
  "condition": "valid_visible",
  "text_pre_gate": "...",
  "text_post_gate": "...",
  "labels": {
    "trusted_visible_provenance": 1,
    "scope_match": 1,
    "host_grant_valid": 1,
    "gate_allowed": 1
  }
}
```

`config/vpd_manifest.yaml` lists each contrast and its allowed condition subset. VPD should fit each contrast separately or include the subset identity in every component record.

## Recommended parameter target

For a BitAgent LoRA or merged adapter, analyze

\[
\Delta W=W_{\text{BitAgent}}-W_{\text{base}}.
\]

A LoRA adapter already supplies low-rank factors. VPD can score and regroup those factors, or decompose the merged delta. This is cheaper and identifies the post-training mechanism more directly than decomposing all pretrained weights.

## Required output 1: components.jsonl

One record per retained component, in the same order as `component_ids` in the NumPy archive:

```json
{
  "component_id": "layer12.mlp.down.rank2.0007",
  "contrast": "exact_scope_match",
  "module_path": "model.layers.12.mlp.down_proj",
  "layer": 12,
  "rank": 2,
  "score": 4.72,
  "checkpoint_ref": "...",
  "intervention_ref": "...",
  "calibration_steps": 5
}
```

`intervention_ref` should identify enough saved state to apply the true parameter component with coefficient `alpha`.

## Required output 2: activation_effects.npz

Required arrays:

```text
component_ids  shape [n_components]
delta_h        shape [n_components, hidden_size]
```

For each component, estimate `delta_h` through a small signed parameter perturbation on the contrast prompts. Keep the anchor and layer fixed. Record the perturbation coefficient and estimator details in component metadata.

Validate the artifact:

```bash
python -m experiments.authz_mechinterp.validate_vpd_artifacts \
  --components artifacts/authz_mechinterp/vpd/components.jsonl \
  --effects artifacts/authz_mechinterp/vpd/activation_effects.npz \
  --hidden-size 1536 \
  --output artifacts/authz_mechinterp/vpd/validation.json
```

## SAE join

```bash
python -m experiments.authz_mechinterp.join_vpd_sae \
  --sae-decoder artifacts/authz_mechinterp/sae/layer_12_proposal/sae_decoder.npy \
  --sae-scores artifacts/authz_mechinterp/sae/scope_features.jsonl \
  --vpd-components artifacts/authz_mechinterp/vpd/components.jsonl \
  --vpd-effects artifacts/authz_mechinterp/vpd/activation_effects.npz \
  --output artifacts/authz_mechinterp/join/vpd_sae_pairs.jsonl
```

Cosine alignment selects candidates. A mechanism claim still requires held-out SAE steering and true parameter-component intervention.

## Calibration gate

The first VPD pass should remain small:

- layers 0, 8, 12, and 13 for the 16-layer HRM path, or matching fractions for Qwen;
- five optimization steps;
- grouped ranks 1, 2, 4, and 8;
- post-hoc Rust rank-k audit when the PyO3 extension is available;
- selected checkpoint retention rather than every layer checkpoint.
