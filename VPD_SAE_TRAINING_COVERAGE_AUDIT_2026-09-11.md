# Saved-activation training coverage: a diagnostic, not a repair result

Date: 2026-09-11. New exploratory analysis of the two archived Q4 captures. No model inference, encoder recomputation, SAE fitting, weight changes or paid compute launch occurred. This is new analysis, not recovered historical research.

## Result relevant to compartmentalization

Generated-row reconstruction error and distance from training trajectories are strongly associated across the ten evaluation examples in each archived site/pass check: descriptive Pearson r = 0.949–0.994. However, the median evaluation-example distance is only 0.972–1.101 times the median training-example leave-one-trajectory-out distance. The evaluation set is therefore not uniformly much farther away under this particular metric than training trajectories are from one another.

This qualifies the quality-repair hypothesis. It is consistent with inadequate transfer between sparsely sampled trajectories, but does not establish that more trajectories will repair reconstruction, identify a memorization mechanism, or distinguish optimization from content/scale effects. Existing generated-phase quality still fails in all four fits. No additional activation category, feature compartment or separate SAE dictionary is admitted.

| Archive / site | Generated distance ratio, pass 0 | Pass 1 | Generated descriptive r, pass 0 | Pass 1 |
|---|---:|---:|---:|---:|
| 2K / layer 11 | 1.0302 | 1.0416 | 0.9851 | 0.9843 |
| 2K / norm | 1.0319 | 1.0237 | 0.9812 | 0.9940 |
| 8K / layer 11 | 1.0222 | 0.9776 | 0.9845 | 0.9492 |
| 8K / norm | 0.9720 | 1.1013 | 0.9771 | 0.9929 |

These are eight related archive/site/pass checks, not eight independent experiments. Both archives reuse seed and example IDs. Prompt ratios are lower, 0.334–0.728, but prompt reconstruction remains a separate endpoint and cannot qualify generated feature groups.

## Measurement and checks

`experiments/authz_mechinterp/audit_q4_sae_training_coverage_v1.py` measures squared Euclidean nearest-row distance separately for each site, recurrent pass and prompt/generated phase. Distance and full saved-code reconstruction error share a denominator: training variance per row for that same phase/pass/site. Euclidean distance reflects both magnitude and direction; it is not a semantic distance or an estimate of manifold density.

For each training row, exclude **all rows of its own trajectory** from the nearest-reference search. For evaluation rows, repeat the search across all eight leave-one-training-trajectory-out reference pools and average the minima. Thus both comparisons use seven reference trajectories, with equal retained row counts. This avoids a self-neighbor shortcut and an eight-versus-seven reference-size advantage. No evaluation trajectory enters a training reference pool.

Average row measurements within each example, then take the median example distance for the evaluation/training ratio. Compute the descriptive correlation over ten evaluation-example means, not hundreds of supposedly independent rows. Family summaries are retained: six ARC examples, two math examples, one campsite and one Dyck example. Family, content, activation scale and repeated tokens may confound the correlation; no p-value, confidence interval or causal claim is supplied. Training leave-one-out distances are **not** leave-one-out SAE errors: the original SAE fitted those training rows.

The runner checks source receipt integrity, capture/SAE manifests, recorded array hashes, IDs, splits and token positions. It reconstructs from saved codes and decoders and verifies the recorded mixed-phase train/evaluation EV before computing this diagnostic. It does not certify fresh data, model revision parity or original experiment validity beyond those checks.

## Subsequent family and influence controls

The large pooled correlations prompted a **post-hoc** sensitivity analysis, not a preregistered confirmation. After subtracting each observed family's mean from both distance and error, generated-phase correlations remain 0.957–0.998. Within-family cross-products account for 29.4–74.7% of the original centered cross-product; this is a covariance decomposition, not explained variance or a causal contribution. Family differences contribute substantially but do not wholly account for the observed association.

| Archive / site | Family-centered r, pass 0 | Pass 1 | Adjusted conditional p, pass 0 | Pass 1 |
|---|---:|---:|---:|---:|
| 2K / layer 11 | 0.9566 | 0.9921 | 0.0444 | 0.0222 |
| 2K / norm | 0.9976 | 0.9984 | 0.0222 | 0.0556 |
| 8K / layer 11 | 0.9701 | 0.9583 | 0.0222 | 0.0444 |
| 8K / norm | 0.9705 | 0.9957 | 0.0556 | 0.0222 |

The conditional calculation enumerates all 6! × 2! = 1,440 within-family assignments of example-mean errors, with distances and family labels fixed, then applies Bonferroni across all 16 prompt/generated × pass × site × archive checks. Six of eight generated checks have adjusted p ≤ 0.05; two do not. These are exploratory sensitivity numbers, not admissions. Their calibration assumes within-family exchangeability, which repeated templates or content can violate. Choosing this diagnostic after seeing the pooled correlation prevents a confirmatory interpretation.

Only the six ARC and two math examples supply within-family variation; the two singleton families have zero centered residuals. ARC-only generated correlations are at least 0.944, but only six examples support each calculation. Across all generated checks, the minimum leave-one-example-out pooled r is 0.925 and the minimum leave-one-family-out pooled r is 0.943. These observations reduce the single-example/family-only explanations; they do not control activation magnitude or other shared content effects and do not establish that breadth training will improve quality.

New implementation: `experiments/authz_mechinterp/audit_q4_sae_coverage_family_controls_v1.py`. The earlier coverage runner and receipt remain unchanged. The new runner independently reproduces the coverage source from saved arrays before recording its derivative result. No category allocation, final holdout, model weight or canonical protocol was changed.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_q4_sae_coverage_family_controls.py tests/test_vpd_q4_sae_training_coverage.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.audit_q4_sae_coverage_family_controls_v1 --verify artifacts/vpd_q4_sae_coverage_family_controls_v1/report.json
```

Nine new tests cover disappearance of a family-only association, retained Simpson reversal, exact factorial permutation calibration, singleton limits, covariance reconstruction, input-order invariance and invalid/unbounded input rejection. Together with the original coverage tests, 21 focused tests pass; the combined thirteen-file audit/planner/adapter suite passes 156 tests in 1.25 seconds. This is not the full project suite. The family-control receipt reproduces exactly, including source-array replay.

- Receipt: `artifacts/vpd_q4_sae_coverage_family_controls_v1/report.json`
- Report ID: `1cdfa34ebde98713b5a75b64c8c5ec582a975b1418d421933d7aa57828b22d1a`
- File SHA-256: `0515b0d15f0b3529a5b10858172f6bae33b49ddce9330df3530779a104b9bd61`

## Original coverage reproduction and handoff

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_q4_sae_training_coverage.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.audit_q4_sae_training_coverage_v1 --verify artifacts/vpd_q4_sae_training_coverage_v1/report.json
```

Twelve focused tests passed, including whole-trajectory exclusion, equal reference pools, translation/scaling invariance, invalid-input rejection, example-level aggregation and undefined-correlation handling. The report reproduced exactly from local arrays in a second run.

- Receipt: `artifacts/vpd_q4_sae_training_coverage_v1/report.json`
- Report ID: `01e2949e876ad93d9c389c77cff2630ddca5395cc95e4134d0836ffb9ee3517d`
- File SHA-256: `83ca7fff7562591feaf7e8fbff0f82e3afd584a1f4ca858a21b8ccb77bb61329`

Q2 remains a proposed matched-budget breadth-versus-density experiment, not a proven remedy or an authorized launch. Preserve this diagnostic as development evidence. Do not select final evaluation examples by favorable distance, error or family results, and do not add a new admission threshold after seeing these values. The fresh-pack shortage, quality/effect criteria, training-entry-point verification and independent provider shutdown requirements remain unresolved. The sealed V52/V44 protocol/runbook and 25-report canonical spec remain unchanged; this diagnostic is supplemental.
