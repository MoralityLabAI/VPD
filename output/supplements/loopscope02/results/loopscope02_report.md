# LOOPSCOPE-02 Automated Replay Summary

- **Bundle SHA-256:** `59459ae0bb60b75db5be017e5f41adead391f2abb0b8fb043cd331c0ee043811`
- **Bootstrap:** `2000` resamples (`seed=20261113`)

## Pre-Registered Verdicts

| Hypothesis | Verdict | Key Quantitative Readout |
|---|---|---|
| **P1** (Capability triangulation) | **Not supported** | TRM s1->s16 `+0.074`; Nanbeige t=2 vs Qwen-4B (S2-S4) `-0.010` (p=0.824) |
| **P2** (ARG-02 causal counterfactuals) | **Not supported** | TRM clean-open `+0.074`, clean-donor `+0.109`, board repair `+0.887` |
| **P3** (SVD operator reuse vs multiplexing) | **Weak** | Nanbeige r(P1,P2)=`0.355`; TRM r_outer(zH)=`0.992`, r_role(zL,zH)=`0.765` |
| **P4** (TCA attractor masking & specificity) | **Not supported** | TRM F1/F6=`1808.359`, max(|z|>3)=`22.04%` |
| **P5** (Q11 compactification & search) | **Supported** | Comm AUC=`1.000`, staircase=`0.653`, stalled solve=`1.000` |

## Stratified Target-Cell Accuracy (256 Distinct Boards)

| Model / Condition | Overall | S1 (Naked-1) | S2 (Prop-2) | S3 (Prop-3+) | S4 (Stalled) |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B-Base (mid-stack L16 lens) | 0.109 | 0.094 | 0.172 | 0.094 | 0.078 |
| Qwen3.5-4B-Base (final L32) | 0.211 | 0.469 | 0.078 | 0.141 | 0.156 |
| Qwen3-4B-Instruct-2507 (final L36) | 0.254 | 0.562 | 0.094 | 0.172 | 0.188 |
| Nanbeige4.2-3B (loop 1) | 0.098 | 0.125 | 0.062 | 0.062 | 0.141 |
| Nanbeige4.2-3B (loop 2, trained) | 0.129 | 0.172 | 0.109 | 0.078 | 0.156 |
| Nanbeige4.2-3B (loop 8, over-looped) | 0.090 | 0.047 | 0.156 | 0.062 | 0.094 |
| TRM 7M (outer step 1) | 0.863 | 1.000 | 1.000 | 0.859 | 0.594 |
| TRM 7M (outer step 16) | 0.938 | 1.000 | 1.000 | 1.000 | 0.750 |
