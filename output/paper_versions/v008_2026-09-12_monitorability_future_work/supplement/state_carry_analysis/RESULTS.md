# Recurrent-state carry control

Actual CPU Nanbeige inference on ten complete held-out prompts. Carry and reset both execute 44 layers. Reset removes the first pass state before the second pass; it does not preserve effective dependent depth or simulate independently trained untied weights. One-pass parity validates the reset.

| Edited layer | Dose | SAE | Carry F | Reset F |
|---|---:|---|---:|---:|
| 12 | -0.1 | adamw | -1.1181 | -1.6537 |
| 12 | +0.1 | adamw | -0.9130 | -3.2121 |
| 12 | -1 | adamw | -0.7048 | -1.9006 |
| 12 | +1 | adamw | -0.7792 | -2.9013 |
| 12 | -10 | adamw | -0.3227 | -1.0932 |
| 12 | +10 | adamw | -0.5841 | -0.3884 |
| 22 | -0.1 | adamw | -0.0146 | -0.0241 |
| 22 | +0.1 | adamw | -0.0146 | -0.0241 |
| 22 | -1 | adamw | -0.0146 | -0.0241 |
| 22 | +1 | adamw | -0.0146 | -0.0241 |
| 22 | -10 | adamw | -3.4850 | -10.5197 |
| 22 | +10 | adamw | -1.8172 | -0.0241 |
| 12 | -0.1 | muon_hybrid | -0.2021 | 0.4458 |
| 12 | +0.1 | muon_hybrid | -0.2568 | 0.0468 |
| 12 | -1 | muon_hybrid | -0.2988 | 0.5723 |
| 12 | +1 | muon_hybrid | -0.2996 | 0.3927 |
| 12 | -10 | muon_hybrid | -0.0457 | 0.3632 |
| 12 | +10 | muon_hybrid | -0.0749 | 0.3860 |
| 22 | -0.1 | muon_hybrid | -0.0057 | -0.0005 |
| 22 | +0.1 | muon_hybrid | -0.0057 | -0.0005 |
| 22 | -1 | muon_hybrid | -12.9531 | -52.3673 |
| 22 | +1 | muon_hybrid | -0.0057 | -0.0005 |
| 22 | -10 | muon_hybrid | -0.4835 | -1.4028 |
| 22 | +10 | muon_hybrid | -0.9688 | -0.0005 |

Capture is the final prompt token at layer 22, final executed pass. Dose is absolute L2 residual-write magnitude along normalized VPD component-0 output direction. All six signed magnitudes and both edited layers are retained. Scores pool energy across ten inputs; conditions are not independent replications.

state_quality.json reports frozen-dictionary reconstruction on this distribution. pooled.json also includes final normalization and layer 12, with full top-k and fixed-baseline-support encodings. crossed_pooled.json separates baseline state and propagated direction in an offline L2=1 assay; those counterfactual vectors were not passed through the model.

This experiment tests state carry within one trained checkpoint. It cannot establish that looped training causes the SAE failure or isolate recurrence from effective depth.
