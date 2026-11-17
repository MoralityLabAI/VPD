# B4 v2 numerical-closure audit | 2026-11-17

Source audit: VPD commit eab20f9f95d4ec0d3767456e4aa398e273a62dec.
Measured follow-up: mechinterp/bpd/results/v2/gate_b4_v2_controls_and_ablation_split.json
on branch feat/mechinterp-bpd (historical receipt retained unchanged).
Model checkpoint: SHA-256 65ecb2ab6c4a144b3563cecf49d6a55b157201fa98ef0b6dc652ff3effc33cbc.
The experiment uses the same 16 evaluation boards and readouts at 20, 41 and 62.

## Numerical identity control

The existing all-components-on plus delta-on control should represent the target
algebraically. Under CUDA BF16, the target executes F.linear(x, W_target) in BF16,
while the masked component path evaluates the factor product and residual in FP32
and casts their sum to BF16. In recurrent passes, a small difference in one
projection can trigger a large change in a stalled trajectory.

| KL to native BF16 target, pass 62 | gaoxin492 | yagizdevre |
|---|---:|---:|
| FP32 matmul at affected target sites | 0.309060 | 0.263795 |
| Entire full-precision target | 0.315935 | 0.718395 |
| All 512 SVD-start factors, delta off | 0.054096 | 0.818602 |
| All trained factors, delta on | 0.365314 | 1.534241 |

The SVD reconstruction has relative Frobenius residual about 3.2–3.5e-7.
The largest trained identity-control board KL at pass 62 was 5.317154 for
gaoxin492 (board 11) and 11.048561 for yagizdevre (board 13).
Most late-pass board KL values in the historical receipt were near zero.
This is a precision-sensitive nonlinear instability concentrated on a few boards.
The 0.15 registered final-KL criterion is therefore below the observed
identity-control divergence on this evaluation set.

The existing entire-FP32-target control loads original full-precision checkpoint
weights. The revised script adds a separate full-FP32 execution of the SAME
previously-loaded BF16-quantized weights, so that checkpoint quantization can be
separated from arithmetic precision. It also adds a same-dtype target rerun and
SVD-start all-components-plus-delta control. Results for these new arms await a
fresh execution.

## Ablation split on the fixed clean-informed dynamic SVD mask

The following three rows of the historical 16-board receipt remain interpretable
at pass 20, on the same frozen top-307/512 mask:

| Pass-20 mean KL | gaoxin492 | yagizdevre |
|---|---:|---:|
| Zero-fill; sphere off | 0.020070 | 0.006205 |
| Pass-mean-fill; sphere off | 0.443553 | 0.162321 |
| Pass-mean-fill; sphere on | 0.621591 | 0.437177 |

Mean replacement drives the main damage at pass 20. Sphere rescaling adds
further damage in this comparison. The originally reported
"mode_zero__sphere_rescale_true" arm is **confounded**: the implementation
zeroed mean_comp_acts but left mean_delta_out nonzero, so delta_mask=0
silently injected the pass-conditional mean residual.

This patch zeroes both pass-mean buffers inside a protected context, then
restores them. Calibrated mean_write_rms is left intact to provide the
sphere target. Rerun before citing corrected zero+sphere values.

## Clean-informed CI gate and SVD masks

BPDTrainer.evaluate() computes ci_net.forward_trajectory(clean_traj) before
running the masked rollout. Its deterministic CI masks are thus derived from
the *clean target* hidden states. The dynamic SVD top-k mask is also selected
from clean target pre-module activations. Both are oracle-informed diagnostic
procedures. The 0.0103159 final-KL learned-mask receipt cannot establish
online closed-loop performance, and its CI weights are missing from the
stage-four factor-only export.

## Reporting rule

Publish all 16 per-board KL values and the backend/dtype, preserving full
precision before summarizing. Use pass 20 as the substantive comparison
point until the 63-pass identity-control floor closes on every board.
Keep Gate B4 marked incomplete; the downstream stages 5a–8c have not run.
These 16 boards are a diagnostic panel, not an adequately powered sample
for mechanistic generalization or a final superiority claim.

For causal gating, recover or refit the CI network with V/U frozen, then
compare clean-informed masks against gates recomputed online from each
perturbed rollout. Pre-register board-stratified performance and use a
larger held-out evaluation with paired uncertainty intervals.

## Run

From the LoopedTransformers checkout that contains this branch, with the
registered B4 V/U checkpoint at mechinterp/bpd/results/v2/:

    python -m pytest mechinterp/bpd/tests/test_b4_v2_closure_controls.py -q
    python mechinterp/bpd/scripts/audit_b4_v2_controls_and_ablation_split.py --device cuda --out mechinterp/bpd/results/v2/gate_b4_v2_closure_controls_corrected.json

The SHA-256 guard rejects a different factor checkpoint by default.
Exploratory --device cpu runs use FP32 as their reference, so they cannot
substitute for the original CUDA BF16 numerical floor. The historical JSON
is retained and is not automatically overwritten.
