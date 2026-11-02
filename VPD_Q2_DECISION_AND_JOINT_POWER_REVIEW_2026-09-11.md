# Q2 decision and joint-power review

New analytic design audit. No activation capture, model inference, SAE fitting or new Monte Carlo simulation was performed. The existing marginal-power receipt was read and checked for content identity and stored count arithmetic; its simulator was not rerun. The proposed research run remains unready and unauthorized.

## Finding: marginal power is not campaign success probability

The existing `vpd_sae_family_test_power_v1` analysis studies one paired family-level functional test under synthetic independent symmetric location-error assumptions, using alpha `0.05/18`. It explicitly excludes reconstruction quality failures and all-pair success. Those results cannot certify Q2's chance of satisfying every required condition.

For success events with marginal probabilities `p_i`, elementary event bounds give:

`max(0, 1 - sum(1-p_i)) <= P(all succeed) <= min(p_i)`.

Only when the success events are independent is the probability their product. Shared evaluation families, shared training pools and schedules make independence an unverified assumption here.

| Required events | Each event succeeds with probability | All-success range, arbitrary dependence | All-success under independence only |
|---:|---:|---:|---:|
| 6 | 80% | 0–80% | 26.21% |
| 6 | 95% | 70–95% | 73.51% |
| 18 | 80% | 0–80% | 1.80% |
| 18 | 95% | 10–95% | 39.72% |

These are exact illustrations for known marginal probabilities, **not forecasts for the real experiment**. Six is an illustrative conjunction size corresponding to Q2's six arm-pair site/seed conditions; it is not an assertion that its unresolved decision rule contains exactly six elementary statistical tests. Eighteen matches the older simulation's comparison count, not a newly assigned Q2 count.

A sufficient equal marginal success probability to guarantee joint success probability at least 80%, without a dependence assumption, is 96.67% for six required events and 98.89% for eighteen. These sufficient bounds can be conservative; they are not a new mandatory power threshold or measured requirement for the model. Adding quality, completion and non-degradation gates cannot increase the chance that all required events succeed.

The saved Gaussian ten-family simulation estimates marginal power 0.95703125 at an excess effect of two family standard deviations above its unset useful margin. Plugging that estimate into six equal-marginal events gives bounds 0.7421875–0.95703125 and an independence-only product 0.7683458. These are plug-in sensitivity calculations, not confidence bounds or calibrated Q2 power; Monte Carlo estimation error and model-assumption uncertainty remain. The inherited alpha, estimand and synthetic noise model have not been converted into Q2-specific ones.

## Decision contract to resolve before capture

The following distinguishes the claims that must not be conflated. It clarifies the review draft but does not fill missing numerical margins or approve a run.

1. **Quality on the fixed benchmark.** Retain the proposed requirement that B's generated-row EV reaches 0.20 for every declared family and pooled, on both passes. The existing evaluator is a point-estimate prerequisite on frozen arrays, not a lower confidence bound for a family population. Report prompt quality, N quality and all failures too. Do not label the point gate a confidence guarantee.
2. **Useful paired improvement.** Evaluate N and B on exactly the same frozen observations. A candidate family-balanced estimand is the mean across evaluation split blocks of `EV_B - EV_N`, separately by pass and phase. On the same family/pass rows, the difference equals `(SSE_N - SSE_B)/TSS`; this gives the effect margin an interpretable scale. Do not exchange it with pooled EV, mean per-trajectory EV, a best pass, or training EV after seeing results. The final estimand is still to be frozen.
3. **Prompt retention.** Define a separate non-degradation tolerance and uncertainty rule before execution. A good generated-phase result does not establish prompt retention. No numerical tolerance has yet been justified or selected.
4. **Replication and conjunction.** Explicitly list which site/seed/pass claims must all hold, and whether any individual condition will receive its own confirmatory claim. A two-fit first-stage quality pilot cannot establish the full twelve-fit replication result. Do not replace a failed seed, choose the best site, or omit stopped/unrun conditions from the denominator.
5. **Inference unit.** Use reviewed family split blocks as the top-level generalization units. Positions, repeated optimizer draws, source-label variants and initialization seeds do not multiply the number of independently sampled families. Keep within-family trajectory uncertainty distinct from across-family generalization.
6. **Null and multiplicity.** Freeze the null, useful-effect margin, testing assumptions and exact decision logic before choosing alpha allocation. A conjunction-only global claim and a menu of individually reportable discoveries have different error events. Do not blindly copy the older 18-comparison adjustment or silently relax it. The existing sign-flip implementation requires joint sign invariance/symmetric independent family errors; pairing alone or zero expected difference does not establish that condition.
7. **Missing or invalid conditions.** A failed N fit does not make a B-vs-N effect measurable. Missing families, undefined EV, numerical failures and censoring stay visible and prevent the corresponding claim. Define the joint handling of these outcomes in the frozen rule.

Minimum-useful improvement in original EV units is an operational choice about what would justify the next research step, informed by calibration, not a parameter that can be estimated uniquely from the current negative results. The current source gives neither that choice nor BF16 N/B variance. The prompt tolerance, final hypothesis set and joint stop/success rule also remain unset. This audit does not invent those values to make the run executable.

## Next substantive analysis

After an actual calibration design is resolved, estimate paired family/block variability without looking at the new final evaluation pack. Evaluate prospective joint success under the chosen model, including dependence across conditions, quality prerequisites, prompt retention, censoring and failed runs. Show sensitivity to assumptions and uncertainty in the calibration estimates. Do not substitute independent products or the prior four-family archive's sample variance for that work.

The next capture/fit decision must jointly resolve source feasibility and a useful statistical question. Adding nominal families without semantic review, or achieving a finer attainable p-value without adequate joint power, does not resolve either issue. Failure to reject a useful-effect null is not proof that interpretability-fueled improvement is impossible; a failed fixed-benchmark quality gate is narrower evidence about that candidate and capture distribution.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_sae_joint_success_power.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.audit_sae_joint_success_power_v1 --verify artifacts/vpd_sae_joint_success_power_v1/report.json
```

Ten focused tests pass; the new receipt reproduces exactly. Tests check bounds against enumerated joint distributions, the six-event calculation, probability validation and the union-bound sufficient requirement. They do not validate the scientific assumptions on model data.

- Receipt: `artifacts/vpd_sae_joint_success_power_v1/report.json`.
- Report ID: `769e4c84e2d5357a5e5728bdf66d07d473ecffc6988a3b6c194fc6c294b79156`.
- SHA-256: `54bbbfe8203737cc6659e0ebfb97e7ef700569d2447d5ca41179ad393b2bfa3d`.

No canonical protocol, historical power receipt or empirical admission was changed. This finding rules out treating the saved marginal analysis as a joint-power certificate; it does not itself certify or authorize Q2.
