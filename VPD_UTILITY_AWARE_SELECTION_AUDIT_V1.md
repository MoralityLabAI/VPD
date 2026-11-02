# Utility-Aware VPD Selection Audit V1

For new comparisons, use the [cross-research addendum](VPD_CROSS_RESEARCH_UTILITY_ADDENDUM_V1.md) and its versioned manifest. It applies this framework to reasoning, Receipt Replay, HRM policy, and authorization, distinguishes selectors from catalog candidates, keeps unmeasured challengers in regret bounds, and replaces unspecified cost penalties with separate utility/resource reporting. The V1 artifact below is retained as the original design; its zero storage/FLOP placeholders are not measurements of local audit work.

## Current status

This is an archive-only evaluation design. It does not load a model, rerun remote work, train a dictionary, make a model call, or edit coordinates. The machine-readable contract is [audit.json](artifacts/vpd_utility_aware_selection_audit_v1/audit.json), which is reproducibly derived from the sealed activation-category specification.

The design starts from the decisive evidence rather than trying to rescue it: equal-total-rank separate pass dictionaries earned zero admissions across 36 rank/site/archive checks (35 losses to monolithic and one tie); the current preferred architecture remains one shared monolithic dictionary with post-hoc factor-labelled feature groups where quality permits. The shared-axis fit used the two existing packs for fitting. It is neither third-domain transfer nor semantic attribution. Loader conformance is also expressly excluded as scientific validation.

## Frozen comparison

The candidate catalog has exactly three selectors, all operating on the same pre-existing shared dictionary, the same frozen candidate pool, total rank 64, and at most 64 edited coordinates. There are zero additional dictionary fits in this archive.

| Candidate | Selection rule | What it cannot establish |
| --- | --- | --- |
| `q_monolithic` | highest calibration mean net utility from the global rank-64 pool | geometric value |
| `q_geometric` | predeclared training-only shared-axis projection with calibration-only threshold/sign | transfer, semantics, or a separate-dictionary case |
| `q_interval_aware` | largest simultaneous lower bound on calibration net utility after regret and benign-loss screens | superiority without final held-out evaluation |

The final-evaluation families cannot choose a candidate. Each selector is frozen using discovery and calibration families only. This prevents the interval-aware selector from becoming a post-hoc winner-picking device.

## Utility, uncertainty, and regret

The target is held-out benign task utility after a bounded coordinate edit, reported per independent semantic-family block relative to no edit. The primary estimand is mean net utility, while raw task utility and each cost are retained separately.

For a selected candidate `q`, report both its improvement over no edit and its regret to the best candidate reachable from the finite, feasible catalog. With simultaneous 95% intervals `[L_qp, U_qp]` for pairwise utility `theta_qp = U(q) - U(p)`, the challenger-aware regret envelope is:

`[max(0, max_p(-U_qp)), max(0, max_p(-L_qp))]`.

The upper endpoint makes missed challengers concrete: it is the largest advantage any evaluated, budget-feasible challenger could still have at simultaneous coverage. The report must name the challenger achieving each bound and list every infeasible or unmeasured challenger with a reason.

Intervals use a 20,000-draw, fixed-seed, paired family-block studentized multiplier bootstrap with a max statistic across all three baseline comparisons, all ordered candidate pairs, and benign-loss comparisons. Individual rows or tokens are never treated as independent samples.

## Costs and controls

Functional budget is held constant: total rank 64, at most 64 edited coordinates, and zero new dictionary fits. Physical access costs are charged and reported independently: model calls, execution-receipt FLOPs, storage bytes and hashes, edited-coordinate count/edit norm, and benign-utility loss. All are zero or unmeasured in this archive; none may be inferred from loader success.

Required controls are held-out semantic families, preseeded rank-64 random subspaces, and a pass-label-only selector that receives neither activation geometry nor coordinate-level effect scores. A geometric gain that a label-only or random-subspace control matches is a falsification, not a weak positive result.

## Decision boundary

A candidate may be reported as better than no edit only if its simultaneous improvement lower bound is positive, benign-loss upper bound is under its preregistered limit, regret-envelope upper bound is under its preregistered tolerance, and every cost/control receipt is present. Otherwise retain no edit and the incumbent shared-monolithic interpretation.

Before any model call, the project must record explicit compute authorization, a frozen family manifest and candidate-pool hash, thresholds for utility/benign loss/regret/costs, and durable execution receipts. A live edit additionally needs explicit live-edit authorization.
