# Fresh Hermes INTELLECT-3 logic comparison

45 / 259 scheduled model calls; Nanbeige4.2-3B Q4_K_M; answer-only; 1,024-token output cap. Unrouted skill outputs are reused exactly across routed arms. Missing calls are unrun, not incorrect.

| Arm | Official correct / observed puzzles | Canonical exact | Literal rule valid | Capped candidates |
|---|---:|---:|---:|---:|
| vanilla | 0 / 20 | 0 | 0 | 11 |
| logic_skill | 0 / 19 | 0 | 0 | 0 |
| routed_candidate | 0 / 19 | 0 | 0 | 0 |
| routed_final | 6 / 19 | 6 | 0 | 0 |
| plain_candidate_routed_repair | 4 / 19 | 4 | 0 | 0 |

The model candidate is preserved before correction. The plain-candidate repair arm isolates the commit-stage helper from the additional retrieved workspace. Reported arms share some calls; their token totals must not be added to estimate actual compute.

- Fresh integration of selected skill and hash-pinned archived helper implementation, not historical replay.
- Synthetic support replaces exposed historical support; all 128 support labels are positive, so critic discrimination is not tested.
- Frozen signature gate was selected on historical outcomes; this is not a clean generalization estimate.
- Answer-only Qwen generation with a 1024-token cap differs from Nanbeige thinking-mode runs.
- TRM auxiliary retrieval/critic/search paths are not neural recurrent transformer passes.
- Nanbeige native template adds its publisher default system message; full rendered prompts differ.
- Same response token cap is not equal compute; model tokenizers, size, runtime and recurrence differ.
- This matched user-message experiment covers Campsite only; ARC/math comparison remains incomplete.
- Answer-only condition is a comparator control, not the native-thinking ledger protocol.
