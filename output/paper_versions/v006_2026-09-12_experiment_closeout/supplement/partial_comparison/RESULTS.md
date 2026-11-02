# Matched user-message Campsite comparison

Each row compares the same procedure on the same public puzzles. Native templates, model architectures, tokenizers and hardware differ. Counts are descriptive for this exposed cohort; they do not isolate recurrence or establish equal compute.

| Procedure | Matched puzzles | Both correct | Qwen only | Nanbeige only | Both incorrect | Nanbeige minus Qwen | Qwen / Nanbeige capped |
|---|---:|---:|---:|---:|---:|---:|---:|
| vanilla | 20 | 0 | 0 | 0 | 20 | -0.0 percentage points | 20 / 11 |
| logic_skill | 19 | 0 | 1 | 0 | 18 | -5.3 percentage points | 0 / 0 |
| routed_candidate | 19 | 0 | 0 | 0 | 19 | -0.0 percentage points | 0 / 0 |
| routed_final | 19 | 3 | 0 | 3 | 13 | 15.8 percentage points | 0 / 0 |
| plain_candidate_routed_repair | 19 | 3 | 1 | 1 | 14 | -0.0 percentage points | 0 / 0 |

Interrupted scheduled prefix only. Each procedure has its own exact matched puzzle intersection. Missing calls are unrun, not incorrect. Differences between procedures may use different puzzles; use within-model paired results for shared denominators. This is not the full planned benchmark.

Capped candidates retain their published-verifier grades and are counted separately. Procedures share candidate calls and puzzles; table rows are not independent replications. No interval or general-population inference is supplied for this selected, previously exposed cohort. ARC and math are outside this comparison.
