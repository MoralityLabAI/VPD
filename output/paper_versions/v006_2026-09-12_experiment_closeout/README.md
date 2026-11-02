# Version 6: experimental closeout

The 15-page manuscript retains the mechanistic argument and completed VPD/SAE results. The new material is in Section 7 and Appendices A, C and D. Earlier numbered revisions remain unchanged.

## Evidence added

- All 45 completed Nanbeige comparator calls graded, including 11 capped outputs: 20 vanilla puzzle outcomes and 19 outcomes for each other reported procedure. The 214 missing calls remain unrun.
- Exact-puzzle comparison to archived Qwen results. Nanbeige routed repair accepts 6/19; Qwen accepts 3/19 on the same puzzles. This is a partial, exposed, answer-only cohort, not a general ranking.
- Published-verifier acceptance and prompt-literal validity remain distinct. All six accepted Nanbeige routed repairs fail the literal check; this known dataset contract discrepancy is explained in the paper.
- Ledger gate: one correct final response, two length-censored responses, nine assigned questions not run, zero ledger episodes. No ledger probe or monitoring result is claimed.
- All 96 observed procedure rows independently checked against grid/count/adjacency and bit-mask matching predicates (192 checks); 16 closeout/resumption/comparison tests pass. The earlier ledger suite records 80 passing tests.

## Reproducibility

The supplement contains immutable grading rows, exact paired IDs, protocol and terminal gate responses, recovery receipt, independent validation, and snapshots of the modified analysis scripts. Earlier supplements are retained as historical evidence, including their original progress records; the current status is in closeout_verification.json.

Run from the VPD workspace using its existing environment and source artifacts. Use fresh output directories when reproducing; the tools refuse to overwrite completed result directories.

```powershell
.venv/Scripts/python.exe -m experiments.nanbeige_reasoning.grade_logic_skill_comparison --run artifacts/nanbeige_reasoning/nanbeige_skill_comparison_run_v2 --preparation artifacts/nanbeige_reasoning/nanbeige_skill_comparison_inputs_v1 --output artifacts/nanbeige_reasoning/partial_grading_reproduction --allow-partial
```

The matching stage uses compare_skill_models with --allow-partial-nanbeige and the archived Qwen grading_v2; exact source paths and hashes are retained in the grading/comparison verification records. Analysis scripts require the repository's pinned verifier and helper dependencies. This revision directory is a manuscript and results bundle, not a replacement for the activation archive or model checkpoint.

No additional model inference was performed during closeout. GPU processes from these experiments are stopped. The original 85 C pause / 87 C hard stop policy is unchanged. The full benchmark and ledger semantic experiment are unfinished; their observed endpoints are now documented rather than treated as pending paper results.
