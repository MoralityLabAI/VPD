# Family separation review for the next SAE pack

New bounded source/development review. This is not a new dataset, an independence certificate, a revised canonical protocol, or permission to run inference. No recovered generator was executed in this review, and no model or SAE was run.

## Finding: labels overstate the available split units

The verified corpus contains 29 eligible source labels. Two conservative co-assignment rules reduce these to **27 candidate split blocks**, of which 25 are still unreviewed singletons. This is a design upper bound under the two rules, not 27 proven independent semantic families.

| Co-assignment block | Source labels | Development-task evidence and decision |
|---|---|---|
| Bracket matching | `dyck_language`, `dyck_language_errors` | The inspected prompts ask for bracket completion versus the first bracket-matching error. Different outputs do not justify treating their common bracket task domain as unseen across roles. Keep them together. |
| Lexical ordering | `word_sorting`, `word_sorting_mistake` | The inspected prompts ask for ordering words under an alphabet versus detecting an error in a word-ordering trace. Keep them together conservatively despite different prompt/answer structures. |

This judgment does not claim that the variants use identical neural mechanisms, are exact duplicates, or have measured statistical dependence. It prevents an avoidable overstatement of family-disjoint generalization. Four public development examples are pinned by ID, source revision and canonical-JSON prompt hash in the receipt; no final-evaluation prompt or private grader was selected for semantic review. The metadata inventory still reads the existing corpus to count labels.

`experiments/authz_mechinterp/audit_sae_family_split_blocks_v1.py::validate_role_blocks` rejects related labels assigned to different experimental roles. It also rejects any identical family crossing roles. Unknown labels remain explicitly `unreviewed`, not certified independent. This reusable checker is **not yet integrated into the sealed protocol or existing row planner**.

For a future pack, count the co-assignment block, not each label, as the family-level split unit. Assign an entire block to one role. Resolve how four examples are selected within a multi-label block before using it; do not supply four per variant and silently count the variants as separate families. Changing the row planner's operative family contract requires a separately versioned integration, not editing old manifests.

## Finding: N-Queens seeds do not create fresh problems

The recovered generator draws only board size; it adds no blocked squares or randomized initial placement. Its reference search is deterministic. Its complete declared public problem domains are:

| Difficulty | Board sizes | Distinct public problems |
|---|---|---:|
| medium | 10, 11, 12 | 3 |
| hard | 14, 15, 16 | 3 |
| expert | 18, 19, 20 | 3 |

Thus four distinct examples at one declared difficulty are impossible for this generator, regardless of the number of seeds. There are nine board sizes across difficulties, so this does not rule out four distinct sizes in a separately reviewed mixed-difficulty design. It does rule out calling repeated instances with new IDs independent fresh puzzles. Neither mixed difficulty nor new seeds proves absence of pretraining or prior evaluation exposure.

The audit extracts the literal ranges with Python's AST reader after verifying the exact reviewed source SHA-256 `83dfb8ba2381961e7c38b0f10434155ac049cc52f6a43cf94c3cde1552096174`. It does not import the generator or execute its solver. The interpretation that board size is the only randomized public constraint comes from reading the complete generator source, not merely from extracting that dictionary.

## Other generator source review: implications, not admissions

Read the complete current `cryptarithm.py`, `bin_packing.py`, `scheduling.py`, `hamiltonian.py`, and `nonogram.py` as source text and verified their hashes against `artifacts/vpd_sae_recovery_source_leads_v1/receipt.json`. No source was executed or changed. These observations inform future adapter/duplicate checks:

- Cryptarithm falls back to the fixed `SEND + MORE = MONEY` puzzle after failed construction attempts. A new seed can therefore return the same classic puzzle. A future pack must record and handle that fallback explicitly under its frozen exposure/duplicate policy, not conceal it with a new ID or unreported retries. Letter-renaming equivalence also requires review.
- Hamiltonian graphs contain a planted cycle plus additional edges. Grade public edge/cycle constraints, not equality to the planted cycle. Cycle rotation/reversal or vertex renaming cannot be counted as evidence of an independent problem.
- Nonogram clues are derived from a generated grid, but the task accepts any grid consistent with the clues. Do not leak the generating grid or require that one reference when alternatives satisfy the public task.
- Bin packing shuffles item identifiers after constructing full bins. The task is exact capacity feasibility; duplicate-size items and bin relabelings require invariant checking. Do not repeat the source description's categorical claim that greedy packing cannot work as a demonstrated property of every sampled instance.
- Scheduling exposes fixed machine assignments, durations, precedences, deadlines and a target makespan. Its source constructs a feasible target and a bottleneck-load bound. A text adapter must check those public constraints independently, keeping source start times private; shuffled job IDs are not an independence certificate.

These are task-level observations, not a complete cross-source semantic taxonomy, license review, validator test suite or freshness audit. All six newly read generators remain unadmitted. The earlier graph-coloring development canary remains only a canary.

## Consequence for F1/Q2

The audited corpus cannot fill the inherited 32-block single-pack layout; the conservative two-merge count leaves a gap of at least five before further semantic/exposure exclusions. Even adding all seven nominal recovered leads would only yield a hypothetical 34 blocks, and their independence and usability are not established. That would still be far short of F1's 64 pairwise-disjoint units across two packs.

Do not weaken the held-out separation or subdivide variants merely to make the count pass. Next substantive data work is completing the taxonomy/exposure review and building actual independently validated task supply, or obtaining review of a separately versioned statistical design with an explicitly different generalization claim. No such design change is implemented by this audit. Activation categories remain empirically evaluated recurrent/phase/feature properties; task-family split blocks are controls against leakage, not labels for routing SAE dictionaries.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_sae_family_split_blocks.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.audit_sae_family_split_blocks_v1 --verify artifacts/vpd_sae_family_split_blocks_v1/report.json
```

Seven focused tests pass; the real-corpus/source receipt reproduces exactly. The prompt-hash regression uses the dataset's canonical JSON-string encoding, not raw UTF-8 prompt bytes. This is not a full-project test result.

- Receipt ID: `081799e63214bc743f92be8c90b7aaee8d02d82f5fbc9a9d0c4a56c3b4379621`.
- Receipt SHA-256: `1878eb447b9ee99b76f0f108e53159a60952a0bcb440df3468425c110cfb77b0`.
- Additional admitted families, new captures and new SAE fits: **zero**.

Historical corpus/source receipts and the canonical V52/V44 artifacts remain unchanged. This review adds constraints and evidence; it does not manufacture the missing pack.
