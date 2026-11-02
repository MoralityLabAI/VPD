# SAE pack source search: recovered leads, not new family admissions

Date: 2026-09-11. Initial search was read-only and bounded. No source or recovery file was modified and no repository was fetched. A subsequent local CPU canary executed two reviewed, hash-pinned generator modules, as recorded below; no model, SAE training or container was run.

## Result

A readable recovered research-environments clone contains procedural source leads that could expand the current corpus. They do not yet resolve F1's 64-family requirement or establish a valid single pack. The current audited corpus still has only 29 eligible labels.

Relevant locations inspected:

- `E:\Recovered_C_projects\Spiral_target\projects\prime_intellect_research_environments`
- `E:\Recovered_C_projects\Remote_clones\prime_intellect_research_environments`
- Corresponding `prime_intellect_community_environments` directories (only `.git` was visible at their roots; not counted as a working dataset).
- `E:\Recovered_C_projects\Spiral_target\projects\trm_observability_harness` (sampled data/config inventory, not an exhaustive content audit).
- `E:\Recovered_C_projects\Spiral_target\projects\HRM-re` (sampled artifact inventory).

The older research-environments copy has recognizable directory names but at least two checked files are entirely zero-filled: `AGENTS.md` (1602 bytes) and `environments/logic_env/README.md` (2191 bytes). This establishes damage to those two files, not loss of the whole repository, other research, or all available copies. The separate recovered clone contains readable instructions, documentation and generator sources.

The readable clone reports HEAD `a4c70f7444c6d35ee211bea9dfca82a6c45ad24e`. Git required a command-local `safe.directory` exception because recovery ownership differs; no persistent Git configuration was changed. This is an observed reference, not proof that the checkout is clean or that these files reproduce the user's original May/June source versions. Individual current-file hashes are the evidence used here.

## Source triage

The clone's `environments/reasoning/i3_logic/README.md` points to INTELLECT-3-RL's logic subset, already represented in the audited corpus. Another environment wrapper does not create another set of independent families.

The clone's `environments/reasoning/prolog` contains nine named procedural task generators. Conservatively map `sudoku` to existing `sudoku` and `zebra` to existing `zebra_puzzle`; do not count either again. The remaining seven names absent from the current eligible-label list are:

- `graph_coloring`
- `nqueens`
- `cryptarithm`
- `scheduling`
- `nonogram`
- `bin_packing`
- `hamiltonian`

These are nominal task-type leads, not seven proven independent semantic families. In particular, mathematical/constraint-solving overlap still needs review. Difficulty levels and random seeds are not new families. Adding seven names to 29 would give only a hypothetical upper bound of 36 labels, still short of 64; no such expanded corpus has been built or admitted.

The documented environment expects an agent to edit a Prolog solution inside a container. It is not directly interchangeable with the proposed no-tool textual capture task. A text-only puzzle adaptation would be a new benchmark variant, with separately versioned prompts, answer schemas and validators; it must not be mislabeled as an unchanged official evaluation. Recovered documentation states container requirements, but no container was launched.

The harness's readable `configs/reasoning_gym.yaml` is another lead: it references leg-counting, GSM8K and Wordle smoke tasks. A config reference is not a verified installed generator or fresh corpus. No additional family count was assigned from it.

## Durable evidence and next checks

### Graph-coloring text adapter: implemented development component

Reviewed `generators/graph_coloring.py`, its `base.py`, package imports, and the graph-coloring verifier function as source text. The generator and base files are bound by the existing receipt; the complete verifier file's observed SHA-256 is `884019c22280cb9e788cc3dcd76aa441de1cde963dfdb7c70daad9e3b74df14d`. No recovered package was imported at that initial checkpoint. The later canary executes only the two reviewed modules, not the package initializer: the initializer imports every generator, exceeding this bounded review.

The generator's `chromatic_number` metadata is a greedy-derived color budget with a reference feasible assignment, not a proof of the graph's minimum chromatic number. Its actual public description asks for *at most* that many colors. The adapter must retain that feasibility task rather than accidentally turning it into an optimal-coloring claim.

New workspace implementation: `experiments/authz_mechinterp/graph_coloring_text_adapter_v1.py`. This is a new text-only interface and independent JSON constraint checker, not recovered historical code or the unchanged official Prolog evaluation. It takes reviewed-shape metadata, exposes only vertex count, permitted color bound and undirected edges, and returns separate question and private-grader objects. It validates the reference against public constraints before packaging, but grades candidate answers by constraints rather than equality to that reference. Any valid coloring using fewer than the maximum colors is accepted.

The interface uses one JSON color array indexed by vertex order. It executes no Prolog, subprocess, model, or container. Edge order and orientation are normalized for the public-instance hash; this does **not** canonicalize graph isomorphisms or prove fresh instances. The current implementation accepts up to 128 vertices, exceeding the source's documented 10–40-node range but not establishing performance or representativeness beyond the tests.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_graph_coloring_text_adapter.py
```

Twenty-four adapter tests pass, including exhaustive checks of all assignments to a three-vertex path/triangle, alternative-reference invariance, invalid-reference rejection, answer-schema checks and public-field whitelisting. These tests use small hand-built fixtures; actual generator checks are reported separately below. Broad generator correctness, text-task quality, licensing/provenance completion, cross-source semantic aliases and historical exposure remain unresolved. Additional families admitted remains zero.

Receipt: `artifacts/vpd_sae_recovery_source_leads_v1/receipt.json`. Fifteen file hashes were verified: eleven nonzero generator/package files, two zero-filled probes, and two readable documentation files. This is a provenance receipt, not an empirical SAE result or a passing generator test suite.

### Completed CPU generator canary

`experiments/authz_mechinterp/check_graph_coloring_generator_canary_v1.py` generated and independently re-generated four medium-difficulty development puzzles. The worker checks both source hashes before executing either file, loads only those exact bytes under a private module namespace, and never runs the recovered package initializer. Its subprocess has a 20-second timeout; this is a local worker bound, not a provider billing safeguard.

| Seed | Vertices | Edges | Maximum colors |
|---|---:|---:|---:|
| 0 | 13 | 20 | 3 |
| 1 | 11 | 11 | 3 |
| 2 | 10 | 9 | 2 |
| 3 | 11 | 14 | 3 |

All four reproduce exactly, accept both the source reference and a cyclic color permutation, and reject a deliberately conflicting edge assignment. Public questions and private references are stored separately in `artifacts/vpd_graph_coloring_generator_canary_v1/questions.json` and `graders.json`. Their IDs and roles explicitly mark them development-only. These checks do not certify a fresh holdout, broad difficulty coverage, semantic independence or new family admission.

```powershell
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.check_graph_coloring_generator_canary_v1 --verify artifacts/vpd_graph_coloring_generator_canary_v1/report.json
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_graph_coloring_generator_canary.py tests/test_vpd_graph_coloring_text_adapter.py
```

Receipt reproduction passed on 2026-09-11. Report ID: `2dce8a3113493ebfcf86faf886cfc9890af59a40be93652908e9dd3b9f7f9631`; file SHA-256: `d476f1f874423e75d9a34321a963c7c7cc4e1147c68c43a30bb654d5d0b46534`. The canary and adapter have 26 focused tests; the combined eleven-file local audit/planner/adapter suite passed 135 tests in 1.09 seconds. This is not a claim that the entire project test suite passes.

The initial source-lead receipt remains unchanged: its no-generation boundary describes the earlier search checkpoint. The separate canary receipt records actual local puzzle generation. No model inference, weight fitting, paid compute launch or additional family admission occurred.

Before incorporating any procedural source:

1. Review the generator and verifier code, dependencies and license obligations without executing the tool-using environment.
2. Establish question-only rendering and independent answer checking; prevent planted solutions or grader metadata from entering prompts.
3. Extend deterministic puzzle and validator checks only after reviewing the relevant source. The four medium-seed graph-coloring canary is complete; other difficulties and other generators are not covered.
4. Define semantic family/alias groups across existing and new sources, and keep related variants together across roles.
5. Check prior exposure and duplicates; freeze actual role assignments and a new pack manifest before inference. New random seeds alone do not prove uncontaminated or independent families.

Additional families admitted: **zero**. F1/Q2 remain design-stage. The source search is not exhaustive across E: or all recovered repositories.

### Subsequent semantic/public-instance review

`VPD_SAE_FAMILY_SPLIT_REVIEW_2026-09-11.md` adds a bounded review of the six remaining nominal generator sources and verifies their existing hashes without executing them. N-Queens has only three public board-size instances per difficulty; cryptarithm can fall back to a fixed classic puzzle. Neither new seeds nor new IDs establish new public instances. Two conservative co-assignment rules for existing bracket and word-ordering variants reduce the 29-label inventory to at most 27 split blocks, with 25 still unreviewed. The receipt at `artifacts/vpd_sae_family_split_blocks_v1/report.json` reproduces; no new family or fresh pack is admitted.

### Nonogram text adapter and actual development generation

The next checkpoint implements `experiments/authz_mechinterp/nonogram_text_adapter_v1.py` and runs `check_nonogram_generator_canary_v1.py`. These are new workspace components, not recovered historical code or the official Prolog evaluation. The adapter exposes only dimensions and ordered row/column run clues. It keeps the source generating grid in a separate private grader object and accepts any binary grid satisfying the public clues, not just an exact reference match. Source difficulty/density, references and starter code are not copied into the question prompt.

The independent checker uses grouped runs and validates dimensions, positive clue lengths, separating gaps, filled-cell totals, exact grid shape and integer binary cells. Empty clues are empty lists, not zero-length runs. The text answer must be bounded JSON; no answer code, Prolog or container is executed. It supports dimensions up to the source's 16-by-16 maximum, but this bound is not a claim of measured task difficulty or model success.

Twenty focused tests pass. In particular, all 64 binary 2-by-3 grids were checked against all 64 corresponding clue sets (4,096 comparisons) using an independently implemented regular-expression clue oracle. A two-solution 2-by-2 puzzle confirms acceptance of alternative valid grids and public-question invariance to which reference is stored. Other tests check private-field exclusion, invalid references and answers, malformed clues, and rejection of altered recovered sources before any execution.

After rereading the complete `base.py` and `nonogram.py`, the bounded local worker executed only their hash-pinned bytes under a private module namespace. The recovered package initializer was not run and no dependency was installed. Its 20-second subprocess timeout bounds this local puzzle-generation worker only; it is not a provider billing control.

Actual development generation produced twelve distinct exact public-clue hashes: seeds 0–3 at each of medium, hard and expert. Every puzzle was independently re-generated with its seed, its source reference accepted, and a single-cell mutation rejected. A subsequent receipt verification independently regenerated and matched the saved questions and private graders exactly.

| Difficulty | Seed 0 size | Seed 1 size | Seed 2 size | Seed 3 size |
|---|---|---|---|---|
| medium | 9×9 | 8×10 | 8×8 | 8×10 |
| hard | 12×12 | 11×13 | 11×11 | 11×13 |
| expert | 15×15 | 14×16 | 14×14 | 14×16 |

These are **development-only** puzzles, not fresh final holdout examples. Distinct clue hashes do not exclude reflection/rotation/transposition equivalents, source/model exposure, or semantic overlap with existing grid/constraint tasks. Three difficulty settings are one nominal task type, not three independent families. Uniqueness of solutions and the source's claims about required search difficulty are not certified.

Artifacts are in `artifacts/vpd_nonogram_generator_canary_v1`:

- `report.json`: ID `350a7fc0a8e8a04781d71fdc2e9a067e36a21b6f3607250a3c596b0f17984722`; SHA-256 `b576f4c4bfee3d59f83668c1c1d4b15f59185a2812e37677578e4e2a7cfdc6a6`.
- `questions.json`: public constraints/prompts only; SHA-256 `79f75f6be117fe4f08c5a712601822dc29666cc0fe308381c9730d3fc565fbbb`.
- `graders.json`: separate private source reference grids; SHA-256 `3cc29ef60e6d0e48eb88952a309b760c3cd9978a97e97527839228723440791b`.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_nonogram_text_adapter.py tests/test_vpd_nonogram_generator_canary.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.check_nonogram_generator_canary_v1 --verify artifacts/vpd_nonogram_generator_canary_v1/report.json
```

The recovered clone's complete root `LICENSE` was read: it contains Apache-2.0 text, a Prime Intellect attribution and a third-party attribution section. Observed file SHA-256 is `fbbf4f7fe6a814ff4245e05f27ef7950fce50e1bf6ad6c1e2a77c8c7f95cd2e0`. This is source-license evidence, not a completed redistribution review or an assignment of that license to the entire VPD workspace. No recovered source file was copied, changed or distributed by this checkpoint; code was executed from its checked local bytes, and the text adapter/checker was written separately.

The nonogram source now has a working text interface, independent constraint checker and a reproducible cross-difficulty development canary. Symmetry/duplicate checking, semantic role grouping, exposure review and actual held-out pack construction remain necessary before it can contribute to Q2. Additional families admitted remains **zero**. No model inference, SAE fitting, paid compute or canonical-protocol change occurred.

### Nonogram public-clue symmetry audit

The new `audit_nonogram_symmetry_duplicates_v1.py` closes the specific rotation/reflection/transposition duplicate check for the twelve saved development puzzles. It transforms only public ordered row/column clues: transposition swaps their roles and dimensions; reversing row order reverses column run order; reversing column order reverses row run order. The eight compositions define a canonical public-clue representation, retaining dimensions. It does not use a private generating grid, solve the puzzle, regenerate source examples or change the original adapter/canary artifacts.

**Result: twelve questions, twelve distinct symmetry classes, zero duplicate classes under these eight transformations.** This strengthens the exact-clue duplicate check; it does not establish non-equivalence under every conceivable task transformation, statistical independence, prior-exposure absence, model difficulty or a new family admission. In particular, these remain exposed development puzzles and three difficulty levels of one nominal task type.

Six new tests and the eighteen adapter tests pass together (24 focused tests). The transformation rule was checked against independently transformed grids and regular-expression-derived clues for every binary 2×3 and 3×3 grid under all eight orientations: 4,608 grid/transformation cases. Canonical-class invariance is checked in each case. A deliberately transposed duplicate with a different exact public hash is detected, while an unrelated puzzle remains separate; input order does not change the audit. Additional tests cover symmetric orbits, duplicate IDs, altered clue fingerprints and invalid transformation flags.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_vpd_nonogram_symmetry_duplicates.py tests/test_vpd_nonogram_text_adapter.py
.\.venv\Scripts\python.exe -m experiments.authz_mechinterp.audit_nonogram_symmetry_duplicates_v1 --verify artifacts/vpd_nonogram_symmetry_duplicates_v1/report.json
```

The receipt at `artifacts/vpd_nonogram_symmetry_duplicates_v1/report.json` reproduces from saved public questions and binds the original canary receipt/question hashes. Report ID: `f4995b148af45386cd5bc4874eb6c5c28b86ee2311e3a2fbfd9895051d150dbc`. The original twelve-puzzle canary and its hashes remain unchanged.

The canonical class key is a reusable instance-level grouping component for a future reviewed pack. It has not been integrated into a real pack builder or exposure index. Broader semantic grouping, historical overlap checks, frozen role allocation and actual capture eligibility remain unresolved; no missing activation rows or SAE quality improvements were produced by this audit.
