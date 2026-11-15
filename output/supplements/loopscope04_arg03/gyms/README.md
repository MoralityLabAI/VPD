# Local Dark Fibers gyms: ARG-03-N and LOOPSCOPE-04

**Status (November 15, 2026):** Local mechanistic-interpretability campaign infrastructure and reproducible runners for `ARG-03-N` (`Nanbeige4.2-3B`, `development.jsonl` $N=96$; held-out `test.jsonl` $N=168$ strictly sealed) and `LOOPSCOPE-04` (`gaoxin492` primary and `yagizdevre` replication 7M Sudoku `TRM`s). Prior `ARG-02` receipts and the `v014` manuscript ([`PAPER.md`](../../papers/shared_weights_dark_fibers/PAPER.md)) are unchanged. Mathematical rationale: [`CAUSAL_QUOTIENT_ARG03.md`](../../papers/shared_weights_dark_fibers/CAUSAL_QUOTIENT_ARG03.md); empirical campaign synthesis: [`DARK_FIBERS_CAMPAIGN_ADDENDUM.md`](../../papers/shared_weights_dark_fibers/DARK_FIBERS_CAMPAIGN_ADDENDUM.md); frozen held-out test protocol: [`ARG03_FROZEN_TEST_PROTOCOL.md`](../../papers/shared_weights_dark_fibers/ARG03_FROZEN_TEST_PROTOCOL.md).

## Models, resources, Python

| Gym / Module | Compute | Local hardware |
| --- | --- | --- |
| `arg03_roster`: fresh algebraic public split, isolated `EvalMath` grader, and dev-only calibration fit | CPU | Any |
| `arg03_nanbeige`: 16 candidate rows, native `Nanbeige4.2-3B` loops `1/2/4/8`, context-fiber swaps, norm-matched random/self controls, equal-compute & temperature grids | BF16 | RTX 5080 16 GiB (`--min-free-gib 9.0`) |
| `sudoku_roster`: solver-certified unique-solution `pilot` (`permuted` or MRV `diverse`) and stratified `representative` splits with full Sudoku automorphism screening | CPU | Any |
| `loopscope04_trm` (`swap` / `gramian`): 7M Sudoku `TRM`, FP32 sphere-preserving `z_H_orth`, `z_L`-only, and joint `z_H_orth + z_L` swaps; batched nonlinear Gramian directional validation | GPU | RTX 3050 4 GiB or RTX 5080 (`<= 1.47 GiB` peak VRAM) |
| `loopscope04_trm` (`cache`): matched-width cellwise ridge + capacity-matched 97-token `SeqRefill97` Transformer (`C^now` vs. `C^trace`) with `Resume` and `m=1, 2` rollouts | GPU + CPU | RTX 3050 + RAM (`<= 0.48 GiB` peak VRAM) |
| `campaign_summary` & `plot_campaign`: deterministic paired bootstrap / McNemar / sign / Wilcoxon aggregator and vector+raster figure builder | CPU | Any |

Run every script as `python -m mechinterp.dark_fiber_gyms.MODULE` from the `LoopedTransformers` repository root. Required local checkouts:

- `LoopedTransformers` at `research/causal-dark-fibers-arg03-20261115`, PR #47.
- `Compactification-Theory` at `research/arg02-recurrent-qwen-20261110`, PR #141, retaining `checkpoint_lock.json` and the pinned `Nanbeige` model adapter.
- `EvalMath` providing `src/spiral_ln/algebraic_reasoning_oracle.py` for `arg03_roster prepare` and `grade` (never imported by `arg03_nanbeige`).

The existing Windows environments from `ARG-02` are used directly: `$HOME/.venvs/bvg-v03/Scripts/python.exe` (`transformers 4.46.3`, `torch 2.11.0+cu128`, `numpy 2.5.4`) for `Nanbeige4.2-3B`, and `$HOME/.venvs/bluebeam/Scripts/python.exe` (`torch 2.11.0+cu128`, `numpy 2.5.3`, `matplotlib 3.11.2`) for `TRM`, CPU regression tests, statistical aggregation, and plotting.

## Smoke (no model weights)

PowerShell:

~~~powershell
$P = "$HOME/.venvs/bluebeam/Scripts/python.exe"
& $P -m pytest -v mechinterp/dark_fiber_gyms/tests
& $P -m compileall -q mechinterp/dark_fiber_gyms
~~~

## ARG-03-N: new algebraic roster, model-only run, independent grade, and dev calibration

~~~powershell
$E = "C:/path/to/EvalMath"
$C = "C:/path/to/Compactification-Theory"
$R = "runs/dark-fibers/arg03"
$N = "$HOME/.venvs/bvg-v03/Scripts/python.exe"

# 1. Export leakage-audited public splits (development.jsonl: 96, test.jsonl: 168)
& $P -m mechinterp.dark_fiber_gyms.arg03_roster prepare --evalmath-root $E --compactification-root $C --out-dir "$R/public"

# 2. 4-case GPU smoke (verifies >= 9.0 GiB free VRAM and SHA-256 checkpoint lock)
& $N -m mechinterp.dark_fiber_gyms.arg03_nanbeige --public-jsonl "$R/public/development.jsonl" --compactification-root $C --out-dir "$R/dev_smoke4" --device cuda:0 --limit 4 --min-free-gib 9.0
& $N -m mechinterp.dark_fiber_gyms.arg03_roster grade --evalmath-root $E --compactification-root $C --split-dir "$R/public" --split development --tape "$R/dev_smoke4/answer_tape.jsonl" --out "$R/dev_smoke4/grade_smoke4.json"

# 3. Full 96-case development evaluation + dev-only temperature/operator calibration fit
& $N -m mechinterp.dark_fiber_gyms.arg03_nanbeige --public-jsonl "$R/public/development.jsonl" --compactification-root $C --out-dir "$R/dev_full96" --device cuda:0 --limit 0 --min-free-gib 9.0
& $N -m mechinterp.dark_fiber_gyms.arg03_roster grade --evalmath-root $E --compactification-root $C --split-dir "$R/public" --split development --full --tape "$R/dev_full96/answer_tape.jsonl" --out "$R/dev_full96/grade_dev96.json" --fit-dev-calibration "$R/dev_full96/dev_calibration.json"
~~~

- **Split isolation:** `development.jsonl` has 96 new positive examples across the 3 calibration generator families (`final operator != R`). `test.jsonl` has 168 positives across all 7 families (`final operator == R`). All prior `EvalMath` and `ARG-02` signatures are forbidden. `arg03_nanbeige` calls `assert_oracle_isolated()` and sees only the 6 public fields.
- **Arms & diagnostics:** At loop 1, `arg03_nanbeige` constructs a same-prompt perturbed donor ($\sigma=0.02$), replaces the `width=8` context positions immediately preceding the 3 scored tail tokens (`context_swap`), constructs an exact per-row/per-position $L_2$-norm-matched Gaussian context perturbation (`random_context_norm_matched`) and a bit-exact `self_context_control`, and asserts pre-continuation full-vocabulary log-probability parity ($\le 10^{-6}$) before continuing through loops `2, 4, 8`. It also evaluates `open_loop` linear extrapolation, `equal_compute` prompt-variant single-pass ensembles (`canonical`, `swap`, `space`, `swap_space`), a 10-point scalar temperature grid (`T0.25..T4.0`), `[16, 3, 4]` raw operator-site logits, hidden-state perturbation $L_2$ norms, delayed tail-readout $L_2$ divergence, and 64-program ranking stability (`top1_changed`, `top12_overlap`, `spearman_rho`, `kl_a_to_b`).
- **Held-out seal:** **Do not run `test.jsonl` until development calibrations and decision rules are frozen and authorized** ([`ARG03_FROZEN_TEST_PROTOCOL.md`](../../papers/shared_weights_dark_fibers/ARG03_FROZEN_TEST_PROTOCOL.md)). `--fit-dev-calibration` raises `RuntimeError` if `--split != development`; frozen parameters are applied via `--apply-dev-calibration`.

## LOOPSCOPE-04: automorphism-disjoint Sudoku splits and causal TRM probes

`sudoku_roster` supports both independent MRV-synthesized `pilot` splits (`--mode pilot --generator diverse`) and solver-certified, deductively stratified `representative` splits (`--mode representative --source-csv <path> --start-row 5000`), screening every puzzle against all `--exclude-csv` files (`.csv` and `.json`) under exact, canonical digit-relabel, **and full Sudoku grid/transpose/digit automorphisms** (`sudoku_wl_signature` + `are_sudoku_isomorphic`, $2 \times 6^8 \times 9!$ group).

~~~powershell
$S = "runs/dark-fibers/loopscope04"
$OldBench = "C:/path/to/test_sudoku_bench.csv"
$OldItems = "$S/exclusions/sudoku_items256.json"
$OldTrain = "C:/path/to/train.csv"
$HardSrc  = "C:/path/to/test_hard.csv"
$TRM_P    = "C:/path/to/gaoxin492/step_21700"
$TRM_R    = "C:/path/to/yagizdevre/model.pt"

# 1. Stratified representative split (360 train, 48 dev, 120 eval: 40 WB-Prop + 40 WB-Stall_partial + 40 S4_stalled)
& $P -m mechinterp.dark_fiber_gyms.sudoku_roster --mode representative --source-csv $HardSrc --start-row 5000 --out-dir "$S/rep_public" --exclude-csv $OldBench --exclude-csv $OldItems --exclude-csv $OldTrain --n-train 360 --n-dev 48 --n-eval 120 --seed 20261115

# 2. 3A: Sphere-calibrated z_H_orth, z_L-only, and joint (z_H_orth + z_L) swaps (t in {2, 4, 8}, horizon 32)
& $P -m mechinterp.dark_fiber_gyms.loopscope04_trm --repo-root . --checkpoint-file $TRM_P --tag gaoxin492 --mode swap --eval-csv "$S/rep_public/eval.csv" --exclude-csv $OldBench --exclude-csv $OldItems --exclude-csv $OldTrain --out "$S/results/gaoxin492_rep_swap_t4_h32.json" --device cuda:0 --limit 0 --patch-step 4 --horizon 32

# 3. 3B: Cellwise ridge + capacity-matched 97-token SeqRefill97 (C^now vs. C^trace) + Resume & m=1, 2 rollouts
& $P -m mechinterp.dark_fiber_gyms.loopscope04_trm --repo-root . --checkpoint-file $TRM_P --tag gaoxin492 --mode cache --train-csv "$S/rep_public/train.csv" --eval-csv "$S/rep_public/eval.csv" --exclude-csv $OldBench --exclude-csv $OldItems --exclude-csv $OldTrain --out "$S/results/gaoxin492_rep_cache_full97_n360.json" --device cuda:0 --train-limit 360 --limit 0 --steps 16 --horizon 32 --full-97-refill

# 4. 3C: Sampled delayed observability Gramian G_{t,k} + finite-angle nonlinear directional validation
& $P -m mechinterp.dark_fiber_gyms.loopscope04_trm --repo-root . --checkpoint-file $TRM_P --tag gaoxin492 --mode gramian --eval-csv "$S/rep_public/eval.csv" --exclude-csv $OldBench --exclude-csv $OldItems --exclude-csv $OldTrain --out "$S/results/gaoxin492_rep_gramian_t4_p8_k4_h12.json" --device cuda:0 --limit 0 --patch-step 4 --probes 8 --angle 0.04 --gramian-horizon 4 --eval-horizon 12
~~~

Repeat with `--checkpoint-file $TRM_R --tag yagizdevre` for the independent replication checkpoint.

## Deterministic Campaign Summary & Publication Figures

~~~powershell
& $P -m mechinterp.dark_fiber_gyms.campaign_summary --repo-root . --out papers/shared_weights_dark_fibers/campaign_20261115/campaign_summary.json
& $P -m mechinterp.dark_fiber_gyms.plot_campaign --summary papers/shared_weights_dark_fibers/campaign_20261115/campaign_summary.json
~~~

## Safety / evidence

All runners enforce `local_files_only` SHA-256 checkpoint verification, never download weights or spend money, and keep `EvalMath` oracle data strictly isolated from `arg03_nanbeige`. Keep generated puzzle CSVs, raw per-case JSONL tapes, and model weights under `runs/` outside version control; commit only code, tests, documentation, figures, and compact provenance-locked summary JSONs.
