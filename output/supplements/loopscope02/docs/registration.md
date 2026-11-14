# LOOPSCOPE-02: Triangulating Looped Virtual Depth vs. Recursive Attractor Search on Stratified 9×9 Sudoku

Registered 2026-11-13 on branch `main`, prior to evaluating any held-out Sudoku test board across any model.

## 1. Motivation and Literature Boundary

Five looped-transformer interpretability papers from the preceding ten weeks establish important pieces of how weight-tied models compute, while leaving a central capability question unresolved:

1. **Wu, Hu, and Zhang (`arXiv:2609.39892`, Sept 2026)** show via Jacobian steering (`J-steering`) that shared weights route to distinct linear transformations across loop iterations. Consequently, showing that `Nanbeige4.2-3B` activates nearly orthogonal SVD rank-one mixtures on Pass 1 vs. Pass 2 is necessary context, not a standalone contribution.
2. **Sesodia et al. (`arXiv:2610.24135`, NeurIPS 2026)** characterize iteration-budget difficulty staircases across recurrent depth, without testing whether the recurrent map commutes with a ground-truth symbolic propagation operator or how post-stall conflicts are repaired.
3. **Olszowiec et al. (`arXiv:2610.18127`, Oct 2026)** track representation refinement across loops without open-loop or donor-displacement counterfactual falsification.
4. **Kaissis et al. (`SDI`, ICML 2026)** use per-step outer products $\phi_t = \sum \delta_t \otimes a_t$ for training-data influence, without closed-form orthogonal rotation nulls or attractor-repair masking analysis.
5. **Blayney et al. (2026)** analyze loop-depth representations without contrasting fixed-pass looped language models against convergent Tiny Recursive Models (`TRM`) on a shared, deductively stratified constraint-satisfaction benchmark.

**Core question.** Can isolating Pass-1 vs. Pass-2 parameter-activation changes in a 2-pass looped 3B language model (`Nanbeige4.2-3B`) explain a capability gain over an untied 4B model (`Qwen3.5-4B-Base`, plus pure-SDPA `Qwen3-4B-Instruct-2507` sensitivity arm) on combinatorial constraint satisfaction, and what mechanistically separates both models from a 7M-parameter Attention-TRM (`TinyRecursiveReasoningModel_ACTV1`) that unrolls a 2-layer transformer across 16 outer ACT steps (`3 H-cycles × 6 L-cycles = 336` passes) on the exact same 9×9 Sudoku boards?

**Blindness audit.** Before committing this registration:
- Two rows of `train.csv` were used solely to verify `PureSudokuTRM` tensor chunking (`flat3` QKV, `gate_first` SwiGLU, Post-RMSNorm, RoPE) against `step_21700`.
- Cached tokenizers were checked on a single synthetic board derived from a textbook solved grid to verify that ending the 3-shot prompt with `"Answer: "` makes digits `'1'..'9'` exact single next tokens across all three LLM tokenizers.
- Zero rows of `test_sudoku_bench.csv` or `test_hard.csv` have been passed through `TRM`, `Nanbeige4.2-3B`, `Qwen3.5-4B-Base`, or `Qwen3-4B-Instruct-2507`.

---

## 2. Pinned Artifacts and SHA-256 Locks

### 2.1 Dataset (`sapientinc/sudoku-extreme-1k`)
- `train.csv`: `887dea10daba811d3a6a565bb138e31ef60a95fdb57045633e4762fc46357cbb` (3 few-shot demonstrations only)
- `test_sudoku_bench.csv`: `0bd94e4843c3ec55e225c8798bf3f63083b2b2207e88ca996c3bbb4b8b1c18e1` (100 canonical boards: 49 propagation-solvable, 51 stalled)
- `test_hard.csv`: `5f9549fb2f38e2cbcb9a515733fdfb19d10c599f21d5c9e82ec20bca28e78c2c` (first 1,500 rows pooled with `test_sudoku_bench.csv` for stratified distinct-board sampling)

### 2.2 Frozen Checkpoints
1. **Primary TRM (7M Attention-TRM):** `gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att`, `step_21700` (`sha256: 932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3`)
2. **Replication TRM (7M Attention-TRM):** `yagizdevre/trm-sudoku-extreme-att`, `model.pt` (`sha256: d67723394a7fe439f1fa498b58196399ee89cd82a794c7da9ce0a8cc3df7fa39`)
3. **Looped 3B LM:** `Nanbeige/Nanbeige4.2-3B` @ `451ed48c3273ecef7ea8faaa43c31ce529763bb1` (`config.json` `f6cb15b22847664f3a6049dc4b58fdd10f1650d112ac99a1da3d051f17c2ca19`; `model.safetensors.index.json` `30d8da0fa8b97abc6d9eddfd017a0cb7a649bcbd58caa57804c56c767db5c0f1`)
4. **Prosaic 4B Base LM:** `Qwen/Qwen3.5-4B-Base` @ `1001bb4d826a52d1f399e183466143f4da7b741b` (`config.json` `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`; `model.safetensors.index.json` `eae340074abb0a5f31a6621f7ae8e8248a7c1790df04a722c4e4b70c2a6d1dbb`)
5. **Pure-SDPA 4B Sensitivity Arm:** `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554` (`config.json` `5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba`; `model.safetensors.index.json` `d6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c`)

---

## 3. Deductive Stratification and Instruments

### 3.1 Exact Synchronous Propagation Lattice $G_{\text{prop}}$ (`sudoku_prop.py`)
Let $G_{\text{prop}}$ denote one synchronous round of naked-single plus hidden-single elimination across all 27 Sudoku units (9 rows, 9 columns, 9 boxes). Iterating $G_{\text{prop}}$ from initial clues assigns every empty cell $i$ an exact deductive round $r(i) \in \{1, 2, \dots, R\}$ or $r(i) = -1$ (stalled). With seed `20261113`, `build_benchmark_bundle()` samples **256 distinct boards** (`64` per stratum, one target cell `?` per board):
- **`S1_naked1` (`n=64`):** Empty cell with 8 distinct given peer digits at round 0 ($r(i) = 1$, immediate local elimination).
- **`S2_prop2` (`n=64`):** Empty cell deduced at synchronous round $r(i) = 2$.
- **`S3_prop3plus` (`n=64`):** Empty cell deduced at synchronous round $r(i) \ge 3$.
- **`S4_stalled` (`n=64`):** Empty cell on a strict zero-single board where $G_{\text{prop}}$ makes zero progress at round 0 (`num_rounds = 0`, backtracking/search required).

Each item $i$ is paired with a deterministic within-stratum donor item $d(i)$ whose ground-truth target digit satisfies $y_{d(i)}^* \neq y_i^*$, guaranteeing donor-state swaps cannot leak the target answer. All LLMs receive the identical 3-shot prompt (`1 naked1`, `1 prop2`, `1 prop3` from `train.csv`) ending with `"Cell ? (RrCc) Answer: "` and are scored over the 9 single-token digit logits `'1'..'9'`.

### 3.2 Causal Counterfactual Suite (`ARG-02` + Compactification Fiber Erasure)
1. **Open-loop extrapolation (`H2a`):**
   - `Nanbeige`: $h_{\text{open}}(t) = h(1) + (t - 1)(h(1) - x)$ at $t \in \{2, 8\}$.
   - `TRM`: $z_{H,\text{open}}(16) = z_H(1) + 15(z_H(1) - H_{\text{init}})$.
2. **Donor recurrent displacement shuffle (`H2b`):**
   - `Nanbeige`: $h_{\text{shuf},i}(t) = h_i(1) + (h_{d(i)}(t) - h_{d(i)}(1))$ on the right-aligned answer tail.
   - `TRM`: $z_{H,\text{shuf},i}(16) = z_{H,i}(1) + (z_{H,d(i)}(16) - z_{H,d(i)}(1))$.
3. **Perturbation repair vs. amplification:**
   - `TRM`: patch 10 random empty cells of $z_H(1)$ with donor states $z_{H,d(i)}(1)$ and unroll steps $2..16$; also measure $0.25 \cdot \text{RMS}(z_H(1))$ Gaussian kick norm ratio $\|z_H^{\text{kick}}(s) - z_H^{\text{clean}}(s)\|_2 / \|\delta_1\|_2$.
   - `Nanbeige`: patch the last 5 answer-site tokens of $h(1)$ with donor $h_{d(i)}(1)$ before Pass 2.
4. **Dark-fiber vs. scratchpad state erasure (`TRM`):**
   - `reset_zL`: reset $z_L \leftarrow L_{\text{init}}$ between every outer ACT step while retaining full 512-dim $z_H$.
   - `fiber_erased`: reset $z_L \leftarrow L_{\text{init}}$ **and** project the 81 board-cell vectors of $z_H$ onto the 11-dimensional row space $P_{\text{head}}$ of `lm_head.weight` ($z_H^{\text{board}} \leftarrow z_H^{\text{board}} P_{\text{head}} + H_{\text{init}}(I - P_{\text{head}})$) between outer ACT steps. This preserves step-$s$ digit logits identically while erasing the 501-dimensional orthogonal fiber of $z_H$.

### 3.3 Weight-Usage Geometry, Unrolled `TCA`, and `Q11` Compactification
- **SVD rank-one usage (`top-64` singular triplets):** Input activation magnitude $|(X v_r)| \sigma_r$ on `Nanbeige` physical layers `(6, 12, 17, 22)` `mlp.down_proj` (Pass 1 vs. Pass 2 Pearson $r_{\text{pass1,pass2}}$) and on all 4 `TRM` `L_level` output matrices (`L0.o_proj`, `L0.down_proj`, `L1.o_proj`, `L1.down_proj`), separating pairwise outer-step correlation $r_{\text{outer}}$ (`s = 2..16`) from within-step scratchpad-vs-commit role correlation $r_{\text{role}}$ (`z_L` vs. `z_H`).
- **Transported Component Attribution (`TCA`):** Log-odds target-digit margin backpropagated through unrolled steps (`s = 1..6` in float32 for `TRM` on `S1_naked1 + S3_prop3plus`; Pass 1–2 for `Nanbeige` and single pass for `Qwen` on `S1_naked1`), yielding fragility $F_s = \|X_s^\top A_s\|_F$ and closed-form orthogonal rotation z-scores $z_{\text{read}}, z_{\text{write}}$.
- **`Q11` $G_{\text{prop}}$ commutation & deductive staircase:** On `bench100` and `items256`, measure commutation error $\rho_s(b) = \text{mean}_{i \in \text{empty}} \|p_{s+1}(b, i) - G_{\text{prop}}(p_s(b), \text{givens}_b)_i\|_2$ against deranged-board $\text{givens}_{b'}$, within-board deductive-round convergence concordance $C_{\text{stair}}$ across empty-cell pairs $1 \le r(i) < r(j)$ across all 48 $H$-cycles (vs. within-board cell-permutation null), and post-stall non-monotone cell-prediction flips.

---

## 4. Pre-Registered Hypotheses and Decision Rules (`score_loopscope02.py`)

All paired binary comparisons use exact two-sided binomial McNemar tests and 2,000-resample paired bootstrap 95% CIs (`seed = 20261113`).

### P1 — Capability Triangulation Across Model Classes
- **Supported** if all hold:
  1. `Nanbeige` fails out-of-distribution loop scaling: $\text{acc}_{\text{NB}}(t=8) - \text{acc}_{\text{NB}}(t=2) \le 0.02$.
  2. `Nanbeige` (`t=2`) fails to exceed `Qwen3.5-4B-Base` by $\ge 0.10$ on multi-step strata (`S2_prop2` $\cup$ `S3_prop3plus` $\cup$ `S4_stalled`). *(Note: if `Nanbeige` exceeds `Qwen3.5-4B-Base` by $\ge +0.10$ with McNemar $p < 0.05$, `P1` is falsified in the positive direction of genuine 2-pass LLM multi-step gain.)*
  3. `TRM` iterative scaling: $\text{acc}_{\text{TRM}}(s=16) - \text{acc}_{\text{TRM}}(s=1) \ge 0.25$.
  4. `TRM` (`s=16`) exceeds $\max(\text{acc}_{\text{NB}}(t=2), \text{acc}_{\text{Qwen4B}})$ by $\ge 0.35$ with exact McNemar $p < 10^{-6}$.
- **Weak** if $\text{acc}_{\text{TRM}}(s=16) - \text{acc}_{\text{TRM}}(s=1) \ge 0.15$ and margin over best LLM $\ge 0.25$.
- **Not supported** otherwise.

### P2 — `ARG-02` Causal Counterfactuals: Input-Conditioned Attractor vs. Virtual Depth
- **Supported** if all hold:
  1. `TRM` open-loop and donor-$z_H$ displacement both collapse accuracy: $\text{acc}_{\text{clean16}} - \text{acc}_{\text{open16}} \ge 0.30$ ($p < 10^{-6}$) and $\text{acc}_{\text{clean16}} - \text{acc}_{\text{donor16}} \ge 0.30$ ($p < 10^{-6}$).
  2. `TRM` actively repairs step-1 perturbations: board or target corruption recovery $\ge +0.15$, or final Gaussian kick norm ratio $< 1.0$.
  3. `Nanbeige` is non-contractive across loops: mean relative step norm at $t=8$ is $\ge 0.10$ with $\text{acc}_{\text{NB}}(t=8) - \text{acc}_{\text{NB}}(t=2) \le 0.02$.
- **Weak** if `TRM` counterfactual drops are in $[0.20, 0.30)$ with repair holding.
- **Not supported** otherwise.

### P3 — SVD Weight-Usage Geometry: Cross-Pass Multiplexing vs. Stationary Outer Operator + Role Split
- **Supported** if all hold:
  1. `Nanbeige` Pass-1 vs. Pass-2 component usage is nearly orthogonal: $\bar{r}_{\text{pass1,pass2}} < 0.30$.
  2. `TRM` reuses a stationary outer operator across steps `2..16`: $\bar{r}_{\text{outer},z_H} \ge 0.80$.
  3. `TRM` multiplexes scratchpad vs. board-commit roles within each cycle: $\bar{r}_{\text{role}}(z_L, z_H) \le \bar{r}_{\text{outer},z_H} - 0.10$.
- **Weak** if $\bar{r}_{\text{pass1,pass2}} < 0.40$, $\bar{r}_{\text{outer},z_H} \ge 0.70$, and $\bar{r}_{\text{role}} < \bar{r}_{\text{outer},z_H} - 0.05$.
- **Not supported** otherwise.

### P4 — Unrolled `TCA`: Attractor Masking of Early Fragility with Specific Subspaces
- **Supported** if `TRM` unrolled 6-step `TCA` exhibits strong early fragility attenuation ($\bar{F}_1 / \bar{F}_6 \le 0.25$) while overall rotation specificity (`|z_read| > 3` or `|z_write| > 3`) is $\ge 1.35\%$ ($\ge 5\times$ the $0.27\%$ two-sided $3\sigma$ Gaussian null).
- **Weak** if $\bar{F}_1 / \bar{F}_6 \le 0.40$ and max specificity $\ge 0.81\%$ ($\ge 3\times$ null).
- **Not supported** otherwise.

### P5 — `Q11` $G_{\text{prop}}$ Compactification, Deductive Staircase, and Dark-Fiber Search
Five sub-criteria:
1. Propagation-solvable commutation separation Mann-Whitney $\text{AUC}(\rho_{\text{clean}} < \rho_{\text{shuf}}) \ge 0.90$ on `bench100`.
2. Within-board deductive-round convergence concordance $C_{\text{stair}} \ge 0.65$ and $\ge C_{\text{shuf\_null}} + 0.10$ across `items256`.
3. `TRM` step-16 exact-solve rate on the 51 stalled `bench100` boards $\ge 0.40$ (where $G_{\text{prop}}$ solves $0\%$).
4. Share of `TRM`-solved stalled boards exhibiting non-monotone empty-cell prediction flips $\ge 0.80$.
5. Erasing inter-step latent carry (`reset_zL` or `fiber_erased`) reduces exact-board solve accuracy by $\ge 0.05$ with exact McNemar $p < 0.05$ on `bench100` or `items256`.
- **Supported** if `5/5` sub-criteria hold; **Weak** if `4/5` hold; **Not supported** if $\le 3/5$ hold.

---

## 5. Execution Audit Note (2026-11-13)

Pre-registration was locked on `origin/main` at commit `1f205d67f940ec198dcbd7f7854ab101e932e283` before passing any held-out board (`test_sudoku_bench.csv` or `test_hard.csv`) through any model. During GPU execution on a shared 16 GB RTX 5080 (`7.4 GiB` held externally), four purely mechanical runtime adjustments were made to `trm_sudoku.py` and `llm_sudoku.py`, with **zero modifications** to `sudoku_prop.py`, `score_loopscope02.py`, `test_loopscope02.py`, dataset sampling (`bundle_sha256 = 59459ae0bb60b75db5be017e5f41adead391f2abb0b8fb043cd331c0ee043811`), prompts, or any pre-registered hypothesis/threshold (`P1`–`P5`):

1. **`TRMBlock` SwiGLU hidden width (`trm_sudoku.py`):** Replaced `hidden_size * expansion // 2` (`1024`) with `round_up(expansion * hidden_size * 2 / 3, 256) = 1536` (`gate_up_proj` shape `(3072, 512)`) required to load `step_21700` state dict shapes cleanly.
2. **`PureSudokuTRM.act_step` ambient `no_grad` & `TCA` chunking (`trm_sudoku.py`):** Respected `torch.is_grad_enabled()` inside `act_step` so `@torch.no_grad()` inference does not build autograd graphs, and lowered float32 6-step unrolled `TCA` `chunk_size` from `8` to `2` with `model.requires_grad_(False)` to fit `< 4 GiB` VRAM.
3. **Exact prompt-vocabulary row slicing & non-reentrant checkpointing (`llm_sudoku.py`):** Sliced `embed_tokens` and `lm_head` at shard-load time to the exact sorted union of prompt and `'1'..'9'` token IDs (`< 1 MB` vs. `~2 GB` full 152k–266k vocabulary matrices, exact logit identity) and enabled `use_reentrant=False` gradient checkpointing during single-item `TCA` backward passes so 3B–4B models execute under `< 7.3 GiB` VRAM and `< 96%` Windows commit charge.
4. **`Nanbeige` loop-2 parity slice (`llm_sudoku.py`):** Evaluated `model.lm_head(h2_0)[:, -1, digit_ids_t]` (`M = seq_len` cuBLAS GEMM, matching `model(ids0).logits` to `0.0` max absolute difference) rather than slicing `h2_0[:, -1:, :]` (`M = 1` GEMV, which differed by 1 `bfloat16` ULP = `0.125`).
