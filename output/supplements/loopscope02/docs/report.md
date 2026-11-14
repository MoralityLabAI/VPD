# LOOPSCOPE-02: Virtual Depth vs. Dark-Fiber Attractor Search — Triangulating Looped LMs, Untied LMs, and Tiny Recursive Models on Stratified 9×9 Sudoku

**Date:** 2026-11-13  
**Pre-registration:** [`docs/registration.md`](registration.md) (locked on `main` at commit `1f205d6` prior to evaluating any held-out test board)  
**Automated replay summary:** [`results/loopscope02_report.md`](../results/loopscope02_report.md) | [`results/loopscope02_report.json`](../results/loopscope02_report.json)  
**Companion program:** [`LOOPSCOPE-01`](../../loopscope01/README.md) (`Q1`–`Q9`, `Q11`: pointer-chase & Lean-CGRT loopers)

---

## Executive Answers to the Core Research Questions

1. **Can isolating Pass-1 vs. Pass-2 parameter-activation changes in `Nanbeige4.2-3B` isolate a capability gain over a prosaic `Qwen-4B` model on Sudoku or ARC-AGI-2?**  
   **It isolates the mechanism of two-pass virtual depth, and proves that two-pass virtual depth yields zero multi-step combinatorial capability gain.** Across physical layers `(6, 12, 17, 22)`, `Nanbeige4.2-3B` routes shared weights through partially to nearly orthogonal SVD mixtures on Pass 1 vs. Pass 2 ($\bar{r}_{\text{pass1,pass2}} = 0.355$; $r = 0.085$ at `L12`, $0.277$ at `L22`). However, on deductively stratified 9×9 Sudoku (`256` distinct boards), Pass 2 lifts `Nanbeige4.2-3B` only from `0.098` (`t=1`) to `0.129` (`t=2`, `+0.031`, exact McNemar $p = 0.341$). At its trained 2-loop horizon, `Nanbeige4.2-3B` **significantly trails** untied 32-layer `Qwen3.5-4B-Base` overall (`0.129` vs. `0.211`, `-0.082`, $p = 0.0031$, driven by single-step `S1_naked1`: `0.172` vs. `0.469`), sits indistinguishable from the `1/9 = 0.111` uniform-guess floor on multi-step deductive and search strata `S2–S4` (`0.115` vs. `0.125`, `-0.010`, $p = 0.824$), and **degrades when unrolled beyond its trained depth** (`0.090` at `t=8`, non-contractive relative step norm `0.712`). Comparing Pass 1 against Pass 2 in a fixed-loop LM therefore isolates **parameter-efficient depth multiplexing** (the regime studied by Wu et al., `arXiv:2609.39892`), not iterative constraint solving.

2. **Can triangulating against a Tiny Recursive Model (`TRM`) across its forward-fed loops isolate what actually drives recursive capability?**  
   **Yes—decisively.** On the exact same 256 boards, a **7M-parameter** 2-layer Attention-TRM (`gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att`, `step_21700`, `430×–570×` smaller than `Nanbeige-3B` and `Qwen-4B`) achieves **`0.938` target-cell accuracy** (`+0.727` over `Qwen3.5-4B-Base`, $p = 3.76 \times 10^{-52}$; `+0.809` over `Nanbeige4.2-3B`, $p = 2.45 \times 10^{-58}`), **`0.883` (`226/256`) full-board exact solve** (`0.082` at outer step `s=1` $\to$ `0.883` at `s=16`, **`+0.801`**), and **`1.000` (`100/100`)** full-board exact solve on `test_sudoku_bench` (`0.210` at `s=1` $\to$ `1.000` at `s=16`; independent checkpoint replication `yagizdevre/trm-sudoku-extreme-att`: **`0.920`**). Triangulation isolates four concrete mechanisms behind this `>70` percentage-point capability gap:
   - **Stationary outer operator + inner scratchpad/commit split (`P3`):** Across outer steps `s = 2..16`, `TRM` reuses virtually the exact same SVD component mixture on `z_H` ($\bar{r}_{\text{outer},z_H} = 0.992$, range `0.982–0.997`), while splitting inner `z_L` scratchpad updates from `z_H` board-commit updates ($\bar{r}_{\text{role}}(z_L, z_H) = 0.765$; `0.635` at `L1.down_proj`).
   - **Exact $G_{\text{prop}}$ compactification followed by non-monotone conflict repair (`P5`, `Supported 5/5`):** On propagation-solvable boards, `TRM`'s visible digit beliefs commute with exact synchronous naked+hidden single elimination $G_{\text{prop}}$ ($\text{AUC} = 1.000$, $\rho_{\text{clean}} \to 0.000$ vs. $\rho_{\text{shuf}} = 0.908$) and lock in along the within-board deductive-depth staircase (`2.14 < 2.35 < 2.97 < 12.22` $H$-cycles across `95,757` cell pairs; concordance `0.653` vs. `0.495` cell-shuffled null). On zero-single **stalled** boards where $G_{\text{prop}}$ solves `0%`, `TRM` solves **`100%` (`51/51`)** on `bench100` (`83.6%` on `items256`), with **`98.0%–99.3%`** passing through explicit Sudoku unit conflicts (`13.9 -> 4.8 -> 1.2 -> 0.0` conflicts) and **`100%`** executing non-monotone cell prediction flips (`128.0` flips vs. `82.5` on propagation-solvable boards).
   - **The 501-dimensional dark-fiber discovery (`P5`):** Resetting `z_L \leftarrow L_{\text{init}}` between outer ACT steps reduces `bench100` exact solve by only `1.0` percentage point (`1.000 -> 0.990`; `0.883 -> 0.832` on `items256`), falsifying the assumption that inter-step search state lives in `z_L`. In contrast, projecting `z_H` onto the **11-dimensional row space of `lm_head`** between outer steps—an intervention that preserves every visible step-$s$ digit logit identically (`< 10^{-5}` difference) while erasing only the **501-dimensional `lm_head`-nullspace dark fiber** $(I - P_{\text{head}})z_H$—**collapses exact solve from `1.000` to `0.070` (`-93.0` pp, $p = 2.02 \times 10^{-28}`)** on `bench100` and from **`0.883` to `0.066` (`-81.6` pp, $p = 2.43 \times 10^{-63}`)** on `items256`.
   - **The Detached-ACT vs. BPTT Lyapunov Dichotomy (`P4` prospectively falsified):** Finite `0.25 · RMS` Gaussian kicks on `z_H(1)` contract across outer steps (`1.000 -> 0.395 -> 0.303`) and 10-cell donor corruptions are repaired (`0.000 -> 0.887` full-board solve, $p = 9.27 \times 10^{-69}`), yet 6-step unrolled float32 `TCA` reveals **`F_1 / F_6 = 1,808.4`** (falsifying BPTT-derived `P4`: `F_1 / F_6 <= 0.25`). Because `TRM` trains with **1-step detached ACT**, its 105-call unrolled tangent Jacobian remains mildly expansive ($\approx 1.074^{105} \approx 1,808$) inside nonlinearly contracting `Post-RMSNorm` attractor basins. Crucially, `TCA`'s closed-form orthogonal rotation denominator cancels this `6,400×` gradient swing identically, isolating a step-invariant **`22.04%` read-specific circuit (`81.6×` above the `0.27%` $3\sigma$ Gaussian null)** where unnormalized gradient outer products (`SDI`, Kaissis et al. 2026) fail.

3. **Does combining `LOOPSCOPE-01` + `LOOPSCOPE-02` support a strong paper that advances beyond the 5 papers from the past 10 weeks?**  
   **Yes—provided the paper is framed around the Triangulated Regime Dichotomy (Virtual Depth vs. Head-Readable Staircases vs. Dark-Fiber Attractor Search) rather than Pass-1/Pass-2 LLM activation diffs alone.** Section 7 lays out the exact positioning matrix against all five 2026 papers and the concrete manuscript structure.

---

## 1. Experimental Architecture and Pre-Registered Verdicts

All datasets, SHA-256 checkpoint hashes, donor pairings, and decision thresholds (`P1`–`P5`) were committed at `1f205d6` prior to held-out inference (`bundle_sha256 = 59459ae0bb60b75db5be017e5f41adead391f2abb0b8fb043cd331c0ee043811`). Following the `LOOPSCOPE-01` protocol, `score_loopscope02.py` is run **unmodified**; where a pre-registered threshold trips for a structural reason, both the unaltered mechanical verdict and the underlying mechanistic cause are reported transparently.

### 1.1 Pre-Registered Hypothesis Scorecard

| ID | Pre-Registered Hypothesis | Verdict (`score_loopscope02.py`) | Primary Quantitative Readout & Structural Interpretation |
|---|---|---|---|
| **P1** | Capability scaling triangulation | **Not supported** *(single-cell `s=1` ceiling)* | `Nanbeige` fails OOD scaling (`t=8 - t=2 = -0.039`) and trails `Qwen3.5-4B` (`0.129` vs. `0.211`, $p=0.0031$; `S2–S4`: `-0.010`, $p=0.824$). `TRM` (`0.938`) beats both LLMs by `+0.727` / `+0.809` ($p < 10^{-51}$). Rule missed `s=1->s=16 >= +0.25` on single cells (`0.863 -> 0.938`, `+0.074`) **only because `s=1` (`18` inner passes) already solves `86.3%` of single cells** (`0.863 + 0.25 > 1.0`), while **full-board exact solve jumps `+0.801` (`0.082 -> 0.883`)** on `items256` and **`+0.790` (`0.210 -> 1.000`)** on `bench100`. |
| **P2** | `ARG-02` causal counterfactuals & perturbation repair | **Not supported** *(same `z_H(1)` ceiling)* | `TRM` repairs 10 donor-corrupted cells from **`0.000` to `0.887` full-board solve** ($p = 9.27 \times 10^{-69}$) and contracts finite kicks (`1.000 -> 0.303`), while `Nanbeige` stays non-contractive (`rel_step_norm = 0.712` at `t=8`). Single-cell `clean16 - open16` (`+0.074`, $p=3.1\times 10^{-4}$) and `clean16 - donor16` (`+0.109`, $p=1.9\times 10^{-6}$) start from `z_H(1)` (`0.863`), capping single-cell drop below `0.30` even as board-level `z_H(1)` solve sits at `0.082`. |
| **P3** | SVD weight-usage: cross-pass multiplexing vs. stationary outer reuse + role split | **Weak** | `TRM` outer reuse is near-unit ($\bar{r}_{\text{outer},z_H} = \mathbf{0.992} \ge 0.80$) with strong `z_L`/`z_H` role split ($\bar{r}_{\text{role}} = \mathbf{0.765} \le 0.892$). `Nanbeige` Pass-1/Pass-2 correlation averages **`0.355`** (`< 0.40` Weak cutoff; `< 0.30` Supported cutoff) because multiplexing is **layer-selective**: `L12` (`0.085`) and `L22` (`0.277`) are orthogonal, while `L6` (`0.448`) and `L17` (`0.610`) partially share active subspaces. |
| **P4** | Unrolled `TCA` early fragility attenuation (`F1/F6 <= 0.25`) + specificity | **Not supported** *(mechanistic discovery)* | Specificity passes by `16×` (`max |z| > 3 = 22.04%` vs. `1.35%` bar, `81.6×` null), **but `F_1 / F_6 = 1,808.4`**. Prospective falsification exposes the **Detached-ACT vs. BPTT Lyapunov Dichotomy** (Section 6). |
| **P5** | `Q11` $G_{\text{prop}}$ compactification, deductive staircase, non-monotone conflict repair, and dark-fiber carry | **Supported (`5/5`)** | Commutation `AUC = 1.000` (`>= 0.90`); 48-$H$-cycle staircase concordance `0.653` (`>= 0.65`, null `0.495`); stalled `bench100` solve `1.000` (`>= 0.40`); non-monotone flips `1.000` (`>= 0.80`); dark-fiber erasure drop **`-0.930`** on `bench100` ($p = 2.02 \times 10^{-28}$) and **`-0.816`** on `items256` ($p = 2.43 \times 10^{-63}$). |

---

## 2. Capability Triangulation Across Deductive Strata (`P1`)

Using exact synchronous naked-single + hidden-single propagation $G_{\text{prop}}$ (`sudoku_prop.py`), `items256` comprises 256 distinct 9×9 boards (`64` per deductive stratum, paired with within-stratum wrong-digit donors $y_{d(i)}^* \neq y_i^*$). All language models receive the identical 3-shot prompt (`1 naked1`, `1 prop2`, `1 prop3` from `train.csv`) ending with `"Cell ? (RrCc) Answer: "`, evaluated over exact single-token logits `'1'..'9'`.

### 2.1 Stratified Accuracy Table (`256` Distinct Boards + `bench100` Full-Board Solve)

| Model / Condition | Params | Compute Passes | Overall (`n=256`) | `S1_naked1` (`r=1`) | `S2_prop2` (`r=2`) | `S3_prop3plus` (`r>=3`) | `S4_stalled` (`search`) | Full-Board `items256` | Full-Board `bench100` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Uniform digit baseline (`1/9`) | — | — | `0.111` | `0.111` | `0.111` | `0.111` | `0.111` | `0.000` | `0.000` |
| `Qwen3.5-4B-Base` (mid-stack `L16` lens) | 4.0B | 16 layers | `0.109` | `0.094` | `0.172` | `0.094` | `0.078` | — | — |
| `Qwen3.5-4B-Base` (final `L32`) | 4.0B | 32 layers | **`0.211`** | **`0.469`** | `0.078` | `0.141` | `0.156` | — | — |
| `Qwen3-4B-Instruct-2507` (`L36`, pure-SDPA) | 4.0B | 36 layers | **`0.254`** (`mid L18: 0.129`) | **`0.562`** | `0.094` | `0.172` | `0.188` | — | — |
| `Nanbeige4.2-3B` (loop `t=1`) | 3.0B | 22 layers | `0.098` | `0.125` | `0.062` | `0.062` | `0.141` | — | — |
| `Nanbeige4.2-3B` (loop `t=2`, trained) | 3.0B | 44 eff. layers | **`0.129`** | `0.172` | `0.109` | `0.078` | `0.156` | — | — |
| `Nanbeige4.2-3B` (loop `t=8`, over-looped) | 3.0B | 176 eff. layers | `0.090` | `0.047` | `0.156` | `0.062` | `0.094` | — | — |
| `TRM` 7M (outer step `s=1`) | **7.0M** | 18 cycles (`36` L) | `0.863` | `1.000` | `1.000` | `0.859` | `0.594` | `0.082` | `0.210` |
| `TRM` 7M (outer step `s=16`) | **7.0M** | 288 cycles (`576` L) | **`0.938`** | **`1.000`** | **`1.000`** | **`1.000`** | **`0.750`** | **`0.883`** | **`1.000`** (`0.920` repl.) |

### 2.2 Three Immediate Lessons from the Stratified Breakdown

1. **Single forward pass of an untied 4B model solves local peer elimination (`S1_naked1`) halfway (`46.9%`), then hits a hard wall at `r >= 2` (`7.8%–15.6%`).** In `Qwen3.5-4B-Base`, accuracy stays at the `1/9` chance floor at mid-stack `L16` (`0.094` on `S1`) and rises to `0.469` only in the second half of the network (`L17–L32`), where attention heads aggregate the 20 row/column/box peers of `?` and suppress already-occupied digits. Once deduction requires even a two-hop chain (`S2_prop2`), single-pass 4B accuracy drops back to chance (`0.125` across `S2–S4`).
2. **Two-pass looping in `Nanbeige4.2-3B` does not unlock `r >= 2` deduction—and fails to extrapolate in `t`.** Across loops `t = 1..8`, `Nanbeige4.2-3B` overall target-cell accuracy traces `[0.098, 0.129, 0.121, 0.102, 0.129, 0.141, 0.109, 0.090]`. Pass 2 lifts `S1_naked1` modestly (`0.125 -> 0.172`) and leaves `S2–S4` at `0.115` (`95%` paired difference CI vs. `Qwen3.5-4B-Base`: `[-0.057, +0.036]`, $p = 0.824$). Unrolling past `t=2` degrades `S1_naked1` to `0.047` by `t=8`.
3. **Why `TRM` outer step `s=1` already scores `0.863` on single cells while scoring only `0.082` on full boards:** A single outer ACT step of `TRM` executes `3 H-cycles × 6 L-cycles = 18` weight-tied 2-layer transformer passes (`36` layer evaluations). Eighteen inner passes suffice to finish almost all local and multi-round propagation (`1.000` on `S1`, `1.000` on `S2`, `0.859` on `S3`), leaving only `+0.074` (`0.863 -> 0.938`, $p = 3.11 \times 10^{-4}$) of single-cell headroom across outer steps `2..16` (`S3`: `0.859 -> 1.000`; `S4`: `0.594 -> 0.750`). Yet at `s=1`, **91.8% of `items256` boards still violate global Sudoku constraints** (`19.7` unit conflicts per stalled board; full-board exact solve `0.082`). Outer steps `2..16` are where global constraint satisfaction happens (`0.082 -> 0.883` on `items256`; `0.210 -> 1.000` on `bench100`).

---

## 3. Causal Counterfactuals (`ARG-02`) and Perturbation Repair (`P2`)

Following the `ARG-02` verification protocol, we test whether recurrent trajectories compute genuine state-dependent updates rather than linear momentum (`open-loop`), input-independent drift (`donor displacement`), or fragile feedforward pipelines (`corrupt-state injection`).

| Counterfactual Intervention | Metric | Clean | Intervention | $\Delta$ (`Clean - Interv.`) | 95% Bootstrap CI | Exact McNemar $p$ |
|---|---|---:|---:|---:|---:|---:|
| `TRM` open-loop `z_H(1) + 15(z_H(1) - H_init)` | Target cell (`n=256`) | `0.938` | `0.863` | **`+0.074`** | `[+0.035, +0.113]` | `3.11e-4` |
| `TRM` donor displacement `z_H,i(1) + (z_H,d(16) - z_H,d(1))` | Target cell (`n=256`) | `0.938` | `0.828` | **`+0.109`** | `[+0.070, +0.152]` | `1.94e-6` |
| `TRM` 10-cell donor patch at `s=1`: step 1 vs. step-16 repaired | **Full board (`n=256`)** | `0.887` *(repaired)* | `0.000` *(at `s=1`)* | **`+0.887`** | `[+0.848, +0.926]` | **`9.27e-69`** |
| `TRM` 10-cell donor patch at `s=1`: step 1 vs. step-16 repaired | Target cell (`n=256`) | `0.957` *(repaired)* | `0.723` *(at `s=1`)* | **`+0.234`** | `[+0.184, +0.289]` | **`2.73e-17`** |
| `Nanbeige` open-loop `h(1) + (h(1) - x)` at `t=2` | Target cell (`n=256`) | `0.129` | `0.098` | `+0.031` | `[-0.024, +0.090]` | `0.350` |
| `Nanbeige` donor displacement on answer tail at `t=2` | Target cell (`n=256`) | `0.129` | `0.098` | `+0.031` | `[-0.016, +0.078]` | `0.243` |
| `Nanbeige` 5-token donor answer-tail patch before Pass 2 | Target cell (`n=256`) | `0.129` | `0.117` | `+0.012` | `[-0.020, +0.043]` | `0.648` |

**Attractor basin verification.** Overwriting 10 random empty cells of `z_H(1)` with donor board states destroys full-board validity at `s=1` (`0/256 = 0.000` exact boards). Running outer steps `2..16` with the original clue embedding `x_emb` **repairs `227/256` (`0.887`) boards to exact global completion**—statistically indistinguishable from uncorrupted `s=16` (`226/256 = 0.883`). Likewise, injecting an isotropic Gaussian perturbation $\delta_1$ of norm $0.25 \cdot \text{RMS}(z_H(1))$ into `z_H(1)` contracts monotonically across outer steps:
$$
\frac{\|z_H^{\text{kick}}(s) - z_H^{\text{clean}}(s)\|_2}{\|\delta_1\|_2}
= [1.000,\; 0.395,\; 0.487,\; 0.469,\; \dots,\; 0.359,\; 0.306,\; \mathbf{0.303}] \quad (s = 1..16).
$$
In contrast, `Nanbeige4.2-3B` relative step norms $\|h(t) - h(t-1)\|_2 / \|h(t-1)\|_2$ follow `[21.21, 6.53, 0.77, 0.87, 0.58, 0.67, 0.66, 0.71]` across `t = 1..8`—never contracting toward a fixed point.

---

## 4. SVD Weight-Usage Geometry: Layer-Selective Multiplexing vs. Stationary Operator + Role Split (`P3`)

To test the user's central mechanistic hypothesis—whether tracking singular-component activation changes across passes isolates how looped models compute—we project layer inputs onto the top-64 right singular vectors $v_r$ of each output projection, weighted by $\sigma_r$: $u_r = \mathbb{E}[|(X v_r)|] \sigma_r$.

### 4.1 `Nanbeige4.2-3B`: Layer-Selective Two-Pass Multiplexing (`mean r = 0.355`)

| Physical Layer (`mlp.down_proj`) | Pass-1 vs. Pass-2 Pearson $r$ | Usage Magnitude Ratio (`Pass 2 / Pass 1`) | Mechanistic Regime |
|---|---:|---:|---|
| `L6` (early core) | `0.448` | `1.657×` | Partially shared feature extraction |
| `L12` (mid core) | **`0.085`** | `0.866×` | **Near-orthogonal cross-pass routing** |
| `L17` (upper-mid core) | `0.610` | `1.211×` | Shared representation transport |
| `L22` (final core / pre-head) | **`0.277`** | `0.378×` | **Pass-specialized readout preparation** |
| **Mean across layers** | **`0.355`** | `1.028×` | **Two-stage virtual depth (`Weak` under `<0.30` / `<0.40` rule)** |

`Nanbeige4.2-3B` does **not** rotate uniformly across every layer: `L12` (`r = 0.085`) and `L22` (`r = 0.277`, with Pass-2 amplitude damped to `0.38×` Pass 1) switch to nearly disjoint singular subspaces on Pass 2, whereas `L6` and `L17` retain moderate overlap (`0.45–0.61`). This confirms that SVD input-usage correlation cleanly isolates cross-pass subspace switching—while proving that **subspace switching is a signature of fixed-pass depth compression, NOT recursive reasoning**.

### 4.2 `TRM` 7M: Near-Unit Outer Stationarity (`r = 0.992`) + Inner Scratchpad/Commit Split (`r = 0.765`)

| `TRM` Tied Weight Matrix | Outer Stationarity $r_{\text{outer}}(z_H)$ (`s=2..16`) | Outer Stationarity $r_{\text{outer}}(z_L)$ (`s=2..16`) | Warmup Shift $r(z_H, s=1 \text{ vs } 2)$ | Inner Role Split $r_{\text{role}}(z_L \text{ vs } z_H)$ |
|---|---:|---:|---:|---:|
| `L0.self_attn.o_proj` | `0.982` | `0.993` | `0.783` | `0.790` |
| `L0.mlp.down_proj` | `0.995` | `0.998` | `0.857` | `0.859` |
| `L1.self_attn.o_proj` | `0.997` | `0.994` | `0.899` | `0.776` |
| `L1.mlp.down_proj` | **`0.996`** | `0.992` | `0.844` | **`0.635`** |
| **4-Matrix Mean** | **`0.992`** | **`0.994`** | `0.846` | **`0.765`** |

The contrast with `Nanbeige4.2-3B` is stark:
1. **True operator iteration across outer steps:** After step-1 initialization (`r = 0.846` between `s=1` and `s=2`), `TRM` applies **the exact same weighted singular spectrum across outer steps `2..16`** ($\bar{r}_{\text{outer},z_H} = 0.992$, $\bar{r}_{\text{outer},z_L} = 0.994$).
2. **Role multiplexing happens *within* each cycle, not across outer depth:** Inside every $H$-cycle, the same 2-layer `L_level` block is called 6 times to update `z_L` (conditioned on `z_H + x_emb`) and once to update `z_H` (conditioned on `z_L`). In `L1.mlp.down_proj`, `r_role(z_L, z_H)` drops to **`0.635`** (a `0.361` gap below `0.996` outer stationarity), demonstrating that the tied MLP separates inner message-passing (`z_L`) from state commitment (`z_H`).

---

## 5. `Q11` Symbolic Compactification, Non-Monotone Conflict Repair, and the 501-Dim Dark Fiber (`P5`, `Supported 5/5`)

Extending `LOOPSCOPE-01` `Q11` (which measured pointer-chase compactification error against exact permutation advance, `AUC = 0.992`), we test `TRM`'s trajectories against the exact Sudoku propagation lattice $G_{\text{prop}}$ and probe where post-stall search state resides.

### 5.1 Exact $G_{\text{prop}}$ Commutation and the 48-$H$-Cycle Deductive Staircase

- **Commutation separation (`AUC = 1.000`):** On propagation-solvable boards, commutation error $\rho_s(b) = \text{mean}_{i \in \text{empty}} \|p_{s+1}(b, i) - G_{\text{prop}}(p_s(b), \text{givens}_b)_i\|_2$ collapses rapidly to **exact zero**:
  - `bench100` (`49` prop boards): `[0.148, 0.013, 0.010, 0.005, 0.000, ...]` vs. deranged-clue null `[0.938, 0.912, 0.908, ...]` (**Mann-Whitney `AUC = 1.000`**, paired win-rate `1.000`).
  - `items256` (`73` prop boards): `[0.193, 0.019, 0.002, 0.000, ...]` vs. deranged-clue null `[0.934, 0.906, 0.900, ...]` (**`AUC = 1.000`**).
- **48-$H$-cycle deductive-depth staircase (`concordance = 0.6531` vs. `0.4947` null):** Measuring fine-grained lock-in cycle (the earliest $H$-cycle $c \in \{1..48\}$ from which a cell's top-1 digit stays locked to the ground-truth answer through $c=48$) across `95,757` within-board empty-cell pairs $(i, j)$ with $1 \le r(i) < r(j)$ on `items256`:
  - Mean lock-in $H$-cycle strictly orders deductive depth: **`2.14` (`r=1`) < `2.35` (`r=2`) < `2.97` (`r>=3`) < `12.22` (`stalled`)** (and cumulative uncertainty `[1.10, 1.27, 1.78, 8.77]`).
  - Within-board pairwise concordance is **`0.6531`** (`+0.1584` above the within-board cell-permutation null `0.4947`).

### 5.2 Post-Stall Search Dynamics: Conflict Traversal and Non-Monotone Revision

On zero-single **stalled** boards (`r(i) = -1` for every empty cell; $G_{\text{prop}}$ solves **`0%`**):
- `TRM` solves **`100.0%` (`51/51`)** of stalled `bench100` boards (`0.020 -> 0.725 -> 0.941 -> 0.980 -> 1.000` at `s=13..16`) and **`83.6%` (`153/183`)** of stalled `items256` boards (`0.005` at `s=1` $\to$ `0.836` at `s=14..16`).
- **Conflict-driven search:** **`98.0%` (`50/51`)** of solved stalled `bench100` boards and **`99.3%` (`152/153`)** of solved stalled `items256` boards pass through explicit duplicate-digit Sudoku unit conflicts at intermediate $H$-cycles before eliminating them (`bench100` mean conflicts on stalled boards: `13.90` at `s=1` $\to$ `4.78` at `s=2` $\to$ `1.22` at `s=4` $\to$ `0.43` at `s=12` $\to$ **`0.00`** at `s=13..16`).
- **Non-monotone revision:** **`100.0%`** of solved stalled boards in both cohorts exhibit non-monotone empty-cell prediction flips across the 48 $H$-cycles (`128.0` flips/board on stalled `items256` vs. `82.5` on propagation-solvable boards).

### 5.3 Where Does Search Memory Live? The 501-Dim Dark-Fiber Ablation

Between outer ACT steps $s$ and $s+1$, `TRM` passes two `[81 + 16, 512]` state tensors `(z_H, z_L)`. Its readout `lm_head` (`11 × 512`, rank 11) projects 512-dim `z_H` onto 11 vocabulary logits (`pad`, `'0'`, `'1'..'9'`), leaving a **501-dimensional orthogonal complement** $(I - P_{\text{head}})z_H$ completely invisible to digit readout at step $s$.

| Inter-Step State Intervention (applied between every outer step $s = 1..15$) | Step-$s$ Digit Logits Preserved? | `bench100` Exact Solve (`n=100`) | `items256` Exact Solve (`n=256`) | Exact McNemar $p$ vs. Clean |
|---|:---:|---:|---:|---:|
| **Clean `TRM` (`s=16`)** | Yes (exact) | **`1.000`** (`100/100`) | **`0.883`** (`226/256`) | — |
| **`reset_zL`:** $z_L \leftarrow L_{\text{init}}$, keep full 512-dim $z_H$ | Yes (exact) | **`0.990`** (`99/100`, `-0.010`) | **`0.832`** (`213/256`, `-0.051`) | `1.00` (`bench100`) / `2.44e-4` (`items256`) |
| **`fiber_erased`:** $z_L \leftarrow L_{\text{init}}$ **and** $z_H^{\text{board}} \leftarrow z_H^{\text{board}} P_{\text{head}} + H_{\text{init}}(I - P_{\text{head}})$ | **Yes (`< 1e-5` logit diff at step $s$)** | **`0.070`** (`7/100`, **`-0.930`**) | **`0.066`** (`17/256`, **`-0.816`**) | **`2.02e-28`** / **`2.43e-63`** |

**Why this resolves the `LOOPSCOPE-01` Monitorability Question (`Q3`/`Q4` vs. `LOOPSCOPE-02` `P5`):**
1. **Inter-step `z_L` carry is mostly dispensable (`-1` to `-5` pp):** Within each outer step, `3 H-cycles × 6 L-cycles` rapidly reconstruct `z_L` from `z_H + x_emb`. `z_L` is a fast *intra-step* message-passing buffer, not the primary repository of multi-step search history.
2. **Visible digit probabilities $P_{\text{head}} z_H$ cannot sustain multi-step search (`-82` to `-93` pp collapse):** If `TRM` were merely performing soft belief propagation on digit distributions (a neural Sinkhorn / mean-field relaxation), retaining the exact 11-dimensional `lm_head` row space $z_H P_{\text{head}}$ at every outer step would preserve solve accuracy. Instead, erasing the 501-dimensional nullspace $(I - P_{\text{head}})z_H$—without altering a single logit at step $s$—**destroys `93%` of `bench100` solves**.
3. **Unifying `LOOPSCOPE-01` and `LOOPSCOPE-02` (The Monitorability Regime Boundary):** In `LOOPSCOPE-01` (`Q3` Lean-CGRT tactic selection and `Q4` pointer chasing), every loop advances deterministically along a Markovian state chain where the current hop *is* the sufficient statistic—so full monitorability through `lm_head` costs zero capability (`Q4`: $\lambda \ge 0.1$ achieves `1.00` extrapolation and `1.00` head readability). In **non-monotone NP search** (`LOOPSCOPE-02` stalled Sudoku), visible cell commitments pass through conflicts and flip `108–128` times per board; without **dark-fiber memory** $(I - P_{\text{head}})z_H$ recording context, candidate eliminations, and conflict history orthogonal to the committed digits, the recurrent operator collapses into unresolved cycles (`0.070`).

---

## 6. The Detached-ACT vs. BPTT Lyapunov Dichotomy and `TCA` Scale Cancellation (`P4`)

In `LOOPSCOPE-01` (`Q2`–`Q5`), every extrapolating looped model trained with multi-step backpropagation (`Huginn-0125`, `Lean S-loop`, `LoopedChase`) exhibited **early first-order fragility attenuation** ($F_1 / F_{\text{late}} \ll 1$: `0.43` in `Huginn`, `0.0065` in `Lean S-loop`, `0.0001–0.01` in `LoopedChase` extrapolators, predicting extrapolation with `AUC = 0.950` in `Q4/Q5`). We prospectively registered `P4` (`F_1 / F_6 <= 0.25`) to test whether this law transfers out-of-family to `TRM`.

### 6.1 Prospective Falsification of `P4a`: `F_1 / F_6 = 1,808.4`

Backpropagating target-cell log-odds margin in float32 across `s = 1..6` full outer steps (`6 × 3 × 7 = 126` `L_level` evaluations; `105` core calls between `s=1` and `s=6`) across `128` boards (`S1_naked1 + S3_prop3plus`):

| `TRM` Matrix | $F_1$ | $F_2$ | $F_3$ | $F_4$ | $F_5$ | $F_6$ | $F_1 / F_6$ | $|z_{\text{read}}| > 3$ (`s=1..6` range) | $|z_{\text{write}}| > 3$ (`s=1..6` range) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `L0.o_proj` | `66,706` | `1,817` | `373.6` | `25.8` | `10.4` | `48.1` | `1,386×` | `16.5% – 20.1%` | `9.6% – 13.2%` |
| `L0.down_proj` | `148,723` | `4,580` | `920.8` | `62.0` | `21.1` | `98.5` | `1,509×` | `19.8% – 28.8%` | `12.2% – 15.6%` |
| `L1.o_proj` | `64,153` | `2,218` | `377.0` | `25.7` | `8.5` | `29.8` | `2,150×` | `8.0% – 12.7%` | `9.2% – 11.5%` |
| `L1.down_proj` | `129,233` | `6,567` | `1,132.5` | `83.6` | `32.4` | `59.1` | `2,188×` | `29.1% – 37.4%` | `10.7% – 18.2%` |
| **4-Matrix Mean** | `102,204` | `3,795` | `701.0` | `49.3` | `18.1` | `58.9` | **`1,808.4×`** | **`22.04%` overall** (`81.6×` null) | **`12.37%` overall** (`45.8×` null) |

*(For comparison on `S1_naked1`, `Nanbeige4.2-3B` has $F_1 / F_2 = [2.00, 0.97, 0.42, 1.96]$ across `L6, L12, L17, L22` with low early specificity `1.0%–5.6%` rising to `8.4%` at `L22` Pass 2; `Qwen3.5-4B-Base` concentrates specificity sharply in final `L32`: `|z_read| > 3 = 19.9%`, `|z_write| > 3 = 24.7%` vs. `0.2%–2.5%` at `L8–L24`.)*

### 6.2 Why Infinitesimal Tangents Explode (`1,808×`) While Finite Kicks Contract (`0.303×`)

How can the same recurrent dynamical system contract finite perturbations (`1.000 -> 0.303`, Section 3) while amplifying infinitesimal backpropagated gradients by `1,808×` (`6,400×` between `s=5` and `s=1`)?
1. **Training-objective root cause (1-Step Detached ACT vs. BPTT):** `TRM` (`TinyRecursiveReasoningModel_ACTV1`) runs its first `H_cycles - 1` inner cycles under `torch.no_grad()` and calls `.detach()` on `(z_H, z_L)` between every outer ACT step during training. Unlike BPTT-trained loopers (`LOOPSCOPE-01`), **multi-step unrolled Jacobians never appear in `TRM`'s training loss**. An average per-call spectral norm of just $1.074$ across `5 × 21 = 105` `L_level` calls yields $1.074^{105} \approx 1,800$.
2. **Geometric resolution (`Post-RMSNorm` sphere folding):** Every `TRMBlock` applies `Post-RMSNorm`, radially projecting states onto the $\sqrt{d}$-sphere while sharp attention/SwiGLU boundaries fold trajectories into discrete digit basins. Along basin boundaries or transverse directions, local first-order derivatives remain $> 1$, even though any *finite* displacement ($0.25 \cdot \text{RMS}$) is radially clamped by `RMSNorm` and pulled back into the same discrete board attractor (`0.303×`).
3. **Methodological payoff over `Kaissis et al. (ICML 2026 SDI)`:** `SDI` attributes step influence via raw per-step gradient outer products $\phi_s = X_s^\top A_s$ (whose Frobenius norm is $F_s$). On detached-ACT models like `TRM`, raw $\phi_s$ is distorted by **`1,808×–6,400×` tangent explosion**, swamping steps `2..6` in favor of step `1`. Because `TCA` divides component removal $\Delta_{s,r} = -v_r^\top X_s^\top A_s u_r$ by the closed-form orthogonal rotation null $\sigma_{\text{read}} = \|v_r\| \|X_s^\top A_s u_r\| / \sqrt{d_{\text{in}}}$, **any scalar Jacobian inflation in $A_s$ cancels identically**, revealing a step-invariant **`22.04%` read-specific circuit** across all 6 steps (matching `LOOPSCOPE-01` `Q4 P2`).

---

## 7. Advancing Beyond the 5 Recent Looped-Transformer Papers: Positioning & Paper Blueprint

### 7.1 Direct Comparison Against the Preceding 10 Weeks of Literature

| Paper (2026) | What That Paper Establishes | What It Cannot Answer | How `LOOPSCOPE-01` + `LOOPSCOPE-02` Advances Beyond It |
|---|---|---|---|
| **Wu, Hu & Zhang** (`arXiv:2609.39892`, Sept 2026) *J-Steering* | Shared weights induce distinct linear transformations across loop passes. | Whether cross-pass subspace switching creates multi-step reasoning capability over an untied model. | Shows `Nanbeige4.2-3B` Pass-1/Pass-2 orthogonality (`L12 r=0.085`, `mean r=0.355`) yields **zero** multi-step Sudoku gain over `Qwen3.5-4B` (`-0.010`, $p=0.824$), whereas `TRM` achieves **`0.992` outer stationarity** + `0.765` inner role split. |
| **Sesodia et al.** (`arXiv:2610.24135`, NeurIPS 2026) *Difficulty Staircases* | Harder inputs require monotonically more recurrent iterations. | Whether the recurrent step commutes with an exact symbolic operator, or what happens *past* deductive stall. | Proves `AUC = 1.000` commutation with exact $G_{\text{prop}}$, 48-$H$-cycle within-board deductive staircase (`2.14 < 2.35 < 2.97 < 12.22`), **and** `98–99%` conflict-driven non-monotone search (`128` flips) past stall. |
| **Olszowiec et al.** (`arXiv:2610.18127`, Oct 2026) *Loop Refinement* | Observational cosine/norm refinement across loops. | Causal necessity of intermediate states vs. momentum, or where hidden search state lives. | `ARG-02` donor/corruption repair (`0.000 -> 0.887`, $p=9.3\times 10^{-69}$) + **501-dim dark-fiber erasure (`1.000 -> 0.070`, `-93` pp)** vs. `reset_zL` (`0.990`), localizing search state to $(I - P_{\text{head}})z_H$. |
| **Kaissis et al.** (`SDI`, ICML 2026) *Step-Resolved Attribution* | Per-step outer product $\phi_t = X_t^\top A_t$ for training-data influence. | Distinguishing generic Jacobian fragility from subspace specificity, especially under detached ACT. | Discovers the **Detached-ACT vs. BPTT Lyapunov Dichotomy** (`F_1/F_6 = 1,808` despite `0.303×` kick contraction), showing raw $\phi_t$ blows up `6,400×` while `TCA` rotation nulls stay invariant (`22.04%`). |
| **Blayney et al.** (2026) *Loop-Depth Representations* | Internal representation trajectories within a single looped family. | Cross-paradigm triangulation (fixed-loop LM vs. untied LM vs. convergent TRM) and monitorability trade-offs. | Establishes the **Monitorability Regime Boundary** across `LOOPSCOPE-01` + `02`: Markovian chains (`Q3/Q4`) stay head-readable; NP conflict search (`P5`) requires unmonitored dark-fiber state. |

### 7.2 Recommended Paper Thesis and `ARC-AGI-2` Extension

- **Recommended Title / Thesis:** *Virtual Depth, Head-Readable Staircases, and Dark-Fiber Search: Mechanistic Regimes of Looped Transformers* (unifying `LOOPSCOPE-01` and `LOOPSCOPE-02`).
- **Why Pass-1/Pass-2 `Nanbeige` vs. `Qwen` should be Act I (Negative Capability Control), not the standalone paper:** On both Sudoku (`LOOPSCOPE-02`) and `ARC-AGI-2` direct grid prediction, a 2-pass general LM (`Nanbeige4.2-3B`) without long chain-of-thought tokens operates in the **virtual-depth compression regime**: Pass 1 $\to$ Pass 2 multiplexing buys parameter efficiency (`3B` matching ~`4B` prose perplexity), not multi-step constraint propagation or hypothesis backtracking. Presenting this negative control *alongside* `TRM` turns a potential dead-end ("why didn't 2-pass looping beat Qwen on Sudoku?") into a sharp architectural theorem: **capability gains from weight tying require stationary operator iteration ($r_{\text{outer}} \approx 1$) over a persistent canvas state with dark-fiber capacity, not cross-pass subspace rotation ($r \approx 0.1\text{–}0.35$).**
- **Low-cost `ARC-AGI-2` extension (if desired as a compact 1-table generalization check in the paper):** Apply the exact `fiber_erased` vs. `reset_zL` ablation (`3.2.4`) and `TCA` scale-invariance check (`6.1`) to an `ARC-AGI` `TRM` checkpoint (`7M` grid-canvas `TRM`). Because `ARC` output grids have 11 color tokens (`0..9` + pad)—the exact same `11 × 512` `lm_head` geometry as Sudoku—the `501`-dim orthogonal dark-fiber projection $(I - P_{\text{head}})z_H$ transfers with **zero code changes** to test whether `ARC` transformation synthesis likewise resides in $(I - P_{\text{head}})z_H$ while `z_L` regenerates intra-step.

---

## 8. Reproducibility & Artifact Map

All code, tests, pre-registration locks, and deterministic JSON receipts live under `mechinterp/loopscope02/`:
- [`docs/registration.md`](registration.md) — Pre-registration (`1f205d6`) + dated execution audit note.
- [`scripts/sudoku_prop.py`](../scripts/sudoku_prop.py) — Exact synchronous $G_{\text{prop}}$ solver, deductive-round stratifier, and deterministic 256-board + `bench100` bundle builder.
- [`scripts/trm_sudoku.py`](../scripts/trm_sudoku.py) — Self-contained `PureSudokuTRM` (`step_21700` + `yagizdevre` replication), `ARG-02` counterfactuals, `reset_zL` vs. `fiber_erased` dark-fiber projection, SVD usage, 6-step float32 `TCA`, and 48-$H$-cycle `Q11` compactification suite.
- [`scripts/llm_sudoku.py`](../scripts/llm_sudoku.py) — Direct safetensors GPU shard streamer with exact prompt-vocabulary row slicing and `use_reentrant=False` checkpointed `TCA` for `Nanbeige4.2-3B`, `Qwen3.5-4B-Base`, and `Qwen3-4B-Instruct-2507`.
- [`scripts/score_loopscope02.py`](../scripts/score_loopscope02.py) — Pure-CPU replay scorer verifying `bundle_sha256`, exact binomial McNemar tests, 2,000-resample paired bootstrap CIs, and `P1`–`P5` verdicts.
- [`tests/test_loopscope02.py`](../tests/test_loopscope02.py) — Unit + receipt-replay verification suite (`6` tests).
