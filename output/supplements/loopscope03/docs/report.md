# LOOPSCOPE-03 Evaluation Report: Three-Ledger Iteration Pricing, Data-Scaled Conditional Refill, and Enforced Path Independence

**Pre-registration commit:** `b1ec77550997cbc5ae2cd84618af74e8f34e38e7` (`2026-11-14T08:28:03+08:00`, `mechinterp/loopscope03/docs/registration.md`, locked prior to Stage 1–3 evaluation commit `b8ec729c` [`2026-11-14T21:43:26+08:00`]; mirrored publicly in `MoralityLabAI/VPD/output/supplements/loopscope03/`)  
**Dates:** 2026-11-14 to 2026-11-15  
**Receipts:**
- `results/stage1_conditional_refill.json` (`5,019.3 s`, Stage 1 capacity ladder `C0`–`C6` on `train_900` + $z_H \times z_L$ cross + $s \in \{16, 32, 64\}$ cycle audit)
- `results/stage2_3_resume_and_fork.json` (`17,551.8 s`, Stage 2 `Resume` + Stage 3 `Fork` / Kubo et al. `C-voting` replication across `Nanbeige4.2-3B`, `LoopedChase`, and both `TRM` checkpoints)
- `results/huginn_cache_and_fork.json` (`411.7 s`, supplementary path-independence & conditional-cache audit on `tomg-group-umd/huginn-0125`)
- `results/stage4_path_independence_and_scaling.json` (`30,831.4 s`, Stage 4 data-size scaling `N = 100..8,100`, periodic refill `m = 1, 2, 4, 8`, unselected `never_lock` + uncapped solver-delay `Resume` audit, and unconditional / enforced path independence across `K = 8` forks)

---

## 1. Executive Summary & Calibrated Verdict Ledger (`H3.1`–`H3.5`)

Fixed-budget ($s=16$) hidden-state erasure confounds two distinct effects in stationary, input-re-injected loops $S_{t+1} = F_x(S_t)$: (i) **recomputable cache** that the operator rebuilds in additional iterations, and (ii) **cross-buffer backup channels** when only $(I - P_{\text{head}})z_H^{\text{board}}$ is ablated while $z_L$ and the 16 prefix tokens $z_H^{\text{pref}}$ remain live. `LOOPSCOPE-03` decomposes the carried state at outer step $t$ into **three disjoint ledgers** $S_t = V_t \oplus C_t \oplus R_t$:

1. **$V_t$ (Visible / Head-Readable State):** $z_H^{\text{board}}(t) P_{\text{head}}$ (`rank = 11` on the 81 Sudoku cells; `rank = 16` in `LoopedChase`).
2. **$C_t$ (Conditional Cache):** $\hat{H}_\theta(x, V_t)$, the sphere-calibrated component of the **entire** hidden state $H_t = \big(z_H^{\text{board}}(t) P_\perp,\; z_H^{\text{pref}}(t),\; z_L(t)\big)$ predictable from puzzle clues $x$ and visible logits $V_t$ alone (fitted strictly on disjoint training puzzles, validated on `val_100 = train.csv[900:1000]`, `0` overlap with `items256` and `bench100`). Because an external monitor observes $(x, V_t)$, computation in $V_t \oplus C_t$ is **reconstructible from the visible trajectory**, and replacing $H_t \leftarrow \hat{H}_\theta(x, V_t)$ **enforces hidden-state path independence** conditioned on $(x, V_t)$.
3. **$R_t$ (Trajectory-Dependent Residual):** $R_t = H_t - \hat{H}_\theta(x, V_t)$, combining genuine trajectory history with any unmodeled refill approximation error. Across our two frozen `TRM` checkpoints, **$R_t$ is causally null on propagation boards and causally active—with checkpoint-dependent sign—only on stalled boards**.

| Hypothesis | Verdict | Pre-registered threshold | Calibrated outcome across Stage 1–4 receipts |
|---|---|---|---|
| **Admission Gate** | **Exact split** | $\bar{r}_{\text{outer}} \ge 0.90$ & transplant $L_\infty < 10^{-5}$ | `Nanbeige4.2-3B` **fails** ($\bar{r}=0.355$, loop-8 acc `0.090` vs. loop-2 `0.129` $\to$ **Regime I**, confounded with absence of input re-injection). `TRM` ($\bar{r}=0.9925$, $L_\infty=0.0$), `LoopedChase`, and `Huginn-0125` ($L_\infty=0.0$) **pass**. |
| **`H3.1`** ($z_L \times z_H$ backup channel) | **Supported** | Joint `V_only` drop $\ge 15$ pp vs. clean & exceeds single-channel erasure | Primary `gaoxin492` `items256` `WB-Stall@16`: single-channel `0.749` (`mean_all_zH`) / `0.765` (`reset_zL`) / `0.792` (`mean_zL`), whereas joint `V_only` drops to **`0.617` / `0.612`** (`-21.9` / `-22.4` pp, $p \le 4.6\times 10^{-10}$) and traps **`33.3%`–`36.6%`** of stalled boards in exact periodic cycles through $s=64$ (`0.519` / `0.486` and `43.2%`–`44.8%` cycles on `yagizdevre`). |
| **`H3.2`** (Propagation cache sufficiency) | **Supported** *(with $\lambda=1.0$ readout blind spot)* | `LoopedChase` ($\lambda \ge 0.1$) & `TRM` `WB-Prop`: `VC` $\ge 97\%$ at $s=16$ (`100%` at $s=32$) & $\text{Price}(R_t) < 0.25$ | `TRM` `WB-Prop`: `VC_C5 = 1.000` (`73/73`) at $s=16, 32, 64$ on primary (`N=900..8,100`) and `1.000` (`N=900`) / `0.973` (`N=8,100`, above `0.877` clean) on replication; primary `WB-Prop` has `0.000` `never_lock` and `0.0` median/mean loop delay at $t \ge 4$. `LoopedChase` $\lambda=0.1$: `VC = 1.000` (`3/3` seeds) and `Price(R_4) = 0.00` loops, whereas densely supervised $\lambda=1.0$ carries `+3.06` and `+5.46` loops in `2/3` seeds (instantiating Sharma & Vu's 2026 readout blind spot). |
| **`H3.3`** (Post-stall $R_t$, scaling, & correlation gate) | **Mixed / Failed correlation gate** | `\|R^2(C6)-R^2(C5)\| < 0.02`; significant `VC` stalled deficit; `Price(R_t) > 0.5`; flip $\rho \ge 0.25$ | Architecture saturated on `train_900` (`0.844 -> 0.863`), **but `N=900` was data-limited**: Stage 4 scaling to `N=8,100` disjoint boards lifts primary `val_100` $R^2$ to **`0.915`** (`0.837` on replication) and narrows closed-loop primary `WB-Stall` gap from `-7.7 / -10.4` pp (`N=900`) to **`-4.9 / -7.1` pp** (`0.787 / 0.825` vs. `0.836 / 0.896`), while lifting replication `WB-Stall@64` to **`0.760` (`+16.9` pp above `0.590` clean)**. Relaxing refill to every $m=2,4,8$ steps on primary restores `WB-Stall@64` to `0.874, 0.885, 0.896` ($p \ge 0.289$), and single-step `Resume` (`N=8,100`) shows **`0.0` median loop delay** among mutual solvers across all $t$ (`+7.65` capped mean on $n=17$ at $t=8$ reflected `3/17` never-lock events with `0.0` median delay on the `14` mutual solvers). **The registered correlation gate (`Price(R_4)` vs. flips $\ge 0.25$) failed** ($\rho = +0.198, p = 0.011$; uncapped mutual-solver $\rho = +0.053, p = 0.508$). |
| **`H3.4`** (Path independence vs. path dependence + `C-voting`) | **Reframed via unconditional PI** | Fork agreement & `C-voting` gain | Unique-solution Sudoku algebraically forces `100%` agreement when conditioned on *solved* forks. Unconditional agreement across all $\binom{8}{2}=28$ fork pairs reveals **asymptotic path dependence** on stalled boards—mild on `gaoxin492` (`PI_board@64 = 0.867`, `OutcomeSplit@16 = 20.8%` at $\sigma=0.35$) and severe on `yagizdevre` (`PI_board@64 = 0.457` at $\sigma=0.35$, **`0.211`** at $\sigma=1.00$; `OutcomeSplit = 69.9%`–`84.7%`; OOD degradation `0.656 -> 0.590`). Replicating Kubo et al.'s (2026) `C-voting` lifts `WB-Stall@16` `0.836 -> 0.940` and `0.656 -> 0.858` by exploiting split outcomes, while Conditional Refill (`VC_C5`, `N=8,100`) **enforces path independence** on `yagizdevre` (`PI_board@64 -> 0.715 / 0.713`, `PI_cell@64 -> 0.938 / 0.937`), reversing OOD degradation (`0.667 -> 0.754`) in agreement with Anil et al. (2022). |
| **`H3.5`** (Causal $R_t$ swap & revisit signature) | **Partially supported** *(chart-local vs. global cycle split)* | `Swap-R_B > Zero-R` ($p<0.01$) on `WB-Stall` (null on `WB-Prop`) + elevated revisits on $R_t$ loss | `Swap_R_B` steers step-$(t+1)$ choices toward donor $B$ above `Zero_R` on `WB-Stall` (`+4.5, +3.7, +2.6` pp on primary, $p \le 0.014$ [vs. unrelated `Swap_R_C`: $p = 0.059\text{--}0.38$]; `+13.5, +11.8, +10.6` pp vs. `Zero_R` and `+14.1, +12.3, +11.7` pp vs. `Swap_R_C` on replication, all $p=0.0002$), and is null on `WB-Prop` ($p \ge 0.32$). Stale `Rewind_R_A1` raises 9-cell unit revisits ($p \le 0.0056$), whereas `Zero_R` (`VC`) raises *global 81-cell limit cycles* on primary (`1.6% -> 5.5%–9.3%`) while *reducing* them on replication (`25.1% -> 13.1%–14.8%`). |

---

## 2. Measurement 1: Conditional Refill, Architecture vs. Data Scaling, and the $z_H \times z_L$ Cross

### 2.1 Manifold Calibration & Stage 1 Architecture Ladder (`train_900` Fit)
Because every `TRMBlock` ends in `Post-RMSNorm` ($\|v\|_2 = \sqrt{512}$), each board token's head-orthogonal radius $\|z_{H,i}^\perp(t)\|_2 = \sqrt{512 - \|V_{t,i}\|_2^2}$ is deterministically fixed by $V_{t,i}$. Recombining $V_t$ with any sphere-calibrated refill $\hat{H}_\theta(x, V_t)$ achieves exact `float32` logit parity (`max_f32_logit_diff = 2.29e-5` on `gaoxin492`, `1.53e-5` on `yagizdevre`) and near-zero continuous hidden-perturbation leak (`1.49e-6` and `3.76e-6`). Without sphere calibration (`VC_C5_raw_uncalibrated`), conditional-mean Jensen shrinkage collapses `items256` `WB-Stall@16` to `0.612` (`gaoxin492`) and `0.415` (`yagizdevre`).

| Capacity tier $\mathcal{F}_k$ (`train_900` fit) | Primary `val_100` Joint $R^2$ (Prop / Stall) | Primary $R^2(z_H^\perp / z_L / z_H^{\text{pref}})$ | Repl. `val_100` Joint $R^2$ (Prop / Stall) | Repl. $R^2(z_H^\perp / z_L / z_H^{\text{pref}})$ |
|---|---:|---:|---:|---:|
| `C0_init` (`H_init`, `L_init` sphere fill) | `-2.104` (`-3.257` / `-1.986`) | `-0.690` / `-2.002` / `-4.818` | `-1.256` (`-1.369` / `-1.240`) | `-0.525` / `-1.285` / `-1.598` |
| `C0_mean` (global `train_900` sphere mean) | `0.000` (`0.000` / `0.000`) | `0.000` / `0.000` / `0.000` | `0.000` (`0.000` / `0.000`) | `0.000` / `0.000` / `0.000` |
| `C1_linear_local` (cell-local ridge) | `0.545` (`0.628` / `0.537`) | `0.380` / `0.542` / `0.759` | `0.335` (`0.323` / `0.337`) | `0.294` / `0.359` / `0.192` |
| `C2_linear_peer` (+ candidate mask & unit probs) | `0.573` (`0.645` / `0.566`) | `0.419` / `0.571` / `0.766` | `0.422` (`0.393` / `0.426`) | `0.378` / `0.456` / `0.205` |
| `C3_tf_2L_128` (`2L d=128` RoPE+SwiGLU) | `0.756` (`0.799` / `0.752`) | `0.722` / `0.753` / `0.837` | `0.640` (`0.589` / `0.647`) | `0.678` / `0.677` / `0.340` |
| `C4_tf_2L_256` (`2L d=256` RoPE+SwiGLU) | `0.803` (`0.846` / `0.799`) | `0.785` / `0.800` / `0.862` | `0.676` (`0.617` / `0.685`) | `0.747` / `0.713` / `0.360` |
| `C5_tf_2L_512` (`2L d=512`, `TRM` capacity match) | `0.844` (`0.888` / `0.840`) | `0.836` / `0.843` / `0.877` | `0.689` (`0.631` / `0.697`) | `0.774` / `0.726` / `0.356` |
| `C6_tf_4L_512` (`4L d=512`, `2×` `TRM` depth) | `0.863` (`0.909` / `0.858`) | `0.853` / `0.862` / `0.889` | `0.722` (`0.660` / `0.730`) | `0.813` / `0.760` / `0.370` |
| `C5_tf_2L_512_stepclock` (+ explicit step $t$) | `0.822` (`0.863` / `0.817`) | `0.815` / `0.818` / `0.872` | — | — |

### 2.2 Stage 4 Data-Size Scaling Curve (`C5_tf_2L_512` across `N = 100..8,100` Disjoint Puzzles)
Saturating model depth at fixed `N = 900` only proves architectural saturation on `900` boards (`14,400` states), not data saturation. Stage 4 fits `C5_tf_2L_512` across $N_{\text{train}} \in \{100, 300, 900, 2700, 8100\}$ puzzles (`1,600` to `129,600` rollout states) guaranteed disjoint from `val_100`, `items256`, and `bench100` under both exact string and canonical digit-relabelling isomorphism:

| Checkpoint & $N_{\text{train}}$ (`16 N` states) | `val_100` Joint $R^2$ (Prop / Stall) | $R^2(z_{H,\perp} / z_L)$ | `items256` Prop `@16 / @64` | `items256` Stall `@16 / @32 / @64` | `items256` Stall Cycle% | `bench100` Stall `@16 / @64` |
|---|---:|---:|---:|---:|---:|---:|
| **Primary `N = 100`** (`1,600`) | `0.632` (`0.707` / `0.624`) | `0.544 / 0.630` | `0.959 / 0.973` | `0.678 / 0.689 / 0.699` | `26.8%` | `0.961 / 0.961` |
| **Primary `N = 300`** (`4,800`) | `0.754` (`0.802` / `0.749`) | `0.729 / 0.752` | `0.986 / 1.000` | `0.721 / 0.732 / 0.749` | `17.5%` | `0.961 / 0.980` |
| **Primary `N = 900`** (`14,400`) | `0.844` (`0.888` / `0.840`) | `0.836 / 0.843` | `1.000 / 1.000` | `0.760 / 0.776 / 0.792` | `9.3%` | `0.980 / 1.000` |
| **Primary `N = 2,700`** (`43,200`) | `0.889` (`0.930` / `0.885`) | `0.887 / 0.885` | `1.000 / 1.000` | `0.765 / 0.798 / 0.820` | `6.6%` | `0.980 / 0.980` |
| **Primary `N = 8,100`** (`129,600`) | **`0.915` (`0.952` / `0.912`)** | **`0.915 / 0.912`** | **`1.000 / 1.000`** | **`0.787 / 0.814 / 0.825`** | **`5.5%`** | **`1.000 / 1.000`** |
| *Primary `Clean` reference* | — | — | `1.000 / 1.000` | `0.836 / 0.863 / 0.896` | `1.6%` | `1.000 / 1.000` |
| **Repl. `N = 900`** (`14,400`) | `0.689` (`0.631` / `0.697`) | `0.774 / 0.726` | `1.000 / 0.986` | `0.683 / 0.721 / 0.743` | `14.8%` | `0.980 / 0.980` |
| **Repl. `N = 8,100`** (`129,600`) | **`0.837` (`0.798` / `0.842`)** | **`0.885 / 0.850`** | `0.973 / 0.945` | **`0.661 / 0.721 / 0.760`** | **`13.1%`** | **`0.980 / 0.980`** |
| *Repl. `Clean` reference* | — | — | `0.877 / 0.890` | `0.656 / 0.617 / 0.590` | `25.1%` | `0.922 / 0.843` |

`N = 900` was indeed still data-limited: `9×` more training puzzles raises primary held-out $R^2$ from `0.844` to **`0.915`** (`95.2%` on `WB-Prop`, `91.2%` on `WB-Stall`), cuts closed-loop primary limit cycles from `9.3%` to `5.5%`, and narrows the every-step closed-loop primary `WB-Stall` deficit by ~`35%` (`-7.7 -> -4.9` pp at $s=16$; `-10.4 -> -7.1` pp at $s=64$), while raising replication `val_100` $R^2$ from `0.689` to **`0.837`** and lifting replication `WB-Stall@64` to **`0.760` (`+16.9` pp above `0.590` clean)**.

### 2.3 Closed-Loop $z_H \times z_L$ Cross (`N = 900`) & Periodic Refill Intervals $m \in \{1, 2, 4, 8\}$ (`N = 8,100`)

In Stage 1 (`train_900`), single-channel ablations (`live_zH__mean_zL = 0.792 / 0.852`, `mean_all_zH__live_zL = 0.749 / 0.836`) retain high solve rates because $z_H$ and $z_L$ mutually reconstruct each other, whereas joint `V_only` (`0.617 / 0.650` on primary, `0.519 / 0.546` on replication) traps `33%`–`45%` of stalled trajectories in deterministic limit cycles.

However, closed-loop every-step refill ($m=1$) feeds $\hat{H}_\theta$ states generated by its *own* modified rollout, compounding off-distribution covariate shift over 16–64 steps. Stage 4 tests refilling every $m \in \{1, 2, 4, 8\}$ steps using the `N = 8,100` `C5` map (`bfloat16` autocast):

| Refill interval (`N = 8,100` `C5`) | Prim. Prop `@16 / @64` | Prim. Stall `@16 / @32 / @64` | Prim. Cycle% `@64` | Prim. McNemar $p_{16} / p_{64}$ | Repl. Prop `@16 / @64` | Repl. Stall `@16 / @32 / @64` | Repl. Cycle% `@64` | Repl. McNemar $p_{16} / p_{64}$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` ($m = \infty$) | `1.000 / 1.000` | `0.836 / 0.863 / 0.896` | `1.6%` | — | `0.877 / 0.890` | `0.656 / 0.617 / 0.590` | `25.1%` | — |
| `refill_every_1` ($m = 1$) | `1.000 / 1.000` | `0.803 / 0.820 / 0.825` | `8.2%` | `0.180 / 2.4e-4` | `0.959 / 0.973` | `0.650 / 0.732 / 0.754` | `13.7%` | `1.00 / 1.5e-5` |
| `refill_every_2` ($m = 2$) | `1.000 / 1.000` | **`0.814 / 0.847 / 0.874`** | **`2.7%`** | `0.388 / 0.289` | `0.932 / 0.932` | **`0.678 / 0.716 / 0.781`** | **`9.8%`** | `0.659 / 1.2e-6` |
| `refill_every_4` ($m = 4$) | `1.000 / 1.000` | **`0.825 / 0.863 / 0.885`** | **`1.1%`** | `0.804 / 0.688` | `0.932 / 0.959` | `0.656 / 0.667 / 0.699` | `16.9%` | `1.00 / 0.0055` |
| `refill_every_8` ($m = 8$) | `1.000 / 1.000` | **`0.842 / 0.869 / 0.896`** | **`1.1%`** | `1.00 / 1.00` | `0.877 / 0.877` | `0.623 / 0.628 / 0.710` | `16.4%` | `0.461 / 0.0016` |

Allowing just **one** unprojected recurrent step between refills ($m=2$) eliminates statistical separation from `clean` on primary `gaoxin492` (`0.874` vs. `0.896`, $p = 0.289$; `0.896` at $m=8$, $p = 1.00$) while achieving the highest stalled solve rate on `yagizdevre` (**`0.781` at $s=64$, `+19.1` pp above `0.590` clean**, $p = 1.2 \times 10^{-6}$, cutting limit cycles from `25.1%` to `9.8%`). Thus `gaoxin492` does not carry long-horizon multi-step hidden tabu state across $4\text{--}8$ steps; its `-4.9` to `-7.1` pp every-step ($m=1$) gap reflects closed-loop approximation drift and single-step local damping when $H_t$ is overwritten on every iteration.

---

## 3. Measurement 2: Resume Test — Unselected Never-Lock Rates & Uncapped Solver Delays

### 3.1 Additive Accounting Decomposition & `LoopedChase` Readout Blind Spot (`12` `Q4` Arms)
For any stationarity-admitted operator with horizon cap $H = 32$, $\min(t, \tau_0(x)) = [\tau_0(x) - \tau(V_t)] + [\tau(V_t) - \tau(V_t \oplus C_t)] + [\tau(V_t \oplus C_t) - \tau(S_t)]$ holds by algebraic telescoping of the intermediate lock times (an accounting definition, not an empirical identity).
- **`LoopedChase` $\lambda = 0.1$ (`3/3` extrapolating seeds):** All-token `V_only` fails (`0.172`), whereas all-token `VC` restores **`1.000` (`3/3` seeds)**: `Price(C_4) = +5.02` loops and **`Price(R_4) = 0.00` loops (`0.00` on every seed)**.
- **`LoopedChase` $\lambda = 1.0$ (`3/3` extrapolating seeds) — Readout Blind Spot:** Seed 3 is pure head-readable (`V_only = 1.000`, `Price(C_4) = 0.00`, `Price(R_4) = 0.00`), whereas seeds 1–2 route intermediate hop state through $R_t$ (**`Price(R_4) = +3.06` and `+5.46` loops**; mean `+2.84`), directly instantiating Sharma & Vu's (2026) finding that dense intermediate logit supervision (`\lambda = 1.0`) still leaves the 48-D head-orthogonal fiber unconstrained against routing causally active state off-axis.

### 3.2 Stage 4 Covariate-Shift-Free `TRM` `Resume` Audit (`N = 8,100` `C5`, Horizon $H = 32$)
Unlike closed-loop refill, single-step `Resume` evaluates $\hat{H}_\theta(x, V_t)$ **once on exact clean rollout states**, eliminating closed-loop covariate shift. Stage 2's capped mean (`+7.65` loops on $n=17$ active clean-locking boards at $t=8$) mixed non-locking failures (`\tau = 33`) with solver delays under post-hoc selection (`\tau_0 > t`). Stage 4 separates (a) the **fraction of boards that never lock within $H=32$** across unselected and conditional cohorts from (b) the **uncapped median (and IQR / mean) loop delay strictly among boards locking within $H=32$ under both compared arms**:

| Checkpoint, Step $t$, & Cohort (`items256`, `N=8,100`) | $n$ | `NeverLock` (`V` / `VC` / `Full`) | McNemar $p$ (`VC` vs. `Full`) | Mutual Lockers $n_{\text{both}}$ | Uncapped $R_t$ Delay (`VC - Full`): **Median** `[IQR]` (`Mean`, $p$) | Uncapped $C_t$ Delay (`V - VC`): **Median** `[IQR]` (`Mean`, $p$) |
|---|---:|---:|---:|---:|---:|---:|
| **Primary `WB-Prop` ($t=1$)** | `73` | `0.000 / 0.000 / 0.000` | `1.00` | `73` | **`+0.0`** `[0.0, 0.0]` (`+0.00`, `1.00`) | `+0.0` `[0.0, 0.0]` (`+0.16`, `0.10`) |
| **Primary `WB-Prop` ($t=4, 8, 12$)** | `73` | `0.000 / 0.000 / 0.000` | `1.00` | `73` | **`+0.0`** `[0.0, 0.0]` (`+0.00`, `1.00`) | `+0.0` `[0.0, 0.0]` (`+0.00`, `1.00`) |
| **Primary Unselected `WB-Stall` ($t=1$)** | `183` | `0.126 / 0.120 / 0.131` | `0.727` | `156` | **`+0.0`** `[-1.0, 0.0]` (`-0.30`, `0.51`) | `+0.0` `[0.0, 1.0]` (`-0.42`, `0.24`) |
| **Primary Unselected `WB-Stall` ($t=2$)** | `183` | `0.142 / 0.142 / 0.131` | `0.754` | `153` | **`+0.0`** `[0.0, 0.0]` (`-0.33`, `0.32`) | `+0.0` `[0.0, 0.0]` (`+0.17`, `0.56`) |
| **Primary Unselected `WB-Stall` ($t=4$)** | `183` | `0.137 / 0.148 / 0.126` | `0.219` | `155` | **`+0.0`** `[0.0, 0.0]` (`+0.17`, `0.71`) | `+0.0` `[0.0, 0.0]` (`+0.01`, `0.99`) |
| **Primary Unselected `WB-Stall` ($t=8$)** | `183` | `0.109 / 0.126 / 0.115` | `0.754` | `156` | **`+0.0`** `[0.0, 0.0]` (`-0.24`, `0.24`) | `+0.0` `[0.0, 0.0]` (`-0.16`, `0.62`) |
| **Primary Unselected `WB-Stall` ($t=12$)** | `183` | `0.104 / 0.104 / 0.115` | `0.727` | `159` | **`+0.0`** `[0.0, 0.0]` (`-0.14`, `0.34`) | `+0.0` `[0.0, 0.0]` (`+0.08`, `0.77`) |
| Primary Unresolved-at-$t$ `WB-Stall` ($t=4$) | `64` | `0.391 / 0.422 / 0.359` | `0.219` | `36` | **`+0.0`** `[-2.0, +4.0]` (`+0.72`, `0.70`) | `+0.0` `[-1.8, +1.8]` (`-0.68`, `0.58`) |
| Primary Unresolved-at-$t$ `WB-Stall` ($t=8$) | `42` | `0.476 / 0.548 / 0.500` | `0.754` | `15` | **`+0.0`** `[-3.5, +0.5]` (`-2.53`, `0.24`) | `+1.0` `[-11.5, +4.5]` (`-1.39`, `0.64`) |
| Primary Active $\tau_0 > 8$ (`N=900` / `N=8,100`) | `17` | `0.176` / `0.235` (`VC`) vs. `0.000` (`Full`) | `0.250` / `0.125` | `14` / `13` | **`+0.0` / `+0.0`** (`-0.71` / `-0.23`, $p \ge 0.82$) | `+0.0` / `+2.0` (`+2.50` / `+1.23`, $p \ge 0.48$) |
| **Repl. Unselected `WB-Stall` ($t=4$)** | `183` | `0.322 / 0.361 / 0.328` | `0.471` | `96` | **`+0.0`** `[-10.5, +8.0]` (`-0.58`, `0.71`) | `+0.0` `[-10.0, +11.0]` (`+0.36`, `0.82`) |
| **Repl. Unselected `WB-Stall` ($t=8$)** | `183` | `0.393 / 0.317 / 0.377` | `0.135` | `97` | **`+0.0`** `[-5.0, +12.0]` (`+2.00`, `0.17`) | `+0.0` `[-8.0, +10.0]` (`+1.29`, `0.44`) |
| **Repl. Unresolved-at-$t$ `WB-Stall` ($t=4$)** | `155` | `0.368 / 0.394 / 0.387` | `1.00` | `73` | **`-2.0`** `[-14.0, +4.0]` (**`-5.12`, $p=0.0014$**) | `+1.0` `[-7.0, +11.0]` (`+1.29`, `0.46`) |
| **Repl. Unresolved-at-$t$ `WB-Stall` ($t=8$)** | `152` | **`0.421 / 0.349 / 0.454`** | **`0.0166`** | `71` | `+1.0` `[-10.0, +9.0]` (`-0.97`, `0.54`) | `+0.0` `[-11.0, +7.5]` (`-0.15`, `0.94`) |
| **Repl. Unresolved-at-$t$ `WB-Stall` ($t=12$)** | `148` | `0.405 / 0.405 / 0.426` | `0.766` | `64` | **`-4.0`** `[-13.2, +2.0]` (**`-5.03`, $p=0.0010$**) | `+0.0` `[-10.0, +8.0]` (`-0.59`, `0.75`) |
| **Repl. `bench100` Unresolved `WB-Stall` ($t=4$)** | `39` | `0.128 / 0.077 / 0.179` | `0.289` | `30` | **`-8.0`** `[-25.8, -1.2]` (**`-11.37`, $p=0.0004$**) | — |

Key lessons from separating `never_lock` and uncapped delay:
1. **Why Stage 2's `+7.65` loops on $n=17$ was an artifact of capping + selection:** On `gaoxin492`'s $n=17$ active clean-locking boards at $t=8$ (where `Full` has `0.000` `never_lock` *by conditioning on clean lock*), `3/17` (`N=900`) and `4/17` (`N=8,100`) `VC` continuations fail to lock within 32 steps ($p = 0.25, 0.125$), whereas **among the `14/17` (`13/17`) boards that lock under both arms, the uncapped median delay is `0.0` loops** (`mean = -0.71` and `-0.23` loops, $p \ge 0.82$)! Across the **unselected `WB-Stall` cohort ($n=183$)**, single-step `VC` (`N=8,100`) changes `never_lock` by at most `+2.2` pp (`p >= 0.219`) with **`0.0` median loop delay at every step $t \in \{1, 2, 4, 8, 12\}$**.
2. **Replication `yagizdevre` shows genuine single-step acceleration on unresolved boards:** Without any closed-loop compounding, a single clean-state refill on `yagizdevre`'s unresolved `WB-Stall` boards reduces `never_lock` by `-10.5` pp at $t=8$ (`0.454 -> 0.349`, exact McNemar $p = 0.0166$) and accelerates mutual solvers by a median of **`-2.0` loops at $t=4$ ($p=0.0014$)** and **`-4.0` loops at $t=12$ ($p=0.0010$)** (`-8.0` loops on `bench100` at $t=4$, $p=0.0004$).
3. **Registered puzzle-level correlation gate failed:** Pre-registered correlation between single-step $\text{Price}(R_4)$ and clean cell-flip count on `WB-Stall` reached $\rho = +0.198$ ($p = 0.011$), below the `0.25` threshold, and uncapped mutual-solver delay correlation at $t=4$ (`N=8,100`) is $\rho = +0.053$ ($p = 0.508$). We do not substitute $\|R_t\|_2^2 / \|H_t\|_2^2$ because prediction-error norm mechanically rises on harder search trajectories even when $R_t$ is inert.

---

## 4. Measurement 3: Unconditional vs. Enforced Path Independence (`K = 8` Forks), `C-Voting`, Causal $R_t$ Swaps, and `Huginn-0125`

### 4.1 Unconditional & Enforced Path Independence Across `K = 8` Forks (Anil et al., 2022; Kubo et al., 2026)
Conditioning fork agreement on *solved* trajectories (`1.000`) is tautological on unique-solution Sudoku. Evaluating **unconditional** exact-board agreement ($\mathrm{PI}_{\text{board}}$), empty-cell agreement ($\mathrm{PI}_{\text{cell}}$), normalized hidden-state cosine ($\mathrm{PI}_{\text{hid}}$), and the fraction of boards with split outcomes across the `K = 8` forks (`OutcomeSplit`: solved on $\ge 1$ fork and unsolved on $\ge 1$ fork) under **Clean** vs. **Enforced Path Independence** (`VC_C5`, `N = 8,100`) directly tests Anil et al. (2022):

| Checkpoint, Noise $\sigma$, & Subset (`items256`) | Rollout Mode | $\mathrm{PI}_{\text{board}}$ `@16 / @64` | $\mathrm{PI}_{\text{cell}}$ `@16 / @64` | $\mathrm{PI}_{\text{hid}}$ `@16 / @64` | `OutcomeSplit` `@16 / @64` | Canonical `@16 / @64` | Fork Mean `@16 / @64` | Kubo `C-Voting` `@16 / @64` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Primary `WB-Prop` ($\sigma=0.35$)** | `Clean` | `1.000 / 0.997` | `1.000 / 0.999` | `0.997 / 0.995` | `0.000 / 0.014` | `1.000 / 1.000` | `1.000 / 0.998` | `1.000 / 1.000` |
| **Primary `WB-Prop` ($\sigma=0.35$)** | `Enforced VC` | `0.993 / 0.997` | `0.998 / 0.999` | `0.996 / 0.998` | `0.027 / 0.014` | `1.000 / 1.000` | `0.997 / 0.998` | `1.000 / 1.000` |
| **Primary `WB-Stall` ($\sigma=0.35$)** | `Clean` | **`0.794 / 0.867`** | **`0.938 / 0.957`** | `0.936 / 0.954` | `0.208 / 0.137` | **`0.836 / 0.896`** | `0.832 / 0.897` | **`0.940 / 0.967`** |
| **Primary `WB-Stall` ($\sigma=0.35$)** | `Enforced VC` | `0.744 / 0.798` | `0.937 / 0.948` | `0.939 / 0.949` | `0.180 / 0.137` | `0.792 / 0.842` | `0.775 / 0.825` | `0.874 / 0.902` |
| **Primary `WB-Stall` ($\sigma=1.00$)** | `Clean` | `0.748 / 0.840` | `0.914 / 0.940` | `0.906 / 0.934` | `0.393 / 0.273` | `0.836 / 0.896` | `0.814 / 0.887` | **`0.962 / 0.973`** |
| **Primary `WB-Stall` ($\sigma=1.00$)** | `Enforced VC` | `0.682 / 0.760` | `0.899 / 0.921` | `0.906 / 0.929` | `0.383 / 0.279` | `0.792 / 0.842` | `0.742 / 0.805` | `0.896 / 0.918` |
| **Repl. `WB-Prop` ($\sigma=0.35$)** | `Clean` | `0.757 / 0.728` | `0.947 / 0.940` | `0.801 / 0.786` | `0.658 / 0.712` | `0.877 / 0.890` | `0.868 / 0.851` | `1.000 / 1.000` |
| **Repl. `WB-Prop` ($\sigma=0.35$)** | `Enforced VC` | **`0.918 / 0.927`** | **`0.989 / 0.990`** | **`0.916 / 0.922`** | **`0.205 / 0.164`** | **`0.932 / 0.973`** | **`0.950 / 0.959`** | `1.000 / 1.000` |
| **Repl. `WB-Stall` ($\sigma=0.35$)** | `Clean` | **`0.446 / 0.457`** | **`0.838 / 0.839`** | `0.772 / 0.767` | **`0.699 / 0.743`** | `0.656 /` **`0.590`** | `0.593 / 0.605` | **`0.847 / 0.874`** |
| **Repl. `WB-Stall` ($\sigma=0.35$)** | `Enforced VC` | **`0.617 / 0.715`** | **`0.911 / 0.938`** | **`0.865 / 0.899`** | **`0.415 / 0.240`** | **`0.667 / 0.754`** | **`0.691 / 0.754`** | `0.842 / 0.842` |
| **Repl. `WB-Stall` ($\sigma=1.00$)** | `Clean` | **`0.191 / 0.211`** | **`0.591 / 0.603`** | `0.663 / 0.671` | **`0.847 / 0.847`** | `0.656 /` **`0.590`** | `0.389 / 0.410` | **`0.852 / 0.842`** |
| **Repl. `WB-Stall` ($\sigma=1.00$)** | `Enforced VC` | **`0.613 / 0.713`** | **`0.913 / 0.937`** | **`0.865 / 0.891`** | **`0.383 / 0.306`** | **`0.667 / 0.754`** | **`0.684 / 0.759`** | `0.831 / 0.863` |

1. **Asymptotic path dependence explains both Kubo et al.'s `C-voting` lift and `yagizdevre`'s OOD iteration degradation:** On `yagizdevre`, `69.9%` ($\sigma=0.35$) to `84.7%` ($\sigma=1.00$) of `WB-Stall` boards end solved on some forks and trapped on others (`PI_board@64 = 0.457` and `0.211`), and running past the training count ($s=16 \to 64$) degrades canonical solve (`0.656 -> 0.590`)—the exact failure mode Anil et al. (2022) identify for path-dependent equilibrium models. Even `gaoxin492` exhibits moderate asymptotic path dependence on its hardest stalled boards (`OutcomeSplit@16 = 20.8%` at $\sigma=0.35$, `39.3%` at $\sigma=1.00$), which replicating Kubo et al.'s (2026) `C-voting` converts into `+10.4` to `+12.6` pp gains (`0.836 -> 0.940 / 0.962`, matching the `K=8` oracle union `100%`).
2. **Enforcing path independence via Conditional Refill (`VC_C5`) helps the path-dependent checkpoint and predicts board-level solve & refill gain:**
   - Across `WB-Stall` ($n=183$), per-board clean $\mathrm{PI}_{\text{cell}}(16)$ strongly predicts clean fork solve rate on **both** checkpoints ($\rho = +0.970$ on `gaoxin492`, $\rho = +0.720$ [$\sigma=0.35$] and $+0.574$ [$\sigma=1.00$] on `yagizdevre`, all $p = 2\times 10^{-4}$).
   - On `yagizdevre`, enforcing $H_t \leftarrow \hat{H}_\theta(x, V_t)$ lifts `PI_cell@64` by **`+9.9` pp** ($\sigma=0.35$) and **`+33.4` pp** ($\sigma=1.00$, both $p = 2\times 10^{-4}$) and `PI_board@64` by **`+25.8` to `+50.2` pp**, restoring monotonic OOD iteration scaling (`0.667 -> 0.754`). Per-board clean $\mathrm{PI}_{\text{cell}}(64)$ negatively predicts `VC` gain ($\rho = -0.354$ at $\sigma=0.35$, $\rho = -0.408$ at $\sigma=1.00$, $p \le 4\times 10^{-4}$): in the lowest-PI quartile (`Q1`), `VC` boosts canonical `s=64` solve from **`0.196 -> 0.587` (`+39.1` pp)** ($\sigma=0.35$) and mean fork solve at $\sigma=1.00$ from **`0.269 -> 0.726` (`+45.7` pp)**, vs. `-2.2` pp in `Q4` (`0.933 -> 0.911`).
   - On `gaoxin492`, `Q2–Q4` (`137/183` stalled boards) already achieve clean $\mathrm{PI}_{\text{cell}}(64) = 1.000$ and `1.000` canonical solve (`1.000` under `VC` in `Q2–Q3`), while the every-step closed-loop loss concentrates in `Q1` (`0.587 -> 0.370`, `-21.7` pp).

### 4.2 Causal $R_t$ Swap & `Huginn-0125` Audit
- **Matched-step causal $R_t$ swap (`items256` `WB-Stall`, $t \in \{3, 5, 8\}$):** Null on `WB-Prop` ($p \ge 0.32$). On `WB-Stall`, `Swap_R_B` steers step-$(t+1)$ commitments toward donor $B$ above `Zero_R` on both checkpoints (`+4.50, +3.71, +2.59` pp on primary, $p \le 0.0136$ [vs. `Swap_R_C`: $p = 0.059\text{--}0.38$]; `+13.50, +11.84, +10.62` pp vs. `Zero_R` and `+14.13, +12.28, +11.69` pp vs. `Swap_R_C` on replication, all $p = 0.0002$). Stale `Rewind_R_A1` raises 9-cell unit revisits ($p \le 0.0056$), whereas `Zero_R` alters global 81-cell limit-cycle rates (`1.6% -> 5.5%–9.3%` on primary; `25.1% -> 13.1%–14.8%` on replication) without raising local 9-cell unit revisits.
- **`Huginn-0125` (`32` prompts $\times$ `K = 4` random Gaussian starts, truncated BPTT $k=8$):** Exact stationarity ($L_\infty = 0.0$); path-dependent variance share contracts from `1.007` ($t=0$) to `0.029` ($t=8$), `0.0041` ($t=16$), and `0.00057` ($t=32$, `99.94%` path-independent cache); leave-one-out `CacheMean` at $t=8$ matches clean top-1 predictions `32/32` (`100%`), consistent with Geiping et al.'s (2025) random-init warmup + truncated-$k=8$ backpropagation training regime.
