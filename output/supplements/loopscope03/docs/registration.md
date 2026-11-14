# LOOPSCOPE-03 Pre-Registration: Three-Ledger Iteration Pricing, Conditional Cache Refill, and Path-Dependent Search Memory

**Date:** 2026-11-14  
**Status:** Locked prior to any `LOOPSCOPE-03` held-out (`items256` / `bench100`) inference  
**Predecessor studies:** `LOOPSCOPE-01` (`Q1`–`Q11`), `LOOPSCOPE-02` (`P1`–`P5`, `v010`–`v012` controls)  
**Checkpoints under test (frozen, zero retraining of base models):**
- `gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att` (`step_21700`, SHA-256 `932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3`)
- `yagizdevre/trm-sudoku-extreme-att` (`model.pt`, SHA-256 `d67723394a7fe439f1fa498b58196399ee89cd82a794c7da9ce0a8cc3df7fa39`)
- `LoopedChase` (`12` `Q4` checkpoints across $\lambda \in \{0, 0.01, 0.1, 1.0\}$, `3` seeds each)
- `Nanbeige/Nanbeige4.2-3B` (stationarity admission control)

---

## 1. Motivation: Why Final-Budget Erasure Cannot Price Where Reasoning Accrued

In a stationary, input-re-injected loop $S_{t+1} = F_x(S_t)$, erasing part of $S_t$ and scoring accuracy only at a fixed terminal step ($t = 16$) confounds two distinct phenomena:
1. **Recomputable state** that the operator rebuilds in $\Delta\tau \ge 1$ additional iterations (e.g., `10`-cell donor corruption repaired to `0.887`; `reset_zL` retaining `0.832`–`0.990`), and
2. **Backup hidden channels** when only one hidden buffer is ablated (`v010`–`v012` manifold controls erased $(I - P_{\text{head}})z_H^{\text{board}}$ while leaving `z_L` and the `16` prefix tokens `z_H[:, :16, :]` live across steps).

`LOOPSCOPE-03` replaces fixed-budget erasure with **iteration pricing** across **three disjoint state ledgers** $S_t = V_t \oplus C_t \oplus R_t$:

- **$V_t$ (Visible / Head-Readable Ledger):** $z_H^{\text{board}}(t) P_{\text{head}} \in \mathbb{R}^{81 \times 512}$ ($\text{rank}(P_{\text{head}}) = 11$), which determines all step-$t$ output logits identically.
- **$C_t$ (Conditional Cache Ledger):** $\hat{H}_\theta(x, V_t)$, the part of the full hidden state $H_t = \big(z_H^{\text{board}}(t) P_\perp,\; z_H^{\text{pref}}(t),\; z_L(t)\big)$ predictable from the puzzle clues $x$ and current visible state $V_t$ alone (fitted strictly on `train.csv` with zero test-board overlap). Because an external monitor observes $(x, V_t)$, anything in $V_t \oplus C_t$ is **monitorable in principle**.
- **$R_t$ (Path-Dependent Memory Ledger):** $R_t = H_t - \hat{H}_\theta(x, V_t)$, the trajectory-dependent residual that cannot be reconstructed from $(x, V_t)$. **Only $R_t$ is genuinely dark.**

Because `TRMBlock` terminates in `Post-RMSNorm` ($\|v\|_2 = \sqrt{512}$ for every token of $z_H(t)$ and $z_L(t)$), orthogonal radius $\|z_{H,i}^\perp(t)\|_2 = \sqrt{512 - \|V_{t,i}\|_2^2}$ is deterministically fixed by $V_{t,i}$. All refill predictors $\hat{H}_\theta(x, V_t)$ therefore output sphere-calibrated vectors ($\hat{z}_{H,i}^\perp \in \ker(\texttt{lm\_head})$ scaled to $\sqrt{512 - \|V_{t,i}\|_2^2}$, and `rms_norm` on $\hat{z}_H^{\text{pref}}, \hat{z}_L$), eliminating off-manifold Jensen shrinkage.

---

## 2. Three Measurements

### Measurement 1: Conditional Refill, Capacity Sweep, and the $z_L \times z_H$ Cross (`scripts/conditional_refill.py`)
1. **Strict train/val/test separation:** Split `train.csv` (`1,000` puzzles, verified disjoint from `items256` and `bench100`) into `train_900` (`14,400` step states) and `val_100` (`1,600` step states).
2. **Capacity ladder ($\mathcal{F}_0 \subset \dots \subset \mathcal{F}_6$):** Sweep refill capacity across:
   - `C0_init` (`0` params: `H_init`/`L_init` sphere fill),
   - `C0_mean` (`train_900` global mean sphere fill, zero `V_t` dependence),
   - `C1_linear_local` (cell-local ridge on $(x_i, V_{t,i})$ with no peer visibility),
   - `C2_linear_peer` (board-peer ridge adding exact Sudoku candidate masks and row/col/box `probs` aggregates),
   - `C3_tf_2L_128` (`2`-layer `d=128` RoPE+SwiGLU 97-token Transformer),
   - `C4_tf_2L_256` (`2`-layer `d=256` RoPE+SwiGLU Transformer),
   - `C5_tf_2L_512` (`2`-layer `d=512` RoPE+SwiGLU Transformer — exact architectural match to `TRM`'s 2-layer `d=512` core),
   - `C6_tf_4L_512` (`4`-layer `d=512` RoPE+SwiGLU Transformer — $2\times$ `TRM` depth/capacity), plus an auxiliary `C5_tf_2L_512_stepclock` control receiving step index $t$.
3. **Full $z_H \times z_L$ cross & extended horizon ($T = 16, 32, 64$):** Cross `{Live, UncondMean, CondRefill}` on $z_H$ hidden (`board_orth + prefix`, plus `board_orth` vs `prefix` isolation) against `{Live, Reset_L_init, UncondMean, CondRefill}` on $z_L$. Track exact solve rates at $s \in \{16, 32, 64\}$ and periodic visible-board cycle/revisit rates across $s = 1..64$.

### Measurement 2: Resume Test — Iteration Pricing in Loop-Equivalents (`scripts/resume_and_fork.py`)
1. **Time-invariance admission gate:** Require operator outer-step stationarity $\bar{r}_{\text{outer}} \ge 0.90$ and exact state-transplant invariance $\|F_x^{(u)}(S_t) - S_{t+u}\|_\infty < 10^{-5}$. Verify that `Nanbeige4.2-3B` fails admission ($\bar{r} = 0.355$, accuracy degrades past $t=2$), while `LoopedChase` and `TRM` pass identically.
2. **Telescoping ledger prices:** At transplant steps $t \in \{1, 2, 4, 8, 12\}$, transplant $\emptyset$ (`Fresh`), $V_t$ (`Uncond`), $V_t \oplus C_t$ (`CondRefill`), or $V_t \oplus C_t \oplus R_t$ (`Full`) into a fresh unperturbed run of horizon $T_{\max} = 32$ steps, recording steps-to-solve $\tau_{\text{res}}(x; t, L)$ and verifying the exact telescoping decomposition:
   $$\min(t, \tau_0(x)) = \underbrace{[\tau_0 - \tau_{\text{res}}(V_t)]}_{\text{Accrued}(V_t)} + \underbrace{[\tau_{\text{res}}(V_t) - \tau_{\text{res}}(V_t \oplus C_t)]}_{\text{Price}(C_t)} + \underbrace{[\tau_{\text{res}}(V_t \oplus C_t) - \tau_{\text{res}}(S_t)]}_{\text{Price}(R_t)}.$$
   Test Spearman correlation between puzzle-level $\text{Price}(R_t)$ and clean conflict-flip / backtrack counts across boards.

### Measurement 3: Fork Test — Multi-Init Path Dependence, $R_t$ Swap, and Refuted-Board Revisit Rate (`scripts/resume_and_fork.py`)
1. **Multi-initialization C-voting ensemble ($K = 8$ forks per puzzle):** Initialize $(z_H^{(m)}(0), z_L^{(m)}(0))$ with perturbed `rms_norm`-calibrated latent starts ($m = 0$ canonical, $m = 1..7$ Gaussian forks). Measure:
   - Asymptotic steady-state agreement among solved forks vs. **transient path divergence** (pairwise empty-cell Hamming distance at steps $t = 1..16$) on `WB-Prop` vs. `WB-Stall` (testing reconciliation of Anil et al. 2022 asymptotic path independence with transient post-stall path dependence), and
   - **C-voting** accuracy (selecting fork $m^*$ with highest mean top-1 visible confidence) vs. canonical single-run accuracy.
2. **Causal $R_t$ swap & refuted-state revisit rate:** At matched steps $t \in \{3, 5, 8\}$ on forked runs $(A, B)$ of the same board $x$, construct continuations sharing identical visible logits $V_t^{(A)}$:
   - `Keep-R_A`: $V_t^{(A)} \oplus C_t^{(A)} \oplus R_t^{(A)}$,
   - `Zero-R`: $V_t^{(A)} \oplus C_t^{(A)} \oplus 0$ (both one-shot at $t$ and persistent $\ge t$),
   - `Swap-R_B`: $V_t^{(A)} \oplus C_t^{(A)} \oplus R_t^{(B)}$ (sphere-projected in $\ker(\texttt{lm\_head})$),
   - `Rewind-R_A(s=1)`: $V_t^{(A)} \oplus C_t^{(A)} \oplus R_1^{(A)}$.
   Score (i) **donor trajectory steering** toward $\hat{y}_{t+1}^{(B)}$ on $(A, B)$-disagreement cells and (ii) **revisit rates of $A$'s already-refuted states** ($1 \le s < t$, abandoned by $t$, revisited at $s' > t$) at full-board, 9-cell Sudoku unit, and wrong-digit cell granularities.

---

## 3. Pre-Registered Predictions & Falsification Criteria

- **H3.1 ($z_L \times z_H$ Backup-Channel Hypothesis):** Joint unconditional erasure of *both* $z_H$ hidden fiber and $z_L$ (`V`-only) degrades `items256` `WB-Stall` substantially more than single-channel `z_H`-only or `reset_zL`-only erasure ($\ge 15$ pp drop vs. clean on `gaoxin492`), demonstrating cross-buffer redundancy when only one channel is ablated.
- **H3.2 (Propagation Cache Sufficiency):** On `LoopedChase` ($\lambda \ge 0.1$) and `TRM` `WB-Prop` (`73` `items256` + `49` `bench100` boards), conditional refill $V_t \oplus C_t$ recovers $\ge 97\%$ of clean exact solves by $s=16$ (`100%` by $s=32$), and Resume-Test $\text{Price}(R_t)$ averages $< 0.25$ outer steps.
- **H3.3 (Post-Stall Non-Null $R_t$ & Capacity Saturation):** On `TRM` `WB-Stall`, held-out `val_100` refill $R^2$ saturates across `C4`–`C6` (`|R^2(C6) - R^2(C5)| < 0.02`), yet persistent `V + C` refill leaves a statistically significant stalled-board deficit ($p < 0.01$ at $s=16$) and/or positive Resume-Test $\text{Price}(R_t) > 0.5$ outer steps on stalled boards that correlates positively ($\rho_{\text{Spearman}} > 0.25$, $p < 0.01$) with puzzle conflict-flip count. *(If `C5`/`C6` conditional refill closes the stalled deficit completely and $\text{Price}(R_t) \approx 0$, the dark-memory hypothesis is falsified in favor of pure board-conditional cache $C_t$!)*
- **H3.4 (Transient Path Dependence vs. Asymptotic Path Independence):** Across $K=8$ initializations on `TRM`, solved forks agree $100\%$ asymptotically while peak intermediate fork divergence ($t \in [2, 8]$) is $\ge 2\times$ larger on `WB-Stall` than `WB-Prop`, and C-voting matches or improves deterministic solve rate.
- **H3.5 (Causal Fork Steering & Lost-Memory Revisit Signature):** On `WB-Stall`, `Swap-R_B` steers step-$(t+1)$ commitments toward donor fork $B$ significantly above `Zero-R` ($p < 0.01$), and erasing/swapping/rewinding $R_t$ significantly increases revisit rates of already-refuted states ($p < 0.01$), while remaining null on `LoopedChase` ($\lambda \ge 0.1$) and `WB-Prop`.
