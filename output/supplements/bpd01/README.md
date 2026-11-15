# BPD — Behavioral Parameter Decomposition

**Behavioral Parameter Decomposition (BPD)** is parameter-space mechanistic interpretability for **weight-tied, recurrent-depth computation**, reducing algebraically and numerically to Goodfire's single-pass [Vector/Variational Parameter Decomposition (VPD)](https://www.goodfire.ai/research/interpreting-lm-parameters) when the pass count is $T = 1$.

Upstream VPD targets feedforward weight sites where each matrix executes once per forward pass (its only weight tying is token embeddings/unembeddings). Once a physical matrix $W_m$ executes $T > 1$ times inside a loop, six structural assumptions of VPD fail simultaneously. BPD keeps one shared factor pair $(V_m, U_m)$ per physical matrix while indexing causal importance and masks by pass $\tau$, training against normalized trajectory-wide behavioral and state targets under a time-shared adversary.

## Where VPD Breaks on Loops, and What BPD Changes

| VPD assumes | What a loop does to it | BPD change |
|---|---|---|
| One site per matrix; importance $g_{b,p,c}$ | One physical matrix $W_m$, $T$ invocations | One $(V_m, U_m)$ set per shared matrix; importance and masks $g_{b,p,\tau,c}$ indexed by pass $\tau$ ([Wang & Reid, arXiv:2609.01924](https://arxiv.org/abs/2609.01924)) |
| KL at the final output measures necessity | A re-read loop rebuilds what was ablated, parking importance at pass $T$ | Match the behavioral readout at every pass $\tau$, or normalized loop-boundary state where no readout exists (e.g. inner $z_L$ cycles in `TRM`) |
| Any output change is evidence of importance | Cross-pass sensitivity swings by orders of magnitude ($F_1 / F_6 = 1{,}808.4$ on `TRM`; `HUG-CAL-02`) | Per-pass tolerance $\mathrm{tol}_\tau$ from an RMS-matched Haar rotation null; report **raw** and **net-of-null** importance side by side |
| Zero is a neutral ablation | Recurrent state carries scale, so zeroed writes shrink norms and compound across passes | Ablate toward the pass-conditional mean write $\mu_{m,\tau,c} u_{m,c}$, rescaled onto the pass-conditional write sphere |
| Minimality counts active positions | A stationary mechanism used on every pass pays $T$ times and artificially splits across time | Pass-pooled level minimality ($T$-invariant at $T \ge 1$) plus sweepable temporal switch penalty $\sum_{\tau} |g_{\tau+1} - g_\tau|^q$ |
| Adversary shares sources across the batch | Unconstrained PGD also resonates across passes | Share adversarial mask sources across batch **and** passes by default (`shared_batch_and_pass`), matching physical weight-edit invariance |

## Pass-Tied Causal Importance Variants (`Clock Gap` & `Visible Gap`)

The causal importance network $f_\phi$ is itself weight-tied across $\tau$ ($O(1)$ parameters in $T$, generalizing zero-shot to OOD pass counts $T' > T$) and trained in three observation spaces (`ci.py`):

1. **`state_only`**: $g_{b,p,\tau} = f_\phi(X_{b,p,\tau})$ — pre-weight module inputs only.
2. **`state_plus_clock`**: $g_{b,p,\tau} = f_\phi(X_{b,p,\tau}, e(\tau))$ — adds sinusoidal + learned pass-clock features.
   - **Clock Gap** $= \mathcal{L}_{\text{recon}}(\text{state\_only}) - \mathcal{L}_{\text{recon}}(\text{state\_plus\_clock})$ tests whether mechanisms are addressed by explicit iteration time or by recurrent state.
3. **`visible_only`**: $g_{b,p,\tau} = f_\phi(x_{b,p}, V_{b,p,\tau})$ — prompt clues $x$ plus head-rowspace projection $V_\tau = S_\tau P_{\text{head}}$, strictly blind to $\ker(\text{head})$ and unheaded latents ($z_L$).
   - **Visible Gap** $= \mathcal{L}_{\text{recon}}(\text{visible\_only}) - \mathcal{L}_{\text{recon}}(\text{state\_only})$ is the parameter-space counterpart of `LOOPSCOPE-03`'s dark-fiber trajectory residual $R_t$.

## What BPD Yields That VPD Cannot

1. **Temporal component schedule** (`extract_schedule`): per-input $[B, T, C]$ activation schedules, duty cycles, center-of-mass pass, temporal entropy, and switch rates.
2. **Transient vs. Sustaining taxonomy** (`classify_transient_sustaining`):
   - *Transient*: alters intermediate trajectory states/readouts when ablated at pass $\tau$, but later passes rebuild the work so final output is unchanged.
   - *Sustaining*: single-pass ablation is repaired by remaining passes, yet all-pass ablation collapses the final readout (holds the fixed point).
3. **Component transition operator & interventional composition** (`compute_transition_operator`, `compose_interventional_path`): pooled 1-step attribution $(c', \tau) \to (c, \tau + 1)$ paired with multi-step nonlinear `do`-interventional path mediation rather than multi-step Jacobian products.

## Layout

```text
mechinterp/bpd/
├── config.py                     # BPDConfig + exact T=1 VPD reduction factory
├── components.py                 # TiedComponentLinear, UnrolledComponentLinear, BPDModel
├── ci.py                         # Pass-tied CI (state_only, state_plus_clock, visible_only) + gaps
├── losses.py                     # Faithfulness, tolerance-weighted trajectory KL/state, switch minimality
├── adversary.py                  # PassSharedPersistentPGD + K_adv step-stability evaluation
├── nulls.py                      # RMS-matched Haar rotation nulls (raw + net importance reporting)
├── analysis.py                   # Schedule, transient/sustaining, transition op, sustaining weight edits
├── trainer.py                    # Hybrid clean-window + closed-loop BPDTrainer & UnrolledVPDTrainer
├── adapters/
│   ├── trm.py                    # 7M PureSudokuTRM adapter (outer z_H commits + inner z_L boundaries)
│   ├── looped_chase.py           # LoopedChase pointer-chase adapter (cursor head projector)
│   └── huginn.py                 # Huginn-0125 recurrent core adapter + HUG-CAL-02 8-factor control
├── scripts/
│   ├── run_trm_bpd.py            # Primary target & Gate B4: 7M TRM hybrid BPD + s=64 edit test
│   ├── bench_unrolled_vs_tied.py # Gate B1: Tied C vs Unrolled T*C stationary Pareto test
│   ├── audit_hug_cal02_null.py   # Gate B2: HUG-CAL-02 8-factor negative control
│   └── run_looped_chase_bpd.py   # Gate B3: LoopedChase lam=0.1 vs lam=1.0 Clock & Visible Gaps
├── docs/
│   ├── design_and_positioning.md # Formal equations, v008 audit, T=1 proof, and risk controls
│   └── registration.md           # Pre-registered falsifiable pass/fail criteria for Gates B1-B4
└── tests/                        # Deterministic CPU unit & smoke tests
```

## Quick Verification

```powershell
# Run full deterministic test suite
& "$HOME\.venvs\bluebeam\Scripts\python.exe" -m pytest mechinterp/bpd/tests -q

# Fast CPU smoke runs across all four pre-registered gates (TRM-first order)
& "$HOME\.venvs\bluebeam\Scripts\python.exe" mechinterp/bpd/scripts/run_trm_bpd.py --smoke
& "$HOME\.venvs\bluebeam\Scripts\python.exe" mechinterp/bpd/scripts/bench_unrolled_vs_tied.py --smoke
& "$HOME\.venvs\bluebeam\Scripts\python.exe" mechinterp/bpd/scripts/audit_hug_cal02_null.py --smoke
& "$HOME\.venvs\bluebeam\Scripts\python.exe" mechinterp/bpd/scripts/run_looped_chase_bpd.py --smoke
```

## References

- Bushnaq et al. (Goodfire, 2026), *Interpreting Language Model Parameters* — https://www.goodfire.ai/research/interpreting-lm-parameters
- Wang & Reid (2026), *Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?*, `arXiv:2609.01924` — https://arxiv.org/abs/2609.01924
- Internal foundations: `mechinterp/loopscope01` (`LoopedChase`, `TCA`), `mechinterp/loopscope02` (`PureSudokuTRM`, $F_1 / F_6 = 1{,}808.4$), `mechinterp/loopscope03` ($V_t / C_t / R_t$ ledger & sphere calibration), and `papers/shared_weights_dark_fibers/PAPER.md` (`v014`).
