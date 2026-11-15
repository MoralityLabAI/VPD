# BPD Design, Positioning, and Mathematical Specification

## 1. Positioning: VPD for Weight-Tied Computation

Goodfire's Vector/Variational Parameter Decomposition ([*Interpreting Language Model Parameters*](https://www.goodfire.ai/research/interpreting-lm-parameters)) decomposes each linear layer $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ into $C$ rank-one components $(V U)^\top = \sum_{c=1}^C u_c v_c^\top$ plus residual $\Delta W = W - (V U)^\top$, gated per input token by a causal importance network $g_{b,p,c} \in [0, 1]$. Upstream `param-decomp` targets strictly feedforward Transformer stacks (`gpt2`, `LlamaForCausalLM`, `Qwen3ForCausalLM`, toy MLPs) where each physical matrix executes **once** per forward pass.

**Behavioral Parameter Decomposition (BPD)** claims the generalization to weight-tied recurrent computation ($T \ge 1$ passes through shared parameters), subject to an exact boundary condition:

> **Single-pass reduction ($T = 1$):** Under `BPDConfig.vpd_single_pass_equivalent(...)` ($T = 1$, `state_only` CI, zero ablation, `coeff_switch = 0`, uniform pass tolerance), BPD's forward pass, faithfulness loss, $L_p + \beta \log_2$ minimality objective, stochastic mask interpolation, and adversarial mask objective reduce **identically** to upstream VPD (`tests/test_bpd_core.py::test_single_pass_reduces_identically_to_vpd`).

### Audit of the `v008` Invocation-Mask Skeleton

Inspection of the earlier `v008` prototype (`experiments/nanbeige_looped/recurrent.py` and `vpd_math.py`) confirms why an invocation-mask wrapper alone is insufficient:

1. **Shared parameter shell only**: `RepeatedLinear` wrapped a shared `ComponentLinear(linear, C)` across `passes`, avoiding $T$-fold duplication of $(V, U)$.
2. **Untied CI network**: `train_calibration` flattened per-pass trajectories under keys `f"{path}::pass_{tau}"`, instantiating a single `CITransformer` whose input/output projections grew linearly with $T$, had untied weights across passes, and could not run at $T' \ne T$.
3. **Final-pass-only KL**: Reconstruction KL was evaluated exclusively against final logits `target(tokens)`, letting prompt-re-injecting loops rebuild early damage and park causal importance at the final pass.
4. **No sensitivity normalization**: Fragile and contractive passes entered with equal raw KL sensitivity despite $>10^3\times$ Jacobian transport swings.
5. **Zero ablation**: Inactive components were zeroed (`(comp_acts * mask) @ U`), shrinking write norms across recurrent steps.
6. **$T$-fold active-pass tax**: Every `path::pass_tau` view was fed independently to `vpd.importance_minimality_loss`, taxing stationary mechanisms $T$ times and incentivizing time-split components.
7. **Pass-resonant adversary**: `PersistentPGD` allocated separate source slices per pass $\tau$, allowing adversarial masks to alternate sign/phase across $\tau$ in ways no physical shared-weight edit can realize.

BPD keeps item (1) and replaces items (2)–(7) with a loop-native objective and diagnostic suite.

---

## 2. Six Structural Modifications for Weight-Tied Loops

### 2.1 Shared $(V_m, U_m)$ Dictionary with Pass-Indexed Importance $g_{b,p,\tau,c}$

Each physical matrix $W_m \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ owns **one** factor pair $V_m \in \mathbb{R}^{d_{\text{in}} \times C_m}$, $U_m \in \mathbb{R}^{C_m \times d_{\text{out}}}$, with faithfulness counted once per physical matrix:
$$
\mathcal{L}_{\text{faith}} = \frac{\sum_m \lVert W_m - (V_m U_m)^\top \rVert_F^2}{\sum_m d_{\text{out},m}\, d_{\text{in},m}}.
$$
Importance $g_{m,b,p,\tau,c} \in [0, 1]$ is indexed by invocation $\tau \in \{0, \dots, T-1\}$. This directly reflects [Wang & Reid (2026, `arXiv:2609.01924`)](https://arxiv.org/abs/2609.01924): under their virtual-unrolling Jacobian lens on `Ouro-2.6B` and `Huginn-0125`, local step-to-step transports repeat within each iteration's own frame (justifying one shared dictionary $(V_m, U_m)$), while end-to-end contribution to the final readout depends strongly on *when* the shared block executes (requiring pass-indexed $g_{b,p,\tau,c}$).

### 2.2 Trajectory Readout & Loop-Boundary State Matching

On any pass $\tau \in \mathcal{T}_{\text{read}}$ with an explicit behavioral readout $y_\tau$ (such as per-loop cursor logits in `LoopedChase`, per-iteration coda logits in `Huginn-0125`, or outer/$H$-cycle $z_H$ commits in `TRM`), BPD matches $\mathrm{KL}(p_{\text{clean}}(y_\tau) \,\|\, p_{\text{pred}}(y_\tau))$. On passes $\tau \in \mathcal{T}_{\text{state}} \setminus \mathcal{T}_{\text{read}}$ without a readout (such as `TRM`'s inner $z_L$ updates), BPD matches scale-normalized + cosine boundary state discrepancy:
$$
\ell_{\text{state}}(\hat{S}_\tau, S_\tau) = \frac{1}{2}\,\mathbb{E}\!\left[\frac{\lVert \hat{S}_\tau - S_\tau \rVert_2^2}{\frac{1}{D}\lVert S_\tau \rVert_2^2 + \varepsilon}\right] + \frac{1}{2}\,\mathbb{E}\!\left[1 - \cos(\hat{S}_\tau, S_\tau)\right].
$$

### 2.3 Per-Pass Sensitivity Tolerance & Raw + Net Rotation Nulls

For each pass $\tau$ and component read vector $v_{m,c}$, BPD samples $K$ Haar-random read vectors $r_{m,\tau,c}^{(k)} \perp v_{m,c}$ rescaled to match empirical pass activation RMS:
$$
\mathbb{E}_{b,p}\!\left[(X_{m,b,p,\tau}^\top r_{m,\tau,c}^{(k)})^2\right] = \mathbb{E}_{b,p}\!\left[(X_{m,b,p,\tau}^\top v_{m,c})^2\right].
$$
During training, per-pass tolerances $\mathrm{tol}_\tau$ estimated from random RMS-matched rank-1 writes weight trajectory losses via $w_\tau \propto (\mathrm{tol}_\tau + \varepsilon)^{-1}$ (mean-normalized to $1$), neutralizing $>10^3\times$ fragility swings ($F_1 / F_6 = 1{,}808.4$ on detached-ACT `TRM`). During evaluation (`nulls.py`), BPD **always reports both** $\mathrm{Imp}^{\text{raw}}$ and $\mathrm{Imp}^{\text{net}} = \mathrm{Imp}^{\text{raw}} - \mu_{\text{rot}}$ alongside $z_{\text{rot}}$ so fragile-pass nulls cannot hide genuine mechanisms.

### 2.4 Pass-Conditional Mean-Write Ablation with Sphere Rescaling

On clean target passes, `TiedComponentLinear` tracks EMA statistics of pass-conditional component activations $\mu_{m,\tau,c} = \mathbb{E}[X_{m,\tau} v_{m,c}]$, residual writes $\mu^{\Delta}_{m,\tau} = \mathbb{E}[\Delta W_m X_{m,\tau}]$, and clean write RMS $r_{m,\tau} = \mathbb{E}[\mathrm{rms}(W_m X_{m,\tau})]$. Under mask $m_\tau \in [0, 1]^C$:
$$
\tilde{a}_{b,p,\tau,c} = m_{b,p,\tau,c}\, a_{b,p,\tau,c} + (1 - m_{b,p,\tau,c})\, \mu_{m,\tau,c}.
$$
Because $\lVert \mathbb{E}[w] \rVert_2 < \mathbb{E}[\lVert w \rVert_2]$ by Jensen's inequality, unrescaled mean writes sit strictly inside the sphere and compound norm decay across recurrent steps. `sphere_rescale=True` lifts ablated writes smoothly back to the pass-conditional radius $r_{m,\tau}$ in proportion to the ablated fraction $\bar{\alpha} = 1 - \frac{1}{C}\sum_c m_c$ (reducing to exact identity $1.0$ at $m \equiv 1$, and preserving per-token clean write RMS under counterfactual single-component ablations).

### 2.5 Pass-Invariant Minimality + Temporal Switch Penalty

Let $\tilde{g}_{b,p,\tau,c}$ denote `upper_leaky_sigmoid` causal importance and $\ell_{b,p,\tau,c} = (\tilde{g}_{b,p,\tau,c} + \varepsilon)^p$ with $p$ annealed from $p_{\text{start}}$ to $p_{\text{end}}$. Pooling across $\tau$ by mean ($\bar{\ell}_{b,p,c} = \frac{1}{T}\sum_{\tau=0}^{T-1} \ell_{b,p,\tau,c}$) inside Goodfire's functional and adding a total-variation switch penalty along $\tau$:
$$
\mathcal{L}_{\text{min}} = \lambda_{\text{level}}\,\mathbb{E}_{b,p}\!\left[\bar{m}_{b,p}\left(1 + \beta \log_2\!\left(1 + \sum_c \bar{\ell}_{b,p,c}\right)\right)\right] + \lambda_{\text{switch}}\,\mathbb{E}_{b,p}\!\left[\frac{1}{T-1}\sum_{\tau=0}^{T-2}\sum_c \lvert \tilde{g}_{b,p,\tau+1,c} - \tilde{g}_{b,p,\tau,c} \rvert^q\right],
$$
where $\bar{m}_{b,p} = \frac{1}{C_{\text{tot}}}\sum_c \bar{\ell}_{b,p,c}$. At $T = 1$, the second term vanishes identically and the first term equals upstream VPD bit-for-bit. At $T > 1$, a stationary mechanism active on all $T$ passes pays the exact same level cost as at $T = 1$ and zero switch cost, while unused passes are still driven to zero and artificial time-splitting pays both switch cost and extra $\log_2$ rank slots.

### 2.6 Time-Shared PersistentPGD & Step-Stability Audit

Because any physical perturbation to $W_m$ acts on every pass $\tau$, `PassSharedPersistentPGD` defaults to `scope="shared_batch_and_pass"` ($s_m \in [0, 1]^{1 \times 1 \times 1 \times (C_m + 1)}$). `evaluate_pgd_stability_curve` sweeps fresh Sign-PGD steps $K_{\text{adv}} \in \{0, 2, 5, 20, 80, 160\}$ under both `shared_batch_and_pass` and `per_pass` scopes to track Goodfire's multi-step adversarial blow-up mode directly.

---

## 3. Pass-Tied CI Variants & Hybrid Training

- **Pass-tied CI (`ci.py`)**: `TiedCINetwork` reshapes $[B, T, P, d_{\text{feat}}] \to [B \cdot T, P, d_{\text{feat}}]$ and applies identical RoPE Transformer blocks across sequence positions within each pass. Comparing `state_only`, `state_plus_clock`, and `visible_only` on frozen shared $(V, U)$ (`train_ci_heads_on_frozen_components`) isolates the **Clock Gap** and **Visible Gap** cleanly from $(V, U)$ optimization noise.
- **Hybrid rollout training (`trainer.py`)**: Following `v014` (`papers/shared_weights_dark_fibers/PAPER.md`), `rollout_mode="hybrid"` mixes full closed-loop rollouts from initial state $S_0$ with short `window_steps` rollouts transplanted onto cached clean states $S_{\tau_0}$, preventing early closed-loop chaos while still training sustaining fixed-point mechanisms over full trajectories.
