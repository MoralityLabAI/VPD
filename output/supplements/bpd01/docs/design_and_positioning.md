# BPD Design, Positioning, and Mathematical Specification

## 1. Positioning: VPD for Weight-Tied Computation

Goodfire's **adVersarial Parameter Decomposition (VPD)** ([*Interpreting Language Model Parameters*](https://www.goodfire.ai/research/interpreting-lm-parameters)) decomposes each linear layer $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ into $C$ rank-one components $(V U)^\top = \sum_{c=1}^C u_c v_c^\top$ plus residual $\Delta W = W - (V U)^\top$, gated per input token by a causal importance network $g_{b,p,c} \in [0, 1]$. Upstream `param-decomp` targets strictly feedforward Transformer stacks (`gpt2`, `LlamaForCausalLM`, `Qwen3ForCausalLM`, toy MLPs) where each physical matrix executes **once** per forward pass.

**Behavioral Parameter Decomposition (BPD)** generalizes parameter decomposition to weight-tied recurrent computation ($T \ge 1$ passes through shared parameters), anchored by a numeric $T = 1$ parity condition:

> **Single-pass parity ($T = 1$):** Under `BPDConfig.vpd_single_pass_equivalent(...)` ($T = 1$, `state_only` CI without post-stack RMSNorm, zero ablation, `coeff_switch = 0`, `use_fragility_hinge = False`), BPD's masked forward pass, faithfulness MSE, $L_p + \beta f_c \log_2$ minimality loss, uniform $k$-subset stochastic routing KL, and persistent Sign-PGD adversarial KL match the **four-term PyTorch VPD port** in `MoralityLabAI/VPD` (`experiments/nanbeige_looped/vpd_math.py` at commit `69509c39e96a912f99b1a83640250ac01797c571`; Goodfire's paper adds a fifth deterministic CI-masked reconstruction term). Numeric parity is verified in `tests/test_bpd_core.py::test_t1_numeric_parity_against_vpd_math_torch_port`.

### Audit of the `v008` Invocation-Mask Skeleton

Inspection of the earlier `v008` prototype (`experiments/nanbeige_looped/recurrent.py` and `vpd_math.py`) confirms why an invocation-mask wrapper alone is insufficient:

1. **Shared parameter shell only**: `RepeatedLinear` wrapped a shared `ComponentLinear(linear, C)` across `passes`, avoiding $T$-fold duplication of $(V, U)$.
2. **Untied CI network**: `train_calibration` flattened per-pass trajectories under keys `f"{path}::pass_{tau}"`, instantiating a single `CITransformer` whose input/output projections grew linearly with $T$, had untied weights across passes, and could not run at $T' \ne T$.
3. **Final-pass-only KL**: Reconstruction KL was evaluated exclusively against final logits `target(tokens)`, letting prompt-re-injecting loops rebuild early damage and park causal importance at the final pass.
4. **No sensitivity calibration**: Fragile and contractive passes entered with equal raw KL sensitivity despite $>10^3\times$ Jacobian transport swings.
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

### 2.2 Trajectory Readout Matching & Optional Inner-Pass State Switch

On passes $\tau \in \mathcal{T}_{\text{read}}$ with an explicit behavioral readout $y_\tau$ (per-loop cursor logits in `LoopedChase`, per-iteration coda logits in `Huginn-0125`, or outer-step $z_H$ readouts in `TRM`), BPD matches $\mathrm{KL}(p_{\text{clean}}(y_\tau) \,\|\, p_{\text{pred}}(y_\tau))$. Inner-pass boundary-state matching is controlled by `BPDConfig.match_inner_boundary_states` (**`False` by default** for adapters that emit a readout at each outer step, such as `TRM`; `run_trm_bpd.py` runs and reports both `False` and `True`). When `match_inner_boundary_states=True` (or on purely read-free trajectories), passes $\tau \in \mathcal{T}_{\text{state}} \setminus \mathcal{T}_{\text{read}}$ also match scale-normalized + cosine boundary state discrepancy:
$$
\ell_{\text{state}}(\hat{S}_\tau, S_\tau) = \frac{1}{2}\,\mathbb{E}\!\left[\frac{\lVert \hat{S}_\tau - S_\tau \rVert_2^2}{\frac{1}{D}\lVert S_\tau \rVert_2^2 + \varepsilon}\right] + \frac{1}{2}\,\mathbb{E}\!\left[1 - \cos(\hat{S}_\tau, S_\tau)\right].
$$

### 2.3 Per-Pass Fragility Hinge & Raw + Net Rotation Nulls

For each active pass $\tau$, `calibrate_pass_null_losses` estimates a per-pass baseline perturbation loss $\mathrm{KL}^{\text{null}}_\tau$ under RMS-matched Haar read rotations at perturbation scale $\eta$. Rather than inverse-fragility reweighting (which suppresses high-sensitivity passes regardless of perturbation excess), BPD applies a **fragility hinge** with registered floor coefficient $\kappa$ (`NullCalibrationConfig.hinge_kappa`, registered in `docs/registration.md`):
$$
\ell_\tau = \max\!\left(0,\, \mathrm{KL}^{\text{mask}}_\tau - \kappa\,\mathrm{KL}^{\text{null}}_\tau\right),
$$
and logs each pass's share of total reconstruction loss (`compute_pass_loss_shares`). During component evaluation (`nulls.py`), BPD samples $K$ Haar-random read vectors $r_{m,\tau,c}^{(k)} \perp v_{m,c}$ matched to empirical pass activation RMS:
$$
\mathbb{E}_{b,p}\!\left[(X_{m,b,p,\tau}^\top r_{m,\tau,c}^{(k)})^2\right] = \mathbb{E}_{b,p}\!\left[(X_{m,b,p,\tau}^\top v_{m,c})^2\right],
$$
and **always reports both** $\mathrm{Imp}^{\text{raw}}$ and $\mathrm{Imp}^{\text{net}} = \mathrm{Imp}^{\text{raw}} - \mu_{\text{rot}}$ alongside $z_{\text{rot}}$.

### 2.4 Pass-Conditional Mean-Write Ablation with Sphere Rescaling

On clean target passes, `TiedComponentLinear` tracks EMA statistics of pass-conditional component activations $\mu_{m,\tau,c} = \mathbb{E}[X_{m,\tau} v_{m,c}]$, residual writes $\mu^{\Delta}_{m,\tau} = \mathbb{E}[\Delta W_m X_{m,\tau}]$, and clean write RMS $r_{m,\tau} = \mathbb{E}[\mathrm{rms}(W_m X_{m,\tau})]$. Under mask $m_\tau \in [0, 1]^C$:
$$
\tilde{a}_{b,p,\tau,c} = m_{b,p,\tau,c}\, a_{b,p,\tau,c} + (1 - m_{b,p,\tau,c})\, \mu_{m,\tau,c}.
$$
Because $\lVert \mathbb{E}[w] \rVert_2 < \mathbb{E}[\lVert w \rVert_2]$ by Jensen's inequality, unrescaled mean writes sit strictly inside the sphere and compound norm decay across recurrent steps. `sphere_rescale=True` lifts ablated writes smoothly back to the pass-conditional radius $r_{m,\tau}$ in proportion to the ablated fraction $\bar{\alpha} = 1 - \frac{1}{C}\sum_c m_c$ (reducing to exact identity $1.0$ at $m \equiv 1$, and preserving per-token clean write RMS under counterfactual single-component ablations).

### 2.5 Max-Over-$\tau$ Component Frequency $f_c$ + Temporal Switch Penalty

Let $\tilde{g}_{b,p,\tau,c}$ denote `upper_leaky_sigmoid` causal importance and $v_{b,\tau,p,c} = (\tilde{g}_{b,p,\tau,c} + \varepsilon)^p$ with $p$ annealed from $p_{\text{start}}$ to $p_{\text{end}}$. For each shared module $m$ and component $c$, BPD separates mean pass activity $\ell_c = \mathbb{E}_{b,\tau,p}[v_{b,\tau,p,c}]$ from **max-over-$\tau$ data frequency** $S_c = \sum_{b,p} \max_{\tau} v_{b,\tau,p,c}$, $f_c = \frac{1}{B P} S_c$, and adds a total-variation switch penalty along $\tau$:
$$
\mathcal{L}_{\text{min}} = \lambda_{\text{level}} \sum_m \sum_{c=1}^{C_m} \left[\ell_{m,c} + \beta\, f_{m,c} \log_2(1 + S_{m,c})\right] + \lambda_{\text{switch}}\,\mathbb{E}_{b,p}\!\left[\frac{1}{T-1}\sum_{\tau=0}^{T-2}\sum_{m,c} \lvert \tilde{g}_{m,b,p,\tau+1,c} - \tilde{g}_{m,b,p,\tau,c} \rvert^q\right].
$$
At $T = 1$, $\max_\tau v = v_{\tau=0}$ so $\ell_c = f_c$, the switch term vanishes identically, and the module sum equals `vpd_math.importance_minimality_loss`. At $T > 1$, a stationary mechanism active on all $T$ passes pays the exact same cost as at $T = 1$ with zero switch cost, whereas splitting one all-pass mechanism into two half-trajectory copies ($[0, T/2)$ and $[T/2, T)$) preserves $\sum_c \ell_c$ while **doubling** $\sum_c f_c \log_2(1 + S_c)$ even at $\lambda_{\text{switch}} = 0$ (`tests/test_bpd_core.py::test_stationary_mechanism_not_cheaper_when_split_into_half_trajectories`).

### 2.6 Time-Shared PersistentPGD & Step-Stability Audit

Because any physical perturbation to $W_m$ acts on every pass $\tau$, `PassSharedPersistentPGD` defaults to `scope="shared_batch_and_pass"` ($s_m \in [0, 1]^{1 \times 1 \times 1 \times (C_m + 1)}$). `evaluate_pgd_stability_curve` sweeps fresh Sign-PGD steps $K_{\text{adv}} \in \{0, 2, 5, 20, 80, 160\}$ under both `shared_batch_and_pass` and `per_pass` scopes to track Goodfire's multi-step adversarial blow-up mode directly.

---

## 3. Pass-Tied CI Variants, Hybrid Training, and Intra-Checkpoint Editing

- **Pass-tied CI (`ci.py`)**: `TiedCINetwork` reshapes $[B, T, P, d_{\text{feat}}] \to [B \cdot T, P, d_{\text{feat}}]$ and applies identical RoPE Transformer blocks across sequence positions within each pass. Comparing `state_only`, `state_plus_clock`, and `visible_only` on frozen shared $(V, U)$ (`train_ci_heads_on_frozen_components`) isolates the **Clock Gap** and **Visible Gap** cleanly from $(V, U)$ optimization noise.
- **Hybrid rollout training (`trainer.py`)**: Following `v014` (`papers/shared_weights_dark_fibers/PAPER.md`), `rollout_mode="hybrid"` mixes full closed-loop rollouts from initial state $S_0$ with short `window_steps` rollouts transplanted onto cached clean states $S_{\tau_0}$.
- **Separate decompositions & intra-checkpoint editing (`analysis.py`)**: Cross-checkpoint weight transplantation and parameter-difference ranking are forbidden. Instead, reference (`gaoxin492`) and target (`yagizdevre`) checkpoints are decomposed separately, components are matched purely by function (temporal activation schedule, 1-step transition profile, and head readout signature), and weight edits scale only components inside `yagizdevre`'s own decomposition ($W_{\text{tgt}} \leftarrow W_{\text{tgt}} + \sum_{c \in S} \gamma_c\, u^{\text{tgt}}_c (v^{\text{tgt}}_c)^\top$), evaluated against matched-count, matched-Frobenius-norm random component controls and RMS-matched Haar rotation-null controls.
