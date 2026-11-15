# Pre-Registered Falsifiable Gates for BPD (`B1`–`B4`)

Every design claim in Behavioral Parameter Decomposition (BPD, extending adVersarial Parameter Decomposition [VPD] to weight-tied recurrent computation) is paired with a pre-registered test that can fail. All gate thresholds, null-calibration parameters, and control arms are fixed and committed in this file prior to any non-smoke run, and the pre-registration commit SHA is recorded in each gate receipt.

---

## Global Objective & Calibration Parameters

- **Trajectory inner-state switch**: `match_inner_boundary_states = False` by default on adapters that emit a behavioral readout at each outer step (`TRM`), supervising behavioral readouts without forcing intermediate inner-pass latent state matching unless explicitly enabled (`match_inner_boundary_states = True`, reported alongside the default in Gate `B4`).
- **Fragility hinge**: `use_fragility_hinge = True`, `hinge_kappa = 0.10`, `perturbation_scale = 0.10`. Per-pass reconstruction loss is hinged against the pass-specific RMS-matched Haar rotation-null loss `KL_null_tau`:
  $$\text{loss}_\tau = \max\bigl(0,\; \text{KL}_{\text{mask},\tau} - \kappa \cdot \text{KL}_{\text{null},\tau}\bigr), \qquad \kappa = 0.10,$$
  logging each pass's share of total reconstruction loss (`pass_loss_shares`).
- **Minimality frequency term**: `f_c` is computed over `(batch, seq)` after taking `max_tau g_{b,p,tau,c}` across loop passes (`pool_mode = "mean"` over `(batch, seq)` of `max_tau`), paired with temporal switch cost `coeff_switch` along `tau`.
- **Single-pass parity reference**: At `T = 1`, `BPDConfig.vpd_single_pass_equivalent(...)` matches the four-term PyTorch port in `MoralityLabAI/VPD` `experiments/nanbeige_looped/vpd_math.py` at commit `69509c39e96a912f99b1a83640250ac01797c571` (faithfulness + `p`-annealed minimality + stochastic k-subset KL + adversarial KL; Goodfire's paper adds a fifth deterministic CI-masked reconstruction term).

---

## Pre-Registered Execution Order

Gates execute strictly one at a time in the following order, stopping after each gate to report pass or fail against the thresholds below:

1. **Gate `B1`** — Unrolled VPD baseline vs weight-tied BPD (`bench_unrolled_vs_tied.py`)
2. **Gate `B3`** — `LoopedChase` planted rank-1 positive control + Q4 ground-truth suite (`run_looped_chase_bpd.py --suite`)
3. **Gate `B2`** — `HUG-CAL-02` 8-factor negative control + planted rank-1 positive arm (`audit_hug_cal02_null.py`)
4. **Gate `B4`** — Separate 7M `TRM` decompositions, functional matching, and intra-checkpoint `yagizdevre` edit on `WB-Stall` at `s = 64` (`run_trm_bpd.py`)

---

## Gate Definitions & Registered Pass/Fail Thresholds

### Gate `B1` — Unrolled VPD Baseline (`bench_unrolled_vs_tied.py`)

- **Protocol**: On `StationaryResidualLooper` (`n_passes = 4`, `c_shared = 4`, `n_steps = 80`, `batch_size = 24`, `seq_len = 8`, `seed = 7`, `hinge_kappa = 0.10`), train:
  1. Weight-tied BPD (`C = 4` shared components across `T = 4` passes),
  2. Unrolled VPD (`T * C = 16` independent components across `4` pass sites),
  3. Budget-matched unrolled VPD (`max(C // T, 1) = 1` component per pass, `4` total components).
- **Registered threshold (`gate_b1_passed`)**:
  - `tied_c * n_passes == unrolled_full_c` (`4 * 4 == 16`),
  - `tied_bpd.recon_loss <= max(1.5 * unrolled_vpd_TxC.recon_loss, 5e-3)`, **and**
  - `tied_bpd.recon_loss < unrolled_vpd_budget_matched.recon_loss`.
- **Failure condition**: Fails if tied BPD cannot match `1.5x` (or `5e-3` floor) of `T * C` unrolled VPD or fails to beat budget-matched unrolled VPD.

### Gate `B3` — `LoopedChase` Planted Positive Control & Q4 Ground Truth (`run_looped_chase_bpd.py`)

- **Protocol**:
  1. **Planted rank-1 positive control**: On real `LoopedChase` checkpoint `lam0.1_s1.final.pt` (`n_passes = 8`), calibrate a sparse `(start_node == 0, cursor p = -1)` ridge read vector on `cal_seed = 777` paired with normalized head-rowspace write `W_head[0] - W_head[1]` (`sigma = 600.0`), plant `u_plant v_plant^T` onto `core.1.fc2.weight`, decompose blindly via `BPDModel(..., init_svd=True)`, and evaluate component `0` out-of-sample on `eval_seed = 9000` against `n_rotations = 16` RMS-matched Haar read rotations (`null_seed = 42`). Report `observed_z_rot` and `margin = observed_z_rot - 2.5`.
  2. **Q4 checkpoint suite**: Fit BPD (`n_passes = 8`, `c_per_mod = 16`, `n_steps = 100`, `ci_head_steps = 50`, `batch_size = 24`, `hinge_kappa = 0.10`) and train `state_only`, `state_plus_clock`, and `visible_only` CI heads on frozen `(V, U)` across `lam0.1_s1`, `lam1_s1`, `lam1_s2`, and `lam1_s3`.
- **Registered threshold (`gate_b3_passed`)**:
  1. **Planted positive control**: `net_importance > 0.0` **and** `observed_z_rot >= 2.5` (`margin >= 0.0`).
  2. **Stationary `lam = 0.1` (`lam0.1_s1`)**: `stationary_component_fraction >= 0.50`, `|clock_gap_recon| <= 0.05`, **and** `|visible_gap_recon| <= 0.05`.
  3. **Off-head `lam = 1.0` visibility separation**:
     - `lam1_s1`: `visible_gap_recon(lam1_s1) > visible_gap_recon(lam1_s3)` **or** `visible_gap_active(lam1_s1) > visible_gap_active(lam1_s3)`, **and**
     - `lam1_s2`: `visible_gap_recon(lam1_s2) > visible_gap_recon(lam1_s3)` **or** `visible_gap_active(lam1_s2) > visible_gap_active(lam1_s3)`.
- **Failure condition**: Fails if the planted rank-1 positive control falls below `z_threshold = 2.5`, if `lam0.1_s1` violates the stationarity/gap bounds, or if either `lam1_s1` or `lam1_s2` fails to exceed `lam1_s3` on `visible_gap`.

### Gate `B2` — `HUG-CAL-02` Dual Negative & Positive Control (`audit_hug_cal02_null.py`)

- **Protocol**: On 4-block `SettlingCoreLooper` (`n_passes = 6`, `n_rotations = 16`, `batch_size = 24`, `seq_len = 12`, `seed = 11`):
  1. **Negative arm**: Audit the 8 leading core settling factors (`core.{0..3}.down_proj`, components `0, 1`) against `16` RMS-matched Haar read rotations, reporting `raw_importance`, `net_importance`, `z_rot`, and `negative_margin = 2.5 - max_abs_z_rot`.
  2. **Positive arm**: Calibrate a rank-1 equilibrium-reversing perturbation (`cancel_fraction = 1.25`) on `cal_seed = 777` targeting `core.3.down_proj`, decompose blindly via `BPDModel(..., init_svd=True)`, and audit component `0` out-of-sample on `seed = 11` under the identical `16`-rotation Haar null, reporting `observed_z_rot` and `margin = observed_z_rot - 2.5`.
- **Registered threshold (`gate_b2_passed`)**:
  1. **Negative arm**: `max_raw_importance > 0.0` **and** `max_{i in 1..8} |z_rot(i)| <= 2.5` (`negative_margin >= 0.0`).
  2. **Positive arm**: `net_importance > 0.0` **and** `observed_z_rot >= 2.5` (`margin >= 0.0`).
- **Failure condition**: Fails if any unaligned settling factor exceeds `|z_rot| = 2.5`, if `max_raw_importance == 0`, or if the planted positive arm fails to reach `observed_z_rot >= 2.5` with positive net importance.

### Gate `B4` — Separate `TRM` Decompositions & Intra-Checkpoint Edit at `s = 64` (`run_trm_bpd.py`)

- **Protocol**:
  1. Decompose SHA-256-pinned 7M `gaoxin492` and `yagizdevre` **independently** (`train_outer_steps = 2`, `c_per_mod = 24`, `n_steps = 120`, `n_train = 32` disjoint training boards, `hinge_kappa = 0.10`), reporting both `match_inner_boundary_states = False` (default) and `match_inner_boundary_states = True` on `gaoxin492`.
  2. Match components across the two independent decompositions strictly by **function** (`match_components_by_function`: schedule cosine similarity, 1-step transition profile similarity, and readout distribution shift similarity — zero cross-checkpoint weight transplant and zero parameter-difference ranking).
  3. Apply a permanent time-shared edit strictly within `yagizdevre`'s own decomposition (`top_k_edit = 4`, `edit_scale_delta = 0.15`): $\Delta W_m = \sum_{c \in S_m} \delta \cdot u^{\text{tgt}}_{m,c} (v^{\text{tgt}}_{m,c})^\top$.
  4. Evaluate on `WB-Stall` (`n = 183` boards from `loopscope02/results/benchmark_bundle.json`) across `eval_horizons = (16, 32, 64)` against two matched controls on `yagizdevre`:
     - `matched_random_component_control`: random non-selected components from `yagizdevre`'s own decomposition with identical per-module count and identical Frobenius norm $\|\Delta W_m\|_F$.
     - `rotation_null_control`: RMS-matched Haar-rotated read vectors and Haar-rotated write vectors, rescaled to identical per-module Frobenius norm $\|\Delta W_m\|_F$.
  5. Report **both** `final_solve_rate` (board exact at `s = 64`) and `ever_solved_rate` (board solved at any evaluated outer step up to `s = 64`).
- **Registered threshold (`gate_b4_passed`)**:
  1. `final_solve_rate(edited) > final_solve_rate(clean_yagizdevre)`,
  2. `period2_oscillation_rate(edited) < period2_oscillation_rate(clean_yagizdevre)`,
  3. `final_solve_rate(edited) > final_solve_rate(matched_random_component_control)`,
  4. `final_solve_rate(edited) > final_solve_rate(rotation_null_control)`, **and**
  5. `board_exact@16(edited) >= board_exact@16(clean_yagizdevre) - 0.02`.
- **Failure condition**: Fails if scaling functionally matched sustaining components inside `yagizdevre`'s own decomposition does not simultaneously improve `final_solve_rate` at `s = 64`, reduce `period2_oscillation_rate`, beat both matched controls, and preserve `s = 16` solve rate within `0.02`.

---

## Mandatory Risk Audits Logged in Gate `B4`

1. **Adversary step-stability curve (`pgd_stability`)**: Reconstruction loss across $K_{\text{adv}} \in \{0, 2, 5, 20, 80, 160\}$ for both `shared_batch_and_pass` and `per_pass` PGD scopes.
2. **Dual raw + net reporting**: Raw importance and rotation-null-calibrated net importance reported side by side.
3. **Switch-penalty sweep (`switch_penalty_sweep`)**: Sweep `coeff_switch` across `(0.0, 2e-4, 2e-3)` on frozen `(V, U)`.
