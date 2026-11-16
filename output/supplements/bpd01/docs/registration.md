# Pre-Registered Falsifiable Gates for BPD (`B1`–`B4`)

Every design claim in Behavioral Parameter Decomposition (BPD, extending adVersarial Parameter Decomposition [VPD] to weight-tied recurrent computation) is paired with a pre-registered test that can fail. All gate thresholds, null-calibration parameters, and control arms are fixed and committed in this file prior to any non-smoke run, and the pre-registration commit SHA is recorded in each gate receipt.

---

## Dated Amendment Log

1. **`2026-11-15` — Initial Registration (`v0`, `e6baa744b82d91cd2812a73799c5d1aea5de2ce8`)**:
   - Defined initial `B1`–`B4` gates (`core.{0..3}.{o_proj, down_proj}` 8-factor negative control in `B2`; cross-checkpoint sustaining-component transplant in `B4`).
2. **`2026-11-15` — Amendment 1 (`v1`, `49c7207e8daf7568c6788b5b127c6c9cc6d1bb25`)**:
   - Added fragility-hinged trajectory reconstruction (`hinge_kappa = 0.10`), `max_tau` frequency minimality `f_c`, planted rank-1 positive controls on `B2` and `B3`, replaced cross-checkpoint parameter transplant in `B4` with separate decompositions + intra-checkpoint `yagizdevre` edit, and narrowed the `B2` negative arm from `core.{0..3}.{o_proj, down_proj}` (`c = 0`) to `core.{0..3}.down_proj` (`c in {0, 1}`).
   - Executed once into `mechinterp/bpd/results/gate_b{1,2,3,4}_receipt.json` (mirrored to `MoralityLabAI/VPD` at `e25b640d5877ff54e8f01e1a3af2d8307352cef8`), returning `B1` PASS, `B2` PASS, `B3` FAIL, `B4` FAIL. All `v1` receipts remain preserved untouched at those paths.
3. **`2026-11-16` — Amendment 2 (`v2`, committed prior to all `mechinterp/bpd/results/v2/` runs)**:
   - Addresses the second-round review defects in experimental design, calibration power, decomposition rank, and statistical resolution while writing all new receipts exclusively under `mechinterp/bpd/results/v2/gate_b{1,2,3,4}_v2_receipt.json`:
     - **`B1 v2`**: Replaces the rank-starved 1-component-per-pass baseline and asymmetric hinge comparison with `StationaryRoutedLooper` (`C = 8` shared routed mechanisms, `k_active = C // T = 2` active per item, so `C // T = 2` per pass can represent any individual item), trains all three arms (`tied_bpd`, `unrolled_vpd_TxC`, `unrolled_vpd_budget_matched`) under identical **unhinged** KL (`use_fragility_hinge = False`), identical `zero` ablation, and identical warmup/LR schedule across `3` seeds (`7, 8, 9`), and drops the `5e-3` floor rescue in favor of `mean_tied <= max(2.0 * mean_TxC, 1e-5)` and `mean_tied <= 0.25 * mean_budget_matched`.
     - **`B2 v2`**: Restores the full `core.{0..3}.{o_proj, down_proj}` (`c = 0`) 8-factor negative arm, upgrades Haar read-rotation nulls from `K = 16` Gaussian `z`-scores to **`K = 199` rotations** with exact Monte Carlo rank $p$-values $p_{\text{rank}} = (1 + \#\{r_k \ge r_{\text{obs}}\}) / 200$ (`min_p_rank > 0.05 / 8 = 0.00625`), and replaces the single `24x`-natural positive arm with a natural-scale **`sigma` sweep across `(0.5x, 1.0x, 2.0x, 4.0x)`** of `max_natural_raw_cal`, auditing both null-instrument sensitivity and out-of-sample blind BPD recovery (`p_rank <= 0.01` required across all multipliers `>= 1.0x`).
     - **`B3 v2`**: Upgrades `LoopedChase` decomposition to full rank `c_per_mod = 256` (`d = 256`) with Goodfire's 5th deterministic CI-masked reconstruction term (`coeff_det = 0.5`), aligns `visible_only` CI with `LOOPSCOPE-03` all-token `V_head` projection, warm-starts `state_plus_clock` from the converged `state_only` checkpoint (`W_clock = 0` at step 0 so nesting is respected up to `clock_gap_recon >= -0.02`), evaluates `3` CI-head seeds per checkpoint requiring `2 * combined_sd` separation margins, and replaces the `sigma = 600` (`z = 2790`) planted test with a `(0.5x, 1.0x, 2.0x, 4.0x)` natural-scale `sigma` sweep at `K = 199` rotations (`p_rank <= 0.01` required at `>= 1.0x` for both instrument power and blind BPD recovery).
     - **`B4 v2`**: Upgrades `TRM` decomposition to full rank `c_per_mod = 512` (`min(d_out, d_in) = 512`, eliminating the 95% residual `Delta_W`), trains on `n_train = 128` disjoint Sudoku boards (`4 x 32` mini-batches) over `train_outer_steps = 3` (`63` inner passes, `n_steps = 240`, `coeff_det = 0.5`, `coeff_adv = 1.0`, `inner_steps = 2`) with `call_idx`-pinned activation checkpointing and OOD `6`-outer-step evaluation, fixes the `compose_interventional_path` input-recording bug, warm-starts the `switch_penalty_sweep` across `3` seeds, strictly enforces `tgt_net_sustaining > 0` in `match_components_by_function`, logs the full off-diagonal `unmatched_similarity_baseline` (`199` permutations), replaces `period2_oscillation_rate` gating with `LOOPSCOPE-03`'s `unsolved_tail_cycle_rate` (`s = 33..64`), and runs **`K = 39` control draws** (`1 / (K + 1) = 0.025`) with exact per-board McNemar discordance counts on **two** hypothesis-testing arms (requiring **both** Arm A intra-`gaoxin492` sustaining causal necessity **and** Arm B intra-`yagizdevre` functional stabilisation, plus `fidelity_ok`, for `gate_b4_passed = True`).

---

## Global Objective & Calibration Parameters

- **Trajectory inner-state switch**: `match_inner_boundary_states = False` by default on adapters that emit a behavioral readout at each outer step (`TRM`), supervising behavioral readouts without forcing intermediate inner-pass latent state matching unless explicitly enabled (`match_inner_boundary_states = True`, reported alongside the default in Gate `B4`).
- **Deterministic + stochastic + adversarial CI reconstruction**: Amendment `v2` multi-pass fits on `LoopedChase` (`B3 v2`) and `TRM` (`B4 v2`) include Goodfire's fifth deterministic CI-masked reconstruction term (`coeff_det = 0.5` at `m_comp = ci, m_delta = 0`) alongside stochastic routing (`coeff_stoch = 0.5`) and persistent time-shared PGD (`coeff_adv = 1.0`, `lr = 0.03`, `inner_steps = 2`). At `T = 1` with `coeff_det = 0.0` (the default in `BPDConfig`), `BPDConfig.vpd_single_pass_equivalent(...)` preserves exact four-term parity with `MoralityLabAI/VPD` `experiments/nanbeige_looped/vpd_math.py` at `69509c39e96a912f99b1a83640250ac01797c571`.
- **Fragility hinge**: On multi-pass checkpoints (`B3`, `B4`), `use_fragility_hinge = True`, `hinge_kappa = 0.10`, `perturbation_scale = 0.10`:
  $$\text{loss}_\tau = \max\bigl(0,\; \text{KL}_{\text{mask},\tau} - \kappa \cdot \text{KL}_{\text{null},\tau}\bigr), \qquad \kappa = 0.10.$$
  In `B1 v2`, `use_fragility_hinge = False` (`hinge_kappa = 0.0`) on all three arms so tied BPD and unrolled VPD optimize the identical unhinged KL objective.
- **Minimality & normalized switch penalty**: `f_c` is computed over `(batch, seq)` after taking `max_tau g_{b,p,tau,c}` across loop passes (`pool_mode = "mean"`), with both `level_raw` and `switch_raw` normalized by `total_c` when `normalize_by_c = True`.
- **Exact Monte Carlo rank $p$-values (`v2`)**: Against $K = 199$ independent Haar rotations $\{r_1, \dots, r_K\}$, upper-tail rank $p$-values are computed without Gaussian tail assumptions as
  $$p_{\text{rank}} = \frac{1 + \#\{k \in \{1,\dots,K\} : r_k \ge r_{\text{obs}}\}}{K + 1}, \qquad p_{\min} = \frac{1}{200} = 0.005.$$

---

## Amendment v2 Gate Definitions & Registered Thresholds (`results/v2/`)

### Gate `B1 v2` — Like-for-Like Unrolled VPD vs Weight-Tied BPD (`bench_unrolled_vs_tied.py`)

- **Protocol**: On `StationaryRoutedLooper` (`vocab_size = 32`, `d_model = 32`, `n_passes = 4`, `c_shared = 8` orthogonal rank-1 mechanisms, `k_active = c_shared // n_passes = 2` active mechanisms per token across all passes, `n_steps = 150`, `batch_size = 32`, `seq_len = 8`, `seeds = (7, 8, 9)`), train all three arms under identical **unhinged** trajectory KL (`use_fragility_hinge = False`, `hinge_kappa = 0.0`), identical `zero` ablation, identical 20-step faithfulness warmup, and identical linear LR decay:
  1. `tied_bpd`: `C = 8` shared components across `T = 4` passes,
  2. `unrolled_vpd_TxC`: `T * C = 32` independent components (`8` per pass),
  3. `unrolled_vpd_budget_matched`: `C // T = 2 = k_active` components per pass (`8` total components — sufficient per-pass rank to represent any single item's active subspace).
- **Registered threshold (`gate_b1_passed`)**:
  - `tied_c * n_passes == unrolled_full_c` (`8 * 4 == 32`),
  - `mean_tied_recon_loss <= max(2.0 * mean_unrolled_TxC_recon_loss, 1e-5)` (no `5e-3` floor), **and**
  - `mean_tied_recon_loss <= 0.25 * mean_unrolled_budget_matched_recon_loss`.

### Gate `B2 v2` — `HUG-CAL-02` Restored `o_proj + down_proj` Negative Arm & Natural-Scale `sigma` Sweep (`audit_hug_cal02_null.py`)

- **Protocol**: On 4-block `SettlingCoreLooper` (`n_passes = 6`, `n_rotations = 199`, `batch_size = 32`, `seq_len = 12`, `seed = 11`):
  1. **Negative arm**: Audit all 8 leading settling factors `core.{0..3}.{o_proj, down_proj}` (`component = 0`) against `K = 199` RMS-matched Haar read rotations, reporting `raw_importance`, `net_importance`, `z_rot`, `exceed_count`, and exact `p_rank`.
  2. **Positive `sigma` sweep**: Measure `R_max = max_natural_raw_cal` across those 8 factors on disjoint `cal_seed = 777`. For each multiplier `m in (0.5, 1.0, 2.0, 4.0)`, binary-search `sigma_m` on `cal_tok` so the planted rank-1 perturbation on `core.3.down_proj` induces raw final KL `m * R_max`. Evaluate out-of-sample on `eval_tokens` (`seed = 11`) under `K = 199` Haar rotations for both:
     - `instrument_power` (exact planted rank-1 factor), and
     - `blind_bpd_recovery` (full-rank `C = 64` BPD trained for `bpd_steps = 80` strictly on `cal_seed`, single component `c*` chosen strictly on `cal_tok`, audited once out-of-sample on `eval_tokens`).
- **Registered threshold (`gate_b2_passed`)**:
  1. **Negative arm**: `max_raw_importance > 0.0` **and** `min_{i in 1..8} p_rank(i) > 0.05 / 8 = 0.00625` (Bonferroni FWER `0.05` across all 8 `o_proj + down_proj` factors).
  2. **Positive `sigma` sweep**: For every multiplier `m >= 1.0` (`1.0x, 2.0x, 4.0x` of `max_natural_raw_cal`), **both** `instrument_power` and `blind_bpd_recovery` achieve `net_importance > 0.0` and `p_rank <= 0.01` (`exceed_count <= 1` out of `199` rotations; `0.5x` is logged transparently to report sub-natural detection limits).

### Gate `B3 v2` — `LoopedChase` Natural-Scale `sigma` Sweep & Nested Multi-Seed Q4 Suite (`run_looped_chase_bpd.py --suite`)

- **Protocol**:
  1. **Natural-scale `sigma` sweep**: On `lam0.1_s1.final.pt` (`n_passes = 8`, `K = 199` rotations), measure `R_max` across `core.*.{proj, fc2}` on `cal_seed = 777`, calibrate `sigma_m` for `m in (0.5, 1.0, 2.0, 4.0)` of `R_max`, and evaluate out-of-sample on `eval_seed = 9000` (`batch_size = 64`) for both `instrument_power` and full-rank (`C = 256`, `bpd_steps = 100`) `blind_bpd_recovery`.
  2. **Q4 checkpoint suite**: Fit full-rank BPD (`n_passes = 8`, `c_per_mod = 256`, `n_steps = 200`, `ci_head_steps = 120`, `batch_size = 64`, `coeff_det = 0.5`, `hinge_kappa = 0.10`) across `lam0.1_s1`, `lam1_s1`, `lam1_s2`, and `lam1_s3`. Refit `state_only`, nested warm-started `state_plus_clock` (`W_clock = 0` at step 0), and all-token `V_head` `visible_only` CI heads across **`n_ci_seeds = 3` seeds** with disjoint `val_tok` checkpoint selection.
- **Registered threshold (`gate_b3_passed`)**:
  1. **Positive `sigma` sweep**: For every multiplier `m >= 1.0` (`1.0x, 2.0x, 4.0x`), both `instrument_power` and `blind_bpd_recovery` satisfy `net_importance > 0.0` and `p_rank <= 0.01`.
  2. **Nested clock validity**: Across all 4 checkpoints, mean `clock_gap_recon = recon(state_only) - recon(state_plus_clock) >= -0.02`.
  3. **Stationary `lam = 0.1` (`lam0.1_s1`)**: `stationary_component_fraction >= 0.50`, `clock_gap_recon <= 0.05`, and `visible_gap_recon <= 0.15`.
  4. **Noise-resolved off-head `lam = 1.0` visibility separation**: Both `lam1_s1` and `lam1_s2` exceed **both** `lam1_s3` and `lam0.1_s1` by more than `2 * combined_sd` (and minimum effect `0.05` nats on `visible_gap_recon` or `2.0` components on `visible_gap_active`).

### Gate `B4 v2` — Full-Rank Separate `TRM` Decompositions, Intra-`gaoxin492` Necessity, & Intra-`yagizdevre` Stabilisation (`run_trm_bpd.py`)

- **Protocol**:
  1. **Full-rank multi-board decompositions**: Decompose SHA-256-pinned 7M `gaoxin492` and `yagizdevre` independently at full rank `c_per_mod = 512` across `layers.{0,1}.{o_proj, down_proj}` on `n_train = 128` disjoint `train.csv` boards (`batch_size = 32`), `train_outer_steps = 3` (`63` inner passes), `n_steps = 240`, `coeff_det = 0.5`, `coeff_stoch = 0.5`, `coeff_adv = 1.0`, `inner_steps = 2`, `hinge_kappa = 0.10`, reporting both `match_inner_boundary_states = False` and `True` plus OOD `6`-outer-step (`126`-pass) reconstruction metrics, PGD stability curves (`0..160` steps), wired `interventional_path_demo`, and 3-seed warm-started `switch_penalty_sweep` across `(0.0, 2e-4, 2e-3, 2e-2)`.
  2. **Duty-cycle-ranked profiling & strict functional matching**: Inspect top `8` active components per module (`32` per checkpoint), match strictly with `require_ref_sustaining = True` and `require_tgt_sustaining = True` (`tgt_net_sustaining > 0`), and log `unmatched_similarity_baseline` (raw + module-centered off-diagonal distribution and `199`-permutation rank $p$-value).
  3. **Arm A (`arm_a_gaoxin492_sustaining_necessity`)**: Ablate top `top_k_edit = 4` sustaining components inside `gaoxin492` (`scale_delta = -1.0`) on `WB-Stall` (`n = 183`, `eval_horizons = (16, 32, 64)`) against **`K = 39` matched-count, matched-Frobenius-norm random active-component ablations** (`duty_cycle > 0.05`, `sign = -1.0`) and **`K = 39` RMS-calibrated Haar rotation draws**, logging exact McNemar discordance counts.
  4. **Arm B (`sustaining_edit_test`, intra-`yagizdevre` stabilisation)**: Scale the `top_k_edit = 4` functionally matched sustaining components inside `yagizdevre` (`edit_scale_delta = 0.15`) on `WB-Stall` (`n = 183`, `eval_horizons = (16, 32, 64)`) against **`K = 39` matched-count, matched-Frobenius-norm random active-component edits** (`duty_cycle > 0.05`, `sign = +1.0`) and **`K = 39` RMS-calibrated Haar rotation draws**, logging `final_solve_rate`, `ever_solved_rate`, `unsolved_tail_cycle_rate` (`s = 33..64`), and exact McNemar discordance counts.
- **Registered threshold (`gate_b4_passed`)**: Requires **all three** sub-gates simultaneously (`fidelity_ok and arm_a_passed and arm_b_passed`):
  1. **`fidelity_ok`**: `final_kl(gaoxin492) <= 0.15`, `final_kl(yagizdevre) <= 0.15`, **and** `interventional_path_demo.direct_source_effect > 0.0`.
  2. **`arm_a_passed` (intra-`gaoxin492` sustaining necessity)**:
     - `final_solve_rate(ablated_gaoxin492) < final_solve_rate(clean_gaoxin492)` with exact two-sided McNemar $p \le 0.05$,
     - `empirical_p_rank <= 0.05` against the `K = 39` random active-component ablation distribution (`(1 + exceed) / 40 <= 0.05`), **and**
     - `empirical_p_rank <= 0.05` against the `K = 39` rotation-null ablation distribution.
  3. **`arm_b_passed` (intra-`yagizdevre` functional stabilisation)**:
     - Every selected match has `tgt_net_sustaining > 0.0`,
     - `final_solve_rate(edited_yagizdevre) > final_solve_rate(clean_yagizdevre)` with exact two-sided McNemar $p \le 0.05$,
     - `unsolved_tail_cycle_rate(edited_yagizdevre) < unsolved_tail_cycle_rate(clean_yagizdevre)`,
     - `empirical_p_rank <= 0.05` against the `K = 39` random active-component edit distribution,
     - `empirical_p_rank <= 0.05` against the `K = 39` rotation-null edit distribution, **and**
     - `board_exact@16(edited_yagizdevre) >= board_exact@16(clean_yagizdevre) - 0.02`.

---

## Historical Archive: Amendment 1 (`v1`, `49c7207`) Thresholds

*(Preserved verbatim for auditability of `mechinterp/bpd/results/gate_b{1,2,3,4}_receipt.json`.)*

- **`B1` (`v1`)**: `StationaryResidualLooper` (`n_passes = 4`, `c_shared = 4`, `hinge_kappa = 0.10`), requiring `tied_bpd.recon_loss <= max(1.5 * unrolled_vpd_TxC.recon_loss, 5e-3)` and `< unrolled_vpd_budget_matched.recon_loss`.
- **`B3` (`v1`)**: `lam0.1_s1` planted rank-1 (`sigma = 600.0`, `K = 16`, `z >= 2.5`) + 1-seed `c_per_mod = 16` CI refits across `lam0.1_s1`, `lam1_s1..3`.
- **`B2` (`v1`)**: `core.{0..3}.down_proj` (`c in {0, 1}`) 8-factor negative control (`K = 16`, `|z_rot| <= 2.5`) + `cancel_fraction = 1.25` planted positive arm (`z_rot >= 2.5`).
- **`B4` (`v1`)**: `c_per_mod = 24`, `n_train = 32`, `train_outer_steps = 2` decompositions + single-draw random/rotation controls + `period2_oscillation_rate` reduction on `yagizdevre`.
