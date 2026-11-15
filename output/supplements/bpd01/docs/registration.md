# Pre-Registered Falsifiable Gates for BPD (`B1`–`B4`)

Every core design claim in BPD is paired with an explicit test that can fail. Metrics and pass/fail boundaries are fixed prior to running full-scale GPU checkpoints.

---

## Execution Order & Compute Discipline

1. **Start with 7M `TRM` (`run_trm_bpd.py`)**: At 7M parameters (`gaoxin492` and `yagizdevre`, SHA-256 pinned), `PureSudokuTRM` is the only pretrained recurrent reasoning model where full closed-loop adversarial BPD training, PGD step-stability sweeps (`0..160` steps), and switch-penalty sweeps are cheap.
2. **Stationary Unrolled Baseline (`bench_unrolled_vs_tied.py`)**: Controlled synthetic and `LoopedChase` (`lam = 0.1`) Pareto comparison against unrolled single-site-per-pass VPD.
3. **`LoopedChase` Ground Truth (`run_looped_chase_bpd.py`)**: `lam = 0.1` vs `lam = 1.0` seeds 1–3 (`Price(R_4) = +3.06, +5.46` vs `0.00`).
4. **`HUG-CAL-02` Negative Control (`audit_hug_cal02_null.py`)**: 8-factor settling control on `SettlingCoreLooper` and `tomg-group-umd/huginn-0125`.

---

## Gate Definitions & Failure Conditions

### Gate `B1` — Unrolled VPD Baseline (`bench_unrolled_vs_tied.py`)

- **Protocol**: Train weight-tied BPD with $C$ components per shared matrix against unrolled VPD with $T \cdot C$ total components ($C$ independent components per pass) and budget-matched unrolled VPD ($\max(C // T, 1)$ per pass) on a stationary weight-tied loop of depth $T \ge 4$.
- **Pass criterion**: Tied BPD ($C$ total components) achieves trajectory reconstruction KL within $1.5\times$ (or $\le 5 \times 10^{-3}$ absolute KL) of $T \cdot C$ unrolled VPD while strictly beating budget-matched unrolled VPD.
- **Failure meaning**: If tied BPD requires $\sim T \cdot C$ components to match unrolled VPD on a stationary loop, tying the parameter decomposition buys nothing.

### Gate `B2` — `HUG-CAL-02` Negative Control (`audit_hug_cal02_null.py`)

- **Protocol**: Evaluate the 8 leading core-block factors across `core.{0..3}.{o_proj, down_proj}` against $K \ge 16$ RMS-matched Haar read rotations, reporting **both** `raw_importance` and `net_importance` side by side.
- **Pass criterion**: `max_raw_importance > 0` (confirming raw fragility is active) **and** $\max_{i \in \{1..8\}} |z_{\text{rot}}(i)| \le 2.5$ (`net_importance` statistically indistinguishable from zero across all 8 settling factors).
- **Failure meaning**: If any of the 8 isotropic settling factors yields large positive `net_importance` ($z_{\text{rot}} > 2.5$), BPD's null calibration fails to reject loop-settling fragility artifacts.

### Gate `B3` — `LoopedChase` Ground Truth (`run_looped_chase_bpd.py`)

- **Protocol**: Fit BPD and evaluate `state_only`, `state_plus_clock`, and `visible_only` CI heads on `LoopedChase` at `lam = 0.1` vs `lam = 1.0` (seeds 1, 2, 3).
- **Pass criterion**:
  1. `lam = 0.1` recovers a stationary hop schedule (`stationary_component_fraction >= 0.5`, `|clock_gap_recon| <= 0.02`, `|visible_gap_recon| <= 0.02`).
  2. `lam = 1.0` seeds 1 & 2 (`Price(R_4) > 3` loops) exhibit strictly larger `visible_gap_recon` (or `visible_gap_active` at matched KL) than `lam = 1.0` seed 3 (`Price(R_4) = 0.00`), surfacing off-head $\ker(\text{head})$ phase components absent in seed 3.
- **Failure meaning**: If `visible_gap` fails to separate `lam = 1.0` seeds 1–2 from seed 3 and `lam = 0.1`, BPD's `visible_only` CI fails to capture dark-fiber parameter routing.

### Gate `B4` — `TRM` Sustaining Weight Edit at `s = 64` (`run_trm_bpd.py`)

- **Protocol**: Fit hybrid clean-window + closed-loop BPD on 7M `gaoxin492`, classify `transient` vs `sustaining` components (reporting raw and net-of-null importance), rank `sustaining` components by Frobenius projection gap against `yagizdevre`, and apply a permanent time-shared rank-$k$ weight edit $W_{\text{tgt}} \leftarrow W_{\text{tgt}} + \sum_{c \in S} (\alpha_{\text{ref},c} - \alpha_{\text{tgt},c}) u_c v_c^\top$ to `yagizdevre` with **no** runtime refill network.
- **Pass criterion**: On held-out 9x9 Sudoku boards at OOD horizon $s = 64$, the edited `yagizdevre` checkpoint increases exact board accuracy and reduces period-2 limit-cycle oscillation relative to unedited `yagizdevre` without degrading $s = 16$ accuracy by $> 2$ percentage points.
- **Failure meaning**: If scaling/transplanting sustaining BPD components cannot reduce `yagizdevre`'s $s = 64$ limit-cycle collapse without an external `C5` refill network, either the stabilization requires nonlinear input-conditional dark-fiber synthesis beyond static rank-$k$ shared weights or BPD's sustaining identification missed the stabilizing subspace.

---

## Mandatory Risk Controls Reported in Every Run

1. **Adversary step curve (`pgd_stability`)**: Report reconstruction KL across $K_{\text{adv}} \in \{0, 2, 5, 20, 80, 160\}$ for both `shared_batch_and_pass` and `per_pass` scopes.
2. **Dual raw + net reporting**: Never drop `raw_importance` when reporting `net_importance`, guarding against over-conservative rotation nulls at fragile passes.
3. **Switch-penalty sweep (`switch_penalty_sweep`)**: Sweep `coeff_switch` across at least three values (including `0.0`) on frozen $(V, U)$ to detect over-merging of distinct same-phase mechanisms.
