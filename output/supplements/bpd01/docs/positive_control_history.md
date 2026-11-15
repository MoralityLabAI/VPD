# `test_bpd_ci_and_nulls.py` Run History (`positive_control_history`)

## Git Recovery Status

Git history cannot recover the four intermediate failing states of `mechinterp/bpd/tests/test_bpd_ci_and_nulls.py` because all five pytest runs occurred in the working tree before the initial branch commit [`e6baa744`](https://github.com/MoralityLabAI/LoopedTransformers/commit/e6baa744b82d91cd2812a73799c5d1aea5de2ce8) on `feat/mechinterp-bpd`.

The exact parameter values, thresholds, returned numbers, and reasons for each change across the four failing runs (`task-143`, `task-177`, `task-185`, `task-193`) and the passing run (`task-198`, committed at `e6baa744`) are preserved in the execution transcript (`ec133c90-6a80-4d63-97ed-b661eab7fba9`) and tabulated below.

---

## Run-by-Run Parameter and Threshold Log

### Summary Table

| Run | Status | `test_visible_gap` LR / steps / threshold | `SettlingCoreLooper` (neg. control) Haar `d_in` / rotations / threshold | Positive control Haar `d_in` / rotations / threshold | Outcome & failure value |
|---|---|---|---|---|---|
| **1 (`task-143`)** | Failed | `main_lr = 1e-2`, `n_steps = 90`, `batch_size = 32`; `total_cost_vis > total_cost_state + 0.03` and (`visible_gap_recon > 0.01` or `visible_gap_active > 0.25`) | `d_model = 48` (`o_proj` `d_in = 48`, `down_proj` `d_in = 96`), `n_passes = 5`, `n_rotations = 16`; `max_abs_z_rot <= 2.8` | `_FragileTwoPass` spike: `d = 32`, `n_rotations = 12`; `raw_importance > rotation_null_mean` | `test_visible_gap`: `NameError: name 'F' is not defined`. Neg. control: `max_abs_z_rot = 5.0049 > 2.8` on `core.0.o_proj` (`raw = 0.3853`, `null_mean = 0.1253`; pos. control line not reached). |
| **2 (`task-177`)** | Failed | `main_lr = 1e-2`, `n_steps = 90`, `batch_size = 32`; same threshold | `d_model = 48` (`down_proj` `d_in = 96`), `n_passes = 5`, `n_rotations = 16`; `max_abs_z_rot <= 2.8` | `_FragileTwoPass` spike: `d = 32`, `n_rotations = 12`; `raw_importance > rotation_null_mean` | `test_visible_gap` passed; neg. control passed; pos. control failed: `raw_importance = 10.1582 <= rotation_null_mean = 44.9369` (`z_rot = -3.2735`). |
| **3 (`task-185`)** | Failed | `main_lr = 1.5e-2`, `n_steps = 45`, `batch_size = 16`; same threshold | `d_model = 32` (`down_proj` `d_in = 64`), `n_passes = 4`, `n_rotations = 8`; `max_abs_z_rot <= 2.8` | `_SelectiveTwoPass` (`q_sign` in `{-1, +1}`): `d = 32`, `n_rotations = 12`; `raw > null_mean`, `net > 0.05`, `z_rot > 2.0` | `test_visible_gap` failed: `0.0705 <= 0.0892 + 0.03`; neg. control failed: `max_abs_z_rot = 4.5194 > 2.8` on `core.3.down_proj` `c=0` (`raw = 0.0165`, `null_mean = 0.0074`, `null_std = 0.0020`; pos. control line not reached). |
| **4 (`task-193`)** | Failed | `main_lr = 1e-2`, `n_steps = 90`, `batch_size = 32`; same threshold | `d_model = 48` (`down_proj` `d_in = 96`), `n_passes = 5`, `n_rotations = 16`; `max_abs_z_rot <= 2.8` | `_SelectiveTwoPass` (`q_sign` in `{-1, +1}`): `d = 32`, `n_rotations = 12`; `raw > null_mean`, `net > 0.05`, `z_rot > 2.0` | `test_visible_gap` passed; neg. control passed; pos. control passed `raw > null_mean` (`0.6691 > 0.5629`) and `net > 0.05` (`0.1061`), failed `z_rot = 0.7691 <= 2.0` (`null_std = 0.1380`). |
| **5 (`task-198`, `e6baa744`)** | Passed | `main_lr = 1e-2`, `n_steps = 90`, `batch_size = 32`; same threshold | `d_model = 48` (`down_proj` `d_in = 96`), `n_passes = 5`, `n_rotations = 16`; `max_abs_z_rot <= 2.8` | `_SelectiveTwoPass` (`active_gate` in `{0.0, 1.0}`, 1-of-8 positions): `d = 32`, `n_rotations = 12`; `raw > null_mean`, `net > 0.05`, `z_rot > 2.0` | All 4 tests in `test_bpd_ci_and_nulls.py` (and all 13 tests in `mechinterp/bpd/tests`) passed. |

---

## Reasons for Each Change

1. **Run 1 → Run 2**:
   - Added missing `import torch.nn.functional as F` in `test_bpd_ci_and_nulls.py`.
   - Fixed an asymmetric intervention bug in `components.py::_apply_intervention`: `mode="override_read"` (rotation null) was entering `sphere_rescale=True` while `mode="add_alpha"` (raw subtraction) was not, clamping null norms while leaving raw subtraction unrescaled.
   - Added `sample_per_pass_rms_matched_rotations` in `nulls.py` so multi-pass rotations match empirical projection RMS on each pass $\tau$ rather than a single pooled scalar across passes, and scoped `evaluate_hug_cal02_eight_factors` in `adapters/huginn.py` to top-2 SVD factors of `core.{0..3}.down_proj` (`d_in = 2 * d_model = 96`).
2. **Run 2 → Run 3**:
   - In Run 2, `_FragileTwoPass` amplified pass-0 writes by $\sqrt{1808.4} \approx 42.53$ into an unselective identity readout; subtracting `(x @ v0) u0` moved logits toward zero (`KL = 10.16`), whereas subtracting an RMS-matched orthogonal `(x @ r_k) u0` left `+170` from `v0` intact and flipped argmax signs (`KL = 44.94`). Replaced the positive-control model with `_SelectiveTwoPass` (`q_sign * y0[..., 0]`) and added explicit `net_importance > 0.05` and `z_rot > 2.0` assertions.
   - Simultaneously reduced `n_steps` (`90 -> 45`, `main_lr: 1e-2 -> 1.5e-2`, `batch_size: 32 -> 16`) in `test_visible_gap` and shrunk `SettlingCoreLooper` (`d_model: 48 -> 32`, `n_rotations: 16 -> 8`, `n_passes: 5 -> 4`, `batch: 20 -> 12`) to reduce CPU pytest runtime.
3. **Run 3 → Run 4**:
   - The Run 3 speedup reductions broke convergence in `test_visible_gap` (`main_lr = 1.5e-2` overshot the `lower_leaky_sigmoid` clamp inside 45 steps) and increased finite-width fourth-moment variance on `core.3.down_proj` at `d_model = 32`, `n_rotations = 8` (`net = +0.0091` nats divided by `null_std = 0.0020` gave `z_rot = 4.52`). Reverted both `test_visible_gap` and `SettlingCoreLooper` to their Run 2 values (`main_lr = 1e-2, n_steps = 90` and `d_model = 48, n_passes = 5, n_rotations = 16`) and used `torch.set_num_threads(2)` in `conftest.py` for CPU runtime instead.
4. **Run 4 → Run 5 (`e6baa744`)**:
   - With dense `q_sign in {-1, +1}` on every position, random orthogonal rotations still flipped the binary margin sign on tail draws (`null_mean = 0.5629` vs `raw = 0.6691`, `z_rot = 0.7691`). Replaced the dense sign multiplication in `_SelectiveTwoPass` with a sparse 1-in-8 position gate (`active_gate[:, 0] = 1.0`, `x_iso[:, :, 0] = active_gate * sqrt(8.0)` so pass-wide RMS stays `1.0` across all coordinates while `v0` fires in phase with the gate), keeping `d = 32`, `n_rotations = 12`, and thresholds `raw > null_mean`, `net > 0.05`, `z_rot > 2.0` unchanged.
