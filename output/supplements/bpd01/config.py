"""Configuration dataclasses for Behavioral Parameter Decomposition (BPD).

BPD extends adVersarial Parameter Decomposition (VPD; Goodfire 2026) to
weight-tied, recurrent-depth computation. At ``T = 1`` under
``BPDConfig.vpd_single_pass_equivalent``, BPD matches the four-term PyTorch VPD
port (``MoralityLabAI/VPD`` ``experiments/nanbeige_looped/vpd_math.py`` at commit
``69509c39e96a912f99b1a83640250ac01797c571``; the Goodfire paper adds a fifth
deterministic CI-masked reconstruction term).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
from typing import Literal


def resolve_registration_commit(explicit: str | None = None) -> str:
    """Return explicit commit hash or latest git SHA of ``mechinterp/bpd/docs/registration.md``."""
    if explicit:
        return explicit.strip()
    reg_path = Path(__file__).resolve().parent / "docs" / "registration.md"
    repo_root = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "log", "-n", "1", "--format=%H", "--", str(reg_path)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
        if out:
            return out
    except Exception:
        pass
    return "uncommitted"

CIVariant = Literal["state_only", "state_plus_clock", "visible_only"]
AblationMode = Literal["pass_mean", "zero"]
MinimalityPoolMode = Literal["mean", "max", "sum_legacy"]
AdversaryScope = Literal["shared_batch_and_pass", "shared_pass_per_batch", "per_pass"]
RolloutMode = Literal["closed_loop", "clean_window", "hybrid"]


@dataclass(frozen=True)
class AblationConfig:
    """Controls neutral ablation baseline and recurrent manifold calibration."""

    mode: AblationMode = "pass_mean"
    sphere_rescale: bool = True
    ema_momentum: float = 0.1
    eps: float = 1e-6


@dataclass(frozen=True)
class MinimalityConfig:
    """Pass-invariant component minimality plus temporal switch penalty along tau."""

    pool_mode: MinimalityPoolMode = "mean"
    p_start: float = 2.0
    p_end: float = 0.4
    eps: float = 1e-12
    beta: float = 0.5
    coeff_level: float = 2e-4
    coeff_switch: float = 2e-4
    switch_q: float = 1.0
    normalize_by_c: bool = True


@dataclass(frozen=True)
class AdversaryConfig:
    """Adversarial mask optimization (time-shared across passes by default)."""

    scope: AdversaryScope = "shared_batch_and_pass"
    lr: float = 0.01
    lr_final_frac: float = 0.1
    beta1: float = 0.5
    beta2: float = 0.99
    eps: float = 1e-8
    inner_steps: int = 2
    eval_step_size: float = 0.1
    eval_steps_grid: tuple[int, ...] = (0, 2, 5, 20, 80, 160)


@dataclass(frozen=True)
class CIConfig:
    """Configuration for the pass-tied Causal Importance (CI) network."""

    variant: CIVariant = "state_only"
    d_model: int = 256
    n_blocks: int = 2
    n_heads: int = 4
    mlp_hidden: int = 512
    max_passes: int = 64
    max_seq_len: int = 512
    clock_dim: int = 32
    rope_base: float = 10000.0
    leaky_alpha: float = 0.01
    final_rms_norm: bool = True


@dataclass(frozen=True)
class NullCalibrationConfig:
    """Norm/RMS-matched Haar rotation null for per-pass fragility hinge calibration."""

    use_fragility_hinge: bool = True
    hinge_kappa: float = 0.10
    n_rotations_train: int = 4
    n_rotations_eval: int = 16
    calibrate_every: int = 50
    perturbation_scale: float = 0.10
    eps: float = 1e-6


@dataclass(frozen=True)
class BPDConfig:
    """Top-level training and evaluation configuration for BPD."""

    c_per_module: dict[str, int]
    n_passes: int = 1
    n_steps: int = 1000
    batch_size: int = 32
    seq_len: int = 97
    seed: int = 0

    # Optimizer & warmup
    init_svd: bool = True
    faithfulness_warmup_steps: int = 100
    faithfulness_warmup_lr: float = 1e-3
    main_lr: float = 1e-4
    main_lr_final_frac: float = 0.1
    grad_clip_components: float = 1.0

    # Objective weights & trajectory state-matching switch
    coeff_faith: float = 1e5
    coeff_stoch: float = 0.5
    coeff_adv: float = 0.5
    match_inner_boundary_states: bool = False
    state_boundary_weight: float = 1.0

    # Hybrid clean-transplant vs closed-loop rollout schedule (v014 distinction)
    rollout_mode: RolloutMode = "hybrid"
    window_steps: int = 2
    closed_loop_prob: float = 0.5

    # Sub-configs
    ci: CIConfig = field(default_factory=CIConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    minimality: MinimalityConfig = field(default_factory=MinimalityConfig)
    adversary: AdversaryConfig = field(default_factory=AdversaryConfig)
    null_calibration: NullCalibrationConfig = field(default_factory=NullCalibrationConfig)

    @classmethod
    def vpd_single_pass_equivalent(
        cls,
        c_per_module: dict[str, int],
        *,
        seq_len: int = 64,
        d_model: int = 128,
        n_blocks: int = 2,
        n_heads: int = 4,
    ) -> "BPDConfig":
        """Construct a T=1 BPDConfig matching the four-term PyTorch VPD port (vpd_math.py @ 69509c39)."""
        return cls(
            c_per_module=dict(c_per_module),
            n_passes=1,
            seq_len=seq_len,
            match_inner_boundary_states=False,
            rollout_mode="closed_loop",
            ci=CIConfig(
                variant="state_only",
                d_model=d_model,
                n_blocks=n_blocks,
                n_heads=n_heads,
                mlp_hidden=2 * d_model,
                max_passes=1,
                max_seq_len=max(seq_len, 16),
                final_rms_norm=False,
            ),
            ablation=AblationConfig(mode="zero", sphere_rescale=False),
            minimality=MinimalityConfig(pool_mode="mean", coeff_switch=0.0, normalize_by_c=False),
            adversary=AdversaryConfig(scope="per_pass"),
            null_calibration=NullCalibrationConfig(use_fragility_hinge=False, hinge_kappa=0.0),
        )
