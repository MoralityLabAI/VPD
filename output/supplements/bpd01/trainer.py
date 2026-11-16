"""BPD training engine (hybrid clean-transplant windows + closed-loop rollouts)
and unrolled single-site-per-pass VPD baseline trainer for Gate B1.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn

from mechinterp.bpd.adversary import PassSharedPersistentPGD
from mechinterp.bpd.ci import (
    CIOutputs,
    TiedCINetwork,
    UnrolledCINetwork,
    active_component_budget,
    compute_ci_variant_gaps,
)
from mechinterp.bpd.components import (
    BPDModel,
    PassMaskSchedule,
    RolloutTrajectory,
    UnrolledComponentLinear,
)
from mechinterp.bpd.config import BPDConfig, CIVariant, MinimalityConfig
from mechinterp.bpd.losses import (
    bpd_minimality_loss,
    compute_pass_loss_shares,
    deterministic_ci_schedules,
    faithfulness_loss,
    kl_divergence_logits,
    sample_stochastic_schedules,
    trajectory_recon_loss,
    unrolled_faithfulness_loss,
)
from mechinterp.bpd.nulls import calibrate_pass_null_losses

# BatchRolloutAdapter executes target_model over `n_passes` starting at `pass_offset`
# (optionally using cached clean state `init_state`) and returns:
#   (readouts, boundary_states, visible_states, prompt_features)
AdapterRolloutFn = Callable[
    [Any, int, int, Any],
    tuple[
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor] | None,
        Tensor | None,
    ],
]
CleanStateSamplerFn = Callable[[Any, int], Any]
ValidWindowStartsFn = Callable[[int, int, bool], Sequence[int]]


@dataclass
class StepMetrics:
    """Per-step scalar training diagnostics."""

    step: int
    mode: str
    total_loss: float
    faith_loss: float
    min_loss: float
    switch_loss: float
    stoch_loss: float
    adv_loss: float
    active_per_pass: float
    pass_loss_shares: dict[int, float] = field(default_factory=dict)


class BPDTrainer:
    """Trains shared ``(V, U)`` components and pass-tied ``TiedCINetwork`` on a looped model."""

    def __init__(
        self,
        bpd_model: BPDModel,
        cfg: BPDConfig,
        rollout_adapter: AdapterRolloutFn,
        *,
        clean_window_state_fn: CleanStateSamplerFn | None = None,
        valid_window_starts_fn: ValidWindowStartsFn | None = None,
        visible_dim: int | None = None,
        prompt_dim: int | None = None,
        is_causal_ci: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.bpd_model = bpd_model.to(self.device)
        self.cfg = cfg
        self.rollout_adapter = rollout_adapter
        self.clean_window_state_fn = clean_window_state_fn
        self.valid_window_starts_fn = valid_window_starts_fn

        module_dims = {name: bpd_model.wrappers[name].d_in for name in bpd_model.module_order}
        self.ci_net = TiedCINetwork(
            module_dims=module_dims,
            c_per_module=bpd_model.c_per_module,
            cfg=cfg.ci,
            visible_dim=visible_dim,
            prompt_dim=prompt_dim,
            is_causal=is_causal_ci,
        ).to(self.device)

        self.pgd = PassSharedPersistentPGD(
            bpd_model.c_per_module, cfg.adversary, device=self.device
        )
        self.pass_null_losses: dict[int, float] | None = None
        self.pass_tolerances: dict[int, float] | None = None
        self.train_pass_loss_accum: dict[int, float] = {}
        self.rng = torch.Generator(device=self.device).manual_seed(cfg.seed)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.bpd_model.component_parameters(), "lr": cfg.main_lr},
                {"params": list(self.ci_net.parameters()), "lr": cfg.main_lr},
            ],
            lr=cfg.main_lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )

    def _effective_hinge_kappa(self) -> float:
        if not self.cfg.null_calibration.use_fragility_hinge or self.cfg.n_passes <= 1:
            return 0.0
        return max(float(self.cfg.null_calibration.hinge_kappa), 0.0)

    def mean_train_pass_loss_shares(self) -> dict[int, float]:
        """Return each active pass's cumulative share of total training trajectory loss."""
        return compute_pass_loss_shares(self.train_pass_loss_accum)

    def _set_lr(self, step: int) -> float:
        if self.cfg.n_steps <= 1:
            lr = self.cfg.main_lr
        else:
            frac = min(max(float(step) / float(self.cfg.n_steps), 0.0), 1.0)
            lr = self.cfg.main_lr * (1.0 + (self.cfg.main_lr_final_frac - 1.0) * frac)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def run_clean_rollout(
        self,
        batch: Any,
        *,
        n_passes: int | None = None,
        pass_offset: int = 0,
        init_state: Any = None,
        update_means: bool = False,
    ) -> RolloutTrajectory:
        passes = self.cfg.n_passes if n_passes is None else int(n_passes)
        with torch.no_grad():
            with self.bpd_model.target_capture(
                pass_offset=pass_offset, update_means=update_means, detach_inputs=True
            ):
                readouts, boundaries, vis, prompt = self.rollout_adapter(
                    batch, passes, pass_offset, init_state
                )
            mod_inputs = self.bpd_model.collect_recorded_inputs(expected_passes=passes)
        return RolloutTrajectory(
            readouts=readouts,
            boundary_states=boundaries,
            module_inputs=mod_inputs,
            visible_states=vis,
            prompt_features=prompt,
        )

    def run_masked_rollout(
        self,
        batch: Any,
        schedules: dict[str, PassMaskSchedule],
        *,
        n_passes: int,
        pass_offset: int = 0,
        init_state: Any = None,
        record_inputs: bool = False,
    ) -> RolloutTrajectory:
        with self.bpd_model.masked_components(
            schedules, pass_offset=pass_offset, record_inputs=record_inputs
        ):
            readouts, boundaries, vis, prompt = self.rollout_adapter(
                batch, n_passes, pass_offset, init_state
            )
            mod_inputs = (
                self.bpd_model.collect_recorded_inputs(expected_passes=n_passes)
                if record_inputs
                else {}
            )
        return RolloutTrajectory(
            readouts=readouts,
            boundary_states=boundaries,
            module_inputs=mod_inputs,
            visible_states=vis,
            prompt_features=prompt,
        )

    def run_faithfulness_warmup(self, steps: int | None = None) -> float:
        """Warm up ``(V, U)`` on matrix reconstruction ``||W - (V U)^T||_F^2`` alone."""
        n_warm = self.cfg.faithfulness_warmup_steps if steps is None else int(steps)
        if n_warm <= 0:
            return float(faithfulness_loss(self.bpd_model.wrappers).detach().item())
        warm_opt = torch.optim.Adam(
            self.bpd_model.component_parameters(), lr=self.cfg.faithfulness_warmup_lr
        )
        last = 0.0
        for _ in range(n_warm):
            warm_opt.zero_grad(set_to_none=True)
            loss = faithfulness_loss(self.bpd_model.wrappers)
            loss.backward()
            warm_opt.step()
            last = float(loss.detach().item())
        return last

    def calibrate_baselines(self, batch: Any) -> dict[int, float]:
        """Calibrate pass-conditional mean writes and per-pass rotation-null hinge baselines."""
        clean_traj = self.run_clean_rollout(
            batch,
            n_passes=self.cfg.n_passes,
            pass_offset=0,
            init_state=None,
            update_means=(self.cfg.ablation.mode == "pass_mean"),
        )
        if self._effective_hinge_kappa() <= 0.0:
            self.pass_null_losses = {tau: 0.0 for tau in range(self.cfg.n_passes)}
            self.pass_tolerances = self.pass_null_losses
            return self.pass_null_losses

        def _plain_rollout() -> RolloutTrajectory:
            readouts, boundaries, vis, prompt = self.rollout_adapter(
                batch, self.cfg.n_passes, 0, None
            )
            return RolloutTrajectory(
                readouts=readouts,
                boundary_states=boundaries,
                module_inputs=clean_traj.module_inputs,
                visible_states=vis,
                prompt_features=prompt,
            )

        self.pass_null_losses = calibrate_pass_null_losses(
            self.bpd_model,
            clean_traj,
            _plain_rollout,
            n_rotations=self.cfg.null_calibration.n_rotations_train,
            perturbation_scale=self.cfg.null_calibration.perturbation_scale,
            match_inner_boundary_states=self.cfg.match_inner_boundary_states,
            state_boundary_weight=self.cfg.state_boundary_weight,
            eps=self.cfg.null_calibration.eps,
            seed=self.cfg.seed,
        )
        self.pass_tolerances = self.pass_null_losses
        return self.pass_null_losses

    def _select_rollout_regime(self, batch: Any) -> tuple[str, int, int, Any]:
        """Choose full closed-loop rollout vs short clean-state transplant window."""
        mode = self.cfg.rollout_mode
        can_window = (
            self.clean_window_state_fn is not None
            and self.cfg.n_passes > self.cfg.window_steps
            and self.cfg.window_steps >= 1
        )
        if mode == "closed_loop" or not can_window:
            return "closed_loop", self.cfg.n_passes, 0, None

        if mode == "hybrid":
            coin = float(torch.rand((1,), generator=self.rng, device=self.device).item())
            if coin < self.cfg.closed_loop_prob:
                return "closed_loop", self.cfg.n_passes, 0, None

        assert self.clean_window_state_fn is not None
        if self.valid_window_starts_fn is not None:
            valid_starts = list(
                self.valid_window_starts_fn(
                    self.cfg.n_passes,
                    self.cfg.window_steps,
                    self.cfg.match_inner_boundary_states,
                )
            )
            if not valid_starts:
                return "closed_loop", self.cfg.n_passes, 0, None
            pick = int(
                torch.randint(0, len(valid_starts), (1,), generator=self.rng, device=self.device).item()
            )
            tau0 = int(valid_starts[pick])
        else:
            max_start = self.cfg.n_passes - self.cfg.window_steps
            tau0 = int(
                torch.randint(0, max_start + 1, (1,), generator=self.rng, device=self.device).item()
            )
        init_state = self.clean_window_state_fn(batch, tau0)
        return "clean_window", self.cfg.window_steps, tau0, init_state

    def train_step(self, batch: Any, step: int) -> StepMetrics:
        """Execute one joint BPD optimization step."""
        self._set_lr(step)
        kappa = self._effective_hinge_kappa()
        if (
            kappa > 0.0
            and (
                self.pass_null_losses is None
                or (
                    self.cfg.null_calibration.calibrate_every > 0
                    and step > 0
                    and step % self.cfg.null_calibration.calibrate_every == 0
                )
            )
        ):
            self.calibrate_baselines(batch)

        exec_mode, n_passes, pass_offset, init_state = self._select_rollout_regime(batch)

        clean_traj = self.run_clean_rollout(
            batch,
            n_passes=n_passes,
            pass_offset=pass_offset,
            init_state=init_state,
            update_means=(self.cfg.ablation.mode == "pass_mean"),
        )

        self.optimizer.zero_grad(set_to_none=True)
        ci_out = self.ci_net.forward_trajectory(clean_traj, pass_offset=pass_offset)
        ci_lower_leaf = {
            k: v.detach().requires_grad_(True) for k, v in ci_out.lower_leaky.items()
        }

        loss_faith = faithfulness_loss(self.bpd_model.wrappers)
        loss_min, min_stats = bpd_minimality_loss(
            ci_out.upper_leaky,
            step=step,
            total_steps=self.cfg.n_steps,
            cfg=self.cfg.minimality,
        )

        base_loss_val = float(
            (self.cfg.coeff_faith * loss_faith.detach() + loss_min.detach()).item()
        )
        step_pass_losses: dict[int, float] = {}

        # 1. Stochastic mask reconstruction branch (backprop immediately to free rollout graph)
        loss_stoch_val = 0.0
        if self.cfg.coeff_stoch > 0.0:
            share_noise = self.cfg.adversary.scope != "per_pass"
            stoch_sched = sample_stochastic_schedules(
                ci_lower_leaf,
                route_subset=True,
                share_noise_across_passes=share_noise,
                generator=self.rng,
            )
            stoch_traj = self.run_masked_rollout(
                batch,
                stoch_sched,
                n_passes=n_passes,
                pass_offset=pass_offset,
                init_state=init_state,
            )
            loss_stoch, stoch_per_pass = trajectory_recon_loss(
                stoch_traj,
                clean_traj,
                match_inner_boundary_states=self.cfg.match_inner_boundary_states,
                pass_null_losses=self.pass_null_losses,
                hinge_kappa=kappa,
                state_boundary_weight=self.cfg.state_boundary_weight,
            )
            for tau, val in stoch_per_pass.items():
                step_pass_losses[tau] = step_pass_losses.get(tau, 0.0) + self.cfg.coeff_stoch * val
            loss_stoch_val = float(loss_stoch.detach().item())
            if loss_stoch.requires_grad:
                (self.cfg.coeff_stoch * loss_stoch).backward()
            del stoch_traj, loss_stoch

        # 2. Persistent PGD adversarial mask reconstruction branch (single backward, no retain_graph)
        loss_adv_val = 0.0
        adv_source_grads: dict[str, Tensor] | None = None
        if self.cfg.coeff_adv > 0.0 and exec_mode == "closed_loop":
            adv_lr = self.pgd.lr_at(step, self.cfg.n_steps)
            if self.cfg.adversary.inner_steps > 0:
                ci_det = {k: v.detach() for k, v in ci_out.lower_leaky.items()}

                def _inner_eval(sched: dict[str, PassMaskSchedule]) -> Tensor:
                    tr = self.run_masked_rollout(
                        batch, sched, n_passes=n_passes, pass_offset=pass_offset, init_state=init_state
                    )
                    l_rec, _ = trajectory_recon_loss(
                        tr,
                        clean_traj,
                        match_inner_boundary_states=self.cfg.match_inner_boundary_states,
                        pass_null_losses=self.pass_null_losses,
                        hinge_kappa=kappa,
                        state_boundary_weight=self.cfg.state_boundary_weight,
                    )
                    return l_rec

                self.pgd.warmup_inner_steps(ci_det, _inner_eval, lr=adv_lr)

            adv_sched, leaf_sources = self.pgd.build_schedules(
                ci_lower_leaf, requires_grad_sources=True
            )
            adv_traj = self.run_masked_rollout(
                batch, adv_sched, n_passes=n_passes, pass_offset=pass_offset, init_state=init_state
            )
            loss_adv, adv_per_pass = trajectory_recon_loss(
                adv_traj,
                clean_traj,
                match_inner_boundary_states=self.cfg.match_inner_boundary_states,
                pass_null_losses=self.pass_null_losses,
                hinge_kappa=kappa,
                state_boundary_weight=self.cfg.state_boundary_weight,
            )
            for tau, val in adv_per_pass.items():
                step_pass_losses[tau] = step_pass_losses.get(tau, 0.0) + self.cfg.coeff_adv * val
            loss_adv_val = float(loss_adv.detach().item())
            if loss_adv.requires_grad:
                (self.cfg.coeff_adv * loss_adv).backward()
                adv_source_grads = {
                    name: (
                        leaf_sources[name].grad.detach() / self.cfg.coeff_adv
                        if leaf_sources[name].grad is not None
                        else torch.zeros_like(leaf_sources[name])
                    )
                    for name in self.bpd_model.module_order
                }
            else:
                adv_source_grads = {
                    name: torch.zeros_like(leaf_sources[name])
                    for name in self.bpd_model.module_order
                }
            del adv_traj, loss_adv
        elif self.cfg.coeff_adv > 0.0 and exec_mode == "clean_window":
            # On variable-length short transplant windows, apply fresh single-step adversarial probe
            # without mutating full-horizon persistent PGD tensor shapes.
            stoch_adv = sample_stochastic_schedules(
                ci_lower_leaf,
                route_subset=False,
                share_noise_across_passes=True,
                generator=self.rng,
            )
            adv_traj = self.run_masked_rollout(
                batch, stoch_adv, n_passes=n_passes, pass_offset=pass_offset, init_state=init_state
            )
            loss_adv, adv_per_pass = trajectory_recon_loss(
                adv_traj,
                clean_traj,
                match_inner_boundary_states=self.cfg.match_inner_boundary_states,
                pass_null_losses=self.pass_null_losses,
                hinge_kappa=kappa,
                state_boundary_weight=self.cfg.state_boundary_weight,
            )
            for tau, val in adv_per_pass.items():
                step_pass_losses[tau] = step_pass_losses.get(tau, 0.0) + self.cfg.coeff_adv * val
            loss_adv_val = float(loss_adv.detach().item())
            if loss_adv.requires_grad:
                (self.cfg.coeff_adv * loss_adv).backward()
            del adv_traj, loss_adv

        # 3. Faithfulness + minimality + accumulated CI leaf gradients through ci_net
        head_loss = self.cfg.coeff_faith * loss_faith + loss_min
        for name, leaf in ci_lower_leaf.items():
            if leaf.grad is not None:
                head_loss = head_loss + (ci_out.lower_leaky[name] * leaf.grad.detach()).sum()
        if head_loss.requires_grad:
            head_loss.backward()

        total_loss_val = (
            base_loss_val
            + self.cfg.coeff_stoch * loss_stoch_val
            + self.cfg.coeff_adv * loss_adv_val
        )
        if self.cfg.grad_clip_components > 0.0:
            nn.utils.clip_grad_norm_(
                self.bpd_model.component_parameters(), max_norm=self.cfg.grad_clip_components
            )
        self.optimizer.step()

        if adv_source_grads is not None:
            self.pgd.ascent_step(adv_source_grads, lr=self.pgd.lr_at(step, self.cfg.n_steps))

        for tau, val in step_pass_losses.items():
            self.train_pass_loss_accum[tau] = self.train_pass_loss_accum.get(tau, 0.0) + val
        step_shares = compute_pass_loss_shares(step_pass_losses)

        budget = active_component_budget(ci_out)
        return StepMetrics(
            step=step,
            mode=exec_mode,
            total_loss=total_loss_val,
            faith_loss=float(loss_faith.detach().item()),
            min_loss=min_stats["total"],
            switch_loss=min_stats["switch_weighted"],
            stoch_loss=loss_stoch_val,
            adv_loss=loss_adv_val,
            active_per_pass=budget["mean_active_per_pass"],
            pass_loss_shares=step_shares,
        )

    @torch.no_grad()
    def evaluate(
        self,
        batch: Any,
        *,
        n_passes: int | None = None,
        delta_value: float = 0.0,
        threshold: float | None = None,
    ) -> tuple[dict[str, float], CIOutputs, RolloutTrajectory]:
        """Evaluate deterministic CI reconstruction, per-pass loss shares, final KL, and budget."""
        passes = self.cfg.n_passes if n_passes is None else int(n_passes)
        clean_traj = self.run_clean_rollout(batch, n_passes=passes, pass_offset=0)
        ci_out = self.ci_net.forward_trajectory(clean_traj, pass_offset=0)
        sched = deterministic_ci_schedules(
            ci_out.lower_leaky, delta_value=delta_value, threshold=threshold
        )
        pred_traj = self.run_masked_rollout(batch, sched, n_passes=passes, pass_offset=0)
        raw_recon, raw_per_pass = trajectory_recon_loss(
            pred_traj,
            clean_traj,
            match_inner_boundary_states=self.cfg.match_inner_boundary_states,
            pass_null_losses=None,
            hinge_kappa=0.0,
            state_boundary_weight=self.cfg.state_boundary_weight,
        )
        kappa_eval = self._effective_hinge_kappa() if passes == self.cfg.n_passes else 0.0
        hinged_recon, hinged_per_pass = trajectory_recon_loss(
            pred_traj,
            clean_traj,
            match_inner_boundary_states=self.cfg.match_inner_boundary_states,
            pass_null_losses=self.pass_null_losses if passes == self.cfg.n_passes else None,
            hinge_kappa=kappa_eval,
            state_boundary_weight=self.cfg.state_boundary_weight,
        )
        raw_shares = compute_pass_loss_shares(raw_per_pass)
        hinged_shares = (
            compute_pass_loss_shares(hinged_per_pass)
            if sum(hinged_per_pass.values()) > 1e-12
            else raw_shares
        )

        final_kl = 0.0
        if clean_traj.readouts:
            f_tau = max(clean_traj.readouts.keys())
            final_kl = float(
                kl_divergence_logits(pred_traj.readouts[f_tau], clean_traj.readouts[f_tau]).item()
            )
        budget = active_component_budget(ci_out)
        metrics: dict[str, float] = {
            "recon_loss": float(raw_recon.item()),
            "hinged_recon_loss": float(hinged_recon.item()),
            "hinge_kappa": kappa_eval,
            "final_kl": final_kl,
            "faith_loss": float(faithfulness_loss(self.bpd_model.wrappers).item()),
            **budget,
        }
        for tau, val in raw_per_pass.items():
            metrics[f"pass_{tau}_loss"] = val
            metrics[f"pass_{tau}_hinged_loss"] = hinged_per_pass[tau]
            metrics[f"pass_{tau}_loss_share"] = hinged_shares[tau]
            metrics[f"pass_{tau}_raw_loss_share"] = raw_shares[tau]
        return metrics, ci_out, clean_traj


def train_ci_heads_on_frozen_components(
    bpd_model: BPDModel,
    base_cfg: BPDConfig,
    rollout_adapter: AdapterRolloutFn,
    batch_fn: Callable[[int], Any],
    eval_batch: Any,
    *,
    visible_dim: int,
    prompt_dim: int,
    pass_null_losses: Mapping[int, float] | None = None,
    pass_tolerances: Mapping[int, float] | None = None,
    hinge_kappa: float | None = None,
    device: torch.device | str = "cpu",
) -> tuple[dict[CIVariant, TiedCINetwork], dict[str, dict[str, float]], dict[str, float]]:
    """Train ``state_only``, ``state_plus_clock``, and ``visible_only`` CI variants on shared ``(V, U)``.

    Holding the physical decomposition ``(V, U)`` fixed isolates the pure causal
    addressing capacity of the three CI observation spaces so that **Clock Gap** and
    **Visible Gap** are unaffected by distinct local minima in ``(V, U)``.
    """
    dev = torch.device(device)
    bpd_model.to(dev)
    module_dims = {name: bpd_model.wrappers[name].d_in for name in bpd_model.module_order}
    variants: tuple[CIVariant, ...] = ("state_only", "state_plus_clock", "visible_only")
    null_map = pass_null_losses if pass_null_losses is not None else pass_tolerances
    raw_kappa = base_cfg.null_calibration.hinge_kappa if hinge_kappa is None else hinge_kappa
    kappa = (
        max(float(raw_kappa), 0.0)
        if base_cfg.null_calibration.use_fragility_hinge and base_cfg.n_passes > 1
        else 0.0
    )

    trained_nets: dict[CIVariant, TiedCINetwork] = {}
    variant_metrics: dict[str, dict[str, float]] = {}

    for var in variants:
        ci_cfg = replace(base_cfg.ci, variant=var)
        net = TiedCINetwork(
            module_dims=module_dims,
            c_per_module=bpd_model.c_per_module,
            cfg=ci_cfg,
            visible_dim=visible_dim if var == "visible_only" else None,
            prompt_dim=prompt_dim if var == "visible_only" else None,
        ).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=base_cfg.main_lr)
        rng = torch.Generator(device=dev).manual_seed(base_cfg.seed)

        for step in range(base_cfg.n_steps):
            batch = batch_fn(step)
            with torch.no_grad():
                with bpd_model.target_capture(
                    pass_offset=0,
                    update_means=(base_cfg.ablation.mode == "pass_mean" and step == 0),
                    detach_inputs=True,
                ):
                    ro, bs, vis, prm = rollout_adapter(batch, base_cfg.n_passes, 0, None)
                mod_in = bpd_model.collect_recorded_inputs(expected_passes=base_cfg.n_passes)
            clean_traj = RolloutTrajectory(
                readouts=ro,
                boundary_states=bs,
                module_inputs=mod_in,
                visible_states=vis,
                prompt_features=prm,
            )

            opt.zero_grad(set_to_none=True)
            ci_out = net.forward_trajectory(clean_traj, pass_offset=0)
            l_min, _ = bpd_minimality_loss(
                ci_out.upper_leaky,
                step=step,
                total_steps=base_cfg.n_steps,
                cfg=base_cfg.minimality,
            )
            sched = sample_stochastic_schedules(
                ci_out.lower_leaky,
                route_subset=True,
                share_noise_across_passes=True,
                generator=rng,
            )
            with bpd_model.masked_components(sched, pass_offset=0):
                p_ro, p_bs, p_vis, p_prm = rollout_adapter(batch, base_cfg.n_passes, 0, None)
            pred_traj = RolloutTrajectory(
                readouts=p_ro,
                boundary_states=p_bs,
                module_inputs={},
                visible_states=p_vis,
                prompt_features=p_prm,
            )
            l_rec, _ = trajectory_recon_loss(
                pred_traj,
                clean_traj,
                match_inner_boundary_states=base_cfg.match_inner_boundary_states,
                pass_null_losses=null_map,
                hinge_kappa=kappa,
                state_boundary_weight=base_cfg.state_boundary_weight,
            )
            loss = l_min + l_rec
            loss.backward()
            opt.step()

        # Deterministic evaluation (un-hinged KL for unbiased cross-variant comparison)
        with torch.no_grad():
            with bpd_model.target_capture(pass_offset=0, detach_inputs=True):
                e_ro, e_bs, e_vis, e_prm = rollout_adapter(eval_batch, base_cfg.n_passes, 0, None)
            e_in = bpd_model.collect_recorded_inputs(expected_passes=base_cfg.n_passes)
            e_clean = RolloutTrajectory(
                readouts=e_ro,
                boundary_states=e_bs,
                module_inputs=e_in,
                visible_states=e_vis,
                prompt_features=e_prm,
            )
            e_ci = net.forward_trajectory(e_clean, pass_offset=0)
            e_sched = deterministic_ci_schedules(e_ci.lower_leaky, delta_value=0.0)
            with bpd_model.masked_components(e_sched, pass_offset=0):
                m_ro, m_bs, m_vis, m_prm = rollout_adapter(eval_batch, base_cfg.n_passes, 0, None)
            e_pred = RolloutTrajectory(
                readouts=m_ro,
                boundary_states=m_bs,
                module_inputs={},
                visible_states=m_vis,
                prompt_features=m_prm,
            )
            e_rec, e_per_pass = trajectory_recon_loss(
                e_pred,
                e_clean,
                match_inner_boundary_states=base_cfg.match_inner_boundary_states,
                pass_null_losses=None,
                hinge_kappa=0.0,
                state_boundary_weight=base_cfg.state_boundary_weight,
            )
            e_shares = compute_pass_loss_shares(e_per_pass)
            e_bud = active_component_budget(e_ci)
            variant_metrics[var] = {
                "recon_loss": float(e_rec.item()),
                **e_bud,
                **{f"pass_{tau}_loss_share": sh for tau, sh in e_shares.items()},
            }
            trained_nets[var] = net

    gaps = compute_ci_variant_gaps(variant_metrics)
    return trained_nets, variant_metrics, gaps


class UnrolledVPDTrainer:
    """Gate B1 baseline: standard VPD with each pass treated as an independent weight site."""

    def __init__(
        self,
        target_model: nn.Module,
        c_per_pass_by_module: Mapping[str, int],
        cfg: BPDConfig,
        rollout_adapter: AdapterRolloutFn,
        *,
        final_pass_only_kl: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.target_model = target_model.to(self.device)
        for p in self.target_model.parameters():
            p.requires_grad_(False)
        self.target_model.eval()

        self.cfg = cfg
        self.rollout_adapter = rollout_adapter
        self.final_pass_only_kl = final_pass_only_kl
        self.module_order = sorted(c_per_pass_by_module.keys())
        self.c_per_pass = {k: int(c_per_pass_by_module[k]) for k in self.module_order}

        self.wrappers: dict[str, UnrolledComponentLinear] = {}
        module_dims: dict[str, int] = {}
        comp_params: list[nn.Parameter] = []
        for path in self.module_order:
            lin = self.target_model.get_submodule(path)
            if not isinstance(lin, nn.Linear):
                raise TypeError(f"Expected nn.Linear at {path!r}")
            w = UnrolledComponentLinear(
                lin, self.c_per_pass[path], cfg.n_passes, init_svd=cfg.init_svd
            ).to(self.device)
            parent_path, _, attr = path.rpartition(".")
            parent = (
                self.target_model.get_submodule(parent_path) if parent_path else self.target_model
            )
            setattr(parent, attr, w)
            self.wrappers[path] = w
            module_dims[path] = w.d_in
            comp_params.extend([w.V, w.U])

        self.ci_net = UnrolledCINetwork(
            module_dims=module_dims,
            c_per_module=self.c_per_pass,
            n_passes=cfg.n_passes,
            cfg=cfg.ci,
        ).to(self.device)
        self.rng = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.optimizer = torch.optim.AdamW(
            comp_params + list(self.ci_net.parameters()), lr=cfg.main_lr
        )
        # Standard VPD taxes every active site independently across passes (no switch penalty)
        self.unrolled_min_cfg = MinimalityConfig(
            pool_mode="sum_legacy",
            p_start=cfg.minimality.p_start,
            p_end=cfg.minimality.p_end,
            eps=cfg.minimality.eps,
            beta=cfg.minimality.beta,
            coeff_level=cfg.minimality.coeff_level / float(max(cfg.n_passes, 1)),
            coeff_switch=0.0,
        )

    def total_unrolled_components(self) -> int:
        return self.cfg.n_passes * sum(self.c_per_pass.values())

    def _capture_clean(self, batch: Any) -> RolloutTrajectory:
        for w in self.wrappers.values():
            w.mode = "target"
            w.schedule = None
            w.reset_pass_state()
            w.record_inputs = True
        with torch.no_grad():
            ro, bs, vis, prm = self.rollout_adapter(batch, self.cfg.n_passes, 0, None)
        mod_in = {
            name: torch.stack(self.wrappers[name].recorded_inputs, dim=1)
            for name in self.module_order
        }
        for w in self.wrappers.values():
            w.record_inputs = False
        return RolloutTrajectory(
            readouts=ro,
            boundary_states=bs,
            module_inputs=mod_in,
            visible_states=vis,
            prompt_features=prm,
        )

    def _run_masked(
        self, batch: Any, schedules: dict[str, PassMaskSchedule]
    ) -> RolloutTrajectory:
        for name, w in self.wrappers.items():
            w.mode = "component"
            w.schedule = schedules[name]
            w.reset_pass_state()
            w.record_inputs = False
        try:
            ro, bs, vis, prm = self.rollout_adapter(batch, self.cfg.n_passes, 0, None)
        finally:
            for w in self.wrappers.values():
                w.mode = "target"
                w.schedule = None
        return RolloutTrajectory(
            readouts=ro,
            boundary_states=bs,
            module_inputs={},
            visible_states=vis,
            prompt_features=prm,
        )

    def train_step(self, batch: Any, step: int) -> dict[str, float]:
        clean_traj = self._capture_clean(batch)
        self.optimizer.zero_grad(set_to_none=True)
        ci_out = self.ci_net(clean_traj.module_inputs)
        l_faith = unrolled_faithfulness_loss(self.wrappers)
        l_min, _ = bpd_minimality_loss(
            ci_out.upper_leaky,
            step=step,
            total_steps=self.cfg.n_steps,
            cfg=self.unrolled_min_cfg,
        )
        sched = sample_stochastic_schedules(
            ci_out.lower_leaky,
            route_subset=True,
            share_noise_across_passes=False,
            generator=self.rng,
        )
        pred_traj = self._run_masked(batch, sched)
        if self.final_pass_only_kl and clean_traj.readouts:
            f_tau = max(clean_traj.readouts.keys())
            l_rec = kl_divergence_logits(pred_traj.readouts[f_tau], clean_traj.readouts[f_tau])
        else:
            l_rec, _ = trajectory_recon_loss(
                pred_traj,
                clean_traj,
                match_inner_boundary_states=self.cfg.match_inner_boundary_states,
                state_boundary_weight=self.cfg.state_boundary_weight,
            )
        loss = self.cfg.coeff_faith * l_faith + l_min + l_rec
        loss.backward()
        self.optimizer.step()
        return {
            "total_loss": float(loss.detach().item()),
            "recon_loss": float(l_rec.detach().item()),
            "faith_loss": float(l_faith.detach().item()),
        }

    @torch.no_grad()
    def evaluate(self, batch: Any, *, delta_value: float = 0.0) -> dict[str, float]:
        clean_traj = self._capture_clean(batch)
        ci_out = self.ci_net(clean_traj.module_inputs)
        sched = deterministic_ci_schedules(ci_out.lower_leaky, delta_value=delta_value)
        pred_traj = self._run_masked(batch, sched)
        rec, per_pass = trajectory_recon_loss(
            pred_traj,
            clean_traj,
            match_inner_boundary_states=self.cfg.match_inner_boundary_states,
            state_boundary_weight=self.cfg.state_boundary_weight,
        )
        shares = compute_pass_loss_shares(per_pass)
        final_kl = 0.0
        if clean_traj.readouts:
            f_tau = max(clean_traj.readouts.keys())
            final_kl = float(
                kl_divergence_logits(pred_traj.readouts[f_tau], clean_traj.readouts[f_tau]).item()
            )
        bud = active_component_budget(ci_out)
        return {
            "recon_loss": float(rec.item()),
            "final_kl": final_kl,
            "faith_loss": float(unrolled_faithfulness_loss(self.wrappers).item()),
            "total_allocated_components": float(self.total_unrolled_components()),
            **bud,
            **{f"pass_{tau}_loss_share": sh for tau, sh in shares.items()},
        }
