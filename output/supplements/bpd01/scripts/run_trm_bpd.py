"""Gate B4 & Primary Target: 7M Attention-TRM BPD training, risk audits, and ``s = 64`` edit/ablation tests.

Executes the TRM-first experimental protocol (Amendment v2):
  1. Full-rank (``C = 512``) hybrid BPD training on multi-board mini-batches of clean-state
     transplant windows + full closed-loop rollouts (with ``call_idx``-pinned activation
     checkpointing), reporting **both** ``match_inner_boundary_states=False`` (default outer-step
     readout matching) and ``match_inner_boundary_states=True``, plus out-of-horizon deterministic
     CI reconstruction KL.
  2. Fragility-hinge calibration ``max(0, KL_mask_tau - kappa * KL_null_tau)`` with logged
     per-pass loss shares.
  3. Adversary stability audit across PGD steps ``K_adv`` (``0..160``), comparing time-shared
     ``shared_batch_and_pass`` against pass-resonant ``per_pass`` PGD, plus warm-started
     multi-seed switch-penalty sweep across ``coeff_switch``.
  4. Separate BPD decompositions of ``gaoxin492`` and ``yagizdevre``, duty-cycle-ranked active
     component inspection, strict ``tgt_net_sustaining > 0`` functional matching alongside the
     full unmatched-pair similarity baseline (raw + module-centered + permutation rank p-value),
     wired 2-step interventional path mediation, and **two 39-draw hypothesis-testing arms**
     on ``WB-Stall`` (``n = 183``) at ``s = 64`` with exact per-board McNemar discordance counts:
       - **Arm A** (intra-``gaoxin492`` sustaining necessity): ablating top-k sustaining components
         vs ``K = 39`` norm-matched random active-component ablations and ``K = 39`` rotation draws.
       - **Arm B** (intra-``yagizdevre`` functional stabilisation): scaling functionally matched
         sustaining components vs ``K = 39`` norm-matched random active-component edits and
         ``K = 39`` rotation draws, tracking both ``final_solve_rate`` and ``unsolved_tail_cycle_rate``.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechinterp.bpd.adapters.trm import (
    SudokuHorizonMetrics,
    TRMBatch,
    TRMBPDAdapter,
    build_mini_trm,
    default_trm_c_per_module,
    encode_sudoku_pairs,
    evaluate_sudoku_horizon,
    load_pretrained_trm,
    load_wb_stall_and_train_sudoku,
)
from mechinterp.bpd.adversary import evaluate_pgd_stability_curve
from mechinterp.bpd.analysis import (
    FunctionalComponentMatch,
    IntraCheckpointEditReceipt,
    apply_intra_checkpoint_sustaining_edit,
    apply_random_component_control_edit,
    apply_rotation_null_control_edit,
    classify_transient_sustaining,
    compose_interventional_path,
    compute_mcnemar_discordance,
    compute_transition_operator,
    compute_unmatched_similarity_baseline,
    extract_component_functional_profiles,
    extract_schedule,
    match_components_by_function,
    summarize_control_draws,
)
from mechinterp.bpd.components import BPDModel, RolloutTrajectory
from mechinterp.bpd.config import (
    AblationConfig,
    AdversaryConfig,
    BPDConfig,
    CIConfig,
    MinimalityConfig,
    NullCalibrationConfig,
    resolve_registration_commit,
)
from mechinterp.bpd.losses import trajectory_recon_loss
from mechinterp.bpd.nulls import evaluate_component_null_report
from mechinterp.bpd.trainer import BPDTrainer
from mechinterp.loopscope02.scripts.trm_sudoku import EXIT_POWER_CAP, PureSudokuTRM, power_limit_watts


def _horizon_to_dict(hm: SudokuHorizonMetrics) -> dict[str, Any]:
    return {
        "board_exact_by_step": {str(k): v for k, v in hm.board_exact_by_step.items()},
        "cell_acc_by_step": {str(k): v for k, v in hm.cell_acc_by_step.items()},
        "final_solve_rate": hm.final_solve_rate,
        "ever_solved_rate": hm.ever_solved_rate,
        "period2_oscillation_rate": hm.period2_oscillation_rate,
        "unsolved_tail_cycle_rate": hm.unsolved_tail_cycle_rate,
        "step_flip_rate_tail": hm.step_flip_rate_tail,
    }


def _receipt_to_dict(rc: IntraCheckpointEditReceipt) -> dict[str, Any]:
    return {
        "edited_modules": rc.edited_modules,
        "total_components_edited": rc.total_components_edited,
        "frobenius_norm_by_module": rc.frobenius_norm_by_module,
        "total_frobenius_norm": rc.total_frobenius_norm,
    }


def _release_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _move_ci_and_traj(
    ci_out: Any,
    traj: RolloutTrajectory,
    target_device: torch.device | str,
) -> None:
    dev = torch.device(target_device)
    for d_attr in ("lower_leaky", "upper_leaky", "pre_sigmoid"):
        mapping = getattr(ci_out, d_attr, None)
        if isinstance(mapping, dict):
            for k in list(mapping.keys()):
                mapping[k] = mapping[k].to(dev)
    traj.readouts = {k: v.to(dev) for k, v in traj.readouts.items()}
    traj.boundary_states = {k: v.to(dev) for k, v in traj.boundary_states.items()}
    traj.module_inputs = {k: v.to(dev) for k, v in traj.module_inputs.items()}
    if isinstance(traj.visible_states, dict):
        traj.visible_states = {k: v.to(dev) for k, v in traj.visible_states.items()}
    elif isinstance(traj.visible_states, torch.Tensor):
        traj.visible_states = traj.visible_states.to(dev)
    if traj.prompt_features is not None:
        traj.prompt_features = traj.prompt_features.to(dev)


def _log_stage(t0: float, msg: str) -> None:
    mem_str = ""
    if torch.cuda.is_available():
        alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        res_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        mem_str = f" [cuda alloc={alloc_mb:.0f}MB reserved={res_mb:.0f}MB]"
    print(f"[B4 +{time.perf_counter() - t0:6.1f}s] {msg}{mem_str}", file=sys.stderr, flush=True)


def _median_draw_index(values: list[float]) -> int:
    if not values:
        return 0
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    return order[len(order) // 2]


def _select_active_pools_and_inspected(
    bpd_model: BPDModel,
    ci_out: Any,
    *,
    n_inspect_per_mod: int,
    top_k_edit: int,
    duty_threshold: float = 0.05,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Rank components within each module by CI schedule duty cycle descending."""
    sched = extract_schedule(ci_out)
    duty_by_mod: dict[str, list[tuple[int, float]]] = {}
    for idx, key in enumerate(sched.keys):
        duty_by_mod.setdefault(key.module, []).append((key.component, float(sched.duty_cycle[idx].item())))

    active_pool: dict[str, list[int]] = {}
    inspected: dict[str, list[int]] = {}
    for mod in bpd_model.module_order:
        ranked = sorted(duty_by_mod.get(mod, []), key=lambda pair: pair[1], reverse=True)
        above = [c for c, d in ranked if d > duty_threshold]
        min_pool = min(len(ranked), max(top_k_edit * 4, 8))
        active_pool[mod] = above if len(above) >= min_pool else [c for c, _ in ranked[:min_pool]]
        inspected[mod] = [c for c, _ in ranked[: min(n_inspect_per_mod, len(ranked))]]
    return active_pool, inspected


def _fit_single_trm_decomposition(
    base_model: PureSudokuTRM,
    train_inputs: torch.Tensor,
    eval_inputs: torch.Tensor,
    cfg: BPDConfig,
    *,
    ood_outer_steps: int | None = None,
    device: str,
    tag: str = "fit",
    t0: float | None = None,
) -> tuple[
    BPDModel,
    BPDTrainer,
    TRMBPDAdapter,
    list[TRMBatch],
    TRMBatch,
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, torch.Tensor],
    Any,
    RolloutTrajectory,
    dict[str, int],
    dict[int, float],
    dict[int, float],
]:
    work_model = copy.deepcopy(base_model).to(device).eval()
    adapter = TRMBPDAdapter(work_model, readout_at="outer_step")
    total_invocations = cfg.n_passes
    ood_invocations = (
        adapter.invocations_for_outer_steps(ood_outer_steps)
        if ood_outer_steps is not None and ood_outer_steps > 0
        else total_invocations
    )
    bpd_mod = BPDModel(
        work_model,
        cfg.c_per_module,
        max_passes=max(total_invocations, ood_invocations),
        ablation=cfg.ablation,
        init_svd=True,
    )

    trainer = BPDTrainer(
        bpd_mod,
        cfg,
        adapter.rollout,
        clean_window_state_fn=adapter.sample_clean_window_state,
        valid_window_starts_fn=adapter.valid_window_starts,
        visible_dim=adapter.visible_dim,
        prompt_dim=adapter.prompt_dim,
        device=device,
    )

    bsz = max(min(int(cfg.batch_size), int(train_inputs.shape[0])), 1)
    cached_train_batches: list[TRMBatch] = []
    for start in range(0, int(train_inputs.shape[0]), bsz):
        chunk = train_inputs[start : start + bsz]
        if int(chunk.shape[0]) < bsz and cached_train_batches:
            break
        cached_train_batches.append(
            adapter.make_batch_with_clean_cache(chunk.to(device), total_invocations)
        )

    trainer.run_faithfulness_warmup()
    null_losses = trainer.calibrate_baselines(cached_train_batches[0])

    mode_counts = {"closed_loop": 0, "clean_window": 0}
    last_shares: dict[int, float] = {}
    log_every = max(cfg.n_steps // 6, 1)
    for step in range(cfg.n_steps):
        batch_step = cached_train_batches[step % len(cached_train_batches)]
        sm = trainer.train_step(batch_step, step)
        mode_counts[sm.mode] = mode_counts.get(sm.mode, 0) + 1
        last_shares = sm.pass_loss_shares
        if t0 is not None and ((step + 1) % log_every == 0 or step + 1 == cfg.n_steps):
            _log_stage(
                t0,
                f"{tag}: step {step + 1}/{cfg.n_steps} ({sm.mode}) "
                f"loss={sm.total_loss:.4f} faith={sm.faith_loss:.2e} active={sm.active_per_pass:.1f}",
            )

    ci_trained_state = {k: v.detach().cpu().clone() for k, v in trainer.ci_net.state_dict().items()}
    del trainer.optimizer
    _release_gpu_memory()

    # Evaluate on a disjoint 16-board eval slice for CI/trajectory diagnostics
    eval_slice_inputs = eval_inputs[: min(int(eval_inputs.shape[0]), 16)]
    cached_eval_slice = adapter.make_batch_with_clean_cache(
        eval_slice_inputs.to(device), total_invocations
    )
    eval_metrics, ci_out, clean_traj = trainer.evaluate(cached_eval_slice)
    ci_out.pre_sigmoid.clear()
    ci_out.upper_leaky.clear()
    _move_ci_and_traj(ci_out, clean_traj, "cpu")
    _release_gpu_memory()

    ood_metrics: dict[str, Any] | None = None
    if ood_invocations > total_invocations:
        ood_slice_inputs = eval_slice_inputs[: min(int(eval_slice_inputs.shape[0]), 8)]
        cached_ood_eval = adapter.make_batch_with_clean_cache(
            ood_slice_inputs.to(device), ood_invocations
        )
        ood_clean = trainer.run_clean_rollout(
            cached_ood_eval,
            n_passes=ood_invocations,
            pass_offset=0,
            update_means=(cfg.ablation.mode == "pass_mean"),
        )
        del ood_clean
        _release_gpu_memory()
        ood_raw, ood_ci, ood_traj = trainer.evaluate(cached_ood_eval, n_passes=ood_invocations)
        ood_metrics = {
            "ood_outer_steps": int(ood_outer_steps or 0),
            "ood_invocations": int(ood_invocations),
            "recon_loss": ood_raw["recon_loss"],
            "final_kl": ood_raw["final_kl"],
            "mean_active_per_pass": ood_raw["mean_active_per_pass"],
        }
        del cached_ood_eval, ood_ci, ood_traj
        _release_gpu_memory()

    # Stage fitted model/trainer on CPU so subsequent independent fits start from ~0 MB VRAM
    work_model.to("cpu")
    bpd_mod.to("cpu")
    trainer.ci_net.to("cpu")
    _release_gpu_memory()

    return (
        bpd_mod,
        trainer,
        adapter,
        cached_train_batches,
        cached_eval_slice,
        eval_metrics,
        ood_metrics,
        ci_trained_state,
        ci_out,
        clean_traj,
        mode_counts,
        null_losses,
        last_shares,
    )


def run_trm_bpd_pipeline(
    ref_model: torch.nn.Module,
    tgt_model: torch.nn.Module,
    train_inputs: torch.Tensor,
    eval_inputs: torch.Tensor,
    eval_targets: torch.Tensor,
    *,
    train_outer_steps: int = 3,
    eval_horizons: tuple[int, ...] = (16, 32, 64),
    c_per_mod: int = 512,
    n_steps: int = 240,
    batch_size: int = 32,
    pgd_steps_grid: tuple[int, ...] = (0, 2, 5, 20, 80, 160),
    switch_sweep: tuple[float, ...] = (0.0, 2e-4, 2e-3, 2e-2),
    top_k_edit: int = 4,
    edit_scale_delta: float = 0.15,
    hinge_kappa: float = 0.10,
    n_control_draws: int | None = None,
    registration_commit: str | None = None,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, object]:
    assert isinstance(ref_model, PureSudokuTRM) and isinstance(tgt_model, PureSudokuTRM)
    t0 = time.perf_counter()
    clean_ref_base = copy.deepcopy(ref_model).to(device).eval()
    clean_tgt_base = copy.deepcopy(tgt_model).to(device).eval()

    is_fast_test = n_steps < 20
    eff_control_draws = (
        int(n_control_draws) if n_control_draws is not None else (3 if is_fast_test else 39)
    )
    n_switch_seeds = 1 if is_fast_test else 3
    n_rot_eval = 4 if is_fast_test else 19
    n_inspect_per_mod = min(4 if is_fast_test else 8, c_per_mod)
    ood_outer = train_outer_steps * 2

    probe_adapter = TRMBPDAdapter(clean_ref_base, readout_at="outer_step")
    total_invocations = probe_adapter.invocations_for_outer_steps(train_outer_steps)
    ood_invocations = probe_adapter.invocations_for_outer_steps(ood_outer)
    window_invocations = probe_adapter.calls_per_outer_step
    c_map = default_trm_c_per_module(clean_ref_base, c=c_per_mod)
    eff_bsz = min(int(train_inputs.shape[0]), max(int(batch_size), 1))

    base_cfg = BPDConfig(
        c_per_module=c_map,
        n_passes=total_invocations,
        n_steps=n_steps,
        batch_size=eff_bsz,
        seq_len=clean_ref_base.total_len,
        seed=seed,
        faithfulness_warmup_steps=15,
        main_lr=1e-3 if c_per_mod >= 128 else 2e-3,
        coeff_faith=1e4,
        coeff_det=0.5,
        coeff_stoch=0.5,
        coeff_adv=1.0,
        match_inner_boundary_states=False,
        rollout_mode="hybrid",
        window_steps=window_invocations,
        closed_loop_prob=0.5,
        ci=CIConfig(
            variant="state_only",
            d_model=min(128, max(32, clean_ref_base.hidden_size)),
            n_blocks=1,
            n_heads=4,
            mlp_hidden=min(256, max(64, 2 * clean_ref_base.hidden_size)),
            max_passes=max(ood_invocations + 8, total_invocations * 2, 16),
            max_seq_len=max(clean_ref_base.total_len, 16),
        ),
        ablation=AblationConfig(mode="pass_mean", sphere_rescale=True),
        minimality=MinimalityConfig(pool_mode="mean", coeff_level=2e-4, coeff_switch=2e-4),
        adversary=AdversaryConfig(
            scope="shared_batch_and_pass",
            lr=0.03,
            inner_steps=1 if is_fast_test else 2,
            eval_steps_grid=pgd_steps_grid,
        ),
        null_calibration=NullCalibrationConfig(
            use_fragility_hinge=True,
            hinge_kappa=hinge_kappa,
            n_rotations_train=2,
            n_rotations_eval=n_rot_eval,
            calibrate_every=max(n_steps // 2, 10),
        ),
    )

    # 1a. Primary decomposition of reference checkpoint (match_inner_boundary_states=False, default)
    _log_stage(t0, f"1a. Fit gaoxin492 (C={c_per_mod}, outer_steps={train_outer_steps}, match_inner=False)")
    (
        bpd_ref,
        ref_trainer,
        ref_adapter,
        cached_train_ref_batches,
        cached_eval_ref,
        ref_eval_metrics,
        ref_ood_metrics,
        ref_ci_state,
        ref_ci_out,
        ref_clean_traj,
        mode_counts,
        ref_null_losses,
        ref_last_shares,
    ) = _fit_single_trm_decomposition(
        clean_ref_base,
        train_inputs,
        eval_inputs,
        base_cfg,
        ood_outer_steps=ood_outer,
        device=device,
        tag="ref_outer",
        t0=t0,
    )
    sched_rep = extract_schedule(ref_ci_out)
    _release_gpu_memory()

    # 1b. Side-by-side trajectory-loss mode comparison: match_inner_boundary_states=True
    _log_stage(t0, "1b. Fit gaoxin492 (match_inner_boundary_states=True)")
    inner_cfg = replace(
        base_cfg,
        match_inner_boundary_states=True,
        window_steps=probe_adapter.calls_per_h_cycle,
        seed=seed + 50,
    )
    (
        _bpd_ref_inner,
        _tr_inner,
        _ad_inner,
        _ct_inner,
        _ce_inner,
        ref_inner_eval_metrics,
        ref_inner_ood_metrics,
        _ci_state_inner,
        _ci_inner,
        _traj_inner,
        inner_mode_counts,
        inner_null_losses,
        inner_last_shares,
    ) = _fit_single_trm_decomposition(
        clean_ref_base,
        train_inputs,
        eval_inputs,
        inner_cfg,
        ood_outer_steps=ood_outer,
        device=device,
        tag="ref_inner",
        t0=t0,
    )
    del _bpd_ref_inner, _tr_inner, _ad_inner, _ct_inner, _ce_inner, _ci_state_inner, _ci_inner, _traj_inner
    _release_gpu_memory()

    # 2. Separate independent decomposition of target checkpoint (yagizdevre)
    _log_stage(t0, f"2. Fit yagizdevre (C={c_per_mod}, outer_steps={train_outer_steps}, separate decomposition)")
    tgt_cfg = replace(base_cfg, seed=seed + 100)
    (
        bpd_tgt,
        _tgt_trainer,
        tgt_adapter,
        _cached_train_tgt_batches,
        cached_eval_tgt,
        tgt_eval_metrics,
        tgt_ood_metrics,
        _tgt_ci_state,
        tgt_ci_out,
        tgt_clean_traj,
        _tgt_modes,
        tgt_null_losses,
        tgt_last_shares,
    ) = _fit_single_trm_decomposition(
        clean_tgt_base,
        train_inputs,
        eval_inputs,
        tgt_cfg,
        ood_outer_steps=ood_outer,
        device=device,
        tag="tgt_outer",
        t0=t0,
    )
    del _tgt_trainer, _cached_train_tgt_batches, _tgt_ci_state
    _release_gpu_memory()

    # Stage gaoxin492 artifacts onto GPU for Risk Audits 1-2 + Stages 5a, 5b, 6a, 7 (tgt stays on CPU)
    bpd_ref.target_model.to(device)
    bpd_ref.to(device)
    ref_trainer.ci_net.to(device)
    _move_ci_and_traj(ref_ci_out, ref_clean_traj, device)

    # Risk Audit 1: PGD adversary step-stability curve (time-shared vs pass-resonant)
    _log_stage(t0, "3a. Risk Audit 1: PGD stability curve (shared_batch_and_pass)")

    def _eval_sched_loss(sched: Any) -> torch.Tensor:
        tr = ref_trainer.run_masked_rollout(cached_eval_ref, sched, n_passes=total_invocations)
        l_rec, _ = trajectory_recon_loss(
            tr,
            ref_clean_traj,
            match_inner_boundary_states=base_cfg.match_inner_boundary_states,
            pass_null_losses=ref_null_losses,
            hinge_kappa=base_cfg.null_calibration.hinge_kappa,
            state_boundary_weight=base_cfg.state_boundary_weight,
        )
        return l_rec

    pgd_shared = evaluate_pgd_stability_curve(
        ref_ci_out.lower_leaky,
        c_map,
        _eval_sched_loss,
        scope="shared_batch_and_pass",
        steps_grid=pgd_steps_grid,
        seed=seed,
    )
    _release_gpu_memory()
    _log_stage(t0, "3b. Risk Audit 1: PGD stability curve (per_pass)")
    pgd_per_pass = evaluate_pgd_stability_curve(
        ref_ci_out.lower_leaky,
        c_map,
        _eval_sched_loss,
        scope="per_pass",
        steps_grid=pgd_steps_grid,
        seed=seed,
    )
    # Park ref_ci_out & ref_clean_traj on CPU so Stage 4 (sw_trainer.train_step) starts from ~125 MB VRAM
    _move_ci_and_traj(ref_ci_out, ref_clean_traj, "cpu")
    ref_trainer.ci_net.to("cpu")
    del ref_trainer
    _release_gpu_memory()

    # Risk Audit 2: Warm-started multi-seed switch penalty sweep on frozen (V, U)
    _log_stage(t0, f"4. Risk Audit 2: Warm-started switch penalty sweep ({n_switch_seeds} seeds)")
    switch_sweep_results: list[dict[str, Any]] = []
    sw_steps = 2 if is_fast_test else max(n_steps // 6, 6)
    for sw_val in switch_sweep:
        seed_recons: list[float] = []
        seed_actives: list[float] = []
        seed_switches: list[float] = []
        for s_idx in range(n_switch_seeds):
            sw_cfg = replace(
                base_cfg,
                n_steps=sw_steps,
                main_lr=5e-4,
                coeff_det=0.5,
                coeff_stoch=0.5,
                coeff_adv=0.0,
                seed=seed + 700 + s_idx,
                minimality=replace(base_cfg.minimality, coeff_switch=sw_val),
                null_calibration=replace(base_cfg.null_calibration, calibrate_every=0),
            )
            sw_trainer = BPDTrainer(
                bpd_ref,
                sw_cfg,
                ref_adapter.rollout,
                clean_window_state_fn=ref_adapter.sample_clean_window_state,
                valid_window_starts_fn=ref_adapter.valid_window_starts,
                device=device,
            )
            sw_trainer.ci_net.load_state_dict(
                {k: v.to(device) for k, v in ref_ci_state.items()}
            )
            sw_trainer.pass_null_losses = dict(ref_null_losses)
            sw_trainer.pass_tolerances = dict(ref_null_losses)
            for p in bpd_ref.component_parameters():
                p.requires_grad_(False)
            try:
                for st in range(sw_cfg.n_steps):
                    b_st = cached_train_ref_batches[st % len(cached_train_ref_batches)]
                    sw_trainer.train_step(b_st, st)
                del sw_trainer.optimizer
                _release_gpu_memory()
                sw_eval, sw_ci, sw_traj = sw_trainer.evaluate(cached_eval_ref)
                sw_sched = extract_schedule(sw_ci)
                seed_recons.append(float(sw_eval["recon_loss"]))
                seed_actives.append(float(sw_eval["mean_active_per_pass"]))
                seed_switches.append(float(sw_sched.switch_rate.mean().item()))
            finally:
                for p in bpd_ref.component_parameters():
                    p.requires_grad_(True)
            del sw_trainer, sw_eval, sw_ci, sw_traj, sw_sched
            _release_gpu_memory()

        t_rec = torch.tensor(seed_recons, dtype=torch.float64)
        t_act = torch.tensor(seed_actives, dtype=torch.float64)
        t_sw = torch.tensor(seed_switches, dtype=torch.float64)
        switch_sweep_results.append(
            {
                "coeff_switch": sw_val,
                "n_seeds": n_switch_seeds,
                "recon_loss": float(t_rec.mean().item()),
                "recon_loss_std": float(t_rec.std(unbiased=False).item()) if n_switch_seeds > 1 else 0.0,
                "mean_active_per_pass": float(t_act.mean().item()),
                "mean_switch_rate": float(t_sw.mean().item()),
                "mean_switch_rate_std": float(t_sw.std(unbiased=False).item()) if n_switch_seeds > 1 else 0.0,
            }
        )

    del cached_train_ref_batches, ref_ci_state
    _release_gpu_memory()
    _move_ci_and_traj(ref_ci_out, ref_clean_traj, device)

    # Raw + Net-of-Rotation-Null Importance & Functional Profiles on BOTH separate decompositions
    def _ref_plain_rollout() -> RolloutTrajectory:
        ro, bs, vis, prm = ref_adapter.rollout(cached_eval_ref, total_invocations, 0, None)
        return RolloutTrajectory(
            readouts=ro,
            boundary_states=bs,
            module_inputs=ref_clean_traj.module_inputs,
            visible_states=vis,
            prompt_features=prm,
        )

    def _tgt_plain_rollout() -> RolloutTrajectory:
        ro, bs, vis, prm = tgt_adapter.rollout(cached_eval_tgt, total_invocations, 0, None)
        return RolloutTrajectory(
            readouts=ro,
            boundary_states=bs,
            module_inputs=tgt_clean_traj.module_inputs,
            visible_states=vis,
            prompt_features=prm,
        )

    active_pool_ref, inspected_comps_ref = _select_active_pools_and_inspected(
        bpd_ref, ref_ci_out, n_inspect_per_mod=n_inspect_per_mod, top_k_edit=top_k_edit
    )
    active_pool_tgt, inspected_comps_tgt = _select_active_pools_and_inspected(
        bpd_tgt, tgt_ci_out, n_inspect_per_mod=n_inspect_per_mod, top_k_edit=top_k_edit
    )

    _log_stage(t0, f"5a. Evaluate component null report on gaoxin492 (K={n_rot_eval} rotations)")
    null_rep = evaluate_component_null_report(
        bpd_ref,
        ref_clean_traj,
        _ref_plain_rollout,
        components_by_module=inspected_comps_ref,
        passes=[total_invocations - 1],
        include_all_pass=True,
        n_rotations=n_rot_eval,
        seed=seed,
    )
    _log_stage(t0, "5b. Classify transient/sustaining on gaoxin492")
    ref_classifications = classify_transient_sustaining(
        bpd_ref,
        ref_clean_traj,
        _ref_plain_rollout,
        components_by_module=inspected_comps_ref,
        min_effect=1e-4,
        sustain_ratio=1.25,
        n_rotations=min(n_rot_eval, 8),
        seed=seed,
    )
    _log_stage(t0, "6a. Extract functional profiles + transition operator on gaoxin492")
    ref_profiles = extract_component_functional_profiles(
        bpd_ref, ref_ci_out, ref_clean_traj, _ref_plain_rollout, ref_classifications
    )
    _release_gpu_memory()

    # 1-step transition operator & wired 2-step interventional path check on top active components
    _log_stage(t0, "7. Compute top-active transition operator + wired 2-step interventional path")
    top_trans_comps = {
        m: inspected_comps_ref[m][: min(2, len(inspected_comps_ref[m]))]
        for m in bpd_ref.module_order
    }
    trans_op = compute_transition_operator(
        bpd_ref, ref_clean_traj, _ref_plain_rollout, components_by_module=top_trans_comps
    )
    path_demo = None
    if total_invocations >= 3 and len(trans_op.keys) >= 2:
        best_prod = -1.0
        demo_chain = [trans_op.keys[0], trans_op.keys[1], trans_op.keys[0]]
        n_k = len(trans_op.keys)
        for i_s in range(n_k):
            for j_m in range(n_k):
                for k_d in range(n_k):
                    prod_val = float(
                        trans_op.matrix[i_s, j_m].item() * trans_op.matrix[j_m, k_d].item()
                    )
                    if prod_val > best_prod:
                        best_prod = prod_val
                        demo_chain = [trans_op.keys[i_s], trans_op.keys[j_m], trans_op.keys[k_d]]
        path_res = compose_interventional_path(
            bpd_ref, ref_clean_traj, _ref_plain_rollout, trans_op, demo_chain, start_pass=0
        )
        path_demo = {
            "chain": [{"module": k.module, "component": k.component} for k in demo_chain],
            "start_pass": 0,
            "direct_source_effect": path_res.direct_source_effect,
            "interventional_mediated_effect": path_res.interventional_mediated_effect,
            "first_order_chain_product": path_res.first_order_chain_product,
            "transport_discrepancy": path_res.transport_discrepancy,
        }
    bpd_ref.target_model.to("cpu")
    bpd_ref.to("cpu")
    _move_ci_and_traj(ref_ci_out, ref_clean_traj, "cpu")
    del ref_adapter, cached_eval_ref, ref_ci_out, trans_op
    _release_gpu_memory()

    # Stage yagizdevre artifacts onto GPU for Stages 5c & 6b (ref is now on CPU)
    bpd_tgt.target_model.to(device)
    bpd_tgt.to(device)
    _move_ci_and_traj(tgt_ci_out, tgt_clean_traj, device)

    _log_stage(t0, "5c. Classify transient/sustaining on yagizdevre")
    tgt_classifications = classify_transient_sustaining(
        bpd_tgt,
        tgt_clean_traj,
        _tgt_plain_rollout,
        components_by_module=inspected_comps_tgt,
        min_effect=1e-4,
        sustain_ratio=1.25,
        n_rotations=min(n_rot_eval, 8),
        seed=seed + 100,
    )
    _log_stage(t0, "6b. Extract functional profiles + transition operator on yagizdevre")
    tgt_profiles = extract_component_functional_profiles(
        bpd_tgt, tgt_ci_out, tgt_clean_traj, _tgt_plain_rollout, tgt_classifications
    )
    bpd_tgt.target_model.to("cpu")
    bpd_tgt.to("cpu")
    _move_ci_and_traj(tgt_ci_out, tgt_clean_traj, "cpu")
    del tgt_adapter, cached_eval_tgt, tgt_ci_out
    _release_gpu_memory()

    # Strict functional matching: requires BOTH ref and tgt components to have positive sustaining effect
    functional_matches = match_components_by_function(
        ref_profiles,
        tgt_profiles,
        require_ref_sustaining=True,
        require_tgt_sustaining=True,
    )
    selected_matches = functional_matches[: min(top_k_edit, len(functional_matches))]
    unmatched_sim_summary = compute_unmatched_similarity_baseline(
        ref_profiles,
        tgt_profiles,
        selected_matches,
        n_permutations=199,
        seed=seed,
    )

    # Clean WB-Stall baselines at eval_horizons (16, 32, 64)
    _log_stage(t0, "8a. Evaluate clean WB-Stall s=64 baselines (gaoxin492 & yagizdevre)")
    ref_baseline_horizon = evaluate_sudoku_horizon(
        clean_ref_base, eval_inputs, eval_targets, eval_outer_steps=eval_horizons
    )
    before_horizon = evaluate_sudoku_horizon(
        clean_tgt_base, eval_inputs, eval_targets, eval_outer_steps=eval_horizons
    )

    # Arm A: Intra-gaoxin492 sustaining causal necessity (ablation vs K=39 random active & K=39 rotation)
    _log_stage(
        t0,
        f"8b. Arm A: Intra-gaoxin492 sustaining ablation vs {eff_control_draws} random & {eff_control_draws} rotation draws",
    )
    ref_sustaining_cands = [
        p for p in ref_profiles if p.role == "sustaining" and p.net_all_pass_final > 0.0
    ]
    if not ref_sustaining_cands:
        ref_sustaining_cands = [p for p in ref_profiles if p.net_all_pass_final > 0.0]
    ref_sustaining_cands.sort(key=lambda p: p.net_all_pass_final, reverse=True)
    ref_self_matches = [
        FunctionalComponentMatch(
            module=p.module,
            ref_component=p.component,
            tgt_component=p.component,
            schedule_similarity=1.0,
            transition_similarity=1.0,
            readout_similarity=1.0,
            functional_similarity=1.0,
            ref_net_sustaining=p.net_all_pass_final,
            tgt_net_sustaining=p.net_all_pass_final,
            priority_score=p.net_all_pass_final,
            centered_functional_similarity=1.0,
        )
        for p in ref_sustaining_cands[: min(top_k_edit, len(ref_sustaining_cands))]
    ]

    ablated_ref = copy.deepcopy(clean_ref_base)
    ablation_ref_receipt = apply_intra_checkpoint_sustaining_edit(
        ablated_ref, bpd_ref, ref_self_matches, scale_delta=-1.0
    )
    ablation_ref_horizon = evaluate_sudoku_horizon(
        ablated_ref, eval_inputs, eval_targets, eval_outer_steps=eval_horizons
    )
    del ablated_ref

    arm_a_rand_horizons: list[SudokuHorizonMetrics] = []
    arm_a_rand_receipt0: IntraCheckpointEditReceipt | None = None
    for d_idx in range(eff_control_draws):
        m_rand_a = copy.deepcopy(clean_ref_base)
        rc_a = apply_random_component_control_edit(
            m_rand_a,
            bpd_ref,
            ablation_ref_receipt,
            exclude_matches=ref_self_matches,
            candidate_pool_by_module=active_pool_ref,
            sign=-1.0,
            seed=seed + 3000 + d_idx,
        )
        if d_idx == 0:
            arm_a_rand_receipt0 = rc_a
        hm_a = evaluate_sudoku_horizon(
            m_rand_a, eval_inputs, eval_targets, eval_outer_steps=eval_horizons
        )
        arm_a_rand_horizons.append(hm_a)
        del m_rand_a

    arm_a_rot_horizons: list[SudokuHorizonMetrics] = []
    arm_a_rot_receipt0: IntraCheckpointEditReceipt | None = None
    for d_idx in range(eff_control_draws):
        m_rot_a = copy.deepcopy(clean_ref_base)
        rc_rot_a = apply_rotation_null_control_edit(
            m_rot_a,
            bpd_ref,
            ref_self_matches,
            ablation_ref_receipt,
            clean_traj=ref_clean_traj,
            seed=seed + 5000 + d_idx,
        )
        if d_idx == 0:
            arm_a_rot_receipt0 = rc_rot_a
        hm_rot_a = evaluate_sudoku_horizon(
            m_rot_a, eval_inputs, eval_targets, eval_outer_steps=eval_horizons
        )
        arm_a_rot_horizons.append(hm_rot_a)
        del m_rot_a

    del bpd_ref, ref_clean_traj
    _release_gpu_memory()

    arm_a_rand_rates = [h.final_solve_rate for h in arm_a_rand_horizons]
    arm_a_rot_rates = [h.final_solve_rate for h in arm_a_rot_horizons]
    arm_a_rand_dist = summarize_control_draws(
        ablation_ref_horizon.final_solve_rate, arm_a_rand_rates, higher_is_better=False
    )
    arm_a_rot_dist = summarize_control_draws(
        ablation_ref_horizon.final_solve_rate, arm_a_rot_rates, higher_is_better=False
    )
    arm_a_med_idx = _median_draw_index(arm_a_rand_rates)
    arm_a_mcnemar_clean = compute_mcnemar_discordance(
        ablation_ref_horizon.solved_final_flags, ref_baseline_horizon.solved_final_flags
    )
    arm_a_mcnemar_cycle = compute_mcnemar_discordance(
        ablation_ref_horizon.tail_cycle_flags, ref_baseline_horizon.tail_cycle_flags
    )
    arm_a_mcnemar_med_rand = compute_mcnemar_discordance(
        ablation_ref_horizon.solved_final_flags,
        arm_a_rand_horizons[arm_a_med_idx].solved_final_flags,
    )
    arm_a_passed = bool(
        len(ref_self_matches) >= 1
        and ablation_ref_horizon.final_solve_rate < ref_baseline_horizon.final_solve_rate
        and arm_a_mcnemar_clean.exact_two_sided_p <= 0.05
        and arm_a_rand_dist.empirical_p_rank <= 0.05
        and arm_a_rot_dist.empirical_p_rank <= 0.05
    )

    # Arm B: Intra-yagizdevre functional sustaining edit vs K=39 random active & K=39 rotation draws
    _log_stage(
        t0,
        f"8c. Arm B: Intra-yagizdevre sustaining edit vs {eff_control_draws} random & {eff_control_draws} rotation draws",
    )
    edited_tgt = copy.deepcopy(clean_tgt_base)
    edit_receipt = apply_intra_checkpoint_sustaining_edit(
        edited_tgt, bpd_tgt, selected_matches, scale_delta=edit_scale_delta
    )
    after_horizon = evaluate_sudoku_horizon(
        edited_tgt, eval_inputs, eval_targets, eval_outer_steps=eval_horizons
    )
    del edited_tgt

    arm_b_rand_horizons: list[SudokuHorizonMetrics] = []
    random_receipt0: IntraCheckpointEditReceipt | None = None
    for d_idx in range(eff_control_draws):
        random_ctrl_tgt = copy.deepcopy(clean_tgt_base)
        rc_b = apply_random_component_control_edit(
            random_ctrl_tgt,
            bpd_tgt,
            edit_receipt,
            exclude_matches=selected_matches,
            candidate_pool_by_module=active_pool_tgt,
            sign=1.0,
            seed=seed + 2026 + d_idx,
        )
        if d_idx == 0:
            random_receipt0 = rc_b
        hm_b = evaluate_sudoku_horizon(
            random_ctrl_tgt, eval_inputs, eval_targets, eval_outer_steps=eval_horizons
        )
        arm_b_rand_horizons.append(hm_b)
        del random_ctrl_tgt

    arm_b_rot_horizons: list[SudokuHorizonMetrics] = []
    rot_receipt0: IntraCheckpointEditReceipt | None = None
    for d_idx in range(eff_control_draws):
        rot_ctrl_tgt = copy.deepcopy(clean_tgt_base)
        rc_rot_b = apply_rotation_null_control_edit(
            rot_ctrl_tgt,
            bpd_tgt,
            selected_matches,
            edit_receipt,
            clean_traj=tgt_clean_traj,
            seed=seed + 4048 + d_idx,
        )
        if d_idx == 0:
            rot_receipt0 = rc_rot_b
        hm_rot_b = evaluate_sudoku_horizon(
            rot_ctrl_tgt, eval_inputs, eval_targets, eval_outer_steps=eval_horizons
        )
        arm_b_rot_horizons.append(hm_rot_b)
        del rot_ctrl_tgt

    del bpd_tgt, tgt_clean_traj
    _release_gpu_memory()
    _log_stage(t0, "9. Done")

    assert random_receipt0 is not None and rot_receipt0 is not None
    assert arm_a_rand_receipt0 is not None and arm_a_rot_receipt0 is not None
    random_horizon0 = arm_b_rand_horizons[0]
    rot_horizon0 = arm_b_rot_horizons[0]

    arm_b_rand_rates = [h.final_solve_rate for h in arm_b_rand_horizons]
    arm_b_rot_rates = [h.final_solve_rate for h in arm_b_rot_horizons]
    arm_b_rand_dist = summarize_control_draws(
        after_horizon.final_solve_rate, arm_b_rand_rates, higher_is_better=True
    )
    arm_b_rot_dist = summarize_control_draws(
        after_horizon.final_solve_rate, arm_b_rot_rates, higher_is_better=True
    )
    arm_b_rand_cycle_dist = summarize_control_draws(
        after_horizon.unsolved_tail_cycle_rate,
        [h.unsolved_tail_cycle_rate for h in arm_b_rand_horizons],
        higher_is_better=False,
    )
    arm_b_med_idx = _median_draw_index(arm_b_rand_rates)
    arm_b_mcnemar_clean = compute_mcnemar_discordance(
        after_horizon.solved_final_flags, before_horizon.solved_final_flags
    )
    arm_b_mcnemar_cycle = compute_mcnemar_discordance(
        after_horizon.tail_cycle_flags, before_horizon.tail_cycle_flags
    )
    arm_b_mcnemar_med_rand = compute_mcnemar_discordance(
        after_horizon.solved_final_flags,
        arm_b_rand_horizons[arm_b_med_idx].solved_final_flags,
    )

    min_h = min(eval_horizons)
    s16_key = 16 if 16 in eval_horizons else min_h
    all_tgt_sustaining_positive = bool(
        len(selected_matches) >= 1 and all(m.tgt_net_sustaining > 0.0 for m in selected_matches)
    )
    arm_b_passed = bool(
        all_tgt_sustaining_positive
        and after_horizon.final_solve_rate > before_horizon.final_solve_rate
        and after_horizon.unsolved_tail_cycle_rate < before_horizon.unsolved_tail_cycle_rate
        and arm_b_mcnemar_clean.exact_two_sided_p <= 0.05
        and arm_b_rand_dist.empirical_p_rank <= 0.05
        and arm_b_rot_dist.empirical_p_rank <= 0.05
        and (
            after_horizon.board_exact_by_step[s16_key]
            >= before_horizon.board_exact_by_step[s16_key] - 0.02
        )
    )

    fidelity_ok = bool(
        float(ref_eval_metrics["final_kl"]) <= 0.15
        and float(tgt_eval_metrics["final_kl"]) <= 0.15
        and path_demo is not None
        and float(path_demo["direct_source_effect"]) > 0.0
    )
    gate_b4_passed = bool(fidelity_ok and arm_a_passed and arm_b_passed)

    return {
        "gate": "B4_v2_trm_separate_decompositions_necessity_and_stabilisation",
        "registration_commit": resolve_registration_commit(registration_commit),
        "c_per_module": c_per_mod,
        "num_train_boards": int(train_inputs.shape[0]),
        "train_batch_size": eff_bsz,
        "total_invocations_trained": total_invocations,
        "hinge_kappa": hinge_kappa,
        "n_control_draws": eff_control_draws,
        "rollout_mode_counts": mode_counts,
        "trajectory_state_matching_modes": {
            "outer_readout_only_default": {
                "match_inner_boundary_states": False,
                "eval_metrics": ref_eval_metrics,
                "ood_eval_metrics": ref_ood_metrics,
                "pass_null_losses": {str(k): v for k, v in ref_null_losses.items()},
                "last_train_pass_loss_shares": {str(k): v for k, v in ref_last_shares.items()},
            },
            "with_inner_boundary_states": {
                "match_inner_boundary_states": True,
                "rollout_mode_counts": inner_mode_counts,
                "eval_metrics": ref_inner_eval_metrics,
                "ood_eval_metrics": ref_inner_ood_metrics,
                "pass_null_losses": {str(k): v for k, v in inner_null_losses.items()},
                "last_train_pass_loss_shares": {str(k): v for k, v in inner_last_shares.items()},
            },
        },
        "target_decomposition_summary": {
            "eval_metrics": tgt_eval_metrics,
            "ood_eval_metrics": tgt_ood_metrics,
            "pass_null_losses": {str(k): v for k, v in tgt_null_losses.items()},
            "last_train_pass_loss_shares": {str(k): v for k, v in tgt_last_shares.items()},
        },
        "stationary_fraction": float(sched_rep.is_stationary.float().mean().item()),
        "active_pool_sizes_ref": {m: len(v) for m, v in active_pool_ref.items()},
        "active_pool_sizes_tgt": {m: len(v) for m, v in active_pool_tgt.items()},
        "pgd_stability": {
            "shared_batch_and_pass": pgd_shared.loss_by_steps,
            "shared_amplification": pgd_shared.amplification_ratio,
            "per_pass_resonant": pgd_per_pass.loss_by_steps,
            "per_pass_amplification": pgd_per_pass.amplification_ratio,
        },
        "switch_penalty_sweep": switch_sweep_results,
        "raw_and_net_null_sample": null_rep.to_dicts()[:12],
        "role_counts_ref": {
            role: sum(1 for e in ref_classifications if e.role == role)
            for role in ("sustaining", "transient", "bottleneck", "inert")
        },
        "role_counts_tgt": {
            role: sum(1 for e in tgt_classifications if e.role == role)
            for role in ("sustaining", "transient", "bottleneck", "inert")
        },
        "interventional_path_demo": path_demo,
        "functional_matches": [
            {
                "module": m.module,
                "ref_component": m.ref_component,
                "tgt_component": m.tgt_component,
                "schedule_similarity": m.schedule_similarity,
                "transition_similarity": m.transition_similarity,
                "readout_similarity": m.readout_similarity,
                "functional_similarity": m.functional_similarity,
                "centered_functional_similarity": m.centered_functional_similarity,
                "ref_net_sustaining": m.ref_net_sustaining,
                "tgt_net_sustaining": m.tgt_net_sustaining,
                "priority_score": m.priority_score,
            }
            for m in selected_matches
        ],
        "unmatched_similarity_baseline": unmatched_sim_summary.to_dict(),
        "arm_a_gaoxin492_sustaining_necessity": {
            "scale_delta": -1.0,
            "selected_components": [
                {
                    "module": m.module,
                    "component": m.ref_component,
                    "ref_net_sustaining": m.ref_net_sustaining,
                }
                for m in ref_self_matches
            ],
            "clean_gaoxin492": _horizon_to_dict(ref_baseline_horizon),
            "targeted_sustaining_ablation": {
                "receipt": _receipt_to_dict(ablation_ref_receipt),
                "metrics": _horizon_to_dict(ablation_ref_horizon),
            },
            "random_active_ablation_distribution": {
                "draw0_receipt": _receipt_to_dict(arm_a_rand_receipt0),
                "final_solve_distribution": arm_a_rand_dist.to_dict(),
                "median_draw_metrics": _horizon_to_dict(arm_a_rand_horizons[arm_a_med_idx]),
            },
            "rotation_null_ablation_distribution": {
                "draw0_receipt": _receipt_to_dict(arm_a_rot_receipt0),
                "final_solve_distribution": arm_a_rot_dist.to_dict(),
            },
            "mcnemar_final_solve_vs_clean": arm_a_mcnemar_clean.to_dict(),
            "mcnemar_tail_cycle_vs_clean": arm_a_mcnemar_cycle.to_dict(),
            "mcnemar_final_solve_vs_median_random": arm_a_mcnemar_med_rand.to_dict(),
            "arm_a_passed": arm_a_passed,
        },
        "sustaining_edit_test": {
            "edit_scale_delta": edit_scale_delta,
            "num_eval_boards": int(eval_inputs.shape[0]),
            "all_tgt_sustaining_positive": all_tgt_sustaining_positive,
            "reference_gaoxin492_clean": _horizon_to_dict(ref_baseline_horizon),
            "target_yagizdevre_clean": _horizon_to_dict(before_horizon),
            "intra_checkpoint_sustaining_edit": {
                "receipt": _receipt_to_dict(edit_receipt),
                "metrics": _horizon_to_dict(after_horizon),
            },
            "matched_random_component_control": {
                "receipt": _receipt_to_dict(random_receipt0),
                "metrics": _horizon_to_dict(random_horizon0),
                "final_solve_distribution": arm_b_rand_dist.to_dict(),
                "unsolved_tail_cycle_distribution": arm_b_rand_cycle_dist.to_dict(),
                "median_draw_metrics": _horizon_to_dict(arm_b_rand_horizons[arm_b_med_idx]),
            },
            "rotation_null_control": {
                "receipt": _receipt_to_dict(rot_receipt0),
                "metrics": _horizon_to_dict(rot_horizon0),
                "final_solve_distribution": arm_b_rot_dist.to_dict(),
            },
            "mcnemar_final_solve_vs_clean": arm_b_mcnemar_clean.to_dict(),
            "mcnemar_tail_cycle_vs_clean": arm_b_mcnemar_cycle.to_dict(),
            "mcnemar_final_solve_vs_median_random": arm_b_mcnemar_med_rand.to_dict(),
            "delta_final_solve_at_max_horizon": (
                after_horizon.final_solve_rate - before_horizon.final_solve_rate
            ),
            "delta_ever_solved_at_max_horizon": (
                after_horizon.ever_solved_rate - before_horizon.ever_solved_rate
            ),
            "delta_unsolved_tail_cycle_at_max_horizon": (
                after_horizon.unsolved_tail_cycle_rate - before_horizon.unsolved_tail_cycle_rate
            ),
            "arm_b_passed": arm_b_passed,
        },
        "checks": {
            "fidelity_ok": fidelity_ok,
            "all_tgt_sustaining_positive": all_tgt_sustaining_positive,
            "arm_a_gaoxin492_necessity_passed": arm_a_passed,
            "arm_b_yagizdevre_stabilisation_passed": arm_b_passed,
        },
        "gate_b4_passed": gate_b4_passed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref-ckpt", type=str, default="gaoxin492")
    ap.add_argument("--tgt-ckpt", type=str, default="yagizdevre")
    ap.add_argument("--sudoku-json", type=Path, default=None, help="Optional JSON list of {question, answer}")
    ap.add_argument("--bundle-json", type=Path, default=None, help="Optional loopscope02 benchmark_bundle.json")
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--train-outer-steps", type=int, default=3)
    ap.add_argument("--c", type=int, default=512)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--top-k-edit", type=int, default=4)
    ap.add_argument("--edit-scale", type=float, default=0.15)
    ap.add_argument("--hinge-kappa", type=float, default=0.10)
    ap.add_argument("--control-draws", type=int, default=39)
    ap.add_argument("--registration-commit", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-watts", type=float, default=300.0)
    ap.add_argument(
        "--max-vram-fraction",
        type=float,
        default=0.65,
        help="Hard per-process CUDA VRAM ceiling (prevents Windows WDDM shared-RAM swap)",
    )
    ap.add_argument("--smoke", action="store_true", help="Fast deterministic CPU smoke run on mini-TRM")
    ap.add_argument(
        "--fast-rehearsal",
        action="store_true",
        help="Fast full-rank 9-stage CUDA rehearsal on real checkpoints (1 step, 2 PGD steps, 1 draw)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.device.startswith("cuda") and not args.smoke:
        if torch.cuda.is_available() and 0.05 <= args.max_vram_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(float(args.max_vram_fraction))
        watts = power_limit_watts()
        if watts is not None and watts > args.max_watts:
            print(
                f"[abort] GPU enforced power limit {watts}W exceeds --max-watts={args.max_watts}W",
                file=sys.stderr,
            )
            return EXIT_POWER_CAP

    if args.smoke:
        ref = build_mini_trm(seed=10)
        tgt = build_mini_trm(seed=20)
        gen = torch.Generator().manual_seed(30)
        train_in = torch.randint(1, 11, (8, ref.seq_len), generator=gen)
        eval_in = torch.randint(1, 11, (8, ref.seq_len), generator=gen)
        eval_tgt = torch.randint(0, 9, (8, ref.seq_len), generator=gen)
        res = run_trm_bpd_pipeline(
            ref,
            tgt,
            train_in,
            eval_in,
            eval_tgt,
            train_outer_steps=2,
            eval_horizons=(2, 4, 8),
            c_per_mod=6,
            n_steps=8,
            batch_size=8,
            pgd_steps_grid=(0, 2, 5),
            switch_sweep=(0.0, 2e-4),
            top_k_edit=2,
            edit_scale_delta=args.edit_scale,
            hinge_kappa=args.hinge_kappa,
            n_control_draws=3,
            registration_commit=args.registration_commit,
            device="cpu",
            seed=0,
        )
    else:
        dt = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
        ref = load_pretrained_trm(args.ref_ckpt, dtype=dt, verify_sha256=True)
        tgt = load_pretrained_trm(args.tgt_ckpt, dtype=dt, verify_sha256=True)
        if args.sudoku_json is not None:
            rows = json.loads(args.sudoku_json.read_text(encoding="utf-8"))
            qs = [r["question"] for r in rows]
            sols = [r["answer"] for r in rows]
            inp, ans = encode_sudoku_pairs(qs, sols, device="cpu")
            n_tr = min(args.n_train, max(inp.shape[0] // 2, 1))
            train_in, eval_in, eval_ans = inp[:n_tr], inp[n_tr:], ans[n_tr:]
            cohort_tag = str(args.sudoku_json)
        else:
            wb = load_wb_stall_and_train_sudoku(
                args.bundle_json, n_train=args.n_train, device="cpu"
            )
            train_in = wb["train_inputs"]
            eval_in = wb["eval_inputs"]
            eval_ans = wb["eval_targets"]
            cohort_tag = f"WB-Stall (n={wb['n_wb_stall']}) from {wb['bundle_path']}"

        res = run_trm_bpd_pipeline(
            ref,
            tgt,
            train_in,
            eval_in,
            eval_ans,
            train_outer_steps=args.train_outer_steps,
            eval_horizons=(16, 32, 64),
            c_per_mod=args.c,
            n_steps=1 if args.fast_rehearsal else args.steps,
            batch_size=args.batch_size,
            pgd_steps_grid=(0, 2) if args.fast_rehearsal else (0, 2, 5, 20, 80, 160),
            switch_sweep=(0.0,) if args.fast_rehearsal else (0.0, 2e-4, 2e-3, 2e-2),
            top_k_edit=args.top_k_edit,
            edit_scale_delta=args.edit_scale,
            hinge_kappa=args.hinge_kappa,
            n_control_draws=1 if args.fast_rehearsal else args.control_draws,
            registration_commit=args.registration_commit,
            device=args.device,
        )
        res["eval_cohort"] = cohort_tag

    text = json.dumps(res, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
