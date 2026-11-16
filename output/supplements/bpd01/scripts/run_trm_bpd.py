"""Gate B4 & Primary Target: 7M Attention-TRM BPD training, risk audits, and ``s = 64`` edit test.

Executes the TRM-first experimental protocol:
  1. Hybrid training on short clean-state transplant windows + full closed-loop rollouts,
     reporting **both** ``match_inner_boundary_states=False`` (default outer-step readout
     matching) and ``match_inner_boundary_states=True``.
  2. Fragility-hinge calibration ``max(0, KL_mask_tau - kappa * KL_null_tau)`` with
     logged per-pass loss shares.
  3. Adversary stability audit across PGD steps ``K_adv`` (``0..160``), comparing
     time-shared ``shared_batch_and_pass`` against pass-resonant ``per_pass`` PGD,
     plus switch-penalty sweep across ``coeff_switch``.
  4. Separate BPD decompositions of ``gaoxin492`` and ``yagizdevre``, functional
     matching (schedule, transition profile, readout effect), and **intra-checkpoint**
     sustaining edits strictly inside ``yagizdevre``'s own decomposition alongside
     matched-count/matched-norm random component and Haar rotation-null controls,
     reporting **final solve** and **ever-solved** on ``WB-Stall`` at ``s = 64``.
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
    IntraCheckpointEditReceipt,
    apply_intra_checkpoint_sustaining_edit,
    apply_random_component_control_edit,
    apply_rotation_null_control_edit,
    classify_transient_sustaining,
    compose_interventional_path,
    compute_transition_operator,
    extract_component_functional_profiles,
    extract_schedule,
    match_components_by_function,
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


def _log_stage(t0: float, msg: str) -> None:
    mem_str = ""
    if torch.cuda.is_available():
        alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        res_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        mem_str = f" [cuda alloc={alloc_mb:.0f}MB reserved={res_mb:.0f}MB]"
    print(f"[B4 +{time.perf_counter() - t0:6.1f}s] {msg}{mem_str}", file=sys.stderr, flush=True)


def _fit_single_trm_decomposition(
    base_model: PureSudokuTRM,
    train_inputs: torch.Tensor,
    eval_inputs: torch.Tensor,
    cfg: BPDConfig,
    *,
    device: str,
    tag: str = "fit",
    t0: float | None = None,
) -> tuple[BPDModel, BPDTrainer, TRMBPDAdapter, Any, Any, dict[str, Any], Any, RolloutTrajectory, dict[str, int], dict[int, float], dict[int, float]]:
    work_model = copy.deepcopy(base_model).to(device).eval()
    adapter = TRMBPDAdapter(work_model, readout_at="outer_step")
    total_invocations = cfg.n_passes
    bpd_mod = BPDModel(
        work_model,
        cfg.c_per_module,
        max_passes=total_invocations,
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
    cached_train = adapter.make_batch_with_clean_cache(train_inputs.to(device), total_invocations)

    trainer.run_faithfulness_warmup()
    null_losses = trainer.calibrate_baselines(cached_train)

    mode_counts = {"closed_loop": 0, "clean_window": 0}
    last_shares: dict[int, float] = {}
    log_every = max(cfg.n_steps // 4, 1)
    for step in range(cfg.n_steps):
        sm = trainer.train_step(cached_train, step)
        mode_counts[sm.mode] = mode_counts.get(sm.mode, 0) + 1
        last_shares = sm.pass_loss_shares
        if t0 is not None and ((step + 1) % log_every == 0 or step + 1 == cfg.n_steps):
            _log_stage(
                t0,
                f"{tag}: step {step + 1}/{cfg.n_steps} ({sm.mode}) loss={sm.total_loss:.4f}",
            )

    # Evaluate on a manageable eval slice for CI/trajectory diagnostics
    eval_slice_inputs = eval_inputs[: min(int(eval_inputs.shape[0]), max(cfg.batch_size, 16))]
    cached_eval_slice = adapter.make_batch_with_clean_cache(
        eval_slice_inputs.to(device), total_invocations
    )
    eval_metrics, ci_out, clean_traj = trainer.evaluate(cached_eval_slice)
    return (
        bpd_mod,
        trainer,
        adapter,
        cached_train,
        cached_eval_slice,
        eval_metrics,
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
    train_outer_steps: int = 2,
    eval_horizons: tuple[int, ...] = (16, 32, 64),
    c_per_mod: int = 16,
    n_steps: int = 60,
    pgd_steps_grid: tuple[int, ...] = (0, 2, 5, 20, 80, 160),
    switch_sweep: tuple[float, ...] = (0.0, 2e-4, 2e-3),
    top_k_edit: int = 4,
    edit_scale_delta: float = 0.15,
    hinge_kappa: float = 0.10,
    registration_commit: str | None = None,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, object]:
    assert isinstance(ref_model, PureSudokuTRM) and isinstance(tgt_model, PureSudokuTRM)
    t0 = time.perf_counter()
    clean_ref_base = copy.deepcopy(ref_model).to(device).eval()
    clean_tgt_base = copy.deepcopy(tgt_model).to(device).eval()

    probe_adapter = TRMBPDAdapter(clean_ref_base, readout_at="outer_step")
    total_invocations = probe_adapter.invocations_for_outer_steps(train_outer_steps)
    window_invocations = probe_adapter.calls_per_outer_step
    c_map = default_trm_c_per_module(clean_ref_base, c=c_per_mod)

    base_cfg = BPDConfig(
        c_per_module=c_map,
        n_passes=total_invocations,
        n_steps=n_steps,
        batch_size=int(train_inputs.shape[0]),
        seq_len=clean_ref_base.total_len,
        seed=seed,
        faithfulness_warmup_steps=15,
        main_lr=2e-3,
        coeff_faith=1e3,
        coeff_stoch=0.5,
        coeff_adv=0.5,
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
            max_passes=max(total_invocations * 2, 16),
            max_seq_len=max(clean_ref_base.total_len, 16),
        ),
        ablation=AblationConfig(mode="pass_mean", sphere_rescale=True),
        minimality=MinimalityConfig(pool_mode="mean", coeff_level=2e-4, coeff_switch=2e-4),
        adversary=AdversaryConfig(
            scope="shared_batch_and_pass",
            inner_steps=1,
            eval_steps_grid=pgd_steps_grid,
        ),
        null_calibration=NullCalibrationConfig(
            use_fragility_hinge=True,
            hinge_kappa=hinge_kappa,
            n_rotations_train=2,
            n_rotations_eval=4,
            calibrate_every=max(n_steps // 2, 10),
        ),
    )

    # 1a. Primary decomposition of reference checkpoint (match_inner_boundary_states=False, default)
    _log_stage(t0, "1a. Fit gaoxin492 (match_inner_boundary_states=False)")
    (
        bpd_ref,
        ref_trainer,
        ref_adapter,
        cached_train_ref,
        cached_eval_ref,
        ref_eval_metrics,
        ref_ci_out,
        ref_clean_traj,
        mode_counts,
        ref_null_losses,
        ref_last_shares,
    ) = _fit_single_trm_decomposition(
        clean_ref_base, train_inputs, eval_inputs, base_cfg, device=device, tag="ref_outer", t0=t0
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
        _ci_inner,
        _traj_inner,
        inner_mode_counts,
        inner_null_losses,
        inner_last_shares,
    ) = _fit_single_trm_decomposition(
        clean_ref_base, train_inputs, eval_inputs, inner_cfg, device=device, tag="ref_inner", t0=t0
    )
    del _bpd_ref_inner, _tr_inner, _ad_inner, _ct_inner, _ce_inner, _ci_inner, _traj_inner
    _release_gpu_memory()

    # 2. Separate independent decomposition of target checkpoint (yagizdevre)
    _log_stage(t0, "2. Fit yagizdevre (separate decomposition)")
    tgt_cfg = replace(base_cfg, seed=seed + 100)
    (
        bpd_tgt,
        _tgt_trainer,
        tgt_adapter,
        _cached_train_tgt,
        cached_eval_tgt,
        tgt_eval_metrics,
        tgt_ci_out,
        tgt_clean_traj,
        _tgt_modes,
        tgt_null_losses,
        tgt_last_shares,
    ) = _fit_single_trm_decomposition(
        clean_tgt_base, train_inputs, eval_inputs, tgt_cfg, device=device, tag="tgt_outer", t0=t0
    )
    del _tgt_trainer, _cached_train_tgt
    _release_gpu_memory()

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
    _release_gpu_memory()

    # Risk Audit 2: Switch penalty sweep on frozen (V, U)
    _log_stage(t0, "4. Risk Audit 2: Switch penalty sweep")
    switch_sweep_results: list[dict[str, float]] = []
    for sw_val in switch_sweep:
        sw_cfg = replace(
            base_cfg,
            n_steps=max(n_steps // 3, 6),
            coeff_adv=0.0,
            minimality=replace(base_cfg.minimality, coeff_switch=sw_val),
        )
        sw_trainer = BPDTrainer(
            bpd_ref,
            sw_cfg,
            ref_adapter.rollout,
            clean_window_state_fn=ref_adapter.sample_clean_window_state,
            valid_window_starts_fn=ref_adapter.valid_window_starts,
            device=device,
        )
        for p in bpd_ref.component_parameters():
            p.requires_grad_(False)
        try:
            for st in range(sw_cfg.n_steps):
                sw_trainer.train_step(cached_train_ref, st)
            sw_eval, sw_ci, _ = sw_trainer.evaluate(cached_eval_ref)
            sw_sched = extract_schedule(sw_ci)
            switch_sweep_results.append(
                {
                    "coeff_switch": sw_val,
                    "recon_loss": sw_eval["recon_loss"],
                    "mean_active_per_pass": sw_eval["mean_active_per_pass"],
                    "mean_switch_rate": float(sw_sched.switch_rate.mean().item()),
                }
            )
        finally:
            for p in bpd_ref.component_parameters():
                p.requires_grad_(True)
        del sw_trainer, sw_eval, sw_ci, sw_sched
        _release_gpu_memory()

    del ref_trainer, cached_train_ref
    _release_gpu_memory()

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

    inspected_comps = {m: list(range(min(6, c_per_mod))) for m in bpd_ref.module_order}
    _log_stage(t0, "5a. Evaluate component null report on gaoxin492")
    null_rep = evaluate_component_null_report(
        bpd_ref,
        ref_clean_traj,
        _ref_plain_rollout,
        components_by_module=inspected_comps,
        passes=[total_invocations - 1],
        include_all_pass=True,
        n_rotations=base_cfg.null_calibration.n_rotations_eval,
        seed=seed,
    )
    _log_stage(t0, "5b. Classify transient/sustaining on gaoxin492")
    ref_classifications = classify_transient_sustaining(
        bpd_ref,
        ref_clean_traj,
        _ref_plain_rollout,
        components_by_module=inspected_comps,
        min_effect=1e-4,
        sustain_ratio=1.25,
        n_rotations=base_cfg.null_calibration.n_rotations_eval,
        seed=seed,
    )
    _log_stage(t0, "5c. Classify transient/sustaining on yagizdevre")
    tgt_classifications = classify_transient_sustaining(
        bpd_tgt,
        tgt_clean_traj,
        _tgt_plain_rollout,
        components_by_module=inspected_comps,
        min_effect=1e-4,
        sustain_ratio=1.25,
        n_rotations=base_cfg.null_calibration.n_rotations_eval,
        seed=seed + 100,
    )

    _log_stage(t0, "6a. Extract functional profiles + transition operator on gaoxin492")
    ref_profiles = extract_component_functional_profiles(
        bpd_ref, ref_ci_out, ref_clean_traj, _ref_plain_rollout, ref_classifications
    )
    _release_gpu_memory()
    _log_stage(t0, "6b. Extract functional profiles + transition operator on yagizdevre")
    tgt_profiles = extract_component_functional_profiles(
        bpd_tgt, tgt_ci_out, tgt_clean_traj, _tgt_plain_rollout, tgt_classifications
    )
    _release_gpu_memory()
    functional_matches = match_components_by_function(ref_profiles, tgt_profiles)
    selected_matches = functional_matches[: min(top_k_edit, len(functional_matches))]

    # 1-step transition operator & 2-step interventional path check on top component per module
    _log_stage(t0, "7. Compute top-1 transition operator + 2-step interventional path")
    top1_comps = {m: [0] for m in bpd_ref.module_order}
    trans_op = compute_transition_operator(
        bpd_ref, ref_clean_traj, _ref_plain_rollout, components_by_module=top1_comps
    )
    path_demo = None
    if total_invocations >= 3 and len(trans_op.keys) >= 2:
        demo_chain = [trans_op.keys[0], trans_op.keys[1], trans_op.keys[0]]
        path_res = compose_interventional_path(
            bpd_ref, ref_clean_traj, _ref_plain_rollout, trans_op, demo_chain, start_pass=0
        )
        path_demo = {
            "direct_source_effect": path_res.direct_source_effect,
            "interventional_mediated_effect": path_res.interventional_mediated_effect,
            "first_order_chain_product": path_res.first_order_chain_product,
            "transport_discrepancy": path_res.transport_discrepancy,
        }
    del bpd_ref, ref_adapter, cached_eval_ref, ref_ci_out, ref_clean_traj, trans_op
    _release_gpu_memory()

    # Gate B4: Intra-checkpoint sustaining edit strictly inside yagizdevre's own decomposition
    # + Matched-count/matched-norm Random Component Control + Rotation-Null Control
    _log_stage(t0, "8. Evaluate WB-Stall s=64 horizons across clean, edited, random, and rotation arms")
    before_horizon = evaluate_sudoku_horizon(
        clean_tgt_base, eval_inputs, eval_targets, eval_outer_steps=eval_horizons
    )
    ref_baseline_horizon = evaluate_sudoku_horizon(
        clean_ref_base, eval_inputs, eval_targets, eval_outer_steps=eval_horizons
    )

    edited_tgt = copy.deepcopy(clean_tgt_base)
    edit_receipt = apply_intra_checkpoint_sustaining_edit(
        edited_tgt, bpd_tgt, selected_matches, scale_delta=edit_scale_delta
    )
    after_horizon = evaluate_sudoku_horizon(
        edited_tgt, eval_inputs, eval_targets, eval_outer_steps=eval_horizons
    )
    del edited_tgt

    random_ctrl_tgt = copy.deepcopy(clean_tgt_base)
    random_receipt = apply_random_component_control_edit(
        random_ctrl_tgt,
        bpd_tgt,
        edit_receipt,
        exclude_matches=selected_matches,
        seed=seed + 2026,
    )
    random_horizon = evaluate_sudoku_horizon(
        random_ctrl_tgt, eval_inputs, eval_targets, eval_outer_steps=eval_horizons
    )
    del random_ctrl_tgt

    rot_ctrl_tgt = copy.deepcopy(clean_tgt_base)
    rot_receipt = apply_rotation_null_control_edit(
        rot_ctrl_tgt,
        bpd_tgt,
        selected_matches,
        edit_receipt,
        clean_traj=tgt_clean_traj,
        seed=seed + 4048,
    )
    rot_horizon = evaluate_sudoku_horizon(
        rot_ctrl_tgt, eval_inputs, eval_targets, eval_outer_steps=eval_horizons
    )
    del rot_ctrl_tgt
    _release_gpu_memory()
    _log_stage(t0, "9. Done")

    min_h = min(eval_horizons)
    s16_key = 16 if 16 in eval_horizons else min_h
    gate_b4_passed = bool(
        after_horizon.final_solve_rate > before_horizon.final_solve_rate
        and after_horizon.period2_oscillation_rate < before_horizon.period2_oscillation_rate
        and after_horizon.final_solve_rate > random_horizon.final_solve_rate
        and after_horizon.final_solve_rate > rot_horizon.final_solve_rate
        and (
            after_horizon.board_exact_by_step[s16_key]
            >= before_horizon.board_exact_by_step[s16_key] - 0.02
        )
    )

    return {
        "gate": "B4_trm_separate_decompositions_and_intra_checkpoint_edit",
        "registration_commit": resolve_registration_commit(registration_commit),
        "total_invocations_trained": total_invocations,
        "hinge_kappa": hinge_kappa,
        "rollout_mode_counts": mode_counts,
        "trajectory_state_matching_modes": {
            "outer_readout_only_default": {
                "match_inner_boundary_states": False,
                "eval_metrics": ref_eval_metrics,
                "pass_null_losses": {str(k): v for k, v in ref_null_losses.items()},
                "last_train_pass_loss_shares": {str(k): v for k, v in ref_last_shares.items()},
            },
            "with_inner_boundary_states": {
                "match_inner_boundary_states": True,
                "rollout_mode_counts": inner_mode_counts,
                "eval_metrics": ref_inner_eval_metrics,
                "pass_null_losses": {str(k): v for k, v in inner_null_losses.items()},
                "last_train_pass_loss_shares": {str(k): v for k, v in inner_last_shares.items()},
            },
        },
        "target_decomposition_summary": {
            "eval_metrics": tgt_eval_metrics,
            "pass_null_losses": {str(k): v for k, v in tgt_null_losses.items()},
            "last_train_pass_loss_shares": {str(k): v for k, v in tgt_last_shares.items()},
        },
        "stationary_fraction": float(sched_rep.is_stationary.float().mean().item()),
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
                "ref_net_sustaining": m.ref_net_sustaining,
                "tgt_net_sustaining": m.tgt_net_sustaining,
                "priority_score": m.priority_score,
            }
            for m in selected_matches
        ],
        "sustaining_edit_test": {
            "edit_scale_delta": edit_scale_delta,
            "num_eval_boards": int(eval_inputs.shape[0]),
            "reference_gaoxin492_clean": _horizon_to_dict(ref_baseline_horizon),
            "target_yagizdevre_clean": _horizon_to_dict(before_horizon),
            "intra_checkpoint_sustaining_edit": {
                "receipt": _receipt_to_dict(edit_receipt),
                "metrics": _horizon_to_dict(after_horizon),
            },
            "matched_random_component_control": {
                "receipt": _receipt_to_dict(random_receipt),
                "metrics": _horizon_to_dict(random_horizon),
            },
            "rotation_null_control": {
                "receipt": _receipt_to_dict(rot_receipt),
                "metrics": _horizon_to_dict(rot_horizon),
            },
            "delta_final_solve_at_max_horizon": (
                after_horizon.final_solve_rate - before_horizon.final_solve_rate
            ),
            "delta_ever_solved_at_max_horizon": (
                after_horizon.ever_solved_rate - before_horizon.ever_solved_rate
            ),
        },
        "gate_b4_passed": gate_b4_passed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref-ckpt", type=str, default="gaoxin492")
    ap.add_argument("--tgt-ckpt", type=str, default="yagizdevre")
    ap.add_argument("--sudoku-json", type=Path, default=None, help="Optional JSON list of {question, answer}")
    ap.add_argument("--bundle-json", type=Path, default=None, help="Optional loopscope02 benchmark_bundle.json")
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--train-outer-steps", type=int, default=2)
    ap.add_argument("--c", type=int, default=24)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--top-k-edit", type=int, default=4)
    ap.add_argument("--edit-scale", type=float, default=0.15)
    ap.add_argument("--hinge-kappa", type=float, default=0.10)
    ap.add_argument("--registration-commit", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-watts", type=float, default=300.0)
    ap.add_argument("--smoke", action="store_true", help="Fast deterministic CPU smoke run on mini-TRM")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.device.startswith("cuda") and not args.smoke:
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
            pgd_steps_grid=(0, 2, 5),
            switch_sweep=(0.0, 2e-4),
            top_k_edit=2,
            edit_scale_delta=args.edit_scale,
            hinge_kappa=args.hinge_kappa,
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
            n_steps=args.steps,
            top_k_edit=args.top_k_edit,
            edit_scale_delta=args.edit_scale,
            hinge_kappa=args.hinge_kappa,
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
