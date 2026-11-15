"""Gate B4 & Primary Target: 7M Attention-TRM BPD training, risk audits, and ``s = 64`` edit test.

Executes the TRM-first experimental protocol:
  1. Hybrid training on short clean-state transplant windows + full closed-loop rollouts
     (the ``v014`` single-transplant vs. closed-loop distinction).
  2. Adversary stability audit across PGD steps ``K_adv`` (e.g. ``0..160``), comparing
     time-shared ``shared_batch_and_pass`` against pass-resonant ``per_pass`` PGD.
  3. Switch-penalty sweep across ``coeff_switch`` to check against merging distinct
     same-phase mechanisms.
  4. Side-by-side raw and net-of-rotation-null importance table + transient vs.
     sustaining classification + 1-step transition operator.
  5. Sustaining weight-edit test (``gaoxin492`` reference -> ``yagizdevre`` target):
     ranks sustaining components differing between checkpoints, applies a permanent
     time-shared rank-``k`` weight edit to ``yagizdevre``, and evaluates board exact
     solve rate and period-2 oscillation up to ``s = 64`` without a refill network.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechinterp.bpd.adapters.trm import (
    TRMBPDAdapter,
    build_mini_trm,
    default_trm_c_per_module,
    evaluate_sudoku_horizon,
    load_pretrained_trm,
)
from mechinterp.bpd.adversary import evaluate_pgd_stability_curve
from mechinterp.bpd.analysis import (
    apply_sustaining_weight_edit,
    classify_transient_sustaining,
    compose_interventional_path,
    compute_transition_operator,
    extract_schedule,
    rank_sustaining_checkpoint_diffs,
)
from mechinterp.bpd.components import BPDModel, RolloutTrajectory
from mechinterp.bpd.config import (
    AblationConfig,
    AdversaryConfig,
    BPDConfig,
    CIConfig,
    MinimalityConfig,
    NullCalibrationConfig,
)
from mechinterp.bpd.losses import trajectory_recon_loss
from mechinterp.bpd.nulls import evaluate_component_null_report
from mechinterp.bpd.trainer import BPDTrainer
from mechinterp.loopscope02.scripts.trm_sudoku import EXIT_POWER_CAP, power_limit_watts


def run_trm_bpd_pipeline(
    ref_model: torch.nn.Module,
    tgt_model: torch.nn.Module,
    train_inputs: torch.Tensor,
    eval_inputs: torch.Tensor,
    eval_targets: torch.Tensor,
    *,
    train_outer_steps: int = 2,
    eval_horizons: tuple[int, ...] = (4, 8, 16),
    c_per_mod: int = 16,
    n_steps: int = 60,
    pgd_steps_grid: tuple[int, ...] = (0, 2, 5, 20, 80, 160),
    switch_sweep: tuple[float, ...] = (0.0, 2e-4, 2e-3),
    top_k_edit: int = 4,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, object]:
    from mechinterp.loopscope02.scripts.trm_sudoku import PureSudokuTRM

    assert isinstance(ref_model, PureSudokuTRM) and isinstance(tgt_model, PureSudokuTRM)
    ref_model = ref_model.to(device).eval()
    tgt_model = tgt_model.to(device).eval()

    adapter = TRMBPDAdapter(ref_model)
    total_invocations = adapter.invocations_for_outer_steps(train_outer_steps)
    window_invocations = adapter.calls_per_h_cycle
    c_map = default_trm_c_per_module(ref_model, c=c_per_mod)

    cfg = BPDConfig(
        c_per_module=c_map,
        n_passes=total_invocations,
        n_steps=n_steps,
        batch_size=int(train_inputs.shape[0]),
        seq_len=ref_model.total_len,
        seed=seed,
        faithfulness_warmup_steps=15,
        main_lr=2e-3,
        coeff_faith=1e3,
        coeff_stoch=0.5,
        coeff_adv=0.5,
        rollout_mode="hybrid",
        window_steps=window_invocations,
        closed_loop_prob=0.5,
        ci=CIConfig(
            variant="state_only",
            d_model=min(128, max(32, ref_model.hidden_size)),
            n_blocks=1,
            n_heads=4,
            mlp_hidden=min(256, max(64, 2 * ref_model.hidden_size)),
            max_passes=max(total_invocations * 2, 16),
            max_seq_len=max(ref_model.total_len, 16),
        ),
        ablation=AblationConfig(mode="pass_mean", sphere_rescale=True),
        minimality=MinimalityConfig(pool_mode="mean", coeff_level=2e-4, coeff_switch=2e-4),
        adversary=AdversaryConfig(
            scope="shared_batch_and_pass",
            inner_steps=1,
            eval_steps_grid=pgd_steps_grid,
        ),
        null_calibration=NullCalibrationConfig(
            use_tolerance_weights=True,
            n_rotations_train=2,
            n_rotations_eval=4,
            calibrate_every=max(n_steps // 2, 10),
        ),
    )

    bpd_ref = BPDModel(
        ref_model, c_map, max_passes=total_invocations, ablation=cfg.ablation, init_svd=True
    )
    trainer = BPDTrainer(
        bpd_ref,
        cfg,
        adapter.rollout,
        clean_window_state_fn=adapter.sample_clean_window_state,
        visible_dim=adapter.visible_dim,
        prompt_dim=adapter.prompt_dim,
        device=device,
    )

    cached_train = adapter.make_batch_with_clean_cache(train_inputs.to(device), total_invocations)
    cached_eval = adapter.make_batch_with_clean_cache(eval_inputs.to(device), total_invocations)

    trainer.run_faithfulness_warmup()
    tolerances = trainer.calibrate_baselines(cached_train)

    mode_counts = {"closed_loop": 0, "clean_window": 0}
    for step in range(n_steps):
        sm = trainer.train_step(cached_train, step)
        mode_counts[sm.mode] = mode_counts.get(sm.mode, 0) + 1

    eval_metrics, ci_out, clean_traj = trainer.evaluate(cached_eval)
    sched_rep = extract_schedule(ci_out)

    # Risk Audit 1: PGD adversary step-stability curve (time-shared vs pass-resonant)
    def _eval_sched_loss(sched):
        tr = trainer.run_masked_rollout(cached_eval, sched, n_passes=total_invocations)
        l_rec, _ = trajectory_recon_loss(
            tr,
            clean_traj,
            pass_tolerances=tolerances,
            state_boundary_weight=cfg.state_boundary_weight,
        )
        return l_rec

    pgd_shared = evaluate_pgd_stability_curve(
        ci_out.lower_leaky,
        c_map,
        _eval_sched_loss,
        scope="shared_batch_and_pass",
        steps_grid=pgd_steps_grid,
        seed=seed,
    )
    pgd_per_pass = evaluate_pgd_stability_curve(
        ci_out.lower_leaky,
        c_map,
        _eval_sched_loss,
        scope="per_pass",
        steps_grid=pgd_steps_grid,
        seed=seed,
    )

    # Risk Audit 2: Switch penalty sweep on frozen (V, U)
    switch_sweep_results: list[dict[str, float]] = []
    for sw_val in switch_sweep:
        sw_cfg = replace(
            cfg,
            n_steps=max(n_steps // 3, 6),
            coeff_adv=0.0,
            minimality=replace(cfg.minimality, coeff_switch=sw_val),
        )
        sw_trainer = BPDTrainer(
            bpd_ref,
            sw_cfg,
            adapter.rollout,
            clean_window_state_fn=adapter.sample_clean_window_state,
            device=device,
        )
        # Freeze physical (V, U) so sweep isolates switch penalty effect on CI schedule
        for p in bpd_ref.component_parameters():
            p.requires_grad_(False)
        try:
            for st in range(sw_cfg.n_steps):
                sw_trainer.train_step(cached_train, st)
            sw_eval, sw_ci, _ = sw_trainer.evaluate(cached_eval)
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

    # Raw + Net-of-Rotation-Null Importance & Transient vs Sustaining Classification
    def _plain_rollout() -> RolloutTrajectory:
        ro, bs, vis, prm = adapter.rollout(cached_eval, total_invocations, 0, None)
        return RolloutTrajectory(
            readouts=ro,
            boundary_states=bs,
            module_inputs=clean_traj.module_inputs,
            visible_states=vis,
            prompt_features=prm,
        )

    inspected_comps = {m: list(range(min(4, c_per_mod))) for m in bpd_ref.module_order}
    null_rep = evaluate_component_null_report(
        bpd_ref,
        clean_traj,
        _plain_rollout,
        components_by_module=inspected_comps,
        passes=[0, total_invocations - 1],
        include_all_pass=True,
        n_rotations=cfg.null_calibration.n_rotations_eval,
        seed=seed,
    )
    classifications = classify_transient_sustaining(
        bpd_ref,
        clean_traj,
        _plain_rollout,
        components_by_module=inspected_comps,
        min_effect=1e-4,
        sustain_ratio=1.25,
        n_rotations=cfg.null_calibration.n_rotations_eval,
        seed=seed,
    )

    # 1-step transition operator & 2-step interventional path check on top component per module
    top1_comps = {m: [0] for m in bpd_ref.module_order}
    trans_op = compute_transition_operator(
        bpd_ref, clean_traj, _plain_rollout, components_by_module=top1_comps
    )
    path_demo = None
    if total_invocations >= 3 and len(trans_op.keys) >= 2:
        demo_chain = [trans_op.keys[0], trans_op.keys[1], trans_op.keys[0]]
        path_res = compose_interventional_path(
            bpd_ref, clean_traj, _plain_rollout, trans_op, demo_chain, start_pass=0
        )
        path_demo = {
            "direct_source_effect": path_res.direct_source_effect,
            "interventional_mediated_effect": path_res.interventional_mediated_effect,
            "first_order_chain_product": path_res.first_order_chain_product,
            "transport_discrepancy": path_res.transport_discrepancy,
        }

    # Gate B4: Sustaining weight-edit test on target checkpoint (yagizdevre) without refill network
    before_horizon = evaluate_sudoku_horizon(
        tgt_model, eval_inputs, eval_targets, eval_outer_steps=eval_horizons
    )
    diffs = rank_sustaining_checkpoint_diffs(bpd_ref, tgt_model, classifications)
    edited_tgt = copy.deepcopy(tgt_model)
    selected_edits = diffs[:top_k_edit]
    edited_counts = apply_sustaining_weight_edit(
        edited_tgt, bpd_ref, selected_edits, interpolation=1.0
    )
    after_horizon = evaluate_sudoku_horizon(
        edited_tgt, eval_inputs, eval_targets, eval_outer_steps=eval_horizons
    )

    max_h = max(eval_horizons)
    return {
        "gate": "B4_trm_hybrid_and_sustaining_edit",
        "total_invocations_trained": total_invocations,
        "rollout_mode_counts": mode_counts,
        "pass_tolerances": {str(k): v for k, v in tolerances.items()},
        "eval_metrics": eval_metrics,
        "stationary_fraction": float(sched_rep.is_stationary.float().mean().item()),
        "pgd_stability": {
            "shared_batch_and_pass": pgd_shared.loss_by_steps,
            "shared_amplification": pgd_shared.amplification_ratio,
            "per_pass_resonant": pgd_per_pass.loss_by_steps,
            "per_pass_amplification": pgd_per_pass.amplification_ratio,
        },
        "switch_penalty_sweep": switch_sweep_results,
        "raw_and_net_null_sample": null_rep.to_dicts()[:12],
        "role_counts": {
            role: sum(1 for e in classifications if e.role == role)
            for role in ("sustaining", "transient", "bottleneck", "inert")
        },
        "interventional_path_demo": path_demo,
        "sustaining_edit_test": {
            "edited_module_counts": edited_counts,
            "num_selected_sustaining_diffs": len(selected_edits),
            "before_board_exact": {str(k): v for k, v in before_horizon.board_exact_by_step.items()},
            "after_board_exact": {str(k): v for k, v in after_horizon.board_exact_by_step.items()},
            "before_cell_acc": {str(k): v for k, v in before_horizon.cell_acc_by_step.items()},
            "after_cell_acc": {str(k): v for k, v in after_horizon.cell_acc_by_step.items()},
            "before_period2_oscillation": before_horizon.period2_oscillation_rate,
            "after_period2_oscillation": after_horizon.period2_oscillation_rate,
            "delta_max_horizon_exact": (
                after_horizon.board_exact_by_step[max_h] - before_horizon.board_exact_by_step[max_h]
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref-ckpt", type=str, default="gaoxin492")
    ap.add_argument("--tgt-ckpt", type=str, default="yagizdevre")
    ap.add_argument("--sudoku-json", type=Path, default=None, help="JSON list of {question, answer} 81-char pairs")
    ap.add_argument("--train-outer-steps", type=int, default=4)
    ap.add_argument("--c", type=int, default=32)
    ap.add_argument("--steps", type=int, default=200)
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
            n_steps=10,
            pgd_steps_grid=(0, 2, 5),
            switch_sweep=(0.0, 2e-4),
            top_k_edit=2,
            device="cpu",
            seed=0,
        )
    else:
        from mechinterp.bpd.adapters.trm import encode_sudoku_pairs

        dt = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
        ref = load_pretrained_trm(args.ref_ckpt, dtype=dt, verify_sha256=True)
        tgt = load_pretrained_trm(args.tgt_ckpt, dtype=dt, verify_sha256=True)
        if args.sudoku_json is None:
            raise ValueError("Non-smoke TRM run requires --sudoku-json pointing to Sudoku puzzles")
        rows = json.loads(args.sudoku_json.read_text(encoding="utf-8"))
        qs = [r["question"] for r in rows]
        sols = [r["answer"] for r in rows]
        inp, ans = encode_sudoku_pairs(qs, sols, device="cpu")
        n_train = min(32, max(inp.shape[0] // 2, 1))
        res = run_trm_bpd_pipeline(
            ref,
            tgt,
            inp[:n_train],
            inp[n_train:],
            ans[n_train:],
            train_outer_steps=args.train_outer_steps,
            eval_horizons=(16, 32, 64),
            c_per_mod=args.c,
            n_steps=args.steps,
            device=args.device,
        )

    text = json.dumps(res, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
