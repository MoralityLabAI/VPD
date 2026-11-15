"""Gate B3: ``LoopedChase`` planted positive control, ground-truth schedule, Clock Gap, and Visible Gap.

Evaluates:
  1. Planted rank-1 positive control on ``lam0.1_s1.final.pt`` against a registered
     rotation-null ``z_threshold`` (reporting ``observed_z_rot`` and ``margin``).
  2. ``lam = 0.1`` (``lam0.1_s1``) stationary hop schedule, Clock Gap, and Visible Gap.
  3. ``lam = 1.0`` seeds 1 & 2 (off-axis ``ker(head)`` phase/relay coordinates) against
     ``lam = 1.0`` seed 3 on Visible Gap and non-stationary component schedules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechinterp.bpd.adapters.looped_chase import (
    DEFAULT_LOOPED_CHASE_CKPT_DIR,
    LoopedChaseBPDAdapter,
    default_looped_chase_c_per_module,
    evaluate_planted_rank1_looped_chase,
    load_looped_chase_checkpoint,
    resolve_looped_chase_checkpoint,
    sample_pointer_chase_batch,
)
from mechinterp.bpd.analysis import classify_transient_sustaining, extract_schedule
from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import (
    AblationConfig,
    AdversaryConfig,
    BPDConfig,
    CIConfig,
    MinimalityConfig,
    NullCalibrationConfig,
    resolve_registration_commit,
)
from mechinterp.bpd.trainer import BPDTrainer, train_ci_heads_on_frozen_components
from mechinterp.loopscope01.scripts.loopscope_toy import LoopedChase


def run_looped_chase_experiment(
    model: LoopedChase,
    *,
    label: str,
    n_passes: int = 8,
    c_per_mod: int = 16,
    n_steps: int = 100,
    ci_head_steps: int = 60,
    batch_size: int = 24,
    hinge_kappa: float = 0.10,
    registration_commit: str | None = None,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, object]:
    model = model.to(device).eval()
    adapter = LoopedChaseBPDAdapter(model)
    c_map = default_looped_chase_c_per_module(model, c=c_per_mod)

    cfg = BPDConfig(
        c_per_module=c_map,
        n_passes=n_passes,
        n_steps=n_steps,
        batch_size=batch_size,
        seq_len=model.n_nodes + 1,
        seed=seed,
        faithfulness_warmup_steps=20,
        main_lr=2e-3,
        coeff_faith=1e3,
        coeff_stoch=0.5,
        coeff_adv=0.5,
        rollout_mode="hybrid",
        window_steps=min(2, max(n_passes - 1, 1)),
        closed_loop_prob=0.5,
        ci=CIConfig(
            variant="state_only",
            d_model=64,
            n_blocks=1,
            n_heads=4,
            mlp_hidden=128,
            max_passes=max(n_passes * 2, 16),
            max_seq_len=max(model.n_nodes + 1, 16),
        ),
        ablation=AblationConfig(mode="pass_mean", sphere_rescale=True),
        minimality=MinimalityConfig(pool_mode="mean", coeff_level=2e-4, coeff_switch=2e-4),
        adversary=AdversaryConfig(scope="shared_batch_and_pass", inner_steps=1),
        null_calibration=NullCalibrationConfig(
            use_fragility_hinge=True,
            hinge_kappa=hinge_kappa,
            n_rotations_train=2,
        ),
    )

    bpd_model = BPDModel(model, c_map, max_passes=n_passes, ablation=cfg.ablation, init_svd=True)
    trainer = BPDTrainer(
        bpd_model,
        cfg,
        adapter.rollout,
        clean_window_state_fn=adapter.sample_clean_window_state,
        visible_dim=adapter.visible_dim,
        prompt_dim=adapter.prompt_dim,
        device=device,
    )

    eval_tok, eval_hops = sample_pointer_chase_batch(
        n_nodes=model.n_nodes, batch_size=batch_size, max_depth=n_passes, seed=seed + 9000
    )
    eval_batch = adapter.make_batch_with_clean_cache(eval_tok, n_passes, hops=eval_hops)

    trainer.run_faithfulness_warmup()
    null_losses = trainer.calibrate_baselines(eval_batch)

    last_train_shares: dict[int, float] = {}
    for step in range(n_steps):
        tok, hops = sample_pointer_chase_batch(
            n_nodes=model.n_nodes, batch_size=batch_size, max_depth=n_passes, seed=seed + step + 1
        )
        batch = adapter.make_batch_with_clean_cache(tok, n_passes, hops=hops)
        sm = trainer.train_step(batch, step)
        last_train_shares = sm.pass_loss_shares

    eval_metrics, ci_out, clean_traj = trainer.evaluate(eval_batch)
    sched_report = extract_schedule(ci_out, position_pool="last")

    # Train all 3 CI variants on the frozen shared (V, U) decomposition to measure Clock & Visible Gaps
    ci_cfg = BPDConfig(
        c_per_module=c_map,
        n_passes=n_passes,
        n_steps=ci_head_steps,
        batch_size=batch_size,
        seq_len=cfg.seq_len,
        seed=seed,
        main_lr=2e-3,
        ci=cfg.ci,
        ablation=cfg.ablation,
        minimality=cfg.minimality,
        null_calibration=cfg.null_calibration,
    )

    def _batch_fn(step_idx: int):
        tok, _ = sample_pointer_chase_batch(
            n_nodes=model.n_nodes,
            batch_size=batch_size,
            max_depth=n_passes,
            seed=seed + 5000 + step_idx,
        )
        return tok.to(device)

    _, variant_metrics, gaps = train_ci_heads_on_frozen_components(
        bpd_model,
        ci_cfg,
        adapter.rollout,
        _batch_fn,
        eval_tok.to(device),
        visible_dim=adapter.visible_dim,
        prompt_dim=adapter.prompt_dim,
        pass_null_losses=trainer.pass_null_losses,
        hinge_kappa=cfg.null_calibration.hinge_kappa,
        device=device,
    )

    def _plain_rollout():
        ro, bs, vis, prm = adapter.rollout(eval_tok.to(device), n_passes, 0, None)
        from mechinterp.bpd.components import RolloutTrajectory

        return RolloutTrajectory(
            readouts=ro,
            boundary_states=bs,
            module_inputs=clean_traj.module_inputs,
            visible_states=vis,
            prompt_features=prm,
        )

    # Classify top-2 components per module for fast summary
    sub_comps = {m: list(range(min(2, c_per_mod))) for m in bpd_model.module_order}
    roles = classify_transient_sustaining(
        bpd_model, clean_traj, _plain_rollout, components_by_module=sub_comps, n_rotations=2
    )

    return {
        "gate": "B3_looped_chase_ground_truth",
        "registration_commit": resolve_registration_commit(registration_commit),
        "label": label,
        "n_passes": n_passes,
        "hinge_kappa": hinge_kappa,
        "pass_null_losses": {str(k): v for k, v in null_losses.items()},
        "last_train_pass_loss_shares": {str(k): v for k, v in last_train_shares.items()},
        "eval_metrics": eval_metrics,
        "stationary_component_fraction": float(sched_report.is_stationary.float().mean().item()),
        "mean_switch_rate": float(sched_report.switch_rate.mean().item()),
        "ci_variants": variant_metrics,
        "gaps": gaps,
        "sample_component_roles": [
            {
                "module": r.module,
                "component": r.component,
                "role": r.role,
                "raw_all_pass_final": r.raw_all_pass_final,
                "net_all_pass_final": r.net_all_pass_final,
            }
            for r in roles
        ],
    }


def run_gate_b3_suite(
    *,
    ckpt_dir: Path = DEFAULT_LOOPED_CHASE_CKPT_DIR,
    n_passes: int = 8,
    c_per_mod: int = 16,
    n_steps: int = 100,
    ci_head_steps: int = 60,
    batch_size: int = 24,
    hinge_kappa: float = 0.10,
    z_threshold: float = 2.5,
    registration_commit: str | None = None,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, object]:
    """Run Gate B3 across ``lam0.1_s1`` + ``lam1_s1..3`` plus the planted rank-1 positive control."""
    reg_sha = resolve_registration_commit(registration_commit)
    lam01_path, lam01_sha = resolve_looped_chase_checkpoint("lam0.1_s1", ckpt_dir=ckpt_dir)
    lam01_orig = load_looped_chase_checkpoint(lam01_path, device=device)

    planted_res = evaluate_planted_rank1_looped_chase(
        lam01_orig,
        n_passes=n_passes,
        z_threshold=z_threshold,
        checkpoint_label="lam0.1_s1",
        checkpoint_sha256=lam01_sha,
    )

    arms = ("lam0.1_s1", "lam1_s1", "lam1_s2", "lam1_s3")
    arm_results: dict[str, dict[str, object]] = {}
    ckpt_shas: dict[str, str] = {}
    for arm_idx, arm in enumerate(arms):
        ck_path, ck_sha = resolve_looped_chase_checkpoint(arm, ckpt_dir=ckpt_dir)
        ckpt_shas[arm] = ck_sha
        m = load_looped_chase_checkpoint(ck_path, device=device)
        arm_results[arm] = run_looped_chase_experiment(
            m,
            label=arm,
            n_passes=n_passes,
            c_per_mod=c_per_mod,
            n_steps=n_steps,
            ci_head_steps=ci_head_steps,
            batch_size=batch_size,
            hinge_kappa=hinge_kappa,
            registration_commit=reg_sha,
            device=device,
            seed=seed + 100 * arm_idx,
        )

    r01 = arm_results["lam0.1_s1"]
    g01 = r01["gaps"]  # type: ignore[assignment]
    g_s1 = arm_results["lam1_s1"]["gaps"]  # type: ignore[assignment]
    g_s2 = arm_results["lam1_s2"]["gaps"]  # type: ignore[assignment]
    g_s3 = arm_results["lam1_s3"]["gaps"]  # type: ignore[assignment]

    lam01_stationary_ok = (
        float(r01["stationary_component_fraction"]) >= 0.50  # type: ignore[arg-type]
        and abs(float(g01["clock_gap_recon"])) <= 0.05  # type: ignore[index]
        and abs(float(g01["visible_gap_recon"])) <= 0.05  # type: ignore[index]
    )
    s1_beats_s3 = (
        float(g_s1["visible_gap_recon"]) > float(g_s3["visible_gap_recon"])  # type: ignore[index]
        or float(g_s1["visible_gap_active"]) > float(g_s3["visible_gap_active"])  # type: ignore[index]
    )
    s2_beats_s3 = (
        float(g_s2["visible_gap_recon"]) > float(g_s3["visible_gap_recon"])  # type: ignore[index]
        or float(g_s2["visible_gap_active"]) > float(g_s3["visible_gap_active"])  # type: ignore[index]
    )
    gate_b3_passed = bool(
        planted_res.passed_positive_control and lam01_stationary_ok and s1_beats_s3 and s2_beats_s3
    )

    return {
        "gate": "B3_looped_chase_suite_with_planted_control",
        "registration_commit": reg_sha,
        "checkpoint_sha256": ckpt_shas,
        "planted_positive_control": planted_res.to_dict(),
        "arms": arm_results,
        "checks": {
            "planted_positive_control_passed": planted_res.passed_positive_control,
            "lam0.1_stationary_and_small_gaps": lam01_stationary_ok,
            "lam1_s1_exceeds_s3_visible_gap": s1_beats_s3,
            "lam1_s2_exceeds_s3_visible_gap": s2_beats_s3,
        },
        "gate_b3_passed": gate_b3_passed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=None, help="Optional single LoopedChase .final.pt")
    ap.add_argument("--ckpt-dir", type=Path, default=DEFAULT_LOOPED_CHASE_CKPT_DIR)
    ap.add_argument("--suite", action="store_true", help="Run full 4-checkpoint Gate B3 + planted control")
    ap.add_argument("--label", type=str, default="smoke_looped_chase")
    ap.add_argument("--passes", type=int, default=8)
    ap.add_argument("--c", type=int, default=16)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--hinge-kappa", type=float, default=0.10)
    ap.add_argument("--z-threshold", type=float, default=2.5)
    ap.add_argument("--registration-commit", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.smoke:
        torch.manual_seed(0)
        mini = LoopedChase(n_nodes=8, d=32, layers=1, heads=4)
        planted_smoke = evaluate_planted_rank1_looped_chase(
            mini,
            n_passes=4,
            cal_batch_size=48,
            eval_batch_size=16,
            n_rotations=8,
            sigma=25.0,
            z_threshold=args.z_threshold,
            checkpoint_label="mini_smoke",
        )
        res = run_looped_chase_experiment(
            mini,
            label=args.label,
            n_passes=4,
            c_per_mod=6,
            n_steps=16,
            ci_head_steps=10,
            batch_size=12,
            hinge_kappa=args.hinge_kappa,
            registration_commit=args.registration_commit,
            device=args.device,
        )
        res["planted_positive_control"] = planted_smoke.to_dict()
    elif args.suite or args.ckpt is None:
        res = run_gate_b3_suite(
            ckpt_dir=args.ckpt_dir,
            n_passes=args.passes,
            c_per_mod=args.c,
            n_steps=args.steps,
            ci_head_steps=max(args.steps // 2, 20),
            hinge_kappa=args.hinge_kappa,
            z_threshold=args.z_threshold,
            registration_commit=args.registration_commit,
            device=args.device,
        )
    else:
        ck_path, ck_sha = resolve_looped_chase_checkpoint(args.ckpt, ckpt_dir=args.ckpt_dir)
        model = load_looped_chase_checkpoint(ck_path, device=args.device)
        res = run_looped_chase_experiment(
            model,
            label=args.label,
            n_passes=args.passes,
            c_per_mod=args.c,
            n_steps=args.steps,
            ci_head_steps=max(args.steps // 2, 20),
            hinge_kappa=args.hinge_kappa,
            registration_commit=args.registration_commit,
            device=args.device,
        )
        res["checkpoint_sha256"] = ck_sha

    text = json.dumps(res, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
