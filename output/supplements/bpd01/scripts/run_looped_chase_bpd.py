"""Gate B3: ``LoopedChase`` ground-truth BPD schedule, Clock Gap, and Visible Gap.

Tests the falsifiable ground-truth predictions on ``LoopedChase``:
  1. At ``lam = 0.1`` (per-step hop supervision), BPD recovers a stationary hop
     schedule nearly constant across ``tau`` with ``Clock Gap ~ 0`` and ``Visible Gap ~ 0``.
  2. At ``lam = 1.0``, seeds 1 & 2 (which carry off-axis ``ker(head)`` phase/relay
     coordinates ``R_t`` with ``Price(R_4) = +3.06, +5.46`` loops) exhibit a positive
     ``Visible Gap`` and non-stationary phase components that seed 3 (``Price(R_4) = 0.00``)
     lacks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechinterp.bpd.adapters.looped_chase import (
    LoopedChaseBPDAdapter,
    default_looped_chase_c_per_module,
    load_looped_chase_checkpoint,
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
)
from mechinterp.bpd.trainer import BPDTrainer, train_ci_heads_on_frozen_components
from mechinterp.loopscope01.scripts.loopscope_toy import LoopedChase


def run_looped_chase_experiment(
    model: LoopedChase,
    *,
    label: str,
    n_passes: int = 6,
    c_per_mod: int = 16,
    n_steps: int = 100,
    ci_head_steps: int = 60,
    batch_size: int = 24,
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
        seq_len=2 * model.n_nodes + 1,
        seed=seed,
        faithfulness_warmup_steps=20,
        main_lr=2e-3,
        coeff_faith=1e3,
        coeff_stoch=0.5,
        coeff_adv=0.5,
        rollout_mode="hybrid",
        window_steps=min(2, max(n_passes - 1, 1)),
        closed_loop_prob=0.5,
        ci=CIConfig(variant="state_only", d_model=64, n_blocks=1, n_heads=4, mlp_hidden=128),
        ablation=AblationConfig(mode="pass_mean", sphere_rescale=True),
        minimality=MinimalityConfig(pool_mode="mean", coeff_level=2e-4, coeff_switch=2e-4),
        adversary=AdversaryConfig(scope="shared_batch_and_pass", inner_steps=1),
        null_calibration=NullCalibrationConfig(use_tolerance_weights=True, n_rotations_train=2),
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
    trainer.calibrate_baselines(eval_batch)

    for step in range(n_steps):
        tok, hops = sample_pointer_chase_batch(
            n_nodes=model.n_nodes, batch_size=batch_size, max_depth=n_passes, seed=seed + step + 1
        )
        batch = adapter.make_batch_with_clean_cache(tok, n_passes, hops=hops)
        trainer.train_step(batch, step)

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
        pass_tolerances=trainer.pass_tolerances,
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
        "label": label,
        "n_passes": n_passes,
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=None, help="Path to trained LoopedChase .final.pt")
    ap.add_argument("--label", type=str, default="smoke_looped_chase")
    ap.add_argument("--passes", type=int, default=6)
    ap.add_argument("--c", type=int, default=12)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.ckpt is not None and not args.smoke:
        model = load_looped_chase_checkpoint(args.ckpt, device=args.device)
        res = run_looped_chase_experiment(
            model,
            label=args.label,
            n_passes=args.passes,
            c_per_mod=args.c,
            n_steps=args.steps,
            ci_head_steps=max(args.steps // 2, 20),
            device=args.device,
        )
    else:
        torch.manual_seed(0)
        mini = LoopedChase(n_nodes=8, d=32, layers=1, heads=4)
        res = run_looped_chase_experiment(
            mini,
            label=args.label,
            n_passes=4,
            c_per_mod=6,
            n_steps=16 if args.smoke else args.steps,
            ci_head_steps=10 if args.smoke else 30,
            batch_size=12,
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
