"""Gate B2: ``HUG-CAL-02`` 8-factor negative control with raw + net-of-null reporting.

Tests the falsifiable negative-control requirement:
  All 8 leading settling factors across the 4 recurrent core blocks
  (``core.{0..3}.{o_proj, down_proj}``) exhibit nonzero **raw** ablation shift due
  to loop-settling Jacobian fragility, yet must come out with ~zero **net**
  importance (``|z_rot| <= z_threshold``) against an RMS-matched Haar rotation null.
  Both raw and net importance are always reported together so fragile-pass nulls
  cannot silently hide genuine aligned mechanisms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechinterp.bpd.adapters.huginn import (
    HuginnBPDAdapter,
    SettlingCoreLooper,
    discover_huginn_core_linears,
    evaluate_hug_cal02_eight_factors,
)
from mechinterp.bpd.components import BPDModel, RolloutTrajectory
from mechinterp.bpd.nulls import evaluate_component_null_report


def run_synthetic_hug_cal02(
    *,
    n_passes: int = 6,
    n_rotations: int = 16,
    batch_size: int = 24,
    seq_len: int = 12,
    z_threshold: float = 2.5,
    seed: int = 11,
) -> dict[str, object]:
    gen = torch.Generator().manual_seed(seed)
    model = SettlingCoreLooper(vocab_size=32, d_model=64, n_heads=4, n_core_blocks=4, seed=seed)
    tokens = torch.randint(0, 32, (batch_size, seq_len), generator=gen)
    audit = evaluate_hug_cal02_eight_factors(
        model,
        tokens,
        n_passes=n_passes,
        n_rotations=n_rotations,
        z_threshold=z_threshold,
        seed=seed,
    )
    return {
        "gate": "B2_hug_cal02_negative_control",
        "target": "SettlingCoreLooper_4block",
        "n_passes": n_passes,
        "n_rotations": n_rotations,
        "z_threshold": z_threshold,
        "max_raw_importance": audit.max_raw_importance,
        "mean_raw_importance": audit.mean_raw_importance,
        "mean_abs_net_importance": audit.mean_abs_net_importance,
        "max_abs_z_rot": audit.max_abs_z_rot,
        "gate_b2_passed": audit.passed_negative_control,
        "factors": [
            {
                "module": e.module,
                "component": e.component,
                "raw_importance": e.raw_importance,
                "rotation_null_mean": e.rotation_null_mean,
                "rotation_null_std": e.rotation_null_std,
                "net_importance": e.net_importance,
                "z_rot": e.z_rot,
            }
            for e in audit.factor_entries
        ],
    }


def run_hf_huginn_audit(
    repo_id: str,
    *,
    n_passes: int = 8,
    n_rotations: int = 16,
    z_threshold: float = 2.5,
    device: str = "cuda",
    seed: int = 11,
) -> dict[str, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    all_core = discover_huginn_core_linears(model, c=4)
    # Pick 8 output-writing projections across the 4 core blocks (2 per block)
    selected = {
        k: 4
        for k in sorted(all_core.keys())
        if any(tag in k for tag in ("proj", "fc_2", "down_proj", "o_proj"))
    }
    selected = dict(list(selected.items())[:8])

    adapter = HuginnBPDAdapter(model)
    bpd_model = BPDModel(model, selected, max_passes=n_passes, init_svd=True).to(device)
    prompts = [
        "In a recurrent transformer, shared weights iterate over latent states to",
        "When sensitivity varies across loop steps, causal attribution requires",
        "Pointer chasing on a directed graph advances one edge per iteration until",
        "Constraint propagation on a 9x9 Sudoku board eliminates invalid digits in",
    ]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=24)
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        with bpd_model.target_capture(pass_offset=0, update_means=True, detach_inputs=True):
            ro, bs, vis, prm = adapter.rollout(input_ids, n_passes, 0, None)
        mod_in = bpd_model.collect_recorded_inputs(expected_passes=n_passes)
    clean_traj = RolloutTrajectory(
        readouts=ro, boundary_states=bs, module_inputs=mod_in, visible_states=vis, prompt_features=prm
    )

    def _run() -> RolloutTrajectory:
        p_ro, p_bs, p_vis, p_prm = adapter.rollout(input_ids, n_passes, 0, None)
        return RolloutTrajectory(
            readouts=p_ro,
            boundary_states=p_bs,
            module_inputs=mod_in,
            visible_states=p_vis,
            prompt_features=p_prm,
        )

    rep = evaluate_component_null_report(
        bpd_model,
        clean_traj,
        _run,
        components_by_module={m: [0] for m in selected},
        passes=[],
        include_all_pass=True,
        n_rotations=n_rotations,
        seed=seed,
    )
    max_z = max(abs(e.z_rot) for e in rep.entries)
    max_raw = max(e.raw_importance for e in rep.entries)
    return {
        "gate": "B2_hug_cal02_negative_control",
        "target": repo_id,
        "n_passes": n_passes,
        "n_rotations": n_rotations,
        "max_raw_importance": max_raw,
        "max_abs_z_rot": max_z,
        "gate_b2_passed": bool(max_raw > 0.0 and max_z <= z_threshold),
        "factors": rep.to_dicts(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-repo", type=str, default=None, help="Optional HF repo e.g. tomg-group-umd/huginn-0125")
    ap.add_argument("--passes", type=int, default=6)
    ap.add_argument("--rotations", type=int, default=16)
    ap.add_argument("--z-threshold", type=float, default=2.5)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.hf_repo is not None and not args.smoke:
        res = run_hf_huginn_audit(
            args.hf_repo,
            n_passes=args.passes,
            n_rotations=args.rotations,
            z_threshold=args.z_threshold,
            device=args.device,
        )
    else:
        res = run_synthetic_hug_cal02(
            n_passes=4 if args.smoke else args.passes,
            n_rotations=8 if args.smoke else args.rotations,
            batch_size=12 if args.smoke else 24,
            z_threshold=args.z_threshold,
        )

    text = json.dumps(res, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
