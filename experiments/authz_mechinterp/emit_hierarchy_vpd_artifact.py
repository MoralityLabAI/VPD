from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.analyze_group_logit_interventions import sha256_file
from experiments.authz_mechinterp.build_hierarchy_vpd import load_jsonl, parse_component


def compact_effect(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_invalid_margin_reduction": row["target_invalid_margin_reduction"],
        "target_invalid_margin_reduction_ci95": row["target_invalid_margin_reduction_ci95"],
        "target_control_advantage": row["target_control_advantage"],
        "target_control_advantage_ci95": row["target_control_advantage_ci95"],
        "invalid_specificity_margin": row["invalid_specificity_margin"],
        "invalid_specificity_margin_ci95": row["invalid_specificity_margin_ci95"],
        "guardrails_passed": row["guardrails_passed"],
    }


def verify_projection_sidecars(path: Path) -> dict[str, Any]:
    component_dir = path / "components"
    tensors = sorted(component_dir.glob("*.pt"))
    if len(tensors) != 252:
        raise ValueError(f"expected 252 base projection tensors, found {len(tensors)}")
    failures: list[str] = []
    total_bytes = 0
    for tensor in tensors:
        sidecar = tensor.with_suffix(tensor.suffix + ".sha256")
        if not sidecar.is_file():
            failures.append(f"missing:{tensor.name}")
            continue
        expected = sidecar.read_text(encoding="utf-8").strip().lower()
        observed = sha256_file(tensor)
        if expected != observed:
            failures.append(f"hash:{tensor.name}")
        total_bytes += tensor.stat().st_size
    if failures:
        raise ValueError(f"base projection integrity failures: {failures[:5]}")
    return {
        "path": str(path.resolve()),
        "rank": 64,
        "tensor_count": len(tensors),
        "tensor_bytes": total_bytes,
        "all_tensor_sidecars_verified": True,
        "inventory_path": str((path / "projection-inventory.jsonl").resolve()),
        "inventory_sha256": sha256_file(path / "projection-inventory.jsonl"),
        "summary_path": str((path / "summary.json").resolve()),
        "summary_sha256": sha256_file(path / "summary.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-inventory", type=Path, required=True)
    parser.add_argument("--group-plan", type=Path, required=True)
    parser.add_argument("--discovery-summary", type=Path, required=True)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--validation-raw", type=Path, required=True)
    parser.add_argument("--adapter-weight", type=Path, required=True)
    parser.add_argument("--base-projection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    inventory = [parse_component(row) for row in load_jsonl(args.component_inventory)]
    if len(inventory) != 4032:
        raise ValueError(f"expected 4032 components, found {len(inventory)}")
    plan = json.loads(args.group_plan.read_text(encoding="utf-8"))
    discovery = json.loads(args.discovery_summary.read_text(encoding="utf-8"))
    validation = json.loads(args.validation_summary.read_text(encoding="utf-8"))
    discovery_by_id = {str(row["group_id"]): row for row in discovery["groups"]}
    validation_by_id = {str(row["group_id"]): row for row in validation["groups"]}
    selected_group_id = str(discovery["selection"]["selected_group_id"])
    plan_by_id = {str(row["group_id"]): row for row in plan["groups"]}
    selected_component_ids = set(
        str(value) for value in plan_by_id[selected_group_id]["target_component_ids"]
    )
    if len(selected_component_ids) != 144:
        raise ValueError("selected cell does not contain exactly 144 components")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    component_output = args.output_dir / "vpd_hierarchy_components.jsonl"
    with component_output.open("w", encoding="utf-8", newline="\n") as handle:
        for component in sorted(inventory, key=lambda row: str(row["component_id"])):
            family = str(component["projection_family"])
            depth = int(component["depth_band"])
            family_group_id = f"family_{family.replace('.', '_')}"
            depth_group_id = f"depth_{depth}"
            cell_group_id = f"cell_{family.replace('.', '_')}_depth_{depth}"
            selected = str(component["component_id"]) in selected_component_ids
            row = {
                "schema": "bonsai8b_vpd_component_evidence_v1",
                "model_id": component["model_id"],
                "revision": component["revision"],
                "component_id": component["component_id"],
                "module": component["module"],
                "layer": component["layer"],
                "rank_index": component["rank_index"],
                "projection_family": family,
                "depth_band": depth,
                "component_fro_norm": component["component_fro_norm"],
                "a_sha256": component["a_sha256"],
                "b_sha256": component["b_sha256"],
                "adapter_scale": component["scale"],
                "fraction_of_module_delta_fro_sq": component["fraction_of_delta_fro_sq"],
                "family_group_id": family_group_id,
                "depth_group_id": depth_group_id,
                "cell_group_id": cell_group_id,
                "discovery_family_joint_effect": compact_effect(discovery_by_id[family_group_id]),
                "discovery_depth_joint_effect": compact_effect(discovery_by_id[depth_group_id]),
                "discovery_cell_joint_effect": compact_effect(discovery_by_id[cell_group_id]),
                "selected_cell": selected,
                "selected_cell_validation_joint_effect": compact_effect(
                    validation_by_id[selected_group_id]
                )
                if selected
                else None,
                "causal_evidence_granularity": (
                    "joint_group_ablation_only_not_individual_component_causality"
                ),
                "causal_status": (
                    "selected_cell_internal_validation_failed"
                    if selected
                    else "jointly_screened_not_selected"
                ),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    projection = verify_projection_sidecars(args.base_projection)
    selected_validation = validation_by_id[selected_group_id]
    hook_path = args.output_dir / "mechanistic_hooks.jsonl"
    hook = {
        "schema_version": "1.0",
        "hook_id": "bonsai8b-authz-v5-selected-cell",
        "kind": "vpd",
        "metric_id": "decision_logit_margin",
        "probe_family_id": "v4-target-logit-perturbation",
        "model_id": discovery["identity"]["model_id"],
        "same_model": True,
        "split": "held_out",
        "causal_intervention": True,
        "intervention": "ablate",
        "target_metric_delta": -float(selected_validation["target_invalid_margin_reduction"]),
        "target_improvement_direction": "decrease",
        "minimum_target_effect": float(validation["frozen_thresholds"]["minimum_target_reduction"]),
        "guardrail_delta": float(selected_validation["target_valid_pose_logprob_effect"]),
        "max_guardrail_regression": abs(
            float(validation["frozen_thresholds"]["minimum_valid_pose_logprob_effect"])
        ),
        "matched_random_control_delta": -float(
            selected_validation["control_invalid_margin_reduction"]
        ),
        "control_margin": float(validation["frozen_thresholds"]["minimum_control_advantage"]),
        "module": "self_attn.k_proj layers 18-26",
        "component_id": selected_group_id,
        "artifact_uri": str(args.validation_raw.resolve()),
        "sha256": sha256_file(args.validation_raw),
    }
    hook_path.write_text(json.dumps(hook, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "bonsai8b_full_vpd_artifact_v1",
        "status": "completed_controlled_null",
        "model_id": discovery["identity"]["model_id"],
        "revision": discovery["identity"]["revision"],
        "decomposition": {
            "kind": "LoRA rank-one outer-product components",
            "module_count": 252,
            "rank_per_module": 16,
            "component_count": 4032,
            "component_formula": ("scale * B[:, rank_index] outer A[rank_index, :]"),
        },
        "adapter_weight": {
            "path": str(args.adapter_weight.resolve()),
            "bytes": args.adapter_weight.stat().st_size,
            "sha256": sha256_file(args.adapter_weight),
        },
        "base_projection_inventory": projection,
        "component_inventory": {
            "path": str(args.component_inventory.resolve()),
            "rows": 4032,
            "sha256": sha256_file(args.component_inventory),
        },
        "component_evidence": {
            "path": str(component_output.resolve()),
            "rows": 4032,
            "sha256": sha256_file(component_output),
            "selected_component_rows": 144,
            "evidence_granularity": (
                "each row locates an exact weight component but causal effects are "
                "joint family, depth, and family-by-depth ablations"
            ),
        },
        "hierarchy": {
            "group_plan_path": str(args.group_plan.resolve()),
            "group_plan_sha256": sha256_file(args.group_plan),
            "projection_family_groups": 7,
            "depth_band_groups": 4,
            "family_x_depth_cells": 28,
            "cell_partition_component_count": 4032,
        },
        "evidence": {
            "discovery_summary_path": str(args.discovery_summary.resolve()),
            "discovery_summary_sha256": sha256_file(args.discovery_summary),
            "validation_summary_path": str(args.validation_summary.resolve()),
            "validation_summary_sha256": sha256_file(args.validation_summary),
            "selected_group_id": selected_group_id,
            "selected_target_component_set_sha256": validation["groups"][0][
                "target_component_set_sha256"
            ],
            "discovery_gate_passed": discovery["stage_gate_passed"],
            "internal_validation_gate_passed": validation["stage_gate_passed"],
            "task3_exposed": False,
            "task3_exposure_permitted": validation["task3_exposure_permitted"],
        },
        "mechanistic_hook": {
            "path": str(hook_path.resolve()),
            "sha256": sha256_file(hook_path),
            "split_label": "held_out_internal_but_not_final_task3",
        },
        "conclusion": (
            "The weight decomposition and joint-ablation evidence artifacts are "
            "complete. The selected architecture cell did not meet the frozen "
            "discovery or internal-validation causal thresholds, so task-3 remains "
            "sealed and no positive authorization-specific mechanism is claimed."
        ),
        "emitter": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    manifest_path = args.output_dir / "vpd_artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
