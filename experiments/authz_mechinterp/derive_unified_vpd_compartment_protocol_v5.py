from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_frozen_vpd_partition_audit import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.finalize_partition_audit_pack_v1 import (
    verify_partition_audit_manifest,
)
from experiments.authz_mechinterp.plan_vpd_partition_audit_gate import (
    verify_gate_plan,
)


SCHEMA = "sch.vpd-activation-category-protocol.v5"


def build_protocol(
    *,
    predecessor_path: Path,
    pack_dir: Path,
    benchmark_path: Path,
    gate_plan_path: Path,
    bundle_dir: Path,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if predecessor.get("schema") != "sch.vpd-activation-category-protocol.v4":
        raise ValueError("partition-audit protocol predecessor is not V4")
    pack_verification = verify_partition_audit_manifest(pack_dir)
    benchmark_verification = verify_benchmark(benchmark_path)
    gate_verification = verify_gate_plan(
        gate_plan_path,
        predecessor_protocol_path=predecessor_path,
        pack_dir=pack_dir,
        benchmark_path=benchmark_path,
        bundle_dir=bundle_dir,
    )
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = predecessor_path.resolve().relative_to(
        project_root.resolve()
    ).as_posix()
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["post_freeze_partition_audit"] = {
        "schema": "sch.vpd-frozen-category-partition-audit.v1",
        "position": "after_category_bundle_freeze_before_category_specific_sae_training",
        "mode": "full_component_reject_only_generalization_audit",
        "fresh_pack_path": pack_dir.resolve().relative_to(
            project_root.resolve()
        ).as_posix(),
        "fresh_pack_id": pack_verification["pack_id"],
        "synthetic_benchmark_path": benchmark_path.resolve().relative_to(
            project_root.resolve()
        ).as_posix(),
        "synthetic_benchmark_id": benchmark_verification["benchmark_id"],
        "gate_plan_path": gate_plan_path.resolve().relative_to(
            project_root.resolve()
        ).as_posix(),
        "gate_plan_id": gate_verification["gate_plan_id"],
        "same_specific_category_required": True,
        "minimum_signed_direction_cosine": 0.5,
        "minimum_dominant_strength_ratio": 0.5,
        "same_stratified_robustness_gates_required": True,
        "zero_specific_failures_required": True,
        "audit_may_select_thresholds_clusters_or_components": False,
        "audit_may_promote_shared_to_specific": False,
        "failure_may_not_tune_candidate_on_same_pack": True,
        "next_candidate_requires_new_pack_version_and_seed": True,
        "final_evaluation_role_consumed": False,
        "required_for_category_specific_sae_training": True,
    }
    result["data_pack"]["partition_audit_pack"] = {
        "separate_from_five_role_capacity_pack": True,
        "path": pack_dir.resolve().relative_to(project_root.resolve()).as_posix(),
        "pack_id": pack_verification["pack_id"],
        "rows": pack_verification["rows"],
        "pairs": pack_verification["pairs"],
        "semantic_template_groups": pack_verification["template_groups"],
        "identity_message_and_template_disjointness_verified": True,
        "role": "partition_audit",
        "selection_use": "none_reject_only",
    }
    result["effect_capture"]["partition_audit_campaign"] = {
        "gate_plan_path": gate_plan_path.resolve().relative_to(
            project_root.resolve()
        ).as_posix(),
        "gate_plan_id": gate_verification["gate_plan_id"],
        "episode_forward_equivalents": gate_verification[
            "episode_forward_equivalents"
        ],
        "work_units": gate_verification["work_units"],
        "capture_adapter_status": "required_not_implemented",
        "execution_authorized": False,
    }
    result["sae_training"]["partition_audit_gate"] = {
        "category_specific_training_requires_pass": True,
        "shared_only_training_not_promoted_by_audit": True,
        "same_pack_failure_cannot_be_used_for_tuning": True,
        "gate_plan_id": gate_verification["gate_plan_id"],
        "current_status": "not_run_no_empirical_category_bundle",
        "execution_authorized": False,
    }
    result["unified_lineage"]["post_freeze_partition_audit_required"] = True
    result["unified_lineage"]["final_evaluation_untouched_by_partition_audit"] = True
    result["unified_lineage"]["model_inference_performed"] = False
    result["unified_lineage"]["execution_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = build_protocol(**kwargs)
    if observed != expected:
        raise ValueError("V5 VPD partition-audit protocol is not reproducible")
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v5",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "predecessor_sha256": sha256_file(kwargs["predecessor_path"]),
        "post_freeze_partition_audit_required": True,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive or verify the post-freeze partition-audit VPD protocol V5."
    )
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--gate-plan", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = {
        "predecessor_path": args.predecessor,
        "pack_dir": args.pack_dir,
        "benchmark_path": args.benchmark,
        "gate_plan_path": args.gate_plan,
        "bundle_dir": args.bundle_dir,
    }
    if args.verify is not None:
        result = verify_protocol(args.verify, **inputs)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_protocol(**inputs)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
