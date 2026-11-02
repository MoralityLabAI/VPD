from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v5 import (
    verify_protocol,
)
from experiments.authz_mechinterp.verify_unified_vpd_compartment_pipeline import (
    verify_receipt as verify_predecessor_receipt,
)


SCHEMA = "sch.vpd-unified-compartment-pipeline-receipt.v3"


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("V5 unified lineage source escapes the project")
    return resolved.relative_to(root.resolve()).as_posix()


def build_receipt(
    *,
    protocol_path: Path,
    predecessor_protocol_path: Path,
    predecessor_receipt_path: Path,
    predecessor_receipt_inputs: dict[str, Path],
    partition_audit_pack_dir: Path,
    partition_audit_benchmark_path: Path,
    partition_audit_gate_plan_path: Path,
    bundle_dir: Path,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    predecessor_verification = verify_predecessor_receipt(
        predecessor_receipt_path, **predecessor_receipt_inputs
    )
    protocol_verification = verify_protocol(
        protocol_path,
        predecessor_path=predecessor_protocol_path,
        pack_dir=partition_audit_pack_dir,
        benchmark_path=partition_audit_benchmark_path,
        gate_plan_path=partition_audit_gate_plan_path,
        bundle_dir=bundle_dir,
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    gate = json.loads(partition_audit_gate_plan_path.read_text(encoding="utf-8"))
    predecessor = json.loads(predecessor_receipt_path.read_text(encoding="utf-8"))
    audit = protocol["component_compartments"]["post_freeze_partition_audit"]
    if audit["gate_plan_id"] != gate["gate_plan_id"]:
        raise ValueError("V5 protocol and partition-audit gate plan IDs differ")
    if audit["fresh_pack_id"] != protocol["data_pack"]["partition_audit_pack"][
        "pack_id"
    ]:
        raise ValueError("V5 protocol partition-audit pack IDs differ")
    if predecessor_verification["pipeline_receipt_id"] != predecessor[
        "pipeline_receipt_id"
    ]:
        raise ValueError("V5 predecessor receipt identity differs")
    body = {
        "schema": SCHEMA,
        "status": "pass_local_lineage_partition_audit_capture_adapter_required",
        "source": {
            "predecessor_unified_receipt": {
                "path": _relative(predecessor_receipt_path, project_root),
                "sha256": sha256_file(predecessor_receipt_path),
                "pipeline_receipt_id": predecessor_verification[
                    "pipeline_receipt_id"
                ],
                "verification": predecessor_verification,
            },
            "protocol_v5": {
                "path": _relative(protocol_path, project_root),
                "sha256": sha256_file(protocol_path),
                "verification": protocol_verification,
            },
            "partition_audit_pack": {
                "path": _relative(partition_audit_pack_dir, project_root),
                "manifest_sha256": sha256_file(
                    partition_audit_pack_dir / "artifact_manifest.json"
                ),
                "pack_id": audit["fresh_pack_id"],
            },
            "partition_audit_benchmark": {
                "path": _relative(partition_audit_benchmark_path, project_root),
                "sha256": sha256_file(partition_audit_benchmark_path),
                "benchmark_id": audit["synthetic_benchmark_id"],
            },
            "partition_audit_gate_plan": {
                "path": _relative(partition_audit_gate_plan_path, project_root),
                "sha256": sha256_file(partition_audit_gate_plan_path),
                "gate_plan_id": audit["gate_plan_id"],
            },
        },
        "lineage_amendment": {
            "predecessor_five_role_split_unchanged": True,
            "predecessor_effect_campaign_unchanged": True,
            "predecessor_sae_plan_preserved_as_planned_work": True,
            "new_partition_audit_pack_is_separate_from_five_role_split": True,
            "category_specific_sae_training_now_requires_partition_audit_pass": True,
            "final_evaluation_remains_untouched": True,
            "capture_adapter_status": gate["implementation"]["capture_adapter"][
                "status"
            ],
        },
        "resources": {
            "predecessor_effect_episode_forward_equivalents": predecessor[
                "resources"
            ]["effect_episode_forward_equivalents"],
            "partition_audit_episode_forward_equivalents": gate["resources"][
                "episode_forward_equivalents"
            ],
            "partition_audit_work_units": len(gate["capture_work_units"]),
        },
        "invariants": {
            "predecessor_receipt_reverified": True,
            "v5_protocol_reverified": True,
            "partition_audit_gate_bound": True,
            "no_same_pack_tuning_after_audit_failure": True,
            "final_evaluation_not_consumed": True,
        },
        "scientific_boundary": {
            "model_inference_performed": False,
            "sae_training_performed": False,
            "weights_loaded_or_mutated": False,
            "empirical_partition_audited": False,
            "execution_authorized": False,
            "v7_negative_release_decision_unchanged": True,
        },
    }
    return {**body, "pipeline_receipt_id": content_sha256(body)}


def verify_receipt(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = copy.deepcopy(observed)
    receipt_id = str(body.pop("pipeline_receipt_id", ""))
    if content_sha256(body) != receipt_id:
        raise ValueError("V5 unified pipeline receipt content address mismatch")
    if build_receipt(**kwargs) != observed:
        raise ValueError("V5 unified pipeline receipt is not reproducible")
    return {
        "schema": "sch.vpd-unified-compartment-pipeline-verification.v3",
        "status": "pass",
        "pipeline_receipt_id": receipt_id,
        "partition_audit_work_units": observed["resources"][
            "partition_audit_work_units"
        ],
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    artifacts = project_root / "artifacts"
    configs = project_root / "configs" / "evals"
    predecessor_inputs = {
        "protocol_path": configs / "vpd_activation_category_compartments_v4.json",
        "predecessor_protocol_path": configs
        / "vpd_activation_category_compartments_v3.json",
        "effect_plan_path": artifacts / "vpd_effect_capture_campaign_v3" / "plan.json",
        "training_plan_path": artifacts
        / "vpd_compartment_sae_training_plan_v3"
        / "plan.json",
        "bundle_dir": artifacts / "bonsai8b_authz_v8" / "vpd_sae_clusters",
        "split_dir": artifacts / "vpd_activation_category_capacity_split_v4",
        "pack_dir": artifacts / "vpd_activation_category_capacity_pack_v3",
        "hierarchy_plan_path": artifacts
        / "bonsai8b_authz_v5"
        / "hierarchy_group_plan.json",
        "capacity_plan_path": artifacts
        / "vpd_compartment_sae_capacity_v2"
        / "plan.json",
        "capture_resource_plan_path": artifacts
        / "vpd_compartment_capture_resources_v1"
        / "plan.json",
        "quality_benchmark_path": artifacts
        / "vpd_compartment_sae_quality_benchmark_v1"
        / "report.json",
    }
    return {
        "protocol_path": configs / "vpd_activation_category_compartments_v5.json",
        "predecessor_protocol_path": configs
        / "vpd_activation_category_compartments_v4.json",
        "predecessor_receipt_path": artifacts
        / "vpd_unified_compartment_pipeline_v2"
        / "receipt.json",
        "predecessor_receipt_inputs": predecessor_inputs,
        "partition_audit_pack_dir": artifacts / "vpd_partition_audit_pack_v1",
        "partition_audit_benchmark_path": artifacts
        / "vpd_frozen_partition_audit_benchmark_v1"
        / "report.json",
        "partition_audit_gate_plan_path": artifacts
        / "vpd_partition_audit_gate_v1"
        / "plan.json",
        "bundle_dir": artifacts / "bonsai8b_authz_v8" / "vpd_sae_clusters",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the V5 partition-audit lineage amendment."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    inputs = canonical_inputs(project_root)
    if args.verify is not None:
        result = verify_receipt(args.verify, **inputs)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_receipt(**inputs)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
