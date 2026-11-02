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
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v6 import (
    canonical_inputs as protocol_v6_inputs,
    verify_protocol as verify_protocol_v6,
)
from experiments.authz_mechinterp.derive_vpd_partition_audit_gate_plan_v2 import (
    verify_gate_plan_v2,
)
from experiments.authz_mechinterp.verify_unified_vpd_compartment_pipeline_v3 import (
    canonical_inputs as receipt_v3_inputs,
    verify_receipt as verify_receipt_v3,
)


SCHEMA = "sch.vpd-unified-compartment-pipeline-receipt.v4"


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("V6 unified lineage source escapes the project")
    return resolved.relative_to(root.resolve()).as_posix()


def build_receipt(
    *,
    predecessor_receipt_path: Path,
    protocol_path: Path,
    gate_plan_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = verify_receipt_v3(
        predecessor_receipt_path, **receipt_v3_inputs(project_root)
    )
    protocol_inputs = protocol_v6_inputs(project_root)
    protocol_verification = verify_protocol_v6(protocol_path, **protocol_inputs)
    gate_verification = verify_gate_plan_v2(
        gate_plan_path,
        predecessor_gate_plan_path=protocol_inputs["predecessor_gate_plan_path"],
        predecessor_protocol_path=protocol_inputs["v4_protocol_path"],
        pack_dir=protocol_inputs["pack_dir"],
        benchmark_path=protocol_inputs["benchmark_path"],
        metadata_bundle_dir=protocol_inputs["bundle_dir"],
        discovery_campaign_plan_path=protocol_inputs[
            "discovery_campaign_plan_path"
        ],
    )
    predecessor = json.loads(predecessor_receipt_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_plan_path.read_text(encoding="utf-8"))
    protocol_gate_id = protocol["effect_capture"]["partition_audit_campaign"][
        "gate_plan_id"
    ]
    capture_pipeline_benchmark = protocol["effect_capture"][
        "partition_audit_campaign"
    ]["model_free_capture_pipeline_benchmark"]
    if protocol_gate_id != gate_verification["gate_plan_id"]:
        raise ValueError("research protocol V6 and gate plan V2 differ")
    if predecessor["pipeline_receipt_id"] != predecessor_verification[
        "pipeline_receipt_id"
    ]:
        raise ValueError("unified lineage V3 identity differs")
    body = {
        "schema": SCHEMA,
        "status": "pass_local_capture_path_implemented_not_run",
        "source": {
            "predecessor_unified_receipt_v3": {
                "path": _relative(predecessor_receipt_path, project_root),
                "sha256": sha256_file(predecessor_receipt_path),
                "pipeline_receipt_id": predecessor_verification[
                    "pipeline_receipt_id"
                ],
                "verification": predecessor_verification,
            },
            "research_protocol_v6": {
                "path": _relative(protocol_path, project_root),
                "sha256": sha256_file(protocol_path),
                "verification": protocol_verification,
            },
            "partition_audit_gate_plan_v2": {
                "path": _relative(gate_plan_path, project_root),
                "sha256": sha256_file(gate_plan_path),
                "gate_plan_id": gate_verification["gate_plan_id"],
                "verification": gate_verification,
            },
            "partition_audit_capture_pipeline_benchmark": copy.deepcopy(
                capture_pipeline_benchmark
            ),
            "implementation": copy.deepcopy(gate["implementation"]),
        },
        "lineage_amendment": {
            "predecessor_receipt_preserved": True,
            "research_protocol_v5_preserved": True,
            "research_protocol_v6_is_not_historical_experiment_v6": True,
            "capture_adapter_implemented_and_hash_bound": True,
            "fresh_pack_attestation_occurs_before_signature": True,
            "dedicated_signed_merge_front_end_required": True,
            "production_reject_only_evaluator_required": True,
            "model_free_signed_pipeline_rehearsal_passed": True,
            "cross_archive_model_adapter_and_measurement_identity_required": True,
            "packet_assigned_worker_signing_required": True,
            "verified_discovery_campaign_capture_parameters_bound": True,
            "audit_model_load_dtype_frozen_by_gate_plan": True,
            "signed_receipt_campaign_measurement_contract_required": True,
            "packet_specific_compute_command_hash_required": True,
            "category_specific_sae_training_still_requires_audit_pass": True,
            "final_evaluation_remains_untouched": True,
        },
        "resources": {
            "partition_audit_episode_forward_equivalents": gate["resources"][
                "episode_forward_equivalents"
            ],
            "partition_audit_work_units": len(gate["capture_work_units"]),
            "prepared_capture_job_groups": len(
                gate["execution_plan"]["job_groups"]
            ),
        },
        "invariants": {
            "predecessor_receipt_reverified": True,
            "research_protocol_v6_reverified": True,
            "partition_audit_gate_v2_reverified": True,
            "capture_pipeline_benchmark_reverified_through_protocol": True,
            "cross_archive_measurement_identity_contract_bound": True,
            "worker_scoped_signing_plan_bound": True,
            "discovery_campaign_numeric_capture_contract_bound": True,
            "signed_receipt_measurement_contract_bound_by_protocol": True,
            "implementation_hashes_bound_by_gate_and_protocol": True,
            "no_same_pack_tuning_after_audit_failure": True,
            "final_evaluation_not_consumed": True,
            "execution_not_authorized": True,
        },
        "scientific_boundary": {
            "model_inference_performed": False,
            "sae_training_performed": False,
            "weights_loaded_or_mutated": False,
            "signed_partition_audit_receipts_created": False,
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
        raise ValueError("V6 unified pipeline receipt content address mismatch")
    if build_receipt(**kwargs) != observed:
        raise ValueError("V6 unified pipeline receipt is not reproducible")
    return {
        "schema": "sch.vpd-unified-compartment-pipeline-verification.v4",
        "status": "pass",
        "pipeline_receipt_id": receipt_id,
        "capture_adapter_status": observed["source"]["implementation"][
            "capture_adapter"
        ]["status"],
        "partition_audit_work_units": observed["resources"][
            "partition_audit_work_units"
        ],
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_receipt_path": project_root
        / "artifacts"
        / "vpd_unified_compartment_pipeline_v3"
        / "receipt.json",
        "protocol_path": project_root
        / "configs"
        / "evals"
        / "vpd_activation_category_compartments_v6.json",
        "gate_plan_path": project_root
        / "artifacts"
        / "vpd_partition_audit_gate_v2"
        / "plan.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the research-protocol V6 lineage amendment."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
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
