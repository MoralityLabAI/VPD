from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.admit_compartment_sae_training_release import (
    PLAN_SCHEMA,
    RECEIPT_SCHEMA,
    RECEIPT_STATUS,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.plan_compartment_sae_training import (
    validate_training_plan_structure,
)


def _relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _validate_addressed(document: dict[str, Any], id_field: str, label: str) -> str:
    body = dict(document)
    identifier = str(body.pop(id_field, ""))
    if content_sha256(body) != identifier:
        raise ValueError(f"{label} content address mismatch")
    return identifier


def build_release_gate_plan(
    *,
    training_plan_path: Path,
    protocol_config_path: Path,
    partition_audit_gate_plan_path: Path,
    partition_audit_pack_dir: Path,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    training_plan = json.loads(training_plan_path.read_text(encoding="utf-8"))
    validate_training_plan_structure(training_plan)
    training_plan_id = _validate_addressed(
        training_plan, "training_plan_id", "compartment SAE training blueprint"
    )
    protocol = json.loads(protocol_config_path.read_text(encoding="utf-8"))
    partition_gate = json.loads(
        partition_audit_gate_plan_path.read_text(encoding="utf-8")
    )
    partition_gate_id = _validate_addressed(
        partition_gate, "gate_plan_id", "partition audit gate plan"
    )
    pack_manifest_path = partition_audit_pack_dir / "artifact_manifest.json"
    pack_manifest = json.loads(pack_manifest_path.read_text(encoding="utf-8"))
    pack_id = _validate_addressed(
        pack_manifest, "pack_id", "partition audit pack manifest"
    )
    protocol_gate = protocol.get("sae_training", {}).get(
        "partition_audit_gate", {}
    )
    if (
        protocol_gate.get("gate_plan_id") != partition_gate_id
        or protocol_gate.get("category_specific_training_requires_pass") is not True
        or protocol_gate.get("execution_authorized") is not False
    ):
        raise ValueError("protocol and partition audit gate differ")
    replica_contract = protocol.get("sae_training", {}).get(
        "replica_training_plan", {}
    )
    if (
        replica_contract.get("path") != _relative(training_plan_path, project_root)
        or replica_contract.get("schema") != training_plan.get("schema")
        or int(replica_contract.get("compartments", -1)) != 9
        or replica_contract.get("status") != training_plan.get("status")
    ):
        raise ValueError("protocol and compartment SAE training blueprint differ")
    frozen_gate = partition_gate.get("frozen_gate", {})
    if (
        frozen_gate.get("position")
        != "after_category_bundle_freeze_before_category_specific_sae_training"
        or frozen_gate.get("failure_effect")
        != "reject_candidate_and_prohibit_category_specific_sae_training"
        or frozen_gate.get("next_candidate_requires_new_pack_version_and_seed")
        is not True
        or frozen_gate.get("threshold_or_cluster_selection_allowed") is not False
    ):
        raise ValueError("partition audit frozen gate contract differs")

    implementation_path = (
        project_root
        / "experiments/authz_mechinterp/admit_compartment_sae_training_release.py"
    )
    receipt_path = "training_release/training_release_receipt.json"
    release_job_body = {
        "schema": "sch.vpd-compartment-sae-training-release-job.v1",
        "output": receipt_path,
        "arguments": [
            "-m",
            "experiments.authz_mechinterp.admit_compartment_sae_training_release",
            "--release-gate-plan",
            "artifacts/vpd_compartment_sae_training_release_gate_v1/plan.json",
            "--training-plan",
            _relative(training_plan_path, project_root),
            "--protocol",
            _relative(protocol_config_path, project_root),
            "--bundle-dir",
            "${FROZEN_CATEGORY_BUNDLE_DIR}",
            "--hierarchy-plan",
            "${COMPONENT_INVENTORY}",
            "--discovery-effects",
            "${DISCOVERY_EFFECTS}",
            "--discovery-merge-manifest",
            "${DISCOVERY_MERGE_MANIFEST}",
            "--audit-merge-dir",
            "${PARTITION_AUDIT_MERGE_DIR}",
            "--partition-audit-pack",
            _relative(partition_audit_pack_dir, project_root),
            "--partition-audit-gate",
            _relative(partition_audit_gate_plan_path, project_root),
            "--partition-audit-report",
            "${PARTITION_AUDIT_REPORT}",
            "--output",
            receipt_path,
        ],
        "runtime_placeholders": [
            "FROZEN_CATEGORY_BUNDLE_DIR",
            "COMPONENT_INVENTORY",
            "DISCOVERY_EFFECTS",
            "DISCOVERY_MERGE_MANIFEST",
            "PARTITION_AUDIT_MERGE_DIR",
            "PARTITION_AUDIT_REPORT",
        ],
        "required_output_schema": RECEIPT_SCHEMA,
        "required_output_status": RECEIPT_STATUS,
        "execution_authorized": False,
    }
    release_job = {
        **release_job_body,
        "job_id": content_sha256(release_job_body),
    }
    guarded_jobs = [
        *(
            {
                "stage": "base_activation_capture",
                "source_job_id": str(job["job_id"]),
                "role": str(job["role"]),
                "depends_on_training_release_job_id": release_job["job_id"],
                "required_training_release_receipt": receipt_path,
            }
            for job in training_plan["capture_jobs"]
        ),
        *(
            {
                "stage": "compartment_sae_training",
                "source_job_id": str(job["job_id"]),
                "compartment": str(job["compartment"]),
                "initialization_seed": int(job["initialization_seed"]),
                "depends_on_training_release_job_id": release_job["job_id"],
                "required_training_release_receipt": receipt_path,
            }
            for job in training_plan["training_jobs"]
        ),
        {
            "stage": "dictionary_bundle_admission",
            "source_job_id": str(training_plan["admission_job"]["job_id"]),
            "depends_on_training_release_job_id": release_job["job_id"],
            "required_training_release_receipt": receipt_path,
        },
        {
            "stage": "calibration_quality",
            "source_job_id": str(training_plan["quality_job"]["job_id"]),
            "depends_on_training_release_job_id": release_job["job_id"],
            "required_training_release_receipt": receipt_path,
        },
    ]
    body = {
        "schema": PLAN_SCHEMA,
        "status": "locally_verified_blocked_until_frozen_bundle_audit_pass_and_compute_authorization",
        "source": {
            "training_blueprint": {
                "path": _relative(training_plan_path, project_root),
                "training_plan_id": training_plan_id,
                "sha256": sha256_file(training_plan_path),
                "schema": training_plan["schema"],
                "preserved_unmodified": True,
            },
            "protocol": {
                "path": _relative(protocol_config_path, project_root),
                "sha256": sha256_file(protocol_config_path),
                "schema": protocol["schema"],
            },
            "partition_audit_gate": {
                "path": _relative(partition_audit_gate_plan_path, project_root),
                "gate_plan_id": partition_gate_id,
                "sha256": sha256_file(partition_audit_gate_plan_path),
            },
            "fresh_partition_audit_pack": {
                "path": _relative(partition_audit_pack_dir, project_root),
                "pack_id": pack_id,
                "manifest_sha256": sha256_file(pack_manifest_path),
            },
            "implementation": {
                "path": _relative(implementation_path, project_root),
                "sha256": sha256_file(implementation_path),
            },
            "planner_sha256": sha256_file(Path(__file__)),
        },
        "release_job": release_job,
        "guarded_execution": {
            "mode": "all_blueprint_jobs_require_one_reproducible_training_release_receipt",
            "guarded_jobs": guarded_jobs,
            "guarded_job_count": len(guarded_jobs),
            "base_capture_jobs": len(training_plan["capture_jobs"]),
            "sae_training_jobs": len(training_plan["training_jobs"]),
            "dictionary_bundle_admission_jobs": 1,
            "calibration_quality_jobs": 1,
            "direct_execution_of_predecessor_blueprint_prohibited": True,
        },
        "candidate_policy": {
            "candidate_identity": "category_aware_bundle_manifest_sha256",
            "candidate_must_be_frozen_before_audit": True,
            "specific_failure_rejects_entire_candidate": True,
            "same_pack_failure_may_tune_candidate": False,
            "next_candidate_requires_new_bundle_manifest": True,
            "next_candidate_requires_new_pack_version_and_seed": True,
            "shared_to_specific_promotion_on_audit": False,
        },
        "required_receipt_joins": [
            "training_blueprint_id_and_sha256",
            "protocol_sha256",
            "category_bundle_manifest_sha256_and_exact_component_census",
            "partition_audit_id_and_report_sha256",
            "fresh_partition_audit_pack_id_and_manifest_sha256",
            "partition_audit_gate_plan_id_and_sha256",
            "bundle_audit_category_names",
            "bundle_audit_component_id_set_sha256",
            "audit_measurement_model_revision_adapter_and_effect_space",
        ],
        "invariants": {
            "partition_audit_pass_required_before_base_capture": True,
            "partition_audit_pass_required_before_category_specific_sae_training": True,
            "partition_audit_pass_required_before_shared_sae_training": True,
            "every_blueprint_job_guarded": True,
            "receipt_content_addressed_and_reproducible": True,
            "fresh_heldout_pack_bound": True,
            "final_evaluation_not_consumed": True,
            "v3_training_blueprint_preserved_as_predecessor_evidence": True,
        },
        "stage_order": [
            "freeze_versioned_category_aware_candidate",
            "capture_and_merge_fresh_signed_partition_audit_effects",
            "reproduce_zero_failure_partition_audit",
            "emit_training_release_receipt",
            "obtain_explicit_compute_authorization",
            "capture_base_prompt_activations",
            "train_compartment_sae_replicas",
            "admit_dictionary_bundle",
            "run_calibration_quality_gate",
        ],
        "scientific_boundary": {
            "model_inference_performed": False,
            "sae_training_performed": False,
            "weights_loaded_or_mutated": False,
            "partition_audit_report_present": False,
            "training_release_receipt_present": False,
            "execution_authorized": False,
            "v7_negative_release_decision_unchanged": True,
        },
    }
    return {**body, "release_gate_plan_id": content_sha256(body)}


def verify_release_gate_plan(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    plan_id = _validate_addressed(
        observed, "release_gate_plan_id", "training release gate plan"
    )
    expected = build_release_gate_plan(**kwargs)
    if observed != expected:
        raise ValueError("training release gate plan is not reproducible")
    guarded = observed["guarded_execution"]
    if guarded["guarded_job_count"] != 22 or not all(
        row["depends_on_training_release_job_id"]
        == observed["release_job"]["job_id"]
        for row in guarded["guarded_jobs"]
    ):
        raise ValueError("training release gate does not guard every blueprint job")
    return {
        "schema": "sch.vpd-compartment-sae-training-release-gate-plan-verification.v1",
        "status": "pass",
        "release_gate_plan_id": plan_id,
        "guarded_jobs": guarded["guarded_job_count"],
        "training_jobs": guarded["sae_training_jobs"],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan the post-audit release gate for compartment SAE training."
    )
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--partition-audit-gate", type=Path, required=True)
    parser.add_argument("--partition-audit-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    kwargs = {
        "training_plan_path": args.training_plan,
        "protocol_config_path": args.protocol,
        "partition_audit_gate_plan_path": args.partition_audit_gate,
        "partition_audit_pack_dir": args.partition_audit_pack,
    }
    if args.verify:
        result = verify_release_gate_plan(args.output, **kwargs)
    else:
        result = build_release_gate_plan(**kwargs)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
