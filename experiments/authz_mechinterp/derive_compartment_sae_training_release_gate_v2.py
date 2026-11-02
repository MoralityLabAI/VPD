from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.admit_compartment_sae_training_release_v2 import (
    PLAN_SCHEMA,
    RECEIPT_SCHEMA,
    RECEIPT_STATUS,
    load_release_gate_plan_v2,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_compartment_sae_training_plan_v4 import (
    canonical_inputs as plan_v4_inputs,
    verify_training_plan_v4,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v8 import (
    canonical_inputs as protocol_v8_inputs,
    verify_protocol as verify_protocol_v8,
)
from experiments.authz_mechinterp.plan_compartment_sae_training_release_gate import (
    verify_release_gate_plan as verify_release_gate_v1,
)


GATE_RELATIVE = "artifacts/vpd_compartment_sae_training_release_gate_v2/plan.json"
PLAN_RELATIVE = "artifacts/vpd_compartment_sae_training_plan_v4/plan.json"
PROTOCOL_V9_RELATIVE = "configs/evals/vpd_activation_category_compartments_v9.json"
RECEIPT_RELATIVE = "training_release/training_release_receipt_v2.json"
COMPUTE_AUTHORIZATION_RELATIVE = (
    "training_release/explicit_compute_authorization.json"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("training release gate V2 source escapes the project")
    return resolved.relative_to(root.resolve()).as_posix()


def _job(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "job_id": content_sha256(body)}


def build_release_gate_v2(
    *,
    predecessor_gate_path: Path,
    training_plan_path: Path,
    protocol_v8_path: Path,
    execution_blueprint_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = verify_release_gate_v1(
        predecessor_gate_path,
        training_plan_path=project_root
        / "artifacts/vpd_compartment_sae_training_plan_v3/plan.json",
        protocol_config_path=project_root
        / "configs/evals/vpd_activation_category_compartments_v6.json",
        partition_audit_gate_plan_path=project_root
        / "artifacts/vpd_partition_audit_gate_v2/plan.json",
        partition_audit_pack_dir=project_root
        / "artifacts/vpd_partition_audit_pack_v1",
    )
    plan_verification = verify_training_plan_v4(
        training_plan_path, **plan_v4_inputs(project_root)
    )
    protocol_verification = verify_protocol_v8(
        protocol_v8_path, **protocol_v8_inputs(project_root)
    )
    predecessor = json.loads(predecessor_gate_path.read_text(encoding="utf-8"))
    plan = json.loads(training_plan_path.read_text(encoding="utf-8"))
    blueprint = json.loads(execution_blueprint_path.read_text(encoding="utf-8"))
    release_runner = project_root / (
        "experiments/authz_mechinterp/admit_compartment_sae_training_release_v2.py"
    )
    planner = Path(__file__)
    result = copy.deepcopy(predecessor)
    result["schema"] = PLAN_SCHEMA
    result["status"] = (
        "locally_verified_blocked_until_bundle_audit_compute_authorization_v2"
    )
    result["source"]["predecessor_release_gate_v1"] = {
        "path": _relative(predecessor_gate_path, project_root),
        "sha256": sha256_file(predecessor_gate_path),
        "release_gate_plan_id": str(predecessor["release_gate_plan_id"]),
        "verification": predecessor_verification,
        "preserved_unmodified": True,
    }
    result["source"]["predecessor_protocol_v6"] = copy.deepcopy(
        predecessor["source"]["protocol"]
    )
    result["source"]["research_protocol_v8"] = {
        "path": _relative(protocol_v8_path, project_root),
        "sha256": sha256_file(protocol_v8_path),
        "verification": protocol_verification,
    }
    result["source"]["training_blueprint"] = {
        "path": _relative(training_plan_path, project_root),
        "schema": str(plan["schema"]),
        "sha256": sha256_file(training_plan_path),
        "training_plan_id": str(plan["training_plan_id"]),
        "verification": plan_verification,
    }
    result["source"]["execution_blueprint"] = {
        "path": _relative(execution_blueprint_path, project_root),
        "sha256": sha256_file(execution_blueprint_path),
        "execution_blueprint_id": str(blueprint["execution_blueprint_id"]),
    }
    result["source"]["implementation"] = {
        "path": _relative(release_runner, project_root),
        "sha256": sha256_file(release_runner),
    }
    result["source"]["planner_sha256"] = sha256_file(planner)

    release_body = {
        "schema": "sch.vpd-compartment-sae-training-release-job.v2",
        "arguments": [
            "-m",
            "experiments.authz_mechinterp.admit_compartment_sae_training_release_v2",
            "--release-gate-plan",
            GATE_RELATIVE,
            "--training-plan",
            PLAN_RELATIVE,
            "--protocol",
            PROTOCOL_V9_RELATIVE,
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
            "artifacts/vpd_partition_audit_pack_v1",
            "--partition-audit-gate",
            "artifacts/vpd_partition_audit_gate_v2/plan.json",
            "--partition-audit-report",
            "${PARTITION_AUDIT_REPORT}",
            "--output",
            RECEIPT_RELATIVE,
        ],
        "output": RECEIPT_RELATIVE,
        "required_output_schema": RECEIPT_SCHEMA,
        "required_output_status": RECEIPT_STATUS,
        "runtime_placeholders": list(
            predecessor["release_job"]["runtime_placeholders"]
        ),
        "execution_authorized": False,
        "compute_authorized": False,
        "explicit_compute_authorization_output": COMPUTE_AUTHORIZATION_RELATIVE,
        "repository_generates_explicit_compute_authorization": False,
    }
    release_job = _job(release_body)
    result["release_job"] = release_job

    guarded_jobs: list[dict[str, Any]] = []
    for job in plan["capture_jobs"]:
        guarded_jobs.append(
            {
                "stage": "base_activation_capture",
                "role": str(job["role"]),
                "source_job_id": str(job["job_id"]),
                "predecessor_job_id": str(job["predecessor_job_id"]),
                "depends_on_training_release_job_id": str(release_job["job_id"]),
                "required_training_release_receipt": RECEIPT_RELATIVE,
                "required_explicit_compute_authorization": (
                    COMPUTE_AUTHORIZATION_RELATIVE
                ),
            }
        )
    for job in plan["training_jobs"]:
        guarded_jobs.append(
            {
                "stage": "compartment_sae_training",
                "compartment": str(job["compartment"]),
                "initialization_seed": int(job["initialization_seed"]),
                "source_job_id": str(job["job_id"]),
                "predecessor_job_id": str(job["predecessor_job_id"]),
                "depends_on_training_release_job_id": str(release_job["job_id"]),
                "required_training_release_receipt": RECEIPT_RELATIVE,
                "required_explicit_compute_authorization": (
                    COMPUTE_AUTHORIZATION_RELATIVE
                ),
                "required_dictionary_manifest_schema": (
                    "sch.vpd-fixed-sae-dictionary-manifest.v2"
                ),
            }
        )
    for stage, job in (
        ("dictionary_bundle_admission", plan["admission_job"]),
        ("calibration_quality", plan["quality_job"]),
    ):
        guarded_jobs.append(
            {
                "stage": stage,
                "source_job_id": str(job["job_id"]),
                "predecessor_job_id": str(job["predecessor_job_id"]),
                "depends_on_training_release_job_id": str(release_job["job_id"]),
                "required_training_release_receipt": RECEIPT_RELATIVE,
                "required_explicit_compute_authorization": (
                    COMPUTE_AUTHORIZATION_RELATIVE
                ),
            }
        )
    if len(guarded_jobs) != 22:
        raise ValueError("release gate V2 did not cover 22 jobs")
    result["guarded_execution"] = {
        "mode": "scientific_release_plus_external_explicit_compute_authorization",
        "guarded_jobs": guarded_jobs,
        "guarded_job_count": len(guarded_jobs),
        "base_capture_jobs": 2,
        "sae_training_jobs": 18,
        "dictionary_bundle_admission_jobs": 1,
        "calibration_quality_jobs": 1,
        "direct_execution_of_predecessor_blueprint_prohibited": True,
        "release_receipt_alone_cannot_start_any_job": True,
        "explicit_compute_authorization_exact_job_scope_required": True,
    }
    result["invariants"].update(
        {
            "v4_training_blueprint_required": True,
            "manifest_v2_required_for_all_dictionaries": True,
            "exact_sample_contract_required_for_all_dictionaries": True,
            "batch_size_and_learning_rate_bound_for_all_dictionaries": True,
            "all_22_jobs_require_release_and_compute_authorization": True,
            "repository_does_not_generate_compute_authorization": True,
        }
    )
    result["stage_order"] = [
        "freeze_versioned_category_aware_candidate",
        "capture_and_merge_fresh_signed_partition_audit_effects",
        "reproduce_zero_failure_partition_audit",
        "emit_training_release_receipt_v2_compute_not_authorized",
        "obtain_external_explicit_compute_authorization",
        "capture_base_prompt_activations_v2",
        "train_compartment_sae_replicas_manifest_v2",
        "admit_dictionary_bundle_v2",
        "run_calibration_quality_gate_v2",
    ]
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["compute_authorized"] = False
    result["scientific_boundary"]["explicit_compute_authorization_present"] = False
    result_body = {
        key: value for key, value in result.items() if key != "release_gate_plan_id"
    }
    result["release_gate_plan_id"] = content_sha256(result_body)
    return result


def verify_release_gate_v2(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = load_release_gate_plan_v2(path)
    expected = build_release_gate_v2(**kwargs)
    if observed != expected:
        raise ValueError("compartment SAE training release gate V2 is not reproducible")
    return {
        "schema": "sch.vpd-compartment-sae-training-release-gate-verification.v2",
        "status": "pass",
        "release_gate_plan_id": str(observed["release_gate_plan_id"]),
        "release_gate_plan_sha256": sha256_file(path),
        "guarded_jobs": len(observed["guarded_execution"]["guarded_jobs"]),
        "execution_authorized": False,
        "compute_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_gate_path": project_root
        / "artifacts/vpd_compartment_sae_training_release_gate_v1/plan.json",
        "training_plan_path": project_root / PLAN_RELATIVE,
        "protocol_v8_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v8.json",
        "execution_blueprint_path": project_root
        / "artifacts/vpd_compartment_sae_training_execution_blueprint_v2/blueprint.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive or verify the manifest-V2 SAE release gate."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None:
        result = verify_release_gate_v2(args.verify, **inputs)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_release_gate_v2(**inputs)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
