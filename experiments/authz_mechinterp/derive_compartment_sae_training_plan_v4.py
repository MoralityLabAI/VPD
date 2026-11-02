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
from experiments.authz_mechinterp.derive_compartment_sae_training_execution_blueprint_v2 import (
    validate_blueprint,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v8 import (
    canonical_inputs as protocol_v8_inputs,
    verify_protocol as verify_protocol_v8,
)
from experiments.authz_mechinterp.plan_compartment_sae_training import (
    verify_training_plan as verify_training_plan_v3,
)
from experiments.authz_mechinterp.run_compartment_sae_training_job_v2 import (
    MANIFEST_V2_SCHEMA,
    PLAN_SCHEMA,
    load_training_plan_v2,
)


PLAN_RELATIVE = "artifacts/vpd_compartment_sae_training_plan_v4/plan.json"
BLUEPRINT_RELATIVE = (
    "artifacts/vpd_compartment_sae_training_execution_blueprint_v2/blueprint.json"
)
RELEASE_GATE_RELATIVE = (
    "artifacts/vpd_compartment_sae_training_release_gate_v2/plan.json"
)
RELEASE_RECEIPT = "training_release/training_release_receipt_v2.json"
COMPUTE_AUTHORIZATION = "training_release/explicit_compute_authorization.json"


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("training plan V4 source escapes the project")
    return resolved.relative_to(root.resolve()).as_posix()


def _job(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "job_id": content_sha256(body)}


def _implementation(path: Path, root: Path) -> dict[str, str]:
    return {"path": _relative(path, root), "sha256": sha256_file(path)}


def _replace_argument(arguments: list[str], flag: str, value: str) -> list[str]:
    result = list(arguments)
    positions = [index for index, item in enumerate(result) if item == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(result):
        raise ValueError(f"predecessor argument is missing or duplicated: {flag}")
    result[positions[0] + 1] = value
    return result


def _verify_predecessor(project_root: Path, predecessor_path: Path) -> None:
    verification = verify_training_plan_v3(
        predecessor_path,
        capacity_plan_path=project_root
        / "artifacts/vpd_compartment_sae_capacity_v2/plan.json",
        capture_resource_plan_path=project_root
        / "artifacts/vpd_compartment_capture_resources_v1/plan.json",
        pack_dir=project_root / "artifacts/vpd_activation_category_capacity_pack_v3",
        split_dir=project_root
        / "artifacts/vpd_activation_category_capacity_split_v4",
        protocol_config_path=project_root
        / "configs/evals/vpd_activation_category_compartments_v4.json",
        quality_benchmark_path=project_root
        / "artifacts/vpd_compartment_sae_quality_benchmark_v1/report.json",
    )
    if verification.get("status") != "pass":
        raise ValueError("predecessor training plan V3 did not verify")


def build_training_plan_v4(
    *,
    predecessor_path: Path,
    protocol_path: Path,
    execution_blueprint_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    _verify_predecessor(project_root, predecessor_path)
    verify_protocol_v8(protocol_path, **protocol_v8_inputs(project_root))
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    blueprint = json.loads(execution_blueprint_path.read_text(encoding="utf-8"))
    validate_blueprint(
        blueprint,
        training_plan_path=predecessor_path,
        project_root=project_root,
    )
    runner_paths = {
        "capture_runner_v2": project_root
        / "experiments/authz_mechinterp/run_compartment_sae_capture_job_v2.py",
        "sae_training_runner_v2": project_root
        / "experiments/authz_mechinterp/run_compartment_sae_training_job_v2.py",
        "dictionary_bundle_admission_v2": project_root
        / "experiments/authz_mechinterp/admit_compartment_sae_dictionary_bundle_v2.py",
        "quality_runner_v2": project_root
        / "experiments/authz_mechinterp/run_compartment_sae_quality_v2.py",
        "release_admission_v2": project_root
        / "experiments/authz_mechinterp/admit_compartment_sae_training_release_v2.py",
        "planner_v2": Path(__file__),
    }
    result = copy.deepcopy(predecessor)
    result["schema"] = PLAN_SCHEMA
    result["source"]["predecessor_training_plan"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "training_plan_id": str(predecessor["training_plan_id"]),
        "preserved_unmodified": True,
    }
    result["source"]["research_protocol_v8"] = {
        "path": _relative(protocol_path, project_root),
        "sha256": sha256_file(protocol_path),
    }
    result["source"]["execution_blueprint_path"] = _relative(
        execution_blueprint_path, project_root
    )
    result["source"]["execution_blueprint_sha256"] = sha256_file(
        execution_blueprint_path
    )
    result["source"]["execution_blueprint_id"] = str(
        blueprint["execution_blueprint_id"]
    )
    result["source"]["training_release_gate_v2"] = {
        "path": RELEASE_GATE_RELATIVE,
        "required_before_any_capture_training_admission_or_quality": True,
        "explicit_compute_authorization_also_required": True,
    }
    predecessor_implementation = predecessor["source"]["implementation"]
    implementation = {
        "predecessor_capture_worker": copy.deepcopy(
            predecessor_implementation["capture_worker"]
        ),
        "predecessor_sae_trainer": copy.deepcopy(
            predecessor_implementation["sae_trainer"]
        ),
        "predecessor_dictionary_bundle_admission": copy.deepcopy(
            predecessor_implementation["dictionary_bundle_admission"]
        ),
        "predecessor_quality_runner": copy.deepcopy(
            predecessor_implementation["quality_runner"]
        ),
        # Compatibility aliases used only by the preserved V1 validation core.
        "capture_worker": copy.deepcopy(predecessor_implementation["capture_worker"]),
        "sae_trainer": copy.deepcopy(predecessor_implementation["sae_trainer"]),
        "capture_semantic_verifier": copy.deepcopy(
            predecessor_implementation["capture_semantic_verifier"]
        ),
        "quality_evaluator": copy.deepcopy(
            predecessor_implementation["quality_evaluator"]
        ),
        **{
            name: _implementation(path, project_root)
            for name, path in runner_paths.items()
        },
    }
    result["source"]["implementation"] = implementation
    result["source"]["planner_sha256"] = implementation["planner_v2"]["sha256"]
    result["profile"]["required_dictionary_manifest_schema"] = MANIFEST_V2_SCHEMA
    result["profile"]["execution_blueprint_id"] = str(
        blueprint["execution_blueprint_id"]
    )

    predecessor_capture_jobs = {
        str(row["role"]): row for row in predecessor["capture_jobs"]
    }
    capture_jobs: list[dict[str, Any]] = []
    for role in ("sae_training", "calibration"):
        old = predecessor_capture_jobs[role]
        body = {
            **{
                key: copy.deepcopy(value)
                for key, value in old.items()
                if key not in ("schema", "job_id", "arguments", "runtime_placeholders")
            },
            "schema": "sch.vpd-compartment-sae-capture-job.v2",
            "predecessor_job_id": str(old["job_id"]),
            "capture_arguments": [str(value) for value in old["arguments"]],
            "arguments": [
                "-m",
                "experiments.authz_mechinterp.run_compartment_sae_capture_job_v2",
                "--training-plan",
                PLAN_RELATIVE,
                "--training-release-receipt",
                "${SAE_TRAINING_RELEASE_RECEIPT}",
                "--compute-authorization-receipt",
                "${SAE_COMPUTE_AUTHORIZATION_RECEIPT}",
                "--role",
                role,
                "--model",
                "${FROZEN_BASE_MODEL}",
                "--revision",
                "${FROZEN_BASE_REVISION}",
            ],
            "runtime_placeholders": [
                "SAE_TRAINING_RELEASE_RECEIPT",
                "SAE_COMPUTE_AUTHORIZATION_RECEIPT",
                "FROZEN_BASE_MODEL",
                "FROZEN_BASE_REVISION",
            ],
            "required_capture_execution_receipt": True,
        }
        capture_jobs.append(_job(body))
    capture_by_role = {str(row["role"]): row for row in capture_jobs}

    blueprint_by_job = {
        str(row["planned_job_id"]): row for row in blueprint["jobs"]
    }
    training_jobs: list[dict[str, Any]] = []
    for old in predecessor["training_jobs"]:
        record = blueprint_by_job[str(old["job_id"])]
        body = {
            **{
                key: copy.deepcopy(value)
                for key, value in old.items()
                if key
                not in (
                    "schema",
                    "job_id",
                    "arguments",
                    "runtime_placeholders",
                    "depends_on_capture_job_id",
                )
            },
            "schema": "sch.vpd-compartment-sae-training-job.v2",
            "predecessor_job_id": str(old["job_id"]),
            "depends_on_capture_job_id": capture_by_role["sae_training"]["job_id"],
            "trainer_arguments": [str(value) for value in old["arguments"]],
            "required_manifest_hyperparameters": copy.deepcopy(
                record["required_manifest_hyperparameters"]
            ),
            "required_sample_contract": copy.deepcopy(
                record["required_sample_contract"]
            ),
            "arguments": [
                "-m",
                "experiments.authz_mechinterp.run_compartment_sae_training_job_v2",
                "--training-plan",
                PLAN_RELATIVE,
                "--execution-blueprint",
                BLUEPRINT_RELATIVE,
                "--training-release-receipt",
                "${SAE_TRAINING_RELEASE_RECEIPT}",
                "--compute-authorization-receipt",
                "${SAE_COMPUTE_AUTHORIZATION_RECEIPT}",
                "--compartment",
                str(old["compartment"]),
                "--initialization-seed",
                str(old["initialization_seed"]),
                "--device",
                "${SAE_TRAINING_DEVICE}",
            ],
            "runtime_placeholders": [
                "SAE_TRAINING_RELEASE_RECEIPT",
                "SAE_COMPUTE_AUTHORIZATION_RECEIPT",
                "SAE_TRAINING_DEVICE",
            ],
            "required_dictionary_manifest_schema": MANIFEST_V2_SCHEMA,
        }
        training_jobs.append(_job(body))

    old_admission = predecessor["admission_job"]
    admission_body = {
        **{
            key: copy.deepcopy(value)
            for key, value in old_admission.items()
            if key
            not in (
                "schema",
                "job_id",
                "arguments",
                "depends_on_capture_job_id",
                "depends_on_training_job_ids",
                "required_receipt_status",
            )
        },
        "schema": "sch.vpd-compartment-sae-dictionary-bundle-admission-job.v2",
        "predecessor_job_id": str(old_admission["job_id"]),
        "depends_on_capture_job_id": capture_by_role["sae_training"]["job_id"],
        "depends_on_training_job_ids": sorted(row["job_id"] for row in training_jobs),
        "arguments": [
            "-m",
            "experiments.authz_mechinterp.admit_compartment_sae_dictionary_bundle_v2",
            "--run-dir",
            ".",
            "--project-root",
            ".",
            "--training-plan",
            PLAN_RELATIVE,
            "--execution-blueprint",
            BLUEPRINT_RELATIVE,
            "--training-release-receipt",
            "${SAE_TRAINING_RELEASE_RECEIPT}",
            "--compute-authorization-receipt",
            "${SAE_COMPUTE_AUTHORIZATION_RECEIPT}",
            "--output",
            str(old_admission["output"]),
        ],
        "runtime_placeholders": [
            "SAE_TRAINING_RELEASE_RECEIPT",
            "SAE_COMPUTE_AUTHORIZATION_RECEIPT",
        ],
        "required_receipt_status": "pass_ready_for_calibration_quality_v2",
        "required_dictionary_manifest_schema": MANIFEST_V2_SCHEMA,
    }
    admission_job = _job(admission_body)

    old_quality = predecessor["quality_job"]
    quality_arguments = _replace_argument(
        [str(value) for value in old_quality["arguments"]],
        "--training-plan",
        PLAN_RELATIVE,
    )
    quality_body = {
        **{
            key: copy.deepcopy(value)
            for key, value in old_quality.items()
            if key
            not in (
                "schema",
                "job_id",
                "arguments",
                "depends_on_capture_job_id",
                "depends_on_training_job_ids",
                "depends_on_admission_job_id",
            )
        },
        "schema": "sch.vpd-compartment-sae-quality-job.v2",
        "predecessor_job_id": str(old_quality["job_id"]),
        "depends_on_capture_job_id": capture_by_role["calibration"]["job_id"],
        "depends_on_training_job_ids": sorted(row["job_id"] for row in training_jobs),
        "depends_on_admission_job_id": admission_job["job_id"],
        "quality_arguments": quality_arguments,
        "arguments": [
            "-m",
            "experiments.authz_mechinterp.run_compartment_sae_quality_v2",
            "--training-plan",
            PLAN_RELATIVE,
            "--execution-blueprint",
            BLUEPRINT_RELATIVE,
            "--training-release-receipt",
            "${SAE_TRAINING_RELEASE_RECEIPT}",
            "--compute-authorization-receipt",
            "${SAE_COMPUTE_AUTHORIZATION_RECEIPT}",
        ],
        "runtime_placeholders": [
            "SAE_TRAINING_RELEASE_RECEIPT",
            "SAE_COMPUTE_AUTHORIZATION_RECEIPT",
        ],
        "required_dictionary_manifest_schema": MANIFEST_V2_SCHEMA,
    }
    quality_job = _job(quality_body)

    result["capture_jobs"] = capture_jobs
    result["training_jobs"] = training_jobs
    result["admission_job"] = admission_job
    result["quality_job"] = quality_job
    result["execution_contract_v2"] = {
        "all_22_jobs_require_scientific_release_receipt": True,
        "all_22_jobs_require_explicit_compute_authorization": True,
        "repository_does_not_generate_compute_authorization": True,
        "all_18_dictionaries_require_manifest_v2": True,
        "all_18_dictionaries_bind_exact_sample_contract": True,
        "all_18_dictionaries_bind_batch_size_and_learning_rate": True,
        "predecessor_plan_v3_direct_execution_prohibited": True,
    }
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["explicit_compute_authorization_present"] = False
    result["scientific_boundary"]["training_release_receipt_present"] = False
    result_body = {
        key: value for key, value in result.items() if key != "training_plan_id"
    }
    result["training_plan_id"] = content_sha256(result_body)
    return result


def verify_training_plan_v4(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = load_training_plan_v2(path)
    expected = build_training_plan_v4(**kwargs)
    if observed != expected:
        raise ValueError("compartment SAE training plan V4 is not reproducible")
    return {
        "schema": "sch.vpd-compartment-sae-training-plan-verification.v2",
        "status": "pass",
        "training_plan_id": str(observed["training_plan_id"]),
        "training_plan_sha256": sha256_file(path),
        "capture_jobs": len(observed["capture_jobs"]),
        "training_jobs": len(observed["training_jobs"]),
        "guarded_jobs": (
            len(observed["capture_jobs"])
            + len(observed["training_jobs"])
            + 2
        ),
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root
        / "artifacts/vpd_compartment_sae_training_plan_v3/plan.json",
        "protocol_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v8.json",
        "execution_blueprint_path": project_root / BLUEPRINT_RELATIVE,
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive or verify the manifest-V2 compartment SAE plan V4."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None:
        result = verify_training_plan_v4(args.verify, **inputs)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_training_plan_v4(**inputs)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
