from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.admit_compartment_sae_training_release_v2 import (
    verify_explicit_compute_authorization,
    verify_training_release_receipt_v2,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.run_compartment_sae_training_job_v2 import (
    load_training_plan_v2,
)


CAPTURE_MANIFEST_SCHEMA = "sch.vpd-activation-capture-manifest.v1"
CAPTURE_EXECUTION_SCHEMA = "sch.vpd-compartment-sae-capture-execution-receipt.v2"


def select_capture_job(plan: dict[str, Any], *, role: str) -> dict[str, Any]:
    matches = [row for row in plan["capture_jobs"] if str(row.get("role")) == role]
    if len(matches) != 1:
        raise ValueError("V2 training plan does not identify one capture role")
    return matches[0]


def upgrade_capture_manifest(
    *,
    predecessor_manifest: dict[str, Any],
    plan: dict[str, Any],
    training_plan_path: Path,
    job: dict[str, Any],
    release_verification: dict[str, Any],
    compute_verification: dict[str, Any],
    runner_path: Path | None = None,
) -> dict[str, Any]:
    if predecessor_manifest.get("schema") != CAPTURE_MANIFEST_SCHEMA:
        raise ValueError("V2 capture wrapper requires capture manifest V1")
    predecessor_body = dict(predecessor_manifest)
    predecessor_id = str(predecessor_body.pop("capture_id", ""))
    if content_sha256(predecessor_body) != predecessor_id:
        raise ValueError("predecessor capture manifest address mismatch")
    implementation = plan["source"]["implementation"]
    if str(predecessor_manifest.get("source", {}).get("runner_sha256")) != str(
        implementation["predecessor_capture_worker"]["sha256"]
    ):
        raise ValueError("capture used an unplanned predecessor worker")
    if (
        release_verification.get("status") != "pass"
        or compute_verification.get("status") != "pass"
        or str(compute_verification.get("required_job_id")) != str(job["job_id"])
        or compute_verification.get("required_action") != "base_activation_capture"
    ):
        raise ValueError("capture execution authorization is incomplete")
    runner = Path(__file__) if runner_path is None else runner_path
    execution_body = {
        "schema": CAPTURE_EXECUTION_SCHEMA,
        "status": "pass_completed_planned_capture_job",
        "training_plan_id": str(plan["training_plan_id"]),
        "training_plan_sha256": sha256_file(training_plan_path),
        "planned_job_id": str(job["job_id"]),
        "predecessor_job_id": str(job["predecessor_job_id"]),
        "role": str(job["role"]),
        "training_release_receipt_id": str(
            release_verification["training_release_receipt_id"]
        ),
        "training_release_receipt_sha256": str(
            release_verification["training_release_receipt_sha256"]
        ),
        "compute_authorization_id": str(
            compute_verification["compute_authorization_id"]
        ),
        "compute_authorization_sha256": str(
            compute_verification["compute_authorization_sha256"]
        ),
        "runner_v2_sha256": sha256_file(runner),
        "predecessor_capture_worker_sha256": str(
            implementation["predecessor_capture_worker"]["sha256"]
        ),
        "planned_capture_arguments_sha256": content_sha256(
            [str(value) for value in job["capture_arguments"]]
        ),
        "predecessor_capture_id": predecessor_id,
        "model_outcomes_generated": False,
        "generated_tokens": 0,
    }
    execution = {
        **execution_body,
        "capture_execution_receipt_id": content_sha256(execution_body),
    }
    result = json.loads(json.dumps(predecessor_manifest))
    result["source"]["execution_authorization"] = execution
    result_body = {key: value for key, value in result.items() if key != "capture_id"}
    result["capture_id"] = content_sha256(result_body)
    return result


def validate_capture_execution(
    *,
    manifest: dict[str, Any],
    plan: dict[str, Any],
    training_plan_path: Path,
    job: dict[str, Any],
) -> dict[str, Any]:
    if manifest.get("schema") != CAPTURE_MANIFEST_SCHEMA:
        raise ValueError("unsupported V2-authorized capture manifest")
    body = dict(manifest)
    capture_id = str(body.pop("capture_id", ""))
    if content_sha256(body) != capture_id:
        raise ValueError("V2-authorized capture manifest address mismatch")
    execution = manifest.get("source", {}).get("execution_authorization", {})
    execution_body = {
        key: value
        for key, value in execution.items()
        if key != "capture_execution_receipt_id"
    }
    execution_id = str(execution.get("capture_execution_receipt_id", ""))
    if content_sha256(execution_body) != execution_id:
        raise ValueError("capture execution receipt address mismatch")
    expected = {
        "training_plan_id": str(plan["training_plan_id"]),
        "training_plan_sha256": sha256_file(training_plan_path),
        "planned_job_id": str(job["job_id"]),
        "predecessor_job_id": str(job["predecessor_job_id"]),
        "role": str(job["role"]),
        "runner_v2_sha256": str(
            plan["source"]["implementation"]["capture_runner_v2"]["sha256"]
        ),
        "predecessor_capture_worker_sha256": str(
            plan["source"]["implementation"]["predecessor_capture_worker"][
                "sha256"
            ]
        ),
        "planned_capture_arguments_sha256": content_sha256(
            [str(value) for value in job["capture_arguments"]]
        ),
        "generated_tokens": 0,
        "model_outcomes_generated": False,
    }
    for field, value in expected.items():
        if execution.get(field) != value:
            raise ValueError(f"capture execution receipt differs: {field}")
    for field in (
        "training_release_receipt_id",
        "training_release_receipt_sha256",
        "compute_authorization_id",
        "compute_authorization_sha256",
        "predecessor_capture_id",
    ):
        if len(str(execution.get(field, ""))) != 64:
            raise ValueError("capture execution receipt has an invalid digest")
    return {
        "schema": "sch.vpd-compartment-sae-capture-execution-verification.v2",
        "status": "pass",
        "capture_id": capture_id,
        "capture_execution_receipt_id": execution_id,
        "planned_job_id": str(job["job_id"]),
        "role": str(job["role"]),
        "training_release_receipt_id": str(
            execution["training_release_receipt_id"]
        ),
        "compute_authorization_id": str(execution["compute_authorization_id"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one release- and user-authorized base capture job V2."
    )
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--training-release-receipt", type=Path, required=True)
    parser.add_argument("--compute-authorization-receipt", type=Path, required=True)
    parser.add_argument("--role", choices=("sae_training", "calibration"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    plan = load_training_plan_v2(args.training_plan)
    job = select_capture_job(plan, role=args.role)
    release_gate_path = Path(plan["source"]["training_release_gate_v2"]["path"])
    release_verification = verify_training_release_receipt_v2(
        args.training_release_receipt,
        release_gate_plan_path=release_gate_path,
        training_plan_path=args.training_plan,
    )
    compute_verification = verify_explicit_compute_authorization(
        args.compute_authorization_receipt,
        release_verification=release_verification,
        release_gate_plan_path=release_gate_path,
        training_plan_path=args.training_plan,
        required_job_id=str(job["job_id"]),
        required_action="base_activation_capture",
    )
    arguments = [str(value) for value in job["capture_arguments"]]
    if arguments[:2] != [
        "-m",
        "experiments.authz_mechinterp.capture_activations",
    ]:
        raise ValueError("capture V2 does not delegate to the preserved worker")
    delegated = [
        args.model
        if value == "${FROZEN_BASE_MODEL}"
        else args.revision
        if value == "${FROZEN_BASE_REVISION}"
        else value
        for value in arguments[2:]
    ]
    if any(value.startswith("${") for value in delegated):
        raise ValueError("capture V2 retains an unresolved placeholder")
    from experiments.authz_mechinterp import capture_activations

    previous_argv = sys.argv
    try:
        sys.argv = [str(Path(capture_activations.__file__).resolve()), *delegated]
        capture_activations.main()
    finally:
        sys.argv = previous_argv
    manifest_path = Path(str(job["output_dir"])) / "capture_manifest.json"
    predecessor = json.loads(manifest_path.read_text(encoding="utf-8"))
    upgraded = upgrade_capture_manifest(
        predecessor_manifest=predecessor,
        plan=plan,
        training_plan_path=args.training_plan,
        job=job,
        release_verification=release_verification,
        compute_verification=compute_verification,
    )
    write_json(manifest_path, upgraded)
    print(
        json.dumps(
            {
                "schema": "sch.vpd-compartment-sae-capture-job-cli.v2",
                "status": "pass",
                "planned_job_id": job["job_id"],
                "capture_id": upgraded["capture_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
