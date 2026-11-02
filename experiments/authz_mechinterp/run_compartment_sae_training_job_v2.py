from __future__ import annotations

import argparse
import json
import sys
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


PLAN_SCHEMA = "sch.vpd-compartment-sae-training-plan.v2"
MANIFEST_V1_SCHEMA = "sch.vpd-fixed-sae-dictionary-manifest.v1"
MANIFEST_V2_SCHEMA = "sch.vpd-fixed-sae-dictionary-manifest.v2"
EXECUTION_SCHEMA = "sch.vpd-compartment-sae-training-execution-receipt.v2"


def _load_addressed(
    path: Path, *, schema: str, id_field: str
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema:
        raise ValueError(f"unsupported content-addressed schema: {path}")
    body = dict(value)
    identifier = str(body.pop(id_field, ""))
    if content_sha256(body) != identifier:
        raise ValueError(f"content address mismatch: {path}")
    return value


def load_training_plan_v2(path: Path) -> dict[str, Any]:
    plan = _load_addressed(
        path, schema=PLAN_SCHEMA, id_field="training_plan_id"
    )
    jobs = plan.get("training_jobs")
    if not isinstance(jobs, list) or len(jobs) != 18:
        raise ValueError("V2 training plan must contain 18 training jobs")
    for job in jobs:
        body = dict(job)
        identifier = str(body.pop("job_id", ""))
        if content_sha256(body) != identifier:
            raise ValueError("V2 training job content address mismatch")
    if plan.get("scientific_boundary", {}).get("execution_authorized") is not False:
        raise ValueError("V2 training plan weakens its execution boundary")
    return plan


def select_training_job(
    plan: dict[str, Any], *, compartment: str, initialization_seed: int
) -> dict[str, Any]:
    matches = [
        row
        for row in plan["training_jobs"]
        if str(row.get("compartment")) == compartment
        and int(row.get("initialization_seed", -1)) == initialization_seed
    ]
    if len(matches) != 1:
        raise ValueError("V2 training plan does not identify one requested job")
    return matches[0]


def _blueprint_record(
    blueprint: dict[str, Any], *, predecessor_job_id: str
) -> dict[str, Any]:
    matches = [
        row
        for row in blueprint.get("jobs", [])
        if str(row.get("planned_job_id")) == predecessor_job_id
    ]
    if len(matches) != 1:
        raise ValueError("execution blueprint does not identify the predecessor job")
    return matches[0]


def validate_job_against_blueprint(
    *,
    plan: dict[str, Any],
    job: dict[str, Any],
    blueprint: dict[str, Any],
) -> dict[str, Any]:
    predecessor_job_id = str(job.get("predecessor_job_id", ""))
    record = _blueprint_record(
        blueprint, predecessor_job_id=predecessor_job_id
    )
    if (
        str(record.get("compartment")) != str(job.get("compartment"))
        or int(record.get("initialization_seed", -1))
        != int(job.get("initialization_seed", -2))
    ):
        raise ValueError("V2 training job differs from its execution blueprint")
    trainer_arguments = [str(value) for value in job.get("trainer_arguments", [])]
    if content_sha256(trainer_arguments) != str(
        record.get("planned_arguments_sha256")
    ):
        raise ValueError("V2 trainer arguments differ from the execution blueprint")
    if job.get("required_manifest_hyperparameters") != record.get(
        "required_manifest_hyperparameters"
    ):
        raise ValueError("V2 job hyperparameters differ from the execution blueprint")
    if job.get("required_sample_contract") != record.get(
        "required_sample_contract"
    ):
        raise ValueError("V2 job sample contract differs from the execution blueprint")
    source = plan.get("source", {})
    if str(source.get("execution_blueprint_id")) != str(
        blueprint.get("execution_blueprint_id")
    ):
        raise ValueError("V2 plan and execution blueprint identities differ")
    return record


def _validate_predecessor_manifest(
    *,
    manifest: dict[str, Any],
    plan: dict[str, Any],
    job: dict[str, Any],
    record: dict[str, Any],
) -> None:
    if manifest.get("schema") != MANIFEST_V1_SCHEMA:
        raise ValueError("V2 upgrade requires a predecessor V1 dictionary manifest")
    body = dict(manifest)
    identifier = str(body.pop("dictionary_id", ""))
    if content_sha256(body) != identifier:
        raise ValueError("predecessor dictionary manifest address mismatch")
    source = manifest.get("source", {})
    implementation = plan.get("source", {}).get("implementation", {})
    if str(source.get("runner_sha256")) != str(
        implementation.get("predecessor_sae_trainer", {}).get("sha256")
    ):
        raise ValueError("predecessor dictionary used an unplanned trainer")
    dictionary = manifest.get("dictionary", {})
    hyperparameters = record["required_manifest_hyperparameters"]
    expected = {
        "steps": int(hyperparameters["steps"]),
        "seed": int(hyperparameters["initialization_seed"]),
        "data_selection_seed": int(hyperparameters["data_selection_seed"]),
    }
    if any(dictionary.get(field) != value for field, value in expected.items()):
        raise ValueError("predecessor dictionary hyperparameters differ from V2")
    compartment = dictionary.get("activation_compartment", {})
    if str(compartment.get("activation_category")) != str(job["compartment"]):
        raise ValueError("predecessor dictionary compartment differs from V2")
    sample = record["required_sample_contract"]
    capacity = compartment.get("capacity_authorization", {})
    if (
        int(capacity.get("selected_activation_rows", -1))
        != int(sample["selected_activation_rows"])
        or int(capacity.get("selected_episodes", -1))
        != int(sample["selected_episodes"])
        or int(capacity.get("selected_unique_pairs", -1))
        != int(sample["selected_unique_pairs"])
    ):
        raise ValueError("predecessor dictionary sample support differs from V2")
    split = dictionary.get("train_validation_split", {})
    expected_partition = sample["pair_atomic_partition"]
    for field in (
        "training_pairs",
        "validation_pairs",
        "training_rows",
        "validation_rows",
        "training_pair_id_set_sha256",
        "validation_pair_id_set_sha256",
    ):
        if split.get(field) != expected_partition.get(field):
            raise ValueError("predecessor dictionary pair partition differs from V2")


def upgrade_dictionary_manifest_v2(
    *,
    predecessor_manifest: dict[str, Any],
    predecessor_manifest_sha256: str,
    plan: dict[str, Any],
    training_plan_path: Path,
    job: dict[str, Any],
    blueprint: dict[str, Any],
    execution_blueprint_path: Path,
    release_verification: dict[str, Any],
    compute_authorization_verification: dict[str, Any],
    runner_path: Path | None = None,
) -> dict[str, Any]:
    record = validate_job_against_blueprint(
        plan=plan, job=job, blueprint=blueprint
    )
    _validate_predecessor_manifest(
        manifest=predecessor_manifest,
        plan=plan,
        job=job,
        record=record,
    )
    for field in (
        "training_release_receipt_id",
        "training_release_receipt_sha256",
    ):
        if len(str(release_verification.get(field, ""))) != 64:
            raise ValueError("V2 training release verification is incomplete")
    if release_verification.get("status") != "pass":
        raise ValueError("V2 training release verification did not pass")
    for field in ("compute_authorization_id", "compute_authorization_sha256"):
        if len(str(compute_authorization_verification.get(field, ""))) != 64:
            raise ValueError("V2 compute authorization verification is incomplete")
    if (
        compute_authorization_verification.get("status") != "pass"
        or str(compute_authorization_verification.get("required_job_id"))
        != str(job["job_id"])
        or compute_authorization_verification.get("required_action")
        != "compartment_sae_training"
    ):
        raise ValueError("V2 compute authorization does not cover the training job")
    runner = Path(__file__) if runner_path is None else runner_path
    execution_body = {
        "schema": EXECUTION_SCHEMA,
        "status": "pass_completed_planned_training_job",
        "training_plan_id": str(plan["training_plan_id"]),
        "training_plan_sha256": sha256_file(training_plan_path),
        "planned_job_id": str(job["job_id"]),
        "predecessor_job_id": str(job["predecessor_job_id"]),
        "execution_blueprint_id": str(blueprint["execution_blueprint_id"]),
        "execution_blueprint_sha256": sha256_file(execution_blueprint_path),
        "training_release_receipt_id": str(
            release_verification["training_release_receipt_id"]
        ),
        "training_release_receipt_sha256": str(
            release_verification["training_release_receipt_sha256"]
        ),
        "compute_authorization_id": str(
            compute_authorization_verification["compute_authorization_id"]
        ),
        "compute_authorization_sha256": str(
            compute_authorization_verification["compute_authorization_sha256"]
        ),
        "runner_v2_sha256": sha256_file(runner),
        "predecessor_trainer_sha256": str(
            plan["source"]["implementation"]["predecessor_sae_trainer"][
                "sha256"
            ]
        ),
        "planned_trainer_arguments_sha256": content_sha256(
            [str(value) for value in job["trainer_arguments"]]
        ),
        "bound_hyperparameters": dict(
            record["required_manifest_hyperparameters"]
        ),
        "exact_sample_contract": dict(record["required_sample_contract"]),
        "predecessor_v1_dictionary_id": str(
            predecessor_manifest["dictionary_id"]
        ),
        "predecessor_v1_manifest_sha256": predecessor_manifest_sha256,
        "dictionary_artifact_sha256": str(
            predecessor_manifest["artifacts"]["dictionary"]["sha256"]
        ),
        "weights_loaded_or_mutated_after_training": False,
    }
    execution = {
        **execution_body,
        "training_execution_receipt_id": content_sha256(execution_body),
    }
    result = json.loads(json.dumps(predecessor_manifest))
    result["schema"] = MANIFEST_V2_SCHEMA
    result["source"]["training_execution"] = execution
    result["dictionary"]["batch_size"] = int(
        record["required_manifest_hyperparameters"]["batch_size"]
    )
    result["dictionary"]["learning_rate"] = float(
        record["required_manifest_hyperparameters"]["learning_rate"]
    )
    result["dictionary"]["exact_sample_contract"] = dict(
        record["required_sample_contract"]
    )
    result["dictionary"]["training_execution_receipt_id"] = execution[
        "training_execution_receipt_id"
    ]
    result_body = {
        key: value for key, value in result.items() if key != "dictionary_id"
    }
    result["dictionary_id"] = content_sha256(result_body)
    return result


def validate_dictionary_manifest_v2(
    *,
    manifest: dict[str, Any],
    plan: dict[str, Any],
    training_plan_path: Path,
    job: dict[str, Any],
    blueprint: dict[str, Any],
    execution_blueprint_path: Path,
) -> dict[str, Any]:
    if manifest.get("schema") != MANIFEST_V2_SCHEMA:
        raise ValueError("unsupported V2 dictionary manifest schema")
    body = dict(manifest)
    dictionary_id = str(body.pop("dictionary_id", ""))
    if content_sha256(body) != dictionary_id:
        raise ValueError("V2 dictionary manifest content address mismatch")
    record = validate_job_against_blueprint(
        plan=plan, job=job, blueprint=blueprint
    )
    execution = manifest.get("source", {}).get("training_execution", {})
    execution_body = {
        key: value
        for key, value in execution.items()
        if key != "training_execution_receipt_id"
    }
    execution_id = str(execution.get("training_execution_receipt_id", ""))
    if content_sha256(execution_body) != execution_id:
        raise ValueError("V2 training execution receipt address mismatch")
    implementation = plan["source"]["implementation"]
    expected_execution = {
        "training_plan_id": str(plan["training_plan_id"]),
        "training_plan_sha256": sha256_file(training_plan_path),
        "planned_job_id": str(job["job_id"]),
        "predecessor_job_id": str(job["predecessor_job_id"]),
        "execution_blueprint_id": str(blueprint["execution_blueprint_id"]),
        "execution_blueprint_sha256": sha256_file(execution_blueprint_path),
        "runner_v2_sha256": str(implementation["sae_training_runner_v2"]["sha256"]),
        "predecessor_trainer_sha256": str(
            implementation["predecessor_sae_trainer"]["sha256"]
        ),
        "planned_trainer_arguments_sha256": content_sha256(
            [str(value) for value in job["trainer_arguments"]]
        ),
        "bound_hyperparameters": record["required_manifest_hyperparameters"],
        "exact_sample_contract": record["required_sample_contract"],
        "dictionary_artifact_sha256": str(
            manifest["artifacts"]["dictionary"]["sha256"]
        ),
    }
    for field, expected in expected_execution.items():
        if execution.get(field) != expected:
            raise ValueError(f"V2 training execution receipt differs: {field}")
    if execution.get("status") != "pass_completed_planned_training_job" or (
        execution.get("weights_loaded_or_mutated_after_training") is not False
    ):
        raise ValueError("V2 training execution receipt weakens its boundary")
    for field in (
        "training_release_receipt_id",
        "training_release_receipt_sha256",
        "compute_authorization_id",
        "compute_authorization_sha256",
        "predecessor_v1_dictionary_id",
        "predecessor_v1_manifest_sha256",
    ):
        if len(str(execution.get(field, ""))) != 64:
            raise ValueError("V2 training execution receipt has an invalid digest")
    dictionary = manifest.get("dictionary", {})
    hyperparameters = record["required_manifest_hyperparameters"]
    if (
        int(dictionary.get("batch_size", -1))
        != int(hyperparameters["batch_size"])
        or float(dictionary.get("learning_rate", -1.0))
        != float(hyperparameters["learning_rate"])
        or dictionary.get("exact_sample_contract")
        != record["required_sample_contract"]
        or str(dictionary.get("training_execution_receipt_id")) != execution_id
    ):
        raise ValueError("V2 dictionary manifest execution bindings differ")

    predecessor = json.loads(json.dumps(manifest))
    predecessor["schema"] = MANIFEST_V1_SCHEMA
    predecessor["source"].pop("training_execution", None)
    predecessor["dictionary"].pop("batch_size", None)
    predecessor["dictionary"].pop("learning_rate", None)
    predecessor["dictionary"].pop("exact_sample_contract", None)
    predecessor["dictionary"].pop("training_execution_receipt_id", None)
    predecessor["dictionary_id"] = str(execution["predecessor_v1_dictionary_id"])
    predecessor_body = {
        key: value for key, value in predecessor.items() if key != "dictionary_id"
    }
    if content_sha256(predecessor_body) != predecessor["dictionary_id"]:
        raise ValueError("V2 manifest does not reconstruct its predecessor V1 address")
    _validate_predecessor_manifest(
        manifest=predecessor,
        plan=plan,
        job=job,
        record=record,
    )
    return {
        "schema": "sch.vpd-compartment-sae-training-execution-verification.v2",
        "status": "pass",
        "planned_job_id": str(job["job_id"]),
        "dictionary_id": dictionary_id,
        "training_execution_receipt_id": execution_id,
        "training_release_receipt_id": str(
            execution["training_release_receipt_id"]
        ),
        "compartment": str(job["compartment"]),
        "initialization_seed": int(job["initialization_seed"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Execute one release-authorized compartment SAE V2 job and emit manifest V2."
        )
    )
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--execution-blueprint", type=Path, required=True)
    parser.add_argument("--training-release-receipt", type=Path, required=True)
    parser.add_argument("--compute-authorization-receipt", type=Path, required=True)
    parser.add_argument("--compartment", required=True)
    parser.add_argument("--initialization-seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    plan = load_training_plan_v2(args.training_plan)
    job = select_training_job(
        plan,
        compartment=args.compartment,
        initialization_seed=args.initialization_seed,
    )
    blueprint = json.loads(args.execution_blueprint.read_text(encoding="utf-8"))
    predecessor_plan_path = Path(
        plan["source"]["predecessor_training_plan"]["path"]
    )
    validate_blueprint(
        blueprint,
        training_plan_path=predecessor_plan_path,
        project_root=Path(".").resolve(),
    )
    validate_job_against_blueprint(plan=plan, job=job, blueprint=blueprint)

    from experiments.authz_mechinterp.admit_compartment_sae_training_release_v2 import (
        verify_explicit_compute_authorization,
        verify_training_release_receipt_v2,
    )

    release_verification = verify_training_release_receipt_v2(
        args.training_release_receipt,
        release_gate_plan_path=Path(
            plan["source"]["training_release_gate_v2"]["path"]
        ),
        training_plan_path=args.training_plan,
    )
    compute_verification = verify_explicit_compute_authorization(
        args.compute_authorization_receipt,
        release_verification=release_verification,
        release_gate_plan_path=Path(
            plan["source"]["training_release_gate_v2"]["path"]
        ),
        training_plan_path=args.training_plan,
        required_job_id=str(job["job_id"]),
        required_action="compartment_sae_training",
    )

    trainer_arguments = [str(value) for value in job["trainer_arguments"]]
    if trainer_arguments[:2] != [
        "-m",
        "experiments.authz_mechinterp.train_sae",
    ]:
        raise ValueError("V2 job does not delegate to the preserved SAE trainer")
    delegated = [
        args.device if value == "${SAE_TRAINING_DEVICE}" else value
        for value in trainer_arguments[2:]
    ]
    if any(value.startswith("${") for value in delegated):
        raise ValueError("V2 trainer arguments retain an unresolved placeholder")
    capture_manifest_positions = [
        index
        for index, value in enumerate(trainer_arguments)
        if value == "--capture-manifest"
    ]
    if len(capture_manifest_positions) != 1:
        raise ValueError("V2 training job lacks one capture manifest argument")
    capture_manifest_path = Path(
        trainer_arguments[capture_manifest_positions[0] + 1]
    )
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    from experiments.authz_mechinterp.run_compartment_sae_capture_job_v2 import (
        select_capture_job,
        validate_capture_execution,
    )

    capture_job = select_capture_job(plan, role="sae_training")
    capture_compute_verification = verify_explicit_compute_authorization(
        args.compute_authorization_receipt,
        release_verification=release_verification,
        release_gate_plan_path=Path(
            plan["source"]["training_release_gate_v2"]["path"]
        ),
        training_plan_path=args.training_plan,
        required_job_id=str(capture_job["job_id"]),
        required_action="base_activation_capture",
    )
    capture_verification = validate_capture_execution(
        manifest=capture_manifest,
        plan=plan,
        training_plan_path=args.training_plan,
        job=capture_job,
    )
    if (
        str(capture_verification["training_release_receipt_id"])
        != str(release_verification["training_release_receipt_id"])
        or str(capture_verification["compute_authorization_id"])
        != str(capture_compute_verification["compute_authorization_id"])
        or str(capture_compute_verification["compute_authorization_id"])
        != str(compute_verification["compute_authorization_id"])
    ):
        raise ValueError("V2 training capture used different authorizations")
    from experiments.authz_mechinterp import train_sae

    previous_argv = sys.argv
    try:
        sys.argv = [str(Path(train_sae.__file__).resolve()), *delegated]
        train_sae.main()
    finally:
        sys.argv = previous_argv

    output_dir = Path(str(job["output_dir"]))
    manifest_path = output_dir / "sae_dictionary_manifest.json"
    predecessor_bytes_sha256 = sha256_file(manifest_path)
    predecessor_manifest = _load_addressed(
        manifest_path,
        schema=MANIFEST_V1_SCHEMA,
        id_field="dictionary_id",
    )
    upgraded = upgrade_dictionary_manifest_v2(
        predecessor_manifest=predecessor_manifest,
        predecessor_manifest_sha256=predecessor_bytes_sha256,
        plan=plan,
        training_plan_path=args.training_plan,
        job=job,
        blueprint=blueprint,
        execution_blueprint_path=args.execution_blueprint,
        release_verification=release_verification,
        compute_authorization_verification=compute_verification,
    )
    write_json(manifest_path, upgraded)
    print(
        json.dumps(
            {
                "schema": "sch.vpd-compartment-sae-training-job-cli.v2",
                "status": "pass",
                "planned_job_id": job["job_id"],
                "dictionary_id": upgraded["dictionary_id"],
                "training_execution_receipt_id": upgraded["source"][
                    "training_execution"
                ]["training_execution_receipt_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
