from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp import admit_compartment_sae_dictionary_bundle as v1
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_compartment_sae_training_execution_blueprint_v2 import (
    validate_blueprint,
)
from experiments.authz_mechinterp.run_compartment_sae_training_job_v2 import (
    MANIFEST_V2_SCHEMA,
    load_training_plan_v2,
    validate_dictionary_manifest_v2,
)


BUNDLE_SCHEMA = "sch.vpd-compartment-sae-dictionary-bundle.v2"


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


def _confined(directory: Path, relative_path: str) -> Path:
    root = directory.resolve()
    path = (directory / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("V2 dictionary artifact escapes its output directory")
    return path


def validate_bundle_receipt_structure_v2(receipt: dict[str, Any]) -> None:
    if receipt.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("unsupported compartment SAE dictionary bundle V2 schema")
    body = dict(receipt)
    identifier = str(body.pop("dictionary_bundle_id", ""))
    if content_sha256(body) != identifier:
        raise ValueError("compartment SAE dictionary bundle V2 address mismatch")
    rows = receipt.get("dictionaries")
    if not isinstance(rows, list) or len(rows) != 18:
        raise ValueError("compartment SAE dictionary bundle V2 requires 18 outputs")
    if len({str(row.get("planned_job_id")) for row in rows}) != 18:
        raise ValueError("compartment SAE dictionary bundle V2 repeats jobs")
    if len({str(row.get("dictionary_id")) for row in rows}) != 18:
        raise ValueError("compartment SAE dictionary bundle V2 repeats dictionaries")
    for row in rows:
        if row.get("manifest_schema") != MANIFEST_V2_SCHEMA:
            raise ValueError("bundle V2 contains a predecessor dictionary manifest")
        for field in (
            "planned_job_id",
            "dictionary_id",
            "dictionary_manifest_sha256",
            "dictionary_sha256",
            "training_execution_receipt_id",
        ):
            if len(str(row.get(field, ""))) != 64:
                raise ValueError("bundle V2 contains an invalid digest")
    source = receipt.get("source", {})
    for field in (
        "training_plan_id",
        "training_plan_sha256",
        "execution_blueprint_id",
        "execution_blueprint_sha256",
        "training_release_receipt_id",
        "training_release_receipt_sha256",
        "compute_authorization_id",
        "compute_authorization_sha256",
        "admission_runner_sha256",
        "predecessor_v1_admission_runner_sha256",
    ):
        if len(str(source.get(field, ""))) != 64:
            raise ValueError("bundle V2 source is incomplete")
    if receipt.get("status") != "pass_ready_for_calibration_quality_v2":
        raise ValueError("bundle V2 status differs")
    invariants = receipt.get("invariants", {})
    for field in (
        "all_dictionary_manifests_are_v2",
        "all_training_execution_receipts_reverified",
        "all_planned_hyperparameters_bound",
        "all_exact_sample_contracts_bound",
        "one_release_receipt_shared_by_every_dictionary",
        "predecessor_v1_bundle_preserved",
    ):
        if invariants.get(field) is not True:
            raise ValueError("bundle V2 weakens its execution invariants")
    boundary = receipt.get("scientific_boundary", {})
    if boundary.get("quality_claimed") is not False or boundary.get(
        "focused_interventions_authorized"
    ) is not False:
        raise ValueError("bundle V2 weakens its scientific boundary")


def _load_and_validate_manifests(
    *,
    run_dir: Path,
    training_plan_path: Path,
    execution_blueprint_path: Path,
    project_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    plan = load_training_plan_v2(training_plan_path)
    blueprint = json.loads(execution_blueprint_path.read_text(encoding="utf-8"))
    predecessor_plan_path = project_root / str(
        plan["source"]["predecessor_training_plan"]["path"]
    )
    validate_blueprint(
        blueprint,
        training_plan_path=predecessor_plan_path,
        project_root=project_root,
    )
    from experiments.authz_mechinterp.run_compartment_sae_capture_job_v2 import (
        select_capture_job,
        validate_capture_execution,
    )

    capture_job = select_capture_job(plan, role="sae_training")
    capture_manifest = json.loads(
        (run_dir / str(capture_job["output_dir"]) / "capture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    capture_verification = validate_capture_execution(
        manifest=capture_manifest,
        plan=plan,
        training_plan_path=training_plan_path,
        job=capture_job,
    )
    verifications: dict[str, dict[str, Any]] = {}
    for job in plan["training_jobs"]:
        directory = run_dir / str(job["output_dir"])
        manifest_path = directory / "sae_dictionary_manifest.json"
        manifest = _load_addressed(
            manifest_path,
            schema=MANIFEST_V2_SCHEMA,
            id_field="dictionary_id",
        )
        verification = validate_dictionary_manifest_v2(
            manifest=manifest,
            plan=plan,
            training_plan_path=training_plan_path,
            job=job,
            blueprint=blueprint,
            execution_blueprint_path=execution_blueprint_path,
        )
        artifact = manifest["artifacts"]["dictionary"]
        artifact_path = _confined(directory, str(artifact["path"]))
        if sha256_file(artifact_path) != str(artifact["sha256"]):
            raise ValueError("V2 dictionary artifact differs from its manifest")
        verification.update(
            {
                "manifest": manifest,
                "manifest_path": manifest_path,
                "artifact_path": artifact_path,
            }
        )
        verifications[str(job["job_id"])] = verification
    release_ids = {
        str(row["training_release_receipt_id"])
        for row in verifications.values()
    }
    if len(release_ids) != 1:
        raise ValueError("V2 dictionaries use different training release receipts")
    return plan, blueprint, verifications, capture_verification


def _delegate_predecessor_admission(
    *,
    run_dir: Path,
    project_root: Path,
    training_plan_path: Path,
) -> dict[str, Any]:
    original_loader = v1._load_content_addressed
    original_validator = v1.validate_training_plan_structure

    def compatible_loader(path: Path, *, schema: str, id_field: str) -> dict[str, Any]:
        if path.resolve() == training_plan_path.resolve():
            return _load_addressed(
                path,
                schema="sch.vpd-compartment-sae-training-plan.v2",
                id_field="training_plan_id",
            )
        if path.name == "sae_dictionary_manifest.json":
            return _load_addressed(
                path, schema=MANIFEST_V2_SCHEMA, id_field="dictionary_id"
            )
        return original_loader(path, schema=schema, id_field=id_field)

    try:
        v1._load_content_addressed = compatible_loader
        v1.validate_training_plan_structure = lambda plan: None
        return v1.admit_run_directory(
            run_dir=run_dir,
            project_root=project_root,
            training_plan_path=training_plan_path,
            output_path=None,
        )
    finally:
        v1._load_content_addressed = original_loader
        v1.validate_training_plan_structure = original_validator


def admit_run_directory_v2(
    *,
    run_dir: Path,
    project_root: Path,
    training_plan_path: Path,
    execution_blueprint_path: Path,
    training_release_receipt_path: Path,
    compute_authorization_receipt_path: Path,
    output_path: Path | None,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    plan, blueprint, verifications, capture_verification = (
        _load_and_validate_manifests(
        run_dir=run_dir,
        training_plan_path=training_plan_path,
        execution_blueprint_path=execution_blueprint_path,
        project_root=project_root,
        )
    )
    from experiments.authz_mechinterp.admit_compartment_sae_training_release_v2 import (
        verify_explicit_compute_authorization,
        verify_training_release_receipt_v2,
    )

    release_verification = verify_training_release_receipt_v2(
        training_release_receipt_path,
        release_gate_plan_path=project_root
        / str(plan["source"]["training_release_gate_v2"]["path"]),
        training_plan_path=training_plan_path,
    )
    compute_verification = verify_explicit_compute_authorization(
        compute_authorization_receipt_path,
        release_verification=release_verification,
        release_gate_plan_path=project_root
        / str(plan["source"]["training_release_gate_v2"]["path"]),
        training_plan_path=training_plan_path,
        required_job_id=str(plan["admission_job"]["job_id"]),
        required_action="dictionary_bundle_admission",
    )
    upstream_authorizations = []
    capture_job = next(
        row for row in plan["capture_jobs"] if row.get("role") == "sae_training"
    )
    upstream_authorizations.append(
        verify_explicit_compute_authorization(
            compute_authorization_receipt_path,
            release_verification=release_verification,
            release_gate_plan_path=project_root
            / str(plan["source"]["training_release_gate_v2"]["path"]),
            training_plan_path=training_plan_path,
            required_job_id=str(capture_job["job_id"]),
            required_action="base_activation_capture",
        )
    )
    for training_job in plan["training_jobs"]:
        upstream_authorizations.append(
            verify_explicit_compute_authorization(
                compute_authorization_receipt_path,
                release_verification=release_verification,
                release_gate_plan_path=project_root
                / str(plan["source"]["training_release_gate_v2"]["path"]),
                training_plan_path=training_plan_path,
                required_job_id=str(training_job["job_id"]),
                required_action="compartment_sae_training",
            )
        )
    if {
        str(row["compute_authorization_id"]) for row in upstream_authorizations
    } != {str(compute_verification["compute_authorization_id"])}:
        raise ValueError("bundle upstream jobs use different compute authorizations")
    observed_release_ids = {
        str(row["training_release_receipt_id"])
        for row in verifications.values()
    }
    if observed_release_ids != {
        str(release_verification["training_release_receipt_id"])
    }:
        raise ValueError("bundle dictionaries use a different release receipt")
    observed_compute_ids = {
        str(row["manifest"]["source"]["training_execution"]["compute_authorization_id"])
        for row in verifications.values()
    }
    if observed_compute_ids != {
        str(compute_verification["compute_authorization_id"])
    }:
        raise ValueError("bundle dictionaries use a different compute authorization")
    if (
        str(capture_verification["training_release_receipt_id"])
        != str(release_verification["training_release_receipt_id"])
        or str(capture_verification["compute_authorization_id"])
        != str(compute_verification["compute_authorization_id"])
    ):
        raise ValueError("bundle training capture used different authorizations")
    predecessor = _delegate_predecessor_admission(
        run_dir=run_dir,
        project_root=project_root,
        training_plan_path=training_plan_path,
    )
    runner = Path(__file__) if runner_path is None else runner_path
    result = json.loads(json.dumps(predecessor))
    result["schema"] = BUNDLE_SCHEMA
    result["status"] = "pass_ready_for_calibration_quality_v2"
    result["source"]["predecessor_v1_admission_runner_sha256"] = str(
        predecessor["source"]["admission_runner_sha256"]
    )
    result["source"]["admission_runner_sha256"] = sha256_file(runner)
    result["source"]["execution_blueprint_id"] = str(
        blueprint["execution_blueprint_id"]
    )
    result["source"]["execution_blueprint_sha256"] = sha256_file(
        execution_blueprint_path
    )
    result["source"]["training_release_receipt_id"] = str(
        release_verification["training_release_receipt_id"]
    )
    result["source"]["training_release_receipt_sha256"] = str(
        release_verification["training_release_receipt_sha256"]
    )
    result["source"]["compute_authorization_id"] = str(
        compute_verification["compute_authorization_id"]
    )
    result["source"]["compute_authorization_sha256"] = str(
        compute_verification["compute_authorization_sha256"]
    )
    by_job = {str(row["planned_job_id"]): row for row in result["dictionaries"]}
    for job_id, verification in verifications.items():
        row = by_job[job_id]
        row["manifest_schema"] = MANIFEST_V2_SCHEMA
        row["training_execution_receipt_id"] = str(
            verification["training_execution_receipt_id"]
        )
    result["invariants"].update(
        {
            "all_dictionary_manifests_are_v2": True,
            "all_training_execution_receipts_reverified": True,
            "all_planned_hyperparameters_bound": True,
            "all_exact_sample_contracts_bound": True,
            "one_release_receipt_shared_by_every_dictionary": True,
            "predecessor_v1_bundle_preserved": True,
        }
    )
    result_body = {
        key: value for key, value in result.items() if key != "dictionary_bundle_id"
    }
    result["dictionary_bundle_id"] = content_sha256(result_body)
    validate_bundle_receipt_structure_v2(result)
    if output_path is not None:
        write_json(output_path, result)
    return result


def verify_bundle_for_quality_v2(
    *,
    receipt_path: Path,
    training_plan_path: Path,
    execution_blueprint_path: Path,
    dictionary_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_bundle_receipt_structure_v2(receipt)
    plan = load_training_plan_v2(training_plan_path)
    blueprint = json.loads(execution_blueprint_path.read_text(encoding="utf-8"))
    project_root = Path(__file__).resolve().parents[2]
    predecessor_plan_path = project_root / str(
        plan["source"]["predecessor_training_plan"]["path"]
    )
    validate_blueprint(
        blueprint,
        training_plan_path=predecessor_plan_path,
        project_root=project_root,
    )
    if (
        str(receipt["source"]["training_plan_id"])
        != str(plan["training_plan_id"])
        or str(receipt["source"]["training_plan_sha256"])
        != sha256_file(training_plan_path)
        or str(receipt["source"]["execution_blueprint_id"])
        != str(blueprint["execution_blueprint_id"])
        or str(receipt["source"]["execution_blueprint_sha256"])
        != sha256_file(execution_blueprint_path)
        or str(plan["source"]["execution_blueprint_id"])
        != str(blueprint["execution_blueprint_id"])
        or str(plan["source"]["execution_blueprint_sha256"])
        != sha256_file(execution_blueprint_path)
        or str(receipt["source"]["admission_runner_sha256"])
        != sha256_file(Path(__file__))
    ):
        raise ValueError("bundle V2 source lineage differs")
    receipt_rows = {
        str(row["dictionary_id"]): row for row in receipt["dictionaries"]
    }
    plan_jobs = {str(row["job_id"]): row for row in plan["training_jobs"]}
    receipt_by_job = {
        str(row["planned_job_id"]): row for row in receipt["dictionaries"]
    }
    if set(receipt_by_job) != set(plan_jobs):
        raise ValueError("bundle V2 planned job coverage differs")
    for job_id, admitted in receipt_by_job.items():
        job = plan_jobs[job_id]
        if (
            str(admitted.get("compartment")) != str(job["compartment"])
            or int(admitted.get("initialization_seed", -1))
            != int(job["initialization_seed"])
            or int(admitted.get("data_selection_seed", -1))
            != int(job["data_selection_seed"])
            or str(admitted.get("planned_output_dir"))
            != str(job["output_dir"])
        ):
            raise ValueError("bundle V2 differs from its planned job")
    sources = {str(row["dictionary_id"]): row for row in dictionary_sources}
    if set(receipt_rows) != set(sources):
        raise ValueError("bundle V2 and quality dictionary coverage differ")
    for dictionary_id, source in sources.items():
        admitted = receipt_rows[dictionary_id]
        for field in (
            "compartment",
            "initialization_seed",
            "data_selection_seed",
            "dictionary_manifest_sha256",
            "dictionary_sha256",
            "training_execution_receipt_id",
        ):
            if admitted.get(field) != source.get(field):
                raise ValueError("bundle V2 quality source identity differs")
    return {
        "schema": "sch.vpd-compartment-sae-dictionary-bundle-authorization.v2",
        "status": "pass",
        "training_plan_id": str(plan["training_plan_id"]),
        "dictionary_bundle_id": str(receipt["dictionary_bundle_id"]),
        "dictionary_bundle_sha256": sha256_file(receipt_path),
        "execution_blueprint_id": str(blueprint["execution_blueprint_id"]),
        "training_release_receipt_id": str(
            receipt["source"]["training_release_receipt_id"]
        ),
        "training_capture": dict(receipt["source"]["training_capture"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Admit exact manifest V2 compartment SAE outputs."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--execution-blueprint", type=Path, required=True)
    parser.add_argument("--training-release-receipt", type=Path, required=True)
    parser.add_argument("--compute-authorization-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    observed = admit_run_directory_v2(
        run_dir=args.run_dir,
        project_root=args.project_root.resolve(),
        training_plan_path=args.training_plan,
        execution_blueprint_path=args.execution_blueprint,
        training_release_receipt_path=args.training_release_receipt,
        compute_authorization_receipt_path=args.compute_authorization_receipt,
        output_path=None,
    )
    if args.verify is not None:
        expected = json.loads(args.verify.read_text(encoding="utf-8"))
        if observed != expected:
            raise ValueError("dictionary bundle V2 receipt is not reproducible")
    elif args.output is not None:
        write_json(args.output, observed)
    else:
        parser.error("use --output or --verify")
    print(
        json.dumps(
            {
                "schema": "sch.vpd-compartment-sae-dictionary-bundle-cli.v2",
                "status": observed["status"],
                "dictionary_bundle_id": observed["dictionary_bundle_id"],
                "dictionaries": len(observed["dictionaries"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
