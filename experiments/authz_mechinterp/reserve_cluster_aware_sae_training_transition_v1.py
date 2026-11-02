from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.admit_cluster_aware_sae_training_v1 import (
    ADMISSION_SCHEMA,
    build_dry_run_admission,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    _existing_ancestor_has_reparse_point,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.run_cluster_aware_sae_training_job_v1 import (
    ClusterAwareSAETrainingLedger,
    reserve_cluster_aware_sae_job_before_runtime_attach,
    select_execution_job,
    validate_execution_blueprint_content,
)


RESERVATION_PARENT = "artifacts/vpd_cluster_aware_sae_training_reservations"
CAMPAIGN_RECEIPT_NAME = "release_campaign_reservation.json"
JOB_RECEIPT_NAME = "job_transition_reservation.json"
CAMPAIGN_SCHEMA = "sch.vpd-cluster-aware-sae-release-campaign-reservation.v1"
TRANSITION_SCHEMA = "sch.vpd-cluster-aware-sae-job-transition-reservation.v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("cluster-aware SAE transition input must be an object")
    return value


def _content_id(value: Any, *, field_name: str) -> str:
    observed = str(value)
    if len(observed) != 64 or any(
        character not in "0123456789abcdef" for character in observed
    ):
        raise ValueError(f"{field_name} is not a lowercase SHA-256 content ID")
    return observed


def _reservation_parent(project_root: Path) -> Path:
    root = project_root.resolve()
    parent = (root / RESERVATION_PARENT).resolve()
    if parent.parent != (root / "artifacts").resolve() or not parent.is_dir():
        raise ValueError("cluster-aware SAE reservation parent is absent or misplaced")
    if _existing_ancestor_has_reparse_point(parent, stop=root):
        raise ValueError("cluster-aware SAE reservation parent crosses a reparse point")
    return parent


def _input_record(path: Path, *, project_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_relative_to(project_root.resolve()) or not resolved.is_file():
        raise ValueError("cluster-aware SAE transition input escapes the project")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _verify_addressed_receipt(
    value: dict[str, Any], *, schema: str, id_field: str, label: str
) -> str:
    if value.get("schema") != schema:
        raise ValueError(f"{label} schema differs")
    body = dict(value)
    identifier = _content_id(body.pop(id_field, ""), field_name=f"{label} ID")
    if identifier != content_sha256(body):
        raise ValueError(f"{label} content address differs")
    return identifier


def _initialize_or_verify_campaign(
    *,
    parent: Path,
    admission: dict[str, Any],
    release_receipt: dict[str, Any],
    release_registry: dict[str, Any],
    simulation_only: bool,
) -> tuple[dict[str, Any], Path, bool]:
    nonce = _content_id(release_receipt.get("nonce", ""), field_name="release nonce")
    release_id = _content_id(
        release_receipt.get("receipt_id", ""), field_name="release receipt ID"
    )
    campaign_root = parent / f"release-nonce-{nonce}"
    created = False
    try:
        campaign_root.mkdir(parents=False, exist_ok=False)
        created = True
    except FileExistsError:
        pass
    receipt_path = campaign_root / CAMPAIGN_RECEIPT_NAME
    if created:
        body = {
            "schema": CAMPAIGN_SCHEMA,
            "status": (
                "reserved_synthetic_release_campaign_no_execution"
                if simulation_only
                else "reserved_real_release_campaign_no_execution"
            ),
            "release_nonce": nonce,
            "scientific_release_receipt_id": release_id,
            "scientific_release_receipt_sha256": content_sha256(release_receipt),
            "release_authority_registry_id": admission[
                "scientific_release_verification"
            ]["authority_registry_id"],
            "release_authority_registry_sha256": content_sha256(release_registry),
            "execution_blueprint_id": admission["execution_blueprint_id"],
            "simulation_only": simulation_only,
            "release_nonce_consumed": True,
            "job_execution_authorized": False,
            "runtime_attached": False,
            "trainer_callback_invoked": False,
        }
        campaign = {**body, "campaign_reservation_id": content_sha256(body)}
        write_json(receipt_path, campaign)
    else:
        if not receipt_path.is_file():
            raise ValueError("release campaign nonce has an incomplete terminal tombstone")
        campaign = _load(receipt_path)
        _verify_addressed_receipt(
            campaign,
            schema=CAMPAIGN_SCHEMA,
            id_field="campaign_reservation_id",
            label="release campaign reservation",
        )
        expected = {
            "release_nonce": nonce,
            "scientific_release_receipt_id": release_id,
            "scientific_release_receipt_sha256": content_sha256(release_receipt),
            "release_authority_registry_id": admission[
                "scientific_release_verification"
            ]["authority_registry_id"],
            "release_authority_registry_sha256": content_sha256(release_registry),
            "execution_blueprint_id": admission["execution_blueprint_id"],
            "simulation_only": simulation_only,
            "release_nonce_consumed": True,
            "job_execution_authorized": False,
            "runtime_attached": False,
            "trainer_callback_invoked": False,
        }
        if any(campaign.get(field) != value for field, value in expected.items()):
            raise ValueError("release campaign nonce is bound to another campaign")
    return campaign, receipt_path, created


def reserve_training_transition(
    *,
    project_root: Path,
    blueprint_path: Path,
    execution_job_id: str,
    device: str,
    release_receipt_path: Path,
    release_registry_path: Path,
    compute_receipt_path: Path,
    compute_registry_path: Path,
    as_of: str,
    simulation_only: bool,
) -> dict[str, Any]:
    root = project_root.resolve()
    parent = _reservation_parent(root)
    input_paths = {
        "execution_blueprint": blueprint_path,
        "scientific_release_receipt": release_receipt_path,
        "scientific_release_registry": release_registry_path,
        "compute_authorization_receipt": compute_receipt_path,
        "compute_authority_registry": compute_registry_path,
    }
    input_records = {
        name: _input_record(path, project_root=root)
        for name, path in input_paths.items()
    }
    blueprint = _load(blueprint_path)
    blueprint_check = validate_execution_blueprint_content(blueprint)
    job = select_execution_job(blueprint, execution_job_id=execution_job_id)
    release_receipt = _load(release_receipt_path)
    release_registry = _load(release_registry_path)
    compute_receipt = _load(compute_receipt_path)
    compute_registry = _load(compute_registry_path)
    admission = build_dry_run_admission(
        blueprint=blueprint,
        blueprint_sha256=sha256_file(blueprint_path),
        execution_job_id=execution_job_id,
        device=device,
        release_receipt=release_receipt,
        release_registry=release_registry,
        compute_receipt=compute_receipt,
        compute_registry=compute_registry,
        as_of=as_of,
    )
    synthetic = bool(admission["scientific_boundary"]["synthetic_fixture_only"])
    if simulation_only is not synthetic:
        raise ValueError("cluster-aware SAE transition simulation boundary differs")
    campaign, campaign_receipt_path, campaign_created = _initialize_or_verify_campaign(
        parent=parent,
        admission=admission,
        release_receipt=release_receipt,
        release_registry=release_registry,
        simulation_only=simulation_only,
    )
    compute_nonce = _content_id(
        compute_receipt.get("nonce", ""), field_name="compute nonce"
    )
    job_root = parent / f"compute-nonce-{compute_nonce}"
    try:
        job_root.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError("compute nonce was already consumed") from exc

    ledger = ClusterAwareSAETrainingLedger(
        execution_blueprint_id=blueprint_check["execution_blueprint_id"]
    )
    reservation = reserve_cluster_aware_sae_job_before_runtime_attach(
        blueprint=blueprint,
        execution_job_id=execution_job_id,
        release_verification=admission["scientific_release_verification"],
        compute_authorization_verification=admission[
            "compute_authorization_verification"
        ],
        ledger=ledger,
        simulation_only=simulation_only,
    )
    del reservation
    exact_output_path = (job_root / str(job["output_dir"])).resolve()
    if not exact_output_path.is_relative_to(job_root):
        raise ValueError("cluster-aware SAE exact output escapes its reserved job root")
    body = {
        "schema": TRANSITION_SCHEMA,
        "status": (
            "reserved_synthetic_job_no_runtime_or_training"
            if simulation_only
            else "reserved_real_job_runtime_not_attached"
        ),
        "source": input_records,
        "as_of": as_of,
        "admission": {
            "schema": ADMISSION_SCHEMA,
            "admission_id": admission["admission_id"],
            "scientific_release_receipt_id": admission[
                "scientific_release_verification"
            ]["scientific_release_receipt_id"],
            "compute_authorization_id": admission[
                "compute_authorization_verification"
            ]["compute_authorization_id"],
        },
        "campaign": {
            "campaign_reservation_id": campaign["campaign_reservation_id"],
            "campaign_receipt_path": str(campaign_receipt_path.resolve()),
            "campaign_created_by_this_transition": campaign_created,
            "release_nonce": campaign["release_nonce"],
            "release_nonce_consumed": True,
        },
        "job": {
            "execution_blueprint_id": blueprint_check["execution_blueprint_id"],
            "execution_job_id": job["execution_job_id"],
            "source_training_job_id": job["source_training_job_id"],
            "compartment_id": job["compartment_id"],
            "compartment_contract_id": job["compartment_contract_id"],
            "global_pair_split_id": job["global_pair_split_id"],
            "exact_sample_contract_sha256": content_sha256(
                job["exact_sample_contract"]
            ),
            "device": device,
            "compute_nonce": compute_nonce,
            "compute_nonce_consumed": True,
            "reserved_job_root": str(job_root.resolve()),
            "reserved_exact_output_path": str(exact_output_path),
            "output_path_created": False,
        },
        "worker_ledger_after_reservation": ledger.snapshot(),
        "transition": {
            "atomic_compute_nonce_and_job_root_reservation": True,
            "release_campaign_nonce_is_durable": True,
            "partial_failure_leaves_terminal_tombstone": True,
            "rollback_or_resume_implemented": False,
            "worker_ledger_reservation_performed": True,
            "runtime_attached": False,
            "trainer_callback_invoked": False,
        },
        "scientific_boundary": {
            "simulation_only": simulation_only,
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "persistent_dictionary_output_created": False,
            "weights_loaded_or_mutated": False,
            "runtime_attached": False,
            "trainer_callback_invoked": False,
            "execution_authorized_by_transition": False,
            "sae_training_authorized_by_transition": False,
            "real_actions_authorized": False,
        },
    }
    receipt = {**body, "transition_reservation_id": content_sha256(body)}
    receipt_path = job_root / JOB_RECEIPT_NAME
    if receipt_path.exists():
        raise ValueError("cluster-aware SAE job transition receipt already exists")
    write_json(receipt_path, receipt)
    return receipt


def verify_training_transition(
    receipt_path: Path, *, project_root: Path
) -> dict[str, Any]:
    root = project_root.resolve()
    parent = _reservation_parent(root)
    resolved_receipt = receipt_path.resolve()
    if resolved_receipt.name != JOB_RECEIPT_NAME or not resolved_receipt.is_file():
        raise ValueError("cluster-aware SAE job transition receipt path differs")
    job_root = resolved_receipt.parent
    if job_root.parent != parent or not job_root.name.startswith("compute-nonce-"):
        raise ValueError("cluster-aware SAE job transition root differs")
    receipt = _load(resolved_receipt)
    transition_id = _verify_addressed_receipt(
        receipt,
        schema=TRANSITION_SCHEMA,
        id_field="transition_reservation_id",
        label="cluster-aware SAE job transition",
    )
    for record in receipt.get("source", {}).values():
        source_path = Path(str(record.get("path", ""))).resolve()
        if (
            not source_path.is_relative_to(root)
            or not source_path.is_file()
            or sha256_file(source_path) != record.get("sha256")
        ):
            raise ValueError("cluster-aware SAE job transition source bytes differ")
    campaign_path = Path(receipt["campaign"]["campaign_receipt_path"]).resolve()
    if campaign_path.parent.parent != parent or campaign_path.name != CAMPAIGN_RECEIPT_NAME:
        raise ValueError("cluster-aware SAE release campaign path differs")
    campaign = _load(campaign_path)
    campaign_id = _verify_addressed_receipt(
        campaign,
        schema=CAMPAIGN_SCHEMA,
        id_field="campaign_reservation_id",
        label="release campaign reservation",
    )
    job = receipt.get("job", {})
    compute_nonce = _content_id(job.get("compute_nonce", ""), field_name="compute nonce")
    exact_output = Path(str(job.get("reserved_exact_output_path", ""))).resolve()
    ledger = receipt.get("worker_ledger_after_reservation", {})
    if (
        job_root.name != f"compute-nonce-{compute_nonce}"
        or job.get("reserved_job_root") != str(job_root)
        or not exact_output.is_relative_to(job_root)
        or exact_output.exists()
        or campaign_id != receipt["campaign"]["campaign_reservation_id"]
        or campaign.get("release_nonce") != receipt["campaign"]["release_nonce"]
        or campaign.get("execution_blueprint_id")
        != job.get("execution_blueprint_id")
        or ledger.get("execution_blueprint_id") != job.get("execution_blueprint_id")
        or ledger.get("reservations", {}).get(job.get("execution_job_id"))
        != "reserved"
        or ledger.get("jobs_reserved") != 1
        or receipt.get("transition", {}).get(
            "atomic_compute_nonce_and_job_root_reservation"
        )
        is not True
        or receipt.get("transition", {}).get("worker_ledger_reservation_performed")
        is not True
        or any(
            receipt.get("scientific_boundary", {}).get(field_name) not in (False, 0)
            for field_name in (
                "model_loaded",
                "model_inference_performed",
                "persistent_dictionary_output_created",
                "weights_loaded_or_mutated",
                "runtime_attached",
                "trainer_callback_invoked",
                "execution_authorized_by_transition",
                "sae_training_authorized_by_transition",
                "real_actions_authorized",
            )
        )
    ):
        raise ValueError("cluster-aware SAE job transition current state differs")
    return {
        "schema": "sch.vpd-cluster-aware-sae-job-transition-verification.v1",
        "status": "pass",
        "transition_reservation_id": transition_id,
        "campaign_reservation_id": campaign_id,
        "execution_blueprint_id": job["execution_blueprint_id"],
        "execution_job_id": job["execution_job_id"],
        "release_nonce_consumed": True,
        "compute_nonce_consumed": True,
        "job_root_reserved": True,
        "worker_ledger_reservation_performed": True,
        "runtime_attached": False,
        "trainer_callback_invoked": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reserve a signed cluster-aware SAE job transition without loading a "
            "runtime or invoking training."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--reserve", action="store_true")
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--execution-blueprint", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--device")
    parser.add_argument("--scientific-release-receipt", type=Path)
    parser.add_argument("--scientific-release-registry", type=Path)
    parser.add_argument("--compute-authorization-receipt", type=Path)
    parser.add_argument("--compute-authority-registry", type=Path)
    parser.add_argument("--as-of")
    parser.add_argument("--simulation-only", action="store_true")
    args = parser.parse_args()
    if args.reserve:
        required = (
            args.execution_blueprint,
            args.job_id,
            args.device,
            args.scientific_release_receipt,
            args.scientific_release_registry,
            args.compute_authorization_receipt,
            args.compute_authority_registry,
            args.as_of,
        )
        if any(value is None for value in required):
            parser.error("--reserve requires all blueprint, job, receipt, registry, device, and time inputs")
        result = reserve_training_transition(
            project_root=args.project_root,
            blueprint_path=args.execution_blueprint,
            execution_job_id=args.job_id,
            device=args.device,
            release_receipt_path=args.scientific_release_receipt,
            release_registry_path=args.scientific_release_registry,
            compute_receipt_path=args.compute_authorization_receipt,
            compute_registry_path=args.compute_authority_registry,
            as_of=args.as_of,
            simulation_only=args.simulation_only,
        )
    else:
        if args.simulation_only:
            parser.error("--simulation-only is valid only with --reserve")
        result = verify_training_transition(args.verify, project_root=args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
