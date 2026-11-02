from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.admit_cluster_aware_sae_training_v1 import (
    build_dry_run_admission,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.reserve_cluster_aware_sae_training_transition_v1 import (
    JOB_RECEIPT_NAME,
    TRANSITION_SCHEMA,
    verify_training_transition,
)
from experiments.authz_mechinterp.run_cluster_aware_sae_training_job_v1 import (
    ClusterAwareSAETrainingLedger,
    ClusterAwareSAETrainingRuntime,
    ClusterAwareSAETrainingSession,
    reserve_cluster_aware_sae_job_before_runtime_attach,
)


CLAIM_DIR_NAME = "runtime-attachment-claim"
CLAIM_RECEIPT_NAME = "inert_runtime_attachment_claim.json"
CLAIM_SCHEMA = "sch.vpd-cluster-aware-sae-inert-runtime-attachment-claim.v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("cluster-aware SAE inert attachment input must be an object")
    return value


def _content_id(value: Any, *, field_name: str) -> str:
    observed = str(value)
    if len(observed) != 64 or any(
        character not in "0123456789abcdef" for character in observed
    ):
        raise ValueError(f"{field_name} is not a lowercase SHA-256 content ID")
    return observed


def _source_path(
    transition: dict[str, Any], *, field_name: str, project_root: Path
) -> Path:
    record = transition.get("source", {}).get(field_name, {})
    path = Path(str(record.get("path", ""))).resolve()
    if (
        not path.is_relative_to(project_root.resolve())
        or not path.is_file()
        or sha256_file(path) != record.get("sha256")
    ):
        raise ValueError(f"cluster-aware SAE transition source differs: {field_name}")
    return path


def _reconstruct_reservation(
    *, transition_receipt_path: Path, project_root: Path
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    transition_verification = verify_training_transition(
        transition_receipt_path, project_root=project_root
    )
    transition = _load(transition_receipt_path)
    if transition.get("schema") != TRANSITION_SCHEMA:
        raise ValueError("cluster-aware SAE transition schema differs")
    if transition.get("scientific_boundary", {}).get("simulation_only") is not True:
        raise ValueError("inert runtime attachment accepts synthetic transitions only")
    paths = {
        field_name: _source_path(
            transition, field_name=field_name, project_root=project_root
        )
        for field_name in (
            "execution_blueprint",
            "scientific_release_receipt",
            "scientific_release_registry",
            "compute_authorization_receipt",
            "compute_authority_registry",
        )
    }
    blueprint = _load(paths["execution_blueprint"])
    job_record = transition["job"]
    admission = build_dry_run_admission(
        blueprint=blueprint,
        blueprint_sha256=sha256_file(paths["execution_blueprint"]),
        execution_job_id=str(job_record["execution_job_id"]),
        device=str(job_record["device"]),
        release_receipt=_load(paths["scientific_release_receipt"]),
        release_registry=_load(paths["scientific_release_registry"]),
        compute_receipt=_load(paths["compute_authorization_receipt"]),
        compute_registry=_load(paths["compute_authority_registry"]),
        as_of=str(transition["as_of"]),
    )
    if (
        admission["admission_id"] != transition["admission"]["admission_id"]
        or admission["execution_blueprint_id"]
        != transition_verification["execution_blueprint_id"]
        or admission["execution_job_id"] != transition_verification["execution_job_id"]
    ):
        raise ValueError("cluster-aware SAE transition admission reconstruction differs")
    ledger = ClusterAwareSAETrainingLedger(
        execution_blueprint_id=transition_verification["execution_blueprint_id"]
    )
    reservation = reserve_cluster_aware_sae_job_before_runtime_attach(
        blueprint=blueprint,
        execution_job_id=transition_verification["execution_job_id"],
        release_verification=admission["scientific_release_verification"],
        compute_authorization_verification=admission[
            "compute_authorization_verification"
        ],
        ledger=ledger,
        simulation_only=True,
    )
    if ledger.snapshot() != transition["worker_ledger_after_reservation"]:
        raise ValueError("cluster-aware SAE transition worker ledger differs")
    return transition, reservation, admission


@dataclass
class ClaimedInertRuntimeSession:
    claim: dict[str, Any]
    claim_receipt_path: Path
    _session: ClusterAwareSAETrainingSession

    def close(self) -> None:
        self._session.close()

    @property
    def callback_invoked(self) -> bool:
        return self._session.invoked


RuntimeFactory = Callable[[], ClusterAwareSAETrainingRuntime]


def claim_inert_runtime_attachment(
    *,
    transition_receipt_path: Path,
    project_root: Path,
    runtime_factory: RuntimeFactory,
) -> ClaimedInertRuntimeSession:
    root = project_root.resolve()
    transition_path = transition_receipt_path.resolve()
    transition, reservation, admission = _reconstruct_reservation(
        transition_receipt_path=transition_path, project_root=root
    )
    job_root = transition_path.parent
    if transition_path.name != JOB_RECEIPT_NAME or job_root != Path(
        transition["job"]["reserved_job_root"]
    ).resolve():
        raise ValueError("cluster-aware SAE transition job root differs")
    claim_root = job_root / CLAIM_DIR_NAME
    try:
        claim_root.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError("cluster-aware SAE runtime attachment was already claimed") from exc
    runtime: ClusterAwareSAETrainingRuntime | None = None
    try:
        runtime = runtime_factory()
        if not isinstance(runtime, ClusterAwareSAETrainingRuntime):
            raise TypeError("cluster-aware SAE runtime factory returned another type")
        session = reservation.attach_runtime(runtime)
    except Exception:
        if runtime is not None:
            runtime.close()
        raise
    body = {
        "schema": CLAIM_SCHEMA,
        "status": "inert_runtime_attached_trainer_callback_not_invoked",
        "source": {
            "transition_receipt_path": str(transition_path),
            "transition_receipt_sha256": sha256_file(transition_path),
            "transition_reservation_id": transition["transition_reservation_id"],
            "admission_id": admission["admission_id"],
        },
        "execution_blueprint_id": transition["job"]["execution_blueprint_id"],
        "execution_job_id": transition["job"]["execution_job_id"],
        "reserved_job_root": str(job_root),
        "claim_root": str(claim_root.resolve()),
        "runtime": {
            "runtime_kind": runtime.runtime_kind,
            "actual_model_loaded": runtime.actual_model_loaded,
            "actual_training_enabled": runtime.actual_training_enabled,
            "persistent_output_enabled": runtime.persistent_output_enabled,
            "attachment_claimed_after_durable_transition": True,
            "trainer_callback_invoked": False,
        },
        "worker": {
            "ledger_reconstructed_exactly": True,
            "runtime_attachment_single_use": True,
            "worker_session_created": True,
            "trainer_callback_invoked": False,
        },
        "scientific_boundary": {
            "synthetic_inert_runtime_only": True,
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "persistent_dictionary_output_created": False,
            "weights_loaded_or_mutated": False,
            "inert_runtime_attached": True,
            "production_runtime_attached": False,
            "trainer_callback_invoked": False,
            "execution_authorized_by_claim": False,
            "sae_training_authorized_by_claim": False,
            "real_actions_authorized": False,
        },
    }
    claim = {**body, "attachment_claim_id": content_sha256(body)}
    claim_receipt_path = claim_root / CLAIM_RECEIPT_NAME
    if claim_receipt_path.exists():
        runtime.close()
        raise ValueError("cluster-aware SAE inert attachment receipt already exists")
    write_json(claim_receipt_path, claim)
    return ClaimedInertRuntimeSession(
        claim=claim,
        claim_receipt_path=claim_receipt_path,
        _session=session,
    )


def verify_inert_runtime_attachment_claim(
    receipt_path: Path, *, project_root: Path
) -> dict[str, Any]:
    root = project_root.resolve()
    resolved = receipt_path.resolve()
    if resolved.name != CLAIM_RECEIPT_NAME or not resolved.is_file():
        raise ValueError("cluster-aware SAE inert attachment receipt path differs")
    claim_root = resolved.parent
    job_root = claim_root.parent
    if claim_root.name != CLAIM_DIR_NAME:
        raise ValueError("cluster-aware SAE inert attachment claim root differs")
    claim = _load(resolved)
    if claim.get("schema") != CLAIM_SCHEMA:
        raise ValueError("cluster-aware SAE inert attachment claim schema differs")
    body = dict(claim)
    claim_id = _content_id(
        body.pop("attachment_claim_id", ""), field_name="attachment claim ID"
    )
    if claim_id != content_sha256(body):
        raise ValueError("cluster-aware SAE inert attachment claim address differs")
    source = claim.get("source", {})
    transition_path = Path(str(source.get("transition_receipt_path", ""))).resolve()
    if (
        transition_path != job_root / JOB_RECEIPT_NAME
        or not transition_path.is_relative_to(root)
        or sha256_file(transition_path) != source.get("transition_receipt_sha256")
    ):
        raise ValueError("cluster-aware SAE inert attachment transition bytes differ")
    transition_verification = verify_training_transition(
        transition_path, project_root=root
    )
    runtime = claim.get("runtime", {})
    worker = claim.get("worker", {})
    boundary = claim.get("scientific_boundary", {})
    if (
        source.get("transition_reservation_id")
        != transition_verification["transition_reservation_id"]
        or claim.get("execution_blueprint_id")
        != transition_verification["execution_blueprint_id"]
        or claim.get("execution_job_id") != transition_verification["execution_job_id"]
        or claim.get("reserved_job_root") != str(job_root)
        or claim.get("claim_root") != str(claim_root)
        or runtime.get("actual_model_loaded") is not False
        or runtime.get("actual_training_enabled") is not False
        or runtime.get("persistent_output_enabled") is not False
        or runtime.get("attachment_claimed_after_durable_transition") is not True
        or runtime.get("trainer_callback_invoked") is not False
        or worker.get("ledger_reconstructed_exactly") is not True
        or worker.get("runtime_attachment_single_use") is not True
        or worker.get("worker_session_created") is not True
        or worker.get("trainer_callback_invoked") is not False
        or boundary.get("synthetic_inert_runtime_only") is not True
        or boundary.get("inert_runtime_attached") is not True
        or any(
            boundary.get(field_name) not in (False, 0)
            for field_name in (
                "model_loaded",
                "model_inference_performed",
                "persistent_dictionary_output_created",
                "weights_loaded_or_mutated",
                "production_runtime_attached",
                "trainer_callback_invoked",
                "execution_authorized_by_claim",
                "sae_training_authorized_by_claim",
                "real_actions_authorized",
            )
        )
    ):
        raise ValueError("cluster-aware SAE inert attachment current state differs")
    return {
        "schema": "sch.vpd-cluster-aware-sae-inert-runtime-attachment-verification.v1",
        "status": "pass",
        "attachment_claim_id": claim_id,
        "transition_reservation_id": transition_verification[
            "transition_reservation_id"
        ],
        "execution_blueprint_id": transition_verification["execution_blueprint_id"],
        "execution_job_id": transition_verification["execution_job_id"],
        "ledger_reconstructed_exactly": True,
        "inert_runtime_attached": True,
        "production_runtime_attached": False,
        "trainer_callback_invoked": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify an inert cluster-aware SAE runtime attachment claim. Runtime "
            "construction is injection-only and has no production CLI."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--verify", type=Path, required=True)
    args = parser.parse_args()
    result = verify_inert_runtime_attachment_claim(
        args.verify, project_root=args.project_root
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
