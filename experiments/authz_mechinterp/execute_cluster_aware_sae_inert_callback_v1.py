from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.attach_cluster_aware_sae_inert_runtime_v1 import (
    CLAIM_RECEIPT_NAME,
    ClaimedInertRuntimeSession,
    verify_inert_runtime_attachment_claim,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.run_cluster_aware_sae_training_job_v1 import (
    WORKER_RECEIPT_SCHEMA,
)


# These names are deliberately short because the parent reservation namespace
# already contains a 64-character nonce and must remain usable on Windows.
EXECUTION_CLAIM_DIR_NAME = "callback-claim"
SUCCESS_RECEIPT_NAME = "success.json"
FAILURE_RECEIPT_NAME = "failure.json"
SUCCESS_SCHEMA = "sch.vpd-cluster-aware-sae-synthetic-callback-execution.v1"
FAILURE_SCHEMA = "sch.vpd-cluster-aware-sae-synthetic-callback-failure.v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("cluster-aware SAE synthetic callback receipt must be an object")
    return value


def _content_id(value: Any, *, field_name: str) -> str:
    observed = str(value)
    if len(observed) != 64 or any(
        character not in "0123456789abcdef" for character in observed
    ):
        raise ValueError(f"{field_name} is not a lowercase SHA-256 content ID")
    return observed


@dataclass
class SyntheticCallbackExecutionError(RuntimeError):
    receipt_path: Path
    cause_type: str

    def __str__(self) -> str:
        return (
            "synthetic cluster-aware SAE callback failed after durable charge; "
            f"failure receipt: {self.receipt_path}"
        )


def _source(
    claimed: ClaimedInertRuntimeSession, claim_verification: dict[str, Any]
) -> dict[str, Any]:
    return {
        "attachment_claim_path": str(claimed.claim_receipt_path.resolve()),
        "attachment_claim_sha256": sha256_file(claimed.claim_receipt_path),
        "attachment_claim_id": claim_verification["attachment_claim_id"],
        "transition_reservation_id": claim_verification["transition_reservation_id"],
    }


def _boundary(*, callback_failed: bool) -> dict[str, Any]:
    return {
        "synthetic_inert_callback_only": True,
        "callback_invoked": True,
        "callback_failed": callback_failed,
        "callback_charge_persisted": True,
        "model_framework_imported": False,
        "model_loaded": False,
        "model_inference_performed": False,
        "training_performed": False,
        "persistent_dictionary_output_created": False,
        "weights_loaded_or_mutated": False,
        "production_runtime_attached": False,
        "execution_authorized_by_receipt": False,
        "sae_training_authorized_by_receipt": False,
        "real_actions_authorized": False,
    }


def execute_claimed_inert_callback(
    *, claimed: ClaimedInertRuntimeSession, project_root: Path
) -> dict[str, Any]:
    root = project_root.resolve()
    claim_path = claimed.claim_receipt_path.resolve()
    claim_verification = verify_inert_runtime_attachment_claim(
        claim_path, project_root=root
    )
    if claimed.callback_invoked:
        raise ValueError("cluster-aware SAE synthetic callback was already invoked")
    execution_claim_root = claim_path.parent / EXECUTION_CLAIM_DIR_NAME
    try:
        execution_claim_root.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError(
            "cluster-aware SAE synthetic callback execution was already claimed"
        ) from exc
    try:
        # The claimed session is deliberately an internal handoff object.  This
        # wrapper owns the durable execution claim before invoking its worker.
        worker_receipt = claimed._session.run()
    except Exception as exc:
        claimed.close()
        body = {
            "schema": FAILURE_SCHEMA,
            "status": "failed_synthetic_callback_charged_terminal",
            "source": _source(claimed, claim_verification),
            "execution_blueprint_id": claim_verification["execution_blueprint_id"],
            "execution_job_id": claim_verification["execution_job_id"],
            "failure": {
                "exception_type": type(exc).__name__,
                "message_persisted": False,
                "retry_authorized": False,
            },
            "ledger_after": claimed._session.ledger.snapshot(),
            "runtime_closed": True,
            "scientific_boundary": _boundary(callback_failed=True),
        }
        receipt = {**body, "synthetic_callback_failure_id": content_sha256(body)}
        receipt_path = execution_claim_root / FAILURE_RECEIPT_NAME
        write_json(receipt_path, receipt)
        raise SyntheticCallbackExecutionError(
            receipt_path=receipt_path, cause_type=type(exc).__name__
        ) from exc
    claimed.close()
    body = {
        "schema": SUCCESS_SCHEMA,
        "status": "complete_synthetic_callback_no_training",
        "source": _source(claimed, claim_verification),
        "execution_blueprint_id": claim_verification["execution_blueprint_id"],
        "execution_job_id": claim_verification["execution_job_id"],
        "runtime_worker_receipt": worker_receipt,
        "ledger_after": claimed._session.ledger.snapshot(),
        "runtime_closed": True,
        "scientific_boundary": _boundary(callback_failed=False),
    }
    receipt = {**body, "synthetic_callback_execution_id": content_sha256(body)}
    write_json(execution_claim_root / SUCCESS_RECEIPT_NAME, receipt)
    return receipt


def _verify_source(
    receipt: dict[str, Any], *, receipt_path: Path, project_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = project_root.resolve()
    resolved = receipt_path.resolve()
    execution_claim_root = resolved.parent
    if execution_claim_root.name != EXECUTION_CLAIM_DIR_NAME:
        raise ValueError("cluster-aware SAE synthetic callback claim root differs")
    claim_path = Path(
        str(receipt.get("source", {}).get("attachment_claim_path", ""))
    ).resolve()
    if (
        claim_path.name != CLAIM_RECEIPT_NAME
        or claim_path.parent != execution_claim_root.parent
        or not claim_path.is_relative_to(root)
        or sha256_file(claim_path)
        != receipt.get("source", {}).get("attachment_claim_sha256")
    ):
        raise ValueError("cluster-aware SAE synthetic callback attachment bytes differ")
    claim_verification = verify_inert_runtime_attachment_claim(
        claim_path, project_root=root
    )
    if (
        receipt.get("source", {}).get("attachment_claim_id")
        != claim_verification["attachment_claim_id"]
        or receipt.get("source", {}).get("transition_reservation_id")
        != claim_verification["transition_reservation_id"]
        or receipt.get("execution_blueprint_id")
        != claim_verification["execution_blueprint_id"]
        or receipt.get("execution_job_id")
        != claim_verification["execution_job_id"]
    ):
        raise ValueError("cluster-aware SAE synthetic callback lineage differs")
    return claim_verification, _load(claim_path)


def _verify_boundary(receipt: dict[str, Any], *, callback_failed: bool) -> None:
    boundary = receipt.get("scientific_boundary", {})
    if (
        boundary.get("synthetic_inert_callback_only") is not True
        or boundary.get("callback_invoked") is not True
        or boundary.get("callback_failed") is not callback_failed
        or boundary.get("callback_charge_persisted") is not True
        or receipt.get("runtime_closed") is not True
        or any(
            boundary.get(field_name) not in (False, 0)
            for field_name in (
                "model_framework_imported",
                "model_loaded",
                "model_inference_performed",
                "training_performed",
                "persistent_dictionary_output_created",
                "weights_loaded_or_mutated",
                "production_runtime_attached",
                "execution_authorized_by_receipt",
                "sae_training_authorized_by_receipt",
                "real_actions_authorized",
            )
        )
    ):
        raise ValueError("cluster-aware SAE synthetic callback boundary differs")


def verify_synthetic_callback_execution(
    receipt_path: Path, *, project_root: Path
) -> dict[str, Any]:
    resolved = receipt_path.resolve()
    if resolved.name != SUCCESS_RECEIPT_NAME or not resolved.is_file():
        raise ValueError("cluster-aware SAE synthetic callback receipt path differs")
    receipt = _load(resolved)
    if receipt.get("schema") != SUCCESS_SCHEMA:
        raise ValueError("cluster-aware SAE synthetic callback schema differs")
    body = dict(receipt)
    receipt_id = _content_id(
        body.pop("synthetic_callback_execution_id", ""),
        field_name="synthetic callback execution ID",
    )
    if receipt_id != content_sha256(body):
        raise ValueError("cluster-aware SAE synthetic callback address differs")
    _verify_source(receipt, receipt_path=resolved, project_root=project_root)
    _verify_boundary(receipt, callback_failed=False)
    worker = receipt.get("runtime_worker_receipt", {})
    worker_body = dict(worker)
    worker_id = _content_id(
        worker_body.pop("runtime_worker_receipt_id", ""),
        field_name="runtime worker receipt ID",
    )
    ledger = receipt.get("ledger_after", {})
    job_id = str(receipt["execution_job_id"])
    callback_result = worker.get("callback_result", {})
    if (
        worker_id != content_sha256(worker_body)
        or worker.get("schema") != WORKER_RECEIPT_SCHEMA
        or worker.get("status") != "complete_synthetic_callback_no_training"
        or worker.get("execution_blueprint_id") != receipt["execution_blueprint_id"]
        or worker.get("execution_job_id") != job_id
        or worker.get("ledger_after") != ledger
        or ledger.get("reservations", {}).get(job_id) != "completed"
        or int(ledger.get("trainer_calls", -1)) != 1
        or callback_result.get("status") != "synthetic_completed_no_training"
        or callback_result.get("execution_job_id") != job_id
        or callback_result.get("model_loaded") is not False
        or callback_result.get("weights_mutated") is not False
        or callback_result.get("persistent_output_written") is not False
        or callback_result.get("dictionary_id") is not None
        or worker.get("scientific_boundary", {}).get("simulation_only") is not True
        or any(
            worker.get("scientific_boundary", {}).get(field_name) not in (False, 0)
            for field_name in (
                "model_loaded",
                "training_performed",
                "persistent_output_written",
                "base_model_weights_mutated",
                "real_actions",
                "receipt_authorizes_additional_execution",
            )
        )
    ):
        raise ValueError("cluster-aware SAE synthetic callback worker receipt differs")
    return {
        "schema": "sch.vpd-cluster-aware-sae-synthetic-callback-execution-verification.v1",
        "status": "pass",
        "synthetic_callback_execution_id": receipt_id,
        "runtime_worker_receipt_id": worker_id,
        "execution_blueprint_id": receipt["execution_blueprint_id"],
        "execution_job_id": job_id,
        "callback_invoked": True,
        "callback_failed": False,
        "callback_charge_persisted": True,
        "model_loaded": False,
        "training_performed": False,
        "persistent_output_written": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def verify_synthetic_callback_failure(
    receipt_path: Path, *, project_root: Path
) -> dict[str, Any]:
    resolved = receipt_path.resolve()
    if resolved.name != FAILURE_RECEIPT_NAME or not resolved.is_file():
        raise ValueError("cluster-aware SAE synthetic callback failure path differs")
    receipt = _load(resolved)
    if receipt.get("schema") != FAILURE_SCHEMA:
        raise ValueError("cluster-aware SAE synthetic callback failure schema differs")
    body = dict(receipt)
    receipt_id = _content_id(
        body.pop("synthetic_callback_failure_id", ""),
        field_name="synthetic callback failure ID",
    )
    if receipt_id != content_sha256(body):
        raise ValueError("cluster-aware SAE synthetic callback failure address differs")
    _verify_source(receipt, receipt_path=resolved, project_root=project_root)
    _verify_boundary(receipt, callback_failed=True)
    ledger = receipt.get("ledger_after", {})
    job_id = str(receipt["execution_job_id"])
    failure = receipt.get("failure", {})
    if (
        receipt.get("status") != "failed_synthetic_callback_charged_terminal"
        or ledger.get("reservations", {}).get(job_id) != "failed"
        or int(ledger.get("trainer_calls", -1)) != 1
        or not str(failure.get("exception_type", ""))
        or failure.get("message_persisted") is not False
        or failure.get("retry_authorized") is not False
    ):
        raise ValueError("cluster-aware SAE synthetic callback failure state differs")
    return {
        "schema": "sch.vpd-cluster-aware-sae-synthetic-callback-failure-verification.v1",
        "status": "pass",
        "synthetic_callback_failure_id": receipt_id,
        "execution_blueprint_id": receipt["execution_blueprint_id"],
        "execution_job_id": job_id,
        "callback_invoked": True,
        "callback_failed": True,
        "callback_charge_persisted": True,
        "retry_authorized": False,
        "model_loaded": False,
        "training_performed": False,
        "persistent_output_written": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a synthetic inert callback execution or failure receipt. "
            "Callback construction and invocation are injection-only."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-success", type=Path)
    mode.add_argument("--verify-failure", type=Path)
    args = parser.parse_args()
    if args.verify_success is not None:
        result = verify_synthetic_callback_execution(
            args.verify_success, project_root=args.project_root
        )
    else:
        result = verify_synthetic_callback_failure(
            args.verify_failure, project_root=args.project_root
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
