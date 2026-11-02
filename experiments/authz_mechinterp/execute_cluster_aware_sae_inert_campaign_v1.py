from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.attach_cluster_aware_sae_inert_runtime_v1 import (
    ClaimedInertRuntimeSession,
    verify_inert_runtime_attachment_claim,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.execute_cluster_aware_sae_inert_callback_v1 import (
    FAILURE_RECEIPT_NAME,
    SUCCESS_RECEIPT_NAME,
    SyntheticCallbackExecutionError,
    execute_claimed_inert_callback,
    verify_synthetic_callback_execution,
    verify_synthetic_callback_failure,
)
from experiments.authz_mechinterp.run_cluster_aware_sae_training_job_v1 import (
    validate_execution_blueprint_content,
)


SUCCESS_RECEIPT_NAME_CAMPAIGN = "campaign.json"
FAILURE_RECEIPT_NAME_CAMPAIGN = "campaign-failure.json"
SUCCESS_SCHEMA = "sch.vpd-cluster-aware-sae-synthetic-inert-campaign.v1"
FAILURE_SCHEMA = "sch.vpd-cluster-aware-sae-synthetic-inert-campaign-failure.v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("cluster-aware SAE synthetic campaign input must be an object")
    return value


def _content_id(value: Any, *, field_name: str) -> str:
    observed = str(value)
    if len(observed) != 64 or any(
        character not in "0123456789abcdef" for character in observed
    ):
        raise ValueError(f"{field_name} is not a lowercase SHA-256 content ID")
    return observed


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("cluster-aware SAE synthetic campaign path escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _boundary(*, callback_failed: bool) -> dict[str, Any]:
    return {
        "synthetic_inert_campaign_only": True,
        "callback_failed": callback_failed,
        "model_framework_imported": False,
        "model_loaded": False,
        "model_inference_performed": False,
        "training_performed": False,
        "persistent_dictionary_output_created": False,
        "weights_loaded_or_mutated": False,
        "production_runtime_attached": False,
        "production_trainer_callback_invoked": False,
        "execution_authorized_by_receipt": False,
        "sae_training_authorized_by_receipt": False,
        "real_actions_authorized": False,
    }


@dataclass
class SyntheticCampaignExecutionError(RuntimeError):
    receipt_path: Path
    failed_execution_job_id: str

    def __str__(self) -> str:
        return (
            "synthetic cluster-aware SAE campaign failed after durable charge; "
            f"failure receipt: {self.receipt_path}"
        )


def _preflight(
    *,
    blueprint_path: Path,
    claimed_sessions: list[ClaimedInertRuntimeSession],
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, ClaimedInertRuntimeSession]]:
    root = project_root.resolve()
    resolved_blueprint = blueprint_path.resolve()
    if not resolved_blueprint.is_relative_to(root) or not resolved_blueprint.is_file():
        raise ValueError("cluster-aware SAE synthetic campaign blueprint path differs")
    blueprint = _load(resolved_blueprint)
    blueprint_check = validate_execution_blueprint_content(blueprint)
    expected_job_ids = [
        str(job["execution_job_id"]) for job in blueprint["execution_jobs"]
    ]
    claimed_by_job: dict[str, ClaimedInertRuntimeSession] = {}
    for claimed in claimed_sessions:
        verification = verify_inert_runtime_attachment_claim(
            claimed.claim_receipt_path, project_root=root
        )
        job_id = str(verification["execution_job_id"])
        if (
            verification["execution_blueprint_id"]
            != blueprint_check["execution_blueprint_id"]
            or job_id in claimed_by_job
            or claimed.callback_invoked
        ):
            raise ValueError("cluster-aware SAE synthetic campaign claim set differs")
        claimed_by_job[job_id] = claimed
    if set(claimed_by_job) != set(expected_job_ids) or len(claimed_by_job) != len(
        expected_job_ids
    ):
        raise ValueError("cluster-aware SAE synthetic campaign job coverage differs")
    return blueprint, blueprint_check, claimed_by_job


def _success_record(
    *,
    job: dict[str, Any],
    receipt: dict[str, Any],
    receipt_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    worker = receipt["runtime_worker_receipt"]
    return {
        "execution_job_id": job["execution_job_id"],
        "source_training_job_id": job["source_training_job_id"],
        "compartment_id": job["compartment_id"],
        "compartment_kind": job["compartment_kind"],
        "activation_categories": list(job["activation_categories"]),
        "cluster_id": job["cluster_id"],
        "output_dir": job["output_dir"],
        "runtime_worker_receipt_id": worker["runtime_worker_receipt_id"],
        "success_receipt_relative_path": _relative(receipt_path, project_root),
        "planned_steps": int(worker["allocation"]["planned_steps"]),
        "planned_activation_rows": int(
            worker["allocation"]["planned_activation_rows"]
        ),
        "dictionary_features": int(worker["allocation"]["dictionary_features"]),
    }


def execute_synthetic_inert_campaign(
    *,
    blueprint_path: Path,
    claimed_sessions: list[ClaimedInertRuntimeSession],
    campaign_root: Path,
    project_root: Path,
) -> dict[str, Any]:
    root = project_root.resolve()
    blueprint, blueprint_check, claimed_by_job = _preflight(
        blueprint_path=blueprint_path,
        claimed_sessions=claimed_sessions,
        project_root=root,
    )
    resolved_campaign_root = campaign_root.resolve()
    if not resolved_campaign_root.is_relative_to(root):
        raise ValueError("cluster-aware SAE synthetic campaign root escapes project")
    try:
        resolved_campaign_root.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError("cluster-aware SAE synthetic campaign was already claimed") from exc
    successes: list[dict[str, Any]] = []
    jobs = list(blueprint["execution_jobs"])
    for index, job in enumerate(jobs):
        job_id = str(job["execution_job_id"])
        claimed = claimed_by_job[job_id]
        try:
            receipt = execute_claimed_inert_callback(claimed=claimed, project_root=root)
        except SyntheticCallbackExecutionError as exc:
            remaining_jobs = jobs[index + 1 :]
            remaining_claims: list[dict[str, Any]] = []
            for remaining_job in remaining_jobs:
                remaining_job_id = str(remaining_job["execution_job_id"])
                remaining_claim = claimed_by_job[remaining_job_id]
                remaining_claim.close()
                remaining_claims.append(
                    {
                        "execution_job_id": remaining_job_id,
                        "attachment_claim_relative_path": _relative(
                            remaining_claim.claim_receipt_path, root
                        ),
                    }
                )
            body = {
                "schema": FAILURE_SCHEMA,
                "status": "failed_synthetic_campaign_charged_terminal",
                "source": {
                    "execution_blueprint_relative_path": _relative(
                        blueprint_path, root
                    ),
                    "execution_blueprint_sha256": sha256_file(blueprint_path),
                    "execution_blueprint_id": blueprint_check[
                        "execution_blueprint_id"
                    ],
                    "global_pair_split_id": blueprint_check[
                        "global_pair_split_id"
                    ],
                },
                "campaign_root_relative_path": _relative(
                    resolved_campaign_root, root
                ),
                "successful_jobs": successes,
                "failed_job": {
                    "execution_job_id": job_id,
                    "failure_receipt_relative_path": _relative(
                        exc.receipt_path, root
                    ),
                },
                "remaining_uninvoked_jobs": remaining_claims,
                "summary": {
                    "expected_jobs": len(jobs),
                    "successful_callbacks": len(successes),
                    "failed_callbacks": 1,
                    "charged_callbacks": len(successes) + 1,
                    "uninvoked_callbacks": len(remaining_jobs),
                    "retry_authorized": False,
                },
                "scientific_boundary": _boundary(callback_failed=True),
            }
            failure_receipt = {
                **body,
                "synthetic_campaign_failure_id": content_sha256(body),
            }
            failure_path = resolved_campaign_root / FAILURE_RECEIPT_NAME_CAMPAIGN
            write_json(failure_path, failure_receipt)
            raise SyntheticCampaignExecutionError(
                receipt_path=failure_path, failed_execution_job_id=job_id
            ) from exc
        success_path = (
            claimed.claim_receipt_path.parent
            / "callback-claim"
            / SUCCESS_RECEIPT_NAME
        )
        successes.append(
            _success_record(
                job=job,
                receipt=receipt,
                receipt_path=success_path,
                project_root=root,
            )
        )
    summary = blueprint["summary"]
    body = {
        "schema": SUCCESS_SCHEMA,
        "status": "complete_exact_synthetic_inert_campaign_no_training",
        "source": {
            "execution_blueprint_relative_path": _relative(blueprint_path, root),
            "execution_blueprint_sha256": sha256_file(blueprint_path),
            "execution_blueprint_id": blueprint_check["execution_blueprint_id"],
            "materialization_id": blueprint_check["materialization_id"],
            "cluster_aware_sae_plan_id": blueprint_check[
                "cluster_aware_sae_plan_id"
            ],
            "global_pair_split_id": blueprint_check["global_pair_split_id"],
        },
        "campaign_root_relative_path": _relative(resolved_campaign_root, root),
        "jobs": successes,
        "summary": {
            "categories": int(summary["categories"]),
            "compartments": int(summary["compartments"]),
            "specific_compartments": int(summary["specific_compartments"]),
            "shared_compartments": int(summary["shared_compartments"]),
            "expected_jobs": len(jobs),
            "successful_callbacks": len(successes),
            "failed_callbacks": 0,
            "charged_callbacks": len(successes),
            "unique_output_dirs": len({row["output_dir"] for row in successes}),
            "planned_steps": sum(row["planned_steps"] for row in successes),
            "planned_activation_rows": sum(
                row["planned_activation_rows"] for row in successes
            ),
            "one_global_pair_split": True,
        },
        "scientific_boundary": _boundary(callback_failed=False),
    }
    receipt = {**body, "synthetic_campaign_id": content_sha256(body)}
    write_json(resolved_campaign_root / SUCCESS_RECEIPT_NAME_CAMPAIGN, receipt)
    return receipt


def _verify_common(
    receipt: dict[str, Any], *, receipt_path: Path, project_root: Path
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    root = project_root.resolve()
    campaign_root = receipt_path.resolve().parent
    if (
        not campaign_root.is_relative_to(root)
        or receipt.get("campaign_root_relative_path") != _relative(campaign_root, root)
    ):
        raise ValueError("cluster-aware SAE synthetic campaign root differs")
    source = receipt.get("source", {})
    blueprint_path = (
        root / str(source.get("execution_blueprint_relative_path", ""))
    ).resolve()
    if (
        not blueprint_path.is_relative_to(root)
        or not blueprint_path.is_file()
        or sha256_file(blueprint_path) != source.get("execution_blueprint_sha256")
    ):
        raise ValueError("cluster-aware SAE synthetic campaign blueprint bytes differ")
    blueprint = _load(blueprint_path)
    blueprint_check = validate_execution_blueprint_content(blueprint)
    if (
        source.get("execution_blueprint_id")
        != blueprint_check["execution_blueprint_id"]
        or source.get("global_pair_split_id")
        != blueprint_check["global_pair_split_id"]
    ):
        raise ValueError("cluster-aware SAE synthetic campaign blueprint lineage differs")
    return blueprint, blueprint_check, root


def _verify_success_records(
    records: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    *,
    project_root: Path,
) -> tuple[int, int]:
    if len(records) != len(jobs):
        raise ValueError("cluster-aware SAE synthetic campaign success count differs")
    planned_steps = 0
    planned_rows = 0
    for record, job in zip(records, jobs, strict=True):
        receipt_path = (
            project_root / str(record.get("success_receipt_relative_path", ""))
        ).resolve()
        verification = verify_synthetic_callback_execution(
            receipt_path, project_root=project_root
        )
        if (
            record.get("execution_job_id") != job["execution_job_id"]
            or record.get("source_training_job_id") != job["source_training_job_id"]
            or record.get("compartment_id") != job["compartment_id"]
            or record.get("compartment_kind") != job["compartment_kind"]
            or record.get("activation_categories") != job["activation_categories"]
            or record.get("cluster_id") != job["cluster_id"]
            or record.get("output_dir") != job["output_dir"]
            or record.get("runtime_worker_receipt_id")
            != verification["runtime_worker_receipt_id"]
            or verification["execution_job_id"] != job["execution_job_id"]
            or int(record.get("planned_steps", -1))
            != int(job["profile"]["steps"])
            or int(record.get("planned_activation_rows", -1))
            != int(job["exact_sample_contract"]["sae_activation_rows"])
            or int(record.get("dictionary_features", -1))
            != int(job["dictionary_features"])
        ):
            raise ValueError("cluster-aware SAE synthetic campaign job receipt differs")
        callback_receipt = _load(receipt_path)
        claim_path = Path(
            str(
                callback_receipt.get("source", {}).get(
                    "attachment_claim_path", ""
                )
            )
        ).resolve()
        claim = _load(claim_path)
        transition_path = Path(
            str(claim.get("source", {}).get("transition_receipt_path", ""))
        ).resolve()
        transition = _load(transition_path)
        reserved_job_root = Path(
            str(transition.get("job", {}).get("reserved_job_root", ""))
        ).resolve()
        exact_output = Path(
            str(transition.get("job", {}).get("reserved_exact_output_path", ""))
        ).resolve()
        if (
            exact_output != (reserved_job_root / str(job["output_dir"])).resolve()
            or exact_output.exists()
        ):
            raise ValueError("cluster-aware SAE synthetic campaign output state differs")
        planned_steps += int(record["planned_steps"])
        planned_rows += int(record["planned_activation_rows"])
    return planned_steps, planned_rows


def _verify_boundary(receipt: dict[str, Any], *, callback_failed: bool) -> None:
    boundary = receipt.get("scientific_boundary", {})
    if (
        boundary.get("synthetic_inert_campaign_only") is not True
        or boundary.get("callback_failed") is not callback_failed
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
                "production_trainer_callback_invoked",
                "execution_authorized_by_receipt",
                "sae_training_authorized_by_receipt",
                "real_actions_authorized",
            )
        )
    ):
        raise ValueError("cluster-aware SAE synthetic campaign boundary differs")


def verify_synthetic_inert_campaign(
    receipt_path: Path, *, project_root: Path
) -> dict[str, Any]:
    resolved = receipt_path.resolve()
    if resolved.name != SUCCESS_RECEIPT_NAME_CAMPAIGN or not resolved.is_file():
        raise ValueError("cluster-aware SAE synthetic campaign receipt path differs")
    receipt = _load(resolved)
    if receipt.get("schema") != SUCCESS_SCHEMA:
        raise ValueError("cluster-aware SAE synthetic campaign schema differs")
    body = dict(receipt)
    campaign_id = _content_id(
        body.pop("synthetic_campaign_id", ""),
        field_name="synthetic campaign ID",
    )
    if campaign_id != content_sha256(body):
        raise ValueError("cluster-aware SAE synthetic campaign address differs")
    blueprint, blueprint_check, root = _verify_common(
        receipt, receipt_path=resolved, project_root=project_root
    )
    _verify_boundary(receipt, callback_failed=False)
    jobs = list(blueprint["execution_jobs"])
    records = receipt.get("jobs", [])
    planned_steps, planned_rows = _verify_success_records(
        records, jobs, project_root=root
    )
    summary = receipt.get("summary", {})
    blueprint_summary = blueprint["summary"]
    if (
        receipt.get("status")
        != "complete_exact_synthetic_inert_campaign_no_training"
        or int(summary.get("categories", -1))
        != int(blueprint_summary["categories"])
        or int(summary.get("compartments", -1))
        != int(blueprint_summary["compartments"])
        or int(summary.get("specific_compartments", -1))
        != int(blueprint_summary["specific_compartments"])
        or int(summary.get("shared_compartments", -1))
        != int(blueprint_summary["shared_compartments"])
        or int(summary.get("expected_jobs", -1)) != len(jobs)
        or int(summary.get("successful_callbacks", -1)) != len(jobs)
        or int(summary.get("failed_callbacks", -1)) != 0
        or int(summary.get("charged_callbacks", -1)) != len(jobs)
        or int(summary.get("unique_output_dirs", -1))
        != len({job["output_dir"] for job in jobs})
        or int(summary.get("planned_steps", -1)) != planned_steps
        or int(summary.get("planned_activation_rows", -1)) != planned_rows
        or summary.get("one_global_pair_split") is not True
    ):
        raise ValueError("cluster-aware SAE synthetic campaign summary differs")
    return {
        "schema": "sch.vpd-cluster-aware-sae-synthetic-inert-campaign-verification.v1",
        "status": "pass",
        "synthetic_campaign_id": campaign_id,
        "execution_blueprint_id": blueprint_check["execution_blueprint_id"],
        "categories": int(summary["categories"]),
        "compartments": int(summary["compartments"]),
        "jobs": len(jobs),
        "charged_callbacks": len(jobs),
        "unique_output_dirs": int(summary["unique_output_dirs"]),
        "model_loaded": False,
        "training_performed": False,
        "persistent_output_written": False,
        "production_trainer_callback_invoked": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def verify_synthetic_inert_campaign_failure(
    receipt_path: Path, *, project_root: Path
) -> dict[str, Any]:
    resolved = receipt_path.resolve()
    if resolved.name != FAILURE_RECEIPT_NAME_CAMPAIGN or not resolved.is_file():
        raise ValueError("cluster-aware SAE synthetic campaign failure path differs")
    receipt = _load(resolved)
    if receipt.get("schema") != FAILURE_SCHEMA:
        raise ValueError("cluster-aware SAE synthetic campaign failure schema differs")
    body = dict(receipt)
    failure_id = _content_id(
        body.pop("synthetic_campaign_failure_id", ""),
        field_name="synthetic campaign failure ID",
    )
    if failure_id != content_sha256(body):
        raise ValueError("cluster-aware SAE synthetic campaign failure address differs")
    blueprint, blueprint_check, root = _verify_common(
        receipt, receipt_path=resolved, project_root=project_root
    )
    _verify_boundary(receipt, callback_failed=True)
    jobs = list(blueprint["execution_jobs"])
    successes = receipt.get("successful_jobs", [])
    failed = receipt.get("failed_job", {})
    failed_index = len(successes)
    if failed_index >= len(jobs):
        raise ValueError("cluster-aware SAE synthetic campaign failure position differs")
    _verify_success_records(successes, jobs[:failed_index], project_root=root)
    failed_job = jobs[failed_index]
    failed_path = (
        root / str(failed.get("failure_receipt_relative_path", ""))
    ).resolve()
    failure_verification = verify_synthetic_callback_failure(
        failed_path, project_root=root
    )
    remaining = receipt.get("remaining_uninvoked_jobs", [])
    remaining_jobs = jobs[failed_index + 1 :]
    if len(remaining) != len(remaining_jobs):
        raise ValueError("cluster-aware SAE synthetic campaign remaining count differs")
    for record, job in zip(remaining, remaining_jobs, strict=True):
        claim_path = (
            root / str(record.get("attachment_claim_relative_path", ""))
        ).resolve()
        claim_verification = verify_inert_runtime_attachment_claim(
            claim_path, project_root=root
        )
        if (
            record.get("execution_job_id") != job["execution_job_id"]
            or claim_verification["execution_job_id"] != job["execution_job_id"]
            or claim_verification["trainer_callback_invoked"] is not False
        ):
            raise ValueError("cluster-aware SAE synthetic campaign remaining job differs")
    summary = receipt.get("summary", {})
    if (
        receipt.get("status") != "failed_synthetic_campaign_charged_terminal"
        or failed.get("execution_job_id") != failed_job["execution_job_id"]
        or failure_verification["execution_job_id"] != failed_job["execution_job_id"]
        or int(summary.get("expected_jobs", -1)) != len(jobs)
        or int(summary.get("successful_callbacks", -1)) != failed_index
        or int(summary.get("failed_callbacks", -1)) != 1
        or int(summary.get("charged_callbacks", -1)) != failed_index + 1
        or int(summary.get("uninvoked_callbacks", -1)) != len(remaining_jobs)
        or summary.get("retry_authorized") is not False
    ):
        raise ValueError("cluster-aware SAE synthetic campaign failure summary differs")
    return {
        "schema": "sch.vpd-cluster-aware-sae-synthetic-inert-campaign-failure-verification.v1",
        "status": "pass",
        "synthetic_campaign_failure_id": failure_id,
        "execution_blueprint_id": blueprint_check["execution_blueprint_id"],
        "jobs": len(jobs),
        "charged_callbacks": failed_index + 1,
        "successful_callbacks": failed_index,
        "failed_callbacks": 1,
        "uninvoked_callbacks": len(remaining_jobs),
        "retry_authorized": False,
        "model_loaded": False,
        "training_performed": False,
        "persistent_output_written": False,
        "production_trainer_callback_invoked": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a synthetic inert 34-job campaign success or failure receipt. "
            "Campaign construction and callback invocation are injection-only."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-success", type=Path)
    mode.add_argument("--verify-failure", type=Path)
    args = parser.parse_args()
    if args.verify_success is not None:
        result = verify_synthetic_inert_campaign(
            args.verify_success, project_root=args.project_root
        )
    else:
        result = verify_synthetic_inert_campaign_failure(
            args.verify_failure, project_root=args.project_root
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
