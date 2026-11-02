from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_execution_blueprint_v1 import (
    _synthetic_materialization,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_inert_campaign_v1 import (
    FAIL_INDEX,
)
from experiments.authz_mechinterp.build_cluster_aware_sae_execution_blueprint_v1 import (
    build_execution_blueprint,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.run_cluster_aware_sae_training_job_v1 import (
    validate_execution_blueprint_content,
)


SCHEMA = "sch.vpd-cluster-aware-sae-compartment-resource-reconciliation.v1"
EXPECTED_CAMPAIGN_BENCHMARK_ID = (
    "34204198cc2cd605e74fff0f24da7b5d09337613f16da74872f354c53f8b32f5"
)
EXPECTED_CAMPAIGN_REPORT_SHA256 = (
    "5a57299a1b912257c174d0daafddf5b89fba9a179de2249a6408bad693ae05a9"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("cluster-aware SAE resource input must be an object")
    return value


def _content_id(value: Any, *, field_name: str) -> str:
    observed = str(value)
    if len(observed) != 64 or any(
        character not in "0123456789abcdef" for character in observed
    ):
        raise ValueError(f"{field_name} is not a lowercase SHA-256 content ID")
    return observed


def _verify_campaign_summary(
    path: Path, *, project_root: Path
) -> dict[str, Any]:
    root = project_root.resolve()
    resolved = path.resolve()
    if (
        not resolved.is_relative_to(root)
        or sha256_file(resolved) != EXPECTED_CAMPAIGN_REPORT_SHA256
    ):
        raise ValueError("pinned inert campaign benchmark report differs")
    report = _load(resolved)
    body = dict(report)
    benchmark_id = _content_id(
        body.pop("benchmark_id", ""), field_name="inert campaign benchmark ID"
    )
    if (
        benchmark_id != EXPECTED_CAMPAIGN_BENCHMARK_ID
        or benchmark_id != content_sha256(body)
        or report.get("schema")
        != "sch.vpd-cluster-aware-sae-inert-campaign-benchmark.v1"
        or report.get("status") != "pass"
        or report.get("summary") != {"controls": 28, "passed": 28}
        or not all(report.get("controls", {}).values())
        or report.get("design", {}).get("exact_activation_categories") != 8
        or report.get("design", {}).get("exact_compartments") != 17
        or report.get("design", {}).get("exact_execution_jobs") != 34
        or report.get("design", {}).get("unique_output_directories") != 34
        or report.get("results", {}).get("success_callbacks_charged") != 34
        or report.get("results", {}).get("failure_campaign_callbacks_charged")
        != FAIL_INDEX + 1
        or report.get("results", {}).get("persistent_campaign_receipts_created")
        is not False
    ):
        raise ValueError("inert campaign benchmark summary contract differs")
    source = report.get("source", {})
    source_paths = {
        "benchmark_sha256": root
        / "experiments/authz_mechinterp/benchmark_vpd_cluster_aware_sae_inert_campaign_v1.py",
        "inert_campaign_execution_sha256": root
        / "experiments/authz_mechinterp/execute_cluster_aware_sae_inert_campaign_v1.py",
        "inert_callback_execution_sha256": root
        / "experiments/authz_mechinterp/execute_cluster_aware_sae_inert_callback_v1.py",
    }
    if any(
        sha256_file(source_path) != source.get(source_field)
        for source_field, source_path in source_paths.items()
    ):
        raise ValueError("inert campaign benchmark source differs")
    return report


def _job_row(job: dict[str, Any], *, index: int) -> dict[str, Any]:
    sample = job["exact_sample_contract"]
    profile = job["profile"]
    return {
        "job_index": index,
        "execution_job_id": job["execution_job_id"],
        "source_training_job_id": job["source_training_job_id"],
        "compartment_id": job["compartment_id"],
        "compartment_kind": job["compartment_kind"],
        "activation_categories": list(job["activation_categories"]),
        "cluster_id": job["cluster_id"],
        "initialization_seed": int(job["initialization_seed"]),
        "data_selection_seed": int(job["data_selection_seed"]),
        "profile_id": profile["id"],
        "activation_key": profile["activation_key"],
        "planned_steps": int(profile["steps"]),
        "planned_activation_rows": int(sample["sae_activation_rows"]),
        "planned_activation_rows_per_episode": int(
            sample["sae_activation_rows_per_episode"]
        ),
        "batch_size": int(profile["batch_size"]),
        "dictionary_features": int(job["dictionary_features"]),
        "hidden_size": int(profile["hidden_size"]),
        "episode_count": len(sample["episode_ids"]),
        "pair_count": len(sample["pair_ids"]),
        "output_dir": job["output_dir"],
        "exact_sample_contract_sha256": content_sha256(sample),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "jobs": len(rows),
        "planned_steps": sum(row["planned_steps"] for row in rows),
        "planned_activation_rows": sum(
            row["planned_activation_rows"] for row in rows
        ),
        "episode_contract_entries": sum(row["episode_count"] for row in rows),
        "pair_contract_entries": sum(row["pair_count"] for row in rows),
        "dictionary_feature_widths": sorted(
            {row["dictionary_features"] for row in rows}
        ),
        "batch_sizes": sorted({row["batch_size"] for row in rows}),
        "initialization_seeds": sorted(
            {row["initialization_seed"] for row in rows}
        ),
        "data_selection_seeds": sorted(
            {row["data_selection_seed"] for row in rows}
        ),
        "unique_output_directories": len({row["output_dir"] for row in rows}),
    }


def _status_resources(
    job_rows: list[dict[str, Any]], statuses: list[str]
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, status in zip(job_rows, statuses, strict=True):
        grouped[status].append(row)
    return {
        status: _aggregate(grouped[status])
        for status in sorted(grouped)
    }


def _job_states(
    job_rows: list[dict[str, Any]], *, failure_index: int | None
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for row in job_rows:
        index = int(row["job_index"])
        if failure_index is None or index < failure_index:
            state = "synthetic_completed_no_training"
            charged = True
        elif index == failure_index:
            state = "synthetic_failed_charged_terminal"
            charged = True
        else:
            state = "synthetic_uninvoked_after_terminal_failure"
            charged = False
        states.append(
            {
                "job_index": index,
                "execution_job_id": row["execution_job_id"],
                "compartment_id": row["compartment_id"],
                "state": state,
                "callback_charged": charged,
                "training_performed": False,
                "dictionary_output_written": False,
                "same_campaign_retry_authorized": False,
                "fresh_authorization_required_for_any_real_execution": True,
            }
        )
    return states


def _reconciliation(
    *,
    job_rows: list[dict[str, Any]],
    campaign_id: str,
    failure_index: int | None,
) -> dict[str, Any]:
    states = _job_states(job_rows, failure_index=failure_index)
    status_values = [str(row["state"]) for row in states]
    charged = sum(bool(row["callback_charged"]) for row in states)
    completed = status_values.count("synthetic_completed_no_training")
    failed = status_values.count("synthetic_failed_charged_terminal")
    uninvoked = status_values.count(
        "synthetic_uninvoked_after_terminal_failure"
    )
    return {
        "source_synthetic_campaign_id": campaign_id,
        "source_raw_campaign_receipt_present": False,
        "reconstruction_basis": (
            "verified_campaign_benchmark_summary_and_pinned_job_order"
        ),
        "job_states": states,
        "planned_resources_by_state": _status_resources(job_rows, status_values),
        "summary": {
            "jobs": len(states),
            "completed_callbacks": completed,
            "failed_callbacks": failed,
            "uninvoked_callbacks": uninvoked,
            "charged_callbacks": charged,
            "training_steps_observed": 0,
            "activation_rows_consumed_observed": 0,
            "dictionary_outputs_observed": 0,
            "same_campaign_retry_authorized": False,
        },
        "fresh_authorization_requirements": {
            "applies_to_every_future_real_job": True,
            "empirical_cluster_aware_blueprint_required": True,
            "fresh_active_scientific_release_receipt_required": True,
            "fresh_exact_job_compute_receipt_required": True,
            "fresh_nonce_and_output_reservation_required": True,
            "explicit_user_authorization_required": True,
            "synthetic_campaign_receipt_transfers_no_authority": True,
        },
    }


def build_resource_reconciliation(
    *, campaign_benchmark_path: Path, project_root: Path
) -> dict[str, Any]:
    root = project_root.resolve()
    campaign = _verify_campaign_summary(
        campaign_benchmark_path, project_root=root
    )
    materialization = _synthetic_materialization(root)
    blueprint = build_execution_blueprint(materialization=materialization)
    blueprint_check = validate_execution_blueprint_content(blueprint)
    if (
        blueprint_check["execution_blueprint_id"]
        != campaign["results"]["execution_blueprint_id"]
        or len(blueprint["execution_jobs"]) != 34
    ):
        raise ValueError("resource reconciliation blueprint differs from campaign")
    job_rows = [
        _job_row(job, index=index)
        for index, job in enumerate(blueprint["execution_jobs"])
    ]
    compartment_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    shared_rows: list[dict[str, Any]] = []
    for row in job_rows:
        compartment_rows[str(row["compartment_id"])].append(row)
        categories = list(row["activation_categories"])
        if row["compartment_kind"] == "shared_fallback":
            shared_rows.append(row)
        elif len(categories) == 1:
            category_rows[str(categories[0])].append(row)
        else:
            raise ValueError("specific compartment category attribution differs")
    if (
        len(compartment_rows) != 17
        or len(category_rows) != 8
        or len(shared_rows) != 2
        or set(map(id, shared_rows))
        & {id(row) for rows in category_rows.values() for row in rows}
    ):
        raise ValueError("resource reconciliation compartment partition differs")
    compartment_accounts = [
        {
            "compartment_id": compartment_id,
            "compartment_kind": rows[0]["compartment_kind"],
            "activation_categories": sorted(
                {category for row in rows for category in row["activation_categories"]}
            ),
            "cluster_ids": sorted({row["cluster_id"] for row in rows}),
            "resources": _aggregate(rows),
        }
        for compartment_id, rows in sorted(compartment_rows.items())
    ]
    category_accounts = [
        {
            "activation_category": category,
            "specific_compartment_ids": sorted(
                {row["compartment_id"] for row in rows}
            ),
            "shared_fallback_accounted_separately": True,
            "resources": _aggregate(rows),
        }
        for category, rows in sorted(category_rows.items())
    ]
    success_id = _content_id(
        campaign["results"]["synthetic_success_campaign_id"],
        field_name="synthetic success campaign ID",
    )
    failure_id = _content_id(
        campaign["results"]["synthetic_failure_campaign_id"],
        field_name="synthetic failure campaign ID",
    )
    body = {
        "schema": SCHEMA,
        "status": (
            "complete_reconstructed_nonexecuting_compartment_resource_and_failure_reconciliation"
        ),
        "source": {
            "campaign_benchmark_path": campaign_benchmark_path.resolve()
            .relative_to(root)
            .as_posix(),
            "campaign_benchmark_sha256": sha256_file(campaign_benchmark_path),
            "campaign_benchmark_id": campaign["benchmark_id"],
            "execution_blueprint_id": blueprint_check["execution_blueprint_id"],
            "materialization_id": blueprint_check["materialization_id"],
            "cluster_aware_sae_plan_id": blueprint_check[
                "cluster_aware_sae_plan_id"
            ],
            "global_pair_split_id": blueprint_check["global_pair_split_id"],
            "raw_campaign_receipts_present": False,
            "reconstructed_from_verified_summary": True,
        },
        "job_resource_rows": job_rows,
        "compartment_resource_accounts": compartment_accounts,
        "category_specific_resource_accounts": category_accounts,
        "shared_fallback_resource_account": {
            "compartment_ids": sorted({row["compartment_id"] for row in shared_rows}),
            "activation_categories": sorted(
                {category for row in shared_rows for category in row["activation_categories"]}
            ),
            "never_double_counted_into_category_totals": True,
            "resources": _aggregate(shared_rows),
        },
        "global_planned_resources": _aggregate(job_rows),
        "success_reconciliation": _reconciliation(
            job_rows=job_rows, campaign_id=success_id, failure_index=None
        ),
        "terminal_failure_reconciliation": _reconciliation(
            job_rows=job_rows, campaign_id=failure_id, failure_index=FAIL_INDEX
        ),
        "summary": {
            "activation_categories": len(category_accounts),
            "compartments": len(compartment_accounts),
            "specific_compartments": sum(
                account["compartment_kind"] != "shared_fallback"
                for account in compartment_accounts
            ),
            "shared_fallback_compartments": sum(
                account["compartment_kind"] == "shared_fallback"
                for account in compartment_accounts
            ),
            "execution_jobs": len(job_rows),
            "unique_output_directories": len(
                {row["output_dir"] for row in job_rows}
            ),
            "shared_fallback_double_counted": False,
            "resource_values_are_planned_not_observed_training_consumption": True,
        },
        "scientific_boundary": {
            "synthetic_reconstruction_only": True,
            "raw_campaign_receipts_recovered": False,
            "resource_accounting_is_empirical_evidence": False,
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "training_performed": False,
            "persistent_dictionary_output_created": False,
            "weights_loaded_or_mutated": False,
            "production_campaign_executed": False,
            "retry_authorized": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
        },
    }
    return {**body, "resource_reconciliation_id": content_sha256(body)}


def verify_resource_reconciliation(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = _content_id(
        body.pop("resource_reconciliation_id", ""),
        field_name="resource reconciliation ID",
    )
    if identifier != content_sha256(body) or observed != build_resource_reconciliation(
        **kwargs
    ):
        raise ValueError("compartment resource reconciliation is not reproducible")
    failure = observed["terminal_failure_reconciliation"]["summary"]
    return {
        "schema": "sch.vpd-cluster-aware-sae-compartment-resource-reconciliation-verification.v1",
        "status": "pass",
        "resource_reconciliation_id": identifier,
        "activation_categories": observed["summary"]["activation_categories"],
        "compartments": observed["summary"]["compartments"],
        "jobs": observed["summary"]["execution_jobs"],
        "unique_output_directories": observed["summary"][
            "unique_output_directories"
        ],
        "planned_steps": observed["global_planned_resources"]["planned_steps"],
        "planned_activation_rows": observed["global_planned_resources"][
            "planned_activation_rows"
        ],
        "failure_charged_callbacks": failure["charged_callbacks"],
        "failure_uninvoked_callbacks": failure["uninvoked_callbacks"],
        "shared_fallback_double_counted": False,
        "raw_campaign_receipts_recovered": False,
        "training_performed": False,
        "retry_authorized": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "campaign_benchmark_path": project_root
        / "artifacts/vpd_cluster_aware_sae_inert_campaign_benchmark_v1/report.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify reconstructed nonexecuting cluster-aware SAE "
            "compartment resource and terminal-failure reconciliation V1."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.output is not None:
        result = build_resource_reconciliation(**inputs)
        write_json(args.output, result)
    else:
        result = verify_resource_reconciliation(args.verify, **inputs)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "resource_reconciliation_id": result["resource_reconciliation_id"],
                "activation_categories": result.get(
                    "activation_categories",
                    result.get("summary", {}).get("activation_categories"),
                ),
                "compartments": result.get(
                    "compartments", result.get("summary", {}).get("compartments")
                ),
                "jobs": result.get("jobs", result.get("summary", {}).get("execution_jobs")),
                "training_performed": result.get(
                    "training_performed",
                    result.get("scientific_boundary", {}).get("training_performed"),
                ),
                "execution_authorized": result.get(
                    "execution_authorized",
                    result.get("scientific_boundary", {}).get("execution_authorized"),
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
