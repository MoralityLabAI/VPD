from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


POLICY_SCHEMA = "sch.vpd-cluster-aware-sae-budget-admission-policy.v1"
PROPOSAL_SCHEMA = "sch.vpd-cluster-aware-sae-budget-proposal.v1"
ADMISSION_SCHEMA = "sch.vpd-cluster-aware-sae-budget-admission.v1"
EXPECTED_LEDGER_ID = (
    "c9759d5e1f89cd3f2e6d3a9fa264e27f0ae7e1eee933c3413de5ac3de54bf0ff"
)
EXPECTED_LEDGER_SHA256 = (
    "3e690917a3cd7ec7797011e77843c3cde973d2015eb40be7999d1f8530de42f5"
)
EXPECTED_POLICY_ID = (
    "5e9f4fc0841d0e71a2eef6f9ac91f5e27d0702c0e0aadfa5a6f511dad50719f2"
)
COUNT_FIELDS = (
    "jobs",
    "planned_steps",
    "planned_activation_rows",
    "episode_contract_entries",
    "pair_contract_entries",
    "unique_output_directories",
)
EXACT_LIST_FIELDS = (
    "batch_sizes",
    "data_selection_seeds",
    "dictionary_feature_widths",
    "initialization_seeds",
)
FALSE_AUTHORITY_CLAIMS = {
    "empirical_blueprint_present": False,
    "execution_authorized": False,
    "model_loading_authorized": False,
    "paid_compute_authorized": False,
    "retry_authorized": False,
    "sae_training_authorized": False,
    "weight_mutation_authorized": False,
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _content_id(value: Any, *, field_name: str) -> str:
    identifier = str(value)
    if len(identifier) != 64 or any(char not in "0123456789abcdef" for char in identifier):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 identifier")
    return identifier


def _resource_ceiling(resources: dict[str, Any]) -> dict[str, Any]:
    ceiling: dict[str, Any] = {}
    for field_name in COUNT_FIELDS:
        value = resources.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"resource {field_name} must be a positive integer")
        ceiling[field_name] = value
    if ceiling["unique_output_directories"] != ceiling["jobs"]:
        raise ValueError("each budgeted job must retain a unique output directory")
    for field_name in EXACT_LIST_FIELDS:
        values = resources.get(field_name)
        if not isinstance(values, list) or not values or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in values
        ):
            raise ValueError(f"resource {field_name} must be a nonempty positive-int list")
        if values != sorted(set(values)):
            raise ValueError(f"resource {field_name} must be sorted and unique")
        ceiling[field_name] = values
    return ceiling


def _validate_ledger(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_LEDGER_SHA256:
        raise ValueError("resource reconciliation ledger SHA-256 mismatch")
    ledger = _load(path)
    body = dict(ledger)
    identifier = _content_id(
        body.pop("resource_reconciliation_id", ""),
        field_name="resource reconciliation ID",
    )
    if identifier != EXPECTED_LEDGER_ID or identifier != content_sha256(body):
        raise ValueError("resource reconciliation ledger content address mismatch")
    if ledger.get("scientific_boundary") != {
        "execution_authorized": False,
        "model_framework_imported": False,
        "model_inference_performed": False,
        "model_loaded": False,
        "persistent_dictionary_output_created": False,
        "production_campaign_executed": False,
        "raw_campaign_receipts_recovered": False,
        "resource_accounting_is_empirical_evidence": False,
        "retry_authorized": False,
        "sae_training_authorized": False,
        "synthetic_reconstruction_only": True,
        "training_performed": False,
        "weights_loaded_or_mutated": False,
    }:
        raise ValueError("resource reconciliation scientific boundary mismatch")
    return ledger


def build_budget_policy(*, resource_ledger_path: Path) -> dict[str, Any]:
    ledger = _validate_ledger(resource_ledger_path)
    category_budgets = []
    for account in ledger["category_specific_resource_accounts"]:
        category_budgets.append(
            {
                "activation_category": str(account["activation_category"]),
                "specific_compartment_ids": sorted(
                    str(value) for value in account["specific_compartment_ids"]
                ),
                "shared_fallback_excluded": account[
                    "shared_fallback_accounted_separately"
                ]
                is True,
                "ceiling": _resource_ceiling(account["resources"]),
            }
        )
    category_budgets.sort(key=lambda row: row["activation_category"])

    specific_budgets = []
    shared_accounts = []
    for account in ledger["compartment_resource_accounts"]:
        row = {
            "compartment_id": str(account["compartment_id"]),
            "compartment_kind": str(account["compartment_kind"]),
            "activation_categories": sorted(
                str(value) for value in account["activation_categories"]
            ),
            "cluster_ids": account["cluster_ids"],
            "ceiling": _resource_ceiling(account["resources"]),
        }
        if row["compartment_kind"] == "shared_fallback":
            shared_accounts.append(row)
        else:
            specific_budgets.append(row)
    specific_budgets.sort(key=lambda row: row["compartment_id"])
    if len(category_budgets) != 8 or len(specific_budgets) != 16 or len(shared_accounts) != 1:
        raise ValueError("expected 8 categories, 16 specific compartments, and 1 shared fallback")
    shared = shared_accounts[0]
    ledger_shared = ledger["shared_fallback_resource_account"]
    if (
        shared["compartment_id"] not in ledger_shared["compartment_ids"]
        or shared["ceiling"] != _resource_ceiling(ledger_shared["resources"])
        or ledger_shared["never_double_counted_into_category_totals"] is not True
    ):
        raise ValueError("shared fallback resource account mismatch")

    body = {
        "schema": POLICY_SCHEMA,
        "status": "sealed_nonexecuting_category_compartment_budget_ceilings",
        "source": {
            "resource_reconciliation_id": ledger["resource_reconciliation_id"],
            "resource_reconciliation_sha256": sha256_file(resource_ledger_path),
            "resource_reconciliation_path": (
                "artifacts/vpd_cluster_aware_sae_compartment_resource_"
                "reconciliation_v1/ledger.json"
            ),
            "raw_campaign_receipts_recovered": False,
            "reconstructed_resource_values_are_empirical_consumption": False,
        },
        "category_specific_budgets": category_budgets,
        "specific_compartment_budgets": specific_budgets,
        "shared_fallback_budget": {
            **shared,
            "accounted_once_globally": True,
            "charged_to_category_budgets": False,
        },
        "global_ceiling": _resource_ceiling(ledger["global_planned_resources"]),
        "admission_contract": {
            "all_eight_categories_required": True,
            "all_sixteen_specific_compartments_required": True,
            "one_shared_fallback_required": True,
            "category_totals_must_equal_specific_compartment_sums": True,
            "global_totals_must_equal_specific_plus_shared_once": True,
            "requested_counts_must_not_exceed_any_bound_ceiling": True,
            "resource_shape_lists_must_match_bound_values_exactly": True,
            "unique_output_directories_must_equal_jobs": True,
            "budget_admission_transfers_execution_authority": False,
            "fresh_empirical_blueprint_and_active_signed_receipts_required": True,
        },
        "scientific_boundary": {
            "synthetic_reconstructed_ceiling_source": True,
            "empirical_budget_observation": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "training_performed": False,
            "weights_loaded_or_mutated": False,
            "persistent_dictionary_output_created": False,
            **FALSE_AUTHORITY_CLAIMS,
        },
    }
    return {**body, "budget_policy_id": content_sha256(body)}


def verify_budget_policy(path: Path, *, resource_ledger_path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = _content_id(body.pop("budget_policy_id", ""), field_name="budget policy ID")
    if identifier != content_sha256(body) or observed != build_budget_policy(
        resource_ledger_path=resource_ledger_path
    ):
        raise ValueError("cluster-aware SAE budget policy is not reproducible")
    return {
        "schema": "sch.vpd-cluster-aware-sae-budget-admission-policy-verification.v1",
        "status": "pass",
        "budget_policy_id": identifier,
        "activation_categories": len(observed["category_specific_budgets"]),
        "specific_compartments": len(observed["specific_compartment_budgets"]),
        "shared_fallback_compartments": 1,
        "jobs_ceiling": observed["global_ceiling"]["jobs"],
        "planned_steps_ceiling": observed["global_ceiling"]["planned_steps"],
        "planned_activation_rows_ceiling": observed["global_ceiling"][
            "planned_activation_rows"
        ],
        "shared_fallback_charged_to_categories": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def _request_from_budget(budget: dict[str, Any]) -> dict[str, Any]:
    return {
        field_name: (list(value) if isinstance(value, list) else value)
        for field_name, value in budget["ceiling"].items()
    }


def build_reference_proposal(policy: dict[str, Any]) -> dict[str, Any]:
    validate_budget_policy_content(policy)
    body = {
        "schema": PROPOSAL_SCHEMA,
        "status": "synthetic_reference_proposal_nonexecuting",
        "source": {"budget_policy_id": policy["budget_policy_id"]},
        "category_specific_requests": [
            {
                "activation_category": budget["activation_category"],
                "specific_compartment_ids": list(budget["specific_compartment_ids"]),
                "requested_resources": _request_from_budget(budget),
            }
            for budget in policy["category_specific_budgets"]
        ],
        "specific_compartment_requests": [
            {
                "compartment_id": budget["compartment_id"],
                "activation_categories": list(budget["activation_categories"]),
                "requested_resources": _request_from_budget(budget),
            }
            for budget in policy["specific_compartment_budgets"]
        ],
        "shared_fallback_request": {
            "compartment_id": policy["shared_fallback_budget"]["compartment_id"],
            "activation_categories": list(
                policy["shared_fallback_budget"]["activation_categories"]
            ),
            "requested_resources": _request_from_budget(
                policy["shared_fallback_budget"]
            ),
            "charged_to_category_budgets": False,
        },
        "global_requested_resources": {
            field_name: (list(value) if isinstance(value, list) else value)
            for field_name, value in policy["global_ceiling"].items()
        },
        "authority_claims": dict(FALSE_AUTHORITY_CLAIMS),
    }
    return {**body, "proposal_id": content_sha256(body)}


def validate_budget_policy_content(policy: dict[str, Any]) -> None:
    if policy.get("schema") != POLICY_SCHEMA:
        raise ValueError("budget policy schema mismatch")
    body = dict(policy)
    identifier = _content_id(body.pop("budget_policy_id", ""), field_name="budget policy ID")
    if identifier != EXPECTED_POLICY_ID or identifier != content_sha256(body):
        raise ValueError("budget policy content address mismatch")
    if policy.get("scientific_boundary", {}).get("execution_authorized") is not False:
        raise ValueError("budget policy cannot transfer execution authority")


def _check_requested_resources(request: dict[str, Any], ceiling: dict[str, Any]) -> None:
    if set(request) != set(COUNT_FIELDS) | set(EXACT_LIST_FIELDS):
        raise ValueError("requested resource fields mismatch")
    for field_name in COUNT_FIELDS:
        value = request[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"requested {field_name} must be a positive integer")
        if value > ceiling[field_name]:
            raise ValueError(f"requested {field_name} exceeds bound ceiling")
    if request["unique_output_directories"] != request["jobs"]:
        raise ValueError("requested outputs must remain unique per job")
    for field_name in EXACT_LIST_FIELDS:
        if request[field_name] != ceiling[field_name]:
            raise ValueError(f"requested {field_name} must match bound resource shape")


def _aggregate_requests(requests: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **{
            field_name: sum(row["requested_resources"][field_name] for row in requests)
            for field_name in COUNT_FIELDS
        },
        **{
            field_name: sorted(
                {
                    value
                    for row in requests
                    for value in row["requested_resources"][field_name]
                }
            )
            for field_name in EXACT_LIST_FIELDS
        },
    }


def evaluate_budget_proposal(
    *, policy: dict[str, Any], proposal: dict[str, Any]
) -> dict[str, Any]:
    validate_budget_policy_content(policy)
    if proposal.get("schema") != PROPOSAL_SCHEMA:
        raise ValueError("budget proposal schema mismatch")
    proposal_body = dict(proposal)
    proposal_id = _content_id(
        proposal_body.pop("proposal_id", ""), field_name="budget proposal ID"
    )
    if proposal_id != content_sha256(proposal_body):
        raise ValueError("budget proposal content address mismatch")
    if proposal.get("source") != {"budget_policy_id": policy["budget_policy_id"]}:
        raise ValueError("budget proposal source mismatch")
    if proposal.get("authority_claims") != FALSE_AUTHORITY_CLAIMS:
        raise ValueError("budget proposal must transfer no execution or training authority")

    category_budgets = {
        row["activation_category"]: row for row in policy["category_specific_budgets"]
    }
    category_requests = proposal.get("category_specific_requests")
    if not isinstance(category_requests, list):
        raise ValueError("category-specific requests must be a list")
    category_by_name: dict[str, dict[str, Any]] = {}
    for row in category_requests:
        name = str(row.get("activation_category", ""))
        if name in category_by_name or name not in category_budgets:
            raise ValueError("category request is duplicated or unknown")
        budget = category_budgets[name]
        if row.get("specific_compartment_ids") != budget["specific_compartment_ids"]:
            raise ValueError("category request compartment binding mismatch")
        _check_requested_resources(row.get("requested_resources", {}), budget["ceiling"])
        category_by_name[name] = row
    if set(category_by_name) != set(category_budgets):
        raise ValueError("all eight activation categories are required")

    specific_budgets = {
        row["compartment_id"]: row for row in policy["specific_compartment_budgets"]
    }
    specific_requests = proposal.get("specific_compartment_requests")
    if not isinstance(specific_requests, list):
        raise ValueError("specific compartment requests must be a list")
    specific_by_id: dict[str, dict[str, Any]] = {}
    for row in specific_requests:
        compartment_id = str(row.get("compartment_id", ""))
        if compartment_id in specific_by_id or compartment_id not in specific_budgets:
            raise ValueError("specific compartment request is duplicated or unknown")
        budget = specific_budgets[compartment_id]
        if row.get("activation_categories") != budget["activation_categories"]:
            raise ValueError("specific compartment category binding mismatch")
        _check_requested_resources(row.get("requested_resources", {}), budget["ceiling"])
        specific_by_id[compartment_id] = row
    if set(specific_by_id) != set(specific_budgets):
        raise ValueError("all sixteen specific compartments are required")

    for name, category_request in category_by_name.items():
        compartment_requests = [
            specific_by_id[compartment_id]
            for compartment_id in category_budgets[name]["specific_compartment_ids"]
        ]
        if _aggregate_requests(compartment_requests) != category_request[
            "requested_resources"
        ]:
            raise ValueError("category totals must equal bound specific-compartment sums")

    shared_request = proposal.get("shared_fallback_request")
    shared_budget = policy["shared_fallback_budget"]
    if not isinstance(shared_request, dict):
        raise ValueError("one shared fallback request is required")
    if (
        shared_request.get("compartment_id") != shared_budget["compartment_id"]
        or shared_request.get("activation_categories")
        != shared_budget["activation_categories"]
        or shared_request.get("charged_to_category_budgets") is not False
    ):
        raise ValueError("shared fallback must remain separately bound and uncharged")
    _check_requested_resources(
        shared_request.get("requested_resources", {}), shared_budget["ceiling"]
    )

    global_requests = specific_requests + [shared_request]
    aggregate = _aggregate_requests(global_requests)
    if proposal.get("global_requested_resources") != aggregate:
        raise ValueError("global request must equal specific compartments plus shared once")
    _check_requested_resources(aggregate, policy["global_ceiling"])

    body = {
        "schema": ADMISSION_SCHEMA,
        "status": "budget_shape_admissible_nonexecuting",
        "source": {
            "budget_policy_id": policy["budget_policy_id"],
            "proposal_id": proposal_id,
        },
        "admitted_shape": {
            "activation_categories": 8,
            "specific_compartments": 16,
            "shared_fallback_compartments": 1,
            "shared_fallback_charged_to_categories": False,
            "requested_resources": aggregate,
        },
        "budget_admissible": True,
        "fresh_empirical_blueprint_required": True,
        "fresh_active_signed_receipts_required": True,
        "fresh_nonces_and_output_reservations_required": True,
        "scientific_boundary": {
            "budget_admission_is_execution_authority": False,
            "empirical_evidence_created": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "training_performed": False,
            "persistent_dictionary_output_created": False,
            **FALSE_AUTHORITY_CLAIMS,
        },
    }
    return {**body, "admission_id": content_sha256(body)}


def canonical_inputs(project_root: Path) -> dict[str, Path]:
    return {
        "resource_ledger_path": project_root
        / "artifacts/vpd_cluster_aware_sae_compartment_resource_reconciliation_v1/ledger.json"
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify nonexecuting cluster-aware SAE budget policy V1."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.output is not None:
        result = build_budget_policy(**inputs)
        write_json(args.output, result)
    else:
        result = verify_budget_policy(args.verify, **inputs)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "budget_policy_id": result["budget_policy_id"],
                "activation_categories": result.get(
                    "activation_categories", len(result.get("category_specific_budgets", []))
                ),
                "specific_compartments": result.get(
                    "specific_compartments", len(result.get("specific_compartment_budgets", []))
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
