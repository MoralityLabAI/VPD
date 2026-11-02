from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.build_cluster_aware_sae_budget_admission_policy_v1 import (
    ADMISSION_SCHEMA as ADMISSION_SCHEMA_V1,
    EXACT_LIST_FIELDS,
    FALSE_AUTHORITY_CLAIMS,
    POLICY_SCHEMA,
    PROPOSAL_SCHEMA as PROPOSAL_SCHEMA_V1,
    evaluate_budget_proposal,
    validate_budget_policy_content,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.run_cluster_aware_sae_training_job_v1 import (
    validate_execution_blueprint_content,
)


PROPOSAL_SCHEMA = "sch.vpd-cluster-aware-sae-blueprint-bound-budget-proposal.v2"
ADMISSION_SCHEMA = "sch.vpd-cluster-aware-sae-blueprint-bound-budget-admission.v2"
DIGEST_RE = re.compile(r"[0-9a-f]{64}")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _digest(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"blueprint-bound budget {field} is not a SHA-256 digest")
    return value


def _aggregate_requests(requests: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "jobs": sum(int(row["jobs"]) for row in requests),
        "planned_steps": sum(int(row["planned_steps"]) for row in requests),
        "planned_activation_rows": sum(
            int(row["planned_activation_rows"]) for row in requests
        ),
        "episode_contract_entries": sum(
            int(row["episode_contract_entries"]) for row in requests
        ),
        "pair_contract_entries": sum(
            int(row["pair_contract_entries"]) for row in requests
        ),
        "unique_output_directories": sum(
            int(row["unique_output_directories"]) for row in requests
        ),
        **{
            field: sorted({item for row in requests for item in row[field]})
            for field in EXACT_LIST_FIELDS
        },
    }


def _job_group_request(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    if not jobs:
        raise ValueError("blueprint-bound budget compartment has no jobs")
    output_dirs = {str(job["output_dir"]) for job in jobs}
    if len(output_dirs) != len(jobs):
        raise ValueError("blueprint-bound budget compartment reuses an output")
    return {
        "jobs": len(jobs),
        "planned_steps": sum(int(job["profile"]["steps"]) for job in jobs),
        "planned_activation_rows": sum(
            int(job["exact_sample_contract"]["sae_activation_rows"])
            for job in jobs
        ),
        "episode_contract_entries": sum(
            len(job["exact_sample_contract"]["episode_ids"]) for job in jobs
        ),
        "pair_contract_entries": sum(
            len(job["exact_sample_contract"]["pair_ids"]) for job in jobs
        ),
        "unique_output_directories": len(output_dirs),
        "batch_sizes": sorted({int(job["profile"]["batch_size"]) for job in jobs}),
        "data_selection_seeds": sorted(
            {int(job["data_selection_seed"]) for job in jobs}
        ),
        "dictionary_feature_widths": sorted(
            {int(job["dictionary_features"]) for job in jobs}
        ),
        "initialization_seeds": sorted(
            {int(job["initialization_seed"]) for job in jobs}
        ),
    }


def _legacy_proposal(
    *, policy: dict[str, Any], proposal: dict[str, Any]
) -> dict[str, Any]:
    body = {
        "schema": PROPOSAL_SCHEMA_V1,
        "status": "blueprint_derived_legacy_policy_projection_nonexecuting",
        "source": {"budget_policy_id": policy["budget_policy_id"]},
        "category_specific_requests": proposal["category_specific_requests"],
        "specific_compartment_requests": proposal["specific_compartment_requests"],
        "shared_fallback_request": proposal["shared_fallback_request"],
        "global_requested_resources": proposal["global_requested_resources"],
        "authority_claims": dict(FALSE_AUTHORITY_CLAIMS),
    }
    return {**body, "proposal_id": content_sha256(body)}


def build_blueprint_bound_proposal(
    *,
    policy: dict[str, Any],
    blueprint: dict[str, Any],
    blueprint_sha256: str,
) -> dict[str, Any]:
    validate_budget_policy_content(policy)
    blueprint_check = validate_execution_blueprint_content(blueprint)
    blueprint_file_sha256 = _digest(
        blueprint_sha256, field="execution blueprint file SHA-256"
    )
    jobs = list(blueprint["execution_jobs"])
    by_compartment: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        by_compartment.setdefault(str(job["compartment_id"]), []).append(job)
    specific_budgets = {
        str(row["compartment_id"]): row
        for row in policy["specific_compartment_budgets"]
    }
    shared_budget = policy["shared_fallback_budget"]
    shared_id = str(shared_budget["compartment_id"])
    if set(by_compartment) != set(specific_budgets) | {shared_id}:
        raise ValueError("blueprint compartments differ from the budget policy")

    specific_requests = []
    request_by_id: dict[str, dict[str, Any]] = {}
    for budget in policy["specific_compartment_budgets"]:
        compartment_id = str(budget["compartment_id"])
        group = by_compartment[compartment_id]
        categories = sorted(
            {str(category) for job in group for category in job["activation_categories"]}
        )
        if categories != list(budget["activation_categories"]) or any(
            job["compartment_kind"] != "specific_cluster" for job in group
        ):
            raise ValueError("blueprint specific-compartment category binding differs")
        request = _job_group_request(group)
        row = {
            "compartment_id": compartment_id,
            "activation_categories": categories,
            "execution_job_ids": sorted(str(job["execution_job_id"]) for job in group),
            "requested_resources": request,
        }
        specific_requests.append(row)
        request_by_id[compartment_id] = request

    category_requests = []
    for budget in policy["category_specific_budgets"]:
        compartment_ids = list(budget["specific_compartment_ids"])
        category_requests.append(
            {
                "activation_category": budget["activation_category"],
                "specific_compartment_ids": compartment_ids,
                "requested_resources": _aggregate_requests(
                    [request_by_id[value] for value in compartment_ids]
                ),
            }
        )

    shared_jobs = by_compartment[shared_id]
    if any(job["compartment_kind"] != "shared_fallback" for job in shared_jobs):
        raise ValueError("blueprint shared fallback kind differs")
    shared_categories = sorted(
        {
            str(category)
            for job in shared_jobs
            for category in job["activation_categories"]
        }
    )
    if shared_categories != list(shared_budget["activation_categories"]):
        raise ValueError("blueprint shared-fallback category binding differs")
    shared_request = {
        "compartment_id": shared_id,
        "activation_categories": shared_categories,
        "execution_job_ids": sorted(
            str(job["execution_job_id"]) for job in shared_jobs
        ),
        "requested_resources": _job_group_request(shared_jobs),
        "charged_to_category_budgets": False,
    }
    global_resources = _aggregate_requests(
        [row["requested_resources"] for row in specific_requests]
        + [shared_request["requested_resources"]]
    )
    body = {
        "schema": PROPOSAL_SCHEMA,
        "status": "exact_execution_blueprint_budget_proposal_nonexecuting",
        "source": {
            "budget_policy_id": policy["budget_policy_id"],
            "execution_blueprint_id": blueprint_check["execution_blueprint_id"],
            "execution_blueprint_sha256": blueprint_file_sha256,
        },
        "blueprint_shape": {
            "evidence_kind": blueprint_check["evidence_kind"],
            "materialization_id": blueprint_check["materialization_id"],
            "cluster_aware_sae_plan_id": blueprint_check[
                "cluster_aware_sae_plan_id"
            ],
            "global_pair_split_id": blueprint_check["global_pair_split_id"],
            "execution_jobs": blueprint_check["execution_jobs"],
            "unique_output_directories": len(
                {str(job["output_dir"]) for job in jobs}
            ),
        },
        "category_specific_requests": category_requests,
        "specific_compartment_requests": specific_requests,
        "shared_fallback_request": shared_request,
        "global_requested_resources": global_resources,
        "authority_claims": dict(FALSE_AUTHORITY_CLAIMS),
    }
    return {**body, "proposal_id": content_sha256(body)}


def evaluate_blueprint_bound_budget_proposal(
    *,
    policy: dict[str, Any],
    blueprint: dict[str, Any],
    blueprint_sha256: str,
    proposal: dict[str, Any],
) -> dict[str, Any]:
    expected = build_blueprint_bound_proposal(
        policy=policy,
        blueprint=blueprint,
        blueprint_sha256=blueprint_sha256,
    )
    if proposal != expected:
        raise ValueError("blueprint-bound budget proposal is not reproducible")
    legacy = _legacy_proposal(policy=policy, proposal=proposal)
    legacy_admission = evaluate_budget_proposal(policy=policy, proposal=legacy)
    if legacy_admission.get("schema") != ADMISSION_SCHEMA_V1:
        raise ValueError("blueprint-bound budget legacy admission differs")
    body = {
        "schema": ADMISSION_SCHEMA,
        "status": "exact_blueprint_budget_admissible_nonexecuting",
        "source": {
            "budget_policy_id": policy["budget_policy_id"],
            "proposal_id": proposal["proposal_id"],
            "execution_blueprint_id": proposal["source"]["execution_blueprint_id"],
            "execution_blueprint_sha256": proposal["source"]
            ["execution_blueprint_sha256"],
            "legacy_policy_projection_id": legacy["proposal_id"],
            "legacy_budget_admission_id": legacy_admission["admission_id"],
        },
        "blueprint_shape": proposal["blueprint_shape"],
        "admitted_shape": legacy_admission["admitted_shape"],
        "budget_admissible": True,
        "blueprint_binding_verified": True,
        "exact_job_resources_derived_from_blueprint": True,
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


def build_from_paths(
    *, policy_path: Path, blueprint_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = _load(policy_path)
    blueprint = _load(blueprint_path)
    blueprint_hash = sha256_file(blueprint_path)
    proposal = build_blueprint_bound_proposal(
        policy=policy,
        blueprint=blueprint,
        blueprint_sha256=blueprint_hash,
    )
    admission = evaluate_blueprint_bound_budget_proposal(
        policy=policy,
        blueprint=blueprint,
        blueprint_sha256=blueprint_hash,
        proposal=proposal,
    )
    return proposal, admission


def verify_proposal(
    path: Path, *, policy_path: Path, blueprint_path: Path
) -> dict[str, Any]:
    observed = _load(path)
    proposal, _ = build_from_paths(
        policy_path=policy_path, blueprint_path=blueprint_path
    )
    if observed != proposal:
        raise ValueError("blueprint-bound budget proposal differs")
    return {
        "schema": "sch.vpd-cluster-aware-sae-blueprint-bound-budget-proposal-verification.v2",
        "status": "pass",
        "proposal_id": observed["proposal_id"],
        "execution_blueprint_id": observed["source"]["execution_blueprint_id"],
        "execution_blueprint_sha256": observed["source"]
        ["execution_blueprint_sha256"],
        "execution_jobs": observed["blueprint_shape"]["execution_jobs"],
        "budget_admissible": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def verify_admission(
    path: Path,
    *,
    proposal_path: Path,
    policy_path: Path,
    blueprint_path: Path,
) -> dict[str, Any]:
    proposal = _load(proposal_path)
    policy = _load(policy_path)
    blueprint = _load(blueprint_path)
    observed = _load(path)
    expected = evaluate_blueprint_bound_budget_proposal(
        policy=policy,
        blueprint=blueprint,
        blueprint_sha256=sha256_file(blueprint_path),
        proposal=proposal,
    )
    if observed != expected:
        raise ValueError("blueprint-bound budget admission differs")
    return {
        "schema": "sch.vpd-cluster-aware-sae-blueprint-bound-budget-admission-verification.v2",
        "status": "pass",
        "admission_id": observed["admission_id"],
        "proposal_id": observed["source"]["proposal_id"],
        "execution_blueprint_id": observed["source"]["execution_blueprint_id"],
        "execution_blueprint_sha256": observed["source"]
        ["execution_blueprint_sha256"],
        "execution_jobs": observed["blueprint_shape"]["execution_jobs"],
        "budget_admissible": True,
        "blueprint_binding_verified": True,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify exact blueprint-bound budget admission V2."
    )
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--blueprint", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path)
    mode.add_argument("--verify-proposal", type=Path)
    mode.add_argument("--verify-admission", type=Path)
    parser.add_argument("--proposal", type=Path)
    args = parser.parse_args()
    if args.output_dir is not None:
        if args.proposal is not None:
            raise ValueError("--proposal is only valid with --verify-admission")
        proposal, admission = build_from_paths(
            policy_path=args.policy, blueprint_path=args.blueprint
        )
        write_json(args.output_dir / "proposal.json", proposal)
        write_json(args.output_dir / "admission.json", admission)
        result = admission
    elif args.verify_proposal is not None:
        if args.proposal is not None:
            raise ValueError("--proposal is only valid with --verify-admission")
        result = verify_proposal(
            args.verify_proposal,
            policy_path=args.policy,
            blueprint_path=args.blueprint,
        )
    else:
        if args.proposal is None:
            raise ValueError("--verify-admission requires --proposal")
        result = verify_admission(
            args.verify_admission,
            proposal_path=args.proposal,
            policy_path=args.policy,
            blueprint_path=args.blueprint,
        )
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "proposal_id": result.get(
                    "proposal_id", result.get("source", {}).get("proposal_id")
                ),
                "admission_id": result.get("admission_id"),
                "execution_blueprint_id": result.get(
                    "execution_blueprint_id",
                    result.get("source", {}).get("execution_blueprint_id"),
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
