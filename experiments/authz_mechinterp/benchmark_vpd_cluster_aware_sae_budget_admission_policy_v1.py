from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.build_cluster_aware_sae_budget_admission_policy_v1 import (
    FALSE_AUTHORITY_CLAIMS,
    build_reference_proposal,
    canonical_inputs as policy_inputs,
    evaluate_budget_proposal,
    verify_budget_policy,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (KeyError, TypeError, ValueError):
        return True
    return False


def _readdress_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in proposal.items() if key != "proposal_id"}
    return {**body, "proposal_id": content_sha256(body)}


def _readdress_policy(policy: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in policy.items() if key != "budget_policy_id"}
    return {**body, "budget_policy_id": content_sha256(body)}


def build_report(project_root: Path) -> dict[str, Any]:
    before_frameworks = MODEL_FRAMEWORKS & set(sys.modules)
    policy_path = (
        project_root
        / "artifacts/vpd_cluster_aware_sae_budget_admission_policy_v1/policy.json"
    )
    verification = verify_budget_policy(policy_path, **policy_inputs(project_root))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    proposal = build_reference_proposal(policy)
    admission = evaluate_budget_proposal(policy=policy, proposal=proposal)

    def mutated_proposal(mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        changed = copy.deepcopy(proposal)
        mutator(changed)
        return _readdress_proposal(changed)

    over_category = mutated_proposal(
        lambda value: value["category_specific_requests"][0]["requested_resources"].__setitem__(
            "planned_steps",
            value["category_specific_requests"][0]["requested_resources"]["planned_steps"]
            + 1,
        )
    )

    def broaden_specific(value: dict[str, Any]) -> None:
        specific = value["specific_compartment_requests"][0]
        category = specific["activation_categories"][0]
        specific["requested_resources"]["planned_steps"] += 1
        next(
            row
            for row in value["category_specific_requests"]
            if row["activation_category"] == category
        )["requested_resources"]["planned_steps"] += 1
        value["global_requested_resources"]["planned_steps"] += 1

    over_specific = mutated_proposal(broaden_specific)

    def broaden_shared(value: dict[str, Any]) -> None:
        value["shared_fallback_request"]["requested_resources"]["planned_steps"] += 1
        value["global_requested_resources"]["planned_steps"] += 1

    over_shared = mutated_proposal(broaden_shared)
    shared_charged = mutated_proposal(
        lambda value: value["shared_fallback_request"].__setitem__(
            "charged_to_category_budgets", True
        )
    )
    global_double_count = mutated_proposal(
        lambda value: value["global_requested_resources"].__setitem__(
            "planned_steps",
            value["global_requested_resources"]["planned_steps"]
            + value["shared_fallback_request"]["requested_resources"]["planned_steps"],
        )
    )
    missing_category = mutated_proposal(
        lambda value: value["category_specific_requests"].pop()
    )
    duplicate_category = mutated_proposal(
        lambda value: value["category_specific_requests"].append(
            copy.deepcopy(value["category_specific_requests"][0])
        )
    )
    unknown_category = mutated_proposal(
        lambda value: value["category_specific_requests"][0].__setitem__(
            "activation_category", "unknown"
        )
    )
    missing_compartment = mutated_proposal(
        lambda value: value["specific_compartment_requests"].pop()
    )
    duplicate_compartment = mutated_proposal(
        lambda value: value["specific_compartment_requests"].append(
            copy.deepcopy(value["specific_compartment_requests"][0])
        )
    )
    unknown_compartment = mutated_proposal(
        lambda value: value["specific_compartment_requests"][0].__setitem__(
            "compartment_id", "unknown"
        )
    )
    wrong_category_binding = mutated_proposal(
        lambda value: value["category_specific_requests"][0].__setitem__(
            "specific_compartment_ids",
            list(value["category_specific_requests"][1]["specific_compartment_ids"]),
        )
    )
    wrong_compartment_binding = mutated_proposal(
        lambda value: value["specific_compartment_requests"][0].__setitem__(
            "activation_categories", ["other_invalid"]
        )
    )
    wrong_specific_width = mutated_proposal(
        lambda value: value["specific_compartment_requests"][0][
            "requested_resources"
        ].__setitem__("dictionary_feature_widths", [8192])
    )
    wrong_shared_width = mutated_proposal(
        lambda value: value["shared_fallback_request"]["requested_resources"].__setitem__(
            "dictionary_feature_widths", [4096]
        )
    )
    reused_output = mutated_proposal(
        lambda value: value["specific_compartment_requests"][0][
            "requested_resources"
        ].__setitem__(
            "unique_output_directories",
            value["specific_compartment_requests"][0]["requested_resources"]["jobs"]
            - 1,
        )
    )
    zero_steps = mutated_proposal(
        lambda value: value["specific_compartment_requests"][0][
            "requested_resources"
        ].__setitem__("planned_steps", 0)
    )
    wrong_source = mutated_proposal(
        lambda value: value["source"].__setitem__("budget_policy_id", "f" * 64)
    )
    authority_broadening = mutated_proposal(
        lambda value: value["authority_claims"].__setitem__(
            "sae_training_authorized", True
        )
    )
    content_tamper = copy.deepcopy(proposal)
    content_tamper["global_requested_resources"]["planned_steps"] -= 1
    policy_tamper = copy.deepcopy(policy)
    policy_tamper["global_ceiling"]["planned_steps"] += 1
    readdressed_policy_tamper = _readdress_policy(policy_tamper)

    def rejects(candidate: dict[str, Any], *, candidate_policy: dict[str, Any] = policy) -> bool:
        return _rejects(
            lambda: evaluate_budget_proposal(
                policy=candidate_policy,
                proposal=candidate,
            )
        )

    controls = {
        "policy_reproduces_from_pinned_reconciliation_ledger": verification["status"]
        == "pass",
        "reference_proposal_is_content_addressed": proposal["proposal_id"]
        == content_sha256(
            {key: value for key, value in proposal.items() if key != "proposal_id"}
        ),
        "exact_reference_shape_is_budget_admissible": admission["budget_admissible"]
        is True,
        "admission_covers_exactly_8_categories": admission["admitted_shape"][
            "activation_categories"
        ]
        == 8,
        "admission_covers_exactly_16_specific_compartments": admission[
            "admitted_shape"
        ]["specific_compartments"]
        == 16,
        "shared_fallback_is_present_once_and_uncharged": admission["admitted_shape"]
        == {
            "activation_categories": 8,
            "specific_compartments": 16,
            "shared_fallback_compartments": 1,
            "shared_fallback_charged_to_categories": False,
            "requested_resources": proposal["global_requested_resources"],
        },
        "global_exact_shape_is_34_jobs_68000_steps_20480_rows": all(
            admission["admitted_shape"]["requested_resources"][field_name]
            == expected
            for field_name, expected in {
                "jobs": 34,
                "planned_steps": 68000,
                "planned_activation_rows": 20480,
                "unique_output_directories": 34,
            }.items()
        ),
        "specific_and_shared_dictionary_widths_remain_distinct": all(
            row["requested_resources"]["dictionary_feature_widths"] == [4096]
            for row in proposal["specific_compartment_requests"]
        )
        and proposal["shared_fallback_request"]["requested_resources"][
            "dictionary_feature_widths"
        ]
        == [8192],
        "category_ceiling_broadening_rejected": rejects(over_category),
        "specific_compartment_ceiling_broadening_rejected": rejects(over_specific),
        "shared_fallback_ceiling_broadening_rejected": rejects(over_shared),
        "shared_fallback_category_charge_rejected": rejects(shared_charged),
        "shared_fallback_global_double_count_rejected": rejects(global_double_count),
        "missing_category_rejected": rejects(missing_category),
        "duplicate_category_rejected": rejects(duplicate_category),
        "unknown_category_rejected": rejects(unknown_category),
        "missing_specific_compartment_rejected": rejects(missing_compartment),
        "duplicate_specific_compartment_rejected": rejects(duplicate_compartment),
        "unknown_specific_compartment_rejected": rejects(unknown_compartment),
        "category_to_compartment_rebinding_rejected": rejects(wrong_category_binding),
        "compartment_to_category_rebinding_rejected": rejects(wrong_compartment_binding),
        "specific_dictionary_width_substitution_rejected": rejects(
            wrong_specific_width
        ),
        "shared_dictionary_width_substitution_rejected": rejects(wrong_shared_width),
        "output_directory_reuse_rejected": rejects(reused_output),
        "nonpositive_resource_request_rejected": rejects(zero_steps),
        "wrong_policy_source_rejected": rejects(wrong_source),
        "authority_broadening_rejected": rejects(authority_broadening),
        "proposal_content_tamper_rejected": rejects(content_tamper),
        "policy_content_tamper_rejected": rejects(proposal, candidate_policy=policy_tamper),
        "readdressed_policy_tamper_rejected": rejects(
            proposal, candidate_policy=readdressed_policy_tamper
        ),
        "budget_admission_transfers_no_authority": admission["scientific_boundary"]
        == {
            "budget_admission_is_execution_authority": False,
            "empirical_evidence_created": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "training_performed": False,
            "persistent_dictionary_output_created": False,
            **FALSE_AUTHORITY_CLAIMS,
        },
        "benchmark_imports_no_model_frameworks": before_frameworks
        == (MODEL_FRAMEWORKS & set(sys.modules)),
    }
    if not all(controls.values()):
        raise ValueError(
            "cluster-aware SAE budget admission benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-aware-sae-budget-admission-policy-benchmark.v1",
        "status": "pass",
        "source": {
            "budget_policy_path": policy_path.relative_to(project_root).as_posix(),
            "budget_policy_sha256": sha256_file(policy_path),
            "budget_policy_id": policy["budget_policy_id"],
            "budget_policy_builder_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/build_cluster_aware_sae_budget_admission_policy_v1.py"
            ),
        },
        "controls": controls,
        "summary": {
            "passed": sum(controls.values()),
            "controls": len(controls),
            "activation_categories": 8,
            "specific_compartments": 16,
            "shared_fallback_compartments": 1,
            "jobs_ceiling": 34,
            "planned_steps_ceiling": 68000,
            "planned_activation_rows_ceiling": 20480,
            "shared_fallback_charged_to_categories": False,
        },
        "reference_admission": admission,
        "scientific_boundary": {
            "synthetic_reference_proposal_only": True,
            "empirical_blueprint_present": False,
            "empirical_evidence_created": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "training_performed": False,
            "weights_mutated": False,
            "persistent_dictionary_output_created": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
            "paid_compute_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body) or observed != build_report(project_root):
        raise ValueError("cluster-aware SAE budget admission benchmark is not reproducible")
    return {
        "schema": "sch.vpd-cluster-aware-sae-budget-admission-policy-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "budget_policy_id": observed["source"]["budget_policy_id"],
        "controls": observed["summary"]["controls"],
        "activation_categories": observed["summary"]["activation_categories"],
        "compartments": observed["summary"]["specific_compartments"]
        + observed["summary"]["shared_fallback_compartments"],
        "jobs_ceiling": observed["summary"]["jobs_ceiling"],
        "planned_steps_ceiling": observed["summary"]["planned_steps_ceiling"],
        "planned_activation_rows_ceiling": observed["summary"][
            "planned_activation_rows_ceiling"
        ],
        "shared_fallback_charged_to_categories": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify cluster-aware SAE budget admission benchmark V1."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.output is not None:
        result = build_report(root)
        write_json(args.output, result)
    else:
        result = verify_benchmark(args.verify, project_root=root)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "benchmark_id": result["benchmark_id"],
                "budget_policy_id": result.get(
                    "budget_policy_id", result.get("source", {}).get("budget_policy_id")
                ),
                "controls": result.get(
                    "controls", result.get("summary", {}).get("controls")
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
