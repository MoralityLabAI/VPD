"""Freeze an archive-only, utility-aware VPD selection audit.

This module deliberately does not load a model, perform inference, fit an SAE, or
select a live edit.  It converts the sealed activation-category evidence into a
small, checkable future-evaluation contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "sch.vpd-utility-aware-selection-audit.v1"
EXPECTED_SOURCE_SPEC_ID = (
    "737c643f880e79b378eb821a394b618f11fdf853282d942b4354a4fb4eb626dd"
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _audit_id(audit: dict[str, Any]) -> str:
    copy = dict(audit)
    copy.pop("audit_id", None)
    return hashlib.sha256(_canonical(copy)).hexdigest()


def _load_source_spec(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("schema") != "sch.vpd-activation-category-sae-experiment-spec.v1":
        raise ValueError("source is not the sealed activation-category SAE specification")
    if source.get("spec_id") != EXPECTED_SOURCE_SPEC_ID:
        raise ValueError("source activation-category specification ID differs")
    authority = source.get("current_authority", {})
    if any(
        authority.get(key) is not False
        for key in (
            "new_inference_authorized",
            "paid_compute_authorized",
            "sae_training_authorized",
            "model_or_adapter_weight_mutation_authorized",
        )
    ):
        raise ValueError("source specification no longer forbids empirical execution")
    evidence = source.get("evidence", {})
    capacity = evidence.get("capacity_screening", {})
    geometry = evidence.get("shared_axis_category_geometry", {})
    if capacity.get("rank_sweep_training_admissions") != 0:
        raise ValueError("separate-dictionary capacity evidence unexpectedly changed")
    if geometry.get("third_domain_transfer_established") is not False:
        raise ValueError("source unexpectedly claims third-domain transfer")
    return source


def build_audit(source_spec_path: Path, *, project_root: Path) -> dict[str, Any]:
    """Build the frozen audit from verified local evidence only."""
    source = _load_source_spec(source_spec_path)
    capacity = source["evidence"]["capacity_screening"]
    geometry = source["evidence"]["shared_axis_category_geometry"]
    source_relative = source_spec_path.resolve().relative_to(project_root.resolve())

    audit: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "archive_only_design_no_model_inference_or_selection_executed",
        "authority": {
            "archive_only": True,
            "new_model_inference_authorized": False,
            "model_or_adapter_weight_mutation_authorized": False,
            "sae_training_authorized": False,
            "live_coordinate_edit_authorized": False,
            "paid_compute_authorized": False,
        },
        "evidence_boundary": {
            "source_spec": {
                "path": source_relative.as_posix(),
                "sha256": _sha256_file(source_spec_path),
                "spec_id": source["spec_id"],
            },
            "separate_pass_dictionary_admissions": capacity[
                "rank_sweep_training_admissions"
            ],
            "equal_total_rank_checks": capacity["rank_sweep_archive_site_rank_checks"],
            "monolithic_wins_or_ties": capacity[
                "rank_sweep_monolithic_wins_vs_pass"
            ]
            + capacity["rank_sweep_monolithic_vs_pass_inconclusive"],
            "shared_axis_is_limited_to_existing_packs": True,
            "third_domain_transfer_established": geometry[
                "third_domain_transfer_established"
            ],
            "semantic_attribution_established": False,
            "loader_conformance_is_not_scientific_validation": True,
            "consequence": (
                "the incumbent shared monolithic dictionary remains the only "
                "current architecture; this audit makes no positive selection claim"
            ),
        },
        "frozen_catalog": {
            "catalog_version": 1,
            "candidate_ids": [
                "q_monolithic",
                "q_geometric",
                "q_interval_aware",
            ],
            "candidate_count": 3,
            "shared_constraints": {
                "base_representation": "one pre-existing shared monolithic dictionary",
                "total_rank": 64,
                "maximum_edited_coordinates": 64,
                "additional_dictionary_fits": 0,
                "additional_model_calls_in_this_archive": 0,
                "same_frozen_case_blocks_and_candidate_pool": True,
                "selection_uses_discovery_and_calibration_blocks_only": True,
                "final_evaluation_blocks_are_never_used_to_choose_a_candidate": True,
            },
            "candidates": [
                {
                    "id": "q_monolithic",
                    "kind": "global_utility_selector",
                    "selection_rule": (
                        "choose the rank-64 coordinate set with greatest calibration "
                        "mean net utility among the frozen candidate pool"
                    ),
                    "prohibited_inputs": [
                        "final_evaluation_outcomes",
                        "pass_labels_as_a_selection_signal",
                        "shared_axis_geometry",
                    ],
                },
                {
                    "id": "q_geometric",
                    "kind": "shared_axis_geometric_selector",
                    "selection_rule": (
                        "choose the rank-64 set fixed by training-only projection onto "
                        "the predeclared shared pass axis, then use calibration only for "
                        "a fixed threshold and sign"
                    ),
                    "prohibited_claims": [
                        "third_domain_transfer",
                        "semantic_attribution",
                        "evidence_for_a_separate_dictionary",
                    ],
                },
                {
                    "id": "q_interval_aware",
                    "kind": "challenger_aware_robust_selector",
                    "selection_rule": (
                        "among the same rank-64 pool, choose the candidate with the "
                        "largest simultaneous lower confidence bound on calibration net "
                        "utility after excluding candidates whose upper regret envelope "
                        "or benign-loss upper bound exceeds its predeclared limit"
                    ),
                    "prohibited_inputs": ["final_evaluation_outcomes"],
                },
            ],
        },
        "utility_target": {
            "name": "held_out_benign_task_utility_after_a_bounded_coordinate_edit",
            "per_case_definition": "task_score_after_edit minus task_score_no_edit",
            "primary_estimand": "mean net utility across independent final family blocks",
            "baseline_improvement_estimand": "mean_utility(q) minus mean_utility(no_edit)",
            "best_reachable_regret_estimand": (
                "max(mean_utility(p) for p in frozen feasible catalog) minus mean_utility(q)"
            ),
            "net_utility": (
                "raw task utility minus a preregistered access-cost penalty; raw utility "
                "and each physical cost remain separately reported"
            ),
            "no_edit_baseline_is_required": True,
        },
        "case_blocks_and_inference": {
            "independence_unit": "semantic family, never individual activation row or token",
            "roles": {
                "discovery": "freeze axis, coordinate pool, and all random seeds",
                "calibration": "run the three frozen selection rules and freeze q",
                "final_evaluation": "estimate utility, baseline improvement, and regret only",
            },
            "family_separation": {
                "pairwise_disjoint_roles": True,
                "held_out_family_control_required": True,
                "minimum_final_family_blocks": 8,
                "all_members_of_a_family_stay_in_one_role": True,
            },
            "simultaneous_interval": {
                "coverage": 0.95,
                "familywise_scope": [
                    "all three candidates versus no-edit baseline",
                    "all ordered candidate-pair utility differences",
                    "all benign-utility-loss comparisons",
                ],
                "method": (
                    "paired family-block studentized multiplier bootstrap using a fixed "
                    "20,000-draw seed and the max statistic over the declared family"
                ),
                "report_block_count_and_raw_block_scores": True,
            },
        },
        "finite_challenger_envelope": {
            "candidate_set": "the three frozen candidates feasible under every shared budget",
            "pairwise_effect": "theta_qp = mean_utility(q) minus mean_utility(p)",
            "pairwise_interval": "[L_qp, U_qp] is the 95% simultaneous interval for theta_qp",
            "baseline_interval": "[L_q0, U_q0] is the simultaneous interval versus no-edit",
            "regret_interval": {
                "lower": "max(0, max_p(-U_qp))",
                "upper": "max(0, max_p(-L_qp))",
                "meaning": "a finite, explicit bound on missed feasible challengers",
            },
            "required_per_selected_q": [
                "selected_q_id",
                "baseline_improvement_point_and_simultaneous_interval",
                "regret_point_and_challenger_aware_interval",
                "identity_of_each_challenger_attaining_the_regret_bounds",
                "infeasible_or_unmeasured_challengers_with_reason",
            ],
        },
        "cost_accounting": {
            "functional_fit_budget": {
                "total_rank_per_candidate": 64,
                "additional_dictionary_fits_per_candidate": 0,
                "maximum_edited_coordinates_per_case": 64,
                "equal_budget_required_before_any_comparison": True,
            },
            "physically_charged_access_costs_report_separately": [
                "model_calls_by_candidate_and_case_role",
                "forward_and_backward_flops_from_execution_receipts",
                "persistent_storage_bytes_and_artifact_hashes",
                "edited_coordinate_count_and_edit_norm",
                "benign_utility_loss_relative_to_no_edit",
            ],
            "archive_values": {
                "model_calls": 0,
                "flops": 0,
                "new_storage_bytes": 0,
                "edited_coordinates": 0,
                "benign_utility_loss_measured": False,
            },
            "costs_are_not_replaced_by_loader_or_format_conformance": True,
        },
        "controls": [
            {
                "id": "held_out_family",
                "rule": "no semantic family may occur in both selection and final evaluation",
                "failure_interpretation": "selection result is exploratory only",
            },
            {
                "id": "random_subspace",
                "rule": (
                    "evaluate a preseeded set of rank-64 random coordinate subspaces with "
                    "the same edit-count, case blocks, and accounting"
                ),
                "failure_interpretation": "geometry or utility selection has no demonstrated value",
            },
            {
                "id": "pass_label_only",
                "rule": (
                    "construct a selector from pass labels without activation geometry or "
                    "coordinate-level effect scores, at the same rank and edit budget"
                ),
                "failure_interpretation": "any geometric result is not separable from label routing",
            },
        ],
        "decision_rule": {
            "selection_is_not_admission": True,
            "a_candidate_is_reportable_as_better_than_no_edit_only_if": [
                "its simultaneous baseline-improvement lower bound exceeds zero",
                "its benign-utility-loss upper bound is at or below the preregistered limit",
                "its challenger-aware regret upper bound is at or below the preregistered tolerance",
                "all physical access costs and all control outcomes are present",
            ],
            "otherwise": "retain no-edit and the incumbent shared monolithic interpretation",
        },
        "falsifiers": [
            "any selected candidate has a nonpositive simultaneous lower bound versus no-edit",
            "a random-subspace control matches or exceeds the selected candidate within simultaneous intervals",
            "the pass-label-only control matches or exceeds the geometric selector within simultaneous intervals",
            "the selected candidate's regret upper bound exceeds its preregistered tolerance",
            "the benign-utility-loss upper bound exceeds its preregistered limit",
            "a final family block was used in discovery, calibration, or threshold setting",
            "rank, coordinate count, fit cost, calls, FLOPs, or storage differ without an explicit infeasibility record",
            "a claim of transfer or semantic attribution is made without a new independently held-out domain",
        ],
        "execution_gate": {
            "current_action": "preserve_as_archive_only_audit",
            "before_any_model_call": [
                "explicit_user_compute_authorization",
                "frozen family manifest and candidate-pool hash",
                "predeclared utility, benign-loss, regret, and access-cost thresholds",
                "durable model-call and resource receipt schema",
            ],
            "before_any_live_coordinate_edit": [
                "all_model_call_requirements",
                "explicit_user_live_edit_authorization",
            ],
        },
    }
    audit["audit_id"] = _audit_id(audit)
    return audit


def verify_audit(path: Path, *, project_root: Path) -> dict[str, Any]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("schema") != SCHEMA:
        raise ValueError("unexpected utility-aware selection audit schema")
    if audit.get("audit_id") != _audit_id(audit):
        raise ValueError("utility-aware selection audit ID differs")
    source_path = project_root / audit["evidence_boundary"]["source_spec"]["path"]
    rebuilt = build_audit(source_path, project_root=project_root)
    if audit != rebuilt:
        raise ValueError("utility-aware selection audit does not reproduce from pinned source")
    return {
        "schema": "sch.vpd-utility-aware-selection-audit-verification.v1",
        "status": "pass",
        "audit_id": audit["audit_id"],
        "execution_authorized": False,
        "model_inference_performed": False,
        "candidate_count": audit["frozen_catalog"]["candidate_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = build_audit(args.source_spec, project_root=args.project_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verify_audit(args.output, project_root=args.project_root), sort_keys=True))


if __name__ == "__main__":
    main()
