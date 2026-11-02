from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.build_activation_category_sae_experiment_spec_v1 import (
    verify_spec,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v51 import (
    canonical_inputs as protocol_v51_inputs,
    verify_protocol as verify_protocol_v51,
)


SCHEMA = "sch.vpd-activation-category-protocol.v52"
EXPECTED_V51_SHA256 = (
    "1b49c83ed86571a9fb9c9f54b768b0ee470123d9a77d63d1bdf7ef637ad2ba2c"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V52 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V51_SHA256:
        raise ValueError("protocol V51 differs from its pinned SHA-256")
    verification = verify_protocol_v51(path, **protocol_v51_inputs(project_root))
    if verification.get("status") != "pass":
        raise ValueError("protocol V51 did not reproduce")
    return verification


def _evidence_summary(spec: dict[str, Any], project_root: Path) -> dict[str, Any]:
    replica_matcher = spec["evidence"]["feature_replica_matching_method"]
    expected_replica_matcher = {
        "version": "v2",
        "synthetic_scenarios": 4,
        "expected_outcomes_reproduced": 4,
        "signed_decoder_cosine": True,
        "mutual_nearest_neighbor": True,
        "one_to_one": True,
        "required_for_fresh_seed_feature_identity": True,
        "v1_matcher_behavior_unchanged": True,
    }
    if any(
        replica_matcher.get(key) != value
        for key, value in expected_replica_matcher.items()
    ):
        raise ValueError("feature replica matching V2 contract differs")
    expected_roles = {
        "arithmetic_pass_replication",
        "broader_reasoning_pass_replication",
        "q4_temporal_region_negative_control_2048_tokens",
        "q4_temporal_region_negative_control_8192_tokens",
        "arithmetic_pass_compartment_linear_proxy_negative",
        "reasoning_pass_compartment_linear_proxy_negative",
        "arithmetic_pass_compartment_rank_sweep_negative",
        "reasoning_pass_compartment_rank_sweep_negative",
        "q4_2048_shared_sae_internal_pass_specialization",
        "q4_8192_shared_sae_internal_pass_specialization",
        "q4_2048_shared_sae_pass_selectivity_robustness",
        "q4_8192_shared_sae_pass_selectivity_robustness",
        "bf16_pass_category_cross_pack_transfer",
        "bf16_pass_category_shared_axis",
        "bf16_pass_category_prompt_phase_transfer",
        "bf16_activation_phase_category_screen",
        "bf16_activation_phase_category_cross_validation",
        "bf16_activation_phase_compartment_linear_proxy_negative",
        "bf16_joint_pass_phase_capacity_proxy_negative",
        "bf16_prompt_frozen_sae_quality_negative",
        "q4_generation_sae_cross_budget_pass_feature_identity",
        "q4_generation_sae_pass_feature_stratum_replication_v2",
        "layer21_sae_seed_objective_stability_negative_control",
        "layer21_sae_category_subspace_stability_negative_control",
        "q4_generation_sae_cross_budget_pass_category_subspace_transfer",
    }
    evidence_rows = spec["evidence"]["reports"]
    roles = {str(row["role"]) for row in evidence_rows}
    if roles != expected_roles:
        raise ValueError("empirical category evidence roles differ")
    reports: dict[str, dict[str, Any]] = {}
    for row in evidence_rows:
        path = project_root / str(row["path"])
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"empirical category report hash differs: {row['role']}")
        report = _load(path)
        if report.get("report_id") != row["report_id"]:
            raise ValueError(f"empirical category report ID differs: {row['role']}")
        reports[str(row["role"])] = report

    pass_reports = [
        reports["arithmetic_pass_replication"],
        reports["broader_reasoning_pass_replication"],
    ]
    region_reports = [
        reports["q4_temporal_region_negative_control_2048_tokens"],
        reports["q4_temporal_region_negative_control_8192_tokens"],
    ]
    linear_proxy_reports = [
        reports["arithmetic_pass_compartment_linear_proxy_negative"],
        reports["reasoning_pass_compartment_linear_proxy_negative"],
    ]
    rank_sweep_reports = [
        reports["arithmetic_pass_compartment_rank_sweep_negative"],
        reports["reasoning_pass_compartment_rank_sweep_negative"],
    ]
    sae_selectivity_reports = [
        reports["q4_2048_shared_sae_internal_pass_specialization"],
        reports["q4_8192_shared_sae_internal_pass_specialization"],
    ]
    sae_robustness_reports = [
        reports["q4_2048_shared_sae_pass_selectivity_robustness"],
        reports["q4_8192_shared_sae_pass_selectivity_robustness"],
    ]
    cross_pack_transfer = reports["bf16_pass_category_cross_pack_transfer"]
    shared_axis = reports["bf16_pass_category_shared_axis"]
    prompt_phase_transfer = reports["bf16_pass_category_prompt_phase_transfer"]
    phase_category = reports["bf16_activation_phase_category_screen"]
    phase_cross_validation = reports[
        "bf16_activation_phase_category_cross_validation"
    ]
    phase_linear_proxy = reports[
        "bf16_activation_phase_compartment_linear_proxy_negative"
    ]
    joint_pass_phase_capacity = reports[
        "bf16_joint_pass_phase_capacity_proxy_negative"
    ]
    prompt_frozen_sae_quality = reports[
        "bf16_prompt_frozen_sae_quality_negative"
    ]
    q4_sae_pass_feature_identity = reports[
        "q4_generation_sae_cross_budget_pass_feature_identity"
    ]
    q4_sae_pass_feature_stratum_v2 = reports[
        "q4_generation_sae_pass_feature_stratum_replication_v2"
    ]
    layer21_seed_objective_stability = reports[
        "layer21_sae_seed_objective_stability_negative_control"
    ]
    layer21_category_subspace_stability = reports[
        "layer21_sae_category_subspace_stability_negative_control"
    ]
    q4_sae_pass_category_subspace_transfer = reports[
        "q4_generation_sae_cross_budget_pass_category_subspace_transfer"
    ]
    pass_sites = sum(int(row["summary"]["sites"]) for row in pass_reports)
    pass_admissions = sum(
        int(row["summary"]["transition_category_admissions"])
        for row in pass_reports
    )
    region_combinations = sum(
        int(row["summary"]["site_pass_combinations"]) for row in region_reports
    )
    region_admissions = sum(
        int(row["summary"]["region_category_admissions"])
        for row in region_reports
    )
    screening_children = sum(
        int(row["summary"]["screening_only_within_region_nontrivial_allocations"])
        for row in region_reports
    )
    admissible_children = sum(
        int(row["summary"]["admissible_within_region_nontrivial_allocations"])
        for row in region_reports
    )
    if pass_admissions != pass_sites or region_admissions or admissible_children:
        raise ValueError("empirical activation-category gate outcome differs")
    linear_proxy_site_checks = sum(
        int(row["summary"]["sites"]) for row in linear_proxy_reports
    )
    linear_proxy_admissions = sum(
        int(row["summary"]["pass_compartment_screening_admissions"])
        for row in linear_proxy_reports
    )
    pass_wins_vs_random = sum(
        site["pass_minus_random_partition_mean_paired_bootstrap_95_ci"][0] > 0.0
        for report in linear_proxy_reports
        for site in report["sites"]
    )
    pass_wins_vs_monolithic = sum(
        site["pass_minus_monolithic_paired_bootstrap_95_ci"][0] > 0.0
        for report in linear_proxy_reports
        for site in report["sites"]
    )
    if linear_proxy_admissions or pass_wins_vs_monolithic:
        raise ValueError("empirical linear capacity screening outcome differs")
    rank_sweep_checks = sum(
        int(row["summary"]["rank_site_checks"]) for row in rank_sweep_reports
    )
    rank_sweep_pass_wins_vs_monolithic = sum(
        int(row["summary"]["pass_wins_vs_monolithic"])
        for row in rank_sweep_reports
    )
    rank_sweep_monolithic_wins_vs_pass = sum(
        int(row["summary"]["monolithic_wins_vs_pass"])
        for row in rank_sweep_reports
    )
    rank_sweep_inconclusive = sum(
        int(row["summary"]["monolithic_vs_pass_inconclusive"])
        for row in rank_sweep_reports
    )
    rank_sweep_pass_wins_vs_random = sum(
        int(row["summary"]["pass_wins_vs_random_mean"])
        for row in rank_sweep_reports
    )
    rank_sweep_admissions = sum(
        int(row["summary"]["pass_compartment_screening_admissions"])
        for row in rank_sweep_reports
    )
    if rank_sweep_pass_wins_vs_monolithic or rank_sweep_admissions:
        raise ValueError("empirical capacity rank-sweep outcome differs")
    shared_sae_checks = sum(
        int(row["summary"]["sites"]) for row in sae_selectivity_reports
    )
    shared_sae_admissions = sum(
        int(row["summary"]["internal_pass_specialization_admissions"])
        for row in sae_selectivity_reports
    )
    shared_sae_decisions = [
        {
            site["site"]: site["monolithic_sae_internal_pass_specialization_admission"]
            for site in report["sites"]
        }
        for report in sae_selectivity_reports
    ]
    expected_shared_sae_decisions = {
        "model.layers.11": "pass",
        "model.layers.21": "fail",
        "model.norm": "pass",
    }
    if shared_sae_admissions != 4 or any(
        decision != expected_shared_sae_decisions for decision in shared_sae_decisions
    ):
        raise ValueError("shared-SAE internal compartment outcome differs")
    robustness_sites = [
        {site["site"]: site for site in report["sites"]}
        for report in sae_robustness_reports
    ]
    if any(
        report["summary"]["sites_passing_all_selectivity_configurations"] != 2
        or report["summary"]["selectivity_configurations_per_site"] != 12
        for report in sae_robustness_reports
    ):
        raise ValueError("shared-SAE robustness summary differs")
    expected_robustness_counts = {
        "model.layers.11": [11, 11],
        "model.layers.21": [12, 12],
        "model.norm": [12, 12],
    }
    observed_robustness_counts = {
        site: [
            int(archive[site]["selectivity_configurations_passing"])
            for archive in robustness_sites
        ]
        for site in expected_robustness_counts
    }
    if observed_robustness_counts != expected_robustness_counts:
        raise ValueError("shared-SAE site robustness outcome differs")
    cross_pack_decisions = {
        row["site"]: row["bidirectional_cross_pack_category_admission"]
        for row in cross_pack_transfer["sites"]
    }
    if cross_pack_transfer["summary"][
        "bidirectional_cross_pack_category_admissions"
    ] != 2 or cross_pack_decisions != {
        "model.layers.11": "fail",
        "model.layers.21": "pass",
        "model.norm": "pass",
    }:
        raise ValueError("cross-pack pass-category transfer outcome differs")
    if shared_axis["summary"]["shared_axis_admissions"] != 3 or any(
        row["shared_axis_admission"] != "pass" for row in shared_axis["sites"]
    ):
        raise ValueError("shared pass-category axis outcome differs")
    if prompt_phase_transfer["summary"]["prompt_phase_transfer_admissions"] != 1:
        raise ValueError("prompt-phase pass-category transfer outcome differs")
    if (
        phase_category["summary"]["activation_phase_category_admissions"] != 0
        or phase_category["summary"]["operative_nontrivial_within_phase_allocations"]
        != 0
    ):
        raise ValueError("activation-phase category screen outcome differs")
    if (
        phase_cross_validation["summary"][
            "provisional_cross_validated_phase_category_supports"
        ]
        != 3
        or phase_cross_validation["summary"]["operative_phase_category_admissions"]
        != 0
    ):
        raise ValueError("activation-phase cross-validation outcome differs")
    if (
        phase_linear_proxy["summary"]["phase_routing_wins_vs_monolithic"] != 0
        or phase_linear_proxy["summary"]["screening_capacity_preferences"] != 0
        or phase_linear_proxy["summary"]["phase_compartment_training_admissions"]
        != 0
    ):
        raise ValueError("activation-phase capacity outcome differs")
    if (
        joint_pass_phase_capacity["summary"]["joint_wins_vs_monolithic"] != 0
        or joint_pass_phase_capacity["summary"]["joint_wins_vs_pass_routed"] != 0
        or joint_pass_phase_capacity["summary"]["joint_wins_vs_phase_routed"] != 0
        or joint_pass_phase_capacity["summary"]["joint_capacity_preferences"] != 0
        or joint_pass_phase_capacity["summary"][
            "joint_compartment_training_admissions"
        ]
        != 0
    ):
        raise ValueError("joint pass/phase capacity outcome differs")
    if (
        prompt_frozen_sae_quality["summary"][
            "feature_interpretation_quality_admissions"
        ]
        != 0
        or prompt_frozen_sae_quality["summary"][
            "prompt_phase_pass_feature_strata_admissions"
        ]
        != 0
    ):
        raise ValueError("prompt frozen-SAE quality outcome differs")
    if (
        q4_sae_pass_feature_identity["summary"][
            "provisional_same_seed_feature_identity_supports"
        ]
        != 1
        or q4_sae_pass_feature_identity["summary"][
            "operative_cross_budget_feature_identity_admissions"
        ]
        != 0
    ):
        raise ValueError("Q4 SAE pass-feature identity outcome differs")
    if (
        q4_sae_pass_feature_stratum_v2["summary"][
            "provisional_same_seed_v2_stratum_replication_supports"
        ]
        != 1
        or q4_sae_pass_feature_stratum_v2["summary"][
            "operative_fresh_seed_stratum_replication_admissions"
        ]
        != 0
    ):
        raise ValueError("Q4 SAE V2 pass-feature stratum outcome differs")
    if (
        layer21_seed_objective_stability["summary"][
            "cross_seed_diagnostic_supports_without_quality_gate"
        ]
        != 0
        or layer21_seed_objective_stability["summary"][
            "same_seed_cross_objective_diagnostic_supports_without_quality_gate"
        ]
        != 3
        or layer21_seed_objective_stability["summary"][
            "operative_stratum_replication_admissions"
        ]
        != 0
    ):
        raise ValueError("layer21 seed/objective stability outcome differs")
    if (
        layer21_category_subspace_stability["summary"][
            "cross_seed_diagnostic_subspace_supports_without_quality_gate"
        ]
        != 0
        or layer21_category_subspace_stability["summary"][
            "same_seed_cross_objective_diagnostic_subspace_supports_without_quality_gate"
        ]
        != 3
        or layer21_category_subspace_stability["summary"][
            "operative_category_subspace_replication_admissions"
        ]
        != 0
    ):
        raise ValueError("layer21 category-subspace stability outcome differs")
    if (
        q4_sae_pass_category_subspace_transfer["summary"][
            "diagnostic_same_seed_category_subspace_transfer_supports"
        ]
        != 3
        or q4_sae_pass_category_subspace_transfer["summary"][
            "provisional_same_seed_quality_qualified_subspace_supports"
        ]
        != 2
        or q4_sae_pass_category_subspace_transfer["summary"][
            "operative_fresh_seed_category_subspace_replication_admissions"
        ]
        != 0
    ):
        raise ValueError("Q4 category-subspace transfer outcome differs")
    return {
        "pass_replication_archives": len(pass_reports),
        "pass_site_admissions": pass_admissions,
        "pass_site_combinations": pass_sites,
        "within_pass_nontrivial_allocations": sum(
            any(int(value) > 1 for value in allocation.values())
            for report in pass_reports
            for allocation in report["summary"]["within_pass_allocations"].values()
        ),
        "q4_region_negative_control_archives": len(region_reports),
        "q4_region_site_pass_admissions": region_admissions,
        "q4_region_site_pass_combinations": region_combinations,
        "screening_only_region_child_allocations": screening_children,
        "admissible_region_child_allocations": admissible_children,
        "linear_proxy_archives": len(linear_proxy_reports),
        "linear_proxy_site_checks": linear_proxy_site_checks,
        "linear_proxy_pass_wins_vs_random": pass_wins_vs_random,
        "linear_proxy_pass_wins_vs_monolithic": pass_wins_vs_monolithic,
        "linear_proxy_training_admissions": linear_proxy_admissions,
        "rank_sweep_archives": len(rank_sweep_reports),
        "rank_sweep_site_rank_checks": rank_sweep_checks,
        "rank_sweep_pass_wins_vs_monolithic": rank_sweep_pass_wins_vs_monolithic,
        "rank_sweep_monolithic_wins_vs_pass": rank_sweep_monolithic_wins_vs_pass,
        "rank_sweep_monolithic_vs_pass_inconclusive": rank_sweep_inconclusive,
        "rank_sweep_pass_wins_vs_random": rank_sweep_pass_wins_vs_random,
        "rank_sweep_training_admissions": rank_sweep_admissions,
        "shared_sae_selectivity_archives": len(sae_selectivity_reports),
        "shared_sae_site_archive_checks": shared_sae_checks,
        "shared_sae_internal_feature_strata_admissions": shared_sae_admissions,
        "shared_sae_admitted_sites_in_both_archives": [
            "model.layers.11",
            "model.norm",
        ],
        "shared_sae_rejected_sites_in_both_archives": ["model.layers.21"],
        "shared_sae_robustness_archives": len(sae_robustness_reports),
        "shared_sae_selectivity_configurations_per_site": 12,
        "shared_sae_selectivity_passing_by_site": observed_robustness_counts,
        "shared_sae_high_confidence_sites_through_ev_0_25": ["model.layers.11"],
        "shared_sae_provisional_sites_at_preregistered_ev_0_20": ["model.norm"],
        "shared_sae_quality_rejected_sites": ["model.layers.21"],
        "cross_pack_transfer_archive_pairs": 1,
        "cross_pack_transfer_site_checks": 3,
        "cross_pack_transfer_bidirectional_admissions": 2,
        "cross_pack_transfer_all_directional_held_out_accuracies": 1.0,
        "cross_pack_transfer_direction_cosines": {
            row["site"]: row["cross_pack_train_direction_cosine"]
            for row in cross_pack_transfer["sites"]
        },
        "cross_pack_transfer_admitted_sites": ["model.layers.21", "model.norm"],
        "sites_meeting_high_confidence_sae_quality_and_direct_pack_direction_replication_gates": [],
        "shared_axis_site_checks": 3,
        "shared_axis_admissions": 3,
        "shared_axis_minimum_held_out_pass_accuracy": 1.0,
        "shared_axis_minimum_cosine_to_pack_train_direction": (
            0.8026508133661998
        ),
        "shared_axis_maximum_exact_upper_tail_p_value": 0.01312255859375,
        "sites_meeting_high_confidence_sae_quality_and_shared_axis_gates": [
            "model.layers.11"
        ],
        "prompt_phase_transfer_site_checks": 3,
        "prompt_phase_transfer_admissions": 1,
        "prompt_phase_transfer_admitted_sites": ["model.norm"],
        "activation_phase_category_site_checks": 3,
        "activation_phase_category_screening_accuracy_hits": 3,
        "activation_phase_category_admissions": 0,
        "activation_phase_exact_p_value_floor": 0.0625,
        "activation_phase_minimum_training_blocks_for_p_0_05": 5,
        "activation_phase_operative_child_clusters": 0,
        "activation_phase_cross_validated_site_checks": 3,
        "activation_phase_provisional_cross_validated_supports": 3,
        "activation_phase_cross_validated_minimum_accuracy": 0.9375,
        "activation_phase_cross_validated_maximum_exact_p_value": 0.0078125,
        "activation_phase_cross_validated_operative_admissions": 0,
        "activation_phase_capacity_site_checks": 3,
        "activation_phase_capacity_wins_vs_monolithic": 0,
        "activation_phase_capacity_wins_vs_random_mean": 3,
        "activation_phase_capacity_preferences": 0,
        "activation_phase_capacity_training_admissions": 0,
        "joint_pass_phase_capacity_site_checks": 3,
        "joint_pass_phase_wins_vs_monolithic": 0,
        "joint_pass_phase_wins_vs_pass_routed": 0,
        "joint_pass_phase_wins_vs_phase_routed": 0,
        "joint_pass_phase_wins_vs_random_mean": 3,
        "joint_pass_phase_capacity_preferences": 0,
        "joint_pass_phase_training_admissions": 0,
        "prompt_frozen_sae_quality_site_checks": 3,
        "prompt_frozen_sae_quality_admissions": 0,
        "prompt_phase_pass_feature_strata_admissions": 0,
        "generation_sae_feature_identity_site_checks": 3,
        "generation_sae_provisional_same_seed_feature_identity_supports": 1,
        "generation_sae_provisional_same_seed_feature_identity_sites": [
            "model.norm"
        ],
        "generation_sae_operative_cross_budget_feature_identity_admissions": 0,
        "generation_sae_v2_stratum_replication_site_checks": 3,
        "generation_sae_v2_provisional_same_seed_supports": 1,
        "generation_sae_v2_provisional_same_seed_sites": ["model.norm"],
        "generation_sae_v2_operative_fresh_seed_admissions": 0,
        "generation_sae_v2_model_norm_primary_matches": {
            "pass_0": 25,
            "pass_1": 28,
        },
        "generation_sae_v2_model_norm_bidirectional_coverage": {
            "pass_0": 25 / 78,
            "pass_1": 28 / 236,
        },
        "layer21_seed_objective_conditions": 6,
        "layer21_seed_objective_quality_qualified_conditions": 0,
        "layer21_cross_seed_pair_checks": 6,
        "layer21_cross_seed_diagnostic_supports_without_quality_gate": 0,
        "layer21_same_seed_cross_objective_pair_checks": 3,
        "layer21_same_seed_cross_objective_diagnostic_supports_without_quality_gate": 3,
        "layer21_seed_objective_operative_admissions": 0,
        "layer21_category_subspace_pair_checks": 9,
        "layer21_cross_seed_subspace_pair_checks": 6,
        "layer21_cross_seed_familywise_corrected_subspace_supports": 0,
        "layer21_same_seed_cross_objective_subspace_pair_checks": 3,
        "layer21_same_seed_cross_objective_subspace_supports": 3,
        "layer21_category_subspace_operative_admissions": 0,
        "generation_sae_category_subspace_site_checks": 3,
        "generation_sae_category_subspace_same_seed_diagnostic_supports": 3,
        "generation_sae_category_subspace_same_seed_quality_supports": 2,
        "generation_sae_category_subspace_same_seed_quality_sites": [
            "model.layers.11",
            "model.norm",
        ],
        "generation_sae_category_subspace_operative_fresh_seed_admissions": 0,
        "feature_replica_matching_version": "v2",
        "feature_replica_matching_synthetic_scenarios": 4,
        "feature_replica_matching_expected_outcomes": 4,
        "feature_replica_matching_signed_cosine": True,
        "feature_replica_matching_mutual_one_to_one": True,
    }


def build_protocol(
    *,
    predecessor_path: Path,
    sae_spec_builder_path: Path,
    sae_spec_path: Path,
    sae_spec_test_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path, project_root)
    spec_verification = verify_spec(sae_spec_path)
    spec = _load(sae_spec_path)
    summary = _evidence_summary(spec, project_root)
    predecessor = _load(predecessor_path)
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["status"] = (
        "recovered_empirical_pass_category_admitted_temporal_region_rejected_"
        "equal_rank_capacity_screen_negative_equal_budget_sae_design_verified_"
        "shared_sae_internal_feature_strata_replication_verified_fresh_capture_"
        "and_training_absent"
    )
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"][
        "nanbeige_empirical_activation_category_sae_experiment_v1"
    ] = {
        "schema": "sch.vpd-empirical-activation-category-sae-lineage.v1",
        "status": (
            "recovered_empirical_pass_category_admitted_temporal_region_rejected_"
            "equal_rank_capacity_screen_negative_equal_budget_sae_design_verified_"
            "shared_sae_internal_feature_strata_replication_verified_capture_and_"
            "training_not_authorized"
        ),
        "builder": {
            "path": _relative(sae_spec_builder_path, project_root),
            "sha256": sha256_file(sae_spec_builder_path),
        },
        "specification": {
            "path": _relative(sae_spec_path, project_root),
            "sha256": sha256_file(sae_spec_path),
            "spec_id": spec["spec_id"],
            "verification": spec_verification,
        },
        "test": {
            "path": _relative(sae_spec_test_path, project_root),
            "sha256": sha256_file(sae_spec_test_path),
        },
        "empirical_summary": summary,
        "admitted_category_domain": spec["evidence"]["admitted_category_domain"],
        "admitted_categories": spec["evidence"]["admitted_categories"],
        "operative_cluster_count_per_category": spec["evidence"][
            "operative_cluster_count_per_category"
        ],
        "sae_comparison_arms": len(spec["sae_comparison"]["arms"]),
        "pretraining_capacity_screen_action": spec["evidence"][
            "capacity_screening"
        ]["current_action"],
        "current_preferred_sae_architecture": spec["evidence"][
            "shared_sae_internal_compartmentalization"
        ]["current_preferred_architecture"],
        "shared_sae_robustness_classification": spec["evidence"][
            "shared_sae_internal_compartmentalization"
        ]["robustness_sweep"],
        "cross_pack_category_transfer": spec["evidence"][
            "cross_pack_category_transfer"
        ],
        "shared_axis_category_geometry": spec["evidence"][
            "shared_axis_category_geometry"
        ],
        "prompt_phase_transfer": spec["evidence"]["prompt_phase_transfer"],
        "activation_phase_category_screen": spec["evidence"][
            "activation_phase_category_screen"
        ],
        "activation_phase_category_cross_validation": spec["evidence"][
            "activation_phase_category_cross_validation"
        ],
        "activation_phase_capacity_screening": spec["evidence"][
            "activation_phase_capacity_screening"
        ],
        "joint_pass_phase_capacity_screening": spec["evidence"][
            "joint_pass_phase_capacity_screening"
        ],
        "prompt_phase_frozen_sae_quality": spec["evidence"][
            "prompt_phase_frozen_sae_quality"
        ],
        "generation_sae_cross_budget_feature_identity": spec["evidence"][
            "generation_sae_cross_budget_feature_identity"
        ],
        "feature_replica_matching_method": spec["evidence"][
            "feature_replica_matching_method"
        ],
        "generation_sae_v2_stratum_replication": spec["evidence"][
            "generation_sae_v2_stratum_replication"
        ],
        "layer21_seed_objective_stability_control": spec["evidence"][
            "layer21_seed_objective_stability_control"
        ],
        "layer21_category_subspace_stability_control": spec["evidence"][
            "layer21_category_subspace_stability_control"
        ],
        "generation_sae_cross_budget_category_subspace_transfer": spec["evidence"][
            "generation_sae_cross_budget_category_subspace_transfer"
        ],
        "matched_trainable_parameters_per_arm": sorted(
            {
                int(row["total_trainable_parameters"])
                for row in spec["sae_comparison"]["arms"]
            }
        ),
        "fresh_pack_families": int(spec["fresh_pack"]["total_families"]),
        "fresh_pack_examples": int(spec["fresh_pack"]["total_examples"]),
        "fresh_pack_present": False,
        "model_inference_performed": False,
        "sae_training_performed": False,
        "execution_authorized": False,
    }
    lineage = result["unified_lineage"]
    lineage["nanbeige_empirical_activation_category_sae_experiment_v1_verified"] = True
    lineage["empirical_pass_category_archives"] = summary[
        "pass_replication_archives"
    ]
    lineage["empirical_temporal_region_negative_control_archives"] = summary[
        "q4_region_negative_control_archives"
    ]
    lineage["empirical_pass_category_site_admissions"] = summary[
        "pass_site_admissions"
    ]
    lineage["empirical_temporal_region_site_admissions"] = summary[
        "q4_region_site_pass_admissions"
    ]
    lineage["empirical_linear_proxy_training_admissions"] = summary[
        "linear_proxy_training_admissions"
    ]
    lineage["empirical_rank_sweep_training_admissions"] = summary[
        "rank_sweep_training_admissions"
    ]
    lineage["shared_sae_internal_feature_strata_admissions"] = summary[
        "shared_sae_internal_feature_strata_admissions"
    ]
    lineage["empirical_activation_category_fresh_pack_present"] = False
    lineage["model_inference_performed"] = False
    lineage["training_performed"] = False
    lineage["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v51_verification": predecessor_verification,
        "empirical_activation_category_sae_spec_verification": spec_verification,
        "empirical_activation_category_sae_derivation_sha256": sha256_file(
            Path(__file__)
        ),
        "deep_predecessor_replay_performed": True,
    }
    boundary = result["scientific_boundary"]
    boundary["nanbeige_empirical_activation_category_sae_experiment_v1_verified"] = True
    boundary["empirical_pass_index_admitted_as_coarse_category"] = True
    boundary["empirical_within_pass_subclusters_admitted"] = False
    boundary["empirical_temporal_region_categories_admitted"] = False
    boundary["empirical_region_child_clusters_admitted"] = False
    boundary["empirical_pass_capacity_screen_admitted"] = False
    boundary["shared_sae_internal_pass_feature_strata_admitted"] = True
    boundary["shared_sae_quality_qualified_sites"] = [
        "model.layers.11",
        "model.norm",
    ]
    boundary["shared_sae_layer21_feature_strata_admitted"] = False
    boundary["shared_sae_high_confidence_site_through_ev_0_25"] = (
        "model.layers.11"
    )
    boundary["shared_sae_provisional_site_at_preregistered_ev_0_20"] = (
        "model.norm"
    )
    boundary["shared_sae_robustness_is_posthoc_and_not_training_authority"] = True
    boundary["codex_goal_pause_is_compute_termination"] = False
    boundary["pass_category_cross_pack_transfer_admitted_sites"] = [
        "model.layers.21",
        "model.norm",
    ]
    boundary["layer11_pass_direction_is_task_dependent"] = True
    boundary[
        "sites_meeting_high_confidence_sae_quality_and_direct_pack_direction_replication_gates"
    ] = []
    boundary["cross_pack_transfer_checkpoint_revision_independently_verified"] = False
    boundary["shared_pass_axis_admitted_at_all_sites"] = True
    boundary["sites_meeting_high_confidence_sae_quality_and_shared_axis_gates"] = [
        "model.layers.11"
    ]
    boundary["shared_axis_third_domain_transfer_established"] = False
    boundary["generation_shared_axis_prompt_phase_transfer_admitted_sites"] = [
        "model.norm"
    ]
    boundary["layer11_generation_shared_axis_is_phase_specific"] = True
    boundary["activation_phase_category_admitted"] = False
    boundary["activation_phase_category_current_exact_test_underpowered"] = True
    boundary["activation_phase_compartment_sae_training_authorized"] = False
    boundary["activation_phase_family_blocked_provisional_support_at_all_sites"] = True
    boundary["activation_phase_fresh_untouched_holdout_present"] = False
    boundary["activation_phase_fixed_capacity_screen_admitted"] = False
    boundary["shared_monolithic_capacity_preferred_over_phase_split"] = True
    boundary["joint_pass_phase_fixed_capacity_screen_admitted"] = False
    boundary["joint_pass_phase_dictionaries_authorized"] = False
    boundary["shared_factor_labeled_features_preferred_over_cartesian_dictionary_split"] = True
    boundary["prompt_frozen_feature_archives_exist"] = True
    boundary["prompt_phase_feature_interpretation_quality_admitted"] = False
    boundary["cross_phase_pass_feature_strata_admitted"] = False
    boundary["generation_sae_same_seed_feature_identity_provisional_site"] = "model.norm"
    boundary["generation_sae_cross_budget_feature_identity_operatively_admitted"] = False
    boundary["fresh_seed_family_disjoint_feature_identity_replication_required"] = True
    boundary["fresh_seed_feature_identity_matcher_version"] = "v2"
    boundary["fresh_seed_feature_identity_requires_signed_cosine"] = True
    boundary[
        "fresh_seed_feature_identity_requires_mutual_one_to_one_matching"
    ] = True
    boundary["v1_feature_replica_matcher_behavior_changed"] = False
    boundary["generation_sae_v2_stratum_replication_provisional_site"] = (
        "model.norm"
    )
    boundary["generation_sae_v2_stratum_replication_operatively_admitted"] = False
    boundary["generation_sae_v2_full_dictionary_replication_claimed"] = False
    boundary["layer21_cross_seed_feature_stability_admitted"] = False
    boundary["same_seed_cross_objective_stability_is_independent_replication"] = False
    boundary["feature_identity_replication_must_change_initialization_seed"] = True
    boundary["layer21_seed_controls_can_override_reconstruction_failure"] = False
    boundary["layer21_cross_seed_category_subspace_stability_admitted"] = False
    boundary["individual_feature_instability_proves_subspace_instability"] = False
    boundary["category_subspace_replication_requires_multiplicity_control"] = True
    boundary["same_seed_category_subspace_transfer_is_independent_replication"] = False
    boundary[
        "generation_sae_cross_budget_category_subspace_operatively_admitted"
    ] = False
    boundary["feature_identity_and_category_subspace_transfer_are_distinct"] = True
    boundary["fresh_nanbeige_pack_present"] = False
    boundary["new_model_inference_performed"] = False
    boundary["sae_training_performed"] = False
    boundary["weights_loaded_or_mutated"] = False
    boundary["sae_dictionary_training_authorized"] = False
    boundary["execution_authorized"] = False
    boundary["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    expected = build_protocol(**kwargs)
    if observed != expected:
        raise ValueError("research protocol V52 is not reproducible")
    component = observed["component_compartments"][
        "nanbeige_empirical_activation_category_sae_experiment_v1"
    ]
    summary = component["empirical_summary"]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v52",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "spec_id": component["specification"]["spec_id"],
        **summary,
        "fresh_pack_present": False,
        "training_performed": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v51.json",
        "sae_spec_builder_path": project_root
        / "experiments/authz_mechinterp/build_activation_category_sae_experiment_spec_v1.py",
        "sae_spec_path": project_root
        / "artifacts/vpd_activation_category_sae_experiment_spec_v1/spec.json",
        "sae_spec_test_path": project_root
        / "tests/test_vpd_activation_category_sae_experiment_spec_v1.py",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V52.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.output is not None:
        result = build_protocol(**inputs)
        write_json(args.output, result)
        protocol_sha256 = sha256_file(args.output)
    else:
        result = verify_protocol(args.verify, **inputs)
        protocol_sha256 = result["protocol_sha256"]
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result.get("status", "derived"),
                "protocol_sha256": protocol_sha256,
                "execution_authorized": result.get("execution_authorized", False),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
