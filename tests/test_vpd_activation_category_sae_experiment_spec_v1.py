from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.build_activation_category_sae_experiment_spec_v1 import (
    verify_spec,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "artifacts/vpd_activation_category_sae_experiment_spec_v1/spec.json"


def test_activation_category_sae_spec_reproduces() -> None:
    verification = verify_spec(SPEC)
    assert verification["status"] == "pass"
    assert verification["execution_authorized"] is False


def test_spec_uses_only_admitted_category_and_exact_matched_budgets() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    region_roles = {
        row["role"]
        for row in spec["evidence"]["reports"]
        if row["role"].startswith("q4_temporal_region_negative_control_")
    }
    assert region_roles == {
        "q4_temporal_region_negative_control_2048_tokens",
        "q4_temporal_region_negative_control_8192_tokens",
    }
    proxy_roles = {
        row["role"]
        for row in spec["evidence"]["reports"]
        if row["role"].endswith("pass_compartment_linear_proxy_negative")
    }
    assert proxy_roles == {
        "arithmetic_pass_compartment_linear_proxy_negative",
        "reasoning_pass_compartment_linear_proxy_negative",
    }
    rank_sweep_roles = {
        row["role"]
        for row in spec["evidence"]["reports"]
        if row["role"].endswith("pass_compartment_rank_sweep_negative")
    }
    assert rank_sweep_roles == {
        "arithmetic_pass_compartment_rank_sweep_negative",
        "reasoning_pass_compartment_rank_sweep_negative",
    }
    selectivity_roles = {
        row["role"]
        for row in spec["evidence"]["reports"]
        if row["role"].endswith("shared_sae_internal_pass_specialization")
    }
    assert selectivity_roles == {
        "q4_2048_shared_sae_internal_pass_specialization",
        "q4_8192_shared_sae_internal_pass_specialization",
    }
    robustness_roles = {
        row["role"]
        for row in spec["evidence"]["reports"]
        if row["role"].endswith("shared_sae_pass_selectivity_robustness")
    }
    assert robustness_roles == {
        "q4_2048_shared_sae_pass_selectivity_robustness",
        "q4_8192_shared_sae_pass_selectivity_robustness",
    }
    assert {
        row["role"]
        for row in spec["evidence"]["reports"]
        if row["role"] == "bf16_pass_category_cross_pack_transfer"
    } == {"bf16_pass_category_cross_pack_transfer"}
    assert {
        row["role"]
        for row in spec["evidence"]["reports"]
        if row["role"] == "bf16_pass_category_shared_axis"
    } == {"bf16_pass_category_shared_axis"}
    assert {
        row["role"]
        for row in spec["evidence"]["reports"]
        if row["role"] in {
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
    } == {
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
    assert spec["evidence"]["shared_sae_internal_compartmentalization"] == {
        "method": "training_only_feature_assignment_with_held_out_pass_validation",
        "archives": 2,
        "site_archive_checks": 6,
        "admissions": 4,
        "admitted_sites_in_both_archives": ["model.layers.11", "model.norm"],
        "rejected_sites_in_both_archives": ["model.layers.21"],
        "layer21_rejection_reason": (
            "held_out_second_pass_reconstruction_below_minimum_despite_selectivity"
        ),
        "current_preferred_architecture": (
            "shared_monolithic_sae_with_posthoc_pass_selective_feature_groups_"
            "at_quality_qualified_sites"
        ),
        "separate_pass_dictionary_status": (
            "experimental_fallback_behind_fresh_pack_capacity_gate"
        ),
        "robustness_sweep": {
            "train_share_thresholds": [0.8, 0.85, 0.9, 0.95],
            "minimum_train_example_support_thresholds": [2, 4, 6],
            "configurations_per_site": 12,
            "layer11_selectivity_configurations_passing_per_archive": [11, 11],
            "layer11_minimum_held_out_explained_variance_per_archive": [
                0.2511399276355816,
                0.2629356381461697,
            ],
            "norm_selectivity_configurations_passing_per_archive": [12, 12],
            "norm_minimum_held_out_explained_variance_per_archive": [
                0.2496276574486168,
                0.2922872645304503,
            ],
            "layer21_selectivity_configurations_passing_per_archive": [12, 12],
            "layer21_minimum_held_out_explained_variance_per_archive": [
                0.011668844551661728,
                -0.1879651772380071,
            ],
            "high_confidence_site_through_ev_0_25": "model.layers.11",
            "provisional_site_at_preregistered_ev_0_20": "model.norm",
            "quality_rejected_site": "model.layers.21",
            "posthoc_classification_is_not_training_authority": True,
        },
    }
    assert spec["evidence"]["capacity_screening"] == {
        "method": "equal_total_rank_held_out_linear_subspace_proxy",
        "current_archives_pass_compartment_admissions": 0,
        "current_archives_site_checks": 6,
        "rank_sweep_total_ranks": [8, 16, 32, 64, 96, 128],
        "rank_sweep_archive_site_rank_checks": 36,
        "rank_sweep_pass_wins_vs_monolithic": 0,
        "rank_sweep_monolithic_wins_vs_pass": 35,
        "rank_sweep_monolithic_vs_pass_inconclusive": 1,
        "rank_sweep_pass_wins_vs_random_mean": 35,
        "rank_sweep_training_admissions": 0,
        "current_action": (
            "do_not_prioritize_pass_compartment_sae_training_on_recovered_archives"
        ),
        "fresh_pack_requirement": (
            "repeat_before_training_and_stop_if_pass_routing_does_not_beat_"
            "monolithic_and_random_controls"
        ),
    }
    assert spec["evidence"]["cross_pack_category_transfer"] == {
        "method": "train_direction_on_one_pack_score_other_pack_held_out",
        "archive_pairs": 1,
        "site_checks": 3,
        "bidirectional_admissions": 2,
        "all_directional_held_out_accuracies": 1.0,
        "cross_pack_train_direction_cosines": {
            "model.layers.11": 0.28849665639484445,
            "model.layers.21": 0.9921854242070439,
            "model.norm": 0.8979968898827502,
        },
        "admitted_sites": ["model.layers.21", "model.norm"],
        "task_dependent_direction_site": "model.layers.11",
        "site_meeting_high_confidence_reconstruction_and_direct_pack_direction_replication_gates": [],
        "interpretation": (
            "pass labels transfer as classifiers at every site, but no site yet "
            "combines cross-task direction stability with non-provisional SAE "
            "reconstruction quality"
        ),
        "checkpoint_revision_limitation": (
            "both sources claim the same revision but neither independently verified it"
        ),
    }
    assert spec["evidence"]["shared_axis_category_geometry"] == {
        "method": "equal_pack_bisector_of_training_mean_pass_transitions",
        "site_checks": 3,
        "admissions": 3,
        "minimum_held_out_pass_accuracy": 1.0,
        "minimum_cosine_to_each_pack_train_direction": 0.8026508133661998,
        "maximum_exact_joint_pair_flip_p_value": 0.01312255859375,
        "layer11_exact_joint_pair_flip_p_value": 0.00048828125,
        "sites_meeting_high_confidence_sae_quality_and_shared_axis_gates": [
            "model.layers.11"
        ],
        "primary_future_capture_site": "model.layers.11",
        "interpretation": (
            "layer11 contains a held-out common pass component plus large "
            "task-specific direction components"
        ),
        "third_domain_transfer_established": False,
        "does_not_authorize_training": True,
    }
    assert spec["evidence"]["prompt_phase_transfer"] == {
        "method": "generation_trained_shared_axis_to_prompt_only_held_out",
        "site_checks": 3,
        "admissions": 1,
        "admitted_site": "model.norm",
        "layer11_prompt_phase_cosine": 0.04767526301390991,
        "layer21_prompt_phase_cosine": 0.7966895787427319,
        "norm_prompt_phase_cosine": 0.9785101628277528,
        "generation_layer11_shared_axis_is_phase_specific": True,
        "prompt_phase_is_third_semantic_domain": False,
        "does_not_authorize_training": True,
    }
    assert spec["evidence"]["activation_phase_category_screen"] == {
        "observable_category": "prompt_vs_generation",
        "site_checks": 3,
        "screening_accuracy_threshold_hits": 3,
        "admissions": 0,
        "current_train_example_blocks": 4,
        "exact_control_configurations": 16,
        "minimum_attainable_exact_p_value": 0.0625,
        "maximum_admissible_exact_p_value": 0.05,
        "minimum_train_example_blocks_for_p_0_05_eligibility": 5,
        "operative_child_clusters": 0,
        "status": "strong_screening_signal_but_statistically_ineligible",
        "phase_compartment_sae_training_authorized": False,
    }
    assert spec["evidence"]["activation_phase_category_cross_validation"] == {
        "method": "leave_one_semantic_family_out",
        "semantic_families": 4,
        "examples": 8,
        "exact_example_block_permutations": 256,
        "site_checks": 3,
        "provisional_supports": 3,
        "minimum_phase_accuracy": 0.9375,
        "maximum_exact_p_value": 0.0078125,
        "operative_admissions": 0,
        "reason_inoperative": "no_fully_untouched_final_holdout",
        "fresh_untouched_holdout_required": True,
        "phase_compartment_sae_training_authorized": False,
    }
    assert spec["evidence"]["activation_phase_capacity_screening"] == {
        "method": "equal_total_rank_held_out_linear_subspace_proxy",
        "site_checks": 3,
        "phase_routing_wins_vs_monolithic": 0,
        "phase_routing_wins_vs_random_mean": 3,
        "screening_capacity_preferences": 0,
        "training_admissions": 0,
        "current_action": "retain_monolithic_capacity_across_activation_phase",
        "interpretation": (
            "phase is structured relative to random partitions but does not "
            "justify splitting fixed dictionary capacity"
        ),
        "cannot_bypass_inoperative_phase_parent": True,
    }
    assert spec["evidence"]["joint_pass_phase_capacity_screening"] == {
        "method": "equal_total_rank_held_out_linear_subspace_proxy",
        "site_checks": 3,
        "joint_routing_wins_vs_monolithic": 0,
        "joint_routing_wins_vs_pass_routed": 0,
        "joint_routing_wins_vs_phase_routed": 0,
        "joint_routing_wins_vs_random_mean": 3,
        "screening_capacity_preferences": 0,
        "training_admissions": 0,
        "current_action": "retain_shared_monolithic_capacity_with_factor_labeled_features",
        "interpretation": (
            "joint pass-by-phase structure exceeds random routing but four-way "
            "capacity fragmentation loses to every simpler structured arm"
        ),
        "four_way_random_control_is_minimal": True,
        "cannot_bypass_inoperative_phase_parent": True,
    }
    assert spec["evidence"]["prompt_phase_frozen_sae_quality"] == {
        "method": "held_out_reconstruction_gate_on_hash_bound_frozen_feature_archives",
        "site_checks": 3,
        "minimum_explained_variance_threshold": 0.2,
        "quality_admissions": 0,
        "pass_feature_strata_admissions": 0,
        "quality_rejected_sites": [
            "model.layers.11",
            "model.layers.21",
            "model.norm",
        ],
        "minimum_held_out_explained_variance_by_site": {
            "model.layers.11": -0.08210552224626455,
            "model.layers.21": -1.2934838949230807,
            "model.norm": 0.07984828542967848,
        },
        "current_action": "reject_prompt_and_cross_phase_feature_interpretation",
        "archive_existence_is_not_feature_interpretation_validity": True,
        "new_dictionary_fitted": False,
    }
    assert spec["evidence"]["generation_sae_cross_budget_feature_identity"] == {
        "method": "same_index_decoder_cosine_and_training_only_pass_assignment",
        "site_checks": 3,
        "quality_qualified_sites": ["model.layers.11", "model.norm"],
        "primary_diagnostic_decoder_cosine": 0.8,
        "minimum_stable_features_per_pass": 16,
        "stable_same_pass_features_at_primary_threshold": {
            "model.layers.11": {"pass_0": 19, "pass_1": 10},
            "model.layers.21": {"pass_0": 43, "pass_1": 19},
            "model.norm": {"pass_0": 22, "pass_1": 28},
        },
        "provisional_same_seed_supports": 1,
        "provisional_same_seed_sites": ["model.norm"],
        "operative_admissions": 0,
        "reason_inoperative": "same_seed_same_training_ids_and_posthoc_cosine_threshold",
        "fresh_seed_family_disjoint_replication_required": True,
        "decoder_alignment_is_not_feature_semantics": True,
    }
    assert spec["evidence"]["feature_replica_matching_method"] == {
        "version": "v2",
        "path": "artifacts/vpd_sae_feature_replica_matching_benchmark_v2/report.json",
        "sha256": "cb94778b3869072fc820858050ffaadd7bed0aca3a39495ff9e69fcae5d60b7e",
        "benchmark_id": "832e6481040319877140d1b1a7f33f1b7a933f6345a96b27af469704c38e47f5",
        "synthetic_scenarios": 4,
        "expected_outcomes_reproduced": 4,
        "signed_decoder_cosine": True,
        "mutual_nearest_neighbor": True,
        "one_to_one": True,
        "clean_permutation_recovered": True,
        "sign_flip_rejected": True,
        "duplicate_collapse_rejected": True,
        "category_profile_swap_rejected": True,
        "v1_matcher_behavior_unchanged": True,
        "required_for_fresh_seed_feature_identity": True,
    }
    assert spec["evidence"]["generation_sae_v2_stratum_replication"] == {
        "method": "signed_mutual_one_to_one_matching_with_training_only_pass_profiles",
        "site_checks": 3,
        "quality_qualified_sites": ["model.layers.11", "model.norm"],
        "primary_diagnostic_decoder_cosine": 0.8,
        "minimum_bidirectional_category_coverage": 0.1,
        "same_category_matches_at_primary_threshold": {
            "model.layers.11": {"pass_0": 19, "pass_1": 10},
            "model.layers.21": {"pass_0": 43, "pass_1": 19},
            "model.norm": {"pass_0": 25, "pass_1": 28},
        },
        "model_norm_bidirectional_category_coverage": {
            "pass_0": 25 / 78,
            "pass_1": 28 / 236,
        },
        "provisional_same_seed_supports": 1,
        "provisional_same_seed_sites": ["model.norm"],
        "operative_fresh_seed_admissions": 0,
        "reason_inoperative": "same_seed_same_training_ids_and_posthoc_thresholds",
        "full_dictionary_replication_claimed": False,
        "fresh_seed_family_disjoint_replication_required": True,
    }
    assert spec["evidence"]["layer21_seed_objective_stability_control"] == {
        "site": "model.layers.21",
        "seeds": [42, 43, 44],
        "objectives": ["raw_mse", "inverse_variance_mse"],
        "quality_qualified_conditions": 0,
        "cross_seed_same_objective_pair_checks": 6,
        "cross_seed_diagnostic_supports_without_quality_gate": 0,
        "same_seed_cross_objective_pair_checks": 3,
        "same_seed_cross_objective_diagnostic_supports_without_quality_gate": 3,
        "operative_admissions": 0,
        "interpretation": (
            "initialization_seed_dominates_apparent_feature_identity_"
            "while_loss_objective_changes_preserve_same_seed_strata"
        ),
        "does_not_test_layer11_or_final_norm": True,
        "quality_gate_cannot_be_bypassed": True,
    }
    assert spec["evidence"]["layer21_category_subspace_stability_control"] == {
        "method": "rank_16_category_decoder_subspaces_with_label_permutation",
        "pair_checks": 9,
        "cross_seed_pair_checks": 6,
        "cross_seed_familywise_corrected_supports_without_quality_gate": 0,
        "same_seed_cross_objective_pair_checks": 3,
        "same_seed_cross_objective_supports_without_quality_gate": 3,
        "operative_admissions": 0,
        "one_nominal_cross_seed_hit_before_multiplicity_correction": True,
        "nominal_hit_same_minus_opposite_below": 0.003,
        "individual_features_and_category_subspaces_both_cross_seed_unstable": True,
        "does_not_test_layer11_or_final_norm": True,
    }
    assert spec["evidence"][
        "generation_sae_cross_budget_category_subspace_transfer"
    ] == {
        "method": (
            "rank_16_training_assigned_pass_decoder_subspaces_with_"
            "label_permutation_and_three_site_bonferroni_control"
        ),
        "site_checks": 3,
        "reported_rank_sensitivity": [4, 8, 16, 32],
        "diagnostic_same_seed_supports": 3,
        "diagnostic_same_seed_sites": [
            "model.layers.11",
            "model.layers.21",
            "model.norm",
        ],
        "quality_qualified_sites": ["model.layers.11", "model.norm"],
        "provisional_same_seed_quality_qualified_supports": 2,
        "provisional_same_seed_quality_qualified_sites": [
            "model.layers.11",
            "model.norm",
        ],
        "primary_rank_same_minus_opposite": {
            "model.layers.11": 0.2371109596382418,
            "model.layers.21": 0.11059298862855133,
            "model.norm": 0.24223160501078123,
        },
        "operative_fresh_seed_admissions": 0,
        "feature_identity_and_category_subspace_transfer_are_distinct": True,
        "reason_inoperative": "same_seed_same_example_ids_and_posthoc_existing_sources",
        "fresh_seed_family_disjoint_replication_required": True,
    }
    assert spec["evidence"]["admitted_categories"] == ["pass_0", "pass_1"]
    assert spec["evidence"]["operative_cluster_count_per_category"] == 1
    assert spec["evidence"]["rejected_child_clusters_are_inoperative"] is True
    assert spec["fresh_pack"]["family_sets_must_be_pairwise_disjoint"] is True
    assert spec["fresh_pack"]["total_families"] == 20
    assert spec["fresh_pack"]["total_examples"] == 80
    arms = spec["sae_comparison"]["arms"]
    assert len({row["total_trainable_parameters"] for row in arms}) == 1
    assert len(
        {
            row["dictionaries"] * row["features_per_dictionary"]
            for row in arms
        }
    ) == 1
    assert len(
        {
            row["dictionaries"] * row["optimizer_steps_per_dictionary"]
            for row in arms
        }
    ) == 1
    assert spec["sae_comparison"]["dictionary_parameterization"]["output_bias"] == (
        "fixed_zero_not_trainable"
    )


def test_spec_cannot_reach_capture_or_training_without_new_authority() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    authority = spec["current_authority"]
    assert authority["new_inference_authorized"] is False
    assert authority["paid_compute_authorized"] is False
    assert authority["sae_training_authorized"] is False
    assert authority["model_or_adapter_weight_mutation_authorized"] is False
    assert authority["codex_goal_pause_is_compute_termination"] is False
    gates = {row["id"]: row for row in spec["stage_gates"]}
    assert "provider_independent_ttl_and_dollar_watchdog" in gates[
        "capture_categories"
    ]["requires"]
    assert "provider_instance_stop_verified_before_goal_pause_or_exit" in gates[
        "capture_categories"
    ]["requires"]
    assert "parent_category_admitted" in gates["admit_child_clusters"]["requires"]
    assert "equal_total_rank_linear_proxy" in gates[
        "screen_capacity_before_sae_training"
    ]["requires"]
    assert "held_out_pass_selectivity" in gates["admit_shared_sae_feature_strata"][
        "requires"
    ]
    assert "held_out_reconstruction_quality_per_pass" in gates[
        "admit_shared_sae_feature_strata"
    ]["requires"]
    assert "at_least_five_independent_training_example_blocks" in gates[
        "admit_activation_phase_category"
    ]["requires"]
    assert "no_child_cluster_without_parent_admission" in gates[
        "admit_activation_phase_category"
    ]["requires"]
    assert "fresh_untouched_final_holdout" in gates[
        "admit_activation_phase_category"
    ]["requires"]
    assert "fresh_pack_linear_capacity_screen_pass" in gates["train_sae_arms"][
        "requires"
    ]
    assert "explicit_user_weight_mutation_authorization" in gates[
        "train_sae_arms"
    ]["requires"]
