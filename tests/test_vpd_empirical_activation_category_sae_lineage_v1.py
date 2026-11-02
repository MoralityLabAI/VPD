from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v52 import (
    canonical_inputs as protocol_inputs,
    verify_protocol,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v44 import (
    canonical_inputs as runbook_inputs,
    verify_runbook,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evals/vpd_activation_category_compartments_v52.json"
RUNBOOK = ROOT / "artifacts/vpd_cheap_executor_runbook_v44/runbook.json"


def test_protocol_v52_reproduces_empirical_category_decision() -> None:
    verification = verify_protocol(PROTOCOL, **protocol_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["pass_replication_archives"] == 2
    assert verification["pass_site_admissions"] == 6
    assert verification["pass_site_combinations"] == 6
    assert verification["within_pass_nontrivial_allocations"] == 0
    assert verification["q4_region_negative_control_archives"] == 2
    assert verification["q4_region_site_pass_admissions"] == 0
    assert verification["q4_region_site_pass_combinations"] == 12
    assert verification["screening_only_region_child_allocations"] == 1
    assert verification["admissible_region_child_allocations"] == 0
    assert verification["linear_proxy_archives"] == 2
    assert verification["linear_proxy_site_checks"] == 6
    assert verification["linear_proxy_pass_wins_vs_random"] == 6
    assert verification["linear_proxy_pass_wins_vs_monolithic"] == 0
    assert verification["linear_proxy_training_admissions"] == 0
    assert verification["rank_sweep_archives"] == 2
    assert verification["rank_sweep_site_rank_checks"] == 36
    assert verification["rank_sweep_pass_wins_vs_monolithic"] == 0
    assert verification["rank_sweep_monolithic_wins_vs_pass"] == 35
    assert verification["rank_sweep_monolithic_vs_pass_inconclusive"] == 1
    assert verification["rank_sweep_pass_wins_vs_random"] == 35
    assert verification["rank_sweep_training_admissions"] == 0
    assert verification["shared_sae_selectivity_archives"] == 2
    assert verification["shared_sae_site_archive_checks"] == 6
    assert verification["shared_sae_internal_feature_strata_admissions"] == 4
    assert verification["shared_sae_admitted_sites_in_both_archives"] == [
        "model.layers.11",
        "model.norm",
    ]
    assert verification["shared_sae_rejected_sites_in_both_archives"] == [
        "model.layers.21"
    ]
    assert verification["shared_sae_robustness_archives"] == 2
    assert verification["shared_sae_selectivity_configurations_per_site"] == 12
    assert verification["shared_sae_selectivity_passing_by_site"] == {
        "model.layers.11": [11, 11],
        "model.layers.21": [12, 12],
        "model.norm": [12, 12],
    }
    assert verification["shared_sae_high_confidence_sites_through_ev_0_25"] == [
        "model.layers.11"
    ]
    assert verification[
        "shared_sae_provisional_sites_at_preregistered_ev_0_20"
    ] == ["model.norm"]
    assert verification["cross_pack_transfer_archive_pairs"] == 1
    assert verification["cross_pack_transfer_site_checks"] == 3
    assert verification["cross_pack_transfer_bidirectional_admissions"] == 2
    assert verification["cross_pack_transfer_admitted_sites"] == [
        "model.layers.21",
        "model.norm",
    ]
    assert verification[
        "sites_meeting_high_confidence_sae_quality_and_direct_pack_direction_replication_gates"
    ] == []
    assert verification["shared_axis_site_checks"] == 3
    assert verification["shared_axis_admissions"] == 3
    assert verification[
        "sites_meeting_high_confidence_sae_quality_and_shared_axis_gates"
    ] == ["model.layers.11"]
    assert verification["prompt_phase_transfer_site_checks"] == 3
    assert verification["prompt_phase_transfer_admissions"] == 1
    assert verification["prompt_phase_transfer_admitted_sites"] == ["model.norm"]
    assert verification["activation_phase_category_site_checks"] == 3
    assert verification["activation_phase_category_admissions"] == 0
    assert verification["activation_phase_exact_p_value_floor"] == 0.0625
    assert verification["activation_phase_minimum_training_blocks_for_p_0_05"] == 5
    assert verification["activation_phase_operative_child_clusters"] == 0
    assert verification["activation_phase_cross_validated_site_checks"] == 3
    assert verification[
        "activation_phase_provisional_cross_validated_supports"
    ] == 3
    assert verification["activation_phase_cross_validated_minimum_accuracy"] == 0.9375
    assert verification[
        "activation_phase_cross_validated_maximum_exact_p_value"
    ] == 0.0078125
    assert verification["activation_phase_cross_validated_operative_admissions"] == 0
    assert verification["activation_phase_capacity_site_checks"] == 3
    assert verification["activation_phase_capacity_wins_vs_monolithic"] == 0
    assert verification["activation_phase_capacity_wins_vs_random_mean"] == 3
    assert verification["activation_phase_capacity_preferences"] == 0
    assert verification["activation_phase_capacity_training_admissions"] == 0
    assert verification["joint_pass_phase_capacity_site_checks"] == 3
    assert verification["joint_pass_phase_wins_vs_monolithic"] == 0
    assert verification["joint_pass_phase_wins_vs_pass_routed"] == 0
    assert verification["joint_pass_phase_wins_vs_phase_routed"] == 0
    assert verification["joint_pass_phase_wins_vs_random_mean"] == 3
    assert verification["joint_pass_phase_capacity_preferences"] == 0
    assert verification["joint_pass_phase_training_admissions"] == 0
    assert verification["prompt_frozen_sae_quality_site_checks"] == 3
    assert verification["prompt_frozen_sae_quality_admissions"] == 0
    assert verification["prompt_phase_pass_feature_strata_admissions"] == 0
    assert verification["generation_sae_feature_identity_site_checks"] == 3
    assert verification[
        "generation_sae_provisional_same_seed_feature_identity_supports"
    ] == 1
    assert verification[
        "generation_sae_provisional_same_seed_feature_identity_sites"
    ] == ["model.norm"]
    assert verification[
        "generation_sae_operative_cross_budget_feature_identity_admissions"
    ] == 0
    assert verification["feature_replica_matching_version"] == "v2"
    assert verification["feature_replica_matching_synthetic_scenarios"] == 4
    assert verification["feature_replica_matching_expected_outcomes"] == 4
    assert verification["feature_replica_matching_signed_cosine"] is True
    assert verification["feature_replica_matching_mutual_one_to_one"] is True
    assert verification["generation_sae_v2_stratum_replication_site_checks"] == 3
    assert verification["generation_sae_v2_provisional_same_seed_supports"] == 1
    assert verification["generation_sae_v2_provisional_same_seed_sites"] == [
        "model.norm"
    ]
    assert verification["generation_sae_v2_operative_fresh_seed_admissions"] == 0
    assert verification["generation_sae_v2_model_norm_primary_matches"] == {
        "pass_0": 25,
        "pass_1": 28,
    }
    assert verification["layer21_seed_objective_conditions"] == 6
    assert verification["layer21_seed_objective_quality_qualified_conditions"] == 0
    assert verification["layer21_cross_seed_pair_checks"] == 6
    assert verification[
        "layer21_cross_seed_diagnostic_supports_without_quality_gate"
    ] == 0
    assert verification["layer21_same_seed_cross_objective_pair_checks"] == 3
    assert verification[
        "layer21_same_seed_cross_objective_diagnostic_supports_without_quality_gate"
    ] == 3
    assert verification["layer21_seed_objective_operative_admissions"] == 0
    assert verification["layer21_category_subspace_pair_checks"] == 9
    assert verification["layer21_cross_seed_subspace_pair_checks"] == 6
    assert verification[
        "layer21_cross_seed_familywise_corrected_subspace_supports"
    ] == 0
    assert verification["layer21_same_seed_cross_objective_subspace_pair_checks"] == 3
    assert verification["layer21_same_seed_cross_objective_subspace_supports"] == 3
    assert verification["layer21_category_subspace_operative_admissions"] == 0
    assert verification["generation_sae_category_subspace_site_checks"] == 3
    assert verification[
        "generation_sae_category_subspace_same_seed_diagnostic_supports"
    ] == 3
    assert verification[
        "generation_sae_category_subspace_same_seed_quality_supports"
    ] == 2
    assert verification[
        "generation_sae_category_subspace_same_seed_quality_sites"
    ] == ["model.layers.11", "model.norm"]
    assert verification[
        "generation_sae_category_subspace_operative_fresh_seed_admissions"
    ] == 0
    assert verification["fresh_pack_present"] is False
    assert verification["execution_authorized"] is False


def test_runbook_v44_reproduces_with_only_local_preflight_safe() -> None:
    verification = verify_runbook(RUNBOOK, **runbook_inputs(ROOT))
    assert verification["status"] == "pass"
    assert verification["safe_stage_now"] == 0
    assert verification["pass_site_admissions"] == 6
    assert verification["q4_region_site_pass_admissions"] == 0
    assert verification["admissible_region_child_allocations"] == 0
    assert verification["linear_proxy_site_checks"] == 6
    assert verification["linear_proxy_pass_wins_vs_random"] == 6
    assert verification["linear_proxy_pass_wins_vs_monolithic"] == 0
    assert verification["linear_proxy_training_admissions"] == 0
    assert verification["rank_sweep_site_rank_checks"] == 36
    assert verification["rank_sweep_pass_wins_vs_monolithic"] == 0
    assert verification["rank_sweep_training_admissions"] == 0
    assert verification["shared_sae_internal_feature_strata_admissions"] == 4
    assert verification["shared_sae_robustness_archives"] == 2
    assert verification["cross_pack_transfer_bidirectional_admissions"] == 2
    assert verification["shared_axis_admissions"] == 3
    assert verification["prompt_phase_transfer_admissions"] == 1
    assert verification["activation_phase_category_admissions"] == 0
    assert verification["activation_phase_provisional_cross_validated_supports"] == 3
    assert verification["activation_phase_capacity_training_admissions"] == 0
    assert verification["joint_pass_phase_training_admissions"] == 0
    assert verification["prompt_frozen_sae_quality_admissions"] == 0
    assert verification["prompt_phase_pass_feature_strata_admissions"] == 0
    assert verification[
        "generation_sae_operative_cross_budget_feature_identity_admissions"
    ] == 0
    assert verification["feature_replica_matching_version"] == "v2"
    assert verification["feature_replica_matching_synthetic_scenarios"] == 4
    assert verification["feature_replica_matching_expected_outcomes"] == 4
    assert verification["generation_sae_v2_stratum_replication_site_checks"] == 3
    assert verification["generation_sae_v2_provisional_same_seed_supports"] == 1
    assert verification["generation_sae_v2_provisional_same_seed_sites"] == [
        "model.norm"
    ]
    assert verification["generation_sae_v2_operative_fresh_seed_admissions"] == 0
    assert verification["layer21_seed_objective_quality_qualified_conditions"] == 0
    assert verification["layer21_cross_seed_pair_checks"] == 6
    assert verification[
        "layer21_cross_seed_diagnostic_supports_without_quality_gate"
    ] == 0
    assert verification["layer21_same_seed_cross_objective_pair_checks"] == 3
    assert verification[
        "layer21_same_seed_cross_objective_diagnostic_supports_without_quality_gate"
    ] == 3
    assert verification["layer21_seed_objective_operative_admissions"] == 0
    assert verification["layer21_category_subspace_pair_checks"] == 9
    assert verification["layer21_cross_seed_subspace_pair_checks"] == 6
    assert verification[
        "layer21_cross_seed_familywise_corrected_subspace_supports"
    ] == 0
    assert verification[
        "layer21_same_seed_cross_objective_subspace_pair_checks"
    ] == 3
    assert verification["layer21_same_seed_cross_objective_subspace_supports"] == 3
    assert verification["layer21_category_subspace_operative_admissions"] == 0
    assert verification["generation_sae_category_subspace_site_checks"] == 3
    assert verification[
        "generation_sae_category_subspace_same_seed_diagnostic_supports"
    ] == 3
    assert verification[
        "generation_sae_category_subspace_same_seed_quality_supports"
    ] == 2
    assert verification[
        "generation_sae_category_subspace_same_seed_quality_sites"
    ] == ["model.layers.11", "model.norm"]
    assert verification[
        "generation_sae_category_subspace_operative_fresh_seed_admissions"
    ] == 0
    assert verification["fresh_pack_present"] is False
    assert verification["capture_authorized"] is False
    assert verification["sae_training_authorized"] is False
    assert verification["execution_authorized"] is False

    runbook = json.loads(RUNBOOK.read_text(encoding="utf-8"))
    safe = [
        stage["id"]
        for stage in runbook["stages"]
        if stage["safe_for_cheap_executor_now"]
    ]
    assert safe == ["local_preflight"]
    command_ids = {row["id"] for row in runbook["stages"][0]["commands"]}
    assert {
        "verify_nanbeige_arithmetic_pass_category_bridge_v1",
        "verify_nanbeige_reasoning_pass_category_bridge_v1",
        "verify_nanbeige_q4_region_category_bridge_2048_v1",
        "verify_nanbeige_q4_region_category_bridge_8192_v1",
        "verify_nanbeige_arithmetic_pass_compartment_linear_proxy_v1",
        "verify_nanbeige_reasoning_pass_compartment_linear_proxy_v1",
        "verify_nanbeige_arithmetic_pass_compartment_rank_sweep_v1",
        "verify_nanbeige_reasoning_pass_compartment_rank_sweep_v1",
        "verify_nanbeige_q4_sae_pass_selectivity_2048_v1",
        "verify_nanbeige_q4_sae_pass_selectivity_8192_v1",
        "verify_nanbeige_q4_sae_pass_selectivity_robustness_2048_v1",
        "verify_nanbeige_q4_sae_pass_selectivity_robustness_8192_v1",
        "verify_nanbeige_pass_category_cross_pack_transfer_v1",
        "verify_nanbeige_pass_category_shared_axis_v1",
        "verify_nanbeige_pass_category_prompt_phase_transfer_v1",
        "verify_nanbeige_activation_phase_category_bridge_v1",
        "verify_nanbeige_activation_phase_category_cross_validation_v1",
        "verify_nanbeige_phase_compartment_linear_proxy_v1",
        "verify_nanbeige_joint_pass_phase_capacity_v1",
        "verify_nanbeige_prompt_frozen_sae_quality_v1",
        "verify_nanbeige_q4_sae_pass_feature_identity_v1",
        "verify_sae_feature_replica_matching_v2",
        "verify_nanbeige_q4_sae_pass_feature_stratum_replication_v2",
        "verify_nanbeige_layer21_sae_seed_objective_stability_v1",
        "verify_nanbeige_layer21_sae_category_subspace_stability_v1",
        "verify_nanbeige_q4_sae_pass_category_subspace_transfer_v1",
        "verify_nanbeige_activation_category_sae_spec_v1",
        "verify_research_protocol_v52",
    }.issubset(command_ids)


def test_lineage_preserves_equal_budget_and_non_authorizing_boundary() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    component = protocol["component_compartments"][
        "nanbeige_empirical_activation_category_sae_experiment_v1"
    ]
    assert component["admitted_category_domain"] == "recurrent_pass_index"
    assert component["admitted_categories"] == ["pass_0", "pass_1"]
    assert component["operative_cluster_count_per_category"] == 1
    assert component["sae_comparison_arms"] == 3
    assert component["matched_trainable_parameters_per_arm"] == [37_754_880]
    assert component["fresh_pack_families"] == 20
    assert component["fresh_pack_examples"] == 80
    assert component["current_preferred_sae_architecture"] == (
        "shared_monolithic_sae_with_posthoc_pass_selective_feature_groups_"
        "at_quality_qualified_sites"
    )
    matcher = component["feature_replica_matching_method"]
    assert matcher["version"] == "v2"
    assert matcher["signed_decoder_cosine"] is True
    assert matcher["mutual_nearest_neighbor"] is True
    assert matcher["one_to_one"] is True
    assert matcher["v1_matcher_behavior_unchanged"] is True
    stratum = component["generation_sae_v2_stratum_replication"]
    assert stratum["provisional_same_seed_sites"] == ["model.norm"]
    assert stratum["operative_fresh_seed_admissions"] == 0
    assert stratum["full_dictionary_replication_claimed"] is False
    seed_control = component["layer21_seed_objective_stability_control"]
    assert seed_control["cross_seed_same_objective_pair_checks"] == 6
    assert seed_control["cross_seed_diagnostic_supports_without_quality_gate"] == 0
    assert seed_control["same_seed_cross_objective_pair_checks"] == 3
    assert (
        seed_control[
            "same_seed_cross_objective_diagnostic_supports_without_quality_gate"
        ]
        == 3
    )
    assert seed_control["operative_admissions"] == 0
    subspace_control = component["layer21_category_subspace_stability_control"]
    assert subspace_control["cross_seed_pair_checks"] == 6
    assert subspace_control[
        "cross_seed_familywise_corrected_supports_without_quality_gate"
    ] == 0
    assert subspace_control["same_seed_cross_objective_supports_without_quality_gate"] == 3
    assert subspace_control["operative_admissions"] == 0
    cross_budget_subspace = component[
        "generation_sae_cross_budget_category_subspace_transfer"
    ]
    assert cross_budget_subspace["diagnostic_same_seed_supports"] == 3
    assert cross_budget_subspace[
        "provisional_same_seed_quality_qualified_supports"
    ] == 2
    assert cross_budget_subspace[
        "provisional_same_seed_quality_qualified_sites"
    ] == ["model.layers.11", "model.norm"]
    assert cross_budget_subspace["operative_fresh_seed_admissions"] == 0
    boundary = protocol["scientific_boundary"]
    assert boundary["empirical_pass_index_admitted_as_coarse_category"] is True
    assert boundary["empirical_within_pass_subclusters_admitted"] is False
    assert boundary["empirical_temporal_region_categories_admitted"] is False
    assert boundary["empirical_region_child_clusters_admitted"] is False
    assert boundary["empirical_pass_capacity_screen_admitted"] is False
    assert boundary["shared_sae_internal_pass_feature_strata_admitted"] is True
    assert boundary["shared_sae_quality_qualified_sites"] == [
        "model.layers.11",
        "model.norm",
    ]
    assert boundary["shared_sae_layer21_feature_strata_admitted"] is False
    assert boundary["shared_sae_high_confidence_site_through_ev_0_25"] == (
        "model.layers.11"
    )
    assert boundary["shared_sae_provisional_site_at_preregistered_ev_0_20"] == (
        "model.norm"
    )
    assert boundary["codex_goal_pause_is_compute_termination"] is False
    assert boundary["pass_category_cross_pack_transfer_admitted_sites"] == [
        "model.layers.21",
        "model.norm",
    ]
    assert boundary["layer11_pass_direction_is_task_dependent"] is True
    assert boundary[
        "sites_meeting_high_confidence_sae_quality_and_direct_pack_direction_replication_gates"
    ] == []
    assert boundary[
        "cross_pack_transfer_checkpoint_revision_independently_verified"
    ] is False
    assert boundary["shared_pass_axis_admitted_at_all_sites"] is True
    assert boundary[
        "sites_meeting_high_confidence_sae_quality_and_shared_axis_gates"
    ] == ["model.layers.11"]
    assert boundary["shared_axis_third_domain_transfer_established"] is False
    assert boundary[
        "generation_shared_axis_prompt_phase_transfer_admitted_sites"
    ] == ["model.norm"]
    assert boundary["layer11_generation_shared_axis_is_phase_specific"] is True
    assert boundary["activation_phase_category_admitted"] is False
    assert boundary["activation_phase_category_current_exact_test_underpowered"] is True
    assert boundary["activation_phase_compartment_sae_training_authorized"] is False
    assert boundary[
        "activation_phase_family_blocked_provisional_support_at_all_sites"
    ] is True
    assert boundary["activation_phase_fresh_untouched_holdout_present"] is False
    assert boundary["activation_phase_fixed_capacity_screen_admitted"] is False
    assert boundary["shared_monolithic_capacity_preferred_over_phase_split"] is True
    assert boundary["joint_pass_phase_fixed_capacity_screen_admitted"] is False
    assert boundary["joint_pass_phase_dictionaries_authorized"] is False
    assert boundary[
        "shared_factor_labeled_features_preferred_over_cartesian_dictionary_split"
    ] is True
    assert boundary["prompt_frozen_feature_archives_exist"] is True
    assert boundary["prompt_phase_feature_interpretation_quality_admitted"] is False
    assert boundary["cross_phase_pass_feature_strata_admitted"] is False
    assert boundary[
        "generation_sae_same_seed_feature_identity_provisional_site"
    ] == "model.norm"
    assert boundary[
        "generation_sae_cross_budget_feature_identity_operatively_admitted"
    ] is False
    assert boundary[
        "fresh_seed_family_disjoint_feature_identity_replication_required"
    ] is True
    assert boundary["fresh_seed_feature_identity_matcher_version"] == "v2"
    assert boundary["fresh_seed_feature_identity_requires_signed_cosine"] is True
    assert boundary[
        "fresh_seed_feature_identity_requires_mutual_one_to_one_matching"
    ] is True
    assert boundary["v1_feature_replica_matcher_behavior_changed"] is False
    assert boundary[
        "generation_sae_v2_stratum_replication_provisional_site"
    ] == "model.norm"
    assert boundary[
        "generation_sae_v2_stratum_replication_operatively_admitted"
    ] is False
    assert boundary["generation_sae_v2_full_dictionary_replication_claimed"] is False
    assert boundary["layer21_cross_seed_feature_stability_admitted"] is False
    assert boundary[
        "same_seed_cross_objective_stability_is_independent_replication"
    ] is False
    assert boundary["feature_identity_replication_must_change_initialization_seed"] is True
    assert boundary["layer21_seed_controls_can_override_reconstruction_failure"] is False
    assert boundary["layer21_cross_seed_category_subspace_stability_admitted"] is False
    assert boundary["individual_feature_instability_proves_subspace_instability"] is False
    assert boundary["category_subspace_replication_requires_multiplicity_control"] is True
    assert boundary[
        "same_seed_category_subspace_transfer_is_independent_replication"
    ] is False
    assert boundary[
        "generation_sae_cross_budget_category_subspace_operatively_admitted"
    ] is False
    assert boundary[
        "feature_identity_and_category_subspace_transfer_are_distinct"
    ] is True
    assert boundary["new_model_inference_performed"] is False
    assert boundary["sae_dictionary_training_authorized"] is False
    assert boundary["execution_authorized"] is False
