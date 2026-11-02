from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_nanbeige_activation_category_bridge_v1 import (
    verify_report as verify_pass_report,
)
from experiments.authz_mechinterp.benchmark_nanbeige_activation_phase_category_bridge_v1 import (
    verify_report as verify_phase_category_report,
)
from experiments.authz_mechinterp.benchmark_nanbeige_activation_phase_category_cross_validation_v1 import (
    verify_report as verify_phase_category_cross_validation_report,
)
from experiments.authz_mechinterp.benchmark_nanbeige_q4_region_category_bridge_v1 import (
    verify_report as verify_region_report,
)
from experiments.authz_mechinterp.benchmark_nanbeige_pass_compartment_linear_proxy_v1 import (
    verify_report as verify_linear_proxy_report,
)
from experiments.authz_mechinterp.benchmark_nanbeige_phase_compartment_linear_proxy_v1 import (
    verify_report as verify_phase_linear_proxy_report,
)
from experiments.authz_mechinterp.benchmark_nanbeige_joint_pass_phase_capacity_v1 import (
    verify_report as verify_joint_pass_phase_capacity_report,
)
from experiments.authz_mechinterp.audit_nanbeige_prompt_frozen_sae_quality_v1 import (
    verify_report as verify_prompt_frozen_sae_quality_report,
)
from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_feature_identity_v1 import (
    verify_report as verify_q4_sae_pass_feature_identity_report,
)
from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_feature_replica_matching_v2 import (
    verify_report as verify_q4_sae_pass_feature_replica_matching_v2_report,
)
from experiments.authz_mechinterp.benchmark_nanbeige_layer21_sae_seed_objective_stability_v1 import (
    verify_report as verify_layer21_seed_objective_stability_report,
)
from experiments.authz_mechinterp.benchmark_nanbeige_layer21_sae_category_subspace_stability_v1 import (
    verify_report as verify_layer21_category_subspace_stability_report,
)
from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_category_subspace_transfer_v1 import (
    verify_report as verify_q4_sae_pass_category_subspace_transfer_report,
)
from experiments.authz_mechinterp.evaluate_sae_feature_replica_matching_v2 import (
    verify_benchmark as verify_feature_replica_matching_v2_benchmark,
)
from experiments.authz_mechinterp.benchmark_nanbeige_pass_compartment_rank_sweep_v1 import (
    verify_report as verify_rank_sweep_report,
)
from experiments.authz_mechinterp.benchmark_nanbeige_pass_category_cross_pack_transfer_v1 import (
    verify_report as verify_cross_pack_transfer_report,
)
from experiments.authz_mechinterp.benchmark_nanbeige_pass_category_shared_axis_v1 import (
    verify_report as verify_shared_axis_report,
)
from experiments.authz_mechinterp.benchmark_nanbeige_pass_category_prompt_phase_transfer_v1 import (
    verify_report as verify_prompt_phase_transfer_report,
)
from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_selectivity_v1 import (
    verify_report as verify_sae_selectivity_report,
)
from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_selectivity_robustness_v1 import (
    verify_report as verify_sae_selectivity_robustness_report,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-activation-category-sae-experiment-spec.v1"


def _read_verified(path: Path, *, kind: str) -> dict[str, Any]:
    verification = (
        verify_pass_report(path) if kind == "pass" else verify_region_report(path)
    )
    if verification.get("status") != "pass":
        raise ValueError("activation-category evidence did not verify")
    return json.loads(path.read_text(encoding="utf-8"))


def build_spec(
    *,
    arithmetic_pass_report_path: Path,
    reasoning_pass_report_path: Path,
    q4_region_report_paths: list[Path],
    arithmetic_linear_proxy_report_path: Path,
    reasoning_linear_proxy_report_path: Path,
    arithmetic_rank_sweep_report_path: Path,
    reasoning_rank_sweep_report_path: Path,
    q4_sae_selectivity_2048_report_path: Path,
    q4_sae_selectivity_8192_report_path: Path,
    q4_sae_selectivity_robustness_2048_report_path: Path,
    q4_sae_selectivity_robustness_8192_report_path: Path,
    pass_category_cross_pack_transfer_report_path: Path,
    pass_category_shared_axis_report_path: Path,
    pass_category_prompt_phase_transfer_report_path: Path,
    activation_phase_category_report_path: Path,
    activation_phase_category_cross_validation_report_path: Path,
    phase_compartment_linear_proxy_report_path: Path,
    joint_pass_phase_capacity_report_path: Path,
    prompt_frozen_sae_quality_report_path: Path,
    q4_sae_pass_feature_identity_report_path: Path,
    q4_sae_pass_feature_replica_matching_v2_report_path: Path,
    layer21_seed_objective_stability_report_path: Path,
    layer21_category_subspace_stability_report_path: Path,
    q4_sae_pass_category_subspace_transfer_report_path: Path,
    feature_replica_matching_v2_benchmark_path: Path,
) -> dict[str, Any]:
    input_dimension = 3072
    total_dictionary_features = 6144
    matched_trainable_parameters = total_dictionary_features * (
        2 * input_dimension + 1
    )
    arithmetic = _read_verified(arithmetic_pass_report_path, kind="pass")
    reasoning = _read_verified(reasoning_pass_report_path, kind="pass")
    if len(q4_region_report_paths) < 2:
        raise ValueError("both short- and long-context Q4 region controls are required")
    region_reports = [
        _read_verified(path, kind="region") for path in q4_region_report_paths
    ]
    region_contexts = {
        int(report["source"]["max_new_tokens"]) for report in region_reports
    }
    if region_contexts != {2048, 8192}:
        raise ValueError("Q4 region controls must cover 2048 and 8192 generated tokens")
    linear_proxy_reports = [
        json.loads(arithmetic_linear_proxy_report_path.read_text(encoding="utf-8")),
        json.loads(reasoning_linear_proxy_report_path.read_text(encoding="utf-8")),
    ]
    for path, report in zip(
        (arithmetic_linear_proxy_report_path, reasoning_linear_proxy_report_path),
        linear_proxy_reports,
    ):
        if verify_linear_proxy_report(path).get("status") != "pass":
            raise ValueError("linear capacity proxy did not verify")
        if report["summary"]["pass_compartment_screening_admissions"] != 0:
            raise ValueError("current linear capacity proxy unexpectedly admits training")
    rank_sweep_reports = [
        json.loads(arithmetic_rank_sweep_report_path.read_text(encoding="utf-8")),
        json.loads(reasoning_rank_sweep_report_path.read_text(encoding="utf-8")),
    ]
    for path, report in zip(
        (arithmetic_rank_sweep_report_path, reasoning_rank_sweep_report_path),
        rank_sweep_reports,
    ):
        if verify_rank_sweep_report(path).get("status") != "pass":
            raise ValueError("linear capacity rank sweep did not verify")
        if report["summary"]["pass_compartment_screening_admissions"] != 0:
            raise ValueError("current capacity rank sweep unexpectedly admits training")
    sae_selectivity_reports = [
        json.loads(q4_sae_selectivity_2048_report_path.read_text(encoding="utf-8")),
        json.loads(q4_sae_selectivity_8192_report_path.read_text(encoding="utf-8")),
    ]
    for path, report in zip(
        (q4_sae_selectivity_2048_report_path, q4_sae_selectivity_8192_report_path),
        sae_selectivity_reports,
    ):
        if verify_sae_selectivity_report(path).get("status") != "pass":
            raise ValueError("shared-SAE pass selectivity did not verify")
        if report["summary"]["internal_pass_specialization_admissions"] != 2:
            raise ValueError("shared-SAE internal pass specialization outcome differs")
    sae_robustness_reports = [
        json.loads(
            q4_sae_selectivity_robustness_2048_report_path.read_text(encoding="utf-8")
        ),
        json.loads(
            q4_sae_selectivity_robustness_8192_report_path.read_text(encoding="utf-8")
        ),
    ]
    for path, report in zip(
        (
            q4_sae_selectivity_robustness_2048_report_path,
            q4_sae_selectivity_robustness_8192_report_path,
        ),
        sae_robustness_reports,
    ):
        if verify_sae_selectivity_robustness_report(path).get("status") != "pass":
            raise ValueError("shared-SAE selectivity robustness did not verify")
        if report["summary"]["sites_passing_all_selectivity_configurations"] != 2:
            raise ValueError("shared-SAE selectivity robustness outcome differs")
    cross_pack_transfer = json.loads(
        pass_category_cross_pack_transfer_report_path.read_text(encoding="utf-8")
    )
    if verify_cross_pack_transfer_report(
        pass_category_cross_pack_transfer_report_path
    ).get("status") != "pass":
        raise ValueError("cross-pack pass-category transfer did not verify")
    if cross_pack_transfer["summary"][
        "bidirectional_cross_pack_category_admissions"
    ] != 2:
        raise ValueError("cross-pack pass-category transfer outcome differs")
    shared_axis = json.loads(
        pass_category_shared_axis_report_path.read_text(encoding="utf-8")
    )
    if verify_shared_axis_report(pass_category_shared_axis_report_path).get(
        "status"
    ) != "pass":
        raise ValueError("shared pass-category axis did not verify")
    if shared_axis["summary"]["shared_axis_admissions"] != 3:
        raise ValueError("shared pass-category axis outcome differs")
    prompt_phase_transfer = json.loads(
        pass_category_prompt_phase_transfer_report_path.read_text(encoding="utf-8")
    )
    if verify_prompt_phase_transfer_report(
        pass_category_prompt_phase_transfer_report_path
    ).get("status") != "pass":
        raise ValueError("prompt-phase pass-category transfer did not verify")
    if prompt_phase_transfer["summary"]["prompt_phase_transfer_admissions"] != 1:
        raise ValueError("prompt-phase pass-category transfer outcome differs")
    phase_category = json.loads(
        activation_phase_category_report_path.read_text(encoding="utf-8")
    )
    if verify_phase_category_report(activation_phase_category_report_path).get(
        "status"
    ) != "pass":
        raise ValueError("activation-phase category screen did not verify")
    if (
        phase_category["summary"]["activation_phase_category_admissions"] != 0
        or phase_category["summary"]["operative_nontrivial_within_phase_allocations"]
        != 0
    ):
        raise ValueError("activation-phase category screen outcome differs")
    phase_cross_validation = json.loads(
        activation_phase_category_cross_validation_report_path.read_text(
            encoding="utf-8"
        )
    )
    if verify_phase_category_cross_validation_report(
        activation_phase_category_cross_validation_report_path
    ).get("status") != "pass":
        raise ValueError("activation-phase category cross-validation did not verify")
    if (
        phase_cross_validation["summary"][
            "provisional_cross_validated_phase_category_supports"
        ]
        != 3
        or phase_cross_validation["summary"]["operative_phase_category_admissions"]
        != 0
    ):
        raise ValueError("activation-phase cross-validation outcome differs")
    phase_linear_proxy = json.loads(
        phase_compartment_linear_proxy_report_path.read_text(encoding="utf-8")
    )
    if verify_phase_linear_proxy_report(phase_compartment_linear_proxy_report_path).get(
        "status"
    ) != "pass":
        raise ValueError("phase compartment linear proxy did not verify")
    if (
        phase_linear_proxy["summary"]["screening_capacity_preferences"] != 0
        or phase_linear_proxy["summary"]["phase_compartment_training_admissions"]
        != 0
    ):
        raise ValueError("phase compartment capacity outcome differs")
    joint_pass_phase_capacity = json.loads(
        joint_pass_phase_capacity_report_path.read_text(encoding="utf-8")
    )
    if verify_joint_pass_phase_capacity_report(joint_pass_phase_capacity_report_path).get(
        "status"
    ) != "pass":
        raise ValueError("joint pass/phase capacity proxy did not verify")
    if (
        joint_pass_phase_capacity["summary"]["joint_capacity_preferences"] != 0
        or joint_pass_phase_capacity["summary"][
            "joint_compartment_training_admissions"
        ]
        != 0
    ):
        raise ValueError("joint pass/phase capacity outcome differs")
    prompt_frozen_sae_quality = json.loads(
        prompt_frozen_sae_quality_report_path.read_text(encoding="utf-8")
    )
    if verify_prompt_frozen_sae_quality_report(prompt_frozen_sae_quality_report_path).get(
        "status"
    ) != "pass":
        raise ValueError("prompt frozen-SAE quality audit did not verify")
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
    q4_sae_pass_feature_identity = json.loads(
        q4_sae_pass_feature_identity_report_path.read_text(encoding="utf-8")
    )
    if verify_q4_sae_pass_feature_identity_report(
        q4_sae_pass_feature_identity_report_path
    ).get("status") != "pass":
        raise ValueError("Q4 SAE pass-feature identity did not verify")
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
    q4_sae_pass_feature_replica_matching_v2 = json.loads(
        q4_sae_pass_feature_replica_matching_v2_report_path.read_text(
            encoding="utf-8"
        )
    )
    if verify_q4_sae_pass_feature_replica_matching_v2_report(
        q4_sae_pass_feature_replica_matching_v2_report_path
    ).get("status") != "pass":
        raise ValueError("Q4 SAE V2 pass-feature stratum replication did not verify")
    if (
        q4_sae_pass_feature_replica_matching_v2["summary"][
            "provisional_same_seed_v2_stratum_replication_supports"
        ]
        != 1
        or q4_sae_pass_feature_replica_matching_v2["summary"][
            "operative_fresh_seed_stratum_replication_admissions"
        ]
        != 0
    ):
        raise ValueError("Q4 SAE V2 pass-feature stratum outcome differs")
    layer21_seed_objective_stability = json.loads(
        layer21_seed_objective_stability_report_path.read_text(encoding="utf-8")
    )
    if verify_layer21_seed_objective_stability_report(
        layer21_seed_objective_stability_report_path
    ).get("status") != "pass":
        raise ValueError("layer21 SAE seed/objective stability control did not verify")
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
        raise ValueError("layer21 SAE seed/objective stability outcome differs")
    layer21_category_subspace_stability = json.loads(
        layer21_category_subspace_stability_report_path.read_text(encoding="utf-8")
    )
    if verify_layer21_category_subspace_stability_report(
        layer21_category_subspace_stability_report_path
    ).get("status") != "pass":
        raise ValueError("layer21 SAE category-subspace stability did not verify")
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
        raise ValueError("layer21 SAE category-subspace outcome differs")
    q4_sae_pass_category_subspace_transfer = json.loads(
        q4_sae_pass_category_subspace_transfer_report_path.read_text(encoding="utf-8")
    )
    if verify_q4_sae_pass_category_subspace_transfer_report(
        q4_sae_pass_category_subspace_transfer_report_path
    ).get("status") != "pass":
        raise ValueError("Q4 SAE pass-category subspace transfer did not verify")
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
        raise ValueError("Q4 SAE pass-category subspace transfer outcome differs")
    feature_replica_matching_v2 = json.loads(
        feature_replica_matching_v2_benchmark_path.read_text(encoding="utf-8")
    )
    if verify_feature_replica_matching_v2_benchmark(
        feature_replica_matching_v2_benchmark_path
    ).get("status") != "pass":
        raise ValueError("V2 feature-replica matching benchmark did not verify")
    if feature_replica_matching_v2["summary"]["expected_outcomes_reproduced"] != 4:
        raise ValueError("V2 feature-replica matching benchmark outcome differs")
    pass_reports = [arithmetic, reasoning]
    if any(
        report["summary"]["transition_category_admissions"]
        != report["summary"]["sites"]
        for report in pass_reports
    ):
        raise ValueError("coarse recurrent-pass evidence is not consistently admitted")
    if any(
        allocation != {"pass_0": 1, "pass_1": 1}
        for report in pass_reports
        for allocation in report["summary"]["within_pass_allocations"].values()
    ):
        raise ValueError("existing recurrent-pass evidence contains a nontrivial split")
    if any(
        report["summary"]["region_category_admissions"] != 0
        or report["summary"]["admissible_within_region_nontrivial_allocations"] != 0
        for report in region_reports
    ):
        raise ValueError("temporal-region evidence unexpectedly authorizes compartments")

    evidence = {
        "admitted_category_domain": "recurrent_pass_index",
        "admitted_categories": ["pass_0", "pass_1"],
        "operative_cluster_count_per_category": 1,
        "rejected_category_domain": "prompt_early_middle_late_region",
        "rejected_child_clusters_are_inoperative": True,
        "reports": [
            {
                "role": "arithmetic_pass_replication",
                "path": arithmetic_pass_report_path.as_posix(),
                "sha256": sha256_file(arithmetic_pass_report_path),
                "report_id": arithmetic["report_id"],
            },
            {
                "role": "broader_reasoning_pass_replication",
                "path": reasoning_pass_report_path.as_posix(),
                "sha256": sha256_file(reasoning_pass_report_path),
                "report_id": reasoning["report_id"],
            },
        ]
        + [
            {
                "role": (
                    "q4_temporal_region_negative_control_"
                    f"{int(report['source']['max_new_tokens'])}_tokens"
                ),
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "report_id": report["report_id"],
            }
            for path, report in sorted(
                zip(q4_region_report_paths, region_reports),
                key=lambda pair: int(pair[1]["source"]["max_new_tokens"]),
            )
        ],
        "capacity_screening": {
            "method": "equal_total_rank_held_out_linear_subspace_proxy",
            "current_archives_pass_compartment_admissions": 0,
            "current_archives_site_checks": sum(
                int(report["summary"]["sites"]) for report in linear_proxy_reports
            ),
            "rank_sweep_total_ranks": [8, 16, 32, 64, 96, 128],
            "rank_sweep_archive_site_rank_checks": sum(
                int(report["summary"]["rank_site_checks"])
                for report in rank_sweep_reports
            ),
            "rank_sweep_pass_wins_vs_monolithic": sum(
                int(report["summary"]["pass_wins_vs_monolithic"])
                for report in rank_sweep_reports
            ),
            "rank_sweep_monolithic_wins_vs_pass": sum(
                int(report["summary"]["monolithic_wins_vs_pass"])
                for report in rank_sweep_reports
            ),
            "rank_sweep_monolithic_vs_pass_inconclusive": sum(
                int(report["summary"]["monolithic_vs_pass_inconclusive"])
                for report in rank_sweep_reports
            ),
            "rank_sweep_pass_wins_vs_random_mean": sum(
                int(report["summary"]["pass_wins_vs_random_mean"])
                for report in rank_sweep_reports
            ),
            "rank_sweep_training_admissions": sum(
                int(report["summary"]["pass_compartment_screening_admissions"])
                for report in rank_sweep_reports
            ),
            "current_action": (
                "do_not_prioritize_pass_compartment_sae_training_on_recovered_archives"
            ),
            "fresh_pack_requirement": (
                "repeat_before_training_and_stop_if_pass_routing_does_not_beat_"
                "monolithic_and_random_controls"
            ),
        },
        "shared_sae_internal_compartmentalization": {
            "method": "training_only_feature_assignment_with_held_out_pass_validation",
            "archives": 2,
            "site_archive_checks": 6,
            "admissions": 4,
            "admitted_sites_in_both_archives": [
                "model.layers.11",
                "model.norm",
            ],
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
        },
        "cross_pack_category_transfer": {
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
        },
        "shared_axis_category_geometry": {
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
        },
        "prompt_phase_transfer": {
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
        },
        "activation_phase_category_screen": {
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
        },
        "activation_phase_category_cross_validation": {
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
        },
        "activation_phase_capacity_screening": {
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
        },
        "joint_pass_phase_capacity_screening": {
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
        },
        "prompt_phase_frozen_sae_quality": {
            "method": "held_out_reconstruction_gate_on_hash_bound_frozen_feature_archives",
            "site_checks": 3,
            "minimum_explained_variance_threshold": 0.20,
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
        },
        "generation_sae_cross_budget_feature_identity": {
            "method": "same_index_decoder_cosine_and_training_only_pass_assignment",
            "site_checks": 3,
            "quality_qualified_sites": ["model.layers.11", "model.norm"],
            "primary_diagnostic_decoder_cosine": 0.80,
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
        },
        "generation_sae_v2_stratum_replication": {
            "method": "signed_mutual_one_to_one_matching_with_training_only_pass_profiles",
            "site_checks": 3,
            "quality_qualified_sites": ["model.layers.11", "model.norm"],
            "primary_diagnostic_decoder_cosine": 0.80,
            "minimum_bidirectional_category_coverage": 0.10,
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
        },
        "layer21_seed_objective_stability_control": {
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
        },
        "layer21_category_subspace_stability_control": {
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
        },
        "generation_sae_cross_budget_category_subspace_transfer": {
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
        },
        "feature_replica_matching_method": {
            "version": "v2",
            "path": feature_replica_matching_v2_benchmark_path.as_posix(),
            "sha256": sha256_file(feature_replica_matching_v2_benchmark_path),
            "benchmark_id": feature_replica_matching_v2["benchmark_id"],
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
        },
    }
    evidence["reports"].extend(
        [
            {
                "role": "arithmetic_pass_compartment_linear_proxy_negative",
                "path": arithmetic_linear_proxy_report_path.as_posix(),
                "sha256": sha256_file(arithmetic_linear_proxy_report_path),
                "report_id": linear_proxy_reports[0]["report_id"],
            },
            {
                "role": "reasoning_pass_compartment_linear_proxy_negative",
                "path": reasoning_linear_proxy_report_path.as_posix(),
                "sha256": sha256_file(reasoning_linear_proxy_report_path),
                "report_id": linear_proxy_reports[1]["report_id"],
            },
        ]
    )
    evidence["reports"].append(
        {
            "role": "bf16_activation_phase_category_cross_validation",
            "path": activation_phase_category_cross_validation_report_path.as_posix(),
            "sha256": sha256_file(
                activation_phase_category_cross_validation_report_path
            ),
            "report_id": phase_cross_validation["report_id"],
        }
    )
    evidence["reports"].append(
        {
            "role": "bf16_activation_phase_compartment_linear_proxy_negative",
            "path": phase_compartment_linear_proxy_report_path.as_posix(),
            "sha256": sha256_file(phase_compartment_linear_proxy_report_path),
            "report_id": phase_linear_proxy["report_id"],
        }
    )
    evidence["reports"].append(
        {
            "role": "bf16_joint_pass_phase_capacity_proxy_negative",
            "path": joint_pass_phase_capacity_report_path.as_posix(),
            "sha256": sha256_file(joint_pass_phase_capacity_report_path),
            "report_id": joint_pass_phase_capacity["report_id"],
        }
    )
    evidence["reports"].append(
        {
            "role": "bf16_prompt_frozen_sae_quality_negative",
            "path": prompt_frozen_sae_quality_report_path.as_posix(),
            "sha256": sha256_file(prompt_frozen_sae_quality_report_path),
            "report_id": prompt_frozen_sae_quality["report_id"],
        }
    )
    evidence["reports"].append(
        {
            "role": "q4_generation_sae_cross_budget_pass_feature_identity",
            "path": q4_sae_pass_feature_identity_report_path.as_posix(),
            "sha256": sha256_file(q4_sae_pass_feature_identity_report_path),
            "report_id": q4_sae_pass_feature_identity["report_id"],
        }
    )
    evidence["reports"].append(
        {
            "role": "q4_generation_sae_pass_feature_stratum_replication_v2",
            "path": q4_sae_pass_feature_replica_matching_v2_report_path.as_posix(),
            "sha256": sha256_file(
                q4_sae_pass_feature_replica_matching_v2_report_path
            ),
            "report_id": q4_sae_pass_feature_replica_matching_v2["report_id"],
        }
    )
    evidence["reports"].append(
        {
            "role": "layer21_sae_seed_objective_stability_negative_control",
            "path": layer21_seed_objective_stability_report_path.as_posix(),
            "sha256": sha256_file(layer21_seed_objective_stability_report_path),
            "report_id": layer21_seed_objective_stability["report_id"],
        }
    )
    evidence["reports"].append(
        {
            "role": "layer21_sae_category_subspace_stability_negative_control",
            "path": layer21_category_subspace_stability_report_path.as_posix(),
            "sha256": sha256_file(layer21_category_subspace_stability_report_path),
            "report_id": layer21_category_subspace_stability["report_id"],
        }
    )
    evidence["reports"].append(
        {
            "role": "q4_generation_sae_cross_budget_pass_category_subspace_transfer",
            "path": q4_sae_pass_category_subspace_transfer_report_path.as_posix(),
            "sha256": sha256_file(q4_sae_pass_category_subspace_transfer_report_path),
            "report_id": q4_sae_pass_category_subspace_transfer["report_id"],
        }
    )
    evidence["reports"].append(
        {
            "role": "bf16_pass_category_cross_pack_transfer",
            "path": pass_category_cross_pack_transfer_report_path.as_posix(),
            "sha256": sha256_file(pass_category_cross_pack_transfer_report_path),
            "report_id": cross_pack_transfer["report_id"],
        }
    )
    evidence["reports"].extend(
        [
            {
                "role": "bf16_pass_category_prompt_phase_transfer",
                "path": pass_category_prompt_phase_transfer_report_path.as_posix(),
                "sha256": sha256_file(pass_category_prompt_phase_transfer_report_path),
                "report_id": prompt_phase_transfer["report_id"],
            },
            {
                "role": "bf16_activation_phase_category_screen",
                "path": activation_phase_category_report_path.as_posix(),
                "sha256": sha256_file(activation_phase_category_report_path),
                "report_id": phase_category["report_id"],
            },
        ]
    )
    evidence["reports"].append(
        {
            "role": "bf16_pass_category_shared_axis",
            "path": pass_category_shared_axis_report_path.as_posix(),
            "sha256": sha256_file(pass_category_shared_axis_report_path),
            "report_id": shared_axis["report_id"],
        }
    )
    evidence["reports"].extend(
        [
            {
                "role": "q4_2048_shared_sae_pass_selectivity_robustness",
                "path": q4_sae_selectivity_robustness_2048_report_path.as_posix(),
                "sha256": sha256_file(q4_sae_selectivity_robustness_2048_report_path),
                "report_id": sae_robustness_reports[0]["report_id"],
            },
            {
                "role": "q4_8192_shared_sae_pass_selectivity_robustness",
                "path": q4_sae_selectivity_robustness_8192_report_path.as_posix(),
                "sha256": sha256_file(q4_sae_selectivity_robustness_8192_report_path),
                "report_id": sae_robustness_reports[1]["report_id"],
            },
        ]
    )
    evidence["reports"].extend(
        [
            {
                "role": "q4_2048_shared_sae_internal_pass_specialization",
                "path": q4_sae_selectivity_2048_report_path.as_posix(),
                "sha256": sha256_file(q4_sae_selectivity_2048_report_path),
                "report_id": sae_selectivity_reports[0]["report_id"],
            },
            {
                "role": "q4_8192_shared_sae_internal_pass_specialization",
                "path": q4_sae_selectivity_8192_report_path.as_posix(),
                "sha256": sha256_file(q4_sae_selectivity_8192_report_path),
                "report_id": sae_selectivity_reports[1]["report_id"],
            },
        ]
    )
    evidence["reports"].extend(
        [
            {
                "role": "arithmetic_pass_compartment_rank_sweep_negative",
                "path": arithmetic_rank_sweep_report_path.as_posix(),
                "sha256": sha256_file(arithmetic_rank_sweep_report_path),
                "report_id": rank_sweep_reports[0]["report_id"],
            },
            {
                "role": "reasoning_pass_compartment_rank_sweep_negative",
                "path": reasoning_rank_sweep_report_path.as_posix(),
                "sha256": sha256_file(reasoning_rank_sweep_report_path),
                "report_id": rank_sweep_reports[1]["report_id"],
            },
        ]
    )
    roles = {
        "discovery": {"families": 4, "examples_per_family": 4},
        "threshold_calibration": {"families": 4, "examples_per_family": 4},
        "cluster_validation": {"families": 4, "examples_per_family": 4},
        "sae_training": {"families": 4, "examples_per_family": 4},
        "final_evaluation": {"families": 4, "examples_per_family": 4},
    }
    body = {
        "schema": SCHEMA,
        "status": "sealed_design_empirical_capture_and_training_not_authorized",
        "evidence": evidence,
        "fresh_pack": {
            "target_model_id": "Nanbeige/Nanbeige4.2-3B",
            "required_revision": "451ed48c3273ecef7ea8faaa43c31ce529763bb1",
            "revision_must_be_verified_before_capture": True,
            "dtype": "torch.bfloat16",
            "roles": roles,
            "total_families": sum(value["families"] for value in roles.values()),
            "total_examples": sum(
                value["families"] * value["examples_per_family"]
                for value in roles.values()
            ),
            "family_sets_must_be_pairwise_disjoint": True,
            "observable_categories_fixed_before_capture": ["pass_0", "pass_1"],
            "capture_sites": [
                "model.layers.11",
                "model.layers.21",
                "model.norm",
            ],
            "rows_per_example_per_category_per_site": 32,
            "artifact_hashes_and_split_membership_frozen_before_inference": True,
            "category_cluster_selector": {
                "version": "v2",
                "projection_replicates": 8,
                "minimum_replicate_adjusted_rand": 0.90,
                "minimum_replicate_mean_silhouette": 0.15,
                "minimum_replicate_silhouette_p10": 0.05,
                "minimum_replicate_boundary_gap_ratio": 1.50,
                "minimum_cluster_size": 4,
                "no_passing_nontrivial_candidate_rule": "one_cluster_for_category",
                "parent_category_must_be_admitted_before_child_cluster": True,
            },
        },
        "sae_comparison": {
            "dictionary_parameterization": {
                "input_dimension": input_dimension,
                "activation_centering": "training_role_mean_frozen_before_evaluation",
                "encoder_weight": "trainable",
                "decoder_weight": "trainable",
                "feature_bias": "trainable",
                "output_bias": "fixed_zero_not_trainable",
                "reason": (
                    "A fixed output bias makes total trainable parameters exactly "
                    "additive in dictionary features across one- and two-dictionary arms."
                ),
            },
            "arms": [
                {
                    "id": "monolithic",
                    "routing": "all admitted pass rows",
                    "dictionaries": 1,
                    "features_per_dictionary": 6144,
                    "optimizer_steps_per_dictionary": 2000,
                    "total_trainable_parameters": matched_trainable_parameters,
                },
                {
                    "id": "pass_compartmentalized",
                    "routing": "one dictionary for each observable pass",
                    "dictionaries": 2,
                    "features_per_dictionary": 3072,
                    "optimizer_steps_per_dictionary": 1000,
                    "total_trainable_parameters": matched_trainable_parameters,
                },
                {
                    "id": "matched_random_partition",
                    "routing": "frozen balanced random split independent of pass",
                    "dictionaries": 2,
                    "features_per_dictionary": 3072,
                    "optimizer_steps_per_dictionary": 1000,
                    "total_trainable_parameters": matched_trainable_parameters,
                },
            ],
            "matched_across_arms": [
                "total_dictionary_features",
                "total_encoder_and_decoder_parameter_budget",
                "total_optimizer_steps",
                "batch_size",
                "optimizer",
                "learning_rate_schedule",
                "activation_rows_seen",
                "seeds",
            ],
            "seeds": [1707, 250829, 260829],
            "primary_endpoints": [
                "held_out_explained_variance_at_matched_l0",
                "cross_category_reconstruction_leakage",
                "feature_activation_category_selectivity",
                "seed_stability",
            ],
            "admission_rule": {
                "pass_compartmentalized_must_beat_monolithic": True,
                "pass_compartmentalized_must_beat_matched_random_partition": True,
                "comparison_uses_only_final_evaluation_role": True,
                "confidence_interval_for_primary_difference_must_exclude_zero": True,
                "no_l0_degradation_greater_than_fraction": 0.05,
                "failed_or_tied_comparison_action": "retain_monolithic_dictionary",
            },
        },
        "stage_gates": [
            {
                "id": "freeze_pack",
                "requires": [
                    "verified_model_revision",
                    "pairwise_family_disjoint_roles",
                    "content_addressed_examples",
                ],
            },
            {
                "id": "capture_categories",
                "requires": [
                    "explicit_user_compute_authorization",
                    "provider_independent_ttl_and_dollar_watchdog",
                    "durable_remote_termination_receipt",
                    "provider_instance_stop_verified_before_goal_pause_or_exit",
                ],
            },
            {
                "id": "admit_parent_categories",
                "requires": [
                    "held_out_category_separation",
                    "block_random_controls",
                    "cross_role_replication",
                ],
            },
            {
                "id": "admit_child_clusters",
                "requires": [
                    "parent_category_admitted",
                    "native_v2_cluster_receipt",
                    "independent_v2_reverification",
                ],
            },
            {
                "id": "admit_shared_sae_feature_strata",
                "requires": [
                    "feature_assignment_uses_training_role_only",
                    "held_out_pass_selectivity",
                    "exact_pair_flip_random_control",
                    "held_out_reconstruction_quality_per_pass",
                    "no_semantic_claim_from_pass_selectivity_alone",
                ],
            },
            {
                "id": "admit_activation_phase_category",
                "requires": [
                    "at_least_five_independent_training_example_blocks",
                    "held_out_prompt_vs_generation_separation",
                    "exact_example_block_label_flip_p_at_most_0_05",
                    "cross_split_direction_stability",
                    "fresh_untouched_final_holdout",
                    "no_child_cluster_without_parent_admission",
                ],
            },
            {
                "id": "screen_capacity_before_sae_training",
                "requires": [
                    "fresh_pack_capture_complete",
                    "equal_total_rank_linear_proxy",
                    "pass_routing_beats_monolithic_with_positive_held_out_interval",
                    "pass_routing_beats_repeated_random_partitions",
                ],
            },
            {
                "id": "train_sae_arms",
                "requires": [
                    "explicit_user_weight_mutation_authorization",
                    "all_predecessor_gates_pass",
                    "fresh_pack_linear_capacity_screen_pass",
                    "equal_budget_arm_receipt",
                ],
            },
        ],
        "current_authority": {
            "local_design_and_verification_authorized": True,
            "new_inference_authorized": False,
            "paid_compute_authorized": False,
            "sae_training_authorized": False,
            "model_or_adapter_weight_mutation_authorized": False,
            "execution_authorized": False,
            "codex_goal_pause_is_compute_termination": False,
        },
        "provenance": {
            "builder_path": Path(__file__).as_posix(),
            "builder_sha256": sha256_file(Path(__file__)),
            "recovery_vs_reconstruction": (
                "Source activation archives are recovered empirical artifacts; "
                "this experiment specification and its gates are newly constructed."
            ),
        },
    }
    return {**body, "spec_id": content_sha256(body)}


def verify_spec(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    spec_id = str(body.pop("spec_id", ""))
    if content_sha256(body) != spec_id:
        raise ValueError("activation-category SAE experiment spec address differs")
    reports = observed["evidence"]["reports"]
    by_role = {str(report["role"]): report for report in reports}
    region_report_paths = [
        Path(report["path"])
        for report in reports
        if str(report["role"]).startswith("q4_temporal_region_negative_control_")
    ]
    expected = build_spec(
        arithmetic_pass_report_path=Path(by_role["arithmetic_pass_replication"]["path"]),
        reasoning_pass_report_path=Path(
            by_role["broader_reasoning_pass_replication"]["path"]
        ),
        q4_region_report_paths=region_report_paths,
        arithmetic_linear_proxy_report_path=Path(
            by_role["arithmetic_pass_compartment_linear_proxy_negative"]["path"]
        ),
        reasoning_linear_proxy_report_path=Path(
            by_role["reasoning_pass_compartment_linear_proxy_negative"]["path"]
        ),
        arithmetic_rank_sweep_report_path=Path(
            by_role["arithmetic_pass_compartment_rank_sweep_negative"]["path"]
        ),
        reasoning_rank_sweep_report_path=Path(
            by_role["reasoning_pass_compartment_rank_sweep_negative"]["path"]
        ),
        q4_sae_selectivity_2048_report_path=Path(
            by_role["q4_2048_shared_sae_internal_pass_specialization"]["path"]
        ),
        q4_sae_selectivity_8192_report_path=Path(
            by_role["q4_8192_shared_sae_internal_pass_specialization"]["path"]
        ),
        q4_sae_selectivity_robustness_2048_report_path=Path(
            by_role["q4_2048_shared_sae_pass_selectivity_robustness"]["path"]
        ),
        q4_sae_selectivity_robustness_8192_report_path=Path(
            by_role["q4_8192_shared_sae_pass_selectivity_robustness"]["path"]
        ),
        pass_category_cross_pack_transfer_report_path=Path(
            by_role["bf16_pass_category_cross_pack_transfer"]["path"]
        ),
        pass_category_shared_axis_report_path=Path(
            by_role["bf16_pass_category_shared_axis"]["path"]
        ),
        pass_category_prompt_phase_transfer_report_path=Path(
            by_role["bf16_pass_category_prompt_phase_transfer"]["path"]
        ),
        activation_phase_category_report_path=Path(
            by_role["bf16_activation_phase_category_screen"]["path"]
        ),
        activation_phase_category_cross_validation_report_path=Path(
            by_role["bf16_activation_phase_category_cross_validation"]["path"]
        ),
        phase_compartment_linear_proxy_report_path=Path(
            by_role["bf16_activation_phase_compartment_linear_proxy_negative"]["path"]
        ),
        joint_pass_phase_capacity_report_path=Path(
            by_role["bf16_joint_pass_phase_capacity_proxy_negative"]["path"]
        ),
        prompt_frozen_sae_quality_report_path=Path(
            by_role["bf16_prompt_frozen_sae_quality_negative"]["path"]
        ),
        q4_sae_pass_feature_identity_report_path=Path(
            by_role["q4_generation_sae_cross_budget_pass_feature_identity"]["path"]
        ),
        q4_sae_pass_feature_replica_matching_v2_report_path=Path(
            by_role["q4_generation_sae_pass_feature_stratum_replication_v2"]["path"]
        ),
        layer21_seed_objective_stability_report_path=Path(
            by_role["layer21_sae_seed_objective_stability_negative_control"]["path"]
        ),
        layer21_category_subspace_stability_report_path=Path(
            by_role["layer21_sae_category_subspace_stability_negative_control"]["path"]
        ),
        q4_sae_pass_category_subspace_transfer_report_path=Path(
            by_role[
                "q4_generation_sae_cross_budget_pass_category_subspace_transfer"
            ]["path"]
        ),
        feature_replica_matching_v2_benchmark_path=Path(
            observed["evidence"]["feature_replica_matching_method"]["path"]
        ),
    )
    if expected != observed:
        raise ValueError("activation-category SAE experiment spec does not reproduce")
    return {
        "schema": "sch.vpd-activation-category-sae-experiment-spec-verification.v1",
        "status": "pass",
        "spec_id": spec_id,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arithmetic-pass-report", type=Path)
    parser.add_argument("--reasoning-pass-report", type=Path)
    parser.add_argument("--q4-region-report", type=Path, action="append")
    parser.add_argument("--arithmetic-linear-proxy-report", type=Path)
    parser.add_argument("--reasoning-linear-proxy-report", type=Path)
    parser.add_argument("--arithmetic-rank-sweep-report", type=Path)
    parser.add_argument("--reasoning-rank-sweep-report", type=Path)
    parser.add_argument("--q4-sae-selectivity-2048-report", type=Path)
    parser.add_argument("--q4-sae-selectivity-8192-report", type=Path)
    parser.add_argument("--q4-sae-selectivity-robustness-2048-report", type=Path)
    parser.add_argument("--q4-sae-selectivity-robustness-8192-report", type=Path)
    parser.add_argument("--pass-category-cross-pack-transfer-report", type=Path)
    parser.add_argument("--pass-category-shared-axis-report", type=Path)
    parser.add_argument("--pass-category-prompt-phase-transfer-report", type=Path)
    parser.add_argument("--activation-phase-category-report", type=Path)
    parser.add_argument("--activation-phase-category-cross-validation-report", type=Path)
    parser.add_argument("--phase-compartment-linear-proxy-report", type=Path)
    parser.add_argument("--joint-pass-phase-capacity-report", type=Path)
    parser.add_argument("--prompt-frozen-sae-quality-report", type=Path)
    parser.add_argument("--q4-sae-pass-feature-identity-report", type=Path)
    parser.add_argument("--q4-sae-pass-feature-replica-matching-v2-report", type=Path)
    parser.add_argument("--layer21-seed-objective-stability-report", type=Path)
    parser.add_argument("--layer21-category-subspace-stability-report", type=Path)
    parser.add_argument("--q4-sae-pass-category-subspace-transfer-report", type=Path)
    parser.add_argument("--feature-replica-matching-v2-benchmark", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_spec(args.verify)
    else:
        required = (
            args.arithmetic_pass_report,
            args.reasoning_pass_report,
            args.arithmetic_linear_proxy_report,
            args.reasoning_linear_proxy_report,
            args.arithmetic_rank_sweep_report,
            args.reasoning_rank_sweep_report,
            args.q4_sae_selectivity_2048_report,
            args.q4_sae_selectivity_8192_report,
            args.q4_sae_selectivity_robustness_2048_report,
            args.q4_sae_selectivity_robustness_8192_report,
            args.pass_category_cross_pack_transfer_report,
            args.pass_category_shared_axis_report,
            args.pass_category_prompt_phase_transfer_report,
            args.activation_phase_category_report,
            args.activation_phase_category_cross_validation_report,
            args.phase_compartment_linear_proxy_report,
            args.joint_pass_phase_capacity_report,
            args.prompt_frozen_sae_quality_report,
            args.q4_sae_pass_feature_identity_report,
            args.q4_sae_pass_feature_replica_matching_v2_report,
            args.layer21_seed_objective_stability_report,
            args.layer21_category_subspace_stability_report,
            args.q4_sae_pass_category_subspace_transfer_report,
            args.feature_replica_matching_v2_benchmark,
            args.output,
        )
        if any(value is None for value in required) or not args.q4_region_report:
            parser.error("all report paths and --output are required unless verifying")
        result = build_spec(
            arithmetic_pass_report_path=args.arithmetic_pass_report,
            reasoning_pass_report_path=args.reasoning_pass_report,
            q4_region_report_paths=args.q4_region_report,
            arithmetic_linear_proxy_report_path=args.arithmetic_linear_proxy_report,
            reasoning_linear_proxy_report_path=args.reasoning_linear_proxy_report,
            arithmetic_rank_sweep_report_path=args.arithmetic_rank_sweep_report,
            reasoning_rank_sweep_report_path=args.reasoning_rank_sweep_report,
            q4_sae_selectivity_2048_report_path=args.q4_sae_selectivity_2048_report,
            q4_sae_selectivity_8192_report_path=args.q4_sae_selectivity_8192_report,
            q4_sae_selectivity_robustness_2048_report_path=(
                args.q4_sae_selectivity_robustness_2048_report
            ),
            q4_sae_selectivity_robustness_8192_report_path=(
                args.q4_sae_selectivity_robustness_8192_report
            ),
            pass_category_cross_pack_transfer_report_path=(
                args.pass_category_cross_pack_transfer_report
            ),
            pass_category_shared_axis_report_path=(
                args.pass_category_shared_axis_report
            ),
            pass_category_prompt_phase_transfer_report_path=(
                args.pass_category_prompt_phase_transfer_report
            ),
            activation_phase_category_report_path=args.activation_phase_category_report,
            activation_phase_category_cross_validation_report_path=(
                args.activation_phase_category_cross_validation_report
            ),
            phase_compartment_linear_proxy_report_path=(
                args.phase_compartment_linear_proxy_report
            ),
            joint_pass_phase_capacity_report_path=(
                args.joint_pass_phase_capacity_report
            ),
            prompt_frozen_sae_quality_report_path=(
                args.prompt_frozen_sae_quality_report
            ),
            q4_sae_pass_feature_identity_report_path=(
                args.q4_sae_pass_feature_identity_report
            ),
            q4_sae_pass_feature_replica_matching_v2_report_path=(
                args.q4_sae_pass_feature_replica_matching_v2_report
            ),
            layer21_seed_objective_stability_report_path=(
                args.layer21_seed_objective_stability_report
            ),
            layer21_category_subspace_stability_report_path=(
                args.layer21_category_subspace_stability_report
            ),
            q4_sae_pass_category_subspace_transfer_report_path=(
                args.q4_sae_pass_category_subspace_transfer_report
            ),
            feature_replica_matching_v2_benchmark_path=(
                args.feature_replica_matching_v2_benchmark
            ),
        )
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
