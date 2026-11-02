from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.build_activation_category_sae_experiment_spec_v1 import (
    verify_spec,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v52 import (
    canonical_inputs as protocol_v52_inputs,
    verify_protocol as verify_protocol_v52,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v43 import (
    canonical_inputs as runbook_v43_inputs,
    verify_runbook as verify_runbook_v43,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v44"
EXPECTED_V43_SHA256 = (
    "ab48bd68b00011acbabc7421a29d2845c23b394bf8f9adc9d2784103f38a6a05"
)
EXPECTED_V43_RUNBOOK_ID = (
    "870a7cbd0622e804cb6c91ad675160de3acc348df4f99256b576023b99017600"
)
EXPECTED_V52_SHA256 = (
    "6c381510ae4b24a362e424c52eec3655860d7879979c98e8ed05f1e4676c7fc6"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V44 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V43_SHA256:
        raise ValueError("runbook V43 differs from its pinned SHA-256")
    observed = _load(path)
    if observed.get("runbook_id") != EXPECTED_V43_RUNBOOK_ID:
        raise ValueError("runbook V43 differs from its pinned content ID")
    verification = verify_runbook_v43(path, **runbook_v43_inputs(project_root))
    if verification.get("status") != "pass":
        raise ValueError("runbook V43 did not reproduce")
    return verification


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v52_path: Path,
    sae_spec_path: Path,
    lineage_test_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path, project_root)
    if sha256_file(protocol_v52_path) != EXPECTED_V52_SHA256:
        raise ValueError("protocol V52 differs from its pinned SHA-256")
    protocol_verification = verify_protocol_v52(
        protocol_v52_path, **protocol_v52_inputs(project_root)
    )
    spec_verification = verify_spec(sae_spec_path)
    predecessor = _load(predecessor_path)
    spec = _load(sae_spec_path)
    protocol = _load(protocol_v52_path)
    component = protocol["component_compartments"][
        "nanbeige_empirical_activation_category_sae_experiment_v1"
    ]
    summary = component["empirical_summary"]

    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "empirical_pass_category_and_temporal_region_controls_verified_equal_rank_"
        "capacity_screen_negative_local_preflight_safe_fresh_capture_and_training_"
        "not_authorized_shared_sae_internal_feature_strata_preferred"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v52"] = {
        "path": _relative(protocol_v52_path, project_root),
        "sha256": sha256_file(protocol_v52_path),
        "verification": protocol_verification,
    }
    result["source"]["nanbeige_activation_category_sae_spec_v1"] = {
        "path": _relative(sae_spec_path, project_root),
        "sha256": sha256_file(sae_spec_path),
        "spec_id": spec["spec_id"],
        "verification": spec_verification,
        "bound_empirical_reports": spec["evidence"]["reports"],
    }
    result["source"]["empirical_activation_category_sae_lineage_test_v1"] = {
        "path": _relative(lineage_test_path, project_root),
        "sha256": sha256_file(lineage_test_path),
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))

    current = result["current_result"]
    current["nanbeige_empirical_activation_category_sae_experiment_v1"] = (
        "pass_index_admitted_6_of_6_site_checks_temporal_region_rejected_"
        "0_of_12_site_pass_checks_no_child_clusters_admitted"
    )
    current["empirical_pass_category_archives"] = summary[
        "pass_replication_archives"
    ]
    current["empirical_pass_category_site_admissions"] = summary[
        "pass_site_admissions"
    ]
    current["empirical_temporal_region_negative_control_archives"] = summary[
        "q4_region_negative_control_archives"
    ]
    current["empirical_temporal_region_site_admissions"] = summary[
        "q4_region_site_pass_admissions"
    ]
    current["empirical_within_pass_nontrivial_allocations"] = summary[
        "within_pass_nontrivial_allocations"
    ]
    current["empirical_admissible_region_child_allocations"] = summary[
        "admissible_region_child_allocations"
    ]
    current["empirical_linear_proxy_site_checks"] = summary[
        "linear_proxy_site_checks"
    ]
    current["empirical_linear_proxy_pass_wins_vs_random"] = summary[
        "linear_proxy_pass_wins_vs_random"
    ]
    current["empirical_linear_proxy_pass_wins_vs_monolithic"] = summary[
        "linear_proxy_pass_wins_vs_monolithic"
    ]
    current["empirical_linear_proxy_training_admissions"] = summary[
        "linear_proxy_training_admissions"
    ]
    current["empirical_rank_sweep_site_rank_checks"] = summary[
        "rank_sweep_site_rank_checks"
    ]
    current["empirical_rank_sweep_pass_wins_vs_monolithic"] = summary[
        "rank_sweep_pass_wins_vs_monolithic"
    ]
    current["empirical_rank_sweep_monolithic_wins_vs_pass"] = summary[
        "rank_sweep_monolithic_wins_vs_pass"
    ]
    current["empirical_rank_sweep_monolithic_vs_pass_inconclusive"] = summary[
        "rank_sweep_monolithic_vs_pass_inconclusive"
    ]
    current["empirical_rank_sweep_training_admissions"] = summary[
        "rank_sweep_training_admissions"
    ]
    current["shared_sae_internal_feature_strata_admissions"] = summary[
        "shared_sae_internal_feature_strata_admissions"
    ]
    current["shared_sae_quality_qualified_sites"] = summary[
        "shared_sae_admitted_sites_in_both_archives"
    ]
    current["shared_sae_high_confidence_sites_through_ev_0_25"] = summary[
        "shared_sae_high_confidence_sites_through_ev_0_25"
    ]
    current["shared_sae_provisional_sites_at_preregistered_ev_0_20"] = summary[
        "shared_sae_provisional_sites_at_preregistered_ev_0_20"
    ]
    current["shared_sae_quality_rejected_sites"] = summary[
        "shared_sae_quality_rejected_sites"
    ]
    current["cross_pack_transfer_bidirectional_admissions"] = summary[
        "cross_pack_transfer_bidirectional_admissions"
    ]
    current["cross_pack_transfer_admitted_sites"] = summary[
        "cross_pack_transfer_admitted_sites"
    ]
    current[
        "sites_meeting_high_confidence_sae_quality_and_direct_pack_direction_replication_gates"
    ] = summary[
        "sites_meeting_high_confidence_sae_quality_and_direct_pack_direction_replication_gates"
    ]
    current["shared_axis_admissions"] = summary["shared_axis_admissions"]
    current[
        "sites_meeting_high_confidence_sae_quality_and_shared_axis_gates"
    ] = summary[
        "sites_meeting_high_confidence_sae_quality_and_shared_axis_gates"
    ]
    current["prompt_phase_transfer_admissions"] = summary[
        "prompt_phase_transfer_admissions"
    ]
    current["prompt_phase_transfer_admitted_sites"] = summary[
        "prompt_phase_transfer_admitted_sites"
    ]
    current["activation_phase_category_admissions"] = summary[
        "activation_phase_category_admissions"
    ]
    current["activation_phase_minimum_training_blocks_for_p_0_05"] = summary[
        "activation_phase_minimum_training_blocks_for_p_0_05"
    ]
    current["activation_phase_provisional_cross_validated_supports"] = summary[
        "activation_phase_provisional_cross_validated_supports"
    ]
    current["activation_phase_cross_validated_operative_admissions"] = summary[
        "activation_phase_cross_validated_operative_admissions"
    ]
    current["activation_phase_capacity_site_checks"] = summary[
        "activation_phase_capacity_site_checks"
    ]
    current["activation_phase_capacity_wins_vs_monolithic"] = summary[
        "activation_phase_capacity_wins_vs_monolithic"
    ]
    current["activation_phase_capacity_wins_vs_random_mean"] = summary[
        "activation_phase_capacity_wins_vs_random_mean"
    ]
    current["activation_phase_capacity_preferences"] = summary[
        "activation_phase_capacity_preferences"
    ]
    current["activation_phase_capacity_training_admissions"] = summary[
        "activation_phase_capacity_training_admissions"
    ]
    current["joint_pass_phase_capacity_site_checks"] = summary[
        "joint_pass_phase_capacity_site_checks"
    ]
    current["joint_pass_phase_wins_vs_monolithic"] = summary[
        "joint_pass_phase_wins_vs_monolithic"
    ]
    current["joint_pass_phase_wins_vs_pass_routed"] = summary[
        "joint_pass_phase_wins_vs_pass_routed"
    ]
    current["joint_pass_phase_wins_vs_phase_routed"] = summary[
        "joint_pass_phase_wins_vs_phase_routed"
    ]
    current["joint_pass_phase_wins_vs_random_mean"] = summary[
        "joint_pass_phase_wins_vs_random_mean"
    ]
    current["joint_pass_phase_capacity_preferences"] = summary[
        "joint_pass_phase_capacity_preferences"
    ]
    current["joint_pass_phase_training_admissions"] = summary[
        "joint_pass_phase_training_admissions"
    ]
    current["prompt_frozen_sae_quality_site_checks"] = summary[
        "prompt_frozen_sae_quality_site_checks"
    ]
    current["prompt_frozen_sae_quality_admissions"] = summary[
        "prompt_frozen_sae_quality_admissions"
    ]
    current["prompt_phase_pass_feature_strata_admissions"] = summary[
        "prompt_phase_pass_feature_strata_admissions"
    ]
    current["generation_sae_feature_identity_site_checks"] = summary[
        "generation_sae_feature_identity_site_checks"
    ]
    current[
        "generation_sae_provisional_same_seed_feature_identity_supports"
    ] = summary["generation_sae_provisional_same_seed_feature_identity_supports"]
    current[
        "generation_sae_provisional_same_seed_feature_identity_sites"
    ] = summary["generation_sae_provisional_same_seed_feature_identity_sites"]
    current[
        "generation_sae_operative_cross_budget_feature_identity_admissions"
    ] = summary["generation_sae_operative_cross_budget_feature_identity_admissions"]
    current["feature_replica_matching_version"] = summary[
        "feature_replica_matching_version"
    ]
    current["feature_replica_matching_synthetic_scenarios"] = summary[
        "feature_replica_matching_synthetic_scenarios"
    ]
    current["feature_replica_matching_expected_outcomes"] = summary[
        "feature_replica_matching_expected_outcomes"
    ]
    current["generation_sae_v2_stratum_replication_site_checks"] = summary[
        "generation_sae_v2_stratum_replication_site_checks"
    ]
    current["generation_sae_v2_provisional_same_seed_supports"] = summary[
        "generation_sae_v2_provisional_same_seed_supports"
    ]
    current["generation_sae_v2_provisional_same_seed_sites"] = summary[
        "generation_sae_v2_provisional_same_seed_sites"
    ]
    current["generation_sae_v2_operative_fresh_seed_admissions"] = summary[
        "generation_sae_v2_operative_fresh_seed_admissions"
    ]
    current["layer21_seed_objective_quality_qualified_conditions"] = summary[
        "layer21_seed_objective_quality_qualified_conditions"
    ]
    current["layer21_cross_seed_pair_checks"] = summary[
        "layer21_cross_seed_pair_checks"
    ]
    current["layer21_cross_seed_diagnostic_supports_without_quality_gate"] = summary[
        "layer21_cross_seed_diagnostic_supports_without_quality_gate"
    ]
    current["layer21_same_seed_cross_objective_pair_checks"] = summary[
        "layer21_same_seed_cross_objective_pair_checks"
    ]
    current[
        "layer21_same_seed_cross_objective_diagnostic_supports_without_quality_gate"
    ] = summary[
        "layer21_same_seed_cross_objective_diagnostic_supports_without_quality_gate"
    ]
    current["layer21_seed_objective_operative_admissions"] = summary[
        "layer21_seed_objective_operative_admissions"
    ]
    current["layer21_category_subspace_pair_checks"] = summary[
        "layer21_category_subspace_pair_checks"
    ]
    current["layer21_cross_seed_subspace_pair_checks"] = summary[
        "layer21_cross_seed_subspace_pair_checks"
    ]
    current["layer21_cross_seed_familywise_corrected_subspace_supports"] = summary[
        "layer21_cross_seed_familywise_corrected_subspace_supports"
    ]
    current["layer21_same_seed_cross_objective_subspace_pair_checks"] = summary[
        "layer21_same_seed_cross_objective_subspace_pair_checks"
    ]
    current["layer21_same_seed_cross_objective_subspace_supports"] = summary[
        "layer21_same_seed_cross_objective_subspace_supports"
    ]
    current["layer21_category_subspace_operative_admissions"] = summary[
        "layer21_category_subspace_operative_admissions"
    ]
    current["generation_sae_category_subspace_site_checks"] = summary[
        "generation_sae_category_subspace_site_checks"
    ]
    current[
        "generation_sae_category_subspace_same_seed_diagnostic_supports"
    ] = summary[
        "generation_sae_category_subspace_same_seed_diagnostic_supports"
    ]
    current[
        "generation_sae_category_subspace_same_seed_quality_supports"
    ] = summary[
        "generation_sae_category_subspace_same_seed_quality_supports"
    ]
    current[
        "generation_sae_category_subspace_operative_fresh_seed_admissions"
    ] = summary[
        "generation_sae_category_subspace_operative_fresh_seed_admissions"
    ]
    current["preferred_sae_architecture"] = component[
        "current_preferred_sae_architecture"
    ]
    current["nanbeige_sae_comparison_arms"] = len(spec["sae_comparison"]["arms"])
    current["nanbeige_fresh_pack_families"] = spec["fresh_pack"]["total_families"]
    current["nanbeige_fresh_pack_examples"] = spec["fresh_pack"]["total_examples"]
    current["nanbeige_fresh_pack_present"] = False
    current["training_performed"] = False
    current["persistent_dictionary_output_created"] = False

    preflight = result["stages"][0]
    preflight["commands"].extend(
        [
            {
                "id": "verify_nanbeige_arithmetic_pass_category_bridge_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_activation_category_bridge_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_activation_category_bridge_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_reasoning_pass_category_bridge_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_activation_category_bridge_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_reasoning_activation_category_bridge_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_q4_region_category_bridge_2048_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_q4_region_category_bridge_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_q4_region_category_bridge_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_q4_region_category_bridge_8192_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_q4_region_category_bridge_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_q4_region_category_bridge_8192_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_arithmetic_pass_compartment_linear_proxy_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_pass_compartment_linear_proxy_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_arithmetic_pass_compartment_linear_proxy_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_reasoning_pass_compartment_linear_proxy_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_pass_compartment_linear_proxy_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_pass_compartment_linear_proxy_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_arithmetic_pass_compartment_rank_sweep_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_pass_compartment_rank_sweep_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_arithmetic_pass_compartment_rank_sweep_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_reasoning_pass_compartment_rank_sweep_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_pass_compartment_rank_sweep_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_reasoning_pass_compartment_rank_sweep_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_q4_sae_pass_selectivity_2048_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_selectivity_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_q4_sae_pass_selectivity_2048_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_q4_sae_pass_selectivity_8192_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_selectivity_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_q4_sae_pass_selectivity_8192_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_q4_sae_pass_selectivity_robustness_2048_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_selectivity_robustness_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_q4_sae_pass_selectivity_robustness_2048_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_q4_sae_pass_selectivity_robustness_8192_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_selectivity_robustness_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_q4_sae_pass_selectivity_robustness_8192_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_pass_category_cross_pack_transfer_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_pass_category_cross_pack_transfer_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_pass_category_cross_pack_transfer_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_pass_category_shared_axis_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_pass_category_shared_axis_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_pass_category_shared_axis_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_pass_category_prompt_phase_transfer_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_pass_category_prompt_phase_transfer_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_pass_category_prompt_phase_transfer_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_activation_phase_category_bridge_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_activation_phase_category_bridge_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_activation_phase_category_bridge_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_activation_phase_category_cross_validation_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_activation_phase_category_cross_validation_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_activation_phase_category_cross_validation_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_phase_compartment_linear_proxy_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_phase_compartment_linear_proxy_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_phase_compartment_linear_proxy_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_joint_pass_phase_capacity_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_joint_pass_phase_capacity_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_joint_pass_phase_capacity_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_prompt_frozen_sae_quality_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.audit_nanbeige_prompt_frozen_sae_quality_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_prompt_frozen_sae_quality_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_q4_sae_pass_feature_identity_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_feature_identity_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_q4_sae_pass_feature_identity_v1/report.json",
                ],
            },
            {
                "id": "verify_sae_feature_replica_matching_v2",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.evaluate_sae_feature_replica_matching_v2",
                    "--verify",
                    "artifacts/vpd_sae_feature_replica_matching_benchmark_v2/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_q4_sae_pass_feature_stratum_replication_v2",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_feature_replica_matching_v2",
                    "--verify",
                    "artifacts/vpd_nanbeige_q4_sae_pass_feature_replica_matching_v2/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_layer21_sae_seed_objective_stability_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_layer21_sae_seed_objective_stability_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_layer21_sae_seed_objective_stability_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_layer21_sae_category_subspace_stability_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_layer21_sae_category_subspace_stability_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_layer21_sae_category_subspace_stability_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_q4_sae_pass_category_subspace_transfer_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_category_subspace_transfer_v1",
                    "--verify",
                    "artifacts/vpd_nanbeige_q4_sae_pass_category_subspace_transfer_v1/report.json",
                ],
            },
            {
                "id": "verify_nanbeige_activation_category_sae_spec_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.build_activation_category_sae_experiment_spec_v1",
                    "--verify",
                    "artifacts/vpd_activation_category_sae_experiment_spec_v1/spec.json",
                ],
            },
            {
                "id": "verify_research_protocol_v52",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v52",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v52.json",
                ],
            },
        ]
    )
    preflight["expected"]["nanbeige_empirical_activation_category_sae_v1"] = (
        "two recovered BF16 archives admit recurrent pass at all six site checks; "
        "two Q4 archives reject temporal region at all twelve site/pass checks; "
        "no child cluster is operative; pass routing beats random but loses to "
        "monolithic in all six equal-rank screens; equal-budget three-arm SAE design "
        "verifies; a six-rank sweep gives zero pass wins and zero admissions across "
        "36 archive/site/rank checks, so training remains deprioritized"
        "; two independent shared SAEs admit internal pass-feature strata at layer "
        "11 and final norm while rejecting layer 21 on reconstruction quality; "
        "the sensitivity sweep makes layer 11 high-confidence through EV 0.25, "
        "final norm provisional at the preregistered EV 0.20 threshold, and layer "
        "21 quality-rejected; cross-pack transfer admits stable directions at layer "
        "21 and norm but not layer 11, leaving no site that clears both the strongest "
        "SAE-quality and direct direction-replication gates; an equal-pack shared-axis "
        "test nevertheless admits all three sites and identifies layer 11 as the sole "
        "site also clearing the strongest SAE-quality gate, pending third-domain transfer; "
        "generation-axis transfer to prompt processing admits only final norm, while "
        "prompt versus generation is a strong but exact-test-underpowered candidate category"
        "; family-blocked cross-validation supports phase at all three sites but remains "
        "inoperative because it reuses every example and lacks a final untouched holdout"
        "; fixed-total-rank phase routing beats matched random partitions at all three "
        "sites but loses to the monolithic subspace at all three, so shared monolithic "
        "capacity remains preferred"
        "; joint pass-by-phase routing also beats its four-way random mean at all "
        "three sites but loses to monolithic, pass-only, and phase-only routing at "
        "all three, rejecting Cartesian dictionary multiplication"
        "; prompt frozen-SAE archives exist but all three sites fail the held-out "
        "EV 0.20 interpretation gate, so cross-phase feature strata are rejected"
        "; cross-budget generation SAE decoder identity is provisional only at final "
        "norm and has zero operative admissions because both fits share seed and IDs"
        "; the V2 matcher recovers a clean noisy permutation and rejects decoder "
        "sign flips, duplicate collapse, and category-profile swaps using signed, "
        "mutual, one-to-one matches while leaving V1 historical receipts unchanged"
        "; applying V2 to training-assigned pass strata leaves final norm as the "
        "sole provisional same-seed site with 25 pass-0 and 28 pass-1 matches, "
        "while making no full-dictionary or fresh-seed replication claim"
        "; the layer-21 seed/objective control has zero matches in all six "
        "cross-seed same-objective pairs but diagnostic stability in all three "
        "same-seed cross-objective pairs, and all six conditions fail quality"
        "; rank-16 category subspaces likewise have zero familywise-corrected "
        "cross-seed supports and three fixed-seed cross-objective supports"
        "; across the two same-seed Q4 budgets, rank-16 pass-category subspaces "
        "transfer at all three sites after site correction, with quality-qualified "
        "provisional support at layer 11 and final norm but zero operative fresh-seed admissions"
    )
    preflight["stop_if"].extend(
        [
            "any empirical activation archive or report hash differs",
            "either pass-category replication loses a site admission",
            "any temporal-region parent category becomes admitted without a new version",
            "any screening-only child cluster is treated as operative",
            "the SAE arms cease to have exactly matched trainable parameter budgets",
            "a negative equal-rank capacity screen is treated as SAE training admission",
            "a paused Codex goal is treated as evidence that a provider pod stopped",
            "within-pack layer-11 pass geometry is treated as a universal cross-family direction",
            "the four-block activation-phase screen is treated as a category admission",
            "family-blocked provisional phase support is treated as an operative admission",
            "a phase capacity preference is claimed despite zero wins over monolithic",
            "the negative phase-capacity screen is treated as a training admission",
            "joint pass-by-phase signal relative to random is treated as capacity preference",
            "Cartesian pass-by-phase dictionaries are introduced despite zero wins over simpler arms",
            "prompt frozen feature archives are interpreted despite zero reconstruction-quality admissions",
            "hidden-state category admission is treated as sufficient SAE feature-quality evidence",
            "same-seed same-ID decoder alignment is treated as independent replication",
            "decoder identity is treated as semantic interpretation or training authority",
            "fresh-seed feature identity uses absolute decoder cosine",
            "non-mutual or collision-permitting feature matches are accepted",
            "the V2 synthetic replica-matching benchmark is not verified",
            "selective-stratum replication is reported as full-dictionary replication",
            "same-seed V2 stratum support is treated as fresh-seed replication",
            "a loss-objective change at fixed seed is treated as independent replication",
            "layer-21 seed controls are used to bypass reconstruction failure",
            "individual-feature instability is assumed to settle subspace stability without a direct test",
            "a nominal category-subspace hit is admitted without multiplicity control",
            "same-seed category-subspace transfer is treated as independent replication",
            "a rank-sensitivity result is used as an alternate admission test",
        ]
    )
    paid_gate = result["stages"][1]
    paid_gate["implementation_status"] = (
        "recovered empirical evidence supports pass index only as the candidate "
        "router and the exact equal-budget SAE comparison is sealed; the fresh "
        "20-family pack is absent, and capture, training, inference, and evaluation "
        "still require new explicit authorization plus independent TTL/dollar controls"
    )
    paid_gate["stop_if"] = list(paid_gate["stop_if"]) + [
        "the fresh Nanbeige pack is not content-addressed before inference",
        "model revision is not independently verified before capture",
        "pass-compartment training starts before the parent category re-admits on the fresh pack",
        "temporal-region routing or finer pass clusters are silently introduced",
        "provider-side pod termination is not independently verified before goal pause or exit",
    ]

    completion = result["completion_definition"]
    completion["nanbeige_empirical_activation_category_sae_experiment_v1_verified"] = True
    completion["empirical_pass_category_site_admissions"] = summary[
        "pass_site_admissions"
    ]
    completion["empirical_temporal_region_site_admissions"] = summary[
        "q4_region_site_pass_admissions"
    ]
    completion["empirical_child_clusters_admitted"] = False
    completion["empirical_linear_proxy_training_admissions"] = 0
    completion["empirical_rank_sweep_training_admissions"] = 0
    completion["shared_sae_internal_feature_strata_admissions"] = 4
    completion["shared_sae_quality_qualified_sites"] = [
        "model.layers.11",
        "model.norm",
    ]
    completion["shared_sae_high_confidence_sites_through_ev_0_25"] = [
        "model.layers.11"
    ]
    completion["shared_sae_provisional_sites_at_preregistered_ev_0_20"] = [
        "model.norm"
    ]
    completion["cross_pack_transfer_bidirectional_admissions"] = 2
    completion["cross_pack_transfer_admitted_sites"] = [
        "model.layers.21",
        "model.norm",
    ]
    completion[
        "sites_meeting_high_confidence_sae_quality_and_direct_pack_direction_replication_gates"
    ] = []
    completion["shared_axis_admissions"] = 3
    completion[
        "sites_meeting_high_confidence_sae_quality_and_shared_axis_gates"
    ] = ["model.layers.11"]
    completion["prompt_phase_transfer_admissions"] = 1
    completion["prompt_phase_transfer_admitted_sites"] = ["model.norm"]
    completion["activation_phase_category_admissions"] = 0
    completion["activation_phase_minimum_training_blocks_for_p_0_05"] = 5
    completion["activation_phase_operative_child_clusters"] = 0
    completion["activation_phase_provisional_cross_validated_supports"] = 3
    completion["activation_phase_cross_validated_operative_admissions"] = 0
    completion["activation_phase_capacity_site_checks"] = 3
    completion["activation_phase_capacity_wins_vs_monolithic"] = 0
    completion["activation_phase_capacity_wins_vs_random_mean"] = 3
    completion["activation_phase_capacity_preferences"] = 0
    completion["activation_phase_capacity_training_admissions"] = 0
    completion["joint_pass_phase_capacity_site_checks"] = 3
    completion["joint_pass_phase_wins_vs_monolithic"] = 0
    completion["joint_pass_phase_wins_vs_pass_routed"] = 0
    completion["joint_pass_phase_wins_vs_phase_routed"] = 0
    completion["joint_pass_phase_wins_vs_random_mean"] = 3
    completion["joint_pass_phase_capacity_preferences"] = 0
    completion["joint_pass_phase_training_admissions"] = 0
    completion["prompt_frozen_sae_quality_site_checks"] = 3
    completion["prompt_frozen_sae_quality_admissions"] = 0
    completion["prompt_phase_pass_feature_strata_admissions"] = 0
    completion["generation_sae_feature_identity_site_checks"] = 3
    completion["generation_sae_provisional_same_seed_feature_identity_supports"] = 1
    completion["generation_sae_provisional_same_seed_feature_identity_sites"] = [
        "model.norm"
    ]
    completion[
        "generation_sae_operative_cross_budget_feature_identity_admissions"
    ] = 0
    completion["feature_replica_matching_version"] = "v2"
    completion["feature_replica_matching_synthetic_scenarios"] = 4
    completion["feature_replica_matching_expected_outcomes"] = 4
    completion["generation_sae_v2_stratum_replication_site_checks"] = 3
    completion["generation_sae_v2_provisional_same_seed_supports"] = 1
    completion["generation_sae_v2_provisional_same_seed_sites"] = ["model.norm"]
    completion["generation_sae_v2_operative_fresh_seed_admissions"] = 0
    completion["layer21_seed_objective_quality_qualified_conditions"] = 0
    completion["layer21_cross_seed_pair_checks"] = 6
    completion["layer21_cross_seed_diagnostic_supports_without_quality_gate"] = 0
    completion["layer21_same_seed_cross_objective_pair_checks"] = 3
    completion[
        "layer21_same_seed_cross_objective_diagnostic_supports_without_quality_gate"
    ] = 3
    completion["layer21_seed_objective_operative_admissions"] = 0
    completion["layer21_category_subspace_pair_checks"] = 9
    completion["layer21_cross_seed_subspace_pair_checks"] = 6
    completion["layer21_cross_seed_familywise_corrected_subspace_supports"] = 0
    completion["layer21_same_seed_cross_objective_subspace_pair_checks"] = 3
    completion["layer21_same_seed_cross_objective_subspace_supports"] = 3
    completion["layer21_category_subspace_operative_admissions"] = 0
    completion["generation_sae_category_subspace_site_checks"] = 3
    completion[
        "generation_sae_category_subspace_same_seed_diagnostic_supports"
    ] = 3
    completion[
        "generation_sae_category_subspace_same_seed_quality_supports"
    ] = 2
    completion["generation_sae_category_subspace_same_seed_quality_sites"] = [
        "model.layers.11",
        "model.norm",
    ]
    completion[
        "generation_sae_category_subspace_operative_fresh_seed_admissions"
    ] = 0
    completion["activation_phase_fresh_untouched_holdout_present"] = False
    completion["nanbeige_fresh_pack_present"] = False
    completion["nanbeige_capture_authorized"] = False
    completion["nanbeige_sae_training_authorized"] = False
    completion["training_performed"] = False
    completion["persistent_dictionary_output_created"] = False

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
    boundary["codex_goal_pause_is_compute_termination"] = False
    boundary["provider_stop_receipt_required_before_goal_pause_or_exit"] = True
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
    boundary["nanbeige_fresh_pack_present"] = False
    boundary["new_model_inference_performed"] = False
    boundary["training_performed"] = False
    boundary["persistent_dictionary_output_created"] = False
    boundary["model_framework_imported"] = False
    boundary["model_or_adapter_weights_opened"] = False
    boundary["weights_loaded_or_mutated"] = False
    boundary["execution_authorized"] = False
    boundary["sae_training_authorized"] = False
    boundary["paid_compute_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    expected = build_runbook(**kwargs)
    if identifier != content_sha256(body) or observed != expected:
        raise ValueError("cheap-executor runbook V44 is not reproducible")
    component = _load(kwargs["protocol_v52_path"])["component_compartments"][
        "nanbeige_empirical_activation_category_sae_experiment_v1"
    ]
    summary = component["empirical_summary"]
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v44",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        **summary,
        "fresh_pack_present": False,
        "capture_authorized": False,
        "training_performed": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root
        / "artifacts/vpd_cheap_executor_runbook_v43/runbook.json",
        "protocol_v52_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v52.json",
        "sae_spec_path": project_root
        / "artifacts/vpd_activation_category_sae_experiment_spec_v1/spec.json",
        "lineage_test_path": project_root
        / "tests/test_vpd_empirical_activation_category_sae_lineage_v1.py",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V44.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.output is not None:
        result = build_runbook(**inputs)
        write_json(args.output, result)
    else:
        result = verify_runbook(args.verify, **inputs)
    print(
        json.dumps(
            {key: result[key] for key in ("schema", "status", "runbook_id")},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
