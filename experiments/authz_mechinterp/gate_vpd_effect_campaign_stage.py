from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    calibrate_activation_compartment_thresholds,
    category_profile_stratified_robustness,
    classify_activation_compartments,
    content_sha256,
    cross_archive_category_direction_cosines,
    load_effect_merge_split_provenance,
    load_effects,
    load_hierarchy_plan,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.effect_uncertainty import (
    leave_one_block_out_strengths,
)
from experiments.authz_mechinterp.merge_vpd_effect_shards import verify_effect_merge


SCHEMA = "sch.vpd-effect-campaign-stage-gate.v1"


def _policy(protocol: dict[str, Any]) -> dict[str, Any]:
    compartments = protocol["component_compartments"]
    threshold = compartments["threshold_calibration"]
    pair_block = compartments[
        "within_category_pair_block_direction_robustness_screen"
    ]
    condition = compartments["within_category_condition_robustness_screen"]
    template = compartments["within_category_template_robustness_screen"]
    uncertainty = compartments["within_split_uncertainty_screen"]
    return {
        "candidate_grid": threshold["candidate_grid"],
        "minimum_replication_rate": float(threshold["minimum_replication_rate"]),
        "minimum_stable_specific_components": int(
            threshold["minimum_stable_specific_components"]
        ),
        "minimum_episodes_per_category_per_archive": int(
            threshold["minimum_episodes_per_category_per_archive"]
        ),
        "minimum_signed_direction_cosine": float(
            threshold["minimum_signed_direction_cosine"]
        ),
        "minimum_dominant_strength_ratio": float(
            threshold["minimum_dominant_strength_ratio"]
        ),
        "minimum_jackknife_stability": float(
            uncertainty["minimum_same_compartment_stability"]
        ),
        "minimum_pair_block_coverage": float(
            pair_block["minimum_pair_block_coverage"]
        ),
        "minimum_pair_block_direction_cosine": float(
            pair_block["minimum_pair_block_direction_cosine"]
        ),
        "minimum_pair_block_relative_norm": float(
            pair_block["minimum_pair_block_relative_norm"]
        ),
        "minimum_condition_coverage": float(condition["minimum_condition_coverage"]),
        "minimum_condition_direction_cosine": float(
            condition["minimum_condition_direction_cosine"]
        ),
        "minimum_condition_relative_norm": float(
            condition["minimum_condition_relative_norm"]
        ),
        "minimum_template_coverage": float(
            template["minimum_template_group_coverage"]
        ),
        "minimum_template_direction_cosine": float(
            template["minimum_template_group_direction_cosine"]
        ),
        "minimum_template_relative_norm": float(
            template["minimum_template_group_relative_norm"]
        ),
        "minimum_sketch_dimension": int(template["minimum_countsketch_dimension"]),
    }


def _robustness(profile: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    return category_profile_stratified_robustness(
        profile,
        minimum_pair_block_coverage=policy["minimum_pair_block_coverage"],
        minimum_pair_block_direction_cosine=policy[
            "minimum_pair_block_direction_cosine"
        ],
        minimum_pair_block_relative_norm=policy["minimum_pair_block_relative_norm"],
        require_pair_block=True,
        minimum_condition_coverage=policy["minimum_condition_coverage"],
        minimum_condition_direction_cosine=policy[
            "minimum_condition_direction_cosine"
        ],
        minimum_condition_relative_norm=policy["minimum_condition_relative_norm"],
        require_condition=True,
        minimum_template_coverage=policy["minimum_template_coverage"],
        minimum_template_direction_cosine=policy[
            "minimum_template_direction_cosine"
        ],
        minimum_template_relative_norm=policy["minimum_template_relative_norm"],
        require_template=True,
        minimum_sketch_dimension=policy["minimum_sketch_dimension"],
    )


def evaluate_discovery_profile(
    profile: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    policy = _policy(protocol)
    category_names = list(profile["category_names"])
    category_counts = [int(value) for value in profile["category_episode_counts"]]
    if min(category_counts) < policy["minimum_episodes_per_category_per_archive"]:
        return {
            "status": "reject_stop_campaign",
            "reason": "discovery category support is below the frozen minimum",
            "policy": policy,
            "metrics": {"minimum_category_episodes": min(category_counts)},
        }
    grid = policy["candidate_grid"]
    optimistic_thresholds = {
        "minimum_dominant_share": min(grid["minimum_dominant_share"]),
        "minimum_dominance_margin": min(grid["minimum_dominance_margin"]),
        "maximum_normalized_entropy": max(grid["maximum_normalized_entropy"]),
        "zero_total_effect_routes_to_shared": True,
    }
    classification = classify_activation_compartments(
        profile["strengths"],
        category_names,
        minimum_dominant_share=optimistic_thresholds["minimum_dominant_share"],
        minimum_dominance_margin=optimistic_thresholds[
            "minimum_dominance_margin"
        ],
        maximum_normalized_entropy=optimistic_thresholds[
            "maximum_normalized_entropy"
        ],
    )
    point_labels = list(classification["compartment_categories"])
    if "uncertainty_block_sums" not in profile:
        raise ValueError("discovery continuation gate requires pair-block evidence")
    replicate_strengths = leave_one_block_out_strengths(
        profile["uncertainty_block_sums"], profile["uncertainty_block_counts"]
    )
    replicate_labels: list[list[str]] = []
    for block_index in range(replicate_strengths.shape[1]):
        result = classify_activation_compartments(
            replicate_strengths[:, block_index, :],
            category_names,
            minimum_dominant_share=optimistic_thresholds[
                "minimum_dominant_share"
            ],
            minimum_dominance_margin=optimistic_thresholds[
                "minimum_dominance_margin"
            ],
            maximum_normalized_entropy=optimistic_thresholds[
                "maximum_normalized_entropy"
            ],
        )
        replicate_labels.append(list(result["compartment_categories"]))
    stability = np.asarray(
        [
            sum(
                replicate_labels[block][component] == point_labels[component]
                for block in range(len(replicate_labels))
            )
            / len(replicate_labels)
            for component in range(len(point_labels))
        ],
        dtype=np.float64,
    )
    eligibility = np.asarray(_robustness(profile, policy)["eligible"], dtype=bool)
    eligible_labels: list[str] = []
    rejection_counts: Counter[str] = Counter()
    for index, label in enumerate(point_labels):
        if label == "shared_cross_category":
            rejection_counts["optimistic_threshold_shared"] += 1
            eligible_labels.append("shared_cross_category")
            continue
        if stability[index] < policy["minimum_jackknife_stability"]:
            rejection_counts["jackknife_unstable"] += 1
            eligible_labels.append("shared_cross_category")
            continue
        category_index = category_names.index(label)
        if not bool(eligibility[index, category_index]):
            rejection_counts["stratified_robustness_failed"] += 1
            eligible_labels.append("shared_cross_category")
            continue
        eligible_labels.append(label)
    eligible_count = sum(label != "shared_cross_category" for label in eligible_labels)
    status = (
        "pass_continue_calibration"
        if eligible_count >= policy["minimum_stable_specific_components"]
        else "reject_stop_campaign"
    )
    return {
        "status": status,
        "reason": (
            "optimistic discovery upper bound meets the frozen minimum"
            if status.startswith("pass_")
            else "even the optimistic discovery upper bound cannot meet the frozen minimum"
        ),
        "policy": policy,
        "optimistic_thresholds": optimistic_thresholds,
        "metrics": {
            "components": len(point_labels),
            "optimistic_specific_count_before_robustness": sum(
                label != "shared_cross_category" for label in point_labels
            ),
            "optimistic_eligible_specific_upper_bound": eligible_count,
            "eligible_specific_by_category": dict(
                sorted(
                    Counter(
                        label
                        for label in eligible_labels
                        if label != "shared_cross_category"
                    ).items()
                )
            ),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "minimum_observed_jackknife_stability": float(np.min(stability)),
            "minimum_category_episodes": min(category_counts),
        },
        "logic": {
            "necessary_condition_only": True,
            "stricter_grid_candidates_cannot_increase_this_specific_upper_bound": True,
            "pass_does_not_predict_calibration_success": True,
        },
    }


def evaluate_calibration_profiles(
    discovery: dict[str, Any],
    calibration: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    policy = _policy(protocol)
    if discovery["category_names"] != calibration["category_names"]:
        raise ValueError("calibration gate category identities differ")
    if np.asarray(discovery["strengths"]).shape != np.asarray(
        calibration["strengths"]
    ).shape:
        raise ValueError("calibration gate component/category shapes differ")
    minimum_support = min(
        *[int(value) for value in discovery["category_episode_counts"]],
        *[int(value) for value in calibration["category_episode_counts"]],
    )
    if minimum_support < policy["minimum_episodes_per_category_per_archive"]:
        return {
            "status": "reject_stop_campaign",
            "reason": "cross-archive category support is below the frozen minimum",
            "policy": policy,
            "metrics": {"minimum_category_episodes": minimum_support},
        }
    eligibility = np.asarray(_robustness(discovery, policy)["eligible"], dtype=bool)
    eligibility &= np.asarray(_robustness(calibration, policy)["eligible"], dtype=bool)
    cosines = cross_archive_category_direction_cosines(
        discovery["category_effect_vectors"],
        calibration["category_effect_vectors"],
    )
    try:
        selected = calibrate_activation_compartment_thresholds(
            discovery["strengths"],
            calibration["strengths"],
            list(discovery["category_names"]),
            direction_cosines=cosines,
            minimum_replication_rate=policy["minimum_replication_rate"],
            minimum_stable_specific_components=policy[
                "minimum_stable_specific_components"
            ],
            minimum_direction_cosine=policy["minimum_signed_direction_cosine"],
            minimum_strength_ratio=policy["minimum_dominant_strength_ratio"],
            robustness_eligibility=eligibility,
            candidate_grid=policy["candidate_grid"],
        )
    except ValueError as exc:
        if "no activation-category threshold candidate" not in str(exc):
            raise
        return {
            "status": "reject_stop_campaign",
            "reason": str(exc),
            "policy": policy,
            "metrics": {
                "minimum_category_episodes": minimum_support,
                "joint_robustness_eligible_cells": int(np.sum(eligibility)),
            },
        }
    return {
        "status": "pass_continue_cluster_validation",
        "reason": "independent calibration selected a passing frozen threshold candidate",
        "policy": policy,
        "calibration": selected,
        "metrics": {
            "minimum_category_episodes": minimum_support,
            "joint_robustness_eligible_cells": int(np.sum(eligibility)),
        },
        "logic": {
            "cluster_validation_not_used": True,
            "selected_thresholds_must_be_recomputed_by_final_cluster_builder": True,
            "pass_does_not_establish_any_nontrivial_cluster_count": True,
        },
    }


def verify_gate_receipt(path: Path, *, required_stage: str | None = None) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != SCHEMA:
        raise ValueError("unsupported VPD effect campaign stage gate schema")
    body = dict(receipt)
    gate_id = str(body.pop("gate_id", ""))
    if content_sha256(body) != gate_id:
        raise ValueError("VPD effect campaign stage gate content address mismatch")
    if required_stage is not None and receipt.get("stage") != required_stage:
        raise ValueError("VPD effect campaign stage gate is for another stage")
    return receipt


def build_stage_gate(
    *,
    stage: str,
    hierarchy_plan_path: Path,
    protocol_config_path: Path,
    discovery_effects_path: Path,
    discovery_merge_manifest_path: Path,
    calibration_effects_path: Path | None = None,
    calibration_merge_manifest_path: Path | None = None,
    discovery_gate_receipt_path: Path | None = None,
) -> dict[str, Any]:
    if stage not in {"discovery", "calibration"}:
        raise ValueError("unsupported VPD effect campaign gate stage")
    protocol = json.loads(protocol_config_path.read_text(encoding="utf-8"))
    rows = load_hierarchy_plan(hierarchy_plan_path)
    _, discovery = load_effects(
        discovery_effects_path, rows, seed=1707, projection_dim=64
    )
    if discovery is None:
        raise ValueError("discovery gate effects omit activation-category payload")
    discovery_provenance = load_effect_merge_split_provenance(
        discovery_effects_path,
        discovery_merge_manifest_path,
        expected_role="discovery",
    )
    discovery_merge_verification = verify_effect_merge(
        discovery_merge_manifest_path
    )
    if int(discovery_merge_verification["components"]) != len(rows):
        raise ValueError("discovery gate merge does not cover the hierarchy inventory")
    source: dict[str, Any] = {
        "hierarchy_plan_sha256": sha256_file(hierarchy_plan_path),
        "protocol_config_sha256": sha256_file(protocol_config_path),
        "discovery_effects_sha256": sha256_file(discovery_effects_path),
        "discovery_merge_manifest_sha256": sha256_file(
            discovery_merge_manifest_path
        ),
        "discovery_split_provenance": discovery_provenance,
        "discovery_merge_verification": discovery_merge_verification,
        "gate_runner_sha256": sha256_file(Path(__file__)),
    }
    if stage == "discovery":
        decision = evaluate_discovery_profile(discovery, protocol)
    else:
        if (
            calibration_effects_path is None
            or calibration_merge_manifest_path is None
            or discovery_gate_receipt_path is None
        ):
            raise ValueError("calibration gate requires both archives and discovery gate")
        prior = verify_gate_receipt(
            discovery_gate_receipt_path, required_stage="discovery"
        )
        if prior.get("decision", {}).get("status") != "pass_continue_calibration":
            raise ValueError("calibration gate requires a passed discovery continuation gate")
        if prior["source"]["discovery_effects_sha256"] != source[
            "discovery_effects_sha256"
        ]:
            raise ValueError("calibration gate discovery archive differs from prior gate")
        _, calibration = load_effects(
            calibration_effects_path, rows, seed=1707, projection_dim=64
        )
        if calibration is None:
            raise ValueError("calibration gate effects omit activation-category payload")
        calibration_provenance = load_effect_merge_split_provenance(
            calibration_effects_path,
            calibration_merge_manifest_path,
            expected_role="calibration",
        )
        calibration_merge_verification = verify_effect_merge(
            calibration_merge_manifest_path
        )
        if int(calibration_merge_verification["components"]) != len(rows):
            raise ValueError("calibration gate merge does not cover the hierarchy inventory")
        identity_keys = (
            "activation_category_split_id",
            "activation_category_split_manifest_sha256",
            "activation_category_split_schema",
            "activation_category_split_semantic_disjointness",
        )
        if any(
            discovery_provenance[key] != calibration_provenance[key]
            for key in identity_keys
        ):
            raise ValueError("calibration gate archives use different split contracts")
        for key in (
            "activation_category_split_episode_id_set_sha256",
            "activation_category_split_pair_id_set_sha256",
            "activation_category_split_template_group_set_sha256",
        ):
            if discovery_provenance[key] == calibration_provenance[key]:
                raise ValueError("calibration gate archive role identities are not distinct")
        source.update(
            {
                "calibration_effects_sha256": sha256_file(calibration_effects_path),
                "calibration_merge_manifest_sha256": sha256_file(
                    calibration_merge_manifest_path
                ),
                "calibration_split_provenance": calibration_provenance,
                "calibration_merge_verification": calibration_merge_verification,
                "discovery_gate_id": prior["gate_id"],
                "discovery_gate_receipt_sha256": sha256_file(
                    discovery_gate_receipt_path
                ),
            }
        )
        decision = evaluate_calibration_profiles(discovery, calibration, protocol)
    body = {
        "schema": SCHEMA,
        "stage": stage,
        "decision": decision,
        "source": source,
        "scientific_boundary": {
            "model_inference_performed_by_gate": False,
            "weights_loaded_or_mutated_by_gate": False,
            "receipt_only_controls_continuation_to_next_expensive_stage": True,
            "does_not_establish_bonsai_cluster_assignments": True,
            "does_not_change_v7_release_decision": True,
        },
    }
    return {**body, "gate_id": content_sha256(body)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate or verify a fail-closed VPD effect campaign stage gate."
    )
    parser.add_argument("--stage", choices=("discovery", "calibration"))
    parser.add_argument("--hierarchy-plan", type=Path)
    parser.add_argument("--protocol-config", type=Path)
    parser.add_argument("--discovery-effects", type=Path)
    parser.add_argument("--discovery-merge-manifest", type=Path)
    parser.add_argument("--calibration-effects", type=Path)
    parser.add_argument("--calibration-merge-manifest", type=Path)
    parser.add_argument("--discovery-gate-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_gate_receipt(args.verify)
    else:
        required = (
            args.stage,
            args.hierarchy_plan,
            args.protocol_config,
            args.discovery_effects,
            args.discovery_merge_manifest,
            args.output,
        )
        if any(value is None for value in required):
            parser.error("gate generation is missing required inputs")
        result = build_stage_gate(
            stage=args.stage,
            hierarchy_plan_path=args.hierarchy_plan,
            protocol_config_path=args.protocol_config,
            discovery_effects_path=args.discovery_effects,
            discovery_merge_manifest_path=args.discovery_merge_manifest,
            calibration_effects_path=args.calibration_effects,
            calibration_merge_manifest_path=args.calibration_merge_manifest,
            discovery_gate_receipt_path=args.discovery_gate_receipt,
        )
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("decision", {}).get("status", "pass").startswith("reject_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
