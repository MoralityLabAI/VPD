from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    classify_activation_compartments,
    content_sha256,
    cross_archive_category_direction_cosines,
    write_json,
)


SCHEMA = "sch.vpd-frozen-category-partition-audit.v1"
SHARED = "shared_cross_category"


def _validate_inputs(
    component_ids: list[str],
    frozen_categories: list[str],
    category_names: list[str],
    discovery_effects: np.ndarray,
    audit_effects: np.ndarray,
    robustness_eligibility: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not component_ids or len(component_ids) != len(set(component_ids)):
        raise ValueError("partition audit component IDs must be nonempty and unique")
    if category_names != sorted(category_names) or len(category_names) < 2:
        raise ValueError("partition audit category names must be sorted and nontrivial")
    if len(category_names) != len(set(category_names)):
        raise ValueError("partition audit category names must be unique")
    if len(frozen_categories) != len(component_ids):
        raise ValueError("partition audit frozen categories do not align")
    valid_categories = set(category_names) | {SHARED}
    if any(category not in valid_categories for category in frozen_categories):
        raise ValueError("partition audit contains an unknown frozen category")
    discovery = np.asarray(discovery_effects, dtype=np.float64)
    audit = np.asarray(audit_effects, dtype=np.float64)
    expected_prefix = (len(component_ids), len(category_names))
    if discovery.ndim != 3 or discovery.shape[:2] != expected_prefix:
        raise ValueError("partition audit discovery effects do not align")
    if audit.shape != discovery.shape:
        raise ValueError("partition audit effect archives do not align")
    if not np.all(np.isfinite(discovery)) or not np.all(np.isfinite(audit)):
        raise ValueError("partition audit effects contain non-finite values")
    robustness = np.asarray(robustness_eligibility, dtype=bool)
    if robustness.shape != expected_prefix:
        raise ValueError("partition audit robustness eligibility does not align")
    return discovery, audit, robustness


def evaluate_frozen_partition(
    *,
    component_ids: list[str],
    frozen_categories: list[str],
    category_names: list[str],
    discovery_effects: np.ndarray,
    audit_effects: np.ndarray,
    robustness_eligibility: np.ndarray,
    specificity_thresholds: dict[str, Any],
    minimum_direction_cosine: float = 0.5,
    minimum_strength_ratio: float = 0.5,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit a frozen partition without allowing the audit to select it.

    The frozen categories are the only candidate assignments.  The audit pack
    can reject a specific assignment, but it cannot promote a shared component,
    choose thresholds, choose clusters, or return a revised partition.
    """

    discovery, audit, robustness = _validate_inputs(
        component_ids,
        frozen_categories,
        category_names,
        discovery_effects,
        audit_effects,
        robustness_eligibility,
    )
    if not -1.0 <= minimum_direction_cosine <= 1.0:
        raise ValueError("partition audit direction threshold is invalid")
    if not 0.0 < minimum_strength_ratio <= 1.0:
        raise ValueError("partition audit strength threshold is invalid")
    required_thresholds = {
        "minimum_dominant_share",
        "minimum_dominance_margin",
        "maximum_normalized_entropy",
    }
    if not required_thresholds <= set(specificity_thresholds):
        raise ValueError("partition audit specificity thresholds are incomplete")

    discovery_strengths = np.linalg.norm(discovery, axis=2)
    audit_strengths = np.linalg.norm(audit, axis=2)
    audit_classification = classify_activation_compartments(
        audit_strengths,
        category_names,
        minimum_dominant_share=float(
            specificity_thresholds["minimum_dominant_share"]
        ),
        minimum_dominance_margin=float(
            specificity_thresholds["minimum_dominance_margin"]
        ),
        maximum_normalized_entropy=float(
            specificity_thresholds["maximum_normalized_entropy"]
        ),
    )
    direction_cosines = cross_archive_category_direction_cosines(discovery, audit)
    maximum_strengths = np.maximum(discovery_strengths, audit_strengths)
    strength_ratios = np.zeros_like(maximum_strengths)
    supported = maximum_strengths > 1e-12
    strength_ratios[supported] = (
        np.minimum(discovery_strengths, audit_strengths)[supported]
        / maximum_strengths[supported]
    )

    decisions: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    specific_passes = 0
    specific_failures = 0
    for index, (component_id, frozen_category) in enumerate(
        zip(component_ids, frozen_categories, strict=True)
    ):
        audit_category = audit_classification["compartment_categories"][index]
        if frozen_category == SHARED:
            decisions.append(
                {
                    "component_id": component_id,
                    "frozen_category": frozen_category,
                    "audit_point_estimate_category": audit_category,
                    "specific_assignment_audited": False,
                    "specific_assignment_passed": None,
                    "rejection_reason": None,
                    "shared_to_specific_promotion_allowed": False,
                }
            )
            continue
        category_index = category_names.index(frozen_category)
        cosine = float(direction_cosines[index, category_index])
        ratio = float(strength_ratios[index, category_index])
        robust = bool(robustness[index, category_index])
        if audit_category != frozen_category:
            reason = "audit_category_mismatch"
        elif cosine < minimum_direction_cosine:
            reason = "audit_direction_instability"
        elif ratio < minimum_strength_ratio:
            reason = "audit_magnitude_instability"
        elif not robust:
            reason = "audit_stratified_robustness_failure"
        else:
            reason = None
        passed = reason is None
        if passed:
            specific_passes += 1
        else:
            specific_failures += 1
            assert reason is not None
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        decisions.append(
            {
                "component_id": component_id,
                "frozen_category": frozen_category,
                "audit_point_estimate_category": audit_category,
                "specific_assignment_audited": True,
                "specific_assignment_passed": passed,
                "rejection_reason": reason,
                "target_direction_cosine": cosine,
                "target_strength_ratio": ratio,
                "target_stratified_robustness_passed": robust,
                "shared_to_specific_promotion_allowed": False,
            }
        )

    specific_count = specific_passes + specific_failures
    eligible = specific_count > 0 and specific_failures == 0
    body = {
        "schema": SCHEMA,
        "status": (
            "pass_partition_audit" if eligible else "reject_partition_audit"
        ),
        "source": copy.deepcopy(source or {}),
        "policy": {
            "mode": "post_freeze_full_component_reject_only_generalization_audit",
            "specificity_thresholds": copy.deepcopy(specificity_thresholds),
            "minimum_direction_cosine": minimum_direction_cosine,
            "minimum_strength_ratio": minimum_strength_ratio,
            "same_specific_category_required": True,
            "stratified_robustness_required": True,
            "zero_specific_failures_required": True,
            "at_least_one_specific_component_required_for_category_specific_training": True,
            "audit_may_select_thresholds_clusters_or_components": False,
            "audit_may_promote_shared_to_specific": False,
            "failure_may_not_tune_candidate_on_same_pack": True,
            "next_candidate_after_failure_requires_fresh_pack_version_and_seed": True,
        },
        "category_names": list(category_names),
        "component_id_set_sha256": content_sha256(sorted(component_ids)),
        "summary": {
            "component_count": len(component_ids),
            "frozen_specific_component_count": specific_count,
            "frozen_shared_component_count": len(component_ids) - specific_count,
            "specific_pass_count": specific_passes,
            "specific_failure_count": specific_failures,
            "specific_failure_reason_counts": dict(sorted(reason_counts.items())),
            "eligible_for_category_specific_sae_training": eligible,
        },
        "decisions": decisions,
        "scientific_boundary": {
            "model_outcomes_used": False,
            "partition_mutated": False,
            "selection_performed_on_audit": False,
        },
    }
    return {**body, "partition_audit_id": content_sha256(body)}


def verify_partition_audit_report(
    path: Path,
    **inputs: Any,
) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = copy.deepcopy(observed)
    audit_id = str(body.pop("partition_audit_id", ""))
    if content_sha256(body) != audit_id:
        raise ValueError("partition audit report content address mismatch")
    expected = evaluate_frozen_partition(**inputs)
    if expected != observed:
        raise ValueError("partition audit report is not reproducible")
    return {
        "schema": "sch.vpd-frozen-category-partition-audit-verification.v1",
        "status": "pass",
        "partition_audit_id": audit_id,
        "audit_decision": observed["status"],
        "component_count": observed["summary"]["component_count"],
        "specific_failure_count": observed["summary"]["specific_failure_count"],
        "eligible_for_category_specific_sae_training": observed["summary"][
            "eligible_for_category_specific_sae_training"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "The production loader is intentionally not implicit: use this module's "
            "evaluate_frozen_partition API with signed, exact-coverage effect archives."
        )
    )
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    if not args.explain:
        parser.error("use --explain or call the verified Python API")
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "model_free_evaluator_ready",
                "production_inputs_required": [
                    "frozen category-aware bundle assignments",
                    "bundle-bound discovery category effects",
                    "fresh signed partition-audit category effects",
                    "fresh partition-audit stratified robustness eligibility",
                ],
                "execution_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
