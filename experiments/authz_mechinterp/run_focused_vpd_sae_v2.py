from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    load_jsonl,
    sha256_file,
    verify_bundle,
    write_json,
)
from experiments.authz_mechinterp.run_focused_vpd_sae import (
    choose_control,
    finalize_study,
    load_dictionary,
    load_effect_inputs,
    prepare_study,
)


SCHEMA = "sch.vpd-focused-sae-intervention-plan.v2"
CONTROL_POLICY = "complete_same_frozen_activation_compartment_out_of_packet_norm_matched_low_alignment"


def _validate_addressed(document: dict[str, Any], id_field: str, label: str) -> str:
    body = copy.deepcopy(document)
    identifier = str(body.pop(id_field, ""))
    if len(identifier) != 64 or content_sha256(body) != identifier:
        raise ValueError(f"{label} content address mismatch")
    return identifier


def apply_category_matched_control_contract(
    *,
    predecessor_plan: dict[str, Any],
    packet: dict[str, Any],
    assignment_categories: dict[str, str],
    effect_component_ids: list[str],
    delta_h: np.ndarray,
    decoder: np.ndarray,
    category_bundle_manifest_sha256: str,
    category_bundle_assignments_sha256: str,
    predecessor_runner_sha256: str,
    runner_v2_sha256: str,
) -> dict[str, Any]:
    """Re-address a focused plan after replacing generic controls by same-compartment controls."""

    predecessor_plan_id = _validate_addressed(
        predecessor_plan, "plan_id", "focused SAE predecessor plan"
    )
    if predecessor_plan.get("schema") != "sch.vpd-focused-sae-intervention-plan.v1":
        raise ValueError("focused SAE V2 requires a V1 predecessor plan")
    if str(predecessor_plan.get("packet_id")) != str(packet.get("packet_id")):
        raise ValueError("focused SAE predecessor plan and packet differ")
    compartment = packet.get("sae_study", {}).get("activation_compartment")
    if not isinstance(compartment, dict):
        raise ValueError("focused SAE V2 requires a frozen activation compartment")
    category = str(compartment.get("category", ""))
    if not category:
        raise ValueError("focused SAE packet activation compartment is absent")
    if predecessor_plan.get("study_inputs", {}).get("activation_category") != category:
        raise ValueError("focused SAE predecessor plan uses a different compartment")

    packet_component_ids = [str(value) for value in packet.get("component_ids", [])]
    packet_component_set = set(packet_component_ids)
    if not packet_component_ids or any(
        assignment_categories.get(component_id) != category
        for component_id in packet_component_ids
    ):
        raise ValueError("focused SAE packet membership crosses activation compartments")
    predecessor_packet_pool = [
        str(value)
        for value in compartment.get("category_matched_control_component_ids", [])
    ]
    if len(predecessor_packet_pool) != len(set(predecessor_packet_pool)):
        raise ValueError("focused SAE predecessor control pool contains duplicates")
    if set(predecessor_packet_pool) & packet_component_set:
        raise ValueError("focused SAE predecessor controls overlap the target packet")
    if any(
        assignment_categories.get(component_id) != category
        for component_id in predecessor_packet_pool
    ):
        raise ValueError("focused SAE predecessor control pool crosses activation compartments")
    pool = sorted(
        component_id
        for component_id, assigned_category in assignment_categories.items()
        if assigned_category == category and component_id not in packet_component_set
    )
    if not pool:
        raise ValueError("focused SAE packet has no complete out-of-packet same-compartment pool")

    component_ids = [str(value) for value in effect_component_ids]
    effect_index = {component_id: index for index, component_id in enumerate(component_ids)}
    if len(effect_index) != len(component_ids):
        raise ValueError("focused SAE effect archive repeats component IDs")
    effects = np.asarray(delta_h, dtype=np.float64)
    decoder_array = np.asarray(decoder, dtype=np.float64)
    if effects.ndim != 2 or effects.shape[0] != len(component_ids):
        raise ValueError("focused SAE effect archive does not align with component IDs")
    if decoder_array.ndim != 2 or effects.shape[1] != decoder_array.shape[1]:
        raise ValueError("focused SAE decoder and effect spaces differ")
    missing_pool = sorted(set(pool) - set(effect_index))
    if missing_pool:
        raise ValueError("focused SAE effect archive omits same-compartment controls")

    candidates: list[dict[str, Any]] = []
    for observed in predecessor_plan.get("candidates", []):
        candidate = copy.deepcopy(observed)
        _validate_addressed(candidate, "candidate_id", "focused SAE candidate")
        component_id = str(candidate.get("component_id", ""))
        if component_id not in packet_component_set:
            raise ValueError("focused SAE candidate is outside its packet")
        if assignment_categories.get(component_id) != category:
            raise ValueError("focused SAE target belongs to a different compartment")
        feature_index = int(candidate.get("feature_index", -1))
        if not 0 <= feature_index < decoder_array.shape[0]:
            raise ValueError("focused SAE candidate feature index is invalid")
        if component_id not in effect_index:
            raise ValueError("focused SAE effect archive omits its target component")
        target_norm = float(np.linalg.norm(effects[effect_index[component_id]]))
        control_id, norm_gap, absolute_cosine = choose_control(
            control_ids=pool,
            effect_index=effect_index,
            delta_h=effects,
            decoder_direction=decoder_array[feature_index],
            target_norm=target_norm,
        )
        candidate.pop("candidate_id", None)
        candidate["matched_control_component_id"] = control_id
        candidate["matched_control_effect_norm_relative_gap"] = norm_gap
        candidate["matched_control_absolute_cosine"] = absolute_cosine
        candidate["matched_control_activation_category"] = category
        candidate["matched_control_policy"] = CONTROL_POLICY
        candidate["matched_control_pool_sha256"] = content_sha256(pool)
        candidates.append(
            {**candidate, "candidate_id": content_sha256(candidate)}
        )

    body = copy.deepcopy(predecessor_plan)
    body.pop("plan_id", None)
    body["schema"] = SCHEMA
    body["candidates"] = candidates
    body["study_inputs"]["predecessor_plan_id"] = predecessor_plan_id
    body["study_inputs"]["predecessor_runner_sha256"] = predecessor_runner_sha256
    body["study_inputs"]["executor_sha256"] = runner_v2_sha256
    body["study_inputs"]["category_bundle_manifest_sha256"] = (
        category_bundle_manifest_sha256
    )
    body["study_inputs"]["category_bundle_assignments_sha256"] = (
        category_bundle_assignments_sha256
    )
    project_root = Path(__file__).resolve().parents[2]
    body["study_inputs"]["intervention_executor_v2_sha256"] = sha256_file(
        project_root
        / "experiments/authz_mechinterp/compute_focused_sae_interventions_v2.py"
    )
    body["study_inputs"]["predecessor_intervention_executor_sha256"] = sha256_file(
        project_root
        / "experiments/authz_mechinterp/compute_focused_sae_interventions.py"
    )
    body["intervention_contract"]["parameter_effect_control"] = (
        "subtract_same_compartment_matched_norm_component_slope"
    )
    body["control_compartment_contract"] = {
        "schema": "sch.vpd-focused-sae-control-compartment.v1",
        "status": "pass",
        "activation_category": category,
        "policy": CONTROL_POLICY,
        "control_pool_component_ids": pool,
        "control_pool_sha256": content_sha256(pool),
        "control_pool_size": len(pool),
        "complete_assignment_derived_pool_required": True,
        "packet_first_32_pool_used_for_selection": False,
        "predecessor_packet_control_pool_component_ids": predecessor_packet_pool,
        "predecessor_packet_control_pool_sha256": content_sha256(
            predecessor_packet_pool
        ),
        "predecessor_packet_control_pool_size": len(predecessor_packet_pool),
        "target_packet_overlap": 0,
        "all_targets_and_controls_share_frozen_compartment": True,
        "generic_cross_compartment_pool_used": False,
        "v2_executor_required": True,
    }
    return {**body, "plan_id": content_sha256(body)}


def validate_v2_plan_control_contract(
    *,
    plan: dict[str, Any],
    packet: dict[str, Any],
    assignment_categories: dict[str, str],
) -> dict[str, Any]:
    plan_id = _validate_addressed(plan, "plan_id", "focused SAE V2 plan")
    if plan.get("schema") != SCHEMA:
        raise ValueError("unsupported focused SAE V2 plan schema")
    contract = plan.get("control_compartment_contract", {})
    compartment = packet.get("sae_study", {}).get("activation_compartment", {})
    category = str(compartment.get("category", ""))
    pool = [str(value) for value in contract.get("control_pool_component_ids", [])]
    if (
        contract.get("status") != "pass"
        or contract.get("activation_category") != category
        or contract.get("policy") != CONTROL_POLICY
        or contract.get("control_pool_sha256") != content_sha256(pool)
        or int(contract.get("control_pool_size", -1)) != len(pool)
        or contract.get("complete_assignment_derived_pool_required") is not True
        or contract.get("packet_first_32_pool_used_for_selection") is not False
        or contract.get("target_packet_overlap") != 0
        or contract.get("all_targets_and_controls_share_frozen_compartment")
        is not True
        or contract.get("generic_cross_compartment_pool_used") is not False
        or contract.get("v2_executor_required") is not True
    ):
        raise ValueError("focused SAE V2 control compartment contract differs")
    expected_pool = sorted(
        component_id
        for component_id, assigned_category in assignment_categories.items()
        if assigned_category == category and component_id not in set(
            str(value) for value in packet.get("component_ids", [])
        )
    )
    predecessor_packet_pool = [
        str(value)
        for value in compartment.get("category_matched_control_component_ids", [])
    ]
    if pool != expected_pool or not pool:
        raise ValueError("focused SAE V2 control pool is not the complete assignment-derived pool")
    if (
        contract.get("predecessor_packet_control_pool_component_ids")
        != predecessor_packet_pool
        or contract.get("predecessor_packet_control_pool_sha256")
        != content_sha256(predecessor_packet_pool)
        or int(contract.get("predecessor_packet_control_pool_size", -1))
        != len(predecessor_packet_pool)
        or len(predecessor_packet_pool) != len(set(predecessor_packet_pool))
        or set(predecessor_packet_pool) - set(pool)
    ):
        raise ValueError("focused SAE V2 predecessor packet control pool is inconsistent")
    targets = [str(value) for value in packet.get("component_ids", [])]
    if any(assignment_categories.get(value) != category for value in [*targets, *pool]):
        raise ValueError("focused SAE V2 plan crosses activation compartments")
    for candidate in plan.get("candidates", []):
        _validate_addressed(candidate, "candidate_id", "focused SAE V2 candidate")
        if (
            str(candidate.get("component_id")) not in targets
            or str(candidate.get("matched_control_component_id")) not in pool
            or candidate.get("matched_control_activation_category") != category
            or candidate.get("matched_control_policy") != CONTROL_POLICY
            or candidate.get("matched_control_pool_sha256") != content_sha256(pool)
        ):
            raise ValueError("focused SAE V2 candidate control compartment differs")
    return {
        "schema": "sch.vpd-focused-sae-control-compartment-verification.v1",
        "status": "pass",
        "plan_id": plan_id,
        "activation_category": category,
        "control_pool_size": len(pool),
        "candidate_count": len(plan.get("candidates", [])),
    }


def recompute_v2_candidate_controls(
    *,
    plan: dict[str, Any],
    packet: dict[str, Any],
    effect_component_ids: list[str],
    delta_h: np.ndarray,
    decoder: np.ndarray,
) -> dict[str, Any]:
    """Recompute every selected control from immutable numeric inputs."""

    contract = plan.get("control_compartment_contract", {})
    pool = [str(value) for value in contract.get("control_pool_component_ids", [])]
    effect_ids = [str(value) for value in effect_component_ids]
    effect_index = {component_id: index for index, component_id in enumerate(effect_ids)}
    effects = np.asarray(delta_h, dtype=np.float64)
    decoder_array = np.asarray(decoder, dtype=np.float64)
    if (
        len(effect_index) != len(effect_ids)
        or effects.ndim != 2
        or effects.shape[0] != len(effect_ids)
        or decoder_array.ndim != 2
        or effects.shape[1] != decoder_array.shape[1]
    ):
        raise ValueError("focused SAE V2 numeric control inputs do not align")
    if set(pool) - set(effect_index):
        raise ValueError("focused SAE V2 numeric archive omits the control pool")
    packet_components = {str(value) for value in packet.get("component_ids", [])}
    for candidate in plan.get("candidates", []):
        component_id = str(candidate.get("component_id", ""))
        feature_index = int(candidate.get("feature_index", -1))
        if (
            component_id not in packet_components
            or component_id not in effect_index
            or not 0 <= feature_index < decoder_array.shape[0]
        ):
            raise ValueError("focused SAE V2 candidate numeric identity differs")
        target_norm = float(np.linalg.norm(effects[effect_index[component_id]]))
        control_id, norm_gap, absolute_cosine = choose_control(
            control_ids=pool,
            effect_index=effect_index,
            delta_h=effects,
            decoder_direction=decoder_array[feature_index],
            target_norm=target_norm,
        )
        if (
            candidate.get("matched_control_component_id") != control_id
            or not math.isclose(
                float(candidate.get("matched_control_effect_norm_relative_gap", -1.0)),
                norm_gap,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(candidate.get("matched_control_absolute_cosine", -1.0)),
                absolute_cosine,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("focused SAE V2 candidate control selection is not reproducible")
    return {
        "schema": "sch.vpd-focused-sae-control-selection-verification.v1",
        "status": "pass",
        "candidate_count": len(plan.get("candidates", [])),
        "control_pool_sha256": content_sha256(pool),
        "numeric_selection_recomputed": True,
    }


def prepare_study_v2(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    predecessor_plan, context = prepare_study(**kwargs)
    bundle_dir = Path(kwargs["bundle_dir"])
    bundle_verification = verify_bundle(bundle_dir)
    if bundle_verification.get("status") != "pass":
        raise ValueError("focused SAE V2 category bundle verification failed")
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assignments_path = bundle_dir / manifest["artifacts"]["assignments"]["path"]
    assignments = load_jsonl(assignments_path)
    assignment_categories = {
        str(row["component_id"]): str(row.get("activation_category", ""))
        for row in assignments
    }
    dictionary = load_dictionary(Path(kwargs["dictionary_path"]))
    component_ids, delta_h, _ = load_effect_inputs(
        Path(kwargs["effect_archive_path"]),
        Path(kwargs["effect_manifest_path"]),
        str(predecessor_plan["study_inputs"]["alignment_activation_key"]),
    )
    project_root = Path(__file__).resolve().parents[2]
    plan = apply_category_matched_control_contract(
        predecessor_plan=predecessor_plan,
        packet=context["packet"],
        assignment_categories=assignment_categories,
        effect_component_ids=component_ids,
        delta_h=delta_h,
        decoder=dictionary["decoder"],
        category_bundle_manifest_sha256=sha256_file(manifest_path),
        category_bundle_assignments_sha256=sha256_file(assignments_path),
        predecessor_runner_sha256=sha256_file(
            project_root / "experiments/authz_mechinterp/run_focused_vpd_sae.py"
        ),
        runner_v2_sha256=sha256_file(Path(__file__)),
    )
    validate_v2_plan_control_contract(
        plan=plan,
        packet=context["packet"],
        assignment_categories=assignment_categories,
    )
    recompute_v2_candidate_controls(
        plan=plan,
        packet=context["packet"],
        effect_component_ids=component_ids,
        delta_h=delta_h,
        decoder=dictionary["decoder"],
    )
    return plan, context


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan a category-compartment-safe focused VPD/SAE study."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--packet-id", required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--dictionary-manifest", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--activation-capture-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--effects", type=Path, required=True)
    parser.add_argument("--effect-merge-manifest", type=Path, required=True)
    parser.add_argument("--intervention-dataset", type=Path, required=True)
    parser.add_argument("--activation-category-split-manifest", type=Path, required=True)
    parser.add_argument("--dictionary-quality-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--intervention-trace", type=Path)
    parser.add_argument("--label", default="semantic_authorization")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--top-features", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--maximum-max-stat-p", type=float, default=0.05)
    parser.add_argument("--minimum-absolute-cosine", type=float, default=0.20)
    parser.add_argument("--minimum-pairs-per-family", type=int, default=2)
    parser.add_argument("--intervention-pairs-per-family", type=int, default=2)
    args = parser.parse_args()
    plan, context = prepare_study_v2(
        bundle_dir=args.bundle,
        packet_id=args.packet_id,
        dictionary_path=args.dictionary,
        dictionary_manifest_path=args.dictionary_manifest,
        activation_capture_path=args.activations,
        activation_capture_manifest_path=args.activation_capture_manifest,
        metadata_path=args.metadata,
        effect_archive_path=args.effects,
        effect_manifest_path=args.effect_merge_manifest,
        intervention_dataset_path=args.intervention_dataset,
        label_name=args.label,
        seed=args.seed,
        permutations=args.permutations,
        top_features=args.top_features,
        max_pairs=args.max_pairs,
        maximum_max_stat_p=args.maximum_max_stat_p,
        minimum_absolute_cosine=args.minimum_absolute_cosine,
        minimum_pairs_per_family=args.minimum_pairs_per_family,
        intervention_pairs_per_family=args.intervention_pairs_per_family,
        activation_category_split_manifest_path=args.activation_category_split_manifest,
        dictionary_quality_report_path=args.dictionary_quality_report,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "intervention_plan.json", plan)
    result: dict[str, Any] = {
        "schema": "sch.vpd-focused-sae-planning-output.v2",
        "plan_id": plan["plan_id"],
        "candidate_pairs": len(plan["candidates"]),
        "plan_path": "intervention_plan.json",
        "control_compartment_status": "pass",
    }
    if args.intervention_trace is not None:
        unsigned = finalize_study(
            plan=plan,
            packet=context["packet"],
            intervention_trace_path=args.intervention_trace,
        )
        write_json(args.output_dir / "unsigned_result.json", unsigned)
        result["unsigned_result_path"] = "unsigned_result.json"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
