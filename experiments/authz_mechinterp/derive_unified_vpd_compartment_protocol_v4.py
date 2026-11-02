from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_activation_category_clustering_geometry import (
    verify_benchmark as verify_clustering_geometry_benchmark,
)
from experiments.authz_mechinterp.benchmark_activation_category_cluster_count_selection import (
    verify_benchmark as verify_cluster_count_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json


SCHEMA = "sch.vpd-activation-category-protocol.v4"


def build_protocol(*, predecessor_path: Path) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if predecessor.get("schema") != "sch.vpd-activation-category-protocol.v3":
        raise ValueError("per-component admission protocol predecessor is not V3")

    geometry_path = (
        project_root
        / "artifacts/vpd_activation_category_clustering_geometry_benchmark_v1/report.json"
    )
    count_path = (
        project_root
        / "artifacts/vpd_activation_category_cluster_count_benchmark_v1/report.json"
    )
    geometry = verify_clustering_geometry_benchmark(geometry_path)
    count = verify_cluster_count_benchmark(count_path)

    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = predecessor_path.resolve().relative_to(
        project_root.resolve()
    ).as_posix()
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    compartments = result["component_compartments"]
    compartments["within_category_clustering_geometry"][
        "synthetic_benchmark_id"
    ] = geometry["benchmark_id"]
    compartments["within_category_cluster_count_selection"][
        "synthetic_benchmark_id"
    ] = count["benchmark_id"]
    calibration = compartments["threshold_calibration"]
    compartments["per_component_cross_archive_admission"] = {
        "schema": "sch.vpd-cross-archive-component-admission.v1",
        "mode": "same_category_direction_magnitude_and_stratified_robustness",
        "applied_after_discovery_within_split_screens": True,
        "calibration_point_estimate_same_specific_category_required": True,
        "minimum_signed_direction_cosine": float(
            calibration["minimum_signed_direction_cosine"]
        ),
        "minimum_dominant_strength_ratio": float(
            calibration["minimum_dominant_strength_ratio"]
        ),
        "same_stratified_robustness_gates_required_in_both_archives": True,
        "global_threshold_replication_does_not_waive_component_admission": True,
        "failed_specific_routes_to": "shared_cross_category",
        "required_for_production_category_bundle": True,
        "assignment_diagnostics_bound_into_bundle": True,
        "independent_verifier_recomputes_every_component": True,
        "model_outcomes_used": False,
    }
    result["effect_capture"]["campaign_plan"]["path"] = (
        "artifacts/vpd_effect_capture_campaign_v3/plan.json"
    )
    result["effect_capture"]["campaign_plan"]["stage_gate_benchmark_path"] = (
        "artifacts/vpd_effect_campaign_stage_gate_benchmark_v3/report.json"
    )
    result["sae_training"]["replica_training_plan"]["path"] = (
        "artifacts/vpd_compartment_sae_training_plan_v3/plan.json"
    )
    result["unified_lineage"][
        "per_component_cross_archive_admission_required"
    ] = True
    result["unified_lineage"]["model_inference_performed"] = False
    result["unified_lineage"]["execution_authorized"] = False
    return result


def verify_protocol(path: Path, *, predecessor_path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = build_protocol(predecessor_path=predecessor_path)
    if observed != expected:
        raise ValueError("V4 VPD compartment protocol is not reproducible")
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v4",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "predecessor_sha256": sha256_file(predecessor_path),
        "per_component_cross_archive_admission_required": True,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive or verify the fail-closed per-component VPD protocol V4."
    )
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        print(
            json.dumps(
                verify_protocol(args.verify, predecessor_path=args.predecessor),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.output is None:
        parser.error("--output is required unless --verify is used")
    protocol = build_protocol(predecessor_path=args.predecessor)
    write_json(args.output, protocol)
    print(
        json.dumps(
            {
                "schema": protocol["schema"],
                "status": "written",
                "output": args.output.as_posix(),
                "protocol_sha256": sha256_file(args.output),
                "execution_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
