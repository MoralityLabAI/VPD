from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.gate_vpd_effect_campaign_stage import (
    evaluate_calibration_profiles,
    evaluate_discovery_profile,
)


def synthetic_profile(
    *,
    specific_components: int = 40,
    swap_categories: bool = False,
    reverse_directions: bool = False,
    magnitude_scale: float = 1.0,
    corrupt_condition_components: int = 0,
) -> dict[str, Any]:
    components = 40
    categories = ["replayed", "valid_authority"]
    vectors = np.zeros((components, 2, 256), dtype=np.float64)
    for index in range(components):
        if index >= specific_components:
            vectors[index, 0, 0] = 1.0
            vectors[index, 1, 1] = 1.0
        elif index % 2 == 0:
            vectors[index, 0, 0] = 2.0
            vectors[index, 1, 1] = 0.05
        else:
            vectors[index, 0, 0] = 0.05
            vectors[index, 1, 1] = 2.0
    if swap_categories:
        vectors = vectors[:, ::-1, :]
    if reverse_directions:
        vectors = -vectors
    vectors *= magnitude_scale
    strengths = np.linalg.norm(vectors, axis=2)
    block_counts = np.ones((2, 8), dtype=np.int64)
    block_sums = np.repeat(vectors[:, :, None, :], 8, axis=2)
    condition_categories = [
        "replayed",
        "replayed",
        "valid_authority",
        "valid_authority",
    ]
    condition_counts = np.asarray([8, 8, 8, 8], dtype=np.int64)
    condition_sums = np.stack(
        [vectors[:, 0] * 8, vectors[:, 0] * 8, vectors[:, 1] * 8, vectors[:, 1] * 8],
        axis=1,
    )
    for index in range(min(corrupt_condition_components, specific_components)):
        target_cell = 0 if index % 2 == 0 else 2
        condition_sums[index, target_cell] *= -1.0
    template_categories = [category for category in categories for _ in range(4)]
    template_counts = np.full(8, 4, dtype=np.int64)
    template_sums = np.stack(
        [
            vectors[:, category_index] * 4
            for category_index in range(2)
            for _ in range(4)
        ],
        axis=1,
    )
    return {
        "category_names": categories,
        "category_episode_counts": [32, 32],
        "strengths": strengths,
        "category_effect_vectors": vectors,
        "uncertainty_block_counts": block_counts,
        "uncertainty_block_sums": block_sums,
        "uncertainty_block_count": 8,
        "uncertainty_countsketch_dimension": 256,
        "uncertainty_countsketch_seed": 1,
        "condition_categories": condition_categories,
        "condition_counts": condition_counts,
        "condition_sums": condition_sums,
        "template_cell_categories": template_categories,
        "template_cell_counts": template_counts,
        "template_cell_sums": template_sums,
    }


def build_benchmark(protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    clean_discovery = synthetic_profile()
    clean_calibration = synthetic_profile(magnitude_scale=0.90)
    scenarios = [
        {
            "scenario_id": "clean_discovery_continues",
            "expected_status": "pass_continue_calibration",
            "result": evaluate_discovery_profile(clean_discovery, protocol),
        },
        {
            "scenario_id": "insufficient_discovery_upper_bound_stops",
            "expected_status": "reject_stop_campaign",
            "result": evaluate_discovery_profile(
                synthetic_profile(specific_components=20), protocol
            ),
        },
        {
            "scenario_id": "condition_artifact_discovery_stops",
            "expected_status": "reject_stop_campaign",
            "result": evaluate_discovery_profile(
                synthetic_profile(corrupt_condition_components=16), protocol
            ),
        },
        {
            "scenario_id": "clean_calibration_continues",
            "expected_status": "pass_continue_cluster_validation",
            "result": evaluate_calibration_profiles(
                clean_discovery, clean_calibration, protocol
            ),
        },
        {
            "scenario_id": "category_swap_calibration_stops",
            "expected_status": "reject_stop_campaign",
            "result": evaluate_calibration_profiles(
                clean_discovery,
                synthetic_profile(swap_categories=True),
                protocol,
            ),
        },
        {
            "scenario_id": "direction_reversal_calibration_stops",
            "expected_status": "reject_stop_campaign",
            "result": evaluate_calibration_profiles(
                clean_discovery,
                synthetic_profile(reverse_directions=True),
                protocol,
            ),
        },
        {
            "scenario_id": "magnitude_collapse_calibration_stops",
            "expected_status": "reject_stop_campaign",
            "result": evaluate_calibration_profiles(
                clean_discovery,
                synthetic_profile(magnitude_scale=0.02),
                protocol,
            ),
        },
    ]
    for row in scenarios:
        row["status"] = (
            "pass"
            if row["result"]["status"] == row["expected_status"]
            else "fail"
        )
    project_root = Path(__file__).resolve().parents[2]
    body = {
        "schema": "sch.vpd-effect-campaign-stage-gate-benchmark.v1",
        "status": "pass" if all(row["status"] == "pass" for row in scenarios) else "fail",
        "design": {
            "synthetic_components": 40,
            "synthetic_categories": 2,
            "pair_blocks": 8,
            "countsketch_dimension": 256,
            "scenarios": len(scenarios),
        },
        "scenarios": scenarios,
        "source": {
            "benchmark_runner_sha256": sha256_file(Path(__file__)),
            "stage_gate_runner_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/gate_vpd_effect_campaign_stage.py"
            ),
            "protocol_sha256": sha256_file(protocol_path),
        },
        "scientific_boundary": {
            "synthetic_effects_only": True,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "does_not_establish_bonsai_stage_gate_outcomes": True,
            "does_not_establish_bonsai_cluster_assignments": True,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, protocol_path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    body = copy.deepcopy(report)
    benchmark_id = str(body.pop("benchmark_id", ""))
    if content_sha256(body) != benchmark_id:
        raise ValueError("effect campaign stage-gate benchmark address differs")
    if report.get("status") != "pass":
        raise ValueError("effect campaign stage-gate benchmark failed")
    if build_benchmark(protocol_path) != report:
        raise ValueError("effect campaign stage-gate benchmark is not reproducible")
    return {
        "schema": "sch.vpd-effect-campaign-stage-gate-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": benchmark_id,
        "scenarios": len(report["scenarios"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or verify the model-free VPD campaign stage-gate benchmark."
    )
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_benchmark(args.verify, protocol_path=args.protocol_config)
    else:
        if args.output is None:
            parser.error("benchmark generation requires --output")
        result = build_benchmark(args.protocol_config)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
