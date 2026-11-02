from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.run_focused_vpd_sae import (
    behavior_screen,
    focused_screen_endpoint,
)


CATEGORIES = (
    "replayed",
    "valid_authority",
    "signature_invalid",
    "scope_invalid",
)


def synthetic_screen_fixture(seed: int) -> tuple[
    list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(seed)
    metadata: list[dict[str, Any]] = []
    features: list[list[float]] = []
    covariates: list[list[float]] = []
    groups: list[str] = []
    blocks: list[str] = []
    for pair_index in range(64):
        family = f"family-{pair_index % 8}"
        prompt_tokens = 80.0 + float(pair_index % 11)
        anchor_position = 48.0 + float(pair_index % 7)
        for category in CATEGORIES:
            outcome = int(category == "valid_authority")
            metadata.append(
                {
                    "activation_category": category,
                    "factors": {"semantic_authorization": outcome},
                }
            )
            noise = rng.normal(0.0, 0.04, size=4)
            features.append(
                [
                    3.0 * int(category == "replayed") + noise[0],
                    3.0 * outcome + noise[1],
                    2.0 * int(category == "signature_invalid") + noise[2],
                    0.02 * prompt_tokens + noise[3],
                ]
            )
            covariates.append([prompt_tokens, anchor_position])
            groups.append(family)
            blocks.append(f"pair-{pair_index:03d}")
    return (
        metadata,
        np.asarray(features, dtype=np.float32),
        np.asarray(covariates, dtype=np.float64),
        np.asarray(groups),
        np.asarray(blocks),
    )


def _screen(
    *,
    metadata: list[dict[str, Any]],
    features: np.ndarray,
    covariates: np.ndarray,
    groups: np.ndarray,
    blocks: np.ndarray,
    compartment: str | None,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels, endpoint = focused_screen_endpoint(
        metadata=metadata,
        label_name="semantic_authorization",
        compartment_category=compartment,
    )
    scores = behavior_screen(
        labels=labels,
        features=features,
        covariates=covariates,
        groups=groups,
        blocks=blocks,
        seed=seed,
        permutations=127,
        top_features=4,
    )
    scores.sort(
        key=lambda row: (
            -abs(float(row["residualized_auc"]) - 0.5),
            int(row["feature_index"]),
        )
    )
    return endpoint, scores


def build_benchmark(seed: int = 290829) -> dict[str, Any]:
    metadata, features, covariates, groups, blocks = synthetic_screen_fixture(seed)
    specific_endpoint, specific_scores = _screen(
        metadata=metadata,
        features=features,
        covariates=covariates,
        groups=groups,
        blocks=blocks,
        compartment="replayed",
        seed=seed,
    )
    legacy_endpoint, legacy_scores = _screen(
        metadata=metadata,
        features=features,
        covariates=covariates,
        groups=groups,
        blocks=blocks,
        compartment=None,
        seed=seed,
    )
    shared_endpoint, shared_scores = _screen(
        metadata=metadata,
        features=features,
        covariates=covariates,
        groups=groups,
        blocks=blocks,
        compartment="shared_cross_category",
        seed=seed,
    )
    scenarios = [
        {
            "scenario_id": "specific_replayed_category_membership",
            "endpoint": specific_endpoint,
            "top_feature_index": int(specific_scores[0]["feature_index"]),
            "expected_top_feature_index": 0,
            "checks": {
                "category_feature_selected": int(
                    specific_scores[0]["feature_index"]
                )
                == 0,
                "outcome_label_excluded": specific_endpoint[
                    "model_outcome_label_used_for_selection"
                ]
                is False,
            },
        },
        {
            "scenario_id": "legacy_global_outcome_leakage_diagnostic",
            "endpoint": legacy_endpoint,
            "top_feature_index": int(legacy_scores[0]["feature_index"]),
            "expected_top_feature_index": 1,
            "checks": {
                "global_outcome_feature_selected": int(
                    legacy_scores[0]["feature_index"]
                )
                == 1,
                "diagnostic_exposes_different_target": int(
                    legacy_scores[0]["feature_index"]
                )
                != int(specific_scores[0]["feature_index"]),
            },
        },
        {
            "scenario_id": "shared_behavior_endpoint_preserved",
            "endpoint": shared_endpoint,
            "top_feature_index": int(shared_scores[0]["feature_index"]),
            "expected_top_feature_index": 1,
            "checks": {
                "behavior_feature_selected": int(
                    shared_scores[0]["feature_index"]
                )
                == 1,
                "shared_behavior_endpoint_preserved": shared_endpoint[
                    "model_outcome_label_used_for_selection"
                ]
                is True,
            },
        },
    ]
    for scenario in scenarios:
        scenario["status"] = (
            "pass" if all(scenario["checks"].values()) else "fail"
        )
    project_root = Path(__file__).resolve().parents[2]
    body = {
        "schema": "sch.vpd-compartment-sae-screening-synthetic-benchmark.v1",
        "status": "pass" if all(row["status"] == "pass" for row in scenarios) else "fail",
        "seed": seed,
        "design": {
            "pairs": 64,
            "rows": len(metadata),
            "categories": list(CATEGORIES),
            "features": 4,
            "prompt_families": 8,
            "paired_within_block_permutations": 127,
            "known_category_feature": 0,
            "known_behavior_outcome_feature": 1,
        },
        "scenarios": scenarios,
        "source": {
            "benchmark_runner_sha256": sha256_file(Path(__file__)),
            "focused_runner_sha256": sha256_file(
                project_root / "experiments/authz_mechinterp/run_focused_vpd_sae.py"
            ),
        },
        "scientific_boundary": {
            "synthetic_activations_only": True,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "does_not_establish_bonsai_category_features": True,
            "heldout_intervention_endpoint_remains_authorization_margin": True,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    body = dict(report)
    benchmark_id = str(body.pop("benchmark_id", ""))
    if content_sha256(body) != benchmark_id:
        raise ValueError("compartment SAE screening benchmark content address mismatch")
    expected = build_benchmark(seed=int(report["seed"]))
    if expected != report:
        raise ValueError("compartment SAE screening benchmark is not reproducible")
    if report.get("status") != "pass":
        raise ValueError("compartment SAE screening benchmark failed")
    return {
        "schema": "sch.vpd-compartment-sae-screening-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": benchmark_id,
        "scenarios": len(report["scenarios"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or verify the model-free compartment SAE screening benchmark."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--seed", type=int, default=290829)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_benchmark(args.verify)
    else:
        if args.output is None:
            parser.error("benchmark generation requires --output")
        result = build_benchmark(seed=args.seed)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
