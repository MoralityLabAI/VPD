from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.benchmark_nanbeige_activation_category_bridge_v1 import (
    _accuracy_and_margins,
    _load_source,
    _paired_signatures,
    _unit,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-nanbeige-pass-category-shared-axis.v1"
MINIMUM_HELD_OUT_ACCURACY = 0.875
MINIMUM_PACK_DIRECTION_COSINE = 0.80
MAXIMUM_EXACT_P_VALUE = 0.05


def _ids(examples: list[dict[str, Any]], split: str) -> list[str]:
    return [str(row["id"]) for row in examples if row["split"] == split]


def _transitions(
    signatures: dict[str, dict[int, np.ndarray]], example_ids: list[str]
) -> dict[str, np.ndarray]:
    return {
        value: signatures[value][1] - signatures[value][0]
        for value in example_ids
    }


def _control_directions(
    transitions: dict[str, np.ndarray],
    examples: list[dict[str, Any]],
) -> list[np.ndarray]:
    pair_by_id = {str(row["id"]): str(row["pair_id"]) for row in examples}
    pairs = sorted({pair_by_id[value] for value in transitions})
    if len(pairs) > 12:
        raise ValueError("exact pair-block sweep exceeds local cap")
    directions = []
    for mask in range(1 << len(pairs)):
        flips = {
            pair: (-1.0 if mask & (1 << index) else 1.0)
            for index, pair in enumerate(pairs)
        }
        mean = np.mean(
            [value * flips[pair_by_id[key]] for key, value in transitions.items()],
            axis=0,
        )
        norm = float(np.linalg.norm(mean))
        directions.append(mean / norm if norm > 1e-12 else np.zeros_like(mean))
    return directions


def _site_report(
    site: str,
    arithmetic_signatures: dict[str, dict[int, np.ndarray]],
    arithmetic_examples: list[dict[str, Any]],
    reasoning_signatures: dict[str, dict[int, np.ndarray]],
    reasoning_examples: list[dict[str, Any]],
) -> dict[str, Any]:
    arithmetic_train_ids = _ids(arithmetic_examples, "train")
    reasoning_train_ids = _ids(reasoning_examples, "train")
    arithmetic_eval_ids = _ids(arithmetic_examples, "eval")
    reasoning_eval_ids = _ids(reasoning_examples, "eval")
    arithmetic_train = _transitions(arithmetic_signatures, arithmetic_train_ids)
    reasoning_train = _transitions(reasoning_signatures, reasoning_train_ids)
    arithmetic_direction = _unit(np.mean(list(arithmetic_train.values()), axis=0))
    reasoning_direction = _unit(np.mean(list(reasoning_train.values()), axis=0))
    shared_direction = _unit(0.5 * arithmetic_direction + 0.5 * reasoning_direction)

    arithmetic_accuracy, arithmetic_margins = _accuracy_and_margins(
        shared_direction, arithmetic_eval_ids, arithmetic_signatures
    )
    reasoning_accuracy, reasoning_margins = _accuracy_and_margins(
        shared_direction, reasoning_eval_ids, reasoning_signatures
    )
    native_arithmetic_accuracy, native_arithmetic_margins = _accuracy_and_margins(
        arithmetic_direction, arithmetic_eval_ids, arithmetic_signatures
    )
    native_reasoning_accuracy, native_reasoning_margins = _accuracy_and_margins(
        reasoning_direction, reasoning_eval_ids, reasoning_signatures
    )
    cosine_to_arithmetic = float(shared_direction @ arithmetic_direction)
    cosine_to_reasoning = float(shared_direction @ reasoning_direction)

    arithmetic_eval_direction = _unit(
        np.mean(
            list(_transitions(arithmetic_signatures, arithmetic_eval_ids).values()),
            axis=0,
        )
    )
    reasoning_eval_direction = _unit(
        np.mean(
            list(_transitions(reasoning_signatures, reasoning_eval_ids).values()),
            axis=0,
        )
    )
    observed_score = float(
        0.5 * (shared_direction @ arithmetic_eval_direction)
        + 0.5 * (shared_direction @ reasoning_eval_direction)
    )
    arithmetic_controls = _control_directions(
        arithmetic_train, arithmetic_examples
    )
    reasoning_controls = _control_directions(reasoning_train, reasoning_examples)
    control_scores: list[float] = []
    for arithmetic_control in arithmetic_controls:
        for reasoning_control in reasoning_controls:
            combined = 0.5 * arithmetic_control + 0.5 * reasoning_control
            norm = float(np.linalg.norm(combined))
            if norm <= 1e-12:
                control_scores.append(0.0)
                continue
            direction = combined / norm
            control_scores.append(
                float(
                    0.5 * (direction @ arithmetic_eval_direction)
                    + 0.5 * (direction @ reasoning_eval_direction)
                )
            )
    at_or_above = sum(value >= observed_score - 1e-12 for value in control_scores)
    p_value = at_or_above / len(control_scores)
    admitted = (
        min(arithmetic_accuracy, reasoning_accuracy) >= MINIMUM_HELD_OUT_ACCURACY
        and min(cosine_to_arithmetic, cosine_to_reasoning)
        >= MINIMUM_PACK_DIRECTION_COSINE
        and p_value <= MAXIMUM_EXACT_P_VALUE
    )
    return {
        "site": site,
        "source_train_direction_cosine": float(
            arithmetic_direction @ reasoning_direction
        ),
        "shared_axis_cosine_to_pack_train_directions": {
            "arithmetic": cosine_to_arithmetic,
            "reasoning": cosine_to_reasoning,
            "minimum": min(cosine_to_arithmetic, cosine_to_reasoning),
        },
        "shared_axis_held_out": {
            "arithmetic": {
                "pass_accuracy": arithmetic_accuracy,
                "minimum_normalized_margin": min(arithmetic_margins),
            },
            "reasoning": {
                "pass_accuracy": reasoning_accuracy,
                "minimum_normalized_margin": min(reasoning_margins),
            },
        },
        "native_pack_axis_held_out": {
            "arithmetic": {
                "pass_accuracy": native_arithmetic_accuracy,
                "minimum_normalized_margin": min(native_arithmetic_margins),
            },
            "reasoning": {
                "pass_accuracy": native_reasoning_accuracy,
                "minimum_normalized_margin": min(native_reasoning_margins),
            },
        },
        "exact_joint_pair_block_label_flip_control": {
            "arithmetic_pair_configurations": len(arithmetic_controls),
            "reasoning_pair_configurations": len(reasoning_controls),
            "joint_configurations": len(control_scores),
            "observed_equal_pack_held_out_alignment": observed_score,
            "minimum": min(control_scores),
            "median": float(np.median(control_scores)),
            "maximum": max(control_scores),
            "at_or_above_observed": at_or_above,
            "exact_upper_tail_p_value": p_value,
        },
        "shared_axis_admission": "pass" if admitted else "fail",
    }


def build_report(arithmetic_dir: Path, reasoning_dir: Path) -> dict[str, Any]:
    arithmetic_dir = arithmetic_dir.resolve()
    reasoning_dir = reasoning_dir.resolve()
    arithmetic_manifest, arithmetic_examples = _load_source(arithmetic_dir)
    reasoning_manifest, reasoning_examples = _load_source(reasoning_dir)
    for field in ("model_id", "expected_revision", "dtype"):
        if arithmetic_manifest["model"][field] != reasoning_manifest["model"][field]:
            raise ValueError(f"shared-axis source mismatch: {field}")
    arithmetic_sites = [str(value) for value in arithmetic_manifest["sites"]]
    reasoning_sites = [str(value) for value in reasoning_manifest["sites"]]
    if arithmetic_sites != reasoning_sites:
        raise ValueError("shared-axis sites differ")
    arithmetic_families = sorted(
        {str(row["family"]) for row in arithmetic_examples}
    )
    reasoning_families = sorted(
        {str(row["family"]) for row in reasoning_examples}
    )
    if set(arithmetic_families) & set(reasoning_families):
        raise ValueError("shared-axis source family sets overlap")
    arithmetic_signatures = _paired_signatures(
        arithmetic_dir, arithmetic_manifest, arithmetic_examples
    )
    reasoning_signatures = _paired_signatures(
        reasoning_dir, reasoning_manifest, reasoning_examples
    )
    sites = [
        _site_report(
            site,
            arithmetic_signatures[site],
            arithmetic_examples,
            reasoning_signatures[site],
            reasoning_examples,
        )
        for site in arithmetic_sites
    ]
    body = {
        "schema": SCHEMA,
        "status": "pass",
        "sources": {
            "arithmetic": {
                "path": arithmetic_dir.as_posix(),
                "manifest_sha256": sha256_file(arithmetic_dir / "manifest.json"),
                "examples_sha256": sha256_file(arithmetic_dir / "examples.json"),
                "families": arithmetic_families,
            },
            "reasoning": {
                "path": reasoning_dir.as_posix(),
                "manifest_sha256": sha256_file(reasoning_dir / "manifest.json"),
                "examples_sha256": sha256_file(reasoning_dir / "examples.json"),
                "families": reasoning_families,
            },
            "model_id": arithmetic_manifest["model"]["model_id"],
            "expected_revision": arithmetic_manifest["model"]["expected_revision"],
            "dtype": arithmetic_manifest["model"]["dtype"],
            "revision_verified_by_both_sources": all(
                bool(value["model"]["revision_verified"])
                for value in (arithmetic_manifest, reasoning_manifest)
            ),
        },
        "design": {
            "axis": "unit_norm_equal_pack_bisector_of_training_mean_pass_transitions",
            "equal_pack_weighting": True,
            "evaluation": "source_designated_held_out_examples_only",
            "task_family_sets_must_be_disjoint": True,
            "random_control": "exact_joint_pair_block_pass_label_flips_across_both_training_packs",
            "admission_thresholds": {
                "minimum_per_pack_held_out_pass_accuracy": MINIMUM_HELD_OUT_ACCURACY,
                "minimum_shared_axis_cosine_to_each_pack_train_direction": (
                    MINIMUM_PACK_DIRECTION_COSINE
                ),
                "maximum_exact_upper_tail_p_value": MAXIMUM_EXACT_P_VALUE,
            },
        },
        "sites": sites,
        "summary": {
            "sites": len(sites),
            "shared_axis_admissions": sum(
                row["shared_axis_admission"] == "pass" for row in sites
            ),
            "minimum_shared_axis_held_out_pass_accuracy": min(
                row["shared_axis_held_out"][pack]["pass_accuracy"]
                for row in sites
                for pack in ("arithmetic", "reasoning")
            ),
            "minimum_shared_axis_cosine_to_pack_train_direction": min(
                row["shared_axis_cosine_to_pack_train_directions"]["minimum"]
                for row in sites
            ),
            "maximum_exact_upper_tail_p_value": max(
                row["exact_joint_pair_block_label_flip_control"][
                    "exact_upper_tail_p_value"
                ]
                for row in sites
            ),
        },
        "scientific_boundary": {
            "new_local_analysis_of_recovered_empirical_arrays": True,
            "new_model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "paid_compute_authorized": False,
            "source_checkpoint_revision_independently_verified": False,
            "pooled_training_uses_both_task_packs": True,
            "held_out_evaluation_does_not_establish_third_domain_transfer": True,
            "shared_axis_is_not_semantic_interpretation": True,
            "does_not_authorize_sae_training": True,
            "execution_authorized": False,
        },
        "provenance": {"runner_sha256": sha256_file(Path(__file__))},
    }
    return {**body, "report_id": content_sha256(body)}


def verify_report(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    report_id = str(body.pop("report_id", ""))
    if content_sha256(body) != report_id:
        raise ValueError("shared-axis report content address differs")
    expected = build_report(
        Path(observed["sources"]["arithmetic"]["path"]),
        Path(observed["sources"]["reasoning"]["path"]),
    )
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("shared-axis report does not reproduce")
    return {
        "schema": "sch.vpd-nanbeige-pass-category-shared-axis-verification.v1",
        "status": "pass",
        "report_id": report_id,
        **observed["summary"],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arithmetic-dir", type=Path)
    parser.add_argument("--reasoning-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_report(args.verify)
    else:
        if args.arithmetic_dir is None or args.reasoning_dir is None or args.output is None:
            parser.error(
                "--arithmetic-dir, --reasoning-dir, and --output are required unless verifying"
            )
        result = build_report(args.arithmetic_dir, args.reasoning_dir)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
