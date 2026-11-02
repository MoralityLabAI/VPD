from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.benchmark_nanbeige_activation_category_bridge_v1 import (
    _load_source,
    _paired_signatures,
    _unit,
)
from experiments.authz_mechinterp.benchmark_nanbeige_pass_category_prompt_phase_transfer_v1 import (
    SITES,
    _load_prompt_capture,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-nanbeige-activation-phase-category-cross-validation.v1"
MINIMUM_CROSS_VALIDATED_ACCURACY = 0.875
MINIMUM_FOLD_DIRECTION_COSINE = 0.80
MAXIMUM_EXACT_P_VALUE = 0.05


def _cross_validated_score(
    transitions: dict[tuple[str, int], np.ndarray],
    family_by_id: dict[str, str],
    flips: dict[str, float],
) -> tuple[float, float, list[np.ndarray]]:
    families = sorted(set(family_by_id.values()))
    scores: list[float] = []
    correct = 0
    total = 0
    fold_directions = []
    for held_out_family in families:
        train = [
            value * flips[example_id]
            for (example_id, _), value in transitions.items()
            if family_by_id[example_id] != held_out_family
        ]
        direction = _unit(np.mean(train, axis=0))
        fold_directions.append(direction)
        for (example_id, _), value in transitions.items():
            if family_by_id[example_id] != held_out_family:
                continue
            target = value * flips[example_id]
            normalized_score = float(
                direction @ target / max(float(np.linalg.norm(target)), 1e-12)
            )
            scores.append(normalized_score)
            correct += int(normalized_score > 0.0)
            total += 1
    return float(np.mean(scores)), correct / total, fold_directions


def _site_report(
    site: str,
    generation: dict[str, dict[int, np.ndarray]],
    prompt: dict[str, dict[int, np.ndarray]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    example_ids = [str(row["id"]) for row in records]
    family_by_id = {str(row["id"]): str(row["family"]) for row in records}
    transitions = {
        (example_id, pass_index): (
            generation[example_id][pass_index] - prompt[example_id][pass_index]
        )
        for example_id in example_ids
        for pass_index in (0, 1)
    }
    observed_flips = {example_id: 1.0 for example_id in example_ids}
    observed_score, observed_accuracy, fold_directions = _cross_validated_score(
        transitions, family_by_id, observed_flips
    )
    full_direction = _unit(np.mean(list(transitions.values()), axis=0))
    fold_cosines = [float(direction @ full_direction) for direction in fold_directions]

    control_scores = []
    control_accuracies = []
    for mask in range(1 << len(example_ids)):
        flips = {
            example_id: (-1.0 if mask & (1 << index) else 1.0)
            for index, example_id in enumerate(example_ids)
        }
        score, accuracy, _ = _cross_validated_score(
            transitions, family_by_id, flips
        )
        control_scores.append(score)
        control_accuracies.append(accuracy)
    at_or_above = sum(value >= observed_score - 1e-12 for value in control_scores)
    p_value = at_or_above / len(control_scores)
    provisionally_supported = (
        observed_accuracy >= MINIMUM_CROSS_VALIDATED_ACCURACY
        and min(fold_cosines) >= MINIMUM_FOLD_DIRECTION_COSINE
        and p_value <= MAXIMUM_EXACT_P_VALUE
    )
    return {
        "site": site,
        "semantic_family_folds": len(set(family_by_id.values())),
        "examples": len(example_ids),
        "phase_pairs": len(transitions),
        "leave_one_family_out_phase_accuracy": observed_accuracy,
        "leave_one_family_out_mean_normalized_phase_alignment": observed_score,
        "leave_one_family_out_minimum_direction_cosine_to_full_direction": min(
            fold_cosines
        ),
        "exact_example_block_phase_label_permutation": {
            "configurations": len(control_scores),
            "minimum": min(control_scores),
            "median": float(np.median(control_scores)),
            "maximum": max(control_scores),
            "at_or_above_observed": at_or_above,
            "exact_upper_tail_p_value": p_value,
            "control_accuracy_minimum": min(control_accuracies),
            "control_accuracy_median": float(np.median(control_accuracies)),
            "control_accuracy_maximum": max(control_accuracies),
        },
        "provisional_cross_validated_phase_category_support": (
            "pass" if provisionally_supported else "fail"
        ),
        "operative_phase_category_admission": "fail",
    }


def build_report(generation_dir: Path, prompt_dir: Path) -> dict[str, Any]:
    generation_dir = generation_dir.resolve()
    prompt_dir = prompt_dir.resolve()
    generation_manifest, generation_examples = _load_source(generation_dir)
    prompt_manifest, prompt_records, prompt_signatures = _load_prompt_capture(
        prompt_dir
    )
    generation_by_id = {str(row["id"]): row for row in generation_examples}
    if any(str(row["id"]) not in generation_by_id for row in prompt_records):
        raise ValueError("prompt record is absent from generation capture")
    for record in prompt_records:
        source = generation_by_id[str(record["id"])]
        if source["family"] != record["family"]:
            raise ValueError("prompt/generation family differs")
    families = sorted({str(row["family"]) for row in prompt_records})
    if len(families) < 4:
        raise ValueError("phase cross-validation requires at least four families")
    generation_signatures = _paired_signatures(
        generation_dir, generation_manifest, generation_examples
    )
    sites = [
        _site_report(
            site,
            generation_signatures[site],
            prompt_signatures[site],
            prompt_records,
        )
        for site in SITES
    ]
    body = {
        "schema": SCHEMA,
        "status": "pass",
        "sources": {
            "generation": {
                "path": generation_dir.as_posix(),
                "manifest_sha256": sha256_file(generation_dir / "manifest.json"),
            },
            "prompt": {
                "path": prompt_dir.as_posix(),
                "manifest_sha256": sha256_file(prompt_dir / "manifest.json"),
            },
            "examples": len(prompt_records),
            "semantic_families": families,
        },
        "design": {
            "observable_category": "activation_phase_prompt_vs_generation",
            "validation": "leave_one_semantic_family_out_cross_validation",
            "pass_indices_are_blocked_within_example": True,
            "permutation_unit": "example_block_phase_label",
            "exact_permutations": 1 << len(prompt_records),
            "admission_thresholds": {
                "minimum_cross_validated_phase_accuracy": (
                    MINIMUM_CROSS_VALIDATED_ACCURACY
                ),
                "minimum_fold_direction_cosine_to_full_direction": (
                    MINIMUM_FOLD_DIRECTION_COSINE
                ),
                "maximum_exact_upper_tail_p_value": MAXIMUM_EXACT_P_VALUE,
            },
            "provisional_support_is_not_operative_admission": True,
            "fresh_untouched_holdout_still_required": True,
        },
        "sites": sites,
        "summary": {
            "sites": len(sites),
            "provisional_cross_validated_phase_category_supports": sum(
                row["provisional_cross_validated_phase_category_support"] == "pass"
                for row in sites
            ),
            "operative_phase_category_admissions": 0,
            "minimum_cross_validated_phase_accuracy": min(
                row["leave_one_family_out_phase_accuracy"] for row in sites
            ),
            "maximum_exact_upper_tail_p_value": max(
                row["exact_example_block_phase_label_permutation"][
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
            "cross_validation_reuses_all_examples_across_folds": True,
            "no_fully_untouched_final_holdout": True,
            "provisional_support_is_not_category_admission": True,
            "does_not_authorize_phase_compartment_sae_training": True,
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
        raise ValueError("phase cross-validation content address differs")
    expected = build_report(
        Path(observed["sources"]["generation"]["path"]),
        Path(observed["sources"]["prompt"]["path"]),
    )
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("phase cross-validation report does not reproduce")
    return {
        "schema": "sch.vpd-nanbeige-activation-phase-category-cross-validation-verification.v1",
        "status": "pass",
        "report_id": report_id,
        **observed["summary"],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", type=Path)
    parser.add_argument("--prompt-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_report(args.verify)
    else:
        if args.generation_dir is None or args.prompt_dir is None or args.output is None:
            parser.error("both source directories and --output are required unless verifying")
        result = build_report(args.generation_dir, args.prompt_dir)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
