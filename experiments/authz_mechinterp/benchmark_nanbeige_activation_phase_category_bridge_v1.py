from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.activation_category_cluster_count_selection_v2 import (
    select_category_cluster_counts_v2,
)
from experiments.authz_mechinterp.benchmark_nanbeige_activation_category_bridge_v1 import (
    PROJECTION_SEEDS,
    _load_source,
    _paired_signatures,
    _unit,
)
from experiments.authz_mechinterp.benchmark_nanbeige_pass_category_prompt_phase_transfer_v1 import (
    SITES,
    _load_prompt_capture,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    _project_effects,
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-nanbeige-activation-phase-category-bridge.v1"
MINIMUM_HELD_OUT_ACCURACY = 0.875
MINIMUM_LEAVE_ONE_EXAMPLE_OUT_COSINE = 0.80
MAXIMUM_EXACT_P_VALUE = 0.05


def _site_report(
    site: str,
    generation: dict[str, dict[int, np.ndarray]],
    prompt: dict[str, dict[int, np.ndarray]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    train_ids = [str(row["id"]) for row in records if row["split"] == "train"]
    eval_ids = [str(row["id"]) for row in records if row["split"] == "eval"]
    transitions = {
        (example_id, pass_index): (
            generation[example_id][pass_index] - prompt[example_id][pass_index]
        )
        for example_id in train_ids
        for pass_index in (0, 1)
    }
    mean_train_transition = np.mean(list(transitions.values()), axis=0)
    direction = _unit(mean_train_transition)
    held_out_transitions = [
        generation[example_id][pass_index] - prompt[example_id][pass_index]
        for example_id in eval_ids
        for pass_index in (0, 1)
    ]
    mean_held_out_transition = np.mean(held_out_transitions, axis=0)
    alignment_scale = max(
        float(
            np.mean([np.linalg.norm(value) for value in transitions.values()])
            * np.mean([np.linalg.norm(value) for value in held_out_transitions])
        ),
        1e-12,
    )
    observed_alignment = float(
        mean_train_transition @ mean_held_out_transition / alignment_scale
    )
    correct = 0
    margins = []
    for example_id in eval_ids:
        for pass_index in (0, 1):
            midpoint = 0.5 * (
                prompt[example_id][pass_index]
                + generation[example_id][pass_index]
            )
            for phase_index, values in enumerate(
                (
                    prompt[example_id][pass_index],
                    generation[example_id][pass_index],
                )
            ):
                residual = values - midpoint
                score = float(residual @ direction)
                predicted = 1 if score >= 0.0 else 0
                correct += int(predicted == phase_index)
                margins.append(abs(score) / max(float(np.linalg.norm(residual)), 1e-12))
    held_out_accuracy = correct / (4 * len(eval_ids))

    leave_one_example_out_cosines = []
    for omitted in train_ids:
        retained = [
            value for (example_id, _), value in transitions.items() if example_id != omitted
        ]
        leave_one_example_out_cosines.append(
            float(_unit(np.mean(retained, axis=0)) @ direction)
        )

    control_alignments = []
    for mask in range(1 << len(train_ids)):
        flips = {
            example_id: (-1.0 if mask & (1 << index) else 1.0)
            for index, example_id in enumerate(train_ids)
        }
        control = np.mean(
            [
                value * flips[example_id]
                for (example_id, _), value in transitions.items()
            ],
            axis=0,
        )
        control_alignments.append(
            float(control @ mean_held_out_transition / alignment_scale)
        )
    at_or_above = sum(
        value >= observed_alignment - 1e-12 for value in control_alignments
    )
    p_value = at_or_above / len(control_alignments)

    component_ids = []
    categories = []
    residuals = []
    for example_id in train_ids:
        for pass_index in (0, 1):
            midpoint = 0.5 * (
                prompt[example_id][pass_index]
                + generation[example_id][pass_index]
            )
            for phase, values in (
                ("prompt", prompt[example_id][pass_index]),
                ("generation", generation[example_id][pass_index]),
            ):
                component_ids.append(f"{example_id}__pass_{pass_index}__{phase}")
                categories.append(phase)
                residuals.append(_unit(values - midpoint))
    residual_matrix = np.asarray(residuals)
    selection = select_category_cluster_counts_v2(
        [
            _project_effects(residual_matrix, seed=seed, projection_dim=64)
            for seed in PROJECTION_SEEDS
        ],
        component_ids,
        categories,
        seed=250829,
        replicate_kinds=["projection"] * len(PROJECTION_SEEDS),
        maximum_clusters_per_category=2,
        minimum_cluster_size=4,
    )
    admitted = (
        held_out_accuracy >= MINIMUM_HELD_OUT_ACCURACY
        and min(leave_one_example_out_cosines)
        >= MINIMUM_LEAVE_ONE_EXAMPLE_OUT_COSINE
        and p_value <= MAXIMUM_EXACT_P_VALUE
    )
    return {
        "site": site,
        "train_examples": len(train_ids),
        "held_out_examples": len(eval_ids),
        "train_phase_pairs": len(transitions),
        "held_out_phase_pairs": len(held_out_transitions),
        "held_out_phase_accuracy": held_out_accuracy,
        "held_out_minimum_normalized_margin": min(margins),
        "leave_one_train_example_block_out_minimum_direction_cosine": min(
            leave_one_example_out_cosines
        ),
        "observed_train_to_held_out_phase_transition_alignment": observed_alignment,
        "exact_train_example_block_label_flip_control": {
            "configurations": len(control_alignments),
            "minimum_attainable_upper_tail_p_value": 1.0 / len(control_alignments),
            "minimum": min(control_alignments),
            "median": float(np.median(control_alignments)),
            "maximum": max(control_alignments),
            "at_or_above_observed": at_or_above,
            "exact_upper_tail_p_value": p_value,
        },
        "activation_phase_category_admission": "pass" if admitted else "fail",
        "within_phase_v2_selection": selection,
        "within_phase_allocation_is_operative": admitted,
    }


def build_report(generation_dir: Path, prompt_dir: Path) -> dict[str, Any]:
    generation_dir = generation_dir.resolve()
    prompt_dir = prompt_dir.resolve()
    generation_manifest, generation_examples = _load_source(generation_dir)
    prompt_manifest, prompt_records, prompt_signatures = _load_prompt_capture(
        prompt_dir
    )
    generation_by_id = {str(row["id"]): row for row in generation_examples}
    prompt_ids = {str(row["id"]) for row in prompt_records}
    if not prompt_ids.issubset(generation_by_id):
        raise ValueError("prompt records are not a subset of generation records")
    for record in prompt_records:
        source = generation_by_id[str(record["id"])]
        if record["split"] != source["split"] or record["family"] != source["family"]:
            raise ValueError("prompt/generation split or family differs")
    if generation_manifest["model"]["model_id"] != prompt_manifest["checkpoint"][
        "model_id"
    ] or generation_manifest["model"]["expected_revision"] != prompt_manifest[
        "checkpoint"
    ]["revision"]:
        raise ValueError("prompt/generation checkpoint claim differs")
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
            "common_examples": len(prompt_records),
            "common_train_examples": sum(
                row["split"] == "train" for row in prompt_records
            ),
            "common_held_out_examples": sum(
                row["split"] == "eval" for row in prompt_records
            ),
        },
        "design": {
            "observable_parent_category": "activation_phase_prompt_vs_generation",
            "pass_index_is_blocked_within_example": True,
            "discovery_split": "four source-designated train examples",
            "held_out_split": "four source-designated eval examples",
            "random_control": "exact_train_example_block_phase_label_flips",
            "exact_control_configurations": 16,
            "minimum_attainable_exact_p_value": 0.0625,
            "maximum_admissible_exact_p_value": MAXIMUM_EXACT_P_VALUE,
            "pack_is_underpowered_for_statistical_admission": True,
            "within_category_selector": "discreteness-aware_v2_over_eight_projections",
            "child_allocations_require_parent_admission": True,
        },
        "sites": sites,
        "summary": {
            "sites": len(sites),
            "activation_phase_category_admissions": sum(
                row["activation_phase_category_admission"] == "pass" for row in sites
            ),
            "screening_sites_with_held_out_accuracy_at_least_0_875": sum(
                row["held_out_phase_accuracy"] >= MINIMUM_HELD_OUT_ACCURACY
                for row in sites
            ),
            "screening_only_nontrivial_within_phase_allocations": sum(
                any(value > 1 for value in row["within_phase_v2_selection"]["allocation"].values())
                for row in sites
            ),
            "operative_nontrivial_within_phase_allocations": sum(
                row["within_phase_allocation_is_operative"]
                and any(
                    value > 1
                    for value in row["within_phase_v2_selection"]["allocation"].values()
                )
                for row in sites
            ),
        },
        "scientific_boundary": {
            "new_local_analysis_of_recovered_empirical_arrays": True,
            "new_model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "paid_compute_authorized": False,
            "activation_phase_category_is_screening_only": True,
            "statistical_admission_impossible_at_current_exact_control_resolution": True,
            "within_phase_child_allocations_are_inoperative": True,
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
        raise ValueError("activation-phase bridge content address differs")
    expected = build_report(
        Path(observed["sources"]["generation"]["path"]),
        Path(observed["sources"]["prompt"]["path"]),
    )
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("activation-phase bridge report does not reproduce")
    return {
        "schema": "sch.vpd-nanbeige-activation-phase-category-bridge-verification.v1",
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
