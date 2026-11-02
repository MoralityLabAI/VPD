from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.activation_category_cluster_count_selection_v2 import (
    select_category_cluster_counts_v2,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    _project_effects,
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-nanbeige-activation-category-bridge.v1"
PROJECTION_SEEDS = (17, 29, 41, 53, 67, 79, 97, 109)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("activation transition has zero norm")
    return vector / norm


def _load_source(source_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    examples = json.loads((source_dir / "examples.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "vpd.multipass.calibration.v1" or manifest.get(
        "status"
    ) != "calibration_complete":
        raise ValueError("unexpected Nanbeige calibration manifest")
    if manifest.get("model", {}).get("kind") != "local_pretrained_checkpoint":
        raise ValueError("Nanbeige bridge requires a pretrained checkpoint receipt")
    expected_examples = int(manifest.get("train_examples", 0)) + int(
        manifest.get("eval_examples", 0)
    )
    if (
        not isinstance(examples, list)
        or len(examples) != expected_examples
        or int(manifest.get("train_examples", 0)) < 8
        or int(manifest.get("eval_examples", 0)) < 4
    ):
        raise ValueError("Nanbeige bridge split counts are absent or underpowered")
    artifacts = manifest.get("artifacts", {})
    required = {"examples.json", *manifest.get("baseline_archives", {}).values()}
    for name in sorted(required):
        if name not in artifacts or sha256_file(source_dir / name) != artifacts[name]:
            raise ValueError(f"Nanbeige source artifact hash differs: {name}")
    return manifest, examples


def _paired_signatures(
    source_dir: Path,
    manifest: dict[str, Any],
    examples: list[dict[str, Any]],
) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    signatures: dict[str, dict[str, dict[int, np.ndarray]]] = {
        str(site): {} for site in manifest["sites"]
    }
    archives = manifest["baseline_archives"]
    for example in examples:
        example_id = str(example["id"])
        archive = np.load(source_dir / str(archives[example_id]), allow_pickle=False)
        for site in signatures:
            by_pass: dict[int, np.ndarray] = {}
            for pass_index in (0, 1):
                values = np.asarray(
                    archive[f"{site}__pass_{pass_index}"], dtype=np.float64
                )
                if values.ndim != 2 or not np.all(np.isfinite(values)):
                    raise ValueError("Nanbeige activation array is invalid")
                by_pass[pass_index] = values.mean(axis=0)
            signatures[site][example_id] = by_pass
    return signatures


def _accuracy_and_margins(
    direction: np.ndarray,
    example_ids: list[str],
    site_signatures: dict[str, dict[int, np.ndarray]],
) -> tuple[float, list[float]]:
    correct = 0
    margins: list[float] = []
    for example_id in example_ids:
        values = site_signatures[example_id]
        midpoint = 0.5 * (values[0] + values[1])
        for pass_index in (0, 1):
            residual = values[pass_index] - midpoint
            score = float(residual @ direction)
            predicted = 1 if score >= 0.0 else 0
            correct += int(predicted == pass_index)
            margins.append(abs(score) / max(float(np.linalg.norm(residual)), 1e-12))
    return correct / (2 * len(example_ids)), margins


def _site_report(
    site: str,
    site_signatures: dict[str, dict[int, np.ndarray]],
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    train = [str(row["id"]) for row in examples if row["split"] == "train"]
    held_out = [str(row["id"]) for row in examples if row["split"] == "eval"]
    pair_by_example = {str(row["id"]): str(row["pair_id"]) for row in examples}
    train_pairs = sorted({pair_by_example[example_id] for example_id in train})
    if len(train_pairs) > 12:
        raise ValueError("exact pair-block randomization exceeds the local sweep cap")
    transitions = {
        example_id: site_signatures[example_id][1]
        - site_signatures[example_id][0]
        for example_id in train
    }
    mean_train_transition = np.mean([transitions[value] for value in train], axis=0)
    direction = _unit(mean_train_transition)
    mean_held_out_transition = np.mean(
        [
            site_signatures[value][1] - site_signatures[value][0]
            for value in held_out
        ],
        axis=0,
    )
    alignment_scale = max(
        float(
            np.mean([np.linalg.norm(transitions[value]) for value in train])
            * np.mean(
                [
                    np.linalg.norm(
                        site_signatures[value][1] - site_signatures[value][0]
                    )
                    for value in held_out
                ]
            )
        ),
        1e-12,
    )
    observed_alignment = float(
        mean_train_transition @ mean_held_out_transition / alignment_scale
    )
    held_out_accuracy, margins = _accuracy_and_margins(
        direction, held_out, site_signatures
    )
    leave_one_pair_out_cosines: list[float] = []
    for omitted_pair in train_pairs:
        retained = [
            transitions[value]
            for value in train
            if pair_by_example[value] != omitted_pair
        ]
        leave_one_pair_out_cosines.append(float(_unit(np.mean(retained, axis=0)) @ direction))

    random_control_accuracies: list[float] = []
    random_control_alignments: list[float] = []
    for mask in range(1 << len(train_pairs)):
        flips = {
            pair: (-1.0 if mask & (1 << index) else 1.0)
            for index, pair in enumerate(train_pairs)
        }
        control = np.mean(
            [transitions[value] * flips[pair_by_example[value]] for value in train],
            axis=0,
        )
        random_control_alignments.append(
            float(control @ mean_held_out_transition / alignment_scale)
        )
        if float(np.linalg.norm(control)) <= 1e-12:
            random_control_accuracies.append(0.5)
        else:
            accuracy, _ = _accuracy_and_margins(
                _unit(control), held_out, site_signatures
            )
            random_control_accuracies.append(accuracy)
    controls_at_or_above_observed = sum(
        value >= observed_alignment - 1e-12 for value in random_control_alignments
    )
    random_p_value = controls_at_or_above_observed / len(random_control_alignments)

    component_ids: list[str] = []
    categories: list[str] = []
    residuals: list[np.ndarray] = []
    for example_id in train:
        values = site_signatures[example_id]
        midpoint = 0.5 * (values[0] + values[1])
        for pass_index in (0, 1):
            component_ids.append(f"{example_id}__pass_{pass_index}")
            categories.append(f"pass_{pass_index}")
            residuals.append(_unit(values[pass_index] - midpoint))
    residual_matrix = np.asarray(residuals)
    replicate_matrices = [
        _project_effects(residual_matrix, seed=seed, projection_dim=64)
        for seed in PROJECTION_SEEDS
    ]
    selection = select_category_cluster_counts_v2(
        replicate_matrices,
        component_ids,
        categories,
        seed=250829,
        replicate_kinds=["projection"] * len(PROJECTION_SEEDS),
        maximum_clusters_per_category=3,
        minimum_cluster_size=4,
    )
    admitted = (
        held_out_accuracy >= 0.875
        and random_p_value <= 0.05
        and min(leave_one_pair_out_cosines) >= 0.80
    )
    return {
        "site": site,
        "transition_category_admission": "pass" if admitted else "fail",
        "train_examples": len(train),
        "held_out_examples": len(held_out),
        "held_out_pass_accuracy": held_out_accuracy,
        "held_out_minimum_normalized_margin": min(margins),
        "leave_one_train_pair_block_out_minimum_direction_cosine": min(
            leave_one_pair_out_cosines
        ),
        "exact_pair_block_label_flip_controls": len(random_control_accuracies),
        "random_control_accuracy_histogram": dict(
            sorted(Counter(str(value) for value in random_control_accuracies).items())
        ),
        "observed_train_to_held_out_transition_alignment": observed_alignment,
        "random_control_transition_alignment": {
            "minimum": min(random_control_alignments),
            "median": float(np.median(random_control_alignments)),
            "maximum": max(random_control_alignments),
            "at_or_above_observed": controls_at_or_above_observed,
            "exact_upper_tail_p_value": random_p_value,
        },
        "within_pass_v2_selection": selection,
    }


def build_report(source_dir: Path) -> dict[str, Any]:
    manifest, examples = _load_source(source_dir)
    signatures = _paired_signatures(source_dir, manifest, examples)
    sites = [
        _site_report(site, signatures[site], examples)
        for site in [str(value) for value in manifest["sites"]]
    ]
    train_families = sorted(
        {str(row["family"]) for row in examples if row["split"] == "train"}
    )
    held_out_families = sorted(
        {str(row["family"]) for row in examples if row["split"] == "eval"}
    )
    body = {
        "schema": SCHEMA,
        "status": "pass",
        "source": {
            "path": str(source_dir.resolve()),
            "manifest_sha256": sha256_file(source_dir / "manifest.json"),
            "examples_sha256": sha256_file(source_dir / "examples.json"),
            "model_id": manifest["model"]["model_id"],
            "expected_revision": manifest["model"]["expected_revision"],
            "revision_verified_by_source": bool(manifest["model"]["revision_verified"]),
            "model_kind": manifest["model"]["kind"],
            "dtype": manifest["model"]["dtype"],
            "runner_sha256": sha256_file(Path(__file__)),
        },
        "design": {
            "category": "recurrent_pass_index_0_vs_1",
            "signature": "per_example_mean_hidden_state_centered_at_paired_pass_midpoint",
            "discovery_split": "source-designated train examples",
            "held_out_split": "source-designated eval examples",
            "train_families": train_families,
            "held_out_families": held_out_families,
            "overlapping_families": sorted(
                set(train_families).intersection(held_out_families)
            ),
            "random_control": "exact train pair-block pass-label flips scored by raw held-out transition alignment",
            "stability": "leave_one_train_pair_block_out transition-direction cosine",
            "within_category_selector": "discreteness-aware v2 over eight projections",
            "admission_thresholds": {
                "minimum_held_out_pass_accuracy": 0.875,
                "maximum_random_control_upper_tail_p_value": 0.05,
                "minimum_leave_one_pair_out_direction_cosine": 0.80,
            },
        },
        "sites": sites,
        "summary": {
            "sites": len(sites),
            "transition_category_admissions": sum(
                row["transition_category_admission"] == "pass" for row in sites
            ),
            "within_pass_allocations": {
                row["site"]: row["within_pass_v2_selection"]["allocation"]
                for row in sites
            },
        },
        "scientific_boundary": {
            "new_model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "stored_empirical_pretrained_model_activations_used": True,
            "small_source_specific_pack": True,
            "family_disjoint": bool(manifest["family_disjoint"]),
            "model_revision_was_not_verified_by_source": not bool(
                manifest["model"]["revision_verified"]
            ),
            "within_pass_cluster_membership_is_screening_only": True,
            "does_not_validate_bonsai_authorization_categories": True,
            "does_not_authorize_sae_training_or_model_release": True,
        },
    }
    return {**body, "report_id": content_sha256(body)}


def verify_report(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    report_id = str(body.pop("report_id", ""))
    if content_sha256(body) != report_id:
        raise ValueError("Nanbeige activation-category bridge content address differs")
    expected = build_report(Path(observed["source"]["path"]))
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("Nanbeige activation-category bridge does not reproduce")
    return {
        "schema": "sch.vpd-nanbeige-activation-category-bridge-verification.v1",
        "status": "pass",
        "report_id": report_id,
        "sites": int(observed["summary"]["sites"]),
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_report(args.verify)
    else:
        if args.source_dir is None or args.output is None:
            parser.error("--source-dir and --output are required unless verifying")
        result = build_report(args.source_dir)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
