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


SCHEMA = "sch.vpd-nanbeige-pass-category-cross-pack-transfer.v1"
MINIMUM_CROSS_PACK_ACCURACY = 0.875
MAXIMUM_EXACT_P_VALUE = 0.05
MINIMUM_CROSS_PACK_DIRECTION_COSINE = 0.80


def _ids(examples: list[dict[str, Any]], split: str) -> list[str]:
    return [str(row["id"]) for row in examples if row["split"] == split]


def _mean_transition(
    signatures: dict[str, dict[int, np.ndarray]], example_ids: list[str]
) -> np.ndarray:
    return np.mean(
        [signatures[value][1] - signatures[value][0] for value in example_ids],
        axis=0,
    )


def _cross_score(
    source_signatures: dict[str, dict[int, np.ndarray]],
    source_examples: list[dict[str, Any]],
    target_signatures: dict[str, dict[int, np.ndarray]],
    target_examples: list[dict[str, Any]],
) -> dict[str, Any]:
    source_train = _ids(source_examples, "train")
    target_eval = _ids(target_examples, "eval")
    pair_by_source = {
        str(row["id"]): str(row["pair_id"]) for row in source_examples
    }
    source_pairs = sorted({pair_by_source[value] for value in source_train})
    if len(source_pairs) > 12:
        raise ValueError("exact source pair-block sweep exceeds local cap")
    transitions = {
        value: source_signatures[value][1] - source_signatures[value][0]
        for value in source_train
    }
    source_mean = np.mean(list(transitions.values()), axis=0)
    direction = _unit(source_mean)
    target_mean = _mean_transition(target_signatures, target_eval)
    alignment_scale = max(
        float(
            np.mean([np.linalg.norm(value) for value in transitions.values()])
            * np.mean(
                [
                    np.linalg.norm(
                        target_signatures[value][1]
                        - target_signatures[value][0]
                    )
                    for value in target_eval
                ]
            )
        ),
        1e-12,
    )
    observed_alignment = float(source_mean @ target_mean / alignment_scale)
    accuracy, margins = _accuracy_and_margins(
        direction, target_eval, target_signatures
    )

    control_alignments: list[float] = []
    control_accuracies: list[float] = []
    for mask in range(1 << len(source_pairs)):
        flips = {
            pair: (-1.0 if mask & (1 << index) else 1.0)
            for index, pair in enumerate(source_pairs)
        }
        control = np.mean(
            [
                transitions[value] * flips[pair_by_source[value]]
                for value in source_train
            ],
            axis=0,
        )
        control_alignments.append(float(control @ target_mean / alignment_scale))
        if float(np.linalg.norm(control)) <= 1e-12:
            control_accuracies.append(0.5)
        else:
            control_accuracy, _ = _accuracy_and_margins(
                _unit(control), target_eval, target_signatures
            )
            control_accuracies.append(control_accuracy)
    p_value = sum(
        value >= observed_alignment - 1e-12 for value in control_alignments
    ) / len(control_alignments)
    return {
        "source_train_examples": len(source_train),
        "source_train_pair_blocks": len(source_pairs),
        "target_held_out_examples": len(target_eval),
        "held_out_pass_accuracy": accuracy,
        "held_out_minimum_normalized_margin": min(margins),
        "observed_source_train_to_target_held_out_transition_alignment": (
            observed_alignment
        ),
        "exact_source_pair_block_label_flip_controls": len(control_alignments),
        "random_control_transition_alignment": {
            "minimum": min(control_alignments),
            "median": float(np.median(control_alignments)),
            "maximum": max(control_alignments),
            "at_or_above_observed": sum(
                value >= observed_alignment - 1e-12
                for value in control_alignments
            ),
            "exact_upper_tail_p_value": p_value,
        },
        "random_control_accuracy": {
            "minimum": min(control_accuracies),
            "median": float(np.median(control_accuracies)),
            "maximum": max(control_accuracies),
        },
        "directional_gate": (
            "pass"
            if accuracy >= MINIMUM_CROSS_PACK_ACCURACY
            and p_value <= MAXIMUM_EXACT_P_VALUE
            else "fail"
        ),
    }


def _source_receipt(
    label: str,
    path: Path,
    manifest: dict[str, Any],
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "label": label,
        "path": path.resolve().as_posix(),
        "manifest_sha256": sha256_file(path / "manifest.json"),
        "examples_sha256": sha256_file(path / "examples.json"),
        "model_id": manifest["model"]["model_id"],
        "expected_revision": manifest["model"]["expected_revision"],
        "revision_verified_by_source": bool(manifest["model"]["revision_verified"]),
        "dtype": manifest["model"]["dtype"],
        "sites": [str(value) for value in manifest["sites"]],
        "train_examples": int(manifest["train_examples"]),
        "held_out_examples": int(manifest["eval_examples"]),
        "families": sorted({str(row["family"]) for row in examples}),
    }


def build_report(arithmetic_dir: Path, reasoning_dir: Path) -> dict[str, Any]:
    arithmetic_dir = arithmetic_dir.resolve()
    reasoning_dir = reasoning_dir.resolve()
    arithmetic_manifest, arithmetic_examples = _load_source(arithmetic_dir)
    reasoning_manifest, reasoning_examples = _load_source(reasoning_dir)
    arithmetic_receipt = _source_receipt(
        "arithmetic", arithmetic_dir, arithmetic_manifest, arithmetic_examples
    )
    reasoning_receipt = _source_receipt(
        "reasoning", reasoning_dir, reasoning_manifest, reasoning_examples
    )
    for field in ("model_id", "expected_revision", "dtype", "sites"):
        if arithmetic_receipt[field] != reasoning_receipt[field]:
            raise ValueError(f"cross-pack source mismatch: {field}")
    if set(arithmetic_receipt["families"]) & set(reasoning_receipt["families"]):
        raise ValueError("cross-pack task families overlap")

    arithmetic_signatures = _paired_signatures(
        arithmetic_dir, arithmetic_manifest, arithmetic_examples
    )
    reasoning_signatures = _paired_signatures(
        reasoning_dir, reasoning_manifest, reasoning_examples
    )
    sites = []
    for site in arithmetic_receipt["sites"]:
        arithmetic_train_mean = _mean_transition(
            arithmetic_signatures[site], _ids(arithmetic_examples, "train")
        )
        reasoning_train_mean = _mean_transition(
            reasoning_signatures[site], _ids(reasoning_examples, "train")
        )
        direction_cosine = float(
            _unit(arithmetic_train_mean) @ _unit(reasoning_train_mean)
        )
        arithmetic_to_reasoning = _cross_score(
            arithmetic_signatures[site],
            arithmetic_examples,
            reasoning_signatures[site],
            reasoning_examples,
        )
        reasoning_to_arithmetic = _cross_score(
            reasoning_signatures[site],
            reasoning_examples,
            arithmetic_signatures[site],
            arithmetic_examples,
        )
        admitted = (
            arithmetic_to_reasoning["directional_gate"] == "pass"
            and reasoning_to_arithmetic["directional_gate"] == "pass"
            and direction_cosine >= MINIMUM_CROSS_PACK_DIRECTION_COSINE
        )
        sites.append(
            {
                "site": site,
                "cross_pack_train_direction_cosine": direction_cosine,
                "arithmetic_train_to_reasoning_held_out": arithmetic_to_reasoning,
                "reasoning_train_to_arithmetic_held_out": reasoning_to_arithmetic,
                "bidirectional_cross_pack_category_admission": (
                    "pass" if admitted else "fail"
                ),
            }
        )

    body = {
        "schema": SCHEMA,
        "status": "pass",
        "sources": [arithmetic_receipt, reasoning_receipt],
        "design": {
            "category": "recurrent_pass_index_0_vs_1",
            "direction": "source_training_mean_pass_transition",
            "target": "other_pack_source_designated_held_out_examples_only",
            "centering": "each_target_example_centered_at_its_paired_pass_midpoint",
            "task_family_sets_must_be_disjoint": True,
            "random_control": (
                "exact_source_training_pair_block_pass_label_flips_scored_on_"
                "other_pack_held_out_transition_alignment"
            ),
            "admission_requires_both_transfer_directions": True,
            "admission_thresholds": {
                "minimum_cross_pack_held_out_pass_accuracy": (
                    MINIMUM_CROSS_PACK_ACCURACY
                ),
                "maximum_exact_upper_tail_p_value": MAXIMUM_EXACT_P_VALUE,
                "minimum_cross_pack_training_direction_cosine": (
                    MINIMUM_CROSS_PACK_DIRECTION_COSINE
                ),
            },
        },
        "sites": sites,
        "summary": {
            "sites": len(sites),
            "bidirectional_cross_pack_category_admissions": sum(
                row["bidirectional_cross_pack_category_admission"] == "pass"
                for row in sites
            ),
            "minimum_cross_pack_train_direction_cosine": min(
                row["cross_pack_train_direction_cosine"] for row in sites
            ),
            "minimum_cross_pack_held_out_pass_accuracy": min(
                row[direction]["held_out_pass_accuracy"]
                for row in sites
                for direction in (
                    "arithmetic_train_to_reasoning_held_out",
                    "reasoning_train_to_arithmetic_held_out",
                )
            ),
        },
        "scientific_boundary": {
            "stored_empirical_activation_archives_used": True,
            "new_model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "paid_compute_authorized": False,
            "source_checkpoint_revision_independently_verified": False,
            "result_is_conditional_on_matching_claimed_checkpoint": True,
            "cross_pack_transfer_is_not_semantic_interpretation": True,
            "does_not_authorize_separate_sae_training": True,
            "does_not_validate_bonsai_or_hrm_transfer": True,
            "execution_authorized": False,
        },
        "provenance": {
            "analysis_kind": "new_local_analysis_of_recovered_empirical_arrays",
            "runner_sha256": sha256_file(Path(__file__)),
        },
    }
    return {**body, "report_id": content_sha256(body)}


def verify_report(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    report_id = str(body.pop("report_id", ""))
    if content_sha256(body) != report_id:
        raise ValueError("cross-pack transfer content address differs")
    sources = {str(row["label"]): row for row in observed["sources"]}
    expected = build_report(
        Path(sources["arithmetic"]["path"]),
        Path(sources["reasoning"]["path"]),
    )
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("cross-pack transfer report does not reproduce")
    return {
        "schema": "sch.vpd-nanbeige-pass-category-cross-pack-transfer-verification.v1",
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
