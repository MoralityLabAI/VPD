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
from experiments.authz_mechinterp.benchmark_nanbeige_pass_category_shared_axis_v1 import (
    _control_directions,
    _ids,
    _transitions,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-nanbeige-pass-category-prompt-phase-transfer.v1"
SITES = ("model.layers.11", "model.layers.21", "model.norm")
MINIMUM_HELD_OUT_ACCURACY = 0.875
MINIMUM_PROMPT_ALIGNMENT = 0.80
MAXIMUM_EXACT_P_VALUE = 0.05


def _load_prompt_capture(
    prompt_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, dict[int, np.ndarray]]]]:
    manifest_path = prompt_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "prompt_capture_complete"
        or manifest.get("scope") != "prompt_only"
        or manifest.get("generated_reasoning") is not False
        or manifest.get("dtype") != "bfloat16"
    ):
        raise ValueError("unexpected prompt-only capture manifest")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("prompt-only capture records are absent")
    files = manifest.get("files", {})
    for name, expected_hash in files.items():
        if sha256_file(prompt_dir / str(name)) != str(expected_hash):
            raise ValueError(f"prompt-only capture artifact hash differs: {name}")
    signatures: dict[str, dict[str, dict[int, np.ndarray]]] = {
        site: {} for site in SITES
    }
    for record in records:
        example_id = str(record["id"])
        archive = np.load(prompt_dir / str(record["archive"]), allow_pickle=False)
        for site in SITES:
            by_pass = {}
            for pass_index in (0, 1):
                values = np.asarray(
                    archive[f"{site}__pass_{pass_index}"], dtype=np.float64
                )
                if values.shape != (32, 3072) or not np.all(np.isfinite(values)):
                    raise ValueError("prompt-only activation array is invalid")
                by_pass[pass_index] = values.mean(axis=0)
            signatures[site][example_id] = by_pass
    return manifest, records, signatures


def _site_report(
    site: str,
    arithmetic_signatures: dict[str, dict[int, np.ndarray]],
    arithmetic_examples: list[dict[str, Any]],
    reasoning_signatures: dict[str, dict[int, np.ndarray]],
    reasoning_examples: list[dict[str, Any]],
    prompt_signatures: dict[str, dict[int, np.ndarray]],
    prompt_records: list[dict[str, Any]],
) -> dict[str, Any]:
    arithmetic_train = _transitions(
        arithmetic_signatures, _ids(arithmetic_examples, "train")
    )
    reasoning_train = _transitions(
        reasoning_signatures, _ids(reasoning_examples, "train")
    )
    arithmetic_direction = _unit(np.mean(list(arithmetic_train.values()), axis=0))
    reasoning_direction = _unit(np.mean(list(reasoning_train.values()), axis=0))
    shared_axis = _unit(0.5 * arithmetic_direction + 0.5 * reasoning_direction)

    prompt_eval_ids = _ids(prompt_records, "eval")
    prompt_all_ids = [str(row["id"]) for row in prompt_records]
    eval_accuracy, eval_margins = _accuracy_and_margins(
        shared_axis, prompt_eval_ids, prompt_signatures
    )
    all_accuracy, all_margins = _accuracy_and_margins(
        shared_axis, prompt_all_ids, prompt_signatures
    )
    prompt_eval_transitions = _transitions(prompt_signatures, prompt_eval_ids)
    prompt_eval_direction = _unit(
        np.mean(list(prompt_eval_transitions.values()), axis=0)
    )
    prompt_alignment = float(shared_axis @ prompt_eval_direction)

    arithmetic_controls = _control_directions(
        arithmetic_train, arithmetic_examples
    )
    reasoning_controls = _control_directions(reasoning_train, reasoning_examples)
    control_alignments: list[float] = []
    for arithmetic_control in arithmetic_controls:
        for reasoning_control in reasoning_controls:
            combined = 0.5 * arithmetic_control + 0.5 * reasoning_control
            norm = float(np.linalg.norm(combined))
            alignment = (
                0.0
                if norm <= 1e-12
                else float((combined / norm) @ prompt_eval_direction)
            )
            control_alignments.append(alignment)
    at_or_above = sum(
        value >= prompt_alignment - 1e-12 for value in control_alignments
    )
    p_value = at_or_above / len(control_alignments)
    admitted = (
        eval_accuracy >= MINIMUM_HELD_OUT_ACCURACY
        and prompt_alignment >= MINIMUM_PROMPT_ALIGNMENT
        and p_value <= MAXIMUM_EXACT_P_VALUE
    )
    return {
        "site": site,
        "prompt_eval_examples": len(prompt_eval_ids),
        "prompt_all_examples": len(prompt_all_ids),
        "shared_axis_prompt_eval_pass_accuracy": eval_accuracy,
        "shared_axis_prompt_eval_minimum_normalized_margin": min(eval_margins),
        "shared_axis_prompt_all_pass_accuracy_descriptive_only": all_accuracy,
        "shared_axis_prompt_all_minimum_normalized_margin_descriptive_only": min(
            all_margins
        ),
        "shared_axis_to_prompt_eval_transition_cosine": prompt_alignment,
        "exact_joint_training_pair_flip_control": {
            "joint_configurations": len(control_alignments),
            "minimum": min(control_alignments),
            "median": float(np.median(control_alignments)),
            "maximum": max(control_alignments),
            "at_or_above_observed": at_or_above,
            "exact_upper_tail_p_value": p_value,
        },
        "prompt_phase_transfer_admission": "pass" if admitted else "fail",
    }


def build_report(
    arithmetic_dir: Path, reasoning_dir: Path, prompt_dir: Path
) -> dict[str, Any]:
    arithmetic_dir = arithmetic_dir.resolve()
    reasoning_dir = reasoning_dir.resolve()
    prompt_dir = prompt_dir.resolve()
    arithmetic_manifest, arithmetic_examples = _load_source(arithmetic_dir)
    reasoning_manifest, reasoning_examples = _load_source(reasoning_dir)
    prompt_manifest, prompt_records, prompt_signatures = _load_prompt_capture(
        prompt_dir
    )
    if arithmetic_manifest["model"]["model_id"] != prompt_manifest["checkpoint"][
        "model_id"
    ] or reasoning_manifest["model"]["model_id"] != prompt_manifest["checkpoint"][
        "model_id"
    ]:
        raise ValueError("prompt-phase model ID differs")
    if arithmetic_manifest["model"]["expected_revision"] != prompt_manifest[
        "checkpoint"
    ]["revision"] or reasoning_manifest["model"]["expected_revision"] != prompt_manifest[
        "checkpoint"
    ]["revision"]:
        raise ValueError("prompt-phase checkpoint revision differs")
    if len(_ids(prompt_records, "eval")) < 4:
        raise ValueError("prompt-phase held-out split is underpowered")
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
            prompt_signatures[site],
            prompt_records,
        )
        for site in SITES
    ]
    body = {
        "schema": SCHEMA,
        "status": "pass",
        "sources": {
            "arithmetic_generation": {
                "path": arithmetic_dir.as_posix(),
                "manifest_sha256": sha256_file(arithmetic_dir / "manifest.json"),
            },
            "reasoning_generation": {
                "path": reasoning_dir.as_posix(),
                "manifest_sha256": sha256_file(reasoning_dir / "manifest.json"),
            },
            "reasoning_prompt_only": {
                "path": prompt_dir.as_posix(),
                "manifest_sha256": sha256_file(prompt_dir / "manifest.json"),
                "checkpoint_revision": prompt_manifest["checkpoint"]["revision"],
                "checkpoint_weight_files_hash_attested": True,
            },
        },
        "design": {
            "axis_training": "equal_pack_generation_phase_training_only_bisector",
            "primary_target": "prompt_only_source_designated_eval_examples",
            "primary_target_examples_overlap_axis_training_examples": False,
            "all_prompt_examples_are_descriptive_only": True,
            "random_control": "exact_16384_joint_generation_training_pair_label_flips",
            "admission_thresholds": {
                "minimum_prompt_eval_pass_accuracy": MINIMUM_HELD_OUT_ACCURACY,
                "minimum_shared_axis_to_prompt_eval_transition_cosine": (
                    MINIMUM_PROMPT_ALIGNMENT
                ),
                "maximum_exact_upper_tail_p_value": MAXIMUM_EXACT_P_VALUE,
            },
        },
        "sites": sites,
        "summary": {
            "sites": len(sites),
            "prompt_phase_transfer_admissions": sum(
                row["prompt_phase_transfer_admission"] == "pass" for row in sites
            ),
            "minimum_prompt_eval_pass_accuracy": min(
                row["shared_axis_prompt_eval_pass_accuracy"] for row in sites
            ),
            "minimum_shared_axis_to_prompt_eval_transition_cosine": min(
                row["shared_axis_to_prompt_eval_transition_cosine"] for row in sites
            ),
            "maximum_exact_upper_tail_p_value": max(
                row["exact_joint_training_pair_flip_control"][
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
            "primary_prompt_eval_examples_were_not_axis_training_examples": True,
            "prompt_phase_is_not_a_third_semantic_domain": True,
            "does_not_establish_third_domain_transfer": True,
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
        raise ValueError("prompt-phase transfer content address differs")
    expected = build_report(
        Path(observed["sources"]["arithmetic_generation"]["path"]),
        Path(observed["sources"]["reasoning_generation"]["path"]),
        Path(observed["sources"]["reasoning_prompt_only"]["path"]),
    )
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("prompt-phase transfer report does not reproduce")
    return {
        "schema": "sch.vpd-nanbeige-pass-category-prompt-phase-transfer-verification.v1",
        "status": "pass",
        "report_id": report_id,
        **observed["summary"],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arithmetic-dir", type=Path)
    parser.add_argument("--reasoning-dir", type=Path)
    parser.add_argument("--prompt-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_report(args.verify)
    else:
        if (
            args.arithmetic_dir is None
            or args.reasoning_dir is None
            or args.prompt_dir is None
            or args.output is None
        ):
            parser.error("all source directories and --output are required unless verifying")
        result = build_report(args.arithmetic_dir, args.reasoning_dir, args.prompt_dir)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
