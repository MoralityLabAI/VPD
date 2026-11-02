from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.activation_categories import (
    CATEGORY_PROTOCOLS,
    activation_category,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    load_jsonl,
    sha256_file,
    stable_hash,
    write_json,
)
from experiments.authz_mechinterp.plan_compartment_sae_training import (
    validate_training_plan_structure,
)


SCHEMA = "sch.vpd-compartment-sae-training-execution-blueprint.v2"
ROWS_PER_EPISODE = 128
TRAINING_FRACTION = 0.90
BOUND_HYPERPARAMETERS = (
    "batch_size",
    "learning_rate",
    "steps",
    "initialization_seed",
    "data_selection_seed",
)


def _argument_value(arguments: list[str], flag: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"training argument is missing or duplicated: {flag}")
    return str(arguments[positions[0] + 1])


def load_training_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != "sch.vpd-compartment-sae-training-plan.v1":
        raise ValueError("unsupported compartment SAE training plan schema")
    body = dict(plan)
    identifier = str(body.pop("training_plan_id", ""))
    if content_sha256(body) != identifier:
        raise ValueError("compartment SAE training plan content address mismatch")
    validate_training_plan_structure(plan)
    return plan


def _selected_episodes(
    episodes: list[dict[str, Any]], *, compartment: str, seed: int
) -> list[dict[str, Any]]:
    if compartment != "shared_cross_category":
        selected = [row for row in episodes if activation_category(row) == compartment]
        if not selected:
            raise ValueError(f"training compartment is empty: {compartment}")
        return sorted(selected, key=lambda row: str(row["episode_id"]))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        grouped.setdefault(activation_category(episode), []).append(episode)
    if set(grouped) != set(CATEGORY_PROTOCOLS):
        raise ValueError("shared training source does not cover every activation category")
    target = min(len(rows) for rows in grouped.values())
    selected: list[dict[str, Any]] = []
    for category, rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda row: stable_hash(
                seed,
                "sae-balanced-category-episode",
                category,
                str(row["episode_id"]),
            ),
        )
        selected.extend(ordered[:target])
    return sorted(selected, key=lambda row: str(row["episode_id"]))


def _pair_atomic_partition(
    selected: list[dict[str, Any]], *, seed: int
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        pair_id = str(row.get("pair_id", ""))
        if not pair_id:
            raise ValueError("training episode lacks a pair ID")
        grouped.setdefault(pair_id, []).append(row)
    if len(grouped) < 2:
        raise ValueError("training partition requires at least two prompt pairs")
    ordered = sorted(
        grouped,
        key=lambda pair_id: stable_hash(seed, "sae-pair-atomic-split", pair_id),
    )
    split = max(1, min(len(ordered) - 1, int(TRAINING_FRACTION * len(ordered))))
    training_pairs = sorted(ordered[:split])
    validation_pairs = sorted(ordered[split:])
    training_pair_set = set(training_pairs)
    training_episodes = [
        str(row["episode_id"])
        for row in selected
        if str(row["pair_id"]) in training_pair_set
    ]
    validation_episodes = [
        str(row["episode_id"])
        for row in selected
        if str(row["pair_id"]) not in training_pair_set
    ]
    return {
        "training_pairs": len(training_pairs),
        "validation_pairs": len(validation_pairs),
        "training_rows": len(training_episodes) * ROWS_PER_EPISODE,
        "validation_rows": len(validation_episodes) * ROWS_PER_EPISODE,
        "training_pair_id_set_sha256": content_sha256(training_pairs),
        "validation_pair_id_set_sha256": content_sha256(validation_pairs),
        "training_episode_id_set_sha256": content_sha256(sorted(training_episodes)),
        "validation_episode_id_set_sha256": content_sha256(
            sorted(validation_episodes)
        ),
        "pair_sets_disjoint": not bool(set(training_pairs) & set(validation_pairs)),
        "exact_selected_episode_partition": (
            len(training_episodes) + len(validation_episodes) == len(selected)
        ),
    }


def derive_sample_contract(
    episodes: list[dict[str, Any]], *, job: dict[str, Any]
) -> dict[str, Any]:
    compartment = str(job["compartment"])
    seed = int(job["data_selection_seed"])
    selected = _selected_episodes(episodes, compartment=compartment, seed=seed)
    categories = Counter(activation_category(row) for row in selected)
    category_rows = {
        category: count * ROWS_PER_EPISODE
        for category, count in sorted(categories.items())
    }
    expected = job["expected_support"]
    observed_rows = len(selected) * ROWS_PER_EPISODE
    observed_pairs = len({str(row["pair_id"]) for row in selected})
    if (
        observed_rows != int(expected["activation_rows"])
        or len(selected) != int(expected["episodes"])
        or observed_pairs != int(expected["unique_pairs"])
    ):
        raise ValueError(f"derived training support differs for {compartment}")
    if compartment == "shared_cross_category":
        if set(categories) != set(CATEGORY_PROTOCOLS) or len(set(categories.values())) != 1:
            raise ValueError("shared training selection is not category-balanced")
    elif set(categories) != {compartment}:
        raise ValueError("specific training selection crosses activation compartments")
    partition = _pair_atomic_partition(selected, seed=seed)
    if not partition["pair_sets_disjoint"] or not partition[
        "exact_selected_episode_partition"
    ]:
        raise ValueError("training pair-atomic partition is invalid")
    return {
        "schema": "sch.vpd-compartment-sae-sample-contract.v2",
        "selection": (
            "equal_whole_episodes_per_observed_activation_category"
            if compartment == "shared_cross_category"
            else "exact_observable_activation_category"
        ),
        "activation_category": compartment,
        "selected_episodes": len(selected),
        "selected_activation_rows": observed_rows,
        "selected_unique_pairs": observed_pairs,
        "selected_episode_id_set_sha256": content_sha256(
            sorted(str(row["episode_id"]) for row in selected)
        ),
        "selected_prompt_window_identity_sha256": content_sha256(
            {
                "episode_ids": sorted(str(row["episode_id"]) for row in selected),
                "token_offset_start": 1 - ROWS_PER_EPISODE,
                "token_offset_end": 0,
            }
        ),
        "selected_episode_counts_by_category": dict(sorted(categories.items())),
        "selected_activation_rows_by_category": category_rows,
        "specific_cross_compartment_rows": (
            0
            if compartment != "shared_cross_category"
            else None
        ),
        "shared_equal_episode_counts": (
            len(set(categories.values())) == 1
            if compartment == "shared_cross_category"
            else None
        ),
        "pair_atomic_partition": partition,
    }


def derive_blueprint(
    *, training_plan_path: Path, project_root: Path
) -> dict[str, Any]:
    plan = load_training_plan(training_plan_path)
    capture_jobs = [
        row for row in plan["capture_jobs"] if row.get("role") == "sae_training"
    ]
    if len(capture_jobs) != 1:
        raise ValueError("training plan must contain one SAE training capture")
    dataset_path = (project_root / str(capture_jobs[0]["input"]["path"])).resolve()
    if not dataset_path.is_relative_to(project_root.resolve()):
        raise ValueError("training dataset escapes the project")
    if sha256_file(dataset_path) != str(capture_jobs[0]["input"]["sha256"]):
        raise ValueError("training dataset differs from the plan")
    episodes = load_jsonl(dataset_path)

    records: list[dict[str, Any]] = []
    sample_by_compartment: dict[str, dict[str, Any]] = {}
    for job in sorted(
        plan["training_jobs"],
        key=lambda row: (str(row["compartment"]), int(row["initialization_seed"])),
    ):
        arguments = [str(value) for value in job["arguments"]]
        hyperparameters = {
            "batch_size": int(_argument_value(arguments, "--batch-size")),
            "learning_rate": float(_argument_value(arguments, "--learning-rate")),
            "steps": int(_argument_value(arguments, "--steps")),
            "initialization_seed": int(_argument_value(arguments, "--seed")),
            "data_selection_seed": int(_argument_value(arguments, "--split-seed")),
        }
        expected_hyperparameters = {
            "batch_size": int(plan["profile"]["batch_size"]),
            "learning_rate": float(plan["profile"]["learning_rate"]),
            "steps": int(plan["profile"]["steps"]),
            "initialization_seed": int(job["initialization_seed"]),
            "data_selection_seed": int(job["data_selection_seed"]),
        }
        if hyperparameters != expected_hyperparameters:
            raise ValueError("planned SAE arguments differ from the frozen profile")
        compartment = str(job["compartment"])
        sample = sample_by_compartment.setdefault(
            compartment, derive_sample_contract(episodes, job=job)
        )
        records.append(
            {
                "planned_job_id": str(job["job_id"]),
                "compartment": compartment,
                "initialization_seed": int(job["initialization_seed"]),
                "planned_arguments_sha256": content_sha256(arguments),
                "required_manifest_hyperparameters": hyperparameters,
                "required_sample_contract": sample,
            }
        )

    body = {
        "schema": SCHEMA,
        "status": "design_only_not_executed",
        "source": {
            "training_plan_id": str(plan["training_plan_id"]),
            "training_plan_sha256": sha256_file(training_plan_path),
            "sae_training_dataset": dataset_path.relative_to(
                project_root.resolve()
            ).as_posix(),
            "sae_training_dataset_sha256": sha256_file(dataset_path),
            "derivation_runner_sha256": sha256_file(Path(__file__)),
        },
        "jobs": records,
        "coverage": {
            "training_jobs": len(records),
            "compartments": len(sample_by_compartment),
            "replicas_per_compartment": 2,
            "specific_compartments": len(CATEGORY_PROTOCOLS),
            "shared_compartments": 1,
        },
        "v2_admission_requirements": {
            "required_dictionary_manifest_schema": (
                "sch.vpd-fixed-sae-dictionary-manifest.v2"
            ),
            "all_planned_hyperparameters_manifest_bound": True,
            "bound_hyperparameters": list(BOUND_HYPERPARAMETERS),
            "exact_sample_contract_manifest_bound": True,
            "specific_cross_compartment_rows_must_equal": 0,
            "shared_equal_whole_episode_balance_required": True,
            "predecessor_v1_manifest_admissible": False,
        },
        "predecessor_gap": {
            "schema": "sch.vpd-fixed-sae-dictionary-manifest.v1",
            "planned_but_not_manifest_bound": ["batch_size", "learning_rate"],
            "training_plan_v3_preserved_as_predecessor_evidence": True,
        },
        "scientific_boundary": {
            "model_inference_performed": False,
            "sae_training_performed": False,
            "weights_loaded_or_mutated": False,
            "execution_authorized": False,
            "v7_negative_release_decision_unchanged": True,
        },
    }
    return {**body, "execution_blueprint_id": content_sha256(body)}


def validate_blueprint(
    blueprint: dict[str, Any], *, training_plan_path: Path, project_root: Path
) -> None:
    if blueprint.get("schema") != SCHEMA:
        raise ValueError("unsupported training execution blueprint schema")
    body = dict(blueprint)
    identifier = str(body.pop("execution_blueprint_id", ""))
    if content_sha256(body) != identifier:
        raise ValueError("training execution blueprint content address mismatch")
    expected = derive_blueprint(
        training_plan_path=training_plan_path, project_root=project_root
    )
    if blueprint != expected:
        raise ValueError("training execution blueprint is not reproducible")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a design-only, exact compartment SAE training execution contract."
        )
    )
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    observed = derive_blueprint(
        training_plan_path=args.training_plan, project_root=args.project_root.resolve()
    )
    if args.verify is not None:
        expected = json.loads(args.verify.read_text(encoding="utf-8"))
        validate_blueprint(
            expected,
            training_plan_path=args.training_plan,
            project_root=args.project_root.resolve(),
        )
        if observed != expected:
            raise ValueError("training execution blueprint differs")
    elif args.output is not None:
        write_json(args.output, observed)
    else:
        parser.error("use --output or --verify")
    print(
        json.dumps(
            {
                "schema": "sch.vpd-compartment-sae-training-execution-cli.v2",
                "status": observed["status"],
                "execution_blueprint_id": observed["execution_blueprint_id"],
                "training_jobs": len(observed["jobs"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
