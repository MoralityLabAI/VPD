from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.benchmark_nanbeige_pass_compartment_linear_proxy_v1 import (
    BOOTSTRAP_SEED,
    RANDOM_PARTITION_SEEDS,
    _bootstrap_delta_interval,
    _fit_subspace,
    _squared_error,
)
from experiments.authz_mechinterp.benchmark_nanbeige_phase_compartment_linear_proxy_v1 import (
    PHASES,
    ROWS_PER_PASS_PER_PHASE,
    _load_rows,
)
from experiments.authz_mechinterp.benchmark_nanbeige_pass_category_prompt_phase_transfer_v1 import (
    SITES,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-nanbeige-joint-pass-phase-capacity.v1"
PASSES = (0, 1)
JOINT_CELLS = tuple((pass_index, phase) for pass_index in PASSES for phase in PHASES)
TOTAL_RANK = 64
BOOTSTRAP_REPLICATES = 4096


def _cell(value: np.ndarray, pass_index: int) -> np.ndarray:
    start = pass_index * ROWS_PER_PASS_PER_PHASE
    stop = start + ROWS_PER_PASS_PER_PHASE
    selected = value[start:stop]
    if selected.shape != (ROWS_PER_PASS_PER_PHASE, 3072):
        raise ValueError("pass/phase cell has an unexpected shape")
    return selected


def _balanced_four_partition(ids: list[str], seed: int, split: str) -> dict[str, int]:
    if len(ids) % 4:
        raise ValueError("four-way random control requires a multiple of four examples")
    ordered = sorted(
        ids,
        key=lambda example_id: hashlib.sha256(
            f"{seed}|joint_pass_phase|{split}|{example_id}".encode("utf-8")
        ).hexdigest(),
    )
    per_group = len(ordered) // 4
    return {
        example_id: min(index // per_group, 3)
        for index, example_id in enumerate(ordered)
    }


def _ev(errors: dict[str, float], denominator: dict[str, float]) -> float:
    return 1.0 - sum(errors.values()) / sum(denominator.values())


def _site_report(
    site: str,
    rows: dict[str, dict[str, dict[str, np.ndarray]]],
    site_index: int,
) -> dict[str, Any]:
    train = rows["train"]
    held_out = rows["eval"]
    train_ids = sorted(train)
    eval_ids = sorted(held_out)
    if len(train_ids) != 4 or len(eval_ids) != 4:
        raise ValueError("expected the sealed four-train/four-eval common-example split")

    def values(source: dict[str, dict[str, np.ndarray]], ids: list[str], cells: tuple[tuple[int, str], ...]) -> np.ndarray:
        return np.concatenate(
            [
                _cell(source[example_id][phase], pass_index)
                for example_id in ids
                for pass_index, phase in cells
            ]
        )

    all_train = values(train, train_ids, JOINT_CELLS)
    global_center, global_basis = _fit_subspace(all_train, TOTAL_RANK)
    pass_models = {
        pass_index: _fit_subspace(
            values(train, train_ids, tuple((pass_index, phase) for phase in PHASES)),
            TOTAL_RANK // 2,
        )
        for pass_index in PASSES
    }
    phase_models = {
        phase: _fit_subspace(
            values(train, train_ids, tuple((pass_index, phase) for pass_index in PASSES)),
            TOTAL_RANK // 2,
        )
        for phase in PHASES
    }
    joint_models = {
        (pass_index, phase): _fit_subspace(
            values(train, train_ids, ((pass_index, phase),)),
            TOTAL_RANK // 4,
        )
        for pass_index, phase in JOINT_CELLS
    }

    denominator = {
        example_id: sum(
            float(
                np.square(
                    _cell(held_out[example_id][phase], pass_index) - global_center,
                    dtype=np.float64,
                ).sum(dtype=np.float64)
            )
            for pass_index, phase in JOINT_CELLS
        )
        for example_id in eval_ids
    }
    monolithic_error = {
        example_id: sum(
            _squared_error(
                _cell(held_out[example_id][phase], pass_index),
                global_center,
                global_basis,
            )
            for pass_index, phase in JOINT_CELLS
        )
        for example_id in eval_ids
    }
    pass_error = {
        example_id: sum(
            _squared_error(
                _cell(held_out[example_id][phase], pass_index),
                *pass_models[pass_index],
            )
            for pass_index, phase in JOINT_CELLS
        )
        for example_id in eval_ids
    }
    phase_error = {
        example_id: sum(
            _squared_error(
                _cell(held_out[example_id][phase], pass_index),
                *phase_models[phase],
            )
            for pass_index, phase in JOINT_CELLS
        )
        for example_id in eval_ids
    }
    joint_error = {
        example_id: sum(
            _squared_error(
                _cell(held_out[example_id][phase], pass_index),
                *joint_models[(pass_index, phase)],
            )
            for pass_index, phase in JOINT_CELLS
        )
        for example_id in eval_ids
    }

    random_results = []
    random_mean_error = {example_id: 0.0 for example_id in eval_ids}
    for seed in RANDOM_PARTITION_SEEDS:
        train_partition = _balanced_four_partition(train_ids, seed, "train")
        eval_partition = _balanced_four_partition(eval_ids, seed, "eval")
        random_models = {
            group: _fit_subspace(
                values(
                    train,
                    [example_id for example_id in train_ids if train_partition[example_id] == group],
                    JOINT_CELLS,
                ),
                TOTAL_RANK // 4,
            )
            for group in range(4)
        }
        errors = {
            example_id: sum(
                _squared_error(
                    _cell(held_out[example_id][phase], pass_index),
                    *random_models[eval_partition[example_id]],
                )
                for pass_index, phase in JOINT_CELLS
            )
            for example_id in eval_ids
        }
        for example_id in eval_ids:
            random_mean_error[example_id] += errors[example_id] / len(RANDOM_PARTITION_SEEDS)
        random_results.append({"seed": seed, "explained_variance": _ev(errors, denominator)})

    arm_errors = {
        "monolithic": monolithic_error,
        "pass_routed": pass_error,
        "phase_routed": phase_error,
        "joint_pass_phase": joint_error,
    }
    arm_evs = {name: _ev(error, denominator) for name, error in arm_errors.items()}
    random_evs = [row["explained_variance"] for row in random_results]
    comparisons = {}
    references = {**arm_errors, "random_partition_mean": random_mean_error}
    for offset, (name, reference_error) in enumerate(references.items()):
        if name == "joint_pass_phase":
            continue
        reference_ev = (
            float(np.mean(random_evs))
            if name == "random_partition_mean"
            else arm_evs[name]
        )
        comparisons[name] = {
            "delta": arm_evs["joint_pass_phase"] - reference_ev,
            "paired_example_block_bootstrap_95_ci": _bootstrap_delta_interval(
                reference_error,
                joint_error,
                denominator,
                seed=BOOTSTRAP_SEED + 400 + site_index * 10 + offset,
            ),
        }
    preference = (
        all(row["paired_example_block_bootstrap_95_ci"][0] > 0.0 for row in comparisons.values())
        and arm_evs["joint_pass_phase"] > max(random_evs)
    )
    return {
        "site": site,
        "train_examples": len(train_ids),
        "held_out_examples": len(eval_ids),
        "arm_explained_variance": arm_evs,
        "joint_minus": comparisons,
        "random_partition_explained_variances": random_evs,
        "random_partition_mean_explained_variance": float(np.mean(random_evs)),
        "joint_capacity_preference": "pass" if preference else "fail",
        "joint_compartment_training_admission": "fail",
    }


def build_report(generation_dir: Path, prompt_dir: Path) -> dict[str, Any]:
    generation_dir = generation_dir.resolve()
    prompt_dir = prompt_dir.resolve()
    generation_manifest, prompt_manifest, records, rows = _load_rows(generation_dir, prompt_dir)
    sites = [_site_report(site, rows[site], index) for index, site in enumerate(SITES)]
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
            "common_examples": len(records),
        },
        "design": {
            "candidate_router": "observable_joint_recurrent_pass_by_activation_phase",
            "comparison": "equal_total_rank_held_out_linear_subspace_proxy",
            "total_rank_per_arm": TOTAL_RANK,
            "arm_ranks": {
                "monolithic": [TOTAL_RANK],
                "pass_routed": [TOTAL_RANK // 2] * 2,
                "phase_routed": [TOTAL_RANK // 2] * 2,
                "joint_pass_phase": [TOTAL_RANK // 4] * 4,
                "random_four_way": [TOTAL_RANK // 4] * 4,
            },
            "random_controls": len(RANDOM_PARTITION_SEEDS),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "held_out_unit": "source_designated_example_block",
            "rows_per_pass_per_phase_per_example": ROWS_PER_PASS_PER_PHASE,
            "all_arms_use_identical_rows_and_denominator": True,
            "four_way_random_train_examples_per_subspace": 1,
            "four_way_random_control_is_minimal_and_screening_only": True,
            "screening_only_because_phase_parent_is_not_operatively_admitted": True,
        },
        "sites": sites,
        "summary": {
            "sites": len(sites),
            "joint_capacity_preferences": sum(row["joint_capacity_preference"] == "pass" for row in sites),
            "joint_wins_vs_monolithic": sum(row["joint_minus"]["monolithic"]["paired_example_block_bootstrap_95_ci"][0] > 0.0 for row in sites),
            "joint_wins_vs_pass_routed": sum(row["joint_minus"]["pass_routed"]["paired_example_block_bootstrap_95_ci"][0] > 0.0 for row in sites),
            "joint_wins_vs_phase_routed": sum(row["joint_minus"]["phase_routed"]["paired_example_block_bootstrap_95_ci"][0] > 0.0 for row in sites),
            "joint_wins_vs_random_mean": sum(row["joint_minus"]["random_partition_mean"]["paired_example_block_bootstrap_95_ci"][0] > 0.0 for row in sites),
            "joint_compartment_training_admissions": 0,
        },
        "scientific_boundary": {
            "new_local_analysis_of_recovered_empirical_arrays": True,
            "new_model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "paid_compute_authorized": False,
            "linear_subspace_proxy_is_not_sae_training": True,
            "phase_parent_category_operatively_admitted": False,
            "capacity_preference_cannot_bypass_parent_category_gate": True,
            "does_not_authorize_joint_compartment_sae_training": True,
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
        raise ValueError("joint pass/phase capacity report content address differs")
    expected = build_report(
        Path(observed["sources"]["generation"]["path"]),
        Path(observed["sources"]["prompt"]["path"]),
    )
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("joint pass/phase capacity report does not reproduce")
    return {
        "schema": "sch.vpd-nanbeige-joint-pass-phase-capacity-verification.v1",
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
