from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.benchmark_nanbeige_pass_category_prompt_phase_transfer_v1 import (
    SITES,
    _load_prompt_capture,
)
from experiments.authz_mechinterp.benchmark_nanbeige_pass_compartment_linear_proxy_v1 import (
    BOOTSTRAP_SEED,
    RANDOM_PARTITION_SEEDS,
    _balanced_partition,
    _bootstrap_delta_interval,
    _fit_subspace,
    _squared_error,
)
from experiments.authz_mechinterp.benchmark_nanbeige_activation_category_bridge_v1 import (
    _load_source,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-nanbeige-phase-compartment-linear-proxy.v1"
PHASES = ("prompt", "generation")
ROWS_PER_PASS_PER_PHASE = 8


def _load_rows(
    generation_dir: Path, prompt_dir: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, dict[str, np.ndarray]]],
]:
    generation_manifest, generation_examples = _load_source(generation_dir)
    prompt_manifest, prompt_records, _ = _load_prompt_capture(prompt_dir)
    generation_by_id = {str(row["id"]): row for row in generation_examples}
    rows: dict[str, dict[str, dict[str, np.ndarray]]] = {
        site: {"train": {}, "eval": {}} for site in SITES
    }
    for record in prompt_records:
        example_id = str(record["id"])
        if example_id not in generation_by_id:
            raise ValueError("prompt example is absent from generation capture")
        source = generation_by_id[example_id]
        if source["split"] != record["split"] or source["family"] != record["family"]:
            raise ValueError("prompt/generation split or family differs")
        prompt_path = prompt_dir / str(record["archive"])
        generation_path = generation_dir / str(
            generation_manifest["baseline_archives"][example_id]
        )
        with np.load(prompt_path, allow_pickle=False) as prompt_archive, np.load(
            generation_path, allow_pickle=False
        ) as generation_archive:
            for site in SITES:
                phase_rows = {}
                for phase, archive in (
                    ("prompt", prompt_archive),
                    ("generation", generation_archive),
                ):
                    selected = []
                    for pass_index in (0, 1):
                        value = np.asarray(
                            archive[f"{site}__pass_{pass_index}"], dtype=np.float64
                        )
                        if (
                            value.ndim != 2
                            or value.shape[1] != 3072
                            or value.shape[0] < ROWS_PER_PASS_PER_PHASE
                            or not np.all(np.isfinite(value))
                        ):
                            raise ValueError("phase activation rows are invalid")
                        indices = np.linspace(
                            0,
                            value.shape[0] - 1,
                            ROWS_PER_PASS_PER_PHASE,
                            dtype=np.int64,
                        )
                        selected.append(value[indices])
                    value = np.concatenate(selected, axis=0)
                    if value.shape != (16, 3072):
                        raise ValueError("phase activation rows are invalid")
                    phase_rows[phase] = value
                rows[site][str(record["split"])][example_id] = phase_rows
    return generation_manifest, prompt_manifest, prompt_records, rows


def _site_report(
    site: str,
    rows: dict[str, dict[str, dict[str, np.ndarray]]],
    site_index: int,
) -> dict[str, Any]:
    train = rows["train"]
    held_out = rows["eval"]
    train_ids = sorted(train)
    eval_ids = sorted(held_out)
    monolithic_rank = 64
    routed_rank = 32
    all_train = np.concatenate(
        [train[example_id][phase] for example_id in train_ids for phase in PHASES]
    )
    global_center, global_basis = _fit_subspace(all_train, monolithic_rank)
    phase_models = {
        phase: _fit_subspace(
            np.concatenate([train[example_id][phase] for example_id in train_ids]),
            routed_rank,
        )
        for phase in PHASES
    }
    denominator = {
        example_id: sum(
            float(
                np.square(
                    held_out[example_id][phase] - global_center,
                    dtype=np.float64,
                ).sum(dtype=np.float64)
            )
            for phase in PHASES
        )
        for example_id in eval_ids
    }
    monolithic_error = {
        example_id: sum(
            _squared_error(
                held_out[example_id][phase], global_center, global_basis
            )
            for phase in PHASES
        )
        for example_id in eval_ids
    }
    phase_error = {
        example_id: sum(
            _squared_error(held_out[example_id][phase], *phase_models[phase])
            for phase in PHASES
        )
        for example_id in eval_ids
    }

    random_results = []
    random_mean_error = {example_id: 0.0 for example_id in eval_ids}
    for seed in RANDOM_PARTITION_SEEDS:
        train_partition = _balanced_partition(train_ids, seed, "phase_train")
        eval_partition = _balanced_partition(eval_ids, seed, "phase_eval")
        random_models = {
            group: _fit_subspace(
                np.concatenate(
                    [
                        train[example_id][phase]
                        for example_id in train_ids
                        if train_partition[example_id] == group
                        for phase in PHASES
                    ]
                ),
                routed_rank,
            )
            for group in (0, 1)
        }
        errors = {
            example_id: sum(
                _squared_error(
                    held_out[example_id][phase],
                    *random_models[eval_partition[example_id]],
                )
                for phase in PHASES
            )
            for example_id in eval_ids
        }
        for example_id in eval_ids:
            random_mean_error[example_id] += errors[example_id] / len(
                RANDOM_PARTITION_SEEDS
            )
        random_results.append(
            {
                "seed": seed,
                "explained_variance": 1.0
                - sum(errors.values()) / sum(denominator.values()),
            }
        )

    denominator_total = sum(denominator.values())
    monolithic_ev = 1.0 - sum(monolithic_error.values()) / denominator_total
    phase_ev = 1.0 - sum(phase_error.values()) / denominator_total
    random_evs = [row["explained_variance"] for row in random_results]
    phase_minus_monolithic_ci = _bootstrap_delta_interval(
        monolithic_error,
        phase_error,
        denominator,
        seed=BOOTSTRAP_SEED + 200 + site_index,
    )
    phase_minus_random_ci = _bootstrap_delta_interval(
        random_mean_error,
        phase_error,
        denominator,
        seed=BOOTSTRAP_SEED + 300 + site_index,
    )
    screening_preference = (
        phase_minus_monolithic_ci[0] > 0.0
        and phase_minus_random_ci[0] > 0.0
        and phase_ev > max(random_evs)
    )
    return {
        "site": site,
        "train_examples": len(train_ids),
        "held_out_examples": len(eval_ids),
        "monolithic_total_rank": monolithic_rank,
        "phase_routed_rank_per_subspace": routed_rank,
        "monolithic_explained_variance": monolithic_ev,
        "phase_routed_explained_variance": phase_ev,
        "phase_minus_monolithic": phase_ev - monolithic_ev,
        "phase_minus_monolithic_paired_bootstrap_95_ci": (
            phase_minus_monolithic_ci
        ),
        "random_partition_explained_variances": random_evs,
        "random_partition_mean_explained_variance": float(np.mean(random_evs)),
        "phase_minus_random_partition_mean": phase_ev - float(np.mean(random_evs)),
        "phase_minus_random_partition_mean_paired_bootstrap_95_ci": (
            phase_minus_random_ci
        ),
        "screening_capacity_preference": "pass" if screening_preference else "fail",
        "phase_compartment_training_admission": "fail",
    }


def build_report(generation_dir: Path, prompt_dir: Path) -> dict[str, Any]:
    generation_dir = generation_dir.resolve()
    prompt_dir = prompt_dir.resolve()
    generation_manifest, prompt_manifest, records, rows = _load_rows(
        generation_dir, prompt_dir
    )
    sites = [
        _site_report(site, rows[site], site_index)
        for site_index, site in enumerate(SITES)
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
            "common_examples": len(records),
        },
        "design": {
            "candidate_router": "observable_activation_phase_prompt_vs_generation",
            "comparison": "equal_total_rank_held_out_linear_subspace_proxy",
            "monolithic_total_rank": 64,
            "phase_routed_total_rank": 64,
            "phase_routed_rank_per_subspace": 32,
            "random_controls": len(RANDOM_PARTITION_SEEDS),
            "bootstrap_replicates": 4096,
            "held_out_unit": "source_designated_example_block",
            "rows_per_pass_per_phase_per_example": ROWS_PER_PASS_PER_PHASE,
            "prompt_row_selection": (
                "eight_deterministic_evenly_spaced_indices_including_endpoints"
            ),
            "generation_row_selection": "all_eight_stored_rows",
            "screening_only_because_phase_parent_is_not_operatively_admitted": True,
        },
        "sites": sites,
        "summary": {
            "sites": len(sites),
            "screening_capacity_preferences": sum(
                row["screening_capacity_preference"] == "pass" for row in sites
            ),
            "phase_routing_wins_vs_monolithic": sum(
                row["phase_minus_monolithic_paired_bootstrap_95_ci"][0] > 0.0
                for row in sites
            ),
            "phase_routing_wins_vs_random_mean": sum(
                row["phase_minus_random_partition_mean_paired_bootstrap_95_ci"][0]
                > 0.0
                for row in sites
            ),
            "phase_compartment_training_admissions": 0,
        },
        "scientific_boundary": {
            "new_local_analysis_of_recovered_empirical_arrays": True,
            "new_model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "paid_compute_authorized": False,
            "linear_subspace_proxy_is_not_sae_training": True,
            "phase_parent_category_operatively_admitted": False,
            "capacity_preference_cannot_bypass_parent_category_gate": True,
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
        raise ValueError("phase capacity proxy content address differs")
    expected = build_report(
        Path(observed["sources"]["generation"]["path"]),
        Path(observed["sources"]["prompt"]["path"]),
    )
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("phase capacity proxy report does not reproduce")
    return {
        "schema": "sch.vpd-nanbeige-phase-compartment-linear-proxy-verification.v1",
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
