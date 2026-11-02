from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.benchmark_nanbeige_pass_compartment_linear_proxy_v1 import (
    BOOTSTRAP_SEED,
    PASSES,
    RANDOM_PARTITION_SEEDS,
    SITES,
    _balanced_partition,
    _bootstrap_delta_interval,
    _fit_subspace,
    _load_json,
    _load_site_rows,
    _squared_error,
    _verify_source,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-nanbeige-pass-compartment-rank-sweep.v1"
TOTAL_RANKS = (8, 16, 32, 64, 96, 128)


def _errors_for_models(
    held_out: dict[str, dict[int, np.ndarray]],
    models: dict[int, tuple[np.ndarray, np.ndarray]],
    rank: int,
) -> dict[str, float]:
    return {
        example_id: sum(
            _squared_error(
                passes[pass_index],
                models[pass_index][0],
                models[pass_index][1][:rank],
            )
            for pass_index in PASSES
        )
        for example_id, passes in held_out.items()
    }


def _site_rank_sweep(
    rows: dict[str, dict[str, dict[int, np.ndarray]]],
    *,
    site_index: int,
) -> list[dict[str, Any]]:
    train = rows["train"]
    held_out = rows["eval"]
    train_ids = sorted(train)
    eval_ids = sorted(held_out)
    maximum_total_rank = max(TOTAL_RANKS)
    maximum_routed_rank = maximum_total_rank // 2
    all_train = np.concatenate(
        [train[example_id][pass_index] for example_id in train_ids for pass_index in PASSES]
    )
    global_center, global_basis = _fit_subspace(all_train, maximum_total_rank)
    pass_models = {
        pass_index: _fit_subspace(
            np.concatenate([train[example_id][pass_index] for example_id in train_ids]),
            maximum_routed_rank,
        )
        for pass_index in PASSES
    }
    denominator = {
        example_id: sum(
            float(
                np.square(
                    held_out[example_id][pass_index] - global_center,
                    dtype=np.float64,
                ).sum(dtype=np.float64)
            )
            for pass_index in PASSES
        )
        for example_id in eval_ids
    }
    denominator_total = sum(denominator.values())

    random_models: list[dict[str, Any]] = []
    for seed in RANDOM_PARTITION_SEEDS:
        train_partition = _balanced_partition(train_ids, seed, "train")
        eval_partition = _balanced_partition(eval_ids, seed, "eval")
        models = {
            group: _fit_subspace(
                np.concatenate(
                    [
                        train[example_id][pass_index]
                        for example_id in train_ids
                        if train_partition[example_id] == group
                        for pass_index in PASSES
                    ]
                ),
                maximum_routed_rank,
            )
            for group in (0, 1)
        }
        random_models.append(
            {
                "seed": seed,
                "eval_partition": eval_partition,
                "models": models,
            }
        )

    results = []
    for rank_index, total_rank in enumerate(TOTAL_RANKS):
        routed_rank = total_rank // 2
        monolithic_error = {
            example_id: sum(
                _squared_error(
                    held_out[example_id][pass_index],
                    global_center,
                    global_basis[:total_rank],
                )
                for pass_index in PASSES
            )
            for example_id in eval_ids
        }
        pass_error = _errors_for_models(held_out, pass_models, routed_rank)
        random_controls = []
        random_mean_error = {example_id: 0.0 for example_id in eval_ids}
        for control in random_models:
            errors = {
                example_id: sum(
                    _squared_error(
                        held_out[example_id][pass_index],
                        control["models"][control["eval_partition"][example_id]][0],
                        control["models"][control["eval_partition"][example_id]][1][
                            :routed_rank
                        ],
                    )
                    for pass_index in PASSES
                )
                for example_id in eval_ids
            }
            for example_id in eval_ids:
                random_mean_error[example_id] += errors[example_id] / len(
                    random_models
                )
            random_controls.append(
                {
                    "seed": int(control["seed"]),
                    "explained_variance": 1.0
                    - sum(errors.values()) / denominator_total,
                }
            )
        monolithic_ev = 1.0 - sum(monolithic_error.values()) / denominator_total
        pass_ev = 1.0 - sum(pass_error.values()) / denominator_total
        random_evs = [float(row["explained_variance"]) for row in random_controls]
        pass_mono_ci = _bootstrap_delta_interval(
            monolithic_error,
            pass_error,
            denominator,
            seed=BOOTSTRAP_SEED + 1000 * site_index + rank_index,
        )
        pass_random_ci = _bootstrap_delta_interval(
            random_mean_error,
            pass_error,
            denominator,
            seed=BOOTSTRAP_SEED + 10000 + 1000 * site_index + rank_index,
        )
        admission = (
            pass_mono_ci[0] > 0.0
            and pass_random_ci[0] > 0.0
            and pass_ev > max(random_evs)
        )
        results.append(
            {
                "total_rank": total_rank,
                "rank_per_routed_dictionary": routed_rank,
                "monolithic_explained_variance": monolithic_ev,
                "pass_compartment_explained_variance": pass_ev,
                "pass_minus_monolithic": pass_ev - monolithic_ev,
                "pass_minus_monolithic_paired_bootstrap_95_ci": pass_mono_ci,
                "random_partition_controls": random_controls,
                "random_partition_mean_explained_variance": float(np.mean(random_evs)),
                "random_partition_range_explained_variance": [
                    float(min(random_evs)),
                    float(max(random_evs)),
                ],
                "pass_minus_random_partition_mean": pass_ev
                - float(np.mean(random_evs)),
                "pass_minus_random_partition_mean_paired_bootstrap_95_ci": (
                    pass_random_ci
                ),
                "screening_admission": "pass" if admission else "fail",
            }
        )
    return results


def build_report(source_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    manifest, examples = _verify_source(source_dir)
    sites = []
    for site_index, site in enumerate(SITES):
        rows = _load_site_rows(source_dir, manifest, examples, site)
        sites.append(
            {
                "site": site,
                "ranks": _site_rank_sweep(rows, site_index=site_index),
            }
        )
    flattened = [rank for site in sites for rank in site["ranks"]]
    body = {
        "schema": SCHEMA,
        "status": "pass",
        "source": {
            "path": source_dir.as_posix(),
            "manifest_sha256": sha256_file(source_dir / "manifest.json"),
            "examples_sha256": sha256_file(source_dir / "examples.json"),
            "model_id": manifest["model"]["model_id"],
            "expected_revision": manifest["model"]["expected_revision"],
            "revision_verified": bool(manifest["model"]["revision_verified"]),
            "dtype": manifest["model"]["dtype"],
            "family_disjoint": bool(manifest["family_disjoint"]),
            "runner_sha256": sha256_file(Path(__file__)),
        },
        "design": {
            "train_examples": int(manifest["train_examples"]),
            "held_out_examples": int(manifest["eval_examples"]),
            "rows_per_example_per_pass": 8,
            "hidden_size": 3072,
            "total_ranks": list(TOTAL_RANKS),
            "routed_rank_rule": "one half of total rank per dictionary",
            "total_rank_matched_across_arms": True,
            "random_partition_seeds": list(RANDOM_PARTITION_SEEDS),
            "maximum_rank_bases_fit_once_then_sliced": True,
            "common_explained_variance_denominator": (
                "held-out squared distance from the all-pass training mean"
            ),
            "screening_admission_rule": (
                "pass routing lower 95% paired-bootstrap delta exceeds zero versus "
                "monolithic and random-mean controls, and point estimate exceeds "
                "every random-partition replicate"
            ),
        },
        "sites": sites,
        "summary": {
            "sites": len(sites),
            "ranks_per_site": len(TOTAL_RANKS),
            "rank_site_checks": len(flattened),
            "pass_wins_vs_monolithic": sum(
                row["pass_minus_monolithic_paired_bootstrap_95_ci"][0] > 0.0
                for row in flattened
            ),
            "monolithic_wins_vs_pass": sum(
                row["pass_minus_monolithic_paired_bootstrap_95_ci"][1] < 0.0
                for row in flattened
            ),
            "monolithic_vs_pass_inconclusive": sum(
                row["pass_minus_monolithic_paired_bootstrap_95_ci"][0] <= 0.0
                <= row["pass_minus_monolithic_paired_bootstrap_95_ci"][1]
                for row in flattened
            ),
            "pass_wins_vs_random_mean": sum(
                row["pass_minus_random_partition_mean_paired_bootstrap_95_ci"][0]
                > 0.0
                for row in flattened
            ),
            "pass_compartment_screening_admissions": sum(
                row["screening_admission"] == "pass" for row in flattened
            ),
        },
        "scientific_boundary": {
            "stored_empirical_bf16_activations_used": True,
            "new_model_inference_performed": False,
            "model_or_sae_weights_loaded_or_mutated": False,
            "new_sae_training_performed": False,
            "linear_rank_sweep_is_not_an_sae_quality_result": True,
            "source_revision_verified": bool(manifest["model"]["revision_verified"]),
            "source_family_disjoint": bool(manifest["family_disjoint"]),
            "screening_result_cannot_authorize_training": True,
            "execution_authorized": False,
            "paid_compute_authorized": False,
        },
    }
    return {**body, "report_id": content_sha256(body)}


def verify_report(path: Path) -> dict[str, Any]:
    observed = _load_json(path)
    body = dict(observed)
    report_id = str(body.pop("report_id", ""))
    if content_sha256(body) != report_id:
        raise ValueError("rank-sweep report content address differs")
    expected = build_report(Path(observed["source"]["path"]))
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("rank-sweep report does not reproduce")
    return {
        "schema": "sch.vpd-nanbeige-pass-compartment-rank-sweep-verification.v1",
        "status": "pass",
        "report_id": report_id,
        **observed["summary"],
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
