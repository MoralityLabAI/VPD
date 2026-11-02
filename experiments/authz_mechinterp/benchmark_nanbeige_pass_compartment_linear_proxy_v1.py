from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-nanbeige-pass-compartment-linear-proxy.v1"
SITES = ("model.layers.11", "model.layers.21", "model.norm")
PASSES = (0, 1)
RANDOM_PARTITION_SEEDS = (17, 29, 43, 59, 71, 89, 101, 127)
BOOTSTRAP_SEED = 250829
BOOTSTRAP_REPLICATES = 4096


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_source(source_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = source_dir / "manifest.json"
    examples_path = source_dir / "examples.json"
    quality_path = source_dir / "sae_quality.json"
    manifest = _load_json(manifest_path)
    examples = _load_json(examples_path)
    if manifest.get("schema") != "vpd.multipass.calibration.v1":
        raise ValueError("unexpected Nanbeige multipass manifest schema")
    if not isinstance(examples, list) or not examples:
        raise ValueError("examples must be a nonempty list")
    if sha256_file(examples_path) != manifest["artifacts"]["examples.json"]:
        raise ValueError("examples hash differs from manifest")
    if sha256_file(quality_path) != manifest["artifacts"]["sae_quality.json"]:
        raise ValueError("SAE quality hash differs from manifest")
    archive_names = set(manifest["baseline_archives"].values())
    if len(archive_names) != len(examples):
        raise ValueError("baseline archive mapping is not one-to-one with examples")
    for name in sorted(archive_names):
        if sha256_file(source_dir / name) != manifest["artifacts"][name]:
            raise ValueError(f"baseline archive hash differs: {name}")
    example_ids = [str(row["id"]) for row in examples]
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("example IDs are not unique")
    if set(example_ids) != set(manifest["baseline_archives"]):
        raise ValueError("example IDs differ from baseline archive mapping")
    split_counts = {
        split: sum(str(row["split"]) == split for row in examples)
        for split in ("train", "eval")
    }
    if split_counts != {
        "train": int(manifest["train_examples"]),
        "eval": int(manifest["eval_examples"]),
    }:
        raise ValueError("example split counts differ from manifest")
    return manifest, examples


def _load_site_rows(
    source_dir: Path,
    manifest: dict[str, Any],
    examples: list[dict[str, Any]],
    site: str,
) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    rows: dict[str, dict[str, dict[int, np.ndarray]]] = {"train": {}, "eval": {}}
    for example in examples:
        example_id = str(example["id"])
        split = str(example["split"])
        archive_path = source_dir / manifest["baseline_archives"][example_id]
        with np.load(archive_path, allow_pickle=False) as archive:
            passes: dict[int, np.ndarray] = {}
            for pass_index in PASSES:
                key = f"{site}__pass_{pass_index}"
                value = np.asarray(archive[key], dtype=np.float64)
                if value.ndim != 2 or value.shape[1] != 3072:
                    raise ValueError(f"unexpected activation shape for {example_id} {key}")
                if not np.isfinite(value).all():
                    raise ValueError(f"nonfinite activation for {example_id} {key}")
                passes[pass_index] = value
            if passes[0].shape != passes[1].shape:
                raise ValueError(f"pass row shapes differ for {example_id}")
            rows[split][example_id] = passes
    return rows


def _fit_subspace(rows: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    center = rows.mean(axis=0, dtype=np.float64)
    centered = rows - center
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    effective_rank = min(rank, right.shape[0])
    return center, right[:effective_rank]


def _squared_error(
    rows: np.ndarray, center: np.ndarray, basis: np.ndarray
) -> float:
    centered = rows - center
    projection = centered @ basis.T
    residual = centered - projection @ basis
    return float(np.square(residual, dtype=np.float64).sum(dtype=np.float64))


def _balanced_partition(ids: list[str], seed: int, split: str) -> dict[str, int]:
    ordered = sorted(
        ids,
        key=lambda example_id: hashlib.sha256(
            f"{seed}|{split}|{example_id}".encode("utf-8")
        ).hexdigest(),
    )
    first_count = len(ordered) // 2
    return {
        example_id: int(index >= first_count)
        for index, example_id in enumerate(ordered)
    }


def _arm_errors(
    rows: dict[str, dict[str, dict[int, np.ndarray]]],
    *,
    monolithic_rank: int,
    routed_rank: int,
) -> tuple[dict[str, float], dict[str, float], list[dict[str, Any]], dict[str, float]]:
    train = rows["train"]
    held_out = rows["eval"]
    train_ids = sorted(train)
    eval_ids = sorted(held_out)
    all_train = np.concatenate(
        [train[example_id][pass_index] for example_id in train_ids for pass_index in PASSES]
    )
    global_center, global_basis = _fit_subspace(all_train, monolithic_rank)
    pass_models = {
        pass_index: _fit_subspace(
            np.concatenate([train[example_id][pass_index] for example_id in train_ids]),
            routed_rank,
        )
        for pass_index in PASSES
    }

    denominator: dict[str, float] = {}
    monolithic_error: dict[str, float] = {}
    pass_error: dict[str, float] = {}
    for example_id in eval_ids:
        denominator[example_id] = sum(
            float(
                np.square(
                    held_out[example_id][pass_index] - global_center,
                    dtype=np.float64,
                ).sum(dtype=np.float64)
            )
            for pass_index in PASSES
        )
        monolithic_error[example_id] = sum(
            _squared_error(
                held_out[example_id][pass_index], global_center, global_basis
            )
            for pass_index in PASSES
        )
        pass_error[example_id] = sum(
            _squared_error(
                held_out[example_id][pass_index], *pass_models[pass_index]
            )
            for pass_index in PASSES
        )

    random_results: list[dict[str, Any]] = []
    random_mean_error = {example_id: 0.0 for example_id in eval_ids}
    for seed in RANDOM_PARTITION_SEEDS:
        train_partition = _balanced_partition(train_ids, seed, "train")
        eval_partition = _balanced_partition(eval_ids, seed, "eval")
        models = {}
        for group in (0, 1):
            group_rows = np.concatenate(
                [
                    train[example_id][pass_index]
                    for example_id in train_ids
                    if train_partition[example_id] == group
                    for pass_index in PASSES
                ]
            )
            models[group] = _fit_subspace(group_rows, routed_rank)
        errors = {
            example_id: sum(
                _squared_error(
                    held_out[example_id][pass_index],
                    *models[eval_partition[example_id]],
                )
                for pass_index in PASSES
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
                "train_group_sizes": {
                    str(group): sum(value == group for value in train_partition.values())
                    for group in (0, 1)
                },
                "held_out_group_sizes": {
                    str(group): sum(value == group for value in eval_partition.values())
                    for group in (0, 1)
                },
                "explained_variance": 1.0
                - sum(errors.values()) / sum(denominator.values()),
            }
        )
    return monolithic_error, pass_error, random_results, random_mean_error


def _bootstrap_delta_interval(
    reference_error: dict[str, float],
    candidate_error: dict[str, float],
    denominator: dict[str, float],
    *,
    seed: int,
) -> list[float]:
    ids = sorted(reference_error)
    numerator = np.asarray(
        [reference_error[key] - candidate_error[key] for key in ids],
        dtype=np.float64,
    )
    scale = np.asarray([denominator[key] for key in ids], dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, len(ids), size=(BOOTSTRAP_REPLICATES, len(ids)), endpoint=False
    )
    deltas = numerator[indices].sum(axis=1) / scale[indices].sum(axis=1)
    return [float(value) for value in np.quantile(deltas, [0.025, 0.975])]


def build_report(source_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    manifest, examples = _verify_source(source_dir)
    monolithic_rank = 64
    routed_rank = 32
    site_results = []
    for site_index, site in enumerate(SITES):
        rows = _load_site_rows(source_dir, manifest, examples, site)
        train_ids = sorted(rows["train"])
        eval_ids = sorted(rows["eval"])
        all_train = np.concatenate(
            [
                rows["train"][example_id][pass_index]
                for example_id in train_ids
                for pass_index in PASSES
            ]
        )
        global_center = all_train.mean(axis=0, dtype=np.float64)
        denominator = {
            example_id: sum(
                float(
                    np.square(
                        rows["eval"][example_id][pass_index] - global_center,
                        dtype=np.float64,
                    ).sum(dtype=np.float64)
                )
                for pass_index in PASSES
            )
            for example_id in eval_ids
        }
        mono_error, pass_error, random_results, random_mean_error = _arm_errors(
            rows, monolithic_rank=monolithic_rank, routed_rank=routed_rank
        )
        denominator_total = sum(denominator.values())
        monolithic_ev = 1.0 - sum(mono_error.values()) / denominator_total
        pass_ev = 1.0 - sum(pass_error.values()) / denominator_total
        random_evs = [float(row["explained_variance"]) for row in random_results]
        pass_minus_monolithic_ci = _bootstrap_delta_interval(
            mono_error,
            pass_error,
            denominator,
            seed=BOOTSTRAP_SEED + site_index,
        )
        pass_minus_random_mean_ci = _bootstrap_delta_interval(
            random_mean_error,
            pass_error,
            denominator,
            seed=BOOTSTRAP_SEED + 100 + site_index,
        )
        admission = (
            pass_minus_monolithic_ci[0] > 0.0
            and pass_minus_random_mean_ci[0] > 0.0
            and pass_ev > max(random_evs)
        )
        site_results.append(
            {
                "site": site,
                "held_out_common_denominator": denominator_total,
                "monolithic_explained_variance": monolithic_ev,
                "pass_compartment_explained_variance": pass_ev,
                "pass_minus_monolithic": pass_ev - monolithic_ev,
                "pass_minus_monolithic_paired_bootstrap_95_ci": pass_minus_monolithic_ci,
                "random_partition_controls": random_results,
                "random_partition_mean_explained_variance": float(np.mean(random_evs)),
                "random_partition_range_explained_variance": [
                    float(min(random_evs)),
                    float(max(random_evs)),
                ],
                "pass_minus_random_partition_mean": pass_ev
                - float(np.mean(random_evs)),
                "pass_minus_random_partition_mean_paired_bootstrap_95_ci": (
                    pass_minus_random_mean_ci
                ),
                "screening_admission": "pass" if admission else "fail",
            }
        )

    quality = _load_json(source_dir / "sae_quality.json")
    existing_quality = {
        site: {
            pass_key: {
                "explained_variance": float(values["explained_variance"]),
                "mean_l0": float(values["mean_l0"]),
            }
            for pass_key, values in quality[site]["held_out_quality"].items()
        }
        for site in SITES
    }
    body = {
        "schema": SCHEMA,
        "status": "pass",
        "source": {
            "path": source_dir.as_posix(),
            "manifest_sha256": sha256_file(source_dir / "manifest.json"),
            "examples_sha256": sha256_file(source_dir / "examples.json"),
            "sae_quality_sha256": sha256_file(source_dir / "sae_quality.json"),
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
            "monolithic_rank": monolithic_rank,
            "pass_compartment_rank_per_dictionary": routed_rank,
            "random_partition_rank_per_dictionary": routed_rank,
            "total_rank_per_arm": monolithic_rank,
            "random_partition_seeds": list(RANDOM_PARTITION_SEEDS),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "common_explained_variance_denominator": (
                "held-out squared distance from the all-pass training mean"
            ),
            "screening_admission_rule": (
                "pass routing lower 95% paired-bootstrap delta exceeds zero versus "
                "monolithic and random-mean controls, and point estimate exceeds "
                "every random-partition replicate"
            ),
        },
        "sites": site_results,
        "summary": {
            "sites": len(site_results),
            "pass_compartment_screening_admissions": sum(
                row["screening_admission"] == "pass" for row in site_results
            ),
        },
        "existing_monolithic_sae_context": {
            "training_loss": manifest["calibration_settings"]["sae_training_loss"],
            "training_steps": int(manifest["calibration_settings"]["sae_steps"]),
            "expansion": int(manifest["calibration_settings"]["sae_expansion"]),
            "held_out_quality": existing_quality,
            "context_only_not_recomputed_by_this_proxy": True,
        },
        "scientific_boundary": {
            "stored_empirical_bf16_activations_used": True,
            "new_model_inference_performed": False,
            "model_or_sae_weights_loaded_or_mutated": False,
            "new_sae_training_performed": False,
            "linear_proxy_is_not_an_sae_quality_result": True,
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
        raise ValueError("linear-proxy report content address differs")
    expected = build_report(Path(observed["source"]["path"]))
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("linear-proxy report does not reproduce")
    return {
        "schema": "sch.vpd-nanbeige-pass-compartment-linear-proxy-verification.v1",
        "status": "pass",
        "report_id": report_id,
        "sites": int(observed["summary"]["sites"]),
        "screening_admissions": int(
            observed["summary"]["pass_compartment_screening_admissions"]
        ),
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
