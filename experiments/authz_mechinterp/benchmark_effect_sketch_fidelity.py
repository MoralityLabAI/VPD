from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.effect_uncertainty import countsketch


HIDDEN_SIZE = 4096
TRIALS_PER_PROFILE_AND_STRUCTURE = 256
SKETCH_DIMENSIONS = (64, 128, 256, 512, 1024)
SKETCH_SEEDS = (17, 29, 41, 53, 67, 79, 97, 109)
MINIMUM_TRUE_DIRECTION_COSINE = 0.50
MINIMUM_TRUE_RELATIVE_NORM = 0.25
MAXIMUM_WORST_SEED_FALSE_ADMISSION_WILSON_UPPER = 0.01
MINIMUM_WORST_SEED_STABLE_PASS_RECALL_WILSON_LOWER = 0.99
MAXIMUM_WORST_SEED_DIRECTION_COSINE_ERROR_P99 = 0.15
MAXIMUM_WORST_SEED_RELATIVE_NORM_ERROR_P99 = 0.075

PROFILES: tuple[dict[str, Any], ...] = (
    {
        "id": "stable_pass_direction",
        "direction_cosine": 0.70,
        "relative_norm": 0.50,
        "class": "stable_pass",
    },
    {
        "id": "stable_pass_norm",
        "direction_cosine": 0.85,
        "relative_norm": 0.40,
        "class": "stable_pass",
    },
    {
        "id": "stable_fail_direction",
        "direction_cosine": 0.30,
        "relative_norm": 0.50,
        "class": "stable_fail",
    },
    {
        "id": "stable_fail_norm",
        "direction_cosine": 0.85,
        "relative_norm": 0.15,
        "class": "stable_fail",
    },
    {
        "id": "near_pass",
        "direction_cosine": 0.55,
        "relative_norm": 0.30,
        "class": "boundary_diagnostic",
    },
    {
        "id": "near_fail_direction",
        "direction_cosine": 0.45,
        "relative_norm": 0.30,
        "class": "boundary_diagnostic",
    },
    {
        "id": "near_fail_norm",
        "direction_cosine": 0.80,
        "relative_norm": 0.20,
        "class": "boundary_diagnostic",
    },
)

OPERATIONAL_THRESHOLDS: tuple[tuple[float, float], ...] = tuple(
    (direction, relative_norm)
    for direction in (0.50, 0.55, 0.60)
    for relative_norm in (0.25, 0.275, 0.30)
)


def _orthonormal_pair(
    rng: np.random.Generator, *, structure: str
) -> tuple[np.ndarray, np.ndarray]:
    if structure == "dense":
        reference = rng.normal(size=HIDDEN_SIZE)
        orthogonal = rng.normal(size=HIDDEN_SIZE)
    elif structure == "sparse_64":
        support = rng.choice(HIDDEN_SIZE, size=64, replace=False)
        reference = np.zeros(HIDDEN_SIZE, dtype=np.float64)
        orthogonal = np.zeros(HIDDEN_SIZE, dtype=np.float64)
        reference[support] = rng.normal(size=len(support))
        orthogonal[support] = rng.normal(size=len(support))
    else:
        raise ValueError(f"unsupported sketch-fidelity structure: {structure}")
    reference /= np.linalg.norm(reference)
    orthogonal -= reference * float(np.dot(orthogonal, reference))
    orthogonal_norm = float(np.linalg.norm(orthogonal))
    if orthogonal_norm <= 1e-12:
        raise ValueError("synthetic orthogonal direction is degenerate")
    orthogonal /= orthogonal_norm
    return reference, orthogonal


def build_cases(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    references: list[np.ndarray] = []
    cells: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for structure in ("dense", "sparse_64"):
        for profile in PROFILES:
            cosine = float(profile["direction_cosine"])
            relative_norm = float(profile["relative_norm"])
            for trial in range(TRIALS_PER_PROFILE_AND_STRUCTURE):
                reference, orthogonal = _orthonormal_pair(
                    rng, structure=structure
                )
                cell = relative_norm * (
                    cosine * reference
                    + np.sqrt(max(0.0, 1.0 - cosine * cosine)) * orthogonal
                )
                references.append(reference)
                cells.append(cell)
                metadata.append(
                    {
                        "structure": structure,
                        "profile_id": str(profile["id"]),
                        "profile_class": str(profile["class"]),
                        "trial": trial,
                        "exact_direction_cosine": cosine,
                        "exact_relative_norm": relative_norm,
                        "exact_pass": (
                            cosine >= MINIMUM_TRUE_DIRECTION_COSINE
                            and relative_norm >= MINIMUM_TRUE_RELATIVE_NORM
                        ),
                    }
                )
    return {
        "references": np.stack(references),
        "cells": np.stack(cells),
        "metadata": metadata,
    }


def projected_metrics(
    references: np.ndarray,
    cells: np.ndarray,
    *,
    dimension: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    projected_references = countsketch(references, dimension=dimension, seed=seed)
    projected_cells = countsketch(cells, dimension=dimension, seed=seed)
    reference_norms = np.linalg.norm(projected_references, axis=1)
    cell_norms = np.linalg.norm(projected_cells, axis=1)
    cosines = np.sum(projected_references * projected_cells, axis=1) / np.maximum(
        reference_norms * cell_norms, 1e-12
    )
    relative_norms = cell_norms / np.maximum(reference_norms, 1e-12)
    return cosines, relative_norms


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("Wilson interval counts are invalid")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return float(center - half_width), float(center + half_width)


def build_benchmark(seed: int = 250829) -> dict[str, Any]:
    cases = build_cases(seed)
    metadata = cases["metadata"]
    stable_pass = np.asarray(
        [row["profile_class"] == "stable_pass" for row in metadata], dtype=bool
    )
    stable_fail = np.asarray(
        [row["profile_class"] == "stable_fail" for row in metadata], dtype=bool
    )
    exact_cosines = np.asarray(
        [row["exact_direction_cosine"] for row in metadata], dtype=np.float64
    )
    exact_relative_norms = np.asarray(
        [row["exact_relative_norm"] for row in metadata], dtype=np.float64
    )
    candidates: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for dimension in SKETCH_DIMENSIONS:
        per_seed: list[dict[str, Any]] = []
        projected_by_seed: list[tuple[np.ndarray, np.ndarray]] = []
        for sketch_seed in SKETCH_SEEDS:
            cosines, relative_norms = projected_metrics(
                cases["references"],
                cases["cells"],
                dimension=dimension,
                seed=sketch_seed,
            )
            projected_by_seed.append((cosines, relative_norms))
            per_seed.append(
                {
                    "seed": sketch_seed,
                    "direction_cosine_absolute_error_p99": float(
                        np.quantile(np.abs(cosines - exact_cosines), 0.99)
                    ),
                    "relative_norm_absolute_error_p99": float(
                        np.quantile(
                            np.abs(relative_norms - exact_relative_norms), 0.99
                        )
                    ),
                }
            )
        diagnostics[str(dimension)] = per_seed
        worst_cosine_error_p99 = max(
            row["direction_cosine_absolute_error_p99"] for row in per_seed
        )
        worst_relative_norm_error_p99 = max(
            row["relative_norm_absolute_error_p99"] for row in per_seed
        )
        for direction_threshold, relative_norm_threshold in OPERATIONAL_THRESHOLDS:
            seed_rows: list[dict[str, Any]] = []
            for sketch_seed, (cosines, relative_norms) in zip(
                SKETCH_SEEDS, projected_by_seed, strict=True
            ):
                predicted_pass = (
                    (cosines >= direction_threshold)
                    & (relative_norms >= relative_norm_threshold)
                )
                stable_pass_successes = int(predicted_pass[stable_pass].sum())
                stable_pass_total = int(stable_pass.sum())
                stable_false_admissions = int(predicted_pass[stable_fail].sum())
                stable_fail_total = int(stable_fail.sum())
                recall_lower, recall_upper = wilson_interval(
                    stable_pass_successes, stable_pass_total
                )
                false_lower, false_upper = wilson_interval(
                    stable_false_admissions, stable_fail_total
                )
                seed_rows.append(
                    {
                        "seed": sketch_seed,
                        "stable_pass_recall": float(predicted_pass[stable_pass].mean()),
                        "stable_false_admission_rate": float(
                            predicted_pass[stable_fail].mean()
                        ),
                        "stable_pass_recall_wilson_95_lower": recall_lower,
                        "stable_pass_recall_wilson_95_upper": recall_upper,
                        "stable_false_admission_wilson_95_lower": false_lower,
                        "stable_false_admission_wilson_95_upper": false_upper,
                        "boundary_pass_rate": float(
                            predicted_pass[~(stable_pass | stable_fail)].mean()
                        ),
                    }
                )
            worst_recall = min(row["stable_pass_recall"] for row in seed_rows)
            worst_false_admission = max(
                row["stable_false_admission_rate"] for row in seed_rows
            )
            worst_recall_lower = min(
                row["stable_pass_recall_wilson_95_lower"] for row in seed_rows
            )
            worst_false_admission_upper = max(
                row["stable_false_admission_wilson_95_upper"] for row in seed_rows
            )
            status = (
                "pass"
                if worst_recall_lower
                >= MINIMUM_WORST_SEED_STABLE_PASS_RECALL_WILSON_LOWER
                and worst_false_admission_upper
                <= MAXIMUM_WORST_SEED_FALSE_ADMISSION_WILSON_UPPER
                and worst_cosine_error_p99
                <= MAXIMUM_WORST_SEED_DIRECTION_COSINE_ERROR_P99
                and worst_relative_norm_error_p99
                <= MAXIMUM_WORST_SEED_RELATIVE_NORM_ERROR_P99
                else "reject"
            )
            candidates.append(
                {
                    "dimension": dimension,
                    "operational_minimum_direction_cosine": direction_threshold,
                    "operational_minimum_relative_norm": relative_norm_threshold,
                    "worst_seed_stable_pass_recall": worst_recall,
                    "worst_seed_stable_false_admission_rate": worst_false_admission,
                    "worst_seed_stable_pass_recall_wilson_95_lower": worst_recall_lower,
                    "worst_seed_stable_false_admission_wilson_95_upper": (
                        worst_false_admission_upper
                    ),
                    "worst_seed_direction_cosine_absolute_error_p99": (
                        worst_cosine_error_p99
                    ),
                    "worst_seed_relative_norm_absolute_error_p99": (
                        worst_relative_norm_error_p99
                    ),
                    "status": status,
                    "per_seed": seed_rows,
                }
            )
    passing = [row for row in candidates if row["status"] == "pass"]
    passing.sort(
        key=lambda row: (
            int(row["dimension"]),
            float(row["operational_minimum_direction_cosine"]),
            float(row["operational_minimum_relative_norm"]),
        )
    )
    selected = None if not passing else dict(passing[0])
    if selected is not None:
        selected.pop("per_seed", None)
    body = {
        "schema": "sch.vpd-effect-sketch-fidelity-benchmark.v1",
        "status": "pass" if selected is not None else "fail_no_supported_candidate",
        "seed": seed,
        "source": {"runner_sha256": sha256_file(Path(__file__))},
        "design": {
            "hidden_size": HIDDEN_SIZE,
            "structures": ["dense", "sparse_64"],
            "trials_per_profile_and_structure": TRIALS_PER_PROFILE_AND_STRUCTURE,
            "profiles": list(PROFILES),
            "sketch_dimensions": list(SKETCH_DIMENSIONS),
            "sketch_seeds": list(SKETCH_SEEDS),
            "exact_gate": {
                "minimum_direction_cosine": MINIMUM_TRUE_DIRECTION_COSINE,
                "minimum_relative_norm": MINIMUM_TRUE_RELATIVE_NORM,
            },
            "selection_requirements": {
                "minimum_worst_seed_stable_pass_recall_wilson_95_lower": (
                    MINIMUM_WORST_SEED_STABLE_PASS_RECALL_WILSON_LOWER
                ),
                "maximum_worst_seed_stable_false_admission_wilson_95_upper": (
                    MAXIMUM_WORST_SEED_FALSE_ADMISSION_WILSON_UPPER
                ),
                "maximum_worst_seed_direction_cosine_absolute_error_p99": (
                    MAXIMUM_WORST_SEED_DIRECTION_COSINE_ERROR_P99
                ),
                "maximum_worst_seed_relative_norm_absolute_error_p99": (
                    MAXIMUM_WORST_SEED_RELATIVE_NORM_ERROR_P99
                ),
                "boundary_profiles_are_diagnostic_not_selection_targets": True,
            },
        },
        "selected_candidate": selected,
        "candidates": candidates,
        "metric_error_diagnostics": diagnostics,
        "scientific_boundary": {
            "synthetic_effects_only": True,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "does_not_establish_bonsai_component_categories": True,
            "purpose": (
                "Select a compact-summary contract that fails closed on stable synthetic "
                "effect profiles at the expected Bonsai hidden dimension."
            ),
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    body = dict(report)
    benchmark_id = str(body.pop("benchmark_id", ""))
    if content_sha256(body) != benchmark_id:
        raise ValueError("effect sketch fidelity benchmark content address mismatch")
    expected = build_benchmark(seed=int(report["seed"]))
    if expected != report:
        raise ValueError("effect sketch fidelity benchmark is not reproducible")
    if report.get("status") != "pass":
        raise ValueError("effect sketch fidelity benchmark has no supported candidate")
    return {
        "schema": "sch.vpd-effect-sketch-fidelity-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": benchmark_id,
        "selected_candidate": report["selected_candidate"],
        "cases": (
            len(PROFILES) * 2 * TRIALS_PER_PROFILE_AND_STRUCTURE
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or verify the production-shaped effect CountSketch benchmark."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--seed", type=int, default=250829)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_benchmark(args.verify)
    else:
        if args.output is None:
            parser.error("benchmark generation requires --output")
        result = build_benchmark(seed=args.seed)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
