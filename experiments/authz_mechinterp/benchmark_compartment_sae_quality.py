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
from experiments.authz_mechinterp.evaluate_compartment_sae_quality import (
    DEFAULT_QUALITY_POLICY,
    evaluate_compartment_bundle,
    evaluate_compartment_bundle_streaming,
)


CATEGORIES = [
    "valid_authority",
    "missing_or_untrusted_authority",
    "signature_invalid",
    "time_inactive",
    "revoked",
    "replayed",
    "scope_invalid",
    "other_invalid",
]


def synthetic_holdout(seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    hidden_size = 4 + 2 * len(CATEGORIES)
    basis = np.eye(hidden_size, dtype=np.float32)
    category_basis = {
        category: basis[:, 4 + 2 * index : 4 + 2 * index + 2]
        for index, category in enumerate(CATEGORIES)
    }
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for category in CATEGORIES:
        for _ in range(64):
            shared = np.asarray(
                [rng.uniform(2.0, 3.0), rng.uniform(0.5, 1.0), *rng.uniform(0.5, 1.5, 2)],
                dtype=np.float32,
            )
            specific = np.asarray(
                [rng.uniform(2.0, 3.0), rng.uniform(0.5, 1.0)], dtype=np.float32
            )
            value = basis[:, :4] @ shared + category_basis[category] @ specific
            value += rng.normal(0.0, 0.005, hidden_size).astype(np.float32)
            rows.append(value)
            labels.append(category)
    return np.stack(rows), np.asarray(labels), {
        "shared": basis[:, :4],
        **category_basis,
    }


def dictionary_from_basis(
    *,
    dictionary_id: str,
    compartment: str,
    replica_id: str,
    basis: np.ndarray,
    transform: np.ndarray | None = None,
    extra_dead_features: int = 0,
) -> dict[str, Any]:
    feature_count = basis.shape[1]
    transform_value = (
        np.eye(feature_count, dtype=np.float32)
        if transform is None
        else np.asarray(transform, dtype=np.float32)
    )
    raw_decoder = (basis @ transform_value).T
    scales = np.linalg.norm(raw_decoder, axis=1)
    decoder = raw_decoder / scales[:, None]
    encoder = np.diag(scales) @ np.linalg.inv(transform_value) @ basis.T
    if extra_dead_features:
        decoder = np.vstack(
            [decoder, np.zeros((extra_dead_features, decoder.shape[1]), dtype=np.float32)]
        )
        encoder = np.vstack(
            [encoder, np.zeros((extra_dead_features, encoder.shape[1]), dtype=np.float32)]
        )
    return {
        "dictionary_id": dictionary_id,
        "compartment": compartment,
        "replica_id": replica_id,
        "decoder": decoder.astype(np.float32),
        "encoder_weight": encoder.astype(np.float32),
        "encoder_bias": np.zeros(decoder.shape[0], dtype=np.float32),
        "pre_bias": np.zeros(decoder.shape[1], dtype=np.float32),
        "top_k": decoder.shape[0],
    }


def clean_dictionaries(bases: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replica in ("seed-17", "seed-29"):
        rows.append(
            dictionary_from_basis(
                dictionary_id=f"shared-{replica}",
                compartment="shared_cross_category",
                replica_id=replica,
                basis=bases["shared"],
            )
        )
        for category in CATEGORIES:
            rows.append(
                dictionary_from_basis(
                    dictionary_id=f"{category}-{replica}",
                    compartment=category,
                    replica_id=replica,
                    basis=np.column_stack([bases["shared"], bases[category]]),
                )
            )
    return rows


def scenario_dictionaries(
    scenario: str, bases: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows = clean_dictionaries(bases)
    if scenario == "clean_replicated_compartments":
        return rows
    if scenario == "collapsed_specific_to_shared":
        for row in rows:
            if row["compartment"] == "shared_cross_category":
                continue
            replacement = dictionary_from_basis(
                dictionary_id=row["dictionary_id"],
                compartment=row["compartment"],
                replica_id=row["replica_id"],
                basis=bases["shared"],
            )
            row.update(replacement)
        return rows
    if scenario == "cyclic_compartment_swap":
        for row in rows:
            if row["compartment"] == "shared_cross_category":
                continue
            target_index = CATEGORIES.index(row["compartment"])
            wrong = CATEGORIES[(target_index + 1) % len(CATEGORIES)]
            row.update(
                dictionary_from_basis(
                    dictionary_id=row["dictionary_id"],
                    compartment=row["compartment"],
                    replica_id=row["replica_id"],
                    basis=np.column_stack([bases["shared"], bases[wrong]]),
                )
            )
        return rows
    if scenario == "dead_feature_overgrowth":
        return [
            dictionary_from_basis(
                dictionary_id=row["dictionary_id"],
                compartment=row["compartment"],
                replica_id=row["replica_id"],
                basis=(
                    bases["shared"]
                    if row["compartment"] == "shared_cross_category"
                    else np.column_stack(
                        [bases["shared"], bases[row["compartment"]]]
                    )
                ),
                extra_dead_features=8,
            )
            for row in rows
        ]
    if scenario == "unstable_replica_basis":
        category = CATEGORIES[0]
        for index, row in enumerate(rows):
            if row["compartment"] == category and row["replica_id"] == "seed-29":
                width = 6
                transform = np.eye(width, dtype=np.float32)
                transform[-2:, -2:] = np.asarray([[1.0, 1.0], [0.0, 1.0]])
                rows[index] = dictionary_from_basis(
                    dictionary_id=row["dictionary_id"],
                    compartment=category,
                    replica_id=row["replica_id"],
                    basis=np.column_stack([bases["shared"], bases[category]]),
                    transform=transform,
                )
        return rows
    raise ValueError(f"unknown synthetic quality scenario: {scenario}")


def build_benchmark(seed: int = 260829) -> dict[str, Any]:
    activations, labels, bases = synthetic_holdout(seed)
    policy = dict(DEFAULT_QUALITY_POLICY)
    scenarios = []
    for scenario_id, expectation in (
        ("clean_replicated_compartments", "pass"),
        ("collapsed_specific_to_shared", "reject"),
        ("cyclic_compartment_swap", "reject"),
        ("dead_feature_overgrowth", "reject"),
        ("unstable_replica_basis", "reject"),
    ):
        result = evaluate_compartment_bundle(
            activations=activations,
            categories=labels,
            dictionaries=scenario_dictionaries(scenario_id, bases),
            required_specific_categories=CATEGORIES,
            policy=policy,
        )
        checks: dict[str, bool] = {"expected_status": result["status"] == expectation}
        if scenario_id == "collapsed_specific_to_shared":
            checks["specificity_gate_rejected"] = all(
                not row["checks"].get("target_cross_category_ev_advantage", True)
                for row in result["dictionary_results"]
                if row["compartment"] != "shared_cross_category"
            )
        elif scenario_id == "cyclic_compartment_swap":
            checks["target_reconstruction_rejected"] = all(
                not row["checks"].get("target_cross_category_ev_advantage", True)
                for row in result["dictionary_results"]
                if row["compartment"] != "shared_cross_category"
            )
        elif scenario_id == "dead_feature_overgrowth":
            checks["dead_feature_gate_rejected"] = all(
                not row["checks"]["dead_feature_fraction"]
                for row in result["dictionary_results"]
            )
        elif scenario_id == "unstable_replica_basis":
            checks["replica_gate_rejected"] = any(
                row["status"] == "reject" for row in result["replica_results"]
            )
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "expectation": expectation,
                "observed_status": result["status"],
                "checks": checks,
                "status": "pass" if all(checks.values()) else "fail",
                "result": result,
            }
        )
    clean = clean_dictionaries(bases)
    by_id = {row["dictionary_id"]: row for row in clean}
    descriptors = [
        {
            "dictionary_id": row["dictionary_id"],
            "compartment": row["compartment"],
            "replica_id": row["replica_id"],
        }
        for row in clean
    ]
    read_sizes: list[int] = []
    loaded_ids: list[str] = []

    def read_rows(indices: np.ndarray) -> np.ndarray:
        read_sizes.append(len(indices))
        return activations[indices]

    def load_dictionary(descriptor: dict[str, Any]) -> dict[str, Any]:
        loaded_ids.append(str(descriptor["dictionary_id"]))
        return by_id[str(descriptor["dictionary_id"])]

    streaming = evaluate_compartment_bundle_streaming(
        row_count=len(activations),
        hidden_size=activations.shape[1],
        read_rows=read_rows,
        categories=labels,
        dictionary_descriptors=descriptors,
        load_dictionary=load_dictionary,
        required_specific_categories=CATEGORIES,
        policy=policy,
        batch_rows=17,
    )
    in_memory_clean = scenarios[0]["result"]
    in_memory_dictionary_gates = {
        row["dictionary_id"]: (row["checks"], row["status"])
        for row in in_memory_clean["dictionary_results"]
    }
    streaming_dictionary_gates = {
        row["dictionary_id"]: (row["checks"], row["status"])
        for row in streaming["dictionary_results"]
    }
    in_memory_replica_gates = {
        (row["compartment"], row["replica_id"]): (row["checks"], row["status"])
        for row in in_memory_clean["replica_results"]
    }
    streaming_replica_gates = {
        (row["compartment"], row["replica_id"]): (row["checks"], row["status"])
        for row in streaming["replica_results"]
    }
    streaming_checks = {
        "expected_status": streaming["status"] == "pass",
        "dictionary_gate_equivalence": (
            streaming_dictionary_gates == in_memory_dictionary_gates
        ),
        "replica_gate_equivalence": streaming_replica_gates == in_memory_replica_gates,
        "bounded_activation_reads": max(read_sizes) <= 17,
        "exact_total_activation_rows_read": sum(read_sizes)
        == len(activations) * len(clean),
        "one_dictionary_load_per_replica": len(loaded_ids) == len(set(loaded_ids))
        == len(clean),
        "forbidden_full_matrices_absent": all(
            streaming["coverage"]["resource_bounds"][field] is False
            for field in (
                "full_activation_matrix_materialized",
                "full_feature_matrix_materialized",
                "full_replica_cosine_matrix_materialized",
            )
        ),
    }
    scenarios.append(
        {
            "scenario_id": "clean_bounded_streaming_equivalence",
            "expectation": "pass",
            "observed_status": streaming["status"],
            "checks": streaming_checks,
            "status": "pass" if all(streaming_checks.values()) else "fail",
            "result": streaming,
        }
    )
    project_root = Path(__file__).resolve().parents[2]
    body = {
        "schema": "sch.vpd-compartment-sae-quality-synthetic-benchmark.v1",
        "status": "pass" if all(row["status"] == "pass" for row in scenarios) else "fail",
        "seed": seed,
        "design": {
            "rows": len(activations),
            "hidden_size": activations.shape[1],
            "categories": CATEGORIES,
            "replicas_per_compartment": 2,
            "known_shared_and_category_subspaces": True,
            "scenarios": len(scenarios),
            "bounded_streaming_equivalence_tested": True,
        },
        "policy": policy,
        "scenarios": scenarios,
        "source": {
            "runner_sha256": sha256_file(Path(__file__)),
            "evaluator_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/evaluate_compartment_sae_quality.py"
            ),
        },
        "scientific_boundary": {
            "synthetic_activations_only": True,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "does_not_establish_bonsai_sae_quality": True,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    body = dict(report)
    benchmark_id = str(body.pop("benchmark_id", ""))
    if content_sha256(body) != benchmark_id:
        raise ValueError("compartment SAE quality benchmark content address mismatch")
    expected = build_benchmark(seed=int(report["seed"]))
    if expected != report:
        raise ValueError("compartment SAE quality benchmark is not reproducible")
    if report["status"] != "pass":
        raise ValueError("compartment SAE quality benchmark expectations failed")
    return {
        "schema": "sch.vpd-compartment-sae-quality-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": benchmark_id,
        "scenarios": len(report["scenarios"]),
        "all_expectations_met": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or verify the synthetic compartment SAE quality benchmark."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--seed", type=int, default=260829)
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
