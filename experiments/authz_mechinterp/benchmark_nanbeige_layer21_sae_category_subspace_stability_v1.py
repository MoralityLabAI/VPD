from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.benchmark_nanbeige_layer21_sae_seed_objective_stability_v1 import (
    ARMS,
    SEEDS,
    _load,
    _load_condition,
    verify_report as verify_seed_objective_report,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-nanbeige-layer21-sae-category-subspace-stability.v1"
SUBSPACE_RANK = 16
PERMUTATION_REPLICATES = 255
PERMUTATION_SEED = 260913
MAXIMUM_MONTE_CARLO_P_VALUE = 0.05


def _basis(rows: np.ndarray, rank: int = SUBSPACE_RANK) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < rank or not np.all(np.isfinite(matrix)):
        raise ValueError("category decoder rows cannot support requested subspace rank")
    _, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    if singular_values[rank - 1] <= 1e-10:
        raise ValueError("category decoder subspace is rank deficient")
    return right[:rank].T


def _similarity(left: np.ndarray, right: np.ndarray) -> float:
    singular_values = np.linalg.svd(left.T @ right, compute_uv=False)
    return float(np.mean(np.square(singular_values)))


def _pair_result(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    pair_index: int,
) -> dict[str, Any]:
    left_bases = {
        pass_index: _basis(left["decoder"][left["labels"] == pass_index])
        for pass_index in (0, 1)
    }
    right_bases = {
        pass_index: _basis(right["decoder"][right["labels"] == pass_index])
        for pass_index in (0, 1)
    }
    same = {
        f"pass_{pass_index}": _similarity(
            left_bases[pass_index], right_bases[pass_index]
        )
        for pass_index in (0, 1)
    }
    opposite = {
        "left_pass_0_right_pass_1": _similarity(left_bases[0], right_bases[1]),
        "left_pass_1_right_pass_0": _similarity(left_bases[1], right_bases[0]),
    }
    observed = float(np.mean(list(same.values())))
    opposite_mean = float(np.mean(list(opposite.values())))

    generator = np.random.default_rng(PERMUTATION_SEED + pair_index)
    random_scores = []
    for _ in range(PERMUTATION_REPLICATES):
        permuted = generator.permutation(right["labels"])
        random_bases = {
            pass_index: _basis(right["decoder"][permuted == pass_index])
            for pass_index in (0, 1)
        }
        random_scores.append(
            float(
                np.mean(
                    [
                        _similarity(left_bases[index], random_bases[index])
                        for index in (0, 1)
                    ]
                )
            )
        )
    random_array = np.asarray(random_scores, dtype=np.float64)
    p_value = float(
        (np.sum(random_array >= observed) + 1) / (PERMUTATION_REPLICATES + 1)
    )
    comparison_class = (
        "cross_seed_same_objective"
        if left["arm"] == right["arm"]
        else "same_seed_cross_objective"
    )
    familywise_alpha = MAXIMUM_MONTE_CARLO_P_VALUE / (
        6 if comparison_class == "cross_seed_same_objective" else 3
    )
    diagnostic = (
        observed > opposite_mean
        and observed > float(np.quantile(random_array, 0.95))
        and p_value <= familywise_alpha
    )
    quality_pair = left["quality_qualified"] and right["quality_qualified"]
    return {
        "left": left["id"],
        "right": right["id"],
        "comparison_class": comparison_class,
        "subspace_rank_per_category": SUBSPACE_RANK,
        "same_category_subspace_similarity": same,
        "mean_same_category_subspace_similarity": observed,
        "opposite_category_subspace_similarity": opposite,
        "mean_opposite_category_subspace_similarity": opposite_mean,
        "same_minus_opposite": observed - opposite_mean,
        "label_permutation_mean": float(np.mean(random_array)),
        "label_permutation_95th_percentile": float(np.quantile(random_array, 0.95)),
        "label_permutation_p_value": p_value,
        "bonferroni_familywise_alpha": familywise_alpha,
        "diagnostic_category_subspace_stability_without_quality_gate": (
            "pass" if diagnostic else "fail"
        ),
        "quality_qualified_pair": quality_pair,
        "operative_category_subspace_replication_admission": (
            "pass" if diagnostic and quality_pair else "fail"
        ),
    }


def build_report(seed_objective_report_path: Path) -> dict[str, Any]:
    seed_report_path = seed_objective_report_path.resolve()
    verification = verify_seed_objective_report(seed_report_path)
    if verification.get("status") != "pass":
        raise ValueError("seed/objective predecessor did not verify")
    seed_report = _load(seed_report_path)
    root = Path(seed_report["source"]["path"])
    root_manifest = _load(root / "manifest.json")
    contract = _load(root / "contract.json")
    conditions = {
        f"{seed}_{arm}": _load_condition(root, root_manifest, contract, seed, arm)
        for seed in SEEDS
        for arm in ARMS
    }
    pair_ids: list[tuple[str, str]] = []
    for arm in ARMS:
        pair_ids.extend(
            (f"{left}_{arm}", f"{right}_{arm}")
            for left, right in ((42, 43), (42, 44), (43, 44))
        )
    pair_ids.extend((f"{seed}_{ARMS[0]}", f"{seed}_{ARMS[1]}") for seed in SEEDS)
    pairs = [
        _pair_result(conditions[left], conditions[right], pair_index=index)
        for index, (left, right) in enumerate(pair_ids)
    ]
    cross_seed = [row for row in pairs if row["comparison_class"] == "cross_seed_same_objective"]
    cross_objective = [row for row in pairs if row["comparison_class"] == "same_seed_cross_objective"]
    body = {
        "schema": SCHEMA,
        "status": "pass",
        "source": {
            "seed_objective_report_path": seed_report_path.as_posix(),
            "seed_objective_report_sha256": sha256_file(seed_report_path),
            "seed_objective_report_id": seed_report["report_id"],
            "site": "model.layers.21",
        },
        "design": {
            "comparison": "equal_rank_category_decoder_row_subspaces",
            "subspace_rank_per_category": SUBSPACE_RANK,
            "seeds": list(SEEDS),
            "objectives": list(ARMS),
            "cross_seed_same_objective_pairs": len(cross_seed),
            "same_seed_cross_objective_pairs": len(cross_objective),
            "label_permutation_replicates": PERMUTATION_REPLICATES,
            "label_permutation_seed": PERMUTATION_SEED,
            "maximum_monte_carlo_p_value": MAXIMUM_MONTE_CARLO_P_VALUE,
            "multiple_comparison_control": (
                "bonferroni_within_cross_seed_6_and_cross_objective_3_pair_families"
            ),
            "quality_gate_applied_after_diagnostic_subspace_test": True,
            "posthoc_existing_source_analysis": True,
        },
        "pairs": pairs,
        "summary": {
            "pair_checks": len(pairs),
            "cross_seed_pair_checks": len(cross_seed),
            "cross_seed_diagnostic_subspace_supports_without_quality_gate": sum(
                row["diagnostic_category_subspace_stability_without_quality_gate"] == "pass"
                for row in cross_seed
            ),
            "same_seed_cross_objective_pair_checks": len(cross_objective),
            "same_seed_cross_objective_diagnostic_subspace_supports_without_quality_gate": sum(
                row["diagnostic_category_subspace_stability_without_quality_gate"] == "pass"
                for row in cross_objective
            ),
            "operative_category_subspace_replication_admissions": sum(
                row["operative_category_subspace_replication_admission"] == "pass"
                for row in pairs
            ),
        },
        "scientific_boundary": {
            "individual_feature_matching_and_subspace_stability_are_distinct": True,
            "layer21_only": True,
            "all_source_conditions_fail_reconstruction_quality": True,
            "diagnostic_subspace_support_cannot_override_quality_gate": True,
            "does_not_test_layer11_or_final_norm": True,
            "new_model_inference_performed": False,
            "new_sae_training_performed": False,
            "sae_decoder_weights_loaded_for_read_only_analysis": True,
            "model_weights_loaded": False,
            "weights_mutated": False,
            "execution_authorized": False,
            "paid_compute_authorized": False,
        },
        "provenance": {"runner_sha256": sha256_file(Path(__file__))},
    }
    return {**body, "report_id": content_sha256(body)}


def verify_report(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    report_id = str(body.pop("report_id", ""))
    if content_sha256(body) != report_id:
        raise ValueError("layer21 category-subspace report content address differs")
    expected = build_report(Path(observed["source"]["seed_objective_report_path"]))
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("layer21 category-subspace report does not reproduce")
    return {
        "schema": "sch.vpd-nanbeige-layer21-sae-category-subspace-stability-verification.v1",
        "status": "pass",
        "report_id": report_id,
        **observed["summary"],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-objective-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_report(args.verify)
    else:
        if args.seed_objective_report is None or args.output is None:
            parser.error("--seed-objective-report and --output are required unless verifying")
        result = build_report(args.seed_objective_report)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
