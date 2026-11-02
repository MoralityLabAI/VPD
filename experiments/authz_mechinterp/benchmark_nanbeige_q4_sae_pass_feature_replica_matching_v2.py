from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_feature_identity_v1 import (
    COSINE_THRESHOLDS,
    MAXIMUM_MONTE_CARLO_P_VALUE,
    MINIMUM_FEATURES_PER_PASS,
    PERMUTATION_REPLICATES,
    PERMUTATION_SEED,
    PRIMARY_DIAGNOSTIC_THRESHOLD,
    _labels,
    _normalized_decoder,
)
from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_selectivity_v1 import (
    SITES,
    _assign_features,
    _verify_and_load,
    verify_report as verify_selectivity_report,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.evaluate_sae_feature_replica_matching_v2 import (
    MINIMUM_PROFILE_COSINE,
    mutual_signed_decoder_matches,
    verify_benchmark as verify_matcher_benchmark,
)


SCHEMA = "sch.vpd-nanbeige-q4-sae-pass-feature-replica-matching.v2"
MINIMUM_BIDIRECTIONAL_CATEGORY_COVERAGE = 0.10


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _category_profiles(train_mass: np.ndarray) -> np.ndarray:
    profiles = train_mass.sum(axis=0, dtype=np.float64).T
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    return np.divide(
        profiles,
        norms,
        out=np.zeros_like(profiles),
        where=norms > 0.0,
    )


def _label_permutation_p_value(
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    matched_left: np.ndarray,
    matched_right: np.ndarray,
    observed_same_category: int,
    *,
    seed: int,
) -> float:
    generator = np.random.default_rng(seed)
    exceed = 0
    for _ in range(PERMUTATION_REPLICATES):
        permuted = generator.permutation(right_labels)
        score = int(
            np.sum(left_labels[matched_left] == permuted[matched_right])
        )
        exceed += score >= observed_same_category
    return (exceed + 1) / (PERMUTATION_REPLICATES + 1)


def build_report(
    report_2048_path: Path,
    report_8192_path: Path,
    matcher_benchmark_path: Path,
) -> dict[str, Any]:
    report_paths = (report_2048_path.resolve(), report_8192_path.resolve())
    reports = [_load(path) for path in report_paths]
    for path in report_paths:
        if verify_selectivity_report(path).get("status") != "pass":
            raise ValueError("source pass-selectivity report did not verify")
    matcher_verification = verify_matcher_benchmark(matcher_benchmark_path)
    if matcher_verification.get("expected_outcomes_reproduced") != 4:
        raise ValueError("V2 matcher benchmark outcomes differ")

    sae_dirs = [Path(report["source"]["path"]) for report in reports]
    loaded_sources = [_verify_and_load(path) for path in sae_dirs]
    manifests = [value[0] for value in loaded_sources]
    qualities = [value[1] for value in loaded_sources]
    features = [value[2] for value in loaded_sources]
    if manifests[0]["seed"] != manifests[1]["seed"]:
        raise ValueError("same-seed diagnostic source seeds differ")
    if manifests[0]["training_ids"] != manifests[1]["training_ids"]:
        raise ValueError("same-seed diagnostic training IDs differ")
    if manifests[0]["evaluation_ids"] != manifests[1]["evaluation_ids"]:
        raise ValueError("same-seed diagnostic evaluation IDs differ")

    quality_admissions = [
        {
            row["site"]: row["monolithic_sae_internal_pass_specialization_admission"]
            for row in report["sites"]
        }
        for report in reports
    ]
    sites = []
    for site_index, site in enumerate(SITES):
        masks = [_assign_features(feature_set[site]["train"]) for feature_set in features]
        full_labels = [_labels(value) for value in masks]
        selective_indices = [np.flatnonzero(labels >= 0) for labels in full_labels]
        selective_labels = [labels[index] for labels, index in zip(full_labels, selective_indices)]
        profiles = [
            _category_profiles(feature_set[site]["train"])[index]
            for feature_set, index in zip(features, selective_indices)
        ]
        decoders = [
            _normalized_decoder(sae_dir / quality[site]["dictionary"])[index]
            for sae_dir, quality, index in zip(sae_dirs, qualities, selective_indices)
        ]

        threshold_rows = []
        for threshold_index, threshold in enumerate(COSINE_THRESHOLDS):
            match = mutual_signed_decoder_matches(
                decoders[0],
                decoders[1],
                minimum_cosine=threshold,
                block_rows=128,
            )
            matched_left = np.asarray(match["matched_left_indices"], dtype=np.int64)
            matched_right = np.asarray(match["matched_right_indices"], dtype=np.int64)
            left_matched_labels = selective_labels[0][matched_left]
            right_matched_labels = selective_labels[1][matched_right]
            same_category = left_matched_labels == right_matched_labels
            profile_cosines = np.sum(
                profiles[0][matched_left] * profiles[1][matched_right], axis=1
            )
            profile_qualified = profile_cosines >= MINIMUM_PROFILE_COSINE
            same_category_by_pass = {
                f"pass_{pass_index}": int(
                    np.sum(same_category & (left_matched_labels == pass_index))
                )
                for pass_index in (0, 1)
            }
            profile_qualified_same_by_pass = {
                f"pass_{pass_index}": int(
                    np.sum(
                        same_category
                        & profile_qualified
                        & (left_matched_labels == pass_index)
                    )
                )
                for pass_index in (0, 1)
            }
            coverage_by_pass = {}
            for pass_index in (0, 1):
                left_count = int(np.sum(selective_labels[0] == pass_index))
                right_count = int(np.sum(selective_labels[1] == pass_index))
                count = same_category_by_pass[f"pass_{pass_index}"]
                coverage_by_pass[f"pass_{pass_index}"] = float(
                    min(count / max(1, left_count), count / max(1, right_count))
                )
            observed_same = int(np.sum(same_category))
            p_value = _label_permutation_p_value(
                selective_labels[0],
                selective_labels[1],
                matched_left,
                matched_right,
                observed_same,
                seed=PERMUTATION_SEED + 100 * site_index + threshold_index,
            )
            threshold_rows.append(
                {
                    "minimum_signed_decoder_cosine": threshold,
                    "mutual_one_to_one_matches": match["mutual_matches"],
                    "same_category_matches": same_category_by_pass,
                    "profile_qualified_same_category_matches": (
                        profile_qualified_same_by_pass
                    ),
                    "bidirectional_category_coverage": coverage_by_pass,
                    "opposite_category_matches": int(np.sum(~same_category)),
                    "profile_rejected_matches": int(np.sum(~profile_qualified)),
                    "target_label_permutation_p_value": p_value,
                }
            )

        primary = next(
            row
            for row in threshold_rows
            if row["minimum_signed_decoder_cosine"] == PRIMARY_DIAGNOSTIC_THRESHOLD
        )
        quality_qualified = all(value[site] == "pass" for value in quality_admissions)
        provisional = (
            quality_qualified
            and min(primary["same_category_matches"].values())
            >= MINIMUM_FEATURES_PER_PASS
            and primary["profile_qualified_same_category_matches"]
            == primary["same_category_matches"]
            and min(primary["bidirectional_category_coverage"].values())
            >= MINIMUM_BIDIRECTIONAL_CATEGORY_COVERAGE
            and primary["opposite_category_matches"] == 0
            and primary["target_label_permutation_p_value"]
            <= MAXIMUM_MONTE_CARLO_P_VALUE
        )
        sites.append(
            {
                "site": site,
                "selective_feature_counts": {
                    "2048": {
                        f"pass_{pass_index}": int(np.sum(selective_labels[0] == pass_index))
                        for pass_index in (0, 1)
                    },
                    "8192": {
                        f"pass_{pass_index}": int(np.sum(selective_labels[1] == pass_index))
                        for pass_index in (0, 1)
                    },
                },
                "threshold_sweep": threshold_rows,
                "quality_qualified_in_both_archives": quality_qualified,
                "provisional_same_seed_v2_stratum_replication_support": (
                    "pass" if provisional else "fail"
                ),
                "operative_fresh_seed_stratum_replication_admission": "fail",
            }
        )

    body = {
        "schema": SCHEMA,
        "status": "pass",
        "sources": [
            {
                "budget": budget,
                "report_path": path.as_posix(),
                "report_sha256": sha256_file(path),
                "report_id": report["report_id"],
                "sae_path": sae_dir.as_posix(),
                "sae_manifest_sha256": sha256_file(sae_dir / "manifest.json"),
            }
            for budget, path, report, sae_dir in zip(
                (2048, 8192), report_paths, reports, sae_dirs
            )
        ],
        "matcher": {
            "version": "v2",
            "benchmark_path": matcher_benchmark_path.resolve().as_posix(),
            "benchmark_sha256": sha256_file(matcher_benchmark_path),
            "benchmark_id": matcher_verification["benchmark_id"],
            "signed_decoder_cosine": True,
            "mutual_nearest_neighbor": True,
            "one_to_one": True,
            "matching_pool": "training_assigned_pass_selective_features_only",
            "minimum_profile_cosine": MINIMUM_PROFILE_COSINE,
        },
        "design": {
            "cosine_threshold_sweep": list(COSINE_THRESHOLDS),
            "primary_diagnostic_cosine_threshold": PRIMARY_DIAGNOSTIC_THRESHOLD,
            "minimum_profile_qualified_features_per_pass": MINIMUM_FEATURES_PER_PASS,
            "minimum_bidirectional_category_coverage": MINIMUM_BIDIRECTIONAL_CATEGORY_COVERAGE,
            "target_label_permutation_replicates": PERMUTATION_REPLICATES,
            "permutation_preserves_target_pass_assignment_counts": True,
            "maximum_monte_carlo_p_value": MAXIMUM_MONTE_CARLO_P_VALUE,
            "thresholds_are_diagnostic_and_posthoc_for_existing_sources": True,
        },
        "sites": sites,
        "summary": {
            "site_checks": len(sites),
            "quality_qualified_sites": [
                row["site"] for row in sites if row["quality_qualified_in_both_archives"]
            ],
            "provisional_same_seed_v2_stratum_replication_supports": sum(
                row["provisional_same_seed_v2_stratum_replication_support"] == "pass"
                for row in sites
            ),
            "provisional_same_seed_v2_stratum_replication_sites": [
                row["site"]
                for row in sites
                if row["provisional_same_seed_v2_stratum_replication_support"] == "pass"
            ],
            "operative_fresh_seed_stratum_replication_admissions": 0,
        },
        "scientific_boundary": {
            "same_seed_and_training_ids_limit_independence": True,
            "matching_pool_is_training_assigned_only": True,
            "cosine_and_coverage_thresholds_are_posthoc_for_these_sources": True,
            "decoder_and_profile_matching_is_not_semantic_interpretation": True,
            "full_dictionary_replication_not_claimed": True,
            "fresh_seed_family_disjoint_replication_required": True,
            "new_model_inference_performed": False,
            "new_sae_training_performed": False,
            "sae_decoder_weights_loaded_for_read_only_analysis": True,
            "model_weights_loaded": False,
            "weights_mutated": False,
            "sae_training_authorized": False,
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
        raise ValueError("V2 stratum-replication report content address differs")
    expected = build_report(
        Path(observed["sources"][0]["report_path"]),
        Path(observed["sources"][1]["report_path"]),
        Path(observed["matcher"]["benchmark_path"]),
    )
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("V2 stratum-replication report does not reproduce")
    return {
        "schema": "sch.vpd-nanbeige-q4-sae-pass-feature-replica-matching-verification.v2",
        "status": "pass",
        "report_id": report_id,
        **observed["summary"],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-2048", type=Path)
    parser.add_argument("--report-8192", type=Path)
    parser.add_argument("--matcher-benchmark", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_report(args.verify)
    else:
        if any(
            value is None
            for value in (
                args.report_2048,
                args.report_8192,
                args.matcher_benchmark,
                args.output,
            )
        ):
            parser.error("source reports, matcher benchmark, and output are required")
        result = build_report(
            args.report_2048,
            args.report_8192,
            args.matcher_benchmark,
        )
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
