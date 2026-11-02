from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.benchmark_nanbeige_layer21_sae_category_subspace_stability_v1 import (
    _basis,
    _similarity,
)
from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_feature_identity_v1 import (
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


SCHEMA = "sch.vpd-nanbeige-q4-sae-pass-category-subspace-transfer.v1"
SUBSPACE_RANKS = (4, 8, 16, 32)
PRIMARY_SUBSPACE_RANK = 16
PERMUTATION_REPLICATES = 255
PERMUTATION_SEED = 260917
MAXIMUM_FAMILYWISE_P_VALUE = 0.05


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _rank_metrics(
    left_bases: dict[int, np.ndarray],
    right_bases: dict[int, np.ndarray],
    rank: int,
) -> tuple[dict[str, float], dict[str, float], float, float]:
    same = {
        f"pass_{pass_index}": _similarity(
            left_bases[pass_index][:, :rank],
            right_bases[pass_index][:, :rank],
        )
        for pass_index in (0, 1)
    }
    opposite = {
        "left_pass_0_right_pass_1": _similarity(
            left_bases[0][:, :rank], right_bases[1][:, :rank]
        ),
        "left_pass_1_right_pass_0": _similarity(
            left_bases[1][:, :rank], right_bases[0][:, :rank]
        ),
    }
    return same, opposite, float(np.mean(list(same.values()))), float(
        np.mean(list(opposite.values()))
    )


def _projected_subset_basis(
    left_projection: np.ndarray,
    right_gram: np.ndarray,
    indices: np.ndarray,
    rank: int,
) -> np.ndarray:
    subset_gram = right_gram[np.ix_(indices, indices)]
    eigenvalues, eigenvectors = np.linalg.eigh(subset_gram)
    order = np.argsort(eigenvalues)[::-1][:rank]
    selected_values = eigenvalues[order]
    if selected_values[-1] <= 1e-10:
        raise ValueError("permuted category decoder subspace is rank deficient")
    return (
        left_projection[:, indices]
        @ eigenvectors[:, order]
        / np.sqrt(selected_values)[None, :]
    )


def _site_result(
    site: str,
    left_decoder: np.ndarray,
    right_decoder: np.ndarray,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    *,
    site_index: int,
    quality_qualified: bool,
) -> dict[str, Any]:
    maximum_rank = max(SUBSPACE_RANKS)
    left_bases = {
        pass_index: _basis(
            left_decoder[left_labels == pass_index], rank=maximum_rank
        )
        for pass_index in (0, 1)
    }
    right_bases = {
        pass_index: _basis(
            right_decoder[right_labels == pass_index], rank=maximum_rank
        )
        for pass_index in (0, 1)
    }
    observed = {
        rank: _rank_metrics(left_bases, right_bases, rank)
        for rank in SUBSPACE_RANKS
    }
    right_gram = right_decoder @ right_decoder.T
    left_projections = {
        pass_index: left_bases[pass_index].T @ right_decoder.T
        for pass_index in (0, 1)
    }
    random_scores = {rank: [] for rank in SUBSPACE_RANKS}
    generator = np.random.default_rng(PERMUTATION_SEED + site_index)
    for _ in range(PERMUTATION_REPLICATES):
        permuted = generator.permutation(right_labels)
        projected_bases = {
            pass_index: _projected_subset_basis(
                left_projections[pass_index],
                right_gram,
                np.flatnonzero(permuted == pass_index),
                maximum_rank,
            )
            for pass_index in (0, 1)
        }
        for rank in SUBSPACE_RANKS:
            random_scores[rank].append(
                float(
                    np.mean(
                        [
                            np.mean(
                                np.square(
                                    np.linalg.svd(
                                        projected_bases[index][:rank, :rank],
                                        compute_uv=False,
                                    )
                                )
                            )
                            for index in (0, 1)
                        ]
                    )
                )
            )

    rank_rows = []
    for rank in SUBSPACE_RANKS:
        same, opposite, score, opposite_score = observed[rank]
        null = np.asarray(random_scores[rank], dtype=np.float64)
        p_value = float(
            (np.sum(null >= score) + 1) / (PERMUTATION_REPLICATES + 1)
        )
        rank_rows.append(
            {
                "subspace_rank_per_category": rank,
                "same_category_subspace_similarity": same,
                "mean_same_category_subspace_similarity": score,
                "opposite_category_subspace_similarity": opposite,
                "mean_opposite_category_subspace_similarity": opposite_score,
                "same_minus_opposite": score - opposite_score,
                "label_permutation_mean": float(np.mean(null)),
                "label_permutation_95th_percentile": float(np.quantile(null, 0.95)),
                "label_permutation_p_value": p_value,
            }
        )
    primary = next(
        row
        for row in rank_rows
        if row["subspace_rank_per_category"] == PRIMARY_SUBSPACE_RANK
    )
    site_familywise_alpha = MAXIMUM_FAMILYWISE_P_VALUE / len(SITES)
    diagnostic = (
        primary["mean_same_category_subspace_similarity"]
        > primary["mean_opposite_category_subspace_similarity"]
        and primary["mean_same_category_subspace_similarity"]
        > primary["label_permutation_95th_percentile"]
        and primary["label_permutation_p_value"] <= site_familywise_alpha
    )
    sensitivity_direction_consistent = all(
        row["same_minus_opposite"] > 0.0 for row in rank_rows
    )
    return {
        "site": site,
        "selective_feature_counts": {
            "2048": {
                f"pass_{pass_index}": int(np.sum(left_labels == pass_index))
                for pass_index in (0, 1)
            },
            "8192": {
                f"pass_{pass_index}": int(np.sum(right_labels == pass_index))
                for pass_index in (0, 1)
            },
        },
        "rank_sensitivity": rank_rows,
        "primary_rank": PRIMARY_SUBSPACE_RANK,
        "bonferroni_site_familywise_alpha": site_familywise_alpha,
        "all_reported_ranks_same_minus_opposite_positive": (
            sensitivity_direction_consistent
        ),
        "quality_qualified_in_both_archives": quality_qualified,
        "diagnostic_same_seed_category_subspace_transfer_support": (
            "pass" if diagnostic else "fail"
        ),
        "provisional_same_seed_quality_qualified_subspace_support": (
            "pass" if diagnostic and quality_qualified else "fail"
        ),
        "operative_fresh_seed_category_subspace_replication_admission": "fail",
    }


def build_report(report_2048_path: Path, report_8192_path: Path) -> dict[str, Any]:
    report_paths = (report_2048_path.resolve(), report_8192_path.resolve())
    reports = [_load(path) for path in report_paths]
    for path in report_paths:
        if verify_selectivity_report(path).get("status") != "pass":
            raise ValueError("source pass-selectivity report did not verify")
    sae_dirs = [Path(report["source"]["path"]) for report in reports]
    loaded = [_verify_and_load(path) for path in sae_dirs]
    manifests = [row[0] for row in loaded]
    qualities = [row[1] for row in loaded]
    features = [row[2] for row in loaded]
    if manifests[0]["seed"] != manifests[1]["seed"]:
        raise ValueError("same-seed transfer source seeds differ")
    if manifests[0]["training_ids"] != manifests[1]["training_ids"]:
        raise ValueError("same-seed transfer training IDs differ")
    if manifests[0]["evaluation_ids"] != manifests[1]["evaluation_ids"]:
        raise ValueError("same-seed transfer evaluation IDs differ")
    quality_admissions = [
        {
            row["site"]: row["monolithic_sae_internal_pass_specialization_admission"]
            for row in report["sites"]
        }
        for report in reports
    ]
    sites = []
    for site_index, site in enumerate(SITES):
        masks = [
            _assign_features(feature_set[site]["train"])
            for feature_set in features
        ]
        labels = [_labels(mask) for mask in masks]
        indices = [np.flatnonzero(value >= 0) for value in labels]
        selective_labels = [
            value[index] for value, index in zip(labels, indices)
        ]
        decoders = [
            _normalized_decoder(sae_dir / quality[site]["dictionary"])[index]
            for sae_dir, quality, index in zip(sae_dirs, qualities, indices)
        ]
        sites.append(
            _site_result(
                site,
                decoders[0],
                decoders[1],
                selective_labels[0],
                selective_labels[1],
                site_index=site_index,
                quality_qualified=all(
                    admissions[site] == "pass"
                    for admissions in quality_admissions
                ),
            )
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
        "design": {
            "comparison": "equal_rank_training_assigned_pass_decoder_row_subspaces",
            "primary_subspace_rank_per_category": PRIMARY_SUBSPACE_RANK,
            "reported_rank_sensitivity": list(SUBSPACE_RANKS),
            "sensitivity_ranks_do_not_supply_alternate_admission_tests": True,
            "label_permutation_replicates": PERMUTATION_REPLICATES,
            "label_permutation_seed": PERMUTATION_SEED,
            "maximum_familywise_p_value": MAXIMUM_FAMILYWISE_P_VALUE,
            "multiple_comparison_control": "bonferroni_across_three_sites_at_primary_rank_only",
            "posthoc_existing_source_analysis": True,
        },
        "sites": sites,
        "summary": {
            "site_checks": len(sites),
            "quality_qualified_sites": [
                row["site"]
                for row in sites
                if row["quality_qualified_in_both_archives"]
            ],
            "diagnostic_same_seed_category_subspace_transfer_supports": sum(
                row["diagnostic_same_seed_category_subspace_transfer_support"]
                == "pass"
                for row in sites
            ),
            "diagnostic_same_seed_category_subspace_transfer_sites": [
                row["site"]
                for row in sites
                if row["diagnostic_same_seed_category_subspace_transfer_support"]
                == "pass"
            ],
            "provisional_same_seed_quality_qualified_subspace_supports": sum(
                row["provisional_same_seed_quality_qualified_subspace_support"]
                == "pass"
                for row in sites
            ),
            "provisional_same_seed_quality_qualified_subspace_sites": [
                row["site"]
                for row in sites
                if row["provisional_same_seed_quality_qualified_subspace_support"]
                == "pass"
            ],
            "operative_fresh_seed_category_subspace_replication_admissions": 0,
        },
        "scientific_boundary": {
            "same_seed_and_example_ids_limit_independence": True,
            "training_assigned_selective_feature_rows_only": True,
            "primary_rank_and_site_multiplicity_control_are_weight_bearing": True,
            "rank_sensitivity_is_not_an_admission_search": True,
            "subspace_similarity_is_not_feature_identity": True,
            "subspace_similarity_is_not_semantic_interpretation": True,
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
        raise ValueError("Q4 pass-category subspace report content address differs")
    expected = build_report(
        Path(observed["sources"][0]["report_path"]),
        Path(observed["sources"][1]["report_path"]),
    )
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("Q4 pass-category subspace report does not reproduce")
    return {
        "schema": "sch.vpd-nanbeige-q4-sae-pass-category-subspace-transfer-verification.v1",
        "status": "pass",
        "report_id": report_id,
        **observed["summary"],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-2048", type=Path)
    parser.add_argument("--report-8192", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_report(args.verify)
    else:
        if args.report_2048 is None or args.report_8192 is None or args.output is None:
            parser.error("--report-2048, --report-8192, and --output are required")
        result = build_report(args.report_2048, args.report_8192)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
