from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_selectivity_v1 import (
    MINIMUM_FEATURES_PER_PASS,
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


SCHEMA = "sch.vpd-nanbeige-q4-sae-pass-feature-identity.v1"
COSINE_THRESHOLDS = (0.50, 0.70, 0.80, 0.90)
PRIMARY_DIAGNOSTIC_THRESHOLD = 0.80
PERMUTATION_REPLICATES = 4096
PERMUTATION_SEED = 260911
MAXIMUM_MONTE_CARLO_P_VALUE = 0.05


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _labels(masks: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    if np.any(masks[0] & masks[1]):
        raise ValueError("a feature is assigned to both passes")
    return np.where(masks[0], 0, np.where(masks[1], 1, -1)).astype(np.int8)


def _normalized_decoder(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        decoder = np.asarray(archive["decoder"], dtype=np.float64)
    if decoder.shape != (6144, 3072) or not np.all(np.isfinite(decoder)):
        raise ValueError("SAE decoder is invalid")
    norm = np.linalg.norm(decoder, axis=1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("SAE decoder contains a zero direction")
    return decoder / norm


def _permutation_p_values(
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    cosine: np.ndarray,
    observed: dict[float, int],
    *,
    seed: int,
) -> dict[float, float]:
    generator = np.random.default_rng(seed)
    exceed = {threshold: 0 for threshold in COSINE_THRESHOLDS}
    source_assigned = source_labels >= 0
    threshold_masks = {
        threshold: source_assigned & (cosine >= threshold)
        for threshold in COSINE_THRESHOLDS
    }
    for _ in range(PERMUTATION_REPLICATES):
        permuted = generator.permutation(target_labels)
        for threshold, mask in threshold_masks.items():
            score = int(np.sum(permuted[mask] == source_labels[mask]))
            exceed[threshold] += score >= observed[threshold]
    return {
        threshold: (exceed[threshold] + 1) / (PERMUTATION_REPLICATES + 1)
        for threshold in COSINE_THRESHOLDS
    }


def build_report(report_2048_path: Path, report_8192_path: Path) -> dict[str, Any]:
    report_paths = (report_2048_path.resolve(), report_8192_path.resolve())
    reports = [_load(path) for path in report_paths]
    for path in report_paths:
        if verify_selectivity_report(path).get("status") != "pass":
            raise ValueError("source pass-selectivity report did not verify")
    sae_dirs = [Path(report["source"]["path"]) for report in reports]
    loaded_sources = [_verify_and_load(path) for path in sae_dirs]
    manifests = [value[0] for value in loaded_sources]
    qualities = [value[1] for value in loaded_sources]
    features = [value[2] for value in loaded_sources]
    if manifests[0]["seed"] != manifests[1]["seed"]:
        raise ValueError("SAE seeds differ; same-index comparison is not identified")
    if manifests[0]["training_ids"] != manifests[1]["training_ids"]:
        raise ValueError("SAE training IDs differ; current controlled comparison requires equality")
    if manifests[0]["evaluation_ids"] != manifests[1]["evaluation_ids"]:
        raise ValueError("SAE evaluation IDs differ")

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
        labels = [_labels(value) for value in masks]
        decoders = [
            _normalized_decoder(sae_dir / quality[site]["dictionary"])
            for sae_dir, quality in zip(sae_dirs, qualities)
        ]
        same_index_cosine = np.sum(decoders[0] * decoders[1], axis=1)
        threshold_rows = []
        observed_total: dict[float, int] = {}
        for threshold in COSINE_THRESHOLDS:
            aligned = same_index_cosine >= threshold
            stable_by_pass = {
                f"pass_{pass_index}": int(
                    np.sum(
                        (labels[0] == pass_index)
                        & (labels[1] == pass_index)
                        & aligned
                    )
                )
                for pass_index in (0, 1)
            }
            opposite = int(
                np.sum(
                    (labels[0] >= 0)
                    & (labels[1] >= 0)
                    & (labels[0] != labels[1])
                    & aligned
                )
            )
            observed_total[threshold] = sum(stable_by_pass.values())
            threshold_rows.append(
                {
                    "minimum_same_index_decoder_cosine": threshold,
                    "stable_same_pass_features": stable_by_pass,
                    "opposite_pass_features": opposite,
                }
            )
        p_values = _permutation_p_values(
            labels[0],
            labels[1],
            same_index_cosine,
            observed_total,
            seed=PERMUTATION_SEED + site_index,
        )
        for row in threshold_rows:
            row["target_label_permutation_p_value"] = p_values[
                row["minimum_same_index_decoder_cosine"]
            ]
        primary = next(
            row
            for row in threshold_rows
            if row["minimum_same_index_decoder_cosine"]
            == PRIMARY_DIAGNOSTIC_THRESHOLD
        )
        quality_qualified = all(value[site] == "pass" for value in quality_admissions)
        provisional = (
            quality_qualified
            and min(primary["stable_same_pass_features"].values())
            >= MINIMUM_FEATURES_PER_PASS
            and primary["opposite_pass_features"] == 0
            and primary["target_label_permutation_p_value"]
            <= MAXIMUM_MONTE_CARLO_P_VALUE
        )
        sites.append(
            {
                "site": site,
                "source_assigned_features": {
                    "2048": {
                        f"pass_{index}": int(masks[0][index].sum())
                        for index in (0, 1)
                    },
                    "8192": {
                        f"pass_{index}": int(masks[1][index].sum())
                        for index in (0, 1)
                    },
                },
                "same_index_decoder_cosine_median_all_features": float(
                    np.median(same_index_cosine)
                ),
                "threshold_sweep": threshold_rows,
                "quality_qualified_in_both_archives": quality_qualified,
                "provisional_same_seed_feature_identity_support": (
                    "pass" if provisional else "fail"
                ),
                "operative_cross_budget_feature_identity_admission": "fail",
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
        "design": {
            "comparison": "same_index_decoder_direction_and_training_only_pass_assignment",
            "same_seed": int(manifests[0]["seed"]),
            "same_training_ids": True,
            "same_evaluation_ids": True,
            "cosine_threshold_sweep": list(COSINE_THRESHOLDS),
            "primary_diagnostic_cosine_threshold": PRIMARY_DIAGNOSTIC_THRESHOLD,
            "minimum_stable_features_per_pass": MINIMUM_FEATURES_PER_PASS,
            "target_label_permutation_replicates": PERMUTATION_REPLICATES,
            "target_label_permutation_seed": PERMUTATION_SEED,
            "permutation_preserves_target_pass_assignment_counts": True,
            "maximum_monte_carlo_p_value": MAXIMUM_MONTE_CARLO_P_VALUE,
            "threshold_and_same_seed_analysis_are_posthoc": True,
        },
        "sites": sites,
        "summary": {
            "site_checks": len(sites),
            "quality_qualified_sites": [
                row["site"] for row in sites if row["quality_qualified_in_both_archives"]
            ],
            "provisional_same_seed_feature_identity_supports": sum(
                row["provisional_same_seed_feature_identity_support"] == "pass"
                for row in sites
            ),
            "provisional_same_seed_feature_identity_sites": [
                row["site"]
                for row in sites
                if row["provisional_same_seed_feature_identity_support"] == "pass"
            ],
            "operative_cross_budget_feature_identity_admissions": 0,
        },
        "scientific_boundary": {
            "same_seed_and_training_ids_limit_independence": True,
            "cosine_threshold_sweep_is_posthoc": True,
            "decoder_alignment_is_not_feature_semantics": True,
            "quality_gate_applied_before_identity_support": True,
            "fresh_seed_and_family_disjoint_replication_required": True,
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
        raise ValueError("cross-budget feature-identity report content address differs")
    expected = build_report(
        Path(observed["sources"][0]["report_path"]),
        Path(observed["sources"][1]["report_path"]),
    )
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("cross-budget feature-identity report does not reproduce")
    return {
        "schema": "sch.vpd-nanbeige-q4-sae-pass-feature-identity-verification.v1",
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
            parser.error("both source reports and --output are required unless verifying")
        result = build_report(args.report_2048, args.report_8192)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
