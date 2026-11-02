from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_selectivity_v1 import (
    MAXIMUM_EXACT_P_VALUE,
    MINIMUM_FEATURES_PER_PASS,
    MINIMUM_HELD_OUT_MASS_COVERAGE,
    MINIMUM_HELD_OUT_MASS_PRECISION,
    SITES,
    _load,
    _score_assignment,
    _verify_and_load,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-nanbeige-q4-sae-pass-selectivity-robustness.v1"
TRAIN_SHARE_THRESHOLDS = (0.80, 0.85, 0.90, 0.95)
TRAIN_SUPPORT_THRESHOLDS = (2, 4, 6)
HELD_OUT_EV_THRESHOLDS = (0.0, 0.10, 0.20, 0.25, 0.30)


def _assign(
    train_mass: np.ndarray, *, share_threshold: float, support_threshold: int
) -> tuple[np.ndarray, np.ndarray]:
    aggregate = train_mass.sum(axis=0, dtype=np.float64)
    total = aggregate[0] + aggregate[1]
    support = (train_mass > 0.0).sum(axis=0)
    shares = np.divide(
        aggregate,
        total[None, :],
        out=np.zeros_like(aggregate),
        where=total[None, :] > 0.0,
    )
    return (
        (shares[0] >= share_threshold) & (support[0] >= support_threshold),
        (shares[1] >= share_threshold) & (support[1] >= support_threshold),
    )


def _evaluate_config(
    train_mass: np.ndarray,
    held_out_mass: np.ndarray,
    *,
    share_threshold: float,
    support_threshold: int,
) -> dict[str, Any]:
    assigned_0, assigned_1 = _assign(
        train_mass,
        share_threshold=share_threshold,
        support_threshold=support_threshold,
    )
    observed_score, details = _score_assignment(held_out_mass, assigned_0, assigned_1)
    controls = []
    for mask in range(1 << train_mass.shape[0]):
        swapped = train_mass.copy()
        for index in range(train_mass.shape[0]):
            if (mask >> index) & 1:
                swapped[index] = swapped[index, ::-1]
        control_0, control_1 = _assign(
            swapped,
            share_threshold=share_threshold,
            support_threshold=support_threshold,
        )
        score, _ = _score_assignment(held_out_mass, control_0, control_1)
        controls.append(score)
    p_value = sum(value >= observed_score - 1e-15 for value in controls) / len(controls)
    pass_details = (details["pass_0"], details["pass_1"])
    selectivity_pass = (
        all(row["assigned_features"] >= MINIMUM_FEATURES_PER_PASS for row in pass_details)
        and all(
            row["held_out_mass_precision"] >= MINIMUM_HELD_OUT_MASS_PRECISION
            for row in pass_details
        )
        and all(
            row["held_out_mass_coverage"] >= MINIMUM_HELD_OUT_MASS_COVERAGE
            for row in pass_details
        )
        and p_value <= MAXIMUM_EXACT_P_VALUE
    )
    return {
        "train_share_threshold": share_threshold,
        "minimum_train_example_support": support_threshold,
        "held_out_signed_selective_mass_score": observed_score,
        "exact_upper_tail_p_value": p_value,
        "control_score_range": [float(min(controls)), float(max(controls))],
        **details,
        "selectivity_gate": "pass" if selectivity_pass else "fail",
    }


def build_report(sae_dir: Path) -> dict[str, Any]:
    sae_dir = sae_dir.resolve()
    manifest, quality, loaded = _verify_and_load(sae_dir)
    sites = []
    for site in SITES:
        configs = [
            _evaluate_config(
                loaded[site]["train"],
                loaded[site]["eval"],
                share_threshold=share_threshold,
                support_threshold=support_threshold,
            )
            for share_threshold in TRAIN_SHARE_THRESHOLDS
            for support_threshold in TRAIN_SUPPORT_THRESHOLDS
        ]
        held_out_ev = {
            pass_name: float(values["explained_variance"])
            for pass_name, values in quality[site]["quality"]["eval"].items()
        }
        minimum_ev = min(held_out_ev.values())
        sites.append(
            {
                "site": site,
                "held_out_explained_variance": held_out_ev,
                "minimum_held_out_explained_variance": minimum_ev,
                "selectivity_configurations": configs,
                "selectivity_configurations_passing": sum(
                    row["selectivity_gate"] == "pass" for row in configs
                ),
                "quality_threshold_sensitivity": [
                    {
                        "minimum_held_out_explained_variance_threshold": threshold,
                        "selectivity_configurations_admitted": sum(
                            row["selectivity_gate"] == "pass" for row in configs
                        )
                        if minimum_ev >= threshold
                        else 0,
                    }
                    for threshold in HELD_OUT_EV_THRESHOLDS
                ],
            }
        )
    all_configs_per_site = len(TRAIN_SHARE_THRESHOLDS) * len(
        TRAIN_SUPPORT_THRESHOLDS
    )
    body = {
        "schema": SCHEMA,
        "status": "pass",
        "source": {
            "path": sae_dir.as_posix(),
            "manifest_sha256": sha256_file(sae_dir / "manifest.json"),
            "sae_quality_sha256": sha256_file(sae_dir / "sae_quality.json"),
            "capture_manifest_sha256": manifest["capture_manifest_sha256"],
            "weights": manifest["weights"],
            "runner_sha256": sha256_file(Path(__file__)),
        },
        "design": {
            "train_share_thresholds": list(TRAIN_SHARE_THRESHOLDS),
            "minimum_train_example_support_thresholds": list(
                TRAIN_SUPPORT_THRESHOLDS
            ),
            "minimum_held_out_explained_variance_thresholds": list(
                HELD_OUT_EV_THRESHOLDS
            ),
            "selectivity_configurations_per_site": all_configs_per_site,
            "minimum_features_per_pass": MINIMUM_FEATURES_PER_PASS,
            "minimum_held_out_mass_precision": MINIMUM_HELD_OUT_MASS_PRECISION,
            "minimum_held_out_mass_coverage": MINIMUM_HELD_OUT_MASS_COVERAGE,
            "maximum_exact_upper_tail_p_value": MAXIMUM_EXACT_P_VALUE,
            "all_configurations_reported_without_selection": True,
        },
        "sites": sites,
        "summary": {
            "sites": len(sites),
            "selectivity_configurations_per_site": all_configs_per_site,
            "sites_passing_all_selectivity_configurations": sum(
                row["selectivity_configurations_passing"] == all_configs_per_site
                for row in sites
            ),
            "quality_threshold_admitted_sites": {
                str(threshold): sum(
                    row["minimum_held_out_explained_variance"] >= threshold
                    and row["selectivity_configurations_passing"] == all_configs_per_site
                    for row in sites
                )
                for threshold in HELD_OUT_EV_THRESHOLDS
            },
        },
        "scientific_boundary": {
            "stored_sparse_feature_archives_used": True,
            "sae_weights_loaded": False,
            "model_weights_loaded": False,
            "new_model_inference_performed": False,
            "new_sae_training_performed": False,
            "sensitivity_grid_is_not_threshold_optimization": True,
            "feature_assignment_is_not_semantic_interpretation": True,
            "does_not_authorize_separate_dictionary_training": True,
            "execution_authorized": False,
            "paid_compute_authorized": False,
        },
    }
    return {**body, "report_id": content_sha256(body)}


def verify_report(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    report_id = str(body.pop("report_id", ""))
    if content_sha256(body) != report_id:
        raise ValueError("SAE pass-selectivity robustness address differs")
    expected = build_report(Path(observed["source"]["path"]))
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("SAE pass-selectivity robustness report does not reproduce")
    return {
        "schema": "sch.vpd-nanbeige-q4-sae-pass-selectivity-robustness-verification.v1",
        "status": "pass",
        "report_id": report_id,
        **observed["summary"],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_report(args.verify)
    else:
        if args.sae_dir is None or args.output is None:
            parser.error("--sae-dir and --output are required unless verifying")
        result = build_report(args.sae_dir)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
