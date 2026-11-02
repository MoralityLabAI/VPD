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


SCHEMA = "sch.vpd-nanbeige-q4-sae-pass-selectivity.v1"
SITES = ("model.layers.11", "model.layers.21", "model.norm")
TRAIN_SHARE_THRESHOLD = 0.90
MINIMUM_TRAIN_EXAMPLE_SUPPORT = 4
MINIMUM_FEATURES_PER_PASS = 16
MINIMUM_HELD_OUT_MASS_PRECISION = 0.75
MINIMUM_HELD_OUT_MASS_COVERAGE = 0.05
MINIMUM_HELD_OUT_EXPLAINED_VARIANCE = 0.20
MAXIMUM_EXACT_P_VALUE = 0.05


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_and_load(
    sae_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    manifest_path = sae_dir / "manifest.json"
    quality_path = sae_dir / "sae_quality.json"
    capture_manifest_path = sae_dir.parent / "q4_activations/manifest.json"
    manifest = _load(manifest_path)
    quality = _load(quality_path)
    if manifest.get("status") != "q4_reasoning_saes_complete":
        raise ValueError("Q4 SAE manifest is not complete")
    if sha256_file(capture_manifest_path) != manifest["capture_manifest_sha256"]:
        raise ValueError("Q4 capture manifest hash differs")
    if sha256_file(quality_path) != manifest["files"]["sae_quality.json"]:
        raise ValueError("Q4 SAE quality hash differs")
    if set(quality) != set(SITES):
        raise ValueError("Q4 SAE quality sites differ")
    expected_train = set(manifest["training_ids"])
    expected_eval = set(manifest["evaluation_ids"])
    if expected_train.intersection(expected_eval):
        raise ValueError("Q4 SAE train/eval IDs overlap")

    loaded: dict[str, dict[str, np.ndarray]] = {}
    for site in SITES:
        entry = quality[site]
        dictionary_path = sae_dir / entry["dictionary"]
        if sha256_file(dictionary_path) != entry["sha256"]:
            raise ValueError(f"Q4 SAE dictionary hash differs for {site}")
        differences = entry["differences"]
        ids = [str(row["id"]) for row in differences]
        if len(ids) != len(set(ids)) or set(ids) != expected_train | expected_eval:
            raise ValueError(f"Q4 SAE feature IDs differ for {site}")
        observed_train = {str(row["id"]) for row in differences if row["split"] == "train"}
        observed_eval = {str(row["id"]) for row in differences if row["split"] == "eval"}
        if observed_train != expected_train or observed_eval != expected_eval:
            raise ValueError(f"Q4 SAE feature split differs for {site}")
        train_values = []
        eval_values = []
        for row in differences:
            archive_path = sae_dir / row["archive"]
            expected_hash = manifest["files"][row["archive"]]
            if row["sha256"] != expected_hash or sha256_file(archive_path) != expected_hash:
                raise ValueError(f"Q4 SAE feature archive hash differs: {row['archive']}")
            with np.load(archive_path, allow_pickle=False) as archive:
                pass_0 = np.asarray(archive["pass_0"], dtype=np.float64)
                pass_1 = np.asarray(archive["pass_1"], dtype=np.float64)
                delta = np.asarray(archive["delta_z"], dtype=np.float64)
            if pass_0.shape != (32, 6144) or pass_1.shape != pass_0.shape:
                raise ValueError(f"unexpected Q4 SAE feature shape: {row['archive']}")
            if not (
                np.isfinite(pass_0).all()
                and np.isfinite(pass_1).all()
                and np.isfinite(delta).all()
            ):
                raise ValueError(f"nonfinite Q4 SAE features: {row['archive']}")
            if np.min(pass_0) < 0.0 or np.min(pass_1) < 0.0:
                raise ValueError(f"negative top-k Q4 SAE feature: {row['archive']}")
            if not np.allclose(delta, pass_1 - pass_0, rtol=1e-6, atol=1e-6):
                raise ValueError(f"Q4 SAE feature delta differs: {row['archive']}")
            mass = np.stack(
                [pass_0.sum(axis=0, dtype=np.float64), pass_1.sum(axis=0, dtype=np.float64)]
            )
            (train_values if row["split"] == "train" else eval_values).append(mass)
        loaded[site] = {
            "train": np.stack(train_values),
            "eval": np.stack(eval_values),
        }
    return manifest, quality, loaded


def _assign_features(train_mass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aggregate = train_mass.sum(axis=0, dtype=np.float64)
    total = aggregate[0] + aggregate[1]
    support = (train_mass > 0.0).sum(axis=0)
    share_0 = np.divide(
        aggregate[0], total, out=np.zeros_like(total), where=total > 0.0
    )
    share_1 = np.divide(
        aggregate[1], total, out=np.zeros_like(total), where=total > 0.0
    )
    assigned_0 = (share_0 >= TRAIN_SHARE_THRESHOLD) & (
        support[0] >= MINIMUM_TRAIN_EXAMPLE_SUPPORT
    )
    assigned_1 = (share_1 >= TRAIN_SHARE_THRESHOLD) & (
        support[1] >= MINIMUM_TRAIN_EXAMPLE_SUPPORT
    )
    return assigned_0, assigned_1


def _score_assignment(
    held_out_mass: np.ndarray, assigned_0: np.ndarray, assigned_1: np.ndarray
) -> tuple[float, dict[str, Any]]:
    aggregate = held_out_mass.sum(axis=0, dtype=np.float64)
    total_mass = float(aggregate.sum(dtype=np.float64))
    signed_mass = float(
        (aggregate[0, assigned_0] - aggregate[1, assigned_0]).sum(dtype=np.float64)
        + (aggregate[1, assigned_1] - aggregate[0, assigned_1]).sum(dtype=np.float64)
    )
    result: dict[str, Any] = {}
    for pass_index, assigned in ((0, assigned_0), (1, assigned_1)):
        target = float(aggregate[pass_index, assigned].sum(dtype=np.float64))
        other = float(aggregate[1 - pass_index, assigned].sum(dtype=np.float64))
        assigned_total = target + other
        result[f"pass_{pass_index}"] = {
            "assigned_features": int(assigned.sum()),
            "held_out_mass_precision": target / assigned_total if assigned_total else 0.0,
            "held_out_mass_coverage": target
            / float(aggregate[pass_index].sum(dtype=np.float64)),
            "held_out_directional_feature_fraction": float(
                np.mean(aggregate[pass_index, assigned] > aggregate[1 - pass_index, assigned])
            )
            if assigned.any()
            else 0.0,
        }
    return signed_mass / total_mass if total_mass else 0.0, result


def _exact_controls(
    train_mass: np.ndarray, held_out_mass: np.ndarray
) -> tuple[float, list[float], dict[str, Any]]:
    assigned_0, assigned_1 = _assign_features(train_mass)
    observed_score, details = _score_assignment(held_out_mass, assigned_0, assigned_1)
    controls = []
    example_count = train_mass.shape[0]
    for mask in range(1 << example_count):
        swapped = train_mass.copy()
        for index in range(example_count):
            if (mask >> index) & 1:
                swapped[index] = swapped[index, ::-1]
        control_0, control_1 = _assign_features(swapped)
        score, _ = _score_assignment(held_out_mass, control_0, control_1)
        controls.append(score)
    upper_tail = sum(value >= observed_score - 1e-15 for value in controls) / len(controls)
    return observed_score, controls, {**details, "exact_upper_tail_p_value": upper_tail}


def build_report(sae_dir: Path) -> dict[str, Any]:
    sae_dir = sae_dir.resolve()
    manifest, quality, loaded = _verify_and_load(sae_dir)
    sites = []
    for site in SITES:
        observed_score, controls, detail = _exact_controls(
            loaded[site]["train"], loaded[site]["eval"]
        )
        held_out_ev = {
            pass_name: float(values["explained_variance"])
            for pass_name, values in quality[site]["quality"]["eval"].items()
        }
        pass_details = [detail["pass_0"], detail["pass_1"]]
        admission = (
            all(row["assigned_features"] >= MINIMUM_FEATURES_PER_PASS for row in pass_details)
            and all(
                row["held_out_mass_precision"] >= MINIMUM_HELD_OUT_MASS_PRECISION
                for row in pass_details
            )
            and all(
                row["held_out_mass_coverage"] >= MINIMUM_HELD_OUT_MASS_COVERAGE
                for row in pass_details
            )
            and min(held_out_ev.values()) >= MINIMUM_HELD_OUT_EXPLAINED_VARIANCE
            and detail["exact_upper_tail_p_value"] <= MAXIMUM_EXACT_P_VALUE
        )
        sites.append(
            {
                "site": site,
                "held_out_explained_variance": held_out_ev,
                "held_out_signed_selective_mass_score": observed_score,
                "exact_training_pair_flip_controls": len(controls),
                "control_score_range": [float(min(controls)), float(max(controls))],
                **detail,
                "monolithic_sae_internal_pass_specialization_admission": (
                    "pass" if admission else "fail"
                ),
            }
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
            "training_examples": len(manifest["training_ids"]),
            "held_out_examples": len(manifest["evaluation_ids"]),
            "training_steps": int(manifest["steps"]),
            "runner_sha256": sha256_file(Path(__file__)),
        },
        "design": {
            "feature_assignment_uses_training_split_only": True,
            "feature_mass_share_threshold": TRAIN_SHARE_THRESHOLD,
            "minimum_train_example_support": MINIMUM_TRAIN_EXAMPLE_SUPPORT,
            "minimum_features_per_pass": MINIMUM_FEATURES_PER_PASS,
            "minimum_held_out_mass_precision": MINIMUM_HELD_OUT_MASS_PRECISION,
            "minimum_held_out_mass_coverage": MINIMUM_HELD_OUT_MASS_COVERAGE,
            "minimum_held_out_explained_variance": MINIMUM_HELD_OUT_EXPLAINED_VARIANCE,
            "maximum_exact_upper_tail_p_value": MAXIMUM_EXACT_P_VALUE,
            "random_control": "exact independent pass-label flip per training example",
            "held_out_signed_selective_mass_score": (
                "assigned target-pass mass minus opposite-pass mass divided by all "
                "held-out feature mass"
            ),
        },
        "sites": sites,
        "summary": {
            "sites": len(sites),
            "internal_pass_specialization_admissions": sum(
                row["monolithic_sae_internal_pass_specialization_admission"] == "pass"
                for row in sites
            ),
        },
        "scientific_boundary": {
            "stored_sparse_feature_archives_used": True,
            "sae_weights_loaded": False,
            "model_weights_loaded": False,
            "new_model_inference_performed": False,
            "new_sae_training_performed": False,
            "feature_assignment_is_not_semantic_interpretation": True,
            "source_q4_weights_are_quantized": True,
            "source_bf16_parity_validated": False,
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
        raise ValueError("Q4 SAE pass-selectivity report content address differs")
    expected = build_report(Path(observed["source"]["path"]))
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("Q4 SAE pass-selectivity report does not reproduce")
    return {
        "schema": "sch.vpd-nanbeige-q4-sae-pass-selectivity-verification.v1",
        "status": "pass",
        "report_id": report_id,
        "sites": int(observed["summary"]["sites"]),
        "admissions": int(observed["summary"]["internal_pass_specialization_admissions"]),
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
