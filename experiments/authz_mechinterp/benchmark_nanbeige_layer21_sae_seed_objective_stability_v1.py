from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_feature_replica_matching_v2 import (
    _category_profiles,
    _label_permutation_p_value,
)
from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_feature_identity_v1 import (
    MAXIMUM_MONTE_CARLO_P_VALUE,
    MINIMUM_FEATURES_PER_PASS,
    PERMUTATION_SEED,
    PRIMARY_DIAGNOSTIC_THRESHOLD,
    _labels,
)
from experiments.authz_mechinterp.benchmark_nanbeige_q4_sae_pass_selectivity_v1 import (
    MINIMUM_HELD_OUT_EXPLAINED_VARIANCE,
    _assign_features,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.evaluate_sae_feature_replica_matching_v2 import (
    MINIMUM_PROFILE_COSINE,
    mutual_signed_decoder_matches,
)


SCHEMA = "sch.vpd-nanbeige-layer21-sae-seed-objective-stability.v1"
MINIMUM_BIDIRECTIONAL_CATEGORY_COVERAGE = 0.10
SEEDS = (42, 43, 44)
ARMS = ("raw_mse", "inverse_variance_mse")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_condition(
    root: Path,
    root_manifest: dict[str, Any],
    contract: dict[str, Any],
    seed: int,
    arm: str,
) -> dict[str, Any]:
    condition_id = f"{seed}_{arm}"
    condition_dir = root / condition_id
    child_manifest_path = condition_dir / "manifest.json"
    expected_manifest_hash = root_manifest["files"][f"{condition_id}/manifest.json"]
    if sha256_file(child_manifest_path) != expected_manifest_hash:
        raise ValueError(f"child manifest hash differs: {condition_id}")
    child = _load(child_manifest_path)
    if (
        child.get("status") != "sae_loss_condition_complete"
        or child.get("seed") != seed
        or child.get("arm") != arm
        or child.get("completed_steps") != contract["steps"]
    ):
        raise ValueError(f"child condition metadata differs: {condition_id}")

    dictionary_path = condition_dir / "dictionary.npz"
    if (
        sha256_file(dictionary_path) != child["dictionary_sha256"]
        or child["files"]["dictionary.npz"] != child["dictionary_sha256"]
        or root_manifest["files"][f"{condition_id}/dictionary.npz"]
        != child["dictionary_sha256"]
    ):
        raise ValueError(f"dictionary hash differs: {condition_id}")
    with np.load(dictionary_path, allow_pickle=False) as archive:
        decoder = np.asarray(archive["decoder"], dtype=np.float64)
    if decoder.shape != (6144, 3072) or not np.all(np.isfinite(decoder)):
        raise ValueError(f"decoder is invalid: {condition_id}")

    records_path = condition_dir / "records.json"
    if (
        sha256_file(records_path) != child["files"]["records.json"]
        or root_manifest["files"][f"{condition_id}/records.json"]
        != child["files"]["records.json"]
    ):
        raise ValueError(f"records hash differs: {condition_id}")
    records = _load(records_path)
    expected_train = set(contract["training_ids"])
    expected_eval = set(contract["evaluation_ids"])
    observed_train = {str(row["id"]) for row in records if row["split"] == "train"}
    observed_eval = {str(row["id"]) for row in records if row["split"] == "eval"}
    if observed_train != expected_train or observed_eval != expected_eval:
        raise ValueError(f"feature split differs: {condition_id}")

    train_mass = []
    for row in records:
        archive_name = str(row["archive"])
        archive_path = condition_dir / archive_name
        if (
            row["sha256"] != child["files"][archive_name]
            or root_manifest["files"][f"{condition_id}/{archive_name}"]
            != row["sha256"]
            or sha256_file(archive_path) != row["sha256"]
        ):
            raise ValueError(f"feature archive hash differs: {condition_id}/{archive_name}")
        if row["split"] != "train":
            continue
        with np.load(archive_path, allow_pickle=False) as archive:
            pass_0 = np.asarray(archive["pass_0"], dtype=np.float64)
            pass_1 = np.asarray(archive["pass_1"], dtype=np.float64)
            delta = np.asarray(archive["delta_z"], dtype=np.float64)
        if pass_0.shape != (8, 6144) or pass_1.shape != pass_0.shape:
            raise ValueError(f"feature shape differs: {condition_id}/{archive_name}")
        if not (
            np.all(np.isfinite(pass_0))
            and np.all(np.isfinite(pass_1))
            and np.all(np.isfinite(delta))
            and np.min(pass_0) >= 0.0
            and np.min(pass_1) >= 0.0
            and np.allclose(delta, pass_1 - pass_0, rtol=1e-6, atol=1e-6)
        ):
            raise ValueError(f"feature values differ: {condition_id}/{archive_name}")
        train_mass.append(
            np.stack(
                [
                    pass_0.sum(axis=0, dtype=np.float64),
                    pass_1.sum(axis=0, dtype=np.float64),
                ]
            )
        )
    train_mass_array = np.stack(train_mass)
    masks = _assign_features(train_mass_array)
    labels = _labels(masks)
    selective = np.flatnonzero(labels >= 0)
    minimum_eval_ev = min(
        float(child["quality"]["eval"][f"pass_{index}"]["explained_variance"])
        for index in (0, 1)
    )
    return {
        "id": condition_id,
        "seed": seed,
        "arm": arm,
        "manifest_sha256": expected_manifest_hash,
        "dictionary_sha256": child["dictionary_sha256"],
        "minimum_held_out_explained_variance": minimum_eval_ev,
        "quality_qualified": minimum_eval_ev >= MINIMUM_HELD_OUT_EXPLAINED_VARIANCE,
        "selective_counts": {
            f"pass_{index}": int(np.sum(labels == index)) for index in (0, 1)
        },
        "decoder": decoder[selective],
        "labels": labels[selective],
        "profiles": _category_profiles(train_mass_array)[selective],
    }


def _compare(left: dict[str, Any], right: dict[str, Any], *, pair_index: int) -> dict[str, Any]:
    match = mutual_signed_decoder_matches(
        left["decoder"],
        right["decoder"],
        minimum_cosine=PRIMARY_DIAGNOSTIC_THRESHOLD,
        block_rows=128,
    )
    matched_left = np.asarray(match["matched_left_indices"], dtype=np.int64)
    matched_right = np.asarray(match["matched_right_indices"], dtype=np.int64)
    left_labels = left["labels"][matched_left]
    right_labels = right["labels"][matched_right]
    same = left_labels == right_labels
    profile_cosine = np.sum(
        left["profiles"][matched_left] * right["profiles"][matched_right], axis=1
    )
    profile_qualified = profile_cosine >= MINIMUM_PROFILE_COSINE
    same_counts = {
        f"pass_{index}": int(np.sum(same & (left_labels == index)))
        for index in (0, 1)
    }
    profile_counts = {
        f"pass_{index}": int(
            np.sum(same & profile_qualified & (left_labels == index))
        )
        for index in (0, 1)
    }
    coverage = {
        f"pass_{index}": float(
            min(
                same_counts[f"pass_{index}"]
                / max(1, left["selective_counts"][f"pass_{index}"]),
                same_counts[f"pass_{index}"]
                / max(1, right["selective_counts"][f"pass_{index}"]),
            )
        )
        for index in (0, 1)
    }
    p_value = _label_permutation_p_value(
        left["labels"],
        right["labels"],
        matched_left,
        matched_right,
        int(np.sum(same)),
        seed=PERMUTATION_SEED + pair_index,
    )
    diagnostic = (
        min(same_counts.values()) >= MINIMUM_FEATURES_PER_PASS
        and min(coverage.values()) >= MINIMUM_BIDIRECTIONAL_CATEGORY_COVERAGE
        and profile_counts == same_counts
        and int(np.sum(~same)) == 0
        and p_value <= MAXIMUM_MONTE_CARLO_P_VALUE
    )
    admitted = diagnostic and left["quality_qualified"] and right["quality_qualified"]
    return {
        "left": left["id"],
        "right": right["id"],
        "comparison_class": (
            "cross_seed_same_objective"
            if left["arm"] == right["arm"]
            else "same_seed_cross_objective"
        ),
        "minimum_signed_decoder_cosine": PRIMARY_DIAGNOSTIC_THRESHOLD,
        "mutual_one_to_one_matches": match["mutual_matches"],
        "same_category_matches": same_counts,
        "profile_qualified_same_category_matches": profile_counts,
        "bidirectional_category_coverage": coverage,
        "opposite_category_matches": int(np.sum(~same)),
        "target_label_permutation_p_value": p_value,
        "diagnostic_stratum_stability_without_quality_gate": (
            "pass" if diagnostic else "fail"
        ),
        "quality_qualified_pair": left["quality_qualified"] and right["quality_qualified"],
        "operative_stratum_replication_admission": "pass" if admitted else "fail",
    }


def build_report(control_root: Path) -> dict[str, Any]:
    root = control_root.resolve()
    root_manifest_path = root / "manifest.json"
    contract_path = root / "contract.json"
    summary_path = root / "summary.json"
    root_manifest = _load(root_manifest_path)
    contract = _load(contract_path)
    summary = _load(summary_path)
    if root_manifest.get("status") != "sae_loss_balance_control_complete":
        raise ValueError("loss-balance control is not complete")
    if (
        sha256_file(contract_path) != root_manifest["contract_sha256"]
        or sha256_file(contract_path) != root_manifest["files"]["contract.json"]
        or sha256_file(summary_path) != root_manifest["files"]["summary.json"]
        or len(summary) != 6
    ):
        raise ValueError("loss-balance control root provenance differs")
    if (
        contract.get("site") != "model.layers.21"
        or tuple(contract.get("seeds", [])) != SEEDS
        or tuple(contract.get("arms", {})) != ARMS
        or not contract.get("exploratory_posthoc")
    ):
        raise ValueError("loss-balance control contract differs")

    conditions = {
        f"{seed}_{arm}": _load_condition(root, root_manifest, contract, seed, arm)
        for seed in SEEDS
        for arm in ARMS
    }
    pair_ids: list[tuple[str, str]] = []
    for arm in ARMS:
        pair_ids.extend(
            (f"{left}_{arm}", f"{right}_{arm}")
            for left, right in itertools.combinations(SEEDS, 2)
        )
    pair_ids.extend((f"{seed}_{ARMS[0]}", f"{seed}_{ARMS[1]}") for seed in SEEDS)
    pairs = [
        _compare(conditions[left], conditions[right], pair_index=index)
        for index, (left, right) in enumerate(pair_ids)
    ]
    public_conditions = [
        {key: value for key, value in condition.items() if key not in {"decoder", "labels", "profiles"}}
        for condition in conditions.values()
    ]
    cross_seed = [row for row in pairs if row["comparison_class"] == "cross_seed_same_objective"]
    cross_objective = [row for row in pairs if row["comparison_class"] == "same_seed_cross_objective"]
    body = {
        "schema": SCHEMA,
        "status": "pass",
        "source": {
            "path": root.as_posix(),
            "manifest_sha256": sha256_file(root_manifest_path),
            "contract_sha256": sha256_file(contract_path),
            "summary_sha256": sha256_file(summary_path),
            "site": "model.layers.21",
            "exploratory_posthoc": True,
        },
        "design": {
            "seeds": list(SEEDS),
            "objectives": list(ARMS),
            "cross_seed_same_objective_pairs": len(cross_seed),
            "same_seed_cross_objective_pairs": len(cross_objective),
            "matching_pool": "training_assigned_pass_selective_features_only",
            "minimum_signed_decoder_cosine": PRIMARY_DIAGNOSTIC_THRESHOLD,
            "minimum_profile_cosine": MINIMUM_PROFILE_COSINE,
            "minimum_features_per_pass": MINIMUM_FEATURES_PER_PASS,
            "minimum_bidirectional_category_coverage": MINIMUM_BIDIRECTIONAL_CATEGORY_COVERAGE,
            "minimum_held_out_explained_variance": MINIMUM_HELD_OUT_EXPLAINED_VARIANCE,
            "target_label_permutation_replicates": 4096,
            "all_thresholds_posthoc_for_existing_sources": True,
        },
        "conditions": public_conditions,
        "pairs": pairs,
        "summary": {
            "conditions": len(public_conditions),
            "quality_qualified_conditions": sum(row["quality_qualified"] for row in public_conditions),
            "cross_seed_pair_checks": len(cross_seed),
            "cross_seed_diagnostic_supports_without_quality_gate": sum(
                row["diagnostic_stratum_stability_without_quality_gate"] == "pass"
                for row in cross_seed
            ),
            "same_seed_cross_objective_pair_checks": len(cross_objective),
            "same_seed_cross_objective_diagnostic_supports_without_quality_gate": sum(
                row["diagnostic_stratum_stability_without_quality_gate"] == "pass"
                for row in cross_objective
            ),
            "operative_stratum_replication_admissions": sum(
                row["operative_stratum_replication_admission"] == "pass" for row in pairs
            ),
        },
        "scientific_boundary": {
            "layer21_only": True,
            "all_conditions_fail_reconstruction_quality": True,
            "diagnostic_matching_cannot_override_quality_gate": True,
            "not_a_quality_qualified_fresh_seed_replication": True,
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
        raise ValueError("layer21 seed-objective report content address differs")
    expected = build_report(Path(observed["source"]["path"]))
    if observed != expected or observed.get("status") != "pass":
        raise ValueError("layer21 seed-objective report does not reproduce")
    return {
        "schema": "sch.vpd-nanbeige-layer21-sae-seed-objective-stability-verification.v1",
        "status": "pass",
        "report_id": report_id,
        **observed["summary"],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_report(args.verify)
    else:
        if args.control_root is None or args.output is None:
            parser.error("--control-root and --output are required unless verifying")
        result = build_report(args.control_root)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
