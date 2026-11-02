from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.activation_category_cluster_count_selection_v2 import (
    select_category_cluster_counts_v2,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    _project_effects,
    content_sha256,
    sha256_file,
    stable_hash,
    write_json,
)


SCHEMA = "sch.vpd-nanbeige-q4-region-category-bridge.v1"
REGIONS = ("prompt", "early", "middle", "late")
SITES = ("model.layers.11", "model.layers.21", "model.norm")
PROJECTION_SEEDS = (17, 29, 41, 53, 67, 79, 97, 109)


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("region signature has zero norm")
    return values / norms


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_source(source_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    records = _read_records(source_dir / "records.jsonl")
    if manifest.get("status") != "q4_reasoning_capture_complete":
        raise ValueError("unexpected Q4 activation manifest")
    if manifest.get("weights") != "Q4_K_M" or not manifest.get(
        "generated_reasoning", False
    ):
        raise ValueError("Q4 region bridge requires recorded generated reasoning")
    if len(records) != int(manifest.get("records", 0)) or len(records) < 12:
        raise ValueError("Q4 region record count differs or is underpowered")
    files = manifest.get("files", {})
    required = {"records.jsonl", *(str(row["archive"]) for row in records)}
    for name in sorted(required):
        if name not in files or sha256_file(source_dir / name) != str(files[name]):
            raise ValueError(f"Q4 region source artifact hash differs: {name}")
    for row in records:
        if row.get("split") not in {"train", "eval"} or set(row.get("regions", {})) != set(
            REGIONS
        ):
            raise ValueError("Q4 region labels or split are invalid")
    if sum(row["split"] == "train" for row in records) < 8 or sum(
        row["split"] == "eval" for row in records
    ) < 4:
        raise ValueError("Q4 region split is underpowered")
    return manifest, records


def _load_signatures(
    source_dir: Path, records: list[dict[str, Any]]
) -> dict[tuple[str, int], np.ndarray]:
    values: dict[tuple[str, int], list[np.ndarray]] = {
        (site, pass_index): [] for site in SITES for pass_index in (0, 1)
    }
    for record in records:
        archive = np.load(source_dir / str(record["archive"]), allow_pickle=False)
        positions = [int(value) for value in record["positions"]]
        region_indices: dict[str, list[int]] = {}
        for region in REGIONS:
            start, end = (int(value) for value in record["regions"][region])
            indices = [
                index
                for index, position in enumerate(positions)
                if start <= position < end
            ]
            if len(indices) != 8:
                raise ValueError("Q4 region does not contain exactly eight recorded rows")
            region_indices[region] = indices
        for site in SITES:
            for pass_index in (0, 1):
                array = np.asarray(
                    archive[f"{site}__pass_{pass_index}"], dtype=np.float64
                )
                if array.ndim != 2 or len(array) != len(positions) or not np.all(
                    np.isfinite(array)
                ):
                    raise ValueError("Q4 activation array is invalid")
                region_means = np.asarray(
                    [array[region_indices[region]].mean(axis=0) for region in REGIONS]
                )
                values[(site, pass_index)].append(
                    region_means - region_means.mean(axis=0, keepdims=True)
                )
    return {key: np.asarray(rows) for key, rows in values.items()}


def _project(
    signatures: np.ndarray, *, seed: int, dimension: int = 64
) -> np.ndarray:
    source_dimension = signatures.shape[-1]
    target_dimension = min(dimension, source_dimension)
    rng = np.random.default_rng(seed)
    projection = rng.normal(
        0.0,
        1.0 / np.sqrt(target_dimension),
        size=(source_dimension, target_dimension),
    )
    return _unit_rows(signatures @ projection)


def _classify(centroids: np.ndarray, examples: np.ndarray) -> tuple[float, list[float]]:
    centers = _unit_rows(centroids)
    rows = examples.reshape(-1, examples.shape[-1])
    labels = np.tile(np.arange(len(REGIONS)), len(examples))
    predictions = np.argmax(rows @ centers.T, axis=1)
    per_category = [
        float(np.mean(predictions[labels == category] == category))
        for category in range(len(REGIONS))
    ]
    return float(np.mean(predictions == labels)), per_category


def _combination_report(
    site: str,
    pass_index: int,
    raw: np.ndarray,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    train_indices = [index for index, row in enumerate(records) if row["split"] == "train"]
    eval_indices = [index for index, row in enumerate(records) if row["split"] == "eval"]
    seed = int(stable_hash(250829, site, str(pass_index))[:8], 16)
    projected = _project(raw, seed=seed)
    train = projected[train_indices]
    held_out = projected[eval_indices]
    centroids = train.mean(axis=0)
    accuracy, per_category = _classify(centroids, held_out)
    leave_one_out_accuracies = [
        _classify(np.delete(train, omitted, axis=0).mean(axis=0), held_out)[0]
        for omitted in range(len(train))
    ]

    # Each control independently either preserves or reverses the region order
    # for each training example. This is an exact 2^N block randomization, not
    # a claim to enumerate every possible four-label permutation.
    control_accuracies: list[float] = []
    reversed_train = train[:, ::-1, :]
    for mask in range(1 << len(train)):
        controlled = np.asarray(
            [
                reversed_train[index] if mask & (1 << index) else train[index]
                for index in range(len(train))
            ]
        )
        control_accuracy, _ = _classify(controlled.mean(axis=0), held_out)
        control_accuracies.append(control_accuracy)
    controls_at_or_above = sum(value >= accuracy - 1e-12 for value in control_accuracies)
    exact_p = controls_at_or_above / len(control_accuracies)

    component_ids = [
        f"{records[index]['id']}__{region}"
        for index in train_indices
        for region in REGIONS
    ]
    categories = [region for _ in train_indices for region in REGIONS]
    raw_train = raw[train_indices].reshape(len(train_indices) * len(REGIONS), -1)
    replicate_matrices = [
        _project_effects(raw_train, seed=value, projection_dim=64)
        for value in PROJECTION_SEEDS
    ]
    selection = select_category_cluster_counts_v2(
        replicate_matrices,
        component_ids,
        categories,
        seed=250829,
        replicate_kinds=["projection"] * len(PROJECTION_SEEDS),
        maximum_clusters_per_category=2,
        minimum_cluster_size=4,
    )
    admitted = (
        accuracy >= 0.75
        and min(per_category) >= 0.60
        and min(leave_one_out_accuracies) >= 0.70
        and exact_p <= 0.05
    )
    screening_allocation = {
        str(key): int(value) for key, value in selection["allocation"].items()
    }
    return {
        "site": site,
        "pass": pass_index,
        "region_category_admission": "pass" if admitted else "fail",
        "held_out_accuracy": accuracy,
        "held_out_accuracy_by_region": dict(zip(REGIONS, per_category, strict=True)),
        "minimum_leave_one_train_example_out_held_out_accuracy": min(
            leave_one_out_accuracies
        ),
        "exact_region_order_reversal_controls": len(control_accuracies),
        "controls_at_or_above_observed": controls_at_or_above,
        "exact_upper_tail_p_value": exact_p,
        "control_accuracy_range": [min(control_accuracies), max(control_accuracies)],
        "within_region_v2_selection": selection,
        "within_region_cluster_gate": (
            "pass_parent_category_admitted"
            if admitted
            else "fail_parent_category_not_admitted"
        ),
        "operative_within_region_allocation": (
            screening_allocation
            if admitted
            else {region: 1 for region in REGIONS}
        ),
    }


def build_report(source_dir: Path) -> dict[str, Any]:
    manifest, records = _load_source(source_dir)
    signatures = _load_signatures(source_dir, records)
    combinations = [
        _combination_report(site, pass_index, signatures[(site, pass_index)], records)
        for site in SITES
        for pass_index in (0, 1)
    ]
    train_families = sorted(
        {str(row["family"]) for row in records if row["split"] == "train"}
    )
    held_out_families = sorted(
        {str(row["family"]) for row in records if row["split"] == "eval"}
    )
    body = {
        "schema": SCHEMA,
        "status": "pass",
        "source": {
            "path": str(source_dir.resolve()),
            "manifest_sha256": sha256_file(source_dir / "manifest.json"),
            "records_sha256": sha256_file(source_dir / "records.jsonl"),
            "model_sha256": manifest["generation_contract"]["model_sha256"],
            "runtime_commit": manifest["generation_contract"]["runtime_commit"],
            "context": int(manifest["generation_contract"]["context"]),
            "max_new_tokens": int(
                manifest["generation_contract"]["max_new_tokens"]
            ),
            "generation_source_sha256": manifest["generation_source_sha256"],
            "weights": manifest["weights"],
            "kv_cache": manifest["kv_cache"],
            "runner_sha256": sha256_file(Path(__file__)),
        },
        "design": {
            "categories": list(REGIONS),
            "sites": list(SITES),
            "passes": [0, 1],
            "signature": "eight-row region mean centered within example across four regions",
            "classification_space": "fixed deterministic 64-dimensional projection with unit rows",
            "train_examples": sum(row["split"] == "train" for row in records),
            "held_out_examples": sum(row["split"] == "eval" for row in records),
            "train_families": train_families,
            "held_out_families": held_out_families,
            "overlapping_families": sorted(
                set(train_families).intersection(held_out_families)
            ),
            "random_control": "exact independent region-order reversal per training example",
            "admission_thresholds": {
                "minimum_held_out_accuracy": 0.75,
                "minimum_per_region_accuracy": 0.60,
                "minimum_leave_one_train_example_out_accuracy": 0.70,
                "maximum_exact_upper_tail_p_value": 0.05,
            },
        },
        "combinations": combinations,
        "summary": {
            "site_pass_combinations": len(combinations),
            "region_category_admissions": sum(
                row["region_category_admission"] == "pass" for row in combinations
            ),
            "screening_only_within_region_nontrivial_allocations": sum(
                any(
                    int(value) > 1
                    for value in row["within_region_v2_selection"]["allocation"].values()
                )
                for row in combinations
            ),
            "admissible_within_region_nontrivial_allocations": sum(
                row["within_region_cluster_gate"]
                == "pass_parent_category_admitted"
                and any(
                    int(value) > 1
                    for value in row["operative_within_region_allocation"].values()
                )
                for row in combinations
            ),
        },
        "scientific_boundary": {
            "new_model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "stored_empirical_quantized_model_activations_used": True,
            "bf16_parity_validated_by_source": bool(
                manifest["generation_contract"].get("bf16_parity_validated", False)
            ),
            "all_continuations_censored": all(
                bool(row.get("max_token_censored", False)) for row in records
            ),
            "region_labels_are_observable_position_windows": True,
            "region_order_reversal_is_a_limited_null_not_all_label_permutations": True,
            "within_region_cluster_membership_is_screening_only": True,
            "does_not_validate_bonsai_authorization_categories": True,
            "does_not_authorize_sae_training_or_model_release": True,
        },
    }
    return {**body, "report_id": content_sha256(body)}


def verify_report(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    report_id = str(body.pop("report_id", ""))
    if content_sha256(body) != report_id:
        raise ValueError("Q4 region-category bridge content address differs")
    expected = build_report(Path(observed["source"]["path"]))
    if expected != observed or observed.get("status") != "pass":
        raise ValueError("Q4 region-category bridge does not reproduce")
    return {
        "schema": "sch.vpd-nanbeige-q4-region-category-bridge-verification.v1",
        "status": "pass",
        "report_id": report_id,
        "combinations": int(observed["summary"]["site_pass_combinations"]),
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
