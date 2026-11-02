from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    verify_bundle,
    write_json,
)
from experiments.authz_mechinterp.merge_vpd_sae_results import (
    load_registry,
    require_digest,
    sign_result,
    verify_signature,
)
from experiments.authz_mechinterp.effect_uncertainty import (
    CONDITION_ROBUSTNESS_ARRAYS,
    TEMPLATE_ROBUSTNESS_ARRAYS,
    UNCERTAINTY_ARRAYS,
    condition_robustness_contract,
    countsketch,
    template_robustness_contract,
    uncertainty_contract,
    validate_condition_robustness_arrays,
    validate_template_robustness_arrays,
    validate_uncertainty_arrays,
)


def array_bytes(array: np.ndarray) -> bytes:
    handle = io.BytesIO()
    np.lib.format.write_array(handle, array, allow_pickle=False)
    return handle.getvalue()


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0
            archive.writestr(info, array_bytes(np.asarray(arrays[name])))


def load_effect_archive(path: Path) -> tuple[list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = {"component_ids", "delta_h"} - set(archive.files)
        if missing:
            raise ValueError(f"effect shard is missing arrays: {sorted(missing)}")
        component_ids = [str(value) for value in archive["component_ids"].tolist()]
        delta_h = np.asarray(archive["delta_h"])
    if delta_h.ndim != 2 or delta_h.shape[0] != len(component_ids):
        raise ValueError("effect shard IDs and delta_h rows do not align")
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("effect shard contains duplicate component IDs")
    if delta_h.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("effect shard delta_h must be float32 or float64")
    if not np.all(np.isfinite(delta_h)):
        raise ValueError("effect shard contains non-finite values")
    return component_ids, delta_h.astype(np.float32)


def load_activation_category_effects(
    path: Path,
) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    with np.load(path, allow_pickle=False) as archive:
        names = {
            "category_names",
            "category_episode_counts",
            "delta_h_by_category",
        }
        present = names & set(archive.files)
        if not present:
            return None
        if present != names:
            raise ValueError("effect shard has an incomplete activation-category payload")
        category_names = [str(value) for value in archive["category_names"].tolist()]
        counts = np.asarray(archive["category_episode_counts"])
        category_delta = np.asarray(archive["delta_h_by_category"])
        component_count = int(np.asarray(archive["component_ids"]).shape[0])
        hidden_size = int(np.asarray(archive["delta_h"]).shape[1])
    if category_names != sorted(category_names) or len(category_names) != len(
        set(category_names)
    ):
        raise ValueError("activation category names must be sorted and unique")
    if counts.ndim != 1 or counts.shape[0] != len(category_names):
        raise ValueError("activation category counts do not align with names")
    if counts.dtype.kind not in "iu" or np.any(counts <= 0):
        raise ValueError("activation category counts must be positive integers")
    expected_shape = (component_count, len(category_names), hidden_size)
    if category_delta.shape != expected_shape:
        raise ValueError("category delta_h shape does not match components/categories/hidden")
    if category_delta.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("category delta_h must be float32 or float64")
    if not np.all(np.isfinite(category_delta)):
        raise ValueError("category delta_h contains non-finite values")
    return category_names, counts.astype(np.int64), category_delta.astype(np.float32)


def load_category_effect_uncertainty(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, int, int, int] | None:
    with np.load(path, allow_pickle=False) as archive:
        present = UNCERTAINTY_ARRAYS & set(archive.files)
        if not present:
            return None
        if present != UNCERTAINTY_ARRAYS:
            raise ValueError("effect shard has an incomplete category uncertainty payload")
        component_count = int(np.asarray(archive["component_ids"]).shape[0])
        category_count = int(np.asarray(archive["category_names"]).shape[0])
        block_count = int(np.asarray(archive["uncertainty_block_count"]).item())
        sketch_dimension = int(
            np.asarray(archive["uncertainty_countsketch_dimension"]).item()
        )
        seed = int(np.asarray(archive["uncertainty_countsketch_seed"]).item())
        counts, sums = validate_uncertainty_arrays(
            block_counts=archive["category_uncertainty_block_counts"],
            block_sums=archive["delta_h_countsketch_block_sums"],
            component_count=component_count,
            category_count=category_count,
            block_count=block_count,
            sketch_dimension=sketch_dimension,
        )
    return counts, sums, block_count, sketch_dimension, seed


def load_category_condition_robustness(
    path: Path,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray] | None:
    with np.load(path, allow_pickle=False) as archive:
        present = CONDITION_ROBUSTNESS_ARRAYS & set(archive.files)
        if not present:
            return None
        if present != CONDITION_ROBUSTNESS_ARRAYS:
            raise ValueError("effect shard has incomplete condition robustness arrays")
        component_count = int(np.asarray(archive["component_ids"]).shape[0])
        category_names = [str(value) for value in archive["category_names"].tolist()]
        sketch_dimension = int(
            np.asarray(archive["uncertainty_countsketch_dimension"]).item()
        )
        return validate_condition_robustness_arrays(
            condition_names=archive["condition_names"],
            condition_category_names=archive["condition_category_names"],
            condition_counts=archive["condition_episode_counts"],
            condition_sums=archive["delta_h_countsketch_condition_sums"],
            component_count=component_count,
            category_names=category_names,
            sketch_dimension=sketch_dimension,
        )


def load_category_template_robustness(
    path: Path,
) -> tuple[list[str], list[str], list[str], np.ndarray, np.ndarray] | None:
    with np.load(path, allow_pickle=False) as archive:
        present = TEMPLATE_ROBUSTNESS_ARRAYS & set(archive.files)
        if not present:
            return None
        if present != TEMPLATE_ROBUSTNESS_ARRAYS:
            raise ValueError("effect shard has incomplete template robustness arrays")
        component_count = int(np.asarray(archive["component_ids"]).shape[0])
        category_names = [str(value) for value in archive["category_names"].tolist()]
        sketch_dimension = int(
            np.asarray(archive["uncertainty_countsketch_dimension"]).item()
        )
        return validate_template_robustness_arrays(
            cell_names=archive["template_cell_names"],
            cell_category_names=archive["template_cell_category_names"],
            cell_group_names=archive["template_cell_group_names"],
            cell_counts=archive["template_cell_episode_counts"],
            cell_sums=archive["delta_h_countsketch_template_cell_sums"],
            component_count=component_count,
            category_names=category_names,
            sketch_dimension=sketch_dimension,
        )


def load_packets(bundle_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    verify_bundle(bundle_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    packets: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for entry in manifest["artifacts"]["packets"]:
        packet = json.loads((bundle_dir / entry["packet_path"]).read_text(encoding="utf-8"))
        packet_id = str(packet["packet_id"])
        packets[packet_id] = packet
        hashes[packet_id] = str(entry["packet_sha256"])
    return packets, hashes


def effect_identity(receipt: dict[str, Any]) -> dict[str, Any]:
    inputs = receipt["inputs"]
    identity = {
        "model_id": str(inputs["model_id"]),
        "revision": str(inputs["revision"]),
        "adapter_sha256": require_digest(inputs["adapter_sha256"], "adapter_sha256"),
        "dataset_sha256": require_digest(inputs["dataset_sha256"], "dataset_sha256"),
        "runner_sha256": require_digest(inputs["runner_sha256"], "runner_sha256"),
        "effect_space_id": str(inputs["effect_space_id"]),
        "phase": str(inputs["phase"]),
        "estimator": str(inputs["estimator"]),
        "perturbation_epsilon": float(inputs["perturbation_epsilon"]),
        "selection_seed": int(inputs["selection_seed"]),
        "episode_order_sha256": require_digest(
            inputs["episode_order_sha256"], "episode_order_sha256"
        ),
        "episodes": int(inputs["episodes"]),
        "generated_tokens": int(inputs["generated_tokens"]),
    }
    if receipt.get("schema") in {
        "sch.vpd-effect-shard-receipt.v2",
        "sch.vpd-effect-shard-receipt.v3",
        "sch.vpd-effect-shard-receipt.v4",
        "sch.vpd-effect-shard-receipt.v5",
    }:
        categories = [str(value) for value in inputs["activation_categories"]]
        counts = {
            str(key): int(value)
            for key, value in inputs["activation_category_episode_counts"].items()
        }
        episode_hashes = {
            str(key): require_digest(value, f"activation category {key} episode hash")
            for key, value in inputs[
                "activation_category_episode_ids_sha256"
            ].items()
        }
        if categories != sorted(categories) or set(categories) != set(counts) or set(
            categories
        ) != set(episode_hashes):
            raise ValueError("activation category receipt identity is inconsistent")
        identity.update(
            {
                "activation_category_contract_sha256": require_digest(
                    inputs["activation_category_contract_sha256"],
                    "activation_category_contract_sha256",
                ),
                "activation_categories": categories,
                "activation_category_episode_counts": counts,
                "activation_category_episode_ids_sha256": episode_hashes,
            }
        )
        split_keys = {
            "activation_category_split_id",
            "activation_category_split_manifest_sha256",
            "activation_category_split_role",
            "activation_category_split_episode_id_set_sha256",
            "activation_category_split_pair_id_set_sha256",
            "activation_category_split_verification_status",
        }
        present_split_keys = split_keys & set(inputs)
        if present_split_keys and present_split_keys != split_keys:
            raise ValueError("activation-category split provenance is incomplete")
        if present_split_keys:
            role = str(inputs["activation_category_split_role"])
            if role not in {"discovery", "calibration", "cluster_validation"}:
                raise ValueError("effect shard uses a forbidden activation-category split role")
            if str(inputs["activation_category_split_verification_status"]) != "pass":
                raise ValueError("effect shard split provenance was not verified")
            identity.update(
                {
                    "activation_category_split_id": require_digest(
                        inputs["activation_category_split_id"],
                        "activation category split ID",
                    ),
                    "activation_category_split_manifest_sha256": require_digest(
                        inputs["activation_category_split_manifest_sha256"],
                        "activation category split manifest hash",
                    ),
                    "activation_category_split_role": role,
                    "activation_category_split_episode_id_set_sha256": require_digest(
                        inputs["activation_category_split_episode_id_set_sha256"],
                        "activation category split episode set hash",
                    ),
                    "activation_category_split_pair_id_set_sha256": require_digest(
                        inputs["activation_category_split_pair_id_set_sha256"],
                        "activation category split pair set hash",
                    ),
                    "activation_category_split_verification_status": "pass",
                }
            )
            split_schema = str(
                inputs.get(
                    "activation_category_split_schema",
                    "sch.vpd-activation-category-split-manifest.v1",
                )
            )
            semantic_status = str(
                inputs.get(
                    "activation_category_split_semantic_disjointness",
                    "not_provided_v1",
                )
            )
            identity.update(
                {
                    "activation_category_split_schema": split_schema,
                    "activation_category_split_semantic_disjointness": semantic_status,
                }
            )
            if split_schema in {
                "sch.vpd-activation-category-split-manifest.v2",
                "sch.vpd-activation-category-split-manifest.v3",
            }:
                if semantic_status != "pass":
                    raise ValueError("semantic split disjointness was not verified")
                identity[
                    "activation_category_split_template_group_set_sha256"
                ] = require_digest(
                    inputs.get(
                        "activation_category_split_template_group_set_sha256"
                    ),
                    "activation category split template-group set hash",
                )
            elif split_schema != "sch.vpd-activation-category-split-manifest.v1":
                raise ValueError("unsupported activation-category split schema")
        if receipt.get("schema") in {
            "sch.vpd-effect-shard-receipt.v3",
            "sch.vpd-effect-shard-receipt.v4",
            "sch.vpd-effect-shard-receipt.v5",
        }:
            contract = dict(inputs["activation_category_uncertainty"])
            block_count = int(contract["block_count"])
            sketch_dimension = int(contract["countsketch_dimension"])
            seed = int(contract["countsketch_seed"])
            if contract != uncertainty_contract(
                block_count=block_count,
                sketch_dimension=sketch_dimension,
                seed=seed,
            ):
                raise ValueError("activation category uncertainty contract is invalid")
            identity.update(
                {
                    "activation_category_uncertainty": contract,
                    "activation_category_pair_block_assignment_sha256": require_digest(
                        inputs["activation_category_pair_block_assignment_sha256"],
                        "activation category pair block assignment hash",
                    ),
                }
            )
        if receipt.get("schema") in {
            "sch.vpd-effect-shard-receipt.v4",
            "sch.vpd-effect-shard-receipt.v5",
        }:
            condition_contract = dict(
                inputs["activation_category_condition_robustness"]
            )
            expected_condition_contract = condition_robustness_contract(
                minimum_coverage=float(condition_contract["minimum_coverage"]),
                minimum_direction_cosine=float(
                    condition_contract["minimum_direction_cosine"]
                ),
                minimum_relative_norm=float(
                    condition_contract["minimum_relative_norm"]
                ),
            )
            if condition_contract != expected_condition_contract:
                raise ValueError("activation category condition contract is invalid")
            condition_names = [
                str(value)
                for value in inputs["activation_category_condition_names"]
            ]
            condition_categories = [
                str(value)
                for value in inputs[
                    "activation_category_condition_categories"
                ]
            ]
            condition_counts = {
                str(key): int(value)
                for key, value in inputs[
                    "activation_category_condition_episode_counts"
                ].items()
            }
            if (
                condition_names != sorted(condition_names)
                or len(condition_names) != len(set(condition_names))
                or len(condition_categories) != len(condition_names)
                or set(condition_counts) != set(condition_names)
                or any(category not in set(categories) for category in condition_categories)
                or require_digest(
                    inputs["activation_category_condition_set_sha256"],
                    "activation category condition set hash",
                )
                != content_sha256(condition_names)
            ):
                raise ValueError("activation category condition identity is inconsistent")
            identity.update(
                {
                    "activation_category_condition_robustness": condition_contract,
                    "activation_category_condition_names": condition_names,
                    "activation_category_condition_categories": condition_categories,
                    "activation_category_condition_episode_counts": condition_counts,
                    "activation_category_condition_set_sha256": content_sha256(
                        condition_names
                    ),
                }
            )
        if receipt.get("schema") == "sch.vpd-effect-shard-receipt.v5":
            template_contract = dict(
                inputs["activation_category_template_robustness"]
            )
            expected_template_contract = template_robustness_contract(
                minimum_coverage=float(template_contract["minimum_coverage"]),
                minimum_direction_cosine=float(
                    template_contract["minimum_direction_cosine"]
                ),
                minimum_relative_norm=float(
                    template_contract["minimum_relative_norm"]
                ),
            )
            if template_contract != expected_template_contract:
                raise ValueError("activation category template contract is invalid")
            cell_names = [
                str(value)
                for value in inputs["activation_category_template_cell_names"]
            ]
            cell_categories = [
                str(value)
                for value in inputs["activation_category_template_cell_categories"]
            ]
            cell_groups = [
                str(value)
                for value in inputs["activation_category_template_cell_groups"]
            ]
            cell_counts = {
                str(key): int(value)
                for key, value in inputs[
                    "activation_category_template_cell_episode_counts"
                ].items()
            }
            if (
                cell_names != sorted(cell_names)
                or len(cell_names) != len(set(cell_names))
                or len(cell_categories) != len(cell_names)
                or len(cell_groups) != len(cell_names)
                or set(cell_counts) != set(cell_names)
                or any(category not in set(categories) for category in cell_categories)
                or any(not group for group in cell_groups)
                or require_digest(
                    inputs["activation_category_template_cell_set_sha256"],
                    "activation category template cell set hash",
                )
                != content_sha256(cell_names)
            ):
                raise ValueError("activation category template identity is inconsistent")
            identity.update(
                {
                    "activation_category_template_robustness": template_contract,
                    "activation_category_template_cell_names": cell_names,
                    "activation_category_template_cell_categories": cell_categories,
                    "activation_category_template_cell_groups": cell_groups,
                    "activation_category_template_cell_episode_counts": cell_counts,
                    "activation_category_template_cell_set_sha256": content_sha256(
                        cell_names
                    ),
                }
            )
    return identity


def validate_shard_receipt(
    receipt: dict[str, Any],
    *,
    shard_dir: Path | None,
    shard_path: Path | None = None,
    packets: dict[str, dict[str, Any]],
    packet_hashes: dict[str, str],
    workers: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, Any],
    list[str],
    np.ndarray,
    np.ndarray | None,
    tuple[np.ndarray, np.ndarray, int, int, int] | None,
    tuple[list[str], list[str], np.ndarray, np.ndarray] | None,
    tuple[list[str], list[str], list[str], np.ndarray, np.ndarray] | None,
]:
    if receipt.get("schema") not in {
        "sch.vpd-effect-shard-receipt.v1",
        "sch.vpd-effect-shard-receipt.v2",
        "sch.vpd-effect-shard-receipt.v3",
        "sch.vpd-effect-shard-receipt.v4",
        "sch.vpd-effect-shard-receipt.v5",
    }:
        raise ValueError("unsupported effect shard receipt schema")
    packet_id = str(receipt["packet_id"])
    if packet_id not in packets:
        raise ValueError(f"effect shard references unknown packet: {packet_id}")
    packet = packets[packet_id]
    if str(receipt["cluster_id"]) != str(packet["cluster_id"]):
        raise ValueError("effect shard cluster does not match packet")
    worker_id = str(receipt["worker_id"])
    worker = workers.get(worker_id)
    if worker is None or not bool(worker.get("enabled", False)):
        raise ValueError(f"effect shard worker is absent or disabled: {worker_id}")
    if worker_id != packet["distribution"].get("assigned_worker"):
        raise ValueError(f"effect shard came from unassigned worker: {worker_id}")
    verify_signature(receipt, worker)
    inputs = receipt["inputs"]
    if require_digest(inputs["packet_sha256"], "packet_sha256") != packet_hashes[packet_id]:
        raise ValueError("effect worker used a different packet artifact")
    identity = effect_identity(receipt)
    if identity["effect_space_id"] != str(packet["effect_refinement"]["effect_space_id"]):
        raise ValueError("effect shard used a different activation effect space")
    if identity["effect_space_id"].endswith("__proposal_start") and identity[
        "generated_tokens"
    ] != 0:
        raise ValueError("proposal_start effects must be captured before token generation")
    shard_filename = str(receipt["shard_filename"])
    if Path(shard_filename).name != shard_filename:
        raise ValueError("effect shard filename must be a basename")
    if shard_path is None:
        if shard_dir is None:
            raise ValueError("effect shard location was not provided")
        shard_path = shard_dir / shard_filename
    if require_digest(receipt["shard_sha256"], "shard_sha256") != sha256_file(shard_path):
        raise ValueError("effect shard artifact hash mismatch")
    component_ids, delta_h = load_effect_archive(shard_path)
    category_payload = load_activation_category_effects(shard_path)
    uncertainty_payload = load_category_effect_uncertainty(shard_path)
    condition_payload = load_category_condition_robustness(shard_path)
    template_payload = load_category_template_robustness(shard_path)
    if receipt.get("schema") in {
        "sch.vpd-effect-shard-receipt.v2",
        "sch.vpd-effect-shard-receipt.v3",
        "sch.vpd-effect-shard-receipt.v4",
        "sch.vpd-effect-shard-receipt.v5",
    }:
        if category_payload is None:
            raise ValueError("v2 effect shard omits activation-category arrays")
        category_names, category_counts, category_delta = category_payload
        if category_names != identity["activation_categories"]:
            raise ValueError("effect shard categories differ from receipt")
        receipt_counts = identity["activation_category_episode_counts"]
        if [int(receipt_counts[name]) for name in category_names] != category_counts.tolist():
            raise ValueError("effect shard category counts differ from receipt")
        if int(receipt.get("activation_category_count", -1)) != len(category_names):
            raise ValueError("effect shard category count differs from receipt")
        weighted = np.average(category_delta, axis=1, weights=category_counts).astype(
            np.float32
        )
        if not np.allclose(weighted, delta_h, rtol=1e-5, atol=1e-6):
            raise ValueError("aggregate delta_h is not the category-count-weighted mean")
    elif category_payload is not None:
        raise ValueError("v1 effect shard may not carry unsigned category semantics")
    if receipt.get("schema") in {
        "sch.vpd-effect-shard-receipt.v3",
        "sch.vpd-effect-shard-receipt.v4",
        "sch.vpd-effect-shard-receipt.v5",
    }:
        if uncertainty_payload is None:
            raise ValueError("v3 effect shard omits category uncertainty arrays")
        uncertainty_counts, uncertainty_sums, block_count, sketch_dimension, seed = (
            uncertainty_payload
        )
        contract = identity["activation_category_uncertainty"]
        if (
            block_count != int(contract["block_count"])
            or sketch_dimension != int(contract["countsketch_dimension"])
            or seed != int(contract["countsketch_seed"])
        ):
            raise ValueError("effect shard uncertainty arrays differ from receipt")
        if uncertainty_counts.sum(axis=1).tolist() != category_counts.tolist():
            raise ValueError("effect shard uncertainty counts differ from category counts")
        sketched_means = uncertainty_sums.sum(axis=2) / uncertainty_counts.sum(axis=1)[
            None, :, None
        ]
        expected_sketches = countsketch(
            category_delta, dimension=sketch_dimension, seed=seed
        )
        if not np.allclose(sketched_means, expected_sketches, rtol=2e-4, atol=2e-5):
            raise ValueError("effect shard uncertainty sums do not reproduce category means")
    elif uncertainty_payload is not None:
        raise ValueError("only v3/v4 effect shards may carry category uncertainty arrays")
    if receipt.get("schema") in {
        "sch.vpd-effect-shard-receipt.v4",
        "sch.vpd-effect-shard-receipt.v5",
    }:
        if condition_payload is None:
            raise ValueError("v4 effect shard omits condition robustness arrays")
        condition_names, condition_categories, condition_counts, condition_sums = (
            condition_payload
        )
        if condition_names != identity["activation_category_condition_names"] or (
            condition_categories
            != identity["activation_category_condition_categories"]
        ):
            raise ValueError("condition robustness arrays differ from receipt")
        receipt_condition_counts = identity[
            "activation_category_condition_episode_counts"
        ]
        if [receipt_condition_counts[name] for name in condition_names] != (
            condition_counts.tolist()
        ):
            raise ValueError("condition robustness counts differ from receipt")
        grouped_counts = {
            category: sum(
                int(condition_counts[index])
                for index, value in enumerate(condition_categories)
                if value == category
            )
            for category in category_names
        }
        if [grouped_counts[name] for name in category_names] != category_counts.tolist():
            raise ValueError("condition robustness counts differ from category counts")
        condition_category_sketches = np.stack(
            [
                condition_sums[
                    :,
                    [
                        index
                        for index, value in enumerate(condition_categories)
                        if value == category
                    ],
                    :,
                ].sum(axis=1)
                / grouped_counts[category]
                for category in category_names
            ],
            axis=1,
        )
        expected_sketches = countsketch(
            category_delta, dimension=sketch_dimension, seed=seed
        )
        if not np.allclose(
            condition_category_sketches,
            expected_sketches,
            rtol=2e-4,
            atol=2e-5,
        ):
            raise ValueError("condition robustness sums do not reproduce category means")
    elif condition_payload is not None:
        raise ValueError("only v4/v5 effect shards may carry condition robustness arrays")
    if receipt.get("schema") == "sch.vpd-effect-shard-receipt.v5":
        if template_payload is None:
            raise ValueError("v5 effect shard omits template robustness arrays")
        cell_names, cell_categories, cell_groups, cell_counts, cell_sums = (
            template_payload
        )
        if (
            cell_names != identity["activation_category_template_cell_names"]
            or cell_categories
            != identity["activation_category_template_cell_categories"]
            or cell_groups != identity["activation_category_template_cell_groups"]
        ):
            raise ValueError("template robustness arrays differ from receipt")
        receipt_cell_counts = identity[
            "activation_category_template_cell_episode_counts"
        ]
        if [receipt_cell_counts[name] for name in cell_names] != cell_counts.tolist():
            raise ValueError("template robustness counts differ from receipt")
        grouped_counts = {
            category: sum(
                int(cell_counts[index])
                for index, value in enumerate(cell_categories)
                if value == category
            )
            for category in category_names
        }
        if [grouped_counts[name] for name in category_names] != category_counts.tolist():
            raise ValueError("template robustness counts differ from category counts")
        template_category_sketches = np.stack(
            [
                cell_sums[
                    :,
                    [
                        index
                        for index, value in enumerate(cell_categories)
                        if value == category
                    ],
                    :,
                ].sum(axis=1)
                / grouped_counts[category]
                for category in category_names
            ],
            axis=1,
        )
        expected_sketches = countsketch(
            category_delta, dimension=sketch_dimension, seed=seed
        )
        if not np.allclose(
            template_category_sketches,
            expected_sketches,
            rtol=2e-4,
            atol=2e-5,
        ):
            raise ValueError("template robustness sums do not reproduce category means")
    elif template_payload is not None:
        raise ValueError("only v5 effect shards may carry template robustness arrays")
    packet_ids = sorted(str(value) for value in packet["component_ids"])
    if sorted(component_ids) != packet_ids:
        raise ValueError("effect shard is not the exact packet component set")
    if content_sha256(sorted(component_ids)) != str(receipt["component_id_set_sha256"]):
        raise ValueError("effect shard component set content address mismatch")
    if int(receipt["rows"]) != len(component_ids):
        raise ValueError("effect shard receipt row count mismatch")
    if int(receipt["hidden_size"]) != int(delta_h.shape[1]):
        raise ValueError("effect shard receipt hidden size mismatch")
    if str(receipt["dtype"]) != "float32":
        raise ValueError("effect shard receipt must freeze float32 merge dtype")
    if not bool(receipt.get("factor_restoration_verified", False)):
        raise ValueError("effect worker did not verify LoRA factor restoration")
    row_index = {component_id: index for index, component_id in enumerate(component_ids)}
    ordered = delta_h[[row_index[component_id] for component_id in packet_ids]]
    ordered_categories = (
        None
        if category_payload is None
        else category_payload[2][[row_index[component_id] for component_id in packet_ids]]
    )
    accepted = {
        "schema": "sch.vpd-accepted-effect-shard.v1",
        "result_id": str(receipt["result_id"]),
        "packet_id": packet_id,
        "cluster_id": str(packet["cluster_id"]),
        "worker_id": worker_id,
        "shard_sha256": str(receipt["shard_sha256"]),
        "component_id_set_sha256": str(receipt["component_id_set_sha256"]),
        "rows": len(packet_ids),
        "hidden_size": int(delta_h.shape[1]),
        "identity": identity,
        "activation_category_count": (
            0 if category_payload is None else len(category_payload[0])
        ),
        "signature_verified": True,
    }
    ordered_uncertainty = (
        None
        if uncertainty_payload is None
        else (
            uncertainty_payload[0],
            uncertainty_payload[1][[row_index[component_id] for component_id in packet_ids]],
            uncertainty_payload[2],
            uncertainty_payload[3],
            uncertainty_payload[4],
        )
    )
    ordered_conditions = (
        None
        if condition_payload is None
        else (
            condition_payload[0],
            condition_payload[1],
            condition_payload[2],
            condition_payload[3][
                [row_index[component_id] for component_id in packet_ids]
            ],
        )
    )
    ordered_templates = (
        None
        if template_payload is None
        else (
            template_payload[0],
            template_payload[1],
            template_payload[2],
            template_payload[3],
            template_payload[4][
                [row_index[component_id] for component_id in packet_ids]
            ],
        )
    )
    return (
        accepted,
        packet_ids,
        ordered,
        ordered_categories,
        ordered_uncertainty,
        ordered_conditions,
        ordered_templates,
    )


def merge_effect_shards(
    *,
    bundle_dir: Path,
    registry_path: Path,
    receipt_paths: list[Path],
    shard_dir: Path,
    output_dir: Path,
    require_complete: bool = True,
    shard_paths_by_result_id: dict[str, Path] | None = None,
) -> dict[str, Any]:
    packets, packet_hashes = load_packets(bundle_dir)
    workers = load_registry(registry_path)
    accepted: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    category_arrays: dict[str, np.ndarray] = {}
    uncertainty_arrays: dict[str, np.ndarray] = {}
    uncertainty_metadata: tuple[np.ndarray, int, int, int] | None = None
    condition_arrays: dict[str, np.ndarray] = {}
    condition_metadata: tuple[list[str], list[str], np.ndarray] | None = None
    template_arrays: dict[str, np.ndarray] = {}
    template_metadata: tuple[list[str], list[str], list[str], np.ndarray] | None = None
    identities: dict[str, dict[str, Any]] = {}
    seen_worker_packet: dict[tuple[str, str], str] = {}
    for path in sorted(receipt_paths, key=lambda item: item.as_posix()):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        resolved_shard_path = None
        if shard_paths_by_result_id is not None:
            resolved_shard_path = shard_paths_by_result_id.get(str(receipt.get("result_id")))
            if resolved_shard_path is None:
                raise ValueError("no content-addressed shard object was provided for receipt")
        (
            accepted_row,
            component_ids,
            delta_h,
            category_delta,
            category_uncertainty,
            category_conditions,
            category_templates,
        ) = validate_shard_receipt(
            receipt,
            shard_dir=shard_dir,
            shard_path=resolved_shard_path,
            packets=packets,
            packet_hashes=packet_hashes,
            workers=workers,
        )
        key = (accepted_row["packet_id"], accepted_row["worker_id"])
        prior = seen_worker_packet.get(key)
        if prior is not None:
            if prior != accepted_row["result_id"]:
                raise ValueError(f"conflicting effect shards for packet: {key}")
            continue
        seen_worker_packet[key] = accepted_row["result_id"]
        identity_hash = content_sha256(accepted_row["identity"])
        identities[identity_hash] = accepted_row["identity"]
        for component_id, vector in zip(component_ids, delta_h, strict=True):
            if component_id in arrays:
                raise ValueError(f"component appears in multiple effect shards: {component_id}")
            arrays[component_id] = vector
        if category_delta is not None:
            for component_id, matrix_row in zip(
                component_ids, category_delta, strict=True
            ):
                category_arrays[component_id] = matrix_row
        if category_uncertainty is not None:
            block_counts, block_sums, block_count, sketch_dimension, sketch_seed = (
                category_uncertainty
            )
            current_metadata = (
                block_counts,
                block_count,
                sketch_dimension,
                sketch_seed,
            )
            if uncertainty_metadata is None:
                uncertainty_metadata = current_metadata
            elif (
                not np.array_equal(uncertainty_metadata[0], block_counts)
                or uncertainty_metadata[1:] != current_metadata[1:]
            ):
                raise ValueError("effect shards have inconsistent uncertainty blocks")
            for component_id, block_row in zip(
                component_ids, block_sums, strict=True
            ):
                uncertainty_arrays[component_id] = block_row
        if category_conditions is not None:
            condition_names, condition_categories, condition_counts, condition_sums = (
                category_conditions
            )
            current_condition_metadata = (
                condition_names,
                condition_categories,
                condition_counts,
            )
            if condition_metadata is None:
                condition_metadata = current_condition_metadata
            elif (
                condition_metadata[0] != condition_names
                or condition_metadata[1] != condition_categories
                or not np.array_equal(condition_metadata[2], condition_counts)
            ):
                raise ValueError("effect shards have inconsistent condition robustness")
            for component_id, condition_row in zip(
                component_ids, condition_sums, strict=True
            ):
                condition_arrays[component_id] = condition_row
        if category_templates is not None:
            (
                template_cell_names,
                template_cell_categories,
                template_cell_groups,
                template_cell_counts,
                template_cell_sums,
            ) = category_templates
            current_template_metadata = (
                template_cell_names,
                template_cell_categories,
                template_cell_groups,
                template_cell_counts,
            )
            if template_metadata is None:
                template_metadata = current_template_metadata
            elif (
                template_metadata[0] != template_cell_names
                or template_metadata[1] != template_cell_categories
                or template_metadata[2] != template_cell_groups
                or not np.array_equal(template_metadata[3], template_cell_counts)
            ):
                raise ValueError("effect shards have inconsistent template robustness")
            for component_id, template_row in zip(
                component_ids, template_cell_sums, strict=True
            ):
                template_arrays[component_id] = template_row
        accepted.append(accepted_row)
    if len(identities) != 1:
        raise ValueError("effect shards do not share one frozen model/effect-space identity")
    expected_ids = {
        str(component_id)
        for packet in packets.values()
        for component_id in packet["component_ids"]
    }
    missing_ids = sorted(expected_ids - set(arrays))
    extra_ids = sorted(set(arrays) - expected_ids)
    completed_packets = {str(row["packet_id"]) for row in accepted}
    missing_packets = sorted(set(packets) - completed_packets)
    if extra_ids:
        raise ValueError(f"effect merge has unexpected components: {len(extra_ids)}")
    if require_complete and (missing_ids or missing_packets):
        raise ValueError(
            f"effect merge is incomplete: {len(missing_packets)} packets, {len(missing_ids)} components"
        )
    ordered_ids = sorted(arrays)
    hidden_sizes = {int(vector.shape[0]) for vector in arrays.values()}
    if len(hidden_sizes) != 1:
        raise ValueError("effect shards have inconsistent hidden dimensions")
    matrix = np.stack([arrays[component_id] for component_id in ordered_ids]).astype(np.float32)
    max_length = max((len(component_id) for component_id in ordered_ids), default=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    effects_path = output_dir / "activation_effects.npz"
    output_arrays = {
        "component_ids": np.asarray(ordered_ids, dtype=f"<U{max_length}"),
        "delta_h": matrix,
    }
    identity = next(iter(identities.values()))
    categories = identity.get("activation_categories")
    category_matrix: np.ndarray | None = None
    if categories is not None:
        if set(category_arrays) != set(ordered_ids):
            raise ValueError("activation-category merge does not cover every component")
        category_matrix = np.stack(
            [category_arrays[component_id] for component_id in ordered_ids]
        ).astype(np.float32)
        counts = np.asarray(
            [identity["activation_category_episode_counts"][name] for name in categories],
            dtype=np.int64,
        )
        max_category_length = max(len(name) for name in categories)
        output_arrays.update(
            {
                "category_names": np.asarray(
                    categories, dtype=f"<U{max_category_length}"
                ),
                "category_episode_counts": counts,
                "delta_h_by_category": category_matrix,
            }
        )
        uncertainty_matrix: np.ndarray | None = None
        if uncertainty_metadata is not None:
            if set(uncertainty_arrays) != set(ordered_ids):
                raise ValueError("category uncertainty merge does not cover every component")
            uncertainty_matrix = np.stack(
                [uncertainty_arrays[component_id] for component_id in ordered_ids]
            ).astype(np.float32)
            block_counts, block_count, sketch_dimension, sketch_seed = uncertainty_metadata
            output_arrays.update(
                {
                    "category_uncertainty_block_counts": block_counts,
                    "delta_h_countsketch_block_sums": uncertainty_matrix,
                    "uncertainty_block_count": np.asarray(block_count, dtype=np.int64),
                    "uncertainty_countsketch_dimension": np.asarray(
                        sketch_dimension, dtype=np.int64
                    ),
                    "uncertainty_countsketch_seed": np.asarray(
                        sketch_seed, dtype=np.int64
                    ),
                }
            )
        elif uncertainty_arrays:
            raise ValueError("category uncertainty arrays exist without shared metadata")
        condition_matrix: np.ndarray | None = None
        if condition_metadata is not None:
            if set(condition_arrays) != set(ordered_ids):
                raise ValueError("condition robustness merge does not cover every component")
            condition_matrix = np.stack(
                [condition_arrays[component_id] for component_id in ordered_ids]
            ).astype(np.float32)
            condition_names, condition_categories, condition_counts = condition_metadata
            output_arrays.update(
                {
                    "condition_names": np.asarray(
                        condition_names,
                        dtype=f"<U{max(len(value) for value in condition_names)}",
                    ),
                    "condition_category_names": np.asarray(
                        condition_categories,
                        dtype=f"<U{max(len(value) for value in condition_categories)}",
                    ),
                    "condition_episode_counts": condition_counts,
                    "delta_h_countsketch_condition_sums": condition_matrix,
                }
            )
        elif condition_arrays:
            raise ValueError("condition robustness arrays exist without shared metadata")
        template_matrix: np.ndarray | None = None
        if template_metadata is not None:
            if set(template_arrays) != set(ordered_ids):
                raise ValueError("template robustness merge does not cover every component")
            template_matrix = np.stack(
                [template_arrays[component_id] for component_id in ordered_ids]
            ).astype(np.float32)
            (
                template_cell_names,
                template_cell_categories,
                template_cell_groups,
                template_cell_counts,
            ) = template_metadata
            output_arrays.update(
                {
                    "template_cell_names": np.asarray(
                        template_cell_names,
                        dtype=f"<U{max(len(value) for value in template_cell_names)}",
                    ),
                    "template_cell_category_names": np.asarray(
                        template_cell_categories,
                        dtype=f"<U{max(len(value) for value in template_cell_categories)}",
                    ),
                    "template_cell_group_names": np.asarray(
                        template_cell_groups,
                        dtype=f"<U{max(len(value) for value in template_cell_groups)}",
                    ),
                    "template_cell_episode_counts": template_cell_counts,
                    "delta_h_countsketch_template_cell_sums": template_matrix,
                }
            )
        elif template_arrays:
            raise ValueError("template robustness arrays exist without shared metadata")
    elif category_arrays:
        raise ValueError("category arrays exist without a frozen category identity")
    else:
        uncertainty_matrix = None
        condition_matrix = None
        template_matrix = None
    write_deterministic_npz(effects_path, output_arrays)
    accepted.sort(key=lambda row: (row["packet_id"], row["worker_id"], row["result_id"]))
    accepted_path = output_dir / "accepted_effect_shards.jsonl"
    from experiments.authz_mechinterp.cluster_vpd_for_sae import write_jsonl

    write_jsonl(accepted_path, accepted)
    merge_body = {
        "schema": "sch.vpd-effect-shard-merge.v1",
        "status": "complete" if not missing_ids and not missing_packets else "incomplete",
        "source": {
            "bundle_manifest_sha256": sha256_file(bundle_dir / "manifest.json"),
            "worker_registry_sha256": sha256_file(registry_path),
        },
        "identity": identity,
        "coverage": {
            "expected_packets": len(packets),
            "accepted_packets": len(completed_packets),
            "missing_packet_ids": missing_packets,
            "expected_components": len(expected_ids),
            "merged_components": len(ordered_ids),
            "missing_component_ids": missing_ids,
            "exact_partition": not missing_ids and not extra_ids,
        },
        "arrays": {
            "rows": len(ordered_ids),
            "hidden_size": int(matrix.shape[1]),
            "dtype": "float32",
            "component_id_set_sha256": content_sha256(ordered_ids),
            "delta_h_content_sha256": content_sha256(matrix.tolist()),
            "activation_category_count": 0 if categories is None else len(categories),
            "delta_h_by_category_content_sha256": (
                None
                if category_matrix is None
                else content_sha256(category_matrix.tolist())
            ),
            "category_uncertainty_block_count": (
                0 if uncertainty_metadata is None else uncertainty_metadata[1]
            ),
            "delta_h_countsketch_block_sums_content_sha256": (
                None
                if uncertainty_matrix is None
                else content_sha256(uncertainty_matrix.tolist())
            ),
            "activation_category_condition_count": (
                0 if condition_metadata is None else len(condition_metadata[0])
            ),
            "delta_h_countsketch_condition_sums_content_sha256": (
                None
                if condition_matrix is None
                else content_sha256(condition_matrix.tolist())
            ),
            "activation_category_template_cell_count": (
                0 if template_metadata is None else len(template_metadata[0])
            ),
            "delta_h_countsketch_template_cell_sums_content_sha256": (
                None
                if template_matrix is None
                else content_sha256(template_matrix.tolist())
            ),
        },
        "artifacts": {
            "activation_effects": {
                "path": effects_path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(effects_path),
            },
            "accepted_effect_shards": {
                "path": accepted_path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(accepted_path),
            },
        },
    }
    manifest = {**merge_body, "merge_id": content_sha256(merge_body)}
    write_json(output_dir / "effect_merge_manifest.json", manifest)
    return manifest


def verify_effect_merge(output_dir: Path) -> dict[str, Any]:
    manifest = json.loads(
        (output_dir / "effect_merge_manifest.json").read_text(encoding="utf-8")
    )
    merge_id = str(manifest.pop("merge_id"))
    if content_sha256(manifest) != merge_id:
        raise ValueError("effect merge manifest content address mismatch")
    for name, entry in manifest["artifacts"].items():
        if sha256_file(output_dir / entry["path"]) != entry["sha256"]:
            raise ValueError(f"effect merge artifact hash mismatch: {name}")
    effect_path = output_dir / manifest["artifacts"]["activation_effects"]["path"]
    ids, delta_h = load_effect_archive(effect_path)
    category_payload = load_activation_category_effects(effect_path)
    uncertainty_payload = load_category_effect_uncertainty(effect_path)
    condition_payload = load_category_condition_robustness(effect_path)
    template_payload = load_category_template_robustness(effect_path)
    if len(ids) != int(manifest["arrays"]["rows"]):
        raise ValueError("effect merge row count mismatch")
    if delta_h.shape[1] != int(manifest["arrays"]["hidden_size"]):
        raise ValueError("effect merge hidden size mismatch")
    if content_sha256(sorted(ids)) != manifest["arrays"]["component_id_set_sha256"]:
        raise ValueError("effect merge component set mismatch")
    if content_sha256(delta_h.tolist()) != manifest["arrays"]["delta_h_content_sha256"]:
        raise ValueError("effect merge delta_h content mismatch")
    expected_category_count = int(manifest["arrays"].get("activation_category_count", 0))
    if (category_payload is None) != (expected_category_count == 0):
        raise ValueError("effect merge activation-category presence mismatch")
    if category_payload is not None:
        category_names, _, category_delta = category_payload
        if len(category_names) != expected_category_count:
            raise ValueError("effect merge activation-category count mismatch")
        if content_sha256(category_delta.tolist()) != manifest["arrays"].get(
            "delta_h_by_category_content_sha256"
        ):
            raise ValueError("effect merge category delta_h content mismatch")
    expected_block_count = int(
        manifest["arrays"].get("category_uncertainty_block_count", 0)
    )
    if (uncertainty_payload is None) != (expected_block_count == 0):
        raise ValueError("effect merge category uncertainty presence mismatch")
    if uncertainty_payload is not None:
        _, block_sums, block_count, _, _ = uncertainty_payload
        if block_count != expected_block_count:
            raise ValueError("effect merge category uncertainty block count mismatch")
        if content_sha256(block_sums.tolist()) != manifest["arrays"].get(
            "delta_h_countsketch_block_sums_content_sha256"
        ):
            raise ValueError("effect merge category uncertainty content mismatch")
    expected_condition_count = int(
        manifest["arrays"].get("activation_category_condition_count", 0)
    )
    if (condition_payload is None) != (expected_condition_count == 0):
        raise ValueError("effect merge condition robustness presence mismatch")
    if condition_payload is not None:
        condition_names, _, _, condition_sums = condition_payload
        if len(condition_names) != expected_condition_count:
            raise ValueError("effect merge condition robustness count mismatch")
        if content_sha256(condition_sums.tolist()) != manifest["arrays"].get(
            "delta_h_countsketch_condition_sums_content_sha256"
        ):
            raise ValueError("effect merge condition robustness content mismatch")
    expected_template_count = int(
        manifest["arrays"].get("activation_category_template_cell_count", 0)
    )
    if (template_payload is None) != (expected_template_count == 0):
        raise ValueError("effect merge template robustness presence mismatch")
    if template_payload is not None:
        template_names, _, _, _, template_sums = template_payload
        if len(template_names) != expected_template_count:
            raise ValueError("effect merge template robustness count mismatch")
        if content_sha256(template_sums.tolist()) != manifest["arrays"].get(
            "delta_h_countsketch_template_cell_sums_content_sha256"
        ):
            raise ValueError("effect merge template robustness content mismatch")
    return {
        "schema": "sch.vpd-effect-merge-verification.v1",
        "status": "pass",
        "merge_status": manifest["status"],
        "components": len(ids),
        "hidden_size": int(delta_h.shape[1]),
        "activation_categories": expected_category_count,
        "category_uncertainty_blocks": expected_block_count,
        "activation_category_conditions": expected_condition_count,
        "activation_category_template_cells": expected_template_count,
        "merge_id": merge_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and merge signed decentralized VPD activation-effect shards."
    )
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--receipts-dir", type=Path)
    parser.add_argument("--shards-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--verify-output", type=Path)
    parser.add_argument("--sign-receipt", type=Path)
    parser.add_argument("--private-key", type=Path)
    parser.add_argument("--signed-output", type=Path)
    args = parser.parse_args()
    if args.sign_receipt is not None:
        if args.private_key is None or args.signed_output is None:
            parser.error("receipt signing requires --private-key and --signed-output")
        from experiments.authz_mechinterp.merge_vpd_sae_results import load_private_key

        receipt = json.loads(args.sign_receipt.read_text(encoding="utf-8"))
        write_json(args.signed_output, sign_result(receipt, load_private_key(args.private_key)))
        return
    if args.verify_output is not None:
        print(json.dumps(verify_effect_merge(args.verify_output), indent=2, sort_keys=True))
        return
    required = (args.bundle, args.registry, args.receipts_dir, args.shards_dir, args.output_dir)
    if any(value is None for value in required):
        parser.error(
            "merge requires --bundle, --registry, --receipts-dir, --shards-dir, and --output-dir"
        )
    receipt_paths = sorted(args.receipts_dir.glob("*.json"))
    if not receipt_paths:
        raise ValueError("receipts directory contains no JSON receipts")
    manifest = merge_effect_shards(
        bundle_dir=args.bundle,
        registry_path=args.registry,
        receipt_paths=receipt_paths,
        shard_dir=args.shards_dir,
        output_dir=args.output_dir,
        require_complete=not args.allow_incomplete,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
