from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.activation_categories import (
    activation_category,
    category_contract,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    stable_hash,
    write_json,
)
from experiments.authz_mechinterp.effect_uncertainty import (
    countsketch,
    validate_condition_robustness_arrays,
    validate_template_robustness_arrays,
    validate_uncertainty_arrays,
)
from experiments.authz_mechinterp.merge_vpd_effect_shards import (
    write_deterministic_npz,
)


SCHEMA = "sch.vpd-canonical-effect-shard-assembly.v1"
COMPONENT_RE = re.compile(r"^(?P<module>.+):svdblock(?P<block>\d{2,})$")
MINIMUM_PRODUCTION_COUNT_SKETCH_DIMENSION = 256
ALLOWED_SPLIT_ROLES = {"discovery", "calibration", "cluster_validation"}
ALLOWED_SEMANTIC_SPLIT_SCHEMAS = {
    "sch.vpd-activation-category-split-manifest.v2",
    "sch.vpd-activation-category-split-manifest.v3",
}


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes(order="C")).hexdigest()


def _require_digest(value: Any, *, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return digest


def _validate_split_provenance(
    split_provenance: dict[str, Any],
    *,
    episode_ids: list[str],
    pair_ids: list[str],
    template_groups: list[str],
) -> dict[str, Any]:
    if not isinstance(split_provenance, dict):
        raise ValueError("canonical effect assembly requires split provenance")
    schema = str(split_provenance.get("activation_category_split_schema", ""))
    role = str(split_provenance.get("activation_category_split_role", ""))
    if schema not in ALLOWED_SEMANTIC_SPLIT_SCHEMAS:
        raise ValueError("canonical effect assembly requires semantic split V2 or V3")
    if role not in ALLOWED_SPLIT_ROLES:
        raise ValueError("canonical effect assembly split role is forbidden")
    if split_provenance.get("activation_category_split_verification_status") != "pass":
        raise ValueError("canonical effect assembly split verification did not pass")
    if split_provenance.get("activation_category_split_semantic_disjointness") != "pass":
        raise ValueError("canonical effect assembly semantic split disjointness did not pass")
    expected_episode_hash = content_sha256(sorted(episode_ids))
    expected_pair_hash = content_sha256(sorted(set(pair_ids)))
    expected_template_hash = content_sha256(sorted(set(template_groups)))
    if _require_digest(
        split_provenance.get("activation_category_split_episode_id_set_sha256"),
        name="split episode set",
    ) != expected_episode_hash:
        raise ValueError("split episode set differs from canonical effect rows")
    if _require_digest(
        split_provenance.get("activation_category_split_pair_id_set_sha256"),
        name="split pair set",
    ) != expected_pair_hash:
        raise ValueError("split pair set differs from canonical effect rows")
    if _require_digest(
        split_provenance.get("activation_category_split_template_group_set_sha256"),
        name="split template-group set",
    ) != expected_template_hash:
        raise ValueError("split template-group set differs from canonical effect rows")
    return {
        **split_provenance,
        "activation_category_split_id": _require_digest(
            split_provenance.get("activation_category_split_id"), name="split ID"
        ),
        "activation_category_split_manifest_sha256": _require_digest(
            split_provenance.get("activation_category_split_manifest_sha256"),
            name="split manifest",
        ),
        "activation_category_split_episode_id_set_sha256": expected_episode_hash,
        "activation_category_split_pair_id_set_sha256": expected_pair_hash,
        "activation_category_split_template_group_set_sha256": expected_template_hash,
    }


def _validate_episode_rows(
    episodes_by_category: dict[str, list[dict[str, Any]]]
) -> tuple[list[str], dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    if not isinstance(episodes_by_category, dict) or not episodes_by_category:
        raise ValueError("canonical effect assembly requires episode rows by category")
    category_names = sorted(episodes_by_category)
    normalized: dict[str, list[dict[str, str]]] = {}
    all_rows: list[dict[str, str]] = []
    episode_ids: list[str] = []
    pair_sets: dict[str, set[str]] = {}
    template_sets: dict[str, set[str]] = {}
    for category in category_names:
        rows = episodes_by_category[category]
        if not isinstance(category, str) or not category or not isinstance(rows, list) or not rows:
            raise ValueError("canonical effect assembly categories and rows must be nonempty")
        normalized_rows: list[dict[str, str]] = []
        for row in rows:
            values = {
                "episode_id": str(row.get("episode_id", "")),
                "pair_id": str(row.get("pair_id", "")),
                "activation_category": category,
                "condition": str(row.get("condition", "")),
                "template_group": str(row.get("template_group", "")),
            }
            if any(not value for value in values.values()):
                raise ValueError("canonical effect row identity is incomplete")
            if activation_category(row) != category:
                raise ValueError("canonical effect row category differs from metadata")
            normalized_rows.append(values)
        if normalized_rows != sorted(normalized_rows, key=lambda row: row["episode_id"]):
            raise ValueError("canonical effect episode rows must be sorted by episode ID")
        normalized[category] = normalized_rows
        all_rows.extend(normalized_rows)
        episode_ids.extend(row["episode_id"] for row in normalized_rows)
        pair_sets[category] = {row["pair_id"] for row in normalized_rows}
        template_sets[category] = {row["template_group"] for row in normalized_rows}
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("canonical effect episode IDs are not globally unique")
    reference_pairs = pair_sets[category_names[0]]
    if len(reference_pairs) < 2 or any(
        pair_sets[category] != reference_pairs for category in category_names
    ):
        raise ValueError("canonical effect categories must share at least two prompt pairs")
    reference_templates = template_sets[category_names[0]]
    if not reference_templates or any(
        template_sets[category] != reference_templates for category in category_names
    ):
        raise ValueError("canonical effect categories must share all template groups")
    return category_names, normalized, all_rows


def _pair_blocks(
    rows_by_category: dict[str, list[dict[str, str]]], *, requested: int, seed: int
) -> tuple[dict[str, int], int]:
    if requested < 2:
        raise ValueError("canonical effect uncertainty requires at least two blocks")
    pair_ids = sorted({row["pair_id"] for row in next(iter(rows_by_category.values()))})
    block_count = min(requested, len(pair_ids))
    ordered = sorted(pair_ids, key=lambda pair_id: stable_hash(seed, "pair-block", pair_id))
    return {pair_id: index % block_count for index, pair_id in enumerate(ordered)}, block_count


def _validate_component_result(
    component_id: str,
    result: dict[str, Any],
    *,
    category_names: list[str],
    category_counts: list[int],
) -> dict[str, np.ndarray]:
    match = COMPONENT_RE.fullmatch(component_id)
    if match is None:
        raise ValueError("canonical effect component ID must identify an SVD block")
    receipt = result.get("receipt")
    if not isinstance(receipt, dict):
        raise ValueError("canonical effect component capture receipt is absent")
    receipt_body = dict(receipt)
    identifier = receipt_body.pop("canonical_category_effect_capture_id", "")
    if identifier != content_sha256(receipt_body):
        raise ValueError("canonical effect component capture receipt address differs")
    component = receipt["component"]
    if (
        str(component.get("module")) != match.group("module")
        or int(component.get("canonical_block_index", -1)) != int(match.group("block"))
    ):
        raise ValueError("canonical effect component ID differs from capture receipt")
    if list(result.get("category_names", [])) != category_names:
        raise ValueError("canonical effect capture categories differ")
    observed_counts = np.asarray(result.get("category_row_counts"))
    if observed_counts.tolist() != category_counts:
        raise ValueError("canonical effect capture row counts differ")
    samples = result.get("delta_samples_by_category")
    if not isinstance(samples, dict) or sorted(samples) != category_names:
        raise ValueError("canonical effect capture samples are incomplete")
    validated: dict[str, np.ndarray] = {}
    hidden_size: int | None = None
    for category, expected_count in zip(category_names, category_counts, strict=True):
        values = np.asarray(samples[category])
        if (
            values.ndim != 2
            or values.shape[0] != expected_count
            or values.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("canonical effect capture sample matrix is invalid")
        if hidden_size is None:
            hidden_size = int(values.shape[1])
        elif values.shape[1] != hidden_size:
            raise ValueError("canonical effect capture hidden sizes differ")
        if receipt["categories"]["delta_sample_sha256"].get(category) != _array_sha256(values):
            raise ValueError("canonical effect capture sample hash differs")
        validated[category] = values.astype(np.float32)
    observed_means = np.asarray(result.get("delta_mean_by_category"), dtype=np.float32)
    expected_means = np.stack(
        [validated[category].mean(axis=0, dtype=np.float64) for category in category_names]
    ).astype(np.float32)
    if not np.array_equal(observed_means, expected_means):
        raise ValueError("canonical effect category means differ from samples")
    observed_global = np.asarray(result.get("delta_global_mean"), dtype=np.float32)
    expected_global = np.concatenate(
        [validated[category].astype(np.float64) for category in category_names], axis=0
    ).mean(axis=0).astype(np.float32)
    if not np.array_equal(observed_global, expected_global):
        raise ValueError("canonical effect global mean differs from samples")
    return validated


def assemble_canonical_effect_shard(
    *,
    episodes_by_category: dict[str, list[dict[str, Any]]],
    component_results: dict[str, dict[str, Any]],
    dataset_sha256: str,
    split_provenance: dict[str, Any],
    requested_uncertainty_blocks: int,
    countsketch_dimension: int,
    countsketch_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if countsketch_dimension < MINIMUM_PRODUCTION_COUNT_SKETCH_DIMENSION:
        raise ValueError("canonical effect CountSketch dimension is below production minimum")
    if not isinstance(countsketch_seed, int) or countsketch_seed < 0:
        raise ValueError("canonical effect CountSketch seed is invalid")
    category_names, normalized_rows, all_rows = _validate_episode_rows(
        episodes_by_category
    )
    category_counts = [len(normalized_rows[category]) for category in category_names]
    episode_ids = [row["episode_id"] for row in all_rows]
    pair_ids = [row["pair_id"] for row in all_rows]
    template_groups = [row["template_group"] for row in all_rows]
    verified_split = _validate_split_provenance(
        split_provenance,
        episode_ids=episode_ids,
        pair_ids=pair_ids,
        template_groups=template_groups,
    )
    dataset_digest = _require_digest(dataset_sha256, name="dataset")
    if not isinstance(component_results, dict) or not component_results:
        raise ValueError("canonical effect assembly requires component captures")
    component_ids = sorted(component_results)
    if any(not isinstance(component_id, str) for component_id in component_ids):
        raise ValueError("canonical effect component IDs must be strings")
    validated_components = {
        component_id: _validate_component_result(
            component_id,
            component_results[component_id],
            category_names=category_names,
            category_counts=category_counts,
        )
        for component_id in component_ids
    }
    hidden_sizes = {
        next(iter(samples.values())).shape[1]
        for samples in validated_components.values()
    }
    if len(hidden_sizes) != 1:
        raise ValueError("canonical effect components use different hidden sizes")
    hidden_size = next(iter(hidden_sizes))
    pair_blocks, block_count = _pair_blocks(
        normalized_rows, requested=requested_uncertainty_blocks, seed=countsketch_seed
    )
    block_counts = np.zeros((len(category_names), block_count), dtype=np.int64)
    condition_cells = sorted(
        {
            (category, row["condition"])
            for category in category_names
            for row in normalized_rows[category]
        }
    )
    condition_ids = [f"{category}::{condition}" for category, condition in condition_cells]
    condition_index = {cell: index for index, cell in enumerate(condition_cells)}
    condition_counts = np.zeros(len(condition_cells), dtype=np.int64)
    template_cells = sorted(
        {
            (category, row["template_group"])
            for category in category_names
            for row in normalized_rows[category]
        }
    )
    template_ids = [f"{category}::{group}" for category, group in template_cells]
    template_index = {cell: index for index, cell in enumerate(template_cells)}
    template_counts = np.zeros(len(template_cells), dtype=np.int64)
    for category_index, category in enumerate(category_names):
        for row in normalized_rows[category]:
            block_counts[category_index, pair_blocks[row["pair_id"]]] += 1
            condition_counts[condition_index[(category, row["condition"])]] += 1
            template_counts[template_index[(category, row["template_group"])]] += 1
    if np.any(block_counts <= 0):
        raise ValueError("canonical effect pair block has empty category support")
    global_effects: list[np.ndarray] = []
    category_effects: list[np.ndarray] = []
    block_sums_by_component: list[np.ndarray] = []
    condition_sums_by_component: list[np.ndarray] = []
    template_sums_by_component: list[np.ndarray] = []
    capture_receipt_ids: dict[str, str] = {}
    for component_id in component_ids:
        samples = validated_components[component_id]
        category_means = np.stack(
            [samples[category].mean(axis=0, dtype=np.float64) for category in category_names]
        ).astype(np.float32)
        category_effects.append(category_means)
        global_effects.append(
            np.concatenate(
                [samples[category].astype(np.float64) for category in category_names], axis=0
            ).mean(axis=0).astype(np.float32)
        )
        component_blocks = np.zeros(
            (len(category_names), block_count, countsketch_dimension), dtype=np.float64
        )
        component_conditions = np.zeros(
            (len(condition_cells), countsketch_dimension), dtype=np.float64
        )
        component_templates = np.zeros(
            (len(template_cells), countsketch_dimension), dtype=np.float64
        )
        for category_index, category in enumerate(category_names):
            sketches = countsketch(
                samples[category], dimension=countsketch_dimension, seed=countsketch_seed
            )
            for row, sketch in zip(normalized_rows[category], sketches, strict=True):
                component_blocks[
                    category_index, pair_blocks[row["pair_id"]]
                ] += sketch
                component_conditions[
                    condition_index[(category, row["condition"])]
                ] += sketch
                component_templates[
                    template_index[(category, row["template_group"])]
                ] += sketch
        block_sums_by_component.append(component_blocks.astype(np.float32))
        condition_sums_by_component.append(component_conditions.astype(np.float32))
        template_sums_by_component.append(component_templates.astype(np.float32))
        capture_receipt_ids[component_id] = component_results[component_id]["receipt"][
            "canonical_category_effect_capture_id"
        ]
    arrays = {
        "component_ids": np.asarray(
            component_ids, dtype=f"<U{max(len(value) for value in component_ids)}"
        ),
        "delta_h": np.stack(global_effects).astype(np.float32),
        "category_names": np.asarray(
            category_names, dtype=f"<U{max(len(value) for value in category_names)}"
        ),
        "category_episode_counts": np.asarray(category_counts, dtype=np.int64),
        "delta_h_by_category": np.stack(category_effects).astype(np.float32),
        "category_uncertainty_block_counts": block_counts,
        "delta_h_countsketch_block_sums": np.stack(block_sums_by_component).astype(
            np.float32
        ),
        "uncertainty_block_count": np.asarray(block_count, dtype=np.int64),
        "uncertainty_countsketch_dimension": np.asarray(
            countsketch_dimension, dtype=np.int64
        ),
        "uncertainty_countsketch_seed": np.asarray(countsketch_seed, dtype=np.int64),
        "condition_names": np.asarray(
            condition_ids, dtype=f"<U{max(len(value) for value in condition_ids)}"
        ),
        "condition_category_names": np.asarray(
            [category for category, _ in condition_cells],
            dtype=f"<U{max(len(value) for value in category_names)}",
        ),
        "condition_episode_counts": condition_counts,
        "delta_h_countsketch_condition_sums": np.stack(
            condition_sums_by_component
        ).astype(np.float32),
        "template_cell_names": np.asarray(
            template_ids, dtype=f"<U{max(len(value) for value in template_ids)}"
        ),
        "template_cell_category_names": np.asarray(
            [category for category, _ in template_cells],
            dtype=f"<U{max(len(value) for value in category_names)}",
        ),
        "template_cell_group_names": np.asarray(
            [group for _, group in template_cells],
            dtype=f"<U{max(len(value) for value in template_groups)}",
        ),
        "template_cell_episode_counts": template_counts,
        "delta_h_countsketch_template_cell_sums": np.stack(
            template_sums_by_component
        ).astype(np.float32),
    }
    validate_uncertainty_arrays(
        block_counts=arrays["category_uncertainty_block_counts"],
        block_sums=arrays["delta_h_countsketch_block_sums"],
        component_count=len(component_ids),
        category_count=len(category_names),
        block_count=block_count,
        sketch_dimension=countsketch_dimension,
    )
    validate_condition_robustness_arrays(
        condition_names=arrays["condition_names"],
        condition_category_names=arrays["condition_category_names"],
        condition_counts=arrays["condition_episode_counts"],
        condition_sums=arrays["delta_h_countsketch_condition_sums"],
        component_count=len(component_ids),
        category_names=category_names,
        sketch_dimension=countsketch_dimension,
    )
    validate_template_robustness_arrays(
        cell_names=arrays["template_cell_names"],
        cell_category_names=arrays["template_cell_category_names"],
        cell_group_names=arrays["template_cell_group_names"],
        cell_counts=arrays["template_cell_episode_counts"],
        cell_sums=arrays["delta_h_countsketch_template_cell_sums"],
        component_count=len(component_ids),
        category_names=category_names,
        sketch_dimension=countsketch_dimension,
    )
    receipt_body = {
        "schema": SCHEMA,
        "status": "complete_model_free_successor_worker_assembly",
        "inputs": {
            "dataset_sha256": dataset_digest,
            "split_provenance": verified_split,
            "activation_category_contract_sha256": content_sha256(category_contract()),
            "capture_episode_order_sha256": content_sha256(episode_ids),
            "episode_row_identity_sha256": content_sha256(all_rows),
            "category_episode_ids_sha256": {
                category: content_sha256(
                    [row["episode_id"] for row in normalized_rows[category]]
                )
                for category in category_names
            },
            "canonical_capture_receipt_ids": capture_receipt_ids,
        },
        "policy": {
            "component_identity": "canonical_svd_spectral_block",
            "raw_lora_rank_identity_used": False,
            "category_names_sorted": True,
            "episode_rows_sorted_within_category": True,
            "pair_set_shared_across_categories": True,
            "template_groups_shared_across_categories": True,
            "countsketch_dimension": countsketch_dimension,
            "countsketch_seed": countsketch_seed,
            "pair_block_assignment_sha256": content_sha256(
                dict(sorted(pair_blocks.items()))
            ),
        },
        "summary": {
            "components": len(component_ids),
            "categories": len(category_names),
            "episodes": len(all_rows),
            "pair_blocks": block_count,
            "conditions": len(condition_ids),
            "template_cells": len(template_ids),
            "hidden_size": hidden_size,
        },
        "array_sha256": {name: _array_sha256(array) for name, array in arrays.items()},
        "scientific_boundary": {
            "model_loaded_by_assembler": False,
            "model_inference_performed_by_assembler": False,
            "adapter_accessed_by_assembler": False,
            "weights_mutated_by_assembler": False,
            "capture_results_supplied_by_caller": True,
            "capture_source_execution_provenance_bound_by_receipt_ids": True,
            "execution_authorized": False,
        },
    }
    return arrays, {
        **receipt_body,
        "canonical_effect_shard_assembly_id": content_sha256(receipt_body),
    }


def write_canonical_effect_shard(
    *,
    output_path: Path,
    receipt_path: Path,
    arrays: dict[str, np.ndarray],
    assembly_receipt: dict[str, Any],
) -> dict[str, Any]:
    if output_path.parent.resolve() != receipt_path.parent.resolve():
        raise ValueError("canonical effect shard and receipt must be colocated")
    if output_path.exists() or receipt_path.exists():
        raise ValueError("canonical effect shard output paths must not already exist")
    assembly_body = dict(assembly_receipt)
    assembly_identifier = assembly_body.pop("canonical_effect_shard_assembly_id", "")
    if assembly_identifier != content_sha256(assembly_body):
        raise ValueError("canonical effect shard assembly receipt address differs")
    if sorted(arrays) != sorted(assembly_receipt.get("array_sha256", {})):
        raise ValueError("canonical effect shard arrays differ from assembly receipt")
    if any(
        _array_sha256(arrays[name]) != assembly_receipt["array_sha256"][name]
        for name in arrays
    ):
        raise ValueError("canonical effect shard arrays do not match assembly hashes")
    write_deterministic_npz(output_path, arrays)
    body = {
        "schema": "sch.vpd-canonical-effect-shard-artifact-receipt.v1",
        "status": "complete_model_free_artifact_write",
        "assembly": assembly_receipt,
        "artifact": {
            "path": output_path.name,
            "sha256": sha256_file(output_path),
            "arrays": sorted(arrays),
        },
        "scientific_boundary": assembly_receipt["scientific_boundary"],
    }
    receipt = {**body, "canonical_effect_shard_artifact_id": content_sha256(body)}
    write_json(receipt_path, receipt)
    return receipt


def verify_canonical_effect_shard(receipt_path: Path) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    body = dict(receipt)
    identifier = body.pop("canonical_effect_shard_artifact_id", "")
    if identifier != content_sha256(body):
        raise ValueError("canonical effect shard artifact receipt address differs")
    assembly = receipt["assembly"]
    assembly_body = dict(assembly)
    assembly_identifier = assembly_body.pop("canonical_effect_shard_assembly_id", "")
    if assembly_identifier != content_sha256(assembly_body):
        raise ValueError("canonical effect shard assembly receipt address differs")
    artifact_path = receipt_path.parent / receipt["artifact"]["path"]
    if sha256_file(artifact_path) != receipt["artifact"]["sha256"]:
        raise ValueError("canonical effect shard artifact hash differs")
    with np.load(artifact_path, allow_pickle=False) as archive:
        if sorted(archive.files) != receipt["artifact"]["arrays"]:
            raise ValueError("canonical effect shard array names differ")
        arrays = {name: archive[name] for name in archive.files}
    if any(
        _array_sha256(arrays[name]) != assembly["array_sha256"].get(name)
        for name in arrays
    ):
        raise ValueError("canonical effect shard array content hash differs")
    component_count = len(arrays["component_ids"])
    category_names = [str(value) for value in arrays["category_names"].tolist()]
    block_count = int(arrays["uncertainty_block_count"].item())
    sketch_dimension = int(arrays["uncertainty_countsketch_dimension"].item())
    validate_uncertainty_arrays(
        block_counts=arrays["category_uncertainty_block_counts"],
        block_sums=arrays["delta_h_countsketch_block_sums"],
        component_count=component_count,
        category_count=len(category_names),
        block_count=block_count,
        sketch_dimension=sketch_dimension,
    )
    validate_condition_robustness_arrays(
        condition_names=arrays["condition_names"],
        condition_category_names=arrays["condition_category_names"],
        condition_counts=arrays["condition_episode_counts"],
        condition_sums=arrays["delta_h_countsketch_condition_sums"],
        component_count=component_count,
        category_names=category_names,
        sketch_dimension=sketch_dimension,
    )
    validate_template_robustness_arrays(
        cell_names=arrays["template_cell_names"],
        cell_category_names=arrays["template_cell_category_names"],
        cell_group_names=arrays["template_cell_group_names"],
        cell_counts=arrays["template_cell_episode_counts"],
        cell_sums=arrays["delta_h_countsketch_template_cell_sums"],
        component_count=component_count,
        category_names=category_names,
        sketch_dimension=sketch_dimension,
    )
    return {
        "schema": "sch.vpd-canonical-effect-shard-artifact-verification.v1",
        "status": "pass",
        "canonical_effect_shard_artifact_id": identifier,
        "canonical_effect_shard_assembly_id": assembly_identifier,
        "components": component_count,
        "categories": len(category_names),
        "execution_authorized": False,
    }
