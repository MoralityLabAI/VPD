from __future__ import annotations

import argparse
import base64
import copy
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.authz_mechinterp.activation_categories import (
    categorize_episodes,
    category_contract,
)
from experiments.authz_mechinterp.attest_partition_audit_effect_receipt import (
    attest_partition_audit_receipt,
    build_dataset_attestation,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    build_bundle,
    content_sha256,
    load_hierarchy_plan,
    load_jsonl,
    sha256_file,
    stable_hash,
    verify_bundle,
    write_json,
)
from experiments.authz_mechinterp.compute_vpd_effect_shard import (
    pair_uncertainty_blocks,
)
from experiments.authz_mechinterp.derive_vpd_partition_audit_gate_plan_v2 import (
    build_gate_plan_v2,
)
from experiments.authz_mechinterp.effect_uncertainty import (
    condition_robustness_contract,
    countsketch,
    template_robustness_contract,
    uncertainty_contract,
)
from experiments.authz_mechinterp.evaluate_vpd_partition_audit import (
    build_partition_audit_report,
    verify_report,
)
from experiments.authz_mechinterp.finalize_partition_audit_pack_v1 import (
    REQUIRED_ARTIFACTS,
    build_partition_audit_manifest,
    verify_partition_audit_manifest,
)
from experiments.authz_mechinterp.merge_vpd_effect_shards import (
    write_deterministic_npz,
)
from experiments.authz_mechinterp.merge_vpd_partition_audit_effect_shards import (
    merge_partition_audit_effect_shards,
    verify_capture_packet_bundle,
    verify_partition_audit_effect_merge,
)
from experiments.authz_mechinterp.merge_vpd_sae_results import (
    crypto_types,
    sign_result,
)
from experiments.authz_mechinterp.plan_vpd_partition_audit_gate import (
    build_gate_plan,
)


SCHEMA = "sch.vpd-partition-audit-capture-pipeline-benchmark.v1"
SKETCH_DIMENSION = 256
SKETCH_SEED = int(
    stable_hash(1707, "category-effect-countsketch", "layer_35__proposal_start"), 16
) % (2**32)


def _component(component_id: str, layer: int, family: str, norm: float) -> dict[str, Any]:
    return {
        "component_id": component_id,
        "module": f"model.layers.{layer}.{family}",
        "layer": layer,
        "rank_index": int(component_id.rsplit("rank", 1)[1]),
        "projection_family": family,
        "component_fro_norm": norm,
        "fraction_of_module_delta_fro_sq": norm,
    }


def _rows() -> list[dict[str, Any]]:
    return [
        _component("model.layers.0.self_attn.q_proj:rank0", 0, "self_attn.q_proj", 0.10),
        _component("model.layers.1.self_attn.q_proj:rank1", 1, "self_attn.q_proj", 0.11),
        _component("model.layers.2.self_attn.q_proj:rank2", 2, "self_attn.q_proj", 0.09),
        _component("model.layers.9.mlp.down_proj:rank3", 9, "mlp.down_proj", 0.20),
        _component("model.layers.10.mlp.down_proj:rank4", 10, "mlp.down_proj", 0.21),
        _component("model.layers.11.mlp.down_proj:rank5", 11, "mlp.down_proj", 0.19),
    ]


def _write_hierarchy(path: Path) -> list[dict[str, Any]]:
    source = _rows()
    groups = []
    for group_id, members in (
        ("fixture-q", source[:3]),
        ("fixture-mlp", source[3:]),
    ):
        groups.append(
            {
                "group_id": group_id,
                "tier": "family_x_depth",
                "projection_family": str(members[0]["projection_family"]),
                "depth_band": 0 if group_id == "fixture-q" else 1,
                "target_component_ids": [row["component_id"] for row in members],
                "matches": [
                    {
                        "target_component_id": row["component_id"],
                        "target_fro_norm": row["component_fro_norm"],
                    }
                    for row in members
                ],
            }
        )
    write_json(
        path,
        {
            "model_id": "model-free/fixture",
            "revision": "fixture-revision",
            "hierarchy": {"cell_target_partition_component_count": len(source)},
            "groups": groups,
        },
    )
    return load_hierarchy_plan(path)


def _copy_small_pack(source: Path, target: Path, *, component_count: int) -> None:
    target.mkdir(parents=True)
    for relative in REQUIRED_ARTIFACTS.values():
        shutil.copy2(source / relative, target / relative)
    source_manifest = json.loads(
        (source / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    manifest = build_partition_audit_manifest(
        target,
        generation_seed=int(source_manifest["generation"]["seed"]),
        pairs_per_family=int(source_manifest["generation"]["pairs_per_family"]),
        component_count=component_count,
        batch_size=4,
    )
    write_json(target / "artifact_manifest.json", manifest)
    verify_partition_audit_manifest(target)


def _category_matrix(
    component_ids: list[str], category_names: list[str], *, scale: float
) -> np.ndarray:
    hidden = 4096
    result = np.zeros((len(component_ids), len(category_names), hidden), dtype=np.float32)
    missing = category_names.index("missing_or_untrusted_authority")
    replayed = category_names.index("replayed")
    for row_index in range(len(component_ids)):
        target = missing if row_index % 2 == 0 else replayed
        for category_index in range(len(category_names)):
            result[row_index, category_index, 8 + category_index] = 0.02 * scale
        direction = 0 if target == missing else 1
        result[row_index, target, 8 + target] = 0.0
        result[row_index, target, direction] = (2.0 + 0.05 * row_index) * scale
    return result


def _effect_payload(
    *,
    component_ids: list[str],
    category_delta: np.ndarray,
    episodes: list[dict[str, Any]],
    condition_mutator: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    category_rows = categorize_episodes(episodes)
    category_names = sorted(category_rows)
    if category_delta.shape[:2] != (len(component_ids), len(category_names)):
        raise ValueError("benchmark category matrix does not align")
    pair_blocks, block_count = pair_uncertainty_blocks(
        category_rows, requested_blocks=8, seed=1707
    )
    category_counts = np.asarray(
        [len(category_rows[name]) for name in category_names], dtype=np.int64
    )
    block_counts = np.zeros((len(category_names), block_count), dtype=np.int64)
    for category_index, category in enumerate(category_names):
        for row in category_rows[category]:
            block_counts[category_index, pair_blocks[str(row["pair_id"])]] += 1

    condition_cells = sorted(
        {
            (category, str(row["condition"]))
            for category in category_names
            for row in category_rows[category]
        }
    )
    condition_names = [f"{category}::{condition}" for category, condition in condition_cells]
    condition_counts = np.asarray(
        [
            sum(str(row["condition"]) == condition for row in category_rows[category])
            for category, condition in condition_cells
        ],
        dtype=np.int64,
    )
    template_cells = sorted(
        {
            (category, str(row["template_group"]))
            for category in category_names
            for row in category_rows[category]
        }
    )
    template_names = [f"{category}::{group}" for category, group in template_cells]
    template_counts = np.asarray(
        [
            sum(str(row["template_group"]) == group for row in category_rows[category])
            for category, group in template_cells
        ],
        dtype=np.int64,
    )
    sketches = countsketch(
        category_delta, dimension=SKETCH_DIMENSION, seed=SKETCH_SEED
    ).astype(np.float32)
    block_sums = (
        sketches[:, :, None, :] * block_counts[None, :, :, None]
    ).astype(np.float32)
    category_index = {name: index for index, name in enumerate(category_names)}
    condition_sums = np.stack(
        [
            sketches[:, category_index[category], :] * int(condition_counts[index])
            for index, (category, _) in enumerate(condition_cells)
        ],
        axis=1,
    ).astype(np.float32)
    template_sums = np.stack(
        [
            sketches[:, category_index[category], :] * int(template_counts[index])
            for index, (category, _) in enumerate(template_cells)
        ],
        axis=1,
    ).astype(np.float32)
    mutable = {
        "category_names": category_names,
        "condition_cells": condition_cells,
        "condition_counts": condition_counts,
        "condition_sums": condition_sums,
        "template_cells": template_cells,
        "template_counts": template_counts,
        "template_sums": template_sums,
    }
    if condition_mutator is not None:
        condition_mutator(mutable)
    arrays = {
        "component_ids": np.asarray(component_ids),
        "delta_h": np.average(category_delta, axis=1, weights=category_counts).astype(
            np.float32
        ),
        "category_names": np.asarray(category_names),
        "category_episode_counts": category_counts,
        "delta_h_by_category": category_delta.astype(np.float32),
        "category_uncertainty_block_counts": block_counts,
        "delta_h_countsketch_block_sums": block_sums,
        "uncertainty_block_count": np.asarray(block_count, dtype=np.int64),
        "uncertainty_countsketch_dimension": np.asarray(
            SKETCH_DIMENSION, dtype=np.int64
        ),
        "uncertainty_countsketch_seed": np.asarray(SKETCH_SEED, dtype=np.int64),
        "condition_names": np.asarray(condition_names),
        "condition_category_names": np.asarray(
            [category for category, _ in condition_cells]
        ),
        "condition_episode_counts": condition_counts,
        "delta_h_countsketch_condition_sums": mutable["condition_sums"],
        "template_cell_names": np.asarray(template_names),
        "template_cell_category_names": np.asarray(
            [category for category, _ in template_cells]
        ),
        "template_cell_group_names": np.asarray(
            [group for _, group in template_cells]
        ),
        "template_cell_episode_counts": template_counts,
        "delta_h_countsketch_template_cell_sums": mutable["template_sums"],
    }
    metadata = {
        "block_count": block_count,
        "pair_block_assignment_sha256": content_sha256(dict(sorted(pair_blocks.items()))),
        "condition_names": condition_names,
        "condition_categories": [category for category, _ in condition_cells],
        "condition_counts": {
            name: int(condition_counts[index]) for index, name in enumerate(condition_names)
        },
        "template_names": template_names,
        "template_categories": [category for category, _ in template_cells],
        "template_groups": [group for _, group in template_cells],
        "template_counts": {
            name: int(template_counts[index]) for index, name in enumerate(template_names)
        },
    }
    return arrays, metadata


def _write_effect_merge_manifest(
    path: Path,
    effects_path: Path,
    *,
    role: str,
    template_hash: str,
    measurement_identity: dict[str, Any],
) -> None:
    role_digit = "4" if role == "discovery" else "5"
    body = {
        "schema": "sch.vpd-effect-shard-merge.v1",
        "status": "complete",
        "identity": {
            **copy.deepcopy(measurement_identity),
            "activation_category_split_id": "2" * 64,
            "activation_category_split_manifest_sha256": "3" * 64,
            "activation_category_split_role": role,
            "activation_category_split_episode_id_set_sha256": role_digit * 64,
            "activation_category_split_pair_id_set_sha256": (
                "6" * 64 if role == "discovery" else "7" * 64
            ),
            "activation_category_split_verification_status": "pass",
            "activation_category_split_schema": (
                "sch.vpd-activation-category-split-manifest.v2"
            ),
            "activation_category_split_semantic_disjointness": "pass",
            "activation_category_split_template_group_set_sha256": template_hash,
        },
        "artifacts": {
            "activation_effects": {
                "path": effects_path.relative_to(path.parent).as_posix(),
                "sha256": sha256_file(effects_path),
            }
        },
    }
    write_json(path, {**body, "merge_id": content_sha256(body)})


def _measurement_identity(
    *, category_names: list[str], metadata: dict[str, Any]
) -> dict[str, Any]:
    return {
        "model_id": "model-free/fixture",
        "revision": "fixture-revision",
        "adapter_sha256": "a" * 64,
        "runner_sha256": sha256_file(
            Path(__file__).with_name("compute_vpd_effect_shard.py")
        ),
        "effect_space_id": "layer_35__proposal_start",
        "phase": "pre_gate",
        "estimator": "mean_central_difference_lora_rank_multiplier",
        "perturbation_epsilon": 0.1,
        "selection_seed": 1707,
        "generated_tokens": 0,
        "activation_category_contract_sha256": content_sha256(category_contract()),
        "activation_categories": category_names,
        "activation_category_uncertainty": uncertainty_contract(
            block_count=int(metadata["block_count"]),
            sketch_dimension=SKETCH_DIMENSION,
            seed=SKETCH_SEED,
        ),
        "activation_category_condition_robustness": condition_robustness_contract(
            minimum_coverage=0.75,
            minimum_direction_cosine=0.5,
            minimum_relative_norm=0.25,
        ),
        "activation_category_template_robustness": template_robustness_contract(
            minimum_coverage=0.75,
            minimum_direction_cosine=0.5,
            minimum_relative_norm=0.25,
        ),
    }


def _public_key_base64(key: object) -> str:
    _, _, serialization, _ = crypto_types()
    raw = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode("ascii")


def _build_base_receipt(
    *,
    packet: dict[str, Any],
    packet_sha256: str,
    shard_path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    dataset: dict[str, Any],
    worker_id: str,
    run_id: str,
    identity_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    component_ids = [str(value) for value in arrays["component_ids"].tolist()]
    receipt = {
        "schema": "sch.vpd-effect-shard-receipt.v5",
        "packet_id": packet["packet_id"],
        "cluster_id": packet["cluster_id"],
        "worker_id": worker_id,
        "run_id": run_id,
        "inputs": {
            "packet_sha256": packet_sha256,
            "model_id": "model-free/fixture",
            "revision": "fixture-revision",
            "adapter_sha256": "a" * 64,
            "dataset_sha256": dataset["dataset_sha256"],
            "runner_sha256": sha256_file(
                Path(__file__).with_name("compute_vpd_effect_shard.py")
            ),
            "effect_space_id": packet["effect_refinement"]["effect_space_id"],
            "phase": "pre_gate",
            "estimator": "mean_central_difference_lora_rank_multiplier",
            "perturbation_epsilon": 0.1,
            "selection_seed": 1707,
            "episode_order_sha256": dataset["episode_order_sha256"],
            "episodes": dataset["episodes"],
            "generated_tokens": 0,
            "activation_category_contract_sha256": dataset[
                "activation_category_contract_sha256"
            ],
            "activation_categories": dataset["activation_categories"],
            "activation_category_episode_counts": dataset[
                "activation_category_episode_counts"
            ],
            "activation_category_episode_ids_sha256": dataset[
                "activation_category_episode_ids_sha256"
            ],
            "activation_category_uncertainty": uncertainty_contract(
                block_count=int(metadata["block_count"]),
                sketch_dimension=SKETCH_DIMENSION,
                seed=SKETCH_SEED,
            ),
            "activation_category_pair_block_assignment_sha256": metadata[
                "pair_block_assignment_sha256"
            ],
            "activation_category_condition_robustness": condition_robustness_contract(
                minimum_coverage=0.75,
                minimum_direction_cosine=0.5,
                minimum_relative_norm=0.25,
            ),
            "activation_category_condition_names": metadata["condition_names"],
            "activation_category_condition_categories": metadata[
                "condition_categories"
            ],
            "activation_category_condition_episode_counts": metadata[
                "condition_counts"
            ],
            "activation_category_condition_set_sha256": content_sha256(
                metadata["condition_names"]
            ),
            "activation_category_template_robustness": template_robustness_contract(
                minimum_coverage=0.75,
                minimum_direction_cosine=0.5,
                minimum_relative_norm=0.25,
            ),
            "activation_category_template_cell_names": metadata["template_names"],
            "activation_category_template_cell_categories": metadata[
                "template_categories"
            ],
            "activation_category_template_cell_groups": metadata["template_groups"],
            "activation_category_template_cell_episode_counts": metadata[
                "template_counts"
            ],
            "activation_category_template_cell_set_sha256": content_sha256(
                metadata["template_names"]
            ),
        },
        "shard_filename": shard_path.name,
        "shard_sha256": sha256_file(shard_path),
        "component_id_set_sha256": content_sha256(sorted(component_ids)),
        "rows": len(component_ids),
        "hidden_size": int(arrays["delta_h"].shape[1]),
        "activation_category_count": len(dataset["activation_categories"]),
        "dtype": "float32",
        "factor_restoration_verified": True,
        "claim_boundary": "model-free fixture receipt",
    }
    receipt["inputs"].update(copy.deepcopy(identity_overrides or {}))
    return receipt


def _subset_arrays(
    arrays: dict[str, np.ndarray], component_ids: list[str]
) -> dict[str, np.ndarray]:
    source_ids = [str(value) for value in arrays["component_ids"].tolist()]
    indices = [source_ids.index(value) for value in component_ids]
    per_component = {
        "delta_h",
        "delta_h_by_category",
        "delta_h_countsketch_block_sums",
        "delta_h_countsketch_condition_sums",
        "delta_h_countsketch_template_cell_sums",
    }
    result: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if key == "component_ids":
            result[key] = np.asarray(component_ids)
        elif key in per_component:
            result[key] = value[indices]
        else:
            result[key] = value
    return result


def _run_audit_merge(
    *,
    root: Path,
    label: str,
    capture_bundle: Path,
    pack_dir: Path,
    gate_path: Path,
    full_arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    key: object,
    registry_path: Path,
    identity_overrides: dict[str, Any] | None = None,
) -> Path:
    manifest = json.loads((capture_bundle / "manifest.json").read_text(encoding="utf-8"))
    packets = {
        str(entry["packet_id"]): (
            entry,
            json.loads((capture_bundle / entry["packet_path"]).read_text(encoding="utf-8")),
        )
        for entry in manifest["artifacts"]["packets"]
    }
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    dataset = build_dataset_attestation(
        pack_dir=pack_dir, selection_seed=1707, maximum_episodes=0
    )
    shards = root / f"{label}-shards"
    receipts = root / f"{label}-receipts"
    output = root / f"{label}-merge"
    shards.mkdir()
    receipts.mkdir()
    receipt_paths: list[Path] = []
    for index, unit in enumerate(gate["capture_work_units"]):
        entry, packet = packets[str(unit["packet_id"])]
        component_ids = sorted(str(value) for value in packet["component_ids"])
        arrays = _subset_arrays(full_arrays, component_ids)
        shard_path = shards / f"{unit['work_unit_id']}.npz"
        write_deterministic_npz(shard_path, arrays)
        base = _build_base_receipt(
            packet=packet,
            packet_sha256=str(entry["packet_sha256"]),
            shard_path=shard_path,
            arrays=arrays,
            metadata=metadata,
            dataset=dataset,
            worker_id="worker-a",
            run_id=f"model-free-{label}-{index}",
            identity_overrides=identity_overrides,
        )
        base_path = root / f"{label}-{index}-base.json"
        write_json(base_path, base)
        adapted = attest_partition_audit_receipt(
            base_receipt_path=base_path,
            pack_dir=pack_dir,
            gate_plan_path=gate_path,
            work_unit_id=str(unit["work_unit_id"]),
        )
        signed_path = receipts / f"{unit['work_unit_id']}.json"
        write_json(signed_path, sign_result(adapted, key))
        receipt_paths.append(signed_path)
    merge_partition_audit_effect_shards(
        bundle_dir=capture_bundle,
        registry_path=registry_path,
        receipt_paths=receipt_paths,
        shard_dir=shards,
        output_dir=output,
        pack_dir=pack_dir,
        gate_plan_path=gate_path,
        require_complete=True,
    )
    verify_partition_audit_effect_merge(
        output, pack_dir=pack_dir, gate_plan_path=gate_path
    )
    return output


def run_benchmark(project_root: Path) -> dict[str, Any]:
    artifacts = project_root / "artifacts"
    with tempfile.TemporaryDirectory(
        prefix="vpd-partition-audit-benchmark-", dir=project_root
    ) as directory:
        root = Path(directory)
        hierarchy_path = root / "hierarchy.json"
        rows = _write_hierarchy(hierarchy_path)
        pack_dir = root / "pack"
        _copy_small_pack(
            artifacts / "vpd_partition_audit_pack_v1",
            pack_dir,
            component_count=len(rows),
        )
        capture_bundle = root / "capture-bundle"
        capture_manifest = build_bundle(
            rows,
            inventory_sha256=sha256_file(hierarchy_path),
            effects_path=None,
            output_dir=capture_bundle,
            cluster_count=2,
            seed=13,
            projection_dim=16,
            metadata_weight=0.2,
            max_iterations=50,
            anchors=["proposal_start"],
            workers=["worker-a"],
        )
        gate_v1_path = root / "gate-v1.json"
        gate_v1 = build_gate_plan(
            predecessor_protocol_path=project_root
            / "configs"
            / "evals"
            / "vpd_activation_category_compartments_v4.json",
            pack_dir=pack_dir,
            benchmark_path=artifacts
            / "vpd_frozen_partition_audit_benchmark_v1"
            / "report.json",
            bundle_dir=capture_bundle,
        )
        write_json(gate_v1_path, gate_v1)
        gate_v2_path = root / "gate-v2.json"
        gate_v2 = build_gate_plan_v2(
            predecessor_gate_plan_path=gate_v1_path,
            predecessor_protocol_path=project_root
            / "configs"
            / "evals"
            / "vpd_activation_category_compartments_v4.json",
            pack_dir=pack_dir,
            benchmark_path=artifacts
            / "vpd_frozen_partition_audit_benchmark_v1"
            / "report.json",
            metadata_bundle_dir=capture_bundle,
            discovery_campaign_plan_path=artifacts
            / "vpd_effect_capture_campaign_v4"
            / "plan.json",
        )
        write_json(gate_v2_path, gate_v2)

        episodes = load_jsonl(pack_dir / "gated_episodes.jsonl")
        category_names = sorted(categorize_episodes(episodes))
        component_ids = [str(row["component_id"]) for row in rows]
        discovery_matrix = _category_matrix(component_ids, category_names, scale=1.0)
        calibration_matrix = _category_matrix(component_ids, category_names, scale=0.9)
        discovery_arrays, discovery_metadata = _effect_payload(
            component_ids=component_ids,
            category_delta=discovery_matrix,
            episodes=episodes,
        )
        calibration_arrays, _ = _effect_payload(
            component_ids=component_ids,
            category_delta=calibration_matrix,
            episodes=episodes,
        )
        discovery_path = root / "discovery-effects.npz"
        calibration_path = root / "calibration-effects.npz"
        write_deterministic_npz(discovery_path, discovery_arrays)
        write_deterministic_npz(calibration_path, calibration_arrays)
        discovery_merge_path = root / "discovery-merge.json"
        calibration_merge_path = root / "calibration-merge.json"
        _write_effect_merge_manifest(
            discovery_merge_path,
            discovery_path,
            role="discovery",
            template_hash="8" * 64,
            measurement_identity=_measurement_identity(
                category_names=category_names, metadata=discovery_metadata
            ),
        )
        _write_effect_merge_manifest(
            calibration_merge_path,
            calibration_path,
            role="calibration",
            template_hash="9" * 64,
            measurement_identity=_measurement_identity(
                category_names=category_names, metadata=discovery_metadata
            ),
        )
        category_bundle = root / "category-bundle"
        category_manifest = build_bundle(
            rows,
            inventory_sha256=sha256_file(hierarchy_path),
            effects_path=discovery_path,
            calibration_effects_path=calibration_path,
            effects_merge_manifest_path=discovery_merge_path,
            calibration_effects_merge_manifest_path=calibration_merge_path,
            output_dir=category_bundle,
            cluster_count=2,
            seed=13,
            projection_dim=16,
            metadata_weight=0.0,
            max_iterations=50,
            anchors=["proposal_start"],
            workers=["worker-a"],
            minimum_stable_specific_components=1,
            minimum_category_calibration_episodes=1,
            require_category_uncertainty=True,
            minimum_category_jackknife_stability=0.75,
            require_category_pair_block_direction_robustness=True,
            minimum_category_pair_block_direction_coverage=0.75,
            require_category_condition_robustness=True,
            minimum_category_condition_coverage=0.75,
            require_category_template_robustness=True,
            minimum_category_template_coverage=0.75,
            minimum_category_robustness_sketch_dimension=SKETCH_DIMENSION,
        )
        verify_bundle(category_bundle)
        capture_packets = {
            str(entry["packet_id"]) for entry in capture_manifest["artifacts"]["packets"]
        }
        category_packets = {
            str(entry["packet_id"]) for entry in category_manifest["artifacts"]["packets"]
        }
        packetizations_differ = capture_packets != category_packets
        if not packetizations_differ:
            raise AssertionError("benchmark did not exercise distinct packetizations")

        Ed25519PrivateKey, _, _, _ = crypto_types()
        key = Ed25519PrivateKey.from_private_bytes(bytes(range(1, 33)))
        registry_path = root / "registry.json"
        write_json(
            registry_path,
            {
                "schema": "sch.vpd-sae-worker-registry.v1",
                "workers": [
                    {
                        "worker_id": "worker-a",
                        "algorithm": "ed25519",
                        "public_key_base64": _public_key_base64(key),
                        "enabled": True,
                    }
                ],
            },
        )

        clean_arrays, clean_metadata = _effect_payload(
            component_ids=component_ids,
            category_delta=_category_matrix(component_ids, category_names, scale=0.95),
            episodes=episodes,
        )
        clean_merge = _run_audit_merge(
            root=root,
            label="clean",
            capture_bundle=capture_bundle,
            pack_dir=pack_dir,
            gate_path=gate_v2_path,
            full_arrays=clean_arrays,
            metadata=clean_metadata,
            key=key,
            registry_path=registry_path,
        )
        clean_report = build_partition_audit_report(
            bundle_dir=category_bundle,
            hierarchy_plan_path=hierarchy_path,
            discovery_effects_path=discovery_path,
            discovery_merge_manifest_path=discovery_merge_path,
            audit_merge_dir=clean_merge,
            pack_dir=pack_dir,
            gate_plan_path=gate_v2_path,
        )
        clean_report_path = root / "clean-report.json"
        write_json(clean_report_path, clean_report)
        verify_report(
            clean_report_path,
            bundle_dir=category_bundle,
            hierarchy_plan_path=hierarchy_path,
            discovery_effects_path=discovery_path,
            discovery_merge_manifest_path=discovery_merge_path,
            audit_merge_dir=clean_merge,
            pack_dir=pack_dir,
            gate_plan_path=gate_v2_path,
        )

        mismatch_matrix = _category_matrix(component_ids, category_names, scale=0.95)
        missing_index = category_names.index("missing_or_untrusted_authority")
        replayed_index = category_names.index("replayed")
        mismatch_matrix[0, [missing_index, replayed_index]] = mismatch_matrix[
            0, [replayed_index, missing_index]
        ]
        mismatch_arrays, mismatch_metadata = _effect_payload(
            component_ids=component_ids,
            category_delta=mismatch_matrix,
            episodes=episodes,
        )
        mismatch_merge = _run_audit_merge(
            root=root,
            label="mismatch",
            capture_bundle=capture_bundle,
            pack_dir=pack_dir,
            gate_path=gate_v2_path,
            full_arrays=mismatch_arrays,
            metadata=mismatch_metadata,
            key=key,
            registry_path=registry_path,
        )
        mismatch_report = build_partition_audit_report(
            bundle_dir=category_bundle,
            hierarchy_plan_path=hierarchy_path,
            discovery_effects_path=discovery_path,
            discovery_merge_manifest_path=discovery_merge_path,
            audit_merge_dir=mismatch_merge,
            pack_dir=pack_dir,
            gate_plan_path=gate_v2_path,
        )

        def make_condition_failure(payload: dict[str, Any]) -> None:
            cells = payload["condition_cells"]
            counts = payload["condition_counts"]
            sums = payload["condition_sums"]
            target_indices = [
                index
                for index, (category, _) in enumerate(cells)
                if category == "missing_or_untrusted_authority"
            ]
            total = int(sum(int(counts[index]) for index in target_indices))
            aggregate = sums[0, target_indices, :].sum(axis=0)
            sums[0, target_indices, :] = 0.0
            sums[0, target_indices[0], :] = aggregate
            if total <= 0 or len(target_indices) < 2:
                raise AssertionError("benchmark condition support is insufficient")

        robustness_arrays, robustness_metadata = _effect_payload(
            component_ids=component_ids,
            category_delta=_category_matrix(component_ids, category_names, scale=0.95),
            episodes=episodes,
            condition_mutator=make_condition_failure,
        )
        robustness_merge = _run_audit_merge(
            root=root,
            label="robustness",
            capture_bundle=capture_bundle,
            pack_dir=pack_dir,
            gate_path=gate_v2_path,
            full_arrays=robustness_arrays,
            metadata=robustness_metadata,
            key=key,
            registry_path=registry_path,
        )
        robustness_report = build_partition_audit_report(
            bundle_dir=category_bundle,
            hierarchy_plan_path=hierarchy_path,
            discovery_effects_path=discovery_path,
            discovery_merge_manifest_path=discovery_merge_path,
            audit_merge_dir=robustness_merge,
            pack_dir=pack_dir,
            gate_plan_path=gate_v2_path,
        )

        def identity_scenario_rejected(
            *,
            label: str,
            overrides: dict[str, Any],
            expected_field: str,
            merge_must_reject: bool = False,
        ) -> bool:
            try:
                merge_dir = _run_audit_merge(
                    root=root,
                    label=label,
                    capture_bundle=capture_bundle,
                    pack_dir=pack_dir,
                    gate_path=gate_v2_path,
                    full_arrays=clean_arrays,
                    metadata=clean_metadata,
                    key=key,
                    registry_path=registry_path,
                    identity_overrides=overrides,
                )
            except ValueError as error:
                return merge_must_reject and expected_field in str(error)
            if merge_must_reject:
                return False
            try:
                build_partition_audit_report(
                    bundle_dir=category_bundle,
                    hierarchy_plan_path=hierarchy_path,
                    discovery_effects_path=discovery_path,
                    discovery_merge_manifest_path=discovery_merge_path,
                    audit_merge_dir=merge_dir,
                    pack_dir=pack_dir,
                    gate_plan_path=gate_v2_path,
                )
            except ValueError as error:
                return (
                    "cross-archive measurement identity differs" in str(error)
                    and expected_field in str(error)
                )
            return False

        wrong_model_rejected = identity_scenario_rejected(
            label="wrong-model",
            overrides={"model_id": "model-free/different-fixture"},
            expected_field="model_id",
        )
        wrong_adapter_rejected = identity_scenario_rejected(
            label="wrong-adapter",
            overrides={"adapter_sha256": "b" * 64},
            expected_field="adapter_sha256",
        )
        wrong_effect_space_rejected = identity_scenario_rejected(
            label="wrong-effect-space",
            overrides={"effect_space_id": "layer_34__proposal_start"},
            expected_field="campaign measurement contract",
            merge_must_reject=True,
        )
        wrong_bundle_rejected = False
        try:
            verify_capture_packet_bundle(bundle_dir=category_bundle, plan=gate_v2)
        except ValueError as error:
            wrong_bundle_rejected = "capture packet bundle differs" in str(error)
        if not wrong_bundle_rejected:
            raise AssertionError("wrong capture packet bundle was not rejected")

    scenario_rows = [
        {
            "name": "distinct_capture_and_category_packetizations_clean_pass",
            "passed": (
                packetizations_differ
                and clean_report["status"] == "pass_partition_audit"
                and clean_report["summary"]["specific_failure_count"] == 0
            ),
            "audit_status": clean_report["status"],
            "specific_failures": clean_report["summary"]["specific_failure_count"],
        },
        {
            "name": "fresh_audit_category_mismatch_rejects",
            "passed": (
                mismatch_report["status"] == "reject_partition_audit"
                and mismatch_report["summary"]["specific_failure_reason_counts"].get(
                    "audit_category_mismatch", 0
                )
                >= 1
            ),
            "audit_status": mismatch_report["status"],
            "reason_counts": mismatch_report["summary"][
                "specific_failure_reason_counts"
            ],
        },
        {
            "name": "fresh_audit_condition_robustness_failure_rejects",
            "passed": (
                robustness_report["status"] == "reject_partition_audit"
                and robustness_report["summary"]["specific_failure_reason_counts"].get(
                    "audit_stratified_robustness_failure", 0
                )
                >= 1
            ),
            "audit_status": robustness_report["status"],
            "reason_counts": robustness_report["summary"][
                "specific_failure_reason_counts"
            ],
        },
        {
            "name": "wrong_capture_packet_bundle_rejects",
            "passed": wrong_bundle_rejected,
        },
        {
            "name": "different_model_identity_rejects",
            "passed": wrong_model_rejected,
            "rejection_stage": "production_cross_archive_identity",
        },
        {
            "name": "different_adapter_identity_rejects",
            "passed": wrong_adapter_rejected,
            "rejection_stage": "production_cross_archive_identity",
        },
        {
            "name": "different_effect_space_rejects",
            "passed": wrong_effect_space_rejected,
            "rejection_stage": "pre_signing_campaign_attestation",
        },
    ]
    body = {
        "schema": SCHEMA,
        "status": "pass" if all(row["passed"] for row in scenario_rows) else "fail",
        "scenarios": scenario_rows,
        "summary": {
            "scenarios": len(scenario_rows),
            "passed": sum(bool(row["passed"]) for row in scenario_rows),
            "failed": sum(not bool(row["passed"]) for row in scenario_rows),
            "components": len(_rows()),
            "capture_work_units": 2,
            "signed_receipt_sets_merged": 5,
            "signed_receipt_sets_rejected_before_merge": 1,
            "model_inference_performed": False,
        },
        "contract": {
            "capture_packetization_is_assignment_agnostic": True,
            "frozen_category_bundle_supplies_assignments_and_discovery_lineage": True,
            "same_component_census_required": True,
            "same_packet_ids_required": False,
            "fresh_pack_attestation_before_signature_required": True,
            "exact_signed_work_unit_coverage_required": True,
            "audit_is_reject_only": True,
        },
        "source": {
            "benchmark_runner_sha256": sha256_file(Path(__file__)),
            "capture_adapter_sha256": sha256_file(
                Path(__file__).with_name("attest_partition_audit_effect_receipt.py")
            ),
            "merge_front_end_sha256": sha256_file(
                Path(__file__).with_name("merge_vpd_partition_audit_effect_shards.py")
            ),
            "production_evaluator_sha256": sha256_file(
                Path(__file__).with_name("evaluate_vpd_partition_audit.py")
            ),
            "gate_plan_v2_builder_sha256": sha256_file(
                Path(__file__).with_name("derive_vpd_partition_audit_gate_plan_v2.py")
            ),
        },
        "scientific_boundary": {
            "synthetic_effects_only": True,
            "model_outcomes_used": False,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "empirical_bonsai_category_claimed": False,
            "execution_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = copy.deepcopy(observed)
    benchmark_id = str(body.pop("benchmark_id", ""))
    if content_sha256(body) != benchmark_id:
        raise ValueError("partition-audit capture benchmark content address mismatch")
    expected = run_benchmark(project_root)
    if expected != observed:
        raise ValueError("partition-audit capture benchmark is not reproducible")
    if observed["status"] != "pass":
        raise ValueError("partition-audit capture benchmark does not pass")
    return {
        "schema": "sch.vpd-partition-audit-capture-pipeline-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": benchmark_id,
        "scenarios": observed["summary"]["scenarios"],
        "passed": observed["summary"]["passed"],
        "model_inference_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or verify the model-free signed partition-audit pipeline benchmark."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.verify is not None:
        result = verify_benchmark(args.verify, project_root=project_root)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = run_benchmark(project_root)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
