from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.assemble_canonical_vpd_effect_shard_v1 import (
    assemble_canonical_effect_shard,
    verify_canonical_effect_shard,
    write_canonical_effect_shard,
)
from experiments.authz_mechinterp.benchmark_authenticated_lora_factor_bundle_extraction_v1 import (
    _fixture_config,
    _fixture_tensors,
    _write_safetensors,
)
from experiments.authz_mechinterp.benchmark_canonical_lora_peft_runtime_v1 import (
    _FakeModel,
    _FakePeftModule,
    _factor_arrays,
)
from experiments.authz_mechinterp.canonical_vpd_category_effect_capture_v1 import (
    capture_canonical_component_effect_by_category,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.emit_canonical_lora_component_inventory_v1 import (
    build_inventory,
)
from experiments.authz_mechinterp.extract_authenticated_lora_factor_bundle_v1 import (
    extract_factor_bundle,
)


def _attempt(function: Any) -> bool:
    try:
        function()
    except (ValueError, KeyError):
        return True
    return False


def _episodes() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for category, conditions in (
        ("missing_or_untrusted_authority", ("none", "user_pasted_valid")),
        ("valid_authority", ("valid_visible", "valid_visible_variant")),
    ):
        prefix = "missing" if category.startswith("missing") else "valid"
        result[category] = [
            {
                "episode_id": f"{prefix}-{index:02d}",
                "pair_id": f"pair-{index // 2}",
                "activation_category": category,
                "condition": conditions[index % 2],
                "template_group": f"template-{index % 2}",
            }
            for index in range(4)
        ]
    return result


def _split_provenance(
    episodes: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    rows = [row for category in sorted(episodes) for row in episodes[category]]
    return {
        "activation_category_split_schema": "sch.vpd-activation-category-split-manifest.v3",
        "activation_category_split_id": "a" * 64,
        "activation_category_split_manifest_sha256": "b" * 64,
        "activation_category_split_role": "discovery",
        "activation_category_split_episode_id_set_sha256": content_sha256(
            sorted(str(row["episode_id"]) for row in rows)
        ),
        "activation_category_split_pair_id_set_sha256": content_sha256(
            sorted({str(row["pair_id"]) for row in rows})
        ),
        "activation_category_split_template_group_set_sha256": content_sha256(
            sorted({str(row["template_group"]) for row in rows})
        ),
        "activation_category_split_verification_status": "pass",
        "activation_category_split_semantic_disjointness": "pass",
    }


def _build_component_results(
    *,
    factor_output: Path,
    inventory_output: Path,
    inventory: dict[str, Any],
    episodes: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    index = json.loads(
        (factor_output / "factor-index.json").read_text(encoding="utf-8")
    )
    module_row = index["modules"][0]
    canonical_module_row = inventory["artifacts"]["modules"][0]
    artifact_path = inventory_output / canonical_module_row["path"]
    artifact_sha256 = canonical_module_row["sha256"]
    component_rows = [
        json.loads(line)
        for line in (inventory_output / "canonical_components.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    component_rows = [
        row for row in component_rows if row["module"] == module_row["module"]
    ]
    with np.load(factor_output / module_row["path"], allow_pickle=False) as archive:
        source_a = np.asarray(archive["a_factor"], dtype=np.float32)
        source_b = np.asarray(archive["b_factor"], dtype=np.float32)
    resolved_path = "base_model.model." + module_row["module"]
    category_batches = {
        category: [
            np.asarray(
                [
                    [
                        0.25 + 0.1 * row_index,
                        (-0.5 if category.startswith("missing") else 0.5)
                        + 0.05 * row_index,
                        1.0 - 0.1 * row_index,
                    ]
                    for row_index, _ in enumerate(rows)
                ],
                dtype=np.float32,
            )
        ]
        for category, rows in episodes.items()
    }
    results: dict[str, dict[str, Any]] = {}
    for component_row in component_rows:
        module = _FakePeftModule(
            source_a, source_b, scale=float(module_row["scale"])
        )

        def capture_callback(category: str, batch: np.ndarray) -> np.ndarray:
            del category
            a_factor, b_factor = _factor_arrays(module)
            return (
                np.asarray(batch)
                @ (float(module_row["scale"]) * b_factor @ a_factor).T
            ).astype(np.float32)

        results[component_row["component_id"]] = (
            capture_canonical_component_effect_by_category(
                model=_FakeModel([(resolved_path, module)]),
                module_path=module_row["module"],
                adapter_name="default",
                canonical_artifact_path=artifact_path,
                expected_canonical_artifact_sha256=artifact_sha256,
                expected_source_a_factor_sha256=component_row[
                    "source_a_factor_sha256"
                ],
                expected_source_b_factor_sha256=component_row[
                    "source_b_factor_sha256"
                ],
                block_index=int(component_row["canonical_block_index"]),
                category_batches=category_batches,
                capture_batch_callback=capture_callback,
                epsilon=0.05,
                no_grad_context_factory=nullcontext,
                maximum_relative_probe_error=2e-6,
            )
        )
    return results


def build_report() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        weights = root / "adapter_model.safetensors"
        config = root / "adapter_config.json"
        _write_safetensors(weights, _fixture_tensors(), metadata={"format": "pt"})
        write_json(config, _fixture_config())
        factor_output = root / "factor-bundle"
        extract_factor_bundle(
            adapter_weights_path=weights,
            adapter_config_path=config,
            expected_adapter_sha256=sha256_file(weights),
            expected_config_sha256=sha256_file(config),
            output_dir=factor_output,
            synthetic_benchmark_only=True,
        )
        inventory_output = root / "canonical-inventory"
        inventory = build_inventory(
            factor_index_path=factor_output / "factor-index.json",
            output_dir=inventory_output,
        )
        episodes = _episodes()
        component_results = _build_component_results(
            factor_output=factor_output,
            inventory_output=inventory_output,
            inventory=inventory,
            episodes=episodes,
        )
        split = _split_provenance(episodes)
        arrays, assembly = assemble_canonical_effect_shard(
            episodes_by_category=episodes,
            component_results=component_results,
            dataset_sha256="c" * 64,
            split_provenance=split,
            requested_uncertainty_blocks=2,
            countsketch_dimension=256,
            countsketch_seed=1707,
        )
        arrays_repeat, assembly_repeat = assemble_canonical_effect_shard(
            episodes_by_category=copy.deepcopy(episodes),
            component_results=copy.deepcopy(component_results),
            dataset_sha256="c" * 64,
            split_provenance=copy.deepcopy(split),
            requested_uncertainty_blocks=2,
            countsketch_dimension=256,
            countsketch_seed=1707,
        )
        deterministic = assembly == assembly_repeat and all(
            np.array_equal(arrays[name], arrays_repeat[name]) for name in arrays
        )
        artifact_dir = root / "artifact"
        artifact_dir.mkdir()
        output_path = artifact_dir / "canonical_effects.npz"
        receipt_path = artifact_dir / "receipt.json"
        artifact_receipt = write_canonical_effect_shard(
            output_path=output_path,
            receipt_path=receipt_path,
            arrays=arrays,
            assembly_receipt=assembly,
        )
        verification = verify_canonical_effect_shard(receipt_path)
        expected_shapes = {
            "delta_h": [len(component_results), 4],
            "delta_h_by_category": [len(component_results), 2, 4],
            "delta_h_countsketch_block_sums": [len(component_results), 2, 2, 256],
            "delta_h_countsketch_condition_sums": [len(component_results), 4, 256],
            "delta_h_countsketch_template_cell_sums": [
                len(component_results),
                4,
                256,
            ],
        }
        shapes_match = all(
            list(arrays[name].shape) == shape for name, shape in expected_shapes.items()
        )
        counts_match = (
            arrays["category_episode_counts"].tolist() == [4, 4]
            and arrays["category_uncertainty_block_counts"].tolist()
            == [[2, 2], [2, 2]]
            and arrays["condition_episode_counts"].tolist() == [2, 2, 2, 2]
            and arrays["template_cell_episode_counts"].tolist() == [2, 2, 2, 2]
        )

        raw_rank_results = copy.deepcopy(component_results)
        first_id = sorted(raw_rank_results)[0]
        raw_rank_results[first_id.replace(":svdblock", ":rank")] = raw_rank_results.pop(
            first_id
        )
        raw_rank_rejected = _attempt(
            lambda: assemble_canonical_effect_shard(
                episodes_by_category=episodes,
                component_results=raw_rank_results,
                dataset_sha256="c" * 64,
                split_provenance=split,
                requested_uncertainty_blocks=2,
                countsketch_dimension=256,
                countsketch_seed=1707,
            )
        )
        receipt_tamper_results = copy.deepcopy(component_results)
        receipt_tamper_results[sorted(receipt_tamper_results)[0]]["receipt"][
            "canonical_category_effect_capture_id"
        ] = "0" * 64
        capture_receipt_tamper_rejected = _attempt(
            lambda: assemble_canonical_effect_shard(
                episodes_by_category=episodes,
                component_results=receipt_tamper_results,
                dataset_sha256="c" * 64,
                split_provenance=split,
                requested_uncertainty_blocks=2,
                countsketch_dimension=256,
                countsketch_seed=1707,
            )
        )
        row_mismatch_episodes = copy.deepcopy(episodes)
        row_mismatch_episodes["valid_authority"].pop()
        row_count_mismatch_rejected = _attempt(
            lambda: assemble_canonical_effect_shard(
                episodes_by_category=row_mismatch_episodes,
                component_results=component_results,
                dataset_sha256="c" * 64,
                split_provenance=_split_provenance(row_mismatch_episodes),
                requested_uncertainty_blocks=2,
                countsketch_dimension=256,
                countsketch_seed=1707,
            )
        )
        split_tamper = dict(split)
        split_tamper["activation_category_split_episode_id_set_sha256"] = "0" * 64
        split_identity_tamper_rejected = _attempt(
            lambda: assemble_canonical_effect_shard(
                episodes_by_category=episodes,
                component_results=component_results,
                dataset_sha256="c" * 64,
                split_provenance=split_tamper,
                requested_uncertainty_blocks=2,
                countsketch_dimension=256,
                countsketch_seed=1707,
            )
        )
        pair_mismatch_episodes = copy.deepcopy(episodes)
        pair_mismatch_episodes["valid_authority"][0]["pair_id"] = "pair-other"
        pair_set_mismatch_rejected = _attempt(
            lambda: assemble_canonical_effect_shard(
                episodes_by_category=pair_mismatch_episodes,
                component_results=component_results,
                dataset_sha256="c" * 64,
                split_provenance=_split_provenance(pair_mismatch_episodes),
                requested_uncertainty_blocks=2,
                countsketch_dimension=256,
                countsketch_seed=1707,
            )
        )
        template_mismatch_episodes = copy.deepcopy(episodes)
        for row in template_mismatch_episodes["valid_authority"]:
            if row["template_group"] == "template-1":
                row["template_group"] = "template-other"
        template_set_mismatch_rejected = _attempt(
            lambda: assemble_canonical_effect_shard(
                episodes_by_category=template_mismatch_episodes,
                component_results=component_results,
                dataset_sha256="c" * 64,
                split_provenance=_split_provenance(template_mismatch_episodes),
                requested_uncertainty_blocks=2,
                countsketch_dimension=256,
                countsketch_seed=1707,
            )
        )
        duplicate_episodes = copy.deepcopy(episodes)
        duplicate_episodes["valid_authority"][0]["episode_id"] = duplicate_episodes[
            "missing_or_untrusted_authority"
        ][0]["episode_id"]
        duplicate_episode_rejected = _attempt(
            lambda: assemble_canonical_effect_shard(
                episodes_by_category=duplicate_episodes,
                component_results=component_results,
                dataset_sha256="c" * 64,
                split_provenance=_split_provenance(duplicate_episodes),
                requested_uncertainty_blocks=2,
                countsketch_dimension=256,
                countsketch_seed=1707,
            )
        )
        small_sketch_rejected = _attempt(
            lambda: assemble_canonical_effect_shard(
                episodes_by_category=episodes,
                component_results=component_results,
                dataset_sha256="c" * 64,
                split_provenance=split,
                requested_uncertainty_blocks=2,
                countsketch_dimension=32,
                countsketch_seed=1707,
            )
        )
        output_path.write_bytes(output_path.read_bytes() + b"tamper")
        artifact_tamper_rejected = _attempt(
            lambda: verify_canonical_effect_shard(receipt_path)
        )
        passed = (
            len(component_results) >= 2
            and deterministic
            and verification["status"] == "pass"
            and artifact_receipt["scientific_boundary"]["execution_authorized"] is False
            and shapes_match
            and counts_match
            and raw_rank_rejected
            and capture_receipt_tamper_rejected
            and row_count_mismatch_rejected
            and split_identity_tamper_rejected
            and pair_set_mismatch_rejected
            and template_set_mismatch_rejected
            and duplicate_episode_rejected
            and small_sketch_rejected
            and artifact_tamper_rejected
        )
    body = {
        "schema": "sch.vpd-canonical-effect-shard-assembly-benchmark.v1",
        "status": "pass" if passed else "fail",
        "source": {
            "runner_sha256": sha256_file(Path(__file__)),
            "assembler_sha256": sha256_file(
                Path(__file__).with_name("assemble_canonical_vpd_effect_shard_v1.py")
            ),
        },
        "design": {
            "authenticated_extraction_chained": True,
            "canonical_category_capture_chained": True,
            "semantic_split_schema": "v3",
            "categories": 2,
            "episodes": 8,
            "pair_blocks": 2,
            "countsketch_dimension": 256,
        },
        "results": {
            "components": len(component_results),
            "deterministic_reassembly": deterministic,
            "written_artifact_verified": verification["status"] == "pass",
            "expected_array_shapes_match": shapes_match,
            "all_support_counts_match": counts_match,
            "raw_rank_component_rejected": raw_rank_rejected,
            "capture_receipt_tamper_rejected": capture_receipt_tamper_rejected,
            "capture_row_count_mismatch_rejected": row_count_mismatch_rejected,
            "split_identity_tamper_rejected": split_identity_tamper_rejected,
            "category_pair_set_mismatch_rejected": pair_set_mismatch_rejected,
            "category_template_set_mismatch_rejected": template_set_mismatch_rejected,
            "duplicate_episode_rejected": duplicate_episode_rejected,
            "small_countsketch_rejected": small_sketch_rejected,
            "written_artifact_tamper_rejected": artifact_tamper_rejected,
        },
        "scientific_boundary": {
            "synthetic_inputs_only": True,
            "fake_peft_objects_only": True,
            "model_loaded": False,
            "model_inference_performed": False,
            "real_adapter_accessed": False,
            "real_effect_archive_emitted": False,
            "execution_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = body.pop("benchmark_id", "")
    if identifier != content_sha256(body):
        raise ValueError("canonical effect assembly benchmark address mismatch")
    if observed != build_report() or observed["status"] != "pass":
        raise ValueError("canonical effect assembly benchmark is not reproducible")
    return {
        "schema": "sch.vpd-canonical-effect-shard-assembly-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "components": observed["results"]["components"],
        "negative_controls": 9,
        "actual_model_execution_verified": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark canonical VPD effect-shard assembly V1."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_benchmark(args.verify)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_report()
        write_json(args.output, result)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "benchmark_id": result["benchmark_id"],
                "components": result.get("results", {}).get(
                    "components", result.get("components")
                ),
                "execution_authorized": result.get("execution_authorized", False),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
