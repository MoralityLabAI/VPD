from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.authz_mechinterp.cluster_vpd_for_sae import build_bundle, content_sha256
from experiments.authz_mechinterp.merge_vpd_effect_shards import (
    effect_identity,
    merge_effect_shards,
    verify_effect_merge,
    write_deterministic_npz,
)
from experiments.authz_mechinterp.merge_vpd_sae_results import sign_result
from experiments.authz_mechinterp.effect_uncertainty import (
    condition_robustness_contract,
    countsketch,
    template_robustness_contract,
    uncertainty_contract,
)
from tests.test_vpd_sae_clusters import synthetic_rows
from tests.test_vpd_sae_result_merge import public_base64


class VpdEffectShardMergeTests(unittest.TestCase):
    def test_effect_identity_requires_complete_verified_split_provenance(self) -> None:
        inputs = {
            "model_id": "model",
            "revision": "revision",
            "adapter_sha256": "a" * 64,
            "dataset_sha256": "b" * 64,
            "runner_sha256": "c" * 64,
            "effect_space_id": "layer_35__proposal_start",
            "phase": "pre_gate",
            "estimator": "central_difference",
            "perturbation_epsilon": 0.1,
            "selection_seed": 19,
            "episode_order_sha256": "d" * 64,
            "episodes": 8,
            "generated_tokens": 0,
            "activation_category_contract_sha256": "e" * 64,
            "activation_categories": ["replayed", "valid_authority"],
            "activation_category_episode_counts": {
                "replayed": 4,
                "valid_authority": 4,
            },
            "activation_category_episode_ids_sha256": {
                "replayed": "f" * 64,
                "valid_authority": "1" * 64,
            },
            "activation_category_split_id": "2" * 64,
            "activation_category_split_manifest_sha256": "3" * 64,
            "activation_category_split_role": "discovery",
            "activation_category_split_episode_id_set_sha256": "4" * 64,
            "activation_category_split_pair_id_set_sha256": "5" * 64,
            "activation_category_split_verification_status": "pass",
        }
        identity = effect_identity(
            {"schema": "sch.vpd-effect-shard-receipt.v2", "inputs": inputs}
        )
        self.assertEqual(identity["activation_category_split_role"], "discovery")
        incomplete = dict(inputs)
        incomplete.pop("activation_category_split_pair_id_set_sha256")
        with self.assertRaisesRegex(ValueError, "split provenance is incomplete"):
            effect_identity(
                {"schema": "sch.vpd-effect-shard-receipt.v2", "inputs": incomplete}
            )

    def prepare(
        self,
        root: Path,
        *,
        with_categories: bool = False,
        with_uncertainty: bool = False,
        with_conditions: bool = False,
        with_templates: bool = False,
    ) -> tuple[Path, Path, Path, list[Path], dict]:
        if with_conditions and not with_uncertainty:
            raise ValueError("condition fixture requires uncertainty")
        if with_templates and not with_conditions:
            raise ValueError("template fixture requires condition robustness")
        rows = synthetic_rows()
        bundle = root / "bundle"
        manifest = build_bundle(
            rows,
            inventory_sha256=content_sha256(rows),
            effects_path=None,
            output_dir=bundle,
            cluster_count=2,
            seed=13,
            projection_dim=3,
            metadata_weight=0.2,
            max_iterations=50,
            anchors=["proposal_start"],
            workers=["worker-a", "worker-b"],
        )
        packet_entries = {entry["packet_id"]: entry for entry in manifest["artifacts"]["packets"]}
        packets = [
            json.loads((bundle / entry["packet_path"]).read_text())
            for entry in manifest["artifacts"]["packets"]
        ]
        worker_ids = {packet["distribution"]["assigned_worker"] for packet in packets}
        keys = {worker_id: Ed25519PrivateKey.generate() for worker_id in worker_ids}
        registry_path = root / "registry.json"
        registry_path.write_text(
            json.dumps(
                {
                    "schema": "sch.vpd-sae-worker-registry.v1",
                    "workers": [
                        {
                            "worker_id": worker_id,
                            "algorithm": "ed25519",
                            "public_key_base64": public_base64(key),
                            "enabled": True,
                        }
                        for worker_id, key in sorted(keys.items())
                    ],
                }
            ),
            encoding="utf-8",
        )
        shard_dir = root / "shards"
        shard_dir.mkdir()
        receipt_paths = []
        vectors = {
            row["component_id"]: np.asarray([float(index), float(index + 1), 1.0], dtype=np.float32)
            for index, row in enumerate(sorted(rows, key=lambda row: row["component_id"]))
        }
        for index, packet in enumerate(packets):
            component_ids = sorted(packet["component_ids"])
            shard_path = shard_dir / f"shard-{index}.npz"
            delta_h = np.stack([vectors[value] for value in component_ids])
            arrays = {
                "component_ids": np.asarray(component_ids),
                "delta_h": delta_h,
            }
            if with_categories:
                category_delta = []
                for component_id, vector in zip(component_ids, delta_h, strict=True):
                    rank = int(component_id.rsplit("rank", 1)[1])
                    multipliers = (2.0, 0.5) if rank % 2 == 0 else (0.5, 2.0)
                    category_delta.append(
                        np.stack([vector * multipliers[0], vector * multipliers[1]])
                    )
                category_matrix = np.stack(category_delta).astype(np.float32)
                arrays.update(
                    {
                        "category_names": np.asarray(
                            ["replayed", "valid_authority"]
                        ),
                        "category_episode_counts": np.asarray(
                            [1, 1], dtype=np.int64
                        ),
                        "delta_h_by_category": category_matrix,
                        "delta_h": category_matrix.mean(axis=1).astype(np.float32),
                    }
                )
                if with_uncertainty:
                    block_counts = np.ones((2, 2), dtype=np.int64)
                    sketches = countsketch(
                        category_matrix, dimension=8, seed=23
                    ).astype(np.float32)
                    block_sums = np.repeat(sketches[:, :, None, :], 2, axis=2)
                    arrays.update(
                        {
                            "category_episode_counts": np.asarray(
                                [2, 2], dtype=np.int64
                            ),
                            "category_uncertainty_block_counts": block_counts,
                            "delta_h_countsketch_block_sums": block_sums,
                            "uncertainty_block_count": np.asarray(2, dtype=np.int64),
                            "uncertainty_countsketch_dimension": np.asarray(
                                8, dtype=np.int64
                            ),
                            "uncertainty_countsketch_seed": np.asarray(
                                23, dtype=np.int64
                            ),
                        }
                    )
                    if with_conditions:
                        condition_vectors = np.stack(
                            [
                                category_matrix[:, 0],
                                category_matrix[:, 0],
                                category_matrix[:, 1],
                                category_matrix[:, 1],
                            ],
                            axis=1,
                        )
                        arrays.update(
                            {
                                "condition_names": np.asarray(
                                    [
                                        "replayed::replay-a",
                                        "replayed::replay-b",
                                        "valid_authority::valid-a",
                                        "valid_authority::valid-b",
                                    ]
                                ),
                                "condition_category_names": np.asarray(
                                    [
                                        "replayed",
                                        "replayed",
                                        "valid_authority",
                                        "valid_authority",
                                    ]
                                ),
                                "condition_episode_counts": np.ones(
                                    4, dtype=np.int64
                                ),
                                "delta_h_countsketch_condition_sums": countsketch(
                                    condition_vectors, dimension=8, seed=23
                                ).astype(np.float32),
                            }
                        )
                        if with_templates:
                            template_vectors = np.stack(
                                [
                                    category_matrix[:, 0],
                                    category_matrix[:, 0],
                                    category_matrix[:, 1],
                                    category_matrix[:, 1],
                                ],
                                axis=1,
                            )
                            arrays.update(
                                {
                                    "template_cell_names": np.asarray(
                                        [
                                            "replayed::template-a",
                                            "replayed::template-b",
                                            "valid_authority::template-a",
                                            "valid_authority::template-b",
                                        ]
                                    ),
                                    "template_cell_category_names": np.asarray(
                                        [
                                            "replayed",
                                            "replayed",
                                            "valid_authority",
                                            "valid_authority",
                                        ]
                                    ),
                                    "template_cell_group_names": np.asarray(
                                        [
                                            "template-a",
                                            "template-b",
                                            "template-a",
                                            "template-b",
                                        ]
                                    ),
                                    "template_cell_episode_counts": np.ones(
                                        4, dtype=np.int64
                                    ),
                                    "delta_h_countsketch_template_cell_sums": countsketch(
                                        template_vectors, dimension=8, seed=23
                                    ).astype(np.float32),
                                }
                            )
            write_deterministic_npz(
                shard_path,
                arrays,
            )
            worker = packet["distribution"]["assigned_worker"]
            receipt = {
                "schema": (
                    "sch.vpd-effect-shard-receipt.v5"
                    if with_templates
                    else
                    "sch.vpd-effect-shard-receipt.v4"
                    if with_conditions
                    else "sch.vpd-effect-shard-receipt.v3"
                    if with_uncertainty
                    else "sch.vpd-effect-shard-receipt.v2"
                    if with_categories
                    else "sch.vpd-effect-shard-receipt.v1"
                ),
                "packet_id": packet["packet_id"],
                "cluster_id": packet["cluster_id"],
                "worker_id": worker,
                "run_id": f"effects-{index}",
                "inputs": {
                    "packet_sha256": packet_entries[packet["packet_id"]]["packet_sha256"],
                    "model_id": "example/model",
                    "revision": "revision-1",
                    "adapter_sha256": "a" * 64,
                    "dataset_sha256": "b" * 64,
                    "runner_sha256": "c" * 64,
                    "effect_space_id": packet["effect_refinement"]["effect_space_id"],
                    "phase": "pre_gate",
                    "estimator": "mean_central_difference_lora_rank_multiplier",
                    "perturbation_epsilon": 0.1,
                    "selection_seed": 1707,
                    "episode_order_sha256": "d" * 64,
                    "episodes": 16,
                    "generated_tokens": 0,
                    **(
                        {
                            "activation_category_contract_sha256": "e" * 64,
                            "activation_categories": ["replayed", "valid_authority"],
                            "activation_category_episode_counts": {
                                "replayed": 2 if with_uncertainty else 1,
                                "valid_authority": 2 if with_uncertainty else 1,
                            },
                            **(
                                {
                                    "activation_category_uncertainty": uncertainty_contract(
                                        block_count=2,
                                        sketch_dimension=8,
                                        seed=23,
                                    ),
                                    "activation_category_pair_block_assignment_sha256": (
                                        "8" * 64
                                    ),
                                }
                                if with_uncertainty
                                else {}
                            ),
                            **(
                                {
                                    "activation_category_template_robustness": template_robustness_contract(
                                        minimum_coverage=0.75,
                                        minimum_direction_cosine=0.50,
                                        minimum_relative_norm=0.25,
                                    ),
                                    "activation_category_template_cell_names": [
                                        "replayed::template-a",
                                        "replayed::template-b",
                                        "valid_authority::template-a",
                                        "valid_authority::template-b",
                                    ],
                                    "activation_category_template_cell_categories": [
                                        "replayed",
                                        "replayed",
                                        "valid_authority",
                                        "valid_authority",
                                    ],
                                    "activation_category_template_cell_groups": [
                                        "template-a",
                                        "template-b",
                                        "template-a",
                                        "template-b",
                                    ],
                                    "activation_category_template_cell_episode_counts": {
                                        "replayed::template-a": 1,
                                        "replayed::template-b": 1,
                                        "valid_authority::template-a": 1,
                                        "valid_authority::template-b": 1,
                                    },
                                    "activation_category_template_cell_set_sha256": content_sha256(
                                        [
                                            "replayed::template-a",
                                            "replayed::template-b",
                                            "valid_authority::template-a",
                                            "valid_authority::template-b",
                                        ]
                                    ),
                                }
                                if with_templates
                                else {}
                            ),
                            **(
                                {
                                    "activation_category_condition_robustness": condition_robustness_contract(
                                        minimum_coverage=0.75,
                                        minimum_direction_cosine=0.50,
                                        minimum_relative_norm=0.25,
                                    ),
                                    "activation_category_condition_names": [
                                        "replayed::replay-a",
                                        "replayed::replay-b",
                                        "valid_authority::valid-a",
                                        "valid_authority::valid-b",
                                    ],
                                    "activation_category_condition_categories": [
                                        "replayed",
                                        "replayed",
                                        "valid_authority",
                                        "valid_authority",
                                    ],
                                    "activation_category_condition_episode_counts": {
                                        "replayed::replay-a": 1,
                                        "replayed::replay-b": 1,
                                        "valid_authority::valid-a": 1,
                                        "valid_authority::valid-b": 1,
                                    },
                                    "activation_category_condition_set_sha256": content_sha256(
                                        [
                                            "replayed::replay-a",
                                            "replayed::replay-b",
                                            "valid_authority::valid-a",
                                            "valid_authority::valid-b",
                                        ]
                                    ),
                                }
                                if with_conditions
                                else {}
                            ),
                            "activation_category_episode_ids_sha256": {
                                "replayed": "f" * 64,
                                "valid_authority": "1" * 64,
                            },
                        }
                        if with_categories
                        else {}
                    ),
                },
                "shard_filename": shard_path.name,
                "shard_sha256": __import__(
                    "experiments.authz_mechinterp.cluster_vpd_for_sae", fromlist=["sha256_file"]
                ).sha256_file(shard_path),
                "component_id_set_sha256": content_sha256(component_ids),
                "rows": len(component_ids),
                "hidden_size": 3,
                **({"activation_category_count": 2} if with_categories else {}),
                "dtype": "float32",
                "factor_restoration_verified": True,
            }
            signed = sign_result(receipt, keys[worker])
            receipt_path = root / f"receipt-{index}.json"
            receipt_path.write_text(json.dumps(signed), encoding="utf-8")
            receipt_paths.append(receipt_path)
        return bundle, registry_path, shard_dir, receipt_paths, keys

    def test_exact_signed_effect_merge_is_reclusterable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, registry, shards, receipts, _ = self.prepare(root)
            output = root / "merged"
            manifest = merge_effect_shards(
                bundle_dir=bundle,
                registry_path=registry,
                receipt_paths=receipts,
                shard_dir=shards,
                output_dir=output,
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["coverage"]["merged_components"], 6)
            self.assertTrue(manifest["coverage"]["exact_partition"])
            self.assertEqual(verify_effect_merge(output)["status"], "pass")
            second = root / "effect-clusters"
            rows = synthetic_rows()
            rebuilt = build_bundle(
                rows,
                inventory_sha256=content_sha256(rows),
                effects_path=output / "activation_effects.npz",
                output_dir=second,
                cluster_count=2,
                seed=13,
                projection_dim=3,
                metadata_weight=0.2,
                max_iterations=50,
                anchors=["proposal_start"],
                workers=["worker-a", "worker-b"],
            )
            self.assertEqual(rebuilt["algorithm"]["feature_mode"], "effect_projection_plus_metadata")

    def test_tampered_effect_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, registry, shards, receipts, _ = self.prepare(root)
            first_shard = next(shards.glob("*.npz"))
            first_shard.write_bytes(first_shard.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                merge_effect_shards(
                    bundle_dir=bundle,
                    registry_path=registry,
                    receipt_paths=receipts,
                    shard_dir=shards,
                    output_dir=root / "merged",
                )

    def test_proposal_start_receipt_rejects_generated_token_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, registry, shards, receipts, keys = self.prepare(root)
            signed = json.loads(receipts[0].read_text(encoding="utf-8"))
            body = {
                key: value
                for key, value in signed.items()
                if key not in {"result_id", "signature"}
            }
            body["inputs"]["generated_tokens"] = 1
            worker_id = str(body["worker_id"])
            receipts[0].write_text(
                json.dumps(sign_result(body, keys[worker_id])), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "before token generation"):
                merge_effect_shards(
                    bundle_dir=bundle,
                    registry_path=registry,
                    receipt_paths=receipts,
                    shard_dir=shards,
                    output_dir=root / "merged",
                )

    def test_signed_category_effects_merge_and_recluster_by_compartment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, registry, shards, receipts, _ = self.prepare(
                root, with_categories=True
            )
            output = root / "merged"
            manifest = merge_effect_shards(
                bundle_dir=bundle,
                registry_path=registry,
                receipt_paths=receipts,
                shard_dir=shards,
                output_dir=output,
            )
            self.assertEqual(manifest["arrays"]["activation_category_count"], 2)
            self.assertEqual(verify_effect_merge(output)["activation_categories"], 2)
            reclustered = build_bundle(
                synthetic_rows(),
                inventory_sha256="inventory",
                effects_path=output / "activation_effects.npz",
                output_dir=root / "category-clusters",
                cluster_count=2,
                seed=13,
                projection_dim=3,
                metadata_weight=0.2,
                max_iterations=50,
                anchors=["proposal_start"],
                workers=["worker-a"],
            )
            self.assertEqual(
                reclustered["algorithm"]["feature_mode"],
                "share_weighted_activation_category_effects_plus_metadata",
            )
            self.assertTrue(
                reclustered["algorithm"]["activation_category_compartments"][
                    "enabled"
                ]
            )

    def test_v3_pair_block_uncertainty_merges_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, registry, shards, receipts, _ = self.prepare(
                root, with_categories=True, with_uncertainty=True
            )
            output = root / "merged"
            manifest = merge_effect_shards(
                bundle_dir=bundle,
                registry_path=registry,
                receipt_paths=receipts,
                shard_dir=shards,
                output_dir=output,
            )
            self.assertEqual(
                manifest["arrays"]["category_uncertainty_block_count"], 2
            )
            verification = verify_effect_merge(output)
            self.assertEqual(verification["category_uncertainty_blocks"], 2)

    def test_v4_condition_robustness_merges_and_reclusters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, registry, shards, receipts, _ = self.prepare(
                root,
                with_categories=True,
                with_uncertainty=True,
                with_conditions=True,
            )
            output = root / "merged"
            manifest = merge_effect_shards(
                bundle_dir=bundle,
                registry_path=registry,
                receipt_paths=receipts,
                shard_dir=shards,
                output_dir=output,
            )
            self.assertEqual(
                manifest["arrays"]["activation_category_condition_count"], 4
            )
            verification = verify_effect_merge(output)
            self.assertEqual(verification["activation_category_conditions"], 4)
            reclustered = build_bundle(
                synthetic_rows(),
                inventory_sha256="inventory",
                effects_path=output / "activation_effects.npz",
                output_dir=root / "condition-clusters",
                cluster_count=2,
                seed=13,
                projection_dim=3,
                metadata_weight=0.2,
                max_iterations=50,
                anchors=["proposal_start"],
                workers=["worker-a"],
                require_category_uncertainty=True,
                require_category_condition_robustness=True,
                minimum_category_robustness_sketch_dimension=8,
            )
            self.assertEqual(
                reclustered["algorithm"]["activation_category_compartments"][
                    "condition_robustness_screen"
                ]["mode"],
                "condition_mean_countsketch_direction_and_relative_norm_coverage",
            )

    def test_v5_template_robustness_merges_and_reclusters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, registry, shards, receipts, _ = self.prepare(
                root,
                with_categories=True,
                with_uncertainty=True,
                with_conditions=True,
                with_templates=True,
            )
            output = root / "merged"
            manifest = merge_effect_shards(
                bundle_dir=bundle,
                registry_path=registry,
                receipt_paths=receipts,
                shard_dir=shards,
                output_dir=output,
            )
            self.assertEqual(
                manifest["arrays"]["activation_category_template_cell_count"], 4
            )
            verification = verify_effect_merge(output)
            self.assertEqual(verification["activation_category_template_cells"], 4)
            reclustered = build_bundle(
                synthetic_rows(),
                inventory_sha256="inventory",
                effects_path=output / "activation_effects.npz",
                output_dir=root / "template-clusters",
                cluster_count=2,
                seed=13,
                projection_dim=3,
                metadata_weight=0.2,
                max_iterations=50,
                anchors=["proposal_start"],
                workers=["worker-a"],
                require_category_uncertainty=True,
                require_category_condition_robustness=True,
                require_category_template_robustness=True,
                minimum_category_robustness_sketch_dimension=8,
            )
            self.assertEqual(
                reclustered["algorithm"]["activation_category_compartments"][
                    "template_robustness_screen"
                ]["mode"],
                "template_group_mean_countsketch_direction_and_relative_norm_coverage",
            )


if __name__ == "__main__":
    unittest.main()
