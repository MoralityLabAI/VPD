from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.authz_mechinterp.activation_shards import (
    ActivationShardWriter,
    ShardedActivationAccessor,
    build_sharded_activation_artifact,
    verify_sharded_activation_artifact,
)
from experiments.authz_mechinterp.capture_activations import (
    load_capture_resource_plan,
    validate_capture_resource_role,
)
from experiments.authz_mechinterp.plan_compartment_capture_resources import (
    verify_capture_resource_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "artifacts" / "vpd_activation_category_capacity_pack_v3"
SPLIT_DIR = ROOT / "artifacts" / "vpd_activation_category_capacity_split_v4"
CAPACITY_PLAN = ROOT / "artifacts" / "vpd_compartment_sae_capacity_v2" / "plan.json"
RESOURCE_PLAN = (
    ROOT / "artifacts" / "vpd_compartment_capture_resources_v1" / "plan.json"
)


class ActivationShardTests(unittest.TestCase):
    def test_writer_verifier_and_accessor_preserve_exact_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = ActivationShardWriter(
                output_dir=root,
                key="layer_35__proposal_start",
                hidden_size=3,
                dtype=np.float16,
                shard_rows=4,
            )
            expected = np.arange(30, dtype=np.float16).reshape(10, 3)
            writer.append(expected[:3])
            writer.append(expected[3:8])
            writer.append(expected[8:])
            artifact = build_sharded_activation_artifact(
                {"layer_35__proposal_start": writer.finalize()}
            )
            verification = verify_sharded_activation_artifact(root, artifact)
            self.assertEqual(verification["rows"], 10)
            self.assertEqual(verification["shards"], 3)
            entry = artifact["keys"]["layer_35__proposal_start"]
            self.assertEqual(
                [shard["rows"] for shard in entry["shards"]], [4, 4, 2]
            )
            self.assertEqual(
                [shard["start_row"] for shard in entry["shards"]], [0, 4, 8]
            )
            with ShardedActivationAccessor(
                root, artifact, "layer_35__proposal_start"
            ) as accessor:
                requested = np.asarray([9, 0, 5, 3], dtype=np.int64)
                np.testing.assert_array_equal(
                    accessor.read_rows(requested), expected[requested].astype(np.float32)
                )
                np.testing.assert_array_equal(
                    accessor.finite_mask(), np.ones(10, dtype=bool)
                )
                np.testing.assert_allclose(
                    accessor.mean_rows(np.arange(10)),
                    expected.astype(np.float32).mean(axis=0),
                )

    def test_tampered_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = ActivationShardWriter(
                output_dir=root,
                key="layer_1__proposal_start",
                hidden_size=2,
                dtype=np.float16,
                shard_rows=2,
            )
            writer.append(np.ones((2, 2), dtype=np.float16))
            artifact = build_sharded_activation_artifact(
                {"layer_1__proposal_start": writer.finalize()}
            )
            path = root / artifact["keys"]["layer_1__proposal_start"]["shards"][0][
                "path"
            ]
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "activation shard differs"):
                verify_sharded_activation_artifact(root, artifact)

    def test_durable_capture_resource_plan_reproduces(self) -> None:
        verification = verify_capture_resource_plan(
            RESOURCE_PLAN,
            capacity_plan_path=CAPACITY_PLAN,
            pack_dir=PACK_DIR,
            split_dir=SPLIT_DIR,
        )
        self.assertEqual(verification["status"], "pass")
        self.assertEqual(
            verification["resource_plan_id"],
            "c99dd5c6e06bc5408aedc8df468d2b5d98a5825b73ca8e52fe6969b3b13d6ee0",
        )
        self.assertEqual(verification["activation_rows"], 805376)
        self.assertEqual(verification["shards_per_layer"], 197)
        self.assertEqual(
            verification["capture_roles"],
            {
                "calibration": {
                    "episodes": 4719,
                    "activation_rows": 604032,
                    "shards_per_layer": 148,
                },
                "sae_training": {
                    "episodes": 6292,
                    "activation_rows": 805376,
                    "shards_per_layer": 197,
                },
            },
        )
        loaded = load_capture_resource_plan(RESOURCE_PLAN)
        self.assertTrue(loaded["invariants"]["monolithic_npz_capture_forbidden"])
        self.assertEqual(
            loaded["compartments"]["shared_cross_category"][
                "fp32_training_bytes"
            ],
            9596567552,
        )
        calibration = validate_capture_resource_role(
            loaded,
            split_role="calibration",
            episode_count=4719,
            rows_per_episode=128,
        )
        self.assertEqual(calibration["activation_rows"], 604032)
        with self.assertRaisesRegex(ValueError, "resource-plan role"):
            validate_capture_resource_role(
                loaded,
                split_role="calibration",
                episode_count=6292,
                rows_per_episode=128,
            )


if __name__ == "__main__":
    unittest.main()
