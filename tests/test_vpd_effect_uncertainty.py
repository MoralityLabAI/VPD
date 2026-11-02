from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    build_bundle,
    content_sha256,
    verify_bundle,
)
from experiments.authz_mechinterp.compute_vpd_effect_shard import (
    pair_uncertainty_blocks,
)
from experiments.authz_mechinterp.effect_uncertainty import (
    condition_robustness_metrics,
    countsketch,
    leave_one_block_out_means,
    leave_one_block_out_strengths,
    pair_block_direction_metrics,
)
from tests.test_vpd_sae_clusters import synthetic_rows


class VpdEffectUncertaintyTests(unittest.TestCase):
    def test_direction_cosine_diagnostics_are_numerically_bounded(self) -> None:
        vector = np.asarray([0.1328649191922372, -0.7132185101509094, 0.6881977915763855])
        block_metrics = pair_block_direction_metrics(
            block_sums=np.repeat(vector[None, None, None, :], 2, axis=2),
            block_counts=np.ones((1, 2), dtype=np.int64),
            minimum_direction_cosine=0.5,
            minimum_relative_norm=0.5,
        )
        condition_metrics = condition_robustness_metrics(
            condition_sums=np.repeat(vector[None, None, :], 2, axis=1),
            condition_counts=np.ones(2, dtype=np.int64),
            condition_categories=["category", "category"],
            category_names=["category"],
            minimum_direction_cosine=0.5,
            minimum_relative_norm=0.5,
        )
        for metrics in (block_metrics, condition_metrics):
            values = metrics["minimum_direction_cosine"]
            self.assertTrue(np.all(values >= -1.0))
            self.assertTrue(np.all(values <= 1.0))

    def test_countsketch_is_linear_and_jackknife_preserves_paired_blocks(self) -> None:
        values = np.asarray([[1.0, 2.0, 3.0], [4.0, -1.0, 0.5]])
        combined = countsketch(values.sum(axis=0), dimension=8, seed=17)
        separate = countsketch(values, dimension=8, seed=17).sum(axis=0)
        np.testing.assert_allclose(combined, separate)

        sums = np.asarray(
            [[[[4.0, 0.0], [0.0, 0.0]], [[0.1, 0.0], [0.1, 0.0]]]],
            dtype=np.float32,
        )
        counts = np.ones((2, 2), dtype=np.int64)
        strengths = leave_one_block_out_strengths(sums, counts)
        means, remaining_counts = leave_one_block_out_means(sums, counts)
        self.assertEqual(means.shape, (1, 2, 2, 2))
        self.assertEqual(remaining_counts.tolist(), [[1, 1], [1, 1]])
        self.assertEqual(strengths.shape, (1, 2, 2))
        self.assertGreater(strengths[0, 0, 1], strengths[0, 0, 0])
        self.assertGreater(strengths[0, 1, 0], strengths[0, 1, 1])

    def test_pair_blocks_require_complete_cross_category_pair_support(self) -> None:
        complete = {
            "replayed": [
                {"pair_id": f"pair-{index}"} for index in range(4)
            ],
            "valid_authority": [
                {"pair_id": f"pair-{index}"} for index in range(4)
            ],
        }
        assignment, blocks = pair_uncertainty_blocks(
            complete, requested_blocks=3, seed=19
        )
        self.assertEqual(blocks, 3)
        self.assertEqual(set(assignment), {f"pair-{index}" for index in range(4)})
        incomplete = {**complete, "valid_authority": complete["valid_authority"][:-1]}
        with self.assertRaisesRegex(ValueError, "same selected pair set"):
            pair_uncertainty_blocks(incomplete, requested_blocks=3, seed=19)

    def test_unstable_specific_point_estimate_routes_to_shared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = synthetic_rows()
            component_ids = np.asarray([row["component_id"] for row in rows])
            category_names = np.asarray(["replayed", "valid_authority"])
            category_delta = np.zeros((len(rows), 2, 3), dtype=np.float32)
            category_delta[:, 0, 0] = 2.0
            category_delta[:, 1, 0] = 0.1
            counts = np.asarray([2, 2], dtype=np.int64)
            block_counts = np.ones((2, 2), dtype=np.int64)
            seed = 23
            block_samples = np.zeros((len(rows), 2, 2, 3), dtype=np.float32)
            block_samples[:, 0, 0, 0] = 4.0
            block_samples[:, 1, :, 0] = 0.1
            block_sums = countsketch(
                block_samples, dimension=8, seed=seed
            ).astype(np.float32)
            effects = root / "effects.npz"
            np.savez(
                effects,
                component_ids=component_ids,
                delta_h=category_delta.mean(axis=1).astype(np.float32),
                category_names=category_names,
                category_episode_counts=counts,
                delta_h_by_category=category_delta,
                category_uncertainty_block_counts=block_counts,
                delta_h_countsketch_block_sums=block_sums,
                uncertainty_block_count=np.asarray(2, dtype=np.int64),
                uncertainty_countsketch_dimension=np.asarray(8, dtype=np.int64),
                uncertainty_countsketch_seed=np.asarray(seed, dtype=np.int64),
            )
            output = root / "bundle"
            manifest = build_bundle(
                rows,
                inventory_sha256=content_sha256(rows),
                effects_path=effects,
                output_dir=output,
                cluster_count=2,
                seed=13,
                projection_dim=3,
                metadata_weight=0.2,
                max_iterations=50,
                anchors=["proposal_start"],
                workers=["worker-a"],
                require_category_uncertainty=True,
                minimum_category_jackknife_stability=0.75,
            )
            contract = manifest["algorithm"]["activation_category_compartments"]
            self.assertEqual(
                contract["uncertainty_screen"]["mode"],
                "leave_one_pair_block_out_countsketch",
            )
            assignments = [
                __import__("json").loads(line)
                for line in (output / "assignments.jsonl").read_text().splitlines()
            ]
            self.assertTrue(
                all(
                    row["point_estimate_activation_category"] == "replayed"
                    and row["activation_category"] == "shared_cross_category"
                    and row["activation_category_jackknife_stability"] == 0.5
                    for row in assignments
                )
            )
            self.assertEqual(verify_bundle(output)["status"], "pass")

    def test_single_condition_specificity_routes_to_shared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = synthetic_rows()
            component_ids = np.asarray([row["component_id"] for row in rows])
            category_names = np.asarray(["replayed", "valid_authority"])
            category_delta = np.zeros((len(rows), 2, 3), dtype=np.float32)
            category_delta[:, 0, 0] = 2.0
            category_delta[:, 1, 0] = 0.1
            seed = 23
            block_counts = np.ones((2, 2), dtype=np.int64)
            block_means = np.repeat(category_delta[:, :, None, :], 2, axis=2)
            block_sums = countsketch(
                block_means, dimension=8, seed=seed
            ).astype(np.float32)
            condition_vectors = np.zeros((len(rows), 4, 3), dtype=np.float32)
            condition_vectors[:, 0, 0] = 4.0
            condition_vectors[:, 1, 0] = 0.0
            condition_vectors[:, 2:, 0] = 0.1
            effects = root / "condition-effects.npz"
            np.savez(
                effects,
                component_ids=component_ids,
                delta_h=category_delta.mean(axis=1).astype(np.float32),
                category_names=category_names,
                category_episode_counts=np.asarray([2, 2], dtype=np.int64),
                delta_h_by_category=category_delta,
                category_uncertainty_block_counts=block_counts,
                delta_h_countsketch_block_sums=block_sums,
                uncertainty_block_count=np.asarray(2, dtype=np.int64),
                uncertainty_countsketch_dimension=np.asarray(8, dtype=np.int64),
                uncertainty_countsketch_seed=np.asarray(seed, dtype=np.int64),
                condition_names=np.asarray(
                    [
                        "replayed::replay-a",
                        "replayed::replay-b",
                        "valid_authority::valid-a",
                        "valid_authority::valid-b",
                    ]
                ),
                condition_category_names=np.asarray(
                    [
                        "replayed",
                        "replayed",
                        "valid_authority",
                        "valid_authority",
                    ]
                ),
                condition_episode_counts=np.ones(4, dtype=np.int64),
                delta_h_countsketch_condition_sums=countsketch(
                    condition_vectors, dimension=8, seed=seed
                ).astype(np.float32),
            )
            with self.assertRaisesRegex(ValueError, "CountSketch dimension is underpowered"):
                build_bundle(
                    rows,
                    inventory_sha256=content_sha256(rows),
                    effects_path=effects,
                    output_dir=root / "underpowered-bundle",
                    cluster_count=2,
                    seed=13,
                    projection_dim=3,
                    metadata_weight=0.2,
                    max_iterations=50,
                    anchors=["proposal_start"],
                    workers=["worker-a"],
                    require_category_uncertainty=True,
                    require_category_condition_robustness=True,
                )
            output = root / "bundle"
            manifest = build_bundle(
                rows,
                inventory_sha256=content_sha256(rows),
                effects_path=effects,
                output_dir=output,
                cluster_count=2,
                seed=13,
                projection_dim=3,
                metadata_weight=0.2,
                max_iterations=50,
                anchors=["proposal_start"],
                workers=["worker-a"],
                require_category_uncertainty=True,
                require_category_condition_robustness=True,
                minimum_category_condition_coverage=0.75,
                minimum_category_robustness_sketch_dimension=8,
            )
            assignments = [
                __import__("json").loads(line)
                for line in (output / "assignments.jsonl").read_text().splitlines()
            ]
            self.assertTrue(
                all(
                    row["post_uncertainty_activation_category"] == "replayed"
                    and row["activation_category"] == "shared_cross_category"
                    and row["activation_category_condition_coverage"]["replayed"]
                    == 0.5
                    for row in assignments
                )
            )
            self.assertEqual(
                manifest["algorithm"]["activation_category_compartments"][
                    "condition_robustness_screen"
                ]["minimum_coverage"],
                0.75,
            )
            self.assertEqual(verify_bundle(output)["status"], "pass")

    def test_opposing_pair_block_directions_route_to_shared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = synthetic_rows()
            component_ids = np.asarray([row["component_id"] for row in rows])
            category_names = np.asarray(["replayed", "valid_authority"])
            category_delta = np.zeros((len(rows), 2, 3), dtype=np.float32)
            category_delta[:, 0, 0] = 2.0
            category_delta[:, 1, 0] = 0.1
            block_counts = np.ones((2, 4), dtype=np.int64)
            block_means = np.repeat(category_delta[:, :, None, :], 4, axis=2)
            block_means[:, 0, :, :] *= np.asarray([3.0, 3.0, -1.0, -1.0])[
                None, :, None
            ]
            seed = 23
            effects = root / "opposing-block-effects.npz"
            np.savez(
                effects,
                component_ids=component_ids,
                delta_h=category_delta.mean(axis=1).astype(np.float32),
                category_names=category_names,
                category_episode_counts=np.asarray([4, 4], dtype=np.int64),
                delta_h_by_category=category_delta,
                category_uncertainty_block_counts=block_counts,
                delta_h_countsketch_block_sums=countsketch(
                    block_means, dimension=8, seed=seed
                ).astype(np.float32),
                uncertainty_block_count=np.asarray(4, dtype=np.int64),
                uncertainty_countsketch_dimension=np.asarray(8, dtype=np.int64),
                uncertainty_countsketch_seed=np.asarray(seed, dtype=np.int64),
            )
            output = root / "bundle"
            manifest = build_bundle(
                rows,
                inventory_sha256=content_sha256(rows),
                effects_path=effects,
                output_dir=output,
                cluster_count=2,
                seed=13,
                projection_dim=3,
                metadata_weight=0.2,
                max_iterations=50,
                anchors=["proposal_start"],
                workers=["worker-a"],
                require_category_uncertainty=True,
                require_category_pair_block_direction_robustness=True,
                minimum_category_pair_block_direction_coverage=0.75,
                minimum_category_robustness_sketch_dimension=8,
            )
            assignments = [
                __import__("json").loads(line)
                for line in (output / "assignments.jsonl").read_text().splitlines()
            ]
            self.assertTrue(
                all(
                    row["point_estimate_activation_category"] == "replayed"
                    and row["post_uncertainty_activation_category"] == "replayed"
                    and row["activation_category_jackknife_stability"] == 1.0
                    and row[
                        "post_pair_block_direction_activation_category"
                    ]
                    == "shared_cross_category"
                    and row["activation_category"] == "shared_cross_category"
                    and row[
                        "activation_category_pair_block_direction_coverage"
                    ]["replayed"]
                    == 0.5
                    for row in assignments
                )
            )
            self.assertTrue(
                manifest["algorithm"]["activation_category_compartments"][
                    "pair_block_direction_robustness_screen"
                ]["required"]
            )
            self.assertEqual(verify_bundle(output)["status"], "pass")

    def test_single_template_specificity_routes_to_shared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = synthetic_rows()
            component_ids = np.asarray([row["component_id"] for row in rows])
            category_names = np.asarray(["replayed", "valid_authority"])
            category_delta = np.zeros((len(rows), 2, 3), dtype=np.float32)
            category_delta[:, 0, 0] = 2.0
            category_delta[:, 1, 0] = 0.1
            seed = 23
            block_counts = np.ones((2, 2), dtype=np.int64)
            block_sums = countsketch(
                np.repeat(category_delta[:, :, None, :], 2, axis=2),
                dimension=8,
                seed=seed,
            ).astype(np.float32)
            condition_vectors = np.stack(
                [
                    category_delta[:, 0],
                    category_delta[:, 0],
                    category_delta[:, 1],
                    category_delta[:, 1],
                ],
                axis=1,
            )
            template_vectors = np.zeros((len(rows), 4, 3), dtype=np.float32)
            template_vectors[:, 0, 0] = 4.0
            template_vectors[:, 1, 0] = 0.0
            template_vectors[:, 2:, 0] = 0.1
            effects = root / "template-effects.npz"
            np.savez(
                effects,
                component_ids=component_ids,
                delta_h=category_delta.mean(axis=1).astype(np.float32),
                category_names=category_names,
                category_episode_counts=np.asarray([2, 2], dtype=np.int64),
                delta_h_by_category=category_delta,
                category_uncertainty_block_counts=block_counts,
                delta_h_countsketch_block_sums=block_sums,
                uncertainty_block_count=np.asarray(2, dtype=np.int64),
                uncertainty_countsketch_dimension=np.asarray(8, dtype=np.int64),
                uncertainty_countsketch_seed=np.asarray(seed, dtype=np.int64),
                condition_names=np.asarray(
                    [
                        "replayed::replay-a",
                        "replayed::replay-b",
                        "valid_authority::valid-a",
                        "valid_authority::valid-b",
                    ]
                ),
                condition_category_names=np.asarray(
                    ["replayed", "replayed", "valid_authority", "valid_authority"]
                ),
                condition_episode_counts=np.ones(4, dtype=np.int64),
                delta_h_countsketch_condition_sums=countsketch(
                    condition_vectors, dimension=8, seed=seed
                ).astype(np.float32),
                template_cell_names=np.asarray(
                    [
                        "replayed::template-a",
                        "replayed::template-b",
                        "valid_authority::template-a",
                        "valid_authority::template-b",
                    ]
                ),
                template_cell_category_names=np.asarray(
                    ["replayed", "replayed", "valid_authority", "valid_authority"]
                ),
                template_cell_group_names=np.asarray(
                    ["template-a", "template-b", "template-a", "template-b"]
                ),
                template_cell_episode_counts=np.ones(4, dtype=np.int64),
                delta_h_countsketch_template_cell_sums=countsketch(
                    template_vectors, dimension=8, seed=seed
                ).astype(np.float32),
            )
            output = root / "bundle"
            manifest = build_bundle(
                rows,
                inventory_sha256=content_sha256(rows),
                effects_path=effects,
                output_dir=output,
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
                minimum_category_template_coverage=0.75,
                minimum_category_robustness_sketch_dimension=8,
            )
            assignments = [
                __import__("json").loads(line)
                for line in (output / "assignments.jsonl").read_text().splitlines()
            ]
            self.assertTrue(
                all(
                    row["post_condition_activation_category"] == "replayed"
                    and row["activation_category"] == "shared_cross_category"
                    and row["activation_category_template_coverage"]["replayed"]
                    == 0.5
                    for row in assignments
                )
            )
            self.assertTrue(
                manifest["algorithm"]["activation_category_compartments"][
                    "template_robustness_screen"
                ]["required"]
            )
            self.assertEqual(verify_bundle(output)["status"], "pass")


if __name__ == "__main__":
    unittest.main()
