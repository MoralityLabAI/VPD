from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    build_bundle,
    calibrate_activation_compartment_thresholds,
    classify_activation_compartments,
    content_sha256,
    sha256_file,
    verify_bundle,
)
from experiments.authz_mechinterp.effect_uncertainty import countsketch
from experiments.authz_mechinterp.activation_categories import category_contract
from experiments.authz_mechinterp.benchmark_activation_category_cluster_discreteness_v2 import (
    SCENARIOS,
    synthetic_effects,
)


def component(component_id: str, layer: int, family: str, norm: float) -> dict:
    return {
        "component_id": component_id,
        "module": f"model.layers.{layer}.{family}",
        "layer": layer,
        "rank_index": int(component_id.rsplit("rank", 1)[1]),
        "projection_family": family,
        "component_fro_norm": norm,
        "fraction_of_module_delta_fro_sq": norm,
    }


def synthetic_rows() -> list[dict]:
    return [
        component("model.layers.0.self_attn.q_proj:rank0", 0, "self_attn.q_proj", 0.10),
        component("model.layers.1.self_attn.q_proj:rank1", 1, "self_attn.q_proj", 0.11),
        component("model.layers.2.self_attn.q_proj:rank2", 2, "self_attn.q_proj", 0.09),
        component("model.layers.9.mlp.down_proj:rank3", 9, "mlp.down_proj", 0.20),
        component("model.layers.10.mlp.down_proj:rank4", 10, "mlp.down_proj", 0.21),
        component("model.layers.11.mlp.down_proj:rank5", 11, "mlp.down_proj", 0.19),
    ]


def write_effects(path: Path, rows: list[dict]) -> None:
    ids = [row["component_id"] for row in rows]
    effects = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.95, -0.05, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
            [-0.05, 0.95, 0.0],
        ],
        dtype=np.float32,
    )
    np.savez(path, component_ids=np.asarray(ids), delta_h=effects)


def write_category_effects(path: Path, rows: list[dict]) -> None:
    ids = [row["component_id"] for row in rows]
    category_names = np.asarray(["replayed", "valid_authority"])
    category_delta = np.zeros((len(rows), 2, 3), dtype=np.float32)
    for index in range(len(rows)):
        if index < 3:
            category_delta[index, 0] = [0.1, 0.0, 0.0]
            category_delta[index, 1] = [2.0, 0.1, 0.0]
        else:
            category_delta[index, 0] = [0.0, 2.0, 0.1]
            category_delta[index, 1] = [0.0, 0.1, 0.0]
    counts = np.asarray([2, 2], dtype=np.int64)
    delta_h = np.average(category_delta, axis=1, weights=counts).astype(np.float32)
    np.savez(
        path,
        component_ids=np.asarray(ids),
        delta_h=delta_h,
        category_names=category_names,
        category_episode_counts=counts,
        delta_h_by_category=category_delta,
    )


def write_category_effects_with_shared(path: Path, rows: list[dict]) -> None:
    ids = [row["component_id"] for row in rows]
    category_names = np.asarray(["replayed", "valid_authority"])
    category_delta = np.zeros((len(rows), 2, 3), dtype=np.float32)
    for index in range(len(rows)):
        if index < 2:
            category_delta[index] = [[0.1, 0.0, 0.0], [2.0, 0.0, 0.0]]
        elif index < 4:
            category_delta[index] = [[0.0, 2.0, 0.0], [0.0, 0.1, 0.0]]
        else:
            category_delta[index] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    counts = np.asarray([1, 1], dtype=np.int64)
    np.savez(
        path,
        component_ids=np.asarray(ids),
        delta_h=category_delta.mean(axis=1).astype(np.float32),
        category_names=category_names,
        category_episode_counts=counts,
        delta_h_by_category=category_delta,
    )


def write_category_calibration_effects(
    path: Path, rows: list[dict], *, swap_categories: bool = False
) -> None:
    ids = [row["component_id"] for row in rows]
    category_names = np.asarray(["replayed", "valid_authority"])
    category_delta = np.zeros((len(rows), 2, 3), dtype=np.float32)
    for index in range(len(rows)):
        if index < 3:
            category_delta[index] = [[0.12, 0.0, 0.0], [1.8, 0.08, 0.0]]
        else:
            category_delta[index] = [[0.0, 1.8, 0.08], [0.0, 0.12, 0.0]]
    if swap_categories:
        category_delta = category_delta[:, ::-1, :]
    counts = np.asarray([3, 3], dtype=np.int64)
    np.savez(
        path,
        component_ids=np.asarray(ids),
        delta_h=np.average(category_delta, axis=1, weights=counts).astype(np.float32),
        category_names=category_names,
        category_episode_counts=counts,
        delta_h_by_category=category_delta,
    )


def write_cluster_geometry_effects(
    path: Path,
    rows: list[dict],
    *,
    crossed_membership: bool = False,
    scale: float = 1.0,
) -> None:
    category_delta = np.zeros((len(rows), 2, 4), dtype=np.float32)
    valid_groups = ((0, 1), (2, 3))
    replayed_groups = ((4, 5), (6, 7))
    if crossed_membership:
        valid_groups = ((0, 2), (1, 3))
        replayed_groups = ((4, 6), (5, 7))
    for dimension, group in enumerate(valid_groups):
        category_delta[list(group), 1, dimension] = 2.0 * scale
    for dimension, group in enumerate(replayed_groups, start=2):
        category_delta[list(group), 0, dimension] = 2.0 * scale
    category_delta[0:4, 0, 0] = 0.01 * scale
    category_delta[4:8, 1, 0] = 0.01 * scale
    block_counts = np.ones((2, 4), dtype=np.int64)
    block_means = np.repeat(category_delta[:, :, None, :], 4, axis=2)
    sketch_seed = 23
    np.savez(
        path,
        component_ids=np.asarray([row["component_id"] for row in rows]),
        delta_h=category_delta.mean(axis=1),
        category_names=np.asarray(["replayed", "valid_authority"]),
        category_episode_counts=np.asarray([4, 4], dtype=np.int64),
        delta_h_by_category=category_delta,
        category_uncertainty_block_counts=block_counts,
        delta_h_countsketch_block_sums=countsketch(
            block_means, dimension=8, seed=sketch_seed
        ).astype(np.float32),
        uncertainty_block_count=np.asarray(4, dtype=np.int64),
        uncertainty_countsketch_dimension=np.asarray(8, dtype=np.int64),
        uncertainty_countsketch_seed=np.asarray(sketch_seed, dtype=np.int64),
    )


def write_effect_merge_manifest(
    path: Path,
    effects_path: Path,
    *,
    role: str,
    split_id: str = "2" * 64,
    split_manifest_sha256: str = "3" * 64,
    semantic_template_hash: str | None = None,
) -> None:
    role_digit = "4" if role == "discovery" else "5"
    body = {
        "schema": "sch.vpd-effect-shard-merge.v1",
        "status": "complete",
        "identity": {
            "activation_category_split_id": split_id,
            "activation_category_split_manifest_sha256": split_manifest_sha256,
            "activation_category_split_role": role,
            "activation_category_split_episode_id_set_sha256": role_digit * 64,
            "activation_category_split_pair_id_set_sha256": (
                "6" * 64 if role == "discovery" else "7" * 64
            ),
            "activation_category_split_verification_status": "pass",
            **(
                {
                    "activation_category_split_schema": (
                        "sch.vpd-activation-category-split-manifest.v2"
                    ),
                    "activation_category_split_semantic_disjointness": "pass",
                    "activation_category_split_template_group_set_sha256": (
                        semantic_template_hash
                    ),
                }
                if semantic_template_hash is not None
                else {}
            ),
        },
        "artifacts": {
            "activation_effects": {
                "path": effects_path.relative_to(path.parent).as_posix(),
                "sha256": sha256_file(effects_path),
            }
        },
    }
    path.write_text(
        json.dumps({**body, "merge_id": content_sha256(body)}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def build(tmp_path: Path, rows: list[dict]) -> dict:
    effects = tmp_path / "effects.npz"
    write_effects(effects, synthetic_rows())
    return build_bundle(
        rows,
        inventory_sha256=content_sha256(rows),
        effects_path=effects,
        output_dir=tmp_path / "bundle",
        cluster_count=2,
        seed=13,
        projection_dim=3,
        metadata_weight=0.2,
        max_iterations=50,
        anchors=["auth_end", "proposal_start"],
        workers=["worker-a", "worker-b"],
    )


class VpdSaeClusterTests(unittest.TestCase):
    def test_independent_threshold_calibration_preserves_only_stable_specificity(self) -> None:
        discovery = np.asarray(
            [[9.0, 1.0], [8.0, 2.0], [1.0, 9.0], [2.0, 8.0], [5.0, 5.0], [5.2, 4.8]]
        )
        calibration = np.asarray(
            [[8.5, 1.0], [7.8, 2.0], [1.0, 8.5], [2.0, 7.8], [4.9, 5.1], [4.8, 5.2]]
        )
        result = calibrate_activation_compartment_thresholds(
            discovery,
            calibration,
            ["replayed", "valid_authority"],
            direction_cosines=np.ones_like(discovery),
            minimum_stable_specific_components=1,
        )
        self.assertEqual(result["selection"]["selected_metrics"]["stable_specific_count"], 4)
        self.assertFalse(
            result["policy"]["cross_archive_stratified_robustness_required"]
        )
        self.assertEqual(
            result["selection"]["selected_metrics"][
                "magnitude_rejected_specific_count"
            ],
            0,
        )
        self.assertGreaterEqual(
            result["selection"]["selected_metrics"]["symmetric_replication_rate"],
            0.90,
        )
        thresholds = result["selection"]["selected_thresholds"]
        classified = classify_activation_compartments(
            discovery,
            ["replayed", "valid_authority"],
            minimum_dominant_share=thresholds["minimum_dominant_share"],
            minimum_dominance_margin=thresholds["minimum_dominance_margin"],
            maximum_normalized_entropy=thresholds["maximum_normalized_entropy"],
        )
        self.assertEqual(
            classified["compartment_categories"],
            [
                "replayed",
                "replayed",
                "valid_authority",
                "valid_authority",
                "shared_cross_category",
                "shared_cross_category",
            ],
        )
        with self.assertRaisesRegex(ValueError, "independent replication policy"):
            calibrate_activation_compartment_thresholds(
                discovery,
                calibration,
                ["replayed", "valid_authority"],
                direction_cosines=-np.ones_like(discovery),
                minimum_stable_specific_components=1,
            )
        with self.assertRaisesRegex(ValueError, "independent replication policy"):
            calibrate_activation_compartment_thresholds(
                discovery,
                calibration * 0.02,
                ["replayed", "valid_authority"],
                direction_cosines=np.ones_like(discovery),
                minimum_stable_specific_components=1,
            )
        robustness_eligibility = np.ones_like(discovery, dtype=bool)
        robustness_eligibility[0, 0] = False
        with self.assertRaisesRegex(ValueError, "independent replication policy"):
            calibrate_activation_compartment_thresholds(
                discovery,
                calibration,
                ["replayed", "valid_authority"],
                direction_cosines=np.ones_like(discovery),
                robustness_eligibility=robustness_eligibility,
                minimum_stable_specific_components=1,
            )

    def test_bundle_is_exact_content_addressed_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            rows = synthetic_rows()
            manifest = build(tmp_path, rows)
            verification = verify_bundle(tmp_path / "bundle")
            self.assertEqual(
                verification,
                {
                    "schema": "sch.vpd-sae-cluster-verification.v1",
                    "status": "pass",
                    "components": 6,
                    "clusters": 2,
                    "packets": 2,
                    "exact_partition": True,
                },
            )
            self.assertTrue(manifest["invariants"]["exact_partition"])
            assignments = [
                json.loads(line)
                for line in (tmp_path / "bundle" / "assignments.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len({row["component_id"] for row in assignments}), 6)
            self.assertEqual(len({row["cluster_id"] for row in assignments}), 2)

    def test_cluster_membership_is_independent_of_inventory_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            rows = synthetic_rows()
            first_dir = tmp_path / "first"
            second_dir = tmp_path / "second"
            effects = tmp_path / "effects.npz"
            write_effects(effects, rows)
            common = {
                "inventory_sha256": "inventory",
                "effects_path": effects,
                "cluster_count": 2,
                "seed": 13,
                "projection_dim": 3,
                "metadata_weight": 0.2,
                "max_iterations": 50,
                "anchors": ["proposal_start"],
                "workers": [],
            }
            build_bundle(rows, output_dir=first_dir, **common)
            build_bundle(list(reversed(rows)), output_dir=second_dir, **common)
            self.assertEqual(
                (first_dir / "assignments.jsonl").read_bytes(),
                (second_dir / "assignments.jsonl").read_bytes(),
            )

    def test_verifier_rejects_tampered_packet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            manifest = build(tmp_path, synthetic_rows())
            entry = manifest["artifacts"]["packets"][0]
            packet_path = tmp_path / "bundle" / entry["packet_path"]
            packet = json.loads(packet_path.read_text())
            packet["component_ids"].pop()
            packet_path.write_text(json.dumps(packet), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "packet artifact hash mismatch"):
                verify_bundle(tmp_path / "bundle")

    def test_category_effects_create_homogeneous_sae_compartments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = synthetic_rows()
            effects = root / "category-effects.npz"
            write_category_effects(effects, rows)
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
                minimum_stable_specific_components=1,
                minimum_category_calibration_episodes=1,
            )
            contract = manifest["algorithm"]["activation_category_compartments"]
            self.assertTrue(contract["enabled"])
            self.assertEqual(
                contract["within_category_clustering_geometry"][
                    "direction_weighting"
                ],
                "sqrt_category_strength_share",
            )
            self.assertEqual(
                contract["cluster_allocation"],
                {"replayed": 1, "valid_authority": 1},
            )
            assignments = [
                json.loads(line)
                for line in (output / "assignments.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                {row["activation_category"] for row in assignments},
                {"replayed", "valid_authority"},
            )
            for entry in manifest["artifacts"]["packets"]:
                packet = json.loads((output / entry["packet_path"]).read_text())
                category = packet["sae_study"]["activation_compartment"]["category"]
                member_categories = {
                    row["activation_category"]
                    for row in assignments
                    if row["component_id"] in packet["component_ids"]
                }
                self.assertEqual(member_categories, {category})
                self.assertIn(
                    "cross_activation_category_holdout",
                    packet["sae_study"]["required_controls"],
                )
            self.assertEqual(verify_bundle(output)["status"], "pass")

    def test_ambiguous_effects_route_to_shared_cross_category_compartment(self) -> None:
        classification = classify_activation_compartments(
            np.asarray([[2.0, 0.1], [1.0, 1.0]], dtype=np.float32),
            ["replayed", "valid_authority"],
            minimum_dominant_share=0.35,
            minimum_dominance_margin=0.10,
            maximum_normalized_entropy=0.80,
        )
        self.assertEqual(
            classification["compartment_categories"],
            ["replayed", "shared_cross_category"],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = synthetic_rows()
            effects = root / "category-effects-shared.npz"
            write_category_effects_with_shared(effects, rows)
            output = root / "bundle"
            manifest = build_bundle(
                rows,
                inventory_sha256=content_sha256(rows),
                effects_path=effects,
                output_dir=output,
                cluster_count=3,
                seed=13,
                projection_dim=3,
                metadata_weight=0.2,
                max_iterations=50,
                anchors=["proposal_start"],
                workers=["worker-a"],
                minimum_stable_specific_components=1,
                minimum_category_calibration_episodes=1,
            )
            contract = manifest["algorithm"]["activation_category_compartments"]
            self.assertEqual(
                contract["observed_component_compartments"],
                ["replayed", "shared_cross_category", "valid_authority"],
            )
            assignments = [
                json.loads(line)
                for line in (output / "assignments.jsonl").read_text().splitlines()
            ]
            shared = [
                row
                for row in assignments
                if row["activation_category"] == "shared_cross_category"
            ]
            self.assertEqual(len(shared), 2)
            self.assertTrue(
                all(row["activation_category_dominance_margin"] == 0.0 for row in shared)
            )
            shared_packet = next(
                json.loads((output / entry["packet_path"]).read_text())
                for entry in manifest["artifacts"]["packets"]
                if json.loads((output / entry["packet_path"]).read_text())[
                    "sae_study"
                ]["activation_compartment"]["category"]
                == "shared_cross_category"
            )
            self.assertEqual(
                shared_packet["sae_study"]["dictionary_scope"],
                "fixed_base_balanced_cross_category",
            )
            self.assertEqual(verify_bundle(output)["status"], "pass")

    def test_projection_ensemble_cluster_count_selection_falls_back_conservatively(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = synthetic_rows()
            effects = root / "category-effects.npz"
            write_category_effects(effects, rows)
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
                workers=[],
                select_category_cluster_counts_by_projection=True,
                category_cluster_projection_replicates=3,
                maximum_clusters_per_category=2,
                minimum_category_cluster_size=2,
            )
            contract = manifest["algorithm"]["activation_category_compartments"]
            receipt = contract["cluster_count_selection"]
            self.assertEqual(
                receipt["mode"],
                "projection_ensemble_stability_and_silhouette",
            )
            self.assertEqual(receipt["selection"]["allocation"], {
                "replayed": 1,
                "valid_authority": 1,
            })
            self.assertIsNone(
                manifest["algorithm"]["requested_manual_cluster_count"]
            )
            self.assertEqual(manifest["algorithm"]["cluster_count"], 2)
            self.assertEqual(verify_bundle(output)["status"], "pass")

    def test_projection_ensemble_cluster_count_selection_builds_nontrivial_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                component(
                    f"model.layers.{index}.self_attn.q_proj:rank{index}",
                    index,
                    "self_attn.q_proj",
                    0.1 + index * 0.01,
                )
                for index in range(8)
            ]
            effects = root / "category-effects.npz"
            category_delta = np.zeros((8, 2, 4), dtype=np.float32)
            category_delta[0:2, 1, 0] = 2.0
            category_delta[2:4, 1, 1] = 2.0
            category_delta[4:6, 0, 2] = 2.0
            category_delta[6:8, 0, 3] = 2.0
            category_delta[0:4, 0, 0] = 0.01
            category_delta[4:8, 1, 0] = 0.01
            block_counts = np.ones((2, 4), dtype=np.int64)
            block_means = np.repeat(category_delta[:, :, None, :], 4, axis=2)
            sketch_seed = 23
            np.savez(
                effects,
                component_ids=np.asarray([row["component_id"] for row in rows]),
                delta_h=category_delta.mean(axis=1),
                category_names=np.asarray(["replayed", "valid_authority"]),
                category_episode_counts=np.asarray([4, 4], dtype=np.int64),
                delta_h_by_category=category_delta,
                category_uncertainty_block_counts=block_counts,
                delta_h_countsketch_block_sums=countsketch(
                    block_means, dimension=8, seed=sketch_seed
                ).astype(np.float32),
                uncertainty_block_count=np.asarray(4, dtype=np.int64),
                uncertainty_countsketch_dimension=np.asarray(8, dtype=np.int64),
                uncertainty_countsketch_seed=np.asarray(sketch_seed, dtype=np.int64),
            )
            output = root / "bundle"
            manifest = build_bundle(
                rows,
                inventory_sha256=content_sha256(rows),
                effects_path=effects,
                output_dir=output,
                cluster_count=2,
                seed=13,
                projection_dim=4,
                metadata_weight=0.0,
                max_iterations=50,
                anchors=["proposal_start"],
                workers=[],
                select_category_cluster_counts_by_projection=True,
                category_cluster_projection_replicates=3,
                maximum_clusters_per_category=2,
                minimum_category_cluster_size=2,
                require_category_cluster_sampling_stability=True,
            )
            contract = manifest["algorithm"]["activation_category_compartments"]
            self.assertEqual(
                contract["cluster_count_selection"]["selection"]["allocation"],
                {"replayed": 2, "valid_authority": 2},
            )
            self.assertEqual(
                contract["cluster_count_selection"]["mode"],
                "projection_and_pair_block_ensemble_stability_and_silhouette",
            )
            self.assertEqual(
                contract["cluster_count_selection"]["selection"]["policy"][
                    "replicate_kind_counts"
                ],
                {"leave_one_pair_block_out": 4, "projection": 3},
            )
            self.assertEqual(manifest["algorithm"]["cluster_count"], 4)
            self.assertEqual(verify_bundle(output)["clusters"], 4)

    def test_v2_selector_runs_through_bundle_and_verifier(self) -> None:
        scenarios = {str(row["id"]): row for row in SCENARIOS}
        for scenario_id, expected_count in (
            ("continuous_direction_arc", 1),
            ("two_discrete_mechanisms", 2),
        ):
            with self.subTest(scenario=scenario_id), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                category_effects, _, _ = synthetic_effects(
                    scenarios[scenario_id], seed=250829
                )
                rows = [
                    component(
                        f"model.layers.{index}.self_attn.q_proj:rank{index}",
                        index,
                        "self_attn.q_proj",
                        1.0,
                    )
                    for index in range(len(category_effects))
                ]
                effects = root / "category-effects.npz"
                category_names = np.asarray(
                    sorted(row["id"] for row in category_contract()["categories"])
                )
                np.savez(
                    effects,
                    component_ids=np.asarray([row["component_id"] for row in rows]),
                    delta_h=category_effects.mean(axis=1).astype(np.float32),
                    category_names=category_names,
                    category_episode_counts=np.full(len(category_names), 8, dtype=np.int64),
                    delta_h_by_category=category_effects.astype(np.float32),
                )
                output = root / "bundle"
                manifest = build_bundle(
                    rows,
                    inventory_sha256=content_sha256(rows),
                    effects_path=effects,
                    output_dir=output,
                    cluster_count=8,
                    seed=250829,
                    projection_dim=64,
                    metadata_weight=0.0,
                    max_iterations=100,
                    anchors=["proposal_start"],
                    workers=[],
                    select_category_cluster_counts_by_projection=True,
                    category_cluster_count_selector_version="v2",
                    category_cluster_projection_replicates=8,
                    maximum_clusters_per_category=4,
                    minimum_stable_specific_components=1,
                    minimum_category_calibration_episodes=1,
                )
                receipt = manifest["algorithm"]["activation_category_compartments"][
                    "cluster_count_selection"
                ]
                self.assertEqual(
                    receipt["schema"],
                    "sch.vpd-category-cluster-count-selection-receipt.v2",
                )
                self.assertTrue(
                    receipt["mode"].endswith("_lower_tail_and_boundary_gap_v2")
                )
                self.assertTrue(
                    all(
                        count == expected_count
                        for count in receipt["selection"]["allocation"].values()
                    )
                )
                self.assertEqual(verify_bundle(output)["status"], "pass")

    def test_v2_selector_receipt_rejects_mutated_boundary_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                component(
                    f"model.layers.{index}.self_attn.q_proj:rank{index}",
                    index,
                    "self_attn.q_proj",
                    0.1 + index * 0.01,
                )
                for index in range(8)
            ]
            effects = root / "category-effects.npz"
            write_cluster_geometry_effects(effects, rows)
            output = root / "bundle"
            build_bundle(
                rows,
                inventory_sha256=content_sha256(rows),
                effects_path=effects,
                output_dir=output,
                cluster_count=2,
                seed=13,
                projection_dim=4,
                metadata_weight=0.0,
                max_iterations=50,
                anchors=["proposal_start"],
                workers=[],
                select_category_cluster_counts_by_projection=True,
                category_cluster_count_selector_version="v2",
                category_cluster_projection_replicates=3,
                maximum_clusters_per_category=2,
                minimum_category_cluster_size=2,
                minimum_category_calibration_episodes=1,
            )
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            receipt = manifest["algorithm"]["activation_category_compartments"][
                "cluster_count_selection"
            ]
            category_result = next(
                result
                for result in receipt["selection"]["categories"].values()
                if int(result["selection"]["cluster_count"]) > 1
            )
            candidate = next(
                row
                for row in category_result["candidates"]
                if int(row["cluster_count"])
                == int(category_result["selection"]["cluster_count"])
            )
            candidate["minimum_replicate_boundary_gap_ratio"] = 0.0
            category_result["selection"]["minimum_replicate_boundary_gap_ratio"] = 0.0
            body = dict(receipt)
            body.pop("selection_id")
            receipt["selection_id"] = content_sha256(body)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "candidate status differs"):
                verify_bundle(output)

    def test_cross_archive_cluster_count_selection_preserves_replicated_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                component(
                    f"model.layers.{index}.self_attn.q_proj:rank{index}",
                    index,
                    "self_attn.q_proj",
                    0.1 + index * 0.01,
                )
                for index in range(8)
            ]
            discovery = root / "discovery-effects.npz"
            validation = root / "cluster-validation-effects.npz"
            write_cluster_geometry_effects(discovery, rows)
            write_cluster_geometry_effects(validation, rows, scale=0.9)
            output = root / "bundle"
            manifest = build_bundle(
                rows,
                inventory_sha256=content_sha256(rows),
                effects_path=discovery,
                cluster_validation_effects_path=validation,
                require_cluster_validation_split_provenance=False,
                output_dir=output,
                cluster_count=2,
                seed=13,
                projection_dim=4,
                metadata_weight=0.0,
                max_iterations=50,
                anchors=["proposal_start"],
                workers=[],
                select_category_cluster_counts_by_projection=True,
                category_cluster_projection_replicates=3,
                maximum_clusters_per_category=2,
                minimum_category_cluster_size=2,
                minimum_category_calibration_episodes=1,
                require_category_cluster_sampling_stability=True,
                require_category_cluster_cross_archive_stability=True,
            )
            receipt = manifest["algorithm"]["activation_category_compartments"][
                "cluster_count_selection"
            ]
            self.assertEqual(
                receipt["mode"],
                "cross_archive_projection_and_pair_block_ensemble_stability_and_silhouette",
            )
            self.assertEqual(
                receipt["selection"]["allocation"],
                {"replayed": 2, "valid_authority": 2},
            )
            self.assertEqual(
                receipt["selection"]["policy"]["replicate_kind_counts"],
                {
                    "independent_cluster_validation_leave_one_pair_block_out": 4,
                    "independent_cluster_validation_projection": 3,
                    "leave_one_pair_block_out": 4,
                    "projection": 3,
                },
            )
            self.assertEqual(verify_bundle(output)["clusters"], 4)

    def test_cross_archive_cluster_count_selection_rejects_rearranged_membership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                component(
                    f"model.layers.{index}.self_attn.q_proj:rank{index}",
                    index,
                    "self_attn.q_proj",
                    0.1 + index * 0.01,
                )
                for index in range(8)
            ]
            discovery = root / "discovery-effects.npz"
            validation = root / "cluster-validation-effects.npz"
            write_cluster_geometry_effects(discovery, rows)
            write_cluster_geometry_effects(
                validation,
                rows,
                crossed_membership=True,
                scale=0.9,
            )
            output = root / "bundle"
            manifest = build_bundle(
                rows,
                inventory_sha256=content_sha256(rows),
                effects_path=discovery,
                cluster_validation_effects_path=validation,
                require_cluster_validation_split_provenance=False,
                output_dir=output,
                cluster_count=2,
                seed=13,
                projection_dim=4,
                metadata_weight=0.0,
                max_iterations=50,
                anchors=["proposal_start"],
                workers=[],
                select_category_cluster_counts_by_projection=True,
                category_cluster_projection_replicates=3,
                maximum_clusters_per_category=2,
                minimum_category_cluster_size=2,
                minimum_category_calibration_episodes=1,
                require_category_cluster_sampling_stability=True,
                require_category_cluster_cross_archive_stability=True,
            )
            receipt = manifest["algorithm"]["activation_category_compartments"][
                "cluster_count_selection"
            ]
            self.assertEqual(
                receipt["selection"]["allocation"],
                {"replayed": 1, "valid_authority": 1},
            )
            for category in ("replayed", "valid_authority"):
                candidate = receipt["selection"]["categories"][category]["candidates"][1]
                self.assertEqual(candidate["status"], "reject_replicate_instability")
            self.assertEqual(verify_bundle(output)["clusters"], 2)

    def test_specificity_threshold_change_changes_packet_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = synthetic_rows()
            effects = root / "category-effects.npz"
            write_category_effects(effects, rows)
            common = {
                "inventory_sha256": content_sha256(rows),
                "effects_path": effects,
                "cluster_count": 2,
                "seed": 13,
                "projection_dim": 3,
                "metadata_weight": 0.2,
                "max_iterations": 50,
                "anchors": ["proposal_start"],
                "workers": ["worker-a"],
            }
            first = build_bundle(
                rows,
                output_dir=root / "first",
                minimum_dominant_category_share=0.35,
                **common,
            )
            second = build_bundle(
                rows,
                output_dir=root / "second",
                minimum_dominant_category_share=0.40,
                **common,
            )
            self.assertEqual(
                [
                    json.loads(line)["cluster_id"]
                    for line in (root / "first" / "clusters.jsonl").read_text().splitlines()
                ],
                [
                    json.loads(line)["cluster_id"]
                    for line in (root / "second" / "clusters.jsonl").read_text().splitlines()
                ],
            )
            self.assertNotEqual(
                {entry["packet_id"] for entry in first["artifacts"]["packets"]},
                {entry["packet_id"] for entry in second["artifacts"]["packets"]},
            )

    def test_bundle_binds_independent_threshold_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = synthetic_rows()
            discovery = root / "discovery-effects.npz"
            calibration = root / "calibration-effects.npz"
            write_category_effects(discovery, rows)
            write_category_calibration_effects(calibration, rows)
            output = root / "bundle"
            manifest = build_bundle(
                rows,
                inventory_sha256=content_sha256(rows),
                effects_path=discovery,
                calibration_effects_path=calibration,
                require_calibration_split_provenance=False,
                output_dir=output,
                cluster_count=2,
                seed=13,
                projection_dim=3,
                metadata_weight=0.2,
                max_iterations=50,
                anchors=["proposal_start"],
                workers=["worker-a"],
                minimum_stable_specific_components=1,
                minimum_category_calibration_episodes=1,
            )
            contract = manifest["algorithm"]["activation_category_compartments"]
            calibration_receipt = contract["threshold_calibration"]
            self.assertEqual(calibration_receipt["status"], "pass_unattested_pilot")
            self.assertTrue(
                calibration_receipt["policy"][
                    "cross_archive_stratified_robustness_required"
                ]
            )
            self.assertEqual(
                calibration_receipt["selection"]["selected_metrics"][
                    "stable_specific_count"
                ],
                6,
            )
            self.assertEqual(
                contract["specificity_thresholds"],
                calibration_receipt["selection"]["selected_thresholds"],
            )
            for entry in manifest["artifacts"]["packets"]:
                packet = json.loads((output / entry["packet_path"]).read_text())
                self.assertEqual(
                    packet["sae_study"]["activation_compartment"][
                        "threshold_calibration"
                    ]["calibration_id"],
                    calibration_receipt["calibration_id"],
                )
            self.assertEqual(verify_bundle(output)["status"], "pass")

    def test_global_calibration_pass_reroutes_minority_component_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                component(
                    f"model.layers.{index}.self_attn.q_proj:rank{index}",
                    index,
                    "self_attn.q_proj",
                    0.10 + index * 0.001,
                )
                for index in range(20)
            ]
            category_names = np.asarray(["replayed", "valid_authority"])
            discovery_delta = np.zeros((20, 2, 3), dtype=np.float32)
            calibration_delta = np.zeros((20, 2, 3), dtype=np.float32)
            for index in range(20):
                if index < 10:
                    discovery_delta[index] = [[0.1, 0.0, 0.0], [2.0, 0.1, 0.0]]
                    calibration_delta[index] = [[0.12, 0.0, 0.0], [1.8, 0.08, 0.0]]
                else:
                    discovery_delta[index] = [[0.0, 2.0, 0.1], [0.0, 0.1, 0.0]]
                    calibration_delta[index] = [[0.0, 1.8, 0.08], [0.0, 0.12, 0.0]]
            # One of twenty components changes its strongest category. Aggregate
            # replication is still 95%, so threshold calibration legitimately passes.
            calibration_delta[0] = [[1.8, 0.08, 0.0], [0.12, 0.0, 0.0]]
            counts = np.asarray([10, 10], dtype=np.int64)
            discovery = root / "discovery-effects.npz"
            calibration = root / "calibration-effects.npz"
            for path, values in (
                (discovery, discovery_delta),
                (calibration, calibration_delta),
            ):
                np.savez(
                    path,
                    component_ids=np.asarray([row["component_id"] for row in rows]),
                    delta_h=np.average(values, axis=1, weights=counts).astype(np.float32),
                    category_names=category_names,
                    category_episode_counts=counts,
                    delta_h_by_category=values,
                )
            output = root / "bundle"
            manifest = build_bundle(
                rows,
                inventory_sha256=content_sha256(rows),
                effects_path=discovery,
                calibration_effects_path=calibration,
                require_calibration_split_provenance=False,
                output_dir=output,
                cluster_count=3,
                seed=13,
                projection_dim=3,
                metadata_weight=0.0,
                max_iterations=50,
                anchors=["proposal_start"],
                workers=[],
                minimum_stable_specific_components=1,
                minimum_category_calibration_episodes=1,
            )
            calibration_metrics = manifest["algorithm"][
                "activation_category_compartments"
            ]["threshold_calibration"]["selection"]["selected_metrics"]
            self.assertEqual(calibration_metrics["stable_specific_count"], 19)
            self.assertEqual(calibration_metrics["symmetric_replication_rate"], 0.95)
            screen = manifest["algorithm"]["activation_category_compartments"][
                "per_component_cross_archive_admission"
            ]
            self.assertEqual(screen["admitted_specific_count"], 19)
            self.assertEqual(screen["rerouted_specific_count"], 1)
            assignments = [
                json.loads(line)
                for line in (output / "assignments.jsonl").read_text().splitlines()
            ]
            rerouted = [
                row
                for row in assignments
                if row["cross_archive_activation_category_rejection_reason"]
                == "calibration_category_mismatch"
            ]
            self.assertEqual(len(rerouted), 1)
            self.assertEqual(
                rerouted[0]["pre_cross_archive_activation_category"],
                "valid_authority",
            )
            self.assertEqual(rerouted[0]["activation_category"], "shared_cross_category")
            self.assertFalse(rerouted[0]["cross_archive_activation_category_admitted"])
            self.assertEqual(verify_bundle(output)["status"], "pass")

            rerouted[0]["cross_archive_activation_category_admitted"] = True
            rerouted[0]["cross_archive_activation_category_rejection_reason"] = None
            assignment_path = output / "assignments.jsonl"
            assignment_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in assignments),
                encoding="utf-8",
            )
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"]["assignments"]["sha256"] = sha256_file(
                assignment_path
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "per-component cross-archive admission is not reproducible"
            ):
                verify_bundle(output)

    def test_bundle_requires_and_binds_verified_split_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = synthetic_rows()
            discovery = root / "discovery-effects.npz"
            calibration = root / "calibration-effects.npz"
            write_category_effects(discovery, rows)
            write_category_calibration_effects(calibration, rows)
            common = {
                "inventory_sha256": content_sha256(rows),
                "effects_path": discovery,
                "calibration_effects_path": calibration,
                "cluster_count": 2,
                "seed": 13,
                "projection_dim": 3,
                "metadata_weight": 0.2,
                "max_iterations": 50,
                "anchors": ["proposal_start"],
                "workers": [],
                "minimum_stable_specific_components": 1,
                "minimum_category_calibration_episodes": 1,
            }
            with self.assertRaisesRegex(ValueError, "requires discovery and calibration"):
                build_bundle(rows, output_dir=root / "missing", **common)
            discovery_manifest = root / "discovery-effect-merge.json"
            calibration_manifest = root / "calibration-effect-merge.json"
            write_effect_merge_manifest(
                discovery_manifest,
                discovery,
                role="discovery",
            )
            write_effect_merge_manifest(
                calibration_manifest,
                calibration,
                role="calibration",
            )
            output = root / "verified"
            manifest = build_bundle(
                rows,
                output_dir=output,
                effects_merge_manifest_path=discovery_manifest,
                calibration_effects_merge_manifest_path=calibration_manifest,
                **common,
            )
            calibration_receipt = manifest["algorithm"][
                "activation_category_compartments"
            ]["threshold_calibration"]
            self.assertEqual(calibration_receipt["status"], "pass")
            self.assertEqual(
                calibration_receipt["split_provenance"]["status"], "verified"
            )
            self.assertEqual(
                manifest["source"]["activation_category_split_id"], "2" * 64
            )
            self.assertEqual(verify_bundle(output)["status"], "pass")

    def test_threshold_calibration_rejects_same_or_unstable_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = synthetic_rows()
            discovery = root / "discovery-effects.npz"
            unstable = root / "unstable-effects.npz"
            write_category_effects(discovery, rows)
            write_category_calibration_effects(unstable, rows, swap_categories=True)
            common = {
                "inventory_sha256": content_sha256(rows),
                "effects_path": discovery,
                "cluster_count": 2,
                "seed": 13,
                "projection_dim": 3,
                "metadata_weight": 0.2,
                "max_iterations": 50,
                "anchors": ["proposal_start"],
                "workers": [],
                "minimum_stable_specific_components": 1,
                "minimum_category_calibration_episodes": 1,
            }
            with self.assertRaisesRegex(ValueError, "must be distinct"):
                build_bundle(
                    rows,
                    calibration_effects_path=discovery,
                    require_calibration_split_provenance=False,
                    output_dir=root / "same",
                    **common,
                )
            with self.assertRaisesRegex(ValueError, "independent replication policy"):
                build_bundle(
                    rows,
                    calibration_effects_path=unstable,
                    require_calibration_split_provenance=False,
                    output_dir=root / "unstable",
                    **common,
                )

    def test_v2_threshold_calibration_requires_cross_template_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = synthetic_rows()
            discovery = root / "discovery-effects.npz"
            calibration = root / "calibration-effects.npz"
            write_category_effects(discovery, rows)
            write_category_calibration_effects(calibration, rows)
            discovery_manifest = root / "discovery-merge.json"
            calibration_manifest = root / "calibration-merge.json"
            write_effect_merge_manifest(
                discovery_manifest,
                discovery,
                role="discovery",
                semantic_template_hash="8" * 64,
            )
            write_effect_merge_manifest(
                calibration_manifest,
                calibration,
                role="calibration",
                semantic_template_hash="9" * 64,
            )
            common = {
                "inventory_sha256": content_sha256(rows),
                "effects_path": discovery,
                "calibration_effects_path": calibration,
                "effects_merge_manifest_path": discovery_manifest,
                "calibration_effects_merge_manifest_path": calibration_manifest,
                "cluster_count": 2,
                "seed": 13,
                "projection_dim": 3,
                "metadata_weight": 0.2,
                "max_iterations": 50,
                "anchors": ["proposal_start"],
                "workers": [],
                "minimum_stable_specific_components": 1,
                "minimum_category_calibration_episodes": 1,
            }
            manifest = build_bundle(rows, output_dir=root / "pass", **common)
            provenance = manifest["algorithm"]["activation_category_compartments"][
                "threshold_calibration"
            ]["split_provenance"]
            self.assertEqual(
                provenance["discovery"][
                    "activation_category_split_template_group_set_sha256"
                ],
                "8" * 64,
            )
            write_effect_merge_manifest(
                calibration_manifest,
                calibration,
                role="calibration",
                semantic_template_hash="8" * 64,
            )
            with self.assertRaisesRegex(ValueError, "template-group sets are not distinct"):
                build_bundle(rows, output_dir=root / "reject", **common)

    def test_verifier_recomputes_threshold_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = synthetic_rows()
            discovery = root / "discovery-effects.npz"
            calibration = root / "calibration-effects.npz"
            write_category_effects(discovery, rows)
            write_category_calibration_effects(calibration, rows)
            output = root / "bundle"
            build_bundle(
                rows,
                inventory_sha256=content_sha256(rows),
                effects_path=discovery,
                calibration_effects_path=calibration,
                require_calibration_split_provenance=False,
                output_dir=output,
                cluster_count=2,
                seed=13,
                projection_dim=3,
                metadata_weight=0.2,
                max_iterations=50,
                anchors=["proposal_start"],
                workers=[],
                minimum_stable_specific_components=1,
                minimum_category_calibration_episodes=1,
            )
            assignment_path = output / "assignments.jsonl"
            assignments = [
                json.loads(line) for line in assignment_path.read_text().splitlines()
            ]
            calibration_strengths = assignments[0][
                "calibration_activation_category_strengths"
            ]
            calibration_strengths["replayed"], calibration_strengths["valid_authority"] = (
                calibration_strengths["valid_authority"],
                calibration_strengths["replayed"],
            )
            assignment_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in assignments),
                encoding="utf-8",
            )
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"]["assignments"]["sha256"] = __import__(
                "experiments.authz_mechinterp.cluster_vpd_for_sae",
                fromlist=["sha256_file"],
            ).sha256_file(assignment_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "calibration strength ratio differs"):
                verify_bundle(output)

    def test_verifier_recomputes_compartment_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = synthetic_rows()
            effects = root / "category-effects.npz"
            write_category_effects(effects, rows)
            output = root / "bundle"
            build_bundle(
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
            )
            assignment_path = output / "assignments.jsonl"
            assignments = [
                json.loads(line) for line in assignment_path.read_text().splitlines()
            ]
            assignments[0]["dominant_activation_category_share"] = 0.0
            assignment_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in assignments),
                encoding="utf-8",
            )
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"]["assignments"]["sha256"] = __import__(
                "experiments.authz_mechinterp.cluster_vpd_for_sae",
                fromlist=["sha256_file"],
            ).sha256_file(assignment_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "activation diagnostic"):
                verify_bundle(output)


if __name__ == "__main__":
    unittest.main()
