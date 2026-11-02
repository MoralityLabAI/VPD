from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.authz_mechinterp.cluster_vpd_for_sae import build_bundle, content_sha256
from experiments.authz_mechinterp.compute_vpd_effect_shard import (
    select_episodes,
    verify_dataset_split_role,
    verify_packet,
)
from experiments.authz_mechinterp.build_activation_category_splits import (
    build_split_bundle,
)
from experiments.authz_mechinterp.activation_categories import (
    activation_category,
    category_contract,
    categorize_episodes,
)
from test_vpd_activation_category_splits import synthetic_episodes
from test_vpd_sae_clusters import synthetic_rows
from experiments.authz_mechinterp.train_sae import (
    activation_compartment_selection,
    balanced_activation_compartment_selection,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    DEFAULT_CATEGORY_CALIBRATION_GRID,
    DEFAULT_MAXIMUM_CATEGORY_NORMALIZED_ENTROPY,
    DEFAULT_MINIMUM_CATEGORY_CALIBRATION_EPISODES,
    DEFAULT_MINIMUM_CATEGORY_DIRECTION_COSINE,
    DEFAULT_MINIMUM_CATEGORY_STRENGTH_RATIO,
    DEFAULT_MINIMUM_CATEGORY_CONDITION_COVERAGE,
    DEFAULT_MINIMUM_CONDITION_DIRECTION_COSINE,
    DEFAULT_MINIMUM_CONDITION_RELATIVE_NORM,
    DEFAULT_MINIMUM_CATEGORY_TEMPLATE_COVERAGE,
    DEFAULT_MINIMUM_TEMPLATE_DIRECTION_COSINE,
    DEFAULT_MINIMUM_TEMPLATE_RELATIVE_NORM,
    DEFAULT_MINIMUM_CATEGORY_ROBUSTNESS_SKETCH_DIMENSION,
    DEFAULT_MINIMUM_CATEGORY_PAIR_BLOCK_DIRECTION_COVERAGE,
    DEFAULT_MINIMUM_PAIR_BLOCK_DIRECTION_COSINE,
    DEFAULT_MINIMUM_PAIR_BLOCK_RELATIVE_NORM,
    DEFAULT_MINIMUM_CATEGORY_JACKKNIFE_STABILITY,
    DEFAULT_MINIMUM_CATEGORY_DOMINANCE_MARGIN,
    DEFAULT_MINIMUM_CATEGORY_REPLICATION_RATE,
    DEFAULT_MINIMUM_DOMINANT_CATEGORY_SHARE,
    DEFAULT_MINIMUM_STABLE_SPECIFIC_COMPONENTS,
    sha256_file,
    write_jsonl,
)


class ComputeVpdEffectShardTests(unittest.TestCase):
    def test_v2_protocol_requires_condition_and_template_robustness(self) -> None:
        protocol_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "evals"
            / "vpd_activation_category_compartments_v2.json"
        )
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        screen = protocol["component_compartments"][
            "within_category_condition_robustness_screen"
        ]
        self.assertEqual(
            screen["minimum_condition_coverage"],
            DEFAULT_MINIMUM_CATEGORY_CONDITION_COVERAGE,
        )
        self.assertEqual(
            screen["minimum_condition_direction_cosine"],
            DEFAULT_MINIMUM_CONDITION_DIRECTION_COSINE,
        )
        self.assertEqual(
            screen["minimum_condition_relative_norm"],
            DEFAULT_MINIMUM_CONDITION_RELATIVE_NORM,
        )
        self.assertTrue(screen["required_for_production_category_bundle"])
        self.assertEqual(
            screen["failed_specific_routes_to"], "shared_cross_category"
        )
        self.assertFalse(screen["model_outcomes_used"])
        self.assertEqual(
            protocol["effect_capture"]["receipt_schema"],
            "sch.vpd-effect-shard-receipt.v5",
        )
        template_screen = protocol["component_compartments"][
            "within_category_template_robustness_screen"
        ]
        self.assertEqual(
            template_screen["minimum_template_group_coverage"],
            DEFAULT_MINIMUM_CATEGORY_TEMPLATE_COVERAGE,
        )
        self.assertEqual(
            template_screen["minimum_template_group_direction_cosine"],
            DEFAULT_MINIMUM_TEMPLATE_DIRECTION_COSINE,
        )
        self.assertEqual(
            template_screen["minimum_template_group_relative_norm"],
            DEFAULT_MINIMUM_TEMPLATE_RELATIVE_NORM,
        )
        self.assertTrue(
            template_screen["required_for_production_category_bundle"]
        )
        self.assertEqual(
            screen["minimum_countsketch_dimension"],
            DEFAULT_MINIMUM_CATEGORY_ROBUSTNESS_SKETCH_DIMENSION,
        )
        self.assertEqual(
            template_screen["minimum_countsketch_dimension"],
            DEFAULT_MINIMUM_CATEGORY_ROBUSTNESS_SKETCH_DIMENSION,
        )
        fidelity = protocol["effect_capture"]["sketch_fidelity_admission"]
        self.assertEqual(
            fidelity["benchmark_id"],
            "4ed9790051cabd86dd4b2faffd68d57bfd1a8fbead2e8ea0df0b107ab29e04dd",
        )
        self.assertEqual(fidelity["selected_countsketch_dimension"], 256)
        self.assertTrue(
            fidelity["production_archives_below_selected_dimension_rejected"]
        )
        block_screen = protocol["component_compartments"][
            "within_category_pair_block_direction_robustness_screen"
        ]
        self.assertEqual(
            block_screen["minimum_pair_block_coverage"],
            DEFAULT_MINIMUM_CATEGORY_PAIR_BLOCK_DIRECTION_COVERAGE,
        )
        self.assertEqual(
            block_screen["minimum_pair_block_direction_cosine"],
            DEFAULT_MINIMUM_PAIR_BLOCK_DIRECTION_COSINE,
        )
        self.assertEqual(
            block_screen["minimum_pair_block_relative_norm"],
            DEFAULT_MINIMUM_PAIR_BLOCK_RELATIVE_NORM,
        )
        self.assertTrue(block_screen["required_for_production_category_bundle"])
        self.assertTrue(block_screen["reuses_signed_uncertainty_arrays"])
        calibration = protocol["component_compartments"]["threshold_calibration"]
        geometry = protocol["component_compartments"][
            "within_category_clustering_geometry"
        ]
        self.assertEqual(
            geometry["direction_weighting"], "sqrt_category_strength_share"
        )
        self.assertEqual(
            geometry["synthetic_benchmark_id"],
            "792a7a337cedd085f8023912178a180d7010b6a61dee9ef69fcd975f1161c8fb",
        )
        self.assertTrue(
            geometry[
                "arbitrarily_small_background_directions_have_arbitrarily_small_weight"
            ]
        )
        count_selection = protocol["component_compartments"][
            "within_category_cluster_count_selection"
        ]
        self.assertEqual(count_selection["discovery_projection_replicates"], 8)
        self.assertEqual(
            count_selection["independent_cluster_validation_projection_replicates"],
            8,
        )
        self.assertEqual(
            count_selection["synthetic_benchmark_id"],
            "6e8fd78406ffea0cb9ba9aea73be858d5a3dc7e8dd5c7c787d7cc51fc49e1b25",
        )
        self.assertTrue(count_selection["required_for_production_category_bundle"])
        self.assertTrue(
            count_selection[
                "pair_block_sampling_stability_required_for_production_category_bundle"
            ]
        )
        self.assertTrue(
            count_selection[
                "independent_cluster_validation_archive_required_for_production_category_bundle"
            ]
        )
        self.assertEqual(
            calibration["minimum_dominant_strength_ratio"],
            DEFAULT_MINIMUM_CATEGORY_STRENGTH_RATIO,
        )
        self.assertTrue(
            calibration["same_stratified_robustness_gates_required_in_both_archives"]
        )
        self.assertTrue(
            calibration[
                "stratified_robustness_rejected_specific_components_do_not_count_as_stable"
            ]
        )

    def test_effect_dataset_role_is_bound_to_verified_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            rows = synthetic_episodes()
            write_jsonl(source, rows)
            output = root / "splits"
            manifest = build_split_bundle(
                rows,
                source_sha256=sha256_file(source),
                output_dir=output,
                seed=19,
                minimum_episodes_per_category_per_split=1,
            )
            discovery = output / manifest["splits"]["discovery"]["artifact"]["path"]
            provenance = verify_dataset_split_role(
                discovery,
                output / "split_manifest.json",
                "discovery",
            )
            self.assertEqual(provenance["activation_category_split_id"], manifest["split_id"])
            self.assertEqual(provenance["activation_category_split_role"], "discovery")
            with self.assertRaisesRegex(ValueError, "final-evaluation rows"):
                verify_dataset_split_role(
                    output / manifest["splits"]["final_evaluation"]["artifact"]["path"],
                    output / "split_manifest.json",
                    "final_evaluation",
                )

    def test_machine_readable_category_protocol_matches_code_defaults(self) -> None:
        protocol_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "evals"
            / "vpd_activation_category_compartments_v1.json"
        )
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        self.assertEqual(protocol["category_contract"], category_contract())
        thresholds = protocol["component_compartments"]["specificity_thresholds"]
        self.assertEqual(
            thresholds["minimum_dominant_share"],
            DEFAULT_MINIMUM_DOMINANT_CATEGORY_SHARE,
        )
        self.assertEqual(
            thresholds["minimum_dominance_margin"],
            DEFAULT_MINIMUM_CATEGORY_DOMINANCE_MARGIN,
        )
        self.assertEqual(
            thresholds["maximum_normalized_entropy"],
            DEFAULT_MAXIMUM_CATEGORY_NORMALIZED_ENTROPY,
        )
        calibration = protocol["component_compartments"]["threshold_calibration"]
        self.assertEqual(
            calibration["minimum_replication_rate"],
            DEFAULT_MINIMUM_CATEGORY_REPLICATION_RATE,
        )
        self.assertEqual(
            calibration["minimum_stable_specific_components"],
            DEFAULT_MINIMUM_STABLE_SPECIFIC_COMPONENTS,
        )
        self.assertEqual(
            calibration["minimum_episodes_per_category_per_archive"],
            DEFAULT_MINIMUM_CATEGORY_CALIBRATION_EPISODES,
        )
        self.assertEqual(
            calibration["minimum_signed_direction_cosine"],
            DEFAULT_MINIMUM_CATEGORY_DIRECTION_COSINE,
        )
        self.assertEqual(calibration["candidate_grid"], DEFAULT_CATEGORY_CALIBRATION_GRID)
        uncertainty = protocol["component_compartments"][
            "within_split_uncertainty_screen"
        ]
        self.assertEqual(
            uncertainty["minimum_same_compartment_stability"],
            DEFAULT_MINIMUM_CATEGORY_JACKKNIFE_STABILITY,
        )
        self.assertTrue(uncertainty["required_for_production_category_bundle"])
        self.assertEqual(
            protocol["effect_capture"]["receipt_schema"],
            "sch.vpd-effect-shard-receipt.v3",
        )
        pack_manifest = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "artifacts"
                / "vpd_activation_category_pack_v1"
                / "artifact_manifest.json"
            ).read_text(encoding="utf-8")
        )
        data_pack = protocol["data_pack"]
        self.assertEqual(data_pack["pack_id"], pack_manifest["pack_id"])
        self.assertEqual(data_pack["split_id"], pack_manifest["split"]["split_id"])
        self.assertEqual(data_pack["rows"], pack_manifest["coverage"]["rows"])
        self.assertEqual(
            data_pack["activation_categories"],
            pack_manifest["coverage"]["activation_categories"],
        )

    def test_authorization_conditions_map_to_activation_categories(self) -> None:
        rows = [
            {
                "episode_id": condition,
                "condition": condition,
                "expected_gate_decision": "allow" if condition == "valid_visible" else "deny",
            }
            for condition in (
                "valid_visible",
                "user_pasted_valid",
                "forged_host_visible",
                "expired_host_visible",
                "revoked_host_visible",
                "replayed_host_visible",
                "scope_mismatch_visible",
            )
        ]
        grouped = categorize_episodes(rows)
        self.assertEqual(
            set(grouped),
            {
                "valid_authority",
                "missing_or_untrusted_authority",
                "signature_invalid",
                "time_inactive",
                "revoked",
                "replayed",
                "scope_invalid",
            },
        )
        self.assertEqual(activation_category(rows[0]), "valid_authority")
        mask, compartment = activation_compartment_selection(rows, "replayed")
        self.assertEqual(mask.tolist(), [False, False, False, False, False, True, False])
        self.assertEqual(compartment["protocol"], "nonce_freshness_rejection")
        self.assertEqual(compartment["selected_rows_before_finite_filter"], 1)

    def test_balanced_shared_dictionary_selection_is_deterministic(self) -> None:
        metadata = [
            {
                "episode_id": f"valid-{index}",
                "condition": "valid_visible",
                "expected_gate_decision": "allow",
            }
            for index in range(3)
        ] + [
            {
                "episode_id": f"replay-{index}",
                "condition": "replayed_host_visible",
                "expected_gate_decision": "deny",
            }
            for index in range(2)
        ]
        mask, compartment = balanced_activation_compartment_selection(
            metadata, seed=19
        )
        self.assertEqual(int(mask.sum()), 4)
        self.assertEqual(compartment["activation_category"], "shared_cross_category")
        self.assertEqual(compartment["rows_per_category"], 2)
        selected_ids = {
            row["episode_id"]
            for row, selected in zip(metadata, mask.tolist())
            if selected
        }
        reversed_rows = list(reversed(metadata))
        reversed_mask, _ = balanced_activation_compartment_selection(
            reversed_rows, seed=19
        )
        self.assertEqual(
            selected_ids,
            {
                row["episode_id"]
                for row, selected in zip(reversed_rows, reversed_mask.tolist())
                if selected
            },
        )

    def test_stratified_selection_is_order_independent(self) -> None:
        rows = [
            {
                "episode_id": f"episode-{index}",
                "template_group": f"family-{index % 3}",
                "condition": f"condition-{index % 2}",
            }
            for index in range(30)
        ]
        first = select_episodes(rows, maximum=12, seed=17)
        second = select_episodes(list(reversed(rows)), maximum=12, seed=17)
        self.assertEqual(
            [row["episode_id"] for row in first],
            [row["episode_id"] for row in second],
        )
        self.assertEqual(
            {(row["template_group"], row["condition"]) for row in first},
            {(f"family-{i}", f"condition-{j}") for i in range(3) for j in range(2)},
        )

    def test_packet_verification_rejects_changed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = build_bundle(
                synthetic_rows(),
                inventory_sha256="inventory",
                effects_path=None,
                output_dir=root,
                cluster_count=2,
                seed=13,
                projection_dim=3,
                metadata_weight=0.2,
                max_iterations=50,
                anchors=["proposal_start"],
                workers=["worker-a"],
            )
            entry = manifest["artifacts"]["packets"][0]
            packet_path = root / entry["packet_path"]
            packet = verify_packet(packet_path, entry["packet_sha256"])
            self.assertEqual(packet["packet_id"], entry["packet_id"])
            changed = json.loads(packet_path.read_text())
            changed["effect_refinement"]["effect_space_id"] = "layer_1__proposal_start"
            packet_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file hash"):
                verify_packet(packet_path, entry["packet_sha256"])


if __name__ == "__main__":
    unittest.main()
