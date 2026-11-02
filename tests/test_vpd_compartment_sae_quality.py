from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.authz_mechinterp.benchmark_compartment_sae_quality import (
    CATEGORIES,
    build_benchmark,
    clean_dictionaries,
    synthetic_holdout,
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, write_json
from experiments.authz_mechinterp.evaluate_compartment_sae_quality import (
    DEFAULT_QUALITY_POLICY,
    DEFAULT_QUALITY_SAMPLING_POLICY,
    build_quality_report,
    evaluate_compartment_bundle,
    evaluate_compartment_bundle_streaming,
    streaming_quality_resource_bounds,
    validate_quality_sampling_receipt,
    verify_quality_report,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = (
    ROOT / "artifacts" / "vpd_compartment_sae_quality_benchmark_v1" / "report.json"
)


def production_sampling_receipt() -> dict:
    return {
        "schema": "sch.vpd-compartment-sae-quality-sampling.v1",
        "seed": 270829,
        "selection_basis": "stable_hash_of_split_category_and_episode_id",
        "observable_activation_category_only": True,
        "model_outcomes_used_for_selection": False,
        "source_rows": 65536,
        "source_episodes": 512,
        "rows_per_episode": 128,
        "episodes_per_category": 64,
        "selected_rows": 65536,
        "selected_episodes": 512,
        "selected_episode_id_set_sha256_by_category": {
            category: "1" * 64 for category in CATEGORIES
        },
        "selected_pair_id_set_sha256_by_category": {
            category: "2" * 64 for category in CATEGORIES
        },
        "selected_row_index_sha256": "3" * 64,
        "whole_prompt_windows_only": True,
        "final_evaluation_rows_used": False,
    }


def admitted_training_provenance() -> dict:
    return {
        "schema": "sch.vpd-compartment-sae-dictionary-bundle-authorization.v1",
        "status": "pass",
        "training_plan_id": "4" * 64,
        "training_plan_sha256": "5" * 64,
        "dictionary_bundle_id": "6" * 64,
        "dictionary_bundle_sha256": "7" * 64,
    }


class CompartmentSaeQualityTests(unittest.TestCase):
    def test_quality_sampling_policy_is_frozen_and_pair_identity_bound(self) -> None:
        validate_quality_sampling_receipt(production_sampling_receipt())
        weak = production_sampling_receipt()
        weak["episodes_per_category"] = 1
        with self.assertRaisesRegex(ValueError, "weakens the frozen sampling"):
            validate_quality_sampling_receipt(weak)

    def test_protocol_config_pins_the_executable_quality_policy(self) -> None:
        config = json.loads(
            (
                ROOT
                / "configs"
                / "evals"
                / "vpd_activation_category_compartments_v2.json"
            ).read_text(encoding="utf-8")
        )
        gate = config["sae_training"]["quality_gate"]
        self.assertEqual(gate["synthetic_benchmark_id"], json.loads(
            BENCHMARK.read_text(encoding="utf-8")
        )["benchmark_id"])
        for name, value in DEFAULT_QUALITY_POLICY.items():
            self.assertEqual(gate[name], value)
        sampling = gate["sampling"]
        self.assertEqual(sampling["seed"], DEFAULT_QUALITY_SAMPLING_POLICY["seed"])
        self.assertEqual(
            sampling["episodes_per_category"],
            DEFAULT_QUALITY_SAMPLING_POLICY["episodes_per_category"],
        )
        self.assertEqual(
            sampling["activation_rows_per_episode"],
            DEFAULT_QUALITY_SAMPLING_POLICY["rows_per_episode"],
        )
        self.assertEqual(
            sampling["selected_rows"],
            DEFAULT_QUALITY_SAMPLING_POLICY["selected_rows"],
        )
        bounds = streaming_quality_resource_bounds(
            hidden_size=4096,
            maximum_features=8192,
            categories=8,
            batch_rows=gate["execution"]["activation_batch_rows"],
        )
        self.assertEqual(
            gate["execution"][
                "bounded_context_pilot_estimated_numeric_peak_bytes_excluding_python_and_npz_overhead"
            ],
            bounds[
                "estimated_numeric_peak_bytes_excluding_python_and_npz_overhead"
            ],
        )

    def test_durable_synthetic_benchmark_reproduces(self) -> None:
        verification = verify_benchmark(BENCHMARK)
        self.assertEqual(verification["status"], "pass")
        self.assertEqual(
            verification["benchmark_id"],
            "01d7c090a7af3f2ffe2046cf9474cb849b6774128172af28fc747b3620370d38",
        )

    def test_clean_passes_and_corruptions_fail_closed(self) -> None:
        benchmark = build_benchmark()
        observed = {
            row["scenario_id"]: row["observed_status"]
            for row in benchmark["scenarios"]
        }
        self.assertEqual(observed["clean_replicated_compartments"], "pass")
        for scenario in (
            "collapsed_specific_to_shared",
            "cyclic_compartment_swap",
            "dead_feature_overgrowth",
            "unstable_replica_basis",
        ):
            self.assertEqual(observed[scenario], "reject")
        checks = {
            row["scenario_id"]: row["checks"] for row in benchmark["scenarios"]
        }
        self.assertTrue(checks["collapsed_specific_to_shared"]["specificity_gate_rejected"])
        self.assertTrue(checks["cyclic_compartment_swap"]["target_reconstruction_rejected"])
        self.assertTrue(checks["dead_feature_overgrowth"]["dead_feature_gate_rejected"])
        self.assertTrue(checks["unstable_replica_basis"]["replica_gate_rejected"])

    def test_streaming_evaluation_matches_in_memory_with_bounded_reads(self) -> None:
        activations, categories, bases = synthetic_holdout(260829)
        dictionaries = clean_dictionaries(bases)
        expected = evaluate_compartment_bundle(
            activations=activations,
            categories=categories,
            dictionaries=dictionaries,
            required_specific_categories=list(dict.fromkeys(categories.tolist())),
        )
        by_id = {row["dictionary_id"]: row for row in dictionaries}
        descriptors = [
            {
                "dictionary_id": row["dictionary_id"],
                "compartment": row["compartment"],
                "replica_id": row["replica_id"],
                "initialization_seed": int(row["replica_id"].removeprefix("seed-")),
                "data_selection_seed": 17,
            }
            for row in dictionaries
        ]
        read_sizes: list[int] = []
        loads: list[str] = []

        def read_rows(indices):
            read_sizes.append(len(indices))
            return activations[indices]

        def load_dictionary(descriptor):
            loads.append(descriptor["dictionary_id"])
            return by_id[descriptor["dictionary_id"]]

        observed = evaluate_compartment_bundle_streaming(
            row_count=len(activations),
            hidden_size=activations.shape[1],
            read_rows=read_rows,
            categories=categories,
            dictionary_descriptors=descriptors,
            load_dictionary=load_dictionary,
            required_specific_categories=list(dict.fromkeys(categories.tolist())),
            batch_rows=17,
        )
        self.assertEqual(observed["status"], expected["status"])
        self.assertEqual(loads, [
            row["dictionary_id"]
            for compartment in ["shared_cross_category", *sorted(set(categories))]
            for row in sorted(
                [item for item in dictionaries if item["compartment"] == compartment],
                key=lambda item: item["replica_id"],
            )
        ])
        self.assertLessEqual(max(read_sizes), 17)
        self.assertEqual(sum(read_sizes), len(activations) * len(dictionaries))
        expected_rows = {row["dictionary_id"]: row for row in expected["dictionary_results"]}
        for row in observed["dictionary_results"]:
            reference = expected_rows[row["dictionary_id"]]
            self.assertEqual(row["checks"], reference["checks"])
            self.assertAlmostEqual(row["dead_feature_fraction"], reference["dead_feature_fraction"])
            self.assertAlmostEqual(row["mean_l0"], reference["mean_l0"])
            for category, value in row["explained_variance_by_category"].items():
                self.assertAlmostEqual(
                    value,
                    reference["explained_variance_by_category"][category],
                    places=6,
                )

    def test_quality_receipt_is_calibration_bound_and_content_addressed(self) -> None:
        clean = build_benchmark()["scenarios"][0]["result"]
        sources = [
            {
                "dictionary_id": row["dictionary_id"],
                "dictionary_manifest_sha256": "a" * 64,
                "dictionary_sha256": "b" * 64,
                "compartment": row["compartment"],
                "replica_id": row["replica_id"],
                "initialization_seed": int(row["replica_id"].removeprefix("seed-")),
                "data_selection_seed": 17,
                "capacity_authorization": {
                    "status": "pass",
                    "capacity_plan_id": "c" * 64,
                    "profile_id": "bounded_context_pilot",
                    "activation_category": row["compartment"],
                },
            }
            for row in clean["dictionary_results"]
        ]
        report = build_quality_report(
            result=clean,
            dictionary_sources=sources,
            activation_category_split={
                "activation_category_split_id": "d" * 64,
                "activation_category_split_role": "calibration",
            },
            holdout_capture={
                "activation_capture_sha256": "e" * 64,
                "capture_manifest_sha256": "f" * 64,
                "quality_sampling": production_sampling_receipt(),
            },
            training_provenance=admitted_training_provenance(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quality-report.json"
            write_json(path, report)
            self.assertEqual(verify_quality_report(path)["status"], "pass")
            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["result"]["status"] = "reject"
            write_json(path, tampered)
            with self.assertRaisesRegex(ValueError, "content address"):
                verify_quality_report(path)

    def test_readdressed_receipt_cannot_lie_about_metric_checks(self) -> None:
        clean = build_benchmark()["scenarios"][0]["result"]
        sources = [
            {
                "dictionary_id": row["dictionary_id"],
                "dictionary_manifest_sha256": "a" * 64,
                "dictionary_sha256": "b" * 64,
                "compartment": row["compartment"],
                "replica_id": row["replica_id"],
                "initialization_seed": int(row["replica_id"].removeprefix("seed-")),
                "data_selection_seed": 17,
                "capacity_authorization": {
                    "status": "pass",
                    "capacity_plan_id": "c" * 64,
                    "profile_id": "bounded_context_pilot",
                    "activation_category": row["compartment"],
                },
            }
            for row in clean["dictionary_results"]
        ]
        report = build_quality_report(
            result=clean,
            dictionary_sources=sources,
            activation_category_split={
                "activation_category_split_id": "d" * 64,
                "activation_category_split_role": "calibration",
            },
            holdout_capture={
                "activation_capture_sha256": "e" * 64,
                "capture_manifest_sha256": "f" * 64,
                "quality_sampling": production_sampling_receipt(),
            },
            training_provenance=admitted_training_provenance(),
        )
        report["result"]["dictionary_results"][0]["checks"][
            "dead_feature_fraction"
        ] = False
        body = {key: value for key, value in report.items() if key != "quality_report_id"}
        report["quality_report_id"] = content_sha256(body)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "readdressed-lie.json"
            write_json(path, report)
            with self.assertRaisesRegex(ValueError, "contradict recorded metrics"):
                verify_quality_report(path)

    def test_final_evaluation_cannot_be_used_for_quality_selection(self) -> None:
        clean = build_benchmark()["scenarios"][0]["result"]
        with self.assertRaisesRegex(ValueError, "calibration split"):
            build_quality_report(
                result=clean,
                dictionary_sources=[],
                activation_category_split={
                    "activation_category_split_id": "d" * 64,
                    "activation_category_split_role": "final_evaluation",
                },
                holdout_capture={
                    "activation_capture_sha256": "e" * 64,
                    "capture_manifest_sha256": "f" * 64,
                },
                training_provenance=admitted_training_provenance(),
            )


if __name__ == "__main__":
    unittest.main()
