from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.authz_mechinterp.benchmark_activation_category_protocol import (
    verify_report,
)


class ActivationCategoryProtocolBenchmarkTests(unittest.TestCase):
    def test_durable_synthetic_benchmark_is_reproducible(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        report_path = (
            project_root
            / "artifacts"
            / "vpd_activation_category_protocol_benchmark_v1"
            / "report.json"
        )
        self.assertEqual(
            verify_report(report_path, project_root=project_root),
            {
                "schema": (
                    "sch.vpd-activation-category-synthetic-benchmark-verification.v1"
                ),
                "status": "pass",
                "benchmark_id": (
                    "eb6ea4b61c17bd124a784b1efe532eb94f7fd79a3743725c284be296e48635c0"
                ),
                "scenarios": 13,
                "all_expectations_met": True,
            },
        )

    def test_benchmark_covers_recovery_and_fail_closed_boundaries(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        report = json.loads(
            (
                project_root
                / "artifacts"
                / "vpd_activation_category_protocol_benchmark_v1"
                / "report.json"
            ).read_text()
        )
        scenarios = {row["scenario_id"]: row for row in report["scenarios"]}
        for scenario_id in ("clean_cross_template", "moderate_noise_and_shift"):
            row = scenarios[scenario_id]
            self.assertEqual(row["discovery"]["exact_accuracy"], 1.0)
            self.assertEqual(row["final_unseen_template_proxy"]["exact_accuracy"], 1.0)
            self.assertGreater(
                row["calibration_metrics"][
                    "minimum_stable_specific_decision_slack"
                ],
                0.0,
            )
        noisy = scenarios["high_pair_block_noise"]
        self.assertGreaterEqual(noisy["jackknife_specific_to_shared_reroutes"], 90)
        self.assertEqual(noisy["discovery"]["specific_precision"], 1.0)
        self.assertEqual(noisy["discovery"]["shared_recall"], 1.0)
        for scenario_id in (
            "direction_reversal",
            "category_label_swap",
            "calibration_magnitude_collapse",
            "calibration_single_condition_category_artifact",
            "calibration_single_template_category_artifact",
            "calibration_opposing_pair_block_directions",
        ):
            self.assertTrue(scenarios[scenario_id]["calibration_rejected"])
        narrow = scenarios["single_condition_category_artifact"]
        self.assertEqual(narrow["jackknife_specific_to_shared_reroutes"], 0)
        self.assertEqual(narrow["condition_specific_to_shared_reroutes"], 8)
        self.assertEqual(narrow["discovery"]["specific_precision"], 1.0)
        template_narrow = scenarios["single_template_category_artifact"]
        self.assertEqual(
            template_narrow["jackknife_specific_to_shared_reroutes"], 0
        )
        self.assertEqual(
            template_narrow["condition_specific_to_shared_reroutes"], 0
        )
        self.assertEqual(
            template_narrow["template_specific_to_shared_reroutes"], 8
        )
        opposing = scenarios["opposing_pair_block_directions"]
        self.assertEqual(opposing["jackknife_specific_to_shared_reroutes"], 0)
        self.assertEqual(
            opposing["pair_block_direction_specific_to_shared_reroutes"], 8
        )
        self.assertEqual(opposing["condition_specific_to_shared_reroutes"], 0)
        self.assertEqual(opposing["template_specific_to_shared_reroutes"], 0)
        minority = scenarios["minority_cross_archive_category_disagreement"]
        self.assertFalse(minority["calibration_rejected"])
        self.assertEqual(
            minority["calibration_metrics"]["symmetric_replication_rate"],
            88 / 96,
        )
        self.assertEqual(minority["cross_archive_specific_to_shared_reroutes"], 8)
        self.assertTrue(
            minority["checks"]["minority_cross_archive_disagreement_rerouted"]
        )


if __name__ == "__main__":
    unittest.main()
