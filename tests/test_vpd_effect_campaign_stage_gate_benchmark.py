from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_effect_campaign_stage_gates import (
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "evals" / "vpd_activation_category_compartments_v4.json"
REPORT = (
    ROOT
    / "artifacts"
    / "vpd_effect_campaign_stage_gate_benchmark_v3"
    / "report.json"
)


class VpdEffectCampaignStageGateBenchmarkTests(unittest.TestCase):
    def test_durable_benchmark_reproduces_and_covers_pass_and_stop_cases(self) -> None:
        verification = verify_benchmark(REPORT, protocol_path=PROTOCOL)
        self.assertEqual(verification["status"], "pass")
        self.assertEqual(
            verification["benchmark_id"],
            "9a12ecaed95889d139c1bf0e504e7172090f17941622d5935752262bc3ec92b4",
        )
        self.assertEqual(verification["scenarios"], 7)

        report = json.loads(REPORT.read_text(encoding="utf-8"))
        outcomes = {
            row["scenario_id"]: row["result"]["status"]
            for row in report["scenarios"]
        }
        self.assertEqual(
            outcomes,
            {
                "clean_discovery_continues": "pass_continue_calibration",
                "insufficient_discovery_upper_bound_stops": "reject_stop_campaign",
                "condition_artifact_discovery_stops": "reject_stop_campaign",
                "clean_calibration_continues": "pass_continue_cluster_validation",
                "category_swap_calibration_stops": "reject_stop_campaign",
                "direction_reversal_calibration_stops": "reject_stop_campaign",
                "magnitude_collapse_calibration_stops": "reject_stop_campaign",
            },
        )
        self.assertTrue(report["scientific_boundary"]["synthetic_effects_only"])
        self.assertTrue(
            report["scientific_boundary"][
                "does_not_establish_bonsai_stage_gate_outcomes"
            ]
        )
        self.assertFalse(report["scientific_boundary"]["model_inference_performed"])


if __name__ == "__main__":
    unittest.main()
