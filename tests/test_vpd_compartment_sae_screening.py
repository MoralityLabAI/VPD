from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.authz_mechinterp.benchmark_compartment_sae_screening import (
    build_benchmark,
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT / "artifacts" / "vpd_compartment_sae_screening_benchmark_v2" / "report.json"
)


class CompartmentSaeScreeningTests(unittest.TestCase):
    def test_synthetic_failure_boundary(self) -> None:
        report = build_benchmark()
        self.assertEqual(report["status"], "pass")
        scenarios = {row["scenario_id"]: row for row in report["scenarios"]}
        self.assertEqual(
            scenarios["specific_replayed_category_membership"]["top_feature_index"],
            0,
        )
        self.assertEqual(
            scenarios["legacy_global_outcome_leakage_diagnostic"][
                "top_feature_index"
            ],
            1,
        )

    def test_durable_benchmark_reproduces(self) -> None:
        verification = verify_benchmark(REPORT)
        self.assertEqual(verification["status"], "pass")
        self.assertEqual(
            verification["benchmark_id"],
            "8f06090f64db0a1fadcaec50e2f14bf73eff3fe8e3e372de836dd55f8a501b12",
        )

    def test_protocol_pins_category_specific_screening(self) -> None:
        protocol = json.loads(
            (
                ROOT
                / "configs"
                / "evals"
                / "vpd_activation_category_compartments_v3.json"
            ).read_text(encoding="utf-8")
        )
        endpoint = protocol["focused_evaluation"]["screening_endpoint"]
        self.assertEqual(
            endpoint["synthetic_benchmark_id"],
            "8f06090f64db0a1fadcaec50e2f14bf73eff3fe8e3e372de836dd55f8a501b12",
        )
        lineage = protocol["focused_evaluation"]["input_lineage"]
        self.assertEqual(
            lineage["synthetic_benchmark_id"],
            "72a45142b1817269e4b991556585ad9e139c02290c31753edcc994d87ec783fe",
        )
        self.assertEqual(
            lineage["category_packet_without_exact_effect_lineage_rule"],
            "reject",
        )
        self.assertFalse(
            endpoint["specific_model_outcome_label_used_for_selection"]
        )


if __name__ == "__main__":
    unittest.main()
