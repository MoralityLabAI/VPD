from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.authz_mechinterp.benchmark_focused_sae_input_lineage import (
    verify_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "artifacts"
    / "vpd_focused_sae_input_lineage_benchmark_v1"
    / "report.json"
)


class FocusedSaeInputLineageTests(unittest.TestCase):
    def test_durable_lineage_benchmark_reproduces_and_fails_closed(self) -> None:
        verification = verify_benchmark(REPORT)
        self.assertEqual(verification["status"], "pass")
        self.assertEqual(
            verification["benchmark_id"],
            "72a45142b1817269e4b991556585ad9e139c02290c31753edcc994d87ec783fe",
        )
        self.assertEqual(verification["scenarios"], 8)
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        outcomes = {
            row["scenario_id"]: row["observed_status"]
            for row in report["scenarios"]
        }
        self.assertEqual(
            outcomes,
            {
                "exact_packet_bound_archive_and_manifest_pass": "pass",
                "substituted_effect_archive_rejects": "reject",
                "substituted_effect_manifest_rejects": "reject",
                "category_packet_without_effect_lineage_rejects": "reject",
                "legacy_metadata_packet_is_explicitly_unattested": (
                    "unattested_legacy_metadata_pilot"
                ),
                "exact_signed_input_lineage_passes_reducer": "pass",
                "substituted_signed_effect_archive_rejects_at_reducer": "reject",
                "misstated_signed_lineage_receipt_rejects_at_reducer": "reject",
            },
        )
        self.assertFalse(report["scientific_boundary"]["model_inference_performed"])


if __name__ == "__main__":
    unittest.main()
