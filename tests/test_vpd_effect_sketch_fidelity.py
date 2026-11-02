from __future__ import annotations

import unittest
from pathlib import Path

from experiments.authz_mechinterp.benchmark_effect_sketch_fidelity import (
    verify_benchmark,
)


class VpdEffectSketchFidelityTests(unittest.TestCase):
    def test_durable_production_shaped_benchmark_reproduces(self) -> None:
        root = Path(__file__).resolve().parents[1]
        report = (
            root
            / "artifacts"
            / "vpd_effect_sketch_fidelity_benchmark_v1"
            / "report.json"
        )
        verification = verify_benchmark(report)
        self.assertEqual(verification["status"], "pass")
        self.assertEqual(verification["cases"], 3584)
        self.assertEqual(
            verification["benchmark_id"],
            "4ed9790051cabd86dd4b2faffd68d57bfd1a8fbead2e8ea0df0b107ab29e04dd",
        )
        self.assertEqual(verification["selected_candidate"]["dimension"], 256)
        self.assertLessEqual(
            verification["selected_candidate"][
                "worst_seed_stable_false_admission_wilson_95_upper"
            ],
            0.01,
        )
        self.assertGreaterEqual(
            verification["selected_candidate"][
                "worst_seed_stable_pass_recall_wilson_95_lower"
            ],
            0.99,
        )


if __name__ == "__main__":
    unittest.main()
