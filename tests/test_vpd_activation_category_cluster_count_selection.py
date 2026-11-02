from __future__ import annotations

import unittest
from pathlib import Path

from experiments.authz_mechinterp.benchmark_activation_category_cluster_count_selection import (
    verify_benchmark,
)


class ActivationCategoryClusterCountSelectionTests(unittest.TestCase):
    def test_durable_production_shaped_benchmark_reproduces(self) -> None:
        root = Path(__file__).resolve().parents[1]
        report = (
            root
            / "artifacts"
            / "vpd_activation_category_cluster_count_benchmark_v1"
            / "report.json"
        )
        verification = verify_benchmark(report)
        self.assertEqual(verification["status"], "pass")
        self.assertEqual(
            verification["benchmark_id"],
            "32ced135dd8ee66af5a1fd56f592c5f407da0a7636f050c75c5a9c3febaaa996",
        )
        self.assertEqual(verification["scenarios"], 6)
        self.assertEqual(verification["projection_seeds"], 8)


if __name__ == "__main__":
    unittest.main()
