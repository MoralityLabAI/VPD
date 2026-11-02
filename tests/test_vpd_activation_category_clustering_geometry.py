from __future__ import annotations

import unittest
from pathlib import Path

from experiments.authz_mechinterp.benchmark_activation_category_clustering_geometry import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    ACTIVATION_CATEGORY_DIRECTION_WEIGHTING,
)


class ActivationCategoryClusteringGeometryTests(unittest.TestCase):
    def test_durable_production_shaped_benchmark_reproduces(self) -> None:
        root = Path(__file__).resolve().parents[1]
        report_path = (
            root
            / "artifacts"
            / "vpd_activation_category_clustering_geometry_benchmark_v1"
            / "report.json"
        )
        verification = verify_benchmark(report_path)
        self.assertEqual(verification["status"], "pass")
        self.assertEqual(
            verification["benchmark_id"],
            "7098172918613a6f74e1f2d4032a5d720440dc654d1465cbc73f25d416c07428",
        )
        self.assertEqual(verification["scenarios"], 3)
        self.assertEqual(
            verification["selected_direction_weighting"],
            ACTIVATION_CATEGORY_DIRECTION_WEIGHTING,
        )


if __name__ == "__main__":
    unittest.main()
