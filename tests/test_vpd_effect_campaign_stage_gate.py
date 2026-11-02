from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, write_json
from experiments.authz_mechinterp.gate_vpd_effect_campaign_stage import (
    evaluate_calibration_profiles,
    evaluate_discovery_profile,
    verify_gate_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads(
    (
        ROOT / "configs" / "evals" / "vpd_activation_category_compartments_v2.json"
    ).read_text(encoding="utf-8")
)


def profile(*, specific_components: int = 40, swap: bool = False) -> dict:
    components = 40
    categories = ["replayed", "valid_authority"]
    vectors = np.zeros((components, 2, 256), dtype=np.float64)
    for index in range(components):
        if index >= specific_components:
            vectors[index, 0, 0] = 1.0
            vectors[index, 1, 1] = 1.0
        elif index % 2 == 0:
            vectors[index, 0, 0] = 2.0
            vectors[index, 1, 1] = 0.05
        else:
            vectors[index, 0, 0] = 0.05
            vectors[index, 1, 1] = 2.0
    if swap:
        vectors = vectors[:, ::-1, :]
    strengths = np.linalg.norm(vectors, axis=2)
    block_counts = np.ones((2, 8), dtype=np.int64)
    block_sums = np.repeat(vectors[:, :, None, :], 8, axis=2)
    condition_categories = ["replayed", "valid_authority"]
    condition_counts = np.asarray([16, 16], dtype=np.int64)
    condition_sums = np.stack(
        [vectors[:, 0] * 16, vectors[:, 1] * 16], axis=1
    )
    template_categories = [category for category in categories for _ in range(4)]
    template_counts = np.full(8, 4, dtype=np.int64)
    template_sums = np.stack(
        [
            vectors[:, category_index] * 4
            for category_index in range(2)
            for _ in range(4)
        ],
        axis=1,
    )
    return {
        "category_names": categories,
        "category_episode_counts": [32, 32],
        "strengths": strengths,
        "category_effect_vectors": vectors,
        "uncertainty_block_counts": block_counts,
        "uncertainty_block_sums": block_sums,
        "uncertainty_block_count": 8,
        "uncertainty_countsketch_dimension": 256,
        "uncertainty_countsketch_seed": 1,
        "condition_categories": condition_categories,
        "condition_counts": condition_counts,
        "condition_sums": condition_sums,
        "template_cell_categories": template_categories,
        "template_cell_counts": template_counts,
        "template_cell_sums": template_sums,
    }


class VpdEffectCampaignStageGateTests(unittest.TestCase):
    def test_discovery_optimistic_upper_bound_passes_clean_fixture(self) -> None:
        result = evaluate_discovery_profile(profile(), PROTOCOL)
        self.assertEqual(result["status"], "pass_continue_calibration")
        self.assertEqual(
            result["metrics"]["optimistic_eligible_specific_upper_bound"], 40
        )
        self.assertTrue(result["logic"]["necessary_condition_only"])
        self.assertTrue(result["logic"]["pass_does_not_predict_calibration_success"])

    def test_discovery_stops_when_even_optimistic_bound_is_too_small(self) -> None:
        result = evaluate_discovery_profile(profile(specific_components=20), PROTOCOL)
        self.assertEqual(result["status"], "reject_stop_campaign")
        self.assertEqual(
            result["metrics"]["optimistic_eligible_specific_upper_bound"], 20
        )
        self.assertEqual(
            result["metrics"]["rejection_counts"]["optimistic_threshold_shared"],
            20,
        )

    def test_calibration_passes_clean_independent_replica(self) -> None:
        discovery = profile()
        calibration = copy.deepcopy(discovery)
        calibration["category_effect_vectors"] *= 0.9
        calibration["strengths"] *= 0.9
        result = evaluate_calibration_profiles(discovery, calibration, PROTOCOL)
        self.assertEqual(result["status"], "pass_continue_cluster_validation")
        selected = result["calibration"]["selection"]
        self.assertEqual(selected["selected_metrics"]["stable_specific_count"], 40)
        self.assertTrue(result["logic"]["cluster_validation_not_used"])

    def test_calibration_stops_on_cross_archive_category_swap(self) -> None:
        result = evaluate_calibration_profiles(profile(), profile(swap=True), PROTOCOL)
        self.assertEqual(result["status"], "reject_stop_campaign")
        self.assertIn("no activation-category threshold candidate", result["reason"])

    def test_gate_receipt_content_address_and_stage_are_enforced(self) -> None:
        body = {
            "schema": "sch.vpd-effect-campaign-stage-gate.v1",
            "stage": "discovery",
            "decision": {"status": "pass_continue_calibration"},
            "source": {"discovery_effects_sha256": "1" * 64},
            "scientific_boundary": {
                "model_inference_performed_by_gate": False,
            },
        }
        receipt = {**body, "gate_id": content_sha256(body)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            write_json(path, receipt)
            self.assertEqual(
                verify_gate_receipt(path, required_stage="discovery")["gate_id"],
                receipt["gate_id"],
            )
            with self.assertRaisesRegex(ValueError, "another stage"):
                verify_gate_receipt(path, required_stage="calibration")
            tampered = copy.deepcopy(receipt)
            tampered["decision"]["status"] = "reject_stop_campaign"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content address mismatch"):
                verify_gate_receipt(path)


if __name__ == "__main__":
    unittest.main()
