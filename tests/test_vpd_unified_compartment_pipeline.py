from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v4 import (
    verify_protocol,
)
from experiments.authz_mechinterp.verify_unified_vpd_compartment_pipeline import (
    build_receipt,
    verify_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "evals" / "vpd_activation_category_compartments_v4.json"
PREDECESSOR = ROOT / "configs" / "evals" / "vpd_activation_category_compartments_v3.json"
EFFECT_PLAN = ROOT / "artifacts" / "vpd_effect_capture_campaign_v3" / "plan.json"
OLD_EFFECT_PLAN = ROOT / "artifacts" / "vpd_effect_capture_campaign_v1" / "plan.json"
TRAINING_PLAN = ROOT / "artifacts" / "vpd_compartment_sae_training_plan_v3" / "plan.json"
BUNDLE = ROOT / "artifacts" / "bonsai8b_authz_v8" / "vpd_sae_clusters"
SPLIT = ROOT / "artifacts" / "vpd_activation_category_capacity_split_v4"
PACK = ROOT / "artifacts" / "vpd_activation_category_capacity_pack_v3"
HIERARCHY = ROOT / "artifacts" / "bonsai8b_authz_v5" / "hierarchy_group_plan.json"
CAPACITY = ROOT / "artifacts" / "vpd_compartment_sae_capacity_v2" / "plan.json"
RESOURCES = ROOT / "artifacts" / "vpd_compartment_capture_resources_v1" / "plan.json"
QUALITY = ROOT / "artifacts" / "vpd_compartment_sae_quality_benchmark_v1" / "report.json"
RECEIPT = ROOT / "artifacts" / "vpd_unified_compartment_pipeline_v2" / "receipt.json"


def common() -> dict[str, Path]:
    return {
        "protocol_path": PROTOCOL,
        "predecessor_protocol_path": PREDECESSOR,
        "effect_plan_path": EFFECT_PLAN,
        "training_plan_path": TRAINING_PLAN,
        "bundle_dir": BUNDLE,
        "split_dir": SPLIT,
        "pack_dir": PACK,
        "hierarchy_plan_path": HIERARCHY,
        "capacity_plan_path": CAPACITY,
        "capture_resource_plan_path": RESOURCES,
        "quality_benchmark_path": QUALITY,
    }


class UnifiedVpdCompartmentPipelineTests(unittest.TestCase):
    def test_unified_protocol_and_pipeline_receipt_reproduce(self) -> None:
        protocol = verify_protocol(
            PROTOCOL,
            predecessor_path=PREDECESSOR,
        )
        self.assertEqual(protocol["status"], "pass")
        self.assertEqual(
            protocol["protocol_sha256"],
            "ff8e42728f77e21c5d61d6425858e16159e0cef887959052ed2de50699c41822",
        )
        verification = verify_receipt(RECEIPT, **common())
        self.assertEqual(verification["status"], "pass")
        self.assertEqual(
            verification["pipeline_receipt_id"],
            "12dea48a1928ec3780f31419aed772f9584962216391bc10aabc5d2ab753ffd3",
        )
        self.assertEqual(verification["roles"], 5)
        self.assertFalse(verification["execution_authorized"])

    def test_receipt_binds_one_split_and_disjoint_role_families(self) -> None:
        receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        semantic = protocol["sae_training"]["capture_semantic_admission"]
        component_admission = protocol["component_compartments"][
            "per_component_cross_archive_admission"
        ]
        self.assertEqual(
            component_admission["mode"],
            "same_category_direction_magnitude_and_stratified_robustness",
        )
        self.assertTrue(
            component_admission[
                "global_threshold_replication_does_not_waive_component_admission"
            ]
        )
        self.assertTrue(component_admission["independent_verifier_recomputes_every_component"])
        self.assertEqual(
            semantic["synthetic_benchmark_id"],
            "51ebf5afb2cf0e1cc109ed8a7672d9ff8f0adb7c65fb4e442128164fbc5f0964",
        )
        self.assertTrue(semantic["required_before_dictionary_training"])
        self.assertTrue(semantic["required_before_dictionary_bundle_admission"])
        self.assertTrue(semantic["required_before_calibration_quality"])
        self.assertEqual(
            receipt["source"]["effect_campaign"]["verification"]["campaign_plan_id"],
            "324dac30e39fa46305ede80eeb00ce5037f219133e2db66e8128a15e506ef6b8",
        )
        self.assertEqual(
            receipt["source"]["sae_training"]["verification"]["training_plan_id"],
            "d4ef3eca0c642d22f3df415bacb61d5a2130fed969570eb6836777589d1301dd",
        )
        role_hashes = [
            row["template_group_set_sha256"] for row in receipt["roles"].values()
        ]
        self.assertEqual(len(role_hashes), 5)
        self.assertEqual(len(set(role_hashes)), 5)
        self.assertEqual(
            receipt["resources"]["effect_episode_forward_equivalents"],
            114_162_048,
        )
        self.assertEqual(
            receipt["resources"]["sae_capture_rows_by_role"],
            {"calibration": 604_032, "sae_training": 805_376},
        )
        self.assertEqual(receipt["resources"]["sae_capture_rows_total"], 1_409_408)
        self.assertFalse(receipt["scientific_boundary"]["model_inference_performed"])

    def test_cross_pack_effect_plan_cannot_join_the_capacity_sae_plan(self) -> None:
        arguments = common()
        arguments["effect_plan_path"] = OLD_EFFECT_PLAN
        with self.assertRaises(ValueError):
            build_receipt(**arguments)


if __name__ == "__main__":
    unittest.main()
