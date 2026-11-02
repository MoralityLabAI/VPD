from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, write_json
from experiments.authz_mechinterp.plan_vpd_effect_capture_campaign import (
    EFFECT_ROLES,
    build_campaign_plan,
    verify_campaign_plan,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "bonsai8b_authz_v8" / "vpd_sae_clusters"
SPLIT = ROOT / "artifacts" / "vpd_activation_category_capacity_split_v4"
PROTOCOL = ROOT / "configs" / "evals" / "vpd_activation_category_compartments_v4.json"
HIERARCHY = ROOT / "artifacts" / "bonsai8b_authz_v5" / "hierarchy_group_plan.json"
PLAN = ROOT / "artifacts" / "vpd_effect_capture_campaign_v3" / "plan.json"


class VpdEffectCaptureCampaignPlanTests(unittest.TestCase):
    def test_durable_campaign_plan_reproduces(self) -> None:
        verification = verify_campaign_plan(
            PLAN,
            bundle_dir=BUNDLE,
            split_dir=SPLIT,
            protocol_config_path=PROTOCOL,
            hierarchy_plan_path=HIERARCHY,
        )
        self.assertEqual(verification["status"], "pass")
        self.assertEqual(
            verification["campaign_plan_id"],
            "324dac30e39fa46305ede80eeb00ce5037f219133e2db66e8128a15e506ef6b8",
        )
        self.assertEqual(verification["roles"], 3)
        self.assertEqual(verification["compute_jobs"], 48)
        self.assertEqual(verification["components_per_role"], 4032)
        self.assertFalse(verification["execution_authorized"])

    def test_campaign_has_exact_packet_coverage_in_every_role(self) -> None:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        packet_ids = {row["packet_id"] for row in plan["packets"]}
        self.assertEqual(len(packet_ids), 16)
        by_role = Counter(row["role"] for row in plan["compute_jobs"])
        self.assertEqual(by_role, Counter({role: 16 for role in EFFECT_ROLES}))
        for role in EFFECT_ROLES:
            jobs = [row for row in plan["compute_jobs"] if row["role"] == role]
            self.assertEqual({row["packet"]["packet_id"] for row in jobs}, packet_ids)
            self.assertEqual(
                sum(row["packet"]["component_count"] for row in jobs), 4032
            )
            self.assertEqual({row["run_id"] for row in jobs}, {plan["run_ids_by_role"][role]})
        self.assertEqual(len({*plan["run_ids_by_role"].values()}), 3)

    def test_only_effect_roles_are_used_and_production_gates_are_frozen(self) -> None:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        encoded_jobs = json.dumps(
            {
                "compute": plan["compute_jobs"],
                "signing": plan["signing_jobs"],
                "merge": plan["merge_jobs"],
                "gates": plan["stage_gate_jobs"],
                "cluster": plan["cluster_job"],
            },
            sort_keys=True,
        )
        self.assertNotIn("final_evaluation", encoded_jobs)
        self.assertNotIn("sae_training", encoded_jobs)
        for role in EFFECT_ROLES:
            self.assertIn(role, encoded_jobs)
        arguments = plan["cluster_job"]["arguments"]
        for flag in (
            "--require-category-uncertainty",
            "--require-category-pair-block-direction-robustness",
            "--require-category-condition-robustness",
            "--require-category-template-robustness",
            "--select-category-cluster-counts-by-projection",
            "--require-category-cluster-sampling-stability",
            "--require-category-cluster-cross-archive-stability",
        ):
            self.assertIn(flag, arguments)
        self.assertNotIn("--allow-unattested-calibration", arguments)
        self.assertNotIn("--allow-unattested-cluster-validation", arguments)
        self.assertTrue(plan["invariants"]["plan_does_not_authorize_execution"])
        self.assertFalse(plan["scientific_boundary"]["model_inference_performed"])

    def test_expensive_roles_are_strictly_stage_gated(self) -> None:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        gates = {row["stage"]: row for row in plan["stage_gate_jobs"]}
        self.assertEqual(set(gates), {"discovery", "calibration"})
        self.assertEqual(
            gates["discovery"]["required_pass_status"],
            "pass_continue_calibration",
        )
        self.assertEqual(
            gates["calibration"]["required_pass_status"],
            "pass_continue_cluster_validation",
        )
        discovery_jobs = [
            row for row in plan["compute_jobs"] if row["role"] == "discovery"
        ]
        self.assertTrue(
            all("depends_on_passed_stage_gate_job_id" not in row for row in discovery_jobs)
        )
        calibration_jobs = [
            row for row in plan["compute_jobs"] if row["role"] == "calibration"
        ]
        self.assertEqual(
            {row["depends_on_passed_stage_gate_job_id"] for row in calibration_jobs},
            {gates["discovery"]["job_id"]},
        )
        validation_jobs = [
            row
            for row in plan["compute_jobs"]
            if row["role"] == "cluster_validation"
        ]
        self.assertEqual(
            {row["depends_on_passed_stage_gate_job_id"] for row in validation_jobs},
            {gates["calibration"]["job_id"]},
        )
        self.assertEqual(
            plan["cluster_job"]["depends_on_gate_job_ids"],
            [gates["calibration"]["job_id"]],
        )

    def test_flat_merge_layout_has_unique_shards_and_signed_receipts(self) -> None:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        self.assertEqual(len(plan["signing_jobs"]), 48)
        compute_by_id = {row["job_id"]: row for row in plan["compute_jobs"]}
        for role in EFFECT_ROLES:
            jobs = [row for row in plan["compute_jobs"] if row["role"] == role]
            shard_names = [Path(row["output"]["shard"]).name for row in jobs]
            receipt_names = [
                Path(row["output"]["unsigned_receipt"]).name for row in jobs
            ]
            self.assertEqual(len(set(shard_names)), 16)
            self.assertEqual(len(set(receipt_names)), 16)
            self.assertTrue(
                all(Path(row["output"]["shard"]).parent.as_posix() == f"runs/{role}/shards" for row in jobs)
            )
            signing = [row for row in plan["signing_jobs"] if row["role"] == role]
            self.assertEqual(len({Path(row["output"]).name for row in signing}), 16)
            self.assertTrue(
                all(row["depends_on_compute_job_id"] in compute_by_id for row in signing)
            )
            merge = next(row for row in plan["merge_jobs"] if row["role"] == role)
            self.assertEqual(
                set(merge["depends_on_signing_job_ids"]),
                {row["job_id"] for row in signing},
            )

    def test_resource_scope_is_explicit_and_worker_load_is_conserved(self) -> None:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        total = plan["resources"]["total"]
        self.assertEqual(total["episode_forward_equivalents"], 114_162_048)
        self.assertEqual(total["batch_forwards_at_planned_batch_size"], 28_667_520)
        self.assertEqual(total["estimated_float32_effect_payload_bytes"], 3_009_871_872)
        self.assertEqual(total["signing_jobs"], 48)
        self.assertEqual(total["stage_gate_jobs"], 2)
        self.assertEqual(sum(total["worker_component_load_per_role"].values()), 4032)
        self.assertEqual(sum(total["worker_packet_load_per_role"].values()), 16)
        self.assertTrue(
            total["tokenized_sequence_lengths_model_memory_wall_time_and_cost_not_estimated"]
        )

    def test_tampering_and_changed_sources_are_rejected(self) -> None:
        original = json.loads(PLAN.read_text(encoding="utf-8"))
        tampered = copy.deepcopy(original)
        tampered["resources"]["total"]["compute_jobs"] = 47
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.json"
            write_json(path, tampered)
            with self.assertRaisesRegex(ValueError, "content address mismatch"):
                verify_campaign_plan(
                    path,
                    bundle_dir=BUNDLE,
                    split_dir=SPLIT,
                    protocol_config_path=PROTOCOL,
                    hierarchy_plan_path=HIERARCHY,
                )

            readdressed = copy.deepcopy(tampered)
            body = dict(readdressed)
            body.pop("campaign_plan_id")
            readdressed["campaign_plan_id"] = content_sha256(body)
            write_json(path, readdressed)
            with self.assertRaisesRegex(ValueError, "not reproducible"):
                verify_campaign_plan(
                    path,
                    bundle_dir=BUNDLE,
                    split_dir=SPLIT,
                    protocol_config_path=PROTOCOL,
                    hierarchy_plan_path=HIERARCHY,
                )

    def test_builder_is_deterministic(self) -> None:
        first = build_campaign_plan(
            bundle_dir=BUNDLE,
            split_dir=SPLIT,
            protocol_config_path=PROTOCOL,
            hierarchy_plan_path=HIERARCHY,
        )
        second = build_campaign_plan(
            bundle_dir=BUNDLE,
            split_dir=SPLIT,
            protocol_config_path=PROTOCOL,
            hierarchy_plan_path=HIERARCHY,
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
