from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.authz_mechinterp.activation_categories import (
    COMPONENT_COMPARTMENT_PROTOCOLS,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    write_json,
)
from experiments.authz_mechinterp.plan_compartment_sae_training import (
    validate_training_plan_structure,
    verify_training_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts" / "vpd_compartment_sae_training_plan_v3" / "plan.json"
HISTORICAL_PLAN = (
    ROOT / "artifacts" / "vpd_compartment_sae_training_plan_v2" / "plan.json"
)
CAPACITY = ROOT / "artifacts" / "vpd_compartment_sae_capacity_v2" / "plan.json"
RESOURCES = ROOT / "artifacts" / "vpd_compartment_capture_resources_v1" / "plan.json"
PACK = ROOT / "artifacts" / "vpd_activation_category_capacity_pack_v3"
SPLIT = ROOT / "artifacts" / "vpd_activation_category_capacity_split_v4"
PROTOCOL = ROOT / "configs" / "evals" / "vpd_activation_category_compartments_v4.json"
QUALITY_BENCHMARK = (
    ROOT / "artifacts" / "vpd_compartment_sae_quality_benchmark_v1" / "report.json"
)


class CompartmentSaeTrainingPlanTests(unittest.TestCase):
    def test_durable_training_plan_reproduces(self) -> None:
        verification = verify_training_plan(
            PLAN,
            capacity_plan_path=CAPACITY,
            capture_resource_plan_path=RESOURCES,
            pack_dir=PACK,
            split_dir=SPLIT,
            protocol_config_path=PROTOCOL,
            quality_benchmark_path=QUALITY_BENCHMARK,
        )
        self.assertEqual(verification["status"], "pass")
        self.assertEqual(
            verification["training_plan_id"],
            "d4ef3eca0c642d22f3df415bacb61d5a2130fed969570eb6836777589d1301dd",
        )
        self.assertEqual(verification["capture_jobs"], 2)
        self.assertEqual(verification["training_jobs"], 18)
        self.assertEqual(verification["compartments"], 9)

    def test_plan_freezes_initialization_only_replication(self) -> None:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        validate_training_plan_structure(plan)
        self.assertEqual(plan["profile"]["initialization_seeds"], [17, 29])
        self.assertEqual(plan["profile"]["data_selection_seed"], 17)
        self.assertEqual(
            plan["source"]["focused_screening_benchmark_id"],
            "8f06090f64db0a1fadcaec50e2f14bf73eff3fe8e3e372de836dd55f8a501b12",
        )
        self.assertEqual(
            plan["source"]["capture_semantic_benchmark_id"],
            "51ebf5afb2cf0e1cc109ed8a7672d9ff8f0adb7c65fb4e442128164fbc5f0964",
        )
        observed = {
            (row["compartment"], row["initialization_seed"])
            for row in plan["training_jobs"]
        }
        self.assertEqual(
            observed,
            {
                (compartment, seed)
                for compartment in COMPONENT_COMPARTMENT_PROTOCOLS
                for seed in (17, 29)
            },
        )
        self.assertEqual(
            {row["data_selection_seed"] for row in plan["training_jobs"]},
            {17},
        )
        self.assertEqual(
            len(plan["quality_job"]["depends_on_training_job_ids"]),
            18,
        )
        self.assertEqual(
            plan["quality_job"]["depends_on_admission_job_id"],
            plan["admission_job"]["job_id"],
        )

    def test_plan_resources_cover_every_training_and_quality_pass(self) -> None:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        resources = plan["resource_summary"]
        self.assertEqual(resources["training_jobs"], 18)
        self.assertEqual(resources["dictionary_bundle_admission_jobs"], 1)
        self.assertEqual(
            resources["capture_rows_by_role"],
            {"calibration": 604032, "sae_training": 805376},
        )
        self.assertEqual(resources["capture_rows_total"], 1409408)
        self.assertEqual(resources["training_activation_row_visits"], 2782208)
        self.assertEqual(resources["quality_selected_rows"], 65536)
        self.assertEqual(resources["quality_activation_row_visits"], 1179648)
        self.assertEqual(
            resources["quality_numeric_resource_bounds"][
                "estimated_numeric_peak_bytes_excluding_python_and_npz_overhead"
            ],
            680525824,
        )
        self.assertTrue(
            resources["checkpoint_optimizer_and_framework_overhead_not_estimated"]
        )

    def test_readdressed_outer_plan_cannot_hide_changed_job_arguments(self) -> None:
        plan = copy.deepcopy(json.loads(PLAN.read_text(encoding="utf-8")))
        job = plan["training_jobs"][0]
        seed_position = job["arguments"].index("--seed") + 1
        job["arguments"][seed_position] = "999"
        body = {key: value for key, value in plan.items() if key != "training_plan_id"}
        plan["training_plan_id"] = content_sha256(body)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "readdressed-plan.json"
            write_json(path, plan)
            with self.assertRaisesRegex(ValueError, "job content address"):
                verify_training_plan(
                    path,
                    capacity_plan_path=CAPACITY,
                    capture_resource_plan_path=RESOURCES,
                    pack_dir=PACK,
                    split_dir=SPLIT,
                    protocol_config_path=PROTOCOL,
                    quality_benchmark_path=QUALITY_BENCHMARK,
                )

    def test_historical_v2_plan_remains_content_addressed_evidence(self) -> None:
        plan = json.loads(HISTORICAL_PLAN.read_text(encoding="utf-8"))
        body = {key: value for key, value in plan.items() if key != "training_plan_id"}
        self.assertEqual(content_sha256(body), plan["training_plan_id"])
        self.assertEqual(
            plan["training_plan_id"],
            "42e69abb5c3837228569085c0f31bf0c04d40fbc027b74fb63477a54c6ee34fb",
        )

    def test_readdressed_plan_cannot_remove_semantic_capture_inputs(self) -> None:
        plan = copy.deepcopy(json.loads(PLAN.read_text(encoding="utf-8")))
        job = plan["training_jobs"][0]
        position = job["arguments"].index("--source-dataset")
        del job["arguments"][position : position + 2]
        job_body = {key: value for key, value in job.items() if key != "job_id"}
        job["job_id"] = content_sha256(job_body)
        with self.assertRaisesRegex(ValueError, "source-dataset"):
            validate_training_plan_structure(plan)

    def test_executable_jobs_never_reference_final_evaluation(self) -> None:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        jobs = {
            "capture_jobs": plan["capture_jobs"],
            "training_jobs": plan["training_jobs"],
            "admission_job": plan["admission_job"],
            "quality_job": plan["quality_job"],
        }
        self.assertNotIn("final_evaluation", json.dumps(jobs, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
