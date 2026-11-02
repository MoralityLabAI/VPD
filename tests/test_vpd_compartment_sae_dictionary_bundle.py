from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.authz_mechinterp.admit_compartment_sae_dictionary_bundle import (
    build_bundle_receipt,
    validate_bundle_receipt_structure,
    verify_bundle_for_quality,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts" / "vpd_compartment_sae_training_plan_v3" / "plan.json"


def fixture_records(plan: dict) -> list[dict]:
    records = []
    for job in plan["training_jobs"]:
        dictionary_id = content_sha256(
            {"job_id": job["job_id"], "fixture": "dictionary"}
        )
        records.append(
            {
                "status": "pass",
                "planned_job_id": job["job_id"],
                "compartment": job["compartment"],
                "initialization_seed": job["initialization_seed"],
                "data_selection_seed": job["data_selection_seed"],
                "dictionary_id": dictionary_id,
                "dictionary_manifest_sha256": content_sha256(
                    {"dictionary_id": dictionary_id, "kind": "manifest"}
                ),
                "dictionary_sha256": content_sha256(
                    {"dictionary_id": dictionary_id, "kind": "npz"}
                ),
                "train_validation_split_sha256": content_sha256(
                    {"dictionary_id": dictionary_id, "kind": "split"}
                ),
                "capacity_authorization": {"status": "pass"},
                "planned_output_dir": job["output_dir"],
            }
        )
    return records


def capture_identity(plan: dict) -> dict:
    semantic_body = {
        "schema": "sch.vpd-prompt-window-capture-semantic-verification.v1",
        "status": "pass",
        "exact_source_episode_order_verified": True,
        "exact_category_labels_verified": True,
        "exact_prompt_window_coordinates_verified": True,
        "model_outcomes_used": False,
    }
    return {
        "capture_id": "1" * 64,
        "capture_manifest_sha256": "2" * 64,
        "activation_capture_sha256": "3" * 64,
        "metadata_sha256": "4" * 64,
        "split_id": plan["source"]["split_id"],
        "semantic_verification": {
            **semantic_body,
            "semantic_verification_id": content_sha256(semantic_body),
        },
        "model": {
            "model_id": "fixture/base-model",
            "revision": "fixture-revision",
            "adapter_sha256": None,
        },
    }


class CompartmentSaeDictionaryBundleTests(unittest.TestCase):
    def test_exact_planned_bundle_is_admitted_for_quality(self) -> None:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        records = fixture_records(plan)
        receipt = build_bundle_receipt(
            training_plan=plan,
            training_plan_path=PLAN,
            capture_identity=capture_identity(plan),
            dictionary_records=records,
        )
        validate_bundle_receipt_structure(receipt)
        sources = [
            {
                key: row[key]
                for key in (
                    "dictionary_id",
                    "dictionary_manifest_sha256",
                    "dictionary_sha256",
                    "compartment",
                    "initialization_seed",
                    "data_selection_seed",
                )
            }
            for row in records
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            write_json(path, receipt)
            authorization = verify_bundle_for_quality(
                receipt_path=path,
                training_plan_path=PLAN,
                dictionary_sources=sources,
            )
        self.assertEqual(authorization["status"], "pass")
        self.assertEqual(
            authorization["dictionary_bundle_id"], receipt["dictionary_bundle_id"]
        )

    def test_readdressed_bundle_cannot_change_a_planned_output(self) -> None:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        receipt = build_bundle_receipt(
            training_plan=plan,
            training_plan_path=PLAN,
            capture_identity=capture_identity(plan),
            dictionary_records=fixture_records(plan),
        )
        tampered = copy.deepcopy(receipt)
        tampered["dictionaries"][0]["planned_output_dir"] = "dictionaries/substitute"
        body = {
            key: value
            for key, value in tampered.items()
            if key != "dictionary_bundle_id"
        }
        tampered["dictionary_bundle_id"] = content_sha256(body)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            write_json(path, tampered)
            sources = [
                {
                    key: row[key]
                    for key in (
                        "dictionary_id",
                        "dictionary_manifest_sha256",
                        "dictionary_sha256",
                        "compartment",
                        "initialization_seed",
                        "data_selection_seed",
                    )
                }
                for row in tampered["dictionaries"]
            ]
            with self.assertRaisesRegex(ValueError, "planned job"):
                verify_bundle_for_quality(
                    receipt_path=path,
                    training_plan_path=PLAN,
                    dictionary_sources=sources,
                )

    def test_builder_rejects_changed_replica_data_selection(self) -> None:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        records = fixture_records(plan)
        records[0]["data_selection_seed"] = 29
        with self.assertRaisesRegex(ValueError, "data selection"):
            build_bundle_receipt(
                training_plan=plan,
                training_plan_path=PLAN,
                capture_identity=capture_identity(plan),
                dictionary_records=records,
            )

    def test_builder_rejects_readdressed_weak_semantic_capture_receipt(self) -> None:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        identity = capture_identity(plan)
        semantic = identity["semantic_verification"]
        semantic["exact_category_labels_verified"] = False
        body = {
            key: value
            for key, value in semantic.items()
            if key != "semantic_verification_id"
        }
        semantic["semantic_verification_id"] = content_sha256(body)
        with self.assertRaisesRegex(ValueError, "weakens"):
            build_bundle_receipt(
                training_plan=plan,
                training_plan_path=PLAN,
                capture_identity=identity,
                dictionary_records=fixture_records(plan),
            )


if __name__ == "__main__":
    unittest.main()
