from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.authz_mechinterp.plan_compartment_sae_capacity import (
    required_observations,
    verify_capacity_plan,
)
from experiments.authz_mechinterp.train_sae import (
    authorize_compartment_capacity,
    load_capacity_plan,
    pair_atomic_train_validation_split,
    validate_prompt_window_metadata,
)
from experiments.authz_mechinterp.capture_activations import (
    prompt_window_metadata_rows,
    prompt_window_positions,
    validate_sae_training_window_request,
)
from experiments.authz_mechinterp.generate_activation_category_capacity_pack_v3 import (
    generate_capacity_episodes_v3,
)
from experiments.authz_mechinterp.generate_activation_category_pack_v2 import (
    generate_activation_category_episodes_v2,
)


ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "artifacts" / "vpd_activation_category_pack_v2"
PLAN_PATH = ROOT / "artifacts" / "vpd_compartment_sae_capacity_v1" / "plan.json"
CAPACITY_PACK_DIR = ROOT / "artifacts" / "vpd_activation_category_capacity_pack_v3"
CAPACITY_SPLIT_DIR = ROOT / "artifacts" / "vpd_activation_category_capacity_split_v4"
READY_PLAN_PATH = ROOT / "artifacts" / "vpd_compartment_sae_capacity_v2" / "plan.json"


class CompartmentSAECapacityTests(unittest.TestCase):
    def test_durable_capacity_plan_reproduces(self) -> None:
        verification = verify_capacity_plan(PLAN_PATH, pack_dir=PACK_DIR)
        self.assertEqual(verification["status"], "pass")
        self.assertEqual(
            verification["capacity_plan_id"],
            "f2f92a86a33ee8193bd6d8df9c3841e7710a9fe220237d5660679ecf040b0067",
        )
        self.assertEqual(verification["plan_status"], "insufficient_current_pack")
        self.assertEqual(verification["ready_profiles"], 0)

    def test_capacity_sized_pack_makes_only_bounded_pilot_ready(self) -> None:
        verification = verify_capacity_plan(
            READY_PLAN_PATH,
            pack_dir=CAPACITY_PACK_DIR,
            split_dir=CAPACITY_SPLIT_DIR,
        )
        self.assertEqual(verification["status"], "pass")
        self.assertEqual(
            verification["capacity_plan_id"],
            "809deb31cba4642ee38cbaf2f7417120610b4fc9cfe6c37d0ed6cbaa885efb7d",
        )
        self.assertEqual(verification["plan_status"], "ready")
        self.assertEqual(verification["ready_profiles"], 1)
        plan = load_capacity_plan(READY_PLAN_PATH)
        statuses = {row["id"]: row["status"] for row in plan["profiles"]}
        self.assertEqual(
            statuses,
            {
                "legacy_anchor_default": "insufficient",
                "bounded_context_pilot": "ready",
                "overcomplete_context_study": "insufficient",
            },
        )

    def test_capacity_pack_identity_namespace_is_disjoint_from_v2(self) -> None:
        v2 = list(generate_activation_category_episodes_v2(seed=320829, pairs_per_family=2))
        v3 = list(generate_capacity_episodes_v3(seed=320829, pairs_per_family=2))
        self.assertFalse(
            {row["episode_id"] for row in v2}
            & {row["episode_id"] for row in v3}
        )
        self.assertFalse(
            {row["pair_id"] for row in v2} & {row["pair_id"] for row in v3}
        )
        self.assertTrue(
            all(
                row["activation_category_pack"]["artifact_version"]
                == "activation-category-capacity-v3"
                for row in v3
            )
        )

    def test_support_floor_arithmetic(self) -> None:
        result = required_observations(
            dictionary_features=8192,
            top_k=16,
            minimum_training_observations_per_feature=8,
            minimum_expected_training_firings_per_feature=32,
            training_fraction=0.90,
        )
        self.assertEqual(result["by_observation_to_feature_ratio"], 65536)
        self.assertEqual(result["by_expected_feature_firings"], 16384)
        self.assertEqual(result["required_training_observations"], 65536)
        self.assertEqual(
            result["required_total_observations_before_train_validation_split"],
            72818,
        )

    def test_current_pack_is_not_sae_ready(self) -> None:
        plan = load_capacity_plan(PLAN_PATH)
        profiles = {row["id"]: row for row in plan["profiles"]}
        self.assertEqual(set(row["status"] for row in profiles.values()), {"insufficient"})
        pilot = profiles["bounded_context_pilot"]
        self.assertEqual(
            pilot["minimum_pairs_per_training_template_group_for_fresh_pack"], 143
        )
        self.assertEqual(pilot["estimated_full_pack_rows"], 25168)
        replayed = profiles["legacy_anchor_default"]["specific_compartments"][
            "replayed"
        ]
        self.assertEqual(replayed["episodes"], 24)
        self.assertEqual(replayed["unique_pairs"], 24)
        self.assertEqual(replayed["training_observations"], 21)

    def test_tampered_plan_is_rejected(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        plan["status"] = "ready"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content address mismatch"):
                load_capacity_plan(path)

    def test_training_gate_rejects_current_insufficient_profile(self) -> None:
        plan = load_capacity_plan(PLAN_PATH)
        capture = self._capture_manifest(plan, rows_per_episode=128)
        with self.assertRaisesRegex(ValueError, "profile is insufficient"):
            authorize_compartment_capacity(
                plan=plan,
                profile_id="bounded_context_pilot",
                capture_manifest=capture,
                activation_category_name="replayed",
                input_dim=4096,
                expansion=2,
                top_k=16,
                selected_metadata=[{"pair_id": "pair-0"}],
            )

    def test_training_gate_binds_sampling_and_observed_support(self) -> None:
        plan = copy.deepcopy(load_capacity_plan(PLAN_PATH))
        profile = next(
            row for row in plan["profiles"] if row["id"] == "legacy_anchor_default"
        )
        plan["status"] = "ready"
        profile["status"] = "ready"
        profile["specific_compartments"]["replayed"]["status"] = "ready"
        profile["requirements"][
            "required_total_observations_before_train_validation_split"
        ] = 32
        capture = self._capture_manifest(plan, rows_per_episode=1)
        metadata = [
            {
                "episode_id": f"episode-{index}",
                "pair_id": f"pair-{index}",
                "condition": "replayed_host_visible",
                "activation_category": "replayed",
                "phase": "pre_gate",
                "token_offset_from_proposal_start": 0,
                "is_proposal_start": True,
            }
            for index in range(32)
        ]
        receipt = authorize_compartment_capacity(
            plan=plan,
            profile_id="legacy_anchor_default",
            capture_manifest=capture,
            activation_category_name="replayed",
            input_dim=4096,
            expansion=8,
            top_k=32,
            selected_metadata=metadata,
        )
        self.assertEqual(receipt["status"], "pass")
        self.assertEqual(receipt["selected_unique_pairs"], 32)
        capture["capture"]["sae_training_sampling"]["generated_tokens"] = 1
        with self.assertRaisesRegex(ValueError, "prompt-only"):
            authorize_compartment_capacity(
                plan=plan,
                profile_id="legacy_anchor_default",
                capture_manifest=capture,
                activation_category_name="replayed",
                input_dim=4096,
                expansion=8,
                top_k=32,
                selected_metadata=metadata,
            )

    def test_prompt_window_contract_is_exact_and_prompt_only(self) -> None:
        self.assertEqual(prompt_window_positions(10, 4), [6, 7, 8, 9])
        with self.assertRaisesRegex(ValueError, "fewer than required"):
            prompt_window_positions(3, 4)
        validate_sae_training_window_request(
            window_size=128,
            split_role="sae_training",
            phases=["pre_gate"],
            adapter=None,
        )
        validate_sae_training_window_request(
            window_size=128,
            split_role="calibration",
            phases=["pre_gate"],
            adapter=None,
        )
        with self.assertRaisesRegex(ValueError, "sae_training or calibration"):
            validate_sae_training_window_request(
                window_size=128,
                split_role="final_evaluation",
                phases=["pre_gate"],
                adapter=None,
            )
        with self.assertRaisesRegex(ValueError, "base-model"):
            validate_sae_training_window_request(
                window_size=128,
                split_role="sae_training",
                phases=["pre_gate"],
                adapter="adapter",
            )

    def test_prompt_window_metadata_preserves_whole_episode(self) -> None:
        episode = {
            "episode_id": "episode-1",
            "pair_id": "pair-1",
            "condition": "replayed_host_visible",
            "template_group": "calendar_reschedule",
            "factors": {},
            "labels": {},
            "observed_gate_decision": "deny",
            "actual_execution": False,
        }
        rows = prompt_window_metadata_rows(
            episode=episode,
            split_provenance={
                "activation_category_split_id": "1" * 64,
                "activation_category_split_role": "sae_training",
            },
            model_variant="base",
            prompt_length=8,
            positions={"proposal_start": 7},
            window_size=4,
            first_row_index=20,
        )
        self.assertEqual([row["row_index"] for row in rows], [20, 21, 22, 23])
        self.assertEqual(
            [row["token_offset_from_proposal_start"] for row in rows],
            [-3, -2, -1, 0],
        )
        self.assertEqual(sum(row["is_proposal_start"] for row in rows), 1)
        self.assertEqual(
            validate_prompt_window_metadata(rows, rows_per_episode=4),
            {"episodes": 1, "rows": 4},
        )

    def test_train_validation_split_is_pair_atomic(self) -> None:
        metadata = [
            {
                "pair_id": f"pair-{pair}",
                "episode_id": f"episode-{pair}-{condition}",
            }
            for pair in range(10)
            for condition in range(2)
            for _token in range(4)
        ]
        selected = np.arange(len(metadata))
        train, validation, receipt = pair_atomic_train_validation_split(
            metadata, selected, seed=19
        )
        train_pairs = {metadata[int(index)]["pair_id"] for index in train}
        validation_pairs = {
            metadata[int(index)]["pair_id"] for index in validation
        }
        self.assertFalse(train_pairs & validation_pairs)
        self.assertEqual(train_pairs | validation_pairs, {f"pair-{i}" for i in range(10)})
        self.assertEqual(receipt["training_pairs"], 9)
        self.assertEqual(receipt["validation_pairs"], 1)
        self.assertTrue(receipt["exact_selected_row_partition"])

    @staticmethod
    def _capture_manifest(plan: dict, *, rows_per_episode: int) -> dict:
        activation_row_count = 32 * rows_per_episode
        return {
            "source": {
                "activation_category_split": {
                    "activation_category_split_id": plan["source"]["split_id"],
                    "activation_category_split_manifest_sha256": plan["source"][
                        "split_manifest_sha256"
                    ],
                },
                "capture_resource_plan": {
                    "resource_plan_id": "2" * 64,
                    "resource_plan_sha256": "3" * 64,
                    "capacity_plan_id": plan["capacity_plan_id"],
                    "pack_id": plan["source"]["pack_id"],
                    "split_id": plan["source"]["split_id"],
                },
            },
            "capture": {
                "rows": activation_row_count,
                "sae_training_sampling": {
                    "schema": "sch.vpd-sae-training-sampling.v1",
                    "mode": "prompt_context_window",
                    "generated_tokens": 0,
                    "activation_rows_per_episode": rows_per_episode,
                    "activation_row_count": activation_row_count,
                    "resource_plan_id": "2" * 64,
                    "activation_storage": "npy_row_shards",
                }
            },
            "artifacts": {
                "activations": {
                    "schema": "sch.vpd-sharded-activation-artifact.v1"
                }
            },
        }


if __name__ == "__main__":
    unittest.main()
