from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json
from experiments.authz_mechinterp.merge_vpd_effect_shards import write_deterministic_npz
from experiments.authz_mechinterp.merge_vpd_sae_results import merge_results, sign_result
from experiments.authz_mechinterp.run_focused_vpd_sae import (
    finalize_study,
    focused_screen_endpoint,
    prepare_study,
)
from tests.test_vpd_sae_result_merge import make_bundle, public_base64


class FocusedVpdSaeRunnerTests(unittest.TestCase):
    def test_specific_screen_uses_observable_category_not_behavior_outcome(self) -> None:
        metadata = [
            {
                "activation_category": category,
                "factors": {"semantic_authorization": outcome},
            }
            for category, outcome in (
                ("replayed", 0),
                ("replayed", 0),
                ("valid_authority", 1),
                ("signature_invalid", 0),
            )
        ]
        labels, receipt = focused_screen_endpoint(
            metadata=metadata,
            label_name="semantic_authorization",
            compartment_category="replayed",
        )
        self.assertEqual(labels.tolist(), [1, 1, 0, 0])
        self.assertEqual(
            receipt["kind"], "observable_activation_category_membership"
        )
        self.assertFalse(receipt["model_outcome_label_used_for_selection"])

    def test_shared_screen_keeps_the_behavior_endpoint(self) -> None:
        metadata = [
            {
                "activation_category": category,
                "factors": {"semantic_authorization": outcome},
            }
            for category, outcome in (
                ("replayed", 0),
                ("valid_authority", 1),
            )
        ]
        labels, receipt = focused_screen_endpoint(
            metadata=metadata,
            label_name="semantic_authorization",
            compartment_category="shared_cross_category",
        )
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertEqual(receipt["kind"], "behavior_label")
        self.assertTrue(receipt["model_outcome_label_used_for_selection"])

    def prepare_inputs(self, root: Path) -> dict:
        bundle, packets, hashes = make_bundle(root)
        worker_ids = {
            str(packet["distribution"]["assigned_worker"])
            for packet in packets.values()
        }
        keys = {worker_id: Ed25519PrivateKey.generate() for worker_id in worker_ids}
        registry = root / "registry.json"
        registry.write_text(
            json.dumps(
                {
                    "schema": "sch.vpd-sae-worker-registry.v1",
                    "workers": [
                        {
                            "worker_id": worker_id,
                            "algorithm": "ed25519",
                            "public_key_base64": public_base64(key),
                            "enabled": True,
                        }
                        for worker_id, key in sorted(keys.items())
                    ],
                }
            ),
            encoding="utf-8",
        )
        packet = next(iter(packets.values()))
        packet_id = str(packet["packet_id"])
        alignment_key = str(packet["sae_study"]["alignment_activation_key"])

        all_component_ids = sorted(
            str(component_id)
            for value in packets.values()
            for component_id in value["component_ids"]
        )
        effects = np.asarray(
            [
                [1.0 + 0.1 * index, 0.1 * (index % 2), 0.05]
                for index in range(len(all_component_ids))
            ],
            dtype=np.float32,
        )
        effects_path = root / "activation_effects.npz"
        write_deterministic_npz(
            effects_path,
            {
                "component_ids": np.asarray(all_component_ids),
                "delta_h": effects,
            },
        )
        effect_body = {
            "schema": "sch.vpd-effect-shard-merge.v1",
            "status": "complete",
            "identity": {
                "model_id": "example/model",
                "revision": "revision-1",
                "adapter_sha256": "a" * 64,
                "dataset_sha256": "b" * 64,
                "runner_sha256": "c" * 64,
                "effect_space_id": alignment_key,
                "phase": "pre_gate",
                "estimator": "mean_central_difference_lora_rank_multiplier",
                "perturbation_epsilon": 0.1,
                "selection_seed": 1707,
                "episode_order_sha256": "d" * 64,
                "episodes": 48,
                "generated_tokens": 0,
            },
            "artifacts": {
                "activation_effects": {
                    "path": effects_path.name,
                    "sha256": sha256_file(effects_path),
                }
            },
        }
        effect_manifest = root / "effect_merge_manifest.json"
        write_json(
            effect_manifest,
            {**effect_body, "merge_id": content_sha256(effect_body)},
        )

        dictionary_path = root / "sae_dictionary.npz"
        write_deterministic_npz(
            dictionary_path,
            {
                "activation_key": np.asarray(alignment_key),
                "decoder": np.eye(3, dtype=np.float32),
                "encoder_weight": np.eye(3, dtype=np.float32),
                "encoder_bias": np.zeros(3, dtype=np.float32),
                "pre_bias": np.zeros(3, dtype=np.float32),
                "top_k": np.asarray(1, dtype=np.int64),
            },
        )
        dictionary_body = {
            "schema": "sch.vpd-fixed-sae-dictionary-manifest.v1",
            "source": {
                "activation_capture_sha256": "2" * 64,
                "capture_manifest_sha256": "3" * 64,
                "capture_id": "4" * 64,
                "runner_sha256": "5" * 64,
            },
            "model": {
                "model_id": "example/model",
                "revision": "revision-1",
                "tokenizer_id": "example/model",
                "model_variant": "base",
                "adapter_sha256": None,
            },
            "dictionary": {
                "activation_key": alignment_key,
                "input_dim": 3,
                "dict_dim": 3,
                "top_k": 1,
                "expansion": 1,
                "steps": 10,
                "seed": 17,
            },
            "artifacts": {
                "dictionary": {
                    "path": dictionary_path.name,
                    "sha256": sha256_file(dictionary_path),
                }
            },
        }
        dictionary_manifest = root / "sae_dictionary_manifest.json"
        write_json(
            dictionary_manifest,
            {**dictionary_body, "dictionary_id": content_sha256(dictionary_body)},
        )

        metadata = []
        activations = []
        families = ["family-a", "family-b", "family-c"]
        for family_index, family in enumerate(families):
            for pair_index in range(8):
                for label in (0, 1):
                    metadata.append(
                        {
                            "phase": "pre_gate",
                            "pair_id": f"{family}-{pair_index}",
                            "template_group": family,
                            "prompt_tokens": 40 + pair_index + family_index,
                            "positions": {"proposal_start": 30 + pair_index},
                            "factors": {"semantic_authorization": label},
                        }
                    )
                    activations.append(
                        [3.0 + 0.1 * pair_index, 0.1, 0.0]
                        if label
                        else [0.1, 3.0 + 0.1 * pair_index, 0.0]
                    )
        activation_path = root / "activations.npz"
        write_deterministic_npz(
            activation_path,
            {alignment_key: np.asarray(activations, dtype=np.float32)},
        )
        metadata_path = root / "metadata.jsonl"
        metadata_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata),
            encoding="utf-8",
        )
        capture_body = {
            "schema": "sch.vpd-activation-capture-manifest.v1",
            "source": {"dataset_sha256": "e" * 64, "runner_sha256": "f" * 64},
            "model": {
                "model_id": "example/model",
                "revision": "revision-1",
                "tokenizer_id": "example/model",
                "model_variant": "base",
                "adapter_sha256": None,
            },
            "capture": {
                "n_layers": 36,
                "captured_layers": [35],
                "hidden_size": 3,
                "rows": len(metadata),
                "activation_keys": [alignment_key],
                "phases": ["pre_gate"],
                "max_new_tokens": 1,
                "dtype": "float32",
            },
            "artifacts": {
                "activations": {
                    "path": activation_path.name,
                    "sha256": sha256_file(activation_path),
                },
                "metadata": {
                    "path": metadata_path.name,
                    "sha256": sha256_file(metadata_path),
                },
                "generations": {"path": "generations.jsonl", "sha256": "1" * 64},
            },
        }
        capture_manifest = root / "capture_manifest.json"
        write_json(
            capture_manifest,
            {**capture_body, "capture_id": content_sha256(capture_body)},
        )
        intervention_dataset = root / "intervention_episodes.jsonl"
        intervention_rows = [
            {
                "episode_id": f"intervention-{family}-{index}",
                "pair_id": f"intervention-{family}-{index}",
                "template_group": family,
                "messages": [
                    {"role": "user", "content": f"{family} prompt {index}"}
                ],
                "expected_candidate_decision": "REQUEST",
            }
            for family in families
            for index in range(2)
        ]
        intervention_dataset.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in intervention_rows),
            encoding="utf-8",
        )
        return {
            "bundle": bundle,
            "registry": registry,
            "packets": packets,
            "hashes": hashes,
            "keys": keys,
            "packet": packet,
            "packet_id": packet_id,
            "alignment_key": alignment_key,
            "effects": effects_path,
            "effect_manifest": effect_manifest,
            "dictionary": dictionary_path,
            "dictionary_manifest": dictionary_manifest,
            "activations": activation_path,
            "capture_manifest": capture_manifest,
            "metadata": metadata_path,
            "intervention_dataset": intervention_dataset,
        }

    def plan(self, fixture: dict) -> tuple[dict, dict]:
        return prepare_study(
            bundle_dir=fixture["bundle"],
            packet_id=fixture["packet_id"],
            dictionary_path=fixture["dictionary"],
            dictionary_manifest_path=fixture["dictionary_manifest"],
            activation_capture_path=fixture["activations"],
            activation_capture_manifest_path=fixture["capture_manifest"],
            metadata_path=fixture["metadata"],
            effect_archive_path=fixture["effects"],
            effect_manifest_path=fixture["effect_manifest"],
            intervention_dataset_path=fixture["intervention_dataset"],
            label_name="semantic_authorization",
            seed=17,
            permutations=31,
            top_features=3,
            max_pairs=1,
            maximum_max_stat_p=1.0,
            minimum_absolute_cosine=0.0,
            minimum_pairs_per_family=1,
            intervention_pairs_per_family=1,
        )

    @staticmethod
    def trace_for(plan: dict) -> dict:
        trials = []
        for candidate in plan["candidates"]:
            targets = {
                "activation": f"sae_feature:{candidate['feature_index']}",
                "parameter": candidate["component_id"],
                "matched_norm_component_control": candidate[
                    "matched_control_component_id"
                ],
            }
            slopes = {
                "activation": 0.10,
                "parameter": 0.08,
                "matched_norm_component_control": 0.01,
            }
            for episode in candidate["intervention_episodes"]:
                family = episode["template_group"]
                pair_id = episode["episode_id"]
                for kind in plan["intervention_contract"]["kinds"]:
                    for coefficient in plan["intervention_contract"]["coefficients"]:
                        trials.append(
                            {
                                "schema": "sch.vpd-focused-sae-intervention-trial.v1",
                                "candidate_id": candidate["candidate_id"],
                                "pair_id": pair_id,
                                "template_group": family,
                                "intervention_kind": kind,
                                "intervention_target_id": targets[kind],
                                "coefficient": coefficient,
                                "endpoint_value": coefficient * slopes[kind],
                                "request_group_logit": coefficient * slopes[kind],
                                "propose_group_logit": 0.0,
                                "predicted_decision": (
                                    "REQUEST" if coefficient * slopes[kind] >= 0 else "PROPOSE"
                                ),
                                "expected_decision": "PROPOSE",
                                "guardrail_violation": False,
                                **(
                                    {}
                                    if episode.get("activation_category") is None
                                    else {
                                        "activation_category": episode[
                                            "activation_category"
                                        ]
                                    }
                                ),
                            }
                        )
        return {
            "schema": "sch.vpd-focused-sae-intervention-trace.v1",
            "plan_id": plan["plan_id"],
            "worker_id": plan["worker_id"],
            "inputs": {
                "dictionary_sha256": plan["study_inputs"]["sae_dictionary_sha256"],
                "dataset_sha256": plan["study_inputs"]["intervention_dataset_sha256"],
                "runner_sha256": "9" * 64,
                "model_id": plan["study_inputs"]["model_id"],
                "revision": plan["study_inputs"]["revision"],
                "adapter_sha256": plan["study_inputs"]["adapter_sha256"],
                "alignment_activation_key": plan["study_inputs"][
                    "alignment_activation_key"
                ],
                "phase": plan["study_inputs"]["phase"],
            },
            "decision_tokens": {
                "request_first_token_ids": [10],
                "propose_first_token_ids": [11],
                "margin": "logsumexp(REQUEST)-logsumexp(PROPOSE)",
            },
            "factor_restoration_verified": True,
            "trials": trials,
        }

    def test_plan_trace_signed_result_and_merge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.prepare_inputs(root)
            plan, context = self.plan(fixture)
            self.assertEqual(plan["study_inputs"]["alignment_activation_key"], fixture["alignment_key"])
            self.assertEqual(
                plan["study_inputs"]["packet_effect_lineage"]["status"],
                "unattested_legacy_metadata_pilot",
            )
            self.assertFalse(
                plan["study_inputs"]["packet_effect_lineage"][
                    "empirical_category_claim_allowed"
                ]
            )
            self.assertEqual(len(plan["candidates"]), 1)
            self.assertEqual(plan["screen"]["features_tested"], 3)
            trace_path = root / "trace.json"
            trace_path.write_text(json.dumps(self.trace_for(plan)), encoding="utf-8")
            result = finalize_study(
                plan=plan,
                packet=context["packet"],
                intervention_trace_path=trace_path,
            )
            self.assertEqual(result["status"], "complete")
            self.assertGreater(result["candidates"][0]["activation_intervention_effect"], 0)
            self.assertGreater(result["candidates"][0]["parameter_intervention_effect"], 0)
            worker_id = str(result["worker_id"])
            result_path = root / "signed-result.json"
            result_path.write_text(
                json.dumps(sign_result(result, fixture["keys"][worker_id])),
                encoding="utf-8",
            )
            manifest = merge_results(
                bundle_dir=fixture["bundle"],
                registry_path=fixture["registry"],
                result_paths=[result_path],
                output_dir=root / "sae-merge",
                require_complete=False,
            )
            self.assertEqual(manifest["coverage"]["accepted_results"], 1)
            self.assertEqual(manifest["evidence"]["candidate_rows"], 1)

    def test_dictionary_at_different_activation_space_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.prepare_inputs(root)
            write_deterministic_npz(
                fixture["dictionary"],
                {
                    "activation_key": np.asarray("layer_0__proposal_start"),
                    "decoder": np.eye(3, dtype=np.float32),
                    "encoder_weight": np.eye(3, dtype=np.float32),
                    "encoder_bias": np.zeros(3, dtype=np.float32),
                    "pre_bias": np.zeros(3, dtype=np.float32),
                    "top_k": np.asarray(1, dtype=np.int64),
                },
            )
            with self.assertRaisesRegex(ValueError, "different activation key"):
                self.plan(fixture)

    def test_compartment_dictionary_without_capacity_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.prepare_inputs(root)
            manifest = json.loads(fixture["dictionary_manifest"].read_text())
            body = {key: value for key, value in manifest.items() if key != "dictionary_id"}
            body["dictionary"]["activation_compartment"] = {
                "schema": "sch.vpd-sae-activation-compartment.v1",
                "activation_category": "replayed",
            }
            write_json(
                fixture["dictionary_manifest"],
                {**body, "dictionary_id": content_sha256(body)},
            )
            with self.assertRaisesRegex(ValueError, "capacity authorization"):
                self.plan(fixture)

    def test_capacity_authorized_compartment_requires_quality_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.prepare_inputs(root)
            manifest = json.loads(fixture["dictionary_manifest"].read_text())
            body = {key: value for key, value in manifest.items() if key != "dictionary_id"}
            capacity = {
                "status": "pass",
                "capacity_plan_id": "6" * 64,
                "profile_id": "bounded_context_pilot",
                "activation_category": "replayed",
            }
            body["source"]["capacity_authorization"] = capacity
            body["source"]["capacity_plan_sha256"] = "7" * 64
            body["dictionary"]["activation_compartment"] = {
                "schema": "sch.vpd-sae-activation-compartment.v1",
                "activation_category": "replayed",
                "capacity_authorization": capacity,
            }
            write_json(
                fixture["dictionary_manifest"],
                {**body, "dictionary_id": content_sha256(body)},
            )
            with self.assertRaisesRegex(ValueError, "quality report"):
                self.plan(fixture)

    def test_incomplete_factorial_trace_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.prepare_inputs(root)
            plan, context = self.plan(fixture)
            trace = self.trace_for(plan)
            trace["trials"].pop()
            trace_path = root / "incomplete-trace.json"
            trace_path.write_text(json.dumps(trace), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "factorial cells"):
                finalize_study(
                    plan=plan,
                    packet=context["packet"],
                    intervention_trace_path=trace_path,
                )

    def test_shared_plan_reports_effects_by_activation_category(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.prepare_inputs(root)
            plan, context = self.plan(fixture)
            candidate = plan["candidates"][0]
            for index, episode in enumerate(candidate["intervention_episodes"]):
                episode["activation_category"] = (
                    "replayed" if index % 2 == 0 else "valid_authority"
                )
            candidate_body = {
                key: value for key, value in candidate.items() if key != "candidate_id"
            }
            candidate["candidate_id"] = content_sha256(candidate_body)
            plan_body = {key: value for key, value in plan.items() if key != "plan_id"}
            plan["plan_id"] = content_sha256(plan_body)
            trace_path = root / "shared-trace.json"
            trace_path.write_text(json.dumps(self.trace_for(plan)), encoding="utf-8")
            result = finalize_study(
                plan=plan,
                packet=context["packet"],
                intervention_trace_path=trace_path,
            )
            category_effects = result["candidates"][0]["activation_category_effects"]
            self.assertEqual(
                set(category_effects), {"replayed", "valid_authority"}
            )
            self.assertTrue(
                all(value["n_pairs"] > 0 for value in category_effects.values())
            )


if __name__ == "__main__":
    unittest.main()
