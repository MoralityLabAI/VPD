from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.authz_mechinterp.activation_categories import (
    COMPONENT_COMPARTMENT_PROTOCOLS,
    category_contract,
)
from experiments.authz_mechinterp.activation_shards import (
    ActivationShardWriter,
    build_sharded_activation_artifact,
)
from experiments.authz_mechinterp.benchmark_compartment_sae_quality import (
    clean_dictionaries,
    synthetic_holdout,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.evaluate_compartment_sae_quality import (
    verify_quality_report,
)
from experiments.authz_mechinterp.merge_vpd_effect_shards import (
    write_deterministic_npz,
)
from experiments.authz_mechinterp.run_compartment_sae_quality import (
    load_calibration_capture,
    run_quality_evaluation,
)


class CompartmentSaeQualityRunnerTests(unittest.TestCase):
    def build_fixture(self, root: Path) -> dict:
        episode_activations, episode_categories, bases = synthetic_holdout(260829)
        rows_per_episode = 128
        activations = np.repeat(episode_activations, rows_per_episode, axis=0)
        categories = np.repeat(episode_categories, rows_per_episode)
        activation_key = "layer_35__proposal_start"
        split_id = "d" * 64
        capture_dir = root / "calibration-capture"
        capture_dir.mkdir()
        writer = ActivationShardWriter(
            output_dir=capture_dir,
            key=activation_key,
            hidden_size=activations.shape[1],
            dtype=np.dtype("float32"),
            shard_rows=4096,
        )
        writer.append(activations)
        artifact = build_sharded_activation_artifact({activation_key: writer.finalize()})
        metadata_path = capture_dir / "metadata.jsonl"
        metadata_path.write_text(
            "".join(
                json.dumps(
                    {
                        "row_index": index,
                        "episode_id": f"calibration-{index // rows_per_episode}",
                        "pair_id": f"calibration-pair-{index // rows_per_episode}",
                        "activation_category": str(category),
                        "activation_category_split_id": split_id,
                        "activation_category_split_role": "calibration",
                        "phase": "pre_gate",
                        "sampled_token_ordinal": index % rows_per_episode,
                        "token_offset_from_proposal_start": (
                            index % rows_per_episode + 1 - rows_per_episode
                        ),
                    },
                    sort_keys=True,
                )
                + "\n"
                for index, category in enumerate(categories)
            ),
            encoding="utf-8",
        )
        model = {
            "model_id": "example/base-model",
            "revision": "revision-1",
            "tokenizer_id": "example/base-model",
            "model_variant": "base",
            "adapter_sha256": None,
        }
        split = {
            "activation_category_split_id": split_id,
            "activation_category_split_role": "calibration",
            "activation_category_split_verification_status": "pass",
        }
        capture_body = {
            "schema": "sch.vpd-activation-capture-manifest.v1",
            "source": {
                "dataset_sha256": "1" * 64,
                "runner_sha256": "2" * 64,
                "activation_category_split": split,
            },
            "model": model,
            "capture": {
                "n_layers": 36,
                "captured_layers": [35],
                "hidden_size": activations.shape[1],
                "rows": len(activations),
                "activation_keys": [activation_key],
                "phases": ["pre_gate"],
                "max_new_tokens": 0,
                "dtype": "float32",
                "sae_training_sampling": {
                    "schema": "sch.vpd-sae-training-sampling.v1",
                    "mode": "prompt_context_window",
                    "generated_tokens": 0,
                    "window_end_anchor": "proposal_start",
                    "activation_rows_per_episode": rows_per_episode,
                    "token_offset_start": 1 - rows_per_episode,
                    "token_offset_end": 0,
                    "exact_rows_per_episode": True,
                    "episode_count": len(episode_activations),
                    "activation_row_count": len(activations),
                    "activation_storage": "npy_row_shards",
                },
            },
            "artifacts": {
                "activations": artifact,
                "metadata": {
                    "path": metadata_path.name,
                    "sha256": sha256_file(metadata_path),
                },
                "generations": {"path": "generations.jsonl", "sha256": "3" * 64},
            },
        }
        capture_manifest_path = capture_dir / "capture_manifest.json"
        write_json(
            capture_manifest_path,
            {**capture_body, "capture_id": content_sha256(capture_body)},
        )

        dictionary_dirs = []
        semantic_body = {
            "schema": "sch.vpd-prompt-window-capture-semantic-verification.v1",
            "status": "pass",
            "exact_source_episode_order_verified": True,
            "exact_category_labels_verified": True,
            "exact_prompt_window_coordinates_verified": True,
            "model_outcomes_used": False,
        }
        for dictionary in clean_dictionaries(bases):
            directory = root / f"dictionary-{dictionary['dictionary_id']}"
            directory.mkdir()
            dictionary_path = directory / "sae_dictionary.npz"
            write_deterministic_npz(
                dictionary_path,
                {
                    "activation_key": np.asarray(activation_key),
                    "decoder": dictionary["decoder"],
                    "encoder_weight": dictionary["encoder_weight"],
                    "encoder_bias": dictionary["encoder_bias"],
                    "pre_bias": dictionary["pre_bias"],
                    "top_k": np.asarray(dictionary["top_k"], dtype=np.int64),
                },
            )
            category = str(dictionary["compartment"])
            capacity = {
                "status": "pass",
                "capacity_plan_id": "4" * 64,
                "profile_id": "bounded_context_pilot",
                "activation_category": category,
            }
            compartment = {
                "schema": "sch.vpd-sae-activation-compartment.v1",
                "activation_category": category,
                "protocol": COMPONENT_COMPARTMENT_PROTOCOLS[category],
                "category_contract_sha256": content_sha256(category_contract()),
                "capacity_authorization": capacity,
            }
            training_split = {
                "activation_category_split_id": split_id,
                "activation_category_split_role": "sae_training",
                "activation_category_split_verification_status": "pass",
            }
            seed = int(str(dictionary["replica_id"]).removeprefix("seed-"))
            manifest_body = {
                "schema": "sch.vpd-fixed-sae-dictionary-manifest.v1",
                "source": {
                    "activation_capture_sha256": "5" * 64,
                    "capture_manifest_sha256": "6" * 64,
                    "capture_id": "7" * 64,
                    "capture_semantic_verification": {
                        **semantic_body,
                        "semantic_verification_id": content_sha256(semantic_body),
                    },
                    "runner_sha256": "8" * 64,
                    "activation_category_split": training_split,
                    "capacity_plan_sha256": "9" * 64,
                    "capacity_authorization": capacity,
                },
                "model": model,
                "dictionary": {
                    "activation_key": activation_key,
                    "input_dim": activations.shape[1],
                    "dict_dim": dictionary["decoder"].shape[0],
                    "top_k": dictionary["top_k"],
                    "expansion": 1,
                    "steps": 10,
                    "seed": seed,
                    "data_selection_seed": 17,
                    "activation_compartment": compartment,
                    "train_validation_split": {"status": "pass", "seed": 17},
                },
                "artifacts": {
                    "dictionary": {
                        "path": dictionary_path.name,
                        "sha256": sha256_file(dictionary_path),
                    }
                },
            }
            write_json(
                directory / "sae_dictionary_manifest.json",
                {
                    **manifest_body,
                    "dictionary_id": content_sha256(manifest_body),
                },
            )
            dictionary_dirs.append(directory)
        return {
            "capture_dir": capture_dir,
            "capture_manifest": capture_manifest_path,
            "metadata": metadata_path,
            "activation_key": activation_key,
            "dictionary_dirs": dictionary_dirs,
        }

    def test_sharded_runner_emits_a_verifiable_pass_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.build_fixture(root)
            output = root / "quality-report.json"
            report = run_quality_evaluation(
                capture_dir=fixture["capture_dir"],
                capture_manifest_path=fixture["capture_manifest"],
                metadata_path=fixture["metadata"],
                activation_key=fixture["activation_key"],
                dictionary_dirs=fixture["dictionary_dirs"],
                output_path=output,
                batch_rows=256,
                episodes_per_category=64,
                sampling_seed=270829,
                training_provenance={
                    "schema": "sch.vpd-compartment-sae-dictionary-bundle-authorization.v1",
                    "status": "pass",
                    "training_plan_id": "4" * 64,
                    "training_plan_sha256": "5" * 64,
                    "dictionary_bundle_id": "6" * 64,
                    "dictionary_bundle_sha256": "7" * 64,
                },
            )
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report, verify_quality_report(output))
            self.assertEqual(
                report["result"]["coverage"]["evaluation_mode"],
                "bounded_memory_streaming",
            )
            self.assertEqual(
                report["result"]["coverage"]["dictionary_loading"],
                "one_replica_at_a_time",
            )
            self.assertEqual(
                report["source"]["holdout_capture"]["quality_sampling"][
                    "selected_rows"
                ],
                65536,
            )

    def test_final_evaluation_capture_is_rejected_before_activation_reads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.build_fixture(root)
            manifest = json.loads(fixture["capture_manifest"].read_text(encoding="utf-8"))
            body = {key: value for key, value in manifest.items() if key != "capture_id"}
            body["source"]["activation_category_split"][
                "activation_category_split_role"
            ] = "final_evaluation"
            write_json(
                fixture["capture_manifest"],
                {**body, "capture_id": content_sha256(body)},
            )
            with self.assertRaisesRegex(ValueError, "must use calibration"):
                load_calibration_capture(
                    capture_dir=fixture["capture_dir"],
                    capture_manifest_path=fixture["capture_manifest"],
                    metadata_path=fixture["metadata"],
                    activation_key=fixture["activation_key"],
                )

    def test_quality_sampling_is_deterministic_and_whole_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.build_fixture(root)
            first = load_calibration_capture(
                capture_dir=fixture["capture_dir"],
                capture_manifest_path=fixture["capture_manifest"],
                metadata_path=fixture["metadata"],
                activation_key=fixture["activation_key"],
            )
            second = load_calibration_capture(
                capture_dir=fixture["capture_dir"],
                capture_manifest_path=fixture["capture_manifest"],
                metadata_path=fixture["metadata"],
                activation_key=fixture["activation_key"],
            )
            try:
                np.testing.assert_array_equal(first[1], second[1])
                np.testing.assert_array_equal(first[3], second[3])
                self.assertEqual(first[4], second[4])
                self.assertTrue(first[4]["whole_prompt_windows_only"])
            finally:
                first[2].close()
                second[2].close()

    def test_tampered_activation_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.build_fixture(root)
            manifest = json.loads(fixture["capture_manifest"].read_text(encoding="utf-8"))
            shard = manifest["artifacts"]["activations"]["keys"][
                fixture["activation_key"]
            ]["shards"][0]
            shard_path = fixture["capture_dir"] / shard["path"]
            data = bytearray(shard_path.read_bytes())
            data[-1] ^= 1
            shard_path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "activation shard differs"):
                load_calibration_capture(
                    capture_dir=fixture["capture_dir"],
                    capture_manifest_path=fixture["capture_manifest"],
                    metadata_path=fixture["metadata"],
                    activation_key=fixture["activation_key"],
                )


if __name__ == "__main__":
    unittest.main()
