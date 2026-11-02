from __future__ import annotations

import copy
import unittest
from pathlib import Path

from experiments.authz_mechinterp.capture_activations import (
    prompt_window_metadata_rows,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256
from experiments.authz_mechinterp.verify_activation_capture_semantics import (
    validate_prompt_window_metadata_against_episodes,
)
from experiments.authz_mechinterp.benchmark_activation_capture_semantics import (
    verify_benchmark,
)


SPLIT_ID = "1" * 64
SPLIT_ROLE = "sae_training"
ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = (
    ROOT / "artifacts" / "vpd_activation_capture_semantic_benchmark_v1" / "report.json"
)


def fixture() -> tuple[list[dict], list[dict], dict]:
    episodes = [
        {
            "episode_id": "episode-a",
            "pair_id": "pair-a",
            "condition": "replayed_host_visible",
            "template_group": "calendar_reschedule",
            "prompt_version": "v3",
        },
        {
            "episode_id": "episode-b",
            "pair_id": "pair-b",
            "condition": "valid_visible",
            "template_group": "document_share",
            "prompt_version": "v3",
        },
    ]
    split = {
        "activation_category_split_id": SPLIT_ID,
        "activation_category_split_role": SPLIT_ROLE,
    }
    metadata: list[dict] = []
    for episode in episodes:
        metadata.extend(
            prompt_window_metadata_rows(
                episode=episode,
                split_provenance=split,
                model_variant="base",
                prompt_length=8,
                positions={"proposal_start": 7},
                window_size=4,
                first_row_index=len(metadata),
            )
        )
    sampling = {
        "schema": "sch.vpd-sae-training-sampling.v1",
        "mode": "prompt_context_window",
        "generated_tokens": 0,
        "window_end_anchor": "proposal_start",
        "activation_rows_per_episode": 4,
        "token_offset_start": -3,
        "token_offset_end": 0,
        "exact_rows_per_episode": True,
        "episode_count": 2,
        "activation_row_count": 8,
        "episode_id_set_sha256": content_sha256(["episode-a", "episode-b"]),
        "pair_id_set_sha256": content_sha256(["pair-a", "pair-b"]),
    }
    manifest_body = {
        "source": {"activation_category_split": split},
        "model": {"model_variant": "base"},
        "capture": {"rows": 8, "sae_training_sampling": sampling},
    }
    manifest = {**manifest_body, "capture_id": content_sha256(manifest_body)}
    return episodes, metadata, manifest


class ActivationCaptureSemanticTests(unittest.TestCase):
    def test_durable_semantic_admission_benchmark_reproduces(self) -> None:
        result = verify_benchmark(BENCHMARK)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["scenarios"], 7)
        self.assertEqual(
            result["benchmark_id"],
            "51ebf5afb2cf0e1cc109ed8a7672d9ff8f0adb7c65fb4e442128164fbc5f0964",
        )

    def test_exact_episode_order_category_and_window_pass(self) -> None:
        episodes, metadata, manifest = fixture()
        result = validate_prompt_window_metadata_against_episodes(
            metadata=metadata,
            episodes=episodes,
            capture_manifest=manifest,
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["rows"], 8)
        self.assertEqual(
            result["activation_categories"], ["replayed", "valid_authority"]
        )

    def test_reordered_episode_windows_are_rejected(self) -> None:
        episodes, metadata, manifest = fixture()
        substituted = metadata[4:] + metadata[:4]
        with self.assertRaisesRegex(ValueError, "source episode"):
            validate_prompt_window_metadata_against_episodes(
                metadata=substituted,
                episodes=episodes,
                capture_manifest=manifest,
            )

    def test_relabelled_activation_category_is_rejected(self) -> None:
        episodes, metadata, manifest = fixture()
        substituted = copy.deepcopy(metadata)
        substituted[0]["activation_category"] = "valid_authority"
        with self.assertRaisesRegex(ValueError, "activation_category"):
            validate_prompt_window_metadata_against_episodes(
                metadata=substituted,
                episodes=episodes,
                capture_manifest=manifest,
            )

    def test_shifted_window_coordinate_is_rejected(self) -> None:
        episodes, metadata, manifest = fixture()
        substituted = copy.deepcopy(metadata)
        substituted[2]["token_offset_from_proposal_start"] = -2
        with self.assertRaisesRegex(ValueError, "token coordinate"):
            validate_prompt_window_metadata_against_episodes(
                metadata=substituted,
                episodes=episodes,
                capture_manifest=manifest,
            )

    def test_substituted_source_episode_digest_is_rejected(self) -> None:
        episodes, metadata, manifest = fixture()
        substituted = copy.deepcopy(metadata)
        substituted[-1]["source_episode_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source_episode_sha256"):
            validate_prompt_window_metadata_against_episodes(
                metadata=substituted,
                episodes=episodes,
                capture_manifest=manifest,
            )


if __name__ == "__main__":
    unittest.main()
