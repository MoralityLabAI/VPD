from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from experiments.authz_mechinterp.activation_categories import activation_category
from experiments.authz_mechinterp.activation_category_split_contract import (
    verify_activation_category_split_role,
)
from experiments.authz_mechinterp.activation_shards import (
    SHARDED_ACTIVATION_SCHEMA,
    verify_sharded_activation_artifact,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
)


SCHEMA = "sch.vpd-prompt-window-capture-semantic-verification.v1"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
                yield value


def _confined(directory: Path, relative_path: str) -> Path:
    root = directory.resolve()
    path = (directory / relative_path).resolve()
    if path != root and root not in path.parents:
        raise ValueError("capture artifact escapes its output directory")
    return path


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"capture metadata has a non-integer {field}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"capture metadata has a non-integer {field}") from exc
    if result != value:
        raise ValueError(f"capture metadata has a non-integral {field}")
    return result


def validate_semantic_verification_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "pass":
        raise ValueError("capture semantic verification receipt did not pass")
    body = dict(receipt)
    identifier = str(body.pop("semantic_verification_id", ""))
    if content_sha256(body) != identifier:
        raise ValueError("capture semantic verification content address mismatch")
    if not all(
        receipt.get(field) is True
        for field in (
            "exact_source_episode_order_verified",
            "exact_category_labels_verified",
            "exact_prompt_window_coordinates_verified",
        )
    ) or receipt.get("model_outcomes_used") is not False:
        raise ValueError("capture semantic verification weakens its boundary")


def validate_prompt_window_metadata_against_episodes(
    *,
    metadata: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    capture_manifest: dict[str, Any],
) -> dict[str, Any]:
    sampling = capture_manifest.get("capture", {}).get("sae_training_sampling", {})
    if sampling.get("schema") != "sch.vpd-sae-training-sampling.v1" or sampling.get(
        "mode"
    ) != "prompt_context_window":
        raise ValueError("capture does not declare prompt-window sampling")
    if int(sampling.get("generated_tokens", -1)) != 0:
        raise ValueError("prompt-window semantic admission requires zero generated tokens")
    rows_per_episode = int(sampling.get("activation_rows_per_episode", 0))
    if rows_per_episode < 1:
        raise ValueError("prompt-window semantic admission requires positive window size")
    if len(metadata) != len(episodes) * rows_per_episode:
        raise ValueError("capture metadata does not exactly cover the source episodes")
    if len({str(row.get("episode_id", "")) for row in episodes}) != len(episodes):
        raise ValueError("source dataset repeats an episode ID")

    split = capture_manifest.get("source", {}).get("activation_category_split", {})
    split_id = str(split.get("activation_category_split_id", ""))
    split_role = str(split.get("activation_category_split_role", ""))
    model_variant = str(capture_manifest.get("model", {}).get("model_variant", ""))
    if not split_id or not split_role or not model_variant:
        raise ValueError("capture manifest lacks semantic identity fields")

    next_row = 0
    observed_pairs: set[str] = set()
    observed_categories: set[str] = set()
    for episode in episodes:
        episode_id = str(episode.get("episode_id", ""))
        if not episode_id:
            raise ValueError("source dataset contains an empty episode ID")
        pair_id = str(episode.get("pair_id", episode_id))
        expected_category = activation_category(episode)
        episode_digest = content_sha256(episode)
        observed_pairs.add(pair_id)
        observed_categories.add(expected_category)
        rows = metadata[next_row : next_row + rows_per_episode]
        if len(rows) != rows_per_episode:
            raise ValueError("capture metadata truncates a prompt window")
        prompt_lengths = {
            _require_int(row.get("prompt_tokens"), "prompt_tokens") for row in rows
        }
        if len(prompt_lengths) != 1:
            raise ValueError("capture metadata changes prompt length within an episode")
        prompt_length = next(iter(prompt_lengths))
        if prompt_length < rows_per_episode:
            raise ValueError("capture metadata window is longer than its prompt")
        proposal_start = prompt_length - 1
        for ordinal, row in enumerate(rows):
            expected_identity = {
                "row_index": next_row + ordinal,
                "episode_id": episode_id,
                "prompt_version": episode.get("prompt_version", "v1"),
                "model_variant": model_variant,
                "pair_id": pair_id,
                "condition": episode["condition"],
                "activation_category": expected_category,
                "template_group": episode["template_group"],
                "phase": "pre_gate",
                "source_episode_sha256": episode_digest,
                "activation_category_split_id": split_id,
                "activation_category_split_role": split_role,
            }
            for field, expected in expected_identity.items():
                if row.get(field) != expected:
                    raise ValueError(
                        f"capture metadata differs from source episode: {episode_id}:{field}"
                    )
            expected_position = prompt_length - rows_per_episode + ordinal
            numeric = {
                "total_tokens": prompt_length,
                "proposal_start_position": proposal_start,
                "sampled_token_position": expected_position,
                "sampled_token_ordinal": ordinal,
                "token_offset_from_proposal_start": ordinal - rows_per_episode + 1,
            }
            for field, expected in numeric.items():
                if _require_int(row.get(field), field) != expected:
                    raise ValueError(
                        f"capture metadata has a non-reproducible token coordinate: "
                        f"{episode_id}:{field}"
                    )
            if row.get("is_proposal_start") is not (ordinal == rows_per_episode - 1):
                raise ValueError(
                    f"capture metadata has a non-reproducible proposal anchor: {episode_id}"
                )
        next_row += rows_per_episode

    expected_episode_hash = content_sha256(
        sorted(str(episode["episode_id"]) for episode in episodes)
    )
    expected_pair_hash = content_sha256(sorted(observed_pairs))
    sampling_expectations = {
        "exact_rows_per_episode": True,
        "episode_count": len(episodes),
        "activation_row_count": len(metadata),
        "episode_id_set_sha256": expected_episode_hash,
        "pair_id_set_sha256": expected_pair_hash,
        "token_offset_start": 1 - rows_per_episode,
        "token_offset_end": 0,
        "window_end_anchor": "proposal_start",
    }
    for field, expected in sampling_expectations.items():
        if sampling.get(field) != expected:
            raise ValueError(f"capture sampling receipt differs: {field}")
    if int(capture_manifest.get("capture", {}).get("rows", -1)) != len(metadata):
        raise ValueError("capture manifest row count differs from semantic metadata")

    body = {
        "schema": SCHEMA,
        "status": "pass",
        "capture_id": str(capture_manifest.get("capture_id", "")),
        "split_id": split_id,
        "split_role": split_role,
        "episodes": len(episodes),
        "rows": len(metadata),
        "rows_per_episode": rows_per_episode,
        "activation_categories": sorted(observed_categories),
        "episode_id_set_sha256": expected_episode_hash,
        "pair_id_set_sha256": expected_pair_hash,
        "exact_source_episode_order_verified": True,
        "exact_category_labels_verified": True,
        "exact_prompt_window_coordinates_verified": True,
        "model_outcomes_used": False,
    }
    return {**body, "semantic_verification_id": content_sha256(body)}


def verify_prompt_window_capture_semantics(
    *,
    capture_dir: Path,
    capture_manifest_path: Path,
    dataset_path: Path,
    split_manifest_path: Path,
    expected_role: str,
) -> dict[str, Any]:
    manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "sch.vpd-activation-capture-manifest.v1":
        raise ValueError("unsupported activation capture manifest schema")
    body = dict(manifest)
    capture_id = str(body.pop("capture_id", ""))
    if content_sha256(body) != capture_id:
        raise ValueError("activation capture manifest content address mismatch")
    if sha256_file(dataset_path) != str(manifest.get("source", {}).get("dataset_sha256")):
        raise ValueError("activation capture source dataset differs")
    split_verification = verify_activation_category_split_role(
        dataset_path, split_manifest_path, expected_role
    )
    if manifest.get("source", {}).get("activation_category_split") != split_verification:
        raise ValueError("activation capture split verification is not reproducible")

    artifacts = manifest.get("artifacts", {})
    metadata_record = artifacts.get("metadata", {})
    metadata_path = _confined(capture_dir, str(metadata_record.get("path", "")))
    if sha256_file(metadata_path) != str(metadata_record.get("sha256", "")):
        raise ValueError("activation capture metadata differs")
    generations_record = artifacts.get("generations", {})
    generations_path = _confined(capture_dir, str(generations_record.get("path", "")))
    if sha256_file(generations_path) != str(generations_record.get("sha256", "")):
        raise ValueError("activation capture generations differ")
    activation_artifact = artifacts.get("activations", {})
    if activation_artifact.get("schema") != SHARDED_ACTIVATION_SCHEMA:
        raise ValueError("prompt-window semantic admission requires sharded activations")
    shard_verification = verify_sharded_activation_artifact(
        capture_dir, activation_artifact
    )
    metadata = list(iter_jsonl(metadata_path))
    episodes = list(iter_jsonl(dataset_path))
    result = validate_prompt_window_metadata_against_episodes(
        metadata=metadata,
        episodes=episodes,
        capture_manifest=manifest,
    )
    if int(shard_verification["rows"]) != int(result["rows"]):
        raise ValueError("activation shard rows differ from semantic metadata")
    addressed = dict(result)
    addressed.pop("semantic_verification_id", None)
    addressed.update(
        {
        "capture_manifest_sha256": sha256_file(capture_manifest_path),
        "dataset_sha256": sha256_file(dataset_path),
        "metadata_sha256": sha256_file(metadata_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "activation_artifact_sha256": content_sha256(activation_artifact),
        }
    )
    receipt = {
        **addressed,
        "semantic_verification_id": content_sha256(addressed),
    }
    validate_semantic_verification_receipt(receipt)
    return receipt
