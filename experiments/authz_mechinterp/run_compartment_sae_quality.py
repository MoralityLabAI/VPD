from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.activation_categories import (
    COMPONENT_COMPARTMENT_PROTOCOLS,
    activation_category,
    category_contract,
)
from experiments.authz_mechinterp.activation_shards import (
    SHARDED_ACTIVATION_SCHEMA,
    ShardedActivationAccessor,
    verify_sharded_activation_artifact,
)
from experiments.authz_mechinterp.admit_compartment_sae_dictionary_bundle import (
    verify_bundle_for_quality,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    stable_hash,
    write_json,
)
from experiments.authz_mechinterp.evaluate_compartment_sae_quality import (
    DEFAULT_QUALITY_SAMPLING_POLICY,
    DEFAULT_SAE_DATA_SELECTION_SEED,
    build_quality_report,
    evaluate_compartment_bundle_streaming,
    verify_quality_report,
)
from experiments.authz_mechinterp.verify_activation_capture_semantics import (
    validate_semantic_verification_receipt,
    verify_prompt_window_capture_semantics,
)


FROZEN_DATA_SELECTION_SEED = DEFAULT_SAE_DATA_SELECTION_SEED


def _content_addressed_json(path: Path, *, schema: str, id_field: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema:
        raise ValueError(f"unsupported manifest schema: {path}")
    body = dict(value)
    identifier = str(body.pop(id_field, ""))
    if content_sha256(body) != identifier:
        raise ValueError(f"manifest content address mismatch: {path}")
    return value


def _confined_artifact(directory: Path, relative_path: str) -> Path:
    root = directory.resolve()
    candidate = (directory / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("dictionary manifest artifact escapes its directory")
    return candidate


def load_calibration_capture(
    *,
    capture_dir: Path,
    capture_manifest_path: Path,
    metadata_path: Path,
    activation_key: str,
    episodes_per_category: int = 64,
    sampling_seed: int = 270829,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    ShardedActivationAccessor,
    np.ndarray,
    dict[str, Any],
]:
    if episodes_per_category < 1 or sampling_seed < 0:
        raise ValueError("quality sampling parameters are invalid")
    manifest = _content_addressed_json(
        capture_manifest_path,
        schema="sch.vpd-activation-capture-manifest.v1",
        id_field="capture_id",
    )
    split = manifest.get("source", {}).get("activation_category_split")
    if not isinstance(split, dict) or split.get(
        "activation_category_split_verification_status"
    ) != "pass":
        raise ValueError("quality capture lacks verified split provenance")
    if split.get("activation_category_split_role") != "calibration":
        raise ValueError("compartment SAE quality capture must use calibration")
    if manifest.get("model", {}).get("adapter_sha256") is not None:
        raise ValueError("compartment SAE quality requires base-model activations")
    capture = manifest["capture"]
    if activation_key not in set(capture["activation_keys"]):
        raise ValueError("quality activation key is absent from the capture")
    sampling = capture.get("sae_training_sampling")
    if not isinstance(sampling, dict) or sampling.get("mode") != (
        "prompt_context_window"
    ):
        raise ValueError("quality capture lacks the prompt-window sampling contract")
    if int(sampling.get("generated_tokens", -1)) != 0 or sampling.get(
        "window_end_anchor"
    ) != "proposal_start":
        raise ValueError("quality capture is not prompt-only at proposal_start")
    if sha256_file(metadata_path) != str(
        manifest["artifacts"]["metadata"]["sha256"]
    ):
        raise ValueError("quality metadata differs from its capture manifest")
    artifact = manifest["artifacts"]["activations"]
    if artifact.get("schema") != SHARDED_ACTIVATION_SCHEMA:
        raise ValueError("quality evaluation requires sharded activation storage")
    verification = verify_sharded_activation_artifact(capture_dir, artifact)
    if int(verification["rows"]) != int(capture["rows"]):
        raise ValueError("quality capture row counts differ")

    rows_per_episode = int(sampling["activation_rows_per_episode"])
    episode_records: list[dict[str, Any]] = []
    seen_episodes: set[str] = set()
    current: dict[str, Any] | None = None
    split_id = str(split["activation_category_split_id"])
    with metadata_path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            row = json.loads(line)
            if int(row.get("row_index", -1)) != row_index:
                raise ValueError("quality metadata row indices are not contiguous")
            if row.get("activation_category_split_role") != "calibration" or str(
                row.get("activation_category_split_id")
            ) != split_id:
                raise ValueError("quality metadata uses different split provenance")
            if row.get("phase") != "pre_gate":
                raise ValueError("quality metadata contains a non-pre_gate row")
            episode_id = str(row.get("episode_id", ""))
            pair_id = str(row.get("pair_id", ""))
            category = activation_category(row)
            if not episode_id or not pair_id:
                raise ValueError("quality metadata lacks episode or pair identity")
            if current is None or current["episode_id"] != episode_id:
                if current is not None:
                    episode_records.append(current)
                if episode_id in seen_episodes:
                    raise ValueError("quality metadata episode rows are not contiguous")
                seen_episodes.add(episode_id)
                current = {
                    "episode_id": episode_id,
                    "pair_id": pair_id,
                    "category": category,
                    "start_row": row_index,
                    "rows": 0,
                }
            assert current is not None
            if current["pair_id"] != pair_id or current["category"] != category:
                raise ValueError("quality metadata changes identity within an episode")
            ordinal = int(row.get("sampled_token_ordinal", -1))
            offset = int(row.get("token_offset_from_proposal_start", 1))
            if ordinal != int(current["rows"]) or offset != 1 - rows_per_episode + ordinal:
                raise ValueError("quality metadata does not preserve the exact prompt window")
            current["rows"] = int(current["rows"]) + 1
    if current is not None:
        episode_records.append(current)
    if sum(int(row["rows"]) for row in episode_records) != int(capture["rows"]):
        raise ValueError("quality metadata and activation rows do not align")
    for row in episode_records:
        if int(row["rows"]) != rows_per_episode:
            raise ValueError("quality metadata contains an incomplete prompt window")
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in episode_records:
        by_category.setdefault(str(row["category"]), []).append(row)
    required_categories = {
        str(row["id"]) for row in category_contract()["categories"]
    }
    if set(by_category) != required_categories:
        raise ValueError("quality calibration capture has incomplete category coverage")
    chosen_records: list[dict[str, Any]] = []
    selected_episode_ids: dict[str, list[str]] = {}
    selected_pair_ids: dict[str, list[str]] = {}
    for category in sorted(by_category):
        ordered = sorted(
            by_category[category],
            key=lambda row: (
                stable_hash(
                    sampling_seed,
                    "compartment-sae-quality-calibration",
                    split_id,
                    category,
                    row["episode_id"],
                ),
                row["episode_id"],
            ),
        )
        if len(ordered) < episodes_per_category:
            raise ValueError(f"quality calibration has too few episodes for {category}")
        selected = ordered[:episodes_per_category]
        chosen_records.extend(selected)
        selected_episode_ids[category] = sorted(
            str(row["episode_id"]) for row in selected
        )
        selected_pair_ids[category] = sorted(
            {str(row["pair_id"]) for row in selected}
        )
    chosen_records.sort(key=lambda row: int(row["start_row"]))
    selected_indices = np.asarray(
        [
            index
            for row in chosen_records
            for index in range(
                int(row["start_row"]), int(row["start_row"]) + rows_per_episode
            )
        ],
        dtype=np.int64,
    )
    categories = np.asarray(
        [
            str(row["category"])
            for row in chosen_records
            for _ in range(rows_per_episode)
        ],
        dtype=object,
    )
    sampling_receipt = {
        "schema": "sch.vpd-compartment-sae-quality-sampling.v1",
        "seed": sampling_seed,
        "selection_basis": "stable_hash_of_split_category_and_episode_id",
        "observable_activation_category_only": True,
        "model_outcomes_used_for_selection": False,
        "source_rows": int(capture["rows"]),
        "source_episodes": len(episode_records),
        "rows_per_episode": rows_per_episode,
        "episodes_per_category": episodes_per_category,
        "selected_rows": len(selected_indices),
        "selected_episodes": len(chosen_records),
        "selected_episode_id_set_sha256_by_category": {
            category: content_sha256(values)
            for category, values in sorted(selected_episode_ids.items())
        },
        "selected_pair_id_set_sha256_by_category": {
            category: content_sha256(values)
            for category, values in sorted(selected_pair_ids.items())
        },
        "selected_row_index_sha256": content_sha256(selected_indices.tolist()),
        "whole_prompt_windows_only": True,
        "final_evaluation_rows_used": False,
    }
    accessor = ShardedActivationAccessor(
        capture_dir,
        artifact,
        activation_key,
        verify_artifact=False,
    )
    return manifest, categories, accessor, selected_indices, sampling_receipt


def load_dictionary_descriptors(
    *,
    dictionary_dirs: list[Path],
    capture_manifest: dict[str, Any],
    activation_key: str,
) -> list[dict[str, Any]]:
    required_categories = {
        str(row["id"]) for row in category_contract()["categories"]
    }
    allowed_compartments = required_categories | {"shared_cross_category"}
    quality_split = capture_manifest["source"]["activation_category_split"]
    quality_split_id = str(quality_split["activation_category_split_id"])
    descriptors: list[dict[str, Any]] = []
    for directory in dictionary_dirs:
        manifest_path = directory / "sae_dictionary_manifest.json"
        manifest = _content_addressed_json(
            manifest_path,
            schema="sch.vpd-fixed-sae-dictionary-manifest.v1",
            id_field="dictionary_id",
        )
        artifact_record = manifest["artifacts"]["dictionary"]
        artifact_path = _confined_artifact(directory, str(artifact_record["path"]))
        if sha256_file(artifact_path) != str(artifact_record["sha256"]):
            raise ValueError("quality dictionary differs from its manifest")
        if manifest.get("model") != capture_manifest.get("model"):
            raise ValueError("quality dictionary and capture use different base models")
        dictionary_contract = manifest["dictionary"]
        if str(dictionary_contract["activation_key"]) != activation_key:
            raise ValueError("quality dictionary uses a different activation key")
        if int(dictionary_contract.get("data_selection_seed", -1)) != (
            FROZEN_DATA_SELECTION_SEED
        ) or int(
            dictionary_contract.get("train_validation_split", {}).get("seed", -1)
        ) != FROZEN_DATA_SELECTION_SEED:
            raise ValueError("quality replicas use different data-selection seeds")
        compartment = dictionary_contract.get("activation_compartment")
        if not isinstance(compartment, dict):
            raise ValueError("quality dictionary lacks an activation compartment")
        category = str(compartment.get("activation_category"))
        if category not in allowed_compartments:
            raise ValueError("quality dictionary uses an unknown compartment")
        if str(compartment.get("protocol")) != COMPONENT_COMPARTMENT_PROTOCOLS[
            category
        ]:
            raise ValueError("quality dictionary uses the wrong category protocol")
        if str(compartment.get("category_contract_sha256")) != content_sha256(
            category_contract()
        ):
            raise ValueError("quality dictionary uses a different category contract")
        source = manifest["source"]
        semantic = source.get("capture_semantic_verification", {})
        validate_semantic_verification_receipt(semantic)
        training_split = source.get("activation_category_split")
        if not isinstance(training_split, dict) or training_split.get(
            "activation_category_split_role"
        ) != "sae_training":
            raise ValueError("quality dictionary lacks the sae_training split role")
        if str(training_split.get("activation_category_split_id")) != quality_split_id:
            raise ValueError("quality dictionary and capture use different splits")
        if training_split.get("activation_category_split_verification_status") != "pass":
            raise ValueError("quality dictionary split provenance is unverified")
        quality_artifact_id = content_sha256(
            capture_manifest["artifacts"]["activations"]
        )
        if str(source.get("activation_capture_sha256")) == quality_artifact_id:
            raise ValueError("quality dictionary training and calibration captures match")
        capacity = source.get("capacity_authorization")
        if not isinstance(capacity, dict) or capacity.get("status") != "pass":
            raise ValueError("quality dictionary lacks passed capacity authorization")
        if str(capacity.get("activation_category")) != category or compartment.get(
            "capacity_authorization"
        ) != capacity:
            raise ValueError("quality dictionary capacity compartment differs")
        if len(str(source.get("capacity_plan_sha256", ""))) != 64 or len(
            str(capacity.get("capacity_plan_id", ""))
        ) != 64:
            raise ValueError("quality dictionary capacity provenance is incomplete")
        replica_id = f"seed-{int(dictionary_contract['seed'])}"
        descriptors.append(
            {
                "dictionary_id": str(manifest["dictionary_id"]),
                "compartment": category,
                "replica_id": replica_id,
                "initialization_seed": int(dictionary_contract["seed"]),
                "data_selection_seed": FROZEN_DATA_SELECTION_SEED,
                "manifest_path": manifest_path,
                "artifact_path": artifact_path,
                "manifest": manifest,
                "source_receipt": {
                    "dictionary_id": str(manifest["dictionary_id"]),
                    "dictionary_manifest_sha256": sha256_file(manifest_path),
                    "dictionary_sha256": sha256_file(artifact_path),
                    "compartment": category,
                    "replica_id": replica_id,
                    "initialization_seed": int(dictionary_contract["seed"]),
                    "data_selection_seed": FROZEN_DATA_SELECTION_SEED,
                    "capacity_authorization": capacity,
                },
            }
        )
    return descriptors


def load_dictionary_arrays(descriptor: dict[str, Any]) -> dict[str, Any]:
    with np.load(descriptor["artifact_path"], allow_pickle=False) as archive:
        required = {
            "activation_key",
            "decoder",
            "encoder_weight",
            "encoder_bias",
            "pre_bias",
            "top_k",
        }
        if required - set(archive.files):
            raise ValueError("quality dictionary artifact is incomplete")
        activation_key = str(np.asarray(archive["activation_key"]).item())
        if activation_key != str(
            descriptor["manifest"]["dictionary"]["activation_key"]
        ):
            raise ValueError("quality dictionary artifact changed its activation key")
        result = {
            "dictionary_id": descriptor["dictionary_id"],
            "compartment": descriptor["compartment"],
            "replica_id": descriptor["replica_id"],
            "decoder": np.asarray(archive["decoder"], dtype=np.float32),
            "encoder_weight": np.asarray(
                archive["encoder_weight"], dtype=np.float32
            ),
            "encoder_bias": np.asarray(archive["encoder_bias"], dtype=np.float32),
            "pre_bias": np.asarray(archive["pre_bias"], dtype=np.float32),
            "top_k": int(np.asarray(archive["top_k"]).item()),
        }
    contract = descriptor["manifest"]["dictionary"]
    expected_shape = (int(contract["dict_dim"]), int(contract["input_dim"]))
    if result["decoder"].shape != expected_shape or result[
        "encoder_weight"
    ].shape != expected_shape:
        raise ValueError("quality dictionary arrays differ from manifest dimensions")
    if result["encoder_bias"].shape != (expected_shape[0],) or result[
        "pre_bias"
    ].shape != (expected_shape[1],):
        raise ValueError("quality dictionary bias arrays differ from its manifest")
    if result["top_k"] != int(contract["top_k"]):
        raise ValueError("quality dictionary top-k differs from its manifest")
    if not all(
        np.isfinite(result[name]).all()
        for name in ("decoder", "encoder_weight", "encoder_bias", "pre_bias")
    ):
        raise ValueError("quality dictionary contains non-finite arrays")
    return result


def run_quality_evaluation(
    *,
    capture_dir: Path,
    capture_manifest_path: Path,
    metadata_path: Path,
    activation_key: str,
    dictionary_dirs: list[Path],
    output_path: Path,
    batch_rows: int,
    episodes_per_category: int = 64,
    sampling_seed: int = 270829,
    source_dataset_path: Path | None = None,
    split_manifest_path: Path | None = None,
    training_plan_path: Path | None = None,
    dictionary_bundle_receipt_path: Path | None = None,
    training_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if episodes_per_category != int(
        DEFAULT_QUALITY_SAMPLING_POLICY["episodes_per_category"]
    ) or sampling_seed != int(DEFAULT_QUALITY_SAMPLING_POLICY["seed"]):
        raise ValueError("quality evaluation differs from the frozen sampling policy")
    if source_dataset_path is not None and split_manifest_path is not None:
        semantic_verification = verify_prompt_window_capture_semantics(
            capture_dir=capture_dir,
            capture_manifest_path=capture_manifest_path,
            dataset_path=source_dataset_path,
            split_manifest_path=split_manifest_path,
            expected_role="calibration",
        )
    elif source_dataset_path is not None or split_manifest_path is not None:
        raise ValueError("quality capture semantic admission inputs are incomplete")
    elif training_provenance is not None:
        semantic_verification = {
            "schema": "sch.vpd-prompt-window-capture-semantic-verification.v1",
            "status": "unattested_injected_test_fixture",
            "empirical_quality_claim_allowed": False,
        }
    else:
        raise ValueError("quality evaluation requires source dataset semantic admission")
    (
        capture_manifest,
        categories,
        accessor,
        selected_indices,
        sampling_receipt,
    ) = load_calibration_capture(
        capture_dir=capture_dir,
        capture_manifest_path=capture_manifest_path,
        metadata_path=metadata_path,
        activation_key=activation_key,
        episodes_per_category=episodes_per_category,
        sampling_seed=sampling_seed,
    )
    try:
        descriptors = load_dictionary_descriptors(
            dictionary_dirs=dictionary_dirs,
            capture_manifest=capture_manifest,
            activation_key=activation_key,
        )
        dictionary_sources = [row["source_receipt"] for row in descriptors]
        if training_provenance is not None and (
            training_plan_path is not None or dictionary_bundle_receipt_path is not None
        ):
            raise ValueError("quality evaluation has conflicting bundle admission inputs")
        if training_provenance is None:
            if training_plan_path is None or dictionary_bundle_receipt_path is None:
                raise ValueError("quality evaluation requires a dictionary bundle receipt")
            training_provenance = verify_bundle_for_quality(
                receipt_path=dictionary_bundle_receipt_path,
                training_plan_path=training_plan_path,
                dictionary_sources=dictionary_sources,
            )
        required_categories = [
            str(row["id"]) for row in category_contract()["categories"]
        ]
        result = evaluate_compartment_bundle_streaming(
            row_count=len(selected_indices),
            hidden_size=accessor.shape[1],
            read_rows=lambda indices: accessor.read_rows(selected_indices[indices]),
            categories=categories,
            dictionary_descriptors=descriptors,
            load_dictionary=load_dictionary_arrays,
            required_specific_categories=required_categories,
            batch_rows=batch_rows,
        )
        artifact = capture_manifest["artifacts"]["activations"]
        report = build_quality_report(
            result=result,
            dictionary_sources=dictionary_sources,
            activation_category_split=capture_manifest["source"][
                "activation_category_split"
            ],
            holdout_capture={
                "capture_id": str(capture_manifest["capture_id"]),
                "capture_manifest_sha256": sha256_file(capture_manifest_path),
                "activation_capture_sha256": content_sha256(artifact),
                "metadata_sha256": sha256_file(metadata_path),
                "activation_key": activation_key,
                "quality_runner_sha256": sha256_file(Path(__file__)),
                "storage_schema": SHARDED_ACTIVATION_SCHEMA,
                "capture_semantic_verification": semantic_verification,
                "quality_sampling": sampling_receipt,
            },
            training_provenance=training_provenance,
        )
        write_json(output_path, report)
        return report
    finally:
        accessor.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate compartment SAE replicas on a sharded calibration capture."
    )
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--capture-manifest", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--source-dataset", type=Path)
    parser.add_argument("--activation-category-split-manifest", type=Path)
    parser.add_argument("--activation-key")
    parser.add_argument("--dictionary-dir", type=Path, action="append", default=[])
    parser.add_argument("--training-plan", type=Path)
    parser.add_argument("--dictionary-bundle-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--episodes-per-category", type=int, default=64)
    parser.add_argument("--sampling-seed", type=int, default=270829)
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()
    if args.verify_report is not None:
        report = verify_quality_report(args.verify_report)
    else:
        required = {
            "--capture-dir": args.capture_dir,
            "--capture-manifest": args.capture_manifest,
            "--metadata": args.metadata,
            "--source-dataset": args.source_dataset,
            "--activation-category-split-manifest": (
                args.activation_category_split_manifest
            ),
            "--activation-key": args.activation_key,
            "--output": args.output,
            "--training-plan": args.training_plan,
            "--dictionary-bundle-receipt": args.dictionary_bundle_receipt,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing or not args.dictionary_dir:
            parser.error(
                "evaluation requires "
                + ", ".join([*missing, *( [] if args.dictionary_dir else ["--dictionary-dir"] )])
            )
        if args.batch_rows < 1:
            parser.error("--batch-rows must be positive")
        report = run_quality_evaluation(
            capture_dir=args.capture_dir,
            capture_manifest_path=args.capture_manifest,
            metadata_path=args.metadata,
            activation_key=args.activation_key,
            dictionary_dirs=args.dictionary_dir,
            output_path=args.output,
            batch_rows=args.batch_rows,
            episodes_per_category=args.episodes_per_category,
            sampling_seed=args.sampling_seed,
            source_dataset_path=args.source_dataset,
            split_manifest_path=args.activation_category_split_manifest,
            training_plan_path=args.training_plan,
            dictionary_bundle_receipt_path=args.dictionary_bundle_receipt,
        )
    print(
        json.dumps(
            {
                "schema": "sch.vpd-compartment-sae-quality-cli-result.v1",
                "status": report["status"],
                "quality_report_id": report["quality_report_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
