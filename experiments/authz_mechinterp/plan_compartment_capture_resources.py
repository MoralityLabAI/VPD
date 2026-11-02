from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.plan_compartment_sae_capacity import (
    verify_capacity_plan,
)


DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}


def build_capture_resource_plan(
    *,
    capacity_plan_path: Path,
    pack_dir: Path,
    split_dir: Path | None = None,
    profile_id: str,
    layer_indices: list[int],
    dtype: str = "float16",
    shard_rows: int = 4096,
    maximum_shard_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    verification = verify_capacity_plan(
        capacity_plan_path,
        pack_dir=pack_dir,
        split_dir=split_dir,
    )
    capacity = json.loads(capacity_plan_path.read_text(encoding="utf-8"))
    profiles = {str(row["id"]): row for row in capacity["profiles"]}
    if profile_id not in profiles:
        raise ValueError(f"unknown capacity profile: {profile_id}")
    profile = profiles[profile_id]
    if profile.get("status") != "ready":
        raise ValueError("capture resource planning requires a support-ready profile")
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported capture dtype: {dtype}")
    if not layer_indices or len(set(layer_indices)) != len(layer_indices):
        raise ValueError("capture layers must be a nonempty unique list")
    if any(index < 0 for index in layer_indices):
        raise ValueError("capture layer indices must be nonnegative")
    if shard_rows < 1:
        raise ValueError("activation shard rows must be positive")
    hidden_size = int(capacity["model_space"]["hidden_size"])
    bytes_per_value = DTYPE_BYTES[dtype]
    rows_per_episode = int(profile["activation_rows_per_episode"])
    episodes = int(capacity["observed_training_role"]["rows"])
    activation_rows = episodes * rows_per_episode
    bytes_per_row = hidden_size * bytes_per_value
    maximum_full_shard_bytes = shard_rows * bytes_per_row
    if maximum_full_shard_bytes > maximum_shard_bytes:
        raise ValueError("activation shard exceeds the frozen maximum byte size")
    shard_count = math.ceil(activation_rows / shard_rows)
    final_shard_rows = activation_rows - shard_rows * (shard_count - 1)
    role_episode_counts = {"sae_training": episodes}
    if split_dir is not None:
        split_manifest_path = split_dir / "split_manifest.json"
        split = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        if sha256_file(split_manifest_path) != str(
            capacity["source"]["split_manifest_sha256"]
        ) or str(split.get("split_id")) != str(capacity["source"]["split_id"]):
            raise ValueError("capture resources use a different activation split")
        role_episode_counts = {
            role: int(split["splits"][role]["rows"])
            for role in ("sae_training", "calibration")
        }
    capture_roles: dict[str, Any] = {}
    for role, role_episodes in sorted(role_episode_counts.items()):
        role_rows = role_episodes * rows_per_episode
        role_shards = math.ceil(role_rows / shard_rows)
        capture_roles[role] = {
            "episodes": role_episodes,
            "activation_rows": role_rows,
            "array_bytes_per_layer": role_rows * bytes_per_row,
            "array_bytes_all_layers": role_rows * bytes_per_row * len(layer_indices),
            "shards_per_layer": role_shards,
            "final_shard_rows": role_rows - shard_rows * (role_shards - 1),
        }
    project_root = Path(__file__).resolve().parents[2]
    implementation_paths = {
        "shard_artifact": (
            project_root / "experiments/authz_mechinterp/activation_shards.py"
        ),
        "capture_worker": (
            project_root / "experiments/authz_mechinterp/capture_activations.py"
        ),
        "sae_trainer": project_root / "experiments/authz_mechinterp/train_sae.py",
        "capture_semantic_verifier": (
            project_root
            / "experiments/authz_mechinterp/verify_activation_capture_semantics.py"
        ),
    }
    categories: dict[str, Any] = {}
    for category, support in sorted(profile["specific_compartments"].items()):
        rows = int(support["total_observations"])
        categories[category] = {
            "episodes": int(support["episodes"]),
            "activation_rows": rows,
            "fp16_capture_bytes_per_layer": rows * hidden_size * 2,
            "fp32_training_bytes": rows * hidden_size * 4,
        }
    shared_rows = int(profile["shared_compartment"]["total_observations"])
    body = {
        "schema": "sch.vpd-compartment-capture-resource-plan.v1",
        "status": "sharded_capture_and_streaming_training_required",
        "source": {
            "capacity_plan_id": verification["capacity_plan_id"],
            "capacity_plan_sha256": sha256_file(capacity_plan_path),
            "pack_id": capacity["source"]["pack_id"],
            "split_id": capacity["source"]["split_id"],
            "planner_sha256": sha256_file(Path(__file__)),
            "implementation": {
                name: {
                    "path": path.relative_to(project_root).as_posix(),
                    "sha256": sha256_file(path),
                }
                for name, path in implementation_paths.items()
            },
        },
        "profile": {
            "id": profile_id,
            "hidden_size": hidden_size,
            "dictionary_features": int(profile["dictionary_features"]),
            "expansion": int(profile["expansion"]),
            "top_k": int(profile["top_k"]),
            "activation_rows_per_episode": rows_per_episode,
        },
        "capture": {
            "layer_indices": layer_indices,
            "layer_count": len(layer_indices),
            "dtype": dtype,
            "bytes_per_value": bytes_per_value,
            "episodes": episodes,
            "activation_rows": activation_rows,
            "bytes_per_row": bytes_per_row,
            "array_bytes_per_layer": activation_rows * bytes_per_row,
            "array_bytes_all_layers": (
                activation_rows * bytes_per_row * len(layer_indices)
            ),
            "roles": capture_roles,
        },
        "sharding": {
            "format": "npy_v1_deterministic_row_shards",
            "rows_per_full_shard": shard_rows,
            "maximum_full_shard_bytes": maximum_full_shard_bytes,
            "maximum_allowed_shard_bytes": maximum_shard_bytes,
            "shards_per_layer": shard_count,
            "final_shard_rows": final_shard_rows,
            "metadata_write_mode": "streaming_jsonl",
            "activation_write_mode": "bounded_row_buffer",
            "training_read_mode": "memory_mapped_shards_no_full_concatenation",
        },
        "compartments": {
            "specific": categories,
            "shared_cross_category": {
                "episodes": int(profile["shared_compartment"]["episodes"]),
                "activation_rows": shared_rows,
                "fp16_capture_bytes_per_layer": shared_rows * hidden_size * 2,
                "fp32_training_bytes": shared_rows * hidden_size * 4,
                "balance_rule": profile["shared_compartment"]["balance_rule"],
            },
        },
        "invariants": {
            "monolithic_npz_capture_forbidden": True,
            "full_activation_concatenation_during_training_forbidden": True,
            "shard_row_ranges_exact_contiguous_partition": True,
            "metadata_rows_align_one_to_one_with_activation_rows": True,
            "capture_episode_and_row_counts_are_split_role_specific": True,
            "resource_plan_does_not_authorize_model_capture_or_training": True,
        },
        "scientific_boundary": {
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "storage_feasibility_is_not_sae_trainability": True,
        },
    }
    return {**body, "resource_plan_id": content_sha256(body)}


def verify_capture_resource_plan(
    path: Path,
    *,
    capacity_plan_path: Path,
    pack_dir: Path,
    split_dir: Path | None = None,
) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    body = dict(plan)
    plan_id = str(body.pop("resource_plan_id", ""))
    if content_sha256(body) != plan_id:
        raise ValueError("capture resource plan content address mismatch")
    expected = build_capture_resource_plan(
        capacity_plan_path=capacity_plan_path,
        pack_dir=pack_dir,
        split_dir=split_dir,
        profile_id=str(plan["profile"]["id"]),
        layer_indices=[int(value) for value in plan["capture"]["layer_indices"]],
        dtype=str(plan["capture"]["dtype"]),
        shard_rows=int(plan["sharding"]["rows_per_full_shard"]),
        maximum_shard_bytes=int(plan["sharding"]["maximum_allowed_shard_bytes"]),
    )
    if expected != plan:
        raise ValueError("capture resource plan is not reproducible")
    return {
        "schema": "sch.vpd-compartment-capture-resource-verification.v1",
        "status": "pass",
        "resource_plan_id": plan_id,
        "capacity_plan_id": plan["source"]["capacity_plan_id"],
        "activation_rows": plan["capture"]["activation_rows"],
        "shards_per_layer": plan["sharding"]["shards_per_layer"],
        "capture_roles": {
            role: {
                "episodes": int(record["episodes"]),
                "activation_rows": int(record["activation_rows"]),
                "shards_per_layer": int(record["shards_per_layer"]),
            }
            for role, record in sorted(plan["capture"].get("roles", {}).items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or verify sharded compartment activation capture resources."
    )
    parser.add_argument("--capacity-plan", type=Path, required=True)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--profile", default="bounded_context_pilot")
    parser.add_argument("--layers", default="35")
    parser.add_argument("--dtype", choices=sorted(DTYPE_BYTES), default="float16")
    parser.add_argument("--shard-rows", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_capture_resource_plan(
            args.verify,
            capacity_plan_path=args.capacity_plan,
            pack_dir=args.pack_dir,
            split_dir=args.split_dir,
        )
    else:
        if args.output is None:
            parser.error("planning requires --output")
        result = build_capture_resource_plan(
            capacity_plan_path=args.capacity_plan,
            pack_dir=args.pack_dir,
            split_dir=args.split_dir,
            profile_id=args.profile,
            layer_indices=[int(value) for value in args.layers.split(",")],
            dtype=args.dtype,
            shard_rows=args.shard_rows,
        )
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
