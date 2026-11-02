from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.merge_vpd_effect_shards import (
    write_deterministic_npz,
)


SCHEMA = "sch.vpd-authenticated-lora-factor-bundle-extraction.v1"
FACTOR_INDEX_SCHEMA = "sch.vpd-lora-factor-bundle-index.v1"

_DTYPES: dict[str, tuple[np.dtype[Any], int]] = {
    "F64": (np.dtype("<f8"), 8),
    "F32": (np.dtype("<f4"), 4),
    "F16": (np.dtype("<f2"), 2),
    "BF16": (np.dtype("<u2"), 2),
}


def _require_digest(value: str, *, name: str) -> str:
    digest = value.lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _read_safetensors(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    payload = path.read_bytes()
    if len(payload) < 8:
        raise ValueError("safetensors file is shorter than its header prefix")
    header_size = struct.unpack("<Q", payload[:8])[0]
    if header_size < 2 or header_size > len(payload) - 8:
        raise ValueError("safetensors header length is invalid")
    try:
        header = json.loads(payload[8 : 8 + header_size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("safetensors header is invalid JSON") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    data = memoryview(payload)[8 + header_size :]
    tensors: dict[str, np.ndarray] = {}
    intervals: list[tuple[int, int, str]] = []
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise ValueError("safetensors metadata must map strings to strings")
    for name in sorted(key for key in header if key != "__metadata__"):
        descriptor = header[name]
        if not isinstance(name, str) or not name or not isinstance(descriptor, dict):
            raise ValueError("safetensors tensor descriptor is invalid")
        dtype_name = descriptor.get("dtype")
        if dtype_name not in _DTYPES:
            raise ValueError(f"unsupported LoRA tensor dtype: {dtype_name}")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(value) is not int or value < 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
        ):
            raise ValueError(f"invalid safetensors shape or offsets: {name}")
        start, end = offsets
        dtype, itemsize = _DTYPES[dtype_name]
        element_count = math.prod(shape)
        if start < 0 or end < start or end > len(data) or end - start != element_count * itemsize:
            raise ValueError(f"safetensors byte range does not match tensor shape: {name}")
        intervals.append((start, end, name))
        raw = np.frombuffer(data[start:end], dtype=dtype, count=element_count).copy()
        if dtype_name == "BF16":
            raw = (raw.astype(np.uint32) << 16).view(np.float32)
        tensors[name] = raw.reshape(tuple(shape))
    intervals.sort()
    for (_, left_end, left_name), (right_start, _, right_name) in zip(
        intervals, intervals[1:]
    ):
        if right_start < left_end:
            raise ValueError(
                f"overlapping safetensors byte ranges: {left_name}, {right_name}"
            )
    if not tensors:
        raise ValueError("safetensors file contains no tensors")
    return tensors, metadata


def _parse_lora_key(key: str) -> tuple[str, str, str] | None:
    suffix = ".weight"
    if not key.endswith(suffix):
        return None
    stem = key[: -len(suffix)]
    for marker, side in ((".lora_A", "A"), (".lora_B", "B")):
        position = stem.rfind(marker)
        if position < 1:
            continue
        module = stem[:position]
        lane = stem[position + len(marker) :]
        if lane and not lane.startswith("."):
            return None
        if lane.startswith("."):
            lane = lane[1:]
        if "." in lane:
            return None
        return module, side, lane
    return None


def _canonical_module_name(source_name: str) -> str:
    prefix = "base_model.model."
    result = source_name[len(prefix) :] if source_name.startswith(prefix) else source_name
    if not result or result.startswith(".") or result.endswith(".") or ".." in result:
        raise ValueError(f"invalid LoRA module name: {source_name}")
    return result


def _pattern_value(
    pattern: Any,
    *,
    source_module: str,
    canonical_module: str,
    default: float,
    name: str,
) -> float:
    if pattern in (None, {}):
        return default
    if not isinstance(pattern, dict):
        raise ValueError(f"adapter config {name} must be an object")
    matches: list[tuple[int, str, Any]] = []
    for raw_key, value in pattern.items():
        key = str(raw_key)
        if (
            source_module == key
            or canonical_module == key
            or source_module.endswith("." + key)
            or canonical_module.endswith("." + key)
        ):
            matches.append((len(key), key, value))
    if not matches:
        return default
    matches.sort(reverse=True)
    longest = matches[0][0]
    winners = [row for row in matches if row[0] == longest]
    if len(winners) != 1:
        raise ValueError(f"adapter config {name} is ambiguous for {canonical_module}")
    try:
        result = float(winners[0][2])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"adapter config {name} is nonnumeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"adapter config {name} is nonfinite")
    return result


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("peft_type", "LORA") != "LORA":
        raise ValueError("adapter config is not LoRA")
    if config.get("fan_in_fan_out", False) is not False:
        raise ValueError("fan_in_fan_out LoRA requires a distinct orientation contract")
    if config.get("use_dora", False) is not False:
        raise ValueError("DoRA magnitude vectors are outside the LoRA factor contract")
    if config.get("bias", "none") != "none":
        raise ValueError("adapter bias deltas are outside the LoRA factor contract")
    if config.get("modules_to_save") not in (None, []):
        raise ValueError("modules_to_save weights are outside the LoRA factor contract")
    if type(config.get("use_rslora", False)) is not bool:
        raise ValueError("use_rslora must be boolean")


def _module_scale(
    config: dict[str, Any], *, source_module: str, canonical_module: str, tensor_rank: int
) -> tuple[float, float]:
    try:
        default_rank = float(config["r"])
        default_alpha = float(config["lora_alpha"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("adapter config requires numeric r and lora_alpha") from exc
    declared_rank = _pattern_value(
        config.get("rank_pattern"),
        source_module=source_module,
        canonical_module=canonical_module,
        default=default_rank,
        name="rank_pattern",
    )
    if not declared_rank.is_integer() or int(declared_rank) != tensor_rank:
        raise ValueError(f"configured rank differs from tensors: {canonical_module}")
    alpha = _pattern_value(
        config.get("alpha_pattern"),
        source_module=source_module,
        canonical_module=canonical_module,
        default=default_alpha,
        name="alpha_pattern",
    )
    divisor = math.sqrt(tensor_rank) if config.get("use_rslora", False) else tensor_rank
    scale = alpha / divisor
    if not math.isfinite(scale):
        raise ValueError(f"adapter scale is nonfinite: {canonical_module}")
    return alpha, scale


def extract_factor_bundle(
    *,
    adapter_weights_path: Path,
    adapter_config_path: Path,
    expected_adapter_sha256: str,
    expected_config_sha256: str,
    output_dir: Path,
    synthetic_benchmark_only: bool = False,
) -> dict[str, Any]:
    adapter_digest = _require_digest(expected_adapter_sha256, name="adapter digest")
    config_digest = _require_digest(expected_config_sha256, name="config digest")
    if sha256_file(adapter_weights_path) != adapter_digest:
        raise ValueError("adapter weights SHA-256 differs")
    if sha256_file(adapter_config_path) != config_digest:
        raise ValueError("adapter config SHA-256 differs")
    try:
        config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("adapter config is invalid JSON") from exc
    if not isinstance(config, dict):
        raise ValueError("adapter config must be an object")
    _validate_config(config)
    tensors, metadata = _read_safetensors(adapter_weights_path)
    pairs: dict[tuple[str, str], dict[str, tuple[str, np.ndarray]]] = {}
    unexpected: list[str] = []
    for tensor_key, tensor in tensors.items():
        parsed = _parse_lora_key(tensor_key)
        if parsed is None:
            unexpected.append(tensor_key)
            continue
        source_module, side, lane = parsed
        slot = pairs.setdefault((source_module, lane), {})
        if side in slot:
            raise ValueError(f"duplicate LoRA {side} tensor: {source_module}")
        slot[side] = (tensor_key, tensor)
    if unexpected:
        raise ValueError(f"unexpected non-LoRA adapter tensors: {unexpected}")
    if not pairs:
        raise ValueError("adapter contains no LoRA factor pairs")
    lanes = {lane for _, lane in pairs}
    if len(lanes) != 1:
        raise ValueError("adapter contains multiple or inconsistent LoRA lanes")
    module_records: list[tuple[str, str, dict[str, tuple[str, np.ndarray]]]] = []
    canonical_names: set[str] = set()
    for (source_module, lane), pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"orphan LoRA factor tensor: {source_module}")
        canonical_module = _canonical_module_name(source_module)
        if canonical_module in canonical_names:
            raise ValueError(f"canonical LoRA module name collision: {canonical_module}")
        canonical_names.add(canonical_module)
        module_records.append((canonical_module, source_module, pair))
    module_records.sort(key=lambda row: row[0])
    validated_records: list[dict[str, Any]] = []
    for module_index, (canonical_module, source_module, pair) in enumerate(module_records):
        a_key, a_raw = pair["A"]
        b_key, b_raw = pair["B"]
        a_factor = np.asarray(a_raw, dtype=np.float64)
        b_factor = np.asarray(b_raw, dtype=np.float64)
        if a_factor.ndim != 2 or b_factor.ndim != 2:
            raise ValueError(f"LoRA factors must be matrices: {canonical_module}")
        if a_factor.shape[0] < 1 or b_factor.shape[1] != a_factor.shape[0]:
            raise ValueError(f"LoRA factor shapes do not compose: {canonical_module}")
        if not np.all(np.isfinite(a_factor)) or not np.all(np.isfinite(b_factor)):
            raise ValueError(f"LoRA factors contain nonfinite values: {canonical_module}")
        rank = int(a_factor.shape[0])
        alpha, scale = _module_scale(
            config,
            source_module=source_module,
            canonical_module=canonical_module,
            tensor_rank=rank,
        )
        validated_records.append(
            {
                "module_index": module_index,
                "canonical_module": canonical_module,
                "source_module": source_module,
                "a_key": a_key,
                "b_key": b_key,
                "a_raw": a_raw,
                "b_raw": b_raw,
                "a_factor": a_factor,
                "b_factor": b_factor,
                "rank": rank,
                "alpha": alpha,
                "scale": scale,
            }
        )
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise ValueError("factor-bundle output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_dir = output_dir / "factors"
    factor_dir.mkdir()
    index_rows: list[dict[str, Any]] = []
    for record in validated_records:
        module_index = int(record["module_index"])
        canonical_module = str(record["canonical_module"])
        source_module = str(record["source_module"])
        a_key = str(record["a_key"])
        b_key = str(record["b_key"])
        a_raw = np.asarray(record["a_raw"])
        b_raw = np.asarray(record["b_raw"])
        a_factor = np.asarray(record["a_factor"])
        b_factor = np.asarray(record["b_factor"])
        rank = int(record["rank"])
        alpha = float(record["alpha"])
        scale = float(record["scale"])
        factor_path = factor_dir / f"module-{module_index:04d}.npz"
        write_deterministic_npz(
            factor_path,
            {"a_factor": a_factor, "b_factor": b_factor},
        )
        index_rows.append(
            {
                "module": canonical_module,
                "path": factor_path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(factor_path),
                "scale": scale,
                "lora_alpha": alpha,
                "rank": rank,
                "input_features": int(a_factor.shape[1]),
                "output_features": int(b_factor.shape[0]),
                "source_module": source_module,
                "adapter_lane": next(iter(lanes)),
                "a_tensor_key": a_key,
                "b_tensor_key": b_key,
                "a_source_dtype": str(a_raw.dtype),
                "b_source_dtype": str(b_raw.dtype),
            }
        )
    index_path = output_dir / "factor-index.json"
    factor_index = {
        "schema": FACTOR_INDEX_SCHEMA,
        "adapter_sha256": adapter_digest,
        "adapter_config_sha256": config_digest,
        "modules": index_rows,
        "synthetic_benchmark_only": synthetic_benchmark_only,
    }
    write_json(index_path, factor_index)
    body = {
        "schema": SCHEMA,
        "status": "complete_authenticated_factor_bundle_extraction",
        "source": {
            "adapter_weights_path": adapter_weights_path.name,
            "adapter_sha256": adapter_digest,
            "adapter_config_path": adapter_config_path.name,
            "adapter_config_sha256": config_digest,
            "safetensors_metadata": metadata,
        },
        "policy": {
            "plain_lora_only": True,
            "exact_source_hashes_required": True,
            "unexpected_tensors_rejected": True,
            "orphan_or_duplicate_factors_rejected": True,
            "module_names_sorted_and_unique": True,
            "factor_output_dtype": "float64",
            "scale_from_peft_config_and_per_module_patterns": True,
            "all_modules_validated_before_output_write": True,
        },
        "artifacts": {
            "factor_index": {
                "path": index_path.name,
                "sha256": sha256_file(index_path),
            },
            "factor_modules": [
                {"module": row["module"], "path": row["path"], "sha256": row["sha256"]}
                for row in index_rows
            ],
        },
        "summary": {"modules": len(index_rows), "adapter_lane": next(iter(lanes))},
        "scientific_boundary": {
            "synthetic_benchmark_only": synthetic_benchmark_only,
            "source_bytes_authenticated_by_expected_sha256": True,
            "real_adapter_source_claimed_by_invocation": not synthetic_benchmark_only,
            "real_adapter_provenance_independently_verified": False,
            "model_loaded": False,
            "torch_or_peft_loaded": False,
            "model_inference_performed": False,
            "weights_mutated": False,
            "effect_capture_performed": False,
            "execution_authorized": False,
        },
    }
    manifest = {**body, "factor_bundle_extraction_id": content_sha256(body)}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def verify_extraction(
    manifest_path: Path,
    *,
    adapter_weights_path: Path,
    adapter_config_path: Path,
) -> dict[str, Any]:
    observed = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = body.pop("factor_bundle_extraction_id", "")
    if identifier != content_sha256(body):
        raise ValueError("factor-bundle extraction content address differs")
    output_dir = manifest_path.parent
    index = observed["artifacts"]["factor_index"]
    if sha256_file(output_dir / index["path"]) != index["sha256"]:
        raise ValueError("factor index hash differs")
    for module in observed["artifacts"]["factor_modules"]:
        if sha256_file(output_dir / module["path"]) != module["sha256"]:
            raise ValueError(f"factor module hash differs: {module['module']}")
    source = observed["source"]
    with tempfile.TemporaryDirectory() as directory:
        expected = extract_factor_bundle(
            adapter_weights_path=adapter_weights_path,
            adapter_config_path=adapter_config_path,
            expected_adapter_sha256=source["adapter_sha256"],
            expected_config_sha256=source["adapter_config_sha256"],
            output_dir=Path(directory),
            synthetic_benchmark_only=bool(
                observed["scientific_boundary"]["synthetic_benchmark_only"]
            ),
        )
    if observed != expected:
        raise ValueError("factor-bundle extraction is not reproducible")
    return {
        "schema": "sch.vpd-authenticated-lora-factor-bundle-extraction-verification.v1",
        "status": "pass",
        "factor_bundle_extraction_id": identifier,
        "modules": observed["summary"]["modules"],
        "synthetic_benchmark_only": observed["scientific_boundary"][
            "synthetic_benchmark_only"
        ],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract an authenticated plain-LoRA safetensors file to factor bundle V1."
    )
    parser.add_argument("--adapter-weights", type=Path, required=True)
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--synthetic-benchmark-only", action="store_true")
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_extraction(
            args.verify,
            adapter_weights_path=args.adapter_weights,
            adapter_config_path=args.adapter_config,
        )
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required unless verifying")
        if args.expected_adapter_sha256 is None or args.expected_config_sha256 is None:
            parser.error(
                "--expected-adapter-sha256 and --expected-config-sha256 are required when extracting"
            )
        result = extract_factor_bundle(
            adapter_weights_path=args.adapter_weights,
            adapter_config_path=args.adapter_config,
            expected_adapter_sha256=args.expected_adapter_sha256,
            expected_config_sha256=args.expected_config_sha256,
            output_dir=args.output_dir,
            synthetic_benchmark_only=args.synthetic_benchmark_only,
        )
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "factor_bundle_extraction_id": result["factor_bundle_extraction_id"],
                "modules": result.get("summary", {}).get("modules", result.get("modules")),
                "synthetic_benchmark_only": result.get("scientific_boundary", {}).get(
                    "synthetic_benchmark_only", result.get("synthetic_benchmark_only")
                ),
                "execution_authorized": result.get("execution_authorized", False),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
