from __future__ import annotations

import argparse
import json
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
from experiments.authz_mechinterp.emit_canonical_lora_component_inventory_v1 import (
    build_inventory,
    verify_inventory,
)
from experiments.authz_mechinterp.extract_authenticated_lora_factor_bundle_v1 import (
    extract_factor_bundle,
    verify_extraction,
)


def _write_safetensors(
    path: Path,
    tensors: dict[str, np.ndarray],
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    dtype_names = {
        np.dtype("float64"): "F64",
        np.dtype("float32"): "F32",
        np.dtype("float16"): "F16",
    }
    header: dict[str, Any] = {}
    data = bytearray()
    for name in sorted(tensors):
        array = np.ascontiguousarray(tensors[name])
        if array.dtype not in dtype_names:
            raise ValueError(f"unsupported synthetic dtype: {array.dtype}")
        start = len(data)
        data.extend(array.tobytes(order="C"))
        header[name] = {
            "dtype": dtype_names[array.dtype],
            "shape": list(array.shape),
            "data_offsets": [start, len(data)],
        }
    if metadata is not None:
        header["__metadata__"] = metadata
    header_bytes = json.dumps(
        header, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    padding = (-len(header_bytes)) % 8
    header_bytes += b" " * padding
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + data)


def _fixture_tensors() -> dict[str, np.ndarray]:
    return {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": np.asarray(
            [[1.0, 0.0, 0.5], [0.0, 1.0, -0.25]], dtype=np.float32
        ),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-0.25, 0.75]],
            dtype=np.float32,
        ),
        "base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight": np.asarray(
            [
                [1.0, 0.0, 0.0, 0.25, 0.0],
                [0.0, 1.0, 0.0, 0.0, -0.5],
                [0.0, 0.0, 1.0, 0.5, 0.25],
            ],
            dtype=np.float32,
        ),
        "base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0],
                [0.0, -0.5, 0.5],
                [0.25, 0.0, 0.75],
            ],
            dtype=np.float32,
        ),
    }


def _fixture_config() -> dict[str, Any]:
    return {
        "peft_type": "LORA",
        "r": 2,
        "lora_alpha": 4,
        "rank_pattern": {"model.layers.1.self_attn.q_proj": 3},
        "alpha_pattern": {"model.layers.1.self_attn.q_proj": 9},
        "use_rslora": False,
        "fan_in_fan_out": False,
        "use_dora": False,
        "bias": "none",
        "modules_to_save": None,
    }


def _attempt(function: Any) -> bool:
    try:
        function()
    except (ValueError, FileExistsError):
        return True
    return False


def build_report() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        weights = root / "adapter_model.safetensors"
        config = root / "adapter_config.json"
        tensors = _fixture_tensors()
        _write_safetensors(weights, tensors, metadata={"format": "pt"})
        write_json(config, _fixture_config())
        adapter_sha256 = sha256_file(weights)
        config_sha256 = sha256_file(config)
        output = root / "factor-bundle"
        manifest = extract_factor_bundle(
            adapter_weights_path=weights,
            adapter_config_path=config,
            expected_adapter_sha256=adapter_sha256,
            expected_config_sha256=config_sha256,
            output_dir=output,
            synthetic_benchmark_only=True,
        )
        verification = verify_extraction(
            output / "manifest.json",
            adapter_weights_path=weights,
            adapter_config_path=config,
        )
        index = json.loads((output / "factor-index.json").read_text(encoding="utf-8"))
        expected_modules = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.1.self_attn.q_proj",
        ]
        extracted_exactly = True
        for row in index["modules"]:
            source_module = row["source_module"]
            with np.load(output / row["path"], allow_pickle=False) as archive:
                extracted_exactly &= np.array_equal(
                    archive["a_factor"],
                    tensors[f"{source_module}.lora_A.weight"].astype(np.float64),
                )
                extracted_exactly &= np.array_equal(
                    archive["b_factor"],
                    tensors[f"{source_module}.lora_B.weight"].astype(np.float64),
                )
        inventory_output = root / "canonical-inventory"
        inventory = build_inventory(
            factor_index_path=output / "factor-index.json",
            output_dir=inventory_output,
        )
        inventory_verification = verify_inventory(
            inventory_output / "manifest.json",
            factor_index_path=output / "factor-index.json",
        )

        tampered_weights = root / "tampered.safetensors"
        tampered_weights.write_bytes(weights.read_bytes() + b"tamper")
        weights_tamper_rejected = _attempt(
            lambda: extract_factor_bundle(
                adapter_weights_path=tampered_weights,
                adapter_config_path=config,
                expected_adapter_sha256=adapter_sha256,
                expected_config_sha256=config_sha256,
                output_dir=root / "tampered-weights-output",
                synthetic_benchmark_only=True,
            )
        )
        tampered_config = root / "tampered-config.json"
        changed_config = _fixture_config()
        changed_config["lora_alpha"] = 5
        write_json(tampered_config, changed_config)
        config_tamper_rejected = _attempt(
            lambda: extract_factor_bundle(
                adapter_weights_path=weights,
                adapter_config_path=tampered_config,
                expected_adapter_sha256=adapter_sha256,
                expected_config_sha256=config_sha256,
                output_dir=root / "tampered-config-output",
                synthetic_benchmark_only=True,
            )
        )
        orphan_weights = root / "orphan.safetensors"
        orphan_tensors = dict(tensors)
        orphan_tensors.pop(
            "base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight"
        )
        _write_safetensors(orphan_weights, orphan_tensors)
        orphan_rejected = _attempt(
            lambda: extract_factor_bundle(
                adapter_weights_path=orphan_weights,
                adapter_config_path=config,
                expected_adapter_sha256=sha256_file(orphan_weights),
                expected_config_sha256=config_sha256,
                output_dir=root / "orphan-output",
                synthetic_benchmark_only=True,
            )
        )
        unexpected_weights = root / "unexpected.safetensors"
        unexpected_tensors = dict(tensors)
        unexpected_tensors["base_model.model.lm_head.weight"] = np.ones(
            (2, 2), dtype=np.float32
        )
        _write_safetensors(unexpected_weights, unexpected_tensors)
        unexpected_tensor_rejected = _attempt(
            lambda: extract_factor_bundle(
                adapter_weights_path=unexpected_weights,
                adapter_config_path=config,
                expected_adapter_sha256=sha256_file(unexpected_weights),
                expected_config_sha256=config_sha256,
                output_dir=root / "unexpected-output",
                synthetic_benchmark_only=True,
            )
        )
        mismatch_config = root / "rank-mismatch.json"
        rank_mismatch = _fixture_config()
        rank_mismatch["rank_pattern"] = {
            "model.layers.1.self_attn.q_proj": 4
        }
        write_json(mismatch_config, rank_mismatch)
        rank_mismatch_output = root / "rank-mismatch-output"
        rank_mismatch_rejected = _attempt(
            lambda: extract_factor_bundle(
                adapter_weights_path=weights,
                adapter_config_path=mismatch_config,
                expected_adapter_sha256=adapter_sha256,
                expected_config_sha256=sha256_file(mismatch_config),
                output_dir=rank_mismatch_output,
                synthetic_benchmark_only=True,
            )
        )
        rank_mismatch_left_no_partial_output = not rank_mismatch_output.exists()
        nonempty_output = root / "nonempty-output"
        nonempty_output.mkdir()
        (nonempty_output / "stale.txt").write_text("stale", encoding="utf-8")
        nonempty_output_rejected = _attempt(
            lambda: extract_factor_bundle(
                adapter_weights_path=weights,
                adapter_config_path=config,
                expected_adapter_sha256=adapter_sha256,
                expected_config_sha256=config_sha256,
                output_dir=nonempty_output,
                synthetic_benchmark_only=True,
            )
        )
        factor_artifact = output / index["modules"][0]["path"]
        factor_artifact.write_bytes(factor_artifact.read_bytes() + b"tamper")
        factor_artifact_tamper_rejected = _attempt(
            lambda: verify_extraction(
                output / "manifest.json",
                adapter_weights_path=weights,
                adapter_config_path=config,
            )
        )
        scales = [float(row["scale"]) for row in index["modules"]]
        passed = (
            verification["status"] == "pass"
            and verification["synthetic_benchmark_only"] is True
            and [row["module"] for row in index["modules"]] == expected_modules
            and scales == [2.0, 3.0]
            and extracted_exactly
            and inventory_verification["status"] == "pass"
            and inventory["summary"]["modules"] == 2
            and weights_tamper_rejected
            and config_tamper_rejected
            and orphan_rejected
            and unexpected_tensor_rejected
            and rank_mismatch_rejected
            and rank_mismatch_left_no_partial_output
            and nonempty_output_rejected
            and factor_artifact_tamper_rejected
        )
    body = {
        "schema": "sch.vpd-authenticated-lora-factor-bundle-extraction-benchmark.v1",
        "status": "pass" if passed else "fail",
        "source": {
            "runner_sha256": sha256_file(Path(__file__)),
            "extractor_sha256": sha256_file(
                Path(__file__).with_name(
                    "extract_authenticated_lora_factor_bundle_v1.py"
                )
            ),
        },
        "design": {
            "synthetic_plain_lora_only": True,
            "synthetic_modules": 2,
            "per_module_rank_and_alpha_patterns": True,
            "canonical_inventory_chained": True,
        },
        "results": {
            "extraction_verified": verification["status"] == "pass",
            "modules": 2,
            "module_names_sorted_and_canonical": [
                row["module"] for row in index["modules"]
            ]
            == expected_modules,
            "factor_arrays_extracted_exactly": extracted_exactly,
            "module_scales": scales,
            "canonical_inventory_verified": inventory_verification["status"]
            == "pass",
            "weights_hash_tamper_rejected": weights_tamper_rejected,
            "config_hash_tamper_rejected": config_tamper_rejected,
            "orphan_factor_rejected": orphan_rejected,
            "unexpected_tensor_rejected": unexpected_tensor_rejected,
            "rank_mismatch_rejected": rank_mismatch_rejected,
            "rank_mismatch_left_no_partial_output": (
                rank_mismatch_left_no_partial_output
            ),
            "nonempty_output_rejected": nonempty_output_rejected,
            "factor_artifact_tamper_rejected": factor_artifact_tamper_rejected,
        },
        "scientific_boundary": {
            "synthetic_adapter_fixture_only": True,
            "real_adapter_accessed": False,
            "torch_or_peft_loaded": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "weights_mutated": False,
            "execution_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = body.pop("benchmark_id", "")
    if identifier != content_sha256(body):
        raise ValueError("factor-bundle extraction benchmark address mismatch")
    if observed != build_report() or observed["status"] != "pass":
        raise ValueError("factor-bundle extraction benchmark is not reproducible")
    return {
        "schema": "sch.vpd-authenticated-lora-factor-bundle-extraction-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "modules": observed["results"]["modules"],
        "negative_controls": 7,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark authenticated LoRA factor-bundle extraction V1."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_benchmark(args.verify)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_report()
        write_json(args.output, result)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "benchmark_id": result["benchmark_id"],
                "modules": result.get("results", {}).get(
                    "modules", result.get("modules")
                ),
                "execution_authorized": result.get("execution_authorized", False),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
