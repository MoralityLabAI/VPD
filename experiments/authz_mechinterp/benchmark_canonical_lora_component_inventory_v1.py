from __future__ import annotations

import argparse
import json
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
from experiments.authz_mechinterp.lora_component_basis_canonicalization_v1 import (
    raw_rank_components,
)
from experiments.authz_mechinterp.merge_vpd_effect_shards import (
    write_deterministic_npz,
)


def _hadamard(order: int) -> np.ndarray:
    result = np.ones((1, 1), dtype=np.float64)
    while len(result) < order:
        result = np.block([[result, result], [result, -result]])
    return result / np.sqrt(order)


def _write_factor_bundle(
    root: Path, *, rotated: bool
) -> tuple[Path, list[tuple[np.ndarray, np.ndarray]]]:
    rotation = _hadamard(4)
    modules = [
        (
            "model.layers.0.self_attn.q_proj",
            np.diag([1.0, 1.2, 1.4, 1.6]),
            np.eye(4),
        ),
        (
            "model.layers.1.self_attn.q_proj",
            np.eye(4),
            np.eye(4),
        ),
    ]
    index_rows: list[dict[str, Any]] = []
    factors: list[tuple[np.ndarray, np.ndarray]] = []
    for index, (module, a_factor, b_factor) in enumerate(modules):
        if rotated:
            a_factor = rotation.T @ a_factor
            b_factor = b_factor @ rotation
        factors.append((a_factor, b_factor))
        path = root / f"factor-{index}.npz"
        write_deterministic_npz(
            path,
            {
                "a_factor": a_factor.astype(np.float64),
                "b_factor": b_factor.astype(np.float64),
            },
        )
        index_rows.append(
            {
                "module": module,
                "path": path.name,
                "sha256": sha256_file(path),
                "scale": 1.0,
            }
        )
    index_path = root / "factor-index.json"
    write_json(
        index_path,
        {
            "schema": "sch.vpd-lora-factor-bundle-index.v1",
            "adapter_sha256": "a" * 64,
            "modules": index_rows,
            "synthetic_benchmark_only": True,
        },
    )
    return index_path, factors


def _load_module(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _semantic_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        row.pop("source_a_factor_sha256")
        row.pop("source_b_factor_sha256")
    return rows


def _compare_semantic_rows(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> tuple[bool, float]:
    if len(left) != len(right):
        return False, float("inf")
    numeric_fields = {
        "singular_value_maximum",
        "singular_value_minimum",
        "component_fro_norm",
        "fraction_of_module_delta_fro_sq",
        "adapter_scale",
        "relative_spectral_gap_tolerance",
        "zero_tolerance",
    }
    maximum_difference = 0.0
    for left_row, right_row in zip(left, right):
        if set(left_row) != set(right_row):
            return False, float("inf")
        for key in left_row:
            if key in numeric_fields:
                maximum_difference = max(
                    maximum_difference,
                    abs(float(left_row[key]) - float(right_row[key])),
                )
            elif left_row[key] != right_row[key]:
                return False, maximum_difference
    return maximum_difference <= 1e-12, maximum_difference


def build_report() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        original_root = root / "original-source"
        rotated_root = root / "rotated-source"
        original_root.mkdir(); rotated_root.mkdir()
        original_index, original_factors = _write_factor_bundle(
            original_root, rotated=False
        )
        rotated_index, rotated_factors = _write_factor_bundle(
            rotated_root, rotated=True
        )
        original_output = root / "original-inventory"
        rotated_output = root / "rotated-inventory"
        original_manifest = build_inventory(
            factor_index_path=original_index, output_dir=original_output
        )
        rotated_manifest = build_inventory(
            factor_index_path=rotated_index, output_dir=rotated_output
        )
        original_verification = verify_inventory(
            original_output / "manifest.json", factor_index_path=original_index
        )
        rotated_verification = verify_inventory(
            rotated_output / "manifest.json", factor_index_path=rotated_index
        )
        module_differences: list[float] = []
        reconstruction_differences: list[float] = []
        for module_index in range(2):
            original_module = _load_module(
                original_output / "modules" / f"module-{module_index:04d}.npz"
            )
            rotated_module = _load_module(
                rotated_output / "modules" / f"module-{module_index:04d}.npz"
            )
            module_differences.extend(
                [
                    float(
                        np.linalg.norm(
                            original_module[name] - rotated_module[name]
                        )
                    )
                    for name in (
                        "canonical_left_factors",
                        "canonical_right_factors",
                        "singular_values",
                    )
                ]
            )
            original_reconstruction = (
                original_module["canonical_left_factors"]
                @ original_module["canonical_right_factors"]
            )
            rotated_reconstruction = (
                rotated_module["canonical_left_factors"]
                @ rotated_module["canonical_right_factors"]
            )
            reconstruction_differences.append(
                float(np.linalg.norm(original_reconstruction - rotated_reconstruction))
            )
        raw_difference = max(
            float(np.linalg.norm(left - right))
            for pair_index in range(2)
            for left, right in zip(
                raw_rank_components(
                    original_factors[pair_index][0], original_factors[pair_index][1]
                ),
                raw_rank_components(
                    rotated_factors[pair_index][0], rotated_factors[pair_index][1]
                ),
            )
        )
        semantic_rows_equal, maximum_semantic_numeric_difference = (
            _compare_semantic_rows(
                _semantic_rows(original_output / "canonical_components.jsonl"),
                _semantic_rows(rotated_output / "canonical_components.jsonl"),
            )
        )
        tamper_path = original_output / original_manifest["artifacts"]["modules"][0]["path"]
        tamper_path.write_bytes(tamper_path.read_bytes() + b"tamper")
        tamper_rejected = False
        try:
            verify_inventory(
                original_output / "manifest.json", factor_index_path=original_index
            )
        except ValueError:
            tamper_rejected = True
        maximum_factor_difference = max(module_differences)
        maximum_reconstruction_difference = max(reconstruction_differences)
        passed = (
            original_verification["status"] == "pass"
            and rotated_verification["status"] == "pass"
            and original_manifest["summary"] == {
                "modules": 2,
                "canonical_components": 5,
                "unresolved_degenerate_blocks": 1,
            }
            and rotated_manifest["summary"] == original_manifest["summary"]
            and semantic_rows_equal
            and raw_difference > 0.5
            and maximum_factor_difference < 1e-10
            and maximum_reconstruction_difference < 1e-10
            and tamper_rejected
        )
    body = {
        "schema": "sch.vpd-canonical-lora-component-inventory-benchmark.v1",
        "status": "pass" if passed else "fail",
        "source": {
            "runner_sha256": sha256_file(Path(__file__)),
            "emitter_sha256": sha256_file(
                Path(__file__).with_name(
                    "emit_canonical_lora_component_inventory_v1.py"
                )
            ),
        },
        "design": {
            "synthetic_modules": 2,
            "raw_rank_per_module": 4,
            "distinct_spectrum_module": True,
            "degenerate_spectrum_module": True,
            "equivalent_hadamard_factor_rotation": True,
        },
        "results": {
            "canonical_modules": 2,
            "canonical_components": 5,
            "unresolved_degenerate_blocks": 1,
            "semantic_inventory_rows_equal": semantic_rows_equal,
            "maximum_semantic_inventory_numeric_difference": (
                maximum_semantic_numeric_difference
            ),
            "maximum_raw_component_frobenius_difference": raw_difference,
            "maximum_canonical_factor_frobenius_difference": maximum_factor_difference,
            "maximum_canonical_reconstruction_frobenius_difference": (
                maximum_reconstruction_difference
            ),
            "original_inventory_verified": original_verification["status"] == "pass",
            "rotated_inventory_verified": rotated_verification["status"] == "pass",
            "tampered_module_artifact_rejected": tamper_rejected,
        },
        "scientific_boundary": {
            "synthetic_factor_bundles_only": True,
            "real_adapter_loaded": False,
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
        raise ValueError("canonical inventory benchmark address mismatch")
    if observed != build_report() or observed["status"] != "pass":
        raise ValueError("canonical inventory benchmark is not reproducible")
    return {
        "schema": "sch.vpd-canonical-lora-component-inventory-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "canonical_components": observed["results"]["canonical_components"],
        "unresolved_degenerate_blocks": observed["results"][
            "unresolved_degenerate_blocks"
        ],
        "tamper_rejected": observed["results"]["tampered_module_artifact_rejected"],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark canonical LoRA inventory V1.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_benchmark(args.verify)
    else:
        if args.output is None: parser.error("--output is required unless verifying")
        result = build_report(); write_json(args.output, result)
    print(json.dumps({
        "schema": result["schema"], "status": result["status"],
        "benchmark_id": result["benchmark_id"],
        "canonical_components": result.get("results", {}).get(
            "canonical_components", result.get("canonical_components")
        ),
        "tamper_rejected": result.get("results", {}).get(
            "tampered_module_artifact_rejected", result.get("tamper_rejected")
        ),
        "execution_authorized": result.get("execution_authorized", False),
    }, indent=2, sort_keys=True))


if __name__ == "__main__": main()
