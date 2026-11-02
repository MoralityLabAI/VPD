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
    write_jsonl,
)
from experiments.authz_mechinterp.lora_component_basis_canonicalization_v1 import (
    canonical_spectral_factorization,
)
from experiments.authz_mechinterp.merge_vpd_effect_shards import (
    write_deterministic_npz,
)


SCHEMA = "sch.vpd-canonical-lora-component-inventory.v1"
FACTOR_INDEX_SCHEMA = "sch.vpd-lora-factor-bundle-index.v1"


def _array_sha256(array: np.ndarray) -> str:
    import hashlib

    values = np.asarray(array).astype(np.float64, copy=False)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _resolve_source(index_path: Path, relative_path: str) -> Path:
    path = (index_path.parent / relative_path).resolve()
    if not path.is_relative_to(index_path.parent.resolve()):
        raise ValueError("factor bundle source escapes its index directory")
    return path


def canonical_subspace_basis(
    orthonormal_basis: np.ndarray, *, tolerance: float = 1e-10
) -> np.ndarray:
    """Choose a deterministic basis using projected coordinate pivots."""

    source = np.asarray(orthonormal_basis, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] < 1:
        raise ValueError("canonical subspace basis is empty")
    selected: list[np.ndarray] = []
    for coordinate in range(source.shape[0]):
        candidate = source @ source[coordinate, :]
        for basis_vector in selected:
            candidate -= float(candidate @ basis_vector) * basis_vector
        norm = float(np.linalg.norm(candidate))
        if norm <= tolerance:
            continue
        candidate /= norm
        pivot = int(np.argmax(np.abs(candidate)))
        if candidate[pivot] < 0.0:
            candidate *= -1.0
        selected.append(candidate)
        if len(selected) == source.shape[1]:
            break
    if len(selected) != source.shape[1]:
        raise ValueError("could not construct complete canonical subspace basis")
    return np.column_stack(selected)


def canonical_coupled_block_factors(
    factorization: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Return basis-invariant low-rank factors for every spectral block."""

    left = np.asarray(factorization["left_singular_vectors"], dtype=np.float64)
    singular_values = np.asarray(factorization["singular_values"], dtype=np.float64)
    right = np.asarray(factorization["right_singular_vectors"], dtype=np.float64)
    canonical_left: list[np.ndarray] = []
    canonical_right: list[np.ndarray] = []
    for start, end in factorization["spectral_blocks"]:
        local_left = canonical_subspace_basis(left[:, start:end])
        local_right = (
            local_left.T
            @ (left[:, start:end] * singular_values[start:end][None, :])
            @ right[start:end, :]
        )
        canonical_left.append(local_left)
        canonical_right.append(local_right)
    return np.column_stack(canonical_left), np.vstack(canonical_right)


def build_inventory(
    *,
    factor_index_path: Path,
    output_dir: Path,
    relative_gap_tolerance: float = 1e-6,
    zero_tolerance: float = 1e-12,
) -> dict[str, Any]:
    index = json.loads(factor_index_path.read_text(encoding="utf-8"))
    if index.get("schema") != FACTOR_INDEX_SCHEMA:
        raise ValueError("canonical inventory requires factor-bundle index V1")
    modules = index.get("modules")
    if not isinstance(modules, list) or not modules:
        raise ValueError("factor-bundle index contains no modules")
    module_names = [str(row.get("module", "")) for row in modules]
    if (
        any(not value for value in module_names)
        or module_names != sorted(module_names)
        or len(module_names) != len(set(module_names))
    ):
        raise ValueError("factor-bundle modules must be sorted and unique")
    adapter_sha256 = str(index.get("adapter_sha256", ""))
    if len(adapter_sha256) != 64:
        raise ValueError("factor-bundle adapter identity is invalid")
    output_dir.mkdir(parents=True, exist_ok=True)
    module_dir = output_dir / "modules"
    module_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows: list[dict[str, Any]] = []
    module_artifacts: list[dict[str, Any]] = []
    unresolved_blocks = 0
    for module_index, source_row in enumerate(modules):
        source_path = _resolve_source(factor_index_path, str(source_row["path"]))
        if sha256_file(source_path) != str(source_row.get("sha256", "")):
            raise ValueError(f"factor source hash differs: {module_names[module_index]}")
        with np.load(source_path, allow_pickle=False) as archive:
            if set(archive.files) != {"a_factor", "b_factor"}:
                raise ValueError("factor source must contain exactly A and B arrays")
            a_factor = np.asarray(archive["a_factor"], dtype=np.float64)
            b_factor = np.asarray(archive["b_factor"], dtype=np.float64)
        scale = float(source_row["scale"])
        factorization = canonical_spectral_factorization(
            a_factor,
            b_factor,
            scale=scale,
            relative_gap_tolerance=relative_gap_tolerance,
            zero_tolerance=zero_tolerance,
        )
        singular_values = factorization["singular_values"]
        blocks = factorization["spectral_blocks"]
        canonical_left, canonical_right = canonical_coupled_block_factors(
            factorization
        )
        block_array = np.asarray(blocks, dtype=np.int64)
        module_filename = f"module-{module_index:04d}.npz"
        module_path = module_dir / module_filename
        write_deterministic_npz(
            module_path,
            {
                "canonical_left_factors": canonical_left.astype(np.float64),
                "singular_values": singular_values.astype(np.float64),
                "canonical_right_factors": canonical_right.astype(np.float64),
                "spectral_blocks": block_array,
                "raw_rank": np.asarray(factorization["raw_rank"], dtype=np.int64),
                "scale": np.asarray(scale, dtype=np.float64),
            },
        )
        module_delta_fro_sq = float(np.sum(singular_values**2))
        module_component_ids: list[str] = []
        for block_index, (start, end) in enumerate(blocks):
            component_id = f"{module_names[module_index]}:svdblock{block_index:02d}"
            block_values = singular_values[start:end]
            unresolved = end - start > 1
            unresolved_blocks += int(unresolved)
            module_component_ids.append(component_id)
            inventory_rows.append(
                {
                    "schema": "sch.vpd-canonical-lora-component.v1",
                    "component_id": component_id,
                    "module": module_names[module_index],
                    "canonical_block_index": block_index,
                    "spectral_start": start,
                    "spectral_end": end,
                    "spectral_rank": end - start,
                    "unresolved_degenerate_block": unresolved,
                    "singular_value_maximum": float(block_values[0]),
                    "singular_value_minimum": float(block_values[-1]),
                    "component_fro_norm": float(np.linalg.norm(block_values)),
                    "fraction_of_module_delta_fro_sq": (
                        float(np.sum(block_values**2)) / module_delta_fro_sq
                        if module_delta_fro_sq > 0.0
                        else 0.0
                    ),
                    "canonical_module_artifact": f"modules/{module_filename}",
                    "source_a_factor_sha256": _array_sha256(a_factor),
                    "source_b_factor_sha256": _array_sha256(b_factor),
                    "adapter_scale": scale,
                    "relative_spectral_gap_tolerance": relative_gap_tolerance,
                    "zero_tolerance": zero_tolerance,
                    "raw_rank_coordinate_identity_used": False,
                }
            )
        module_artifacts.append(
            {
                "module": module_names[module_index],
                "path": f"modules/{module_filename}",
                "sha256": sha256_file(module_path),
                "source_path": str(source_row["path"]),
                "source_sha256": str(source_row["sha256"]),
                "raw_rank": factorization["raw_rank"],
                "active_rank": len(singular_values),
                "canonical_component_ids": module_component_ids,
                "canonical_component_count": len(blocks),
                "unresolved_degenerate_blocks": factorization[
                    "unresolved_degenerate_blocks"
                ],
            }
        )
    inventory_path = output_dir / "canonical_components.jsonl"
    write_jsonl(inventory_path, inventory_rows)
    body = {
        "schema": SCHEMA,
        "status": "complete_model_free_factor_bundle_canonicalization",
        "source": {
            "factor_index_path": factor_index_path.name,
            "factor_index_sha256": sha256_file(factor_index_path),
            "adapter_sha256": adapter_sha256,
        },
        "policy": {
            "decomposition": "thin_svd_spectral_components_and_unresolved_blocks",
            "relative_spectral_gap_tolerance": relative_gap_tolerance,
            "zero_tolerance": zero_tolerance,
            "raw_lora_rank_coordinate_identity_used": False,
            "module_names_sorted_and_unique": True,
        },
        "artifacts": {
            "component_inventory": {
                "path": inventory_path.name,
                "sha256": sha256_file(inventory_path),
                "rows": len(inventory_rows),
            },
            "modules": module_artifacts,
        },
        "summary": {
            "modules": len(modules),
            "canonical_components": len(inventory_rows),
            "unresolved_degenerate_blocks": unresolved_blocks,
        },
        "scientific_boundary": {
            "model_loaded": False,
            "model_inference_performed": False,
            "weights_mutated": False,
            "real_adapter_extraction_performed": False,
            "effect_capture_performed": False,
            "execution_authorized": False,
        },
    }
    manifest = {**body, "canonical_inventory_id": content_sha256(body)}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def verify_inventory(
    manifest_path: Path, *, factor_index_path: Path
) -> dict[str, Any]:
    observed = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = body.pop("canonical_inventory_id", "")
    if identifier != content_sha256(body):
        raise ValueError("canonical inventory content address differs")
    output_dir = manifest_path.parent
    inventory = observed["artifacts"]["component_inventory"]
    if sha256_file(output_dir / inventory["path"]) != inventory["sha256"]:
        raise ValueError("canonical component inventory hash differs")
    for module in observed["artifacts"]["modules"]:
        if sha256_file(output_dir / module["path"]) != module["sha256"]:
            raise ValueError("canonical module artifact hash differs")
    with tempfile.TemporaryDirectory() as directory:
        expected = build_inventory(
            factor_index_path=factor_index_path,
            output_dir=Path(directory),
            relative_gap_tolerance=float(
                observed["policy"]["relative_spectral_gap_tolerance"]
            ),
            zero_tolerance=float(observed["policy"]["zero_tolerance"]),
        )
    if observed != expected:
        raise ValueError("canonical inventory is not reproducible")
    return {
        "schema": "sch.vpd-canonical-lora-component-inventory-verification.v1",
        "status": "pass",
        "canonical_inventory_id": identifier,
        "modules": observed["summary"]["modules"],
        "canonical_components": observed["summary"]["canonical_components"],
        "unresolved_degenerate_blocks": observed["summary"][
            "unresolved_degenerate_blocks"
        ],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit canonical LoRA spectral inventory V1.")
    parser.add_argument("--factor-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--relative-gap-tolerance", type=float, default=1e-6)
    parser.add_argument("--zero-tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_inventory(args.verify, factor_index_path=args.factor_index)
    else:
        if args.output_dir is None: parser.error("--output-dir is required unless verifying")
        result = build_inventory(
            factor_index_path=args.factor_index,
            output_dir=args.output_dir,
            relative_gap_tolerance=args.relative_gap_tolerance,
            zero_tolerance=args.zero_tolerance,
        )
    print(json.dumps({
        "schema": result["schema"], "status": result["status"],
        "canonical_inventory_id": result["canonical_inventory_id"],
        "modules": result.get("summary", {}).get("modules", result.get("modules")),
        "canonical_components": result.get("summary", {}).get(
            "canonical_components", result.get("canonical_components")
        ),
        "execution_authorized": result.get("execution_authorized", False),
    }, indent=2, sort_keys=True))


if __name__ == "__main__": main()
