from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
    _canonical_absolute,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


MANIFEST_SCHEMA = "sch.vpd-cluster-to-sae-model-files-manifest.v1"
PREFLIGHT_SCHEMA = "sch.vpd-cluster-to-sae-empirical-preflight.v2"
ALLOWED_ADAPTER_WEIGHT_FILES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
)
ADAPTER_CONFIG_FILE = "adapter_config.json"


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def _require_plain_directory(path: Path, *, field: str) -> Path:
    resolved = path.resolve()
    if not path.exists() or not path.is_dir():
        raise ValueError(f"bound content {field} is not a directory")
    if _is_reparse_point(path):
        raise ValueError(f"bound content {field} is a reparse point")
    return resolved


def _relative_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("model-files manifest path is not canonical POSIX-relative")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise ValueError("model-files manifest path escapes its root")
    if any(part in {"", "."} for part in path.parts):
        raise ValueError("model-files manifest path contains an empty segment")
    return path


def _scan_files(root: Path) -> list[Path]:
    resolved_root = _require_plain_directory(root, field="model root")
    rows: list[Path] = []
    for path in resolved_root.rglob("*"):
        if _is_reparse_point(path):
            raise ValueError("model-files tree contains a reparse point")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("model-files tree contains a non-regular entry")
        rows.append(path)
    rows.sort(key=lambda path: path.relative_to(resolved_root).as_posix())
    if not rows:
        raise ValueError("model-files tree is empty")
    return rows


def build_model_files_manifest(
    *, model_root: Path, model_id: str, model_revision: str
) -> dict[str, Any]:
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model-files manifest model ID is absent")
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise ValueError("model-files manifest revision is absent")
    root = _require_plain_directory(model_root, field="model root")
    files = _scan_files(root)
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    body = {
        "schema": MANIFEST_SCHEMA,
        "model_id": model_id,
        "model_revision": model_revision,
        "model_root": str(root),
        "files": rows,
        "summary": {
            "files": len(rows),
            "total_bytes": sum(row["size_bytes"] for row in rows),
        },
        "cycle_boundary": {
            "preflight_id_in_manifest": False,
            "preflight_digest_in_manifest": False,
            "manifest_is_built_before_preflight_binding": True,
        },
        "scientific_boundary": {
            "streaming_digest_only": True,
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "weights_mutated": False,
            "execution_authorized": False,
        },
    }
    return {**body, "manifest_id": content_sha256(body)}


def verify_model_files_manifest(
    manifest_path: Path, *, expected_file_sha256: str | None = None
) -> dict[str, Any]:
    observed = json.loads(manifest_path.read_text(encoding="utf-8"))
    if observed.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("model-files manifest schema differs")
    identifier = str(observed.get("manifest_id", ""))
    body = {key: value for key, value in observed.items() if key != "manifest_id"}
    if identifier != content_sha256(body):
        raise ValueError("model-files manifest content address differs")
    if expected_file_sha256 is not None and sha256_file(manifest_path) != expected_file_sha256:
        raise ValueError("model-files manifest file digest differs from preflight")
    if any(key.startswith("preflight") for key in observed):
        raise ValueError("model-files manifest contains a preflight back-reference")
    cycle = observed.get("cycle_boundary", {})
    if cycle != {
        "preflight_id_in_manifest": False,
        "preflight_digest_in_manifest": False,
        "manifest_is_built_before_preflight_binding": True,
    }:
        raise ValueError("model-files manifest cycle boundary differs")

    root = _canonical_absolute(observed.get("model_root"), field="model_root")
    root = _require_plain_directory(root, field="model root")
    if manifest_path.resolve().is_relative_to(root):
        raise ValueError("model-files manifest is inside the tree it addresses")
    expected_paths = [path.relative_to(root).as_posix() for path in _scan_files(root)]
    rows = observed.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("model-files manifest has no rows")
    row_paths = [str(row.get("path", "")) for row in rows]
    if row_paths != sorted(row_paths) or len(row_paths) != len(set(row_paths)):
        raise ValueError("model-files manifest paths are duplicated or unsorted")
    if row_paths != expected_paths:
        raise ValueError("model-files manifest does not exactly cover the model tree")
    total_bytes = 0
    for row in rows:
        relative = _relative_path(row.get("path"))
        path = root.joinpath(*relative.parts)
        if _is_reparse_point(path) or not path.is_file():
            raise ValueError("model-files manifest row is not a regular file")
        size = path.stat().st_size
        if type(row.get("size_bytes")) is not int or row["size_bytes"] != size:
            raise ValueError("model-files manifest size differs")
        if row.get("sha256") != sha256_file(path):
            raise ValueError("model-files manifest content digest differs")
        total_bytes += size
    if observed.get("summary") != {"files": len(rows), "total_bytes": total_bytes}:
        raise ValueError("model-files manifest summary differs")
    return {
        "schema": "sch.vpd-cluster-to-sae-model-files-manifest-verification.v1",
        "status": "pass_streaming_digest_only",
        "manifest_id": identifier,
        "manifest_file_sha256": sha256_file(manifest_path),
        "model_id": observed["model_id"],
        "model_revision": observed["model_revision"],
        "model_root": str(root),
        "files": len(rows),
        "total_bytes": total_bytes,
        "model_framework_imported": False,
        "model_loaded": False,
        "execution_authorized": False,
    }


def _verify_preflight_address(preflight: dict[str, Any]) -> str:
    if preflight.get("schema") != PREFLIGHT_SCHEMA:
        raise ValueError("bound content preflight schema differs")
    identifier = str(preflight.get("preflight_id", ""))
    body = {key: value for key, value in preflight.items() if key != "preflight_id"}
    if identifier != content_sha256(body):
        raise ValueError("bound content preflight address differs")
    if preflight.get("missing_required_bindings"):
        raise ValueError("bound content requires an identity-complete preflight")
    return identifier


def _adapter_files(adapter_path: Path) -> tuple[Path, Path]:
    root = _require_plain_directory(adapter_path, field="adapter path")
    config = root / ADAPTER_CONFIG_FILE
    if not config.is_file() or _is_reparse_point(config):
        raise ValueError("bound content adapter configuration is absent or indirect")
    candidates = [
        root / filename
        for filename in ALLOWED_ADAPTER_WEIGHT_FILES
        if (root / filename).is_file()
    ]
    if len(candidates) != 1 or any(_is_reparse_point(path) for path in candidates):
        raise ValueError("bound content adapter weight file is absent, ambiguous, or indirect")
    return candidates[0], config


def verify_bound_content(
    *, preflight_path: Path, model_manifest_path: Path
) -> dict[str, Any]:
    before = MODEL_FRAMEWORKS & set(sys.modules)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight_id = _verify_preflight_address(preflight)
    bindings = preflight["immutable_bindings"]
    model = verify_model_files_manifest(
        model_manifest_path,
        expected_file_sha256=str(bindings["model_files_manifest_sha256"]),
    )
    if model["model_id"] != bindings["model_id"]:
        raise ValueError("bound content model ID differs")
    if model["model_revision"] != bindings["model_revision"]:
        raise ValueError("bound content model revision differs")
    adapter_root = _canonical_absolute(bindings["adapter_path"], field="adapter_path")
    weights, config = _adapter_files(adapter_root)
    weights_sha256 = sha256_file(weights)
    config_sha256 = sha256_file(config)
    if weights_sha256 != bindings["adapter_weights_sha256"]:
        raise ValueError("bound content adapter weight digest differs")
    if config_sha256 != bindings["adapter_config_sha256"]:
        raise ValueError("bound content adapter configuration digest differs")
    after = MODEL_FRAMEWORKS & set(sys.modules)
    if before != after:
        raise ValueError("bound content verification imported a model framework")
    return {
        "schema": "sch.vpd-cluster-to-sae-bound-content-verification.v1",
        "status": "pass_exact_bytes_unopened_by_model_framework",
        "preflight_id": preflight_id,
        "model_manifest_id": model["manifest_id"],
        "model_manifest_file_sha256": model["manifest_file_sha256"],
        "model_files": model["files"],
        "model_total_bytes": model["total_bytes"],
        "adapter_root": str(adapter_root),
        "adapter_weight_file": weights.name,
        "adapter_weights_sha256": weights_sha256,
        "adapter_config_sha256": config_sha256,
        "streaming_digest_only": True,
        "model_framework_imported": False,
        "model_loaded": False,
        "adapter_loaded": False,
        "model_inference_performed": False,
        "weights_mutated": False,
        "authorization_receipt_present": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a model-files manifest or verify content against preflight V2."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-model-manifest")
    build.add_argument("--model-root", type=Path, required=True)
    build.add_argument("--model-id", required=True)
    build.add_argument("--model-revision", required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify-bound-content")
    verify.add_argument("--preflight", type=Path, required=True)
    verify.add_argument("--model-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build-model-manifest":
        result = build_model_files_manifest(
            model_root=args.model_root,
            model_id=args.model_id,
            model_revision=args.model_revision,
        )
        if args.output.resolve().is_relative_to(args.model_root.resolve()):
            raise ValueError("model-files manifest output must be outside model root")
        write_json(args.output, result)
    else:
        result = verify_bound_content(
            preflight_path=args.preflight,
            model_manifest_path=args.model_manifest,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
