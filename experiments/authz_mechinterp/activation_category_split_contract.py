from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.build_activation_category_splits import (
    SPLIT_ROLES,
    verify_split_artifact_role as verify_v1_role,
    verify_split_bundle as verify_v1_bundle,
)
from experiments.authz_mechinterp.build_activation_category_splits_v2 import (
    SCHEMA as V2_SCHEMA,
    verify_semantic_split_artifact_role,
    verify_semantic_split_bundle,
)
from experiments.authz_mechinterp.build_activation_category_splits_v3 import (
    SCHEMA as V3_SCHEMA,
    SPLIT_ROLES as V3_SPLIT_ROLES,
    verify_semantic_split_artifact_role as verify_v3_role,
    verify_semantic_split_bundle as verify_v3_bundle,
)


V1_SCHEMA = "sch.vpd-activation-category-split-manifest.v1"


def split_schema(manifest_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return str(manifest.get("schema", ""))


def verify_activation_category_split_bundle(
    output_dir: Path, *, source_path: Path | None = None
) -> dict[str, Any]:
    schema = split_schema(output_dir / "split_manifest.json")
    if schema == V1_SCHEMA:
        return verify_v1_bundle(output_dir, source_path=source_path)
    if schema == V2_SCHEMA:
        return verify_semantic_split_bundle(output_dir, source_path=source_path)
    if schema == V3_SCHEMA:
        return verify_v3_bundle(output_dir, source_path=source_path)
    raise ValueError("unsupported activation-category split manifest schema")


def verify_activation_category_split_role(
    dataset_path: Path, split_manifest_path: Path, split_role: str
) -> dict[str, Any]:
    schema = split_schema(split_manifest_path)
    if schema == V1_SCHEMA:
        if split_role not in SPLIT_ROLES:
            raise ValueError("unknown activation-category split role")
        result = verify_v1_role(dataset_path, split_manifest_path, split_role)
        return {
            **result,
            "activation_category_split_schema": V1_SCHEMA,
            "activation_category_split_semantic_disjointness": "not_provided_v1",
        }
    if schema == V2_SCHEMA:
        if split_role not in SPLIT_ROLES:
            raise ValueError("unknown activation-category split role")
        return verify_semantic_split_artifact_role(
            dataset_path, split_manifest_path, split_role
        )
    if schema == V3_SCHEMA:
        if split_role not in V3_SPLIT_ROLES:
            raise ValueError("unknown activation-category split role")
        return verify_v3_role(dataset_path, split_manifest_path, split_role)
    raise ValueError("unsupported activation-category split manifest schema")
