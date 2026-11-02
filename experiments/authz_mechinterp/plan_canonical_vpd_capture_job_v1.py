from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.activation_categories import (
    CATEGORY_PROTOCOLS,
    activation_category,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


PLAN_SCHEMA = "sch.vpd-canonical-category-capture-job-plan.v1"
PLAN_STATUS = "planned_model_execution_not_authorized"
MODEL_IDENTITY_SCHEMA = "sch.vpd-frozen-model-identity.v1"
AUTHORIZATION_SCHEMA = "sch.vpd-explicit-canonical-capture-authorization.v1"
EXTRACTION_SCHEMA = "sch.vpd-authenticated-lora-factor-bundle-extraction.v1"
INVENTORY_SCHEMA = "sch.vpd-canonical-lora-component-inventory.v1"
ALLOWED_SPLIT_SCHEMAS = {
    "sch.vpd-activation-category-split-manifest.v2",
    "sch.vpd-activation-category-split-manifest.v3",
}
ALLOWED_SPLIT_ROLES = {"discovery", "calibration", "cluster_validation"}
ALLOWED_DTYPES = {"float16", "bfloat16", "float32"}
COMPONENT_RE = re.compile(r"^(?P<module>.+):svdblock(?P<block>\d{2,})$")
ACTIVATION_KEY_RE = re.compile(r"^layer_(?P<layer>\d+)__proposal_start$")
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _digest(value: Any, *, name: str) -> str:
    result = str(value).lower()
    if DIGEST_RE.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _load_addressed(
    path: Path, *, schema: str, identifier_field: str
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema:
        raise ValueError(f"unsupported schema: {path}")
    body = dict(value)
    identifier = str(body.pop(identifier_field, ""))
    if identifier != content_sha256(body):
        raise ValueError(f"content address mismatch: {path}")
    return value


def _project_relative(project_root: Path, path: Path, *, name: str) -> str:
    root = project_root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"{name} must be inside the project root") from exc


def _resolve_relative(base: Path, value: Any, *, name: str) -> Path:
    text = str(value)
    relative = Path(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must be a safe relative path")
    resolved = (base / relative).resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError(f"{name} escapes its artifact root") from exc
    return resolved


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank dataset row: {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"dataset row is not an object: {line_number}")
            rows.append(value)
    if not rows:
        raise ValueError("canonical capture dataset is empty")
    return rows


def _validate_model_identity(path: Path) -> dict[str, Any]:
    value = _load_addressed(
        path, schema=MODEL_IDENTITY_SCHEMA, identifier_field="model_identity_id"
    )
    if value.get("status") != "frozen_reproducible_model_identity":
        raise ValueError("model identity is not frozen")
    for field in ("model_id", "revision"):
        text = str(value.get(field, ""))
        if not text or text.lower() in {"main", "master", "latest"}:
            raise ValueError(f"model identity {field} is mutable or absent")
    _digest(value.get("model_files_manifest_sha256"), name="model files manifest")
    loader = value.get("loader_contract", {})
    if (
        loader.get("trust_remote_code") is not False
        or loader.get("immutable_revision_required") is not True
        or type(loader.get("local_files_only")) is not bool
        or loader.get("generated_tokens") != 0
    ):
        raise ValueError("model identity loader contract is unsafe")
    boundary = value.get("scientific_boundary", {})
    if (
        boundary.get("model_loaded") is not False
        or boundary.get("inference_performed") is not False
        or boundary.get("execution_authorized") is not False
    ):
        raise ValueError("model identity improperly claims execution")
    return value


def _validate_adapter_lineage(
    *,
    adapter_weights_path: Path,
    adapter_config_path: Path,
    extraction_manifest_path: Path,
    inventory_manifest_path: Path,
    synthetic_benchmark_only: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    extraction = _load_addressed(
        extraction_manifest_path,
        schema=EXTRACTION_SCHEMA,
        identifier_field="factor_bundle_extraction_id",
    )
    source = extraction.get("source", {})
    adapter_sha256 = _digest(source.get("adapter_sha256"), name="adapter")
    config_sha256 = _digest(
        source.get("adapter_config_sha256"), name="adapter config"
    )
    if sha256_file(adapter_weights_path) != adapter_sha256:
        raise ValueError("adapter weights differ from extraction manifest")
    if sha256_file(adapter_config_path) != config_sha256:
        raise ValueError("adapter config differs from extraction manifest")
    extraction_synthetic = extraction.get("scientific_boundary", {}).get(
        "synthetic_benchmark_only"
    )
    if extraction_synthetic is not synthetic_benchmark_only:
        raise ValueError("adapter extraction synthetic provenance differs")
    factor_index_record = extraction.get("artifacts", {}).get("factor_index", {})
    factor_index_path = _resolve_relative(
        extraction_manifest_path.parent,
        factor_index_record.get("path"),
        name="factor index",
    )
    factor_index_sha256 = _digest(
        factor_index_record.get("sha256"), name="factor index"
    )
    if sha256_file(factor_index_path) != factor_index_sha256:
        raise ValueError("factor index hash differs")

    inventory = _load_addressed(
        inventory_manifest_path,
        schema=INVENTORY_SCHEMA,
        identifier_field="canonical_inventory_id",
    )
    inventory_source = inventory.get("source", {})
    if (
        _digest(inventory_source.get("factor_index_sha256"), name="inventory factor index")
        != factor_index_sha256
        or _digest(inventory_source.get("adapter_sha256"), name="inventory adapter")
        != adapter_sha256
    ):
        raise ValueError("canonical inventory source differs from extraction")
    inventory_record = inventory.get("artifacts", {}).get("component_inventory", {})
    inventory_path = _resolve_relative(
        inventory_manifest_path.parent,
        inventory_record.get("path"),
        name="component inventory",
    )
    if sha256_file(inventory_path) != _digest(
        inventory_record.get("sha256"), name="component inventory"
    ):
        raise ValueError("canonical component inventory hash differs")
    component_rows = _read_jsonl(inventory_path)
    if len(component_rows) != int(inventory_record.get("rows", -1)):
        raise ValueError("canonical component inventory row count differs")
    component_ids = [str(row.get("component_id", "")) for row in component_rows]
    if (
        component_ids != sorted(component_ids)
        or len(component_ids) != len(set(component_ids))
        or any(COMPONENT_RE.fullmatch(value) is None for value in component_ids)
    ):
        raise ValueError("canonical component identities are invalid")
    module_artifacts = inventory.get("artifacts", {}).get("modules", [])
    for artifact in module_artifacts:
        artifact_path = _resolve_relative(
            inventory_manifest_path.parent,
            artifact.get("path"),
            name="canonical module artifact",
        )
        if sha256_file(artifact_path) != _digest(
            artifact.get("sha256"), name="canonical module artifact"
        ):
            raise ValueError("canonical module artifact hash differs")
    return extraction, inventory, component_rows


def _validate_split(
    *, split_manifest_path: Path, split_role: str, dataset_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    split = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split.get("schema") not in ALLOWED_SPLIT_SCHEMAS:
        raise ValueError("canonical capture requires semantic split V2 or V3")
    split_body = dict(split)
    split_id = str(split_body.pop("split_id", ""))
    if split_id != content_sha256(split_body):
        raise ValueError("semantic split manifest address mismatch")
    if split_role not in ALLOWED_SPLIT_ROLES:
        raise ValueError("canonical capture split role is forbidden")
    role = split.get("splits", {}).get(split_role)
    if not isinstance(role, dict):
        raise ValueError("semantic split role is absent")
    artifact = role.get("artifact", {})
    expected_dataset = _resolve_relative(
        split_manifest_path.parent, artifact.get("path"), name="split dataset"
    )
    if dataset_path.resolve() != expected_dataset:
        raise ValueError("dataset path differs from semantic split role")
    dataset_sha256 = _digest(artifact.get("sha256"), name="split dataset")
    if sha256_file(dataset_path) != dataset_sha256:
        raise ValueError("semantic split dataset hash differs")
    rows = _read_jsonl(dataset_path)
    if len(rows) != int(role.get("rows", -1)):
        raise ValueError("semantic split dataset row count differs")
    episode_ids: list[str] = []
    pair_ids: list[str] = []
    template_groups: list[str] = []
    category_counts: Counter[str] = Counter()
    pairs_by_category: dict[str, set[str]] = {
        category: set() for category in CATEGORY_PROTOCOLS
    }
    for row in rows:
        episode_id = str(row.get("episode_id", ""))
        pair_id = str(row.get("pair_id", ""))
        template_group = str(
            row.get("template_group")
            or row.get("activation_category_pack", {}).get("semantic_template_group", "")
        )
        if not episode_id or not pair_id or not template_group:
            raise ValueError("semantic split row identity is incomplete")
        category = activation_category(row)
        if (
            row.get("actual_execution") is not False
            or row.get("execution_path_exposed") is not False
            or row.get("labels", {}).get("candidate_effect") != "none"
        ):
            raise ValueError("canonical capture dataset crosses the no-action boundary")
        episode_ids.append(episode_id)
        pair_ids.append(pair_id)
        template_groups.append(template_group)
        category_counts[category] += 1
        pairs_by_category[category].add(pair_id)
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("semantic split episode IDs are not unique")
    if set(category_counts) != set(CATEGORY_PROTOCOLS):
        raise ValueError("canonical capture dataset does not cover all activation categories")
    expected_counts = {str(key): int(value) for key, value in role.get("activation_category_counts", {}).items()}
    if dict(sorted(category_counts.items())) != dict(sorted(expected_counts.items())):
        raise ValueError("semantic split activation-category counts differ")
    shared_pairs = set(pair_ids)
    if any(pairs_by_category[category] != shared_pairs for category in CATEGORY_PROTOCOLS):
        raise ValueError("activation categories do not share the exact prompt-pair set")
    expected_hashes = {
        "episode_id_set_sha256": content_sha256(sorted(episode_ids)),
        "pair_id_set_sha256": content_sha256(sorted(set(pair_ids))),
        "template_group_set_sha256": content_sha256(sorted(set(template_groups))),
    }
    for field, expected in expected_hashes.items():
        if _digest(role.get(field), name=f"split {field}") != expected:
            raise ValueError(f"semantic split {field} differs from rows")
    normalized = {
        "split_id": split_id,
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "split_schema": str(split["schema"]),
        "split_role": split_role,
        "dataset_sha256": dataset_sha256,
        "rows": len(rows),
        "pairs": len(shared_pairs),
        "categories": sorted(category_counts),
        "activation_category_counts": dict(sorted(category_counts.items())),
        **expected_hashes,
    }
    return split, rows, normalized


def build_capture_job_plan(
    *,
    project_root: Path,
    model_identity_path: Path,
    adapter_weights_path: Path,
    adapter_config_path: Path,
    extraction_manifest_path: Path,
    inventory_manifest_path: Path,
    split_manifest_path: Path,
    split_role: str,
    dataset_path: Path,
    component_ids: list[str] | None,
    adapter_name: str,
    activation_key: str,
    epsilon: float,
    batch_size: int,
    device: str,
    dtype: str,
    output_dir: str,
    probe_seed: int = 1707,
    probe_count: int = 8,
    maximum_relative_probe_error: float = 2e-3,
    countsketch_dimension: int = 256,
    countsketch_seed: int = 270829,
    uncertainty_blocks: int = 8,
    synthetic_benchmark_only: bool = False,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    if adapter_weights_path.resolve().parent != adapter_config_path.resolve().parent:
        raise ValueError("adapter weights and config must share one PEFT directory")
    if adapter_weights_path.name != "adapter_model.safetensors" or adapter_config_path.name != "adapter_config.json":
        raise ValueError("canonical capture requires standard PEFT adapter filenames")
    if not adapter_name or any(character.isspace() for character in adapter_name):
        raise ValueError("canonical capture adapter name is invalid")
    model_identity = _validate_model_identity(model_identity_path)
    extraction, inventory, inventory_rows = _validate_adapter_lineage(
        adapter_weights_path=adapter_weights_path,
        adapter_config_path=adapter_config_path,
        extraction_manifest_path=extraction_manifest_path,
        inventory_manifest_path=inventory_manifest_path,
        synthetic_benchmark_only=synthetic_benchmark_only,
    )
    _, _, split_summary = _validate_split(
        split_manifest_path=split_manifest_path,
        split_role=split_role,
        dataset_path=dataset_path,
    )
    available = {str(row["component_id"]): row for row in inventory_rows}
    selected_ids = sorted(available) if component_ids is None else list(component_ids)
    if (
        not selected_ids
        or selected_ids != sorted(selected_ids)
        or len(selected_ids) != len(set(selected_ids))
        or any(component_id not in available for component_id in selected_ids)
    ):
        raise ValueError("selected canonical component list is invalid")
    if ACTIVATION_KEY_RE.fullmatch(activation_key) is None:
        raise ValueError("canonical capture activation key must be layer_N__proposal_start")
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ValueError("canonical capture epsilon must be in (0, 1)")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("canonical capture batch size is invalid")
    if dtype not in ALLOWED_DTYPES or not device:
        raise ValueError("canonical capture device or dtype is invalid")
    if type(probe_seed) is not int or probe_seed < 0 or probe_count < 1:
        raise ValueError("canonical capture probe contract is invalid")
    if not math.isfinite(maximum_relative_probe_error) or maximum_relative_probe_error <= 0:
        raise ValueError("canonical capture maximum probe error is invalid")
    if countsketch_dimension < 256 or countsketch_seed < 0 or uncertainty_blocks < 2:
        raise ValueError("canonical capture semantic-sketch contract is invalid")
    _resolve_relative(project_root, output_dir, name="capture output directory")
    component_records: list[dict[str, Any]] = []
    inventory_root = inventory_manifest_path.parent
    module_records = {
        str(row["module"]): row for row in inventory["artifacts"]["modules"]
    }
    for component_id in selected_ids:
        row = available[component_id]
        match = COMPONENT_RE.fullmatch(component_id)
        assert match is not None
        module = str(row["module"])
        module_record = module_records[module]
        artifact_path = _resolve_relative(
            inventory_root,
            row["canonical_module_artifact"],
            name="canonical component module artifact",
        )
        artifact_sha256 = sha256_file(artifact_path)
        if artifact_sha256 != str(module_record["sha256"]):
            raise ValueError("component module artifact differs from inventory manifest")
        component_records.append(
            {
                "component_id": component_id,
                "module": module,
                "canonical_block_index": int(row["canonical_block_index"]),
                "canonical_module_artifact_path": _project_relative(
                    project_root, artifact_path, name="canonical module artifact"
                ),
                "canonical_module_artifact_sha256": artifact_sha256,
                "source_a_factor_sha256": _digest(
                    row["source_a_factor_sha256"], name="source A factor"
                ),
                "source_b_factor_sha256": _digest(
                    row["source_b_factor_sha256"], name="source B factor"
                ),
                "unresolved_degenerate_block": bool(
                    row["unresolved_degenerate_block"]
                ),
                "raw_rank_coordinate_identity_used": False,
            }
        )
    batches_per_lane = sum(
        math.ceil(count / batch_size)
        for count in split_summary["activation_category_counts"].values()
    )
    maximum_forward_passes = 2 * len(component_records) * batches_per_lane
    body = {
        "schema": PLAN_SCHEMA,
        "status": PLAN_STATUS,
        "source": {
            "model_identity": {
                "path": _project_relative(project_root, model_identity_path, name="model identity"),
                "sha256": sha256_file(model_identity_path),
                "model_identity_id": model_identity["model_identity_id"],
                "model_id": model_identity["model_id"],
                "revision": model_identity["revision"],
                "model_files_manifest_sha256": model_identity["model_files_manifest_sha256"],
                "loader_contract": model_identity["loader_contract"],
            },
            "adapter_weights": {
                "path": _project_relative(project_root, adapter_weights_path, name="adapter weights"),
                "sha256": sha256_file(adapter_weights_path),
            },
            "adapter_config": {
                "path": _project_relative(project_root, adapter_config_path, name="adapter config"),
                "sha256": sha256_file(adapter_config_path),
            },
            "factor_extraction": {
                "path": _project_relative(project_root, extraction_manifest_path, name="extraction manifest"),
                "sha256": sha256_file(extraction_manifest_path),
                "factor_bundle_extraction_id": extraction["factor_bundle_extraction_id"],
            },
            "canonical_inventory": {
                "path": _project_relative(project_root, inventory_manifest_path, name="inventory manifest"),
                "sha256": sha256_file(inventory_manifest_path),
                "canonical_inventory_id": inventory["canonical_inventory_id"],
            },
            "semantic_split": {
                "path": _project_relative(project_root, split_manifest_path, name="split manifest"),
                **split_summary,
            },
            "dataset": {
                "path": _project_relative(project_root, dataset_path, name="dataset"),
                "sha256": split_summary["dataset_sha256"],
            },
        },
        "capture_contract": {
            "activation_key": activation_key,
            "adapter_name": adapter_name,
            "layer": int(ACTIVATION_KEY_RE.fullmatch(activation_key).group("layer")),
            "phase": "proposal_start",
            "component_basis": "canonical_thin_svd_spectral_blocks",
            "component_records": component_records,
            "category_names": split_summary["categories"],
            "category_row_counts": split_summary["activation_category_counts"],
            "epsilon": float(epsilon),
            "multipliers": [1.0 + float(epsilon), 1.0 - float(epsilon)],
            "batch_size": batch_size,
            "device": device,
            "dtype": dtype,
            "probe_seed": probe_seed,
            "probe_count": probe_count,
            "maximum_relative_probe_error": maximum_relative_probe_error,
            "generated_tokens": 0,
            "output_dir": output_dir,
            "semantic_sketch": {
                "countsketch_dimension": countsketch_dimension,
                "countsketch_seed": countsketch_seed,
                "uncertainty_blocks": uncertainty_blocks,
            },
        },
        "resource_ceiling": {
            "dataset_rows": split_summary["rows"],
            "canonical_components": len(component_records),
            "lanes_per_component": 2,
            "batches_per_lane": batches_per_lane,
            "maximum_forward_passes": maximum_forward_passes,
            "generated_tokens": 0,
        },
        "policy": {
            "exact_source_hashes_required": True,
            "semantic_split_v2_or_v3_required": True,
            "all_activation_categories_required": True,
            "exact_shared_pair_set_required": True,
            "canonical_svd_components_only": True,
            "raw_lora_rank_identity_forbidden": True,
            "central_difference_lanes": True,
            "per_example_category_effects_required": True,
            "condition_template_countsketch_required": True,
            "authorization_receipt_must_preexist": True,
            "planner_cannot_create_authorization": True,
        },
        "scientific_boundary": {
            "synthetic_benchmark_only": synthetic_benchmark_only,
            "model_loaded": False,
            "torch_transformers_or_peft_imported": False,
            "model_inference_performed": False,
            "adapter_parameters_mutated": False,
            "weights_persisted": False,
            "effect_capture_performed": False,
            "real_actions_authorized": False,
            "execution_authorized": False,
        },
    }
    return {**body, "capture_job_plan_id": content_sha256(body)}


def verify_capture_job_plan(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = _load_addressed(
        path, schema=PLAN_SCHEMA, identifier_field="capture_job_plan_id"
    )
    source = observed["source"]
    split = source["semantic_split"]
    contract = observed["capture_contract"]
    expected = build_capture_job_plan(
        project_root=project_root,
        model_identity_path=project_root / source["model_identity"]["path"],
        adapter_weights_path=project_root / source["adapter_weights"]["path"],
        adapter_config_path=project_root / source["adapter_config"]["path"],
        extraction_manifest_path=project_root / source["factor_extraction"]["path"],
        inventory_manifest_path=project_root / source["canonical_inventory"]["path"],
        split_manifest_path=project_root / split["path"],
        split_role=split["split_role"],
        dataset_path=project_root / source["dataset"]["path"],
        component_ids=[row["component_id"] for row in contract["component_records"]],
        adapter_name=str(contract["adapter_name"]),
        activation_key=contract["activation_key"],
        epsilon=float(contract["epsilon"]),
        batch_size=int(contract["batch_size"]),
        device=str(contract["device"]),
        dtype=str(contract["dtype"]),
        output_dir=str(contract["output_dir"]),
        probe_seed=int(contract["probe_seed"]),
        probe_count=int(contract["probe_count"]),
        maximum_relative_probe_error=float(contract["maximum_relative_probe_error"]),
        countsketch_dimension=int(contract["semantic_sketch"]["countsketch_dimension"]),
        countsketch_seed=int(contract["semantic_sketch"]["countsketch_seed"]),
        uncertainty_blocks=int(contract["semantic_sketch"]["uncertainty_blocks"]),
        synthetic_benchmark_only=bool(
            observed["scientific_boundary"]["synthetic_benchmark_only"]
        ),
    )
    if observed != expected:
        raise ValueError("canonical capture job plan is not reproducible")
    return {
        "schema": "sch.vpd-canonical-category-capture-job-plan-verification.v1",
        "status": "pass",
        "capture_job_plan_id": observed["capture_job_plan_id"],
        "components": len(contract["component_records"]),
        "rows": observed["resource_ceiling"]["dataset_rows"],
        "maximum_forward_passes": observed["resource_ceiling"]["maximum_forward_passes"],
        "execution_authorized": False,
    }


def verify_capture_authorization(
    path: Path, *, plan_path: Path, project_root: Path
) -> dict[str, Any]:
    plan_verification = verify_capture_job_plan(plan_path, project_root=project_root)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    authorization = _load_addressed(
        path,
        schema=AUTHORIZATION_SCHEMA,
        identifier_field="capture_authorization_id",
    )
    if authorization.get("status") != "authorized_by_user":
        raise ValueError("canonical capture authorization status differs")
    source = authorization.get("source", {})
    if (
        source.get("capture_job_plan_id") != plan["capture_job_plan_id"]
        or source.get("capture_job_plan_sha256") != sha256_file(plan_path)
    ):
        raise ValueError("canonical capture authorization plan source differs")
    scope = authorization.get("scope", {})
    required = {
        "authorized_action": "canonical_category_effect_capture",
        "authorized_job_id": plan["capture_job_plan_id"],
        "model_inference_authorized": True,
        "temporary_adapter_parameter_mutation_authorized": True,
        "base_model_weights_mutable": False,
        "adapter_weight_persistence_authorized": False,
        "generated_tokens_authorized": False,
        "real_actions_authorized": False,
        "output_dir": plan["capture_contract"]["output_dir"],
        "maximum_forward_passes": plan["resource_ceiling"]["maximum_forward_passes"],
    }
    for field, expected in required.items():
        if scope.get(field) != expected:
            raise ValueError(f"canonical capture authorization scope differs: {field}")
    if bool(authorization.get("synthetic_authorization_only")) != bool(
        plan["scientific_boundary"]["synthetic_benchmark_only"]
    ):
        raise ValueError("canonical capture authorization provenance differs")
    return {
        "schema": "sch.vpd-explicit-canonical-capture-authorization-verification.v1",
        "status": "pass",
        "capture_authorization_id": authorization["capture_authorization_id"],
        "capture_authorization_sha256": sha256_file(path),
        "capture_job_plan_id": plan_verification["capture_job_plan_id"],
        "authorized_action": "canonical_category_effect_capture",
        "maximum_forward_passes": required["maximum_forward_passes"],
        "real_actions_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or verify an exact canonical VPD category-capture job V1."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--verify-authorization", type=Path)
    parser.add_argument("--model-identity", type=Path)
    parser.add_argument("--adapter-weights", type=Path)
    parser.add_argument("--adapter-config", type=Path)
    parser.add_argument("--extraction-manifest", type=Path)
    parser.add_argument("--inventory-manifest", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split-role", choices=sorted(ALLOWED_SPLIT_ROLES))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--component-id", action="append")
    parser.add_argument("--adapter-name", default="default")
    parser.add_argument("--activation-key", default="layer_35__proposal_start")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=sorted(ALLOWED_DTYPES), default="bfloat16")
    parser.add_argument("--output-dir")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--synthetic-benchmark-only", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    if args.verify is not None:
        if args.verify_authorization is not None:
            result = verify_capture_authorization(
                args.verify_authorization,
                plan_path=args.verify,
                project_root=project_root,
            )
        else:
            result = verify_capture_job_plan(args.verify, project_root=project_root)
    else:
        required = {
            "model_identity": args.model_identity,
            "adapter_weights": args.adapter_weights,
            "adapter_config": args.adapter_config,
            "extraction_manifest": args.extraction_manifest,
            "inventory_manifest": args.inventory_manifest,
            "split_manifest": args.split_manifest,
            "split_role": args.split_role,
            "dataset": args.dataset,
            "device": args.device,
            "output_dir": args.output_dir,
            "output": args.output,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"missing planning arguments: {', '.join(missing)}")
        result = build_capture_job_plan(
            project_root=project_root,
            model_identity_path=args.model_identity,
            adapter_weights_path=args.adapter_weights,
            adapter_config_path=args.adapter_config,
            extraction_manifest_path=args.extraction_manifest,
            inventory_manifest_path=args.inventory_manifest,
            split_manifest_path=args.split_manifest,
            split_role=args.split_role,
            dataset_path=args.dataset,
            component_ids=args.component_id,
            adapter_name=args.adapter_name,
            activation_key=args.activation_key,
            epsilon=args.epsilon,
            batch_size=args.batch_size,
            device=args.device,
            dtype=args.dtype,
            output_dir=args.output_dir,
            synthetic_benchmark_only=args.synthetic_benchmark_only,
        )
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
