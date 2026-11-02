from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.activation_categories import categorize_episodes
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
)
from experiments.authz_mechinterp.plan_canonical_vpd_capture_job_v1 import (
    PLAN_SCHEMA as PLAN_SCHEMA_V1,
    _read_jsonl,
)
from experiments.authz_mechinterp.plan_canonical_vpd_capture_job_v2 import (
    PLAN_SCHEMA as PLAN_SCHEMA_V2,
)
from experiments.authz_mechinterp.route_vpd_cluster_episodes_v1 import (
    _array_sha256,
    _canonical_groups,
)
from experiments.authz_mechinterp.run_canonical_vpd_capture_job_v1 import (
    verify_capture_execution_receipt as verify_execution_v1,
)
from experiments.authz_mechinterp.run_canonical_vpd_capture_job_v2 import (
    verify_capture_execution_receipt as verify_execution_v2,
)


ADAPTER_SCHEMA = "sch.vpd-capture-receipt-router-input-adapter.v1"


def split_role_identity_v1(split: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    body = {
        "schema": "sch.vpd-semantic-split-role-identity.v1",
        "semantic_split_id": str(split["split_id"]),
        "split_role": str(split["split_role"]),
        "dataset_sha256": str(split["dataset_sha256"]),
        "episode_id_set_sha256": str(split["episode_id_set_sha256"]),
        "pair_id_set_sha256": str(split["pair_id_set_sha256"]),
        "template_group_set_sha256": str(split["template_group_set_sha256"]),
    }
    return content_sha256(body), body


def _normalize_episode(row: dict[str, Any], *, category: str) -> dict[str, str]:
    condition = str(row.get("condition_id") or row.get("condition") or "")
    template = str(
        row.get("template_group_id")
        or row.get("template_group")
        or row.get("activation_category_pack", {}).get(
            "semantic_template_group", ""
        )
    )
    normalized = {
        "episode_id": str(row.get("episode_id", "")),
        "pair_id": str(row.get("pair_id", "")),
        "activation_category": category,
        "condition_id": condition,
        "template_group_id": template,
    }
    if any(not value for value in normalized.values()):
        raise ValueError("capture-to-router episode identity is incomplete")
    return normalized


def load_capture_router_inputs_v1(
    *,
    receipt_path: Path,
    plan_path: Path,
    authorization_path: Path,
    project_root: Path,
    activation_category: str,
    partition_identity_admission_id: str,
    partition_admission_status: str,
    category_partition_status: str,
    category_partition_sha256: str,
    component_membership: dict[str, str],
    minimum_response_rms: float,
    minimum_margin_absolute: float,
    minimum_margin_ratio: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    project_root = project_root.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_schema = str(plan.get("schema", ""))
    if plan_schema == PLAN_SCHEMA_V1:
        execution_verification = verify_execution_v1(
            receipt_path,
            plan_path=plan_path,
            authorization_path=authorization_path,
            project_root=project_root,
        )
    elif plan_schema == PLAN_SCHEMA_V2:
        execution_verification = verify_execution_v2(
            receipt_path,
            plan_path=plan_path,
            authorization_path=authorization_path,
            project_root=project_root,
        )
    else:
        raise ValueError("capture-to-router adapter plan schema is unsupported")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    split = plan["source"]["semantic_split"]
    split_role = str(split["split_role"])
    if split_role not in {"discovery", "cluster_validation", "sae_training"}:
        raise ValueError("capture-to-router adapter split role is forbidden")
    if activation_category not in plan["capture_contract"]["category_names"]:
        raise ValueError("capture-to-router category is absent from capture plan")
    dataset_path = project_root / plan["source"]["dataset"]["path"]
    if sha256_file(dataset_path) != plan["source"]["dataset"]["sha256"]:
        raise ValueError("capture-to-router dataset hash differs")
    grouped = categorize_episodes(_read_jsonl(dataset_path))
    category_rows = grouped.get(activation_category, [])
    episodes = [
        _normalize_episode(row, category=activation_category) for row in category_rows
    ]
    episode_ids = [row["episode_id"] for row in episodes]
    if not episodes or episode_ids != sorted(episode_ids):
        raise ValueError("capture-to-router episode order is not exact sorted order")
    capture_episode_order_sha256 = content_sha256(episode_ids)
    split_role_id, split_role_identity_body = split_role_identity_v1(
        {
            "split_id": split["split_id"],
            "split_role": split_role,
            "dataset_sha256": plan["source"]["dataset"]["sha256"],
            "episode_id_set_sha256": split["episode_id_set_sha256"],
            "pair_id_set_sha256": split["pair_id_set_sha256"],
            "template_group_set_sha256": split["template_group_set_sha256"],
        }
    )

    component_records = receipt["artifacts"]["components"]
    if {str(row["component_id"]) for row in component_records} != set(
        component_membership
    ):
        raise ValueError("capture-to-router components differ from partition membership")
    groups, _ = _canonical_groups(component_membership)
    if content_sha256(groups) != category_partition_sha256:
        raise ValueError("capture-to-router partition hash differs from membership")
    component_effects: dict[str, np.ndarray] = {}
    effect_sha256: dict[str, str] = {}
    artifact_sha256: dict[str, str] = {}
    component_provenance: dict[str, Any] = {}
    key = f"delta_samples__{activation_category}"
    for record in sorted(component_records, key=lambda row: str(row["component_id"])):
        component_id = str(record["component_id"])
        if record["episode_ids_by_category_sha256"].get(
            activation_category
        ) != capture_episode_order_sha256:
            raise ValueError("capture-to-router component episode order differs")
        artifact_path = receipt_path.parent / str(record["path"])
        if sha256_file(artifact_path) != record["sha256"]:
            raise ValueError("capture-to-router component artifact hash differs")
        with np.load(artifact_path, allow_pickle=False) as archive:
            if sorted(archive.files) != sorted(record["arrays"]):
                raise ValueError("capture-to-router component array inventory differs")
            if key not in archive.files:
                raise ValueError("capture-to-router category effect array is absent")
            values = np.asarray(archive[key], dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(episodes):
            raise ValueError("capture-to-router category effect shape differs")
        component_effects[component_id] = values
        effect_sha256[component_id] = _array_sha256(values)
        artifact_sha256[component_id] = str(record["sha256"])
        component_provenance[component_id] = {
            "artifact_path": str(record["path"]),
            "artifact_sha256": str(record["sha256"]),
            "effect_array": key,
            "effect_array_sha256": effect_sha256[component_id],
            "capture_receipt_id": record["capture_receipt"][
                "canonical_category_effect_capture_id"
            ],
        }
    execution_id = str(receipt["capture_execution_id"])
    routing_inputs = {
        "activation_category": activation_category,
        "split_role": split_role,
        "split_id": split_role_id,
        "partition_identity_admission_id": partition_identity_admission_id,
        "partition_admission_status": partition_admission_status,
        "category_partition_status": category_partition_status,
        "category_partition_sha256": category_partition_sha256,
        "capture_execution_id": execution_id,
        "episodes": episodes,
        "capture_episode_order_sha256": capture_episode_order_sha256,
        "component_membership": dict(sorted(component_membership.items())),
        "component_effects": component_effects,
        "expected_component_effect_sha256": effect_sha256,
        "component_artifact_sha256": artifact_sha256,
        "activation_rows_per_episode": 1,
        "minimum_response_rms": minimum_response_rms,
        "minimum_margin_absolute": minimum_margin_absolute,
        "minimum_margin_ratio": minimum_margin_ratio,
    }
    body = {
        "schema": ADAPTER_SCHEMA,
        "status": "complete_verified_capture_receipt_to_router_inputs",
        "source": {
            "plan_schema": plan_schema,
            "plan_path": plan_path.resolve().relative_to(project_root).as_posix(),
            "plan_sha256": sha256_file(plan_path),
            "authorization_path": authorization_path.resolve()
            .relative_to(project_root)
            .as_posix(),
            "authorization_sha256": sha256_file(authorization_path),
            "execution_receipt_path": receipt_path.resolve()
            .relative_to(project_root)
            .as_posix(),
            "capture_execution_id": execution_id,
            "execution_verification_schema": execution_verification["schema"],
            "semantic_split_id": str(split["split_id"]),
            "split_role_id": split_role_id,
            "split_role_identity": split_role_identity_body,
            "split_role": split_role,
            "dataset_sha256": plan["source"]["dataset"]["sha256"],
            "activation_category": activation_category,
            "partition_identity_admission_id": partition_identity_admission_id,
            "category_partition_sha256": category_partition_sha256,
        },
        "episode_contract": {
            "episodes": len(episodes),
            "episode_order_sha256": capture_episode_order_sha256,
            "activation_rows_per_episode": 1,
            "effect_rows": len(episodes),
        },
        "component_provenance": component_provenance,
        "boundary": {
            "model_loaded_by_adapter": False,
            "model_inference_performed_by_adapter": False,
            "weights_loaded_or_mutated_by_adapter": False,
            "capture_artifacts_read_only": True,
            "sae_training_authorized": False,
            "execution_authorized": False,
        },
    }
    adapter_receipt = {
        **body,
        "capture_router_adapter_id": content_sha256(body),
    }
    return routing_inputs, adapter_receipt


def verify_capture_router_adapter_receipt_v1(receipt: dict[str, Any]) -> dict[str, Any]:
    body = dict(receipt)
    identifier = str(body.pop("capture_router_adapter_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("capture-to-router adapter receipt address differs")
    if receipt.get("schema") != ADAPTER_SCHEMA:
        raise ValueError("capture-to-router adapter receipt schema differs")
    if receipt.get("status") != "complete_verified_capture_receipt_to_router_inputs":
        raise ValueError("capture-to-router adapter receipt status differs")
    if receipt.get("boundary", {}).get("sae_training_authorized") is not False:
        raise ValueError("capture-to-router adapter receipt grants SAE training")
    return {
        "schema": "sch.vpd-capture-receipt-router-input-adapter-verification.v1",
        "status": "pass",
        "capture_router_adapter_id": identifier,
        "split_role": receipt["source"]["split_role"],
        "activation_category": receipt["source"]["activation_category"],
        "components": len(receipt["component_provenance"]),
        "episodes": receipt["episode_contract"]["episodes"],
        "sae_training_authorized": False,
    }
