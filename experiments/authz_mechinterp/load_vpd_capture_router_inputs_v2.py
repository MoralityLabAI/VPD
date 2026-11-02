from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
)
from experiments.authz_mechinterp.load_vpd_capture_router_inputs_v1 import (
    load_capture_router_inputs_v1,
    verify_capture_router_adapter_receipt_v1,
)
from experiments.authz_mechinterp.run_canonical_vpd_capture_job_v3 import (
    verify_capture_execution_receipt as verify_v3_execution_receipt,
)


ADAPTER_SCHEMA = "sch.vpd-capture-receipt-router-input-adapter.v2"


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    project = root.resolve()
    if not resolved.is_relative_to(project):
        raise ValueError("V3 capture-to-router source escapes project")
    return resolved.relative_to(project).as_posix()


def _require_content_id(value: Any, *, field: str) -> str:
    observed = str(value)
    if len(observed) != 64 or any(char not in "0123456789abcdef" for char in observed):
        raise ValueError(f"{field} must be a lowercase SHA-256 content ID")
    return observed


def load_capture_router_inputs_v2(
    *,
    v3_receipt_path: Path,
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
    allocation_manifest_path: Path | None = None,
    reservation_receipt_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load router inputs while retaining the guarded V3 execution identity.

    V3 wraps a byte-identical, independently verifiable V2 execution receipt.
    The historical V1 adapter reads that V2 payload. This successor first
    verifies the V3 allocation/reservation/guard lineage, then delegates the
    array loading to V1 and replaces the router-facing execution identity with
    the V3 content ID. No model framework or weights are opened here.
    """

    project_root = project_root.resolve()
    v3_verification = verify_v3_execution_receipt(
        v3_receipt_path,
        plan_path=plan_path,
        authorization_path=authorization_path,
        project_root=project_root,
        allocation_manifest_path=allocation_manifest_path,
        reservation_receipt_path=reservation_receipt_path,
    )
    wrapper = json.loads(v3_receipt_path.read_text(encoding="utf-8"))
    predecessor_record = wrapper["predecessor_v2_execution"]
    predecessor_path = v3_receipt_path.parent / str(predecessor_record["path"])
    if sha256_file(predecessor_path) != str(predecessor_record["sha256"]):
        raise ValueError("V3 capture-to-router predecessor bytes differ")

    routing_inputs, predecessor_adapter = load_capture_router_inputs_v1(
        receipt_path=predecessor_path,
        plan_path=plan_path,
        authorization_path=authorization_path,
        project_root=project_root,
        activation_category=activation_category,
        partition_identity_admission_id=partition_identity_admission_id,
        partition_admission_status=partition_admission_status,
        category_partition_status=category_partition_status,
        category_partition_sha256=category_partition_sha256,
        component_membership=component_membership,
        minimum_response_rms=minimum_response_rms,
        minimum_margin_absolute=minimum_margin_absolute,
        minimum_margin_ratio=minimum_margin_ratio,
    )
    predecessor_verification = verify_capture_router_adapter_receipt_v1(
        predecessor_adapter
    )
    if (
        predecessor_verification["status"] != "pass"
        or predecessor_adapter["source"]["capture_execution_id"]
        != v3_verification["capture_execution_v2_id"]
        or predecessor_record["capture_execution_id"]
        != v3_verification["capture_execution_v2_id"]
    ):
        raise ValueError("V3 capture-to-router predecessor identity differs")

    v3_id = _require_content_id(
        v3_verification["capture_execution_v3_id"], field="V3 execution ID"
    )
    guarded_id = _require_content_id(
        v3_verification["guarded_job_receipt_id"], field="guarded job receipt ID"
    )
    routing_inputs = dict(routing_inputs)
    routing_inputs["capture_execution_id"] = v3_id
    component_effect_hashes = dict(
        sorted(
            (str(key), str(value))
            for key, value in routing_inputs["expected_component_effect_sha256"].items()
        )
    )
    synthetic = bool(wrapper["scientific_boundary"]["synthetic_benchmark_only"])
    body = {
        "schema": ADAPTER_SCHEMA,
        "status": "complete_verified_guarded_v3_capture_receipt_to_router_inputs",
        "source": {
            "v3_execution_receipt_path": _relative(v3_receipt_path, project_root),
            "v3_execution_receipt_sha256": sha256_file(v3_receipt_path),
            "capture_execution_v3_id": v3_id,
            "capture_execution_v2_id": v3_verification["capture_execution_v2_id"],
            "guarded_job_receipt_id": guarded_id,
            "job_allocation_id": wrapper["source"].get("job_allocation_id"),
            "allocation_manifest_id": wrapper["source"].get(
                "allocation_manifest_id"
            ),
            "reservation_transition_id": wrapper["source"].get(
                "reservation_transition_id"
            ),
            "plan_path": _relative(plan_path, project_root),
            "plan_sha256": sha256_file(plan_path),
            "authorization_path": _relative(authorization_path, project_root),
            "authorization_sha256": sha256_file(authorization_path),
            "activation_category": activation_category,
            "split_role": v3_verification["split_role"],
        },
        "predecessor_v1_adapter": predecessor_adapter,
        "router_input_contract": {
            "capture_execution_id": v3_id,
            "capture_episode_order_sha256": routing_inputs[
                "capture_episode_order_sha256"
            ],
            "component_effect_sha256": component_effect_hashes,
            "components": len(component_effect_hashes),
            "episodes": len(routing_inputs["episodes"]),
            "activation_rows_per_episode": routing_inputs[
                "activation_rows_per_episode"
            ],
            "partition_identity_admission_id": partition_identity_admission_id,
            "category_partition_sha256": category_partition_sha256,
        },
        "scientific_boundary": {
            "synthetic_benchmark_only": synthetic,
            "v3_guarded_lineage_preserved": True,
            "allocation_and_reservation_replayed_for_production": not synthetic,
            "capture_artifacts_read_only": True,
            "model_loaded_by_adapter": False,
            "model_inference_performed_by_adapter": False,
            "weights_loaded_or_mutated_by_adapter": False,
            "sae_training_authorized": False,
            "execution_authorized": False,
        },
    }
    adapter_receipt = {**body, "capture_router_adapter_v2_id": content_sha256(body)}
    return routing_inputs, adapter_receipt


def verify_capture_router_adapter_receipt_v2(
    receipt: dict[str, Any],
) -> dict[str, Any]:
    body = dict(receipt)
    identifier = str(body.pop("capture_router_adapter_v2_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("V3 capture-to-router adapter receipt address differs")
    if receipt.get("schema") != ADAPTER_SCHEMA or receipt.get("status") != (
        "complete_verified_guarded_v3_capture_receipt_to_router_inputs"
    ):
        raise ValueError("V3 capture-to-router adapter receipt contract differs")
    predecessor = receipt.get("predecessor_v1_adapter")
    if not isinstance(predecessor, dict):
        raise ValueError("V3 capture-to-router predecessor adapter is absent")
    predecessor_verification = verify_capture_router_adapter_receipt_v1(predecessor)
    source = receipt.get("source", {})
    contract = receipt.get("router_input_contract", {})
    boundary = receipt.get("scientific_boundary", {})
    v3_id = _require_content_id(
        source.get("capture_execution_v3_id"), field="V3 execution ID"
    )
    v2_id = _require_content_id(
        source.get("capture_execution_v2_id"), field="V2 execution ID"
    )
    guarded_id = _require_content_id(
        source.get("guarded_job_receipt_id"), field="guarded job receipt ID"
    )
    if (
        predecessor_verification["status"] != "pass"
        or predecessor["source"]["capture_execution_id"] != v2_id
        or contract.get("capture_execution_id") != v3_id
        or contract.get("components") != predecessor_verification["components"]
        or contract.get("episodes") != predecessor_verification["episodes"]
        or source.get("activation_category")
        != predecessor["source"]["activation_category"]
        or source.get("split_role") != predecessor["source"]["split_role"]
        or boundary.get("v3_guarded_lineage_preserved") is not True
        or boundary.get("capture_artifacts_read_only") is not True
        or any(
            boundary.get(field) is not False
            for field in (
                "model_loaded_by_adapter",
                "model_inference_performed_by_adapter",
                "weights_loaded_or_mutated_by_adapter",
                "sae_training_authorized",
                "execution_authorized",
            )
        )
    ):
        raise ValueError("V3 capture-to-router adapter lineage differs")
    return {
        "schema": "sch.vpd-capture-receipt-router-input-adapter-verification.v2",
        "status": "pass",
        "capture_router_adapter_v2_id": identifier,
        "capture_execution_v3_id": v3_id,
        "capture_execution_v2_id": v2_id,
        "guarded_job_receipt_id": guarded_id,
        "split_role": source["split_role"],
        "activation_category": source["activation_category"],
        "components": contract["components"],
        "episodes": contract["episodes"],
        "v3_guarded_lineage_preserved": True,
        "sae_training_authorized": False,
    }
