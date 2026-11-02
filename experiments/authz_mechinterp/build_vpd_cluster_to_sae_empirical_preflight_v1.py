from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_five_role_recovered_incidence_audit_v1 import (
    verify_audit,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


SCHEMA = "sch.vpd-cluster-to-sae-empirical-preflight.v1"
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
SELECTION_ROLES = ("discovery", "calibration", "cluster_validation")
ROUTING_EXTENSION_ROLE = "sae_training"
REQUIRED_BINDINGS = (
    "model_id",
    "model_revision",
    "model_files_manifest_sha256",
    "adapter_path",
    "adapter_weights_sha256",
    "adapter_config_sha256",
    "device",
    "wall_time_ceiling_seconds",
    "cost_ceiling_usd",
    "output_root",
    "authorization_receipt_id",
    "authorization_receipt_sha256",
    "explicit_user_authorization_statement",
)


def _verify_campaign_address(campaign: dict[str, Any]) -> str:
    body = dict(campaign)
    identifier = str(body.pop("campaign_plan_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("effect campaign V4 content address differs")
    if identifier != "ffcc7351bade510166c449099f19d43bd1a28a89e83bc85bc912f883c742b528":
        raise ValueError("effect campaign V4 identity differs")
    if tuple(sorted(campaign["resources"]["by_role"])) != tuple(
        sorted(SELECTION_ROLES)
    ):
        raise ValueError("effect campaign selection roles differ")
    return identifier


def _validate_bindings(bindings: dict[str, Any]) -> list[str]:
    unknown = sorted(set(bindings) - set(REQUIRED_BINDINGS))
    if unknown:
        raise ValueError("empirical preflight has unknown bindings: " + ", ".join(unknown))
    missing: list[str] = []
    for field in REQUIRED_BINDINGS:
        value = bindings.get(field)
        if value is None or value == "":
            missing.append(field)
            continue
        if field.endswith("_sha256") or field == "authorization_receipt_id":
            if not isinstance(value, str) or DIGEST_RE.fullmatch(value) is None:
                raise ValueError(f"empirical preflight binding {field} is not a digest")
        elif field == "wall_time_ceiling_seconds":
            if type(value) is not int or value <= 0:
                raise ValueError("empirical preflight wall-time ceiling is invalid")
        elif field == "cost_ceiling_usd":
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("empirical preflight cost ceiling is invalid")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"empirical preflight binding {field} is invalid")
    return missing


def _role_resource_from_manifest(
    role: dict[str, Any],
    *,
    component_count: int,
    batch_size: int,
    hidden_size: int,
    uncertainty_blocks: int,
    sketch_dimension: int,
) -> dict[str, Any]:
    category_counts = {
        str(key): int(value)
        for key, value in sorted(role["activation_category_counts"].items())
    }
    template_cells = len(role["template_group_counts"]) * len(category_counts)
    condition_cells = 11
    floats_per_component = (
        hidden_size
        + len(category_counts) * hidden_size
        + len(category_counts) * uncertainty_blocks * sketch_dimension
        + condition_cells * sketch_dimension
        + template_cells * sketch_dimension
    )
    return {
        "episodes": int(role["rows"]),
        "pairs": int(role["pairs"]),
        "categories": len(category_counts),
        "expected_hidden_size": hidden_size,
        "category_episode_counts": category_counts,
        "condition_cells": condition_cells,
        "template_cells": template_cells,
        "signed_perturbation_lanes": 2,
        "episode_forward_equivalents": 2 * component_count * int(role["rows"]),
        "batch_forwards_at_planned_batch_size": component_count
        * 2
        * sum(math.ceil(value / batch_size) for value in category_counts.values()),
        "estimated_float32_effect_payload_bytes": component_count
        * floats_per_component
        * 4,
        "estimate_excludes_npz_container_ids_counters_model_and_runtime_overhead": True,
    }


def build_preflight(
    *,
    project_root: Path,
    bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    audit_path = project_root / "artifacts/vpd_five_role_recovered_incidence_audit_v1/report.json"
    audit_verification = verify_audit(audit_path, project_root=project_root)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    split_path = project_root / "artifacts/vpd_activation_category_capacity_split_v4/split_manifest.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    campaign_path = project_root / "artifacts/vpd_effect_capture_campaign_v4/plan.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign_id = _verify_campaign_address(campaign)
    bundle_path = project_root / "artifacts/bonsai8b_authz_v8/vpd_sae_clusters/manifest.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if (
        bundle["invariants"]["component_count"] != 4032
        or bundle["invariants"]["assigned_component_count"] != 4032
        or not bundle["invariants"]["exact_partition"]
        or len(bundle["artifacts"]["packets"]) != 16
    ):
        raise ValueError("V8 component packet census differs")
    for record in bundle["artifacts"]["packets"]:
        packet_path = bundle_path.parent / record["packet_path"]
        if sha256_file(packet_path) != record["packet_sha256"]:
            raise ValueError("V8 packet artifact hash differs")

    parameters = campaign["parameters"]
    component_count = int(bundle["invariants"]["component_count"])
    selection_resources = campaign["resources"]
    sae_resource = _role_resource_from_manifest(
        split["splits"][ROUTING_EXTENSION_ROLE],
        component_count=component_count,
        batch_size=int(parameters["batch_size"]),
        hidden_size=int(parameters["expected_hidden_size"]),
        uncertainty_blocks=int(parameters["uncertainty_blocks"]),
        sketch_dimension=int(parameters["uncertainty_sketch_dimension"]),
    )
    full_resources = {
        "roles": 4,
        "components_per_role": component_count,
        "packets_per_role": 16,
        "selection_compute_jobs": int(selection_resources["total"]["compute_jobs"]),
        "sae_routing_extension_compute_jobs": 16,
        "maximum_compute_jobs": int(selection_resources["total"]["compute_jobs"])
        + 16,
        "maximum_episode_forward_equivalents": int(
            selection_resources["total"]["episode_forward_equivalents"]
        )
        + sae_resource["episode_forward_equivalents"],
        "maximum_batch_forwards_at_batch_size_4": int(
            selection_resources["total"]["batch_forwards_at_planned_batch_size"]
        )
        + sae_resource["batch_forwards_at_planned_batch_size"],
        "estimated_float32_effect_payload_bytes": int(
            selection_resources["total"]["estimated_float32_effect_payload_bytes"]
        )
        + sae_resource["estimated_float32_effect_payload_bytes"],
        "generated_tokens": 0,
        "not_estimated": [
            "tokenized_sequence_lengths",
            "model_memory",
            "runtime_overhead",
            "npz_overhead",
            "wall_time",
            "monetary_cost",
        ],
    }
    supplied = dict(bindings or {})
    missing = _validate_bindings(supplied)
    normalized_bindings = {
        field: supplied.get(field) for field in REQUIRED_BINDINGS
    }
    status = (
        "blocked_missing_immutable_identities_and_explicit_authorization"
        if missing
        else "identity_complete_requires_separate_authenticated_runtime_authorization"
    )
    output_root = supplied.get("output_root") or "${OUTPUT_ROOT}"
    body = {
        "schema": SCHEMA,
        "status": status,
        "source": {
            "five_role_audit": {
                "path": audit_path.relative_to(project_root).as_posix(),
                "sha256": sha256_file(audit_path),
                "audit_id": audit_verification["audit_id"],
            },
            "semantic_split": {
                "path": split_path.relative_to(project_root).as_posix(),
                "sha256": sha256_file(split_path),
                "split_id": split["split_id"],
            },
            "selection_campaign_v4": {
                "path": campaign_path.relative_to(project_root).as_posix(),
                "sha256": sha256_file(campaign_path),
                "campaign_plan_id": campaign_id,
            },
            "component_packet_bundle_v8": {
                "path": bundle_path.relative_to(project_root).as_posix(),
                "sha256": sha256_file(bundle_path),
                "component_count": component_count,
                "component_id_set_sha256": bundle["invariants"][
                    "component_id_set_sha256"
                ],
                "packets": len(bundle["artifacts"]["packets"]),
            },
            "runners": {
                "selection_effect_worker_v2_sha256": sha256_file(
                    project_root
                    / "experiments/authz_mechinterp/compute_vpd_effect_shard_v2.py"
                ),
                "sae_training_capture_worker_v2_sha256": sha256_file(
                    project_root
                    / "experiments/authz_mechinterp/run_canonical_vpd_capture_job_v2.py"
                ),
            },
        },
        "capture_roles": {
            role: {
                "purpose": split["protocol"]["role_uses"][role],
                "dataset_path": (
                    split_path.parent / split["splits"][role]["artifact"]["path"]
                ).relative_to(project_root).as_posix(),
                "dataset_sha256": split["splits"][role]["artifact"]["sha256"],
                "split_role_id": audit["role_ids"][role],
                "preflight_use": (
                    "cluster_selection_effect_capture"
                    if role in SELECTION_ROLES
                    else "routing_assignment_effect_capture_only_before_dictionary_learning"
                ),
                "resource_ceiling": (
                    selection_resources["by_role"][role]
                    if role in SELECTION_ROLES
                    else sae_resource
                ),
            }
            for role in (*SELECTION_ROLES, ROUTING_EXTENSION_ROLE)
        },
        "stage_order": [
            "discovery_capture",
            "discovery_gate",
            "calibration_capture",
            "calibration_gate",
            "cluster_validation_capture",
            "partition_identity_admission",
            "sae_training_routing_capture",
            "router_and_manifest_planner_v2",
        ],
        "resource_ceiling": full_resources,
        "immutable_bindings": normalized_bindings,
        "missing_required_bindings": missing,
        "output_contract": {
            "root": output_root,
            "selection_campaign": f"{output_root}/selection_effect_campaign",
            "sae_routing_extension": f"{output_root}/sae_training_routing_effects",
            "preflight_receipt": f"{output_root}/preflight_receipt.json",
            "must_not_exist_before_authorized_execution": True,
        },
        "stop_conditions": [
            "any_source_hash_or_content_address_differs",
            "any_required_binding_is_missing",
            "authorization_receipt_is_not_authenticated_and_scope_exact",
            "output_root_already_exists",
            "resource_ceiling_would_be_exceeded",
            "model_framework_would_be_imported_during_this_preflight",
            "model_or_adapter_weights_would_be_opened_during_this_preflight",
        ],
        "scientific_boundary": {
            "preflight_only": True,
            "metadata_and_manifests_read_only": True,
            "model_framework_imported": False,
            "model_or_adapter_weights_opened": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "effects_captured": False,
            "route_allocations_created": False,
            "sae_dictionary_trained": False,
            "weights_loaded_or_mutated": False,
            "final_evaluation_consumed": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
            "release_authorized": False,
        },
    }
    return {**body, "preflight_id": content_sha256(body)}


def verify_preflight(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    identifier = observed.get("preflight_id")
    body = {key: value for key, value in observed.items() if key != "preflight_id"}
    if identifier != content_sha256(body):
        raise ValueError("cluster-to-SAE preflight address differs")
    expected = build_preflight(
        project_root=project_root,
        bindings={
            key: value
            for key, value in observed["immutable_bindings"].items()
            if value is not None
        },
    )
    if observed != expected:
        raise ValueError("cluster-to-SAE preflight is not reproducible")
    return {
        "schema": "sch.vpd-cluster-to-sae-empirical-preflight-verification.v1",
        "status": "pass",
        "preflight_id": identifier,
        "readiness": observed["status"],
        "missing_required_bindings": len(observed["missing_required_bindings"]),
        "maximum_batch_forwards": observed["resource_ceiling"][
            "maximum_batch_forwards_at_batch_size_4"
        ],
        "estimated_effect_payload_bytes": observed["resource_ceiling"][
            "estimated_float32_effect_payload_bytes"
        ],
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify a no-model cluster-to-SAE empirical preflight."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--bindings", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    bindings = (
        json.loads(args.bindings.read_text(encoding="utf-8"))
        if args.bindings is not None
        else {}
    )
    if args.output is not None:
        result = build_preflight(project_root=project_root, bindings=bindings)
        write_json(args.output, result)
    else:
        if args.bindings is not None:
            raise ValueError("--bindings is only valid with --output")
        result = verify_preflight(args.verify, project_root=project_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
