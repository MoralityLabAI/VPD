from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.activation_categories import categorize_episodes
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_empirical_preflight_v2 import (
    verify_preflight,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.plan_canonical_vpd_capture_job_v1 import _read_jsonl


SCHEMA = "sch.vpd-guarded-capture-job-allocation-manifest.v1"
ROLE_ORDER = ["discovery", "calibration", "cluster_validation", "sae_training"]
BATCH_SIZE = 4


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_packet_bundle(
    *, project_root: Path, preflight: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = preflight["source"]["component_packet_bundle_v8"]
    manifest_path = project_root / source["path"]
    if sha256_file(manifest_path) != source["sha256"]:
        raise ValueError("guarded allocation packet manifest differs")
    manifest = _load(manifest_path)
    entries = manifest.get("artifacts", {}).get("packets", [])
    if len(entries) != int(source["packets"]):
        raise ValueError("guarded allocation packet count differs")
    packets: list[dict[str, Any]] = []
    all_component_ids: list[str] = []
    bundle_dir = manifest_path.parent
    for entry in entries:
        packet_path = bundle_dir / str(entry["packet_path"])
        if sha256_file(packet_path) != str(entry["packet_sha256"]):
            raise ValueError("guarded allocation packet file differs")
        packet = _load(packet_path)
        body = dict(packet)
        packet_id = str(body.pop("packet_id", ""))
        distribution = dict(body.get("distribution", {}))
        distribution.pop("assigned_worker", None)
        body["distribution"] = distribution
        if packet_id != str(entry["packet_id"]) or packet_id != content_sha256(body):
            raise ValueError("guarded allocation packet address differs")
        component_ids = [str(value) for value in packet.get("component_ids", [])]
        if (
            not component_ids
            or component_ids != sorted(component_ids)
            or len(component_ids) != len(set(component_ids))
            or content_sha256(component_ids)
            != str(packet.get("cluster_component_set_sha256", ""))
        ):
            raise ValueError("guarded allocation packet component set differs")
        packets.append(
            {
                "packet_id": packet_id,
                "packet_path": packet_path.relative_to(project_root).as_posix(),
                "packet_sha256": sha256_file(packet_path),
                "component_id_set_sha256": packet["cluster_component_set_sha256"],
                "component_ids": component_ids,
            }
        )
        all_component_ids.extend(component_ids)
    if (
        len(all_component_ids) != int(source["component_count"])
        or len(all_component_ids) != len(set(all_component_ids))
        or content_sha256(sorted(all_component_ids))
        != str(source["component_id_set_sha256"])
    ):
        raise ValueError("guarded allocation bundle is not an exact component partition")
    packets.sort(key=lambda row: row["packet_id"])
    return manifest, packets


def _role_shape(
    *, project_root: Path, preflight: dict[str, Any], role: str
) -> dict[str, Any]:
    role_record = preflight["capture_roles"][role]
    dataset_path = project_root / role_record["dataset_path"]
    if sha256_file(dataset_path) != role_record["dataset_sha256"]:
        raise ValueError("guarded allocation role dataset differs")
    rows = _read_jsonl(dataset_path)
    categorized = categorize_episodes(rows)
    category_counts = {
        category: len(category_rows)
        for category, category_rows in sorted(categorized.items())
    }
    resource = role_record["resource_ceiling"]
    if (
        len(rows) != int(resource["episodes"])
        or category_counts != resource["category_episode_counts"]
    ):
        raise ValueError("guarded allocation role incidence differs")
    conditions = {
        (category, str(row.get("condition", "")))
        for category, category_rows in categorized.items()
        for row in category_rows
    }
    templates = {
        (category, str(row.get("template_group", "")))
        for category, category_rows in categorized.items()
        for row in category_rows
    }
    if (
        len(conditions) != int(resource["condition_cells"])
        or len(templates) != int(resource["template_cells"])
    ):
        raise ValueError("guarded allocation robustness cell shape differs")
    return {
        "dataset_path": role_record["dataset_path"],
        "dataset_sha256": role_record["dataset_sha256"],
        "split_role_id": role_record["split_role_id"],
        "episodes": len(rows),
        "category_counts": category_counts,
        "categories": len(category_counts),
        "condition_cells": len(conditions),
        "template_cells": len(templates),
        "hidden_size": int(resource["expected_hidden_size"]),
        "uncertainty_blocks": 8,
        "countsketch_dimension": 256,
    }


def _packet_allocation(
    *, packet: dict[str, Any], role_shape: dict[str, Any]
) -> dict[str, int]:
    components = len(packet["component_ids"])
    batches_per_lane = sum(
        math.ceil(count / BATCH_SIZE)
        for count in role_shape["category_counts"].values()
    )
    floats_per_component = (
        role_shape["hidden_size"]
        + role_shape["categories"] * role_shape["hidden_size"]
        + role_shape["categories"]
        * role_shape["uncertainty_blocks"]
        * role_shape["countsketch_dimension"]
        + role_shape["condition_cells"] * role_shape["countsketch_dimension"]
        + role_shape["template_cells"] * role_shape["countsketch_dimension"]
    )
    return {
        "compute_jobs": 1,
        "batch_forwards": 2 * components * batches_per_lane,
        "episode_forward_equivalents": 2 * components * role_shape["episodes"],
        "float32_effect_payload_bytes": components * floats_per_component * 4,
    }


def _build_body(
    *, project_root: Path, preflight: dict[str, Any], preflight_path: Path
) -> dict[str, Any]:
    _manifest, packets = _load_packet_bundle(
        project_root=project_root, preflight=preflight
    )
    roles: dict[str, Any] = {}
    global_totals = {
        "compute_jobs": 0,
        "batch_forwards": 0,
        "episode_forward_equivalents": 0,
        "float32_effect_payload_bytes": 0,
    }
    for role in ROLE_ORDER:
        shape = _role_shape(project_root=project_root, preflight=preflight, role=role)
        jobs: list[dict[str, Any]] = []
        role_totals = {name: 0 for name in global_totals}
        for packet_index, packet in enumerate(packets):
            allocation = _packet_allocation(packet=packet, role_shape=shape)
            for name, value in allocation.items():
                role_totals[name] += value
                global_totals[name] += value
            job_body = {
                "role": role,
                "split_role_id": shape["split_role_id"],
                "packet_index": packet_index,
                "packet_id": packet["packet_id"],
                "packet_path": packet["packet_path"],
                "packet_sha256": packet["packet_sha256"],
                "component_id_set_sha256": packet["component_id_set_sha256"],
                "component_count": len(packet["component_ids"]),
                "dataset_path": shape["dataset_path"],
                "dataset_sha256": shape["dataset_sha256"],
                "allocation": allocation,
            }
            jobs.append({**job_body, "job_allocation_id": content_sha256(job_body)})
        expected = preflight["capture_roles"][role]["resource_ceiling"]
        expected_totals = {
            "compute_jobs": int(preflight["resource_ceiling"]["packets_per_role"]),
            "batch_forwards": int(
                expected["batch_forwards_at_planned_batch_size"]
            ),
            "episode_forward_equivalents": int(
                expected["episode_forward_equivalents"]
            ),
            "float32_effect_payload_bytes": int(
                expected["estimated_float32_effect_payload_bytes"]
            ),
        }
        if role_totals != expected_totals:
            raise ValueError(f"guarded allocation role total differs: {role}")
        roles[role] = {
            "split_role_id": shape["split_role_id"],
            "dataset_path": shape["dataset_path"],
            "dataset_sha256": shape["dataset_sha256"],
            "resource_shape": {
                key: shape[key]
                for key in (
                    "episodes",
                    "category_counts",
                    "categories",
                    "condition_cells",
                    "template_cells",
                    "hidden_size",
                    "uncertainty_blocks",
                    "countsketch_dimension",
                )
            },
            "jobs": jobs,
            "totals": role_totals,
        }
    global_expected = {
        "compute_jobs": int(preflight["resource_ceiling"]["maximum_compute_jobs"]),
        "batch_forwards": int(
            preflight["resource_ceiling"]["maximum_batch_forwards_at_batch_size_4"]
        ),
        "episode_forward_equivalents": int(
            preflight["resource_ceiling"]["maximum_episode_forward_equivalents"]
        ),
        "float32_effect_payload_bytes": int(
            preflight["resource_ceiling"]["estimated_float32_effect_payload_bytes"]
        ),
    }
    if global_totals != global_expected:
        raise ValueError("guarded allocation global total differs")
    return {
        "schema": SCHEMA,
        "status": "complete_model_free_exact_four_role_packet_allocations",
        "preflight_id": preflight["preflight_id"],
        "source": {
            "preflight_path": preflight_path.relative_to(project_root).as_posix(),
            "preflight_sha256": sha256_file(preflight_path),
            "packet_bundle": copy.deepcopy(
                preflight["source"]["component_packet_bundle_v8"]
            ),
        },
        "contract": {
            "role_order": ROLE_ORDER,
            "packets_per_role": len(packets),
            "batch_size": BATCH_SIZE,
            "job_allocation_must_be_reserved_before_runtime_load": True,
            "job_plan_component_membership_must_equal_packet": True,
            "job_plan_dataset_and_role_must_equal_allocation": True,
            "final_evaluation_forbidden": True,
            "identity_bindings_complete": not bool(
                preflight.get("missing_required_bindings")
            ),
            "missing_required_bindings": list(
                preflight.get("missing_required_bindings", [])
            ),
        },
        "roles": roles,
        "global_totals": global_totals,
        "scientific_boundary": {
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "capture_started": False,
            "output_reserved": False,
            "weights_mutated": False,
            "generated_tokens": 0,
            "real_actions": 0,
            "sae_training_authorized": False,
            "execution_authorized": False,
        },
    }


def build_manifest(
    *, project_root: Path, preflight_path: Path | None = None
) -> dict[str, Any]:
    project_root = project_root.resolve()
    preflight_path = (
        preflight_path.resolve()
        if preflight_path is not None
        else (
            project_root
            / "artifacts/vpd_cluster_to_sae_empirical_preflight_v2/preflight.json"
        ).resolve()
    )
    if not preflight_path.is_relative_to(project_root):
        raise ValueError("guarded allocation preflight escapes project")
    verify_preflight(preflight_path, project_root=project_root)
    preflight = _load(preflight_path)
    body = _build_body(
        project_root=project_root,
        preflight=preflight,
        preflight_path=preflight_path,
    )
    return {**body, "allocation_manifest_id": content_sha256(body)}


def verify_manifest(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = _load(path)
    body = {key: value for key, value in observed.items() if key != "allocation_manifest_id"}
    if observed.get("allocation_manifest_id") != content_sha256(body):
        raise ValueError("guarded allocation manifest address differs")
    source_path = project_root / str(observed.get("source", {}).get("preflight_path", ""))
    if observed != build_manifest(
        project_root=project_root, preflight_path=source_path
    ):
        raise ValueError("guarded allocation manifest is not reproducible")
    return {
        "schema": "sch.vpd-guarded-capture-job-allocation-manifest-verification.v1",
        "status": "pass",
        "allocation_manifest_id": observed["allocation_manifest_id"],
        "roles": len(observed["roles"]),
        "jobs": sum(len(role["jobs"]) for role in observed["roles"].values()),
        "identity_bindings_complete": observed["contract"][
            "identity_bindings_complete"
        ],
        "capture_started": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan exact guarded VPD capture allocations.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.output is not None:
        result = build_manifest(
            project_root=project_root,
            preflight_path=args.preflight,
        )
        write_json(args.output, result)
    else:
        result = verify_manifest(args.verify, project_root=project_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
