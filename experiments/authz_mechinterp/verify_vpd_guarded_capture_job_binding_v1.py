from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
)
from experiments.authz_mechinterp.plan_canonical_vpd_capture_job_v2 import (
    verify_capture_authorization,
    verify_capture_job_plan,
)
from experiments.authz_mechinterp.plan_vpd_guarded_capture_job_allocations_v1 import (
    verify_manifest,
)
from experiments.authz_mechinterp.run_canonical_vpd_capture_job_v1 import (
    _safe_output_path,
)


SCHEMA = "sch.vpd-guarded-capture-job-binding-verification.v1"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_job(manifest: dict[str, Any], job_allocation_id: str) -> dict[str, Any]:
    matches = [
        job
        for role in manifest.get("roles", {}).values()
        for job in role.get("jobs", [])
        if job.get("job_allocation_id") == job_allocation_id
    ]
    if len(matches) != 1:
        raise ValueError("guarded capture job allocation is absent or ambiguous")
    return matches[0]


def _validate_plan_binding(
    *,
    plan: dict[str, Any],
    job: dict[str, Any],
    preflight: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    role = str(job["role"])
    role_record = preflight["capture_roles"].get(role)
    if not isinstance(role_record, dict):
        raise ValueError("guarded capture plan role differs")
    split = plan["source"]["semantic_split"]
    dataset = plan["source"]["dataset"]
    if (
        split.get("split_role") != role
        or dataset.get("path") != job["dataset_path"]
        or dataset.get("sha256") != job["dataset_sha256"]
        or split.get("dataset_sha256") != job["dataset_sha256"]
        or job["split_role_id"] != role_record["split_role_id"]
    ):
        raise ValueError("guarded capture plan role or dataset binding differs")
    packet_path = project_root / str(job["packet_path"])
    if sha256_file(packet_path) != job["packet_sha256"]:
        raise ValueError("guarded capture allocation packet bytes differ")
    packet = _load(packet_path)
    packet_body = dict(packet)
    packet_id = str(packet_body.pop("packet_id", ""))
    distribution = dict(packet_body.get("distribution", {}))
    distribution.pop("assigned_worker", None)
    packet_body["distribution"] = distribution
    packet_components = [str(value) for value in packet["component_ids"]]
    if (
        packet_id != job["packet_id"]
        or packet_id != content_sha256(packet_body)
        or content_sha256(packet_components) != job["component_id_set_sha256"]
    ):
        raise ValueError("guarded capture allocation packet identity differs")
    plan_components = sorted(
        str(record["component_id"])
        for record in plan["capture_contract"]["component_records"]
    )
    if plan_components != packet_components or len(plan_components) != int(
        job["component_count"]
    ):
        raise ValueError("guarded capture plan component membership differs")
    contract = plan["capture_contract"]
    resource = plan["resource_ceiling"]
    allocation = job["allocation"]
    role_shape = role_record["resource_ceiling"]
    if (
        int(contract["batch_size"]) != 4
        or int(resource["dataset_rows"]) != int(role_shape["episodes"])
        or int(resource["canonical_components"]) != len(plan_components)
        or int(resource["maximum_forward_passes"]) != allocation["batch_forwards"]
        or 2 * len(plan_components) * int(resource["dataset_rows"])
        != allocation["episode_forward_equivalents"]
        or contract["category_names"]
        != sorted(role_shape["category_episode_counts"])
        or contract["category_row_counts"]
        != [
            role_shape["category_episode_counts"][category]
            for category in sorted(role_shape["category_episode_counts"])
        ]
        or int(contract["semantic_sketch"]["uncertainty_blocks"]) != 8
        or int(contract["semantic_sketch"]["countsketch_dimension"]) != 256
    ):
        raise ValueError("guarded capture plan resource allocation differs")
    bindings = preflight["immutable_bindings"]
    bindings_complete = not bool(preflight.get("missing_required_bindings"))
    if bindings_complete:
        identity = plan["source"]["model_identity"]
        if (
            identity.get("model_id") != bindings["model_id"]
            or identity.get("revision") != bindings["model_revision"]
            or identity.get("model_files_manifest_sha256")
            != bindings["model_files_manifest_sha256"]
            or plan["source"]["adapter_weights"].get("sha256")
            != bindings["adapter_weights_sha256"]
            or plan["source"]["adapter_config"].get("sha256")
            != bindings["adapter_config_sha256"]
            or contract.get("device") != bindings["device"]
        ):
            raise ValueError("guarded capture plan execution identity differs")
        output_root = Path(str(bindings["output_root"])).resolve()
        output_dir = _safe_output_path(
            project_root, str(contract.get("output_dir", ""))
        )
        if output_dir.parent != output_root:
            raise ValueError("guarded capture plan output is not a direct run-root child")
    return {
        "role": role,
        "split_role_id": job["split_role_id"],
        "packet_id": job["packet_id"],
        "job_allocation_id": job["job_allocation_id"],
        "allocation": dict(allocation),
        "identity_bindings_complete": bindings_complete,
        "execution_ready": False,
    }


def verify_job_binding(
    *,
    plan_path: Path,
    capture_authorization_path: Path,
    allocation_manifest_path: Path,
    job_allocation_id: str,
    project_root: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    plan_check = verify_capture_job_plan(plan_path, project_root=project_root)
    authorization_check = verify_capture_authorization(
        capture_authorization_path,
        plan_path=plan_path,
        project_root=project_root,
    )
    allocation_check = verify_manifest(
        allocation_manifest_path, project_root=project_root
    )
    plan = _load(plan_path)
    manifest = _load(allocation_manifest_path)
    preflight_path = project_root / manifest["source"]["preflight_path"]
    preflight = _load(preflight_path)
    if (
        allocation_check["allocation_manifest_id"]
        != manifest["allocation_manifest_id"]
        or manifest["preflight_id"] != preflight["preflight_id"]
    ):
        raise ValueError("guarded capture allocation preflight binding differs")
    job = _find_job(manifest, job_allocation_id)
    binding = _validate_plan_binding(
        plan=plan,
        job=job,
        preflight=preflight,
        project_root=project_root,
    )
    if (
        plan_check["capture_job_plan_id"] != plan["capture_job_plan_id"]
        or authorization_check["capture_job_plan_id"]
        != plan["capture_job_plan_id"]
        or authorization_check["authorized_split_role"] != binding["role"]
    ):
        raise ValueError("guarded capture V2 plan authorization join differs")
    return {
        "schema": SCHEMA,
        "status": (
            "pass_job_exact_but_global_runtime_admission_still_required"
            if binding["identity_bindings_complete"]
            else "blocked_incomplete_execution_identity_bindings"
        ),
        "capture_job_plan_id": plan["capture_job_plan_id"],
        "capture_authorization_id": authorization_check[
            "capture_authorization_id"
        ],
        "allocation_manifest_id": manifest["allocation_manifest_id"],
        **binding,
        "global_runtime_admission_verified": False,
        "output_reservation_verified": False,
        "campaign_guard_reserved": False,
        "capture_started": False,
        "model_loaded": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a VPD plan against one exact guarded allocation.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--capture-authorization", type=Path, required=True)
    parser.add_argument("--allocation-manifest", type=Path, required=True)
    parser.add_argument("--job-allocation-id", required=True)
    args = parser.parse_args()
    result = verify_job_binding(
        plan_path=args.plan,
        capture_authorization_path=args.capture_authorization,
        allocation_manifest_path=args.allocation_manifest,
        job_allocation_id=args.job_allocation_id,
        project_root=args.project_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
