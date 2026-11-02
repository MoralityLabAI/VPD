from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.plan_vpd_effect_capture_campaign import (
    verify_campaign_plan as verify_predecessor_campaign,
)


SCHEMA = "sch.vpd-effect-capture-campaign-plan.v2"
WRAPPER_MODULE = "experiments.authz_mechinterp.compute_vpd_effect_shard_v2"
WRAPPER_PATH = "experiments/authz_mechinterp/compute_vpd_effect_shard_v2.py"


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("V2 effect campaign source escapes the project")
    return resolved.relative_to(root.resolve()).as_posix()


def _readdress_job(row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    body = copy.deepcopy(row)
    predecessor_id = str(body.pop("job_id"))
    return {**body, "job_id": content_sha256(body)}, predecessor_id


def build_campaign_plan_v2(
    *,
    predecessor_plan_path: Path,
    bundle_dir: Path,
    split_dir: Path,
    protocol_config_path: Path,
    hierarchy_plan_path: Path,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    predecessor_verification = verify_predecessor_campaign(
        predecessor_plan_path,
        bundle_dir=bundle_dir,
        split_dir=split_dir,
        protocol_config_path=protocol_config_path,
        hierarchy_plan_path=hierarchy_plan_path,
    )
    predecessor = json.loads(predecessor_plan_path.read_text(encoding="utf-8"))
    if predecessor.get("schema") != "sch.vpd-effect-capture-campaign-plan.v1":
        raise ValueError("V2 effect campaign predecessor is not V1")
    body = copy.deepcopy(predecessor)
    body.pop("campaign_plan_id")
    body["schema"] = SCHEMA
    body["status"] = (
        "locally_verified_seed_corrected_wrapper_requires_frozen_identities_"
        "signing_registry_and_compute_authorization"
    )

    compute_id_map: dict[str, str] = {}
    compute_jobs = []
    for original in predecessor["compute_jobs"]:
        row = copy.deepcopy(original)
        arguments = [str(value) for value in row["arguments"]]
        if arguments[:2] != [
            "-m",
            "experiments.authz_mechinterp.compute_vpd_effect_shard",
        ]:
            raise ValueError("V2 effect campaign predecessor worker command differs")
        arguments[1] = WRAPPER_MODULE
        row["arguments"] = arguments
        addressed, predecessor_id = _readdress_job(row)
        compute_id_map[predecessor_id] = addressed["job_id"]
        compute_jobs.append(addressed)

    signing_id_map: dict[str, str] = {}
    signing_jobs = []
    for original in predecessor["signing_jobs"]:
        row = copy.deepcopy(original)
        row["depends_on_compute_job_id"] = compute_id_map[
            str(row["depends_on_compute_job_id"])
        ]
        addressed, predecessor_id = _readdress_job(row)
        signing_id_map[predecessor_id] = addressed["job_id"]
        signing_jobs.append(addressed)

    merge_id_map: dict[str, str] = {}
    merge_jobs = []
    for original in predecessor["merge_jobs"]:
        row = copy.deepcopy(original)
        row["depends_on_signing_job_ids"] = [
            signing_id_map[str(value)] for value in row["depends_on_signing_job_ids"]
        ]
        addressed, predecessor_id = _readdress_job(row)
        merge_id_map[predecessor_id] = addressed["job_id"]
        merge_jobs.append(addressed)

    gate_id_map: dict[str, str] = {}
    stage_gate_jobs = []
    for original in predecessor["stage_gate_jobs"]:
        row = copy.deepcopy(original)
        row["depends_on_merge_job_ids"] = [
            merge_id_map[str(value)] for value in row["depends_on_merge_job_ids"]
        ]
        if "depends_on_gate_job_ids" in row:
            row["depends_on_gate_job_ids"] = [
                gate_id_map[str(value)] for value in row["depends_on_gate_job_ids"]
            ]
        addressed, predecessor_id = _readdress_job(row)
        gate_id_map[predecessor_id] = addressed["job_id"]
        stage_gate_jobs.append(addressed)

    cluster_job = copy.deepcopy(predecessor["cluster_job"])
    cluster_job["depends_on_merge_job_ids"] = [
        merge_id_map[str(value)] for value in cluster_job["depends_on_merge_job_ids"]
    ]
    cluster_job["depends_on_gate_job_ids"] = [
        gate_id_map[str(value)] for value in cluster_job["depends_on_gate_job_ids"]
    ]
    cluster_job, _ = _readdress_job(cluster_job)

    body["compute_jobs"] = compute_jobs
    body["signing_jobs"] = signing_jobs
    body["merge_jobs"] = merge_jobs
    body["stage_gate_jobs"] = stage_gate_jobs
    body["cluster_job"] = cluster_job
    body["source"]["predecessor_campaign"] = {
        "path": _relative(predecessor_plan_path, project_root),
        "sha256": sha256_file(predecessor_plan_path),
        "campaign_plan_id": predecessor_verification["campaign_plan_id"],
        "verification": predecessor_verification,
    }
    body["source"]["implementation"]["effect_worker_wrapper"] = {
        "path": WRAPPER_PATH,
        "sha256": sha256_file(project_root / WRAPPER_PATH),
        "preserved_worker_modified": False,
        "correction": "convert_hex_countsketch_seed_to_integer_before_modulo",
    }
    body["source"]["implementation"]["planner_v2"] = {
        "path": _relative(Path(__file__), project_root),
        "sha256": sha256_file(Path(__file__)),
    }
    body["execution_correction"] = {
        "schema": "sch.vpd-effect-worker-execution-correction.v1",
        "predecessor_worker_preserved_byte_for_byte": True,
        "predecessor_failure": "hex_digest_modulo_integer_type_error",
        "corrected_seed_rule": "int(full_sha256_hex,16) modulo 2**32",
        "wrapper_module": WRAPPER_MODULE,
        "wrapper_sha256": sha256_file(project_root / WRAPPER_PATH),
        "model_inference_used_to_discover_defect": False,
        "execution_authorized": False,
    }
    body["invariants"]["preserved_effect_worker_remains_unmodified"] = True
    body["invariants"]["all_compute_jobs_use_seed_corrected_v2_wrapper"] = True
    body["scientific_boundary"]["model_inference_performed"] = False
    body["scientific_boundary"]["campaign_v2_execution_performed"] = False
    body["scientific_boundary"]["execution_authorized"] = False
    return {**body, "campaign_plan_id": content_sha256(body)}


def verify_campaign_plan_v2(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = copy.deepcopy(observed)
    plan_id = str(body.pop("campaign_plan_id", ""))
    if content_sha256(body) != plan_id:
        raise ValueError("V2 effect campaign content address mismatch")
    if build_campaign_plan_v2(**kwargs) != observed:
        raise ValueError("V2 effect campaign is not reproducible")
    return {
        "schema": "sch.vpd-effect-capture-campaign-verification.v2",
        "status": "pass",
        "campaign_plan_id": plan_id,
        "compute_jobs": len(observed["compute_jobs"]),
        "components_per_role": observed["resources"]["total"]["components_per_role"],
        "worker_wrapper_sha256": observed["execution_correction"]["wrapper_sha256"],
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Path]:
    artifacts = project_root / "artifacts"
    return {
        "predecessor_plan_path": artifacts
        / "vpd_effect_capture_campaign_v3"
        / "plan.json",
        "bundle_dir": artifacts / "bonsai8b_authz_v8" / "vpd_sae_clusters",
        "split_dir": artifacts / "vpd_activation_category_capacity_split_v4",
        "protocol_config_path": project_root
        / "configs"
        / "evals"
        / "vpd_activation_category_compartments_v4.json",
        "hierarchy_plan_path": artifacts
        / "bonsai8b_authz_v5"
        / "hierarchy_group_plan.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive or verify the seed-corrected VPD effect campaign V2."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None:
        result = verify_campaign_plan_v2(args.verify, **inputs)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_campaign_plan_v2(**inputs)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
