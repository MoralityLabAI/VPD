from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
)


AUDIT_SCHEMA = "sch.vpd-compartment-sae-cluster-routing-gap-audit.v1"


def audit_training_plan(plan_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    jobs = plan.get("training_jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("SAE training plan requires nonempty training_jobs")

    compartments: dict[str, int] = {}
    cluster_bound_jobs: list[str] = []
    selection_modes: dict[str, int] = {}
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError("SAE training job must be an object")
        compartment = str(job.get("compartment", ""))
        job_id = str(job.get("job_id", f"index:{index}"))
        if not compartment:
            raise ValueError("SAE training job requires a compartment")
        compartments[compartment] = compartments.get(compartment, 0) + 1
        sample_contract = job.get("required_sample_contract", {})
        if not isinstance(sample_contract, dict):
            raise ValueError("SAE training job sample contract must be an object")
        selection = str(sample_contract.get("selection", "missing"))
        selection_modes[selection] = selection_modes.get(selection, 0) + 1
        cluster_fields = (
            "cluster_id",
            "cluster_members",
            "component_ids",
            "component_membership_sha256",
            "cluster_allocation_sha256",
            "routing_receipt_id",
        )
        if any(field in job or field in sample_contract for field in cluster_fields):
            cluster_bound_jobs.append(job_id)

    shared_jobs = compartments.get("shared_cross_category", 0)
    category_specific_jobs = len(jobs) - shared_jobs
    finding = (
        "cluster_bound_training_jobs_present"
        if cluster_bound_jobs
        else "category_only_no_within_category_cluster_routing"
    )
    resolved_plan = plan_path.resolve()
    try:
        source_path = resolved_plan.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        source_path = resolved_plan.as_posix()
    body = {
        "schema": AUDIT_SCHEMA,
        "status": "pass",
        "source": {
            "path": source_path,
            "sha256": sha256_file(plan_path),
            "training_plan_id": str(plan.get("training_plan_id", "")),
        },
        "observed": {
            "training_jobs": len(jobs),
            "unique_compartments": len(compartments),
            "jobs_by_compartment": dict(sorted(compartments.items())),
            "category_specific_jobs": category_specific_jobs,
            "shared_jobs": shared_jobs,
            "cluster_bound_training_jobs": len(cluster_bound_jobs),
            "cluster_bound_job_ids": sorted(cluster_bound_jobs),
            "selection_modes": dict(sorted(selection_modes.items())),
        },
        "finding": finding,
        "interpretation": (
            "The plan preserves observable activation categories but does not bind "
            "SAE jobs to admitted within-category component clusters."
            if not cluster_bound_jobs
            else "At least one SAE job contains a cluster-routing identity field."
        ),
        "boundary": {
            "model_loaded": False,
            "inference_performed": False,
            "weights_written": False,
            "sae_training_authorized": False,
        },
    }
    return {**body, "audit_id": content_sha256(body)}


def verify_audit(report_path: Path, *, plan_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    body = dict(report)
    audit_id = str(body.pop("audit_id", ""))
    if audit_id != content_sha256(body):
        raise ValueError("cluster-routing gap audit content address differs")
    observed = audit_training_plan(plan_path)
    if report != observed:
        raise ValueError("cluster-routing gap audit is not reproducible")
    return {
        "schema": "sch.vpd-compartment-sae-cluster-routing-gap-verification.v1",
        "status": "pass",
        "audit_id": audit_id,
        "finding": report["finding"],
        "training_jobs": report["observed"]["training_jobs"],
        "cluster_bound_training_jobs": report["observed"][
            "cluster_bound_training_jobs"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether an SAE training plan binds jobs to VPD clusters."
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("artifacts/vpd_compartment_sae_training_plan_v4/plan.json"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        result = audit_training_plan(args.plan)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        result = verify_audit(args.verify, plan_path=args.plan)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
