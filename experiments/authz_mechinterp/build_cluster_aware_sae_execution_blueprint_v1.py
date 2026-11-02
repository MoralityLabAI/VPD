from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    write_json,
)
from experiments.authz_mechinterp.materialize_vpd_cluster_aware_sae_plan_v1 import (
    MATERIALIZATION_SCHEMA,
)
from experiments.authz_mechinterp.plan_cluster_aware_sae_training_v2 import (
    verify_cluster_aware_sae_plan_v2,
)


SCHEMA = "sch.vpd-cluster-aware-sae-execution-blueprint.v1"
JOB_SCHEMA = "sch.vpd-cluster-aware-sae-execution-job.v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("cluster-aware SAE materialization must be an object")
    return value


def _require_content_id(value: Any, *, field: str) -> str:
    observed = str(value)
    if len(observed) != 64 or any(char not in "0123456789abcdef" for char in observed):
        raise ValueError(f"{field} must be a lowercase SHA-256 content ID")
    return observed


def validate_materialization_content(materialization: dict[str, Any]) -> dict[str, Any]:
    body = dict(materialization)
    identifier = str(body.pop("materialization_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("cluster-aware SAE materialization address differs")
    if materialization.get("schema") != MATERIALIZATION_SCHEMA or materialization.get(
        "status"
    ) != "complete_manifest_only_guarded_cluster_aware_sae_plan":
        raise ValueError("cluster-aware SAE materialization contract differs")
    plan = materialization.get("cluster_aware_sae_plan")
    if not isinstance(plan, dict):
        raise ValueError("cluster-aware SAE materialized plan is absent")
    plan_check = verify_cluster_aware_sae_plan_v2(plan)
    bundle = materialization.get("bundle_verification", {})
    boundary = materialization.get("boundary", {})
    evidence_kind = str(bundle.get("evidence_kind", ""))
    if (
        int(bundle.get("categories", -1)) != int(plan["summary"]["categories"])
        or int(bundle.get("roles_per_category", -1)) != 3
        or evidence_kind not in {"synthetic_benchmark", "empirical_guarded_v3_capture"}
        or boundary.get("manifest_only") is not True
        or boundary.get("evidence_kind") != evidence_kind
        or boundary.get("empirical_plan")
        != (evidence_kind == "empirical_guarded_v3_capture")
        or any(
            boundary.get(field) is not False
            for field in (
                "model_loaded",
                "model_inference_performed",
                "weights_loaded_or_mutated",
                "sae_training_authorized",
                "execution_authorized",
            )
        )
    ):
        raise ValueError("cluster-aware SAE materialization boundary differs")
    return {
        "schema": "sch.vpd-cluster-aware-sae-materialization-content-verification.v1",
        "status": "pass",
        "materialization_id": identifier,
        "cluster_aware_sae_plan_id": plan_check["cluster_aware_sae_plan_id"],
        "evidence_kind": evidence_kind,
        "categories": int(bundle["categories"]),
        "compartments": plan_check["compartments"],
        "training_jobs": plan_check["training_jobs"],
        "shared_pair_safe": plan_check["shared_pair_safe"],
        "sae_training_authorized": False,
    }


def build_execution_blueprint(
    *, materialization: dict[str, Any]
) -> dict[str, Any]:
    materialization_check = validate_materialization_content(materialization)
    plan = materialization["cluster_aware_sae_plan"]
    global_split = plan["global_pair_split"]
    global_split_id = _require_content_id(
        global_split["global_pair_split_id"], field="global pair split ID"
    )
    contracts = {
        str(row["compartment_id"]): row for row in plan["compartments"]
    }
    if len(contracts) != len(plan["compartments"]):
        raise ValueError("cluster-aware SAE compartment identity repeats")
    execution_jobs: list[dict[str, Any]] = []
    output_dirs: set[str] = set()
    for source_job in sorted(
        plan["training_jobs"],
        key=lambda row: (str(row["compartment_id"]), int(row["initialization_seed"])),
    ):
        compartment_id = str(source_job["compartment_id"])
        contract = contracts.get(compartment_id)
        if contract is None or source_job["compartment_contract_id"] != contract[
            "compartment_contract_id"
        ]:
            raise ValueError("cluster-aware SAE job compartment binding differs")
        support = contract["support"]
        split = contract["pair_atomic_split"]
        if (
            source_job["required_sample_contract"]["global_pair_split_id"]
            != global_split_id
            or split["global_pair_split_id"] != global_split_id
            or source_job["required_sample_contract"]["episode_id_set_sha256"]
            != support["episode_id_set_sha256"]
            or source_job["required_sample_contract"]["pair_id_set_sha256"]
            != support["pair_id_set_sha256"]
            or not support["episode_ids"]
            or not support["pair_ids"]
            or not contract["routing_receipt_ids"]
            or not contract["routing_stability_ids"]
        ):
            raise ValueError("cluster-aware SAE job sample lineage differs")
        for value in contract["routing_receipt_ids"] + contract[
            "routing_stability_ids"
        ]:
            _require_content_id(value, field="cluster routing evidence ID")
        seed = int(source_job["initialization_seed"])
        output_dir = f"dictionaries/cluster-aware/{compartment_id}/seed-{seed}"
        if output_dir in output_dirs:
            raise ValueError("cluster-aware SAE execution output repeats")
        output_dirs.add(output_dir)
        job_body = {
            "schema": JOB_SCHEMA,
            "source_training_job_id": source_job["job_id"],
            "compartment_id": compartment_id,
            "compartment_contract_id": contract["compartment_contract_id"],
            "compartment_kind": contract["kind"],
            "activation_categories": list(contract["activation_categories"]),
            "cluster_id": contract["cluster_id"],
            "component_membership_sha256": contract[
                "component_membership_sha256"
            ],
            "routing_receipt_ids": list(contract["routing_receipt_ids"]),
            "routing_stability_ids": list(contract["routing_stability_ids"]),
            "global_pair_split_id": global_split_id,
            "exact_sample_contract": {
                "selection": "exact_cluster_routed_episode_set",
                "episode_ids": list(support["episode_ids"]),
                "episode_id_set_sha256": support["episode_id_set_sha256"],
                "pair_ids": list(support["pair_ids"]),
                "pair_id_set_sha256": support["pair_id_set_sha256"],
                "training_pair_ids": list(split["training_pair_ids"]),
                "training_pair_id_set_sha256": split[
                    "training_pair_id_set_sha256"
                ],
                "validation_pair_ids": list(split["validation_pair_ids"]),
                "validation_pair_id_set_sha256": split[
                    "validation_pair_id_set_sha256"
                ],
                "training_episode_ids": list(split["training_episode_ids"]),
                "training_episode_id_set_sha256": split[
                    "training_episode_id_set_sha256"
                ],
                "validation_episode_ids": list(split["validation_episode_ids"]),
                "validation_episode_id_set_sha256": split[
                    "validation_episode_id_set_sha256"
                ],
                "sae_activation_rows": int(support["sae_activation_rows"]),
                "sae_activation_rows_per_episode": int(
                    support["sae_activation_rows_per_episode"]
                ),
                "cross_compartment_pair_assignment_globally_consistent": True,
            },
            "profile": dict(source_job["profile"]),
            "dictionary_features": int(source_job["dictionary_features"]),
            "initialization_seed": seed,
            "data_selection_seed": int(source_job["data_selection_seed"]),
            "output_dir": output_dir,
            "executable": ".venv/Scripts/python.exe",
            "arguments": [
                "-m",
                "experiments.authz_mechinterp.run_cluster_aware_sae_training_job_v1",
                "--execution-blueprint",
                "${CLUSTER_AWARE_SAE_EXECUTION_BLUEPRINT}",
                "--job-id",
                "${CLUSTER_AWARE_SAE_EXECUTION_JOB_ID}",
                "--scientific-release-receipt",
                "${CLUSTER_AWARE_SAE_RELEASE_RECEIPT}",
                "--compute-authorization-receipt",
                "${CLUSTER_AWARE_SAE_COMPUTE_AUTHORIZATION_RECEIPT}",
                "--device",
                "${SAE_TRAINING_DEVICE}",
            ],
            "runtime_placeholders": [
                "CLUSTER_AWARE_SAE_EXECUTION_BLUEPRINT",
                "CLUSTER_AWARE_SAE_EXECUTION_JOB_ID",
                "CLUSTER_AWARE_SAE_RELEASE_RECEIPT",
                "CLUSTER_AWARE_SAE_COMPUTE_AUTHORIZATION_RECEIPT",
                "SAE_TRAINING_DEVICE",
            ],
            "boundary": {
                "manifest_only": True,
                "runtime_worker_implemented": False,
                "scientific_release_receipt_present": False,
                "explicit_compute_authorization_present": False,
                "model_loaded": False,
                "weights_written": False,
                "sae_training_authorized": False,
                "execution_authorized": False,
            },
        }
        execution_jobs.append({**job_body, "execution_job_id": content_sha256(job_body)})
    if len(execution_jobs) != int(plan["summary"]["training_jobs"]):
        raise ValueError("cluster-aware SAE execution job total differs")
    body = {
        "schema": SCHEMA,
        "status": "complete_manifest_only_cluster_routed_execution_blueprint",
        "source": {
            "materialization_id": materialization_check["materialization_id"],
            "cluster_aware_sae_plan_id": materialization_check[
                "cluster_aware_sae_plan_id"
            ],
            "routing_evidence_bundle_id": materialization["source"][
                "routing_evidence_bundle_id"
            ],
            "evidence_kind": materialization_check["evidence_kind"],
            "global_pair_split_id": global_split_id,
        },
        "global_pair_split": dict(global_split),
        "execution_jobs": execution_jobs,
        "summary": {
            "categories": materialization_check["categories"],
            "compartments": len(contracts),
            "specific_compartments": int(plan["summary"]["specific_compartments"]),
            "shared_compartments": int(plan["summary"]["shared_compartments"]),
            "execution_jobs": len(execution_jobs),
            "unique_output_dirs": len(output_dirs),
            "one_global_pair_split": True,
            "exact_cluster_routed_episode_sets": True,
        },
        "boundary": {
            "manifest_only": True,
            "runtime_worker_implemented": False,
            "empirical_plan": materialization_check["evidence_kind"]
            == "empirical_guarded_v3_capture",
            "scientific_release_receipt_present": False,
            "explicit_compute_authorization_present": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "sae_training_authorized": False,
            "execution_authorized": False,
        },
    }
    return {**body, "execution_blueprint_id": content_sha256(body)}


def verify_execution_blueprint(
    blueprint: dict[str, Any], *, materialization: dict[str, Any]
) -> dict[str, Any]:
    body = dict(blueprint)
    identifier = str(body.pop("execution_blueprint_id", ""))
    if identifier != content_sha256(body) or blueprint != build_execution_blueprint(
        materialization=materialization
    ):
        raise ValueError("cluster-aware SAE execution blueprint is not reproducible")
    return {
        "schema": "sch.vpd-cluster-aware-sae-execution-blueprint-verification.v1",
        "status": "pass",
        "execution_blueprint_id": identifier,
        "materialization_id": blueprint["source"]["materialization_id"],
        "cluster_aware_sae_plan_id": blueprint["source"][
            "cluster_aware_sae_plan_id"
        ],
        "global_pair_split_id": blueprint["source"]["global_pair_split_id"],
        "categories": blueprint["summary"]["categories"],
        "compartments": blueprint["summary"]["compartments"],
        "execution_jobs": blueprint["summary"]["execution_jobs"],
        "runtime_worker_implemented": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify a cluster-aware SAE execution blueprint V1."
    )
    parser.add_argument("--materialization", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    materialization = _load(args.materialization)
    if args.output is not None:
        result = build_execution_blueprint(materialization=materialization)
        write_json(args.output, result)
    else:
        result = verify_execution_blueprint(
            _load(args.verify), materialization=materialization
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
