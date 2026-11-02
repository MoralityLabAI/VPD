from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.activation_categories import CATEGORY_PROTOCOLS
from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_materialization_v1 import (
    _capture_role_v3,
)
from experiments.authz_mechinterp.benchmark_vpd_three_role_capture_to_sae_integration_v1 import (
    _build_assets,
)
from experiments.authz_mechinterp.build_cluster_aware_sae_execution_blueprint_v1 import (
    build_execution_blueprint,
    verify_execution_blueprint,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.materialize_vpd_cluster_aware_sae_plan_v1 import (
    ROLES,
    build_materialization,
    build_routing_evidence_bundle,
)
from experiments.authz_mechinterp.load_vpd_capture_router_inputs_v2 import (
    load_capture_router_inputs_v2,
)
from experiments.authz_mechinterp.route_vpd_cluster_episodes_v1 import (
    assess_routing_stability_v1,
    route_cluster_episodes_v1,
)


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (KeyError, TypeError, ValueError):
        return True
    return False


def _synthetic_materialization(project_root: Path) -> dict[str, Any]:
    predecessor_source = (
        project_root / "artifacts/vpd_compartment_sae_training_plan_v4/plan.json"
    )
    with tempfile.TemporaryDirectory(prefix="vpd-blueprint-v1-") as temporary:
        root = Path(temporary)
        predecessor_path = root / "predecessor/plan.json"
        predecessor_path.parent.mkdir(parents=True)
        predecessor_path.write_bytes(predecessor_source.read_bytes())
        predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
        assets = _build_assets(root)
        captures = {
            role: _capture_role_v3(root, assets, role) for role in ROLES
        }
        admission_id = content_sha256(
            {"fixture": "cluster-aware-execution-blueprint-v1"}
        )
        category_files: dict[str, dict[str, Any]] = {}
        for category in sorted(CATEGORY_PROTOCOLS):
            category_root = root / "evidence" / category
            roles: dict[str, Any] = {}
            routes: dict[str, dict[str, Any]] = {}
            for role in ROLES:
                capture = captures[role]
                routing_inputs, adapter = load_capture_router_inputs_v2(
                    v3_receipt_path=capture["receipt_path"],
                    plan_path=capture["plan_path"],
                    authorization_path=capture["authorization_path"],
                    project_root=root,
                    activation_category=category,
                    partition_identity_admission_id=admission_id,
                    partition_admission_status="pass_candidate_partition_identity_v4",
                    category_partition_status="pass_exact_partition_identity",
                    category_partition_sha256=assets["partition_sha256"],
                    component_membership=assets["membership"],
                    minimum_response_rms=0.0,
                    minimum_margin_absolute=0.0,
                    minimum_margin_ratio=0.5,
                )
                route = route_cluster_episodes_v1(**routing_inputs)
                role_root = category_root / role
                adapter_path = role_root / "adapter.json"
                routing_path = role_root / "routing.json"
                write_json(adapter_path, adapter)
                write_json(routing_path, route)
                roles[role] = {
                    "adapter_path": adapter_path,
                    "routing_path": routing_path,
                }
                routes[role] = route
            stability = assess_routing_stability_v1(
                routes["discovery"],
                routes["cluster_validation"],
                maximum_total_variation=0.0,
                maximum_condition_total_variation=0.0,
                maximum_shared_rate_difference=0.0,
                minimum_condition_support_per_role=5,
            )
            stability_path = category_root / "stability.json"
            write_json(stability_path, stability)
            category_files[category] = {
                "roles": roles,
                "stability_path": stability_path,
            }
        policy = {
            "initialization_seeds": [17, 29],
            "split_seed": 1707,
            "validation_fraction": 0.25,
            "minimum_specific_episodes": 4,
            "minimum_specific_pairs": 4,
            "minimum_shared_episodes": 8,
            "minimum_shared_pairs": 2,
            "specific_dictionary_features": 4096,
            "shared_dictionary_features": 8192,
            "sae_activation_rows_per_episode": int(
                predecessor["profile"]["activation_rows_per_episode"]
            ),
        }
        bundle = build_routing_evidence_bundle(
            category_files=category_files,
            evidence_kind="synthetic_benchmark",
            planner_policy=policy,
            project_root=root,
        )
        bundle_path = root / "routing-evidence-bundle.json"
        write_json(bundle_path, bundle)
        return build_materialization(
            bundle_path=bundle_path,
            predecessor_training_plan_path=predecessor_path,
            project_root=root,
        )


def build_report(project_root: Path) -> dict[str, Any]:
    materialization = _synthetic_materialization(project_root)
    blueprint = build_execution_blueprint(materialization=materialization)
    verification = verify_execution_blueprint(
        blueprint, materialization=materialization
    )
    plan = materialization["cluster_aware_sae_plan"]
    jobs = blueprint["execution_jobs"]
    specific = [job for job in jobs if job["compartment_kind"] == "specific_cluster"]
    shared = [job for job in jobs if job["compartment_kind"] == "shared_fallback"]
    global_id = blueprint["source"]["global_pair_split_id"]

    changed_blueprint = copy.deepcopy(blueprint)
    changed_blueprint["boundary"]["sae_training_authorized"] = True
    blueprint_tamper_rejected = _rejects(
        lambda: verify_execution_blueprint(
            changed_blueprint, materialization=materialization
        )
    )
    changed_materialization = copy.deepcopy(materialization)
    changed_materialization["cluster_aware_sae_plan"]["training_jobs"][0][
        "compartment_contract_id"
    ] = "f" * 64
    changed_body = {
        key: value
        for key, value in changed_materialization.items()
        if key != "materialization_id"
    }
    changed_materialization["materialization_id"] = content_sha256(changed_body)
    materialization_job_tamper_rejected = _rejects(
        lambda: build_execution_blueprint(materialization=changed_materialization)
    )
    changed_job = copy.deepcopy(blueprint)
    changed_job["execution_jobs"][0]["exact_sample_contract"][
        "episode_ids"
    ] = []
    changed_job_body = {
        key: value for key, value in changed_job.items() if key != "execution_blueprint_id"
    }
    changed_job["execution_blueprint_id"] = content_sha256(changed_job_body)
    job_sample_tamper_rejected = _rejects(
        lambda: verify_execution_blueprint(
            changed_job, materialization=materialization
        )
    )

    contract_by_id = {
        row["compartment_id"]: row for row in plan["compartments"]
    }
    controls = {
        "blueprint_verifies": verification["status"] == "pass",
        "every_materialized_job_has_execution_contract": len(jobs)
        == int(plan["summary"]["training_jobs"]),
        "two_seed_jobs_per_compartment": len(jobs)
        == 2 * len(plan["compartments"]),
        "execution_jobs_bind_source_job_and_compartment": all(
            len(job["source_training_job_id"]) == 64
            and job["compartment_contract_id"]
            == contract_by_id[job["compartment_id"]]["compartment_contract_id"]
            for job in jobs
        ),
        "specific_jobs_bind_component_membership": all(
            job["cluster_id"] is not None
            and len(str(job["component_membership_sha256"])) == 64
            for job in specific
        ),
        "shared_jobs_have_no_fake_cluster_identity": all(
            job["cluster_id"] is None
            and job["component_membership_sha256"] is None
            for job in shared
        ),
        "exact_episode_and_pair_lists_bound": all(
            contract["exact_sample_contract"]["episode_ids"]
            and contract["exact_sample_contract"]["pair_ids"]
            and content_sha256(contract["exact_sample_contract"]["episode_ids"])
            == contract["exact_sample_contract"]["episode_id_set_sha256"]
            and content_sha256(contract["exact_sample_contract"]["pair_ids"])
            == contract["exact_sample_contract"]["pair_id_set_sha256"]
            for contract in jobs
        ),
        "one_global_pair_split_for_all_jobs": all(
            job["global_pair_split_id"] == global_id
            and job["exact_sample_contract"][
                "cross_compartment_pair_assignment_globally_consistent"
            ]
            is True
            for job in jobs
        ),
        "training_and_validation_pair_lists_are_disjoint": all(
            not (
                set(job["exact_sample_contract"]["training_pair_ids"])
                & set(job["exact_sample_contract"]["validation_pair_ids"])
            )
            for job in jobs
        ),
        "routing_and_stability_ids_bound": all(
            job["routing_receipt_ids"] and job["routing_stability_ids"]
            for job in jobs
        ),
        "output_directories_are_unique_and_relative": len(
            {job["output_dir"] for job in jobs}
        )
        == len(jobs)
        and all(not Path(job["output_dir"]).is_absolute() for job in jobs),
        "authorization_placeholders_are_explicit": all(
            "CLUSTER_AWARE_SAE_RELEASE_RECEIPT" in job["runtime_placeholders"]
            and "CLUSTER_AWARE_SAE_COMPUTE_AUTHORIZATION_RECEIPT"
            in job["runtime_placeholders"]
            for job in jobs
        ),
        "runtime_worker_absence_is_explicit": blueprint["boundary"][
            "runtime_worker_implemented"
        ]
        is False
        and all(job["boundary"]["runtime_worker_implemented"] is False for job in jobs),
        "blueprint_tamper_rejected": blueprint_tamper_rejected,
        "materialization_job_tamper_rejected": materialization_job_tamper_rejected,
        "job_sample_tamper_rejected": job_sample_tamper_rejected,
        "manifest_only_no_training_authority": blueprint["boundary"]["manifest_only"]
        is True
        and blueprint["boundary"]["sae_training_authorized"] is False
        and blueprint["boundary"]["execution_authorized"] is False,
    }
    body = {
        "schema": "sch.vpd-cluster-aware-sae-execution-blueprint-benchmark.v1",
        "status": "pass" if all(controls.values()) else "fail",
        "source": {
            "blueprint_builder_sha256": sha256_file(
                Path(__file__).with_name(
                    "build_cluster_aware_sae_execution_blueprint_v1.py"
                )
            ),
            "benchmark_sha256": sha256_file(Path(__file__)),
            "planner_v2_sha256": sha256_file(
                Path(__file__).with_name("plan_cluster_aware_sae_training_v2.py")
            ),
        },
        "design": {
            "synthetic_materialization_only": True,
            "exact_cluster_routed_episode_sets": True,
            "one_global_pair_assignment": True,
            "two_initialization_seeds": True,
            "runtime_worker_implemented": False,
            "manifest_only": True,
        },
        "results": {
            "materialization_id": materialization["materialization_id"],
            "cluster_aware_sae_plan_id": plan["cluster_aware_sae_plan_id"],
            "execution_blueprint_id": blueprint["execution_blueprint_id"],
            "global_pair_split_id": global_id,
            "verification": verification,
            "summary": blueprint["summary"],
        },
        "controls": controls,
        "summary": {
            "controls": len(controls),
            "passed": sum(bool(value) for value in controls.values()),
        },
        "scientific_boundary": {
            "synthetic_materialization_only": True,
            "runtime_worker_implemented": False,
            "empirical_blueprint_produced": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "sae_training_authorized": False,
            "execution_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("cluster-aware SAE execution blueprint benchmark address differs")
    if observed != build_report(project_root):
        raise ValueError("cluster-aware SAE execution blueprint benchmark is not reproducible")
    if observed.get("status") != "pass":
        raise ValueError("cluster-aware SAE execution blueprint benchmark did not pass")
    return {
        "schema": "sch.vpd-cluster-aware-sae-execution-blueprint-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "controls": observed["summary"]["controls"],
        "execution_jobs": observed["results"]["verification"]["execution_jobs"],
        "runtime_worker_implemented": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark cluster-aware SAE execution blueprint V1."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.output is not None:
        result = build_report(project_root)
        write_json(args.output, result)
    else:
        result = verify_benchmark(args.verify, project_root=project_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
