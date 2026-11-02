from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.activation_categories import CATEGORY_PROTOCOLS
from experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_runtime_v3 import (
    _allocation,
    _guard,
)
from experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_plan_v1 import (
    _split as _fixture_split,
)
from experiments.authz_mechinterp.benchmark_canonical_vpd_sae_training_capture_plan_v2 import (
    _authorization,
    _rewrite_split_role,
)
from experiments.authz_mechinterp.benchmark_vpd_three_role_capture_to_sae_integration_v1 import (
    _build_assets,
    _rows,
    _runtime_factory,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.load_vpd_capture_router_inputs_v2 import (
    load_capture_router_inputs_v2,
    verify_capture_router_adapter_receipt_v2,
)
from experiments.authz_mechinterp.materialize_vpd_cluster_aware_sae_plan_v1 import (
    ROLES,
    build_materialization,
    build_routing_evidence_bundle,
    verify_materialization,
)
from experiments.authz_mechinterp.plan_canonical_vpd_capture_job_v2 import (
    build_capture_job_plan,
)
from experiments.authz_mechinterp.route_vpd_cluster_episodes_v1 import (
    assess_routing_stability_v1,
    route_cluster_episodes_v1,
)
from experiments.authz_mechinterp.run_canonical_vpd_capture_job_v3 import (
    run_capture_job,
)
def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (KeyError, RuntimeError, TypeError, ValueError):
        return True
    return False


def _split(root: Path, role: str) -> tuple[Path, Path]:
    manifest_path, dataset_path = _fixture_split(
        root, _rows(role), name=f"split-{role}"
    )
    if role != "discovery":
        _rewrite_split_role(manifest_path, role=role)
    return manifest_path, dataset_path


def _capture_role_v3(
    root: Path, assets: dict[str, Any], role: str
) -> dict[str, Any]:
    split_path, dataset_path = _split(root, role)
    plan = build_capture_job_plan(
        project_root=root,
        model_identity_path=root / "model_identity.json",
        adapter_weights_path=assets["weights"],
        adapter_config_path=assets["config"],
        extraction_manifest_path=assets["factor_root"] / "manifest.json",
        inventory_manifest_path=assets["inventory_root"] / "manifest.json",
        split_manifest_path=split_path,
        split_role=role,
        dataset_path=dataset_path,
        component_ids=assets["component_ids"],
        adapter_name="default",
        activation_key="layer_35__proposal_start",
        epsilon=0.05,
        batch_size=5,
        device="synthetic-cpu",
        dtype="float32",
        output_dir=f"captures/materialization-{role}",
        synthetic_benchmark_only=True,
    )
    plan_path = root / f"capture-plan-{role}.json"
    authorization_path = root / f"capture-authorization-{role}.json"
    write_json(plan_path, plan)
    authorization = _authorization(plan, plan_path)
    authorization["scope"]["authorized_split_role"] = role
    authorization = _readdress(authorization, "capture_authorization_id")
    write_json(authorization_path, authorization)
    allocation = _allocation(plan, payload_bytes=1_000_000)
    guard = _guard(plan, allocation)
    factory, state = _runtime_factory(assets["inventory_rows"])
    execution = run_capture_job(
        plan_path=plan_path,
        authorization_path=authorization_path,
        project_root=root,
        campaign_guard=guard,
        resource_observer=lambda: (1.0, 0.01),
        runtime_factory=factory,
        synthetic_job_allocation=allocation,
    )
    output = root / plan["capture_contract"]["output_dir"]
    return {
        "plan": plan,
        "plan_path": plan_path,
        "authorization_path": authorization_path,
        "receipt_path": output / "capture_execution_receipt.json",
        "execution": execution,
        "runtime_state": state,
    }


def _readdress(record: dict[str, Any], identifier: str) -> dict[str, Any]:
    body = {key: value for key, value in record.items() if key != identifier}
    return {**body, identifier: content_sha256(body)}


def build_report(project_root: Path) -> dict[str, Any]:
    before_frameworks = MODEL_FRAMEWORKS & set(sys.modules)
    predecessor_source = (
        project_root / "artifacts/vpd_compartment_sae_training_plan_v4/plan.json"
    )
    with tempfile.TemporaryDirectory(prefix="vpd-sae-materialization-") as temporary:
        root = Path(temporary)
        predecessor_path = root / "predecessor/plan.json"
        predecessor_path.parent.mkdir(parents=True)
        predecessor_path.write_bytes(predecessor_source.read_bytes())
        predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
        assets = _build_assets(root)
        captures = {
            role: _capture_role_v3(root, assets, role) for role in ROLES
        }
        evidence_root = root / "evidence"
        category_files: dict[str, dict[str, Any]] = {}
        adapter_ids: set[str] = set()
        routing_ids: set[str] = set()
        admission_id = content_sha256(
            {"fixture": "guarded-v3-eight-category-materialization"}
        )
        for category in sorted(CATEGORY_PROTOCOLS):
            category_root = evidence_root / category
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
                adapter_check = verify_capture_router_adapter_receipt_v2(adapter)
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
                adapter_ids.add(adapter_check["capture_router_adapter_v2_id"])
                routing_ids.add(route["cluster_episode_routing_id"])
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
        materialization_path = root / "materialization.json"
        write_json(bundle_path, bundle)
        materialization = build_materialization(
            bundle_path=bundle_path,
            predecessor_training_plan_path=predecessor_path,
            project_root=root,
        )
        write_json(materialization_path, materialization)
        verification = verify_materialization(
            materialization_path,
            bundle_path=bundle_path,
            predecessor_training_plan_path=predecessor_path,
            project_root=root,
        )
        plan = materialization["cluster_aware_sae_plan"]
        specific = [
            row for row in plan["compartments"] if row["kind"] == "specific_cluster"
        ]
        shared = [
            row for row in plan["compartments"] if row["kind"] == "shared_fallback"
        ]

        original_bundle = bundle_path.read_bytes()
        missing = copy.deepcopy(bundle)
        missing_category = sorted(CATEGORY_PROTOCOLS)[0]
        del missing["categories"][missing_category]
        missing = _readdress(missing, "routing_evidence_bundle_id")
        write_json(bundle_path, missing)
        missing_category_rejected = _rejects(
            lambda: build_materialization(
                bundle_path=bundle_path,
                predecessor_training_plan_path=predecessor_path,
                project_root=root,
            )
        )
        bundle_path.write_bytes(original_bundle)

        escalated = copy.deepcopy(bundle)
        escalated["evidence_kind"] = "empirical_guarded_v3_capture"
        escalated["boundary"]["persistent_empirical_evidence"] = True
        escalated = _readdress(escalated, "routing_evidence_bundle_id")
        write_json(bundle_path, escalated)
        synthetic_as_empirical_rejected = _rejects(
            lambda: build_materialization(
                bundle_path=bundle_path,
                predecessor_training_plan_path=predecessor_path,
                project_root=root,
            )
        )
        bundle_path.write_bytes(original_bundle)

        substituted = copy.deepcopy(bundle)
        first = sorted(CATEGORY_PROTOCOLS)[0]
        substituted["categories"][first]["roles"]["discovery"]["adapter"] = copy.deepcopy(
            substituted["categories"][first]["roles"]["cluster_validation"][
                "adapter"
            ]
        )
        substituted = _readdress(substituted, "routing_evidence_bundle_id")
        write_json(bundle_path, substituted)
        role_substitution_rejected = _rejects(
            lambda: build_materialization(
                bundle_path=bundle_path,
                predecessor_training_plan_path=predecessor_path,
                project_root=root,
            )
        )
        bundle_path.write_bytes(original_bundle)

        route_path = Path(
            category_files[first]["roles"]["discovery"]["routing_path"]
        )
        route_bytes = route_path.read_bytes()
        changed_route = json.loads(route_bytes.decode("utf-8"))
        changed_route["source"]["capture_execution_id"] = "f" * 64
        write_json(route_path, changed_route)
        routing_tamper_rejected = _rejects(
            lambda: build_materialization(
                bundle_path=bundle_path,
                predecessor_training_plan_path=predecessor_path,
                project_root=root,
            )
        )
        route_path.write_bytes(route_bytes)

        predecessor_bytes = predecessor_path.read_bytes()
        changed_predecessor = json.loads(predecessor_bytes.decode("utf-8"))
        changed_predecessor["profile"]["steps"] = 1
        write_json(predecessor_path, changed_predecessor)
        predecessor_tamper_rejected = _rejects(
            lambda: verify_materialization(
                materialization_path,
                bundle_path=bundle_path,
                predecessor_training_plan_path=predecessor_path,
                project_root=root,
            )
        )
        predecessor_path.write_bytes(predecessor_bytes)

        materialization_bytes = materialization_path.read_bytes()
        changed_materialization = json.loads(materialization_bytes.decode("utf-8"))
        changed_materialization["boundary"]["sae_training_authorized"] = True
        write_json(materialization_path, changed_materialization)
        materialization_tamper_rejected = _rejects(
            lambda: verify_materialization(
                materialization_path,
                bundle_path=bundle_path,
                predecessor_training_plan_path=predecessor_path,
                project_root=root,
            )
        )
        materialization_path.write_bytes(materialization_bytes)
        after_frameworks = MODEL_FRAMEWORKS & set(sys.modules)

        controls = {
            "materialization_verifies": verification["status"] == "pass",
            "exact_eight_categories_bound": verification["categories"] == 8,
            "twenty_four_adapter_receipts_bound": len(adapter_ids) == 24
            and materialization["bundle_verification"]["adapter_receipts"] == 24,
            "twenty_four_router_receipts_bound": len(routing_ids) == 24
            and materialization["bundle_verification"]["routing_receipts"] == 24,
            "three_guarded_v3_role_captures_bound": len(
                materialization["bundle_verification"]["v3_capture_ids_by_role"]
            )
            == 3,
            "sixteen_specific_clusters_and_shared_fallback": len(specific) == 16
            and len(shared) == 1,
            "two_seed_jobs_per_compartment": verification["training_jobs"] == 34,
            "all_eighty_training_episodes_routed_once": plan["summary"][
                "routed_training_episodes"
            ]
            == 80
            and plan["summary"]["primary_compartment_episode_sets_disjoint"]
            is True,
            "shared_pair_safe_global_split": verification["shared_pair_safe"] is True
            and plan["summary"]["global_pair_split_consistent"] is True,
            "v3_identity_is_router_source_for_every_role_category": all(
                json.loads(
                    Path(category_files[category]["roles"][role]["routing_path"])
                    .read_text(encoding="utf-8")
                )["source"]["capture_execution_id"]
                == captures[role]["execution"]["capture_execution_v3_id"]
                for category in sorted(CATEGORY_PROTOCOLS)
                for role in ROLES
            ),
            "missing_category_rejected": missing_category_rejected,
            "synthetic_evidence_as_empirical_rejected": synthetic_as_empirical_rejected,
            "role_adapter_substitution_rejected": role_substitution_rejected,
            "routing_file_tamper_rejected": routing_tamper_rejected,
            "predecessor_plan_tamper_rejected": predecessor_tamper_rejected,
            "materialization_tamper_rejected": materialization_tamper_rejected,
            "manifest_only_no_training_authority": materialization["boundary"][
                "manifest_only"
            ]
            is True
            and materialization["boundary"]["empirical_plan"] is False
            and materialization["boundary"]["sae_training_authorized"] is False,
            "model_frameworks_absent_before_and_after": not before_frameworks
            and not after_frameworks,
        }
        body = {
            "schema": "sch.vpd-cluster-aware-sae-plan-materialization-benchmark.v1",
            "status": "pass" if all(controls.values()) else "fail",
            "source": {
                "materializer_sha256": sha256_file(
                    Path(__file__).with_name(
                        "materialize_vpd_cluster_aware_sae_plan_v1.py"
                    )
                ),
                "benchmark_sha256": sha256_file(Path(__file__)),
                "capture_router_adapter_v2_sha256": sha256_file(
                    Path(__file__).with_name("load_vpd_capture_router_inputs_v2.py")
                ),
                "planner_v2_sha256": sha256_file(
                    Path(__file__).with_name("plan_cluster_aware_sae_training_v2.py")
                ),
                "predecessor_plan_sha256": sha256_file(predecessor_source),
            },
            "design": {
                "categories": sorted(CATEGORY_PROTOCOLS),
                "roles": list(ROLES),
                "guarded_v3_capture_receipts": 3,
                "adapter_receipts": 24,
                "routing_receipts": 24,
                "independent_stability_receipts": 8,
                "synthetic_fixture_only": True,
                "manifest_only": True,
            },
            "results": {
                "routing_evidence_bundle_id": bundle[
                    "routing_evidence_bundle_id"
                ],
                "materialization_id": materialization["materialization_id"],
                "cluster_aware_sae_plan_id": plan["cluster_aware_sae_plan_id"],
                "global_pair_split_id": plan["global_pair_split"][
                    "global_pair_split_id"
                ],
                "verification": verification,
                "summary": plan["summary"],
            },
            "controls": controls,
            "summary": {
                "controls": len(controls),
                "passed": sum(bool(value) for value in controls.values()),
            },
            "scientific_boundary": {
                "synthetic_guarded_v3_captures_only": True,
                "persistent_empirical_bundle_produced": False,
                "empirical_plan_produced": False,
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
        raise ValueError("cluster-aware SAE materialization benchmark address differs")
    if observed != build_report(project_root):
        raise ValueError("cluster-aware SAE materialization benchmark is not reproducible")
    if observed.get("status") != "pass":
        raise ValueError("cluster-aware SAE materialization benchmark did not pass")
    return {
        "schema": "sch.vpd-cluster-aware-sae-plan-materialization-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "controls": observed["summary"]["controls"],
        "categories": 8,
        "roles": 3,
        "cluster_bound_training_jobs": observed["results"]["verification"][
            "training_jobs"
        ],
        "empirical_plan_produced": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark guarded cluster-aware SAE plan materialization V1."
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
