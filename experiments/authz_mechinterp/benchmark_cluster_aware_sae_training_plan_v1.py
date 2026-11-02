from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.benchmark_vpd_cluster_episode_routing_v1 import (
    _condition_fixture,
    _core_fixture,
    _make_inputs,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.plan_cluster_aware_sae_training_v1 import (
    SHARED_COMPARTMENT,
    build_cluster_aware_sae_plan_v1,
    verify_cluster_aware_sae_plan_v1,
)
from experiments.authz_mechinterp.route_vpd_cluster_episodes_v1 import (
    assess_routing_stability_v1,
    route_cluster_episodes_v1,
)


CATEGORIES = ["missing_or_untrusted_authority", "valid_authority"]


def _with_category(inputs: dict[str, Any], category: str) -> dict[str, Any]:
    changed = copy.deepcopy(inputs)
    changed["activation_category"] = category
    for row in changed["episodes"]:
        row["activation_category"] = category
    return changed


def _routing(inputs: dict[str, Any], category: str) -> dict[str, Any]:
    return route_cluster_episodes_v1(**_with_category(inputs, category))


def _category_evidence(
    category: str,
    *,
    prefix: str,
    sparse_training_cluster: bool,
    swapped_validation: bool = False,
    overlap_training: bool = False,
    training_margin_ratio: float | None = None,
) -> dict[str, Any]:
    discovery_inputs = _condition_fixture("discovery", f"{prefix}-d", swapped=False)
    validation_inputs = _condition_fixture(
        "cluster_validation", f"{prefix}-v", swapped=swapped_validation
    )
    if sparse_training_cluster:
        training_inputs = _make_inputs(
            role="sae_training",
            prefix=f"{prefix}-t",
            kinds_by_condition={
                "condition-a": ["a", "a", "a", "a", "b", "b"],
                "condition-b": ["ambiguous", "ambiguous"],
            },
        )
    else:
        training_inputs = _core_fixture("discovery", f"{prefix}-t")
        training_inputs["split_role"] = "sae_training"
        training_inputs["split_id"] = f"synthetic-sae-training-{prefix}"
    if overlap_training:
        training_inputs = _condition_fixture("discovery", f"{prefix}-d", swapped=False)
        training_inputs["split_role"] = "sae_training"
        training_inputs["split_id"] = f"synthetic-sae-training-overlap-{prefix}"
    if training_margin_ratio is not None:
        training_inputs["minimum_margin_ratio"] = training_margin_ratio
    discovery = _routing(discovery_inputs, category)
    validation = _routing(validation_inputs, category)
    training = _routing(training_inputs, category)
    stability = assess_routing_stability_v1(
        discovery,
        validation,
        maximum_total_variation=0.05,
        maximum_condition_total_variation=0.05,
        maximum_shared_rate_difference=0.05,
        minimum_condition_support_per_role=4,
    )
    return {
        "discovery": discovery,
        "cluster_validation": validation,
        "stability": stability,
        "sae_training": training,
    }


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except ValueError:
        return True
    return False


def _plan_kwargs(project_root: Path) -> dict[str, Any]:
    predecessor_path = (
        project_root / "artifacts/vpd_compartment_sae_training_plan_v4/plan.json"
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    evidence = {
        CATEGORIES[0]: _category_evidence(
            CATEGORIES[0], prefix="invalid", sparse_training_cluster=False
        ),
        CATEGORIES[1]: _category_evidence(
            CATEGORIES[1], prefix="valid", sparse_training_cluster=True
        ),
    }
    return {
        "category_evidence": evidence,
        "predecessor_training_plan_id": predecessor["training_plan_id"],
        "predecessor_training_plan_sha256": sha256_file(predecessor_path),
        "profile": predecessor["profile"],
        "initialization_seeds": [17, 29],
        "split_seed": 1707,
        "validation_fraction": 0.25,
        "minimum_specific_episodes": 4,
        "minimum_specific_pairs": 2,
        "minimum_shared_episodes": 4,
        "minimum_shared_pairs": 2,
        "specific_dictionary_features": 4096,
        "shared_dictionary_features": 8192,
        "sae_activation_rows_per_episode": int(
            predecessor["profile"]["activation_rows_per_episode"]
        ),
    }


def build_report(project_root: Path) -> dict[str, Any]:
    kwargs = _plan_kwargs(project_root)
    before_evidence = copy.deepcopy(kwargs["category_evidence"])
    before_profile = copy.deepcopy(kwargs["profile"])
    plan = build_cluster_aware_sae_plan_v1(**kwargs)
    verification = verify_cluster_aware_sae_plan_v1(plan)

    contracts = plan["compartments"]
    specific = [row for row in contracts if row["kind"] == "specific_cluster"]
    shared = [row for row in contracts if row["kind"] == "shared_fallback"]
    job_counts: dict[str, int] = {}
    for job in plan["training_jobs"]:
        key = str(job["compartment_id"])
        job_counts[key] = job_counts.get(key, 0) + 1
    total_contract_episodes = sum(row["support"]["episodes"] for row in contracts)
    training_episode_ids = {
        row["episode_id"]
        for evidence in kwargs["category_evidence"].values()
        for row in evidence["sae_training"]["routing"]["rows"]
    }
    under_supported = plan["routing_adjustments"]["under_supported_routes"]

    unstable_kwargs = copy.deepcopy(kwargs)
    unstable_kwargs["category_evidence"][CATEGORIES[0]] = _category_evidence(
        CATEGORIES[0],
        prefix="unstable",
        sparse_training_cluster=False,
        swapped_validation=True,
    )
    overlap_kwargs = copy.deepcopy(kwargs)
    overlap_kwargs["category_evidence"][CATEGORIES[0]] = _category_evidence(
        CATEGORIES[0],
        prefix="overlap",
        sparse_training_cluster=False,
        overlap_training=True,
    )
    policy_kwargs = copy.deepcopy(kwargs)
    policy_kwargs["category_evidence"][CATEGORIES[0]] = _category_evidence(
        CATEGORIES[0],
        prefix="policy",
        sparse_training_cluster=False,
        training_margin_ratio=0.3,
    )
    tampered_stability_kwargs = copy.deepcopy(kwargs)
    tampered_stability_kwargs["category_evidence"][CATEGORIES[0]]["stability"][
        "policy"
    ]["maximum_total_variation"] = 0.9

    contract_pair_hashes = [row["support"]["pair_id_set_sha256"] for row in contracts]
    controls = {
        "manifest_plan_verifies": verification["status"] == "pass",
        "two_categories_bound": plan["summary"]["categories"] == 2,
        "three_specific_routes_and_one_shared": len(specific) == 3 and len(shared) == 1,
        "two_initialization_jobs_per_compartment": all(
            value == 2 for value in job_counts.values()
        )
        and len(job_counts) == len(contracts),
        "under_supported_route_merged_to_shared": len(under_supported) == 1
        and under_supported[0]["episodes"] == 2
        and under_supported[0]["pairs"] == 1,
        "all_training_episodes_assigned_once": total_contract_episodes
        == len(training_episode_ids)
        == plan["summary"]["routed_training_episodes"],
        "compartment_pair_sets_disjoint": plan["summary"][
            "primary_compartment_pair_sets_disjoint"
        ]
        and len(contract_pair_hashes) == len(set(contract_pair_hashes)),
        "specific_membership_hashes_bound": all(
            row["component_members"] and row["component_membership_sha256"]
            for row in specific
        ),
        "shared_has_no_fake_component_membership": shared[0]["cluster_id"] is None
        and shared[0]["component_members"] == []
        and shared[0]["component_membership_sha256"] is None,
        "pair_atomic_train_validation_splits": all(
            row["pair_atomic_split"]["pair_sets_disjoint"]
            and row["pair_atomic_split"]["episode_sets_disjoint"]
            and row["pair_atomic_split"]["training_pairs"] >= 1
            and row["pair_atomic_split"]["validation_pairs"] >= 1
            for row in contracts
        ),
        "routing_and_sae_row_units_separated": all(
            row["support"]["routing_effect_rows_per_episode"] == 2
            and row["support"]["sae_activation_rows_per_episode"] == 128
            and row["support"]["sae_activation_rows"]
            == row["support"]["episodes"] * 128
            for row in contracts
        ),
        "unstable_routing_rejected": _rejects(
            lambda: build_cluster_aware_sae_plan_v1(**unstable_kwargs)
        ),
        "training_discovery_overlap_rejected": _rejects(
            lambda: build_cluster_aware_sae_plan_v1(**overlap_kwargs)
        ),
        "routing_policy_drift_rejected": _rejects(
            lambda: build_cluster_aware_sae_plan_v1(**policy_kwargs)
        ),
        "stability_receipt_tamper_rejected": _rejects(
            lambda: build_cluster_aware_sae_plan_v1(**tampered_stability_kwargs)
        ),
        "predecessor_and_evidence_not_mutated": before_evidence
        == kwargs["category_evidence"]
        and before_profile == kwargs["profile"],
        "no_model_or_weight_authority": plan["boundary"]["model_loaded"] is False
        and plan["boundary"]["weights_loaded_or_mutated"] is False
        and plan["boundary"]["sae_training_authorized"] is False,
    }
    predecessor_path = (
        project_root / "artifacts/vpd_compartment_sae_training_plan_v4/plan.json"
    )
    body = {
        "schema": "sch.vpd-cluster-aware-sae-training-plan-benchmark.v1",
        "status": "pass" if all(controls.values()) else "fail",
        "source": {
            "runner_sha256": sha256_file(Path(__file__)),
            "planner_sha256": sha256_file(
                Path(__file__).with_name("plan_cluster_aware_sae_training_v1.py")
            ),
            "router_sha256": sha256_file(
                Path(__file__).with_name("route_vpd_cluster_episodes_v1.py")
            ),
            "predecessor_plan_path": predecessor_path.relative_to(
                project_root
            ).as_posix(),
            "predecessor_plan_sha256": sha256_file(predecessor_path),
        },
        "design": {
            "categories": CATEGORIES,
            "routing_roles": ["discovery", "cluster_validation", "sae_training"],
            "stable_routing_required_before_planning": True,
            "under_supported_routes_merge_to_shared": True,
            "pair_atomic_train_validation": True,
            "two_initialization_seeds": True,
            "manifest_only": True,
        },
        "results": {
            "cluster_aware_sae_plan_id": plan["cluster_aware_sae_plan_id"],
            "verification": verification,
            "summary": plan["summary"],
            "under_supported_routes": under_supported,
            "compartment_support": {
                row["compartment_id"]: row["support"] for row in contracts
            },
        },
        "controls": controls,
        "summary": {
            "controls": len(controls),
            "passed": sum(bool(value) for value in controls.values()),
        },
        "scientific_boundary": {
            "synthetic_routing_receipts_only": True,
            "model_loaded": False,
            "model_inference_performed": False,
            "weights_loaded_or_mutated": False,
            "empirical_plan_produced": False,
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
        raise ValueError("cluster-aware SAE plan benchmark address differs")
    if observed != build_report(project_root):
        raise ValueError("cluster-aware SAE plan benchmark is not reproducible")
    if observed.get("status") != "pass":
        raise ValueError("cluster-aware SAE plan benchmark did not pass")
    return {
        "schema": "sch.vpd-cluster-aware-sae-training-plan-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "controls": observed["summary"]["controls"],
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark manifest-only cluster-aware SAE planning V1."
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
