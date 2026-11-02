from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
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
from experiments.authz_mechinterp.build_cluster_aware_sae_blueprint_budget_admission_v2 import (
    build_from_paths,
)
from experiments.authz_mechinterp.build_cluster_aware_sae_execution_blueprint_v1 import (
    build_execution_blueprint,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.dispatch_cluster_aware_sae_semantic_slice_v1 import (
    ALLOWED_INPUT_PARENT,
    dispatch_semantic_slice,
)
from experiments.authz_mechinterp.load_vpd_capture_router_inputs_v2 import (
    load_capture_router_inputs_v2,
)
from experiments.authz_mechinterp.materialize_vpd_cluster_aware_sae_plan_v1 import (
    ROLES,
    build_materialization,
    build_routing_evidence_bundle,
)
from experiments.authz_mechinterp.route_vpd_cluster_episodes_v1 import (
    assess_routing_stability_v1,
    route_cluster_episodes_v1,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (KeyError, RuntimeError, TypeError, ValueError):
        return True
    return False


def _readdress(record: dict[str, Any], identifier: str) -> dict[str, Any]:
    changed = copy.deepcopy(record)
    body = {key: value for key, value in changed.items() if key != identifier}
    changed[identifier] = content_sha256(body)
    return changed


def _build_fixture(project_root: Path, root: Path) -> dict[str, str]:
    predecessor_source = (
        project_root / "artifacts/vpd_compartment_sae_training_plan_v4/plan.json"
    )
    predecessor_path = root / "predecessor/plan.json"
    predecessor_path.parent.mkdir(parents=True)
    predecessor_path.write_bytes(predecessor_source.read_bytes())
    predecessor = _load(predecessor_path)
    assets = _build_assets(root)
    captures = {role: _capture_role_v3(root, assets, role) for role in ROLES}
    category_files: dict[str, dict[str, Any]] = {}
    admission_id = content_sha256(
        {"fixture": "cluster-aware-execution-blueprint-v1"}
    )
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
    planner_policy = {
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
        planner_policy=planner_policy,
        project_root=root,
    )
    bundle_path = root / "routing-evidence-bundle.json"
    materialization_path = root / "materialization.json"
    blueprint_path = root / "execution-blueprint.json"
    policy_path = root / "budget-policy.json"
    proposal_path = root / "budget-proposal.json"
    admission_path = root / "budget-admission.json"
    write_json(bundle_path, bundle)
    materialization = build_materialization(
        bundle_path=bundle_path,
        predecessor_training_plan_path=predecessor_path,
        project_root=root,
    )
    write_json(materialization_path, materialization)
    write_json(blueprint_path, build_execution_blueprint(materialization=materialization))
    canonical_policy_path = (
        project_root
        / "artifacts/vpd_cluster_aware_sae_budget_admission_policy_v1/policy.json"
    )
    write_json(policy_path, _load(canonical_policy_path))
    proposal, admission = build_from_paths(
        policy_path=policy_path, blueprint_path=blueprint_path
    )
    write_json(proposal_path, proposal)
    write_json(admission_path, admission)
    return {
        "materialization": materialization_path.relative_to(root).as_posix(),
        "routing_bundle": bundle_path.relative_to(root).as_posix(),
        "predecessor_training_plan": predecessor_path.relative_to(root).as_posix(),
        "execution_blueprint": blueprint_path.relative_to(root).as_posix(),
        "budget_policy": policy_path.relative_to(root).as_posix(),
        "budget_proposal": proposal_path.relative_to(root).as_posix(),
        "budget_admission": admission_path.relative_to(root).as_posix(),
    }


def build_report(project_root: Path) -> dict[str, Any]:
    before_frameworks = MODEL_FRAMEWORKS & set(sys.modules)
    allowed_parent = project_root / ALLOWED_INPUT_PARENT
    allowed_parent.mkdir(parents=True, exist_ok=True)
    temporary_root: Path | None = None
    with tempfile.TemporaryDirectory(
        prefix="semantic-slice-v1-", dir=allowed_parent
    ) as value:
        root = Path(value)
        temporary_root = root
        files = _build_fixture(project_root, root)
        common = {
            "project_root": project_root,
            "artifact_root": root.relative_to(project_root).as_posix(),
            **files,
        }
        clean = dispatch_semantic_slice(
            **common, expected_graph_kind="synthetic_benchmark_fixture"
        )

        wrong_kind_rejected = _rejects(
            lambda: dispatch_semantic_slice(
                **common, expected_graph_kind="empirical_candidate"
            )
        )
        traversal = dict(common)
        traversal["materialization"] = "../materialization.json"
        path_traversal_rejected = _rejects(
            lambda: dispatch_semantic_slice(
                **traversal, expected_graph_kind="synthetic_benchmark_fixture"
            )
        )
        absolute = dict(common)
        absolute["materialization"] = str(
            (root / files["materialization"]).resolve()
        )
        absolute_path_rejected = _rejects(
            lambda: dispatch_semantic_slice(
                **absolute, expected_graph_kind="synthetic_benchmark_fixture"
            )
        )
        missing = dict(common)
        missing["budget_admission"] = "missing-admission.json"
        missing_file_rejected = _rejects(
            lambda: dispatch_semantic_slice(
                **missing, expected_graph_kind="synthetic_benchmark_fixture"
            )
        )

        materialization_path = root / files["materialization"]
        materialization_bytes = materialization_path.read_bytes()
        changed_materialization = _load(materialization_path)
        changed_materialization["boundary"]["sae_training_authorized"] = True
        write_json(materialization_path, changed_materialization)
        unaddressed_materialization_rejected = _rejects(
            lambda: dispatch_semantic_slice(
                **common, expected_graph_kind="synthetic_benchmark_fixture"
            )
        )
        readdressed_materialization = _readdress(
            changed_materialization, "materialization_id"
        )
        write_json(materialization_path, readdressed_materialization)
        readdressed_materialization_rejected = _rejects(
            lambda: dispatch_semantic_slice(
                **common, expected_graph_kind="synthetic_benchmark_fixture"
            )
        )
        materialization_path.write_bytes(materialization_bytes)

        predecessor_path = root / files["predecessor_training_plan"]
        substituted_predecessor = _load(predecessor_path)
        substituted_predecessor["profile"]["steps"] += 1
        substituted_predecessor = _readdress(
            substituted_predecessor, "training_plan_id"
        )
        substituted_predecessor_path = root / "substituted-predecessor.json"
        write_json(substituted_predecessor_path, substituted_predecessor)
        substituted = dict(common)
        substituted["predecessor_training_plan"] = (
            substituted_predecessor_path.relative_to(root).as_posix()
        )
        companion_substitution_rejected = _rejects(
            lambda: dispatch_semantic_slice(
                **substituted, expected_graph_kind="synthetic_benchmark_fixture"
            )
        )

        blueprint_path = root / files["execution_blueprint"]
        blueprint_bytes = blueprint_path.read_bytes()
        changed_blueprint = _load(blueprint_path)
        changed_job = copy.deepcopy(changed_blueprint["execution_jobs"][0])
        changed_job["output_dir"] = "changed-output"
        changed_job = _readdress(changed_job, "execution_job_id")
        changed_blueprint["execution_jobs"][0] = changed_job
        changed_blueprint = _readdress(
            changed_blueprint, "execution_blueprint_id"
        )
        write_json(blueprint_path, changed_blueprint)
        readdressed_blueprint_rejected = _rejects(
            lambda: dispatch_semantic_slice(
                **common, expected_graph_kind="synthetic_benchmark_fixture"
            )
        )
        blueprint_path.write_bytes(blueprint_bytes)
        blueprint_path.write_bytes(blueprint_bytes + b" ")
        blueprint_byte_substitution_rejected = _rejects(
            lambda: dispatch_semantic_slice(
                **common, expected_graph_kind="synthetic_benchmark_fixture"
            )
        )
        blueprint_path.write_bytes(blueprint_bytes)

        proposal_path = root / files["budget_proposal"]
        proposal_bytes = proposal_path.read_bytes()
        changed_proposal = _load(proposal_path)
        changed_proposal["authority_claims"]["sae_training_authorized"] = True
        changed_proposal = _readdress(changed_proposal, "proposal_id")
        write_json(proposal_path, changed_proposal)
        readdressed_proposal_rejected = _rejects(
            lambda: dispatch_semantic_slice(
                **common, expected_graph_kind="synthetic_benchmark_fixture"
            )
        )
        proposal_path.write_bytes(proposal_bytes)

        admission_path = root / files["budget_admission"]
        admission_bytes = admission_path.read_bytes()
        changed_admission = _load(admission_path)
        changed_admission["blueprint_binding_verified"] = False
        changed_admission = _readdress(changed_admission, "admission_id")
        write_json(admission_path, changed_admission)
        readdressed_admission_rejected = _rejects(
            lambda: dispatch_semantic_slice(
                **common, expected_graph_kind="synthetic_benchmark_fixture"
            )
        )
        admission_path.write_bytes(admission_bytes)

        policy_path = root / files["budget_policy"]
        substituted_policy = _load(policy_path)
        substituted_policy["global_ceiling"]["planned_steps"] += 1
        substituted_policy = _readdress(substituted_policy, "budget_policy_id")
        substituted_policy_path = root / "substituted-policy.json"
        write_json(substituted_policy_path, substituted_policy)
        policy_substitution = dict(common)
        policy_substitution["budget_policy"] = (
            substituted_policy_path.relative_to(root).as_posix()
        )
        policy_substitution_rejected = _rejects(
            lambda: dispatch_semantic_slice(
                **policy_substitution,
                expected_graph_kind="synthetic_benchmark_fixture",
            )
        )

        link_path = root / "linked-materialization.json"
        reparse_created = False
        junction_path: Path | None = None
        try:
            os.symlink(materialization_path, link_path, target_is_directory=False)
            reparse_created = True
        except (NotImplementedError, OSError):
            real_dir = root / "real-linked-source"
            real_dir.mkdir()
            linked_source = real_dir / "materialization.json"
            linked_source.write_bytes(materialization_bytes)
            junction_path = root / "linked-dir"
            result = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(junction_path), str(real_dir)],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                reparse_created = True
                link_path = junction_path / "materialization.json"
        if reparse_created:
            linked = dict(common)
            linked["materialization"] = link_path.relative_to(root).as_posix()
            reparse_rejected = _rejects(
                lambda: dispatch_semantic_slice(
                    **linked, expected_graph_kind="synthetic_benchmark_fixture"
                )
            )
        else:
            reparse_rejected = False
        if link_path.is_symlink():
            link_path.unlink()
        if junction_path is not None and junction_path.exists():
            os.rmdir(junction_path)

        after_frameworks = MODEL_FRAMEWORKS & set(sys.modules)

    controls = {
        "clean_slice_is_content_addressed": clean["semantic_dispatch_slice_id"]
        == content_sha256(
            {
                key: value
                for key, value in clean.items()
                if key != "semantic_dispatch_slice_id"
            }
        ),
        "four_real_semantic_verifiers_dispatch": clean["slice_summary"][
            "dispatched_nodes"
        ]
        == 4
        and all(
            row["verifier_status"] == "pass"
            for row in clean["dispatched_nodes"]
        ),
        "all_four_nodes_compare_full_reproduction": clean["slice_summary"][
            "full_reproduction_comparisons"
        ]
        == 4
        and all(
            row["full_reproduction_compared"] is True
            for row in clean["dispatched_nodes"]
        ),
        "exact_34_job_shape_dispatches": clean["slice_summary"]["execution_jobs"]
        == 34,
        "synthetic_kind_remains_explicit": clean["slice_summary"]["graph_kind"]
        == "synthetic_benchmark_fixture"
        and clean["scientific_boundary"]["empirical_graph_present"] is False,
        "three_terminal_gaps_are_returned": clean["terminal_gaps"]
        == [
            "budget_admission_to_signed_release",
            "budget_admission_to_runtime_admission",
            "production_runtime_attachment",
        ],
        "slice_transfers_no_authority": all(
            clean["scientific_boundary"][field] is False
            for field in (
                "end_to_end_semantic_dispatch_complete",
                "signed_authority_dispatched",
                "durable_transition_dispatched",
                "production_runtime_attachment_dispatched",
                "model_loaded",
                "model_inference_performed",
                "training_performed",
                "weights_loaded_or_mutated",
                "persistent_dictionary_output_created",
                "execution_authorized",
                "sae_training_authorized",
                "paid_compute_authorized",
            )
        ),
        "wrong_graph_kind_rejected": wrong_kind_rejected,
        "path_traversal_rejected": path_traversal_rejected,
        "absolute_path_rejected": absolute_path_rejected,
        "missing_file_rejected": missing_file_rejected,
        "symbolic_link_or_junction_rejected": reparse_rejected,
        "unaddressed_materialization_mutation_rejected": unaddressed_materialization_rejected,
        "readdressed_materialization_mutation_rejected_by_reproduction": readdressed_materialization_rejected,
        "readdressed_predecessor_companion_substitution_rejected": companion_substitution_rejected,
        "readdressed_blueprint_mutation_rejected_by_reproduction": readdressed_blueprint_rejected,
        "blueprint_byte_substitution_rejected_by_exact_hash_binding": blueprint_byte_substitution_rejected,
        "readdressed_proposal_authority_broadening_rejected": readdressed_proposal_rejected,
        "readdressed_admission_mutation_rejected_by_reproduction": readdressed_admission_rejected,
        "readdressed_policy_substitution_rejected": policy_substitution_rejected,
        "temporary_fixture_root_removed": bool(
            temporary_root is not None and not temporary_root.exists()
        ),
        "benchmark_imports_no_model_frameworks": not before_frameworks
        and not after_frameworks,
    }
    if not all(controls.values()):
        raise ValueError(
            "semantic dispatch slice benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-aware-sae-semantic-dispatch-slice-benchmark.v1",
        "status": "pass",
        "source": {
            "semantic_graph_contract_id": clean["source"][
                "semantic_graph_contract_id"
            ],
            "dispatcher_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/dispatch_cluster_aware_sae_semantic_slice_v1.py"
            ),
            "benchmark_builder_sha256": sha256_file(Path(__file__)),
        },
        "controls": controls,
        "results": {
            "semantic_dispatch_slice_id": clean["semantic_dispatch_slice_id"],
            "dispatched_node_ids": {
                row["node"]: row["id"] for row in clean["dispatched_nodes"]
            },
            "exact_blueprint_file_sha256": clean["verified_joins"][
                "exact_blueprint_file_sha256"
            ],
        },
        "summary": {
            "passed": sum(controls.values()),
            "controls": len(controls),
            "dispatched_nodes": 4,
            "full_reproduction_comparisons": 4,
            "execution_jobs": 34,
            "terminal_gap_count": 3,
            "synthetic_fixture_persisted": False,
        },
        "scientific_boundary": clean["scientific_boundary"],
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body) or observed != build_report(project_root):
        raise ValueError("semantic dispatch slice benchmark is not reproducible")
    return {
        "schema": "sch.vpd-cluster-aware-sae-semantic-dispatch-slice-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "semantic_dispatch_slice_id": observed["results"][
            "semantic_dispatch_slice_id"
        ],
        "controls": observed["summary"]["controls"],
        "dispatched_nodes": observed["summary"]["dispatched_nodes"],
        "full_reproduction_comparisons": observed["summary"][
            "full_reproduction_comparisons"
        ],
        "execution_jobs": observed["summary"]["execution_jobs"],
        "terminal_gap_count": observed["summary"]["terminal_gap_count"],
        "synthetic_fixture_persisted": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the model-free semantic dispatcher slice V1."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.output is not None:
        result = build_report(root)
        write_json(args.output, result)
    else:
        result = verify_benchmark(args.verify, project_root=root)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "benchmark_id": result.get("benchmark_id"),
                "semantic_dispatch_slice_id": result.get(
                    "semantic_dispatch_slice_id",
                    result.get("results", {}).get("semantic_dispatch_slice_id"),
                ),
                "controls": result.get(
                    "controls", result.get("summary", {}).get("controls")
                ),
                "execution_authorized": result.get(
                    "execution_authorized",
                    result.get("scientific_boundary", {}).get(
                        "execution_authorized"
                    ),
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
