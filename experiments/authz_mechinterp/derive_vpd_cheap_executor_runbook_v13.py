from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.audit_compartment_sae_cluster_routing_gap_v1 import (
    verify_audit,
)
from experiments.authz_mechinterp.benchmark_cluster_aware_sae_training_plan_v1 import (
    verify_benchmark as verify_planner_benchmark,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_episode_routing_v1 import (
    verify_benchmark as verify_router_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v21 import (
    canonical_inputs as protocol_v21_inputs,
    verify_protocol as verify_protocol_v21,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v13"
EXPECTED_V12_RUNBOOK_ID = (
    "61e128644cce58c4c873ff2fd01d26b44ec6c021c70c60a6a70a7379259e20da"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V13 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V12_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V12 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v12":
        raise ValueError("pinned predecessor is not runbook V12")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V12 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "method": "content_address_and_boundary_fields",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v21_path: Path,
    router_benchmark_path: Path,
    planner_benchmark_path: Path,
    gap_audit_path: Path,
    predecessor_sae_plan_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v21(
        protocol_v21_path, **protocol_v21_inputs(project_root)
    )
    router_verification = verify_router_benchmark(router_benchmark_path)
    planner_verification = verify_planner_benchmark(
        planner_benchmark_path, project_root=project_root
    )
    audit_verification = verify_audit(
        gap_audit_path, plan_path=predecessor_sae_plan_path
    )
    predecessor = _load(predecessor_path)
    protocol = _load(protocol_v21_path)
    router_benchmark = _load(router_benchmark_path)
    planner_benchmark = _load(planner_benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "ready_for_local_preflight_router_and_manifest_planner_ready_"
        "empirical_three_role_capture_pending"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v21"] = {
        "path": _relative(protocol_v21_path, project_root),
        "sha256": sha256_file(protocol_v21_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_episode_routing_benchmark_v1"] = {
        "path": _relative(router_benchmark_path, project_root),
        "sha256": sha256_file(router_benchmark_path),
        "benchmark_id": router_benchmark["benchmark_id"],
        "verification": router_verification,
    }
    result["source"]["cluster_aware_sae_plan_benchmark_v1"] = {
        "path": _relative(planner_benchmark_path, project_root),
        "sha256": sha256_file(planner_benchmark_path),
        "benchmark_id": planner_benchmark["benchmark_id"],
        "verification": planner_verification,
    }
    result["source"]["category_only_sae_plan_gap_audit_v1"] = {
        "path": _relative(gap_audit_path, project_root),
        "sha256": sha256_file(gap_audit_path),
        "audit_id": audit_verification["audit_id"],
        "verification": audit_verification,
    }
    result["source"]["cluster_episode_routing_runtime_v1"] = protocol[
        "component_compartments"
    ]["cluster_episode_routing_v1"]["runtime"]
    result["source"]["cluster_aware_sae_plan_runtime_v1"] = protocol[
        "component_compartments"
    ]["cluster_aware_sae_training_plan_v1"]["runtime"]
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["category_only_sae_plan_gap_verified"] = True
    result["current_result"]["current_sae_plan_cluster_bound_jobs"] = 0
    result["current_result"]["cluster_episode_routing_benchmark"] = "pass_16_of_16"
    result["current_result"]["cluster_aware_sae_plan_benchmark"] = "pass_16_of_16"
    result["current_result"]["empirical_cluster_routing_receipts_present"] = False
    result["current_result"]["empirical_cluster_aware_sae_plan_present"] = False

    preflight = result["stages"][0]
    legacy_partition_ids = {
        "verify_cluster_partition_identity_benchmark_v4",
        "verify_cluster_partition_identity_admission_benchmark_v4",
    }
    preflight["commands"] = [
        row for row in preflight["commands"] if row.get("id") not in legacy_partition_ids
    ]
    preflight["commands"].extend(
        [
            {
                "id": "verify_partition_identity_evidence_wrapper_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.verify_vpd_partition_identity_evidence_v1",
                ],
            },
            {
                "id": "verify_category_only_sae_plan_gap_audit_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.audit_compartment_sae_cluster_routing_gap_v1",
                    "--verify",
                    "artifacts/vpd_compartment_sae_cluster_routing_gap_audit_v1/report.json",
                ],
            },
            {
                "id": "verify_cluster_episode_routing_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_episode_routing_v1",
                    "--verify",
                    "artifacts/vpd_cluster_episode_routing_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_cluster_aware_sae_plan_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_cluster_aware_sae_training_plan_v1",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_training_plan_benchmark_v1/report.json",
                ],
            },
        ]
    )
    preflight["expected"]["category_only_sae_plan_gap_audit"] = (
        "18 jobs, 9 compartments, zero cluster-bound jobs"
    )
    preflight["expected"]["cluster_episode_routing_benchmark"] = (
        "16 controls pass; condition-confound and artifact tamper rejected"
    )
    preflight["expected"]["cluster_aware_sae_plan_benchmark"] = (
        "16 controls pass; weak route merges to shared; pair-atomic jobs disjoint"
    )
    preflight["stop_if"].append(
        "protocol V21, routing benchmark, planner benchmark, or gap audit differs"
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "model-free cluster router and manifest-only cluster-aware SAE planner ready; "
        "empirical discovery, cluster-validation, and SAE-training routing receipts absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "discovery and cluster-validation routing stability receipt is absent or rejects",
        "condition-stratified routing stability rejects despite aggregate agreement",
        "disjoint SAE-training routing receipt is absent",
        "specific route support is below the frozen gate and was not merged to shared",
        "any pair or episode appears in more than one primary SAE compartment",
        "job dictionary capacity differs from its resolved profile",
        "empirical cluster-aware SAE plan receipt is absent",
    ]
    result["completion_definition"]["cluster_episode_routing_runtime_ready"] = True
    result["completion_definition"]["cluster_aware_sae_plan_runtime_ready"] = True
    result["completion_definition"]["category_archive_duplication_for_clusters_forbidden"] = True
    result["completion_definition"]["sae_training_role_capture_planner_ready"] = False
    result["completion_definition"]["empirical_cluster_routing_ready"] = False
    result["completion_definition"]["empirical_cluster_aware_sae_plan_ready"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V13 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v13",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "cluster_episode_routing_runtime_ready": True,
        "cluster_aware_sae_plan_runtime_ready": True,
        "sae_training_role_capture_planner_ready": False,
        "empirical_cluster_routing_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v12/runbook.json"
        ),
        "protocol_v21_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v21.json"
        ),
        "router_benchmark_path": (
            project_root / "artifacts/vpd_cluster_episode_routing_benchmark_v1/report.json"
        ),
        "planner_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_training_plan_benchmark_v1/report.json"
        ),
        "gap_audit_path": (
            project_root
            / "artifacts/vpd_compartment_sae_cluster_routing_gap_audit_v1/report.json"
        ),
        "predecessor_sae_plan_path": (
            project_root / "artifacts/vpd_compartment_sae_training_plan_v4/plan.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive or verify cheap-executor runbook V13."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.output is not None:
        result = build_runbook(**inputs)
        write_json(args.output, result)
    else:
        result = verify_runbook(args.verify, **inputs)
    print(
        json.dumps(
            {key: result[key] for key in ("schema", "status", "runbook_id")},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
