from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_activation_category_cluster_discreteness_v2 import (
    verify_benchmark as verify_discreteness_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v1 import (
    canonical_inputs as runbook_v1_inputs,
    verify_runbook as verify_runbook_v1,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v2"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("cheap-executor runbook V2 source escapes the project")
    return resolved.relative_to(root.resolve()).as_posix()


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v10_path: Path,
    discreteness_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = verify_runbook_v1(
        predecessor_path, **runbook_v1_inputs(project_root)
    )
    benchmark_verification = verify_discreteness_benchmark(
        discreteness_benchmark_path
    )
    predecessor = _load(predecessor_path)
    protocol = _load(protocol_v10_path)
    benchmark = _load(discreteness_benchmark_path)
    if protocol.get("schema") != "sch.vpd-activation-category-protocol.v10":
        raise ValueError("cheap-executor runbook V2 requires protocol V10")
    contract = protocol.get("component_compartments", {}).get(
        "within_category_cluster_discreteness_v2", {}
    )
    if (
        contract.get("benchmark", {}).get("benchmark_id")
        != benchmark["benchmark_id"]
        or contract.get("selector", {}).get("sha256")
        != benchmark["source"]["v2_selector_sha256"]
    ):
        raise ValueError("protocol V10 cluster discreteness lineage differs")

    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "ready_for_low_cost_executor_preflight_cluster_candidate_v2_runtime_pending"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v10"] = {
        "path": _relative(protocol_v10_path, project_root),
        "sha256": sha256_file(protocol_v10_path),
    }
    result["source"]["cluster_discreteness_benchmark_v2"] = {
        "path": _relative(discreteness_benchmark_path, project_root),
        "sha256": sha256_file(discreteness_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["cluster_count_selector_v2"] = {
        "path": contract["selector"]["path"],
        "sha256": contract["selector"]["sha256"],
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["legacy_continuous_false_split_scenarios"] = int(
        benchmark["summary"]["legacy_false_splits_reproduced"]
    )
    result["current_result"]["v2_continuous_false_split_scenarios"] = 0
    result["current_result"]["cluster_discreteness_benchmark_status"] = "pass"
    result["executor_policy"]["never_accept_v1_only_cluster_count_selection"] = True
    result["executor_policy"][
        "next_empirical_candidate_requires_cluster_count_selection_v2"
    ] = True

    local_preflight = result["stages"][0]
    local_preflight["commands"].append(
        {
            "id": "verify_cluster_discreteness_benchmark_v2",
            "executable": ".venv/Scripts/python.exe",
            "arguments": [
                "-m",
                "experiments.authz_mechinterp.benchmark_activation_category_cluster_discreteness_v2",
                "--verify",
                "artifacts/vpd_activation_category_cluster_discreteness_benchmark_v2/report.json",
            ],
        }
    )
    local_preflight["expected"]["cluster_discreteness_benchmark"] = (
        "4/4 pass; two legacy continuous false splits reproduced; V2 false splits zero"
    )
    local_preflight["stop_if"].append(
        "protocol V10 or cluster discreteness benchmark lineage differs"
    )

    candidate_gate = {
        "stage": 1,
        "id": "cluster_candidate_discreteness_v2",
        "authorization": "model_free_after_fresh_empirical_effect_inputs_exist",
        "safe_for_cheap_executor_now": False,
        "implementation_status": (
            "selector_and_synthetic_benchmark_implemented; production_candidate_"
            "wrapper_and_independent_admission_not_yet_implemented"
        ),
        "required_inputs": [
            "frozen candidate component inventory",
            "discovery activation-category effects",
            "independent cluster-validation activation-category effects",
            "discovery and validation pair-block sufficient statistics",
        ],
        "required_selection_schema": "sch.vpd-category-cluster-count-selection.v2",
        "required_checks": [
            "cross-archive adjusted-Rand stability",
            "mean silhouette",
            "silhouette lower 10th percentile",
            "minimum cluster-boundary gap ratio",
            "minimum cluster size",
        ],
        "continuous_geometry_action": "one_cluster_for_activation_category",
        "stop_if": [
            "production V2 candidate wrapper is absent",
            "independent V2 admission is absent",
            "candidate uses V1-only cluster count selection",
            "any V2 candidate status is not pass or pass_fallback",
        ],
    }
    for stage in result["stages"][1:]:
        stage["stage"] = int(stage["stage"]) + 1
    result["stages"].insert(1, candidate_gate)
    scientific_release = next(
        stage
        for stage in result["stages"]
        if stage["id"] == "scientific_release_prerequisite"
    )
    scientific_release["stop_if"].append(
        "cluster candidate lacks independently admitted V2 discreteness receipt"
    )
    scientific_release["additional_required_prerequisite"] = (
        "independently admitted cluster-count selection V2 receipt"
    )
    result["completion_definition"]["cheap_executor_task_complete_after_stage_0"] = True
    result["completion_definition"]["paid_pipeline_complete_only_after_stage_8"] = True
    result["completion_definition"].pop("paid_pipeline_complete_only_after_stage_7", None)
    result["completion_definition"][
        "cluster_candidate_runtime_ready"
    ] = False
    result["scientific_boundary"]["execution_authorized"] = False
    body = result
    return {**body, "runbook_id": content_sha256(body)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if content_sha256(body) != identifier:
        raise ValueError("cheap-executor runbook V2 content address mismatch")
    expected = build_runbook(**kwargs)
    if observed != expected:
        raise ValueError("cheap-executor runbook V2 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v2",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "cluster_candidate_runtime_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root
        / "artifacts/vpd_cheap_executor_runbook_v1/runbook.json",
        "protocol_v10_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v10.json",
        "discreteness_benchmark_path": project_root
        / "artifacts/vpd_activation_category_cluster_discreteness_benchmark_v2/report.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive or verify the V2 cheap-executor clustering handoff."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None:
        result = verify_runbook(args.verify, **inputs)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_runbook(**inputs)
        write_json(args.output, result)
    print(
        json.dumps(
            {
                "schema": str(result["schema"]),
                "status": str(result["status"]),
                "runbook_id": str(result["runbook_id"]),
                "execution_authorized": bool(
                    result.get("execution_authorized", False)
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
