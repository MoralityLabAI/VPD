from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_materialization_v1 import (
    verify_benchmark as verify_materialization_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v35 import (
    canonical_inputs as protocol_v35_inputs,
    verify_protocol as verify_protocol_v35,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v27"
EXPECTED_V26_RUNBOOK_ID = (
    "746a4ed9b85be93e7dee9b95e949b6a685e7c98a2bda08d91ffd7e8822a81727"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V27 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V26_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V26 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v26":
        raise ValueError("pinned predecessor is not runbook V26")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V26 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v35_path: Path,
    materializer_path: Path,
    materialization_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v35(
        protocol_v35_path, **protocol_v35_inputs(project_root)
    )
    benchmark_verification = verify_materialization_benchmark(
        materialization_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    benchmark = _load(materialization_benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "eight_category_guarded_route_materializer_ready_synthetic_controls_pass_"
        "empirical_bundle_and_plan_absent"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v35"] = {
        "path": _relative(protocol_v35_path, project_root),
        "sha256": sha256_file(protocol_v35_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_aware_sae_materializer_v1"] = {
        "path": _relative(materializer_path, project_root),
        "sha256": sha256_file(materializer_path),
    }
    result["source"]["cluster_aware_sae_materialization_benchmark_v1"] = {
        "path": _relative(materialization_benchmark_path, project_root),
        "sha256": sha256_file(materialization_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["cluster_aware_sae_materializer_v1"] = (
        "pass_18_of_18_exact_eight_category_three_role_global_pair_safe"
    )
    result["current_result"]["guarded_v3_role_captures_required"] = 3
    result["current_result"]["adapter_and_routing_receipts_required"] = 24
    result["current_result"]["independent_stability_receipts_required"] = 8
    result["current_result"]["cluster_bound_training_jobs_in_fixture"] = 34
    result["current_result"]["persistent_empirical_routing_bundle_present"] = False
    result["current_result"]["empirical_cluster_aware_plan_present"] = False
    result["current_result"]["identity_bindings_complete"] = False
    result["current_result"]["capture_started"] = False
    result["current_result"]["empirical_capture_ready"] = False

    preflight_stage = result["stages"][0]
    preflight_stage["commands"].extend(
        [
            {
                "id": "verify_cluster_aware_sae_materialization_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_materialization_v1",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_materialization_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v35",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v35",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v35.json",
                ],
            },
        ]
    )
    preflight_stage["expected"]["cluster_aware_sae_materializer_v1"] = (
        "18 controls pass; eight categories, three roles, 24 V3-bound routes, eight stability receipts, and one global pair split"
    )
    preflight_stage["stop_if"].extend(
        [
            "protocol V35 or cluster-aware materialization benchmark differs",
            "any activation category or required role is absent",
            "a routing receipt does not bind the guarded V3 execution identity",
            "synthetic routing evidence is relabeled as empirical",
            "a cluster-aware plan lacks one global pair assignment across compartments",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "exact eight-category guarded routing evidence can materialize shared-pair-safe "
        "cluster-bound SAE jobs; persistent empirical route bundles and plans remain absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "a plan is materialized from fewer than 24 category-role routing receipts",
        "discovery and cluster-validation stability is omitted or rejected",
        "manifest materialization is treated as authority to train dictionaries",
    ]
    result["completion_definition"]["cluster_aware_sae_materializer_ready"] = True
    result["completion_definition"]["exact_eight_category_route_bundle_contract_ready"] = True
    result["completion_definition"]["global_pair_safe_cluster_jobs_ready"] = True
    result["completion_definition"]["persistent_empirical_routing_bundle_present"] = False
    result["completion_definition"]["empirical_cluster_aware_plan_present"] = False
    result["completion_definition"]["identity_bindings_complete"] = False
    result["completion_definition"]["capture_started"] = False
    result["completion_definition"]["empirical_capture_ready"] = False
    result["scientific_boundary"]["synthetic_guarded_route_bundle_only"] = True
    result["scientific_boundary"]["exact_eight_category_materializer_present"] = True
    result["scientific_boundary"]["persistent_empirical_routing_bundle_present"] = False
    result["scientific_boundary"]["empirical_cluster_aware_plan_present"] = False
    result["scientific_boundary"]["model_framework_imported"] = False
    result["scientific_boundary"]["model_or_adapter_weights_opened"] = False
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["sae_training_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V27 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v27",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "cluster_aware_sae_materializer_ready": True,
        "activation_categories": 8,
        "category_role_routing_receipts_required": 24,
        "persistent_empirical_routing_bundle_present": False,
        "empirical_cluster_aware_plan_present": False,
        "identity_bindings_complete": False,
        "capture_started": False,
        "empirical_capture_ready": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v26/runbook.json"
        ),
        "protocol_v35_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v35.json"
        ),
        "materializer_path": (
            project_root
            / "experiments/authz_mechinterp/materialize_vpd_cluster_aware_sae_plan_v1.py"
        ),
        "materialization_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_materialization_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V27.")
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
