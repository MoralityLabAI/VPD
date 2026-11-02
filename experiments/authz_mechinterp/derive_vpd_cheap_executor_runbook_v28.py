from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_execution_blueprint_v1 import (
    verify_benchmark as verify_blueprint_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v36 import (
    canonical_inputs as protocol_v36_inputs,
    verify_protocol as verify_protocol_v36,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v28"
EXPECTED_V27_RUNBOOK_ID = (
    "d9de56b5b51ebfd48dc7625e853c78264da640700d1a1623bf82385358285c53"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V28 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V27_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V27 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v27":
        raise ValueError("pinned predecessor is not runbook V27")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V27 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v36_path: Path,
    blueprint_builder_path: Path,
    blueprint_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v36(
        protocol_v36_path, **protocol_v36_inputs(project_root)
    )
    benchmark_verification = verify_blueprint_benchmark(
        blueprint_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    benchmark = _load(blueprint_benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "cluster_routed_sae_execution_contracts_ready_synthetic_controls_pass_"
        "runtime_worker_empirical_blueprint_and_authority_absent"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v36"] = {
        "path": _relative(protocol_v36_path, project_root),
        "sha256": sha256_file(protocol_v36_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_aware_sae_execution_blueprint_v1"] = {
        "path": _relative(blueprint_builder_path, project_root),
        "sha256": sha256_file(blueprint_builder_path),
    }
    result["source"]["cluster_aware_sae_execution_blueprint_benchmark_v1"] = {
        "path": _relative(blueprint_benchmark_path, project_root),
        "sha256": sha256_file(blueprint_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["cluster_aware_sae_execution_blueprint_v1"] = (
        "pass_17_of_17_exact_samples_routes_global_pairs_and_outputs"
    )
    result["current_result"]["execution_jobs_in_exact_fixture"] = 34
    result["current_result"]["cluster_aware_runtime_worker_implemented"] = False
    result["current_result"]["empirical_cluster_aware_blueprint_present"] = False
    result["current_result"]["scientific_release_receipt_present"] = False
    result["current_result"]["explicit_compute_authorization_present"] = False
    result["current_result"]["identity_bindings_complete"] = False
    result["current_result"]["capture_started"] = False

    preflight_stage = result["stages"][0]
    preflight_stage["commands"].extend(
        [
            {
                "id": "verify_cluster_aware_sae_execution_blueprint_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_execution_blueprint_v1",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_execution_blueprint_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v36",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v36",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v36.json",
                ],
            },
        ]
    )
    preflight_stage["expected"]["cluster_aware_sae_execution_blueprint_v1"] = (
        "17 controls pass; 34 exact cluster-routed job contracts, unique outputs, one global pair split, worker absent"
    )
    preflight_stage["stop_if"].extend(
        [
            "protocol V36 or cluster-aware execution blueprint benchmark differs",
            "a job omits exact episode or pair membership",
            "a job uses a different global pair assignment",
            "a specific job omits component or routing identity",
            "the absent runtime worker is described as implemented",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "exact cluster-routed SAE execution contracts pass synthetic controls; runtime "
        "worker, empirical blueprint, scientific release, and compute authority are absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "a blueprint is treated as an executable worker",
        "scientific release or explicit compute authorization is bypassed",
        "exact routed samples are replaced by whole-category selection",
    ]
    result["completion_definition"]["cluster_aware_sae_execution_blueprint_ready"] = True
    result["completion_definition"]["exact_cluster_routed_job_contracts_ready"] = True
    result["completion_definition"]["cluster_aware_runtime_worker_implemented"] = False
    result["completion_definition"]["empirical_cluster_aware_blueprint_present"] = False
    result["completion_definition"]["scientific_release_receipt_present"] = False
    result["completion_definition"]["explicit_compute_authorization_present"] = False
    result["completion_definition"]["identity_bindings_complete"] = False
    result["completion_definition"]["capture_started"] = False
    result["scientific_boundary"]["synthetic_execution_blueprint_fixture_only"] = True
    result["scientific_boundary"]["cluster_aware_execution_contract_present"] = True
    result["scientific_boundary"]["cluster_aware_runtime_worker_implemented"] = False
    result["scientific_boundary"]["empirical_cluster_aware_blueprint_present"] = False
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
        raise ValueError("cheap-executor runbook V28 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v28",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "cluster_aware_sae_execution_blueprint_ready": True,
        "execution_jobs_in_fixture": 34,
        "cluster_aware_runtime_worker_implemented": False,
        "empirical_cluster_aware_blueprint_present": False,
        "scientific_release_receipt_present": False,
        "explicit_compute_authorization_present": False,
        "identity_bindings_complete": False,
        "capture_started": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v27/runbook.json"
        ),
        "protocol_v36_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v36.json"
        ),
        "blueprint_builder_path": (
            project_root
            / "experiments/authz_mechinterp/build_cluster_aware_sae_execution_blueprint_v1.py"
        ),
        "blueprint_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_execution_blueprint_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V28.")
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
