from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_activation_category_cluster_model_order_v2 import verify_diagnostic
from experiments.authz_mechinterp.benchmark_activation_category_cluster_semantic_strata_v3 import verify_benchmark as verify_matrix_benchmark
from experiments.authz_mechinterp.benchmark_vpd_cluster_candidate_semantic_strata_admission_v3 import verify_benchmark as verify_production_benchmark
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v12 import canonical_inputs as protocol_v12_inputs, verify_protocol as verify_protocol_v12
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v3 import canonical_inputs as runbook_v3_inputs, verify_runbook as verify_runbook_v3


SCHEMA = "sch.vpd-cheap-executor-runbook.v4"


def _load(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()): raise ValueError("runbook V4 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def build_runbook(
    *, predecessor_path: Path, protocol_v12_path: Path, model_order_path: Path,
    matrix_benchmark_path: Path, production_benchmark_path: Path, project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = verify_runbook_v3(predecessor_path, **runbook_v3_inputs(project_root))
    protocol_verification = verify_protocol_v12(protocol_v12_path, **protocol_v12_inputs(project_root))
    model_order_verification = verify_diagnostic(model_order_path)
    matrix_verification = verify_matrix_benchmark(matrix_benchmark_path)
    production_verification = verify_production_benchmark(production_benchmark_path, project_root=project_root)
    predecessor = _load(predecessor_path); protocol = _load(protocol_v12_path)
    model_order = _load(model_order_path); matrix_benchmark = _load(matrix_benchmark_path); production = _load(production_benchmark_path)
    contract = protocol["component_compartments"]["cluster_candidate_semantic_strata_admission_v3"]
    if contract["production_shaped_benchmark"]["benchmark_id"] != production["benchmark_id"]:
        raise ValueError("protocol V12 production benchmark lineage differs")
    result = json.loads(json.dumps(predecessor)); result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = "ready_for_local_preflight_and_semantic_strata_admission_when_inputs_exist"
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root), "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"], "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v12"] = {
        "path": _relative(protocol_v12_path, project_root), "sha256": sha256_file(protocol_v12_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_model_order_diagnostic_v2"] = {
        "path": _relative(model_order_path, project_root), "sha256": sha256_file(model_order_path),
        "diagnostic_id": model_order["diagnostic_id"], "verification": model_order_verification,
    }
    result["source"]["cluster_semantic_strata_benchmark_v3"] = {
        "path": _relative(matrix_benchmark_path, project_root), "sha256": sha256_file(matrix_benchmark_path),
        "benchmark_id": matrix_benchmark["benchmark_id"], "verification": matrix_verification,
    }
    result["source"]["cluster_semantic_strata_admission_benchmark_v3"] = {
        "path": _relative(production_benchmark_path, project_root), "sha256": sha256_file(production_benchmark_path),
        "benchmark_id": production["benchmark_id"], "verification": production_verification,
    }
    result["source"]["cluster_semantic_strata_admission_runner_v3"] = contract["runner"]
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["semantic_strata_confound_scenarios_rejected"] = 2
    result["current_result"]["production_condition_confound_rejected_by_v3"] = True
    result["current_result"]["empirical_cluster_candidate_admission_receipt_present"] = False
    result["executor_policy"]["never_accept_v2_only_candidate_admission"] = True
    preflight = result["stages"][0]
    for command_id, module, artifact in (
        ("verify_cluster_model_order_diagnostic_v2", "experiments.authz_mechinterp.benchmark_activation_category_cluster_model_order_v2", "artifacts/vpd_activation_category_cluster_model_order_diagnostic_v2/report.json"),
        ("verify_cluster_semantic_strata_benchmark_v3", "experiments.authz_mechinterp.benchmark_activation_category_cluster_semantic_strata_v3", "artifacts/vpd_activation_category_cluster_semantic_strata_benchmark_v3/report.json"),
        ("verify_cluster_semantic_strata_admission_benchmark_v3", "experiments.authz_mechinterp.benchmark_vpd_cluster_candidate_semantic_strata_admission_v3", "artifacts/vpd_cluster_candidate_semantic_strata_admission_benchmark_v3/report.json"),
    ):
        flag = "--verify"
        preflight["commands"].append({"id": command_id, "executable": ".venv/Scripts/python.exe", "arguments": ["-m", module, flag, artifact]})
    preflight["expected"]["cluster_model_order"] = "4/4 exact for known 2-4 modes"
    preflight["expected"]["semantic_strata_matrix_benchmark"] = "3/3 pass; two confounds rejected"
    preflight["expected"]["semantic_strata_production_benchmark"] = "2/2 pass; V2 confound admitted and V3 rejected"
    preflight["stop_if"].append("protocol V12 or semantic-strata evidence lineage differs")
    candidate = result["stages"][1]
    candidate["id"] = "cluster_candidate_semantic_strata_v3"
    candidate["required_selection_schema"] = "sch.vpd-cluster-candidate-semantic-strata-admission.v3"
    candidate["implementation_status"] = "independent semantic-strata V3 admission implemented and production-shaped benchmark passed"
    candidate["command_template"]["arguments"][1] = "experiments.authz_mechinterp.admit_vpd_cluster_candidate_semantic_strata_v3"
    candidate["expected"] = {
        "base_bundle_verification": "pass",
        "admission_status": "pass_candidate_semantic_strata_v3",
        "mismatch_count": 0,
        "condition_and_template_statistics_in_both_archives": True,
        "candidate_mutated_or_relabelled": False,
    }
    candidate["required_checks"].extend([
        "leave-one-condition-out membership stability in discovery and cluster validation",
        "leave-one-template-group-out membership stability in discovery and cluster validation",
    ])
    candidate["stop_if"] = [
        "any required empirical input is missing or its merge provenance fails",
        "condition or template sufficient statistics are absent in either archive",
        "base V1-format candidate verification is not pass",
        "semantic-strata V3 allocation differs from the frozen candidate allocation",
        "admission status is not pass_candidate_semantic_strata_v3",
        "candidate was mutated or relabelled",
    ]
    result["completion_definition"]["semantic_strata_admission_runtime_ready"] = True
    result["completion_definition"]["semantic_strata_empirical_inputs_present"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path); body = dict(observed); identifier = body.pop("runbook_id", "")
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V4 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v4", "status": "pass",
        "runbook_id": identifier, "stages": len(observed["stages"]), "safe_stage_now": 0,
        "semantic_strata_admission_runtime_ready": True, "empirical_inputs_present": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root / "artifacts/vpd_cheap_executor_runbook_v3/runbook.json",
        "protocol_v12_path": project_root / "configs/evals/vpd_activation_category_compartments_v12.json",
        "model_order_path": project_root / "artifacts/vpd_activation_category_cluster_model_order_diagnostic_v2/report.json",
        "matrix_benchmark_path": project_root / "artifacts/vpd_activation_category_cluster_semantic_strata_benchmark_v3/report.json",
        "production_benchmark_path": project_root / "artifacts/vpd_cluster_candidate_semantic_strata_admission_benchmark_v3/report.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V4.")
    parser.add_argument("--output", type=Path); parser.add_argument("--verify", type=Path)
    args = parser.parse_args(); inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None: result = verify_runbook(args.verify, **inputs)
    else:
        if args.output is None: parser.error("--output is required unless verifying")
        result = build_runbook(**inputs); write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("schema", "status", "runbook_id")}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
