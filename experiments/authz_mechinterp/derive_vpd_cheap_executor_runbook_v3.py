from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_candidate_discreteness_admission_v2 import (
    verify_benchmark as verify_admission_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v11 import (
    canonical_inputs as protocol_v11_inputs,
    verify_protocol as verify_protocol_v11,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v2 import (
    canonical_inputs as runbook_v2_inputs,
    verify_runbook as verify_runbook_v2,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v3"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("cheap-executor runbook V3 source escapes the project")
    return resolved.relative_to(root.resolve()).as_posix()


def build_runbook(
    *, predecessor_path: Path, protocol_v11_path: Path,
    admission_benchmark_path: Path, project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = verify_runbook_v2(
        predecessor_path, **runbook_v2_inputs(project_root)
    )
    protocol_verification = verify_protocol_v11(
        protocol_v11_path, **protocol_v11_inputs(project_root)
    )
    benchmark_verification = verify_admission_benchmark(
        admission_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    protocol = _load(protocol_v11_path)
    benchmark = _load(admission_benchmark_path)
    contract = protocol["component_compartments"][
        "cluster_candidate_discreteness_admission_v2"
    ]
    if contract["benchmark"]["benchmark_id"] != benchmark["benchmark_id"]:
        raise ValueError("protocol V11 admission benchmark lineage differs")
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = "ready_for_local_preflight_and_empirical_admission_when_inputs_exist"
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v11"] = {
        "path": _relative(protocol_v11_path, project_root),
        "sha256": sha256_file(protocol_v11_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_candidate_admission_benchmark_v2"] = {
        "path": _relative(admission_benchmark_path, project_root),
        "sha256": sha256_file(admission_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["cluster_candidate_admission_runner_v2"] = contract["runner"]
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["production_cluster_candidate_admission_runtime_present"] = True
    result["current_result"]["empirical_cluster_candidate_admission_receipt_present"] = False
    result["executor_policy"]["never_relabel_or_mutate_rejected_candidate"] = True
    result["stages"][0]["commands"].append({
        "id": "verify_cluster_candidate_admission_benchmark_v2",
        "executable": ".venv/Scripts/python.exe",
        "arguments": [
            "-m",
            "experiments.authz_mechinterp.benchmark_vpd_cluster_candidate_discreteness_admission_v2",
            "--verify",
            "artifacts/vpd_cluster_candidate_discreteness_admission_benchmark_v2/report.json",
        ],
    })
    result["stages"][0]["expected"]["cluster_candidate_admission_benchmark"] = (
        "2/2 pass; discrete admitted; continuous strength gradient rejected"
    )
    result["stages"][0]["stop_if"].append(
        "protocol V11 or candidate-admission benchmark lineage differs"
    )
    candidate = result["stages"][1]
    candidate["implementation_status"] = (
        "independent V2 admission implemented and production-shaped benchmark passed"
    )
    candidate["safe_for_cheap_executor_now"] = False
    candidate["command_template"] = {
        "executable": ".venv/Scripts/python.exe",
        "arguments": [
            "-m", "experiments.authz_mechinterp.admit_vpd_cluster_candidate_discreteness_v2",
            "--bundle-dir", "<FROZEN_CATEGORY_BUNDLE_DIR>",
            "--components", "<COMPONENT_INVENTORY>",
            "--discovery-effects", "<DISCOVERY_EFFECTS>",
            "--discovery-merge-manifest", "<DISCOVERY_MERGE_MANIFEST>",
            "--cluster-validation-effects", "<CLUSTER_VALIDATION_EFFECTS>",
            "--cluster-validation-merge-manifest", "<CLUSTER_VALIDATION_MERGE_MANIFEST>",
            "--output", "<CLUSTER_DISCRETENESS_ADMISSION_RECEIPT>",
        ],
    }
    candidate["expected"] = {
        "base_bundle_verification": "pass",
        "admission_status": "pass_candidate_cluster_discreteness_v2",
        "mismatch_count": 0,
        "candidate_mutated_or_relabelled": False,
    }
    candidate["stop_if"] = [
        "any required empirical input is missing or its merge provenance fails",
        "base V1-format candidate verification is not pass",
        "V2 allocation differs from the frozen candidate allocation",
        "admission status is not pass_candidate_cluster_discreteness_v2",
        "candidate was mutated or relabelled",
    ]
    result["placeholders"]["cluster_candidate_admission_inputs"] = [
        "FROZEN_CATEGORY_BUNDLE_DIR", "COMPONENT_INVENTORY", "DISCOVERY_EFFECTS",
        "DISCOVERY_MERGE_MANIFEST", "CLUSTER_VALIDATION_EFFECTS",
        "CLUSTER_VALIDATION_MERGE_MANIFEST", "CLUSTER_DISCRETENESS_ADMISSION_RECEIPT",
    ]
    result["completion_definition"]["cluster_candidate_runtime_ready"] = True
    result["completion_definition"]["cluster_candidate_empirical_inputs_present"] = False
    result["completion_definition"]["cheap_executor_task_complete_after_stage_0"] = True
    result["scientific_boundary"]["execution_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = body.pop("runbook_id", "")
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V3 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v3", "status": "pass",
        "runbook_id": identifier, "stages": len(observed["stages"]),
        "safe_stage_now": 0, "cluster_candidate_runtime_ready": True,
        "empirical_inputs_present": False, "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root / "artifacts/vpd_cheap_executor_runbook_v2/runbook.json",
        "protocol_v11_path": project_root / "configs/evals/vpd_activation_category_compartments_v11.json",
        "admission_benchmark_path": project_root / "artifacts/vpd_cluster_candidate_discreteness_admission_benchmark_v2/report.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V3.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None:
        result = verify_runbook(args.verify, **inputs)
    else:
        if args.output is None: parser.error("--output is required unless verifying")
        result = build_runbook(**inputs); write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("schema", "status", "runbook_id")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
