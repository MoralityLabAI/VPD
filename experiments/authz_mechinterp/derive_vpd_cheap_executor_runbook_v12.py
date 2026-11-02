from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_activation_category_cluster_partition_identity_v4 import (
    verify_benchmark as verify_partition_benchmark,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_candidate_partition_identity_admission_v4 import (
    verify_benchmark as verify_admission_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v20 import (
    canonical_inputs as protocol_v20_inputs,
    verify_protocol as verify_protocol_v20,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v12"
EXPECTED_V11_RUNBOOK_ID = (
    "ea909e4c81514150353be464e87f4f464469184d070bc72c4121bc67893af5bc"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V12 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_pinned_runbook_v11(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V11_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V11 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v11":
        raise ValueError("pinned predecessor is not runbook V11")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V11 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "method": "content_address_and_boundary_fields",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v20_path: Path,
    partition_benchmark_path: Path,
    admission_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_pinned_runbook_v11(predecessor_path)
    protocol_verification = verify_protocol_v20(
        protocol_v20_path, **protocol_v20_inputs(project_root)
    )
    partition_verification = verify_partition_benchmark(partition_benchmark_path)
    admission_verification = verify_admission_benchmark(
        admission_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    protocol = _load(protocol_v20_path)
    partition_benchmark = _load(partition_benchmark_path)
    admission_benchmark = _load(admission_benchmark_path)
    contract = protocol["component_compartments"][
        "cluster_candidate_partition_identity_admission_v4"
    ]
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "ready_for_local_preflight_exact_partition_admission_required_"
        "real_capture_pending"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v20"] = {
        "path": _relative(protocol_v20_path, project_root),
        "sha256": sha256_file(protocol_v20_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_partition_identity_benchmark_v4"] = {
        "path": _relative(partition_benchmark_path, project_root),
        "sha256": sha256_file(partition_benchmark_path),
        "benchmark_id": partition_benchmark["benchmark_id"],
        "verification": partition_verification,
    }
    result["source"]["cluster_partition_identity_admission_benchmark_v4"] = {
        "path": _relative(admission_benchmark_path, project_root),
        "sha256": sha256_file(admission_benchmark_path),
        "benchmark_id": admission_benchmark["benchmark_id"],
        "verification": admission_verification,
    }
    result["source"]["cluster_partition_identity_runtime_v4"] = contract[
        "partition_identity_runtime"
    ]
    result["source"]["cluster_partition_identity_admission_runtime_v4"] = contract[
        "production_admission_runtime"
    ]
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["cluster_partition_identity_benchmark"] = "pass"
    result["current_result"]["cluster_partition_identity_admission_benchmark"] = "pass"
    result["current_result"]["count_only_candidate_admission_prohibited"] = True
    result["current_result"]["empirical_partition_identity_admission_receipt_present"] = False
    preflight = result["stages"][0]
    preflight["commands"].extend(
        [
            {
                "id": "verify_cluster_partition_identity_benchmark_v4",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_activation_category_cluster_partition_identity_v4",
                    "--verify",
                    "artifacts/vpd_activation_category_cluster_partition_identity_benchmark_v4/report.json",
                ],
            },
            {
                "id": "verify_cluster_partition_identity_admission_benchmark_v4",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_candidate_partition_identity_admission_v4",
                    "--verify",
                    "artifacts/vpd_cluster_candidate_partition_identity_admission_benchmark_v4/report.json",
                ],
            },
        ]
    )
    preflight["expected"]["cluster_partition_identity_benchmark"] = (
        "same-count membership substitution rejected; label renaming invariant; six controls pass"
    )
    preflight["expected"]["cluster_partition_identity_admission_benchmark"] = (
        "production-shaped stable candidate admitted and semantic confound rejected"
    )
    preflight["stop_if"].append(
        "protocol V20 or exact partition-identity evidence differs"
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "capture pipeline implementation ready and exact label-invariant partition admission "
        "benchmarked; real capture and empirical V4 admission receipt absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "candidate cluster counts agree but exact label-invariant component co-assignment differs",
        "any discovery, validation, pair-block, condition, or template replicate partition differs",
        "empirical partition-identity admission V4 receipt is absent or rejects",
    ]
    result["completion_definition"]["cluster_partition_identity_runtime_ready"] = True
    result["completion_definition"]["cluster_partition_identity_admission_runtime_ready"] = True
    result["completion_definition"]["count_only_candidate_admission_forbidden"] = True
    result["completion_definition"]["empirical_partition_identity_admission_ready"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V12 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v12",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "cluster_partition_identity_admission_runtime_ready": True,
        "empirical_partition_identity_admission_ready": False,
        "actual_model_category_capture_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v11/runbook.json"
        ),
        "protocol_v20_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v20.json"
        ),
        "partition_benchmark_path": (
            project_root
            / "artifacts/vpd_activation_category_cluster_partition_identity_benchmark_v4/report.json"
        ),
        "admission_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_candidate_partition_identity_admission_benchmark_v4/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V12.")
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
            {key: result[key] for key in ("schema", "status", "runbook_id")},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
