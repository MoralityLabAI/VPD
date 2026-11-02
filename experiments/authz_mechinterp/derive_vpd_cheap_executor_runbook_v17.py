from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_empirical_preflight_v1 import (
    verify_benchmark as verify_preflight_benchmark,
)
from experiments.authz_mechinterp.benchmark_vpd_five_role_recovered_incidence_audit_v1 import (
    verify_audit,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_empirical_preflight_v1 import (
    verify_preflight,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v25 import (
    canonical_inputs as protocol_v25_inputs,
    verify_protocol as verify_protocol_v25,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v17"
EXPECTED_V16_RUNBOOK_ID = (
    "321b0cab4fc95d0270395f39ba95d0139a3ea258e3cb9c05822b6c82f6ce5dd3"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V17 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V16_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V16 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v16":
        raise ValueError("pinned predecessor is not runbook V16")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V16 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "method": "content_address_and_boundary_fields",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v25_path: Path,
    five_role_audit_path: Path,
    empirical_preflight_path: Path,
    empirical_preflight_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v25(
        protocol_v25_path, **protocol_v25_inputs(project_root)
    )
    audit_verification = verify_audit(
        five_role_audit_path, project_root=project_root
    )
    preflight_verification = verify_preflight(
        empirical_preflight_path, project_root=project_root
    )
    benchmark_verification = verify_preflight_benchmark(
        empirical_preflight_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    protocol = _load(protocol_v25_path)
    audit = _load(five_role_audit_path)
    preflight = _load(empirical_preflight_path)
    benchmark = _load(empirical_preflight_benchmark_path)
    contract = protocol["component_compartments"][
        "cluster_to_sae_empirical_preflight_v1"
    ]
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "ready_for_local_model_free_preflight_verification_"
        "empirical_capture_blocked_on_13_bindings"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v25"] = {
        "path": _relative(protocol_v25_path, project_root),
        "sha256": sha256_file(protocol_v25_path),
        "verification": protocol_verification,
    }
    result["source"]["five_role_recovered_incidence_audit_v1"] = {
        "path": _relative(five_role_audit_path, project_root),
        "sha256": sha256_file(five_role_audit_path),
        "audit_id": audit["audit_id"],
        "verification": audit_verification,
    }
    result["source"]["cluster_to_sae_empirical_preflight_v1"] = {
        "path": _relative(empirical_preflight_path, project_root),
        "sha256": sha256_file(empirical_preflight_path),
        "preflight_id": preflight["preflight_id"],
        "verification": preflight_verification,
    }
    result["source"]["cluster_to_sae_empirical_preflight_benchmark_v1"] = {
        "path": _relative(empirical_preflight_benchmark_path, project_root),
        "sha256": sha256_file(empirical_preflight_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["five_role_recovered_incidence_audit"] = (
        "pass_15_of_15_25168_rows_2288_pairs_16_templates"
    )
    result["current_result"]["cluster_to_sae_empirical_preflight"] = (
        "pass_15_of_15_blocked_on_13_bindings"
    )
    result["current_result"]["cluster_to_sae_capture_roles"] = [
        "discovery",
        "calibration",
        "cluster_validation",
        "sae_training",
    ]
    result["current_result"]["final_evaluation_excluded"] = True
    result["current_result"]["maximum_batch_forwards_at_batch_size_4"] = (
        preflight["resource_ceiling"]["maximum_batch_forwards_at_batch_size_4"]
    )
    result["current_result"]["empirical_capture_ready"] = False

    preflight_stage = result["stages"][0]
    preflight_stage["commands"].extend(
        [
            {
                "id": "verify_five_role_recovered_incidence_audit_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_five_role_recovered_incidence_audit_v1",
                    "--verify",
                    "artifacts/vpd_five_role_recovered_incidence_audit_v1/report.json",
                ],
            },
            {
                "id": "verify_cluster_to_sae_empirical_preflight_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.build_vpd_cluster_to_sae_empirical_preflight_v1",
                    "--verify",
                    "artifacts/vpd_cluster_to_sae_empirical_preflight_v1/preflight.json",
                ],
            },
            {
                "id": "verify_cluster_to_sae_empirical_preflight_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_to_sae_empirical_preflight_v1",
                    "--verify",
                    "artifacts/vpd_cluster_to_sae_empirical_preflight_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v25",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v25",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v25.json",
                ],
            },
        ]
    )
    preflight_stage["expected"]["five_role_incidence_audit"] = (
        "15 controls pass; all five roles exact and pairwise disjoint"
    )
    preflight_stage["expected"]["cluster_to_sae_empirical_preflight"] = (
        "15 controls pass; exactly 13 bindings missing; execution remains false"
    )
    preflight_stage["stop_if"].extend(
        [
            "protocol V25, five-role audit, empirical preflight, or benchmark differs",
            "preflight missing-binding count is not exactly 13",
            "final_evaluation appears in any capture role or stage",
            "preflight imports a model framework or opens model/adapter weights",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "four-role staged cluster-to-SAE capture preflight is model-free and verified; "
        "all immutable identities, authenticated authorization, empirical receipts, "
        "and partition admission remain absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "any of the 13 immutable bindings is absent",
        "stage order is not discovery, calibration, cluster validation, partition admission, then SAE-training routing capture",
        "resource ceiling exceeds 41352192 batch forwards without a new versioned plan",
        "authorization scope includes SAE dictionary training or final evaluation",
    ]
    result["completion_definition"]["five_role_incidence_audit_ready"] = True
    result["completion_definition"]["cluster_to_sae_empirical_preflight_ready"] = True
    result["completion_definition"]["empirical_capture_ready"] = False
    result["scientific_boundary"]["model_framework_imported"] = False
    result["scientific_boundary"]["model_or_adapter_weights_opened"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["sae_training_authorized"] = False
    result["scientific_boundary"]["final_evaluation_consumed"] = False
    if len(contract["empirical_preflight"]["missing_required_bindings"]) != 13:
        raise ValueError("protocol V25 missing-binding count differs")
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V17 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v17",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "missing_required_bindings": 13,
        "empirical_capture_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v16/runbook.json"
        ),
        "protocol_v25_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v25.json"
        ),
        "five_role_audit_path": (
            project_root
            / "artifacts/vpd_five_role_recovered_incidence_audit_v1/report.json"
        ),
        "empirical_preflight_path": (
            project_root
            / "artifacts/vpd_cluster_to_sae_empirical_preflight_v1/preflight.json"
        ),
        "empirical_preflight_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_to_sae_empirical_preflight_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive or verify cheap-executor runbook V17."
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
