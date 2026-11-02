from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.audit_cluster_aware_sae_semantic_dispatch_coverage_v1 import (
    canonical_inputs as audit_inputs,
    verify_audit,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_semantic_dispatch_coverage_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v47 import (
    canonical_inputs as protocol_v47_inputs,
    verify_protocol as verify_protocol_v47,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v39"
EXPECTED_V38_RUNBOOK_ID = (
    "ab7b2c5658ef05d4ab9ea25b7f14f577c60b6c740a31d227fc45080a69ca5c33"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V39 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V38_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V38 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v38":
        raise ValueError("pinned predecessor is not runbook V38")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V38 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v47_path: Path,
    audit_builder_path: Path,
    audit_path: Path,
    benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v47(
        protocol_v47_path, **protocol_v47_inputs(project_root)
    )
    audit_verification = verify_audit(audit_path, **audit_inputs(project_root))
    benchmark_verification = verify_benchmark(
        benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    audit = _load(audit_path)
    benchmark = _load(benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "v1_graph_semantic_mismatch_audited_corrected_successor_required_"
        "empirical_graph_training_and_execution_absent"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v47"] = {
        "path": _relative(protocol_v47_path, project_root),
        "sha256": sha256_file(protocol_v47_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_aware_sae_semantic_dispatch_coverage_audit_builder_v1"] = {
        "path": _relative(audit_builder_path, project_root),
        "sha256": sha256_file(audit_builder_path),
    }
    result["source"]["cluster_aware_sae_semantic_dispatch_coverage_audit_v1"] = {
        "path": _relative(audit_path, project_root),
        "sha256": sha256_file(audit_path),
        "semantic_dispatch_coverage_audit_id": audit[
            "semantic_dispatch_coverage_audit_id"
        ],
        "verification": audit_verification,
    }
    result["source"]["cluster_aware_sae_semantic_dispatch_coverage_benchmark_v1"] = {
        "path": _relative(benchmark_path, project_root),
        "sha256": sha256_file(benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    current = result["current_result"]
    current["cluster_aware_sae_semantic_dispatch_coverage_v1"] = (
        "pass_23_of_23_fail_closed_identity_edge_callable_gap_and_authority_controls"
    )
    current["v1_cross_artifact_structural_graph_preserved"] = True
    current["v1_cross_artifact_semantic_graph_verified"] = False
    current["semantic_dispatch_graph_nodes"] = 20
    current["semantic_dispatch_schema_and_id_matching_nodes"] = 12
    current["semantic_dispatch_ready_nodes_under_v1_graph"] = 4
    current["semantic_dispatch_graph_edges"] = 37
    current["semantic_dispatch_exact_implemented_edges"] = 9
    current["semantic_dispatch_mismatched_edges"] = 28
    current["corrected_semantic_graph_successor_required"] = True
    current["production_runtime_attachment_verifier_present"] = False
    current["empirical_cross_artifact_graph_present"] = False
    current["semantic_dispatch_allowed"] = False
    current["training_performed"] = False
    current["persistent_dictionary_output_created"] = False

    stage = result["stages"][0]
    stage["commands"].extend(
        [
            {
                "id": "verify_cluster_aware_sae_semantic_dispatch_coverage_audit_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.audit_cluster_aware_sae_semantic_dispatch_coverage_v1",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_semantic_dispatch_coverage_audit_v1/audit.json",
                ],
            },
            {
                "id": "verify_cluster_aware_sae_semantic_dispatch_coverage_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_semantic_dispatch_coverage_v1",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_semantic_dispatch_coverage_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v47",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v47",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v47.json",
                ],
            },
        ]
    )
    stage["expected"]["cluster_aware_sae_semantic_dispatch_coverage_v1"] = (
        "23 controls pass; V1 remains structural-only; 12/20 node identities and "
        "9/37 field edges match implementation; semantic dispatch stays disabled"
    )
    stage["stop_if"].extend(
        [
            "protocol V47, semantic coverage audit, or benchmark differs",
            "V1 structural graph is described as semantically dispatchable",
            "a mismatched schema ID field edge or embedded receipt is silently coerced",
            "synthetic inert runtime attachment is presented as production attachment",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "semantic dispatcher intentionally withheld after reproducible V1 contract mismatch audit; "
        "a corrected successor must use implemented guarded capture routing materialization "
        "blueprint budget signed-receipt and combined-transition representations"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "the corrected successor mutates or relabels V1 instead of preserving it",
        "budget admission lacks exact execution-blueprint ID and file-hash binding",
        "raw capture manifest substitutes for guarded V3 capture execution lineage",
        "production runtime attachment is inferred from an inert synthetic claim",
    ]
    completion = result["completion_definition"]
    completion["cluster_aware_sae_semantic_dispatch_coverage_audited"] = True
    completion["v1_cross_artifact_structural_graph_preserved"] = True
    completion["v1_cross_artifact_semantic_graph_verified"] = False
    completion["semantic_dispatch_schema_and_id_matching_nodes"] = 12
    completion["semantic_dispatch_ready_nodes_under_v1_graph"] = 4
    completion["semantic_dispatch_exact_implemented_edges"] = 9
    completion["semantic_dispatch_mismatched_edges"] = 28
    completion["corrected_semantic_graph_successor_required"] = True
    completion["production_runtime_attachment_verifier_present"] = False
    completion["empirical_cross_artifact_graph_present"] = False
    completion["semantic_dispatch_allowed"] = False
    completion["training_performed"] = False
    completion["persistent_dictionary_output_created"] = False
    boundary = result["scientific_boundary"]
    boundary["cluster_aware_sae_semantic_dispatch_coverage_audited"] = True
    boundary["v1_cross_artifact_structural_graph_preserved"] = True
    boundary["v1_cross_artifact_semantic_graph_verified"] = False
    boundary["semantic_dispatch_allowed"] = False
    boundary["corrected_semantic_graph_successor_required"] = True
    boundary["production_runtime_attachment_verifier_present"] = False
    boundary["empirical_cross_artifact_graph_present"] = False
    boundary["semantic_verifiers_invoked_on_empirical_inputs"] = False
    boundary["production_campaign_executed"] = False
    boundary["production_trainer_callback_invoked"] = False
    boundary["training_performed"] = False
    boundary["persistent_dictionary_output_created"] = False
    boundary["model_framework_imported"] = False
    boundary["model_or_adapter_weights_opened"] = False
    boundary["model_inference_performed"] = False
    boundary["execution_authorized"] = False
    boundary["sae_training_authorized"] = False
    boundary["paid_compute_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V39 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v39",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "semantic_dispatch_coverage_audited": True,
        "schema_and_id_matching_nodes": 12,
        "dispatch_ready_nodes_under_v1_graph": 4,
        "exact_implemented_field_edges": 9,
        "mismatched_edges": 28,
        "corrected_semantic_graph_successor_required": True,
        "production_runtime_attachment_verifier_present": False,
        "empirical_graph_present": False,
        "semantic_dispatch_allowed": False,
        "training_performed": False,
        "persistent_dictionary_output_created": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v38/runbook.json"
        ),
        "protocol_v47_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v47.json"
        ),
        "audit_builder_path": (
            project_root
            / "experiments/authz_mechinterp/audit_cluster_aware_sae_semantic_dispatch_coverage_v1.py"
        ),
        "audit_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_semantic_dispatch_coverage_audit_v1/audit.json"
        ),
        "benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_semantic_dispatch_coverage_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V39.")
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
