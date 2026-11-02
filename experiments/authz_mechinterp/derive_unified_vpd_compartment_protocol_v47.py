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
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json


SCHEMA = "sch.vpd-activation-category-protocol.v47"
EXPECTED_V46_SHA256 = (
    "02d046536a330c56da723080738277ba22c786bdb3737c6c30bde82aac6dd0a1"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V47 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V46_SHA256:
        raise ValueError("protocol V46 differs from its pinned SHA-256")
    observed = _load(path)
    if observed.get("schema") != "sch.vpd-activation-category-protocol.v46":
        raise ValueError("pinned predecessor is not protocol V46")
    if observed.get("scientific_boundary", {}).get("execution_authorized") is not False:
        raise ValueError("protocol V46 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-protocol-predecessor-verification.v1",
        "status": "pass",
        "protocol_sha256": EXPECTED_V46_SHA256,
        "schema_verified": True,
        "non_authorizing_boundary_verified": True,
        "deep_predecessor_replay_performed": False,
    }


def build_protocol(
    *,
    predecessor_path: Path,
    audit_builder_path: Path,
    audit_path: Path,
    benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    audit_verification = verify_audit(audit_path, **audit_inputs(project_root))
    benchmark_verification = verify_benchmark(
        benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    audit = _load(audit_path)
    benchmark = _load(benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"][
        "cluster_aware_sae_semantic_dispatch_coverage_v1"
    ] = {
        "schema": "sch.vpd-cluster-aware-sae-semantic-dispatch-gap-contract.v1",
        "status": (
            "fail_closed_v1_structural_graph_preserved_semantic_successor_"
            "required_empirical_graph_training_and_execution_absent"
        ),
        "audit_builder": {
            "path": _relative(audit_builder_path, project_root),
            "sha256": sha256_file(audit_builder_path),
        },
        "audit": {
            "path": _relative(audit_path, project_root),
            "sha256": sha256_file(audit_path),
            "semantic_dispatch_coverage_audit_id": audit[
                "semantic_dispatch_coverage_audit_id"
            ],
            "verification": audit_verification,
        },
        "benchmark": {
            "path": _relative(benchmark_path, project_root),
            "sha256": sha256_file(benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
        },
        "graph_nodes": 20,
        "schema_and_id_matching_nodes": 12,
        "dispatch_ready_nodes_under_v1_graph": 4,
        "graph_edges": 37,
        "exact_implemented_field_edges": 9,
        "mismatched_edges": 28,
        "production_runtime_attachment_verifier_present": False,
        "v1_structural_graph_preserved": True,
        "v1_semantic_graph_verified": False,
        "semantic_dispatch_allowed": False,
        "empirical_graph_present": False,
        "training_performed": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    lineage = result["unified_lineage"]
    lineage["cluster_aware_sae_semantic_dispatch_coverage_audited"] = True
    lineage["v1_cross_artifact_structural_graph_preserved"] = True
    lineage["v1_cross_artifact_semantic_graph_verified"] = False
    lineage["semantic_dispatch_schema_and_id_matching_nodes"] = 12
    lineage["semantic_dispatch_ready_nodes_under_v1_graph"] = 4
    lineage["semantic_dispatch_exact_implemented_edges"] = 9
    lineage["semantic_dispatch_mismatched_edges"] = 28
    lineage["corrected_semantic_graph_successor_required"] = True
    lineage["production_runtime_attachment_verifier_present"] = False
    lineage["empirical_cross_artifact_graph_present"] = False
    lineage["production_trainer_callback_invoked"] = False
    lineage["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v46_verification": predecessor_verification,
        "semantic_dispatch_coverage_audit_verification": audit_verification,
        "semantic_dispatch_coverage_benchmark_verification": benchmark_verification,
        "semantic_dispatch_coverage_derivation_sha256": sha256_file(Path(__file__)),
        "pinned_predecessor_hash_schema_and_boundary_verified": True,
        "deep_predecessor_replay_performed": False,
    }
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
    boundary["weights_loaded_or_mutated"] = False
    boundary["sae_dictionary_training_authorized"] = False
    boundary["execution_authorized"] = False
    boundary["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    if observed != build_protocol(**kwargs):
        raise ValueError("research protocol V47 is not reproducible")
    component = observed["component_compartments"][
        "cluster_aware_sae_semantic_dispatch_coverage_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v47",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "semantic_dispatch_coverage_audit_id": component["audit"][
            "semantic_dispatch_coverage_audit_id"
        ],
        "semantic_dispatch_coverage_benchmark_id": component["benchmark"]
        ["benchmark_id"],
        "controls": component["benchmark"]["controls"]["controls"],
        "schema_and_id_matching_nodes": 12,
        "dispatch_ready_nodes_under_v1_graph": 4,
        "exact_implemented_field_edges": 9,
        "mismatched_edges": 28,
        "semantic_dispatch_allowed": False,
        "empirical_graph_present": False,
        "training_performed": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v46.json"
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
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V47.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.output is not None:
        result = build_protocol(**inputs)
        write_json(args.output, result)
        protocol_sha256 = sha256_file(args.output)
    else:
        result = verify_protocol(args.verify, **inputs)
        protocol_sha256 = result["protocol_sha256"]
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result.get("status", "derived"),
                "protocol_sha256": protocol_sha256,
                "execution_authorized": result.get("execution_authorized", False),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
