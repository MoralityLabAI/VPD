from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.audit_cluster_aware_sae_semantic_dispatch_coverage_v1 import (
    build_audit,
    canonical_inputs,
    verify_audit,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def build_report(project_root: Path) -> dict[str, Any]:
    before_frameworks = MODEL_FRAMEWORKS & set(sys.modules)
    audit_path = (
        project_root
        / "artifacts/vpd_cluster_aware_sae_semantic_dispatch_coverage_audit_v1/audit.json"
    )
    inputs = canonical_inputs(project_root)
    verification = verify_audit(audit_path, **inputs)
    audit = _load(audit_path)
    rebuilt = build_audit(**inputs)
    nodes = {row["node"]: row for row in audit["node_rows"]}
    edges = {row["edge"]: row for row in audit["edge_rows"]}
    tampered = copy.deepcopy(audit)
    tampered["summary"]["dispatch_ready_nodes_under_v1_graph"] += 1
    tampered_body = {
        key: value
        for key, value in tampered.items()
        if key != "semantic_dispatch_coverage_audit_id"
    }
    controls = {
        "audit_reproduces": verification["status"] == "pass"
        and audit == rebuilt,
        "all_20_graph_nodes_audited": verification["graph_nodes"] == 20,
        "exactly_12_node_schema_and_id_shapes_match": verification[
            "schema_and_id_matching_nodes"
        ]
        == 12,
        "only_4_nodes_are_dispatch_ready_under_v1": verification[
            "dispatch_ready_nodes_under_v1_graph"
        ]
        == 4,
        "all_37_graph_edges_audited": verification["graph_edges"] == 37,
        "only_9_field_edges_match_implementation": verification[
            "exact_implemented_field_edges"
        ]
        == 9,
        "remaining_28_edges_are_explicit_mismatches": verification[
            "mismatched_edges"
        ]
        == 28,
        "all_referenced_verifier_callables_exist": all(
            row["verifier"]["callable_exists"] is True
            for row in audit["node_rows"]
        ),
        "capture_receipt_id_mismatch_is_explicit": nodes[
            "capture_authorization_receipt"
        ]["graph_id_field"]
        == "receipt_id"
        and nodes["capture_authorization_receipt"]["implemented_id_field"]
        == "authorization_receipt_id",
        "capture_manifest_id_and_sidecar_gap_is_explicit": nodes[
            "empirical_capture_manifest"
        ]["implemented_id_field"]
        == "capture_id"
        and "sidecars" in nodes["empirical_capture_manifest"]["gap"],
        "routing_bundle_id_mismatch_is_explicit": nodes[
            "empirical_routing_bundle"
        ]["implemented_id_field"]
        == "routing_evidence_bundle_id",
        "global_split_embedding_is_explicit": nodes["global_pair_split"]
        ["verification_mode"]
        == "embedded_inside_cluster_aware_plan",
        "budget_blueprint_binding_gap_is_explicit": "blueprint" in nodes[
            "budget_proposal"
        ]["gap"]
        and "blueprint" in nodes["budget_admission"]["gap"],
        "compute_receipt_representation_gap_is_explicit": nodes[
            "compute_authorization_manifest"
        ]["implemented_schema"]
        == "sch.vpd-cluster-aware-sae-compute-authorization-receipt.v1"
        and nodes["compute_authorization_manifest"]["implemented_id_field"]
        == "receipt_id",
        "durable_transition_is_one_combined_receipt": len(
            {
                nodes[name]["implemented_schema"]
                for name in (
                    "nonce_consumption_manifest",
                    "output_reservation_manifest",
                    "worker_ledger_claim",
                )
            }
        )
        == 1
        and nodes["worker_ledger_claim"]["implemented_id_field"]
        == "transition_reservation_id",
        "production_runtime_attachment_gap_is_explicit": verification[
            "production_runtime_attachment_verifier_present"
        ]
        is False
        and "synthetic inert" in nodes["runtime_attachment_claim"]["gap"],
        "capture_preflight_id_edge_is_one_of_exact_nine": edges[
            "capture_receipt_to_preflight_id"
        ]["implemented_field_path_and_target_match"]
        is True,
        "capture_preflight_file_hash_edge_is_rejected_as_representation_mismatch": edges[
            "capture_receipt_to_preflight_hash"
        ]["implemented_field_path_and_target_match"]
        is False,
        "successor_corrections_cover_capture_budget_compute_transition_and_runtime": all(
            any(term in correction for correction in audit["required_successor_corrections"])
            for term in ("guarded V3", "budget", "compute", "transition", "production")
        ),
        "audit_content_tamper_changes_address": tampered[
            "semantic_dispatch_coverage_audit_id"
        ]
        != content_sha256(tampered_body),
        "semantic_dispatch_remains_disallowed": audit["summary"][
            "semantic_dispatch_allowed"
        ]
        is False,
        "audit_transfers_no_authority": all(
            audit["scientific_boundary"][field] is False
            for field in (
                "v1_semantic_graph_verified",
                "empirical_graph_present",
                "semantic_verifiers_invoked_on_empirical_inputs",
                "model_framework_imported",
                "model_loaded",
                "model_inference_performed",
                "training_performed",
                "weights_loaded_or_mutated",
                "persistent_dictionary_output_created",
                "execution_authorized",
                "sae_training_authorized",
                "paid_compute_authorized",
            )
        ),
        "benchmark_imports_no_model_frameworks": before_frameworks
        == (MODEL_FRAMEWORKS & set(sys.modules)),
    }
    if not all(controls.values()):
        raise ValueError(
            "semantic dispatch coverage benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-aware-sae-semantic-dispatch-coverage-benchmark.v1",
        "status": "pass_fail_closed_gap_detection",
        "source": {
            "audit_path": audit_path.relative_to(project_root).as_posix(),
            "audit_sha256": sha256_file(audit_path),
            "semantic_dispatch_coverage_audit_id": audit[
                "semantic_dispatch_coverage_audit_id"
            ],
            "audit_builder_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/audit_cluster_aware_sae_semantic_dispatch_coverage_v1.py"
            ),
        },
        "controls": controls,
        "summary": {
            "passed": sum(controls.values()),
            "controls": len(controls),
            **audit["summary"],
        },
        "scientific_boundary": {
            "gap_detection_only": True,
            "semantic_dispatch_allowed": False,
            "empirical_graph_present": False,
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "training_performed": False,
            "weights_loaded_or_mutated": False,
            "persistent_dictionary_output_created": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
            "paid_compute_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body) or observed != build_report(project_root):
        raise ValueError("semantic dispatch coverage benchmark is not reproducible")
    return {
        "schema": "sch.vpd-cluster-aware-sae-semantic-dispatch-coverage-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "semantic_dispatch_coverage_audit_id": observed["source"][
            "semantic_dispatch_coverage_audit_id"
        ],
        "controls": observed["summary"]["controls"],
        "graph_nodes": observed["summary"]["graph_nodes"],
        "schema_and_id_matching_nodes": observed["summary"]
        ["schema_and_id_matching_nodes"],
        "dispatch_ready_nodes_under_v1_graph": observed["summary"]
        ["dispatch_ready_nodes_under_v1_graph"],
        "graph_edges": observed["summary"]["graph_edges"],
        "exact_implemented_field_edges": observed["summary"]
        ["exact_implemented_field_edges"],
        "semantic_dispatch_allowed": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify semantic-dispatch coverage benchmark V1."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.output is not None:
        result = build_report(root)
        write_json(args.output, result)
    else:
        result = verify_benchmark(args.verify, project_root=root)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "benchmark_id": result.get("benchmark_id"),
                "controls": result.get("controls", result.get("summary", {}).get("controls")),
                "semantic_dispatch_allowed": result.get(
                    "semantic_dispatch_allowed",
                    result.get("scientific_boundary", {}).get(
                        "semantic_dispatch_allowed"
                    ),
                ),
                "execution_authorized": result.get(
                    "execution_authorized",
                    result.get("scientific_boundary", {}).get("execution_authorized"),
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
