from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_semantic_dispatch_slice_v2 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json


SCHEMA = "sch.vpd-activation-category-protocol.v50"
EXPECTED_V49_SHA256 = (
    "0c3798bdca4dd501e8f674afdfb368319ebb6010a4e8ea4f82aff92afec48fbb"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V50 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V49_SHA256:
        raise ValueError("protocol V49 differs from its pinned SHA-256")
    observed = _load(path)
    if observed.get("schema") != "sch.vpd-activation-category-protocol.v49":
        raise ValueError("pinned predecessor is not protocol V49")
    if observed.get("scientific_boundary", {}).get("execution_authorized") is not False:
        raise ValueError("protocol V49 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-protocol-predecessor-verification.v1",
        "status": "pass",
        "protocol_sha256": EXPECTED_V49_SHA256,
        "schema_verified": True,
        "non_authorizing_boundary_verified": True,
        "deep_predecessor_replay_performed": False,
    }


def build_protocol(
    *,
    predecessor_path: Path,
    release_runtime_verifier_path: Path,
    dispatcher_path: Path,
    benchmark_path: Path,
    test_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    benchmark_verification = verify_benchmark(
        benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    benchmark = _load(benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"][
        "cluster_aware_sae_semantic_dispatch_slice_v2"
    ] = {
        "schema": "sch.vpd-cluster-aware-sae-semantic-dispatch-slice-lineage.v2",
        "status": (
            "synthetic_budget_bound_signed_release_and_runtime_admission_verified_"
            "durable_transition_and_production_attachment_absent"
        ),
        "release_and_runtime_verifier": {
            "path": _relative(release_runtime_verifier_path, project_root),
            "sha256": sha256_file(release_runtime_verifier_path),
        },
        "dispatcher": {
            "path": _relative(dispatcher_path, project_root),
            "sha256": sha256_file(dispatcher_path),
        },
        "benchmark": {
            "path": _relative(benchmark_path, project_root),
            "sha256": sha256_file(benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "semantic_dispatch_slice_id": benchmark["results"][
                "semantic_dispatch_slice_id"
            ],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
        },
        "test": {
            "path": _relative(test_path, project_root),
            "sha256": sha256_file(test_path),
        },
        "dispatched_nodes": benchmark["summary"]["dispatched_nodes"],
        "full_reproduction_comparisons": benchmark["summary"][
            "full_reproduction_comparisons"
        ],
        "cryptographic_receipt_verifications": benchmark["summary"][
            "cryptographic_receipt_verifications"
        ],
        "execution_jobs": benchmark["summary"]["execution_jobs"],
        "closed_terminal_joins": [
            "budget_admission_to_signed_release",
            "budget_admission_to_runtime_admission",
        ],
        "remaining_terminal_gaps": [
            "runtime_admission_to_durable_transition",
            "production_runtime_attachment",
        ],
        "terminal_gap_count": benchmark["summary"]["terminal_gap_count"],
        "synthetic_fixture_persisted": False,
        "synthetic_signed_receipts_verified": True,
        "real_signed_authority_dispatched": False,
        "empirical_graph_present": False,
        "end_to_end_semantic_dispatch_complete": False,
        "nonce_consumed": False,
        "output_reserved": False,
        "durable_transition_dispatched": False,
        "production_runtime_attachment_dispatched": False,
        "training_performed": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    lineage = result["unified_lineage"]
    lineage["cluster_aware_sae_semantic_dispatch_slice_v2_verified"] = True
    lineage["semantic_dispatch_slice_nodes"] = 7
    lineage["semantic_dispatch_full_reproduction_comparisons"] = 5
    lineage["semantic_dispatch_cryptographic_receipt_verifications"] = 2
    lineage["semantic_dispatch_execution_jobs"] = 34
    lineage["semantic_dispatch_closed_terminal_joins"] = 2
    lineage["semantic_dispatch_terminal_gap_count"] = 2
    lineage["synthetic_signed_receipts_verified"] = True
    lineage["real_signed_authority_dispatched"] = False
    lineage["semantic_dispatch_synthetic_fixture_persisted"] = False
    lineage["semantic_dispatch_empirical_graph_present"] = False
    lineage["end_to_end_semantic_dispatch_complete"] = False
    lineage["nonce_consumed"] = False
    lineage["output_reserved"] = False
    lineage["durable_transition_dispatched"] = False
    lineage["production_runtime_attachment_dispatched"] = False
    lineage["production_trainer_callback_invoked"] = False
    lineage["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v49_verification": predecessor_verification,
        "semantic_dispatch_slice_v2_benchmark_verification": benchmark_verification,
        "semantic_dispatch_slice_v2_derivation_sha256": sha256_file(Path(__file__)),
        "pinned_predecessor_hash_schema_and_boundary_verified": True,
        "deep_predecessor_replay_performed": False,
    }
    boundary = result["scientific_boundary"]
    boundary["cluster_aware_sae_semantic_dispatch_slice_v2_verified"] = True
    boundary["semantic_dispatch_slice_nodes"] = 7
    boundary["semantic_dispatch_full_reproduction_comparisons"] = 5
    boundary["semantic_dispatch_cryptographic_receipt_verifications"] = 2
    boundary["semantic_dispatch_closed_terminal_joins"] = 2
    boundary["semantic_dispatch_terminal_gap_count"] = 2
    boundary["synthetic_signed_receipts_verified"] = True
    boundary["real_signed_authority_dispatched"] = False
    boundary["semantic_dispatch_synthetic_fixture_persisted"] = False
    boundary["semantic_dispatch_allowed"] = False
    boundary["end_to_end_semantic_dispatch_complete"] = False
    boundary["nonce_consumed"] = False
    boundary["output_reserved"] = False
    boundary["durable_transition_dispatched"] = False
    boundary["production_runtime_attachment_dispatched"] = False
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
        raise ValueError("research protocol V50 is not reproducible")
    component = observed["component_compartments"][
        "cluster_aware_sae_semantic_dispatch_slice_v2"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v50",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "semantic_dispatch_benchmark_id": component["benchmark"]["benchmark_id"],
        "semantic_dispatch_slice_id": component["benchmark"][
            "semantic_dispatch_slice_id"
        ],
        "controls": component["benchmark"]["controls"]["controls"],
        "dispatched_nodes": component["dispatched_nodes"],
        "full_reproduction_comparisons": component[
            "full_reproduction_comparisons"
        ],
        "cryptographic_receipt_verifications": component[
            "cryptographic_receipt_verifications"
        ],
        "execution_jobs": component["execution_jobs"],
        "closed_terminal_joins": len(component["closed_terminal_joins"]),
        "terminal_gap_count": component["terminal_gap_count"],
        "semantic_dispatch_allowed": False,
        "empirical_graph_present": False,
        "training_performed": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v49.json",
        "release_runtime_verifier_path": project_root
        / "experiments/authz_mechinterp/admit_cluster_aware_sae_training_v2.py",
        "dispatcher_path": project_root
        / "experiments/authz_mechinterp/dispatch_cluster_aware_sae_semantic_slice_v2.py",
        "benchmark_path": project_root
        / "artifacts/vpd_cluster_aware_sae_semantic_dispatch_slice_benchmark_v2/report.json",
        "test_path": project_root
        / "tests/test_vpd_cluster_aware_sae_semantic_dispatch_slice_v2.py",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V50.")
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
