from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v10 import (
    canonical_inputs as protocol_v10_inputs,
    verify_protocol as verify_protocol_v10,
)
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v2 import (
    canonical_inputs as runbook_v2_inputs,
    verify_runbook as verify_runbook_v2,
)
from experiments.authz_mechinterp.verify_unified_vpd_compartment_pipeline_v7 import (
    canonical_inputs as receipt_v7_inputs,
    verify_receipt as verify_receipt_v7,
)


SCHEMA = "sch.vpd-unified-compartment-pipeline-receipt.v8"


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("V10 unified lineage source escapes the project")
    return resolved.relative_to(root.resolve()).as_posix()


def build_receipt(
    *,
    predecessor_receipt_path: Path,
    protocol_path: Path,
    runbook_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = verify_receipt_v7(
        predecessor_receipt_path, **receipt_v7_inputs(project_root)
    )
    protocol_verification = verify_protocol_v10(
        protocol_path, **protocol_v10_inputs(project_root)
    )
    runbook_verification = verify_runbook_v2(
        runbook_path, **runbook_v2_inputs(project_root)
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    runbook = json.loads(runbook_path.read_text(encoding="utf-8"))
    contract = protocol["component_compartments"][
        "within_category_cluster_discreteness_v2"
    ]
    runbook_benchmark = runbook["source"]["cluster_discreteness_benchmark_v2"]
    if contract["benchmark"]["benchmark_id"] != runbook_benchmark["benchmark_id"]:
        raise ValueError("protocol V10 and runbook V2 benchmark lineage differs")
    if contract["selector"]["sha256"] != runbook["source"][
        "cluster_count_selector_v2"
    ]["sha256"]:
        raise ValueError("protocol V10 and runbook V2 selector lineage differs")
    body = {
        "schema": SCHEMA,
        "status": "pass_local_cluster_discreteness_v2_benchmarked_runtime_pending",
        "source": {
            "predecessor_unified_receipt_v7": {
                "path": _relative(predecessor_receipt_path, project_root),
                "sha256": sha256_file(predecessor_receipt_path),
                "pipeline_receipt_id": predecessor_verification[
                    "pipeline_receipt_id"
                ],
                "verification": predecessor_verification,
            },
            "research_protocol_v10": {
                "path": _relative(protocol_path, project_root),
                "sha256": sha256_file(protocol_path),
                "verification": protocol_verification,
            },
            "cheap_executor_runbook_v2": {
                "path": _relative(runbook_path, project_root),
                "sha256": sha256_file(runbook_path),
                "runbook_id": runbook["runbook_id"],
                "verification": runbook_verification,
            },
            "cluster_discreteness_benchmark_v2": json.loads(
                json.dumps(runbook_benchmark)
            ),
            "cluster_count_selector_v2": json.loads(
                json.dumps(runbook["source"]["cluster_count_selector_v2"])
            ),
        },
        "lineage_amendment": {
            "predecessor_receipt_v7_preserved_and_reverified": True,
            "research_protocol_v10_is_not_historical_experiment_v10": True,
            "legacy_continuous_false_split_reproduced": True,
            "continuous_direction_arc_negative_control_present": True,
            "continuous_strength_gradient_negative_control_present": True,
            "discrete_two_mechanism_positive_control_present": True,
            "next_empirical_candidate_requires_selection_v2": True,
            "v1_only_cluster_count_selection_prohibited_for_next_candidate": True,
            "production_v2_candidate_wrapper_and_admission_still_required": True,
            "historical_v7_release_rejection_unchanged": True,
        },
        "resources": {
            "synthetic_scenarios": int(contract["benchmark"]["scenarios"]),
            "legacy_false_split_scenarios": int(
                contract["benchmark"]["legacy_false_splits_reproduced"]
            ),
            "cheap_executor_stages": len(runbook["stages"]),
            "safe_stage_now": 0,
        },
        "invariants": {
            "predecessor_receipt_v7_reverified": True,
            "research_protocol_v10_reverified": True,
            "cheap_executor_runbook_v2_reverified": True,
            "cluster_discreteness_benchmark_reverified": True,
            "cluster_selector_hash_joined": True,
            "execution_not_authorized": True,
        },
        "scientific_boundary": {
            "model_inference_performed": False,
            "activation_capture_performed": False,
            "sae_training_performed": False,
            "weights_loaded_or_mutated": False,
            "empirical_cluster_discreteness_audit_present": False,
            "production_cluster_candidate_v2_runtime_present": False,
            "scientific_release_receipt_present": False,
            "external_compute_authorization_present": False,
            "execution_authorized": False,
        },
    }
    return {**body, "pipeline_receipt_id": content_sha256(body)}


def verify_receipt(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("pipeline_receipt_id", ""))
    if content_sha256(body) != identifier:
        raise ValueError("V10 unified pipeline receipt content address mismatch")
    expected = build_receipt(**kwargs)
    if observed != expected:
        raise ValueError("V10 unified pipeline receipt is not reproducible")
    return {
        "schema": "sch.vpd-unified-compartment-pipeline-verification.v8",
        "status": "pass",
        "pipeline_receipt_id": identifier,
        "legacy_false_split_scenarios": observed["resources"][
            "legacy_false_split_scenarios"
        ],
        "production_cluster_candidate_v2_runtime_present": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_receipt_path": project_root
        / "artifacts/vpd_unified_compartment_pipeline_v7/receipt.json",
        "protocol_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v10.json",
        "runbook_path": project_root
        / "artifacts/vpd_cheap_executor_runbook_v2/runbook.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the protocol V10 cluster-discreteness lineage."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None:
        result = verify_receipt(args.verify, **inputs)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_receipt(**inputs)
        write_json(args.output, result)
    print(
        json.dumps(
            {
                "schema": str(result["schema"]),
                "status": str(result["status"]),
                "pipeline_receipt_id": str(result["pipeline_receipt_id"]),
                "execution_authorized": bool(
                    result.get("execution_authorized", False)
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
