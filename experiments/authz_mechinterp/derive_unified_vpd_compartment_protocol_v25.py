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
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json


SCHEMA = "sch.vpd-activation-category-protocol.v25"
EXPECTED_V24_SHA256 = (
    "c42dc6e43296c7e4a3d6054174e015620221cf141377bff44773d8cfc0add2b2"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V25 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V24_SHA256:
        raise ValueError("protocol V24 differs from its pinned SHA-256")
    predecessor = json.loads(path.read_text(encoding="utf-8"))
    if predecessor.get("schema") != "sch.vpd-activation-category-protocol.v24":
        raise ValueError("pinned predecessor is not protocol V24")
    if predecessor["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("protocol V24 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-predecessor-verification.v1",
        "status": "pass",
        "method": "exact_sha256_and_boundary_fields",
        "sha256": EXPECTED_V24_SHA256,
    }


def build_protocol(
    *,
    predecessor_path: Path,
    five_role_audit_path: Path,
    empirical_preflight_path: Path,
    empirical_preflight_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    audit_verification = verify_audit(
        five_role_audit_path, project_root=project_root
    )
    preflight_verification = verify_preflight(
        empirical_preflight_path, project_root=project_root
    )
    benchmark_verification = verify_preflight_benchmark(
        empirical_preflight_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    audit = json.loads(five_role_audit_path.read_text(encoding="utf-8"))
    preflight = json.loads(empirical_preflight_path.read_text(encoding="utf-8"))
    benchmark = json.loads(
        empirical_preflight_benchmark_path.read_text(encoding="utf-8")
    )
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["cluster_to_sae_empirical_preflight_v1"] = {
        "schema": "sch.vpd-cluster-to-sae-empirical-preflight-contract.v1",
        "status": "model_free_preflight_verified_execution_blocked_on_13_bindings",
        "five_role_incidence_audit": {
            "path": _relative(five_role_audit_path, project_root),
            "sha256": sha256_file(five_role_audit_path),
            "audit_id": audit["audit_id"],
            "verification": audit_verification,
            "roles": len(audit["role_summaries"]),
            "rows": audit["aggregate"]["rows"],
            "pairs": audit["aggregate"]["pairs"],
            "template_groups": audit["aggregate"]["template_groups"],
            "pairwise_identity_overlap_zero": True,
        },
        "empirical_preflight": {
            "path": _relative(empirical_preflight_path, project_root),
            "sha256": sha256_file(empirical_preflight_path),
            "preflight_id": preflight["preflight_id"],
            "verification": preflight_verification,
            "capture_roles": list(preflight["capture_roles"]),
            "stage_order": preflight["stage_order"],
            "missing_required_bindings": preflight["missing_required_bindings"],
            "resource_ceiling": preflight["resource_ceiling"],
        },
        "preflight_benchmark": {
            "path": _relative(empirical_preflight_benchmark_path, project_root),
            "sha256": sha256_file(empirical_preflight_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"]["controls"],
        },
        "cluster_selection_roles": [
            "discovery",
            "calibration",
            "cluster_validation",
        ],
        "sae_routing_extension_role": "sae_training",
        "final_evaluation_excluded": True,
        "preflight_imports_model_frameworks": False,
        "preflight_opens_model_or_adapter_weights": False,
        "empirical_capture_receipts_present": False,
        "empirical_partition_admission_present": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    result["unified_lineage"]["five_role_recovered_incidence_audit_ready"] = True
    result["unified_lineage"]["cluster_to_sae_empirical_preflight_v1_ready"] = True
    result["unified_lineage"]["cluster_to_sae_preflight_controls"] = 15
    result["unified_lineage"]["cluster_to_sae_missing_required_bindings"] = 13
    result["unified_lineage"]["empirical_capture_ready"] = False
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v24_verification": predecessor_verification,
        "five_role_recovered_incidence_audit_verification": audit_verification,
        "cluster_to_sae_empirical_preflight_verification": preflight_verification,
        "cluster_to_sae_empirical_preflight_benchmark_verification": benchmark_verification,
        "bounded_predecessor_replay": True,
        "research_protocol_v25_is_distinct_from_historical_experiment_labels": True,
    }
    result["scientific_boundary"]["model_framework_imported"] = False
    result["scientific_boundary"]["model_or_adapter_weights_opened"] = False
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["real_effect_archive_emitted"] = False
    result["scientific_boundary"]["final_evaluation_consumed"] = False
    result["scientific_boundary"]["weights_loaded_or_mutated"] = False
    result["scientific_boundary"]["sae_dictionary_training_authorized"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != build_protocol(**kwargs):
        raise ValueError("research protocol V25 is not reproducible")
    contract = observed["component_compartments"][
        "cluster_to_sae_empirical_preflight_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v25",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "five_role_audit_id": contract["five_role_incidence_audit"]["audit_id"],
        "empirical_preflight_id": contract["empirical_preflight"]["preflight_id"],
        "preflight_benchmark_id": contract["preflight_benchmark"]["benchmark_id"],
        "missing_required_bindings": len(
            contract["empirical_preflight"]["missing_required_bindings"]
        ),
        "empirical_capture_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v24.json"
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
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V25.")
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
