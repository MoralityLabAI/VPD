from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_capture_campaign_guard_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v29 import (
    canonical_inputs as protocol_v29_inputs,
    verify_protocol as verify_protocol_v29,
)


SCHEMA = "sch.vpd-activation-category-protocol.v30"
EXPECTED_V29_SHA256 = (
    "8dfdfe72a6fd245d4d5a9b4d5612a1463bada0cdc3d2ae61afe32a342bef4a8b"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V30 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, *, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V29_SHA256:
        raise ValueError("protocol V29 differs from its pinned SHA-256")
    verification = verify_protocol_v29(path, **protocol_v29_inputs(project_root))
    if verification["execution_authorized"] is not False:
        raise ValueError("protocol V29 authorization boundary differs")
    return verification


def build_protocol(
    *,
    predecessor_path: Path,
    campaign_guard_path: Path,
    campaign_guard_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(
        predecessor_path, project_root=project_root
    )
    benchmark_verification = verify_benchmark(
        campaign_guard_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    benchmark = json.loads(campaign_guard_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["cluster_capture_campaign_guard_v1"] = {
        "schema": "sch.vpd-cluster-capture-campaign-guard-contract.v1",
        "status": "guard_ready_simulated_counters_only_real_capture_absent",
        "guard": {
            "path": _relative(campaign_guard_path, project_root),
            "sha256": sha256_file(campaign_guard_path),
            "receipt_schema": "sch.vpd-cluster-capture-campaign-guard-receipt.v1",
        },
        "benchmark": {
            "path": _relative(campaign_guard_benchmark_path, project_root),
            "sha256": sha256_file(campaign_guard_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
        },
        "role_prerequisites": benchmark["results"]["role_prerequisites"],
        "counter_fields": [
            "compute_jobs",
            "batch_forwards",
            "episode_forward_equivalents",
            "float32_effect_payload_bytes",
        ],
        "counter_increment_must_precede_compute": True,
        "all_role_ceilings_sum_to_global_ceiling": True,
        "wall_time_and_cost_must_be_monotonic_and_bounded": True,
        "generated_tokens_real_actions_and_persistent_mutations_forbidden": True,
        "final_evaluation_role_forbidden": True,
        "runtime_admission_and_output_reservation_required_for_production": True,
        "real_guard_receipt_present": False,
        "capture_started": False,
        "empirical_capture_ready": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    result["unified_lineage"]["cluster_capture_campaign_guard_ready"] = True
    result["unified_lineage"]["campaign_guard_controls"] = benchmark["summary"][
        "controls"
    ]
    result["unified_lineage"]["four_role_counter_partition_exact"] = True
    result["unified_lineage"]["real_guard_receipt_present"] = False
    result["unified_lineage"]["capture_started"] = False
    result["unified_lineage"]["empirical_capture_ready"] = False
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v29_verification": predecessor_verification,
        "campaign_guard_benchmark_verification": benchmark_verification,
        "campaign_guard_sha256": sha256_file(campaign_guard_path),
        "campaign_guard_derivation_sha256": sha256_file(Path(__file__)),
        "campaign_meter_is_distinct_from_model_worker": True,
        "bounded_predecessor_replay": True,
    }
    result["scientific_boundary"]["synthetic_campaign_counter_fixture_only"] = True
    result["scientific_boundary"]["real_guard_receipt_present"] = False
    result["scientific_boundary"]["persistent_output_root_created"] = False
    result["scientific_boundary"]["capture_started"] = False
    result["scientific_boundary"]["model_framework_imported"] = False
    result["scientific_boundary"]["model_or_adapter_weights_opened"] = False
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["weights_loaded_or_mutated"] = False
    result["scientific_boundary"]["sae_dictionary_training_authorized"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != build_protocol(**kwargs):
        raise ValueError("research protocol V30 is not reproducible")
    contract = observed["component_compartments"]["cluster_capture_campaign_guard_v1"]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v30",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "campaign_guard_benchmark_id": contract["benchmark"]["benchmark_id"],
        "campaign_guard_controls": contract["benchmark"]["controls"]["controls"],
        "roles": len(contract["role_prerequisites"]),
        "real_guard_receipt_present": False,
        "capture_started": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v29.json"
        ),
        "campaign_guard_path": (
            project_root
            / "experiments/authz_mechinterp/vpd_cluster_capture_campaign_guard_v1.py"
        ),
        "campaign_guard_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_capture_campaign_guard_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V30.")
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
