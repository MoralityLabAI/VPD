from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_inert_campaign_v1 import (
    verify_benchmark as verify_campaign_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v41 import (
    canonical_inputs as protocol_v41_inputs,
    verify_protocol as verify_protocol_v41,
)


SCHEMA = "sch.vpd-activation-category-protocol.v42"
EXPECTED_V41_SHA256 = (
    "87c7b79aac37bc094ddfe06501f020eda9857c5ef08767eff492f5c0cb21476b"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V42 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, *, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V41_SHA256:
        raise ValueError("protocol V41 differs from its pinned SHA-256")
    verification = verify_protocol_v41(path, **protocol_v41_inputs(project_root))
    if verification["execution_authorized"] is not False:
        raise ValueError("protocol V41 authorization boundary differs")
    return verification


def build_protocol(
    *,
    predecessor_path: Path,
    campaign_execution_path: Path,
    campaign_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(
        predecessor_path, project_root=project_root
    )
    benchmark_verification = verify_campaign_benchmark(
        campaign_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    benchmark = json.loads(campaign_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["cluster_aware_sae_inert_campaign_v1"] = {
        "schema": "sch.vpd-cluster-aware-sae-inert-campaign-contract.v1",
        "status": (
            "exact_34_job_synthetic_campaign_success_and_terminal_failure_"
            "receipts_ready_production_campaign_empirical_inputs_and_training_absent"
        ),
        "execution": {
            "path": _relative(campaign_execution_path, project_root),
            "sha256": sha256_file(campaign_execution_path),
            "success_schema": (
                "sch.vpd-cluster-aware-sae-synthetic-inert-campaign.v1"
            ),
            "failure_schema": (
                "sch.vpd-cluster-aware-sae-synthetic-inert-campaign-failure.v1"
            ),
        },
        "benchmark": {
            "path": _relative(campaign_benchmark_path, project_root),
            "sha256": sha256_file(campaign_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
        },
        "activation_categories": 8,
        "compartments": 17,
        "specific_compartments": 16,
        "shared_fallback_compartments": 1,
        "execution_jobs": 34,
        "unique_output_directories": 34,
        "one_global_pair_split": True,
        "exact_job_coverage_required_before_campaign_claim": True,
        "campaign_and_per_job_claims_precede_callbacks": True,
        "success_callbacks_charged": 34,
        "controlled_failure_callbacks_charged": 8,
        "controlled_failure_remaining_uninvoked": 26,
        "synthetic_inert_campaign_executed": True,
        "production_campaign_executed": False,
        "production_trainer_callback_invoked": False,
        "training_performed": False,
        "persistent_dictionary_output_created": False,
        "empirical_blueprint_present": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    lineage = result["unified_lineage"]
    lineage["cluster_aware_sae_inert_campaign_ready"] = True
    lineage["exact_synthetic_campaign_job_coverage"] = True
    lineage["synthetic_inert_campaign_executed"] = True
    lineage["synthetic_campaign_success_callbacks_charged"] = 34
    lineage["synthetic_campaign_failure_callbacks_charged"] = 8
    lineage["production_campaign_executed"] = False
    lineage["production_trainer_callback_invoked"] = False
    lineage["empirical_cluster_aware_blueprint_present"] = False
    lineage["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v41_verification": predecessor_verification,
        "inert_campaign_benchmark_verification": benchmark_verification,
        "inert_campaign_derivation_sha256": sha256_file(Path(__file__)),
        "bounded_predecessor_replay": True,
    }
    boundary = result["scientific_boundary"]
    boundary["temporary_inert_campaign_fixture_only"] = True
    boundary["cluster_aware_inert_campaign_ready"] = True
    boundary["exact_synthetic_campaign_job_coverage"] = True
    boundary["synthetic_inert_campaign_executed"] = True
    boundary["production_campaign_executed"] = False
    boundary["production_trainer_callback_invoked"] = False
    boundary["training_performed"] = False
    boundary["persistent_dictionary_output_created"] = False
    boundary["empirical_cluster_aware_blueprint_present"] = False
    boundary["model_framework_imported"] = False
    boundary["model_or_adapter_weights_opened"] = False
    boundary["model_inference_performed"] = False
    boundary["weights_loaded_or_mutated"] = False
    boundary["sae_dictionary_training_authorized"] = False
    boundary["execution_authorized"] = False
    boundary["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != build_protocol(**kwargs):
        raise ValueError("research protocol V42 is not reproducible")
    component = observed["component_compartments"][
        "cluster_aware_sae_inert_campaign_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v42",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "inert_campaign_benchmark_id": component["benchmark"]["benchmark_id"],
        "inert_campaign_controls": component["benchmark"]["controls"]["controls"],
        "categories": component["activation_categories"],
        "compartments": component["compartments"],
        "jobs": component["execution_jobs"],
        "unique_output_directories": component["unique_output_directories"],
        "success_callbacks_charged": component["success_callbacks_charged"],
        "controlled_failure_callbacks_charged": component[
            "controlled_failure_callbacks_charged"
        ],
        "synthetic_inert_campaign_executed": True,
        "production_campaign_executed": False,
        "production_trainer_callback_invoked": False,
        "training_performed": False,
        "persistent_dictionary_output_created": False,
        "empirical_blueprint_present": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v41.json"
        ),
        "campaign_execution_path": (
            project_root
            / "experiments/authz_mechinterp/execute_cluster_aware_sae_inert_campaign_v1.py"
        ),
        "campaign_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_inert_campaign_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V42.")
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
