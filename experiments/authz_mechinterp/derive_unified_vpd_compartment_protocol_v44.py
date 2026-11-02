from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_budget_admission_policy_v1 import (
    verify_benchmark as verify_budget_benchmark,
)
from experiments.authz_mechinterp.build_cluster_aware_sae_budget_admission_policy_v1 import (
    canonical_inputs as budget_policy_inputs,
    verify_budget_policy,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v43 import (
    canonical_inputs as protocol_v43_inputs,
    verify_protocol as verify_protocol_v43,
)


SCHEMA = "sch.vpd-activation-category-protocol.v44"
EXPECTED_V43_SHA256 = (
    "b7d2f07b19295176c5afcd06349c07dcb2f5a4aff5679d2c74f493744f5626e5"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V44 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, *, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V43_SHA256:
        raise ValueError("protocol V43 differs from its pinned SHA-256")
    verification = verify_protocol_v43(path, **protocol_v43_inputs(project_root))
    if verification["execution_authorized"] is not False:
        raise ValueError("protocol V43 authorization boundary differs")
    return verification


def build_protocol(
    *,
    predecessor_path: Path,
    budget_policy_builder_path: Path,
    budget_policy_path: Path,
    budget_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(
        predecessor_path, project_root=project_root
    )
    policy_verification = verify_budget_policy(
        budget_policy_path, **budget_policy_inputs(project_root)
    )
    benchmark_verification = verify_budget_benchmark(
        budget_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    benchmark = json.loads(budget_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"][
        "cluster_aware_sae_budget_admission_policy_v1"
    ] = {
        "schema": "sch.vpd-cluster-aware-sae-budget-admission-contract.v1",
        "status": (
            "sealed_nonexecuting_category_compartment_and_shared_fallback_"
            "ceilings_ready_empirical_inputs_training_and_execution_absent"
        ),
        "builder": {
            "path": _relative(budget_policy_builder_path, project_root),
            "sha256": sha256_file(budget_policy_builder_path),
        },
        "policy": {
            "path": _relative(budget_policy_path, project_root),
            "sha256": sha256_file(budget_policy_path),
            "budget_policy_id": policy_verification["budget_policy_id"],
            "verification": policy_verification,
        },
        "benchmark": {
            "path": _relative(budget_benchmark_path, project_root),
            "sha256": sha256_file(budget_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
        },
        "activation_categories": 8,
        "compartments": 17,
        "specific_compartments": 16,
        "shared_fallback_compartments": 1,
        "jobs_ceiling": 34,
        "planned_steps_ceiling": 68000,
        "planned_activation_rows_ceiling": 20480,
        "specific_dictionary_features": 4096,
        "shared_fallback_dictionary_features": 8192,
        "shared_fallback_charged_to_categories": False,
        "budget_admission_transfers_execution_authority": False,
        "fresh_empirical_blueprint_required": True,
        "fresh_active_signed_receipts_required": True,
        "fresh_nonces_and_output_reservations_required": True,
        "retry_authorized": False,
        "training_performed": False,
        "empirical_blueprint_present": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    lineage = result["unified_lineage"]
    lineage["cluster_aware_sae_budget_admission_policy_ready"] = True
    lineage["budget_policy_activation_categories"] = 8
    lineage["budget_policy_compartments"] = 17
    lineage["budget_policy_jobs_ceiling"] = 34
    lineage["budget_policy_planned_steps_ceiling"] = 68000
    lineage["budget_policy_planned_activation_rows_ceiling"] = 20480
    lineage["shared_fallback_charged_to_categories"] = False
    lineage["budget_admission_transfers_execution_authority"] = False
    lineage["production_trainer_callback_invoked"] = False
    lineage["empirical_cluster_aware_blueprint_present"] = False
    lineage["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v43_verification": predecessor_verification,
        "budget_policy_verification": policy_verification,
        "budget_policy_benchmark_verification": benchmark_verification,
        "budget_policy_derivation_sha256": sha256_file(Path(__file__)),
        "bounded_predecessor_replay": True,
    }
    boundary = result["scientific_boundary"]
    boundary["temporary_budget_reference_proposal_only"] = True
    boundary["cluster_aware_sae_budget_admission_policy_ready"] = True
    boundary["budget_admission_is_empirical_evidence"] = False
    boundary["shared_fallback_charged_to_categories"] = False
    boundary["budget_admission_transfers_execution_authority"] = False
    boundary["terminal_failure_retry_authorized"] = False
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
        raise ValueError("research protocol V44 is not reproducible")
    component = observed["component_compartments"][
        "cluster_aware_sae_budget_admission_policy_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v44",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "budget_policy_id": component["policy"]["budget_policy_id"],
        "budget_benchmark_id": component["benchmark"]["benchmark_id"],
        "budget_controls": component["benchmark"]["controls"]["controls"],
        "categories": component["activation_categories"],
        "compartments": component["compartments"],
        "jobs_ceiling": component["jobs_ceiling"],
        "planned_steps_ceiling": component["planned_steps_ceiling"],
        "planned_activation_rows_ceiling": component[
            "planned_activation_rows_ceiling"
        ],
        "shared_fallback_charged_to_categories": False,
        "budget_admission_transfers_execution_authority": False,
        "training_performed": False,
        "empirical_blueprint_present": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v43.json"
        ),
        "budget_policy_builder_path": (
            project_root
            / "experiments/authz_mechinterp/build_cluster_aware_sae_budget_admission_policy_v1.py"
        ),
        "budget_policy_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_budget_admission_policy_v1/policy.json"
        ),
        "budget_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_budget_admission_policy_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V44.")
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
