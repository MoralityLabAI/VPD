from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_compartment_resource_reconciliation_v1 import (
    verify_benchmark as verify_resource_benchmark,
)
from experiments.authz_mechinterp.build_cluster_aware_sae_compartment_resource_reconciliation_v1 import (
    canonical_inputs as resource_inputs,
    verify_resource_reconciliation,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v42 import (
    canonical_inputs as protocol_v42_inputs,
    verify_protocol as verify_protocol_v42,
)


SCHEMA = "sch.vpd-activation-category-protocol.v43"
EXPECTED_V42_SHA256 = (
    "fc612123f8e293c25e57c6b470722ce2a6c5876f1e9a43c477b31945bedab383"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V43 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, *, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V42_SHA256:
        raise ValueError("protocol V42 differs from its pinned SHA-256")
    verification = verify_protocol_v42(path, **protocol_v42_inputs(project_root))
    if verification["execution_authorized"] is not False:
        raise ValueError("protocol V42 authorization boundary differs")
    return verification


def build_protocol(
    *,
    predecessor_path: Path,
    resource_builder_path: Path,
    resource_ledger_path: Path,
    resource_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(
        predecessor_path, project_root=project_root
    )
    resource_verification = verify_resource_reconciliation(
        resource_ledger_path, **resource_inputs(project_root)
    )
    benchmark_verification = verify_resource_benchmark(
        resource_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    benchmark = json.loads(resource_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"][
        "cluster_aware_sae_compartment_resource_reconciliation_v1"
    ] = {
        "schema": "sch.vpd-cluster-aware-sae-compartment-resource-reconciliation-contract.v1",
        "status": (
            "reconstructed_nonexecuting_exact_compartment_resources_and_terminal_"
            "failure_reconciliation_ready_raw_receipts_empirical_inputs_training_and_retry_absent"
        ),
        "builder": {
            "path": _relative(resource_builder_path, project_root),
            "sha256": sha256_file(resource_builder_path),
        },
        "ledger": {
            "path": _relative(resource_ledger_path, project_root),
            "sha256": sha256_file(resource_ledger_path),
            "resource_reconciliation_id": resource_verification[
                "resource_reconciliation_id"
            ],
            "verification": resource_verification,
        },
        "benchmark": {
            "path": _relative(resource_benchmark_path, project_root),
            "sha256": sha256_file(resource_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
        },
        "reconstructed_from_verified_campaign_summary": True,
        "raw_campaign_receipts_recovered": False,
        "activation_categories": 8,
        "compartments": 17,
        "specific_compartments": 16,
        "shared_fallback_compartments": 1,
        "execution_jobs": 34,
        "unique_output_directories": 34,
        "planned_steps": resource_verification["planned_steps"],
        "planned_activation_rows": resource_verification[
            "planned_activation_rows"
        ],
        "specific_dictionary_features": 4096,
        "shared_fallback_dictionary_features": 8192,
        "shared_fallback_double_counted": False,
        "failure_charged_callbacks": 8,
        "failure_uninvoked_callbacks": 26,
        "fresh_authority_required_for_every_future_real_job": True,
        "retry_authorized": False,
        "training_performed": False,
        "empirical_blueprint_present": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    lineage = result["unified_lineage"]
    lineage["cluster_aware_sae_compartment_resource_reconciliation_ready"] = True
    lineage["resource_reconciliation_reconstructed_not_raw"] = True
    lineage["planned_cluster_aware_sae_steps"] = resource_verification[
        "planned_steps"
    ]
    lineage["planned_cluster_aware_sae_activation_rows"] = resource_verification[
        "planned_activation_rows"
    ]
    lineage["shared_fallback_double_counted"] = False
    lineage["terminal_failure_retry_authorized"] = False
    lineage["production_trainer_callback_invoked"] = False
    lineage["empirical_cluster_aware_blueprint_present"] = False
    lineage["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v42_verification": predecessor_verification,
        "resource_reconciliation_verification": resource_verification,
        "resource_reconciliation_benchmark_verification": benchmark_verification,
        "resource_reconciliation_derivation_sha256": sha256_file(Path(__file__)),
        "bounded_predecessor_replay": True,
    }
    boundary = result["scientific_boundary"]
    boundary["temporary_resource_reconciliation_fixture_only"] = True
    boundary["cluster_aware_compartment_resource_reconciliation_ready"] = True
    boundary["resource_reconciliation_reconstructed_not_raw"] = True
    boundary["raw_campaign_receipts_recovered"] = False
    boundary["shared_fallback_double_counted"] = False
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
        raise ValueError("research protocol V43 is not reproducible")
    component = observed["component_compartments"][
        "cluster_aware_sae_compartment_resource_reconciliation_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v43",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "resource_reconciliation_id": component["ledger"][
            "resource_reconciliation_id"
        ],
        "resource_benchmark_id": component["benchmark"]["benchmark_id"],
        "resource_controls": component["benchmark"]["controls"]["controls"],
        "categories": component["activation_categories"],
        "compartments": component["compartments"],
        "jobs": component["execution_jobs"],
        "planned_steps": component["planned_steps"],
        "planned_activation_rows": component["planned_activation_rows"],
        "raw_campaign_receipts_recovered": False,
        "shared_fallback_double_counted": False,
        "retry_authorized": False,
        "training_performed": False,
        "empirical_blueprint_present": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v42.json"
        ),
        "resource_builder_path": (
            project_root
            / "experiments/authz_mechinterp/build_cluster_aware_sae_compartment_resource_reconciliation_v1.py"
        ),
        "resource_ledger_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_compartment_resource_reconciliation_v1/ledger.json"
        ),
        "resource_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_compartment_resource_reconciliation_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V43.")
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
