from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_empirical_budget_binding_readiness_v1 import (
    verify_benchmark as verify_binding_benchmark,
)
from experiments.authz_mechinterp.build_cluster_aware_sae_empirical_budget_binding_readiness_v1 import (
    canonical_inputs as binding_inputs,
    verify_binding_readiness,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v44 import (
    canonical_inputs as protocol_v44_inputs,
    verify_protocol as verify_protocol_v44,
)


SCHEMA = "sch.vpd-activation-category-protocol.v45"
EXPECTED_V44_SHA256 = (
    "d7771004061a7521fd743da293af5c20d16bd1b8667970e54f7d1f61496e8828"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V45 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, *, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V44_SHA256:
        raise ValueError("protocol V44 differs from its pinned SHA-256")
    verification = verify_protocol_v44(path, **protocol_v44_inputs(project_root))
    if verification["execution_authorized"] is not False:
        raise ValueError("protocol V44 authorization boundary differs")
    return verification


def build_protocol(
    *,
    predecessor_path: Path,
    binding_builder_path: Path,
    binding_readiness_path: Path,
    binding_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(
        predecessor_path, project_root=project_root
    )
    readiness_verification = verify_binding_readiness(
        binding_readiness_path, **binding_inputs(project_root)
    )
    benchmark_verification = verify_binding_benchmark(
        binding_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    benchmark = json.loads(binding_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"][
        "cluster_aware_sae_empirical_budget_binding_readiness_v1"
    ] = {
        "schema": "sch.vpd-cluster-aware-sae-empirical-budget-binding-contract.v1",
        "status": (
            "blocked_exact_50_bindings_missing_empirical_artifact_graph_"
            "signed_authority_durable_transition_training_and_execution"
        ),
        "builder": {
            "path": _relative(binding_builder_path, project_root),
            "sha256": sha256_file(binding_builder_path),
        },
        "readiness": {
            "path": _relative(binding_readiness_path, project_root),
            "sha256": sha256_file(binding_readiness_path),
            "binding_readiness_id": readiness_verification["binding_readiness_id"],
            "verification": readiness_verification,
        },
        "benchmark": {
            "path": _relative(binding_benchmark_path, project_root),
            "sha256": sha256_file(binding_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
        },
        "binding_phases": 6,
        "required_bindings": 50,
        "capture_execution_identity_bindings": 10,
        "artifact_graph_bindings": 40,
        "supplied_bindings": 0,
        "missing_bindings": 50,
        "syntax_complete": False,
        "cross_artifact_content_verified": False,
        "budget_binding_verified": False,
        "signed_authority_verified": False,
        "durable_runtime_transition_verified": False,
        "synthetic_complete_fixture_persisted": False,
        "empirical_blueprint_present": False,
        "training_performed": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    lineage = result["unified_lineage"]
    lineage["cluster_aware_sae_empirical_budget_binding_readiness_ready"] = True
    lineage["empirical_budget_binding_required_bindings"] = 50
    lineage["empirical_budget_binding_missing_bindings"] = 50
    lineage["empirical_budget_binding_phases"] = 6
    lineage["empirical_budget_binding_syntax_complete"] = False
    lineage["empirical_budget_binding_verified"] = False
    lineage["signed_authority_verified"] = False
    lineage["durable_runtime_transition_verified"] = False
    lineage["production_trainer_callback_invoked"] = False
    lineage["empirical_cluster_aware_blueprint_present"] = False
    lineage["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v44_verification": predecessor_verification,
        "empirical_budget_binding_readiness_verification": readiness_verification,
        "empirical_budget_binding_benchmark_verification": benchmark_verification,
        "empirical_budget_binding_derivation_sha256": sha256_file(Path(__file__)),
        "bounded_predecessor_replay": True,
    }
    boundary = result["scientific_boundary"]
    boundary["temporary_complete_binding_fixture_in_memory_only"] = True
    boundary["cluster_aware_sae_empirical_budget_binding_readiness_ready"] = True
    boundary["empirical_budget_binding_syntax_complete"] = False
    boundary["empirical_budget_binding_verified"] = False
    boundary["signed_authority_verified"] = False
    boundary["durable_runtime_transition_verified"] = False
    boundary["synthetic_complete_binding_fixture_persisted"] = False
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
        raise ValueError("research protocol V45 is not reproducible")
    component = observed["component_compartments"][
        "cluster_aware_sae_empirical_budget_binding_readiness_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v45",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "binding_readiness_id": component["readiness"]["binding_readiness_id"],
        "binding_benchmark_id": component["benchmark"]["benchmark_id"],
        "binding_controls": component["benchmark"]["controls"]["controls"],
        "binding_phases": component["binding_phases"],
        "required_bindings": component["required_bindings"],
        "missing_bindings": component["missing_bindings"],
        "syntax_complete": False,
        "budget_binding_verified": False,
        "signed_authority_verified": False,
        "durable_runtime_transition_verified": False,
        "training_performed": False,
        "empirical_blueprint_present": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v44.json"
        ),
        "binding_builder_path": (
            project_root
            / "experiments/authz_mechinterp/build_cluster_aware_sae_empirical_budget_binding_readiness_v1.py"
        ),
        "binding_readiness_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_empirical_budget_binding_readiness_v1/readiness.json"
        ),
        "binding_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_empirical_budget_binding_readiness_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V45.")
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
