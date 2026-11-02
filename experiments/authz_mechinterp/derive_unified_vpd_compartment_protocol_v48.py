from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_semantic_graph_contract_v2 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.build_cluster_aware_sae_semantic_graph_contract_v2 import (
    canonical_inputs as contract_inputs,
    verify_contract,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json


SCHEMA = "sch.vpd-activation-category-protocol.v48"
EXPECTED_V47_SHA256 = (
    "a7bd3d23ad2ecce3b5933dae59b70a7de72212fec103228a149931048b6dc15c"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V48 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V47_SHA256:
        raise ValueError("protocol V47 differs from its pinned SHA-256")
    observed = _load(path)
    if observed.get("schema") != "sch.vpd-activation-category-protocol.v47":
        raise ValueError("pinned predecessor is not protocol V47")
    if observed.get("scientific_boundary", {}).get("execution_authorized") is not False:
        raise ValueError("protocol V47 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-protocol-predecessor-verification.v1",
        "status": "pass",
        "protocol_sha256": EXPECTED_V47_SHA256,
        "schema_verified": True,
        "non_authorizing_boundary_verified": True,
        "deep_predecessor_replay_performed": False,
    }


def build_protocol(
    *,
    predecessor_path: Path,
    contract_builder_path: Path,
    contract_path: Path,
    benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    contract_verification = verify_contract(
        contract_path, **contract_inputs(project_root)
    )
    benchmark_verification = verify_benchmark(
        benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    contract = _load(contract_path)
    benchmark = _load(benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"][
        "cluster_aware_sae_semantic_graph_contract_v2"
    ] = {
        "schema": "sch.vpd-cluster-aware-sae-semantic-graph-lineage-contract.v2",
        "status": (
            "corrected_implemented_interface_graph_contract_verified_"
            "three_terminal_joins_empirical_graph_training_and_execution_absent"
        ),
        "builder": {
            "path": _relative(contract_builder_path, project_root),
            "sha256": sha256_file(contract_builder_path),
        },
        "contract": {
            "path": _relative(contract_path, project_root),
            "sha256": sha256_file(contract_path),
            "semantic_graph_contract_id": contract["semantic_graph_contract_id"],
            "verification": contract_verification,
        },
        "benchmark": {
            "path": _relative(benchmark_path, project_root),
            "sha256": sha256_file(benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
        },
        "logical_nodes": contract["summary"]["logical_nodes"],
        "implemented_node_representations": contract["summary"][
            "implemented_node_representations"
        ],
        "semantic_joins": contract["summary"]["semantic_joins"],
        "terminal_gap_nodes": contract["summary"]["terminal_gap_nodes"],
        "terminal_gap_joins": contract["summary"]["terminal_gap_joins"],
        "v1_mismatches_accounted": contract["summary"][
            "v1_mismatches_accounted"
        ],
        "blueprint_budget_v2_controls_passed": contract["summary"][
            "blueprint_budget_v2_controls_passed"
        ],
        "v1_structural_graph_preserved": True,
        "corrected_v2_contract_verified": True,
        "end_to_end_semantic_dispatch_allowed": False,
        "empirical_graph_present": False,
        "production_runtime_attachment_verifier_present": False,
        "training_performed": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    lineage = result["unified_lineage"]
    lineage["cluster_aware_sae_semantic_graph_contract_v2_verified"] = True
    lineage["semantic_graph_v2_logical_nodes"] = 21
    lineage["semantic_graph_v2_implemented_node_representations"] = 20
    lineage["semantic_graph_v2_semantic_joins"] = 38
    lineage["semantic_graph_v2_terminal_gap_joins"] = 3
    lineage["semantic_graph_v1_mismatches_accounted"] = 28
    lineage["blueprint_bound_budget_v2_verified"] = True
    lineage["budget_admission_to_signed_release_join_present"] = False
    lineage["budget_admission_to_runtime_admission_join_present"] = False
    lineage["production_runtime_attachment_verifier_present"] = False
    lineage["end_to_end_semantic_dispatch_allowed"] = False
    lineage["empirical_cross_artifact_graph_present"] = False
    lineage["production_trainer_callback_invoked"] = False
    lineage["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v47_verification": predecessor_verification,
        "semantic_graph_contract_v2_verification": contract_verification,
        "semantic_graph_contract_v2_benchmark_verification": benchmark_verification,
        "semantic_graph_contract_v2_derivation_sha256": sha256_file(Path(__file__)),
        "pinned_predecessor_hash_schema_and_boundary_verified": True,
        "deep_predecessor_replay_performed": False,
    }
    boundary = result["scientific_boundary"]
    boundary["cluster_aware_sae_semantic_graph_contract_v2_verified"] = True
    boundary["v1_cross_artifact_structural_graph_preserved"] = True
    boundary["v1_cross_artifact_semantic_graph_verified"] = False
    boundary["semantic_graph_v2_contract_only"] = True
    boundary["semantic_dispatch_allowed"] = False
    boundary["budget_admission_to_signed_release_join_present"] = False
    boundary["budget_admission_to_runtime_admission_join_present"] = False
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
        raise ValueError("research protocol V48 is not reproducible")
    component = observed["component_compartments"][
        "cluster_aware_sae_semantic_graph_contract_v2"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v48",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "semantic_graph_contract_id": component["contract"][
            "semantic_graph_contract_id"
        ],
        "semantic_graph_benchmark_id": component["benchmark"]["benchmark_id"],
        "controls": component["benchmark"]["controls"]["controls"],
        "logical_nodes": component["logical_nodes"],
        "implemented_node_representations": component[
            "implemented_node_representations"
        ],
        "semantic_joins": component["semantic_joins"],
        "terminal_gap_joins": component["terminal_gap_joins"],
        "semantic_dispatch_allowed": False,
        "empirical_graph_present": False,
        "training_performed": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v47.json",
        "contract_builder_path": project_root
        / "experiments/authz_mechinterp/build_cluster_aware_sae_semantic_graph_contract_v2.py",
        "contract_path": project_root
        / "artifacts/vpd_cluster_aware_sae_semantic_graph_contract_v2/contract.json",
        "benchmark_path": project_root
        / "artifacts/vpd_cluster_aware_sae_semantic_graph_contract_benchmark_v2/report.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V48.")
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
