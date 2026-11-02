from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_execution_blueprint_v1 import (
    verify_benchmark as verify_blueprint_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v35 import (
    canonical_inputs as protocol_v35_inputs,
    verify_protocol as verify_protocol_v35,
)


SCHEMA = "sch.vpd-activation-category-protocol.v36"
EXPECTED_V35_SHA256 = (
    "6b9650239cb69cb51a0d72dd003c9b407086007ffc5dada7a14f616b0d4f3b9c"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V36 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path, *, project_root: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V35_SHA256:
        raise ValueError("protocol V35 differs from its pinned SHA-256")
    verification = verify_protocol_v35(path, **protocol_v35_inputs(project_root))
    if verification["execution_authorized"] is not False:
        raise ValueError("protocol V35 authorization boundary differs")
    return verification


def build_protocol(
    *,
    predecessor_path: Path,
    blueprint_builder_path: Path,
    blueprint_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(
        predecessor_path, project_root=project_root
    )
    benchmark_verification = verify_blueprint_benchmark(
        blueprint_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    benchmark = json.loads(blueprint_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["cluster_aware_sae_execution_blueprint_v1"] = {
        "schema": "sch.vpd-cluster-aware-sae-execution-blueprint-contract.v1",
        "status": "exact_cluster_routed_execution_contract_ready_worker_absent",
        "builder": {
            "path": _relative(blueprint_builder_path, project_root),
            "sha256": sha256_file(blueprint_builder_path),
            "blueprint_schema": "sch.vpd-cluster-aware-sae-execution-blueprint.v1",
            "job_schema": "sch.vpd-cluster-aware-sae-execution-job.v1",
        },
        "benchmark": {
            "path": _relative(blueprint_benchmark_path, project_root),
            "sha256": sha256_file(blueprint_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
            "results": benchmark["results"]["summary"],
        },
        "exact_activation_categories_in_fixture": 8,
        "specific_compartments_in_fixture": 16,
        "shared_fallback_compartments_in_fixture": 1,
        "execution_jobs_in_fixture": 34,
        "exact_episode_and_pair_lists_required": True,
        "one_global_pair_split_required": True,
        "routing_and_stability_ids_required": True,
        "unique_relative_output_directories_required": True,
        "scientific_release_and_compute_authorization_placeholders_required": True,
        "runtime_worker_implemented": False,
        "empirical_blueprint_present": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    result["unified_lineage"]["cluster_aware_sae_execution_blueprint_ready"] = True
    result["unified_lineage"]["exact_cluster_routed_job_contracts_ready"] = True
    result["unified_lineage"]["cluster_aware_sae_runtime_worker_implemented"] = False
    result["unified_lineage"]["empirical_cluster_aware_blueprint_present"] = False
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v35_verification": predecessor_verification,
        "execution_blueprint_benchmark_verification": benchmark_verification,
        "execution_blueprint_derivation_sha256": sha256_file(Path(__file__)),
        "bounded_predecessor_replay": True,
    }
    result["scientific_boundary"]["synthetic_execution_blueprint_fixture_only"] = True
    result["scientific_boundary"]["cluster_aware_execution_contract_present"] = True
    result["scientific_boundary"]["cluster_aware_runtime_worker_implemented"] = False
    result["scientific_boundary"]["empirical_cluster_aware_blueprint_present"] = False
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
        raise ValueError("research protocol V36 is not reproducible")
    component = observed["component_compartments"][
        "cluster_aware_sae_execution_blueprint_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v36",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "execution_blueprint_benchmark_id": component["benchmark"]["benchmark_id"],
        "execution_blueprint_controls": component["benchmark"]["controls"]["controls"],
        "activation_categories": component["exact_activation_categories_in_fixture"],
        "execution_jobs": component["execution_jobs_in_fixture"],
        "runtime_worker_implemented": False,
        "empirical_blueprint_present": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v35.json"
        ),
        "blueprint_builder_path": (
            project_root
            / "experiments/authz_mechinterp/build_cluster_aware_sae_execution_blueprint_v1.py"
        ),
        "blueprint_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_execution_blueprint_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V36.")
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
