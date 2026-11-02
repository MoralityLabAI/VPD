from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_observable_authz_retraining_experiment_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_vpd_observable_authz_retraining_experiment_v1 import (
    verify_experiment_spec,
)


SCHEMA = "sch.vpd-activation-category-protocol.v51"
EXPECTED_V50_SHA256 = (
    "55dc12c6f948c6fc8ecaeb1d9d06d037fbc5a72ab2fd0a5990b453bd7271a951"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V51 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V50_SHA256:
        raise ValueError("protocol V50 differs from its pinned SHA-256")
    observed = _load(path)
    if observed.get("schema") != "sch.vpd-activation-category-protocol.v50":
        raise ValueError("pinned predecessor is not protocol V50")
    if observed.get("scientific_boundary", {}).get("execution_authorized") is not False:
        raise ValueError("protocol V50 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-protocol-predecessor-verification.v1",
        "status": "pass",
        "protocol_sha256": EXPECTED_V50_SHA256,
        "schema_verified": True,
        "non_authorizing_boundary_verified": True,
        "deep_predecessor_replay_performed": False,
    }


def build_protocol(
    *,
    predecessor_path: Path,
    experiment_derivation_path: Path,
    experiment_spec_path: Path,
    benchmark_path: Path,
    test_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    spec_verification = verify_experiment_spec(
        experiment_spec_path, project_root=project_root
    )
    benchmark_verification = verify_benchmark(
        benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    spec = _load(experiment_spec_path)
    benchmark = _load(benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"][
        "observable_authz_category_retraining_experiment_v1"
    ] = {
        "schema": "sch.vpd-observable-authz-category-retraining-experiment-lineage.v1",
        "status": (
            "design_only_pair_template_condition_disjoint_category_compartment_"
            "experiment_verified_pack_training_inference_and_execution_absent"
        ),
        "derivation": {
            "path": _relative(experiment_derivation_path, project_root),
            "sha256": sha256_file(experiment_derivation_path),
        },
        "specification": {
            "path": _relative(experiment_spec_path, project_root),
            "sha256": sha256_file(experiment_spec_path),
            "experiment_spec_id": spec["experiment_spec_id"],
            "verification": spec_verification,
        },
        "benchmark": {
            "path": _relative(benchmark_path, project_root),
            "sha256": sha256_file(benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": benchmark["summary"],
        },
        "test": {
            "path": _relative(test_path, project_root),
            "sha256": sha256_file(test_path),
        },
        "activation_categories": 8,
        "disjoint_partitions": 7,
        "candidate_variants": 4,
        "planned_pairs": 840,
        "planned_rows": 1680,
        "primary_candidate_id": "fresh_lora_margin_v1",
        "pair_template_and_condition_disjoint": True,
        "observable_invalid_maps_to_request": True,
        "observable_valid_maps_to_propose": True,
        "fresh_pack_present": False,
        "training_performed": False,
        "model_inference_performed": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
        "candidate_training_authorized": False,
    }
    lineage = result["unified_lineage"]
    lineage["observable_authz_category_retraining_experiment_v1_verified"] = True
    lineage["observable_retraining_activation_categories"] = 8
    lineage["observable_retraining_disjoint_partitions"] = 7
    lineage["observable_retraining_candidate_variants"] = 4
    lineage["observable_retraining_planned_pairs"] = 840
    lineage["observable_retraining_planned_rows"] = 1680
    lineage["observable_retraining_pair_template_condition_disjoint"] = True
    lineage["observable_retraining_fresh_pack_present"] = False
    lineage["model_inference_performed"] = False
    lineage["training_performed"] = False
    lineage["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v50_verification": predecessor_verification,
        "observable_retraining_experiment_spec_verification": spec_verification,
        "observable_retraining_experiment_benchmark_verification": benchmark_verification,
        "observable_retraining_experiment_derivation_sha256": sha256_file(Path(__file__)),
        "pinned_predecessor_hash_schema_and_boundary_verified": True,
        "deep_predecessor_replay_performed": False,
    }
    boundary = result["scientific_boundary"]
    boundary["observable_authz_category_retraining_experiment_v1_verified"] = True
    boundary["observable_retraining_activation_categories"] = 8
    boundary["observable_retraining_disjoint_partitions"] = 7
    boundary["observable_retraining_candidate_variants"] = 4
    boundary["observable_retraining_pair_template_condition_disjoint"] = True
    boundary["observable_retraining_fresh_pack_present"] = False
    boundary["observable_retraining_pack_generation_authorized"] = False
    boundary["semantic_dispatch_allowed"] = False
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
    boundary["candidate_training_authorized"] = False
    boundary["execution_authorized"] = False
    boundary["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    if observed != build_protocol(**kwargs):
        raise ValueError("research protocol V51 is not reproducible")
    component = observed["component_compartments"][
        "observable_authz_category_retraining_experiment_v1"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v51",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "experiment_spec_id": component["specification"]["experiment_spec_id"],
        "benchmark_id": component["benchmark"]["benchmark_id"],
        "controls": component["benchmark"]["controls"]["controls"],
        "activation_categories": component["activation_categories"],
        "disjoint_partitions": component["disjoint_partitions"],
        "candidate_variants": component["candidate_variants"],
        "planned_pairs": component["planned_pairs"],
        "planned_rows": component["planned_rows"],
        "fresh_pack_present": False,
        "training_performed": False,
        "sae_training_authorized": False,
        "candidate_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v50.json",
        "experiment_derivation_path": project_root
        / "experiments/authz_mechinterp/derive_vpd_observable_authz_retraining_experiment_v1.py",
        "experiment_spec_path": project_root
        / "configs/evals/vpd_observable_authz_retraining_experiment_v1.json",
        "benchmark_path": project_root
        / "artifacts/vpd_observable_authz_retraining_experiment_benchmark_v1/report.json",
        "test_path": project_root
        / "tests/test_vpd_observable_authz_retraining_experiment_v1.py",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V51.")
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
