from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_activation_category_cluster_model_order_v2 import verify_diagnostic
from experiments.authz_mechinterp.benchmark_activation_category_cluster_semantic_strata_v3 import verify_benchmark as verify_matrix_benchmark
from experiments.authz_mechinterp.benchmark_vpd_cluster_candidate_semantic_strata_admission_v3 import verify_benchmark as verify_production_benchmark
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v11 import canonical_inputs as protocol_v11_inputs, verify_protocol as verify_protocol_v11


SCHEMA = "sch.vpd-activation-category-protocol.v12"


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("research protocol V12 source escapes the project")
    return resolved.relative_to(root.resolve()).as_posix()


def build_protocol(
    *, predecessor_path: Path, model_order_path: Path,
    matrix_benchmark_path: Path, production_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = verify_protocol_v11(
        predecessor_path, **protocol_v11_inputs(project_root)
    )
    model_order_verification = verify_diagnostic(model_order_path)
    matrix_verification = verify_matrix_benchmark(matrix_benchmark_path)
    production_verification = verify_production_benchmark(
        production_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    model_order = json.loads(model_order_path.read_text(encoding="utf-8"))
    matrix_benchmark = json.loads(matrix_benchmark_path.read_text(encoding="utf-8"))
    production_benchmark = json.loads(production_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    runner = project_root / "experiments/authz_mechinterp/admit_vpd_cluster_candidate_semantic_strata_v3.py"
    result["component_compartments"]["cluster_candidate_semantic_strata_admission_v3"] = {
        "schema": "sch.vpd-cluster-candidate-semantic-strata-admission.v3",
        "status": "implemented_model_free_benchmarked_not_empirically_run",
        "runner": {"path": _relative(runner, project_root), "sha256": sha256_file(runner)},
        "model_order_diagnostic": {
            "path": _relative(model_order_path, project_root),
            "sha256": sha256_file(model_order_path),
            "diagnostic_id": model_order["diagnostic_id"],
            "verification": model_order_verification,
            "known_two_three_four_mode_scenarios_exact": 4,
        },
        "semantic_matrix_benchmark": {
            "path": _relative(matrix_benchmark_path, project_root),
            "sha256": sha256_file(matrix_benchmark_path),
            "benchmark_id": matrix_benchmark["benchmark_id"],
            "verification": matrix_verification,
            "condition_and_template_confounds_rejected": 2,
        },
        "production_shaped_benchmark": {
            "path": _relative(production_benchmark_path, project_root),
            "sha256": sha256_file(production_benchmark_path),
            "benchmark_id": production_benchmark["benchmark_id"],
            "verification": production_verification,
            "v2_admitted_condition_confound_reproduced": True,
            "v3_rejected_condition_confound": True,
            "v3_preserved_genuine_four_mechanisms": True,
        },
        "condition_and_template_statistics_required_in_discovery": True,
        "condition_and_template_statistics_required_in_cluster_validation": True,
        "leave_one_condition_out_membership_stability_required": True,
        "leave_one_template_group_out_membership_stability_required": True,
        "exact_candidate_allocation_agreement_required": True,
        "candidate_mutation_or_relabelling_prohibited": True,
        "v2_only_admission_for_next_empirical_candidate_prohibited": True,
        "empirical_admission_receipt_present": False,
        "execution_authorized": False,
    }
    result["component_compartments"]["cluster_candidate_discreteness_admission_v2"][
        "superseded_for_next_empirical_candidate_by"
    ] = "cluster_candidate_semantic_strata_admission_v3"
    result["unified_lineage"]["cluster_candidate_semantic_strata_admission_v3_required"] = True
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v11_verification": predecessor_verification,
        "cluster_model_order_verification": model_order_verification,
        "semantic_strata_matrix_benchmark_verification": matrix_verification,
        "semantic_strata_production_benchmark_verification": production_verification,
        "semantic_strata_admission_runner_sha256": sha256_file(runner),
        "research_protocol_v12_is_distinct_from_historical_experiment_labels": True,
    }
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["sae_training_performed"] = False
    result["scientific_boundary"]["weights_loaded_or_mutated"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != build_protocol(**kwargs):
        raise ValueError("research protocol V12 is not reproducible")
    contract = observed["component_compartments"]["cluster_candidate_semantic_strata_admission_v3"]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v12",
        "status": "pass", "protocol_sha256": sha256_file(path),
        "predecessor_sha256": sha256_file(kwargs["predecessor_path"]),
        "production_benchmark_id": contract["production_shaped_benchmark"]["benchmark_id"],
        "semantic_strata_admission_runtime_present": True,
        "empirical_admission_receipt_present": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root / "configs/evals/vpd_activation_category_compartments_v11.json",
        "model_order_path": project_root / "artifacts/vpd_activation_category_cluster_model_order_diagnostic_v2/report.json",
        "matrix_benchmark_path": project_root / "artifacts/vpd_activation_category_cluster_semantic_strata_benchmark_v3/report.json",
        "production_benchmark_path": project_root / "artifacts/vpd_cluster_candidate_semantic_strata_admission_benchmark_v3/report.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V12.")
    parser.add_argument("--output", type=Path); parser.add_argument("--verify", type=Path)
    args = parser.parse_args(); inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None: result = verify_protocol(args.verify, **inputs)
    else:
        if args.output is None: parser.error("--output is required unless verifying")
        result = build_protocol(**inputs); write_json(args.output, result)
    print(json.dumps({
        "schema": result["schema"], "status": result.get("status", "derived"),
        "protocol_sha256": result.get("protocol_sha256", sha256_file(args.output)),
        "execution_authorized": result.get("execution_authorized", False),
    }, indent=2, sort_keys=True))


if __name__ == "__main__": main()
