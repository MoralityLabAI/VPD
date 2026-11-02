from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.activation_category_cluster_count_selection_v2 import (
    DEFAULT_MINIMUM_CATEGORY_CLUSTER_BOUNDARY_GAP_RATIO,
    DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE_P10,
)
from experiments.authz_mechinterp.benchmark_activation_category_cluster_discreteness_v2 import (
    verify_benchmark as verify_discreteness_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v9 import (
    canonical_inputs as protocol_v9_inputs,
    verify_protocol as verify_protocol_v9,
)


SCHEMA = "sch.vpd-activation-category-protocol.v10"


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("research protocol V10 source escapes the project")
    return resolved.relative_to(root.resolve()).as_posix()


def build_protocol(
    *,
    predecessor_path: Path,
    discreteness_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = verify_protocol_v9(
        predecessor_path, **protocol_v9_inputs(project_root)
    )
    benchmark_verification = verify_discreteness_benchmark(
        discreteness_benchmark_path
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    benchmark = json.loads(discreteness_benchmark_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    selector_path = (
        project_root
        / "experiments/authz_mechinterp/activation_category_cluster_count_selection_v2.py"
    )
    benchmark_runner_path = (
        project_root
        / "experiments/authz_mechinterp/benchmark_activation_category_cluster_discreteness_v2.py"
    )
    result["component_compartments"]["within_category_cluster_discreteness_v2"] = {
        "schema": "sch.vpd-category-cluster-count-selection.v2",
        "status": "implemented_model_free_benchmarked_not_empirically_run",
        "selector": {
            "path": _relative(selector_path, project_root),
            "sha256": sha256_file(selector_path),
        },
        "benchmark": {
            "path": _relative(discreteness_benchmark_path, project_root),
            "sha256": sha256_file(discreteness_benchmark_path),
            "benchmark_id": str(benchmark["benchmark_id"]),
            "verification": benchmark_verification,
            "scenarios": int(benchmark["summary"]["scenarios"]),
            "legacy_false_splits_reproduced": int(
                benchmark["summary"]["legacy_false_splits_reproduced"]
            ),
        },
        "minimum_replicate_silhouette_p10": (
            DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE_P10
        ),
        "minimum_replicate_boundary_gap_ratio": (
            DEFAULT_MINIMUM_CATEGORY_CLUSTER_BOUNDARY_GAP_RATIO
        ),
        "continuous_direction_or_strength_geometry_routes_to": (
            "one_cluster_for_activation_category"
        ),
        "discrete_separated_mechanisms_remain_selectable": True,
        "cross_archive_and_pair_block_stability_still_required": True,
        "v1_mean_silhouette_and_ari_alone_are_insufficient": True,
        "required_for_next_empirical_category_candidate": True,
        "direct_v1_cluster_count_selection_for_next_candidate_prohibited": True,
        "does_not_select_empirical_bonsai_cluster_counts": True,
    }
    result["component_compartments"]["within_category_cluster_count_selection"][
        "superseded_for_next_empirical_candidate_by"
    ] = "within_category_cluster_discreteness_v2"
    result["unified_lineage"]["cluster_discreteness_v2_required"] = True
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v9_verification": predecessor_verification,
        "cluster_discreteness_v2_benchmark_verification": benchmark_verification,
        "cluster_discreteness_v2_selector_sha256": sha256_file(selector_path),
        "cluster_discreteness_v2_benchmark_runner_sha256": sha256_file(
            benchmark_runner_path
        ),
        "research_protocol_v10_is_distinct_from_historical_experiment_labels": True,
    }
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["sae_training_performed"] = False
    result["scientific_boundary"]["weights_loaded_or_mutated"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = build_protocol(**kwargs)
    if observed != expected:
        raise ValueError("research protocol V10 is not reproducible")
    contract = observed["component_compartments"][
        "within_category_cluster_discreteness_v2"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v10",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "predecessor_sha256": sha256_file(kwargs["predecessor_path"]),
        "cluster_discreteness_benchmark_id": contract["benchmark"]["benchmark_id"],
        "legacy_false_splits_reproduced": contract["benchmark"][
            "legacy_false_splits_reproduced"
        ],
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v9.json",
        "discreteness_benchmark_path": project_root
        / "artifacts/vpd_activation_category_cluster_discreteness_benchmark_v2/report.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive or verify research protocol V10 cluster discreteness."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None:
        result = verify_protocol(args.verify, **inputs)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_protocol(**inputs)
        write_json(args.output, result)
    print(
        json.dumps(
            {
                "schema": str(result["schema"]),
                "status": str(result.get("status", "derived")),
                "protocol_sha256": str(
                    result.get("protocol_sha256", sha256_file(args.output))
                )
                if args.output is not None
                else str(result["protocol_sha256"]),
                "execution_authorized": bool(
                    result.get("execution_authorized", False)
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
