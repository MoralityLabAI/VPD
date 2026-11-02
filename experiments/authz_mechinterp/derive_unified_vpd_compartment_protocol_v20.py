from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_activation_category_cluster_partition_identity_v4 import (
    verify_benchmark as verify_partition_benchmark,
)
from experiments.authz_mechinterp.benchmark_vpd_cluster_candidate_partition_identity_admission_v4 import (
    verify_benchmark as verify_admission_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json


SCHEMA = "sch.vpd-activation-category-protocol.v20"
EXPECTED_V19_SHA256 = (
    "a689a19dc28ba3a2822b8f762625e467ba2515ba9f6faf665da140628da888a7"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V20 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_pinned_predecessor(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V19_SHA256:
        raise ValueError("protocol V19 differs from its pinned SHA-256")
    predecessor = json.loads(path.read_text(encoding="utf-8"))
    if predecessor.get("schema") != "sch.vpd-activation-category-protocol.v19":
        raise ValueError("pinned predecessor is not protocol V19")
    if predecessor["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("protocol V19 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-predecessor-verification.v1",
        "status": "pass",
        "method": "exact_sha256_and_boundary_fields",
        "sha256": EXPECTED_V19_SHA256,
    }


def build_protocol(
    *,
    predecessor_path: Path,
    partition_benchmark_path: Path,
    admission_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_pinned_predecessor(predecessor_path)
    partition_verification = verify_partition_benchmark(partition_benchmark_path)
    admission_verification = verify_admission_benchmark(
        admission_benchmark_path, project_root=project_root
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    partition_benchmark = json.loads(
        partition_benchmark_path.read_text(encoding="utf-8")
    )
    admission_benchmark = json.loads(
        admission_benchmark_path.read_text(encoding="utf-8")
    )
    partition_runtime = (
        project_root
        / "experiments/authz_mechinterp/activation_category_cluster_partition_identity_v4.py"
    )
    admission_runtime = (
        project_root
        / "experiments/authz_mechinterp/admit_vpd_cluster_candidate_partition_identity_v4.py"
    )
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    predecessor_contract = result["component_compartments"][
        "cluster_candidate_semantic_strata_admission_v3"
    ]
    contract = json.loads(json.dumps(predecessor_contract))
    contract["schema"] = "sch.vpd-cluster-candidate-partition-identity-contract.v4"
    contract["status"] = "required_successor_admission_before_empirical_sae_compartments"
    contract["predecessor_semantic_strata_v3_required"] = True
    contract["cluster_count_agreement_alone_is_insufficient"] = True
    contract["comparison"] = "pairwise_coassignment_label_invariant"
    contract["exact_partition_agreement_required_across_all_replicates"] = True
    contract["exact_candidate_consensus_membership_required"] = True
    contract["unstable_or_mismatched_candidate_mutated"] = False
    contract["unstable_or_mismatched_candidate_relabelled"] = False
    contract["partition_identity_runtime"] = {
        "path": _relative(partition_runtime, project_root),
        "sha256": sha256_file(partition_runtime),
        "schema": "sch.vpd-activation-category-cluster-partition-identity.v4",
    }
    contract["partition_identity_benchmark"] = {
        "path": _relative(partition_benchmark_path, project_root),
        "sha256": sha256_file(partition_benchmark_path),
        "benchmark_id": partition_benchmark["benchmark_id"],
        "verification": partition_verification,
        "negative_controls": 6,
    }
    contract["production_admission_runtime"] = {
        "path": _relative(admission_runtime, project_root),
        "sha256": sha256_file(admission_runtime),
        "schema": "sch.vpd-cluster-candidate-partition-identity-admission.v4",
    }
    contract["production_admission_benchmark"] = {
        "path": _relative(admission_benchmark_path, project_root),
        "sha256": sha256_file(admission_benchmark_path),
        "benchmark_id": admission_benchmark["benchmark_id"],
        "verification": admission_verification,
        "scenarios": 2,
    }
    result["component_compartments"][
        "cluster_candidate_partition_identity_admission_v4"
    ] = contract
    result["component_compartments"][
        "cluster_candidate_semantic_strata_admission_v3"
    ]["successor_partition_identity_v4_required"] = True
    result["unified_lineage"]["cluster_partition_identity_v4_required"] = True
    result["unified_lineage"]["count_only_candidate_admission_prohibited"] = True
    result["unified_lineage"]["exact_candidate_membership_required_for_sae_compartments"] = True
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v19_verification": predecessor_verification,
        "partition_identity_benchmark_verification": partition_verification,
        "production_partition_identity_admission_benchmark_verification": admission_verification,
        "partition_identity_runtime_sha256": sha256_file(partition_runtime),
        "partition_identity_admission_runtime_sha256": sha256_file(admission_runtime),
        "bounded_predecessor_replay": True,
        "research_protocol_v20_is_distinct_from_historical_experiment_labels": True,
    }
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["real_effect_archive_emitted"] = False
    result["scientific_boundary"]["weights_loaded_or_mutated"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != build_protocol(**kwargs):
        raise ValueError("research protocol V20 is not reproducible")
    contract = observed["component_compartments"][
        "cluster_candidate_partition_identity_admission_v4"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v20",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "predecessor_sha256": sha256_file(kwargs["predecessor_path"]),
        "partition_benchmark_id": contract["partition_identity_benchmark"][
            "benchmark_id"
        ],
        "admission_benchmark_id": contract["production_admission_benchmark"][
            "benchmark_id"
        ],
        "count_only_candidate_admission_prohibited": True,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v19.json"
        ),
        "partition_benchmark_path": (
            project_root
            / "artifacts/vpd_activation_category_cluster_partition_identity_benchmark_v4/report.json"
        ),
        "admission_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_candidate_partition_identity_admission_benchmark_v4/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V20.")
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
    protocol_sha256 = result.get("protocol_sha256")
    if protocol_sha256 is None:
        protocol_sha256 = sha256_file(args.output)
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
