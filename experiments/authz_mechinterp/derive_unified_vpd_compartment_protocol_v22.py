from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_canonical_vpd_sae_training_capture_plan_v2 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json
from experiments.authz_mechinterp.plan_canonical_vpd_capture_job_v2 import (
    _validate_split,
)


SCHEMA = "sch.vpd-activation-category-protocol.v22"
EXPECTED_V21_SHA256 = (
    "b26f5668a877923c4f45a877edd8836e65ec061d4132b9ca309abb3d9e74556a"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V22 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V21_SHA256:
        raise ValueError("protocol V21 differs from its pinned SHA-256")
    predecessor = json.loads(path.read_text(encoding="utf-8"))
    if predecessor.get("schema") != "sch.vpd-activation-category-protocol.v21":
        raise ValueError("pinned predecessor is not protocol V21")
    if predecessor["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("protocol V21 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-predecessor-verification.v1",
        "status": "pass",
        "method": "exact_sha256_and_boundary_fields",
        "sha256": EXPECTED_V21_SHA256,
    }


def build_protocol(
    *,
    predecessor_path: Path,
    capture_plan_benchmark_path: Path,
    capture_planner_path: Path,
    split_manifest_path: Path,
    sae_training_dataset_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    benchmark_verification = verify_benchmark(capture_plan_benchmark_path)
    benchmark = json.loads(capture_plan_benchmark_path.read_text(encoding="utf-8"))
    _, _, split_summary = _validate_split(
        split_manifest_path=split_manifest_path,
        split_role="sae_training",
        dataset_path=sae_training_dataset_path,
    )
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    result["component_compartments"]["sae_training_effect_capture_plan_v2"] = {
        "schema": "sch.vpd-canonical-sae-training-effect-capture-plan-contract.v2",
        "status": "planner_and_authorization_verifier_ready_worker_pending",
        "predecessor_v1_source_unchanged": True,
        "sae_training_split_role_allowed_only_in_v2": True,
        "effect_capture_input_role_not_dictionary_training_authority": True,
        "exact_split_role_dataset_and_forward_ceiling_authorization_required": True,
        "planner": {
            "path": _relative(capture_planner_path, project_root),
            "sha256": sha256_file(capture_planner_path),
            "plan_schema": "sch.vpd-canonical-category-capture-job-plan.v2",
            "authorization_schema": (
                "sch.vpd-explicit-canonical-capture-authorization.v2"
            ),
        },
        "benchmark": {
            "path": _relative(capture_plan_benchmark_path, project_root),
            "sha256": sha256_file(capture_plan_benchmark_path),
            "benchmark_id": benchmark["benchmark_id"],
            "verification": benchmark_verification,
            "controls": 12,
            "synthetic_plan_id": benchmark["results"]["capture_job_plan_id"],
        },
        "recovered_sae_training_split": {
            "manifest_path": _relative(split_manifest_path, project_root),
            "manifest_sha256": sha256_file(split_manifest_path),
            "split_id": split_summary["split_id"],
            "dataset_path": _relative(sae_training_dataset_path, project_root),
            "dataset_sha256": split_summary["dataset_sha256"],
            "episodes": split_summary["rows"],
            "pairs": split_summary["pairs"],
            "activation_category_counts": split_summary[
                "activation_category_counts"
            ],
            "episode_id_set_sha256": split_summary["episode_id_set_sha256"],
            "pair_id_set_sha256": split_summary["pair_id_set_sha256"],
            "template_group_set_sha256": split_summary[
                "template_group_set_sha256"
            ],
            "validation_status": "pass_exact_v2_split_contract",
        },
        "capture_worker_v2_present": False,
        "empirical_plan_present": False,
        "empirical_authorization_present": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }
    result["unified_lineage"]["sae_training_effect_capture_plan_v2_ready"] = True
    result["unified_lineage"]["sae_training_effect_capture_worker_v2_ready"] = False
    result["unified_lineage"]["recovered_sae_training_split_v2_validated"] = True
    result["unified_lineage"]["effect_capture_authority_separate_from_sae_training"] = True
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v21_verification": predecessor_verification,
        "sae_training_capture_plan_benchmark_verification": benchmark_verification,
        "recovered_sae_training_split_summary": split_summary,
        "sae_training_capture_planner_v2_sha256": sha256_file(capture_planner_path),
        "bounded_predecessor_replay": True,
        "research_protocol_v22_is_distinct_from_historical_experiment_labels": True,
    }
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["real_effect_archive_emitted"] = False
    result["scientific_boundary"]["empirical_sae_training_capture_plan_emitted"] = False
    result["scientific_boundary"]["weights_loaded_or_mutated"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != build_protocol(**kwargs):
        raise ValueError("research protocol V22 is not reproducible")
    contract = observed["component_compartments"][
        "sae_training_effect_capture_plan_v2"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v22",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "capture_plan_benchmark_id": contract["benchmark"]["benchmark_id"],
        "recovered_sae_training_episodes": contract[
            "recovered_sae_training_split"
        ]["episodes"],
        "capture_worker_v2_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    split_root = project_root / "artifacts/vpd_activation_category_capacity_split_v4"
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v21.json"
        ),
        "capture_plan_benchmark_path": (
            project_root
            / "artifacts/vpd_canonical_sae_training_capture_plan_benchmark_v2/report.json"
        ),
        "capture_planner_path": (
            project_root
            / "experiments/authz_mechinterp/plan_canonical_vpd_capture_job_v2.py"
        ),
        "split_manifest_path": split_root / "split_manifest.json",
        "sae_training_dataset_path": split_root / "sae_training_episodes.jsonl",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V22.")
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
