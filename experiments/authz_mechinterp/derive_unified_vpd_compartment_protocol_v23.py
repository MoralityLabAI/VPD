from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_canonical_vpd_sae_training_capture_runtime_v2 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file, write_json


SCHEMA = "sch.vpd-activation-category-protocol.v23"
EXPECTED_V22_SHA256 = (
    "4a59df7c095e407be02ebfa5f01504eee6f98bc0872b4c9379c3f40ea98d016c"
)


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("protocol V23 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_V22_SHA256:
        raise ValueError("protocol V22 differs from its pinned SHA-256")
    predecessor = json.loads(path.read_text(encoding="utf-8"))
    if predecessor.get("schema") != "sch.vpd-activation-category-protocol.v22":
        raise ValueError("pinned predecessor is not protocol V22")
    if predecessor["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("protocol V22 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-predecessor-verification.v1",
        "status": "pass",
        "method": "exact_sha256_and_boundary_fields",
        "sha256": EXPECTED_V22_SHA256,
    }


def build_protocol(
    *,
    predecessor_path: Path,
    runtime_benchmark_path: Path,
    capture_worker_path: Path,
    assembler_v2_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    benchmark_verification = verify_benchmark(runtime_benchmark_path)
    benchmark = json.loads(runtime_benchmark_path.read_text(encoding="utf-8"))
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(predecessor))
    result["schema"] = SCHEMA
    result["predecessor"] = _relative(predecessor_path, project_root)
    result["predecessor_sha256"] = sha256_file(predecessor_path)
    contract = result["component_compartments"][
        "sae_training_effect_capture_plan_v2"
    ]
    contract["status"] = (
        "planner_authorization_verifier_and_guarded_worker_ready_"
        "empirical_inputs_pending"
    )
    contract["capture_worker_v2_present"] = True
    contract["capture_worker_v2"] = {
        "path": _relative(capture_worker_path, project_root),
        "sha256": sha256_file(capture_worker_path),
        "execution_schema": "sch.vpd-canonical-category-capture-job-execution.v2",
        "sae_training_role_effect_capture_only": True,
        "dictionary_training_authority_granted": False,
    }
    contract["effect_shard_assembler_v2"] = {
        "path": _relative(assembler_v2_path, project_root),
        "sha256": sha256_file(assembler_v2_path),
        "assembly_schema": "sch.vpd-canonical-effect-shard-assembly.v2",
        "role_extension": "sae_training",
        "predecessor_v1_source_unchanged": True,
        "array_kernel_role_independent": True,
    }
    contract["runtime_benchmark"] = {
        "path": _relative(runtime_benchmark_path, project_root),
        "sha256": sha256_file(runtime_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
        "controls": benchmark["summary"]["controls"],
        "forward_passes": benchmark["scenario"]["forward_passes"],
        "split_role": benchmark["scenario"]["split_role"],
    }
    contract["synthetic_worker_verification_ready"] = True
    contract["empirical_plan_present"] = False
    contract["empirical_authorization_present"] = False
    contract["empirical_capture_receipt_present"] = False
    contract["execution_authorized"] = False
    contract["sae_training_authorized"] = False
    result["unified_lineage"]["sae_training_effect_capture_worker_v2_ready"] = True
    result["unified_lineage"]["sae_training_effect_shard_assembler_v2_ready"] = True
    result["unified_lineage"]["sae_training_capture_worker_controls"] = 16
    result["unified_lineage"]["empirical_sae_training_effect_capture_ready"] = False
    result["unified_lineage"]["execution_authorized"] = False
    result["protocol_derivation"] = {
        **result["protocol_derivation"],
        "predecessor_v22_verification": predecessor_verification,
        "sae_training_capture_runtime_benchmark_verification": benchmark_verification,
        "sae_training_capture_worker_v2_sha256": sha256_file(capture_worker_path),
        "sae_training_effect_shard_assembler_v2_sha256": sha256_file(
            assembler_v2_path
        ),
        "bounded_predecessor_replay": True,
        "research_protocol_v23_is_distinct_from_historical_experiment_labels": True,
    }
    result["scientific_boundary"]["model_inference_performed"] = False
    result["scientific_boundary"]["real_effect_archive_emitted"] = False
    result["scientific_boundary"]["empirical_sae_training_capture_plan_emitted"] = False
    result["scientific_boundary"]["empirical_sae_training_capture_receipt_emitted"] = False
    result["scientific_boundary"]["weights_loaded_or_mutated"] = False
    result["scientific_boundary"]["sae_dictionary_training_authorized"] = False
    result["scientific_boundary"]["execution_authorized"] = False
    result["scientific_boundary"]["paid_compute_authorized"] = False
    return result


def verify_protocol(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != build_protocol(**kwargs):
        raise ValueError("research protocol V23 is not reproducible")
    contract = observed["component_compartments"][
        "sae_training_effect_capture_plan_v2"
    ]
    return {
        "schema": "sch.vpd-activation-category-protocol-verification.v23",
        "status": "pass",
        "protocol_sha256": sha256_file(path),
        "runtime_benchmark_id": contract["runtime_benchmark"]["benchmark_id"],
        "capture_worker_v2_ready": True,
        "empirical_capture_ready": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v22.json"
        ),
        "runtime_benchmark_path": (
            project_root
            / "artifacts/vpd_canonical_sae_training_capture_runtime_benchmark_v2/report.json"
        ),
        "capture_worker_path": (
            project_root
            / "experiments/authz_mechinterp/run_canonical_vpd_capture_job_v2.py"
        ),
        "assembler_v2_path": (
            project_root
            / "experiments/authz_mechinterp/assemble_canonical_vpd_effect_shard_v2.py"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify research protocol V23.")
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
