from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.verify_vpd_guarded_capture_job_binding_v1 import (
    _validate_plan_binding,
)


def _rejects(call: Callable[[], Any]) -> bool:
    try:
        call()
    except (KeyError, TypeError, ValueError):
        return True
    return False


def _fixture(project_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = json.loads(
        (
            project_root
            / "artifacts/vpd_guarded_capture_job_allocations_v1/manifest.json"
        ).read_text(encoding="utf-8")
    )
    preflight = json.loads(
        (project_root / manifest["source"]["preflight_path"]).read_text(
            encoding="utf-8"
        )
    )
    job = manifest["roles"]["discovery"]["jobs"][0]
    packet = json.loads((project_root / job["packet_path"]).read_text(encoding="utf-8"))
    role_resource = preflight["capture_roles"]["discovery"]["resource_ceiling"]
    plan = {
        "source": {
            "semantic_split": {
                "split_role": "discovery",
                "dataset_sha256": job["dataset_sha256"],
            },
            "dataset": {
                "path": job["dataset_path"],
                "sha256": job["dataset_sha256"],
            },
            "model_identity": {
                "model_id": "synthetic/not-loaded",
                "revision": "synthetic-not-loaded",
                "model_files_manifest_sha256": "1" * 64,
            },
            "adapter_weights": {"sha256": "2" * 64},
            "adapter_config": {"sha256": "3" * 64},
        },
        "capture_contract": {
            "component_records": [
                {"component_id": component_id}
                for component_id in packet["component_ids"]
            ],
            "batch_size": 4,
            "category_names": sorted(role_resource["category_episode_counts"]),
            "category_row_counts": [
                role_resource["category_episode_counts"][category]
                for category in sorted(role_resource["category_episode_counts"])
            ],
            "semantic_sketch": {
                "uncertainty_blocks": 8,
                "countsketch_dimension": 256,
            },
            "device": "cpu",
            "output_dir": (
                "artifacts/vpd_cluster_to_sae_empirical_capture_runs/"
                "synthetic-binding-do-not-create/discovery-packet-00"
            ),
        },
        "resource_ceiling": {
            "dataset_rows": role_resource["episodes"],
            "canonical_components": job["component_count"],
            "maximum_forward_passes": job["allocation"]["batch_forwards"],
        },
    }
    return plan, job, preflight


def build_report(project_root: Path) -> dict[str, Any]:
    before = MODEL_FRAMEWORKS & set(sys.modules)
    plan, job, preflight = _fixture(project_root)
    incomplete_result = _validate_plan_binding(
        plan=plan,
        job=job,
        preflight=preflight,
        project_root=project_root,
    )
    complete_preflight = copy.deepcopy(preflight)
    complete_preflight["missing_required_bindings"] = []
    complete_preflight["immutable_bindings"] = {
        "model_id": "synthetic/not-loaded",
        "model_revision": "synthetic-not-loaded",
        "model_files_manifest_sha256": "1" * 64,
        "adapter_path": str(
            (project_root / "synthetic-adapter-do-not-open").resolve()
        ),
        "adapter_weights_sha256": "2" * 64,
        "adapter_config_sha256": "3" * 64,
        "device": "cpu",
        "wall_time_ceiling_seconds": 3600,
        "cost_ceiling_usd": 1.0,
        "output_root": str(
            (
                project_root
                / "artifacts/vpd_cluster_to_sae_empirical_capture_runs/"
                "synthetic-binding-do-not-create"
            ).resolve()
        ),
    }
    complete_result = _validate_plan_binding(
        plan=plan,
        job=job,
        preflight=complete_preflight,
        project_root=project_root,
    )

    def rejected_after(mutator: Callable[[dict[str, Any]], None]) -> bool:
        changed = copy.deepcopy(plan)
        mutator(changed)
        return _rejects(
            lambda: _validate_plan_binding(
                plan=changed,
                job=job,
                preflight=preflight,
                project_root=project_root,
            )
        )

    controls = {
        "exact_recovered_plan_shape_binds": incomplete_result["packet_id"]
        == job["packet_id"],
        "incomplete_preflight_remains_not_ready": (
            incomplete_result["identity_bindings_complete"] is False
            and incomplete_result["execution_ready"] is False
        ),
        "identity_complete_shape_binds_without_execution": (
            complete_result["identity_bindings_complete"] is True
            and complete_result["execution_ready"] is False
        ),
        "component_substitution_rejected": rejected_after(
            lambda value: value["capture_contract"]["component_records"][0].__setitem__(
                "component_id", "substituted:rank0"
            )
        ),
        "role_substitution_rejected": rejected_after(
            lambda value: value["source"]["semantic_split"].__setitem__(
                "split_role", "calibration"
            )
        ),
        "dataset_substitution_rejected": rejected_after(
            lambda value: value["source"]["dataset"].__setitem__(
                "sha256", "0" * 64
            )
        ),
        "forward_allocation_tamper_rejected": rejected_after(
            lambda value: value["resource_ceiling"].__setitem__(
                "maximum_forward_passes",
                value["resource_ceiling"]["maximum_forward_passes"] - 1,
            )
        ),
        "episode_allocation_tamper_rejected": rejected_after(
            lambda value: value["resource_ceiling"].__setitem__(
                "dataset_rows", value["resource_ceiling"]["dataset_rows"] - 1
            )
        ),
        "batch_size_tamper_rejected": rejected_after(
            lambda value: value["capture_contract"].__setitem__("batch_size", 8)
        ),
        "category_count_tamper_rejected": rejected_after(
            lambda value: value["capture_contract"]["category_row_counts"].__setitem__(
                0, 1
            )
        ),
        "sketch_shape_tamper_rejected": rejected_after(
            lambda value: value["capture_contract"]["semantic_sketch"].__setitem__(
                "countsketch_dimension", 128
            )
        ),
        "complete_identity_tamper_rejected": (
            lambda changed: (
                changed["source"]["model_identity"].__setitem__(
                    "revision", "wrong"
                )
                is None
                and _rejects(
                    lambda: _validate_plan_binding(
                        plan=changed,
                        job=job,
                        preflight=complete_preflight,
                        project_root=project_root,
                    )
                )
            )
        )(copy.deepcopy(plan)),
        "complete_output_escape_rejected": (
            lambda changed: (
                changed["capture_contract"].__setitem__(
                    "output_dir", "artifacts/outside-run-root/job"
                )
                is None
                and _rejects(
                    lambda: _validate_plan_binding(
                        plan=changed,
                        job=job,
                        preflight=complete_preflight,
                        project_root=project_root,
                    )
                )
            )
        )(copy.deepcopy(plan)),
        "binding_imports_no_model_frameworks": before
        == (MODEL_FRAMEWORKS & set(sys.modules)),
        "binding_never_authorizes_execution": (
            incomplete_result["execution_ready"] is False
            and complete_result["execution_ready"] is False
        ),
    }
    if not all(controls.values()):
        raise ValueError(
            "guarded capture job binding benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-guarded-capture-job-binding-benchmark.v1",
        "status": "pass",
        "controls": controls,
        "summary": {"controls": len(controls), "passed": sum(controls.values())},
        "results": {
            "job_allocation_id": job["job_allocation_id"],
            "packet_id": job["packet_id"],
            "role": job["role"],
            "incomplete_bindings_ready": False,
            "complete_identity_shape_still_requires_runtime_admission": True,
        },
        "source": {
            "binding_verifier_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/verify_vpd_guarded_capture_job_binding_v1.py"
            ),
            "benchmark_sha256": sha256_file(Path(__file__)),
        },
        "scientific_boundary": {
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "capture_started": False,
            "output_reserved": False,
            "weights_mutated": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in observed.items() if key != "benchmark_id"}
    if observed.get("benchmark_id") != content_sha256(body):
        raise ValueError("guarded job binding benchmark address differs")
    if observed != build_report(project_root):
        raise ValueError("guarded job binding benchmark is not reproducible")
    return {
        "schema": "sch.vpd-guarded-capture-job-binding-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": observed["benchmark_id"],
        "controls": observed["summary"]["controls"],
        "capture_started": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark VPD guarded job bindings.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.output is not None:
        result = build_report(project_root)
        write_json(args.output, result)
    else:
        result = verify_benchmark(args.verify, project_root=project_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
