from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
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
from experiments.authz_mechinterp.plan_vpd_guarded_capture_job_allocations_v1 import (
    ROLE_ORDER,
    build_manifest,
    verify_manifest,
)


def _rejects(call: Callable[[], Any]) -> bool:
    try:
        call()
    except (KeyError, TypeError, ValueError):
        return True
    return False


def _tamper_rejected(
    *, manifest: dict[str, Any], project_root: Path, mutate: Callable[[dict[str, Any]], None]
) -> bool:
    value = copy.deepcopy(manifest)
    mutate(value)
    with tempfile.TemporaryDirectory(prefix="vpd-allocation-tamper-") as temporary:
        path = Path(temporary) / "manifest.json"
        write_json(path, value)
        return _rejects(lambda: verify_manifest(path, project_root=project_root))


def build_report(project_root: Path) -> dict[str, Any]:
    before = MODEL_FRAMEWORKS & set(sys.modules)
    manifest_path = (
        project_root / "artifacts/vpd_guarded_capture_job_allocations_v1/manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verification = verify_manifest(manifest_path, project_root=project_root)
    role_packet_ids = {
        role: [job["packet_id"] for job in manifest["roles"][role]["jobs"]]
        for role in ROLE_ORDER
    }
    all_jobs = [
        job
        for role in ROLE_ORDER
        for job in manifest["roles"][role]["jobs"]
    ]
    role_totals = {
        role: manifest["roles"][role]["totals"] for role in ROLE_ORDER
    }
    after = MODEL_FRAMEWORKS & set(sys.modules)

    controls = {
        "allocation_manifest_reproduces": verification["status"] == "pass",
        "four_roles_exact": list(manifest["roles"]) == sorted(ROLE_ORDER),
        "sixteen_jobs_per_role": all(
            len(manifest["roles"][role]["jobs"]) == 16 for role in ROLE_ORDER
        ),
        "sixty_four_jobs_global": len(all_jobs) == 64,
        "same_exact_packet_ids_each_role": all(
            role_packet_ids[role] == role_packet_ids[ROLE_ORDER[0]]
            for role in ROLE_ORDER[1:]
        ),
        "job_allocation_ids_are_unique": len(
            {job["job_allocation_id"] for job in all_jobs}
        )
        == 64,
        "every_job_is_content_addressed": all(
            job["job_allocation_id"]
            == content_sha256(
                {key: value for key, value in job.items() if key != "job_allocation_id"}
            )
            for job in all_jobs
        ),
        "every_job_reserves_one_compute_job": all(
            job["allocation"]["compute_jobs"] == 1 for job in all_jobs
        ),
        "role_totals_are_positive_and_exact": all(
            all(value > 0 for value in role_totals[role].values())
            for role in ROLE_ORDER
        ),
        "global_totals_match_recovered_preflight": manifest["global_totals"]
        == {
            "compute_jobs": 64,
            "batch_forwards": 41_352_192,
            "episode_forward_equivalents": 164_900_736,
            "float32_effect_payload_bytes": 4_046_192_640,
        },
        "final_evaluation_absent": "final_evaluation" not in manifest["roles"],
        "role_removal_rejected": _tamper_rejected(
            manifest=manifest,
            project_root=project_root,
            mutate=lambda value: value["roles"].pop("calibration"),
        ),
        "job_payload_tamper_rejected": _tamper_rejected(
            manifest=manifest,
            project_root=project_root,
            mutate=lambda value: value["roles"]["discovery"]["jobs"][0][
                "allocation"
            ].__setitem__("float32_effect_payload_bytes", 1),
        ),
        "packet_substitution_rejected": _tamper_rejected(
            manifest=manifest,
            project_root=project_root,
            mutate=lambda value: value["roles"]["discovery"]["jobs"][0].__setitem__(
                "packet_id", value["roles"]["discovery"]["jobs"][1]["packet_id"]
            ),
        ),
        "dataset_substitution_rejected": _tamper_rejected(
            manifest=manifest,
            project_root=project_root,
            mutate=lambda value: value["roles"]["calibration"]["jobs"][0].__setitem__(
                "dataset_sha256",
                value["roles"]["discovery"]["jobs"][0]["dataset_sha256"],
            ),
        ),
        "manifest_address_tamper_rejected": _tamper_rejected(
            manifest=manifest,
            project_root=project_root,
            mutate=lambda value: value.__setitem__("allocation_manifest_id", "0" * 64),
        ),
        "allocation_imports_no_model_frameworks": before == after,
        "allocation_remains_non_authorizing": all(
            manifest["scientific_boundary"][field] in (False, 0)
            for field in (
                "model_framework_imported",
                "model_loaded",
                "model_inference_performed",
                "capture_started",
                "output_reserved",
                "weights_mutated",
                "generated_tokens",
                "real_actions",
                "sae_training_authorized",
                "execution_authorized",
            )
        ),
    }
    if not all(controls.values()):
        raise ValueError(
            "guarded capture allocation benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-guarded-capture-job-allocation-benchmark.v1",
        "status": "pass",
        "controls": controls,
        "summary": {"controls": len(controls), "passed": sum(controls.values())},
        "results": {
            "allocation_manifest_id": manifest["allocation_manifest_id"],
            "roles": len(manifest["roles"]),
            "jobs": len(all_jobs),
            "global_totals": manifest["global_totals"],
        },
        "source": {
            "allocation_planner_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/plan_vpd_guarded_capture_job_allocations_v1.py"
            ),
            "allocation_manifest_sha256": sha256_file(manifest_path),
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
        raise ValueError("guarded capture allocation benchmark address differs")
    if observed != build_report(project_root):
        raise ValueError("guarded capture allocation benchmark is not reproducible")
    return {
        "schema": "sch.vpd-guarded-capture-job-allocation-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": observed["benchmark_id"],
        "controls": observed["summary"]["controls"],
        "roles": observed["results"]["roles"],
        "jobs": observed["results"]["jobs"],
        "capture_started": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark exact guarded VPD allocations.")
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
