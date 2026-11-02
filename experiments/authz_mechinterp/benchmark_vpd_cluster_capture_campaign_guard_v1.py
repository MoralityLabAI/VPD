from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
    OUTPUT_PARENT,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_empirical_preflight_v2 import (
    build_preflight,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.vpd_cluster_capture_campaign_guard_v1 import (
    COUNTER_FIELDS,
    ROLE_PREREQUISITES,
    build_guard,
    verify_guard_receipt,
)


def _rejects(call: Callable[[], Any]) -> bool:
    try:
        call()
    except ValueError:
        return True
    return False


def _bindings(project_root: Path) -> dict[str, Any]:
    return {
        "model_id": "synthetic/model-not-loaded",
        "model_revision": "synthetic-revision-not-loaded",
        "model_files_manifest_sha256": "1" * 64,
        "adapter_path": str(Path("C:/synthetic/adapter-not-opened").resolve()),
        "adapter_weights_sha256": "2" * 64,
        "adapter_config_sha256": "3" * 64,
        "device": "cpu",
        "wall_time_ceiling_seconds": 3600,
        "cost_ceiling_usd": 1.0,
        "output_root": str(
            (project_root / OUTPUT_PARENT / "synthetic-guard-do-not-create").resolve()
        ),
    }


def _new_guard(preflight: dict[str, Any], role: str = "discovery") -> Any:
    return build_guard(
        preflight=preflight,
        role=role,
        completed_events=ROLE_PREREQUISITES[role],
        simulation_only=True,
    )


def build_report(project_root: Path) -> dict[str, Any]:
    before = MODEL_FRAMEWORKS & set(sys.modules)
    preflight = build_preflight(project_root=project_root, bindings=_bindings(project_root))
    discovery = _new_guard(preflight)
    discovery.record(
        **discovery.ceilings,
        elapsed_seconds=300.0,
        cost_usd=0.25,
    )
    receipt = discovery.finalize()
    with tempfile.TemporaryDirectory(prefix="vpd-campaign-guard-") as temporary:
        receipt_path = Path(temporary) / "guard-receipt.json"
        write_json(receipt_path, receipt)
        receipt_verification = verify_guard_receipt(receipt_path, preflight=preflight)

    def over_counter(name: str) -> bool:
        guard = _new_guard(preflight)
        values = {field: 0 for field in COUNTER_FIELDS}
        values[name] = guard.ceilings[name] + 1
        return _rejects(
            lambda: guard.record(
                **values,
                elapsed_seconds=1.0,
                cost_usd=0.0,
            )
        )

    incomplete = _new_guard(preflight)
    incomplete.record(
        compute_jobs=1,
        elapsed_seconds=1.0,
        cost_usd=0.0,
    )
    after = MODEL_FRAMEWORKS & set(sys.modules)
    controls = {
        "discovery_is_only_role_with_no_prerequisites": (
            _new_guard(preflight, "discovery").prerequisites == []
        ),
        "calibration_stage_skip_rejected": _rejects(
            lambda: build_guard(
                preflight=preflight,
                role="calibration",
                completed_events=[],
                simulation_only=True,
            )
        ),
        "cluster_validation_stage_skip_rejected": _rejects(
            lambda: build_guard(
                preflight=preflight,
                role="cluster_validation",
                completed_events=ROLE_PREREQUISITES["calibration"],
                simulation_only=True,
            )
        ),
        "sae_training_stage_skip_rejected": _rejects(
            lambda: build_guard(
                preflight=preflight,
                role="sae_training",
                completed_events=ROLE_PREREQUISITES["cluster_validation"],
                simulation_only=True,
            )
        ),
        "final_evaluation_role_rejected": _rejects(
            lambda: build_guard(
                preflight=preflight,
                role="final_evaluation",
                completed_events=[],
                simulation_only=True,
            )
        ),
        "production_start_without_admission_rejected": _rejects(
            lambda: build_guard(
                preflight=preflight,
                role="discovery",
                completed_events=[],
                simulation_only=False,
            )
        ),
        "compute_job_overage_rejected": over_counter("compute_jobs"),
        "batch_forward_overage_rejected": over_counter("batch_forwards"),
        "episode_forward_overage_rejected": over_counter(
            "episode_forward_equivalents"
        ),
        "payload_overage_rejected": over_counter("float32_effect_payload_bytes"),
        "generated_token_attempt_rejected": _rejects(
            lambda: _new_guard(preflight).record(
                elapsed_seconds=1.0, cost_usd=0.0, generated_tokens=1
            )
        ),
        "real_action_attempt_rejected": _rejects(
            lambda: _new_guard(preflight).record(
                elapsed_seconds=1.0, cost_usd=0.0, real_actions=1
            )
        ),
        "persistent_mutation_attempt_rejected": _rejects(
            lambda: _new_guard(preflight).record(
                elapsed_seconds=1.0,
                cost_usd=0.0,
                persistent_weight_mutations=1,
            )
        ),
        "wall_time_overage_rejected": _rejects(
            lambda: _new_guard(preflight).record(
                elapsed_seconds=3600.001, cost_usd=0.0
            )
        ),
        "cost_overage_rejected": _rejects(
            lambda: _new_guard(preflight).record(
                elapsed_seconds=1.0, cost_usd=1.001
            )
        ),
        "nonmonotonic_wall_time_rejected": (
            lambda guard: (
                guard.record(elapsed_seconds=2.0, cost_usd=0.1) is not None
                and _rejects(
                    lambda: guard.record(elapsed_seconds=1.0, cost_usd=0.1)
                )
            )
        )(_new_guard(preflight)),
        "nonmonotonic_cost_rejected": (
            lambda guard: (
                guard.record(elapsed_seconds=1.0, cost_usd=0.2) is not None
                and _rejects(
                    lambda: guard.record(elapsed_seconds=2.0, cost_usd=0.1)
                )
            )
        )(_new_guard(preflight)),
        "incomplete_role_cannot_finalize": _rejects(incomplete.finalize),
        "exact_role_receipt_verifies": (
            receipt_verification["status"] == "pass"
            and receipt_verification["role"] == "discovery"
            and receipt_verification["execution_authorized"] is False
        ),
        "all_four_role_ceilings_sum_to_global_ceiling": (
            sum(
                _new_guard(preflight, role).ceilings["compute_jobs"]
                for role in ROLE_PREREQUISITES
            )
            == preflight["resource_ceiling"]["maximum_compute_jobs"]
            and sum(
                _new_guard(preflight, role).ceilings["batch_forwards"]
                for role in ROLE_PREREQUISITES
            )
            == preflight["resource_ceiling"][
                "maximum_batch_forwards_at_batch_size_4"
            ]
            and sum(
                _new_guard(preflight, role).ceilings[
                    "episode_forward_equivalents"
                ]
                for role in ROLE_PREREQUISITES
            )
            == preflight["resource_ceiling"]["maximum_episode_forward_equivalents"]
            and sum(
                _new_guard(preflight, role).ceilings[
                    "float32_effect_payload_bytes"
                ]
                for role in ROLE_PREREQUISITES
            )
            == preflight["resource_ceiling"][
                "estimated_float32_effect_payload_bytes"
            ]
        ),
        "guard_imports_no_model_frameworks": before == after,
        "guard_receipt_remains_non_authorizing": all(
            receipt["scientific_boundary"][field] in (False, 0)
            for field in (
                "model_framework_imported",
                "model_loaded_by_guard",
                "model_inference_performed_by_guard",
                "generated_tokens",
                "real_actions",
                "persistent_weight_mutations",
                "sae_dictionary_training_authorized",
                "final_evaluation_authorized",
                "execution_authorized",
            )
        ),
    }
    if not all(controls.values()):
        raise ValueError(
            "cluster capture campaign guard benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-capture-campaign-guard-benchmark.v1",
        "status": "pass",
        "controls": controls,
        "summary": {"controls": len(controls), "passed": sum(controls.values())},
        "results": {
            "synthetic_preflight_id": preflight["preflight_id"],
            "discovery_guard_receipt_id": receipt["guard_receipt_id"],
            "role_prerequisites": ROLE_PREREQUISITES,
        },
        "source": {
            "guard_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/vpd_cluster_capture_campaign_guard_v1.py"
            ),
            "benchmark_sha256": sha256_file(Path(__file__)),
        },
        "scientific_boundary": {
            "synthetic_counter_simulation_only": True,
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "capture_started": False,
            "persistent_output_created": False,
            "weights_mutated": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    identifier = str(observed.get("benchmark_id", ""))
    body = {key: value for key, value in observed.items() if key != "benchmark_id"}
    if identifier != content_sha256(body) or observed != build_report(project_root):
        raise ValueError("cluster capture campaign guard benchmark is not reproducible")
    return {
        "schema": "sch.vpd-cluster-capture-campaign-guard-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "controls": observed["summary"]["controls"],
        "roles": len(observed["results"]["role_prerequisites"]),
        "capture_started": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the VPD cluster capture campaign guard."
    )
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
