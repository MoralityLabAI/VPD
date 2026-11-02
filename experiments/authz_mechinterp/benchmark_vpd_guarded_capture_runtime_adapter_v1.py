from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
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
from experiments.authz_mechinterp.run_canonical_vpd_capture_job_v1 import CaptureRuntime
from experiments.authz_mechinterp.vpd_cluster_capture_campaign_guard_v1 import (
    ROLE_PREREQUISITES,
    build_guard,
)
from experiments.authz_mechinterp.vpd_guarded_capture_runtime_adapter_v1 import (
    CaptureJobAllocation,
    reserve_guarded_capture_job,
)


def _rejects(call: Callable[[], Any]) -> bool:
    try:
        call()
    except (TypeError, ValueError):
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
            (project_root / OUTPUT_PARENT / "synthetic-adapter-do-not-create").resolve()
        ),
    }


def _guard(preflight: dict[str, Any]) -> Any:
    return build_guard(
        preflight=preflight,
        role="discovery",
        completed_events=ROLE_PREREQUISITES["discovery"],
        simulation_only=True,
    )


def _runtime(callback: Callable[[str, list[dict[str, Any]]], Any], state: dict[str, Any]) -> CaptureRuntime:
    def close() -> None:
        state["close_calls"] = state.get("close_calls", 0) + 1

    return CaptureRuntime(
        model=object(),
        adapter_name="synthetic-not-loaded",
        capture_batch_callback=callback,
        no_grad_context_factory=nullcontext,
        close=close,
        runtime_kind="synthetic_callback_no_model_framework",
        actual_model_loaded=False,
    )


def build_report(project_root: Path) -> dict[str, Any]:
    before = MODEL_FRAMEWORKS & set(sys.modules)
    preflight = build_preflight(project_root=project_root, bindings=_bindings(project_root))

    observer_state = {"elapsed": 1.0, "cost": 0.01}
    observer = lambda: (observer_state["elapsed"], observer_state["cost"])
    main_state: dict[str, Any] = {"callback_snapshots": [], "close_calls": 0}
    main_guard = _guard(preflight)

    def main_callback(category: str, batch: list[dict[str, Any]]) -> list[str]:
        main_state["callback_snapshots"].append(main_guard.snapshot())
        return [category for _ in batch]

    main_session = reserve_guarded_capture_job(
        runtime=_runtime(main_callback, main_state),
        guard=main_guard,
        allocation=CaptureJobAllocation(
            batch_forwards=2,
            episode_forward_equivalents=3,
            float32_effect_payload_bytes=100,
        ),
        resource_observer=observer,
    )
    reservation_snapshot = main_guard.snapshot()
    wrapped = main_session.as_capture_runtime()
    first_result = wrapped.capture_batch_callback("valid_authority", [{}, {}])
    observer_state.update(elapsed=2.0, cost=0.02)
    second_result = wrapped.capture_batch_callback("revoked", [{}])
    receipt = main_session.finalize_job()
    receipt_body = {key: value for key, value in receipt.items() if key != "job_receipt_id"}
    wrapped.close()
    wrapped.close()

    failed_state = {"calls": 0}
    failed_guard = _guard(preflight)

    def failed_callback(_category: str, _batch: list[dict[str, Any]]) -> Any:
        failed_state["calls"] += 1
        raise RuntimeError("synthetic callback failure")

    failed_session = reserve_guarded_capture_job(
        runtime=_runtime(failed_callback, failed_state),
        guard=failed_guard,
        allocation=CaptureJobAllocation(1, 1, 4),
        resource_observer=lambda: (1.0, 0.0),
    )
    try:
        failed_session.capture_batch("revoked", [{}])
    except RuntimeError:
        pass

    forward_state = {"calls": 0}
    forward_guard = _guard(preflight)

    def counted_callback(_category: str, batch: list[dict[str, Any]]) -> list[Any]:
        forward_state["calls"] += 1
        return batch

    forward_session = reserve_guarded_capture_job(
        runtime=_runtime(counted_callback, forward_state),
        guard=forward_guard,
        allocation=CaptureJobAllocation(1, 2, 8),
        resource_observer=lambda: (1.0, 0.0),
    )
    forward_session.capture_batch("valid_authority", [{}, {}])
    forward_over_rejected = _rejects(
        lambda: forward_session.capture_batch("valid_authority", [{}])
    )

    episode_state = {"calls": 0}
    episode_session = reserve_guarded_capture_job(
        runtime=_runtime(counted_callback, episode_state),
        guard=_guard(preflight),
        allocation=CaptureJobAllocation(2, 2, 8),
        resource_observer=lambda: (1.0, 0.0),
    )
    episode_over_rejected = _rejects(
        lambda: episode_session.capture_batch("valid_authority", [{}, {}, {}])
    )

    incomplete_session = reserve_guarded_capture_job(
        runtime=_runtime(counted_callback, {}),
        guard=_guard(preflight),
        allocation=CaptureJobAllocation(2, 2, 8),
        resource_observer=lambda: (1.0, 0.0),
    )

    nonmonotonic_state = {"calls": 0}
    observations = iter(((1.0, 0.1), (0.5, 0.1)))
    nonmonotonic_session = reserve_guarded_capture_job(
        runtime=_runtime(counted_callback, nonmonotonic_state),
        guard=_guard(preflight),
        allocation=CaptureJobAllocation(1, 1, 4),
        resource_observer=lambda: next(observations),
    )
    nonmonotonic_rejected = _rejects(
        lambda: nonmonotonic_session.capture_batch("revoked", [{}])
    )

    controls = {
        "job_and_payload_reserved_before_first_callback": (
            reservation_snapshot["counters"]["compute_jobs"] == 1
            and reservation_snapshot["counters"]["float32_effect_payload_bytes"] == 100
            and reservation_snapshot["counters"]["batch_forwards"] == 0
            and reservation_snapshot["counters"]["episode_forward_equivalents"] == 0
        ),
        "first_callback_observes_preincremented_counters": (
            main_state["callback_snapshots"][0]["counters"]["batch_forwards"] == 1
            and main_state["callback_snapshots"][0]["counters"]["episode_forward_equivalents"] == 2
        ),
        "second_partial_batch_charged_by_actual_length": (
            main_state["callback_snapshots"][1]["counters"]["batch_forwards"] == 2
            and main_state["callback_snapshots"][1]["counters"]["episode_forward_equivalents"] == 3
        ),
        "synthetic_callback_results_preserved": (
            first_result == ["valid_authority", "valid_authority"]
            and second_result == ["revoked"]
        ),
        "failed_callback_remains_charged": (
            failed_state["calls"] == 1
            and failed_guard.counters["batch_forwards"] == 1
            and failed_guard.counters["episode_forward_equivalents"] == 1
        ),
        "per_job_forward_overage_stops_before_callback": (
            forward_over_rejected and forward_state["calls"] == 1
        ),
        "per_job_episode_overage_stops_before_callback": (
            episode_over_rejected and episode_state["calls"] == 0
        ),
        "nonmonotonic_observation_stops_before_callback": (
            nonmonotonic_rejected and nonmonotonic_state["calls"] == 0
        ),
        "empty_batch_rejected": _rejects(
            lambda: incomplete_session.capture_batch("revoked", [])
        ),
        "incomplete_job_cannot_finalize": _rejects(incomplete_session.finalize_job),
        "exact_job_receipt_is_content_addressed": (
            receipt["job_receipt_id"] == content_sha256(receipt_body)
            and receipt["observed"] == receipt["allocation"]
        ),
        "job_finalize_is_single_use": _rejects(main_session.finalize_job),
        "close_delegates_exactly_once": main_state["close_calls"] == 1,
        "capture_after_close_rejected": _rejects(
            lambda: main_session.capture_batch("revoked", [{}])
        ),
        "invalid_allocation_rejected": _rejects(
            lambda: reserve_guarded_capture_job(
                runtime=_runtime(counted_callback, {}),
                guard=_guard(preflight),
                allocation=CaptureJobAllocation(0, 1, 4),
                resource_observer=lambda: (1.0, 0.0),
            )
        ),
        "role_payload_overage_rejected_before_session": _rejects(
            lambda: reserve_guarded_capture_job(
                runtime=_runtime(counted_callback, {}),
                guard=_guard(preflight),
                allocation=CaptureJobAllocation(
                    1,
                    1,
                    _guard(preflight).ceilings["float32_effect_payload_bytes"] + 1,
                ),
                resource_observer=lambda: (1.0, 0.0),
            )
        ),
        "role_forward_allocation_overage_rejected_before_session": _rejects(
            lambda: reserve_guarded_capture_job(
                runtime=_runtime(counted_callback, {}),
                guard=_guard(preflight),
                allocation=CaptureJobAllocation(
                    _guard(preflight).ceilings["batch_forwards"] + 1,
                    1,
                    4,
                ),
                resource_observer=lambda: (1.0, 0.0),
            )
        ),
        "wrapped_runtime_preserves_nonloading_boundary": (
            wrapped.runtime_kind == "guarded:synthetic_callback_no_model_framework"
            and wrapped.actual_model_loaded is False
        ),
        "adapter_imports_no_model_frameworks": before == (MODEL_FRAMEWORKS & set(sys.modules)),
        "receipt_remains_non_authorizing": all(
            receipt["scientific_boundary"][field] in (False, 0)
            for field in (
                "adapter_authorizes_capture",
                "adapter_authorizes_model_loading",
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
            "guarded capture runtime adapter benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-guarded-capture-runtime-adapter-benchmark.v1",
        "status": "pass",
        "controls": controls,
        "summary": {"controls": len(controls), "passed": sum(controls.values())},
        "results": {
            "synthetic_preflight_id": preflight["preflight_id"],
            "exact_job_receipt_id": receipt["job_receipt_id"],
            "failed_callback_charged_batch_forwards": failed_guard.counters["batch_forwards"],
            "failed_callback_charged_episode_forward_equivalents": failed_guard.counters[
                "episode_forward_equivalents"
            ],
        },
        "source": {
            "adapter_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/vpd_guarded_capture_runtime_adapter_v1.py"
            ),
            "benchmark_sha256": sha256_file(Path(__file__)),
        },
        "scientific_boundary": {
            "synthetic_callbacks_only": True,
            "model_framework_imported": False,
            "model_loaded": False,
            "real_model_inference_performed": False,
            "persistent_output_created": False,
            "weights_mutated": False,
            "capture_authorized": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in observed.items() if key != "benchmark_id"}
    if observed.get("benchmark_id") != content_sha256(body) or observed != build_report(project_root):
        raise ValueError("guarded capture runtime adapter benchmark is not reproducible")
    return {
        "schema": "sch.vpd-guarded-capture-runtime-adapter-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": observed["benchmark_id"],
        "controls": observed["summary"]["controls"],
        "real_model_inference_performed": False,
        "capture_authorized": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the guarded capture runtime adapter.")
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
