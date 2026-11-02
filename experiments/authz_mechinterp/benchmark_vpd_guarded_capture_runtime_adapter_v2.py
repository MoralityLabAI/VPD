from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.benchmark_vpd_cluster_capture_campaign_guard_v1 import (
    _bindings,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
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
from experiments.authz_mechinterp.vpd_cluster_capture_campaign_guard_v1 import build_guard
from experiments.authz_mechinterp.vpd_guarded_capture_runtime_adapter_v1 import (
    CaptureJobAllocation,
)
from experiments.authz_mechinterp.vpd_guarded_capture_runtime_adapter_v2 import (
    reserve_guarded_capture_job_before_runtime_load,
)


def _rejects(call: Callable[[], Any]) -> bool:
    try:
        call()
    except (RuntimeError, TypeError, ValueError):
        return True
    return False


def _guard(preflight: dict[str, Any]) -> Any:
    return build_guard(
        preflight=preflight,
        role="discovery",
        completed_events=[],
        simulation_only=True,
    )


def _runtime(
    callback: Callable[[str, list[dict[str, Any]]], Any], state: dict[str, Any]
) -> CaptureRuntime:
    state["loader_calls"] = state.get("loader_calls", 0) + 1

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
    preflight = build_preflight(
        project_root=project_root, bindings=_bindings(project_root)
    )
    guard = _guard(preflight)
    state: dict[str, Any] = {
        "loader_calls": 0,
        "close_calls": 0,
        "callback_snapshots": [],
    }
    observer_state = {"elapsed": 1.0, "cost": 0.01}
    observer = lambda: (observer_state["elapsed"], observer_state["cost"])
    reservation = reserve_guarded_capture_job_before_runtime_load(
        guard=guard,
        allocation=CaptureJobAllocation(2, 3, 100),
        resource_observer=observer,
    )
    post_reservation = guard.snapshot()

    def callback(category: str, batch: list[dict[str, Any]]) -> list[str]:
        state["callback_snapshots"].append(guard.snapshot())
        return [category for _ in batch]

    session = reservation.attach_runtime(_runtime(callback, state))
    wrapped = session.as_capture_runtime()
    first = wrapped.capture_batch_callback("valid_authority", [{}, {}])
    observer_state.update(elapsed=2.0, cost=0.02)
    second = wrapped.capture_batch_callback("revoked", [{}])
    receipt = session.finalize_job()
    wrapped.close()
    wrapped.close()

    over_guard = _guard(preflight)
    over_state = {"loader_calls": 0}
    over_rejected = _rejects(
        lambda: reserve_guarded_capture_job_before_runtime_load(
            guard=over_guard,
            allocation=CaptureJobAllocation(
                over_guard.ceilings["batch_forwards"] + 1,
                1,
                4,
            ),
            resource_observer=lambda: (1.0, 0.0),
        )
    )

    failure_guard = _guard(preflight)
    failure_state = {"loader_calls": 0, "callback_calls": 0}
    failure_reservation = reserve_guarded_capture_job_before_runtime_load(
        guard=failure_guard,
        allocation=CaptureJobAllocation(1, 1, 4),
        resource_observer=lambda: (1.0, 0.0),
    )

    def failed_callback(_category: str, _batch: list[dict[str, Any]]) -> Any:
        failure_state["callback_calls"] += 1
        raise RuntimeError("synthetic callback failure")

    failed_session = failure_reservation.attach_runtime(
        _runtime(failed_callback, failure_state)
    )
    callback_failure_rejected = _rejects(
        lambda: failed_session.capture_batch("revoked", [{}])
    )

    no_attach_guard = _guard(preflight)
    no_attach_reservation = reserve_guarded_capture_job_before_runtime_load(
        guard=no_attach_guard,
        allocation=CaptureJobAllocation(1, 1, 4),
        resource_observer=lambda: (1.0, 0.0),
    )
    no_attach_state = {"loader_calls": 0}

    controls = {
        "job_and_payload_reserved_before_loader": (
            post_reservation["counters"]["compute_jobs"] == 1
            and post_reservation["counters"]["float32_effect_payload_bytes"] == 100
            and state["loader_calls"] == 1
            and reservation.guard_before_reservation["counters"]["compute_jobs"] == 0
        ),
        "reservation_object_contains_no_runtime": not hasattr(reservation, "runtime"),
        "runtime_attaches_after_reservation": reservation.runtime_attached is True,
        "runtime_attachment_is_single_use": _rejects(
            lambda: reservation.attach_runtime(_runtime(callback, {}))
        ),
        "first_callback_observes_preincrement": (
            state["callback_snapshots"][0]["counters"]["batch_forwards"] == 1
            and state["callback_snapshots"][0]["counters"][
                "episode_forward_equivalents"
            ]
            == 2
        ),
        "partial_batch_uses_actual_episode_count": (
            state["callback_snapshots"][1]["counters"][
                "episode_forward_equivalents"
            ]
            == 3
        ),
        "callback_values_preserved": (
            first == ["valid_authority", "valid_authority"]
            and second == ["revoked"]
        ),
        "exact_job_receipt_preserved": receipt["observed"] == receipt["allocation"],
        "close_delegates_exactly_once": state["close_calls"] == 1,
        "impossible_allocation_rejected_before_loader": (
            over_rejected and over_state["loader_calls"] == 0
        ),
        "failed_callback_remains_charged": (
            callback_failure_rejected
            and failure_state["callback_calls"] == 1
            and failure_guard.counters["batch_forwards"] == 1
            and failure_guard.counters["episode_forward_equivalents"] == 1
        ),
        "reservation_can_stop_without_loading": (
            no_attach_reservation.runtime_attached is False
            and no_attach_state["loader_calls"] == 0
            and no_attach_guard.counters["compute_jobs"] == 1
        ),
        "adapter_v2_imports_no_model_frameworks": before
        == (MODEL_FRAMEWORKS & set(sys.modules)),
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
            "guarded runtime adapter V2 benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-guarded-capture-runtime-adapter-benchmark.v2",
        "status": "pass",
        "controls": controls,
        "summary": {"controls": len(controls), "passed": sum(controls.values())},
        "results": {
            "synthetic_preflight_id": preflight["preflight_id"],
            "exact_job_receipt_id": receipt["job_receipt_id"],
            "reservation_precedes_runtime_load": True,
        },
        "source": {
            "adapter_v1_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/vpd_guarded_capture_runtime_adapter_v1.py"
            ),
            "adapter_v2_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/vpd_guarded_capture_runtime_adapter_v2.py"
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
    if observed.get("benchmark_id") != content_sha256(body):
        raise ValueError("guarded runtime adapter V2 benchmark address differs")
    if observed != build_report(project_root):
        raise ValueError("guarded runtime adapter V2 benchmark is not reproducible")
    return {
        "schema": "sch.vpd-guarded-capture-runtime-adapter-benchmark-verification.v2",
        "status": "pass",
        "benchmark_id": observed["benchmark_id"],
        "controls": observed["summary"]["controls"],
        "reservation_precedes_runtime_load": True,
        "capture_authorized": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark guarded runtime adapter V2.")
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
