from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_runtime_v1 import (
    _fake_runtime_factory,
)
from experiments.authz_mechinterp.benchmark_canonical_vpd_sae_training_capture_runtime_v2 import (
    _factors_restored,
    _fixture,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.run_canonical_vpd_capture_job_v3 import (
    run_capture_job,
    verify_capture_execution_receipt,
)
from experiments.authz_mechinterp.vpd_cluster_capture_campaign_guard_v1 import (
    CaptureCampaignGuard,
)
from experiments.authz_mechinterp.vpd_guarded_capture_runtime_adapter_v1 import (
    CaptureJobAllocation,
)


def _rejects(call: Callable[[], Any]) -> bool:
    try:
        call()
    except (RuntimeError, TypeError, ValueError):
        return True
    return False


def _allocation(plan: dict[str, Any], payload_bytes: int = 1000) -> CaptureJobAllocation:
    return CaptureJobAllocation(
        batch_forwards=int(plan["resource_ceiling"]["maximum_forward_passes"]),
        episode_forward_equivalents=(
            2
            * int(plan["resource_ceiling"]["canonical_components"])
            * int(plan["resource_ceiling"]["dataset_rows"])
        ),
        float32_effect_payload_bytes=payload_bytes,
    )


def _guard(plan: dict[str, Any], allocation: CaptureJobAllocation) -> CaptureCampaignGuard:
    ceilings = {
        "compute_jobs": 1,
        "batch_forwards": allocation.batch_forwards,
        "episode_forward_equivalents": allocation.episode_forward_equivalents,
        "float32_effect_payload_bytes": allocation.float32_effect_payload_bytes,
    }
    return CaptureCampaignGuard(
        preflight_id="synthetic-preflight-not-production",
        role=str(plan["source"]["semantic_split"]["split_role"]),
        split_role_id="synthetic-split-role-not-production",
        prerequisites=[],
        ceilings=ceilings,
        global_ceilings=dict(ceilings),
        wall_time_ceiling_seconds=3600,
        cost_ceiling_usd=1.0,
        simulation_only=True,
        runtime_admission_real=False,
        output_reserved=False,
    )


def _tracked_factory(
    guard: CaptureCampaignGuard,
    *,
    fail_callback: bool = False,
    adapter_name: str = "default",
) -> tuple[Callable[..., Any], dict[str, Any]]:
    base, state = _fake_runtime_factory(
        fail_callback=fail_callback, adapter_name=adapter_name
    )
    state["loader_calls"] = 0
    state["guard_at_load"] = None

    def factory(plan: dict[str, Any], project_root: Path) -> Any:
        state["loader_calls"] += 1
        state["guard_at_load"] = guard.snapshot()
        return base(plan, project_root)

    return factory, state


def build_report(project_root: Path) -> dict[str, Any]:
    del project_root
    before = MODEL_FRAMEWORKS & set(sys.modules)
    with tempfile.TemporaryDirectory(prefix="vpd-capture-v3-") as temporary:
        root = Path(temporary)
        plan, plan_path, authorization_path = _fixture(
            root, output_dir="captures/v3-success"
        )
        allocation = _allocation(plan)
        guard = _guard(plan, allocation)
        factory, state = _tracked_factory(guard)
        execution = run_capture_job(
            plan_path=plan_path,
            authorization_path=authorization_path,
            project_root=root,
            campaign_guard=guard,
            resource_observer=lambda: (1.0, 0.01),
            runtime_factory=factory,
            synthetic_job_allocation=allocation,
        )
        output = root / plan["capture_contract"]["output_dir"]
        receipt_path = output / "capture_execution_receipt.json"
        verification = verify_capture_execution_receipt(
            receipt_path,
            plan_path=plan_path,
            authorization_path=authorization_path,
            project_root=root,
        )
        guard_receipt = json.loads(
            (output / "guarded_job_receipt.json").read_text(encoding="utf-8")
        )

        bad_plan, bad_plan_path, bad_auth_path = _fixture(
            root, output_dir="captures/v3-bad-allocation"
        )
        bad_allocation = _allocation(bad_plan)
        bad_allocation = CaptureJobAllocation(
            bad_allocation.batch_forwards - 1,
            bad_allocation.episode_forward_equivalents,
            bad_allocation.float32_effect_payload_bytes,
        )
        bad_guard = _guard(bad_plan, bad_allocation)
        bad_factory, bad_state = _tracked_factory(bad_guard)
        bad_allocation_rejected = _rejects(
            lambda: run_capture_job(
                plan_path=bad_plan_path,
                authorization_path=bad_auth_path,
                project_root=root,
                campaign_guard=bad_guard,
                resource_observer=lambda: (1.0, 0.0),
                runtime_factory=bad_factory,
                synthetic_job_allocation=bad_allocation,
            )
        )

        fail_plan, fail_plan_path, fail_auth_path = _fixture(
            root, output_dir="captures/v3-callback-failure"
        )
        fail_allocation = _allocation(fail_plan)
        fail_guard = _guard(fail_plan, fail_allocation)
        fail_factory, fail_state = _tracked_factory(fail_guard, fail_callback=True)
        callback_failure_rejected = _rejects(
            lambda: run_capture_job(
                plan_path=fail_plan_path,
                authorization_path=fail_auth_path,
                project_root=root,
                campaign_guard=fail_guard,
                resource_observer=lambda: (1.0, 0.0),
                runtime_factory=fail_factory,
                synthetic_job_allocation=fail_allocation,
            )
        )

        wrong_plan, wrong_plan_path, wrong_auth_path = _fixture(
            root, output_dir="captures/v3-wrong-adapter"
        )
        wrong_allocation = _allocation(wrong_plan)
        wrong_guard = _guard(wrong_plan, wrong_allocation)
        wrong_factory, wrong_state = _tracked_factory(
            wrong_guard, adapter_name="wrong"
        )
        wrong_adapter_rejected = _rejects(
            lambda: run_capture_job(
                plan_path=wrong_plan_path,
                authorization_path=wrong_auth_path,
                project_root=root,
                campaign_guard=wrong_guard,
                resource_observer=lambda: (1.0, 0.0),
                runtime_factory=wrong_factory,
                synthetic_job_allocation=wrong_allocation,
            )
        )

        original_receipt = receipt_path.read_text(encoding="utf-8")
        changed = json.loads(original_receipt)
        changed["scientific_boundary"]["sae_training_authorized"] = True
        write_json(receipt_path, changed)
        wrapper_tamper_rejected = _rejects(
            lambda: verify_capture_execution_receipt(
                receipt_path,
                plan_path=plan_path,
                authorization_path=authorization_path,
                project_root=root,
            )
        )
        receipt_path.write_text(original_receipt, encoding="utf-8")

        production_context_rejected_for_synthetic = _rejects(
            lambda: run_capture_job(
                plan_path=plan_path,
                authorization_path=authorization_path,
                project_root=root,
                campaign_guard=_guard(plan, allocation),
                resource_observer=lambda: (1.0, 0.0),
                runtime_factory=factory,
                allocation_manifest_path=root / "not-used.json",
                job_allocation_id="0" * 64,
                reservation_receipt_path=root / "not-used-receipt.json",
                synthetic_job_allocation=allocation,
            )
        )

        controls = {
            "synthetic_v3_execution_completes": execution["status"]
            == "complete_guarded_canonical_category_capture_v3",
            "reservation_precedes_loader": (
                state["guard_at_load"]["counters"]["compute_jobs"] == 1
                and state["guard_at_load"]["counters"][
                    "float32_effect_payload_bytes"
                ]
                == allocation.float32_effect_payload_bytes
            ),
            "all_callbacks_are_preaccounted": guard.counters["batch_forwards"]
            == allocation.batch_forwards,
            "all_episode_equivalents_are_exact": guard.counters[
                "episode_forward_equivalents"
            ]
            == allocation.episode_forward_equivalents,
            "guarded_job_receipt_is_exact": guard_receipt["observed"]
            == guard_receipt["allocation"],
            "v3_and_v2_receipts_verify": (
                verification["status"] == "pass"
                and verification["capture_execution_v3_id"]
                == execution["capture_execution_v3_id"]
            ),
            "three_receipts_are_persisted": all(
                (output / name).is_file()
                for name in (
                    "capture_execution_receipt.json",
                    "capture_execution_receipt_v2.json",
                    "guarded_job_receipt.json",
                )
            ),
            "runtime_closes_and_factors_restore": (
                state["closed"] is True
                and _factors_restored(state)
            ),
            "bad_allocation_rejected_before_loader": (
                bad_allocation_rejected and bad_state["loader_calls"] == 0
            ),
            "bad_allocation_creates_no_output": not (
                root / bad_plan["capture_contract"]["output_dir"]
            ).exists(),
            "callback_failure_remains_charged": (
                callback_failure_rejected
                and fail_guard.counters["batch_forwards"] == 1
                and fail_guard.counters["episode_forward_equivalents"] > 0
            ),
            "callback_failure_creates_no_output": not (
                root / fail_plan["capture_contract"]["output_dir"]
            ).exists(),
            "wrong_adapter_rejected_after_reserved_load": (
                wrong_adapter_rejected
                and wrong_state["loader_calls"] == 1
                and wrong_guard.counters["compute_jobs"] == 1
            ),
            "wrong_adapter_creates_no_output": not (
                root / wrong_plan["capture_contract"]["output_dir"]
            ).exists(),
            "synthetic_production_context_mix_rejected": production_context_rejected_for_synthetic,
            "v3_wrapper_tamper_rejected": wrapper_tamper_rejected,
            "runner_imports_no_model_frameworks": before
            == (MODEL_FRAMEWORKS & set(sys.modules)),
            "runner_remains_non_authorizing": all(
                execution["scientific_boundary"][field] in (False, 0)
                for field in (
                    "base_model_weights_mutated",
                    "adapter_weights_persisted",
                    "generated_tokens",
                    "real_actions_authorized",
                    "sae_dictionary_training_authorized",
                    "authorization_control_release_authorized",
                    "sae_training_authorized",
                )
            ),
        }
    if not all(controls.values()):
        raise ValueError(
            "canonical capture V3 runtime benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-canonical-capture-job-runtime-benchmark.v3",
        "status": "pass",
        "controls": controls,
        "summary": {"controls": len(controls), "passed": sum(controls.values())},
        "results": {
            "capture_execution_v3_id": execution["capture_execution_v3_id"],
            "guarded_job_receipt_id": verification["guarded_job_receipt_id"],
            "forward_passes": allocation.batch_forwards,
            "episode_forward_equivalents": allocation.episode_forward_equivalents,
        },
        "source": {
            "runner_v3_sha256": sha256_file(
                Path(__file__).with_name("run_canonical_vpd_capture_job_v3.py")
            ),
            "benchmark_sha256": sha256_file(Path(__file__)),
        },
        "scientific_boundary": {
            "synthetic_runtime_only": True,
            "production_runner_code_path_present": True,
            "production_runner_empirically_exercised": False,
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
        raise ValueError("canonical capture V3 benchmark address differs")
    if observed != build_report(project_root):
        raise ValueError("canonical capture V3 benchmark is not reproducible")
    return {
        "schema": "sch.vpd-canonical-capture-job-runtime-benchmark-verification.v3",
        "status": "pass",
        "benchmark_id": observed["benchmark_id"],
        "controls": observed["summary"]["controls"],
        "production_runner_empirically_exercised": False,
        "capture_authorized": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark canonical capture runner V3.")
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
