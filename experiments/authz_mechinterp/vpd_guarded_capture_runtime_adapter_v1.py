from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256
from experiments.authz_mechinterp.run_canonical_vpd_capture_job_v1 import CaptureRuntime
from experiments.authz_mechinterp.vpd_cluster_capture_campaign_guard_v1 import (
    CaptureCampaignGuard,
)


SCHEMA = "sch.vpd-guarded-capture-runtime-job-receipt.v1"
ResourceObserver = Callable[[], tuple[float, float]]


@dataclass(frozen=True)
class CaptureJobAllocation:
    batch_forwards: int
    episode_forward_equivalents: int
    float32_effect_payload_bytes: int

    def validate(self) -> None:
        values = (
            self.batch_forwards,
            self.episode_forward_equivalents,
            self.float32_effect_payload_bytes,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("guarded capture job allocation is invalid")


@dataclass
class GuardedCaptureRuntimeSession:
    runtime: CaptureRuntime
    guard: CaptureCampaignGuard
    allocation: CaptureJobAllocation
    resource_observer: ResourceObserver
    guard_before_reservation: dict[str, Any]
    calls: int = 0
    episodes: int = 0
    closed: bool = False
    job_finalized: bool = False

    def _observe(self) -> tuple[float, float]:
        observed = self.resource_observer()
        if (
            not isinstance(observed, tuple)
            or len(observed) != 2
        ):
            raise ValueError("guarded capture resource observation is invalid")
        return observed

    def capture_batch(
        self, category: str, batch: list[dict[str, Any]]
    ) -> Any:
        if self.closed:
            raise ValueError("guarded capture runtime is closed")
        if self.job_finalized:
            raise ValueError("guarded capture job is already finalized")
        if not isinstance(batch, list) or not batch:
            raise ValueError("guarded capture batch is empty or invalid")
        if self.calls + 1 > self.allocation.batch_forwards:
            raise ValueError("guarded capture job exceeded its forward allocation")
        if self.episodes + len(batch) > self.allocation.episode_forward_equivalents:
            raise ValueError("guarded capture job exceeded its episode allocation")
        elapsed_seconds, cost_usd = self._observe()
        self.guard.record(
            batch_forwards=1,
            episode_forward_equivalents=len(batch),
            elapsed_seconds=elapsed_seconds,
            cost_usd=cost_usd,
        )
        # These counters intentionally advance before the callback. A callback
        # failure still consumed the attempted compute and must remain charged.
        self.calls += 1
        self.episodes += len(batch)
        return self.runtime.capture_batch_callback(category, batch)

    def as_capture_runtime(self) -> CaptureRuntime:
        return CaptureRuntime(
            model=self.runtime.model,
            adapter_name=self.runtime.adapter_name,
            capture_batch_callback=self.capture_batch,
            no_grad_context_factory=self.runtime.no_grad_context_factory,
            close=self.close,
            runtime_kind=f"guarded:{self.runtime.runtime_kind}",
            actual_model_loaded=self.runtime.actual_model_loaded,
        )

    def finalize_job(self) -> dict[str, Any]:
        if self.job_finalized:
            raise ValueError("guarded capture job is already finalized")
        if (
            self.calls != self.allocation.batch_forwards
            or self.episodes != self.allocation.episode_forward_equivalents
        ):
            raise ValueError("guarded capture job cannot finalize incomplete counters")
        self.job_finalized = True
        body = {
            "schema": SCHEMA,
            "status": "complete_metered_job_capture_authority_separate",
            "preflight_id": self.guard.preflight_id,
            "role": self.guard.role,
            "split_role_id": self.guard.split_role_id,
            "allocation": {
                "compute_jobs": 1,
                "batch_forwards": self.allocation.batch_forwards,
                "episode_forward_equivalents": (
                    self.allocation.episode_forward_equivalents
                ),
                "float32_effect_payload_bytes": (
                    self.allocation.float32_effect_payload_bytes
                ),
            },
            "observed": {
                "compute_jobs": 1,
                "batch_forwards": self.calls,
                "episode_forward_equivalents": self.episodes,
                "float32_effect_payload_bytes": (
                    self.allocation.float32_effect_payload_bytes
                ),
            },
            "guard_after_job": self.guard.snapshot(),
            "runtime_boundary": {
                "counter_increment_preceded_each_callback": True,
                "failed_callbacks_remain_charged": True,
                "payload_reserved_before_first_callback": True,
                "job_receipt_does_not_finalize_role_guard": True,
            },
            "scientific_boundary": {
                "adapter_authorizes_capture": False,
                "adapter_authorizes_model_loading": False,
                "generated_tokens": 0,
                "real_actions": 0,
                "persistent_weight_mutations": 0,
                "sae_dictionary_training_authorized": False,
                "final_evaluation_authorized": False,
                "execution_authorized": False,
            },
        }
        return {**body, "job_receipt_id": content_sha256(body)}

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.runtime.close()


def reserve_guarded_capture_job(
    *,
    runtime: CaptureRuntime,
    guard: CaptureCampaignGuard,
    allocation: CaptureJobAllocation,
    resource_observer: ResourceObserver,
) -> GuardedCaptureRuntimeSession:
    if guard.finalized:
        raise ValueError("guarded capture job cannot use a finalized role guard")
    allocation.validate()
    before = guard.snapshot()
    requested = {
        "compute_jobs": 1,
        "batch_forwards": allocation.batch_forwards,
        "episode_forward_equivalents": allocation.episode_forward_equivalents,
        "float32_effect_payload_bytes": allocation.float32_effect_payload_bytes,
    }
    if any(requested[name] > before["remaining"][name] for name in requested):
        raise ValueError("guarded capture job allocation exceeds the remaining role budget")
    observed = resource_observer()
    if not isinstance(observed, tuple) or len(observed) != 2:
        raise ValueError("guarded capture resource observation is invalid")
    guard.record(
        compute_jobs=1,
        float32_effect_payload_bytes=allocation.float32_effect_payload_bytes,
        elapsed_seconds=observed[0],
        cost_usd=observed[1],
    )
    return GuardedCaptureRuntimeSession(
        runtime=runtime,
        guard=guard,
        allocation=allocation,
        resource_observer=resource_observer,
        guard_before_reservation=before,
    )
