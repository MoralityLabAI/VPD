from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from experiments.authz_mechinterp.run_canonical_vpd_capture_job_v1 import CaptureRuntime
from experiments.authz_mechinterp.vpd_cluster_capture_campaign_guard_v1 import (
    CaptureCampaignGuard,
)
from experiments.authz_mechinterp.vpd_guarded_capture_runtime_adapter_v1 import (
    CaptureJobAllocation,
    GuardedCaptureRuntimeSession,
    ResourceObserver,
)


@dataclass
class GuardedCaptureJobReservation:
    guard: CaptureCampaignGuard
    allocation: CaptureJobAllocation
    resource_observer: ResourceObserver
    guard_before_reservation: dict[str, Any]
    runtime_attached: bool = False

    def attach_runtime(self, runtime: CaptureRuntime) -> GuardedCaptureRuntimeSession:
        if self.runtime_attached:
            raise ValueError("guarded capture reservation already attached a runtime")
        if self.guard.finalized:
            raise ValueError("guarded capture reservation cannot attach after finalization")
        self.runtime_attached = True
        return GuardedCaptureRuntimeSession(
            runtime=runtime,
            guard=self.guard,
            allocation=self.allocation,
            resource_observer=self.resource_observer,
            guard_before_reservation=self.guard_before_reservation,
        )


def reserve_guarded_capture_job_before_runtime_load(
    *,
    guard: CaptureCampaignGuard,
    allocation: CaptureJobAllocation,
    resource_observer: ResourceObserver,
) -> GuardedCaptureJobReservation:
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
    return GuardedCaptureJobReservation(
        guard=guard,
        allocation=allocation,
        resource_observer=resource_observer,
        guard_before_reservation=before,
    )
