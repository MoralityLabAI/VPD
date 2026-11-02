from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, write_json


SCHEMA = "sch.vpd-cluster-capture-campaign-guard-receipt.v1"
EXPECTED_STAGE_ORDER = [
    "discovery_capture",
    "discovery_gate",
    "calibration_capture",
    "calibration_gate",
    "cluster_validation_capture",
    "partition_identity_admission",
    "sae_training_routing_capture",
    "router_and_manifest_planner_v2",
]
ROLE_PREREQUISITES = {
    "discovery": [],
    "calibration": ["discovery_capture_complete", "discovery_gate_pass"],
    "cluster_validation": [
        "discovery_capture_complete",
        "discovery_gate_pass",
        "calibration_capture_complete",
        "calibration_gate_pass",
    ],
    "sae_training": [
        "discovery_capture_complete",
        "discovery_gate_pass",
        "calibration_capture_complete",
        "calibration_gate_pass",
        "cluster_validation_capture_complete",
        "partition_identity_admission_pass",
    ],
}
COUNTER_FIELDS = (
    "compute_jobs",
    "batch_forwards",
    "episode_forward_equivalents",
    "float32_effect_payload_bytes",
)


def _verify_preflight(preflight: dict[str, Any]) -> str:
    if preflight.get("schema") != "sch.vpd-cluster-to-sae-empirical-preflight.v2":
        raise ValueError("campaign guard preflight schema differs")
    identifier = str(preflight.get("preflight_id", ""))
    body = {key: value for key, value in preflight.items() if key != "preflight_id"}
    if identifier != content_sha256(body):
        raise ValueError("campaign guard preflight address differs")
    if preflight.get("missing_required_bindings"):
        raise ValueError("campaign guard requires an identity-complete preflight")
    if preflight.get("stage_order") != EXPECTED_STAGE_ORDER:
        raise ValueError("campaign guard stage order differs")
    if set(preflight.get("capture_roles", {})) != set(ROLE_PREREQUISITES):
        raise ValueError("campaign guard capture roles differ")
    return identifier


def _role_ceilings(preflight: dict[str, Any], role: str) -> dict[str, int]:
    role_resource = preflight["capture_roles"][role]["resource_ceiling"]
    ceilings = {
        "compute_jobs": int(preflight["resource_ceiling"]["packets_per_role"]),
        "batch_forwards": int(role_resource["batch_forwards_at_planned_batch_size"]),
        "episode_forward_equivalents": int(
            role_resource["episode_forward_equivalents"]
        ),
        "float32_effect_payload_bytes": int(
            role_resource["estimated_float32_effect_payload_bytes"]
        ),
    }
    if any(value <= 0 for value in ceilings.values()):
        raise ValueError("campaign guard role ceiling is invalid")
    return ceilings


@dataclass
class CaptureCampaignGuard:
    preflight_id: str
    role: str
    split_role_id: str
    prerequisites: list[str]
    ceilings: dict[str, int]
    global_ceilings: dict[str, int]
    wall_time_ceiling_seconds: int
    cost_ceiling_usd: float
    simulation_only: bool
    runtime_admission_real: bool
    output_reserved: bool
    counters: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in COUNTER_FIELDS}
    )
    elapsed_seconds: float = 0.0
    cost_usd: float = 0.0
    finalized: bool = False

    def record(
        self,
        *,
        compute_jobs: int = 0,
        batch_forwards: int = 0,
        episode_forward_equivalents: int = 0,
        float32_effect_payload_bytes: int = 0,
        elapsed_seconds: float,
        cost_usd: float,
        generated_tokens: int = 0,
        real_actions: int = 0,
        persistent_weight_mutations: int = 0,
    ) -> dict[str, Any]:
        if self.finalized:
            raise ValueError("campaign guard is already finalized")
        deltas = {
            "compute_jobs": compute_jobs,
            "batch_forwards": batch_forwards,
            "episode_forward_equivalents": episode_forward_equivalents,
            "float32_effect_payload_bytes": float32_effect_payload_bytes,
        }
        if any(type(value) is not int or value < 0 for value in deltas.values()):
            raise ValueError("campaign guard counter delta is invalid")
        if type(generated_tokens) is not int or generated_tokens != 0:
            raise ValueError("campaign guard forbids generated tokens")
        if type(real_actions) is not int or real_actions != 0:
            raise ValueError("campaign guard forbids real actions")
        if type(persistent_weight_mutations) is not int or persistent_weight_mutations != 0:
            raise ValueError("campaign guard forbids persistent weight mutations")
        if (
            not isinstance(elapsed_seconds, (int, float))
            or isinstance(elapsed_seconds, bool)
            or not math.isfinite(elapsed_seconds)
            or elapsed_seconds < self.elapsed_seconds
            or elapsed_seconds > self.wall_time_ceiling_seconds
        ):
            raise ValueError("campaign guard wall-time observation is invalid")
        if (
            not isinstance(cost_usd, (int, float))
            or isinstance(cost_usd, bool)
            or not math.isfinite(cost_usd)
            or cost_usd < self.cost_usd
            or cost_usd > self.cost_ceiling_usd
        ):
            raise ValueError("campaign guard cost observation is invalid")
        next_counters = {
            name: self.counters[name] + deltas[name] for name in COUNTER_FIELDS
        }
        for name, observed in next_counters.items():
            if observed > self.ceilings[name] or observed > self.global_ceilings[name]:
                raise ValueError(f"campaign guard exceeded ceiling: {name}")
        self.counters = next_counters
        self.elapsed_seconds = float(elapsed_seconds)
        self.cost_usd = float(cost_usd)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "counters": dict(self.counters),
            "elapsed_seconds": self.elapsed_seconds,
            "cost_usd": self.cost_usd,
            "remaining": {
                name: self.ceilings[name] - self.counters[name]
                for name in COUNTER_FIELDS
            },
            "generated_tokens": 0,
            "real_actions": 0,
            "persistent_weight_mutations": 0,
            "finalized": self.finalized,
            "execution_authorized": False,
        }

    def finalize(self) -> dict[str, Any]:
        if self.finalized:
            raise ValueError("campaign guard is already finalized")
        if self.counters != self.ceilings:
            raise ValueError("campaign guard cannot finalize incomplete role counters")
        self.finalized = True
        body = {
            "schema": SCHEMA,
            "status": (
                "complete_simulated_meter_only"
                if self.simulation_only
                else "complete_meter_receipt_capture_authority_separate"
            ),
            "preflight_id": self.preflight_id,
            "role": self.role,
            "split_role_id": self.split_role_id,
            "prerequisites": self.prerequisites,
            "ceilings": self.ceilings,
            "observed": {
                **self.counters,
                "elapsed_seconds": self.elapsed_seconds,
                "cost_usd": self.cost_usd,
                "generated_tokens": 0,
                "real_actions": 0,
                "persistent_weight_mutations": 0,
            },
            "runtime_boundary": {
                "simulation_only": self.simulation_only,
                "runtime_admission_real": self.runtime_admission_real,
                "output_reserved": self.output_reserved,
                "counter_increment_must_precede_each_compute_call": True,
                "ceiling_violation_must_stop_before_compute": True,
                "reservation_does_not_start_capture": True,
            },
            "scientific_boundary": {
                "model_framework_imported": False,
                "model_loaded_by_guard": False,
                "model_inference_performed_by_guard": False,
                "generated_tokens": 0,
                "real_actions": 0,
                "persistent_weight_mutations": 0,
                "sae_dictionary_training_authorized": False,
                "final_evaluation_authorized": False,
                "execution_authorized": False,
            },
        }
        return {**body, "guard_receipt_id": content_sha256(body)}


def build_guard(
    *,
    preflight: dict[str, Any],
    role: str,
    completed_events: list[str],
    simulation_only: bool,
    runtime_admission_real: bool = False,
    output_reserved: bool = False,
) -> CaptureCampaignGuard:
    preflight_id = _verify_preflight(preflight)
    if role not in ROLE_PREREQUISITES:
        raise ValueError("campaign guard role is unsupported")
    required = ROLE_PREREQUISITES[role]
    if completed_events != required:
        raise ValueError("campaign guard prerequisites differ or skip a stage")
    if not simulation_only and (not runtime_admission_real or not output_reserved):
        raise ValueError("campaign guard production start lacks admission or reservation")
    bindings = preflight["immutable_bindings"]
    global_resource = preflight["resource_ceiling"]
    return CaptureCampaignGuard(
        preflight_id=preflight_id,
        role=role,
        split_role_id=preflight["capture_roles"][role]["split_role_id"],
        prerequisites=list(required),
        ceilings=_role_ceilings(preflight, role),
        global_ceilings={
            "compute_jobs": int(global_resource["maximum_compute_jobs"]),
            "batch_forwards": int(
                global_resource["maximum_batch_forwards_at_batch_size_4"]
            ),
            "episode_forward_equivalents": int(
                global_resource["maximum_episode_forward_equivalents"]
            ),
            "float32_effect_payload_bytes": int(
                global_resource["estimated_float32_effect_payload_bytes"]
            ),
        },
        wall_time_ceiling_seconds=int(bindings["wall_time_ceiling_seconds"]),
        cost_ceiling_usd=float(bindings["cost_ceiling_usd"]),
        simulation_only=simulation_only,
        runtime_admission_real=runtime_admission_real,
        output_reserved=output_reserved,
    )


def verify_guard_receipt(path: Path, *, preflight: dict[str, Any]) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in observed.items() if key != "guard_receipt_id"}
    identifier = str(observed.get("guard_receipt_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("campaign guard receipt address differs")
    preflight_id = _verify_preflight(preflight)
    role = str(observed.get("role", ""))
    if role not in ROLE_PREREQUISITES or observed.get("preflight_id") != preflight_id:
        raise ValueError("campaign guard receipt source differs")
    if observed.get("prerequisites") != ROLE_PREREQUISITES[role]:
        raise ValueError("campaign guard receipt prerequisites differ")
    if observed.get("ceilings") != _role_ceilings(preflight, role):
        raise ValueError("campaign guard receipt ceilings differ")
    if any(observed["observed"].get(name) != observed["ceilings"][name] for name in COUNTER_FIELDS):
        raise ValueError("campaign guard receipt counters differ")
    boundary = observed.get("scientific_boundary", {})
    if any(
        boundary.get(field) not in (False, 0)
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
    ):
        raise ValueError("campaign guard receipt scientific boundary differs")
    return {
        "schema": "sch.vpd-cluster-capture-campaign-guard-verification.v1",
        "status": "pass",
        "guard_receipt_id": identifier,
        "role": role,
        "simulation_only": observed["runtime_boundary"]["simulation_only"],
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a VPD capture campaign guard receipt.")
    parser.add_argument("--verify", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    result = verify_guard_receipt(args.verify, preflight=preflight)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
