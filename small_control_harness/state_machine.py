from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RunPhase(str, Enum):
    DRAFT = "draft"
    SIMULATED = "simulated"
    AWAITING_AUTHORIZATION = "awaiting_authorization"
    AUTHORIZED = "authorized"
    PROPOSED = "proposed"
    GATE_DENIED = "gate_denied"
    GATE_ALLOWED = "gate_allowed"
    EXECUTED = "executed"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"


_ALLOWED_TRANSITIONS: dict[RunPhase, frozenset[RunPhase]] = {
    RunPhase.DRAFT: frozenset({RunPhase.SIMULATED, RunPhase.FAILED, RunPhase.EXPIRED}),
    RunPhase.SIMULATED: frozenset(
        {RunPhase.AWAITING_AUTHORIZATION, RunPhase.AUTHORIZED, RunPhase.FAILED, RunPhase.EXPIRED}
    ),
    RunPhase.AWAITING_AUTHORIZATION: frozenset(
        {RunPhase.AUTHORIZED, RunPhase.FAILED, RunPhase.EXPIRED}
    ),
    RunPhase.AUTHORIZED: frozenset({RunPhase.PROPOSED, RunPhase.FAILED, RunPhase.EXPIRED}),
    RunPhase.PROPOSED: frozenset(
        {RunPhase.GATE_ALLOWED, RunPhase.GATE_DENIED, RunPhase.FAILED, RunPhase.EXPIRED}
    ),
    RunPhase.GATE_DENIED: frozenset(
        {RunPhase.AWAITING_AUTHORIZATION, RunPhase.PROPOSED, RunPhase.FAILED, RunPhase.EXPIRED}
    ),
    RunPhase.GATE_ALLOWED: frozenset({RunPhase.EXECUTED, RunPhase.FAILED, RunPhase.EXPIRED}),
    RunPhase.EXECUTED: frozenset({RunPhase.VERIFIED, RunPhase.FAILED}),
    RunPhase.VERIFIED: frozenset(),
    RunPhase.FAILED: frozenset(),
    RunPhase.EXPIRED: frozenset(),
}


@dataclass(frozen=True)
class TransitionRecord:
    from_phase: RunPhase
    to_phase: RunPhase
    evidence_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunLifecycle:
    phase: RunPhase = RunPhase.DRAFT
    history: list[TransitionRecord] = field(default_factory=list)

    def transition(
        self,
        to_phase: RunPhase,
        *,
        evidence_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> TransitionRecord:
        if not evidence_id.strip():
            raise ValueError("every lifecycle transition requires an evidence identifier")
        if to_phase not in _ALLOWED_TRANSITIONS[self.phase]:
            raise ValueError(f"invalid run transition: {self.phase.value} -> {to_phase.value}")
        record = TransitionRecord(
            from_phase=self.phase,
            to_phase=to_phase,
            evidence_id=evidence_id,
            metadata=dict(metadata or {}),
        )
        self.history.append(record)
        self.phase = to_phase
        return record
