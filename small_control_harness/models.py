from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ScopedAction:
    tool: str
    verb: str
    constraints: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScopedAction":
        return cls(
            tool=str(data["tool"]),
            verb=str(data["verb"]),
            constraints=dict(data.get("constraints", {})),
        )


@dataclass(frozen=True)
class RunContract:
    version: str
    contract_id: str
    subject: str
    objective: Mapping[str, Any]
    action_scope: tuple[ScopedAction, ...]
    resources: Mapping[str, Any]
    invariants: tuple[str, ...]
    evidence: tuple[Mapping[str, Any], ...]
    created_at: str
    expires_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunContract":
        if data.get("version") != "sch.run-contract.v1":
            raise ValueError("unsupported run contract version")
        return cls(
            version=str(data["version"]),
            contract_id=str(data["contract_id"]),
            subject=str(data["subject"]),
            objective=dict(data["objective"]),
            action_scope=tuple(ScopedAction.from_dict(x) for x in data["action_scope"]),
            resources=dict(data["resources"]),
            invariants=tuple(str(x) for x in data["invariants"]),
            evidence=tuple(dict(x) for x in data["evidence"]),
            created_at=str(data["created_at"]),
            expires_at=str(data["expires_at"]),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["action_scope"] = [asdict(x) for x in self.action_scope]
        value["invariants"] = list(self.invariants)
        value["evidence"] = [dict(x) for x in self.evidence]
        return value


@dataclass(frozen=True)
class AuthorizationGrant:
    version: str
    grant_id: str
    contract_id: str
    issuer: str
    subject: str
    audience: str
    allowed_actions: tuple[ScopedAction, ...]
    limits: Mapping[str, Any]
    issued_at: str
    expires_at: str
    nonce: str
    one_shot: bool
    signature: Mapping[str, str]
    parent_grant_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuthorizationGrant":
        if data.get("version") != "sch.authorization-grant.v1":
            raise ValueError("unsupported authorization grant version")
        return cls(
            version=str(data["version"]),
            grant_id=str(data["grant_id"]),
            contract_id=str(data["contract_id"]),
            issuer=str(data["issuer"]),
            subject=str(data["subject"]),
            audience=str(data["audience"]),
            allowed_actions=tuple(ScopedAction.from_dict(x) for x in data["allowed_actions"]),
            limits=dict(data["limits"]),
            issued_at=str(data["issued_at"]),
            expires_at=str(data["expires_at"]),
            nonce=str(data["nonce"]),
            one_shot=bool(data["one_shot"]),
            signature=dict(data["signature"]),
            parent_grant_id=(
                str(data["parent_grant_id"])
                if data.get("parent_grant_id") is not None
                else None
            ),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["allowed_actions"] = [asdict(x) for x in self.allowed_actions]
        return value


@dataclass(frozen=True)
class ActionProposal:
    version: str
    proposal_id: str
    contract_id: str
    grant_id: str | None
    tool: str
    verb: str
    args: Mapping[str, Any]
    estimated_effects: Mapping[str, Any]
    created_at: str
    justification: str = ""
    model_context_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionProposal":
        if data.get("version") != "sch.action-proposal.v1":
            raise ValueError("unsupported action proposal version")
        return cls(
            version=str(data["version"]),
            proposal_id=str(data["proposal_id"]),
            contract_id=str(data["contract_id"]),
            grant_id=str(data["grant_id"]) if data.get("grant_id") is not None else None,
            tool=str(data["tool"]),
            verb=str(data["verb"]),
            args=dict(data["args"]),
            estimated_effects=dict(data["estimated_effects"]),
            created_at=str(data["created_at"]),
            justification=str(data.get("justification", "")),
            model_context_hash=(
                str(data["model_context_hash"])
                if data.get("model_context_hash") is not None
                else None
            ),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
