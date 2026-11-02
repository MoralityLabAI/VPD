from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


_SCHEMA_BY_VERSION = {
    "sch.run-contract.v1": "run_contract.schema.json",
    "sch.authorization-grant.v1": "authorization_grant.schema.json",
    "sch.action-proposal.v1": "action_proposal.schema.json",
    "sch.decision-receipt.v1": "decision_receipt.schema.json",
}


def schema_root() -> Path:
    return Path(__file__).resolve().parents[1] / "schemas"


def validate_document(document: Mapping[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
        raise RuntimeError("jsonschema is required for boundary validation") from exc

    version = str(document.get("version", ""))
    schema_name = _SCHEMA_BY_VERSION.get(version)
    if schema_name is None:
        raise ValueError(f"unsupported document version: {version!r}")
    schema = json.loads((schema_root() / schema_name).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(
        dict(document)
    )
