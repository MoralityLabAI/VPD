from __future__ import annotations

from copy import deepcopy

from experiments.authz_mechinterp.generate_independent_authority_pack_v2 import (
    HOST_VISIBLE_CONDITIONS,
    generate_episodes,
)
from experiments.authz_mechinterp.validate_independent_authority_pack_v2 import (
    conflicting_prompt_labels,
    extract_status,
    redact_status,
)


def generated_rows() -> list[dict]:
    return list(generate_episodes(seed=2713, pairs_per_family=1))


def test_all_expected_decisions_are_model_visibly_identifiable() -> None:
    rows = generated_rows()
    assert len(rows) == 40
    assert conflicting_prompt_labels(rows) == {}
    for row in rows:
        status = extract_status(row)
        assert (status is not None) == (row["condition"] in HOST_VISIBLE_CONDITIONS)
        assert row["labels"]["observably_identifiable_pre_gate"] is True


def test_revocation_and_replay_change_only_trusted_visible_status() -> None:
    rows = generated_rows()
    pairs: dict[str, dict[str, dict]] = {}
    for row in rows:
        pairs.setdefault(row["pair_id"], {})[row["condition"]] = row
    for by_condition in pairs.values():
        normalized = {
            condition: redact_status(by_condition[condition]["messages"])
            for condition in (
                "valid_visible",
                "revoked_host_visible",
                "replayed_host_visible",
            )
        }
        assert normalized["valid_visible"] == normalized["revoked_host_visible"]
        assert normalized["valid_visible"] == normalized["replayed_host_visible"]
        assert extract_status(by_condition["valid_visible"])["revoked"] is False
        assert extract_status(by_condition["revoked_host_visible"])["revoked"] is True
        assert extract_status(by_condition["valid_visible"])["nonce_state"] == "fresh"
        assert extract_status(by_condition["replayed_host_visible"])["nonce_state"] == "consumed"


def test_collision_guard_rejects_conflicting_labels_on_same_prompt() -> None:
    rows = generated_rows()
    pair_id = rows[0]["pair_id"]
    by_condition = {
        row["condition"]: row for row in rows if row["pair_id"] == pair_id
    }
    tampered = deepcopy(rows)
    tampered_replay = next(
        row
        for row in tampered
        if row["pair_id"] == pair_id and row["condition"] == "replayed_host_visible"
    )
    tampered_replay["messages"] = deepcopy(by_condition["valid_visible"]["messages"])
    conflicts = conflicting_prompt_labels(tampered)
    assert len(conflicts) == 1
    assert next(iter(conflicts.values())) == ["PROPOSE", "REQUEST"]
