import pytest
from experiments.authz_mechinterp.audit_sae_source_family_inventory_v1 import inventory


def row(id, family="a", split="train", group=None):
    return dict(id=id, family=family, split=split, group_id=group or id, prompt_sha256="fixture-hash", source_revision="fixture-revision")


def test_multiple_tests_in_one_group_do_not_meet_four_group_quota():
    result = inventory([row(str(i), group="same-task") for i in range(8)])
    assert result["eligible_family_labels_any_allowed_split"] == []
    assert result["families"][0]["splits"]["train"]["unique_declared_groups"] == 1


def test_historical_and_quarantined_rows_do_not_supply_families():
    records = [row(str(i), "history", "historical_eval") for i in range(4)]
    records += [row(str(i+4), "alias", "quarantine") for i in range(4)]
    assert inventory(records)["eligible_family_labels_any_allowed_split"] == []


def test_same_label_across_splits_is_not_multiple_disjoint_families():
    records = [row(f"{split}-{i}", split=split) for split in ("train", "dev", "eval") for i in range(4)]
    result = inventory(records)
    assert result["eligible_family_labels_any_allowed_split"] == ["a"]
    assert result["F1_necessary_family_count_check"]["available_label_upper_bound"] == 1


def test_four_distinct_groups_meet_label_count_only():
    result = inventory([row(str(i)) for i in range(4)])
    assert result["eligible_families_by_source_split"]["train"] == ["a"]
    assert result["F1_necessary_family_count_check"]["is_sufficient_feasibility_test"] is False


def test_duplicate_ids_fail():
    with pytest.raises(ValueError, match="duplicate"):
        inventory([row("a"), row("a")])


def test_no_silent_unknown_role():
    with pytest.raises(ValueError, match="unknown"):
        inventory([row("a", split="mystery")])


def test_empty_source_fails():
    with pytest.raises(ValueError, match="empty"):
        inventory([])
