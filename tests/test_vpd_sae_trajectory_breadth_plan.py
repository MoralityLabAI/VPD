import copy
import pytest
from experiments.authz_mechinterp.plan_sae_trajectory_breadth_v1 import build_plan, build_report, synthetic_metadata


def test_equal_budgets_different_trajectory_breadth():
    plan = build_plan(synthetic_metadata())
    n, b = plan["arms"]["Q2-N"], plan["arms"]["Q2-B"]
    assert n["rows_per_pass"] == b["rows_per_pass"] == 320
    assert n["phase_counts_per_pass"] == b["phase_counts_per_pass"] == {"prompt": 80, "generated": 240}
    assert set(n["family_counts_per_pass"].values()) == {32}
    assert n["trajectory_count"] == 10 and b["trajectory_count"] == 40
    assert n["roster_sha256"] != b["roster_sha256"]
    assert plan["shared_retained_rows_per_pass"] == 80
    assert plan["common_preprocessing_pool"]["rows_per_pass"] == 1280


def test_input_order_does_not_change_choices_or_rosters():
    metadata = synthetic_metadata()
    before = build_plan(metadata)
    metadata["records"].reverse()
    metadata["expected_training_families"].reverse()
    assert build_plan(metadata) == before


def test_quality_fields_and_holdout_positions_do_not_change_selection():
    metadata = synthetic_metadata()
    before = build_plan(metadata)
    for r in metadata["records"]:
        r["quality"] = -999
    metadata["records"][-1]["positions"] = [999999]
    assert build_plan(metadata) == before
    assert all("holdout" not in r["id"] for arm in before["arms"].values() for r in arm["roster"])


def test_seed_changes_only_narrow_choices():
    metadata = synthetic_metadata()
    before = build_plan(metadata)
    metadata["data_selection_seed"] += 1
    after = build_plan(metadata)
    assert before["narrow_trajectory_choices"] != after["narrow_trajectory_choices"]
    assert before["arms"]["Q2-B"] == after["arms"]["Q2-B"]
    assert before["common_preprocessing_pool"] == after["common_preprocessing_pool"]


@pytest.mark.parametrize("change", ["duplicate_id", "role_overlap", "missing_example", "extra_example", "wrong_position", "wrong_window", "short_trajectory", "bad_seed", "wrong_family", "bool_position"])
def test_invalid_layout_fails_without_replacement(change):
    metadata = synthetic_metadata()
    r = metadata["records"][0]
    if change == "duplicate_id": metadata["records"][1]["id"] = r["id"]
    elif change == "role_overlap": metadata["records"][-1]["family"] = r["family"]
    elif change == "missing_example": metadata["records"].pop(0)
    elif change == "extra_example":
        added = copy.deepcopy(r)
        added["id"] += "-extra"
        metadata["records"].append(added)
    elif change == "wrong_position": r["positions"][0] += 1
    elif change == "wrong_window": r["regions"]["early"] = r["regions"]["middle"]
    elif change == "short_trajectory": r["generated_tokens"] = 8
    elif change == "bad_seed": metadata["data_selection_seed"] = True
    elif change == "wrong_family": metadata["expected_training_families"][0] = "missing"
    elif change == "bool_position": r["positions"][0] = True
    with pytest.raises(ValueError):
        build_plan(metadata)


def test_synthetic_report_never_claims_fresh_data_or_authorization():
    report = build_report()
    assert report["source"] == {"kind": "synthetic_demo", "fresh_pack": False}
    assert not any(report["boundary"].values())


def test_smaller_family_inventory_cannot_silently_redefine_q2():
    metadata = synthetic_metadata()
    omitted = metadata["expected_training_families"].pop()
    metadata["records"] = [r for r in metadata["records"] if r["family"] != omitted]
    with pytest.raises(ValueError, match="exactly ten"):
        build_plan(metadata)
