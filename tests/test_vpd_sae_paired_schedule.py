import copy

import pytest

from experiments.authz_mechinterp.plan_sae_trajectory_breadth_v1 import build_plan, synthetic_metadata
from experiments.authz_mechinterp.plan_sae_paired_schedule_v1 import aligned_slots, draw_slots, paired_batch_indices, schedule_summary
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256


@pytest.fixture
def plan():
    return build_plan(synthetic_metadata())


def test_native_order_would_mismatch_factors_but_aligned_slots_do_not(plan):
    narrow, broad = (plan["arms"][a]["roster"] for a in ("Q2-N", "Q2-B"))
    assert sum(a["phase"] != b["phase"] for a, b in zip(narrow, broad)) == 120
    assert sum(a["region"] != b["region"] for a, b in zip(narrow, broad)) == 240
    mappings, factors = aligned_slots(plan)
    assert len(factors) == 320
    for a in mappings:
        assert sorted(mappings[a]) == list(range(320))
    for slot in range(320):
        a, b = narrow[mappings["Q2-N"][slot]], broad[mappings["Q2-B"][slot]]
        assert (a["family"], a["region"], a["phase"]) == (b["family"], b["region"], b["phase"])


def test_paired_passes_use_identical_indices(plan):
    mappings, _ = aligned_slots(plan)
    indices = paired_batch_indices(mappings, [0, 7, 8, 31, 319, 0])
    for arm in indices:
        assert indices[arm]["pass_0"] == indices[arm]["pass_1"]
        assert indices[arm]["pass_0"][0] == indices[arm]["pass_0"][-1]


def test_full_schedule_exact_budget_and_factor_matching(plan):
    slots = draw_slots()
    assert len(slots) == 1000 and all(len(batch) == 64 for batch in slots)
    mappings, _ = aligned_slots(plan)
    summary = schedule_summary(plan, mappings, slots)
    assert sum(summary["matched_phase_draws_per_arm_per_pass"].values()) == 64000
    assert sum(summary["matched_family_draws_per_arm_per_pass"].values()) == 64000
    assert summary["all_draws_factor_matched"]
    assert summary["all_draws_pass_index_matched"]
    assert all(r["unique_rows_drawn_per_pass"] == 320 for r in summary["arms"].values())


def test_seed_reproduces_and_different_seed_changes_schedule():
    assert draw_slots(updates=3) == draw_slots(updates=3)
    assert draw_slots(seed=1, updates=3) != draw_slots(seed=2, updates=3)


def test_roster_reordering_preserves_draw_identity(plan):
    other = copy.deepcopy(plan)
    for arm in other["arms"].values():
        arm["roster"].reverse()
        arm["roster_sha256"] = content_sha256(arm["roster"])
    original_maps, _ = aligned_slots(plan)
    other_maps, _ = aligned_slots(other)
    for arm in original_maps:
        assert [plan["arms"][arm]["roster"][i] for i in original_maps[arm]] == [other["arms"][arm]["roster"][i] for i in other_maps[arm]]


def test_hash_mismatch_rejected(plan):
    plan["arms"]["Q2-N"]["roster"][0]["position"] += 100
    with pytest.raises(ValueError, match="hash"):
        aligned_slots(plan)


def test_resigned_duplicate_and_factor_mismatch_rejected(plan):
    other = copy.deepcopy(plan)
    arm = other["arms"]["Q2-N"]
    arm["roster"][0] = arm["roster"][1]
    arm["roster_sha256"] = content_sha256(arm["roster"])
    with pytest.raises(ValueError, match="duplicate"):
        aligned_slots(other)
    arm = plan["arms"]["Q2-N"]
    arm["roster"][0]["phase"] = "generated"
    arm["roster_sha256"] = content_sha256(arm["roster"])
    with pytest.raises(ValueError, match="phase"):
        aligned_slots(plan)


@pytest.mark.parametrize("kwargs", [{"seed": True}, {"updates": 0}, {"updates": 1001}, {"batch_rows": 65}, {"seed": -1}])
def test_unbounded_or_invalid_schedule_rejected(kwargs):
    with pytest.raises(ValueError):
        draw_slots(**kwargs)


@pytest.mark.parametrize("batch", [[], [320], [-1], [True], [1.0]])
def test_invalid_draw_indices_rejected(plan, batch):
    mappings, _ = aligned_slots(plan)
    with pytest.raises(ValueError):
        paired_batch_indices(mappings, batch)


def test_non_bijective_map_rejected(plan):
    mappings, _ = aligned_slots(plan)
    mappings["Q2-N"][0] = mappings["Q2-N"][1]
    with pytest.raises(ValueError, match="bijection"):
        paired_batch_indices(mappings, [0])
