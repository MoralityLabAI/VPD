import copy

import numpy as np
import pytest

from experiments.authz_mechinterp.bind_sae_paired_activation_rows_v1 import (
    SITES, PASSES, array_digest, bind_rows, iter_paired_batches, synthetic_fixture,
)
from experiments.authz_mechinterp.plan_sae_paired_schedule_v1 import aligned_slots


@pytest.fixture
def fixture():
    return synthetic_fixture()


def test_bound_rosters_preserve_identity_and_exclude_holdout(fixture):
    bound = bind_rows(*fixture, expected_width=5)
    assert bound["complete_pool_rows_per_pass"] == 1280
    for si, site in enumerate(SITES):
        for arm, views in bound["views"][site].items():
            roster = bound["plan"]["arms"][arm]["roster"]
            for pi, p in enumerate(PASSES):
                assert views[p].shape == (320, 5)
                assert not views[p].flags.writeable
                assert np.all(views[p][:, 2] == si) and np.all(views[p][:, 3] == pi)
                assert views[p][:, 4].tolist() == [r["position"] for r in roster]


def test_complete_pool_bias_not_narrow_arm_bias(fixture):
    bound = bind_rows(*fixture, expected_width=5)
    for site in SITES:
        # Complete fixture has equal representation of all four example indices.
        assert bound["common_pre_bias"][site][1] == 1.5
        assert bound["common_pre_bias"][site][3] == 0.5
        assert bound["common_pre_bias"][site].dtype == np.float32
        assert not bound["common_pre_bias"][site].flags.writeable
    assert bound["views"][SITES[0]]["Q2-N"]["pass_0"][:, 1].mean() != 1.5


def test_legitimate_storage_reordering_keeps_bound_arrays_identical(fixture):
    expected = bind_rows(*fixture, expected_width=5)
    metadata, manifest, arrays = copy.deepcopy(fixture)
    example = next(iter(arrays))
    for site in SITES:
        for p in PASSES:
            arrays[example][site][p] = arrays[example][site][p][::-1]
            entry = manifest["records"][example]["sites"][site][p]
            entry["positions"].reverse()
            entry["array_content_sha256"] = array_digest(arrays[example][site][p])
    observed = bind_rows(metadata, manifest, arrays, expected_width=5)
    for site in SITES:
        for arm in expected["views"][site]:
            for p in PASSES:
                np.testing.assert_array_equal(expected["views"][site][arm][p], observed["views"][site][arm][p])


@pytest.mark.parametrize("mutation", ["swap_passes", "shuffle_rows", "swap_sites", "change_value"])
def test_array_changes_rejected_against_frozen_per_key_hashes(fixture, mutation):
    metadata, manifest, arrays = fixture
    example = next(iter(arrays))
    if mutation == "swap_passes":
        arrays[example][SITES[0]]["pass_0"], arrays[example][SITES[0]]["pass_1"] = arrays[example][SITES[0]]["pass_1"], arrays[example][SITES[0]]["pass_0"]
    elif mutation == "swap_sites":
        arrays[example][SITES[0]], arrays[example][SITES[1]] = arrays[example][SITES[1]], arrays[example][SITES[0]]
    elif mutation == "shuffle_rows":
        arrays[example][SITES[0]]["pass_0"] = arrays[example][SITES[0]]["pass_0"][::-1]
    else:
        arrays[example][SITES[0]]["pass_0"][0, 0] += 1
    with pytest.raises(ValueError, match="array hash"):
        bind_rows(metadata, manifest, arrays, expected_width=5)


@pytest.mark.parametrize("mutation", ["duplicate_position", "wrong_family", "missing_example", "extra_eval", "wrong_width", "nonfinite"])
def test_invalid_capture_contract_rejected(fixture, mutation):
    metadata, manifest, arrays = fixture
    example = next(iter(arrays))
    if mutation == "duplicate_position":
        positions = manifest["records"][example]["sites"][SITES[0]]["pass_0"]["positions"]
        positions[0] = positions[1]
    elif mutation == "wrong_family":
        manifest["records"][example]["family"] = "wrong"
    elif mutation == "missing_example":
        arrays.pop(example)
    elif mutation == "extra_eval":
        arrays["synthetic-holdout/example-0"] = arrays[example]
    elif mutation == "wrong_width":
        arrays[example][SITES[0]]["pass_0"] = arrays[example][SITES[0]]["pass_0"][:, :4]
    else:
        arrays[example][SITES[0]]["pass_0"][0, 0] = np.nan
    with pytest.raises(ValueError):
        bind_rows(metadata, manifest, arrays, expected_width=5)


def test_batch_iterator_preserves_both_passes_and_returns_copies(fixture):
    bound = bind_rows(*fixture, expected_width=5)
    mappings, _ = aligned_slots(bound["plan"])
    step, batch = next(iter_paired_batches(bound, mappings, [[0, 9, 319, 0]]))
    assert step == 0
    for site in SITES:
        for arm in batch[site]:
            a, b = batch[site][arm]["pass_0"], batch[site][arm]["pass_1"]
            np.testing.assert_array_equal(a[:, [0, 1, 2, 4]], b[:, [0, 1, 2, 4]])
            assert np.all(a[:, 3] == 0) and np.all(b[:, 3] == 1)
    batch[SITES[0]]["Q2-N"]["pass_0"][0] = -99
    assert not np.any(bound["views"][SITES[0]]["Q2-N"]["pass_0"] == -99)


def test_explicit_width_and_dtype_are_not_silently_coerced(fixture):
    with pytest.raises(ValueError):
        bind_rows(*fixture, expected_width=3072)
    with pytest.raises(ValueError):
        array_digest(np.ones((32, 5), dtype=np.int32))


def test_changed_bijective_batch_map_is_rejected(fixture):
    bound = bind_rows(*fixture, expected_width=5)
    mappings, _ = aligned_slots(bound["plan"])
    mappings["Q2-N"][0], mappings["Q2-N"][9] = mappings["Q2-N"][9], mappings["Q2-N"][0]
    with pytest.raises(ValueError, match="factor-aligned"):
        next(iter_paired_batches(bound, mappings, [[0]]))
