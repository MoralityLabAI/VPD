import copy
import pytest

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256
from experiments.authz_mechinterp.evaluate_sae_family_quality_v1 import evaluate_quality, evaluate_receipt


def row(family, values, examples=1):
    return {"family": family, "per_pass_ev": values, "examples": examples}


def test_pooled_pass_cannot_hide_family_failure():
    result = evaluate_quality([.8, .8], [row("a", [.9, .9], 1000), row("b", [.19, .8])], ["a", "b"])
    assert result["pooled_status"] == "pass"
    assert not result["quality_prerequisite_pass"]
    assert result["failed_families"] == ["b"]


def test_exact_boundary_passes_quality_but_not_clustering():
    result = evaluate_quality([.2, .2], [row("a", [.2, .2])], ["a"])
    assert result["quality_prerequisite_pass"]
    assert not result["clustering_admission"]


def test_pooled_failure_cannot_be_overridden():
    assert not evaluate_quality([-.1, .5], [row("a", [.9, .9])], ["a"])["quality_prerequisite_pass"]


@pytest.mark.parametrize("rows", [[], [row("a", [None, .8])]])
def test_missing_or_undefined_family_is_not_a_pass(rows):
    result = evaluate_quality([.8, .8], rows, ["a"])
    assert result["unevaluable_families"] == ["a"]
    assert not result["quality_prerequisite_pass"]


@pytest.mark.parametrize("bad", [[float("nan"), .5], [float("inf"), .5], [1.01, .5], [True, .5], [.5], ["0.5", .5]])
def test_malformed_ev_rejected(bad):
    with pytest.raises(ValueError):
        evaluate_quality(bad, [row("a", [.5, .5])], ["a"])


@pytest.mark.parametrize("families", [[], ["a", "a"], [""]])
def test_declared_scope_required(families):
    with pytest.raises(ValueError):
        evaluate_quality([.5, .5], [], families)


@pytest.mark.parametrize("rows", [[row("b", [.5, .5])], [row("a", [.5, .5])] * 2, [row("a", [.5, .5], 0)]])
def test_unexpected_duplicate_or_empty_family_rejected(rows):
    with pytest.raises(ValueError):
        evaluate_quality([.5, .5], rows, ["a"])


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -.1, 1.1, True])
def test_bad_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        evaluate_quality([.5, .5], [row("a", [.5, .5])], ["a"], threshold)


def receipt():
    body = {"schema": "sch.vpd-q4-sae-family-contributions.v1", "archives": [
        {"path": "fixture", "evaluation_families": ["a"], "sites": [
            {"site": "fixture.site", "pooled_ev": [.5, .5], "family_reconstruction_quality": [row("a", [.5, .5])]}]}]}
    return {**body, "report_id": content_sha256(body)}


def test_receipt_tampering_rejected():
    source = receipt()
    source["archives"][0]["sites"][0]["pooled_ev"] = [.8, .8]
    with pytest.raises(ValueError, match="hash"):
        evaluate_receipt(source, ["a"])


def test_summary_and_claim_scope():
    result = evaluate_receipt(receipt(), ["a"])
    assert result["summary"] == {"fits": 1, "pooled_passes": 1, "all_family_quality_passes": 1, "family_checks": 1, "family_passes": 1}
    assert result["source_arrays_verified_by_this_command"] is False


def test_posthoc_narrowing_receipt_family_scope_rejected():
    with pytest.raises(ValueError, match="scope"):
        evaluate_receipt(receipt(), ["a", "b"])


def test_duplicate_fit_rejected_even_with_valid_hash():
    source = receipt()
    source["archives"].append(copy.deepcopy(source["archives"][0]))
    source["report_id"] = content_sha256({k: v for k, v in source.items() if k != "report_id"})
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_receipt(source, ["a"])
