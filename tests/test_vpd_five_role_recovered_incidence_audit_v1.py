from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.benchmark_vpd_five_role_recovered_incidence_audit_v1 import (
    assert_role_use,
    build_report,
    verify_audit,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/vpd_five_role_recovered_incidence_audit_v1/report.json"


def test_five_role_recovered_incidence_audit_reproduces() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert observed == build_report(ROOT)
    assert observed["audit_id"] == (
        "c14b80e53ea6fe39e171fbd201ab0a87b86dc39737c091fd83616bb813d2cefc"
    )
    assert observed["summary"] == {"controls": 15, "passed": 15}
    assert observed["aggregate"] == {
        "episode_id_set_sha256": (
            "b72d01345b939b68f4b2c1025e945fcfb2bb3305c07f81b3b9faa2e5828d6676"
        ),
        "pair_id_set_sha256": (
            "7924d0068da5d2c4d022bff32af1e2d024754667e04065cbd52d7715fa7251a9"
        ),
        "pairs": 2288,
        "rows": 25168,
        "template_group_set_sha256": (
            "846f6578e75dc17d6e33a80710e2112163bd7d4a0c62bbf7f39c4a982fcfe987"
        ),
        "template_groups": 16,
    }
    assert len(set(observed["role_ids"].values())) == 5
    assert all(
        overlap == {"episodes": 0, "pairs": 0, "template_groups": 0}
        for overlap in observed["pairwise_overlap"].values()
    )


def test_five_role_verifier_preserves_no_execution_boundaries() -> None:
    verification = verify_audit(REPORT, project_root=ROOT)
    assert verification["status"] == "pass"
    assert verification["roles"] == 5
    assert verification["rows"] == 25168
    assert verification["pairs"] == 2288
    assert verification["execution_authorized"] is False
    assert verification["sae_training_authorized"] is False


@pytest.mark.parametrize("use", ["selection", "routing", "dictionary_training"])
def test_final_evaluation_cannot_leak_into_upstream_uses(use: str) -> None:
    with pytest.raises(ValueError, match="final_evaluation is forbidden"):
        assert_role_use("final_evaluation", use)


def test_role_use_policy_allows_only_intended_roles() -> None:
    assert_role_use("discovery", "selection")
    assert_role_use("cluster_validation", "routing")
    assert_role_use("sae_training", "dictionary_training")
    assert_role_use("final_evaluation", "final_intervention_evaluation")
    with pytest.raises(ValueError, match="calibration is forbidden"):
        assert_role_use("calibration", "routing")

