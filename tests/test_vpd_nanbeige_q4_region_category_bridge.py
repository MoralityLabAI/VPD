from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_nanbeige_q4_region_category_bridge_v1 import (
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]
SHORT_REPORT = ROOT / "artifacts/vpd_nanbeige_q4_region_category_bridge_v1/report.json"
LONG_REPORT = (
    ROOT / "artifacts/vpd_nanbeige_q4_region_category_bridge_8192_v1/report.json"
)


def test_q4_region_category_bridge_reproduces() -> None:
    for path in (SHORT_REPORT, LONG_REPORT):
        verification = verify_report(path)
        assert verification["status"] == "pass"
        assert verification["combinations"] == 6
        assert verification["execution_authorized"] is False


def test_q4_region_categories_fail_and_child_split_is_inoperative() -> None:
    report = json.loads(SHORT_REPORT.read_text(encoding="utf-8"))
    assert report["summary"] == {
        "admissible_within_region_nontrivial_allocations": 0,
        "region_category_admissions": 0,
        "screening_only_within_region_nontrivial_allocations": 1,
        "site_pass_combinations": 6,
    }
    assert all(
        row["region_category_admission"] == "fail"
        and row["within_region_cluster_gate"]
        == "fail_parent_category_not_admitted"
        and row["operative_within_region_allocation"]
        == {"prompt": 1, "early": 1, "middle": 1, "late": 1}
        for row in report["combinations"]
    )
    raw_nontrivial = [
        row
        for row in report["combinations"]
        if any(
            int(value) > 1
            for value in row["within_region_v2_selection"]["allocation"].values()
        )
    ]
    assert len(raw_nontrivial) == 1
    assert raw_nontrivial[0]["site"] == "model.layers.21"
    assert raw_nontrivial[0]["pass"] == 0
    assert report["scientific_boundary"][
        "region_order_reversal_is_a_limited_null_not_all_label_permutations"
    ] is True


def test_long_q4_region_control_rejects_categories_without_child_splits() -> None:
    report = json.loads(LONG_REPORT.read_text(encoding="utf-8"))
    assert report["source"]["max_new_tokens"] == 8192
    assert report["summary"] == {
        "admissible_within_region_nontrivial_allocations": 0,
        "region_category_admissions": 0,
        "screening_only_within_region_nontrivial_allocations": 0,
        "site_pass_combinations": 6,
    }
    assert all(
        row["region_category_admission"] == "fail"
        and row["within_region_cluster_gate"]
        == "fail_parent_category_not_admitted"
        and row["operative_within_region_allocation"]
        == {"prompt": 1, "early": 1, "middle": 1, "late": 1}
        for row in report["combinations"]
    )
