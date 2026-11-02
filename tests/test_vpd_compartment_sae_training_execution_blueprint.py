from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.authz_mechinterp.activation_categories import CATEGORY_PROTOCOLS
from experiments.authz_mechinterp.benchmark_compartment_sae_training_execution_contract import (
    run_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256
from experiments.authz_mechinterp.derive_compartment_sae_training_execution_blueprint_v2 import (
    validate_blueprint,
)
from experiments.authz_mechinterp.verify_unified_vpd_compartment_pipeline_v6 import (
    canonical_inputs as lineage_v6_inputs,
    verify_receipt as verify_lineage_v6,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts/vpd_compartment_sae_training_plan_v3/plan.json"
BLUEPRINT = (
    ROOT
    / "artifacts/vpd_compartment_sae_training_execution_blueprint_v2/blueprint.json"
)
BENCHMARK = (
    ROOT
    / "artifacts/vpd_compartment_sae_training_execution_contract_benchmark_v1/report.json"
)
LINEAGE_V6 = ROOT / "artifacts/vpd_unified_compartment_pipeline_v6/receipt.json"


def test_durable_execution_blueprint_reproduces() -> None:
    blueprint = json.loads(BLUEPRINT.read_text(encoding="utf-8"))
    validate_blueprint(
        blueprint, training_plan_path=PLAN, project_root=ROOT
    )
    assert blueprint["execution_blueprint_id"] == (
        "33bc09fac66fcec62bc0374c39d7ad4209a5d17eafd9b1003b4353073687e40e"
    )
    assert blueprint["coverage"] == {
        "training_jobs": 18,
        "compartments": 9,
        "replicas_per_compartment": 2,
        "specific_compartments": 8,
        "shared_compartments": 1,
    }
    assert blueprint["scientific_boundary"]["execution_authorized"] is False
    assert blueprint["scientific_boundary"]["sae_training_performed"] is False


def test_exact_specific_and_shared_sample_contracts_are_frozen() -> None:
    blueprint = json.loads(BLUEPRINT.read_text(encoding="utf-8"))
    by_compartment = {
        row["compartment"]: row["required_sample_contract"]
        for row in blueprint["jobs"]
    }
    for category in CATEGORY_PROTOCOLS:
        sample = by_compartment[category]
        assert sample["specific_cross_compartment_rows"] == 0
        assert set(sample["selected_episode_counts_by_category"]) == {category}
    shared = by_compartment["shared_cross_category"]
    assert shared["shared_equal_episode_counts"] is True
    assert shared["selected_episode_counts_by_category"] == {
        category: 572 for category in sorted(CATEGORY_PROTOCOLS)
    }
    assert shared["selected_episodes"] == 4_576
    assert shared["selected_activation_rows"] == 585_728


def test_v2_blueprint_binds_predecessor_missing_hyperparameters() -> None:
    blueprint = json.loads(BLUEPRINT.read_text(encoding="utf-8"))
    assert blueprint["predecessor_gap"]["planned_but_not_manifest_bound"] == [
        "batch_size",
        "learning_rate",
    ]
    for row in blueprint["jobs"]:
        assert row["required_manifest_hyperparameters"]["batch_size"] == 128
        assert row["required_manifest_hyperparameters"]["learning_rate"] == 0.0003


def test_readdressed_hyperparameter_substitution_rejects() -> None:
    changed = copy.deepcopy(json.loads(BLUEPRINT.read_text(encoding="utf-8")))
    changed["jobs"][0]["required_manifest_hyperparameters"]["batch_size"] = 64
    body = {
        key: value
        for key, value in changed.items()
        if key != "execution_blueprint_id"
    }
    changed["execution_blueprint_id"] = content_sha256(body)
    with pytest.raises(ValueError, match="not reproducible"):
        validate_blueprint(changed, training_plan_path=PLAN, project_root=ROOT)


def test_training_execution_contract_benchmark_reproduces() -> None:
    expected = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    observed = run_benchmark(training_plan_path=PLAN, project_root=ROOT)
    assert observed == expected
    assert observed["benchmark_id"] == (
        "08255dba491199e4963080c8e059c8ebd0fdddaffb3e4fbd70c781a8685b2ff4"
    )
    assert observed["summary"] == {
        "scenarios": 8,
        "passed": 8,
        "v2_accepts_exact_contract": True,
        "v2_rejects_hyperparameter_or_sample_contract_substitution": True,
        "v1_batch_size_and_learning_rate_gap_reproduced": True,
    }


def test_unified_training_execution_lineage_v6_reproduces() -> None:
    verification = verify_lineage_v6(LINEAGE_V6, **lineage_v6_inputs(ROOT))
    assert verification == {
        "schema": "sch.vpd-unified-compartment-pipeline-verification.v6",
        "status": "pass",
        "pipeline_receipt_id": (
            "f4391a7bed1c8047d2b4ea08247b667a53619586f3931802b4146dee0d34187f"
        ),
        "training_jobs": 18,
        "execution_contract_scenarios": 8,
        "execution_authorized": False,
    }
