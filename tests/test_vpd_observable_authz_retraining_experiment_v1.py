from __future__ import annotations

import json
from pathlib import Path

from experiments.authz_mechinterp.benchmark_vpd_observable_authz_retraining_experiment_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.derive_vpd_observable_authz_retraining_experiment_v1 import (
    verify_experiment_spec,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/evals/vpd_observable_authz_retraining_experiment_v1.json"
REPORT = (
    ROOT
    / "artifacts/vpd_observable_authz_retraining_experiment_benchmark_v1/report.json"
)
SPEC_ID = "0c2f14cad8a05efeddda7a2e3428f2d323f49ee1a48f173dfc778a71f855dd86"
BENCHMARK_ID = "c0c88eeb6cdce5fc00564a38712d6e4ac1f771d59df216985d18fd8d497611a3"


def test_observable_authz_retraining_spec_and_benchmark_reproduce() -> None:
    spec = verify_experiment_spec(SPEC, project_root=ROOT)
    assert spec["status"] == "pass"
    assert spec["experiment_spec_id"] == SPEC_ID
    assert spec["activation_categories"] == 8
    assert spec["partitions"] == 7
    assert spec["candidate_variants"] == 4
    assert spec["fresh_pack_present"] is False
    assert spec["training_authorized"] is False
    benchmark = verify_benchmark(REPORT, project_root=ROOT)
    assert benchmark["status"] == "pass"
    assert benchmark["benchmark_id"] == BENCHMARK_ID
    assert benchmark["experiment_spec_id"] == SPEC_ID
    assert benchmark["controls"] == 25
    assert benchmark["planned_pairs"] == 840
    assert benchmark["planned_rows"] == 1680
    assert benchmark["fresh_pack_present"] is False
    assert benchmark["training_authorized"] is False
    assert benchmark["execution_authorized"] is False


def test_observable_authz_retraining_contract_is_disjoint_and_nonexecuting() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    partitions = spec["fresh_pack_contract"]["partitions"]
    for field in (
        "pair_namespace",
        "semantic_template_namespace",
        "condition_archetype_namespace",
    ):
        values = [row[field] for row in partitions]
        assert len(values) == len(set(values)) == 7
    assert spec["observable_label_contract"]["category_targets"] == {
        "valid_authority": "PROPOSE",
        "missing_or_untrusted_authority": "REQUEST",
        "signature_invalid": "REQUEST",
        "time_inactive": "REQUEST",
        "revoked": "REQUEST",
        "replayed": "REQUEST",
        "scope_invalid": "REQUEST",
        "other_invalid": "REQUEST",
    }
    assert spec["candidate_registry"]["primary_candidate_id"] == (
        "fresh_lora_margin_v1"
    )
    assert spec["activation_category_compartments"][
        "final_evaluation_may_change_clusters_or_thresholds"
    ] is False
    assert spec["source"]["v7_validator_successor"]["provenance"].startswith(
        "reconstructed_successor_not_hash_identical"
    )
    boundary = spec["scientific_boundary"]
    assert boundary["design_only"] is True
    for field in (
        "fresh_pack_present",
        "empirical_inputs_created",
        "model_framework_imported",
        "model_or_adapter_loaded",
        "model_inference_performed",
        "gpu_or_external_job_launched",
        "training_performed",
        "weights_or_dictionaries_mutated",
        "real_actions_performed",
        "execution_authorized",
        "sae_training_authorized",
        "candidate_training_authorized",
        "paid_compute_authorized",
    ):
        assert boundary[field] is False


def test_observable_authz_retraining_mutation_controls_are_pinned() -> None:
    observed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert observed["benchmark_id"] == BENCHMARK_ID
    for control in (
        "hidden_label_enablement_rejected",
        "conflicting_visible_label_enablement_rejected",
        "invalid_category_propose_target_rejected",
        "pair_namespace_overlap_rejected",
        "template_namespace_overlap_rejected",
        "condition_namespace_overlap_rejected",
        "missing_partition_category_rejected",
        "partition_use_substitution_rejected",
        "premature_pack_presence_rejected",
        "premature_generation_authority_rejected",
        "candidate_removal_rejected",
        "second_primary_candidate_rejected",
        "executed_action_invariant_broadening_rejected",
        "model_load_boundary_broadening_rejected",
        "v7_validator_reconstruction_provenance_explicit",
        "prior_split_not_misrepresented_as_condition_disjoint",
        "fresh_pack_training_and_inference_absent",
        "benchmark_imports_no_model_frameworks",
    ):
        assert observed["controls"][control] is True
