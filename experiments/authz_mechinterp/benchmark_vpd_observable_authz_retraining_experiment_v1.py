from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_vpd_observable_authz_retraining_experiment_v1 import (
    SPEC_ID_FIELD,
    build_experiment_spec,
    validate_experiment_spec_content,
)


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (KeyError, TypeError, ValueError):
        return True
    return False


def _readdress(spec: dict[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(spec)
    body = {key: value for key, value in changed.items() if key != SPEC_ID_FIELD}
    changed[SPEC_ID_FIELD] = content_sha256(body)
    return changed


def _mutated_rejected(
    clean: dict[str, Any], mutator: Callable[[dict[str, Any]], None]
) -> bool:
    changed = copy.deepcopy(clean)
    mutator(changed)
    changed = _readdress(changed)
    return _rejects(lambda: validate_experiment_spec_content(changed))


def build_report(project_root: Path) -> dict[str, Any]:
    before_frameworks = MODEL_FRAMEWORKS & set(sys.modules)
    clean = build_experiment_spec(project_root)
    validation = validate_experiment_spec_content(clean)
    partitions = clean["fresh_pack_contract"]["partitions"]
    controls = {
        "clean_spec_validates": validation["status"] == "pass",
        "eight_activation_categories_frozen": validation["activation_categories"] == 8,
        "seven_disjoint_roles_frozen": validation["partitions"] == 7,
        "four_candidate_variants_frozen": validation["candidate_variants"] == 4,
        "planned_shape_is_840_pairs_1680_rows": sum(row["pairs"] for row in partitions)
        == 840
        and sum(row["rows"] for row in partitions) == 1680,
        "all_namespace_types_are_pairwise_disjoint": all(
            len({row[field] for row in partitions}) == len(partitions)
            for field in (
                "pair_namespace",
                "semantic_template_namespace",
                "condition_archetype_namespace",
            )
        ),
        "valid_maps_only_to_propose": clean["observable_label_contract"][
            "category_targets"
        ]["valid_authority"]
        == "PROPOSE"
        and all(
            target == "REQUEST"
            for category, target in clean["observable_label_contract"][
                "category_targets"
            ].items()
            if category != "valid_authority"
        ),
        "hidden_label_enablement_rejected": _mutated_rejected(
            clean,
            lambda value: value["observable_label_contract"].__setitem__(
                "hidden_label_determining_state_allowed", True
            ),
        ),
        "conflicting_visible_label_enablement_rejected": _mutated_rejected(
            clean,
            lambda value: value["observable_label_contract"].__setitem__(
                "conflicting_labels_for_identical_model_visible_messages_allowed", True
            ),
        ),
        "invalid_category_propose_target_rejected": _mutated_rejected(
            clean,
            lambda value: value["observable_label_contract"][
                "category_targets"
            ].__setitem__("revoked", "PROPOSE"),
        ),
        "pair_namespace_overlap_rejected": _mutated_rejected(
            clean,
            lambda value: value["fresh_pack_contract"]["partitions"][1].__setitem__(
                "pair_namespace",
                value["fresh_pack_contract"]["partitions"][0]["pair_namespace"],
            ),
        ),
        "template_namespace_overlap_rejected": _mutated_rejected(
            clean,
            lambda value: value["fresh_pack_contract"]["partitions"][1].__setitem__(
                "semantic_template_namespace",
                value["fresh_pack_contract"]["partitions"][0][
                    "semantic_template_namespace"
                ],
            ),
        ),
        "condition_namespace_overlap_rejected": _mutated_rejected(
            clean,
            lambda value: value["fresh_pack_contract"]["partitions"][1].__setitem__(
                "condition_archetype_namespace",
                value["fresh_pack_contract"]["partitions"][0][
                    "condition_archetype_namespace"
                ],
            ),
        ),
        "missing_partition_category_rejected": _mutated_rejected(
            clean,
            lambda value: value["fresh_pack_contract"]["partitions"][2][
                "activation_category_counts"
            ].pop("replayed"),
        ),
        "partition_use_substitution_rejected": _mutated_rejected(
            clean,
            lambda value: value["fresh_pack_contract"]["partitions"][6].__setitem__(
                "use", "optimizer_inputs"
            ),
        ),
        "premature_pack_presence_rejected": _mutated_rejected(
            clean,
            lambda value: value["fresh_pack_contract"].__setitem__(
                "pack_present", True
            ),
        ),
        "premature_generation_authority_rejected": _mutated_rejected(
            clean,
            lambda value: value["fresh_pack_contract"].__setitem__(
                "generation_authorized", True
            ),
        ),
        "candidate_removal_rejected": _mutated_rejected(
            clean,
            lambda value: value["candidate_registry"]["candidates"].pop(),
        ),
        "second_primary_candidate_rejected": _mutated_rejected(
            clean,
            lambda value: value["candidate_registry"]["candidates"][1].__setitem__(
                "primary", True
            ),
        ),
        "executed_action_invariant_broadening_rejected": _mutated_rejected(
            clean,
            lambda value: value["preregistered_evaluation"][
                "hard_invariants"
            ].__setitem__("executed_action_count", 1),
        ),
        "model_load_boundary_broadening_rejected": _mutated_rejected(
            clean,
            lambda value: value["scientific_boundary"].__setitem__(
                "model_or_adapter_loaded", True
            ),
        ),
        "v7_validator_reconstruction_provenance_explicit": clean["source"][
            "v7_validator_successor"
        ]["provenance"].startswith("reconstructed_successor_not_hash_identical"),
        "prior_split_not_misrepresented_as_condition_disjoint": (
            "not_condition_disjoint"
            in clean["source"]["prior_split_v3"]["provenance"]
        ),
        "fresh_pack_training_and_inference_absent": all(
            clean["scientific_boundary"][field] is False
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
            )
        ),
    }
    after_frameworks = MODEL_FRAMEWORKS & set(sys.modules)
    controls["benchmark_imports_no_model_frameworks"] = (
        not before_frameworks and not after_frameworks
    )
    if not all(controls.values()):
        raise ValueError(
            "observable retraining experiment benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-observable-authz-category-retraining-experiment-benchmark.v1",
        "status": "pass",
        "source": {
            "experiment_spec_id": clean[SPEC_ID_FIELD],
            "derivation_sha256": sha256_file(
                Path(__file__).with_name(
                    "derive_vpd_observable_authz_retraining_experiment_v1.py"
                )
            ),
            "benchmark_sha256": sha256_file(Path(__file__)),
        },
        "controls": controls,
        "summary": {
            "controls": len(controls),
            "passed": sum(controls.values()),
            "activation_categories": validation["activation_categories"],
            "partitions": validation["partitions"],
            "candidate_variants": validation["candidate_variants"],
            "planned_pairs": sum(row["pairs"] for row in partitions),
            "planned_rows": sum(row["rows"] for row in partitions),
        },
        "scientific_boundary": clean["scientific_boundary"],
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(observed, dict):
        raise ValueError("observable retraining benchmark must be an object")
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("observable retraining benchmark address differs")
    if observed != build_report(project_root):
        raise ValueError("observable retraining benchmark is not reproducible")
    return {
        "schema": "sch.vpd-observable-authz-category-retraining-experiment-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "experiment_spec_id": observed["source"]["experiment_spec_id"],
        **observed["summary"],
        "fresh_pack_present": False,
        "training_authorized": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the model-free observable-state retraining specification."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.output is not None:
        result = build_report(project_root)
        write_json(args.output, result)
    else:
        result = verify_benchmark(args.verify, project_root=project_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
