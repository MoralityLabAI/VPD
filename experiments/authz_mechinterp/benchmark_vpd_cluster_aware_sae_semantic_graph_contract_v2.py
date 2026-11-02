from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.build_cluster_aware_sae_semantic_graph_contract_v2 import (
    V1_MISMATCH_CORRECTIONS,
    build_contract,
    canonical_inputs,
    verify_contract,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (KeyError, TypeError, ValueError):
        return True
    return False


def _readdress(contract: dict[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(contract)
    body = {
        key: value
        for key, value in changed.items()
        if key != "semantic_graph_contract_id"
    }
    changed["semantic_graph_contract_id"] = content_sha256(body)
    return changed


def build_report(project_root: Path) -> dict[str, Any]:
    before_frameworks = MODEL_FRAMEWORKS & set(sys.modules)
    inputs = canonical_inputs(project_root)
    contract_path = (
        project_root
        / "artifacts/vpd_cluster_aware_sae_semantic_graph_contract_v2/contract.json"
    )
    observed = _load(contract_path)
    expected = build_contract(**inputs)
    verification = verify_contract(contract_path, **inputs)
    nodes = {row["node"]: row for row in observed["node_specs"]}
    joins = {row["join"]: row for row in observed["join_specs"]}
    corrections = {
        row["v1_edge"]: row["v2_representation"]
        for row in observed["v1_mismatch_corrections"]
    }

    with tempfile.TemporaryDirectory(prefix="vpd-semantic-graph-v2-") as value:
        temporary = Path(value)

        def rejected(candidate: dict[str, Any], name: str) -> bool:
            path = temporary / f"{name}.json"
            write_json(path, candidate)
            return _rejects(lambda: verify_contract(path, **inputs))

        unaddressed = copy.deepcopy(observed)
        unaddressed["summary"]["end_to_end_semantic_dispatch_allowed"] = True

        readdressed_dispatch = _readdress(unaddressed)

        missing_correction = copy.deepcopy(observed)
        missing_correction["v1_mismatch_corrections"] = missing_correction[
            "v1_mismatch_corrections"
        ][1:]
        missing_correction = _readdress(missing_correction)

        fictional_capture_id = copy.deepcopy(observed)
        next(
            row
            for row in fictional_capture_id["node_specs"]
            if row["node"] == "capture_authorization_receipt"
        )["id_field"] = "receipt_id"
        fictional_capture_id = _readdress(fictional_capture_id)

        independent_split = copy.deepcopy(observed)
        next(
            row
            for row in independent_split["node_specs"]
            if row["node"] == "cluster_aware_sae_plan_with_embedded_split"
        )["verification_mode"] = "standalone_plan_and_independent_split"
        independent_split = _readdress(independent_split)

        old_budget_schema = copy.deepcopy(observed)
        next(
            row
            for row in old_budget_schema["node_specs"]
            if row["node"] == "blueprint_bound_budget_proposal_v2"
        )["schema"] = "sch.vpd-cluster-aware-sae-budget-proposal.v1"
        old_budget_schema = _readdress(old_budget_schema)

        split_transition = copy.deepcopy(observed)
        next(
            row
            for row in split_transition["join_specs"]
            if row["join"] == "transition_combines_nonce_output_and_worker_ledger"
        )["join_kind"] = "three_independent_manifests"
        split_transition = _readdress(split_transition)

        fake_attachment = copy.deepcopy(observed)
        attachment = next(
            row
            for row in fake_attachment["node_specs"]
            if row["node"] == "production_runtime_attachment"
        )
        attachment.update(
            {
                "implemented": True,
                "schema": "sch.fake-production-attachment.v1",
                "id_field": "attachment_id",
                "module": "experiments.authz_mechinterp.attach_cluster_aware_sae_inert_runtime_v1",
                "callable": "verify_inert_runtime_attachment_claim",
            }
        )
        fake_attachment = _readdress(fake_attachment)

        weakened_paths = copy.deepcopy(observed)
        weakened_paths["path_policy"]["path_traversal_forbidden"] = False
        weakened_paths = _readdress(weakened_paths)

        temporary_removed_after_context = temporary
        attack_results = {
            "unaddressed_contract_tamper_rejected": rejected(
                unaddressed, "unaddressed"
            ),
            "readdressed_dispatch_enablement_rejected": rejected(
                readdressed_dispatch, "readdressed-dispatch"
            ),
            "readdressed_missing_v1_correction_rejected": rejected(
                missing_correction, "missing-correction"
            ),
            "readdressed_fictional_capture_id_rejected": rejected(
                fictional_capture_id, "fictional-capture-id"
            ),
            "readdressed_independent_split_rejected": rejected(
                independent_split, "independent-split"
            ),
            "readdressed_v1_budget_schema_rejected": rejected(
                old_budget_schema, "old-budget-schema"
            ),
            "readdressed_split_transition_rejected": rejected(
                split_transition, "split-transition"
            ),
            "readdressed_fake_production_attachment_rejected": rejected(
                fake_attachment, "fake-attachment"
            ),
            "readdressed_path_policy_weakening_rejected": rejected(
                weakened_paths, "weakened-path-policy"
            ),
        }

    controls = {
        "contract_matches_reproduction": observed == expected,
        "contract_is_content_addressed": observed["semantic_graph_contract_id"]
        == content_sha256(
            {
                key: value
                for key, value in observed.items()
                if key != "semantic_graph_contract_id"
            }
        ),
        "contract_verifier_passes": verification["status"] == "pass",
        "twenty_one_logical_nodes_are_declared": observed["summary"]["logical_nodes"]
        == 21,
        "twenty_implemented_representations_are_evidenced": observed["summary"][
            "implemented_node_representations"
        ]
        == 20,
        "one_terminal_node_is_explicit": observed["summary"]["terminal_gap_nodes"]
        == 1,
        "all_28_v1_mismatches_are_accounted": corrections
        == V1_MISMATCH_CORRECTIONS,
        "capture_receipt_uses_actual_id": nodes["capture_authorization_receipt"][
            "id_field"
        ]
        == "authorization_receipt_id",
        "guarded_v3_capture_replaces_raw_manifest_lineage": nodes[
            "guarded_v3_capture_receipt"
        ]["id_field"]
        == "capture_execution_v3_id"
        and nodes["capture_execution_v2_receipt"]["id_field"]
        == "capture_execution_id"
        and joins["v3_capture_to_predecessor_v2"]["target_node"]
        == "capture_execution_v2_receipt"
        and corrections["capture_manifest_to_receipt_id"].startswith(
            "removed_raw_manifest_edge"
        ),
        "routing_bundle_uses_actual_id": nodes["routing_evidence_bundle"][
            "id_field"
        ]
        == "routing_evidence_bundle_id",
        "global_pair_split_is_embedded": nodes[
            "cluster_aware_sae_plan_with_embedded_split"
        ]["required_companions"]
        == ["embedded_global_pair_split"]
        and joins["plan_to_global_split"]["join_kind"]
        == "embedded_content_addressed_subdocument",
        "materialization_requires_real_companions": nodes["materialization"][
            "required_companions"
        ]
        == ["routing_evidence_bundle", "predecessor_training_plan"],
        "budget_nodes_use_blueprint_bound_v2": nodes[
            "blueprint_bound_budget_proposal_v2"
        ]["schema"].endswith("blueprint-bound-budget-proposal.v2")
        and nodes["blueprint_bound_budget_admission_v2"]["schema"].endswith(
            "blueprint-bound-budget-admission.v2"
        ),
        "budget_proposal_binds_blueprint_id_and_hash": {
            "proposal_to_blueprint_id",
            "proposal_to_blueprint_hash",
        }.issubset(joins),
        "budget_admission_binds_blueprint_id_and_hash": {
            "budget_admission_to_blueprint_id",
            "budget_admission_to_blueprint_hash",
        }.issubset(joins),
        "capture_preflight_digest_is_not_mislabeled_file_hash": joins[
            "capture_receipt_to_preflight_canonical_digest"
        ]["join_kind"]
        == "exact_field_to_canonical_object_digest_not_file_hash",
        "registry_joins_are_cryptographic": all(
            joins[name]["join_kind"] == "cryptographic_authority_membership_join"
            for name in (
                "capture_receipt_to_registry",
                "release_receipt_to_registry",
                "compute_receipt_to_registry",
            )
        ),
        "durable_transition_is_one_combined_receipt": nodes[
            "durable_job_transition"
        ]["id_field"]
        == "transition_reservation_id"
        and joins["transition_combines_nonce_output_and_worker_ledger"][
            "join_kind"
        ]
        == "single_atomic_combined_receipt",
        "budget_to_signed_release_gap_is_explicit": joins[
            "budget_admission_to_release_receipt"
        ]["join_kind"]
        == "terminal_missing_signed_join",
        "budget_to_runtime_admission_gap_is_explicit": joins[
            "budget_admission_to_runtime_admission"
        ]["join_kind"]
        == "terminal_missing_runtime_join",
        "production_attachment_remains_absent": nodes[
            "production_runtime_attachment"
        ]["implemented"]
        is False
        and nodes["production_runtime_attachment"]["verifier_evidence"] is None,
        "safe_path_policy_is_mandatory": all(
            observed["path_policy"][field] is True
            for field in (
                "candidate_inputs_must_use_safe_relative_paths",
                "candidate_inputs_must_remain_under_dedicated_graph_root",
                "path_traversal_forbidden",
                "symlink_and_windows_junction_components_forbidden",
                "file_sha256_required_before_semantic_dispatch",
            )
        ),
        "all_implemented_verifier_modules_are_local_and_hashed": all(
            row["verifier_evidence"] is not None
            and row["verifier_evidence"]["callable_exists"] is True
            and len(row["verifier_evidence"]["module_sha256"]) == 64
            for row in observed["node_specs"]
            if row["implemented"]
        ),
        "end_to_end_dispatch_remains_disabled": observed["summary"][
            "end_to_end_semantic_dispatch_allowed"
        ]
        is False,
        "contract_transfers_no_authority": all(
            observed["scientific_boundary"][field] is False
            for field in (
                "execution_authorized",
                "sae_training_authorized",
                "paid_compute_authorized",
                "model_loaded",
                "model_inference_performed",
                "training_performed",
                "weights_loaded_or_mutated",
            )
        ),
        "benchmark_imports_no_model_frameworks": before_frameworks
        == (MODEL_FRAMEWORKS & set(sys.modules)),
        "temporary_attack_files_removed": not temporary_removed_after_context.exists(),
        **attack_results,
    }
    if not all(controls.values()):
        raise ValueError(
            "semantic graph V2 contract benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-aware-sae-semantic-graph-contract-benchmark.v2",
        "status": "pass",
        "source": {
            "semantic_graph_contract_path": contract_path.relative_to(
                project_root
            ).as_posix(),
            "semantic_graph_contract_sha256": sha256_file(contract_path),
            "semantic_graph_contract_id": observed["semantic_graph_contract_id"],
            "contract_builder_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/build_cluster_aware_sae_semantic_graph_contract_v2.py"
            ),
            "benchmark_builder_sha256": sha256_file(Path(__file__)),
        },
        "controls": controls,
        "summary": {
            "passed": sum(controls.values()),
            "controls": len(controls),
            **observed["summary"],
        },
        "remaining_gaps": observed["terminal_gaps"],
        "scientific_boundary": {
            "contract_benchmark_only": True,
            "semantic_dispatch_performed": False,
            "empirical_graph_present": False,
            "model_framework_imported": False,
            "model_loaded": False,
            "model_inference_performed": False,
            "training_performed": False,
            "weights_loaded_or_mutated": False,
            "persistent_dictionary_output_created": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
            "paid_compute_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body) or observed != build_report(project_root):
        raise ValueError("semantic graph V2 contract benchmark is not reproducible")
    return {
        "schema": "sch.vpd-cluster-aware-sae-semantic-graph-contract-benchmark-verification.v2",
        "status": "pass",
        "benchmark_id": identifier,
        "semantic_graph_contract_id": observed["source"][
            "semantic_graph_contract_id"
        ],
        "controls": observed["summary"]["controls"],
        "logical_nodes": observed["summary"]["logical_nodes"],
        "semantic_joins": observed["summary"]["semantic_joins"],
        "terminal_gap_joins": observed["summary"]["terminal_gap_joins"],
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark corrected semantic graph interface contract V2."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.output is not None:
        result = build_report(root)
        write_json(args.output, result)
    else:
        result = verify_benchmark(args.verify, project_root=root)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "benchmark_id": result.get("benchmark_id"),
                "semantic_graph_contract_id": result.get(
                    "semantic_graph_contract_id",
                    result.get("source", {}).get("semantic_graph_contract_id"),
                ),
                "controls": result.get(
                    "controls", result.get("summary", {}).get("controls")
                ),
                "execution_authorized": result.get(
                    "execution_authorized",
                    result.get("scientific_boundary", {}).get(
                        "execution_authorized"
                    ),
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
