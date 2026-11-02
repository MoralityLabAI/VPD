from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.build_cluster_aware_sae_compartment_resource_reconciliation_v1 import (
    canonical_inputs,
    verify_resource_reconciliation,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return True
    return False


def build_report(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    ledger_path = (
        root
        / "artifacts/vpd_cluster_aware_sae_compartment_resource_reconciliation_v1/ledger.json"
    )
    verification = verify_resource_reconciliation(
        ledger_path, **canonical_inputs(root)
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    categories = ledger["category_specific_resource_accounts"]
    compartments = ledger["compartment_resource_accounts"]
    shared = ledger["shared_fallback_resource_account"]
    global_resources = ledger["global_planned_resources"]
    success = ledger["success_reconciliation"]
    failure = ledger["terminal_failure_reconciliation"]
    category_steps = sum(row["resources"]["planned_steps"] for row in categories)
    category_rows = sum(
        row["resources"]["planned_activation_rows"] for row in categories
    )
    category_jobs = sum(row["resources"]["jobs"] for row in categories)
    failure_state_resources = failure["planned_resources_by_state"]
    failure_state_steps = sum(
        row["planned_steps"] for row in failure_state_resources.values()
    )
    failure_state_rows = sum(
        row["planned_activation_rows"] for row in failure_state_resources.values()
    )
    with tempfile.TemporaryDirectory(
        prefix="vpd-sae-resource-reconciliation-v1-"
    ) as temporary:
        temp_root = Path(temporary)
        original_text = ledger_path.read_text(encoding="utf-8")

        changed = json.loads(original_text)
        changed["global_planned_resources"]["planned_steps"] += 1
        changed_path = temp_root / "changed-global.json"
        write_json(changed_path, changed)
        global_tamper_rejected = _rejects(
            lambda: verify_resource_reconciliation(
                changed_path, **canonical_inputs(root)
            )
        )

        changed = json.loads(original_text)
        changed["category_specific_resource_accounts"][0][
            "shared_fallback_accounted_separately"
        ] = False
        changed_path = temp_root / "changed-category.json"
        write_json(changed_path, changed)
        category_tamper_rejected = _rejects(
            lambda: verify_resource_reconciliation(
                changed_path, **canonical_inputs(root)
            )
        )

        changed = json.loads(original_text)
        changed["terminal_failure_reconciliation"]["summary"][
            "same_campaign_retry_authorized"
        ] = True
        changed_path = temp_root / "changed-retry.json"
        write_json(changed_path, changed)
        retry_tamper_rejected = _rejects(
            lambda: verify_resource_reconciliation(
                changed_path, **canonical_inputs(root)
            )
        )

        changed = json.loads(original_text)
        changed["source"]["raw_campaign_receipts_present"] = True
        changed_path = temp_root / "changed-provenance.json"
        write_json(changed_path, changed)
        provenance_tamper_rejected = _rejects(
            lambda: verify_resource_reconciliation(
                changed_path, **canonical_inputs(root)
            )
        )

        campaign_path = canonical_inputs(root)["campaign_benchmark_path"]
        original_campaign = campaign_path.read_text(encoding="utf-8")
        try:
            campaign_path.write_text(original_campaign + " ", encoding="utf-8")
            campaign_source_tamper_rejected = _rejects(
                lambda: verify_resource_reconciliation(
                    ledger_path, **canonical_inputs(root)
                )
            )
        finally:
            campaign_path.write_text(original_campaign, encoding="utf-8")

    controls = {
        "ledger_reproduces_from_pinned_campaign_summary": verification["status"]
        == "pass",
        "provenance_explicitly_reconstructed_not_raw_receipt": ledger["source"][
            "raw_campaign_receipts_present"
        ]
        is False
        and ledger["source"]["reconstructed_from_verified_summary"] is True,
        "exact_8_category_accounts": len(categories) == 8
        and all(row["resources"]["jobs"] == 4 for row in categories),
        "each_category_has_two_specific_compartments": all(
            len(row["specific_compartment_ids"]) == 2 for row in categories
        ),
        "exact_17_compartment_accounts": len(compartments) == 17,
        "each_compartment_preserves_two_seed_jobs": all(
            row["resources"]["jobs"] == 2 for row in compartments
        ),
        "shared_fallback_is_separate_two_job_account": shared["resources"]["jobs"]
        == 2
        and shared["never_double_counted_into_category_totals"] is True,
        "specific_and_shared_jobs_sum_to_global": category_jobs
        + shared["resources"]["jobs"]
        == global_resources["jobs"]
        == 34,
        "specific_and_shared_steps_sum_to_global": category_steps
        + shared["resources"]["planned_steps"]
        == global_resources["planned_steps"]
        == 68000,
        "specific_and_shared_rows_sum_to_global": category_rows
        + shared["resources"]["planned_activation_rows"]
        == global_resources["planned_activation_rows"]
        == 20480,
        "dictionary_widths_remain_compartment_specific": all(
            row["resources"]["dictionary_feature_widths"] == [4096]
            for row in categories
        )
        and shared["resources"]["dictionary_feature_widths"] == [8192],
        "all_34_output_directories_remain_unique": global_resources[
            "unique_output_directories"
        ]
        == 34,
        "success_reconciliation_charges_34_without_training": success["summary"][
            "charged_callbacks"
        ]
        == 34
        and success["summary"]["training_steps_observed"] == 0,
        "failure_reconciliation_is_7_success_1_failure_26_uninvoked": failure[
            "summary"
        ]["completed_callbacks"]
        == 7
        and failure["summary"]["failed_callbacks"] == 1
        and failure["summary"]["uninvoked_callbacks"] == 26,
        "failure_state_planned_steps_reconcile_to_global": failure_state_steps
        == global_resources["planned_steps"],
        "failure_state_planned_rows_reconcile_to_global": failure_state_rows
        == global_resources["planned_activation_rows"],
        "failure_and_uninvoked_jobs_require_fresh_authority": all(
            row["fresh_authorization_required_for_any_real_execution"] is True
            and row["same_campaign_retry_authorized"] is False
            for row in failure["job_states"]
        ),
        "synthetic_receipts_transfer_no_authority": failure[
            "fresh_authorization_requirements"
        ]["synthetic_campaign_receipt_transfers_no_authority"]
        is True,
        "global_resource_tamper_rejected": global_tamper_rejected,
        "category_partition_tamper_rejected": category_tamper_rejected,
        "retry_authority_tamper_rejected": retry_tamper_rejected,
        "reconstruction_provenance_tamper_rejected": provenance_tamper_rejected,
        "campaign_source_byte_tamper_rejected": campaign_source_tamper_rejected,
        "resource_ledger_remains_nonexecuting": all(
            ledger["scientific_boundary"][field_name] in (False, 0)
            for field_name in (
                "raw_campaign_receipts_recovered",
                "resource_accounting_is_empirical_evidence",
                "model_framework_imported",
                "model_loaded",
                "model_inference_performed",
                "training_performed",
                "persistent_dictionary_output_created",
                "weights_loaded_or_mutated",
                "production_campaign_executed",
                "retry_authorized",
                "execution_authorized",
                "sae_training_authorized",
            )
        ),
    }
    if not all(controls.values()):
        raise ValueError(
            "cluster-aware SAE resource reconciliation benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-aware-sae-compartment-resource-reconciliation-benchmark.v1",
        "status": "pass",
        "source": {
            "resource_reconciliation_path": ledger_path.relative_to(root).as_posix(),
            "resource_reconciliation_sha256": sha256_file(ledger_path),
            "resource_reconciliation_id": ledger["resource_reconciliation_id"],
            "builder_sha256": sha256_file(
                Path(__file__).with_name(
                    "build_cluster_aware_sae_compartment_resource_reconciliation_v1.py"
                )
            ),
            "benchmark_sha256": sha256_file(Path(__file__)),
        },
        "design": {
            "reconstructed_from_verified_campaign_summary": True,
            "raw_campaign_receipts_present": False,
            "activation_categories": 8,
            "specific_compartments": 16,
            "shared_fallback_compartments": 1,
            "execution_jobs": 34,
            "unique_output_directories": 34,
            "shared_fallback_accounted_separately": True,
        },
        "controls": controls,
        "summary": {"controls": len(controls), "passed": sum(controls.values())},
        "results": {
            "planned_steps": global_resources["planned_steps"],
            "planned_activation_rows": global_resources[
                "planned_activation_rows"
            ],
            "dictionary_feature_widths": global_resources[
                "dictionary_feature_widths"
            ],
            "success_charged_callbacks": success["summary"]["charged_callbacks"],
            "failure_charged_callbacks": failure["summary"]["charged_callbacks"],
            "failure_uninvoked_callbacks": failure["summary"][
                "uninvoked_callbacks"
            ],
        },
        "scientific_boundary": {
            "synthetic_reconstruction_only": True,
            "raw_campaign_receipts_recovered": False,
            "empirical_evidence_created": False,
            "model_loaded": False,
            "inference_performed": False,
            "training_performed": False,
            "weights_mutated": False,
            "persistent_dictionary_output_created": False,
            "retry_authorized": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("resource reconciliation benchmark address differs")
    if observed != build_report(project_root):
        raise ValueError("resource reconciliation benchmark is not reproducible")
    return {
        "schema": "sch.vpd-cluster-aware-sae-compartment-resource-reconciliation-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "controls": observed["summary"]["controls"],
        "resource_reconciliation_id": observed["source"][
            "resource_reconciliation_id"
        ],
        "activation_categories": observed["design"]["activation_categories"],
        "compartments": observed["design"]["specific_compartments"]
        + observed["design"]["shared_fallback_compartments"],
        "jobs": observed["design"]["execution_jobs"],
        "planned_steps": observed["results"]["planned_steps"],
        "planned_activation_rows": observed["results"][
            "planned_activation_rows"
        ],
        "raw_campaign_receipts_recovered": False,
        "training_performed": False,
        "retry_authorized": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark compartment resource and failure reconciliation V1."
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
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
