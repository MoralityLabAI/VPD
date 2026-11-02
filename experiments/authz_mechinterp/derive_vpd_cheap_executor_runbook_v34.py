from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_inert_campaign_v1 import (
    verify_benchmark as verify_campaign_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v42 import (
    canonical_inputs as protocol_v42_inputs,
    verify_protocol as verify_protocol_v42,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v34"
EXPECTED_V33_RUNBOOK_ID = (
    "23e7c03df8c0b73f71e85cfaea4977b230e129d82211fa7e944e71920eaeecc8"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V34 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V33_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V33 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v33":
        raise ValueError("pinned predecessor is not runbook V33")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V33 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v42_path: Path,
    campaign_execution_path: Path,
    campaign_benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    protocol_verification = verify_protocol_v42(
        protocol_v42_path, **protocol_v42_inputs(project_root)
    )
    benchmark_verification = verify_campaign_benchmark(
        campaign_benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    benchmark = _load(campaign_benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "exact_34_job_synthetic_inert_campaign_ready_production_campaign_"
        "empirical_blueprint_real_receipts_training_and_execution_absent"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v42"] = {
        "path": _relative(protocol_v42_path, project_root),
        "sha256": sha256_file(protocol_v42_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_aware_sae_inert_campaign_execution_v1"] = {
        "path": _relative(campaign_execution_path, project_root),
        "sha256": sha256_file(campaign_execution_path),
    }
    result["source"]["cluster_aware_sae_inert_campaign_benchmark_v1"] = {
        "path": _relative(campaign_benchmark_path, project_root),
        "sha256": sha256_file(campaign_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    current = result["current_result"]
    current["cluster_aware_sae_inert_campaign_v1"] = (
        "pass_28_of_28_exact_coverage_charging_isolation_failure_replay_and_tamper_controls"
    )
    current["synthetic_campaign_activation_categories"] = 8
    current["synthetic_campaign_compartments"] = 17
    current["synthetic_campaign_execution_jobs"] = 34
    current["synthetic_campaign_unique_output_directories"] = 34
    current["synthetic_campaign_success_callbacks_charged"] = 34
    current["synthetic_campaign_failure_callbacks_charged"] = 8
    current["synthetic_campaign_failure_callbacks_uninvoked"] = 26
    current["synthetic_inert_campaign_executed"] = True
    current["production_campaign_executed"] = False
    current["production_trainer_callback_invoked"] = False
    current["training_performed"] = False
    current["persistent_dictionary_output_created"] = False
    current["empirical_cluster_aware_blueprint_present"] = False
    current["real_scientific_release_receipt_present"] = False
    current["real_compute_authorization_present"] = False

    stage = result["stages"][0]
    stage["commands"].extend(
        [
            {
                "id": "verify_cluster_aware_sae_inert_campaign_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_inert_campaign_v1",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_inert_campaign_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v42",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v42",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v42.json",
                ],
            },
        ]
    )
    stage["expected"]["cluster_aware_sae_inert_campaign_v1"] = (
        "28 controls pass; exact 8-category/17-compartment/34-job coverage; 34 isolated outputs; success and failure charges exact; no training"
    )
    stage["stop_if"].extend(
        [
            "protocol V42 or inert-campaign benchmark differs",
            "campaign coverage is not exactly 34 jobs across 17 compartments and 8 categories",
            "a campaign or per-job callback runs before its durable claim exists",
            "a failure invokes a later callback or weakens per-compartment output isolation",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "exact synthetic inert 34-job campaign success/failure receipts pass controls; "
        "production campaign, empirical blueprint, real receipts, training, and execution authority are absent"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "synthetic campaign evidence is treated as empirical or production training authority",
        "a terminal campaign failure is deleted or retried",
        "two compartment jobs share an output directory",
    ]
    completion = result["completion_definition"]
    completion["cluster_aware_sae_inert_campaign_ready"] = True
    completion["exact_synthetic_campaign_job_coverage"] = True
    completion["synthetic_campaign_activation_categories"] = 8
    completion["synthetic_campaign_compartments"] = 17
    completion["synthetic_campaign_execution_jobs"] = 34
    completion["synthetic_campaign_unique_output_directories"] = 34
    completion["synthetic_inert_campaign_executed"] = True
    completion["production_campaign_executed"] = False
    completion["production_trainer_callback_invoked"] = False
    completion["training_performed"] = False
    completion["persistent_dictionary_output_created"] = False
    completion["empirical_cluster_aware_blueprint_present"] = False
    completion["real_scientific_release_receipt_present"] = False
    completion["real_compute_authorization_present"] = False
    boundary = result["scientific_boundary"]
    boundary["temporary_inert_campaign_fixture_only"] = True
    boundary["cluster_aware_inert_campaign_ready"] = True
    boundary["exact_synthetic_campaign_job_coverage"] = True
    boundary["synthetic_inert_campaign_executed"] = True
    boundary["production_campaign_executed"] = False
    boundary["production_trainer_callback_invoked"] = False
    boundary["training_performed"] = False
    boundary["persistent_dictionary_output_created"] = False
    boundary["empirical_cluster_aware_blueprint_present"] = False
    boundary["model_framework_imported"] = False
    boundary["model_or_adapter_weights_opened"] = False
    boundary["model_inference_performed"] = False
    boundary["execution_authorized"] = False
    boundary["sae_training_authorized"] = False
    boundary["paid_compute_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V34 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v34",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "cluster_aware_inert_campaign_ready": True,
        "categories": 8,
        "compartments": 17,
        "jobs": 34,
        "unique_output_directories": 34,
        "synthetic_inert_campaign_executed": True,
        "production_campaign_executed": False,
        "production_trainer_callback_invoked": False,
        "training_performed": False,
        "persistent_dictionary_output_created": False,
        "empirical_cluster_aware_blueprint_present": False,
        "real_scientific_release_receipt_present": False,
        "real_compute_authorization_present": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": (
            project_root / "artifacts/vpd_cheap_executor_runbook_v33/runbook.json"
        ),
        "protocol_v42_path": (
            project_root / "configs/evals/vpd_activation_category_compartments_v42.json"
        ),
        "campaign_execution_path": (
            project_root
            / "experiments/authz_mechinterp/execute_cluster_aware_sae_inert_campaign_v1.py"
        ),
        "campaign_benchmark_path": (
            project_root
            / "artifacts/vpd_cluster_aware_sae_inert_campaign_benchmark_v1/report.json"
        ),
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V34.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.output is not None:
        result = build_runbook(**inputs)
        write_json(args.output, result)
    else:
        result = verify_runbook(args.verify, **inputs)
    print(
        json.dumps(
            {key: result[key] for key in ("schema", "status", "runbook_id")},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
