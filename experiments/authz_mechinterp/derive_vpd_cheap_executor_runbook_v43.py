from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_observable_authz_retraining_experiment_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v51 import (
    canonical_inputs as protocol_v51_inputs,
    verify_protocol as verify_protocol_v51,
)
from experiments.authz_mechinterp.derive_vpd_observable_authz_retraining_experiment_v1 import (
    verify_experiment_spec,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v43"
EXPECTED_V42_RUNBOOK_ID = (
    "917c9582a1e4d71015eaef89507b335ad54bbebd5c2930482c1aab81d6842589"
)
EXPECTED_V51_SHA256 = (
    "1b49c83ed86571a9fb9c9f54b768b0ee470123d9a77d63d1bdf7ef637ad2ba2c"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V43 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V42_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V42 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v42":
        raise ValueError("pinned predecessor is not runbook V42")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V42 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v51_path: Path,
    experiment_derivation_path: Path,
    experiment_spec_path: Path,
    benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    if sha256_file(protocol_v51_path) != EXPECTED_V51_SHA256:
        raise ValueError("protocol V51 differs from its pinned SHA-256")
    protocol_verification = verify_protocol_v51(
        protocol_v51_path, **protocol_v51_inputs(project_root)
    )
    spec_verification = verify_experiment_spec(
        experiment_spec_path, project_root=project_root
    )
    benchmark_verification = verify_benchmark(
        benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    spec = _load(experiment_spec_path)
    benchmark = _load(benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "observable_category_retraining_design_verified_fresh_pack_training_"
        "inference_durable_transition_and_execution_absent"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v51"] = {
        "path": _relative(protocol_v51_path, project_root),
        "sha256": sha256_file(protocol_v51_path),
        "verification": protocol_verification,
    }
    result["source"]["observable_authz_retraining_derivation_v1"] = {
        "path": _relative(experiment_derivation_path, project_root),
        "sha256": sha256_file(experiment_derivation_path),
    }
    result["source"]["observable_authz_retraining_spec_v1"] = {
        "path": _relative(experiment_spec_path, project_root),
        "sha256": sha256_file(experiment_spec_path),
        "experiment_spec_id": spec["experiment_spec_id"],
        "verification": spec_verification,
    }
    result["source"]["observable_authz_retraining_benchmark_v1"] = {
        "path": _relative(benchmark_path, project_root),
        "sha256": sha256_file(benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    current = result["current_result"]
    current["observable_authz_category_retraining_experiment_v1"] = (
        "pass_25_of_25_disjointness_label_candidate_invariant_provenance_and_boundary_controls"
    )
    current["observable_retraining_activation_categories"] = 8
    current["observable_retraining_disjoint_partitions"] = 7
    current["observable_retraining_candidate_variants"] = 4
    current["observable_retraining_planned_pairs"] = 840
    current["observable_retraining_planned_rows"] = 1680
    current["observable_retraining_primary_candidate"] = "fresh_lora_margin_v1"
    current["observable_retraining_pair_template_condition_disjoint"] = True
    current["observable_retraining_fresh_pack_present"] = False
    current["observable_retraining_pack_generation_authorized"] = False
    current["training_performed"] = False
    current["persistent_dictionary_output_created"] = False

    stage = result["stages"][0]
    stage["commands"].extend(
        [
            {
                "id": "verify_observable_authz_retraining_experiment_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_vpd_observable_authz_retraining_experiment_v1",
                    "--verify",
                    "configs/evals/vpd_observable_authz_retraining_experiment_v1.json",
                ],
            },
            {
                "id": "verify_observable_authz_retraining_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_observable_authz_retraining_experiment_v1",
                    "--verify",
                    "artifacts/vpd_observable_authz_retraining_experiment_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v51",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v51",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v51.json",
                ],
            },
        ]
    )
    stage["expected"]["observable_authz_category_retraining_experiment_v1"] = (
        "25 controls pass; seven role namespaces are pair template and condition "
        "disjoint; eight activation categories and four candidates are frozen; "
        "840 pairs and 1680 rows are planned; no pack or training exists"
    )
    stage["stop_if"].extend(
        [
            "protocol V51 experiment specification or benchmark differs",
            "any pair template or condition namespace overlaps another role",
            "invalid observable host state maps to PROPOSE",
            "valid trusted observable host state maps to REQUEST",
            "hidden or conflicting labels are introduced",
            "fresh pack generation model load inference or training begins",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "the fresh observable-state retraining and category-compartment experiment "
        "is fully specified but no held-out pack exists; generation, empirical "
        "capture, SAE or candidate training, inference, and evaluation require new "
        "explicit user authorization and approved compute/runtime receipts"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "fresh pack generation lacks new explicit authorization",
        "candidate or dictionary training lacks an approved compute ceiling",
        "final evaluation is exposed before all hashes and thresholds freeze",
    ]
    completion = result["completion_definition"]
    completion["observable_authz_category_retraining_experiment_v1_verified"] = True
    completion["observable_retraining_activation_categories"] = 8
    completion["observable_retraining_disjoint_partitions"] = 7
    completion["observable_retraining_candidate_variants"] = 4
    completion["observable_retraining_planned_pairs"] = 840
    completion["observable_retraining_planned_rows"] = 1680
    completion["observable_retraining_pair_template_condition_disjoint"] = True
    completion["observable_retraining_fresh_pack_present"] = False
    completion["observable_retraining_pack_generation_authorized"] = False
    completion["training_performed"] = False
    completion["persistent_dictionary_output_created"] = False
    boundary = result["scientific_boundary"]
    boundary["observable_authz_category_retraining_experiment_v1_verified"] = True
    boundary["observable_retraining_pair_template_condition_disjoint"] = True
    boundary["observable_retraining_fresh_pack_present"] = False
    boundary["observable_retraining_pack_generation_authorized"] = False
    boundary["empirical_cross_artifact_graph_present"] = False
    boundary["semantic_verifiers_invoked_on_empirical_inputs"] = False
    boundary["production_campaign_executed"] = False
    boundary["production_trainer_callback_invoked"] = False
    boundary["training_performed"] = False
    boundary["persistent_dictionary_output_created"] = False
    boundary["model_framework_imported"] = False
    boundary["model_or_adapter_weights_opened"] = False
    boundary["model_inference_performed"] = False
    boundary["weights_loaded_or_mutated"] = False
    boundary["execution_authorized"] = False
    boundary["sae_training_authorized"] = False
    boundary["candidate_training_authorized"] = False
    boundary["paid_compute_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V43 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v43",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "observable_retraining_experiment_verified": True,
        "activation_categories": 8,
        "disjoint_partitions": 7,
        "candidate_variants": 4,
        "planned_pairs": 840,
        "planned_rows": 1680,
        "fresh_pack_present": False,
        "pack_generation_authorized": False,
        "training_performed": False,
        "persistent_dictionary_output_created": False,
        "sae_training_authorized": False,
        "candidate_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root
        / "artifacts/vpd_cheap_executor_runbook_v42/runbook.json",
        "protocol_v51_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v51.json",
        "experiment_derivation_path": project_root
        / "experiments/authz_mechinterp/derive_vpd_observable_authz_retraining_experiment_v1.py",
        "experiment_spec_path": project_root
        / "configs/evals/vpd_observable_authz_retraining_experiment_v1.json",
        "benchmark_path": project_root
        / "artifacts/vpd_observable_authz_retraining_experiment_benchmark_v1/report.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V43.")
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
