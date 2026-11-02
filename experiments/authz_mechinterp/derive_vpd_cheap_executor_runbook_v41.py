from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_semantic_dispatch_slice_v1 import (
    verify_benchmark,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v49 import (
    canonical_inputs as protocol_v49_inputs,
    verify_protocol as verify_protocol_v49,
)


SCHEMA = "sch.vpd-cheap-executor-runbook.v41"
EXPECTED_V40_RUNBOOK_ID = (
    "08444884365069ea91a362f0fc4be7ca4f7552947780c6a016f739553c0c4681"
)
EXPECTED_V49_SHA256 = (
    "0c3798bdca4dd501e8f674afdfb368319ebb6010a4e8ea4f82aff92afec48fbb"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("runbook V41 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_predecessor(path: Path) -> dict[str, Any]:
    observed = _load(path)
    body = dict(observed)
    identifier = str(body.pop("runbook_id", ""))
    if identifier != EXPECTED_V40_RUNBOOK_ID or identifier != content_sha256(body):
        raise ValueError("runbook V40 differs from its pinned content ID")
    if observed.get("schema") != "sch.vpd-cheap-executor-runbook.v40":
        raise ValueError("pinned predecessor is not runbook V40")
    if observed["scientific_boundary"].get("execution_authorized") is not False:
        raise ValueError("runbook V40 authorization boundary differs")
    return {
        "schema": "sch.vpd-pinned-runbook-predecessor-verification.v1",
        "status": "pass",
        "runbook_id": identifier,
    }


def build_runbook(
    *,
    predecessor_path: Path,
    protocol_v49_path: Path,
    dispatcher_path: Path,
    benchmark_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = _verify_predecessor(predecessor_path)
    if sha256_file(protocol_v49_path) != EXPECTED_V49_SHA256:
        raise ValueError("protocol V49 differs from its pinned SHA-256")
    protocol_verification = verify_protocol_v49(
        protocol_v49_path, **protocol_v49_inputs(project_root)
    )
    benchmark_verification = verify_benchmark(
        benchmark_path, project_root=project_root
    )
    predecessor = _load(predecessor_path)
    benchmark = _load(benchmark_path)
    result = json.loads(json.dumps(predecessor))
    result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = (
        "four_node_full_reproduction_semantic_slice_verified_three_terminal_"
        "gaps_empirical_dispatch_training_and_execution_absent"
    )
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root),
        "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"],
        "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v49"] = {
        "path": _relative(protocol_v49_path, project_root),
        "sha256": sha256_file(protocol_v49_path),
        "verification": protocol_verification,
    }
    result["source"]["cluster_aware_sae_semantic_dispatcher_v1"] = {
        "path": _relative(dispatcher_path, project_root),
        "sha256": sha256_file(dispatcher_path),
    }
    result["source"]["cluster_aware_sae_semantic_dispatch_benchmark_v1"] = {
        "path": _relative(benchmark_path, project_root),
        "sha256": sha256_file(benchmark_path),
        "benchmark_id": benchmark["benchmark_id"],
        "semantic_dispatch_slice_id": benchmark["results"][
            "semantic_dispatch_slice_id"
        ],
        "verification": benchmark_verification,
    }
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    current = result["current_result"]
    current["cluster_aware_sae_semantic_dispatch_slice_v1"] = (
        "pass_22_of_22_full_reproduction_companion_path_mutation_gap_and_authority_controls"
    )
    current["semantic_dispatch_slice_nodes"] = 4
    current["semantic_dispatch_full_reproduction_comparisons"] = 4
    current["semantic_dispatch_execution_jobs"] = 34
    current["semantic_dispatch_terminal_gap_count"] = 3
    current["semantic_dispatch_synthetic_fixture_persisted"] = False
    current["semantic_dispatch_empirical_graph_present"] = False
    current["end_to_end_semantic_dispatch_complete"] = False
    current["signed_authority_dispatched"] = False
    current["durable_transition_dispatched"] = False
    current["production_runtime_attachment_dispatched"] = False
    current["semantic_dispatch_allowed"] = False
    current["training_performed"] = False
    current["persistent_dictionary_output_created"] = False

    stage = result["stages"][0]
    stage["commands"].extend(
        [
            {
                "id": "verify_cluster_aware_sae_semantic_dispatch_slice_benchmark_v1",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.benchmark_vpd_cluster_aware_sae_semantic_dispatch_slice_v1",
                    "--verify",
                    "artifacts/vpd_cluster_aware_sae_semantic_dispatch_slice_benchmark_v1/report.json",
                ],
            },
            {
                "id": "verify_research_protocol_v49",
                "executable": ".venv/Scripts/python.exe",
                "arguments": [
                    "-m",
                    "experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v49",
                    "--verify",
                    "configs/evals/vpd_activation_category_compartments_v49.json",
                ],
            },
        ]
    )
    stage["expected"]["cluster_aware_sae_semantic_dispatch_slice_v1"] = (
        "22 controls pass; four real semantic verifiers reproduce four full artifacts; "
        "34 exact jobs remain bound; three terminal gaps remain; no authority transfers"
    )
    stage["stop_if"].extend(
        [
            "protocol V49 or semantic dispatch slice benchmark differs",
            "a dispatcher compares only schema or content ID without full reproduction",
            "a companion path or exact blueprint byte hash is substituted",
            "signed authority durable transition or production attachment is inferred",
            "synthetic dispatcher evidence is presented as empirical",
        ]
    )
    gate = result["stages"][1]
    gate["implementation_status"] = (
        "the materialization blueprint proposal admission slice is semantically dispatched; "
        "end-to-end dispatch remains blocked on budget-bound signed release, budget-bound "
        "runtime admission, and a separately authorized production attachment verifier"
    )
    gate["stop_if"] = list(gate["stop_if"]) + [
        "budget admission is not bound by a new signed release verifier version",
        "runtime admission does not reproduce the exact budget admission",
        "production attachment verifier remains absent",
    ]
    completion = result["completion_definition"]
    completion["cluster_aware_sae_semantic_dispatch_slice_v1_verified"] = True
    completion["semantic_dispatch_slice_nodes"] = 4
    completion["semantic_dispatch_full_reproduction_comparisons"] = 4
    completion["semantic_dispatch_execution_jobs"] = 34
    completion["semantic_dispatch_terminal_gap_count"] = 3
    completion["semantic_dispatch_synthetic_fixture_persisted"] = False
    completion["semantic_dispatch_empirical_graph_present"] = False
    completion["end_to_end_semantic_dispatch_complete"] = False
    completion["signed_authority_dispatched"] = False
    completion["durable_transition_dispatched"] = False
    completion["production_runtime_attachment_dispatched"] = False
    completion["semantic_dispatch_allowed"] = False
    completion["training_performed"] = False
    completion["persistent_dictionary_output_created"] = False
    boundary = result["scientific_boundary"]
    boundary["cluster_aware_sae_semantic_dispatch_slice_v1_verified"] = True
    boundary["semantic_dispatch_slice_nodes"] = 4
    boundary["semantic_dispatch_full_reproduction_comparisons"] = 4
    boundary["semantic_dispatch_synthetic_fixture_persisted"] = False
    boundary["semantic_dispatch_allowed"] = False
    boundary["end_to_end_semantic_dispatch_complete"] = False
    boundary["signed_authority_dispatched"] = False
    boundary["durable_transition_dispatched"] = False
    boundary["production_runtime_attachment_dispatched"] = False
    boundary["empirical_cross_artifact_graph_present"] = False
    boundary["semantic_verifiers_invoked_on_empirical_inputs"] = False
    boundary["production_campaign_executed"] = False
    boundary["production_trainer_callback_invoked"] = False
    boundary["training_performed"] = False
    boundary["persistent_dictionary_output_created"] = False
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
        raise ValueError("cheap-executor runbook V41 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v41",
        "status": "pass",
        "runbook_id": identifier,
        "stages": len(observed["stages"]),
        "safe_stage_now": 0,
        "semantic_dispatch_slice_v1_verified": True,
        "dispatched_nodes": 4,
        "full_reproduction_comparisons": 4,
        "execution_jobs": 34,
        "terminal_gap_count": 3,
        "production_runtime_attachment_verifier_present": False,
        "empirical_graph_present": False,
        "semantic_dispatch_allowed": False,
        "training_performed": False,
        "persistent_dictionary_output_created": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root
        / "artifacts/vpd_cheap_executor_runbook_v40/runbook.json",
        "protocol_v49_path": project_root
        / "configs/evals/vpd_activation_category_compartments_v49.json",
        "dispatcher_path": project_root
        / "experiments/authz_mechinterp/dispatch_cluster_aware_sae_semantic_slice_v1.py",
        "benchmark_path": project_root
        / "artifacts/vpd_cluster_aware_sae_semantic_dispatch_slice_benchmark_v1/report.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V41.")
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
