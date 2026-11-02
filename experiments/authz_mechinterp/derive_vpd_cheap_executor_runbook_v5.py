from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.benchmark_lora_component_basis_identifiability_v1 import verify_benchmark
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json
from experiments.authz_mechinterp.derive_unified_vpd_compartment_protocol_v13 import canonical_inputs as protocol_v13_inputs, verify_protocol as verify_protocol_v13
from experiments.authz_mechinterp.derive_vpd_cheap_executor_runbook_v4 import canonical_inputs as runbook_v4_inputs, verify_runbook as verify_runbook_v4


SCHEMA = "sch.vpd-cheap-executor-runbook.v5"


def _load(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()): raise ValueError("runbook V5 source escapes project")
    return resolved.relative_to(root.resolve()).as_posix()


def build_runbook(
    *, predecessor_path: Path, protocol_v13_path: Path,
    basis_benchmark_path: Path, project_root: Path,
) -> dict[str, Any]:
    predecessor_verification = verify_runbook_v4(predecessor_path, **runbook_v4_inputs(project_root))
    protocol_verification = verify_protocol_v13(protocol_v13_path, **protocol_v13_inputs(project_root))
    benchmark_verification = verify_benchmark(basis_benchmark_path)
    predecessor = _load(predecessor_path); protocol = _load(protocol_v13_path); benchmark = _load(basis_benchmark_path)
    contract = protocol["component_compartments"]["lora_component_basis_identifiability_v1"]
    if contract["benchmark"]["benchmark_id"] != benchmark["benchmark_id"]:
        raise ValueError("protocol V13 basis benchmark lineage differs")
    result = json.loads(json.dumps(predecessor)); result.pop("runbook_id", None)
    result["schema"] = SCHEMA
    result["status"] = "ready_for_local_preflight_canonical_production_integration_pending"
    result["predecessor"] = {
        "path": _relative(predecessor_path, project_root), "sha256": sha256_file(predecessor_path),
        "runbook_id": predecessor["runbook_id"], "verification": predecessor_verification,
    }
    result["source"]["research_protocol_v13"] = {
        "path": _relative(protocol_v13_path, project_root), "sha256": sha256_file(protocol_v13_path),
        "verification": protocol_verification,
    }
    result["source"]["lora_component_basis_identifiability_benchmark_v1"] = {
        "path": _relative(basis_benchmark_path, project_root), "sha256": sha256_file(basis_benchmark_path),
        "benchmark_id": benchmark["benchmark_id"], "verification": benchmark_verification,
    }
    result["source"]["lora_component_basis_canonicalizer_v1"] = contract["canonicalizer"]
    result["source"]["derivation_runner_sha256"] = sha256_file(Path(__file__))
    result["current_result"]["raw_rank_specific_component_count_change_under_basis_rotation"] = -16
    result["current_result"]["canonical_spectral_basis_invariance_benchmark"] = "pass"
    result["current_result"]["production_canonical_capture_integration_present"] = False
    result["executor_policy"]["never_use_raw_lora_factor_rank_as_next_empirical_component"] = True
    preflight = result["stages"][0]
    preflight["commands"].append({
        "id": "verify_lora_component_basis_identifiability_v1",
        "executable": ".venv/Scripts/python.exe",
        "arguments": ["-m", "experiments.authz_mechinterp.benchmark_lora_component_basis_identifiability_v1", "--verify", "artifacts/vpd_lora_component_basis_identifiability_benchmark_v1/report.json"],
    })
    preflight["expected"]["lora_component_basis_identifiability"] = "raw specific count 16 to 0; canonical spectral result invariant"
    preflight["stop_if"].append("protocol V13 or LoRA basis-identifiability lineage differs")
    canonical_gate = {
        "stage": 1,
        "id": "canonical_component_inventory_and_capture_runtime",
        "authorization": "implementation_only_model_free_until_real_adapter_access_is_explicitly_authorized",
        "safe_for_cheap_executor_now": False,
        "implementation_status": "thin-SVD spectral canonicalizer and same-shape multiplier math implemented; production inventory emitter and effect-capture integration absent",
        "required_component_schema": "canonical_spectral_component_or_unresolved_degenerate_block",
        "required_inputs": [
            "exact frozen adapter identity and module factor hashes",
            "per-module LoRA A and B factors plus adapter scale",
            "frozen relative spectral-gap tolerance",
        ],
        "required_checks": [
            "exact baseline delta reconstruction",
            "equivalent-factorization invariance",
            "near-degenerate singular directions grouped into unresolved blocks",
            "canonical block multiplier factor reconstruction",
            "full original factor-byte restoration after every future intervention",
        ],
        "stop_if": [
            "raw LoRA factor rank is used as the component identity",
            "production canonical inventory emitter is absent",
            "production canonical effect-capture integration is absent",
            "spectral-gap tolerance is not frozen and benchmarked",
            "any baseline delta reconstruction or restoration check fails",
        ],
    }
    for stage in result["stages"][1:]: stage["stage"] = int(stage["stage"]) + 1
    result["stages"].insert(1, canonical_gate)
    semantic = next(stage for stage in result["stages"] if stage["id"] == "cluster_candidate_semantic_strata_v3")
    semantic["additional_precondition"] = "passing canonical component inventory and capture receipt"
    semantic["stop_if"].append("candidate component inventory is raw LoRA factor-rank coordinates")
    result["completion_definition"].pop("paid_pipeline_complete_only_after_stage_8", None)
    result["completion_definition"]["paid_pipeline_complete_only_after_stage_9"] = True
    result["completion_definition"]["canonical_component_runtime_ready"] = False
    result["completion_definition"]["cheap_executor_task_complete_after_stage_0"] = True
    result["scientific_boundary"]["execution_authorized"] = False
    return {**result, "runbook_id": content_sha256(result)}


def verify_runbook(path: Path, **kwargs: Any) -> dict[str, Any]:
    observed = _load(path); body = dict(observed); identifier = body.pop("runbook_id", "")
    if identifier != content_sha256(body) or observed != build_runbook(**kwargs):
        raise ValueError("cheap-executor runbook V5 is not reproducible")
    return {
        "schema": "sch.vpd-cheap-executor-runbook-verification.v5", "status": "pass",
        "runbook_id": identifier, "stages": len(observed["stages"]), "safe_stage_now": 0,
        "canonical_component_runtime_ready": False, "execution_authorized": False,
    }


def canonical_inputs(project_root: Path) -> dict[str, Any]:
    return {
        "predecessor_path": project_root / "artifacts/vpd_cheap_executor_runbook_v4/runbook.json",
        "protocol_v13_path": project_root / "configs/evals/vpd_activation_category_compartments_v13.json",
        "basis_benchmark_path": project_root / "artifacts/vpd_lora_component_basis_identifiability_benchmark_v1/report.json",
        "project_root": project_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive or verify cheap-executor runbook V5.")
    parser.add_argument("--output", type=Path); parser.add_argument("--verify", type=Path)
    args = parser.parse_args(); inputs = canonical_inputs(Path(__file__).resolve().parents[2])
    if args.verify is not None: result = verify_runbook(args.verify, **inputs)
    else:
        if args.output is None: parser.error("--output is required unless verifying")
        result = build_runbook(**inputs); write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("schema", "status", "runbook_id")}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
