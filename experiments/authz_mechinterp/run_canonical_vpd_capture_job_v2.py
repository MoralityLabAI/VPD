from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.authz_mechinterp.activation_categories import categorize_episodes
from experiments.authz_mechinterp.assemble_canonical_vpd_effect_shard_v2 import (
    assemble_canonical_effect_shard,
    verify_canonical_effect_shard,
    write_canonical_effect_shard,
)
from experiments.authz_mechinterp.canonical_vpd_category_effect_capture_v1 import (
    capture_canonical_component_effect_by_category,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.plan_canonical_vpd_capture_job_v1 import _read_jsonl
from experiments.authz_mechinterp.plan_canonical_vpd_capture_job_v2 import (
    verify_capture_authorization,
    verify_capture_job_plan,
)
from experiments.authz_mechinterp.run_canonical_vpd_capture_job_v1 import (
    CaptureRuntime,
    _load_transformer_runtime,
    _persist_component,
    _safe_output_path,
)


EXECUTION_SCHEMA = "sch.vpd-canonical-category-capture-job-execution.v2"


def preflight_capture_job(
    *, plan_path: Path, authorization_path: Path, project_root: Path
) -> dict[str, Any]:
    plan_verification = verify_capture_job_plan(plan_path, project_root=project_root)
    authorization_verification = verify_capture_authorization(
        authorization_path, plan_path=plan_path, project_root=project_root
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    output_dir = _safe_output_path(
        project_root, str(plan["capture_contract"]["output_dir"])
    )
    if output_dir.exists():
        raise ValueError("canonical capture V2 output directory already exists")
    split_role = str(plan["source"]["semantic_split"]["split_role"])
    if authorization_verification["authorized_split_role"] != split_role:
        raise ValueError("canonical capture V2 authorization split role differs")
    return {
        "schema": "sch.vpd-canonical-category-capture-job-preflight.v2",
        "status": "pass_ready_for_exact_authorized_execution_v2",
        "capture_job_plan_id": plan_verification["capture_job_plan_id"],
        "capture_authorization_id": authorization_verification[
            "capture_authorization_id"
        ],
        "split_role": split_role,
        "dataset_sha256": plan["source"]["dataset"]["sha256"],
        "components": plan_verification["components"],
        "rows": plan_verification["rows"],
        "maximum_forward_passes": plan_verification["maximum_forward_passes"],
        "output_dir": str(plan["capture_contract"]["output_dir"]),
        "sae_dictionary_training_authorized": False,
        "real_actions_authorized": False,
    }


def run_capture_job(
    *,
    plan_path: Path,
    authorization_path: Path,
    project_root: Path,
    runtime_factory: Callable[[dict[str, Any], Path], CaptureRuntime] | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    preflight = preflight_capture_job(
        plan_path=plan_path,
        authorization_path=authorization_path,
        project_root=project_root,
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    authorization = verify_capture_authorization(
        authorization_path, plan_path=plan_path, project_root=project_root
    )
    synthetic = bool(plan["scientific_boundary"]["synthetic_benchmark_only"])
    if synthetic and runtime_factory is None:
        raise ValueError("synthetic V2 capture plans require an injected fake runtime")
    if not synthetic and runtime_factory is not None:
        raise ValueError("production V2 capture plans forbid injected runtimes")
    output_dir = _safe_output_path(
        project_root, str(plan["capture_contract"]["output_dir"])
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".canonical-capture-v2-staging-", dir=output_dir.parent)
    )
    runtime: CaptureRuntime | None = None
    try:
        rows = _read_jsonl(project_root / plan["source"]["dataset"]["path"])
        rows_by_category = categorize_episodes(rows)
        batch_size = int(plan["capture_contract"]["batch_size"])
        category_batches = {
            category: [
                category_rows[start : start + batch_size]
                for start in range(0, len(category_rows), batch_size)
            ]
            for category, category_rows in rows_by_category.items()
        }
        runtime = (
            runtime_factory(plan, project_root)
            if runtime_factory is not None
            else _load_transformer_runtime(plan=plan, project_root=project_root)
        )
        if runtime.adapter_name != plan["capture_contract"]["adapter_name"]:
            raise ValueError("canonical capture V2 runtime adapter name differs")
        component_results: dict[str, dict[str, Any]] = {}
        component_artifacts: list[dict[str, Any]] = []
        component_dir = staging / "components"
        component_dir.mkdir()
        callback_calls = 0

        def counted_capture(
            category: str, batch: list[dict[str, Any]]
        ) -> np.ndarray:
            nonlocal callback_calls
            callback_calls += 1
            if callback_calls > int(preflight["maximum_forward_passes"]):
                raise ValueError("canonical capture V2 exceeded its forward-pass ceiling")
            assert runtime is not None
            return runtime.capture_batch_callback(category, batch)

        for index, component in enumerate(
            plan["capture_contract"]["component_records"]
        ):
            component_id = str(component["component_id"])
            result = capture_canonical_component_effect_by_category(
                model=runtime.model,
                module_path=str(component["module"]),
                adapter_name=runtime.adapter_name,
                canonical_artifact_path=(
                    project_root / component["canonical_module_artifact_path"]
                ),
                expected_canonical_artifact_sha256=str(
                    component["canonical_module_artifact_sha256"]
                ),
                expected_source_a_factor_sha256=str(
                    component["source_a_factor_sha256"]
                ),
                expected_source_b_factor_sha256=str(
                    component["source_b_factor_sha256"]
                ),
                block_index=int(component["canonical_block_index"]),
                category_batches=category_batches,
                capture_batch_callback=counted_capture,
                epsilon=float(plan["capture_contract"]["epsilon"]),
                no_grad_context_factory=runtime.no_grad_context_factory,
                probe_seed=int(plan["capture_contract"]["probe_seed"]),
                probe_count=int(plan["capture_contract"]["probe_count"]),
                maximum_relative_probe_error=float(
                    plan["capture_contract"]["maximum_relative_probe_error"]
                ),
            )
            component_results[component_id] = result
            component_artifacts.append(
                _persist_component(
                    index=index,
                    component_id=component_id,
                    result=result,
                    rows_by_category=rows_by_category,
                    output_dir=component_dir,
                )
            )
        if callback_calls != int(preflight["maximum_forward_passes"]):
            raise ValueError("canonical capture V2 forward-pass count differs from plan")
        split = plan["source"]["semantic_split"]
        split_provenance = {
            "activation_category_split_schema": split["split_schema"],
            "activation_category_split_role": split["split_role"],
            "activation_category_split_verification_status": "pass",
            "activation_category_split_semantic_disjointness": "pass",
            "activation_category_split_id": split["split_id"],
            "activation_category_split_manifest_sha256": split[
                "split_manifest_sha256"
            ],
            "activation_category_split_episode_id_set_sha256": split[
                "episode_id_set_sha256"
            ],
            "activation_category_split_pair_id_set_sha256": split[
                "pair_id_set_sha256"
            ],
            "activation_category_split_template_group_set_sha256": split[
                "template_group_set_sha256"
            ],
        }
        sketch = plan["capture_contract"]["semantic_sketch"]
        arrays, assembly_receipt = assemble_canonical_effect_shard(
            episodes_by_category=rows_by_category,
            component_results=component_results,
            dataset_sha256=plan["source"]["dataset"]["sha256"],
            split_provenance=split_provenance,
            requested_uncertainty_blocks=int(sketch["uncertainty_blocks"]),
            countsketch_dimension=int(sketch["countsketch_dimension"]),
            countsketch_seed=int(sketch["countsketch_seed"]),
        )
        shard_path = staging / "canonical_effect_shard.npz"
        shard_receipt_path = staging / "canonical_effect_shard_receipt.json"
        shard_receipt = write_canonical_effect_shard(
            output_path=shard_path,
            receipt_path=shard_receipt_path,
            arrays=arrays,
            assembly_receipt=assembly_receipt,
        )
        split_role = str(split["split_role"])
        execution_body = {
            "schema": EXECUTION_SCHEMA,
            "status": "complete_exact_authorized_canonical_category_capture_v2",
            "capture_job_plan": {
                "sha256": sha256_file(plan_path),
                "capture_job_plan_id": plan["capture_job_plan_id"],
                "split_role": split_role,
                "dataset_sha256": plan["source"]["dataset"]["sha256"],
            },
            "capture_authorization": {
                "sha256": sha256_file(authorization_path),
                "capture_authorization_id": authorization[
                    "capture_authorization_id"
                ],
                "authorized_split_role": authorization["authorized_split_role"],
            },
            "runtime": {
                "kind": runtime.runtime_kind,
                "actual_model_loaded": runtime.actual_model_loaded,
                "model_id": plan["source"]["model_identity"]["model_id"],
                "revision": plan["source"]["model_identity"]["revision"],
                "adapter_sha256": plan["source"]["adapter_weights"]["sha256"],
                "activation_key": plan["capture_contract"]["activation_key"],
                "forward_passes": callback_calls,
                "generated_tokens": 0,
            },
            "artifacts": {
                "components": component_artifacts,
                "effect_shard": {
                    "path": shard_path.name,
                    "sha256": sha256_file(shard_path),
                    "receipt_path": shard_receipt_path.name,
                    "receipt_sha256": sha256_file(shard_receipt_path),
                    "canonical_effect_shard_artifact_id": shard_receipt[
                        "canonical_effect_shard_artifact_id"
                    ],
                },
            },
            "scientific_boundary": {
                "synthetic_benchmark_only": synthetic,
                "model_inference_performed": runtime.actual_model_loaded,
                "capture_callback_executed": True,
                "temporary_adapter_parameter_mutation_restored": True,
                "base_model_weights_mutated": False,
                "adapter_weights_persisted": False,
                "generated_tokens": 0,
                "sae_training_role_effect_capture_only": split_role == "sae_training",
                "sae_dictionary_training_authorized": False,
                "real_actions_authorized": False,
                "authorization_control_release_authorized": False,
                "sae_training_authorized": False,
            },
        }
        execution = {
            **execution_body,
            "capture_execution_id": content_sha256(execution_body),
        }
        write_json(staging / "capture_execution_receipt.json", execution)
        staging.replace(output_dir)
        return execution
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        if runtime is not None:
            runtime.close()


def verify_capture_execution_receipt(
    receipt_path: Path,
    *,
    plan_path: Path,
    authorization_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    plan_check = verify_capture_job_plan(plan_path, project_root=project_root)
    auth_check = verify_capture_authorization(
        authorization_path, plan_path=plan_path, project_root=project_root
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    expected_output = _safe_output_path(
        project_root, str(plan["capture_contract"]["output_dir"])
    )
    if receipt_path.parent.resolve() != expected_output:
        raise ValueError("canonical capture V2 execution receipt is outside planned output")
    if receipt.get("schema") != EXECUTION_SCHEMA:
        raise ValueError("unsupported canonical capture V2 execution receipt")
    body = dict(receipt)
    identifier = str(body.pop("capture_execution_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("canonical capture V2 execution receipt address mismatch")
    split = plan["source"]["semantic_split"]
    if (
        receipt["capture_job_plan"]["capture_job_plan_id"]
        != plan_check["capture_job_plan_id"]
        or receipt["capture_job_plan"]["sha256"] != sha256_file(plan_path)
        or receipt["capture_job_plan"]["split_role"] != split["split_role"]
        or receipt["capture_job_plan"]["dataset_sha256"]
        != plan["source"]["dataset"]["sha256"]
        or receipt["capture_authorization"]["capture_authorization_id"]
        != auth_check["capture_authorization_id"]
        or receipt["capture_authorization"]["sha256"]
        != sha256_file(authorization_path)
        or receipt["capture_authorization"]["authorized_split_role"]
        != split["split_role"]
    ):
        raise ValueError("canonical capture V2 execution source differs")
    if receipt["runtime"]["forward_passes"] != plan_check["maximum_forward_passes"]:
        raise ValueError("canonical capture V2 execution forward-pass count differs")
    root = receipt_path.parent
    if len(receipt["artifacts"]["components"]) != plan_check["components"]:
        raise ValueError("canonical capture V2 component artifact count differs")
    for component in receipt["artifacts"]["components"]:
        if sha256_file(root / component["path"]) != component["sha256"]:
            raise ValueError("canonical capture V2 component artifact hash differs")
        capture_body = dict(component["capture_receipt"])
        capture_id = str(
            capture_body.pop("canonical_category_effect_capture_id", "")
        )
        if capture_id != content_sha256(capture_body):
            raise ValueError("canonical capture V2 component receipt address differs")
    shard = receipt["artifacts"]["effect_shard"]
    shard_path = root / shard["path"]
    shard_receipt_path = root / shard["receipt_path"]
    if (
        sha256_file(shard_path) != shard["sha256"]
        or sha256_file(shard_receipt_path) != shard["receipt_sha256"]
    ):
        raise ValueError("canonical capture V2 effect shard hash differs")
    shard_check = verify_canonical_effect_shard(shard_receipt_path)
    if (
        shard_check["canonical_effect_shard_artifact_id"]
        != shard["canonical_effect_shard_artifact_id"]
    ):
        raise ValueError("canonical capture V2 effect shard receipt differs")
    boundary = receipt.get("scientific_boundary", {})
    for field in (
        "base_model_weights_mutated",
        "adapter_weights_persisted",
        "sae_dictionary_training_authorized",
        "real_actions_authorized",
        "authorization_control_release_authorized",
        "sae_training_authorized",
    ):
        if boundary.get(field) is not False:
            raise ValueError(f"canonical capture V2 boundary differs: {field}")
    if split["split_role"] == "sae_training" and boundary.get(
        "sae_training_role_effect_capture_only"
    ) is not True:
        raise ValueError("canonical capture V2 SAE-training role boundary differs")
    return {
        "schema": "sch.vpd-canonical-category-capture-job-execution-verification.v2",
        "status": "pass",
        "capture_execution_id": identifier,
        "capture_job_plan_id": plan_check["capture_job_plan_id"],
        "capture_authorization_id": auth_check["capture_authorization_id"],
        "split_role": split["split_role"],
        "components": len(receipt["artifacts"]["components"]),
        "effect_shard_artifact_id": shard_check[
            "canonical_effect_shard_artifact_id"
        ],
        "sae_dictionary_training_authorized": False,
        "real_actions_authorized": False,
        "release_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight, execute, or verify a canonical VPD capture job V2."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify-receipt", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.preflight:
        result = preflight_capture_job(
            plan_path=args.plan,
            authorization_path=args.authorization,
            project_root=root,
        )
    elif args.execute:
        result = run_capture_job(
            plan_path=args.plan,
            authorization_path=args.authorization,
            project_root=root,
        )
    else:
        result = verify_capture_execution_receipt(
            args.verify_receipt,
            plan_path=args.plan,
            authorization_path=args.authorization,
            project_root=root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
