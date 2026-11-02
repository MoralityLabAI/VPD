from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager

import numpy as np

from experiments.authz_mechinterp.activation_categories import (
    activation_category,
    categorize_episodes,
)
from experiments.authz_mechinterp.assemble_canonical_vpd_effect_shard_v1 import (
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
from experiments.authz_mechinterp.merge_vpd_effect_shards import (
    write_deterministic_npz,
)
from experiments.authz_mechinterp.plan_canonical_vpd_capture_job_v1 import (
    _read_jsonl,
    verify_capture_authorization,
    verify_capture_job_plan,
)


EXECUTION_SCHEMA = "sch.vpd-canonical-category-capture-job-execution.v1"


@dataclass
class CaptureRuntime:
    model: Any
    adapter_name: str
    capture_batch_callback: Callable[[str, list[dict[str, Any]]], np.ndarray]
    no_grad_context_factory: Callable[[], ContextManager[Any]]
    close: Callable[[], None]
    runtime_kind: str
    actual_model_loaded: bool


def _safe_output_path(project_root: Path, relative_path: str) -> Path:
    value = Path(relative_path)
    if not relative_path or value.is_absolute() or ".." in value.parts:
        raise ValueError("canonical capture output path is unsafe")
    result = (project_root / value).resolve()
    try:
        result.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError("canonical capture output escapes project root") from exc
    return result


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
        raise ValueError("canonical capture output directory already exists")
    return {
        "schema": "sch.vpd-canonical-category-capture-job-preflight.v1",
        "status": "pass_ready_for_exact_authorized_execution",
        "capture_job_plan_id": plan_verification["capture_job_plan_id"],
        "capture_authorization_id": authorization_verification[
            "capture_authorization_id"
        ],
        "components": plan_verification["components"],
        "rows": plan_verification["rows"],
        "maximum_forward_passes": plan_verification["maximum_forward_passes"],
        "output_dir": str(plan["capture_contract"]["output_dir"]),
        "real_actions_authorized": False,
    }


def _load_transformer_runtime(
    *, plan: dict[str, Any], project_root: Path
) -> CaptureRuntime:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ValueError(
            "authorized capture requires Torch, Transformers, and PEFT"
        ) from exc
    from experiments.authz_mechinterp.capture_activations import (
        discover_layers,
        extract_hidden,
        render_prompt,
    )

    source = plan["source"]
    contract = plan["capture_contract"]
    identity = source["model_identity"]
    loader = identity["loader_contract"]
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[contract["dtype"]]
    common = {
        "revision": identity["revision"],
        "trust_remote_code": False,
        "local_files_only": bool(loader["local_files_only"]),
    }
    tokenizer = AutoTokenizer.from_pretrained(identity["model_id"], **common)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("canonical capture tokenizer lacks pad and EOS tokens")
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        identity["model_id"], torch_dtype=dtype, **common
    )
    adapter_dir = (project_root / source["adapter_weights"]["path"]).resolve().parent
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        adapter_name=contract["adapter_name"],
        is_trainable=False,
    )
    model.to(contract["device"])
    model.eval()
    layers = discover_layers(model)
    layer_index = int(contract["layer"])
    if layer_index >= len(layers):
        raise ValueError("canonical capture layer is absent from model")
    layer = layers[layer_index]

    def capture_batch(
        category: str, batch: list[dict[str, Any]]
    ) -> np.ndarray:
        if not batch or any(activation_category(row) != category for row in batch):
            raise ValueError("canonical capture runtime batch category differs")
        prompts = [render_prompt(tokenizer, row["messages"]) for row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        if "attention_mask" not in encoded:
            raise ValueError("canonical capture tokenizer omitted attention mask")
        encoded = {
            key: value.to(contract["device"]) for key, value in encoded.items()
        }
        captured: dict[str, Any] = {}

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if "hidden" in captured:
                raise ValueError("canonical capture layer hook fired repeatedly")
            captured["hidden"] = extract_hidden(output)

        handle = layer.register_forward_hook(hook)
        try:
            model(**encoded, use_cache=False)
        finally:
            handle.remove()
        hidden = captured.get("hidden")
        if hidden is None:
            raise ValueError("canonical capture layer hook did not fire")
        rows: list[Any] = []
        for row_index in range(encoded["attention_mask"].shape[0]):
            positions = torch.nonzero(
                encoded["attention_mask"][row_index], as_tuple=False
            ).flatten()
            if positions.numel() < 1:
                raise ValueError("canonical capture prompt contains no tokens")
            rows.append(hidden[row_index, int(positions[-1])])
        return torch.stack(rows).detach().float().cpu().numpy().astype(np.float32)

    def close() -> None:
        nonlocal model, base_model, tokenizer
        model = None
        base_model = None
        tokenizer = None
        if str(contract["device"]).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return CaptureRuntime(
        model=model,
        adapter_name=str(contract["adapter_name"]),
        capture_batch_callback=capture_batch,
        no_grad_context_factory=torch.no_grad,
        close=close,
        runtime_kind="torch_transformers_peft_exact_plan",
        actual_model_loaded=True,
    )


def _persist_component(
    *,
    index: int,
    component_id: str,
    result: dict[str, Any],
    rows_by_category: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> dict[str, Any]:
    path = output_dir / f"component-{index:04d}.npz"
    arrays: dict[str, np.ndarray] = {
        f"delta_samples__{category}": np.asarray(values, dtype=np.float32)
        for category, values in result["delta_samples_by_category"].items()
    }
    arrays.update(
        {
            "category_names": np.asarray(result["category_names"]),
            "category_row_counts": np.asarray(
                result["category_row_counts"], dtype=np.int64
            ),
            "delta_mean_by_category": np.asarray(
                result["delta_mean_by_category"], dtype=np.float32
            ),
            "delta_global_mean": np.asarray(
                result["delta_global_mean"], dtype=np.float32
            ),
        }
    )
    write_deterministic_npz(path, arrays)
    return {
        "component_id": component_id,
        "path": f"components/{path.name}",
        "sha256": sha256_file(path),
        "arrays": sorted(arrays),
        "episode_ids_by_category_sha256": {
            category: content_sha256(
                [str(row["episode_id"]) for row in rows_by_category[category]]
            )
            for category in sorted(rows_by_category)
        },
        "capture_receipt": result["receipt"],
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
        raise ValueError("synthetic plans require an injected fake runtime")
    if not synthetic and runtime_factory is not None:
        raise ValueError("production plans forbid injected runtimes")
    output_dir = _safe_output_path(
        project_root, str(plan["capture_contract"]["output_dir"])
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".canonical-capture-staging-", dir=output_dir.parent)
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
            raise ValueError("canonical capture runtime adapter name differs")
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
                raise ValueError("canonical capture exceeded its forward-pass ceiling")
            return runtime.capture_batch_callback(category, batch)

        for index, component in enumerate(
            plan["capture_contract"]["component_records"]
        ):
            component_id = str(component["component_id"])
            result = capture_canonical_component_effect_by_category(
                model=runtime.model,
                module_path=str(component["module"]),
                adapter_name=runtime.adapter_name,
                canonical_artifact_path=project_root
                / component["canonical_module_artifact_path"],
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
            raise ValueError("canonical capture forward-pass count differs from plan")
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
        execution_body = {
            "schema": EXECUTION_SCHEMA,
            "status": "complete_exact_authorized_canonical_category_capture",
            "capture_job_plan": {
                "sha256": sha256_file(plan_path),
                "capture_job_plan_id": plan["capture_job_plan_id"],
            },
            "capture_authorization": {
                "sha256": sha256_file(authorization_path),
                "capture_authorization_id": authorization[
                    "capture_authorization_id"
                ],
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
        raise ValueError("canonical capture execution receipt is outside planned output")
    if receipt.get("schema") != EXECUTION_SCHEMA:
        raise ValueError("unsupported canonical capture execution receipt")
    body = dict(receipt)
    identifier = str(body.pop("capture_execution_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("canonical capture execution receipt address mismatch")
    if (
        receipt["capture_job_plan"]["capture_job_plan_id"]
        != plan_check["capture_job_plan_id"]
        or receipt["capture_job_plan"]["sha256"] != sha256_file(plan_path)
        or receipt["capture_authorization"]["capture_authorization_id"]
        != auth_check["capture_authorization_id"]
        or receipt["capture_authorization"]["sha256"]
        != sha256_file(authorization_path)
    ):
        raise ValueError("canonical capture execution source differs")
    if receipt["runtime"]["forward_passes"] != plan_check["maximum_forward_passes"]:
        raise ValueError("canonical capture execution forward-pass count differs")
    root = receipt_path.parent
    for component in receipt["artifacts"]["components"]:
        if sha256_file(root / component["path"]) != component["sha256"]:
            raise ValueError("canonical component artifact hash differs")
        capture_receipt = component["capture_receipt"]
        capture_body = dict(capture_receipt)
        capture_id = str(
            capture_body.pop("canonical_category_effect_capture_id", "")
        )
        if capture_id != content_sha256(capture_body):
            raise ValueError("canonical category capture receipt address differs")
    shard = receipt["artifacts"]["effect_shard"]
    shard_path = root / shard["path"]
    shard_receipt_path = root / shard["receipt_path"]
    if (
        sha256_file(shard_path) != shard["sha256"]
        or sha256_file(shard_receipt_path) != shard["receipt_sha256"]
    ):
        raise ValueError("canonical effect shard hash differs")
    shard_check = verify_canonical_effect_shard(shard_receipt_path)
    if (
        shard_check["canonical_effect_shard_artifact_id"]
        != shard["canonical_effect_shard_artifact_id"]
    ):
        raise ValueError("canonical effect shard receipt ID differs")
    boundary = receipt.get("scientific_boundary", {})
    for field in (
        "base_model_weights_mutated",
        "adapter_weights_persisted",
        "real_actions_authorized",
        "authorization_control_release_authorized",
        "sae_training_authorized",
    ):
        if boundary.get(field) is not False:
            raise ValueError(f"canonical capture execution boundary differs: {field}")
    return {
        "schema": "sch.vpd-canonical-category-capture-job-execution-verification.v1",
        "status": "pass",
        "capture_execution_id": identifier,
        "capture_job_plan_id": plan_check["capture_job_plan_id"],
        "capture_authorization_id": auth_check["capture_authorization_id"],
        "components": len(receipt["artifacts"]["components"]),
        "effect_shard_artifact_id": shard_check[
            "canonical_effect_shard_artifact_id"
        ],
        "real_actions_authorized": False,
        "release_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight, execute, or verify a canonical VPD capture job V1."
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
