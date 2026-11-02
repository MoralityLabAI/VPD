from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.activation_categories import CATEGORY_PROTOCOLS
from experiments.authz_mechinterp.benchmark_authenticated_lora_factor_bundle_extraction_v1 import (
    _fixture_config,
    _fixture_tensors,
    _write_safetensors,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
    write_jsonl,
)
from experiments.authz_mechinterp.emit_canonical_lora_component_inventory_v1 import (
    build_inventory,
)
from experiments.authz_mechinterp.extract_authenticated_lora_factor_bundle_v1 import (
    extract_factor_bundle,
)
from experiments.authz_mechinterp.plan_canonical_vpd_capture_job_v1 import (
    AUTHORIZATION_SCHEMA,
    MODEL_IDENTITY_SCHEMA,
    build_capture_job_plan,
    verify_capture_authorization,
    verify_capture_job_plan,
)


def _attempt(function: Callable[[], Any]) -> bool:
    try:
        function()
    except (ValueError, FileNotFoundError):
        return True
    return False


def _address(value: dict[str, Any], identifier_field: str) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != identifier_field}
    return {**body, identifier_field: content_sha256(body)}


def _model_identity(path: Path, *, revision: str = "synthetic-revision-001") -> dict[str, Any]:
    body = {
        "schema": MODEL_IDENTITY_SCHEMA,
        "status": "frozen_reproducible_model_identity",
        "model_id": "synthetic/bonsai-shaped",
        "revision": revision,
        "model_files_manifest_sha256": "9" * 64,
        "loader_contract": {
            "trust_remote_code": False,
            "immutable_revision_required": True,
            "local_files_only": True,
            "generated_tokens": 0,
        },
        "scientific_boundary": {
            "model_loaded": False,
            "inference_performed": False,
            "execution_authorized": False,
        },
    }
    result = {**body, "model_identity_id": content_sha256(body)}
    write_json(path, result)
    return result


def _rows(*, omit_category: str | None = None, executed: bool = False) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for pair_index in range(2):
        pair_id = f"synthetic-pair-{pair_index:02d}"
        template_group = f"synthetic-template-{pair_index:02d}"
        for category in sorted(CATEGORY_PROTOCOLS):
            if category == omit_category:
                continue
            result.append(
                {
                    "episode_id": f"{pair_id}-{category}",
                    "pair_id": pair_id,
                    "template_group": template_group,
                    "condition": f"synthetic-{category}",
                    "activation_category": category,
                    "actual_execution": executed,
                    "execution_path_exposed": False,
                    "labels": {"candidate_effect": "none"},
                    "messages": [
                        {"role": "system", "content": "Synthetic benchmark only."},
                        {"role": "user", "content": f"Inspect {pair_id}."},
                    ],
                }
            )
    return result


def _split(root: Path, rows: list[dict[str, Any]], *, name: str) -> tuple[Path, Path]:
    split_root = root / name
    split_root.mkdir()
    dataset_path = split_root / "discovery.jsonl"
    write_jsonl(dataset_path, rows)
    episode_ids = sorted(str(row["episode_id"]) for row in rows)
    pair_ids = sorted({str(row["pair_id"]) for row in rows})
    template_groups = sorted({str(row["template_group"]) for row in rows})
    counts = {
        category: sum(row["activation_category"] == category for row in rows)
        for category in sorted({str(row["activation_category"]) for row in rows})
    }
    body = {
        "schema": "sch.vpd-activation-category-split-manifest.v3",
        "status": "complete_synthetic_semantic_split",
        "source": {"synthetic_benchmark_only": True},
        "protocol": {"semantic_disjointness_required": True},
        "invariants": {"pair_atomic": True},
        "observed": {"real_actions": 0},
        "splits": {
            "discovery": {
                "artifact": {
                    "path": dataset_path.name,
                    "sha256": sha256_file(dataset_path),
                },
                "rows": len(rows),
                "pairs": len(pair_ids),
                "activation_category_counts": counts,
                "episode_id_set_sha256": content_sha256(episode_ids),
                "pair_id_set_sha256": content_sha256(pair_ids),
                "template_group_set_sha256": content_sha256(template_groups),
            }
        },
    }
    manifest = {**body, "split_id": content_sha256(body)}
    manifest_path = split_root / "split_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path, dataset_path


def _authorization(plan: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    body = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "authorized_by_user",
        "source": {
            "capture_job_plan_id": plan["capture_job_plan_id"],
            "capture_job_plan_sha256": sha256_file(plan_path),
        },
        "scope": {
            "authorized_action": "canonical_category_effect_capture",
            "authorized_job_id": plan["capture_job_plan_id"],
            "model_inference_authorized": True,
            "temporary_adapter_parameter_mutation_authorized": True,
            "base_model_weights_mutable": False,
            "adapter_weight_persistence_authorized": False,
            "generated_tokens_authorized": False,
            "real_actions_authorized": False,
            "output_dir": plan["capture_contract"]["output_dir"],
            "maximum_forward_passes": plan["resource_ceiling"]["maximum_forward_passes"],
        },
        "synthetic_authorization_only": True,
    }
    return {**body, "capture_authorization_id": content_sha256(body)}


def build_report() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        weights = root / "adapter_model.safetensors"
        config = root / "adapter_config.json"
        _write_safetensors(weights, _fixture_tensors(), metadata={"format": "pt"})
        write_json(config, _fixture_config())
        factor_root = root / "factor-bundle"
        extract_factor_bundle(
            adapter_weights_path=weights,
            adapter_config_path=config,
            expected_adapter_sha256=sha256_file(weights),
            expected_config_sha256=sha256_file(config),
            output_dir=factor_root,
            synthetic_benchmark_only=True,
        )
        inventory_root = root / "inventory"
        inventory = build_inventory(
            factor_index_path=factor_root / "factor-index.json",
            output_dir=inventory_root,
        )
        model_path = root / "model_identity.json"
        _model_identity(model_path)
        split_path, dataset_path = _split(root, _rows(), name="split")
        inventory_rows = [
            json.loads(line)
            for line in (inventory_root / "canonical_components.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        component_ids = sorted(row["component_id"] for row in inventory_rows)[:2]
        kwargs = {
            "project_root": root,
            "model_identity_path": model_path,
            "adapter_weights_path": weights,
            "adapter_config_path": config,
            "extraction_manifest_path": factor_root / "manifest.json",
            "inventory_manifest_path": inventory_root / "manifest.json",
            "split_manifest_path": split_path,
            "split_role": "discovery",
            "dataset_path": dataset_path,
            "component_ids": component_ids,
            "adapter_name": "default",
            "activation_key": "layer_35__proposal_start",
            "epsilon": 0.05,
            "batch_size": 4,
            "device": "synthetic-cpu",
            "dtype": "float32",
            "output_dir": "captures/synthetic-canonical-v1",
            "synthetic_benchmark_only": True,
        }
        plan = build_capture_job_plan(**kwargs)
        plan_path = root / "capture_plan.json"
        write_json(plan_path, plan)
        verification = verify_capture_job_plan(plan_path, project_root=root)
        reproduced = build_capture_job_plan(**kwargs) == plan
        authorization = _authorization(plan, plan_path)
        authorization_path = root / "authorization.json"
        write_json(authorization_path, authorization)
        authorization_verification = verify_capture_authorization(
            authorization_path, plan_path=plan_path, project_root=root
        )

        tampered_weights = root / "adapter_tampered.safetensors"
        tampered_weights.write_bytes(weights.read_bytes() + b"tamper")
        adapter_tamper_rejected = _attempt(
            lambda: build_capture_job_plan(
                **{**kwargs, "adapter_weights_path": tampered_weights}
            )
        )
        raw_rank_rejected = _attempt(
            lambda: build_capture_job_plan(
                **{**kwargs, "component_ids": ["model.layers.0.self_attn.q_proj:rank00"]}
            )
        )
        bad_split, bad_dataset = _split(
            root, _rows(executed=True), name="executed-split"
        )
        action_boundary_rejected = _attempt(
            lambda: build_capture_job_plan(
                **{
                    **kwargs,
                    "split_manifest_path": bad_split,
                    "dataset_path": bad_dataset,
                }
            )
        )
        incomplete_split, incomplete_dataset = _split(
            root,
            _rows(omit_category="revoked"),
            name="incomplete-split",
        )
        category_omission_rejected = _attempt(
            lambda: build_capture_job_plan(
                **{
                    **kwargs,
                    "split_manifest_path": incomplete_split,
                    "dataset_path": incomplete_dataset,
                }
            )
        )
        mutable_model_path = root / "mutable_model.json"
        _model_identity(mutable_model_path, revision="main")
        mutable_model_rejected = _attempt(
            lambda: build_capture_job_plan(
                **{**kwargs, "model_identity_path": mutable_model_path}
            )
        )
        tampered_plan = copy.deepcopy(plan)
        tampered_plan["resource_ceiling"]["maximum_forward_passes"] += 1
        tampered_plan = _address(tampered_plan, "capture_job_plan_id")
        tampered_plan_path = root / "tampered_plan.json"
        write_json(tampered_plan_path, tampered_plan)
        readdressed_plan_tamper_rejected = _attempt(
            lambda: verify_capture_job_plan(tampered_plan_path, project_root=root)
        )
        wrong_job = copy.deepcopy(authorization)
        wrong_job["scope"]["authorized_job_id"] = "0" * 64
        wrong_job = _address(wrong_job, "capture_authorization_id")
        wrong_job_path = root / "wrong_job_authorization.json"
        write_json(wrong_job_path, wrong_job)
        wrong_job_authorization_rejected = _attempt(
            lambda: verify_capture_authorization(
                wrong_job_path, plan_path=plan_path, project_root=root
            )
        )
        persistence = copy.deepcopy(authorization)
        persistence["scope"]["adapter_weight_persistence_authorized"] = True
        persistence = _address(persistence, "capture_authorization_id")
        persistence_path = root / "persistence_authorization.json"
        write_json(persistence_path, persistence)
        persistence_authorization_rejected = _attempt(
            lambda: verify_capture_authorization(
                persistence_path, plan_path=plan_path, project_root=root
            )
        )
        synthetic_provenance_rejected = _attempt(
            lambda: build_capture_job_plan(
                **{**kwargs, "synthetic_benchmark_only": False}
            )
        )
        controls = {
            "adapter_tamper_rejected": adapter_tamper_rejected,
            "raw_rank_component_rejected": raw_rank_rejected,
            "action_boundary_rejected": action_boundary_rejected,
            "category_omission_rejected": category_omission_rejected,
            "mutable_model_revision_rejected": mutable_model_rejected,
            "readdressed_plan_tamper_rejected": readdressed_plan_tamper_rejected,
            "wrong_job_authorization_rejected": wrong_job_authorization_rejected,
            "weight_persistence_authorization_rejected": persistence_authorization_rejected,
            "synthetic_provenance_mismatch_rejected": synthetic_provenance_rejected,
        }
        if not reproduced or not all(controls.values()):
            failed = [name for name, passed in controls.items() if not passed]
            if not reproduced:
                failed.append("plan_reproduced")
            raise ValueError(
                "canonical capture-job plan benchmark control failed: "
                + ", ".join(failed)
            )
        body = {
            "schema": "sch.vpd-canonical-category-capture-job-plan-benchmark.v1",
            "status": "pass",
            "scenario": {
                "components": len(component_ids),
                "activation_categories": len(CATEGORY_PROTOCOLS),
                "rows": len(_rows()),
                "maximum_forward_passes": plan["resource_ceiling"]["maximum_forward_passes"],
                "plan_reproduced": reproduced,
                "plan_verification": verification,
                "synthetic_authorization_verification": authorization_verification,
            },
            "negative_controls": controls,
            "source": {
                "planner_sha256": sha256_file(Path(__file__).with_name("plan_canonical_vpd_capture_job_v1.py")),
                "synthetic_inventory_id": inventory["canonical_inventory_id"],
            },
            "scientific_boundary": {
                "synthetic_benchmark_only": True,
                "synthetic_authorization_is_not_user_authorization": True,
                "model_loaded": False,
                "torch_transformers_or_peft_imported": False,
                "model_inference_performed": False,
                "weights_mutated": False,
                "effect_capture_performed": False,
                "execution_authorized": False,
            },
        }
        return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("canonical capture-job plan benchmark address mismatch")
    expected = build_report()
    if observed != expected:
        raise ValueError("canonical capture-job plan benchmark is not reproducible")
    return {
        "schema": "sch.vpd-canonical-category-capture-job-plan-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "negative_controls": len(observed["negative_controls"]),
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the model-free canonical VPD capture-job planner V1."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify_benchmark(args.verify)
    else:
        if args.output is None:
            parser.error("--output is required unless verifying")
        result = build_report()
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
