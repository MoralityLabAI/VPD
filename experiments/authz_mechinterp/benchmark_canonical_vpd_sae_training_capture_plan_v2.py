from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.benchmark_authenticated_lora_factor_bundle_extraction_v1 import (
    _fixture_config,
    _fixture_tensors,
    _write_safetensors,
)
from experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_plan_v1 import (
    _model_identity,
    _rows,
    _split,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.emit_canonical_lora_component_inventory_v1 import (
    build_inventory,
)
from experiments.authz_mechinterp.extract_authenticated_lora_factor_bundle_v1 import (
    extract_factor_bundle,
)
from experiments.authz_mechinterp.plan_canonical_vpd_capture_job_v1 import (
    build_capture_job_plan as build_v1_plan,
)
from experiments.authz_mechinterp.plan_canonical_vpd_capture_job_v2 import (
    AUTHORIZATION_SCHEMA,
    build_capture_job_plan,
    verify_capture_authorization,
    verify_capture_job_plan,
)


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (ValueError, FileNotFoundError):
        return True
    return False


def _rewrite_split_role(split_path: Path, *, role: str) -> None:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    original = split["splits"]["discovery"]
    body = {key: value for key, value in split.items() if key != "split_id"}
    body["splits"] = {role: original}
    write_json(split_path, {**body, "split_id": content_sha256(body)})


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
            "authorized_split_role": "sae_training",
            "authorized_dataset_sha256": plan["source"]["dataset"]["sha256"],
            "model_inference_authorized": True,
            "temporary_adapter_parameter_mutation_authorized": True,
            "base_model_weights_mutable": False,
            "adapter_weight_persistence_authorized": False,
            "sae_dictionary_training_authorized": False,
            "generated_tokens_authorized": False,
            "real_actions_authorized": False,
            "output_dir": plan["capture_contract"]["output_dir"],
            "maximum_forward_passes": plan["resource_ceiling"][
                "maximum_forward_passes"
            ],
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
        _rewrite_split_role(split_path, role="sae_training")
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
            "split_role": "sae_training",
            "dataset_path": dataset_path,
            "component_ids": component_ids,
            "adapter_name": "default",
            "activation_key": "layer_35__proposal_start",
            "epsilon": 0.05,
            "batch_size": 4,
            "device": "synthetic-cpu",
            "dtype": "float32",
            "output_dir": "captures/synthetic-canonical-sae-training-v2",
            "synthetic_benchmark_only": True,
        }
        plan = build_capture_job_plan(**kwargs)
        plan_path = root / "capture_plan_v2.json"
        write_json(plan_path, plan)
        plan_verification = verify_capture_job_plan(plan_path, project_root=root)
        authorization = _authorization(plan, plan_path)
        authorization_path = root / "authorization_v2.json"
        write_json(authorization_path, authorization)
        authorization_verification = verify_capture_authorization(
            authorization_path, plan_path=plan_path, project_root=root
        )

        wrong_role_authorization = copy.deepcopy(authorization)
        wrong_role_authorization["scope"]["authorized_split_role"] = "discovery"
        wrong_role_body = {
            key: value
            for key, value in wrong_role_authorization.items()
            if key != "capture_authorization_id"
        }
        wrong_role_authorization["capture_authorization_id"] = content_sha256(
            wrong_role_body
        )
        wrong_role_path = root / "wrong_role_authorization.json"
        write_json(wrong_role_path, wrong_role_authorization)

        training_authorization = copy.deepcopy(authorization)
        training_authorization["scope"]["sae_dictionary_training_authorized"] = True
        training_body = {
            key: value
            for key, value in training_authorization.items()
            if key != "capture_authorization_id"
        }
        training_authorization["capture_authorization_id"] = content_sha256(
            training_body
        )
        training_path = root / "training_authorization.json"
        write_json(training_path, training_authorization)

        tampered_split = json.loads(split_path.read_text(encoding="utf-8"))
        tampered_split["splits"]["sae_training"]["rows"] += 1
        tampered_body = {
            key: value for key, value in tampered_split.items() if key != "split_id"
        }
        tampered_split["split_id"] = content_sha256(tampered_body)
        tampered_split_path = root / "tampered_split.json"
        write_json(tampered_split_path, tampered_split)

        expected_batches_per_lane = sum(
            (count + kwargs["batch_size"] - 1) // kwargs["batch_size"]
            for count in plan["capture_contract"]["category_row_counts"].values()
        )
        expected_passes = 2 * len(component_ids) * expected_batches_per_lane
        controls = {
            "v1_role_restriction_preserved": _rejects(
                lambda: build_v1_plan(**kwargs)
            ),
            "v2_sae_training_plan_reproduces": plan_verification["status"] == "pass"
            and build_capture_job_plan(**kwargs) == plan,
            "split_role_exactly_bound": plan["source"]["semantic_split"][
                "split_role"
            ]
            == "sae_training"
            and plan["policy"]["planned_split_role"] == "sae_training",
            "dataset_and_split_hashes_bound": plan["source"]["dataset"]["sha256"]
            == sha256_file(dataset_path)
            and plan["source"]["semantic_split"]["split_manifest_sha256"]
            == sha256_file(split_path),
            "forward_pass_ceiling_exact": plan["resource_ceiling"][
                "maximum_forward_passes"
            ]
            == expected_passes,
            "authorization_binds_role_dataset_and_ceiling": (
                authorization_verification["status"] == "pass"
                and authorization_verification["authorized_split_role"]
                == "sae_training"
                and authorization_verification["maximum_forward_passes"]
                == expected_passes
            ),
            "authorization_does_not_grant_dictionary_training": (
                authorization_verification["sae_dictionary_training_authorized"]
                is False
                and plan["scientific_boundary"]["sae_training_authorized"] is False
            ),
            "wrong_role_authorization_rejected": _rejects(
                lambda: verify_capture_authorization(
                    wrong_role_path, plan_path=plan_path, project_root=root
                )
            ),
            "dictionary_training_authorization_rejected": _rejects(
                lambda: verify_capture_authorization(
                    training_path, plan_path=plan_path, project_root=root
                )
            ),
            "split_support_tamper_rejected": _rejects(
                lambda: build_capture_job_plan(
                    **{**kwargs, "split_manifest_path": tampered_split_path}
                )
            ),
            "planner_imports_no_model_framework": all(
                name not in Path(__file__).with_name(
                    "plan_canonical_vpd_capture_job_v2.py"
                ).read_text(encoding="utf-8")
                for name in ("import torch", "import transformers", "import peft")
            ),
            "planning_performs_no_execution": plan["scientific_boundary"][
                "model_loaded"
            ]
            is False
            and plan["scientific_boundary"]["effect_capture_performed"] is False
            and plan["scientific_boundary"]["execution_authorized"] is False,
        }
        body = {
            "schema": "sch.vpd-canonical-sae-training-capture-plan-benchmark.v2",
            "status": "pass" if all(controls.values()) else "fail",
            "source": {
                "runner_sha256": sha256_file(Path(__file__)),
                "planner_v2_sha256": sha256_file(
                    Path(__file__).with_name("plan_canonical_vpd_capture_job_v2.py")
                ),
                "planner_v1_sha256": plan["predecessor_contract"][
                    "planner_v1_sha256"
                ],
            },
            "design": {
                "synthetic_sae_training_split": True,
                "canonical_components": len(component_ids),
                "all_activation_categories": True,
                "exact_shared_pair_set": True,
                "central_difference_lanes": 2,
                "authorization_is_exact_job_scoped": True,
                "dictionary_training_authority_separate": True,
            },
            "results": {
                "capture_job_plan_id": plan["capture_job_plan_id"],
                "capture_authorization_id": authorization[
                    "capture_authorization_id"
                ],
                "split_role": "sae_training",
                "dataset_rows": plan["resource_ceiling"]["dataset_rows"],
                "maximum_forward_passes": expected_passes,
                "canonical_inventory_id": inventory["canonical_inventory_id"],
            },
            "controls": controls,
            "summary": {
                "controls": len(controls),
                "passed": sum(bool(value) for value in controls.values()),
            },
            "scientific_boundary": {
                "synthetic_artifacts_only": True,
                "model_loaded": False,
                "model_inference_performed": False,
                "weights_loaded_or_mutated": False,
                "effect_capture_performed": False,
                "empirical_plan_produced": False,
                "sae_training_authorized": False,
                "execution_authorized": False,
            },
        }
        return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("SAE-training capture-plan benchmark address differs")
    if observed != build_report():
        raise ValueError("SAE-training capture-plan benchmark is not reproducible")
    if observed.get("status") != "pass":
        raise ValueError("SAE-training capture-plan benchmark did not pass")
    return {
        "schema": "sch.vpd-canonical-sae-training-capture-plan-benchmark-verification.v2",
        "status": "pass",
        "benchmark_id": identifier,
        "controls": observed["summary"]["controls"],
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark canonical SAE-training capture planning V2."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        result = build_report()
        write_json(args.output, result)
    else:
        result = verify_benchmark(args.verify)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
