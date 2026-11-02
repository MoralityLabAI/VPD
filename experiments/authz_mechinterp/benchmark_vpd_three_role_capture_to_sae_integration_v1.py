from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.authz_mechinterp.activation_categories import CATEGORY_PROTOCOLS
from experiments.authz_mechinterp.benchmark_authenticated_lora_factor_bundle_extraction_v1 import (
    _fixture_config,
    _fixture_tensors,
    _write_safetensors,
)
from experiments.authz_mechinterp.benchmark_canonical_lora_peft_runtime_v1 import (
    _factor_arrays,
)
from experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_plan_v1 import (
    _authorization as _authorization_v1,
    _model_identity,
    _split,
)
from experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_runtime_v1 import (
    _fake_runtime_factory,
)
from experiments.authz_mechinterp.benchmark_canonical_vpd_sae_training_capture_plan_v2 import (
    _authorization as _authorization_v2,
    _rewrite_split_role,
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
from experiments.authz_mechinterp.load_vpd_capture_router_inputs_v1 import (
    load_capture_router_inputs_v1,
    verify_capture_router_adapter_receipt_v1,
)
from experiments.authz_mechinterp.plan_canonical_vpd_capture_job_v1 import (
    build_capture_job_plan as build_capture_plan_v1,
)
from experiments.authz_mechinterp.plan_canonical_vpd_capture_job_v2 import (
    build_capture_job_plan as build_capture_plan_v2,
)
from experiments.authz_mechinterp.plan_cluster_aware_sae_training_v1 import (
    build_cluster_aware_sae_plan_v1,
)
from experiments.authz_mechinterp.plan_cluster_aware_sae_training_v2 import (
    build_cluster_aware_sae_plan_v2,
    verify_cluster_aware_sae_plan_v2,
)
from experiments.authz_mechinterp.route_vpd_cluster_episodes_v1 import (
    _canonical_groups,
    assess_routing_stability_v1,
    route_cluster_episodes_v1,
    verify_routing_receipt_v1,
    verify_routing_stability_v1,
)
from experiments.authz_mechinterp.run_canonical_vpd_capture_job_v1 import (
    CaptureRuntime,
    run_capture_job as run_capture_v1,
)
from experiments.authz_mechinterp.run_canonical_vpd_capture_job_v2 import (
    run_capture_job as run_capture_v2,
)


ROLES = ("discovery", "cluster_validation", "sae_training")


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (ValueError, FileNotFoundError, RuntimeError, KeyError):
        return True
    return False


def _rows(role: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_index in range(10):
        if pair_index < 4:
            route_group: int | str = 0
        elif pair_index < 8:
            route_group = 1
        else:
            route_group = "shared"
        pair_id = f"{role}-pair-{pair_index:02d}"
        for category in sorted(CATEGORY_PROTOCOLS):
            rows.append(
                {
                    "episode_id": f"{pair_id}-{category}",
                    "pair_id": pair_id,
                    "template_group": f"{role}-template-{pair_index % 2}",
                    "condition": f"condition-{pair_index % 2}",
                    "activation_category": category,
                    "synthetic_route_group": route_group,
                    "actual_execution": False,
                    "execution_path_exposed": False,
                    "labels": {"candidate_effect": "none"},
                    "messages": [
                        {"role": "system", "content": "Synthetic integration only."},
                        {"role": "user", "content": f"Inspect {pair_id}."},
                    ],
                }
            )
    return rows


def _runtime_factory(
    inventory_rows: list[dict[str, Any]],
) -> tuple[Callable[[dict[str, Any], Path], CaptureRuntime], dict[str, Any]]:
    base_factory, state = _fake_runtime_factory()
    module_names = sorted({str(row["module"]) for row in inventory_rows})
    norm_by_module = {
        module: float(
            np.sqrt(
                np.mean(
                    [
                        float(row["component_fro_norm"]) ** 2
                        for row in inventory_rows
                        if str(row["module"]) == module
                    ]
                )
            )
        )
        for module in module_names
    }
    if len(module_names) != 2 or any(value <= 0.0 for value in norm_by_module.values()):
        raise ValueError("three-role integration fixture requires two nonzero modules")

    def factory(plan: dict[str, Any], project_root: Path) -> CaptureRuntime:
        runtime = base_factory(plan, project_root)
        if len(state["modules"]) != len(module_names):
            raise ValueError("three-role integration runtime module inventory differs")
        flattened_sizes = [
            int(np.prod(_factor_arrays(module)[1].shape[:-1]))
            * int(_factor_arrays(module)[0].shape[1])
            for module in state["modules"]
        ]
        flattened_sizes = [
            int((float(module.scaling["default"]) * (b_factor @ a_factor)).size)
            for module in state["modules"]
            for a_factor, b_factor in [_factor_arrays(module)]
        ]
        offsets = np.cumsum([0, *flattened_sizes])

        def capture_batch(category: str, batch: list[dict[str, Any]]) -> np.ndarray:
            values: list[np.ndarray] = []
            for row in batch:
                route_group = row["synthetic_route_group"]
                hidden = np.zeros(int(offsets[-1]), dtype=np.float64)
                for module_index, (module_name, module) in enumerate(
                    zip(module_names, state["modules"], strict=True)
                ):
                    a_factor, b_factor = _factor_arrays(module)
                    delta = (
                        float(module.scaling["default"]) * (b_factor @ a_factor)
                    ).ravel()
                    if route_group == "shared":
                        relative_weight = 1.0
                    elif int(route_group) == module_index:
                        relative_weight = 20.0
                    else:
                        relative_weight = 0.01
                    weight = relative_weight / norm_by_module[module_name]
                    hidden[offsets[module_index] : offsets[module_index + 1]] = (
                        weight * delta
                    )
                values.append(hidden)
            return np.asarray(values, dtype=np.float32)

        runtime.capture_batch_callback = capture_batch
        return runtime

    return factory, state


def _build_assets(root: Path) -> dict[str, Any]:
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
    build_inventory(
        factor_index_path=factor_root / "factor-index.json",
        output_dir=inventory_root,
    )
    _model_identity(root / "model_identity.json")
    inventory_rows = [
        json.loads(line)
        for line in (inventory_root / "canonical_components.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    membership = {
        str(row["component_id"]): str(row["module"]) for row in inventory_rows
    }
    groups, _ = _canonical_groups(membership)
    return {
        "weights": weights,
        "config": config,
        "factor_root": factor_root,
        "inventory_root": inventory_root,
        "inventory_rows": inventory_rows,
        "component_ids": sorted(membership),
        "membership": membership,
        "partition_sha256": content_sha256(groups),
    }


def _capture_role(root: Path, assets: dict[str, Any], role: str) -> dict[str, Any]:
    split_path, dataset_path = _split(root, _rows(role), name=f"split-{role}")
    if role != "discovery":
        _rewrite_split_role(split_path, role=role)
    kwargs = {
        "project_root": root,
        "model_identity_path": root / "model_identity.json",
        "adapter_weights_path": assets["weights"],
        "adapter_config_path": assets["config"],
        "extraction_manifest_path": assets["factor_root"] / "manifest.json",
        "inventory_manifest_path": assets["inventory_root"] / "manifest.json",
        "split_manifest_path": split_path,
        "split_role": role,
        "dataset_path": dataset_path,
        "component_ids": assets["component_ids"],
        "adapter_name": "default",
        "activation_key": "layer_35__proposal_start",
        "epsilon": 0.05,
        "batch_size": 5,
        "device": "synthetic-cpu",
        "dtype": "float32",
        "output_dir": f"captures/integration-{role}",
        "synthetic_benchmark_only": True,
    }
    if role == "sae_training":
        plan = build_capture_plan_v2(**kwargs)
        runner = run_capture_v2
        authorization_builder = _authorization_v2
    else:
        plan = build_capture_plan_v1(**kwargs)
        runner = run_capture_v1
        authorization_builder = _authorization_v1
    plan_path = root / f"capture-plan-{role}.json"
    write_json(plan_path, plan)
    authorization_path = root / f"capture-authorization-{role}.json"
    write_json(authorization_path, authorization_builder(plan, plan_path))
    factory, state = _runtime_factory(assets["inventory_rows"])
    execution = runner(
        plan_path=plan_path,
        authorization_path=authorization_path,
        project_root=root,
        runtime_factory=factory,
    )
    receipt_path = (
        root / plan["capture_contract"]["output_dir"] / "capture_execution_receipt.json"
    )
    return {
        "plan": plan,
        "plan_path": plan_path,
        "authorization_path": authorization_path,
        "execution": execution,
        "receipt_path": receipt_path,
        "runtime_state": state,
    }


def build_report(project_root: Path) -> dict[str, Any]:
    torch_absent_before = "torch" not in sys.modules
    transformers_absent_before = "transformers" not in sys.modules
    peft_absent_before = "peft" not in sys.modules
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assets = _build_assets(root)
        captures = {role: _capture_role(root, assets, role) for role in ROLES}
        admission_id = content_sha256(
            {"fixture": "three-role-exact-partition-identity-v4"}
        )
        category_evidence: dict[str, dict[str, Any]] = {}
        adapter_receipts: dict[str, dict[str, dict[str, Any]]] = {}
        for category in sorted(CATEGORY_PROTOCOLS):
            routes: dict[str, dict[str, Any]] = {}
            adapter_receipts[category] = {}
            for role in ROLES:
                capture = captures[role]
                routing_inputs, adapter_receipt = load_capture_router_inputs_v1(
                    receipt_path=capture["receipt_path"],
                    plan_path=capture["plan_path"],
                    authorization_path=capture["authorization_path"],
                    project_root=root,
                    activation_category=category,
                    partition_identity_admission_id=admission_id,
                    partition_admission_status="pass_candidate_partition_identity_v4",
                    category_partition_status="pass_exact_partition_identity",
                    category_partition_sha256=assets["partition_sha256"],
                    component_membership=assets["membership"],
                    minimum_response_rms=0.0,
                    minimum_margin_absolute=0.0,
                    minimum_margin_ratio=0.5,
                )
                verify_capture_router_adapter_receipt_v1(adapter_receipt)
                routes[role] = route_cluster_episodes_v1(**routing_inputs)
                verify_routing_receipt_v1(routes[role])
                adapter_receipts[category][role] = adapter_receipt
            stability = assess_routing_stability_v1(
                routes["discovery"],
                routes["cluster_validation"],
                maximum_total_variation=0.0,
                maximum_condition_total_variation=0.0,
                maximum_shared_rate_difference=0.0,
                minimum_condition_support_per_role=5,
            )
            stability_check = verify_routing_stability_v1(stability)
            if not stability_check["eligible_for_cluster_aware_sae_planning"]:
                raise ValueError("three-role integration routing stability was rejected")
            category_evidence[category] = {
                "discovery": routes["discovery"],
                "cluster_validation": routes["cluster_validation"],
                "stability": stability,
                "sae_training": routes["sae_training"],
            }

        predecessor_path = (
            project_root / "artifacts/vpd_compartment_sae_training_plan_v4/plan.json"
        )
        predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
        planner_kwargs = {
            "category_evidence": category_evidence,
            "predecessor_training_plan_id": predecessor["training_plan_id"],
            "predecessor_training_plan_sha256": sha256_file(predecessor_path),
            "profile": predecessor["profile"],
            "initialization_seeds": [17, 29],
            "split_seed": 1707,
            "validation_fraction": 0.25,
            "minimum_specific_episodes": 4,
            "minimum_specific_pairs": 4,
            "minimum_shared_episodes": 8,
            "minimum_shared_pairs": 2,
            "specific_dictionary_features": 4096,
            "shared_dictionary_features": 8192,
            "sae_activation_rows_per_episode": int(
                predecessor["profile"]["activation_rows_per_episode"]
            ),
        }
        plan_v1 = build_cluster_aware_sae_plan_v1(**planner_kwargs)
        plan_v2 = build_cluster_aware_sae_plan_v2(**planner_kwargs)
        plan_verification = verify_cluster_aware_sae_plan_v2(plan_v2)
        specific_v1 = [
            row for row in plan_v1["compartments"] if row["kind"] == "specific_cluster"
        ]
        specific_v2 = [
            row for row in plan_v2["compartments"] if row["kind"] == "specific_cluster"
        ]
        shared_v2 = [
            row for row in plan_v2["compartments"] if row["kind"] == "shared_fallback"
        ]

        first_category = sorted(CATEGORY_PROTOCOLS)[0]
        first_role = "sae_training"
        first_capture = captures[first_role]
        receipt_path = first_capture["receipt_path"]
        original_receipt_bytes = receipt_path.read_bytes()
        changed_receipt = json.loads(original_receipt_bytes.decode("utf-8"))
        changed_receipt["artifacts"]["components"][0][
            "episode_ids_by_category_sha256"
        ][first_category] = "0" * 64
        changed_body = {
            key: value
            for key, value in changed_receipt.items()
            if key != "capture_execution_id"
        }
        changed_receipt["capture_execution_id"] = content_sha256(changed_body)
        write_json(receipt_path, changed_receipt)
        episode_order_tamper_rejected = _rejects(
            lambda: load_capture_router_inputs_v1(
                receipt_path=receipt_path,
                plan_path=first_capture["plan_path"],
                authorization_path=first_capture["authorization_path"],
                project_root=root,
                activation_category=first_category,
                partition_identity_admission_id=admission_id,
                partition_admission_status="pass_candidate_partition_identity_v4",
                category_partition_status="pass_exact_partition_identity",
                category_partition_sha256=assets["partition_sha256"],
                component_membership=assets["membership"],
                minimum_response_rms=0.0,
                minimum_margin_absolute=0.0,
                minimum_margin_ratio=0.5,
            )
        )
        receipt_path.write_bytes(original_receipt_bytes)

        first_inputs, _ = load_capture_router_inputs_v1(
            receipt_path=receipt_path,
            plan_path=first_capture["plan_path"],
            authorization_path=first_capture["authorization_path"],
            project_root=root,
            activation_category=first_category,
            partition_identity_admission_id=admission_id,
            partition_admission_status="pass_candidate_partition_identity_v4",
            category_partition_status="pass_exact_partition_identity",
            category_partition_sha256=assets["partition_sha256"],
            component_membership=assets["membership"],
            minimum_response_rms=0.0,
            minimum_margin_absolute=0.0,
            minimum_margin_ratio=0.5,
        )
        effect_hash_tamper = copy.deepcopy(first_inputs)
        first_component = sorted(effect_hash_tamper["component_effects"])[0]
        effect_hash_tamper["expected_component_effect_sha256"][first_component] = "0" * 64
        adapter_receipt_tamper = copy.deepcopy(
            adapter_receipts[first_category][first_role]
        )
        adapter_receipt_tamper["episode_contract"]["episodes"] += 1
        training_as_validation_rejected = _rejects(
            lambda: assess_routing_stability_v1(
                category_evidence[first_category]["discovery"],
                category_evidence[first_category]["sae_training"],
                maximum_total_variation=0.0,
                maximum_condition_total_variation=0.0,
                maximum_shared_rate_difference=0.0,
                minimum_condition_support_per_role=5,
            )
        )
        all_states_restored_and_closed = all(
            capture["runtime_state"]["closed"] is True
            and all(
                np.array_equal(observed, expected)
                for module, initial in zip(
                    capture["runtime_state"]["modules"],
                    capture["runtime_state"]["initial"],
                    strict=True,
                )
                for observed, expected in zip(
                    _factor_arrays(module), initial, strict=True
                )
            )
            for capture in captures.values()
        )
        frameworks_absent = (
            torch_absent_before
            and transformers_absent_before
            and peft_absent_before
            and "torch" not in sys.modules
            and "transformers" not in sys.modules
            and "peft" not in sys.modules
        )
        shared_pair_ids_reused_across_categories = all(
            len(
                {
                    row["pair_id"]
                    for row in category_evidence[category]["sae_training"]["routing"][
                        "rows"
                    ]
                }
            )
            == 10
            for category in category_evidence
        ) and plan_v2["summary"]["routed_training_pairs"] == 10
        controls = {
            "three_verified_capture_roles": set(captures) == set(ROLES),
            "all_eight_categories_chained": len(category_evidence) == 8,
            "all_capture_factors_restored_and_runtimes_closed": all_states_restored_and_closed,
            "capture_receipts_drive_router_arrays": all(
                verify_capture_router_adapter_receipt_v1(receipt)["status"] == "pass"
                for by_role in adapter_receipts.values()
                for receipt in by_role.values()
            ),
            "discovery_validation_stability_passes_all_categories": all(
                evidence["stability"]["status"] == "pass_routing_stability_v1"
                for evidence in category_evidence.values()
            ),
            "sae_training_not_used_for_stability_selection": training_as_validation_rejected,
            "shared_pair_ids_reused_across_categories": shared_pair_ids_reused_across_categories,
            "v1_planner_collapses_real_shape_shared_pairs": len(specific_v1) == 0
            and plan_v1["summary"]["shared_compartments"] == 1,
            "v2_planner_preserves_two_clusters_per_category": len(specific_v2) == 16,
            "v2_planner_has_one_shared_fallback": len(shared_v2) == 1,
            "v2_plan_verifies": plan_verification["status"] == "pass",
            "v2_global_pair_split_consistent": plan_v2["summary"][
                "global_pair_split_consistent"
            ]
            and all(
                row["pair_atomic_split"]["global_pair_split_id"]
                == plan_v2["global_pair_split"]["global_pair_split_id"]
                for row in plan_v2["compartments"]
            ),
            "episodes_assigned_once_without_category_archive_duplication": sum(
                row["support"]["episodes"] for row in plan_v2["compartments"]
            )
            == 80
            == plan_v2["summary"]["routed_training_episodes"],
            "effect_and_sae_activation_row_units_separate": sum(
                row["support"]["routing_effect_rows"]
                for row in plan_v2["compartments"]
            )
            == 80
            and sum(
                row["support"]["sae_activation_rows"]
                for row in plan_v2["compartments"]
            )
            == 80 * 128,
            "episode_order_tamper_rejected": episode_order_tamper_rejected,
            "effect_hash_tamper_rejected": _rejects(
                lambda: route_cluster_episodes_v1(**effect_hash_tamper)
            ),
            "adapter_receipt_address_tamper_rejected": _rejects(
                lambda: verify_capture_router_adapter_receipt_v1(
                    adapter_receipt_tamper
                )
            ),
            "model_frameworks_absent_before_and_after": frameworks_absent,
        }
        if not all(controls.values()):
            raise ValueError(
                "three-role capture-to-SAE integration failed: "
                + ", ".join(name for name, passed in controls.items() if not passed)
            )
        body = {
            "schema": "sch.vpd-three-role-capture-to-sae-integration-benchmark.v1",
            "status": "pass",
            "design": {
                "split_roles": list(ROLES),
                "activation_categories": sorted(CATEGORY_PROTOCOLS),
                "pairs_per_role": 10,
                "same_pair_ids_across_categories_within_role": True,
                "canonical_component_clusters": 2,
                "receipt_to_router_adapter": True,
                "global_pair_split_planner_v2": True,
            },
            "results": {
                "capture_execution_ids": {
                    role: captures[role]["execution"]["capture_execution_id"]
                    for role in ROLES
                },
                "adapter_receipt_ids": {
                    category: {
                        role: adapter_receipts[category][role][
                            "capture_router_adapter_id"
                        ]
                        for role in ROLES
                    }
                    for category in sorted(adapter_receipts)
                },
                "routing_stability_ids": {
                    category: category_evidence[category]["stability"][
                        "routing_stability_id"
                    ]
                    for category in sorted(category_evidence)
                },
                "v1_specific_compartments": len(specific_v1),
                "v2_cluster_aware_sae_plan_id": plan_v2[
                    "cluster_aware_sae_plan_id"
                ],
                "v2_global_pair_split_id": plan_v2["global_pair_split"][
                    "global_pair_split_id"
                ],
                "v2_summary": plan_v2["summary"],
                "v2_compartment_support": {
                    row["compartment_id"]: row["support"]
                    for row in plan_v2["compartments"]
                },
            },
            "controls": controls,
            "summary": {
                "controls": len(controls),
                "passed": sum(bool(value) for value in controls.values()),
            },
            "source": {
                "benchmark_sha256": sha256_file(Path(__file__)),
                "capture_router_adapter_sha256": sha256_file(
                    Path(__file__).with_name("load_vpd_capture_router_inputs_v1.py")
                ),
                "planner_v2_sha256": sha256_file(
                    Path(__file__).with_name("plan_cluster_aware_sae_training_v2.py")
                ),
                "router_v1_sha256": sha256_file(
                    Path(__file__).with_name("route_vpd_cluster_episodes_v1.py")
                ),
            },
            "scientific_boundary": {
                "synthetic_artifacts_only": True,
                "model_loaded": False,
                "model_inference_performed": False,
                "weights_loaded_or_mutated": False,
                "sae_dictionary_trained": False,
                "empirical_effect_archive_produced": False,
                "execution_authorized": False,
                "sae_training_authorized": False,
                "release_authorized": False,
            },
        }
        return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("three-role integration benchmark address differs")
    if observed != build_report(project_root):
        raise ValueError("three-role integration benchmark is not reproducible")
    if observed.get("status") != "pass":
        raise ValueError("three-role integration benchmark did not pass")
    return {
        "schema": "sch.vpd-three-role-capture-to-sae-integration-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "controls": observed["summary"]["controls"],
        "specific_compartments": observed["results"]["v2_summary"][
            "specific_compartments"
        ],
        "shared_compartments": observed["results"]["v2_summary"][
            "shared_compartments"
        ],
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark three-role capture-to-router-to-SAE integration V1."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.output is not None:
        result = build_report(project_root)
        write_json(args.output, result)
    else:
        result = verify_benchmark(args.verify, project_root=project_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
