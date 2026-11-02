from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.authz_mechinterp.activation_categories import categorize_episodes
from experiments.authz_mechinterp.assemble_canonical_vpd_effect_shard_v2 import (
    assemble_canonical_effect_shard,
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
)
from experiments.authz_mechinterp.run_canonical_vpd_capture_job_v1 import (
    CaptureRuntime,
    _load_transformer_runtime,
    _persist_component,
    _safe_output_path,
)
from experiments.authz_mechinterp.run_canonical_vpd_capture_job_v2 import (
    EXECUTION_SCHEMA as V2_EXECUTION_SCHEMA,
    preflight_capture_job as preflight_capture_job_v2,
    verify_capture_execution_receipt as verify_v2_execution_receipt,
)
from experiments.authz_mechinterp.verify_vpd_guarded_capture_job_binding_v1 import (
    verify_job_binding,
)
from experiments.authz_mechinterp.vpd_cluster_capture_campaign_guard_v1 import (
    CaptureCampaignGuard,
)
from experiments.authz_mechinterp.vpd_cluster_to_sae_output_reservation_v1 import (
    verify_output_reservation,
)
from experiments.authz_mechinterp.vpd_guarded_capture_runtime_adapter_v1 import (
    CaptureJobAllocation,
    ResourceObserver,
)
from experiments.authz_mechinterp.vpd_guarded_capture_runtime_adapter_v2 import (
    reserve_guarded_capture_job_before_runtime_load,
)


EXECUTION_SCHEMA = "sch.vpd-canonical-category-capture-job-execution.v3"
RuntimeFactory = Callable[[dict[str, Any], Path], CaptureRuntime]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_guarded_context(
    *,
    plan_path: Path,
    authorization_path: Path,
    project_root: Path,
    campaign_guard: CaptureCampaignGuard,
    runtime_factory: RuntimeFactory | None,
    allocation_manifest_path: Path | None,
    job_allocation_id: str | None,
    reservation_receipt_path: Path | None,
    synthetic_job_allocation: CaptureJobAllocation | None,
) -> dict[str, Any]:
    preflight_v2 = preflight_capture_job_v2(
        plan_path=plan_path,
        authorization_path=authorization_path,
        project_root=project_root,
    )
    plan = _load(plan_path)
    synthetic = bool(plan["scientific_boundary"]["synthetic_benchmark_only"])
    role = str(plan["source"]["semantic_split"]["split_role"])
    if campaign_guard.role != role:
        raise ValueError("canonical capture V3 campaign guard role differs")
    if synthetic:
        if (
            runtime_factory is None
            or synthetic_job_allocation is None
            or any(
                value is not None
                for value in (
                    allocation_manifest_path,
                    job_allocation_id,
                    reservation_receipt_path,
                )
            )
            or not campaign_guard.simulation_only
        ):
            raise ValueError("canonical capture V3 synthetic context differs")
        allocation = synthetic_job_allocation
        expected_episodes = (
            2
            * int(preflight_v2["components"])
            * int(preflight_v2["rows"])
        )
        if (
            allocation.batch_forwards != int(preflight_v2["maximum_forward_passes"])
            or allocation.episode_forward_equivalents != expected_episodes
        ):
            raise ValueError("canonical capture V3 synthetic allocation differs")
        return {
            "synthetic": True,
            "allocation": allocation,
            "job_binding": None,
            "reservation": None,
            "preflight_v2": preflight_v2,
        }
    if (
        runtime_factory is not None
        or synthetic_job_allocation is not None
        or allocation_manifest_path is None
        or not job_allocation_id
        or reservation_receipt_path is None
    ):
        raise ValueError("canonical capture V3 production context is incomplete")
    job_binding = verify_job_binding(
        plan_path=plan_path,
        capture_authorization_path=authorization_path,
        allocation_manifest_path=allocation_manifest_path,
        job_allocation_id=job_allocation_id,
        project_root=project_root,
    )
    reservation = verify_output_reservation(
        reservation_receipt_path, project_root=project_root
    )
    if (
        job_binding["identity_bindings_complete"] is not True
        or reservation["runtime_admission_real"] is not True
        or reservation["simulation_only"] is not False
        or reservation["preflight_id"] != campaign_guard.preflight_id
        or campaign_guard.simulation_only
        or not campaign_guard.runtime_admission_real
        or not campaign_guard.output_reserved
    ):
        raise ValueError("canonical capture V3 real admission or guard differs")
    allocation = CaptureJobAllocation(
        batch_forwards=int(job_binding["allocation"]["batch_forwards"]),
        episode_forward_equivalents=int(
            job_binding["allocation"]["episode_forward_equivalents"]
        ),
        float32_effect_payload_bytes=int(
            job_binding["allocation"]["float32_effect_payload_bytes"]
        ),
    )
    return {
        "synthetic": False,
        "allocation": allocation,
        "job_binding": job_binding,
        "reservation": reservation,
        "preflight_v2": preflight_v2,
    }


def run_capture_job(
    *,
    plan_path: Path,
    authorization_path: Path,
    project_root: Path,
    campaign_guard: CaptureCampaignGuard,
    resource_observer: ResourceObserver,
    runtime_factory: RuntimeFactory | None = None,
    allocation_manifest_path: Path | None = None,
    job_allocation_id: str | None = None,
    reservation_receipt_path: Path | None = None,
    synthetic_job_allocation: CaptureJobAllocation | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    context = _prepare_guarded_context(
        plan_path=plan_path,
        authorization_path=authorization_path,
        project_root=project_root,
        campaign_guard=campaign_guard,
        runtime_factory=runtime_factory,
        allocation_manifest_path=allocation_manifest_path,
        job_allocation_id=job_allocation_id,
        reservation_receipt_path=reservation_receipt_path,
        synthetic_job_allocation=synthetic_job_allocation,
    )
    plan = _load(plan_path)
    authorization = verify_capture_authorization(
        authorization_path, plan_path=plan_path, project_root=project_root
    )
    output_dir = _safe_output_path(
        project_root, str(plan["capture_contract"]["output_dir"])
    )
    reservation = reserve_guarded_capture_job_before_runtime_load(
        guard=campaign_guard,
        allocation=context["allocation"],
        resource_observer=resource_observer,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".canonical-capture-v3-staging-", dir=output_dir.parent)
    )
    runtime: CaptureRuntime | None = None
    guarded_session: Any = None
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
        underlying_runtime = (
            runtime_factory(plan, project_root)
            if runtime_factory is not None
            else _load_transformer_runtime(plan=plan, project_root=project_root)
        )
        guarded_session = reservation.attach_runtime(underlying_runtime)
        runtime = guarded_session.as_capture_runtime()
        if runtime.adapter_name != plan["capture_contract"]["adapter_name"]:
            raise ValueError("canonical capture V3 runtime adapter name differs")
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
            if callback_calls > int(context["preflight_v2"]["maximum_forward_passes"]):
                raise ValueError("canonical capture V3 exceeded its forward-pass ceiling")
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
        if callback_calls != int(context["preflight_v2"]["maximum_forward_passes"]):
            raise ValueError("canonical capture V3 forward-pass count differs from plan")
        guard_job_receipt = guarded_session.finalize_job()
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
        v2_body = {
            "schema": V2_EXECUTION_SCHEMA,
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
                "synthetic_benchmark_only": context["synthetic"],
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
        v2_execution = {
            **v2_body,
            "capture_execution_id": content_sha256(v2_body),
        }
        v2_receipt_path = staging / "capture_execution_receipt_v2.json"
        guard_receipt_path = staging / "guarded_job_receipt.json"
        write_json(v2_receipt_path, v2_execution)
        write_json(guard_receipt_path, guard_job_receipt)
        v3_body = {
            "schema": EXECUTION_SCHEMA,
            "status": "complete_guarded_canonical_category_capture_v3",
            "source": {
                "capture_job_plan_id": plan["capture_job_plan_id"],
                "capture_authorization_id": authorization[
                    "capture_authorization_id"
                ],
                "job_allocation_id": (
                    context["job_binding"]["job_allocation_id"]
                    if context["job_binding"] is not None
                    else None
                ),
                "allocation_manifest_id": (
                    context["job_binding"]["allocation_manifest_id"]
                    if context["job_binding"] is not None
                    else None
                ),
                "reservation_transition_id": (
                    context["reservation"]["reservation_transition_id"]
                    if context["reservation"] is not None
                    else None
                ),
            },
            "predecessor_v2_execution": {
                "path": v2_receipt_path.name,
                "sha256": sha256_file(v2_receipt_path),
                "capture_execution_id": v2_execution["capture_execution_id"],
            },
            "guarded_job_receipt": {
                "path": guard_receipt_path.name,
                "sha256": sha256_file(guard_receipt_path),
                "job_receipt_id": guard_job_receipt["job_receipt_id"],
                "allocation": guard_job_receipt["allocation"],
            },
            "scientific_boundary": {
                "synthetic_benchmark_only": context["synthetic"],
                "reservation_preceded_runtime_load": True,
                "counter_increment_preceded_each_callback": True,
                "failed_callbacks_remain_charged": True,
                "base_model_weights_mutated": False,
                "adapter_weights_persisted": False,
                "generated_tokens": 0,
                "real_actions_authorized": False,
                "sae_dictionary_training_authorized": False,
                "authorization_control_release_authorized": False,
                "sae_training_authorized": False,
            },
        }
        v3_execution = {
            **v3_body,
            "capture_execution_v3_id": content_sha256(v3_body),
        }
        write_json(staging / "capture_execution_receipt.json", v3_execution)
        staging.replace(output_dir)
        return v3_execution
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
    allocation_manifest_path: Path | None = None,
    reservation_receipt_path: Path | None = None,
) -> dict[str, Any]:
    receipt = _load(receipt_path)
    body = {
        key: value for key, value in receipt.items() if key != "capture_execution_v3_id"
    }
    if (
        receipt.get("schema") != EXECUTION_SCHEMA
        or receipt.get("capture_execution_v3_id") != content_sha256(body)
    ):
        raise ValueError("canonical capture V3 receipt address differs")
    root = receipt_path.parent
    predecessor = receipt["predecessor_v2_execution"]
    predecessor_path = root / predecessor["path"]
    if sha256_file(predecessor_path) != predecessor["sha256"]:
        raise ValueError("canonical capture V3 predecessor bytes differ")
    predecessor_verification = verify_v2_execution_receipt(
        predecessor_path,
        plan_path=plan_path,
        authorization_path=authorization_path,
        project_root=project_root,
    )
    if predecessor_verification["capture_execution_id"] != predecessor[
        "capture_execution_id"
    ]:
        raise ValueError("canonical capture V3 predecessor identity differs")
    guard_record = receipt["guarded_job_receipt"]
    guard_path = root / guard_record["path"]
    if sha256_file(guard_path) != guard_record["sha256"]:
        raise ValueError("canonical capture V3 guard receipt bytes differ")
    guard_receipt = _load(guard_path)
    guard_body = {
        key: value for key, value in guard_receipt.items() if key != "job_receipt_id"
    }
    if (
        guard_receipt.get("job_receipt_id") != content_sha256(guard_body)
        or guard_receipt["job_receipt_id"] != guard_record["job_receipt_id"]
        or guard_receipt["observed"] != guard_receipt["allocation"]
    ):
        raise ValueError("canonical capture V3 guard receipt differs")
    plan = _load(plan_path)
    synthetic = bool(plan["scientific_boundary"]["synthetic_benchmark_only"])
    source = receipt["source"]
    if synthetic:
        if (
            allocation_manifest_path is not None
            or reservation_receipt_path is not None
            or source.get("job_allocation_id") is not None
            or source.get("allocation_manifest_id") is not None
            or source.get("reservation_transition_id") is not None
        ):
            raise ValueError("canonical capture V3 synthetic receipt context differs")
    else:
        if allocation_manifest_path is None or reservation_receipt_path is None:
            raise ValueError("canonical capture V3 production receipt inputs are absent")
        job_binding = verify_job_binding(
            plan_path=plan_path,
            capture_authorization_path=authorization_path,
            allocation_manifest_path=allocation_manifest_path,
            job_allocation_id=str(source.get("job_allocation_id", "")),
            project_root=project_root,
        )
        reservation_verification = verify_output_reservation(
            reservation_receipt_path, project_root=project_root
        )
        if (
            job_binding["allocation_manifest_id"]
            != source.get("allocation_manifest_id")
            or reservation_verification["reservation_transition_id"]
            != source.get("reservation_transition_id")
            or reservation_verification["preflight_id"]
            != guard_receipt["preflight_id"]
            or reservation_verification["runtime_admission_real"] is not True
        ):
            raise ValueError("canonical capture V3 production receipt lineage differs")
    boundary = receipt["scientific_boundary"]
    if (
        boundary.get("reservation_preceded_runtime_load") is not True
        or boundary.get("counter_increment_preceded_each_callback") is not True
        or any(
            boundary.get(field) not in (False, 0)
            for field in (
                "base_model_weights_mutated",
                "adapter_weights_persisted",
                "generated_tokens",
                "real_actions_authorized",
                "sae_dictionary_training_authorized",
                "authorization_control_release_authorized",
                "sae_training_authorized",
            )
        )
    ):
        raise ValueError("canonical capture V3 scientific boundary differs")
    return {
        "schema": "sch.vpd-canonical-category-capture-job-execution-verification.v3",
        "status": "pass",
        "capture_execution_v3_id": receipt["capture_execution_v3_id"],
        "capture_execution_v2_id": predecessor["capture_execution_id"],
        "guarded_job_receipt_id": guard_receipt["job_receipt_id"],
        "split_role": predecessor_verification["split_role"],
        "components": predecessor_verification["components"],
        "capture_started": True,
        "sae_dictionary_training_authorized": False,
        "real_actions_authorized": False,
        "release_authorized": False,
    }
