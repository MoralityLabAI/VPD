from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_runtime_v3 import (
    _allocation,
    _guard,
    _tracked_factory,
)
from experiments.authz_mechinterp.benchmark_canonical_vpd_sae_training_capture_runtime_v2 import (
    _fixture,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.load_vpd_capture_router_inputs_v2 import (
    load_capture_router_inputs_v2,
    verify_capture_router_adapter_receipt_v2,
)
from experiments.authz_mechinterp.route_vpd_cluster_episodes_v1 import (
    _canonical_groups,
    route_cluster_episodes_v1,
)
from experiments.authz_mechinterp.run_canonical_vpd_capture_job_v3 import (
    run_capture_job,
)


CATEGORY = "valid_authority"


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (KeyError, RuntimeError, TypeError, ValueError):
        return True
    return False


def _adapter_kwargs(
    *,
    root: Path,
    receipt_path: Path,
    plan_path: Path,
    authorization_path: Path,
    membership: dict[str, str],
) -> dict[str, Any]:
    groups, _ = _canonical_groups(membership)
    return {
        "v3_receipt_path": receipt_path,
        "plan_path": plan_path,
        "authorization_path": authorization_path,
        "project_root": root,
        "activation_category": CATEGORY,
        "partition_identity_admission_id": "1" * 64,
        "partition_admission_status": "pass_candidate_partition_identity_v4",
        "category_partition_status": "pass_exact_partition_identity",
        "category_partition_sha256": content_sha256(groups),
        "component_membership": membership,
        "minimum_response_rms": 0.0,
        "minimum_margin_absolute": 0.0,
        "minimum_margin_ratio": 0.0,
    }


def build_report(project_root: Path) -> dict[str, Any]:
    before_frameworks = MODEL_FRAMEWORKS & set(sys.modules)
    with tempfile.TemporaryDirectory(prefix="vpd-router-adapter-v2-") as temporary:
        root = Path(temporary)
        plan, plan_path, authorization_path = _fixture(
            root, output_dir="captures/router-adapter-v2"
        )
        allocation = _allocation(plan)
        guard = _guard(plan, allocation)
        factory, state = _tracked_factory(guard)
        execution = run_capture_job(
            plan_path=plan_path,
            authorization_path=authorization_path,
            project_root=root,
            campaign_guard=guard,
            resource_observer=lambda: (1.0, 0.01),
            runtime_factory=factory,
            synthetic_job_allocation=allocation,
        )
        output = root / plan["capture_contract"]["output_dir"]
        receipt_path = output / "capture_execution_receipt.json"
        predecessor_execution = json.loads(
            (output / "capture_execution_receipt_v2.json").read_text(
                encoding="utf-8"
            )
        )
        component_ids = sorted(
            str(row["component_id"])
            for row in predecessor_execution["artifacts"]["components"]
        )
        membership = {
            component_id: f"cluster-{index}"
            for index, component_id in enumerate(component_ids)
        }
        kwargs = _adapter_kwargs(
            root=root,
            receipt_path=receipt_path,
            plan_path=plan_path,
            authorization_path=authorization_path,
            membership=membership,
        )
        component_paths = [
            output / str(row["path"])
            for row in predecessor_execution["artifacts"]["components"]
        ]
        artifact_hashes_before = [sha256_file(path) for path in component_paths]
        routing_inputs, adapter_receipt = load_capture_router_inputs_v2(**kwargs)
        adapter_verification = verify_capture_router_adapter_receipt_v2(
            adapter_receipt
        )
        routing_receipt = route_cluster_episodes_v1(**routing_inputs)
        artifact_hashes_after = [sha256_file(path) for path in component_paths]

        changed_adapter = copy.deepcopy(adapter_receipt)
        changed_adapter["scientific_boundary"]["sae_training_authorized"] = True
        adapter_receipt_tamper_rejected = _rejects(
            lambda: verify_capture_router_adapter_receipt_v2(changed_adapter)
        )

        wrapper_bytes = receipt_path.read_bytes()
        changed_wrapper = json.loads(wrapper_bytes.decode("utf-8"))
        changed_wrapper["scientific_boundary"]["v3_guarded_lineage_preserved"] = True
        write_json(receipt_path, changed_wrapper)
        wrapper_tamper_rejected = _rejects(
            lambda: load_capture_router_inputs_v2(**kwargs)
        )
        receipt_path.write_bytes(wrapper_bytes)

        predecessor_path = output / "capture_execution_receipt_v2.json"
        predecessor_bytes = predecessor_path.read_bytes()
        changed_predecessor = json.loads(predecessor_bytes.decode("utf-8"))
        changed_predecessor["scientific_boundary"]["generated_tokens"] = 1
        write_json(predecessor_path, changed_predecessor)
        predecessor_tamper_rejected = _rejects(
            lambda: load_capture_router_inputs_v2(**kwargs)
        )
        predecessor_path.write_bytes(predecessor_bytes)

        guard_path = output / "guarded_job_receipt.json"
        guard_bytes = guard_path.read_bytes()
        changed_guard = json.loads(guard_bytes.decode("utf-8"))
        changed_guard["observed"]["compute_jobs"] = 0
        write_json(guard_path, changed_guard)
        guard_tamper_rejected = _rejects(
            lambda: load_capture_router_inputs_v2(**kwargs)
        )
        guard_path.write_bytes(guard_bytes)

        component_bytes = component_paths[0].read_bytes()
        component_paths[0].write_bytes(component_bytes + b"tamper")
        component_tamper_rejected = _rejects(
            lambda: load_capture_router_inputs_v2(**kwargs)
        )
        component_paths[0].write_bytes(component_bytes)

        category_kwargs = dict(kwargs)
        category_kwargs["activation_category"] = "not_a_category"
        membership_kwargs = dict(kwargs)
        membership_kwargs["component_membership"] = {
            **membership,
            "f" * 64: "cluster-extra",
        }
        production_context_kwargs = dict(kwargs)
        production_context_kwargs["allocation_manifest_path"] = root / "absent.json"
        production_context_kwargs["reservation_receipt_path"] = (
            root / "absent-reservation.json"
        )
        after_frameworks = MODEL_FRAMEWORKS & set(sys.modules)

        controls = {
            "adapter_v2_receipt_verifies": adapter_verification["status"] == "pass",
            "v3_execution_identity_is_router_identity": routing_inputs[
                "capture_execution_id"
            ]
            == execution["capture_execution_v3_id"],
            "router_receipt_preserves_v3_execution_identity": routing_receipt[
                "source"
            ]["capture_execution_id"]
            == execution["capture_execution_v3_id"],
            "predecessor_v2_identity_preserved": adapter_receipt["source"][
                "capture_execution_v2_id"
            ]
            == execution["predecessor_v2_execution"]["capture_execution_id"],
            "guarded_job_identity_preserved": adapter_receipt["source"][
                "guarded_job_receipt_id"
            ]
            == execution["guarded_job_receipt"]["job_receipt_id"],
            "synthetic_context_remains_explicit": adapter_receipt[
                "scientific_boundary"
            ]["synthetic_benchmark_only"]
            is True
            and adapter_receipt["source"]["allocation_manifest_id"] is None
            and adapter_receipt["source"]["reservation_transition_id"] is None,
            "capture_artifacts_are_read_only": artifact_hashes_before
            == artifact_hashes_after,
            "adapter_receipt_tamper_rejected": adapter_receipt_tamper_rejected,
            "v3_wrapper_tamper_rejected": wrapper_tamper_rejected,
            "v2_predecessor_tamper_rejected": predecessor_tamper_rejected,
            "guard_receipt_tamper_rejected": guard_tamper_rejected,
            "component_artifact_tamper_rejected": component_tamper_rejected,
            "category_substitution_rejected": _rejects(
                lambda: load_capture_router_inputs_v2(**category_kwargs)
            ),
            "component_membership_substitution_rejected": _rejects(
                lambda: load_capture_router_inputs_v2(**membership_kwargs)
            ),
            "production_context_rejected_for_synthetic": _rejects(
                lambda: load_capture_router_inputs_v2(**production_context_kwargs)
            ),
            "runtime_loaded_only_after_guard_reservation": state["guard_at_load"][
                "counters"
            ]["compute_jobs"]
            == 1,
            "model_frameworks_absent_before_and_after": not before_frameworks
            and not after_frameworks,
            "no_training_or_execution_authority": adapter_receipt[
                "scientific_boundary"
            ]["sae_training_authorized"]
            is False
            and adapter_receipt["scientific_boundary"]["execution_authorized"]
            is False,
        }
        body = {
            "schema": "sch.vpd-capture-receipt-router-input-adapter-benchmark.v2",
            "status": "pass" if all(controls.values()) else "fail",
            "source": {
                "adapter_v2_sha256": sha256_file(
                    Path(__file__).with_name("load_vpd_capture_router_inputs_v2.py")
                ),
                "benchmark_sha256": sha256_file(Path(__file__)),
                "capture_runner_v3_sha256": sha256_file(
                    Path(__file__).with_name("run_canonical_vpd_capture_job_v3.py")
                ),
                "router_v1_sha256": sha256_file(
                    Path(__file__).with_name("route_vpd_cluster_episodes_v1.py")
                ),
            },
            "design": {
                "synthetic_guarded_v3_capture": True,
                "router_consumes_v3_execution_identity": True,
                "predecessor_v2_array_loader_reused": True,
                "capture_artifacts_read_only": True,
                "production_allocation_and_reservation_replay_code_path_present": True,
                "production_path_empirically_exercised": False,
            },
            "results": {
                "capture_execution_v3_id": execution["capture_execution_v3_id"],
                "capture_execution_v2_id": execution["predecessor_v2_execution"][
                    "capture_execution_id"
                ],
                "guarded_job_receipt_id": execution["guarded_job_receipt"][
                    "job_receipt_id"
                ],
                "capture_router_adapter_v2_id": adapter_receipt[
                    "capture_router_adapter_v2_id"
                ],
                "cluster_episode_routing_id": routing_receipt[
                    "cluster_episode_routing_id"
                ],
                "adapter_verification": adapter_verification,
                "components": len(component_ids),
                "episodes": len(routing_inputs["episodes"]),
            },
            "controls": controls,
            "summary": {
                "controls": len(controls),
                "passed": sum(bool(value) for value in controls.values()),
            },
            "scientific_boundary": {
                "synthetic_capture_and_effect_arrays_only": True,
                "production_path_empirically_exercised": False,
                "persistent_empirical_router_receipt_produced": False,
                "model_loaded": False,
                "model_inference_performed": False,
                "weights_loaded_or_mutated": False,
                "sae_training_authorized": False,
                "execution_authorized": False,
            },
        }
        return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("V3 capture-to-router adapter benchmark address differs")
    if observed != build_report(project_root):
        raise ValueError("V3 capture-to-router adapter benchmark is not reproducible")
    if observed.get("status") != "pass":
        raise ValueError("V3 capture-to-router adapter benchmark did not pass")
    return {
        "schema": "sch.vpd-capture-receipt-router-input-adapter-benchmark-verification.v2",
        "status": "pass",
        "benchmark_id": identifier,
        "controls": observed["summary"]["controls"],
        "v3_guarded_lineage_preserved": True,
        "production_path_empirically_exercised": False,
        "sae_training_authorized": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark guarded V3 capture-to-router adapter V2."
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
