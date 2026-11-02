from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.authz_mechinterp.benchmark_authenticated_lora_factor_bundle_extraction_v1 import (
    _fixture_config,
    _fixture_tensors,
    _write_safetensors,
)
from experiments.authz_mechinterp.benchmark_canonical_lora_peft_runtime_v1 import (
    _factor_arrays,
)
from experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_plan_v1 import (
    _model_identity,
    _rows,
    _split,
)
from experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_runtime_v1 import (
    _fake_runtime_factory,
)
from experiments.authz_mechinterp.benchmark_canonical_vpd_sae_training_capture_plan_v2 import (
    _authorization,
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
from experiments.authz_mechinterp.plan_canonical_vpd_capture_job_v2 import (
    build_capture_job_plan,
)
from experiments.authz_mechinterp.run_canonical_vpd_capture_job_v1 import (
    verify_capture_execution_receipt as verify_v1_execution_receipt,
)
from experiments.authz_mechinterp.run_canonical_vpd_capture_job_v2 import (
    run_capture_job,
    verify_capture_execution_receipt,
)


def _rejects(callback: Callable[[], Any]) -> bool:
    try:
        callback()
    except (ValueError, FileNotFoundError, RuntimeError, KeyError):
        return True
    return False


def _readdress(record: dict[str, Any], field: str) -> dict[str, Any]:
    body = {key: value for key, value in record.items() if key != field}
    return {**body, field: content_sha256(body)}


def _factors_restored(state: dict[str, Any]) -> bool:
    return bool(state.get("modules")) and all(
        np.array_equal(observed, expected)
        for module, initial in zip(
            state["modules"], state["initial"], strict=True
        )
        for observed, expected in zip(_factor_arrays(module), initial, strict=True)
    )


def _fixture(
    root: Path, *, output_dir: str
) -> tuple[dict[str, Any], Path, Path]:
    weights = root / "adapter_model.safetensors"
    config = root / "adapter_config.json"
    if not weights.exists():
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
        build_inventory(
            factor_index_path=factor_root / "factor-index.json",
            output_dir=root / "inventory",
        )
        _model_identity(root / "model_identity.json")
        split_path, _ = _split(root, _rows(), name="split")
        _rewrite_split_role(split_path, role="sae_training")
    inventory_rows = [
        json.loads(line)
        for line in (root / "inventory/canonical_components.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    component_ids = sorted(row["component_id"] for row in inventory_rows)[:2]
    plan = build_capture_job_plan(
        project_root=root,
        model_identity_path=root / "model_identity.json",
        adapter_weights_path=weights,
        adapter_config_path=config,
        extraction_manifest_path=root / "factor-bundle/manifest.json",
        inventory_manifest_path=root / "inventory/manifest.json",
        split_manifest_path=root / "split/split_manifest.json",
        split_role="sae_training",
        dataset_path=root / "split/discovery.jsonl",
        component_ids=component_ids,
        adapter_name="default",
        activation_key="layer_35__proposal_start",
        epsilon=0.05,
        batch_size=4,
        device="synthetic-cpu",
        dtype="float32",
        output_dir=output_dir,
        synthetic_benchmark_only=True,
    )
    suffix = output_dir.replace("/", "-")
    plan_path = root / f"plan-{suffix}.json"
    write_json(plan_path, plan)
    authorization_path = root / f"authorization-{suffix}.json"
    write_json(authorization_path, _authorization(plan, plan_path))
    return plan, plan_path, authorization_path


def build_report() -> dict[str, Any]:
    torch_absent_before = "torch" not in sys.modules
    transformers_absent_before = "transformers" not in sys.modules
    peft_absent_before = "peft" not in sys.modules
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        plan, plan_path, authorization_path = _fixture(
            root, output_dir="captures/synthetic-sae-training-runtime-v2-success"
        )
        actual_loader_blocked = _rejects(
            lambda: run_capture_job(
                plan_path=plan_path,
                authorization_path=authorization_path,
                project_root=root,
            )
        )
        factory, state = _fake_runtime_factory()
        execution = run_capture_job(
            plan_path=plan_path,
            authorization_path=authorization_path,
            project_root=root,
            runtime_factory=factory,
        )
        output = root / plan["capture_contract"]["output_dir"]
        receipt_path = output / "capture_execution_receipt.json"
        verification = verify_capture_execution_receipt(
            receipt_path,
            plan_path=plan_path,
            authorization_path=authorization_path,
            project_root=root,
        )
        success_restored = _factors_restored(state)
        expected_callbacks = int(plan["resource_ceiling"]["maximum_forward_passes"])
        no_weight_artifacts = not any(
            path
            for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt")
            for path in output.rglob(pattern)
        )

        wrong_plan, wrong_plan_path, wrong_auth_path = _fixture(
            root, output_dir="captures/synthetic-sae-training-runtime-v2-wrong-adapter"
        )
        wrong_factory, wrong_state = _fake_runtime_factory(adapter_name="wrong")
        wrong_adapter_rejected = _rejects(
            lambda: run_capture_job(
                plan_path=wrong_plan_path,
                authorization_path=wrong_auth_path,
                project_root=root,
                runtime_factory=wrong_factory,
            )
        )
        wrong_output_absent = not (
            root / wrong_plan["capture_contract"]["output_dir"]
        ).exists()

        fail_plan, fail_plan_path, fail_auth_path = _fixture(
            root, output_dir="captures/synthetic-sae-training-runtime-v2-callback-failure"
        )
        failure_factory, failure_state = _fake_runtime_factory(fail_callback=True)
        callback_failure_rejected = _rejects(
            lambda: run_capture_job(
                plan_path=fail_plan_path,
                authorization_path=fail_auth_path,
                project_root=root,
                runtime_factory=failure_factory,
            )
        )
        callback_output_absent = not (
            root / fail_plan["capture_contract"]["output_dir"]
        ).exists()

        original_receipt = copy.deepcopy(execution)
        changed_role = copy.deepcopy(original_receipt)
        changed_role["capture_job_plan"]["split_role"] = "discovery"
        write_json(
            receipt_path, _readdress(changed_role, "capture_execution_id")
        )
        readdressed_role_tamper_rejected = _rejects(
            lambda: verify_capture_execution_receipt(
                receipt_path,
                plan_path=plan_path,
                authorization_path=authorization_path,
                project_root=root,
            )
        )

        changed_authority = copy.deepcopy(original_receipt)
        changed_authority["scientific_boundary"][
            "sae_dictionary_training_authorized"
        ] = True
        changed_authority["scientific_boundary"]["sae_training_authorized"] = True
        write_json(
            receipt_path, _readdress(changed_authority, "capture_execution_id")
        )
        readdressed_training_authority_rejected = _rejects(
            lambda: verify_capture_execution_receipt(
                receipt_path,
                plan_path=plan_path,
                authorization_path=authorization_path,
                project_root=root,
            )
        )
        write_json(receipt_path, original_receipt)

        component_path = output / execution["artifacts"]["components"][0]["path"]
        component_bytes = component_path.read_bytes()
        component_path.write_bytes(component_bytes + b"tamper")
        component_tamper_rejected = _rejects(
            lambda: verify_capture_execution_receipt(
                receipt_path,
                plan_path=plan_path,
                authorization_path=authorization_path,
                project_root=root,
            )
        )
        component_path.write_bytes(component_bytes)

        shard_path = output / execution["artifacts"]["effect_shard"]["path"]
        shard_bytes = shard_path.read_bytes()
        shard_path.write_bytes(shard_bytes + b"tamper")
        effect_shard_tamper_rejected = _rejects(
            lambda: verify_capture_execution_receipt(
                receipt_path,
                plan_path=plan_path,
                authorization_path=authorization_path,
                project_root=root,
            )
        )
        shard_path.write_bytes(shard_bytes)
        restored_verification = verify_capture_execution_receipt(
            receipt_path,
            plan_path=plan_path,
            authorization_path=authorization_path,
            project_root=root,
        )

        v1_worker_rejects_v2_schema = _rejects(
            lambda: verify_v1_execution_receipt(
                receipt_path,
                plan_path=plan_path,
                authorization_path=authorization_path,
                project_root=root,
            )
        )
        controls = {
            "synthetic_plan_cannot_enter_actual_loader": actual_loader_blocked,
            "exact_planned_callback_count": execution["runtime"]["forward_passes"]
            == expected_callbacks
            == 32,
            "execution_receipt_verifies": verification["status"] == "pass"
            and restored_verification["status"] == "pass",
            "split_role_exactly_sae_training": verification["split_role"]
            == "sae_training",
            "success_factors_restored": success_restored,
            "success_runtime_closed": state["closed"] is True,
            "wrong_adapter_rejected_without_output": wrong_adapter_rejected
            and wrong_output_absent,
            "wrong_adapter_runtime_closed_and_restored": wrong_state["closed"] is True
            and _factors_restored(wrong_state),
            "callback_failure_rejected_without_output": callback_failure_rejected
            and callback_output_absent,
            "callback_failure_runtime_closed_and_restored": failure_state["closed"]
            is True
            and _factors_restored(failure_state),
            "readdressed_split_role_tamper_rejected": readdressed_role_tamper_rejected,
            "readdressed_training_authority_tamper_rejected": readdressed_training_authority_rejected,
            "component_artifact_tamper_rejected": component_tamper_rejected,
            "effect_shard_tamper_rejected": effect_shard_tamper_rejected,
            "v1_worker_rejects_v2_schema": v1_worker_rejects_v2_schema,
            "no_weight_or_dictionary_artifacts_persisted": no_weight_artifacts,
        }
        frameworks_absent = (
            torch_absent_before
            and transformers_absent_before
            and peft_absent_before
            and "torch" not in sys.modules
            and "transformers" not in sys.modules
            and "peft" not in sys.modules
        )
        if not all(controls.values()) or not frameworks_absent:
            failed = [name for name, passed in controls.items() if not passed]
            if not frameworks_absent:
                failed.append("model_frameworks_absent_before_and_after")
            raise ValueError(
                "canonical SAE-training capture runtime V2 benchmark failed: "
                + ", ".join(failed)
            )
        body = {
            "schema": "sch.vpd-canonical-sae-training-capture-runtime-benchmark.v2",
            "status": "pass",
            "scenario": {
                "components": len(execution["artifacts"]["components"]),
                "activation_categories": len(
                    plan["capture_contract"]["category_names"]
                ),
                "rows": plan["resource_ceiling"]["dataset_rows"],
                "split_role": verification["split_role"],
                "forward_passes": execution["runtime"]["forward_passes"],
                "execution_verification": verification,
            },
            "controls": controls,
            "summary": {
                "controls": len(controls),
                "passed": sum(bool(value) for value in controls.values()),
            },
            "source": {
                "benchmark_sha256": sha256_file(Path(__file__)),
                "runner_v2_sha256": sha256_file(
                    Path(__file__).with_name("run_canonical_vpd_capture_job_v2.py")
                ),
                "planner_v2_sha256": sha256_file(
                    Path(__file__).with_name("plan_canonical_vpd_capture_job_v2.py")
                ),
                "fake_runtime_predecessor_sha256": sha256_file(
                    Path(__file__).with_name(
                        "benchmark_canonical_vpd_capture_job_runtime_v1.py"
                    )
                ),
            },
            "scientific_boundary": {
                "synthetic_artifacts_only": True,
                "synthetic_authorization_is_not_user_authorization": True,
                "model_frameworks_absent_before_and_after": frameworks_absent,
                "actual_model_loaded": False,
                "model_inference_performed": False,
                "weights_loaded_or_mutated": False,
                "weights_or_sae_dictionary_persisted": False,
                "empirical_effect_capture_produced": False,
                "sae_dictionary_training_authorized": False,
                "sae_training_authorized": False,
                "real_actions_authorized": False,
                "release_authorized": False,
            },
        }
        return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = str(body.pop("benchmark_id", ""))
    if identifier != content_sha256(body):
        raise ValueError("SAE-training capture runtime V2 benchmark address differs")
    if observed != build_report():
        raise ValueError("SAE-training capture runtime V2 benchmark is not reproducible")
    if observed.get("status") != "pass":
        raise ValueError("SAE-training capture runtime V2 benchmark did not pass")
    return {
        "schema": "sch.vpd-canonical-sae-training-capture-runtime-benchmark-verification.v2",
        "status": "pass",
        "benchmark_id": identifier,
        "controls": observed["summary"]["controls"],
        "split_role": observed["scenario"]["split_role"],
        "actual_model_loaded": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the guarded canonical SAE-training capture runtime V2."
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
