from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.authz_mechinterp.benchmark_authenticated_lora_factor_bundle_extraction_v1 import (
    _fixture_config,
    _fixture_tensors,
    _write_safetensors,
)
from experiments.authz_mechinterp.benchmark_canonical_lora_peft_runtime_v1 import (
    _FakeModel,
    _FakePeftModule,
    _factor_arrays,
)
from experiments.authz_mechinterp.benchmark_canonical_vpd_capture_job_plan_v1 import (
    _address,
    _authorization,
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
    build_capture_job_plan,
)
from experiments.authz_mechinterp.run_canonical_vpd_capture_job_v1 import (
    CaptureRuntime,
    run_capture_job,
    verify_capture_execution_receipt,
)


def _attempt(function: Callable[[], Any]) -> bool:
    try:
        function()
    except (ValueError, FileNotFoundError, RuntimeError):
        return True
    return False


def _fake_runtime_factory(
    *, fail_callback: bool = False, adapter_name: str = "default"
) -> tuple[Callable[[dict[str, Any], Path], CaptureRuntime], dict[str, Any]]:
    state: dict[str, Any] = {"closed": False, "modules": []}

    def factory(plan: dict[str, Any], project_root: Path) -> CaptureRuntime:
        extraction_path = project_root / plan["source"]["factor_extraction"]["path"]
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
        index_record = extraction["artifacts"]["factor_index"]
        index_path = extraction_path.parent / index_record["path"]
        index = json.loads(index_path.read_text(encoding="utf-8"))
        named: list[tuple[str, _FakePeftModule]] = []
        for row in index["modules"]:
            with np.load(index_path.parent / row["path"], allow_pickle=False) as archive:
                a_factor = np.asarray(archive["a_factor"], dtype=np.float64)
                b_factor = np.asarray(archive["b_factor"], dtype=np.float64)
            module = _FakePeftModule(a_factor, b_factor, scale=float(row["scale"]))
            named.append((str(row["module"]), module))
        model = _FakeModel(named)
        state["modules"] = [module for _, module in named]
        state["initial"] = [
            tuple(array.copy() for array in _factor_arrays(module))
            for _, module in named
        ]

        def capture_batch(category: str, batch: list[dict[str, Any]]) -> np.ndarray:
            if fail_callback:
                raise RuntimeError("synthetic capture callback failure")
            hidden_size = 12
            combined = np.zeros(hidden_size, dtype=np.float64)
            for _, module in named:
                a_factor, b_factor = _factor_arrays(module)
                delta = (float(module.scaling["default"]) * (b_factor @ a_factor)).ravel()
                combined[: min(hidden_size, len(delta))] += delta[:hidden_size]
            category_index = sorted(
                {
                    "missing_or_untrusted_authority",
                    "other_invalid",
                    "replayed",
                    "revoked",
                    "scope_invalid",
                    "signature_invalid",
                    "time_inactive",
                    "valid_authority",
                }
            ).index(category)
            values = []
            for row_index, _row in enumerate(batch):
                coefficient = 1.0 + 0.1 * category_index + 0.01 * row_index
                baseline = np.linspace(0.0, 0.11, hidden_size)
                values.append(baseline + coefficient * combined)
            return np.asarray(values, dtype=np.float32)

        def close() -> None:
            state["closed"] = True

        return CaptureRuntime(
            model=model,
            adapter_name=adapter_name,
            capture_batch_callback=capture_batch,
            no_grad_context_factory=nullcontext,
            close=close,
            runtime_kind="synthetic_fake_peft_no_torch",
            actual_model_loaded=False,
        )

    return factory, state


def _fixture(root: Path, *, output_dir: str) -> tuple[dict[str, Any], Path, Path]:
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
        _split(root, _rows(), name="split")
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
        split_role="discovery",
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
    authorization = _authorization(plan, plan_path)
    authorization_path = root / f"authorization-{suffix}.json"
    write_json(authorization_path, authorization)
    return plan, plan_path, authorization_path


def build_report() -> dict[str, Any]:
    torch_absent_before = "torch" not in sys.modules
    peft_absent_before = "peft" not in sys.modules
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        plan, plan_path, authorization_path = _fixture(
            root, output_dir="captures/synthetic-runtime-success"
        )
        actual_loader_blocked_for_synthetic = _attempt(
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
        receipt_path = root / plan["capture_contract"]["output_dir"] / "capture_execution_receipt.json"
        verification = verify_capture_execution_receipt(
            receipt_path,
            plan_path=plan_path,
            authorization_path=authorization_path,
            project_root=root,
        )
        factor_restored = all(
            np.array_equal(observed, expected)
            for module, initial in zip(state["modules"], state["initial"], strict=True)
            for observed, expected in zip(_factor_arrays(module), initial, strict=True)
        )

        bad_plan, bad_plan_path, bad_authorization_path = _fixture(
            root, output_dir="captures/synthetic-runtime-wrong-adapter"
        )
        wrong_factory, wrong_state = _fake_runtime_factory(adapter_name="wrong")
        wrong_adapter_rejected = _attempt(
            lambda: run_capture_job(
                plan_path=bad_plan_path,
                authorization_path=bad_authorization_path,
                project_root=root,
                runtime_factory=wrong_factory,
            )
        )
        wrong_adapter_output_absent = not (
            root / bad_plan["capture_contract"]["output_dir"]
        ).exists()

        fail_plan, fail_plan_path, fail_authorization_path = _fixture(
            root, output_dir="captures/synthetic-runtime-callback-failure"
        )
        failure_factory, failure_state = _fake_runtime_factory(fail_callback=True)
        callback_failure_rejected = _attempt(
            lambda: run_capture_job(
                plan_path=fail_plan_path,
                authorization_path=fail_authorization_path,
                project_root=root,
                runtime_factory=failure_factory,
            )
        )
        callback_failure_restored = all(
            np.array_equal(observed, expected)
            for module, initial in zip(
                failure_state["modules"], failure_state["initial"], strict=True
            )
            for observed, expected in zip(_factor_arrays(module), initial, strict=True)
        )
        callback_failure_output_absent = not (
            root / fail_plan["capture_contract"]["output_dir"]
        ).exists()

        component_path = (
            receipt_path.parent / execution["artifacts"]["components"][0]["path"]
        )
        original_component = component_path.read_bytes()
        component_path.write_bytes(original_component + b"tamper")
        component_tamper_rejected = _attempt(
            lambda: verify_capture_execution_receipt(
                receipt_path,
                plan_path=plan_path,
                authorization_path=authorization_path,
                project_root=root,
            )
        )
        component_path.write_bytes(original_component)

        changed = copy.deepcopy(execution)
        changed["scientific_boundary"]["sae_training_authorized"] = True
        changed = _address(changed, "capture_execution_id")
        write_json(receipt_path, changed)
        boundary_tamper_rejected = _attempt(
            lambda: verify_capture_execution_receipt(
                receipt_path,
                plan_path=plan_path,
                authorization_path=authorization_path,
                project_root=root,
            )
        )
        write_json(receipt_path, execution)
        restored_verification = verify_capture_execution_receipt(
            receipt_path,
            plan_path=plan_path,
            authorization_path=authorization_path,
            project_root=root,
        )
        controls = {
            "synthetic_plan_cannot_enter_actual_loader": actual_loader_blocked_for_synthetic,
            "wrong_runtime_adapter_rejected": wrong_adapter_rejected,
            "wrong_runtime_adapter_output_absent": wrong_adapter_output_absent,
            "callback_failure_rejected": callback_failure_rejected,
            "callback_failure_restored_factors": callback_failure_restored,
            "callback_failure_output_absent": callback_failure_output_absent,
            "component_artifact_tamper_rejected": component_tamper_rejected,
            "readdressed_release_boundary_tamper_rejected": boundary_tamper_rejected,
        }
        if not factor_restored or not state["closed"] or not all(controls.values()):
            failed = [name for name, passed in controls.items() if not passed]
            if not factor_restored:
                failed.append("success_factors_restored")
            if not state["closed"]:
                failed.append("success_runtime_closed")
            raise ValueError("canonical capture runtime benchmark failed: " + ", ".join(failed))
        body = {
            "schema": "sch.vpd-canonical-category-capture-job-runtime-benchmark.v1",
            "status": "pass",
            "scenario": {
                "components": len(execution["artifacts"]["components"]),
                "activation_categories": len(plan["capture_contract"]["category_names"]),
                "rows": plan["resource_ceiling"]["dataset_rows"],
                "forward_passes": execution["runtime"]["forward_passes"],
                "factor_restored_after_success": factor_restored,
                "runtime_closed_after_success": state["closed"],
                "execution_verification": verification,
                "restored_execution_verification": restored_verification,
            },
            "negative_controls": controls,
            "source": {
                "runner_sha256": sha256_file(
                    Path(__file__).with_name("run_canonical_vpd_capture_job_v1.py")
                ),
                "planner_sha256": sha256_file(
                    Path(__file__).with_name("plan_canonical_vpd_capture_job_v1.py")
                ),
            },
            "scientific_boundary": {
                "synthetic_benchmark_only": True,
                "synthetic_authorization_is_not_user_authorization": True,
                "fake_peft_objects_only": True,
                "torch_absent_before_and_after": torch_absent_before and "torch" not in sys.modules,
                "peft_absent_before_and_after": peft_absent_before and "peft" not in sys.modules,
                "actual_model_loaded": False,
                "actual_model_inference_performed": False,
                "weights_persisted": False,
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
        raise ValueError("canonical capture runtime benchmark address mismatch")
    if observed != build_report():
        raise ValueError("canonical capture runtime benchmark is not reproducible")
    return {
        "schema": "sch.vpd-canonical-category-capture-job-runtime-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "negative_controls": len(observed["negative_controls"]),
        "actual_model_loaded": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the guarded canonical VPD capture runtime V1."
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
