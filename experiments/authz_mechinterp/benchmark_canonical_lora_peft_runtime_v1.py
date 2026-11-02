from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.benchmark_authenticated_lora_factor_bundle_extraction_v1 import (
    _fixture_config,
    _fixture_tensors,
    _write_safetensors,
)
from experiments.authz_mechinterp.canonical_lora_factor_runtime_v1 import (
    load_canonical_module_artifact,
    multiplier_factors_from_artifact,
)
from experiments.authz_mechinterp.canonical_lora_peft_runtime_v1 import (
    run_with_canonical_multiplier_peft,
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


class _FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values).copy()

    def detach(self) -> _FakeTensor:
        return self

    def to(self, device: str) -> _FakeTensor:
        if device != "cpu":
            raise ValueError("fake tensor supports only CPU inspection")
        return self

    def contiguous(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values

    def float(self) -> _FakeTensor:
        return _FakeTensor(self.values.astype(np.float32))

    def clone(self) -> _FakeTensor:
        return _FakeTensor(self.values)

    def equal(self, other: _FakeTensor) -> bool:
        return self.values.dtype == other.values.dtype and np.array_equal(
            self.values, other.values
        )

    def new_tensor(self, values: np.ndarray) -> _FakeTensor:
        return _FakeTensor(np.asarray(values, dtype=self.values.dtype))

    def copy_(self, source: _FakeTensor) -> _FakeTensor:
        np.copyto(self.values, source.values)
        return self


class _FakeLinear:
    def __init__(self, values: np.ndarray) -> None:
        self.weight = _FakeTensor(values)


class _FakePeftModule:
    def __init__(
        self, a_factor: np.ndarray, b_factor: np.ndarray, *, scale: float
    ) -> None:
        self.lora_A = {"default": _FakeLinear(a_factor)}
        self.lora_B = {"default": _FakeLinear(b_factor)}
        self.scaling = {"default": scale}


class _FakeModel:
    def __init__(self, rows: list[tuple[str, Any]]) -> None:
        self.rows = rows

    def named_modules(self) -> list[tuple[str, Any]]:
        return self.rows


def _sha256_float64(values: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(
        np.asarray(values).astype(np.float64, copy=False).tobytes(order="C")
    ).hexdigest()


def _attempt(function: Any) -> bool:
    try:
        function()
    except (ValueError, RuntimeError):
        return True
    return False


def _factor_arrays(module: _FakePeftModule) -> tuple[np.ndarray, np.ndarray]:
    return (
        module.lora_A["default"].weight.values.copy(),
        module.lora_B["default"].weight.values.copy(),
    )


def build_report() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        weights = root / "adapter_model.safetensors"
        config = root / "adapter_config.json"
        _write_safetensors(weights, _fixture_tensors(), metadata={"format": "pt"})
        write_json(config, _fixture_config())
        factor_output = root / "factor-bundle"
        extract_factor_bundle(
            adapter_weights_path=weights,
            adapter_config_path=config,
            expected_adapter_sha256=sha256_file(weights),
            expected_config_sha256=sha256_file(config),
            output_dir=factor_output,
            synthetic_benchmark_only=True,
        )
        index = json.loads(
            (factor_output / "factor-index.json").read_text(encoding="utf-8")
        )
        inventory_output = root / "canonical-inventory"
        inventory = build_inventory(
            factor_index_path=factor_output / "factor-index.json",
            output_dir=inventory_output,
        )
        component_rows = [
            json.loads(line)
            for line in (inventory_output / "canonical_components.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        module_row = index["modules"][0]
        component_row = next(
            row for row in component_rows if row["module"] == module_row["module"]
        )
        canonical_module_row = inventory["artifacts"]["modules"][0]
        artifact_path = inventory_output / canonical_module_row["path"]
        artifact_sha256 = canonical_module_row["sha256"]
        artifact = load_canonical_module_artifact(
            artifact_path, expected_sha256=artifact_sha256
        )
        with np.load(factor_output / module_row["path"], allow_pickle=False) as archive:
            source_a = np.asarray(archive["a_factor"], dtype=np.float32)
            source_b = np.asarray(archive["b_factor"], dtype=np.float32)
        module_path = module_row["module"]
        resolved_path = "base_model.model." + module_path
        multiplier_rows: list[dict[str, Any]] = []
        for multiplier in (-1.0, 0.0, 1.0, 2.0):
            module = _FakePeftModule(
                source_a, source_b, scale=float(module_row["scale"])
            )
            model = _FakeModel([(resolved_path, module)])
            original_a, original_b = _factor_arrays(module)
            expected_a, expected_b = multiplier_factors_from_artifact(
                artifact, block_index=0, multiplier=multiplier
            )

            def callback() -> np.ndarray:
                installed_a, installed_b = _factor_arrays(module)
                return float(module_row["scale"]) * installed_b @ installed_a

            output, receipt = run_with_canonical_multiplier_peft(
                model=model,
                module_path=module_path,
                adapter_name="default",
                canonical_artifact_path=artifact_path,
                expected_canonical_artifact_sha256=artifact_sha256,
                expected_source_a_factor_sha256=component_row[
                    "source_a_factor_sha256"
                ],
                expected_source_b_factor_sha256=component_row[
                    "source_b_factor_sha256"
                ],
                block_index=0,
                multiplier=multiplier,
                callback=callback,
                no_grad_context_factory=nullcontext,
                maximum_relative_probe_error=2e-6,
            )
            expected_delta = float(artifact["scale"]) * expected_b @ expected_a
            restored_a, restored_b = _factor_arrays(module)
            multiplier_rows.append(
                {
                    "multiplier": multiplier,
                    "delta_frobenius_error": float(
                        np.linalg.norm(output.astype(np.float64) - expected_delta)
                    ),
                    "a_factor_restored_exactly": np.array_equal(
                        restored_a, original_a
                    ),
                    "b_factor_restored_exactly": np.array_equal(
                        restored_b, original_b
                    ),
                    "observed_relative_probe_error": receipt[
                        "observed_relative_probe_error"
                    ],
                    "resolved_module_suffix_matched": receipt["resolved_module"]
                    == resolved_path,
                }
            )

        failure_module = _FakePeftModule(
            source_a, source_b, scale=float(module_row["scale"])
        )
        failure_model = _FakeModel([(resolved_path, failure_module)])
        failure_original = _factor_arrays(failure_module)

        def fail_callback() -> None:
            raise RuntimeError("synthetic callback failure")

        callback_failure_propagated = _attempt(
            lambda: run_with_canonical_multiplier_peft(
                model=failure_model,
                module_path=module_path,
                adapter_name="default",
                canonical_artifact_path=artifact_path,
                expected_canonical_artifact_sha256=artifact_sha256,
                expected_source_a_factor_sha256=component_row[
                    "source_a_factor_sha256"
                ],
                expected_source_b_factor_sha256=component_row[
                    "source_b_factor_sha256"
                ],
                block_index=0,
                multiplier=2.0,
                callback=fail_callback,
                no_grad_context_factory=nullcontext,
                maximum_relative_probe_error=2e-6,
            )
        )
        callback_failure_restored = all(
            np.array_equal(observed, expected)
            for observed, expected in zip(
                _factor_arrays(failure_module), failure_original, strict=True
            )
        )

        source_mismatch_module = _FakePeftModule(
            source_a, source_b, scale=float(module_row["scale"])
        )
        source_mismatch_original = _factor_arrays(source_mismatch_module)
        source_mismatch_rejected = _attempt(
            lambda: run_with_canonical_multiplier_peft(
                model=_FakeModel([(resolved_path, source_mismatch_module)]),
                module_path=module_path,
                adapter_name="default",
                canonical_artifact_path=artifact_path,
                expected_canonical_artifact_sha256=artifact_sha256,
                expected_source_a_factor_sha256="0" * 64,
                expected_source_b_factor_sha256=component_row[
                    "source_b_factor_sha256"
                ],
                block_index=0,
                multiplier=1.0,
                callback=lambda: None,
                no_grad_context_factory=nullcontext,
            )
        )
        source_mismatch_never_mutated = all(
            np.array_equal(observed, expected)
            for observed, expected in zip(
                _factor_arrays(source_mismatch_module),
                source_mismatch_original,
                strict=True,
            )
        )
        scale_mismatch_rejected = _attempt(
            lambda: run_with_canonical_multiplier_peft(
                model=_FakeModel(
                    [
                        (
                            resolved_path,
                            _FakePeftModule(
                                source_a,
                                source_b,
                                scale=float(module_row["scale"]) + 0.5,
                            ),
                        )
                    ]
                ),
                module_path=module_path,
                adapter_name="default",
                canonical_artifact_path=artifact_path,
                expected_canonical_artifact_sha256=artifact_sha256,
                expected_source_a_factor_sha256=component_row[
                    "source_a_factor_sha256"
                ],
                expected_source_b_factor_sha256=component_row[
                    "source_b_factor_sha256"
                ],
                block_index=0,
                multiplier=1.0,
                callback=lambda: None,
                no_grad_context_factory=nullcontext,
            )
        )
        ambiguous_module_rejected = _attempt(
            lambda: run_with_canonical_multiplier_peft(
                model=_FakeModel(
                    [
                        (resolved_path, _FakePeftModule(source_a, source_b, scale=2.0)),
                        (
                            "replica." + resolved_path,
                            _FakePeftModule(source_a, source_b, scale=2.0),
                        ),
                    ]
                ),
                module_path=module_path,
                adapter_name="default",
                canonical_artifact_path=artifact_path,
                expected_canonical_artifact_sha256=artifact_sha256,
                expected_source_a_factor_sha256=component_row[
                    "source_a_factor_sha256"
                ],
                expected_source_b_factor_sha256=component_row[
                    "source_b_factor_sha256"
                ],
                block_index=0,
                multiplier=1.0,
                callback=lambda: None,
                no_grad_context_factory=nullcontext,
            )
        )
        tampered_artifact = root / "tampered-module.npz"
        tampered_artifact.write_bytes(artifact_path.read_bytes() + b"tamper")
        artifact_tamper_rejected = _attempt(
            lambda: run_with_canonical_multiplier_peft(
                model=_FakeModel(
                    [(resolved_path, _FakePeftModule(source_a, source_b, scale=2.0))]
                ),
                module_path=module_path,
                adapter_name="default",
                canonical_artifact_path=tampered_artifact,
                expected_canonical_artifact_sha256=artifact_sha256,
                expected_source_a_factor_sha256=component_row[
                    "source_a_factor_sha256"
                ],
                expected_source_b_factor_sha256=component_row[
                    "source_b_factor_sha256"
                ],
                block_index=0,
                multiplier=1.0,
                callback=lambda: None,
                no_grad_context_factory=nullcontext,
            )
        )
        maximum_delta_error = max(
            row["delta_frobenius_error"] for row in multiplier_rows
        )
        maximum_probe_error = max(
            row["observed_relative_probe_error"] for row in multiplier_rows
        )
        passed = (
            len(multiplier_rows) == 4
            and maximum_delta_error < 1e-6
            and maximum_probe_error < 2e-6
            and all(
                row["a_factor_restored_exactly"]
                and row["b_factor_restored_exactly"]
                and row["resolved_module_suffix_matched"]
                for row in multiplier_rows
            )
            and callback_failure_propagated
            and callback_failure_restored
            and source_mismatch_rejected
            and source_mismatch_never_mutated
            and scale_mismatch_rejected
            and ambiguous_module_rejected
            and artifact_tamper_rejected
        )
    body = {
        "schema": "sch.vpd-canonical-lora-peft-runtime-benchmark.v1",
        "status": "pass" if passed else "fail",
        "source": {
            "runner_sha256": sha256_file(Path(__file__)),
            "runtime_sha256": sha256_file(
                Path(__file__).with_name("canonical_lora_peft_runtime_v1.py")
            ),
        },
        "design": {
            "authenticated_extraction_chained": True,
            "canonical_inventory_chained": True,
            "fake_peft_module_and_tensor_only": True,
            "multipliers": [-1.0, 0.0, 1.0, 2.0],
        },
        "results": {
            "multiplier_rows": multiplier_rows,
            "maximum_delta_frobenius_error": maximum_delta_error,
            "maximum_relative_probe_error": maximum_probe_error,
            "callback_failure_propagated": callback_failure_propagated,
            "callback_failure_restored": callback_failure_restored,
            "source_hash_mismatch_rejected": source_mismatch_rejected,
            "source_hash_mismatch_never_mutated": source_mismatch_never_mutated,
            "live_scale_mismatch_rejected": scale_mismatch_rejected,
            "ambiguous_module_match_rejected": ambiguous_module_rejected,
            "canonical_artifact_tamper_rejected": artifact_tamper_rejected,
        },
        "scientific_boundary": {
            "fake_peft_objects_only": True,
            "torch_imported": False,
            "peft_imported": False,
            "model_loaded": False,
            "real_adapter_accessed": False,
            "model_inference_performed": False,
            "real_weights_mutated": False,
            "actual_torch_peft_execution_verified": False,
            "execution_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed)
    identifier = body.pop("benchmark_id", "")
    if identifier != content_sha256(body):
        raise ValueError("canonical PEFT runtime benchmark address mismatch")
    if observed != build_report() or observed["status"] != "pass":
        raise ValueError("canonical PEFT runtime benchmark is not reproducible")
    return {
        "schema": "sch.vpd-canonical-lora-peft-runtime-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "multipliers": len(observed["results"]["multiplier_rows"]),
        "negative_controls": 5,
        "actual_torch_peft_execution_verified": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark canonical PEFT factor runtime V1 without Torch."
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
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "benchmark_id": result["benchmark_id"],
                "multipliers": result.get("results", {}).get(
                    "multiplier_rows", result.get("multipliers")
                )
                if result.get("multipliers") is not None
                else len(result["results"]["multiplier_rows"]),
                "execution_authorized": result.get("execution_authorized", False),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
