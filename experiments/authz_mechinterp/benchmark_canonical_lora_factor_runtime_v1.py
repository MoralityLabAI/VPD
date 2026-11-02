from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.benchmark_canonical_lora_component_inventory_v1 import (
    _write_factor_bundle,
)
from experiments.authz_mechinterp.canonical_lora_factor_runtime_v1 import (
    load_canonical_module_artifact,
    multiplier_factors_from_artifact,
    run_with_canonical_multiplier_numpy,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.emit_canonical_lora_component_inventory_v1 import (
    build_inventory,
)


def build_report() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"; source.mkdir()
        index_path, factors = _write_factor_bundle(source, rotated=True)
        output = root / "inventory"
        manifest = build_inventory(factor_index_path=index_path, output_dir=output)
        module_entry = manifest["artifacts"]["modules"][0]
        artifact = load_canonical_module_artifact(
            output / module_entry["path"], expected_sha256=module_entry["sha256"]
        )
        original_a = factors[0][0].astype(np.float32)
        original_b = factors[0][1].astype(np.float32)
        baseline_delta = original_b.astype(np.float64) @ original_a.astype(np.float64)
        block_index = 1
        baseline_a, baseline_b = multiplier_factors_from_artifact(
            artifact, block_index=block_index, multiplier=1.0
        )
        zero_a, zero_b = multiplier_factors_from_artifact(
            artifact, block_index=block_index, multiplier=0.0
        )
        target_component = baseline_b @ baseline_a - zero_b @ zero_a
        rows: list[dict[str, Any]] = []
        for multiplier in (-1.0, 0.0, 1.0, 2.0):
            a_target = original_a.copy()
            b_target = original_b.copy()
            before_a = a_target.copy(); before_b = b_target.copy()

            def callback() -> np.ndarray:
                return b_target.astype(np.float64) @ a_target.astype(np.float64)

            observed_delta, receipt = run_with_canonical_multiplier_numpy(
                a_target=a_target,
                b_target=b_target,
                artifact=artifact,
                block_index=block_index,
                multiplier=multiplier,
                callback=callback,
                maximum_relative_probe_error=2e-6,
            )
            expected_delta = baseline_b @ baseline_a + (
                multiplier - 1.0
            ) * target_component
            rows.append(
                {
                    "multiplier": multiplier,
                    "delta_frobenius_error": float(
                        np.linalg.norm(observed_delta - expected_delta)
                    ),
                    "a_factor_restored_exactly": bool(
                        np.array_equal(a_target, before_a)
                    ),
                    "b_factor_restored_exactly": bool(
                        np.array_equal(b_target, before_b)
                    ),
                    "relative_probe_error": receipt[
                        "observed_relative_probe_error"
                    ],
                }
            )
        failing_a = original_a.copy(); failing_b = original_b.copy()
        before_failing_a = failing_a.copy(); before_failing_b = failing_b.copy()
        callback_failure_restored = False
        try:
            run_with_canonical_multiplier_numpy(
                a_target=failing_a,
                b_target=failing_b,
                artifact=artifact,
                block_index=block_index,
                multiplier=0.0,
                callback=lambda: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
                maximum_relative_probe_error=2e-6,
            )
        except RuntimeError:
            callback_failure_restored = bool(
                np.array_equal(failing_a, before_failing_a)
                and np.array_equal(failing_b, before_failing_b)
            )
        baseline_reconstruction_error = float(
            np.linalg.norm(baseline_b @ baseline_a - baseline_delta)
        )
        maximum_delta_error = max(row["delta_frobenius_error"] for row in rows)
        maximum_probe_error = max(row["relative_probe_error"] for row in rows)
        passed = (
            baseline_reconstruction_error < 1e-6
            and maximum_delta_error < 1e-6
            and maximum_probe_error <= 2e-6
            and all(row["a_factor_restored_exactly"] for row in rows)
            and all(row["b_factor_restored_exactly"] for row in rows)
            and callback_failure_restored
        )
    body = {
        "schema": "sch.vpd-canonical-lora-factor-runtime-benchmark.v1",
        "status": "pass" if passed else "fail",
        "source": {
            "runner_sha256": sha256_file(Path(__file__)),
            "runtime_sha256": sha256_file(
                Path(__file__).with_name("canonical_lora_factor_runtime_v1.py")
            ),
        },
        "design": {
            "target_dtype": "float32",
            "multipliers": [-1.0, 0.0, 1.0, 2.0],
            "probe_count": 8,
            "callback_failure_injected": True,
            "model_execution": False,
        },
        "results": {
            "baseline_reconstruction_frobenius_error": baseline_reconstruction_error,
            "maximum_intervention_delta_frobenius_error": maximum_delta_error,
            "maximum_relative_probe_error": maximum_probe_error,
            "multiplier_rows": rows,
            "callback_failure_restored": callback_failure_restored,
        },
        "scientific_boundary": {
            "synthetic_factor_bundle_only": True,
            "torch_or_model_loaded": False,
            "model_inference_performed": False,
            "real_weights_mutated": False,
            "production_torch_adapter_integration_present": False,
            "execution_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    body = dict(observed); identifier = body.pop("benchmark_id", "")
    if identifier != content_sha256(body):
        raise ValueError("canonical factor runtime benchmark address mismatch")
    if observed != build_report() or observed["status"] != "pass":
        raise ValueError("canonical factor runtime benchmark is not reproducible")
    return {
        "schema": "sch.vpd-canonical-lora-factor-runtime-benchmark-verification.v1",
        "status": "pass", "benchmark_id": identifier,
        "multipliers": len(observed["results"]["multiplier_rows"]),
        "callback_failure_restored": observed["results"]["callback_failure_restored"],
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark canonical LoRA factor runtime V1.")
    parser.add_argument("--output", type=Path); parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None: result = verify_benchmark(args.verify)
    else:
        if args.output is None: parser.error("--output is required unless verifying")
        result = build_report(); write_json(args.output, result)
    print(json.dumps({
        "schema": result["schema"], "status": result["status"],
        "benchmark_id": result["benchmark_id"],
        "multipliers": len(result.get("results", {}).get("multiplier_rows", [])) or result.get("multipliers"),
        "callback_failure_restored": result.get("results", {}).get(
            "callback_failure_restored", result.get("callback_failure_restored")
        ),
        "execution_authorized": result.get("execution_authorized", False),
    }, indent=2, sort_keys=True))


if __name__ == "__main__": main()
