from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.authz_mechinterp.build_vpd_cluster_to_sae_binding_readiness_v1 import (
    MODEL_FRAMEWORKS,
    OUTPUT_PARENT,
)
from experiments.authz_mechinterp.build_vpd_cluster_to_sae_empirical_preflight_v2 import (
    build_preflight,
)
from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp import verify_vpd_cluster_to_sae_bound_content_v1 as content_v1
from experiments.authz_mechinterp.verify_vpd_cluster_to_sae_bound_content_v1 import (
    build_model_files_manifest,
    verify_bound_content,
    verify_model_files_manifest,
)


def _rejects(call: Callable[[], Any]) -> bool:
    try:
        call()
    except (ValueError, json.JSONDecodeError):
        return True
    return False


def _address_manifest(value: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "manifest_id"}
    return {**body, "manifest_id": content_sha256(body)}


def build_report(project_root: Path) -> dict[str, Any]:
    before = MODEL_FRAMEWORKS & set(sys.modules)
    with tempfile.TemporaryDirectory(prefix="vpd-bound-content-") as temporary:
        root = Path(temporary)
        model_root = root / "model"
        adapter_root = root / "adapter"
        model_root.mkdir()
        adapter_root.mkdir()
        model_files = {
            "config.json": b'{"model_type":"synthetic"}\n',
            "model-00001-of-00002.safetensors": b"synthetic-shard-one\x00\x01",
            "model-00002-of-00002.safetensors": b"synthetic-shard-two\x02\x03",
        }
        for name, payload in model_files.items():
            (model_root / name).write_bytes(payload)
        weight_payload = b"synthetic-adapter-weights\x00\x01"
        config_payload = b'{"r":8,"target_modules":["q_proj"]}\n'
        weights_path = adapter_root / "adapter_model.safetensors"
        config_path = adapter_root / "adapter_config.json"
        weights_path.write_bytes(weight_payload)
        config_path.write_bytes(config_payload)

        manifest_path = root / "model-files-manifest.json"
        original_manifest = build_model_files_manifest(
            model_root=model_root,
            model_id="synthetic/model-not-loaded",
            model_revision="synthetic-revision-not-loaded",
        )
        write_json(manifest_path, original_manifest)
        bindings = {
            "model_id": "synthetic/model-not-loaded",
            "model_revision": "synthetic-revision-not-loaded",
            "model_files_manifest_sha256": sha256_file(manifest_path),
            "adapter_path": str(adapter_root.resolve()),
            "adapter_weights_sha256": sha256_file(weights_path),
            "adapter_config_sha256": sha256_file(config_path),
            "device": "cpu",
            "wall_time_ceiling_seconds": 3600,
            "cost_ceiling_usd": 1.0,
            "output_root": str(
                (
                    project_root
                    / OUTPUT_PARENT
                    / "synthetic-content-fixture-do-not-create"
                ).resolve()
            ),
        }
        preflight = build_preflight(project_root=project_root, bindings=bindings)
        preflight_path = root / "preflight.json"
        write_json(preflight_path, preflight)
        clean = verify_bound_content(
            preflight_path=preflight_path, model_manifest_path=manifest_path
        )

        def restore_manifest() -> None:
            write_json(manifest_path, original_manifest)

        first_model_path = model_root / "config.json"
        original_first = first_model_path.read_bytes()
        first_model_path.write_bytes(original_first + b"tamper")
        model_mutation_rejected = _rejects(
            lambda: verify_bound_content(
                preflight_path=preflight_path, model_manifest_path=manifest_path
            )
        )
        first_model_path.write_bytes(original_first)

        extra_path = model_root / "unlisted.bin"
        extra_path.write_bytes(b"unlisted")
        unlisted_rejected = _rejects(
            lambda: verify_bound_content(
                preflight_path=preflight_path, model_manifest_path=manifest_path
            )
        )
        extra_path.unlink()

        wrong_hash = json.loads(json.dumps(original_manifest))
        wrong_hash["files"][0]["sha256"] = "f" * 64
        write_json(manifest_path, _address_manifest(wrong_hash))
        wrong_hash_rejected = _rejects(
            lambda: verify_model_files_manifest(manifest_path)
        )
        restore_manifest()

        traversal = json.loads(json.dumps(original_manifest))
        traversal["files"][0]["path"] = "../escape.bin"
        write_json(manifest_path, _address_manifest(traversal))
        traversal_rejected = _rejects(lambda: verify_model_files_manifest(manifest_path))
        restore_manifest()

        duplicated = json.loads(json.dumps(original_manifest))
        duplicated["files"][1]["path"] = duplicated["files"][0]["path"]
        write_json(manifest_path, _address_manifest(duplicated))
        duplicate_rejected = _rejects(lambda: verify_model_files_manifest(manifest_path))
        restore_manifest()

        unsorted = json.loads(json.dumps(original_manifest))
        unsorted["files"] = list(reversed(unsorted["files"]))
        write_json(manifest_path, _address_manifest(unsorted))
        unsorted_rejected = _rejects(lambda: verify_model_files_manifest(manifest_path))
        restore_manifest()

        weights_path.write_bytes(weight_payload + b"tamper")
        adapter_weight_mutation_rejected = _rejects(
            lambda: verify_bound_content(
                preflight_path=preflight_path, model_manifest_path=manifest_path
            )
        )
        weights_path.write_bytes(weight_payload)

        config_path.write_bytes(config_payload + b"tamper")
        adapter_config_mutation_rejected = _rejects(
            lambda: verify_bound_content(
                preflight_path=preflight_path, model_manifest_path=manifest_path
            )
        )
        config_path.write_bytes(config_payload)

        second_weight = adapter_root / "adapter_model.bin"
        second_weight.write_bytes(b"ambiguous")
        ambiguous_adapter_rejected = _rejects(
            lambda: verify_bound_content(
                preflight_path=preflight_path, model_manifest_path=manifest_path
            )
        )
        second_weight.unlink()

        wrong_manifest_bindings = {
            **bindings,
            "model_files_manifest_sha256": "0" * 64,
        }
        wrong_manifest_preflight = build_preflight(
            project_root=project_root, bindings=wrong_manifest_bindings
        )
        wrong_manifest_preflight_path = root / "wrong-manifest-preflight.json"
        write_json(wrong_manifest_preflight_path, wrong_manifest_preflight)
        preflight_manifest_mismatch_rejected = _rejects(
            lambda: verify_bound_content(
                preflight_path=wrong_manifest_preflight_path,
                model_manifest_path=manifest_path,
            )
        )

        other_adapter = root / "other-adapter"
        other_adapter.mkdir()
        wrong_adapter_preflight = build_preflight(
            project_root=project_root,
            bindings={**bindings, "adapter_path": str(other_adapter.resolve())},
        )
        wrong_adapter_preflight_path = root / "wrong-adapter-preflight.json"
        write_json(wrong_adapter_preflight_path, wrong_adapter_preflight)
        adapter_path_mismatch_rejected = _rejects(
            lambda: verify_bound_content(
                preflight_path=wrong_adapter_preflight_path,
                model_manifest_path=manifest_path,
            )
        )

        incomplete_preflight_path = root / "incomplete-preflight.json"
        write_json(incomplete_preflight_path, build_preflight(project_root=project_root))
        incomplete_preflight_rejected = _rejects(
            lambda: verify_bound_content(
                preflight_path=incomplete_preflight_path,
                model_manifest_path=manifest_path,
            )
        )

        inside_manifest_path = model_root / "manifest.json"
        write_json(inside_manifest_path, original_manifest)
        inside_manifest_rejected = _rejects(
            lambda: verify_model_files_manifest(inside_manifest_path)
        )
        inside_manifest_path.unlink()

        empty_root = root / "empty-model"
        empty_root.mkdir()
        empty_model_rejected = _rejects(
            lambda: build_model_files_manifest(
                model_root=empty_root,
                model_id="synthetic/empty",
                model_revision="none",
            )
        )

        original_reparse_check = content_v1._is_reparse_point
        try:
            content_v1._is_reparse_point = lambda path: (
                path.name == "config.json" or original_reparse_check(path)
            )
            synthetic_reparse_rejected = _rejects(
                lambda: verify_model_files_manifest(manifest_path)
            )
        finally:
            content_v1._is_reparse_point = original_reparse_check

        after = MODEL_FRAMEWORKS & set(sys.modules)
        controls = {
            "clean_exact_byte_verification_passes_without_loading": (
                clean["status"] == "pass_exact_bytes_unopened_by_model_framework"
                and clean["model_files"] == 3
                and clean["model_loaded"] is False
                and clean["adapter_loaded"] is False
            ),
            "manifest_is_one_way_and_cycle_free": (
                "preflight_id" not in original_manifest
                and "preflight_sha256" not in original_manifest
                and original_manifest["cycle_boundary"]["manifest_is_built_before_preflight_binding"]
                is True
            ),
            "model_file_mutation_rejected": model_mutation_rejected,
            "unlisted_model_file_rejected": unlisted_rejected,
            "readdressed_wrong_file_hash_rejected": wrong_hash_rejected,
            "path_traversal_rejected": traversal_rejected,
            "duplicate_manifest_path_rejected": duplicate_rejected,
            "unsorted_manifest_path_rejected": unsorted_rejected,
            "adapter_weight_mutation_rejected": adapter_weight_mutation_rejected,
            "adapter_config_mutation_rejected": adapter_config_mutation_rejected,
            "ambiguous_adapter_weights_rejected": ambiguous_adapter_rejected,
            "preflight_manifest_digest_mismatch_rejected": preflight_manifest_mismatch_rejected,
            "preflight_adapter_path_mismatch_rejected": adapter_path_mismatch_rejected,
            "incomplete_preflight_rejected": incomplete_preflight_rejected,
            "manifest_inside_addressed_tree_rejected": inside_manifest_rejected,
            "empty_model_tree_rejected": empty_model_rejected,
            "synthetic_reparse_point_rejected": synthetic_reparse_rejected,
            "verifier_imports_no_model_frameworks": before == after,
            "all_verification_paths_remain_non_authorizing": all(
                clean[field] is False
                for field in (
                    "model_loaded",
                    "adapter_loaded",
                    "model_inference_performed",
                    "weights_mutated",
                    "execution_authorized",
                    "sae_training_authorized",
                )
            ),
        }
    if not all(controls.values()):
        raise ValueError(
            "cluster-to-SAE bound-content benchmark failed: "
            + ", ".join(name for name, passed in controls.items() if not passed)
        )
    body = {
        "schema": "sch.vpd-cluster-to-sae-bound-content-benchmark.v1",
        "status": "pass",
        "controls": controls,
        "summary": {"controls": len(controls), "passed": sum(controls.values())},
        "source": {
            "content_verifier_sha256": sha256_file(
                project_root
                / "experiments/authz_mechinterp/verify_vpd_cluster_to_sae_bound_content_v1.py"
            ),
            "benchmark_sha256": sha256_file(Path(__file__)),
        },
        "scientific_boundary": {
            "synthetic_temporary_files_only": True,
            "temporary_files_persisted": False,
            "real_model_or_adapter_content_opened": False,
            "model_framework_imported": False,
            "model_loaded": False,
            "adapter_loaded": False,
            "model_inference_performed": False,
            "weights_mutated": False,
            "authorization_receipt_present": False,
            "execution_authorized": False,
            "sae_training_authorized": False,
        },
    }
    return {**body, "benchmark_id": content_sha256(body)}


def verify_benchmark(path: Path, *, project_root: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    identifier = str(observed.get("benchmark_id", ""))
    body = {key: value for key, value in observed.items() if key != "benchmark_id"}
    if identifier != content_sha256(body) or observed != build_report(project_root):
        raise ValueError("cluster-to-SAE bound-content benchmark is not reproducible")
    return {
        "schema": "sch.vpd-cluster-to-sae-bound-content-benchmark-verification.v1",
        "status": "pass",
        "benchmark_id": identifier,
        "controls": observed["summary"]["controls"],
        "real_model_or_adapter_content_opened": False,
        "model_loaded": False,
        "execution_authorized": False,
        "sae_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark cluster-to-SAE streaming bound-content verification."
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
