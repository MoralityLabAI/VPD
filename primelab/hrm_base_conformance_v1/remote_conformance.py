"""Non-mutating HRM-Text-1B native-base and no-op instrumentation audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


TARGET_FAMILIES = ("attn.gqkv_proj", "attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")
STACKS = ("L_module", "H_module")
EXPECTED_LAYER_COUNT = 16
EXPECTED_WEIGHT_SHA256 = "f8fe2b2bf6948414e8e8d6538659198726d98f967c55b533b7aabe8a1fa9a584"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = [json_safe(item) for item in value]
        return sorted(items, key=str) if isinstance(value, set) else items
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def tensor_finiteness(model: torch.nn.Module) -> dict[str, Any]:
    tensor_count = 0
    element_count = 0
    nonfinite_count = 0
    bad_tensors: list[dict[str, Any]] = []
    for name, value in model.state_dict().items():
        if not torch.is_tensor(value) or not (value.is_floating_point() or value.is_complex()):
            continue
        tensor_count += 1
        element_count += value.numel()
        count = int((~torch.isfinite(value)).sum().item())
        nonfinite_count += count
        if count:
            bad_tensors.append(
                {"name": name, "elements": value.numel(), "nonfinite": count}
            )
    return {
        "tensor_count": tensor_count,
        "element_count": element_count,
        "nonfinite_count": nonfinite_count,
        "bad_tensors": bad_tensors,
    }


def loading_info_counts(loading_info: dict[str, Any]) -> dict[str, Any]:
    fields = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    result: dict[str, Any] = {}
    for field in fields:
        values = loading_info.get(field) or []
        result[field] = {"count": len(values), "values": json_safe(values)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-pack", type=Path, required=True)
    parser.add_argument("--repo-id", default="sapientinc/HRM-Text-1B")
    parser.add_argument(
        "--revision", default="46e495977e4aff73b4b8bbb7fed6ff8e3f92d9f0"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--probe-limit", type=int, default=32)
    parser.add_argument("--local-model-dir", type=Path)
    parser.add_argument(
        "--model-code-dir",
        type=Path,
        help=(
            "Optional reconstructed loader overlay. Used only with a downloaded "
            "snapshot; files are copied over a symlinked immutable snapshot."
        ),
    )
    parser.add_argument("--weight-file", type=Path)
    args = parser.parse_args()

    started = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    termination: dict[str, Any] = {"status": "started", "started_unix": started}

    try:
        probe_payload = json.loads(args.probe_pack.read_text(encoding="utf-8"))
        all_prompts = probe_payload["prompts"]
        if len(all_prompts) != 32 or len(set(all_prompts)) != 32:
            raise ValueError("probe pack must contain 32 distinct prompts")
        if not 0 <= args.probe_limit <= 32:
            raise ValueError("probe-limit must be between 0 and 32")
        prompts = all_prompts[: args.probe_limit]

        if args.local_model_dir is None:
            downloaded_snapshot_path = Path(
                snapshot_download(
                    repo_id=args.repo_id,
                    revision=args.revision,
                    allow_patterns=[
                        "*.json",
                        "*.py",
                        "*.safetensors",
                        "*.model",
                        "*.txt",
                        ".gitattributes",
                    ],
                )
            ).resolve()
            snapshot_path = downloaded_snapshot_path
            source_kind = "pinned_huggingface_snapshot"
            if args.model_code_dir is not None:
                code_dir = args.model_code_dir.resolve()
                overlay_path = output_dir / "model_snapshot_overlay"
                overlay_path.mkdir(parents=True, exist_ok=True)
                for source in sorted(downloaded_snapshot_path.rglob("*")):
                    if not source.is_file():
                        continue
                    relative = source.relative_to(downloaded_snapshot_path)
                    destination = overlay_path / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.exists():
                        destination.symlink_to(source.resolve())
                for source in sorted(code_dir.iterdir()):
                    if source.is_file() and source.suffix == ".py":
                        destination = overlay_path / source.name
                        if destination.exists() or destination.is_symlink():
                            destination.unlink()
                        shutil.copy2(source, destination)
                snapshot_path = overlay_path.resolve()
                source_kind = (
                    "pinned_huggingface_snapshot_with_reconstructed_fused_loader_overlay"
                )
        else:
            snapshot_path = args.local_model_dir.resolve()
            source_kind = "recovered_local_directory"
        weight_path = (
            args.weight_file.resolve()
            if args.weight_file is not None
            else snapshot_path / "model.safetensors"
        )
        actual_weight_sha256 = sha256(weight_path)
        source_manifest = {
            "repo_id": args.repo_id,
            "revision": args.revision,
            "source_kind": source_kind,
            "snapshot_path": str(snapshot_path),
            "model_code_overlay": (
                {
                    path.name: {
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                    for path in sorted(args.model_code_dir.resolve().iterdir())
                    if path.is_file() and path.suffix == ".py"
                }
                if args.model_code_dir is not None
                else None
            ),
            "files": {
                str(path.relative_to(snapshot_path)): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in sorted(snapshot_path.rglob("*"))
                if path.is_file()
            },
            "expected_weight_sha256": EXPECTED_WEIGHT_SHA256,
            "actual_weight_sha256": actual_weight_sha256,
            "weight_hash_matches_recovered_base": (
                actual_weight_sha256 == EXPECTED_WEIGHT_SHA256
            ),
            "probe_pack_sha256": sha256(args.probe_pack),
            "runner_sha256": sha256(Path(__file__).resolve()),
        }
        write_json(output_dir / "source_manifest.json", source_manifest)

        if actual_weight_sha256 != EXPECTED_WEIGHT_SHA256:
            raise RuntimeError(
                "pinned remote model.safetensors does not match the recovered local base"
            )

        torch.manual_seed(20260908)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(20260908)
        torch.use_deterministic_algorithms(True, warn_only=False)

        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_path, trust_remote_code=True
        )
        model, loading_info = AutoModelForCausalLM.from_pretrained(
            snapshot_path,
            trust_remote_code=True,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
        model.eval()

        load_counts = loading_info_counts(loading_info)
        newly_initialized_keys = load_counts["missing_keys"]["values"]
        finiteness = tensor_finiteness(model)

        modules = dict(model.named_modules())
        expected_modules = [
            f"model.{stack}.layers.{layer}.{family}"
            for stack in STACKS
            for layer in range(EXPECTED_LAYER_COUNT)
            for family in TARGET_FAMILIES
        ]
        missing_modules = [name for name in expected_modules if name not in modules]
        discovered_modules = sorted(
            name
            for name in modules
            if any(name.endswith(family) for family in TARGET_FAMILIES)
            and (".L_module." in name or ".H_module." in name)
        )
        extra_modules = sorted(set(discovered_modules) - set(expected_modules))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        baseline_tokens: list[list[int]] = []
        baseline_logits: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=96,
            )
            baseline_tokens.extend(encoded["input_ids"].tolist())
            attention_masks.extend(encoded["attention_mask"].clone())
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded, use_cache=False).logits.detach().cpu()
            baseline_logits.extend(
                logits[index, : int(attention_masks[start + index].sum().item())].clone()
                for index in range(len(batch))
            )

        handles = [
            modules[name].register_forward_hook(
                lambda _module, _inputs, output: output
            )
            for name in expected_modules
            if name in modules
        ]
        instrumented_logits: list[torch.Tensor] = []
        retokenized: list[list[int]] = []
        tokenizer_second = AutoTokenizer.from_pretrained(
            snapshot_path, trust_remote_code=True
        )
        try:
            for start in range(0, len(prompts), args.batch_size):
                batch = prompts[start : start + args.batch_size]
                encoded = tokenizer_second(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=96,
                )
                retokenized.extend(encoded["input_ids"].tolist())
                masks = encoded["attention_mask"].clone()
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.inference_mode():
                    logits = model(**encoded, use_cache=False).logits.detach().cpu()
                instrumented_logits.extend(
                    logits[index, : int(masks[index].sum().item())].clone()
                    for index in range(len(batch))
                )
        finally:
            for handle in handles:
                handle.remove()

        probe_rows: list[dict[str, Any]] = []
        global_max_abs_error = 0.0
        token_mismatches = 0
        nonfinite_logit_values = 0
        for index, prompt in enumerate(prompts):
            tokens_equal = baseline_tokens[index] == retokenized[index]
            token_mismatches += int(not tokens_equal)
            baseline = baseline_logits[index]
            instrumented = instrumented_logits[index]
            same_shape = tuple(baseline.shape) == tuple(instrumented.shape)
            if same_shape:
                difference = (baseline - instrumented).abs()
                max_abs_error = float(difference.max().item())
                nonfinite_values = int((~torch.isfinite(baseline)).sum().item())
                nonfinite_values += int((~torch.isfinite(instrumented)).sum().item())
            else:
                max_abs_error = float("inf")
                nonfinite_values = -1
            global_max_abs_error = max(global_max_abs_error, max_abs_error)
            nonfinite_logit_values += max(0, nonfinite_values)
            probe_rows.append(
                {
                    "probe_index": index,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "tokens_equal": tokens_equal,
                    "shape_equal": same_shape,
                    "baseline_shape": list(baseline.shape),
                    "instrumented_shape": list(instrumented.shape),
                    "max_abs_error": max_abs_error,
                    "nonfinite_logit_values": nonfinite_values,
                }
            )
        with (output_dir / "probe_results.jsonl").open("w", encoding="utf-8") as handle:
            for row in probe_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

        environment_lock = {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: package_version(name)
                for name in (
                    "torch",
                    "transformers",
                    "huggingface-hub",
                    "accelerate",
                    "safetensors",
                    "psutil",
                )
            },
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "cuda_peak_memory_bytes": (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            ),
        }
        write_json(output_dir / "environment_lock.json", environment_lock)

        checks = {
            "weight_hash_matches_recovered_base": source_manifest[
                "weight_hash_matches_recovered_base"
            ],
            "missing_keys_zero": load_counts["missing_keys"]["count"] == 0,
            "unexpected_keys_zero": load_counts["unexpected_keys"]["count"] == 0,
            "mismatched_keys_zero": load_counts["mismatched_keys"]["count"] == 0,
            "loading_errors_zero": load_counts["error_msgs"]["count"] == 0,
            "newly_initialized_keys_zero": len(newly_initialized_keys) == 0,
            "model_tensors_finite": finiteness["nonfinite_count"] == 0,
            "native_fused_inventory_complete": not missing_modules
            and not extra_modules
            and len(discovered_modules) == 128,
            "registered_hook_count_128": len(handles) == 128,
            "probe_count_32": len(prompts) == 32,
            "token_mismatches_zero": token_mismatches == 0,
            "logits_finite": nonfinite_logit_values == 0,
            "fp32_logit_max_abs_error_lte_1e_5": global_max_abs_error <= 1e-5,
        }
        passed = all(checks.values())
        receipt = {
            "schema_version": "1.0.0",
            "run_id": os.environ.get("RUN_ID", "PL-HRM-BASE-CONFORMANCE-V1"),
            "status": "PASS" if passed else "FAIL",
            "mutated_weights": False,
            "optimizer_created": False,
            "checks": checks,
            "loading_info": load_counts,
            "newly_initialized_keys": newly_initialized_keys,
            "model_finiteness": finiteness,
            "module_inventory": {
                "expected_count": 128,
                "discovered_count": len(discovered_modules),
                "missing": missing_modules,
                "extra": extra_modules,
                "modules": discovered_modules,
            },
            "probe_summary": {
                "probe_count": len(prompts),
                "token_mismatches": token_mismatches,
                "nonfinite_logit_values": nonfinite_logit_values,
                "global_max_abs_error": global_max_abs_error,
            },
            "runtime_seconds": time.time() - started,
        }
        write_json(output_dir / "base_conformance.json", receipt)
        termination.update(
            {
                "status": "completed",
                "gate_status": receipt["status"],
                "finished_unix": time.time(),
                "exit_code": 0 if passed else 2,
            }
        )
        write_json(output_dir / "termination_receipt.json", termination)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if passed else 2
    except Exception as exc:
        termination.update(
            {
                "status": "exception",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "finished_unix": time.time(),
                "exit_code": 3,
            }
        )
        write_json(output_dir / "termination_receipt.json", termination)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
