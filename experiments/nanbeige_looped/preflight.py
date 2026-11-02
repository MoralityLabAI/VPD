"""Real-model compatibility and pass-capture check, before VPD optimization.

This produces baseline activations and reference outputs. It does not perform a
parameter decomposition or claim that its small capture is SAE-training-ready.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from functools import partial
from pathlib import Path

from experiments.nanbeige_looped.capture import RepeatedForwardCapture, torch_tail_rows

MODEL_ID = "Nanbeige/Nanbeige4.2-3B"
REVISION = "451ed48c3273ecef7ea8faaa43c31ce529763bb1"


def prompts() -> list[dict]:
    items = []
    for index, (a, b) in enumerate([(17, 28), (24, 19), (38, 27), (46, 18),
                                   (59, 26), (63, 29), (78, 17), (86, 28)]):
        for variant in range(2):
            first = a + variant
            choices = [first + b, first + b + 10, first + b - 10, first + b + 1]
            shift = index % 4
            choices = choices[shift:] + choices[:shift]
            label = "ABCD"[choices.index(first + b)]
            text = f"Calculate {first} + {b}.\n" + "\n".join(
                f"{letter}. {value}" for letter, value in zip("ABCD", choices)
            ) + "\nReturn only the letter of the correct answer."
            items.append({"id": f"carry-{index}-{variant}", "pair_id": f"carry-{index}",
                          "family": "addition", "text": text, "answer": label})
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--layers", default="11,21")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise SystemExit("Choose an empty preflight output directory")
    torch.manual_seed(42)
    model_path = str(args.model_dir) if args.model_dir else snapshot_download(
        MODEL_ID, revision=REVISION,
        allow_patterns=["*.json", "*.py", "*.model", "*.safetensors", "*.txt"],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map=args.device,
    ).eval()
    if (model.config.num_hidden_layers, model.config.num_loops, model.config.hidden_size) != (22, 2, 3072):
        raise RuntimeError("Model architecture differs from the pinned Nanbeige contract")
    layer_indices = [int(value) for value in args.layers.split(",")]
    modules = {f"model.layers.{i}": model.model.layers[i] for i in layer_indices}
    modules["model.norm"] = model.model.norm
    metadata = []
    started = time.monotonic()
    for item in prompts():
        for thinking in (False, True):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": item["text"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
            )
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            if encoded.input_ids.shape[1] > args.max_prompt_tokens:
                raise RuntimeError("Prompt exceeds the frozen preflight length limit")
            inputs = {key: value.to(args.device) for key, value in encoded.items()}
            capture = RepeatedForwardCapture(modules, expected_calls=2,
                                            extract=partial(torch_tail_rows, rows=args.rows))
            with torch.inference_mode():
                plain = model(**inputs, use_cache=False).logits[:, -1].float().cpu()
                with capture:
                    hooked = model(**inputs, use_cache=False).logits[:, -1].float().cpu()
            torch.testing.assert_close(plain, hooked, rtol=1e-5, atol=1e-5)
            filename = f"{item['id']}__think_{int(thinking)}.npz"
            arrays = {f"pass_{invocation}__{name.replace('.', '_')}": value
                      for (name, invocation), value in capture.values.items()}
            activation_keys = list(arrays)
            arrays["reference_last_logits"] = plain.numpy()
            np.savez_compressed(args.output_dir / filename, **arrays)
            row = {**item, "thinking": thinking, "rendered_prompt": prompt,
                   "prompt_token_ids": encoded.input_ids[0].tolist(), "archive": filename,
                   "calls": capture.counts, "activation_keys": activation_keys,
                   "rows_per_site": min(args.rows, encoded.input_ids.shape[1]),
                   "hook_logit_max_abs_error": float((plain - hooked).abs().max()),
                   "next_token_id": int(plain.argmax()),
                   "sha256": hashlib.sha256((args.output_dir / filename).read_bytes()).hexdigest()}
            metadata.append(row)
            with (args.output_dir / "metadata.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(row) + "\n")
            print(json.dumps({"status": "captured", "id": item["id"], "thinking": thinking}), flush=True)
    manifest = {"status": "preflight_passed", "model_id": MODEL_ID, "revision": REVISION,
                "dtype": "bfloat16", "physical_layers": 22, "num_loops": 2,
                "hidden_size": 3072, "module_paths": list(modules), "examples": len(metadata),
                "capture_semantics": "physical_module_output; model.norm is the loop boundary",
                "invocation_index_base": 0, "parameter_decomposition_trained": False,
                "sae_training_ready": False, "elapsed_seconds": time.monotonic() - started,
                "torch_version": torch.__version__,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
