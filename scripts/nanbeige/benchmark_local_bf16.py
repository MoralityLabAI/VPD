"""One full-checkpoint CPU forward pair to validate the local fallback."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.nanbeige_looped.capture import RepeatedForwardCapture, torch_tail_rows
from functools import partial

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("model_dir", type=Path)
p.add_argument("output_dir", type=Path)
args = p.parse_args()
if args.output_dir.exists():
    raise SystemExit("Choose a new benchmark directory")
args.output_dir.mkdir(parents=True)
torch.set_num_threads(6)
torch.manual_seed(42)
started = time.monotonic()
print("Loading full pretrained BF16 checkpoint on CPU", flush=True)
model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="sdpa").eval()
model.requires_grad_(False)
if (model.config.num_hidden_layers, model.config.num_loops, model.config.hidden_size) != (22, 2, 3072):
    raise RuntimeError("Unexpected model architecture")
tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True, use_fast=False)
text = tokenizer.apply_chat_template([{"role": "user", "content": "Calculate 17 + 28. Answer with the number only."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)
tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
loaded = time.monotonic()
print(f"Loaded in {loaded - started:.1f}s; running {tokens.shape[1]} tokens", flush=True)
sites = {s: model.get_submodule(s) for s in ("model.layers.11", "model.layers.21", "model.norm")}
capture = RepeatedForwardCapture(sites, expected_calls=2, extract=partial(torch_tail_rows, rows=16))
with torch.inference_mode():
    plain = model(tokens, use_cache=False).logits[:, -1].float().cpu()
    first = time.monotonic()
    print(f"First forward completed in {first - loaded:.1f}s", flush=True)
    with capture:
        hooked = model(tokens, use_cache=False).logits[:, -1].float().cpu()
torch.testing.assert_close(plain, hooked, rtol=1e-5, atol=1e-5)
archive = args.output_dir / "pretrained_capture.npz"
np.savez_compressed(archive, **{f"{s}__pass_{i}": v for (s, i), v in capture.values.items()},
                    token_ids=tokens[0].numpy(), last_logits=plain.numpy())
report = {"status": "pretrained_cpu_capture_passed", "dtype": "bfloat16", "device": "cpu",
          "load_seconds": loaded - started, "plain_forward_seconds": first - loaded,
          "hooked_forward_seconds": time.monotonic() - first, "calls": capture.counts,
          "installation_logit_max_abs_error": float((plain - hooked).abs().max()),
          "model_dir": str(args.model_dir.resolve()), "prompt": text,
          "next_token": tokenizer.decode([int(plain.argmax())]),
          "archive": archive.name, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
          "vpd_trained": False, "sae_trained": False}
(args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
