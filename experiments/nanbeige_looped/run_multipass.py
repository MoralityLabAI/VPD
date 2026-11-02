"""Executable VPD/SAE calibration and pass-specific effect capture.

Use --canary for a tiny random recurrent network, or --model-dir for the pinned
BF16 Nanbeige checkpoint. Both paths run actual optimization, but neither small
calibration establishes semantic features or an explanation of Astra.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.nanbeige_looped import vpd_math as vpd
from experiments.nanbeige_looped.measure import capture_forward, measure_component
from experiments.nanbeige_looped.preflight import MODEL_ID, REVISION, prompts
from experiments.nanbeige_looped.recurrent import RecurrentTarget, train_calibration
from experiments.nanbeige_looped.sae import encode_rows, fit_shared_sae


class TinyRecurrentLM(nn.Module):
    """Random test fixture; no pretrained language capabilities."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.shared = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 32)

    def forward(self, tokens):
        hidden = self.embedding(tokens)
        for _ in range(2):
            hidden = self.norm(hidden + torch.tanh(self.shared(hidden)))
        return self.head(hidden)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def architecture_canary(source):
    """Load already-downloaded, hash-checked publisher code without pretrained weights."""
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8-sig"))
    if provenance.get("revision") != REVISION:
        raise ValueError("Architecture canary requires the pinned source revision")
    for filename in ("configuration_nanbeige.py", "modeling_nanbeige.py"):
        if digest(source / filename) != provenance["files"][filename]:
            raise ValueError(f"Publisher source hash mismatch: {filename}")
    package = types.ModuleType("_nanbeige_multipass_canary")
    package.__path__ = [str(source.resolve())]
    sys.modules[package.__name__] = package
    config_module = importlib.import_module(package.__name__ + ".configuration_nanbeige")
    model_module = importlib.import_module(package.__name__ + ".modeling_nanbeige")
    config = config_module.NanbeigeConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=22, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, kv_channels=8, max_position_embeddings=512, num_loops=2,
        bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=False)
    config._attn_implementation = "sdpa"
    return model_module.NanbeigeForCausalLM(config).eval(), provenance


def validate_split(examples):
    if not examples or {row.get("split") for row in examples} != {"train", "eval"}:
        raise ValueError("Provide nonempty explicit train and eval splits")
    if len({row["id"] for row in examples}) != len(examples):
        raise ValueError("Duplicate example IDs")
    for field in ("pair_id", "token_ids"):
        groups = {split: {json.dumps(row[field], sort_keys=True)
                          for row in examples if row["split"] == split}
                  for split in ("train", "eval")}
        if groups["train"] & groups["eval"]:
            raise ValueError(f"Training/evaluation leakage in {field}")
    if any(not row["pair_id"] or not row["token_ids"] for row in examples):
        raise ValueError("Empty pair ID or token sequence")
    if any(not row.get("family") for row in examples):
        raise ValueError("Each example requires a family label")


def write_report(output_dir, manifest, quality):
    lines = ["# Multipass VPD / SAE calibration", "",
        f"Model source: **{manifest['model']['kind']}**.", "",
        "This is an implementation calibration, not a validated mechanism result.", "",
        f"VPD optimization: {manifest['vpd_training_steps']} steps; "
        f"{manifest['effect_records']} held-out component-effect records.",
        f"Adapter installation maximum logit difference: {manifest['installation_logit_max_abs_error']:.8g}.",
        "", "Each physical matrix has one V/U parameter set. Caches, CI, and intervention masks "
        "distinguish both invocations. First-only, second-only, and both-pass effects are saved "
        "at positive and negative coefficients, with exact token alignment.", "",
        "## Held-out SAE reconstruction", "",
        "| Site | First-pass explained variance | Second-pass explained variance |",
        "| --- | ---: | ---: |"]
    negative = []
    for site, entry in quality.items():
        numbers = [entry["held_out_quality"][f"pass_{p}"]["explained_variance"] for p in (0, 1)]
        formatted = ["undefined" if number is None else f"{number:.4f}" for number in numbers]
        lines.append(f"| {site} | {formatted[0]} | {formatted[1]} |")
        if any(number is not None and number < 0 for number in numbers):
            negative.append(site)
    if negative:
        lines += ["", "**Reconstruction limitation:** some passes have negative explained variance "
                  "(worse squared error than predicting their held-out mean). Feature differences "
                  "at these sites are not accepted for mechanistic interpretation: " + ", ".join(negative) + "."]
    checkpoint_next = ("Run pretrained BF16 capture on a GPU with the pinned checkpoint. "
                       if manifest["model"]["kind"] != "local_pretrained_checkpoint" else "")
    lines += ["", "The common encoder is fitted only on training examples, with balanced rows "
              "from both passes. Quality is evaluated separately by pass. No semantic validation "
              "of SAE features is claimed.", "", "## Artifacts", "",
              "- `vpd_factors.npz`: one component basis per physical matrix.",
              "- `ci_state.pt` and `vpd_training.json`: invocation-aware CI state and optimizer history.",
              "- `sae_*.npz` and `sae_quality.json`: common dictionaries and per-pass quality.",
              "- `baseline_*.npz`: original activations, token positions, and output logits.",
              "- `effect_*.npz` and `effects.json`: hidden/feature effects, central differences, "
              "interaction residuals, and next-token output proxies.",
              "- `manifest.json`: source, split, budget, and artifact hashes.", "",
              "## Remaining empirical work", "",
              checkpoint_next + "Improve and "
              "validate SAE reconstruction on each pass before interpreting features. Supply "
              "teacher-forced CoT prefixes and a task-specific behavioral endpoint for the tuck "
              "hypothesis; default arithmetic prompts only test prefill. Test stability across "
              "seeds and perturbation sizes before causal interpretation. These results provide "
              "no evidence about Astra's undisclosed internals.", ""]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def training_prefixes(train, mode, seq_len):
    """Select unpadded prefixes; generated endpoints retain all prior context."""
    if not train or any(row['split']!='train' for row in train):
        raise ValueError('VPD fitting accepts a nonempty training split only')
    if mode not in ('fixed_prefix', 'generated_prefixes'):
        raise ValueError('Unknown training-window mode')
    fixed = min(seq_len, min(len(row['token_ids']) for row in train))
    prefixes, receipt = [], []
    for row in train:
        tokens = row['token_ids']
        prompt = row.get('prompt_tokens')
        if mode=='generated_prefixes':
            if type(prompt) is not int or not 0<prompt<len(tokens):
                raise ValueError('Generated-prefix fitting requires prompt_tokens and a continuation')
            remaining = len(tokens)-prompt
            endpoints = sorted({prompt+max(1,round(remaining*f)) for f in (.25,.5,1.)})
        else:
            endpoints = [fixed]
        for end in endpoints:
            prefixes.append(tokens[:end])
            receipt.append(dict(id=row['id'],start=0,end=end,prompt_tokens=prompt,
                generated_tokens_in_prefix=max(0,end-prompt) if prompt is not None else None))
    return prefixes, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--canary", action="store_true")
    source.add_argument("--architecture-canary-dir", type=Path,
                        help="Pinned publisher source and provenance, with tiny random weights")
    source.add_argument("--model-dir", type=Path)
    parser.add_argument("--dataset", type=Path, help="JSONL with id, pair_id, family, split, token_ids")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-bf16", action="store_true", help="Keep the full pretrained checkpoint in BF16 on CPU")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--saved-tensors-on-cpu", action="store_true",
                        help="Offload autograd-saved tensors without truncation or forward recomputation")
    parser.add_argument("--layers", default="11,21", help="Zero-based physical layers")
    parser.add_argument("--vpd-steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--sae-steps", type=int, default=100)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--measure-components", type=int, default=2)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--training-window", choices=['fixed_prefix','generated_prefixes'], default='fixed_prefix',
                        help='Generated prefixes retain full conditioning through 25%, 50%, and 100% of each training continuation')
    parser.add_argument("--max-example-tokens", type=int, default=512,
                        help="Explicit full-context bound; --seq-len applies only to fixed-prefix fitting")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    if (min(args.vpd_steps, args.sae_steps, args.rows, args.seq_len, args.components,
            args.measure_components, args.cpu_threads) < 1 or args.warmup_steps < 0
            or args.measure_components > args.components or not np.isfinite(args.alpha)
            or args.alpha <= 0):
        parser.error("Invalid training, measurement, or coefficient budget")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Choose an empty output directory")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(args.cpu_threads)
    is_canary = args.canary or args.architecture_canary_dir is not None
    if is_canary:
        if args.architecture_canary_dir:
            model, source_provenance = architecture_canary(args.architecture_canary_dir)
            component_counts = {f"model.layers.{i}.mlp.down_proj": args.components for i in (11, 21)}
            sites = ["model.layers.11", "model.layers.21", "model.norm"]
            model_source = {"kind": "random_nanbeige_architecture_fixture", "seed": 42,
                            "publisher_source": source_provenance, "hidden_size": 32,
                            "physical_layers": 22, "pretrained_weights": False}
        else:
            model = TinyRecurrentLM().eval()
            component_counts = {"shared": args.components}
            sites = ["norm"]
            model_source = {"kind": "random_tiny_fixture", "seed": 42}
        device = "cpu"
        generator = np.random.default_rng(42)
        examples = [{"id": f"fixture-{i}", "pair_id": f"pair-{i}",
                     "family": f"fixture-family-{i // 4}", "split": "train" if i < 4 else "eval",
                     "token_ids": generator.integers(0, 32, size=12).tolist()}
                    for i in range(6)]
    else:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            parser.error("CUDA PyTorch and a GPU are required for the requested BF16 run")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = args.device
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") or args.cpu_bf16 else torch.float32,
            attn_implementation="sdpa", device_map=device,
        ).eval()
        if (model.config.num_hidden_layers, model.config.num_loops, model.config.hidden_size) != (22, 2, 3072):
            raise ValueError("Model differs from the Nanbeige architecture contract")
        model.config.use_cache = False
        layers = [int(value) for value in args.layers.split(",")]
        if len(set(layers)) != len(layers) or any(not 0 <= i < 22 for i in layers):
            raise ValueError("Select distinct physical layers from 0 to 21")
        component_counts = {f"model.layers.{i}.mlp.down_proj": args.components for i in layers}
        sites = [f"model.layers.{i}" for i in layers] + ["model.norm"]
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True, use_fast=False)
        examples = []
        for index, item in enumerate(prompts()):
            text = tokenizer.apply_chat_template([{"role": "user", "content": item["text"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            examples.append({**item, "split": "train" if index < 12 else "eval",
                "token_ids": tokenizer(text, add_special_tokens=False)["input_ids"]})
        model_source = {"kind": "local_pretrained_checkpoint", "model_id": MODEL_ID,
                        "expected_revision": REVISION, "revision_verified": False,
                        "path": str(args.model_dir.resolve()),
                        "dtype": str(model.dtype),
                        "checkpoint_files": {path.name: digest(path) for path in args.model_dir.iterdir()
                            if path.is_file() and path.suffix in {".json", ".py", ".safetensors"}}}
    if args.dataset:
        examples = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
    validate_split(examples)
    vocabulary = model.embedding.num_embeddings if args.canary else model.config.vocab_size
    if any(any(type(token) is not int or not 0 <= token < vocabulary for token in row["token_ids"])
           for row in examples):
        raise ValueError("Dataset token IDs are outside the model vocabulary")
    if not 1 <= args.max_example_tokens <= 8192:
        raise ValueError("Full-context capture bound must be between 1 and 8192")
    if any(len(row["token_ids"]) > args.max_example_tokens for row in examples):
        raise ValueError("An example exceeds the explicit full-context capture bound")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "run_status.json", {"status": "running", "canary": is_canary})
    write_json(args.output_dir / "examples.json", examples)
    first_tokens = torch.tensor([examples[0]["token_ids"]], device=device)
    with torch.inference_mode():
        original_output = model(first_tokens)
        original_logits = (original_output.logits if hasattr(original_output, "logits")
                           else original_output).float().cpu()
    del original_output
    target = RecurrentTarget(model, component_counts)
    with torch.inference_mode(), target.autocast():
        wrapped_logits = target(first_tokens).float().cpu()
    torch.testing.assert_close(original_logits, wrapped_logits, rtol=1e-5, atol=1e-5)
    installation_error = float((original_logits - wrapped_logits).abs().max())
    del original_logits, wrapped_logits
    train = [row for row in examples if row["split"] == "train"]
    evaluation = [row for row in examples if row["split"] == "eval"]
    prefixes, training_receipt = training_prefixes(train,args.training_window,args.seq_len)
    sequence_length = max(map(len,prefixes))
    batches = [torch.tensor([tokens], device=device) for tokens in prefixes]
    cfg = vpd.Config(C_per_module={name: view.C for name, view in target.views.items()},
                     batch_size=1, seq_len=sequence_length, n_steps=args.vpd_steps,
                     faithfulness_warmup_steps=args.warmup_steps,
                     ci_d_model=32 if is_canary else 128, ci_n_blocks=1, ci_n_heads=4,
                     ci_mlp_hidden=64 if is_canary else 256, use_wandb=False)
    print("Optimizing recurrent VPD calibration", flush=True)
    training_started = time.monotonic()
    def record_progress(row):
        row['elapsed_seconds'] = time.monotonic() - training_started
        if str(device).startswith('cuda'):
            row.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                       cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        with (args.output_dir / 'vpd_progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    try:
        ci, history = train_calibration(target, batches, cfg,
            variable_lengths=args.training_window=='generated_prefixes',
            saved_tensors_on_cpu=args.saved_tensors_on_cpu, progress_callback=record_progress)
    except Exception as error:
        write_json(args.output_dir / 'run_status.json', dict(status='calibration_failed',
            stage='vpd_training', exception_type=type(error).__name__, message=str(error)))
        raise
    factors = {f"{path}__{name}": getattr(wrapper.core, name).detach().cpu().numpy()
               for path, wrapper in target.physical.items() for name in ("V", "U")}
    np.savez_compressed(args.output_dir / "vpd_factors.npz", **factors)
    torch.save(ci.state_dict(), args.output_dir / "ci_state.pt")
    write_json(args.output_dir / "vpd_training.json", {"config": asdict(cfg), "history": history,
        "physical_matrices": list(component_counts), "invocation_sites": list(target.views),
        "parameter_sets_per_physical_matrix": 1, "learning_rate_schedule": "constant calibration",
        "training_prefixes":training_receipt,"training_window":args.training_window})
    print("Capturing aligned activations and fitting common SAE dictionaries", flush=True)
    captures = {}
    for index, row in enumerate(examples):
        tokens = torch.tensor([row["token_ids"]], device=device)
        captures[row["id"]], logits = capture_forward(target, tokens, sites, rows=args.rows)
        archive = {f"{site}__pass_{invocation}": value
                   for (site, invocation), value in captures[row["id"]].items()}
        archive["token_ids"] = np.asarray(row["token_ids"][-args.rows:])
        archive["token_positions"] = np.arange(max(0, len(row["token_ids"]) - args.rows), len(row["token_ids"]))
        archive["last_logits"] = logits
        np.savez_compressed(args.output_dir / f"baseline_{index:04d}.npz", **archive)
    dictionaries, sae_quality = {}, {}
    for site_index, site in enumerate(sites):
        print(f"Fitting shared SAE at {site}", flush=True)
        pass_rows = [np.concatenate([captures[row["id"]][(site, invocation)] for row in train])
                     for invocation in range(2)]
        width = pass_rows[0].shape[1]
        sae = fit_shared_sae(pass_rows, steps=args.sae_steps, top_k=min(32, width), seed=42, device=device)
        dictionaries[site] = sae
        filename = f"sae_{site_index:02d}.npz"
        sae.export(args.output_dir / filename, site=site)
        quality = {}
        for invocation in range(2):
            rows = np.concatenate([captures[row["id"]][(site, invocation)] for row in evaluation])
            _, quality[f"pass_{invocation}"] = encode_rows(sae, rows)
        sae_quality[site] = {"dictionary": filename, "sha256": digest(args.output_dir / filename),
                             "held_out_quality": quality, "train_example_ids": [r["id"] for r in train]}
    print("Measuring signed first-pass, second-pass, and shared-weight effects", flush=True)
    records = []
    for row in evaluation:
        tokens = torch.tensor([row["token_ids"]], device=device)
        for path in component_counts:
            for component in range(args.measure_components):
                arrays, summary = measure_component(target, tokens, sites, dictionaries,
                    path=path, component=component, alpha=args.alpha, rows=args.rows)
                filename = f"effect_{len(records):04d}.npz"
                np.savez_compressed(args.output_dir / filename, **arrays)
                records.append({"example_id": row["id"], "pair_id": row["pair_id"],
                    "archive": filename, "sha256": digest(args.output_dir / filename), **summary})
                print(f"Saved component-effect record {len(records)}: {row['id']} {path} component {component}", flush=True)
    write_json(args.output_dir / "effects.json", records)
    write_json(args.output_dir / "sae_quality.json", sae_quality)
    train_families = {r["family"] for r in train}
    eval_families = {r["family"] for r in evaluation}
    manifest = {"schema": "vpd.multipass.calibration.v1", "status": "calibration_complete",
        "model": model_source, "passes": 2, "pass_index_base": 0, "sites": sites,
        "vpd_training_steps": len(history), "effect_records": len(records),
        "train_examples": len(train), "eval_examples": len(evaluation),
        "family_disjoint": not bool(train_families & eval_families),
        "parameter_decomposition_optimized": True, "mechanism_validated": False,
        "sae_semantics_validated": False,
        "calibration_settings": {"seed": 42, "sae_steps": args.sae_steps,
            "sae_expansion": 2, "sae_training_loss": "balanced-pass raw activation MSE",
            "components_per_matrix": args.components, "measured_component_indices": list(range(args.measure_components)),
            "saved_tensors_on_cpu": args.saved_tensors_on_cpu,
            "alpha": args.alpha, "tail_rows": args.rows},
        "baseline_archives": {row["id"]: f"baseline_{index:04d}.npz" for index, row in enumerate(examples)},
        "implementation_sha256": {path.name: digest(path) for path in Path(__file__).parent.iterdir()
            if path.suffix == ".py" or path.name in {"vpd_backend_provenance.json", "VPD_BACKEND_LICENSE"}},
        "torch_version": torch.__version__, "device": device, "cpu_threads": args.cpu_threads,
        "installation_logit_max_abs_error": installation_error,
        "capture_scope": "Teacher-forced input tokens; defaults are prompt prefill, not generated CoT",
        "vpd_training_window": {"kind": args.training_window, "max_tokens": sequence_length,
                                "prefixes":training_receipt,"unpadded":True},
        "limitations": ["Small calibration only", "SAE quality has no automatic semantic acceptance",
                        "Single-pass interventions temporarily break weight sharing",
                        "Output metrics are next-token proxies, not task accuracy",
                        "No inference about Astra architecture or behavior"],
        }
    write_report(args.output_dir, manifest, sae_quality)
    manifest["artifacts"] = {path.name: digest(path) for path in args.output_dir.iterdir()
                            if path.is_file() and path.name != "run_status.json"}
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "run_status.json", {"status": "calibration_complete", "canary": is_canary})
    print(json.dumps({"status": manifest["status"], "output": str(args.output_dir),
                      "canary": is_canary, "effect_records": len(records)}), flush=True)


if __name__ == "__main__":
    main()
