from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from experiments.authz_mechinterp.behavior import score_output


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def get_attr_path(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        value = getattr(value, part)
    return value


def discover_layers(model: Any) -> Sequence[Any]:
    for path in (
        "model.layers",
        "model.decoder.layers",
        "transformer.h",
        "gpt_neox.layers",
        "base_model.model.model.layers",
        "base_model.model.model.decoder.layers",
        "base_model.model.transformer.h",
        "base_model.model.gpt_neox.layers",
    ):
        try:
            layers = get_attr_path(model, path)
            if len(layers):
                return layers
        except (AttributeError, TypeError):
            continue
    raise ValueError("could not locate transformer block list")


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (KeyError, TypeError, ValueError):
            pass
    return "\n".join(
        f"[{message['role'].upper()}]\n{message['content']}" for message in messages
    ) + "\n[ASSISTANT]\n"


def load_vector(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.sae_decoder:
        decoder = np.load(args.sae_decoder).astype(np.float32)
        if args.feature_index is None:
            raise ValueError("--feature-index is required with --sae-decoder")
        return decoder[args.feature_index], f"sae_feature_{args.feature_index}"
    if args.vpd_effects:
        archive = np.load(args.vpd_effects, allow_pickle=True)
        ids = [str(x) for x in archive["component_ids"].tolist()]
        if args.component_id:
            index = ids.index(args.component_id)
        elif args.component_index is not None:
            index = args.component_index
        else:
            raise ValueError("select a VPD component by id or index")
        return archive["delta_h"][index].astype(np.float32), ids[index]
    raise ValueError("supply --sae-decoder or --vpd-effects")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, help="base or merged model path")
    parser.add_argument("--adapter", help="optional PEFT adapter path")
    parser.add_argument("--tokenizer", help="optional tokenizer path; defaults to --model")
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sae-decoder", type=Path)
    parser.add_argument("--feature-index", type=int)
    parser.add_argument("--vpd-effects", type=Path)
    parser.add_argument("--component-index", type=int)
    parser.add_argument("--component-id")
    parser.add_argument("--coefficients", default="-2,-1,-0.5,0,0.5,1,2")
    parser.add_argument("--phase", choices=["pre_gate", "post_gate"], default="pre_gate")
    parser.add_argument("--conditions", default="")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("intervention runner requires torch and transformers") from exc

    vector_np, vector_id = load_vector(args)
    vector_np = vector_np / max(float(np.linalg.norm(vector_np)), 1e-12)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if args.adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("loading --adapter requires peft") from exc
        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.to(args.device)
    model.eval()
    layers = discover_layers(model)
    if args.layer < 0 or args.layer >= len(layers):
        raise ValueError(f"layer {args.layer} outside 0..{len(layers)-1}")
    hidden_size = int(
        getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 0))
    )
    if hidden_size <= 0:
        raise ValueError("could not determine model hidden size")
    if vector_np.shape != (hidden_size,):
        raise ValueError(
            f"vector shape {vector_np.shape} does not match hidden size {hidden_size}"
        )
    vector = torch.from_numpy(vector_np).to(args.device, dtype=model.dtype)
    coefficients = [float(x) for x in args.coefficients.split(",")]
    selected_conditions = {
        x.strip() for x in args.conditions.split(",") if x.strip()
    }
    episodes = [
        episode
        for episode in iter_jsonl(args.input)
        if not selected_conditions or episode["condition"] in selected_conditions
    ]
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as out:
        for episode in episodes:
            messages = (
                episode["messages"]
                if args.phase == "pre_gate"
                else episode["messages_post_gate"]
            )
            prompt = render_prompt(tokenizer, messages)
            encoded = tokenizer(prompt, return_tensors="pt").to(args.device)

            for coefficient in coefficients:
                handle = None
                if coefficient != 0.0:
                    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                        hidden = output[0] if isinstance(output, tuple) else output
                        modified = hidden.clone()
                        modified[:, -1, :] += coefficient * vector
                        if isinstance(output, tuple):
                            return (modified, *output[1:])
                        return modified
                    handle = layers[args.layer].register_forward_hook(hook)
                try:
                    with torch.inference_mode():
                        generated = model.generate(
                            **encoded,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            pad_token_id=(
                                tokenizer.pad_token_id
                                if tokenizer.pad_token_id is not None
                                else tokenizer.eos_token_id
                            ),
                        )
                finally:
                    if handle is not None:
                        handle.remove()
                generated_text = tokenizer.decode(
                    generated[0, encoded["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                score = score_output(generated_text)
                out.write(
                    json.dumps(
                        {
                            "episode_id": episode["episode_id"],
                            "condition": episode["condition"],
                            "phase": args.phase,
                            "layer": args.layer,
                            "vector_id": vector_id,
                            "coefficient": coefficient,
                            "generated_text": generated_text,
                            **score,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
