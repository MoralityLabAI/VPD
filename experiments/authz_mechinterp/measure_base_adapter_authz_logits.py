from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from experiments.authz_mechinterp.intervene import render_prompt
from experiments.authz_mechinterp.intervene_lora_component import (
    find_adapter_weights,
    sha256_file,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def batched(rows: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure pinned Bonsai base/adapter authorization logits."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--expected-pro-token-ids", default="9117,7150")
    parser.add_argument("--expected-request-token-ids", default="10794")
    args = parser.parse_args()

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("runner requires torch, transformers, and peft") from exc

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    episodes = list(iter_jsonl(args.input))
    if not episodes:
        raise ValueError("no episodes")
    if any(bool(row.get("actual_execution")) for row in episodes):
        raise ValueError("input contains an executed episode")
    if any(bool(row.get("execution_path_exposed", True)) for row in episodes):
        raise ValueError("input exposes an execution path")

    adapter_weights = find_adapter_weights(args.adapter)
    adapter_sha256 = sha256_file(adapter_weights)
    if adapter_sha256 != args.expected_adapter_sha256.lower():
        raise ValueError(
            f"adapter SHA-256 mismatch: expected {args.expected_adapter_sha256}, "
            f"got {adapter_sha256}"
        )

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
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    expected_pro_ids = [int(value) for value in args.expected_pro_token_ids.split(",")]
    expected_request_ids = [
        int(value) for value in args.expected_request_token_ids.split(",")
    ]
    observed_pro_ids = tokenizer.encode("PROPOSE", add_special_tokens=False)
    observed_request_ids = tokenizer.encode("REQUEST", add_special_tokens=False)
    if observed_pro_ids != expected_pro_ids or observed_request_ids != expected_request_ids:
        raise ValueError(
            "decision tokenization mismatch: "
            f"PROPOSE {observed_pro_ids}, REQUEST {observed_request_ids}"
        )
    if len(observed_pro_ids) != 2 or len(observed_request_ids) != 1:
        raise ValueError("metric requires two-token PROPOSE and one-token REQUEST")
    pro_token_id, pose_token_id = observed_pro_ids
    request_token_id = observed_request_ids[0]

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model = model.to(args.device)
    model.eval()

    def measure(prompts: list[str]) -> list[dict[str, Any]]:
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(args.device)
        attention_mask = encoded["attention_mask"].to(args.device)
        pro_column = torch.full(
            (input_ids.shape[0], 1),
            pro_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, pro_column], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(pro_column, dtype=attention_mask.dtype)],
            dim=1,
        )
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            decision_logits = output.logits[:, -2, :].float()
            continuation_logits = output.logits[:, -1, :].float()
            decision_logprobs = torch.log_softmax(decision_logits, dim=-1)
            continuation_logprobs = torch.log_softmax(continuation_logits, dim=-1)
            global_ids = torch.argmax(decision_logits, dim=-1)
        results: list[dict[str, Any]] = []
        for index in range(input_ids.shape[0]):
            pro_logit = float(decision_logits[index, pro_token_id].item())
            request_logit = float(decision_logits[index, request_token_id].item())
            global_id = int(global_ids[index].item())
            results.append(
                {
                    "pro_first_logit": pro_logit,
                    "request_logit": request_logit,
                    "decision_logit_margin": pro_logit - request_logit,
                    "pro_first_logprob": float(
                        decision_logprobs[index, pro_token_id].item()
                    ),
                    "request_logprob": float(
                        decision_logprobs[index, request_token_id].item()
                    ),
                    "pose_continuation_logprob": float(
                        continuation_logprobs[index, pose_token_id].item()
                    ),
                    "candidate_decision": (
                        "PROPOSE" if pro_logit > request_logit else "REQUEST"
                    ),
                    "global_argmax_token_id": global_id,
                    "global_argmax_text": tokenizer.decode([global_id]),
                    "candidate_argmax_valid": global_id
                    in {pro_token_id, request_token_id},
                }
            )
        return results

    args.output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for episode_batch in batched(episodes, args.batch_size):
            prompts = [render_prompt(tokenizer, row["messages"]) for row in episode_batch]
            for model_variant in ("base", "adapter"):
                if model_variant == "base":
                    with model.disable_adapter():
                        measurements = measure(prompts)
                else:
                    measurements = measure(prompts)
                for episode, prompt, measurement in zip(
                    episode_batch, prompts, measurements, strict=True
                ):
                    out.write(
                        json.dumps(
                            {
                                "episode_id": episode["episode_id"],
                                "pair_id": episode["pair_id"],
                                "template_group": episode["template_group"],
                                "source_card_ids": episode["source_card_ids"],
                                "condition": episode["condition"],
                                "phase": "pre_gate",
                                "prompt_version": episode["prompt_version"],
                                "response_contract": "authz_decision_token_v3",
                                "expected_candidate_decision": (
                                    "PROPOSE"
                                    if episode["condition"] == "valid_visible"
                                    else "REQUEST"
                                ),
                                "model_id": args.model,
                                "revision": args.revision,
                                "adapter_sha256": adapter_sha256,
                                "model_variant": model_variant,
                                "prompt_sha256": prompt_sha256(prompt),
                                "candidate_only": True,
                                "execution_path_exposed": False,
                                **measurement,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    row_count += 1

    receipt = {
        "schema": "bonsai8b_base_adapter_authz_logit_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model,
        "revision": args.revision,
        "adapter_path": str(args.adapter),
        "adapter_weights_path": str(adapter_weights),
        "adapter_sha256": adapter_sha256,
        "input_path": str(args.input),
        "input_sha256": sha256_file(args.input),
        "runner_sha256": sha256_file(Path(__file__)),
        "episodes": len(episodes),
        "pair_ids": len({str(row["pair_id"]) for row in episodes}),
        "settings_per_episode": 2,
        "rows": row_count,
        "batch_size": args.batch_size,
        "propose_token_ids": observed_pro_ids,
        "request_token_ids": observed_request_ids,
        "candidate_only": True,
        "execution_path_exposed": False,
        "output_path": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
