from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from experiments.authz_mechinterp.activation_categories import activation_category

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.compute_vpd_effect_shard import find_adapter_weights
from experiments.authz_mechinterp.merge_vpd_effect_shards import write_deterministic_npz
from experiments.authz_mechinterp.activation_category_split_contract import (
    verify_activation_category_split_role,
)
from experiments.authz_mechinterp.activation_shards import (
    ActivationShardWriter,
    build_sharded_activation_artifact,
    verify_sharded_activation_artifact,
)


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
    candidates = (
        "model.layers",
        "model.decoder.layers",
        "transformer.h",
        "gpt_neox.layers",
        "base_model.model.model.layers",
        "base_model.model.model.decoder.layers",
        "base_model.model.transformer.h",
        "base_model.model.gpt_neox.layers",
    )
    for path in candidates:
        try:
            layers = get_attr_path(model, path)
            if len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            continue
    raise ValueError("could not locate transformer block list")


def find_last_subsequence(sequence: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(sequence):
        return None
    for start in range(len(sequence) - len(pattern), -1, -1):
        if sequence[start : start + len(pattern)] == pattern:
            return start + len(pattern) - 1
    return None


def extract_hidden(output: Any) -> Any:
    hidden = output[0] if isinstance(output, tuple) else output
    if not hasattr(hidden, "ndim") or hidden.ndim != 3:
        raise ValueError("layer hook expected [batch, sequence, hidden] output")
    return hidden


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


def validate_sae_training_window_request(
    *,
    window_size: int,
    split_role: str | None,
    phases: Sequence[str],
    adapter: str | None,
) -> None:
    if window_size < 0:
        raise ValueError("SAE training window must be nonnegative")
    if window_size == 0:
        return
    if split_role not in {"sae_training", "calibration"}:
        raise ValueError(
            "SAE prompt-window capture requires the sae_training or calibration split"
        )
    if list(phases) != ["pre_gate"]:
        raise ValueError("SAE training window capture requires only the pre_gate phase")
    if adapter is not None:
        raise ValueError("SAE training window capture requires base-model activations")


def load_capture_resource_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != "sch.vpd-compartment-capture-resource-plan.v1":
        raise ValueError("unsupported compartment capture resource plan schema")
    body = dict(plan)
    plan_id = str(body.pop("resource_plan_id", ""))
    if content_sha256(body) != plan_id:
        raise ValueError("capture resource plan content address mismatch")
    if plan.get("status") != "sharded_capture_and_streaming_training_required":
        raise ValueError("capture resource plan has an unsupported status")
    return plan


def validate_capture_resource_role(
    plan: dict[str, Any],
    *,
    split_role: str,
    episode_count: int,
    rows_per_episode: int,
) -> dict[str, Any]:
    role = plan.get("capture", {}).get("roles", {}).get(split_role)
    if not isinstance(role, dict):
        raise ValueError("capture resource plan omits the requested split role")
    if episode_count != int(role.get("episodes", -1)):
        raise ValueError("dataset episode count differs from its resource-plan role")
    if episode_count * rows_per_episode != int(role.get("activation_rows", -1)):
        raise ValueError(
            "dataset activation row count differs from its resource-plan role"
        )
    return role


def prompt_window_positions(prompt_length: int, window_size: int) -> list[int]:
    if prompt_length < 1:
        raise ValueError("prompt must contain at least one token")
    if window_size < 1:
        raise ValueError("prompt window must contain at least one token")
    if prompt_length < window_size:
        raise ValueError(
            f"prompt has {prompt_length} tokens, fewer than required window {window_size}"
        )
    return list(range(prompt_length - window_size, prompt_length))


def prompt_window_metadata_rows(
    *,
    episode: dict[str, Any],
    split_provenance: dict[str, Any],
    model_variant: str,
    prompt_length: int,
    positions: dict[str, int | None],
    window_size: int,
    first_row_index: int,
) -> list[dict[str, Any]]:
    token_positions = prompt_window_positions(prompt_length, window_size)
    proposal_start = prompt_length - 1
    rows: list[dict[str, Any]] = []
    for ordinal, token_position in enumerate(token_positions):
        rows.append(
            {
                "row_index": first_row_index + ordinal,
                "episode_id": episode["episode_id"],
                "prompt_version": episode.get("prompt_version", "v1"),
                "model_variant": model_variant,
                "pair_id": episode.get("pair_id", episode["episode_id"]),
                "condition": episode["condition"],
                "activation_category": activation_category(episode),
                "template_group": episode["template_group"],
                "phase": "pre_gate",
                "prompt_tokens": prompt_length,
                "total_tokens": prompt_length,
                "proposal_start_position": positions.get("proposal_start"),
                "sampled_token_position": token_position,
                "sampled_token_ordinal": ordinal,
                "token_offset_from_proposal_start": token_position - proposal_start,
                "is_proposal_start": token_position == proposal_start,
                "source_episode_sha256": content_sha256(episode),
                "activation_category_split_id": split_provenance[
                    "activation_category_split_id"
                ],
                "activation_category_split_role": split_provenance[
                    "activation_category_split_role"
                ],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--activation-category-split-manifest", type=Path)
    parser.add_argument(
        "--activation-category-split-role",
        choices=["sae_training", "calibration", "final_evaluation"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, help="base or merged model path")
    parser.add_argument("--adapter", help="optional PEFT adapter path")
    parser.add_argument("--tokenizer", help="optional tokenizer path; defaults to --model")
    parser.add_argument("--revision")
    parser.add_argument("--model-variant", default="model")
    parser.add_argument("--layers", default="0,0.5,0.75,0.875")
    parser.add_argument("--phases", default="pre_gate,post_gate")
    parser.add_argument(
        "--sae-training-window",
        type=int,
        default=0,
        help=(
            "capture this exact number of prompt-token states per episode, ending "
            "at proposal_start; requires the base model and a sae_training or "
            "calibration pre_gate split"
        ),
    )
    parser.add_argument("--capture-resource-plan", type=Path)
    parser.add_argument("--activation-shard-rows", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    if (args.activation_category_split_manifest is None) != (
        args.activation_category_split_role is None
    ):
        raise ValueError("activation capture split provenance requires manifest and role")
    split_provenance = (
        None
        if args.activation_category_split_manifest is None
        else verify_activation_category_split_role(
            args.input,
            args.activation_category_split_manifest,
            args.activation_category_split_role,
        )
    )
    phases = [x.strip() for x in args.phases.split(",") if x.strip()]
    validate_sae_training_window_request(
        window_size=args.sae_training_window,
        split_role=args.activation_category_split_role,
        phases=phases,
        adapter=args.adapter,
    )
    resource_plan: dict[str, Any] | None = None
    resource_role: dict[str, Any] | None = None
    if args.sae_training_window > 0:
        if args.capture_resource_plan is None or args.activation_shard_rows < 1:
            raise ValueError(
                "SAE training window capture requires a resource plan and shards"
            )
        if args.max_episodes != 0:
            raise ValueError("capacity-bound SAE capture requires the complete split role")
        resource_plan = load_capture_resource_plan(args.capture_resource_plan)
        if split_provenance is None or resource_plan["source"]["split_id"] != (
            split_provenance["activation_category_split_id"]
        ):
            raise ValueError("capture resource plan and dataset use different split IDs")
        if int(resource_plan["profile"]["activation_rows_per_episode"]) != (
            args.sae_training_window
        ):
            raise ValueError("capture window differs from the resource plan")
        if int(resource_plan["sharding"]["rows_per_full_shard"]) != (
            args.activation_shard_rows
        ):
            raise ValueError("activation shard size differs from the resource plan")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("capture requires torch and transformers") from exc

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
    layer_specs = [float(x.strip()) for x in args.layers.split(",") if x.strip()]
    layer_indices = sorted(
        {
            int(round(spec * (len(layers) - 1))) if 0.0 <= spec <= 1.0 else int(spec)
            for spec in layer_specs
        }
    )
    for index in layer_indices:
        if index < 0 or index >= len(layers):
            raise ValueError(f"layer index {index} outside 0..{len(layers)-1}")

    hidden_size = int(
        getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 0))
    )
    if hidden_size <= 0:
        raise ValueError("could not determine model hidden size")
    if resource_plan is not None:
        if [int(value) for value in resource_plan["capture"]["layer_indices"]] != (
            layer_indices
        ):
            raise ValueError("captured layers differ from the resource plan")
        if int(resource_plan["profile"]["hidden_size"]) != hidden_size:
            raise ValueError("model hidden size differs from the resource plan")
        if str(resource_plan["capture"]["dtype"]) != args.dtype:
            raise ValueError("capture dtype differs from the resource plan")

    marker_map = {
        "auth_end": "</AUTH_BLOCK>",
        "task_end": "<TASK_END>",
        "gate_end": "</GATE_RESULT>",
    }
    anchor_names = [
        "auth_end",
        "task_end",
        "proposal_start",
        "gate_end",
        "post_gate_start",
    ]
    activation_rows: dict[str, list[np.ndarray]] = defaultdict(list)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = args.output_dir / "generations.jsonl"
    metadata_path = args.output_dir / "metadata.jsonl"
    metadata_row_count = 0
    shard_writers: dict[str, ActivationShardWriter] = {}
    if resource_plan is not None:
        if args.dtype != "float16":
            raise ValueError("sharded SAE capture currently requires float16 storage")
        shard_writers = {
            f"layer_{layer_index}__proposal_start": ActivationShardWriter(
                output_dir=args.output_dir,
                key=f"layer_{layer_index}__proposal_start",
                hidden_size=hidden_size,
                dtype=np.float16,
                shard_rows=args.activation_shard_rows,
            )
            for layer_index in layer_indices
        }

    episodes = list(iter_jsonl(args.input))
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    if resource_plan is not None:
        resource_role = validate_capture_resource_role(
            resource_plan,
            split_role=str(args.activation_category_split_role),
            episode_count=len(episodes),
            rows_per_episode=args.sae_training_window,
        )

    with (
        outputs_path.open("w", encoding="utf-8") as generations_out,
        metadata_path.open("w", encoding="utf-8", newline="\n") as metadata_out,
    ):
        for episode in episodes:
            for phase in phases:
                if phase == "pre_gate":
                    messages = episode["messages"]
                    generation_anchor = "proposal_start"
                elif phase == "post_gate":
                    messages = episode["messages_post_gate"]
                    generation_anchor = "post_gate_start"
                else:
                    raise ValueError(f"unknown phase: {phase}")

                prompt_text = render_prompt(tokenizer, messages)
                encoded = tokenizer(prompt_text, return_tensors="pt")
                input_ids = encoded["input_ids"].to(args.device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(args.device)
                prompt_len = int(input_ids.shape[1])

                if args.sae_training_window > 0:
                    full_ids = input_ids
                else:
                    with torch.inference_mode():
                        generated_ids = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            pad_token_id=(
                                tokenizer.pad_token_id
                                if tokenizer.pad_token_id is not None
                                else tokenizer.eos_token_id
                            ),
                        )
                    full_ids = generated_ids
                full_sequence = full_ids[0].tolist()
                generated_text = tokenizer.decode(
                    full_ids[0, prompt_len:], skip_special_tokens=True
                )

                positions: dict[str, int | None] = {}
                for anchor_name, marker in marker_map.items():
                    marker_ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
                    positions[anchor_name] = find_last_subsequence(
                        full_sequence[:prompt_len], marker_ids
                    )
                positions[generation_anchor] = (
                    prompt_len - 1 if prompt_len > 0 else None
                )
                if generation_anchor == "proposal_start":
                    positions["post_gate_start"] = None
                else:
                    positions["proposal_start"] = None

                if args.sae_training_window > 0:
                    assert split_provenance is not None
                    token_positions = prompt_window_positions(
                        prompt_len, args.sae_training_window
                    )
                    captured_windows: dict[int, np.ndarray] = {}
                    hooks = []
                    for layer_index in layer_indices:
                        def make_window_hook(index: int):
                            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                                hidden = extract_hidden(output)
                                captured_windows[index] = (
                                    hidden[0, token_positions]
                                    .detach()
                                    .to(torch.float32)
                                    .cpu()
                                    .numpy()
                                )
                            return hook
                        hooks.append(
                            layers[layer_index].register_forward_hook(
                                make_window_hook(layer_index)
                            )
                        )
                    try:
                        with torch.inference_mode():
                            model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=False,
                            )
                    finally:
                        for hook in hooks:
                            hook.remove()
                    for layer_index in layer_indices:
                        matrix = captured_windows.get(layer_index)
                        if matrix is None or matrix.shape != (
                            args.sae_training_window,
                            hidden_size,
                        ):
                            raise ValueError(
                                "SAE prompt-window hook did not capture the exact matrix"
                            )
                        key = f"layer_{layer_index}__proposal_start"
                        shard_writers[key].append(matrix)
                    window_rows = prompt_window_metadata_rows(
                        episode=episode,
                        split_provenance=split_provenance,
                        model_variant=args.model_variant,
                        prompt_length=prompt_len,
                        positions=positions,
                        window_size=args.sae_training_window,
                        first_row_index=metadata_row_count,
                    )
                    for row in window_rows:
                        metadata_out.write(json.dumps(row, sort_keys=True) + "\n")
                    metadata_row_count += len(window_rows)
                    generations_out.write(
                        json.dumps(
                            {
                                "episode_id": episode["episode_id"],
                                "pair_id": episode.get(
                                    "pair_id", episode["episode_id"]
                                ),
                                "activation_category": activation_category(episode),
                                "phase": "pre_gate",
                                "prompt_tokens": prompt_len,
                                "generated_tokens": 0,
                                "prompt_text": prompt_text,
                                "generated_text": "",
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    continue

                captured: dict[tuple[int, str], np.ndarray] = {}
                hooks = []
                for layer_index in layer_indices:
                    def make_hook(index: int):
                        def hook(_module: Any, _inputs: Any, output: Any) -> None:
                            hidden = extract_hidden(output)
                            for anchor_name, token_index in positions.items():
                                if token_index is None or token_index >= hidden.shape[1]:
                                    continue
                                captured[(index, anchor_name)] = (
                                    hidden[0, token_index]
                                    .detach()
                                    .to(torch.float32)
                                    .cpu()
                                    .numpy()
                                )
                        return hook
                    hooks.append(layers[layer_index].register_forward_hook(make_hook(layer_index)))

                try:
                    with torch.inference_mode():
                        model(input_ids=full_ids, use_cache=False)
                finally:
                    for hook in hooks:
                        hook.remove()

                for layer_index in layer_indices:
                    for anchor_name in anchor_names:
                        key = f"layer_{layer_index}__{anchor_name}"
                        vector = captured.get((layer_index, anchor_name))
                        if vector is None:
                            vector = np.full((hidden_size,), np.nan, dtype=np.float32)
                        activation_rows[key].append(vector.astype(np.float16))

                metadata = {
                    "row_index": metadata_row_count,
                    "episode_id": episode["episode_id"],
                    "prompt_version": episode.get("prompt_version", "v1"),
                    "model_variant": args.model_variant,
                    "pair_id": episode.get("pair_id", episode["episode_id"]),
                    "condition": episode["condition"],
                    "activation_category": activation_category(episode),
                    "template_group": episode["template_group"],
                    "phase": phase,
                    "prompt_tokens": prompt_len,
                    "total_tokens": int(full_ids.shape[1]),
                    "positions": positions,
                    "factors": episode["factors"],
                    "labels": episode["labels"],
                    "observed_gate_decision": episode["observed_gate_decision"],
                    "actual_execution": episode["actual_execution"],
                    **(
                        {}
                        if split_provenance is None
                        else {
                            "activation_category_split_id": split_provenance[
                                "activation_category_split_id"
                            ],
                            "activation_category_split_role": split_provenance[
                                "activation_category_split_role"
                            ],
                        }
                    ),
                }
                metadata_out.write(json.dumps(metadata, sort_keys=True) + "\n")
                metadata_row_count += 1
                generations_out.write(
                    json.dumps(
                        {
                            **metadata,
                            "prompt_text": prompt_text,
                            "generated_text": generated_text,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    if resource_plan is None:
        activations_path = args.output_dir / "activations.npz"
        write_deterministic_npz(
            activations_path,
            {key: np.stack(rows, axis=0) for key, rows in activation_rows.items()},
        )
        activations_artifact: dict[str, Any] = {
            "path": activations_path.name,
            "sha256": sha256_file(activations_path),
        }
        activation_keys = sorted(activation_rows)
    else:
        activations_artifact = build_sharded_activation_artifact(
            {key: writer.finalize() for key, writer in shard_writers.items()}
        )
        verification = verify_sharded_activation_artifact(
            args.output_dir, activations_artifact
        )
        if int(verification["rows"]) != metadata_row_count:
            raise ValueError("activation shards and metadata have different row counts")
        activation_keys = list(activations_artifact["activation_keys"])
    adapter_sha256 = (
        sha256_file(find_adapter_weights(Path(args.adapter))) if args.adapter else None
    )
    sae_training_sampling = (
        None
        if args.sae_training_window == 0
        else {
            "schema": "sch.vpd-sae-training-sampling.v1",
            "mode": "prompt_context_window",
            "generated_tokens": 0,
            "window_end_anchor": "proposal_start",
            "activation_rows_per_episode": args.sae_training_window,
            "token_offset_start": 1 - args.sae_training_window,
            "token_offset_end": 0,
            "exact_rows_per_episode": True,
            "episode_count": len(episodes),
            "activation_row_count": metadata_row_count,
            "episode_id_set_sha256": content_sha256(
                sorted(str(episode["episode_id"]) for episode in episodes)
            ),
            "pair_id_set_sha256": content_sha256(
                sorted(
                    {
                        str(episode.get("pair_id", episode["episode_id"]))
                        for episode in episodes
                    }
                )
            ),
            "resource_plan_id": (
                resource_plan["resource_plan_id"] if resource_plan else None
            ),
            "resource_plan_role": args.activation_category_split_role,
            "capacity_plan_id": (
                resource_plan["source"]["capacity_plan_id"]
                if resource_plan
                else None
            ),
            "activation_storage": "npy_row_shards",
            "activation_shard_rows": args.activation_shard_rows,
        }
    )
    manifest_body = {
        "schema": "sch.vpd-activation-capture-manifest.v1",
        "source": {
            "dataset_sha256": sha256_file(args.input),
            "runner_sha256": sha256_file(Path(__file__)),
            "activation_category_split": split_provenance,
            "capture_resource_plan": (
                None
                if resource_plan is None
                else {
                    "resource_plan_id": resource_plan["resource_plan_id"],
                    "resource_plan_sha256": sha256_file(args.capture_resource_plan),
                    "capacity_plan_id": resource_plan["source"]["capacity_plan_id"],
                    "pack_id": resource_plan["source"]["pack_id"],
                    "split_id": resource_plan["source"]["split_id"],
                    "capture_role": args.activation_category_split_role,
                    "role_episodes": int(resource_role["episodes"]),
                    "role_activation_rows": int(resource_role["activation_rows"]),
                }
            ),
        },
        "model": {
            "model_id": args.model,
            "revision": args.revision,
            "tokenizer_id": args.tokenizer or args.model,
            "model_variant": args.model_variant,
            "adapter_sha256": adapter_sha256,
        },
        "capture": {
            "n_layers": len(layers),
            "captured_layers": layer_indices,
            "hidden_size": hidden_size,
            "rows": metadata_row_count,
            "activation_keys": activation_keys,
            "phases": phases,
            "max_new_tokens": (
                0 if args.sae_training_window > 0 else args.max_new_tokens
            ),
            "dtype": args.dtype,
            **(
                {}
                if sae_training_sampling is None
                else {"sae_training_sampling": sae_training_sampling}
            ),
        },
        "artifacts": {
            "activations": activations_artifact,
            "metadata": {
                "path": metadata_path.name,
                "sha256": sha256_file(metadata_path),
            },
            "generations": {
                "path": outputs_path.name,
                "sha256": sha256_file(outputs_path),
            },
        },
    }
    write_json(
        args.output_dir / "capture_manifest.json",
        {**manifest_body, "capture_id": content_sha256(manifest_body)},
    )


if __name__ == "__main__":
    main()
