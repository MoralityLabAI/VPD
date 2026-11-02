from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    stable_hash,
    write_json,
)
from experiments.authz_mechinterp.activation_categories import (
    categorize_episodes,
    category_contract,
)
from experiments.authz_mechinterp.merge_vpd_effect_shards import write_deterministic_npz
from experiments.authz_mechinterp.activation_category_split_contract import (
    verify_activation_category_split_role,
)
from experiments.authz_mechinterp.effect_uncertainty import (
    condition_robustness_contract,
    countsketch,
    template_robustness_contract,
    uncertainty_contract,
)


EFFECT_SPACE_RE = re.compile(r"^layer_(?P<layer>\d+)__(?P<anchor>proposal_start)$")
MINIMUM_PRODUCTION_COUNT_SKETCH_DIMENSION = 256
EFFECT_SPLIT_ROLES = {"discovery", "calibration", "cluster_validation"}


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


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError(f"dataset contains null bytes: {path}")
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def verify_dataset_split_role(
    dataset_path: Path, split_manifest_path: Path, split_role: str
) -> dict[str, Any]:
    if split_role not in EFFECT_SPLIT_ROLES:
        raise ValueError(
            f"{split_role.replace('_', '-')} rows cannot be used for effect threshold selection"
        )
    return verify_activation_category_split_role(
        dataset_path, split_manifest_path, split_role
    )


def verify_packet(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256.lower():
        raise ValueError("packet file hash does not match coordinator identity")
    packet = json.loads(path.read_text(encoding="utf-8"))
    packet_body = json.loads(json.dumps(packet))
    packet_id = str(packet_body.pop("packet_id"))
    distribution = dict(packet_body["distribution"])
    distribution.pop("assigned_worker", None)
    packet_body["distribution"] = distribution
    if content_sha256(packet_body) != packet_id:
        raise ValueError("packet content address is invalid")
    return packet


def select_episodes(
    rows: list[dict[str, Any]], *, maximum: int, seed: int
) -> list[dict[str, Any]]:
    ids = [str(row["episode_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset episode IDs are not unique")
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(str(row["template_group"]), str(row["condition"]))].append(row)
    for key in strata:
        strata[key].sort(key=lambda row: stable_hash(seed, "episode", str(row["episode_id"])))
    stratum_order = sorted(
        strata,
        key=lambda key: stable_hash(seed, "stratum", key[0], key[1]),
    )
    selected: list[dict[str, Any]] = []
    offset = 0
    target = len(rows) if maximum <= 0 else min(maximum, len(rows))
    while len(selected) < target:
        progressed = False
        for key in stratum_order:
            if offset < len(strata[key]) and len(selected) < target:
                selected.append(strata[key][offset])
                progressed = True
        if not progressed:
            break
        offset += 1
    return selected


def find_adapter_weights(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "adapter_model.safetensors"
    if not candidate.is_file():
        raise FileNotFoundError(f"adapter weights not found under {path}")
    return candidate


def factor_digest(tensor: Any) -> str:
    array = tensor.detach().to("cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def resolve_lora_factor(model: Any, component_id: str, adapter_name: str) -> tuple[Any, Any, int]:
    try:
        module_path, rank_text = component_id.rsplit(":rank", 1)
        rank_index = int(rank_text)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid LoRA component ID: {component_id}") from exc
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith(module_path)
        and hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one PEFT module for {module_path}, found {len(matches)}")
    _, module = matches[0]
    if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
        raise ValueError(f"adapter lane {adapter_name} is absent from {module_path}")
    a_weight = module.lora_A[adapter_name].weight
    b_weight = module.lora_B[adapter_name].weight
    if not 0 <= rank_index < int(a_weight.shape[0]) or rank_index >= int(b_weight.shape[1]):
        raise ValueError(f"rank index is outside LoRA factors: {component_id}")
    return a_weight, b_weight, rank_index


def prepare_batches(
    *,
    tokenizer: Any,
    episodes: list[dict[str, Any]],
    batch_size: int,
    device: str,
) -> list[tuple[Any, Any]]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [render_prompt(tokenizer, row["messages"]) for row in episodes]
    batches: list[tuple[Any, Any]] = []
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        batches.append((input_ids.to("cpu"), attention_mask.to("cpu")))
    return batches


def capture_effect_space(
    *, model: Any, layer: Any, input_ids: Any, attention_mask: Any, device: str
) -> np.ndarray:
    import torch

    captured: list[Any] = []

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3:
            raise ValueError("effect-space layer output must be batch x sequence x hidden")
        captured.append(hidden[:, -1, :].detach().to(torch.float32).cpu())

    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                use_cache=False,
            )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("effect-space hook did not capture exactly one layer output")
    return captured[0].numpy()


def component_effect_samples(
    *,
    model: Any,
    layer: Any,
    b_weight: Any,
    rank_index: int,
    batches: list[tuple[Any, Any]],
    epsilon: float,
    device: str,
) -> np.ndarray:
    import torch

    original = b_weight[:, rank_index].detach().clone()
    original_digest = factor_digest(original)
    plus_rows: np.ndarray | None = None
    minus_rows: np.ndarray | None = None
    try:
        for multiplier, lane in ((1.0 + epsilon, "plus"), (1.0 - epsilon, "minus")):
            with torch.no_grad():
                b_weight[:, rank_index].copy_(original * multiplier)
            lane_rows: list[np.ndarray] = []
            for input_ids, attention_mask in batches:
                values = capture_effect_space(
                    model=model,
                    layer=layer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    device=device,
                )
                lane_rows.append(values.astype(np.float64))
            current_rows = np.concatenate(lane_rows, axis=0) if lane_rows else None
            if lane == "plus":
                plus_rows = current_rows
            else:
                minus_rows = current_rows
                if plus_rows is not None and minus_rows is not None and minus_rows.shape != plus_rows.shape:
                    raise ValueError("signed perturbation lanes have different row counts")
    finally:
        with torch.no_grad():
            b_weight[:, rank_index].copy_(original)
    if factor_digest(b_weight[:, rank_index]) != original_digest:
        raise ValueError("LoRA factor restoration failed")
    if plus_rows is None or minus_rows is None or len(plus_rows) <= 0:
        raise ValueError("signed perturbation produced no activation rows")
    return ((plus_rows - minus_rows) / (2.0 * epsilon)).astype(np.float32)


def component_effect(**kwargs: Any) -> np.ndarray:
    return component_effect_samples(**kwargs).mean(axis=0).astype(np.float32)


def pair_uncertainty_blocks(
    category_rows: dict[str, list[dict[str, Any]]], *, requested_blocks: int, seed: int
) -> tuple[dict[str, int], int]:
    if requested_blocks < 2:
        raise ValueError("uncertainty block count must be at least two")
    pair_sets: dict[str, set[str]] = {}
    for category, rows in category_rows.items():
        pair_ids = [str(row.get("pair_id", "")) for row in rows]
        if any(not pair_id for pair_id in pair_ids):
            raise ValueError("category uncertainty requires pair_id on every episode")
        pair_sets[category] = set(pair_ids)
    reference = next(iter(pair_sets.values()))
    if any(pair_ids != reference for pair_ids in pair_sets.values()):
        raise ValueError(
            "category uncertainty requires the same selected pair set in every category; "
            "use the full split role rather than an episode-truncated selection"
        )
    block_count = min(requested_blocks, len(reference))
    if block_count < 2:
        raise ValueError("category uncertainty requires at least two prompt pairs")
    ordered = sorted(reference, key=lambda pair_id: stable_hash(seed, "pair-block", pair_id))
    return {pair_id: index % block_count for index, pair_id in enumerate(ordered)}, block_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute one packet's exact-LoRA signed activation-effect shard."
    )
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--expected-packet-sha256", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--activation-category-split-manifest", type=Path)
    parser.add_argument(
        "--activation-category-split-role",
        choices=sorted(EFFECT_SPLIT_ROLES),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--output-shard", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--adapter-name", default="default")
    parser.add_argument("--maximum-episodes", type=int, default=0)
    parser.add_argument("--selection-seed", type=int, default=1707)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--uncertainty-blocks", type=int, default=8)
    parser.add_argument(
        "--uncertainty-sketch-dimension",
        type=int,
        default=MINIMUM_PRODUCTION_COUNT_SKETCH_DIMENSION,
    )
    parser.add_argument("--minimum-condition-coverage", type=float, default=0.75)
    parser.add_argument("--minimum-condition-direction-cosine", type=float, default=0.50)
    parser.add_argument("--minimum-condition-relative-norm", type=float, default=0.25)
    parser.add_argument("--minimum-template-group-coverage", type=float, default=0.75)
    parser.add_argument(
        "--minimum-template-group-direction-cosine", type=float, default=0.50
    )
    parser.add_argument(
        "--minimum-template-group-relative-norm", type=float, default=0.25
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.epsilon < 1.0:
        raise ValueError("epsilon must be between zero and one")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.uncertainty_sketch_dimension < MINIMUM_PRODUCTION_COUNT_SKETCH_DIMENSION:
        raise ValueError(
            "production effect robustness requires CountSketch dimension at least "
            f"{MINIMUM_PRODUCTION_COUNT_SKETCH_DIMENSION}"
        )
    condition_contract = condition_robustness_contract(
        minimum_coverage=args.minimum_condition_coverage,
        minimum_direction_cosine=args.minimum_condition_direction_cosine,
        minimum_relative_norm=args.minimum_condition_relative_norm,
    )
    template_contract = template_robustness_contract(
        minimum_coverage=args.minimum_template_group_coverage,
        minimum_direction_cosine=args.minimum_template_group_direction_cosine,
        minimum_relative_norm=args.minimum_template_group_relative_norm,
    )
    if (args.activation_category_split_manifest is None) != (
        args.activation_category_split_role is None
    ):
        raise ValueError("effect split provenance requires both manifest and role")
    split_provenance = (
        {}
        if args.activation_category_split_manifest is None
        else verify_dataset_split_role(
            args.dataset,
            args.activation_category_split_manifest,
            args.activation_category_split_role,
        )
    )
    packet = verify_packet(args.packet, args.expected_packet_sha256)
    if args.worker_id != packet["distribution"].get("assigned_worker"):
        raise ValueError("worker is not assigned to this packet")
    effect_space_id = str(packet["effect_refinement"]["effect_space_id"])
    match = EFFECT_SPACE_RE.fullmatch(effect_space_id)
    if match is None:
        raise ValueError(f"unsupported effect space: {effect_space_id}")
    layer_index = int(match.group("layer"))
    episodes = select_episodes(
        list(iter_jsonl(args.dataset)),
        maximum=args.maximum_episodes,
        seed=args.selection_seed,
    )
    if not episodes:
        raise ValueError("effect dataset selection is empty")
    adapter_weights = find_adapter_weights(args.adapter)
    adapter_sha256 = sha256_file(adapter_weights)
    if adapter_sha256 != args.expected_adapter_sha256.lower():
        raise ValueError("adapter weights do not match frozen identity")
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("effect shard computation requires torch, transformers, and peft") from exc
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter)).to(args.device)
    model.eval()
    layers = discover_layers(model)
    if not 0 <= layer_index < len(layers):
        raise ValueError(f"effect-space layer is outside model: {layer_index}")
    category_rows = categorize_episodes(episodes)
    category_names = sorted(category_rows)
    pair_blocks, uncertainty_blocks = pair_uncertainty_blocks(
        category_rows,
        requested_blocks=args.uncertainty_blocks,
        seed=args.selection_seed,
    )
    uncertainty_seed = stable_hash(
        args.selection_seed, "category-effect-countsketch", effect_space_id
    ) % (2**32)
    category_batches = {
        category: prepare_batches(
            tokenizer=tokenizer,
            episodes=category_rows[category],
            batch_size=args.batch_size,
            device=args.device,
        )
        for category in category_names
    }
    category_counts = np.asarray(
        [len(category_rows[category]) for category in category_names], dtype=np.int64
    )
    condition_cells = sorted(
        {
            (category, str(row.get("condition", "")))
            for category in category_names
            for row in category_rows[category]
        }
    )
    if any(not condition for _, condition in condition_cells):
        raise ValueError("category condition robustness requires condition on every episode")
    condition_ids = [f"{category}::{condition}" for category, condition in condition_cells]
    condition_index = {
        (category, condition): index
        for index, (category, condition) in enumerate(condition_cells)
    }
    condition_counts = np.zeros(len(condition_cells), dtype=np.int64)
    for category in category_names:
        for row in category_rows[category]:
            condition_counts[
                condition_index[(category, str(row["condition"]))]
            ] += 1
    if np.any(condition_counts <= 0):
        raise ValueError("category condition robustness has empty condition support")
    template_cells = sorted(
        {
            (category, str(row.get("template_group", "")))
            for category in category_names
            for row in category_rows[category]
        }
    )
    if any(not template_group for _, template_group in template_cells):
        raise ValueError("category template robustness requires template_group on every episode")
    template_groups = sorted({template_group for _, template_group in template_cells})
    if any(
        {group for cell_category, group in template_cells if cell_category == category}
        != set(template_groups)
        for category in category_names
    ):
        raise ValueError("category template robustness requires every category in every template")
    template_cell_ids = [
        f"{category}::{template_group}" for category, template_group in template_cells
    ]
    template_cell_index = {
        cell: index for index, cell in enumerate(template_cells)
    }
    template_cell_counts = np.zeros(len(template_cells), dtype=np.int64)
    for category in category_names:
        for row in category_rows[category]:
            template_cell_counts[
                template_cell_index[(category, str(row["template_group"]))]
            ] += 1
    if np.any(template_cell_counts <= 0):
        raise ValueError("category template robustness has empty template support")
    component_ids = sorted(str(value) for value in packet["component_ids"])
    effects: list[np.ndarray] = []
    effects_by_category: list[np.ndarray] = []
    uncertainty_sums_by_component: list[np.ndarray] = []
    condition_sums_by_component: list[np.ndarray] = []
    template_sums_by_component: list[np.ndarray] = []
    uncertainty_counts = np.zeros(
        (len(category_names), uncertainty_blocks), dtype=np.int64
    )
    for category_index, category in enumerate(category_names):
        for row in category_rows[category]:
            uncertainty_counts[category_index, pair_blocks[str(row["pair_id"])]] += 1
    if np.any(uncertainty_counts <= 0):
        raise ValueError("category uncertainty produced an empty pair block")
    for component_id in component_ids:
        _, b_weight, rank_index = resolve_lora_factor(
            model, component_id, args.adapter_name
        )
        category_samples = [
                component_effect_samples(
                    model=model,
                    layer=layers[layer_index],
                    b_weight=b_weight,
                    rank_index=rank_index,
                    batches=category_batches[category],
                    epsilon=args.epsilon,
                    device=args.device,
                ) for category in category_names
        ]
        category_effects = np.stack(
            [samples.mean(axis=0) for samples in category_samples]
        ).astype(np.float32)
        component_block_sums = np.zeros(
            (
                len(category_names),
                uncertainty_blocks,
                args.uncertainty_sketch_dimension,
            ),
            dtype=np.float64,
        )
        component_condition_sums = np.zeros(
            (len(condition_cells), args.uncertainty_sketch_dimension),
            dtype=np.float64,
        )
        component_template_sums = np.zeros(
            (len(template_cells), args.uncertainty_sketch_dimension),
            dtype=np.float64,
        )
        for category_index, (category, samples) in enumerate(
            zip(category_names, category_samples, strict=True)
        ):
            sketches = countsketch(
                samples,
                dimension=args.uncertainty_sketch_dimension,
                seed=uncertainty_seed,
            )
            for row, sketch in zip(category_rows[category], sketches, strict=True):
                component_block_sums[
                    category_index, pair_blocks[str(row["pair_id"])]
                ] += sketch
                component_condition_sums[
                    condition_index[(category, str(row["condition"]))]
                ] += sketch
                component_template_sums[
                    template_cell_index[(category, str(row["template_group"]))]
                ] += sketch
        uncertainty_sums_by_component.append(component_block_sums.astype(np.float32))
        condition_sums_by_component.append(
            component_condition_sums.astype(np.float32)
        )
        template_sums_by_component.append(component_template_sums.astype(np.float32))
        effects_by_category.append(category_effects)
        effects.append(
            np.average(category_effects, axis=0, weights=category_counts).astype(
                np.float32
            )
        )
    matrix = np.stack(effects).astype(np.float32)
    category_matrix = np.stack(effects_by_category).astype(np.float32)
    uncertainty_sums = np.stack(uncertainty_sums_by_component).astype(np.float32)
    condition_sums = np.stack(condition_sums_by_component).astype(np.float32)
    template_sums = np.stack(template_sums_by_component).astype(np.float32)
    max_id_length = max(len(value) for value in component_ids)
    max_category_length = max(len(value) for value in category_names)
    write_deterministic_npz(
        args.output_shard,
        {
            "component_ids": np.asarray(component_ids, dtype=f"<U{max_id_length}"),
            "delta_h": matrix,
            "category_names": np.asarray(
                category_names, dtype=f"<U{max_category_length}"
            ),
            "category_episode_counts": category_counts,
            "delta_h_by_category": category_matrix,
            "category_uncertainty_block_counts": uncertainty_counts,
            "delta_h_countsketch_block_sums": uncertainty_sums,
            "uncertainty_block_count": np.asarray(uncertainty_blocks, dtype=np.int64),
            "uncertainty_countsketch_dimension": np.asarray(
                args.uncertainty_sketch_dimension, dtype=np.int64
            ),
            "uncertainty_countsketch_seed": np.asarray(uncertainty_seed, dtype=np.int64),
            "condition_names": np.asarray(
                condition_ids,
                dtype=f"<U{max(len(value) for value in condition_ids)}",
            ),
            "condition_category_names": np.asarray(
                [category for category, _ in condition_cells],
                dtype=f"<U{max_category_length}",
            ),
            "condition_episode_counts": condition_counts,
            "delta_h_countsketch_condition_sums": condition_sums,
            "template_cell_names": np.asarray(
                template_cell_ids,
                dtype=f"<U{max(len(value) for value in template_cell_ids)}",
            ),
            "template_cell_category_names": np.asarray(
                [category for category, _ in template_cells],
                dtype=f"<U{max_category_length}",
            ),
            "template_cell_group_names": np.asarray(
                [template_group for _, template_group in template_cells],
                dtype=f"<U{max(len(value) for value in template_groups)}",
            ),
            "template_cell_episode_counts": template_cell_counts,
            "delta_h_countsketch_template_cell_sums": template_sums,
        },
    )
    episode_ids = [str(row["episode_id"]) for row in episodes]
    category_episode_ids = {
        category: [str(row["episode_id"]) for row in category_rows[category]]
        for category in category_names
    }
    activation_contract = category_contract()
    receipt = {
        "schema": "sch.vpd-effect-shard-receipt.v5",
        "packet_id": packet["packet_id"],
        "cluster_id": packet["cluster_id"],
        "worker_id": args.worker_id,
        "run_id": args.run_id,
        "inputs": {
            "packet_sha256": sha256_file(args.packet),
            "model_id": args.model,
            "revision": args.revision,
            "adapter_sha256": adapter_sha256,
            "dataset_sha256": sha256_file(args.dataset),
            **split_provenance,
            "runner_sha256": sha256_file(Path(__file__)),
            "effect_space_id": effect_space_id,
            "phase": "pre_gate",
            "estimator": "mean_central_difference_lora_rank_multiplier",
            "perturbation_epsilon": args.epsilon,
            "selection_seed": args.selection_seed,
            "episode_order_sha256": content_sha256(episode_ids),
            "episodes": len(episode_ids),
            "generated_tokens": 0,
            "activation_category_contract_sha256": content_sha256(
                activation_contract
            ),
            "activation_categories": category_names,
            "activation_category_episode_counts": {
                category: len(category_rows[category]) for category in category_names
            },
            "activation_category_episode_ids_sha256": {
                category: content_sha256(category_episode_ids[category])
                for category in category_names
            },
            "activation_category_uncertainty": uncertainty_contract(
                block_count=uncertainty_blocks,
                sketch_dimension=args.uncertainty_sketch_dimension,
                seed=uncertainty_seed,
            ),
            "activation_category_pair_block_assignment_sha256": content_sha256(
                dict(sorted(pair_blocks.items()))
            ),
            "activation_category_condition_robustness": condition_contract,
            "activation_category_condition_names": condition_ids,
            "activation_category_condition_categories": [
                category for category, _ in condition_cells
            ],
            "activation_category_condition_episode_counts": {
                condition_id: int(condition_counts[index])
                for index, condition_id in enumerate(condition_ids)
            },
            "activation_category_condition_set_sha256": content_sha256(
                condition_ids
            ),
            "activation_category_template_robustness": template_contract,
            "activation_category_template_cell_names": template_cell_ids,
            "activation_category_template_cell_categories": [
                category for category, _ in template_cells
            ],
            "activation_category_template_cell_groups": [
                template_group for _, template_group in template_cells
            ],
            "activation_category_template_cell_episode_counts": {
                cell_id: int(template_cell_counts[index])
                for index, cell_id in enumerate(template_cell_ids)
            },
            "activation_category_template_cell_set_sha256": content_sha256(
                template_cell_ids
            ),
        },
        "shard_filename": args.output_shard.name,
        "shard_sha256": sha256_file(args.output_shard),
        "component_id_set_sha256": content_sha256(component_ids),
        "rows": len(component_ids),
        "hidden_size": int(matrix.shape[1]),
        "activation_category_count": len(category_names),
        "dtype": "float32",
        "factor_restoration_verified": True,
        "claim_boundary": (
            "Signed split-role effect vectors only; cluster screening does not establish "
            "individual component causality."
        ),
    }
    write_json(args.output_receipt, receipt)


if __name__ == "__main__":
    main()
