from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    load_jsonl,
    sha256_file,
    write_json,
)
from experiments.authz_mechinterp.activation_categories import activation_category
from experiments.authz_mechinterp.compute_vpd_effect_shard import (
    discover_layers,
    factor_digest,
    find_adapter_weights,
    render_prompt,
    resolve_lora_factor,
)
from experiments.authz_mechinterp.run_focused_vpd_sae import (
    INTERVENTION_COEFFICIENTS,
    INTERVENTION_KINDS,
    load_dictionary,
)


ALIGNMENT_KEY_RE = re.compile(r"^layer_(?P<layer>\d+)__(?P<anchor>proposal_start)$")


def load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != "sch.vpd-focused-sae-intervention-plan.v1":
        raise ValueError("unsupported focused SAE intervention plan schema")
    body = dict(plan)
    plan_id = str(body.pop("plan_id"))
    if content_sha256(body) != plan_id:
        raise ValueError("focused SAE intervention plan content address mismatch")
    return plan


def decision_token_ids(tokenizer: Any, word: str) -> list[int]:
    token_ids: set[int] = set()
    for prefix in ("", " ", "\n"):
        encoded = tokenizer(prefix + word, add_special_tokens=False)["input_ids"]
        if encoded:
            token_ids.add(int(encoded[0]))
    if not token_ids:
        raise ValueError(f"tokenizer produced no first-token candidate for {word}")
    return sorted(token_ids)


def expected_decision(episode: dict[str, Any]) -> str:
    value = str(episode.get("expected_candidate_decision", "")).upper()
    if value in {"REQUEST", "PROPOSE"}:
        return value
    if "labels" in episode and "gate_allowed" in episode["labels"]:
        return "PROPOSE" if bool(episode["labels"]["gate_allowed"]) else "REQUEST"
    raise ValueError(f"episode omits a REQUEST/PROPOSE expectation: {episode.get('episode_id')}")


def score_decision_logits(logits: Any, request_ids: list[int], propose_ids: list[int]) -> dict[str, Any]:
    import torch

    request_score = torch.logsumexp(logits[request_ids].to(torch.float32), dim=0)
    propose_score = torch.logsumexp(logits[propose_ids].to(torch.float32), dim=0)
    margin = float((request_score - propose_score).item())
    return {
        "endpoint_value": margin,
        "request_group_logit": float(request_score.item()),
        "propose_group_logit": float(propose_score.item()),
        "predicted_decision": "REQUEST" if margin >= 0.0 else "PROPOSE",
    }


def activation_hook(vector: Any, coefficient: float) -> Any:
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3:
            raise ValueError("activation intervention expected batch x sequence x hidden")
        modified = hidden.clone()
        modified[:, -1, :] += coefficient * vector
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    return hook


def run_interventions(
    *,
    plan_path: Path,
    dictionary_path: Path,
    dataset_path: Path,
    model_id: str,
    revision: str,
    adapter_path: Path,
    expected_adapter_sha256: str,
    worker_id: str,
    output_path: Path,
    tokenizer_id: str | None = None,
    adapter_name: str = "default",
    device: str = "cuda",
    dtype_name: str = "float16",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    plan = load_plan(plan_path)
    inputs = plan["study_inputs"]
    if worker_id != str(plan["worker_id"]):
        raise ValueError("worker is not assigned to this focused SAE plan")
    if model_id != str(inputs["model_id"]) or revision != str(inputs["revision"]):
        raise ValueError("model identity differs from the focused SAE plan")
    if sha256_file(dictionary_path) != str(inputs["sae_dictionary_sha256"]):
        raise ValueError("SAE dictionary differs from the focused SAE plan")
    if sha256_file(dataset_path) != str(inputs["intervention_dataset_sha256"]):
        raise ValueError("intervention dataset differs from the focused SAE plan")
    adapter_weights = find_adapter_weights(adapter_path)
    adapter_sha256 = sha256_file(adapter_weights)
    if adapter_sha256 != expected_adapter_sha256.lower() or adapter_sha256 != str(
        inputs["adapter_sha256"]
    ):
        raise ValueError("adapter weights differ from the focused SAE plan")
    alignment_key = str(inputs["alignment_activation_key"])
    match = ALIGNMENT_KEY_RE.fullmatch(alignment_key)
    if match is None:
        raise ValueError(f"unsupported focused SAE activation space: {alignment_key}")
    layer_index = int(match.group("layer"))
    phase = str(inputs["phase"])
    messages_field = "messages" if phase == "pre_gate" else "messages_post_gate"
    dictionary = load_dictionary(dictionary_path)
    if str(dictionary["activation_key"]) != alignment_key:
        raise ValueError("SAE dictionary activation key changed after planning")
    episodes = load_jsonl(dataset_path)
    episodes_by_id: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        if episode_id in episodes_by_id:
            raise ValueError(f"duplicate intervention episode ID: {episode_id}")
        episodes_by_id[episode_id] = episode

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "focused SAE interventions require torch, transformers, and peft"
        ) from exc
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id or model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path)).to(device)
    model.eval()
    layers = discover_layers(model)
    if not 0 <= layer_index < len(layers):
        raise ValueError("focused SAE activation layer is outside the model")
    hidden_size = int(
        getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 0))
    )
    if hidden_size != int(dictionary["hidden_size"]):
        raise ValueError("model and SAE dictionary hidden dimensions differ")
    request_ids = decision_token_ids(tokenizer, "REQUEST")
    propose_ids = decision_token_ids(tokenizer, "PROPOSE")
    coefficients = [float(value) for value in plan["intervention_contract"]["coefficients"]]
    kinds = [str(value) for value in plan["intervention_contract"]["kinds"]]
    if coefficients != list(INTERVENTION_COEFFICIENTS) or kinds != list(
        INTERVENTION_KINDS
    ):
        raise ValueError("focused SAE plan changed the supported factorial intervention grid")
    trials: list[dict[str, Any]] = []
    factor_restoration_verified = True

    for candidate in plan["candidates"]:
        feature_index = int(candidate["feature_index"])
        activation_np = dictionary["decoder"][feature_index]
        activation_vector = torch.from_numpy(activation_np).to(device=device, dtype=dtype)
        _, target_b, target_rank = resolve_lora_factor(
            model, str(candidate["component_id"]), adapter_name
        )
        _, control_b, control_rank = resolve_lora_factor(
            model, str(candidate["matched_control_component_id"]), adapter_name
        )
        target_original = target_b[:, target_rank].detach().clone()
        control_original = control_b[:, control_rank].detach().clone()
        target_digest = factor_digest(target_original)
        control_digest = factor_digest(control_original)
        for planned_episode in candidate["intervention_episodes"]:
            episode_id = str(planned_episode["episode_id"])
            episode = episodes_by_id.get(episode_id)
            if episode is None:
                raise ValueError(f"planned intervention episode is absent: {episode_id}")
            if str(episode["template_group"]) != str(planned_episode["template_group"]):
                raise ValueError("intervention episode family changed after planning")
            planned_category = planned_episode.get("activation_category")
            observed_category = activation_category(episode)
            if planned_category is not None and str(planned_category) != observed_category:
                raise ValueError("intervention episode activation category changed after planning")
            if content_sha256(episode[messages_field]) != str(
                planned_episode["prompt_messages_sha256"]
            ):
                raise ValueError("intervention prompt changed after planning")
            prompt = render_prompt(tokenizer, episode[messages_field])
            encoded = tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            expected = expected_decision(episode)
            for kind in kinds:
                for coefficient in coefficients:
                    handle = None
                    try:
                        with torch.no_grad():
                            target_b[:, target_rank].copy_(target_original)
                            control_b[:, control_rank].copy_(control_original)
                            if kind == "parameter":
                                target_b[:, target_rank].copy_(
                                    target_original * (1.0 + coefficient)
                                )
                            elif kind == "matched_norm_component_control":
                                control_b[:, control_rank].copy_(
                                    control_original * (1.0 + coefficient)
                                )
                            elif kind == "activation" and coefficient != 0.0:
                                handle = layers[layer_index].register_forward_hook(
                                    activation_hook(activation_vector, coefficient)
                                )
                        with torch.inference_mode():
                            output = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=False,
                            )
                        scored = score_decision_logits(
                            output.logits[0, -1], request_ids, propose_ids
                        )
                    finally:
                        if handle is not None:
                            handle.remove()
                        with torch.no_grad():
                            target_b[:, target_rank].copy_(target_original)
                            control_b[:, control_rank].copy_(control_original)
                    if factor_digest(target_b[:, target_rank]) != target_digest:
                        factor_restoration_verified = False
                        raise ValueError("target LoRA factor restoration failed")
                    if factor_digest(control_b[:, control_rank]) != control_digest:
                        factor_restoration_verified = False
                        raise ValueError("control LoRA factor restoration failed")
                    target_id = {
                        "activation": f"sae_feature:{feature_index}",
                        "parameter": str(candidate["component_id"]),
                        "matched_norm_component_control": str(
                            candidate["matched_control_component_id"]
                        ),
                    }[kind]
                    trials.append(
                        {
                            "schema": "sch.vpd-focused-sae-intervention-trial.v1",
                            "candidate_id": str(candidate["candidate_id"]),
                            "pair_id": episode_id,
                            "template_group": str(episode["template_group"]),
                            **(
                                {}
                                if planned_category is None
                                else {"activation_category": observed_category}
                            ),
                            "intervention_kind": kind,
                            "intervention_target_id": target_id,
                            "coefficient": coefficient,
                            **scored,
                            "expected_decision": expected,
                            "guardrail_violation": (
                                expected == "REQUEST"
                                and scored["predicted_decision"] == "PROPOSE"
                            ),
                        }
                    )
    trace = {
        "schema": "sch.vpd-focused-sae-intervention-trace.v1",
        "plan_id": str(plan["plan_id"]),
        "worker_id": worker_id,
        "inputs": {
            "plan_sha256": sha256_file(plan_path),
            "dictionary_sha256": sha256_file(dictionary_path),
            "dataset_sha256": sha256_file(dataset_path),
            "runner_sha256": sha256_file(Path(__file__)),
            "model_id": model_id,
            "revision": revision,
            "adapter_sha256": adapter_sha256,
            "alignment_activation_key": alignment_key,
            "phase": phase,
        },
        "decision_tokens": {
            "request_first_token_ids": request_ids,
            "propose_first_token_ids": propose_ids,
            "margin": "logsumexp(REQUEST)-logsumexp(PROPOSE)",
        },
        "factor_restoration_verified": factor_restoration_verified,
        "trials": trials,
    }
    write_json(output_path, trace)
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute a frozen focused SAE activation/parameter intervention plan."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--adapter-name", default="default")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    trace = run_interventions(
        plan_path=args.plan,
        dictionary_path=args.dictionary,
        dataset_path=args.dataset,
        model_id=args.model,
        revision=args.revision,
        adapter_path=args.adapter,
        expected_adapter_sha256=args.expected_adapter_sha256,
        worker_id=args.worker_id,
        output_path=args.output,
        tokenizer_id=args.tokenizer,
        adapter_name=args.adapter_name,
        device=args.device,
        dtype_name=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    print(
        json.dumps(
            {
                "schema": "sch.vpd-focused-sae-intervention-output.v1",
                "plan_id": trace["plan_id"],
                "trials": len(trace["trials"]),
                "output_sha256": sha256_file(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
