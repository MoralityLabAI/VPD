"""Q2 capture-only replay building block; no CLI, model loading or generation.

Calling replay_q2 performs one model forward and therefore belongs only in an
explicitly authorized capture job. Importing this module does not run a model.
Checkpoint identity, generation provenance and provider lifecycle are upstream
requirements, not certified by this structural capture helper.
"""
from __future__ import annotations

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256
from experiments.authz_mechinterp.plan_sae_trajectory_breadth_v1 import validate_windows
from experiments.nanbeige_looped.capture import RepeatedForwardCapture

SITES = ("model.layers.11", "model.norm")
WIDTH = 3072


def replay_q2(model, token_ids, record, *, max_sequence_tokens):
    """Replay one frozen, unpadded native trajectory and return four arrays.

    record supplies id/family/split, prompt_tokens/generated_tokens,
    positions/regions and token_ids_content_sha256 (canonical JSON list hash).
    No weights, config, input metadata or files are modified. This is not a
    freshness, checkpoint-revision or capture-authorization validator.
    """
    import torch

    if type(max_sequence_tokens) is not int or not 1 <= max_sequence_tokens <= 8192:
        raise ValueError("explicit sequence bound in 1..8192 required")
    if (not isinstance(token_ids, list) or not token_ids
            or len(token_ids) > max_sequence_tokens
            or any(type(t) is not int or t < 0 for t in token_ids)):
        raise ValueError("bounded nonnegative integer token list required")
    for field in ("id", "family"):
        if not isinstance(record[field], str) or not record[field]:
            raise ValueError("explicit record identity required")
    if record["split"] not in {"train", "eval", "discovery", "calibration", "validation"}:
        raise ValueError("unknown record role")
    validate_windows(record)
    if (len(token_ids) != record["prompt_tokens"] + record["generated_tokens"]
            or content_sha256(token_ids) != record["token_ids_content_sha256"]):
        raise ValueError("frozen full trajectory differs")
    config = model.config
    if (config.num_hidden_layers, config.num_loops, config.hidden_size) != (22, 2, WIDTH):
        raise ValueError("Q2 requires the reviewed 22-layer two-pass width-3072 architecture")
    if config.loop_loss_weights or getattr(config, "enable_double_loop_split", False):
        raise ValueError("unsupported loop configuration")
    if model.training or model.dtype != torch.bfloat16 or getattr(model, "is_quantized", False):
        raise ValueError("nonquantized BF16 evaluation model required")
    if type(config.vocab_size) is not int or any(t >= config.vocab_size for t in token_ids):
        raise ValueError("token outside model vocabulary")

    modules = {SITES[0]: model.model.layers[11], SITES[1]: model.model.norm}
    if modules[SITES[0]] is modules[SITES[1]]:
        raise ValueError("distinct physical sites required")
    positions = record["positions"]

    def extract(output):
        hidden = output[0] if isinstance(output, tuple) else output
        if (not isinstance(hidden, torch.Tensor)
                or tuple(hidden.shape) != (1, len(token_ids), WIDTH)
                or hidden.dtype != torch.bfloat16):
            raise ValueError("expected full unpadded BF16 physical module output")
        rows = hidden[0, positions].float().cpu().numpy().copy()
        if not np.isfinite(rows).all():
            raise ValueError("nonfinite selected activations")
        return rows

    capture = RepeatedForwardCapture(modules, expected_calls=2, extract=extract)
    with torch.inference_mode():
        tokens = torch.tensor([token_ids], dtype=torch.long, device=model.device)
        with capture:
            model.model(tokens, use_cache=False)
    arrays = {f"{site}__pass_{p}": capture.values[(site, p)] for site in SITES for p in (0, 1)}
    return {
        "arrays": arrays,
        "record": {
            "id": record["id"], "family": record["family"], "split": record["split"],
            "token_ids_content_sha256": record["token_ids_content_sha256"],
            "token_positions": list(positions),
            "regions": {k: list(v) for k, v in record["regions"].items()},
            "sites": list(SITES), "calls": dict(capture.counts), "passes": 2,
            "execution_output_dtype": "bfloat16", "storage_dtype": "float32",
            "capture_semantics": "physical module outputs; invocation indices 0 and 1 on one full-context replay",
        },
    }
