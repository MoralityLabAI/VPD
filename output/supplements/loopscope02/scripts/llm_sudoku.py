"""LOOPSCOPE-02: Nanbeige4.2-3B (looped 3B) and Qwen3.5-4B-Base / Qwen3-4B (untied 4B) runner on 9x9 Sudoku.

Runs on the exact stratified 256-item target-cell benchmark (`benchmark_bundle.json`):
  - `nanbeige`: Nanbeige/Nanbeige4.2-3B @ 451ed48c (bvg-v03 venv, transformers 4.46.3),
    evaluating loops t = 1..max_loops, ARG-02 counterfactuals (open-loop sharpening at t=2 and t=8,
    donor recurrent displacement shuffle at t=2 and t=8, step-1 answer-site patch repair),
    and per-pass SVD component usage + Transported Component Attribution (TCA) across Pass 1 vs Pass 2
    on physical layers (6, 12, 17, 22) `mlp.down_proj`.
  - `qwen4b`: Qwen/Qwen3.5-4B-Base @ 1001bb4d (bluebeam venv, transformers 5.17), plus optional
    `qwen3_4b_inst` (Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f), evaluating single-pass accuracy,
    untied mid-stack vs final-stack component usage correlation, and TCA specificity.

Resumable and power-cap safe: checks `enforced.power.limit` between batches and exits 75 if capped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mechinterp.loopscope02.scripts import sudoku_prop as SP  # noqa: E402

EXIT_POWER_CAP = 75
TOP_SVD = 64

MODEL_LOCKS = {
    "nanbeige": {
        "repo": "Nanbeige/Nanbeige4.2-3B",
        "revision": "451ed48c3273ecef7ea8faaa43c31ce529763bb1",
        "bos_token_id": 166100,
        "files_sha256": {
            "config.json": "f6cb15b22847664f3a6049dc4b58fdd10f1650d112ac99a1da3d051f17c2ca19",
            "model.safetensors.index.json": "30d8da0fa8b97abc6d9eddfd017a0cb7a649bcbd58caa57804c56c767db5c0f1",
        },
        "tca_layers": (5, 11, 16, 21),  # 0-based -> physical layers 6, 12, 17, 22
    },
    "qwen4b": {
        "repo": "Qwen/Qwen3.5-4B-Base",
        "revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
        "bos_token_id": None,
        "files_sha256": {
            "config.json": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
            "model.safetensors.index.json": "eae340074abb0a5f31a6621f7ae8e8248a7c1790df04a722c4e4b70c2a6d1dbb",
        },
        "tca_layers": (7, 15, 23, 31),  # quartiles of the 32-layer stack
    },
    "qwen3_4b_inst": {
        "repo": "Qwen/Qwen3-4B-Instruct-2507",
        "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "bos_token_id": None,
        "files_sha256": {
            "config.json": "5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba",
            "model.safetensors.index.json": "d6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c",
        },
        "tca_layers": (8, 17, 26, 35),  # quartiles of the 36-layer stack
    },
}


def power_limit_watts() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=enforced.power.limit", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip().splitlines()[0]
        return float(out.strip())
    except Exception:
        return None


def snapshot_path(entry: dict) -> Path:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    root = home / "hub" / ("models--" + entry["repo"].replace("/", "--")) / "snapshots" / entry["revision"]
    for fname, expected in entry["files_sha256"].items():
        p = root / fname
        if not p.is_file():
            raise FileNotFoundError(f"missing snapshot file: {p}")
        actual = SP.sha256_file(p)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch on {p}: {actual} != {expected}")
    return root


def verify_and_resolve_digit_tokens(encode_fn, sample_prompt: str) -> list[int]:
    """Verify prompt + d == encode(prompt) + [digit_id[d]] for all digits 1..9."""
    base_ids = encode_fn(sample_prompt)
    digit_ids = []
    for d in range(1, 10):
        full_ids = encode_fn(sample_prompt + str(d))
        if len(full_ids) != len(base_ids) + 1 or full_ids[:-1] != base_ids:
            raise RuntimeError(
                f"Tokenizer boundary violation for digit {d}: "
                f"len(base)={len(base_ids)}, len(full)={len(full_ids)}, tail={full_ids[-3:]}"
            )
        digit_ids.append(int(full_ids[-1]))
    if len(set(digit_ids)) != 9:
        raise RuntimeError(f"Digit token IDs are not 9 distinct tokens: {digit_ids}")
    return digit_ids


def collect_used_token_ids(encode_fn, bundle: dict) -> list[int]:
    """Collect exact sorted union of prompt token IDs and single-digit answer token IDs."""
    shots = bundle["shots"]
    prompts = [SP.format_target_prompt(shots, it["question"], int(it["target_idx"])) for it in bundle["items"]]
    digit_ids = verify_and_resolve_digit_tokens(encode_fn, prompts[0])
    used: set[int] = set(digit_ids)
    for p in prompts:
        used.update(encode_fn(p))
    return sorted(used)


def _read_vocab_rows_from_shard(f, base_off: int, spec: dict, used_ids: list[int], dt: torch.dtype) -> torch.Tensor:
    s0, s1 = spec["data_offsets"]
    n_rows, dim = int(spec["shape"][0]), int(spec["shape"][1])
    elem_bytes = torch.tensor([], dtype=dt).element_size()
    row_bytes = dim * elem_bytes
    if n_rows * row_bytes != (s1 - s0):
        raise RuntimeError(f"Unexpected vocab tensor byte layout: {spec['shape']} vs {s1 - s0}")
    out = torch.empty((len(used_ids), dim), dtype=dt)
    for c_idx, orig_id in enumerate(used_ids):
        f.seek(base_off + s0 + int(orig_id) * row_bytes)
        raw = bytearray(f.read(row_bytes))
        out[c_idx].copy_(torch.frombuffer(raw, dtype=dt))
    return out


def load_nanbeige(root: Path, device: str, bundle: dict):
    """Stream safetensors shards straight onto GPU, slicing exact used vocab rows (<1 MB vs 2 GB)."""
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

    tok_raw = PreTrainedTokenizerFast(tokenizer_file=str(root / "tokenizer.json"))
    bos = MODEL_LOCKS["nanbeige"]["bos_token_id"]

    def raw_encode_fn(text: str) -> list[int]:
        return [bos] + tok_raw.encode(text, add_special_tokens=False)

    used_ids = collect_used_token_ids(raw_encode_fn, bundle)
    orig_to_compact = {tid: idx for idx, tid in enumerate(used_ids)}

    def encode_fn(text: str) -> list[int]:
        return [orig_to_compact[tid] for tid in raw_encode_fn(text)]

    config = AutoConfig.from_pretrained(str(root), trust_remote_code=True, local_files_only=True)
    config._attn_implementation = "sdpa"
    config.vocab_size = len(used_ids)
    config.bos_token_id = orig_to_compact[bos]
    config.eos_token_id = 0
    config.pad_token_id = 0
    prev_dt = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation="sdpa")
    finally:
        torch.set_default_dtype(prev_dt)

    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    params = model.state_dict()
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    loaded = set()
    for shard in sorted(set(index["weight_map"].values())):
        with open(root / shard, "rb") as f:
            hdr_len = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(hdr_len))
            base_off = 8 + hdr_len
            for key, spec in header.items():
                if key == "__metadata__":
                    continue
                dt = dtypes[spec["dtype"]]
                if key in ("model.embed_tokens.weight", "lm_head.weight"):
                    t = _read_vocab_rows_from_shard(f, base_off, spec, used_ids, dt)
                else:
                    s0, s1 = spec["data_offsets"]
                    f.seek(base_off + s0)
                    raw = bytearray(f.read(s1 - s0))
                    t = torch.frombuffer(raw, dtype=dt).reshape(spec["shape"])
                with torch.no_grad():
                    params[key].copy_(t.to(device=device, dtype=params[key].dtype))
                del t
                loaded.add(key)
    missing = sorted(k for k in set(params) - loaded if "rotary" not in k)
    if missing:
        raise RuntimeError(f"Nanbeige missing weights: {missing[:5]}")
    model.eval()
    model.requires_grad_(False)
    return model, encode_fn


def load_qwen(profile: str, root: Path, device: str, bundle: dict):
    """Stream Qwen3.5-4B-Base (text CausalLM) or Qwen3-4B safetensors shards straight onto GPU."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(root), local_files_only=True)

    def raw_encode_fn(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    used_ids = collect_used_token_ids(raw_encode_fn, bundle)
    orig_to_compact = {tid: idx for idx, tid in enumerate(used_ids)}

    def encode_fn(text: str) -> list[int]:
        return [orig_to_compact[tid] for tid in raw_encode_fn(text)]

    raw_cfg = AutoConfig.from_pretrained(str(root), local_files_only=True)
    cfg = raw_cfg.text_config if hasattr(raw_cfg, "text_config") else raw_cfg
    cfg._attn_implementation = "sdpa"
    cfg.vocab_size = len(used_ids)
    cfg.bos_token_id = 0
    cfg.eos_token_id = 0
    cfg.pad_token_id = 0

    prev_dt = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            model = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa")
    finally:
        torch.set_default_dtype(prev_dt)

    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    params = model.state_dict()
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    loaded: set[str] = set()
    for shard in sorted(set(index["weight_map"].values())):
        with open(root / shard, "rb") as f:
            hdr_len = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(hdr_len))
            base_off = 8 + hdr_len
            for key, spec in header.items():
                if key == "__metadata__" or key.startswith("model.visual.") or key.startswith("mtp."):
                    continue
                mapped = "model." + key[len("model.language_model."):] if key.startswith("model.language_model.") else key
                if mapped not in params:
                    raise RuntimeError(f"Unexpected Qwen weight key {key} -> {mapped}")
                dt = dtypes[spec["dtype"]]
                if mapped in ("model.embed_tokens.weight", "lm_head.weight"):
                    t = _read_vocab_rows_from_shard(f, base_off, spec, used_ids, dt)
                else:
                    s0, s1 = spec["data_offsets"]
                    f.seek(base_off + s0)
                    raw = bytearray(f.read(s1 - s0))
                    t = torch.frombuffer(raw, dtype=dt).reshape(spec["shape"])
                with torch.no_grad():
                    params[mapped].copy_(t.to(device=device, dtype=params[mapped].dtype))
                del t
                loaded.add(mapped)

    if getattr(cfg, "tie_word_embeddings", False) and "lm_head.weight" not in loaded:
        model.tie_weights()
        loaded.add("lm_head.weight")

    missing = sorted(k for k in set(params) - loaded if "rotary" not in k and "inv_freq" not in k)
    if missing:
        raise RuntimeError(f"Qwen ({profile}) missing weights: {missing[:5]}")
    model.eval()
    model.requires_grad_(False)
    return model, encode_fn


def nanbeige_prepare(model, ids: torch.Tensor):
    base = model.model
    emb = base.embed_tokens(ids)
    rows, length = ids.shape
    cache_pos = torch.arange(length, device=ids.device)
    pos = cache_pos.unsqueeze(0).expand(rows, -1)
    mask = base._update_causal_mask(None, emb, cache_pos, None, False)
    return emb, {"mask": mask, "pos": pos, "cache_position": cache_pos}


def nanbeige_step(model, h: torch.Tensor, ctx: dict, pass_box: list[int] | None = None, pass_val: int = 0) -> torch.Tensor:
    from torch.utils.checkpoint import checkpoint

    base = model.model
    use_ckpt = torch.is_grad_enabled()
    for layer in base.layers:
        if use_ckpt:
            def _run_lyr(h_in: torch.Tensor, lyr=layer, pv=pass_val) -> torch.Tensor:
                if pass_box is not None:
                    pass_box[0] = pv
                return lyr(
                    h_in,
                    attention_mask=ctx["mask"],
                    position_ids=ctx["pos"],
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=ctx["cache_position"],
                    loop_idx=0,
                )[0]

            h = checkpoint(_run_lyr, h, use_reentrant=False)
        else:
            h = layer(
                h,
                attention_mask=ctx["mask"],
                position_ids=ctx["pos"],
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=ctx["cache_position"],
                loop_idx=0,
            )[0]
    return base.norm(h)


def compute_svd_top(weight: torch.Tensor, top_k: int = TOP_SVD):
    # Compute truncated SVD in float32 on GPU (fast, low host RAM)
    w = weight.detach().float()
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    r = min(top_k, s.numel())
    return u[:, :r].contiguous(), s[:r].contiguous(), vh[:r, :].T.contiguous()


def run_nanbeige_suite(model, encode_fn, bundle: dict, max_loops: int, min_watts: float) -> dict:
    dev = next(model.parameters()).device
    items = bundle["items"]
    shots = bundle["shots"]
    n = len(items)
    prompts = [SP.format_target_prompt(shots, it["question"], int(it["target_idx"])) for it in items]
    digit_ids = verify_and_resolve_digit_tokens(encode_fn, prompts[0])
    digit_ids_t = torch.tensor(digit_ids, dtype=torch.long, device=dev)

    # Verify parity of 2-step manual loop against native model.forward on item 0
    ids0 = torch.tensor([encode_fn(prompts[0])], dtype=torch.long, device=dev)
    with torch.no_grad():
        native_logits = model(input_ids=ids0, use_cache=False).logits[:, -1, digit_ids_t].float()
        emb0, ctx0 = nanbeige_prepare(model, ids0)
        h1_0 = nanbeige_step(model, emb0, ctx0)
        h2_0 = nanbeige_step(model, h1_0, ctx0)
        manual_logits = model.lm_head(h2_0)[:, -1, digit_ids_t].float()
        parity_max_diff = float((native_logits - manual_logits).abs().max().item())
    if parity_max_diff > 1e-3:
        raise RuntimeError(f"Nanbeige loop-2 parity check failed: max diff {parity_max_diff}")

    encoded = [encode_fn(p) for p in prompts]
    # Group indices by exact token length so zero padding is introduced
    by_len: dict[int, list[int]] = {}
    for idx, seq in enumerate(encoded):
        by_len.setdefault(len(seq), []).append(idx)

    tca_layers = MODEL_LOCKS["nanbeige"]["tca_layers"]
    facs = {
        L: compute_svd_top(model.model.layers[L].mlp.down_proj.weight)
        for L in tca_layers
    }
    if str(dev).startswith("cuda"):
        torch.cuda.empty_cache()
    usage_pass = {L: {1: np.zeros(TOP_SVD, dtype=np.float64), 2: np.zeros(TOP_SVD, dtype=np.float64)} for L in tca_layers}

    probs_loops = np.zeros((max_loops, n, 9), dtype=np.float32)
    h1_tail = torch.zeros((n, 5, model.config.hidden_size), dtype=torch.bfloat16)
    h2_tail = torch.zeros((n, 5, model.config.hidden_size), dtype=torch.bfloat16)
    hmax_tail = torch.zeros((n, 5, model.config.hidden_size), dtype=torch.bfloat16)
    x_tail = torch.zeros((n, 5, model.config.hidden_size), dtype=torch.bfloat16)
    rel_step_norms = np.zeros((max_loops, n), dtype=np.float32)

    batch_size = 2
    with torch.no_grad():
        for seq_len, group in sorted(by_len.items()):
            for b0 in range(0, len(group), batch_size):
                watts = power_limit_watts()
                if watts is not None and watts < min_watts:
                    raise SystemExit(EXIT_POWER_CAP)
                sub_idx = group[b0:b0 + batch_size]
                ids = torch.tensor([encoded[i] for i in sub_idx], dtype=torch.long, device=dev)
                emb, ctx = nanbeige_prepare(model, ids)
                x_tail[sub_idx] = emb[:, -5:, :].cpu()
                prev = emb

                active_loop = {"t": 0}
                handles = []
                for L in tca_layers:
                    _, s_vec, v_mat = facs[L]

                    def make_hk(lay=L, sv=s_vec, vm=v_mat):
                        def _hk(_m, inp, _out):
                            t_now = active_loop["t"]
                            if t_now in (1, 2):
                                xin = inp[0][:, -1, :].float()  # answer token [b, d_in]
                                mag = (xin @ vm).abs() * sv
                                usage_pass[lay][t_now] += mag.sum(0).cpu().numpy()
                        return _hk

                    handles.append(model.model.layers[L].mlp.down_proj.register_forward_hook(make_hk()))

                try:
                    h = emb
                    for t in range(1, max_loops + 1):
                        active_loop["t"] = t
                        h = nanbeige_step(model, h, ctx)
                        logits = model.lm_head(h[:, -1:, :])[:, 0, digit_ids_t].float()
                        probs_loops[t - 1, sub_idx] = torch.softmax(logits, dim=-1).cpu().numpy()
                        step_norm = (h.float() - prev.float()).flatten(1).norm(dim=1)
                        prev_norm = prev.float().flatten(1).norm(dim=1).clamp_min(1e-8)
                        rel_step_norms[t - 1, sub_idx] = (step_norm / prev_norm).cpu().numpy()
                        prev = h
                        if t == 1:
                            h1_tail[sub_idx] = h[:, -5:, :].cpu()
                        if t == 2:
                            h2_tail[sub_idx] = h[:, -5:, :].cpu()
                        if t == max_loops:
                            hmax_tail[sub_idx] = h[:, -5:, :].cpu()
                finally:
                    for hk in handles:
                        hk.remove()

    # Open-loop & donor-shuffle counterfactuals at depth 2 and depth max_loops (exact right-aligned last token)
    id_to_idx = {it["item_id"]: idx for idx, it in enumerate(items)}
    donor_idx = torch.tensor([id_to_idx[it["donor_item_id"]] for it in items], dtype=torch.long)

    def readout_tail_last(h_tail_cpu: torch.Tensor) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i0 in range(0, n, 64):
                last_h = h_tail_cpu[i0:i0 + 64, -1:, :].to(dev)
                lg = model.lm_head(last_h)[:, 0, digit_ids_t].float()
                out.append(torch.softmax(lg, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0)

    probs_open2 = readout_tail_last(2 * h1_tail - x_tail)
    probs_open_max = readout_tail_last(max_loops * h1_tail - (max_loops - 1) * x_tail)
    probs_donor2 = readout_tail_last(h1_tail + (h2_tail[donor_idx] - h1_tail[donor_idx]))
    probs_donor_max = readout_tail_last(h1_tail + (hmax_tail[donor_idx] - h1_tail[donor_idx]))

    # Step-1 answer-site donor patch repair (patch last 5 positions after loop 1, then run loop 2)
    probs_patch_repaired2 = np.zeros((n, 9), dtype=np.float32)
    with torch.no_grad():
        for seq_len, group in sorted(by_len.items()):
            for b0 in range(0, len(group), batch_size):
                sub_idx = group[b0:b0 + batch_size]
                ids = torch.tensor([encoded[i] for i in sub_idx], dtype=torch.long, device=dev)
                emb, ctx = nanbeige_prepare(model, ids)
                h1 = nanbeige_step(model, emb, ctx)
                h1_patched = h1.clone()
                d_sub = donor_idx[sub_idx]
                h1_patched[:, -5:, :] = h1_tail[d_sub].to(dev)
                h2_rep = nanbeige_step(model, h1_patched, ctx)
                lg = model.lm_head(h2_rep[:, -1:, :])[:, 0, digit_ids_t].float()
                probs_patch_repaired2[sub_idx] = torch.softmax(lg, dim=-1).cpu().numpy()

    # Pass 1 vs Pass 2 component usage correlations
    usage_corrs = {}
    for L in tca_layers:
        u1 = usage_pass[L][1] / n
        u2 = usage_pass[L][2] / n
        r_val = float(np.corrcoef(u1, u2)[0, 1])
        usage_corrs[f"L{L + 1}"] = {
            "r_pass1_vs_pass2": r_val,
            "usage_ratio_pass2_over_pass1": float(u2.sum() / max(u1.sum(), 1e-12)),
        }

    # Unrolled 2-pass TCA on the 64 S1_naked1 items (parameter-grad-free activation backprop!)
    s1_indices = [i for i, it in enumerate(items) if it["stratum"] == "S1_naked1"]
    m_cnt = len(tca_layers)
    frag = np.zeros((m_cnt, 2, len(s1_indices)), dtype=np.float32)
    zr_gt3 = np.zeros((m_cnt, 2), dtype=np.int64)
    zw_gt3 = np.zeros((m_cnt, 2), dtype=np.int64)

    for k0 in range(0, len(s1_indices), 2):
        watts = power_limit_watts()
        if watts is not None and watts < min_watts:
            raise SystemExit(EXIT_POWER_CAP)
        sub_local = list(range(k0, min(len(s1_indices), k0 + 2)))
        sub_global = [s1_indices[j] for j in sub_local]
        # Run 1 item at a time if lengths differ so zero padding is needed
        for loc_j, glob_i in zip(sub_local, sub_global):
            ids = torch.tensor([encoded[glob_i]], dtype=torch.long, device=dev)
            tgt = int(items[glob_i]["target_digit"]) - 1
            emb, ctx = nanbeige_prepare(model, ids)
            emb = emb.detach().requires_grad_(True)

            cap_x: dict[int, dict[int, torch.Tensor]] = {mi: {} for mi in range(m_cnt)}
            cap_g: dict[int, dict[int, torch.Tensor]] = {mi: {} for mi in range(m_cnt)}
            pass_box = [0]
            handles = []
            for mi, L in enumerate(tca_layers):
                mod = model.model.layers[L].mlp.down_proj

                def make_tca_hk(m_idx=mi):
                    def _hk(_m, inp, out):
                        p_idx = pass_box[0]
                        cap_x[m_idx][p_idx] = inp[0].detach()

                        def _sv(grad, m_i=m_idx, p_i=p_idx):
                            cap_g[m_i][p_i] = grad.detach()

                        out.register_hook(_sv)
                    return _hk

                handles.append(mod.register_forward_hook(make_tca_hk()))

            try:
                h1 = nanbeige_step(model, emb, ctx, pass_box=pass_box, pass_val=0)
                h2 = nanbeige_step(model, h1, ctx, pass_box=pass_box, pass_val=1)
                logits9 = model.lm_head(h2[:, -1:, :])[:, 0, digit_ids_t].float()  # [1, 9]
                zc = logits9[0, tgt]
                other = logits9[0].clone()
                other[tgt] = float("-inf")
                margin = zc - torch.logsumexp(other, dim=0)
                margin.backward()

                for mi, L in enumerate(tca_layers):
                    u_mat, s_vec, v_mat = facs[L]
                    din, dout = v_mat.shape[0], u_mat.shape[0]
                    for p_idx in range(2):
                        xb = cap_x[mi][p_idx].float()  # [1, P, din]
                        ab = cap_g[mi][p_idx].float()  # [1, P, dout]
                        xv = xb @ v_mat
                        au = ab @ u_mat
                        delta = -s_vec * (xv * au).sum(dim=1)  # [1, r]
                        xtau = torch.einsum("bpi,bpr->bir", xb, au)
                        atxv = torch.einsum("bpo,bpr->bor", ab, xv)
                        sdr = s_vec * xtau.norm(dim=1) / math.sqrt(din)
                        sdw = s_vec * atxv.norm(dim=1) / math.sqrt(dout)
                        zr = (delta / sdr.clamp_min(1e-12)).abs()
                        zw = (delta / sdw.clamp_min(1e-12)).abs()
                        zr_gt3[mi, p_idx] += int((zr > 3.0).sum().item())
                        zw_gt3[mi, p_idx] += int((zw > 3.0).sum().item())
                        # Norm of answer-position + last-16 positions gradient-input outer product
                        x_sub, a_sub = xb[:, -32:, :], ab[:, -32:, :]
                        frag[mi, p_idx, loc_j] = float(
                            ((x_sub @ x_sub.transpose(1, 2)) * (a_sub @ a_sub.transpose(1, 2)))
                            .sum().clamp_min(0).sqrt().item()
                        )
            finally:
                for hk in handles:
                    hk.remove()
                del cap_x, cap_g

    tgt_all = np.array([int(it["target_digit"]) - 1 for it in items], dtype=np.int64)
    denom = float(len(s1_indices) * TOP_SVD)
    f_mean = frag.mean(axis=2)

    return {
        "model": MODEL_LOCKS["nanbeige"]["repo"],
        "revision": MODEL_LOCKS["nanbeige"]["revision"],
        "bundle_sha256": bundle["bundle_sha256"],
        "loop2_native_parity_max_diff": parity_max_diff,
        "max_loops": max_loops,
        "items_target_correct_by_loop": (probs_loops.argmax(axis=-1) == tgt_all[None, :]).astype(int).tolist(),
        "items_target_logprob_by_loop": [
            np.log(np.clip(probs_loops[t, np.arange(n), tgt_all], 1e-12, 1.0)).mean().item()
            for t in range(max_loops)
        ],
        "items_open2_correct": (probs_open2.argmax(axis=-1) == tgt_all).astype(int).tolist(),
        "items_open_max_correct": (probs_open_max.argmax(axis=-1) == tgt_all).astype(int).tolist(),
        "items_donor2_correct": (probs_donor2.argmax(axis=-1) == tgt_all).astype(int).tolist(),
        "items_donor_max_correct": (probs_donor_max.argmax(axis=-1) == tgt_all).astype(int).tolist(),
        "items_donor2_top1_agree_with_loop2": (probs_donor2.argmax(axis=-1) == probs_loops[1].argmax(axis=-1)).astype(int).tolist(),
        "items_patch_repaired2_top1_agree_with_loop2": (
            probs_patch_repaired2.argmax(axis=-1) == probs_loops[1].argmax(axis=-1)
        ).astype(int).tolist(),
        "items_patch_repaired2_correct": (probs_patch_repaired2.argmax(axis=-1) == tgt_all).astype(int).tolist(),
        "rel_step_norm_mean_by_loop": rel_step_norms.mean(axis=1).tolist(),
        "usage_pass1_vs_pass2": usage_corrs,
        "mean_r_pass1_vs_pass2": float(np.mean([v["r_pass1_vs_pass2"] for v in usage_corrs.values()])),
        "tca_s1": {
            "layers": [f"L{L + 1}" for L in tca_layers],
            "frag_pass1_pass2": f_mean.tolist(),
            "f1_over_f2_by_layer": (f_mean[:, 0] / np.maximum(f_mean[:, 1], 1e-12)).tolist(),
            "zr_gt3_share_pass1_pass2": (zr_gt3 / denom).tolist(),
            "zw_gt3_share_pass1_pass2": (zw_gt3 / denom).tolist(),
        },
    }


def get_qwen_layers_and_norm(model):
    # Qwen3_5ForCausalLM vs Qwen3ForCausalLM
    base = model.model
    if hasattr(base, "layers"):
        return base.layers, base.norm
    if hasattr(base, "language_model") and hasattr(base.language_model, "layers"):
        return base.language_model.layers, base.language_model.norm
    raise RuntimeError(f"Unsupported Qwen base structure: {type(base)}")


def run_qwen_suite(profile: str, model, encode_fn, bundle: dict, min_watts: float) -> dict:
    dev = next(model.parameters()).device
    items = bundle["items"]
    shots = bundle["shots"]
    n = len(items)
    prompts = [SP.format_target_prompt(shots, it["question"], int(it["target_idx"])) for it in items]
    digit_ids = verify_and_resolve_digit_tokens(encode_fn, prompts[0])
    digit_ids_t = torch.tensor(digit_ids, dtype=torch.long, device=dev)

    encoded = [encode_fn(p) for p in prompts]
    by_len: dict[int, list[int]] = {}
    for idx, seq in enumerate(encoded):
        by_len.setdefault(len(seq), []).append(idx)

    layers_mod, final_norm = get_qwen_layers_and_norm(model)
    n_layers = len(layers_mod)
    mid_layer = (n_layers // 2) - 1
    tca_layers = MODEL_LOCKS[profile]["tca_layers"]
    facs = {L: compute_svd_top(layers_mod[L].mlp.down_proj.weight) for L in tca_layers}
    if str(dev).startswith("cuda"):
        torch.cuda.empty_cache()
    usage_by_layer = {L: np.zeros(TOP_SVD, dtype=np.float64) for L in tca_layers}

    probs_final = np.zeros((n, 9), dtype=np.float32)
    probs_mid_lens = np.zeros((n, 9), dtype=np.float32)

    batch_size = 1
    with torch.no_grad():
        for _seq_len, group in sorted(by_len.items()):
            for b0 in range(0, len(group), batch_size):
                watts = power_limit_watts()
                if watts is not None and watts < min_watts:
                    raise SystemExit(EXIT_POWER_CAP)
                sub_idx = group[b0:b0 + batch_size]
                ids = torch.tensor([encoded[i] for i in sub_idx], dtype=torch.long, device=dev)

                mid_capture = {}
                handles = []

                def mid_hk(_m, _inp, out):
                    h_out = out[0] if isinstance(out, tuple) else out
                    mid_capture["h"] = h_out[:, -1:, :].detach()

                handles.append(layers_mod[mid_layer].register_forward_hook(mid_hk))
                for L in tca_layers:
                    _, s_vec, v_mat = facs[L]

                    def make_u_hk(lay=L, sv=s_vec, vm=v_mat):
                        def _hk(_m, inp, _out):
                            xin = inp[0][:, -1, :].float()
                            mag = (xin @ vm).abs() * sv
                            usage_by_layer[lay] += mag.sum(0).cpu().numpy()
                        return _hk

                    handles.append(layers_mod[L].mlp.down_proj.register_forward_hook(make_u_hk()))

                try:
                    logits = model(input_ids=ids, use_cache=False).logits[:, -1, digit_ids_t].float()
                    probs_final[sub_idx] = torch.softmax(logits, dim=-1).cpu().numpy()
                    mid_lg = model.lm_head(final_norm(mid_capture["h"]))[:, 0, digit_ids_t].float()
                    probs_mid_lens[sub_idx] = torch.softmax(mid_lg, dim=-1).cpu().numpy()
                finally:
                    for hk in handles:
                        hk.remove()

    # Single-pass TCA specificity on the 64 S1_naked1 items
    s1_indices = [i for i, it in enumerate(items) if it["stratum"] == "S1_naked1"]
    m_cnt = len(tca_layers)
    zr_gt3 = np.zeros(m_cnt, dtype=np.int64)
    zw_gt3 = np.zeros(m_cnt, dtype=np.int64)
    frag = np.zeros((m_cnt, len(s1_indices)), dtype=np.float32)

    embed_mod = model.get_input_embeddings()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for lyr in layers_mod:
        lyr.training = True
    if str(dev).startswith("cuda"):
        torch.cuda.empty_cache()
    for loc_j, glob_i in enumerate(s1_indices):
        if loc_j % 8 == 0:
            watts = power_limit_watts()
            if watts is not None and watts < min_watts:
                raise SystemExit(EXIT_POWER_CAP)
        ids = torch.tensor([encoded[glob_i]], dtype=torch.long, device=dev)
        tgt = int(items[glob_i]["target_digit"]) - 1
        inp_emb = embed_mod(ids).detach().requires_grad_(True)

        cap_x: dict[int, torch.Tensor] = {}
        cap_g: dict[int, torch.Tensor] = {}
        handles = []
        for mi, L in enumerate(tca_layers):
            mod = layers_mod[L].mlp.down_proj

            def make_tca_hk(m_idx=mi):
                def _hk(_m, inp, out):
                    cap_x[m_idx] = inp[0].detach()

                    def _sv(grad, m_i=m_idx):
                        cap_g[m_i] = grad.detach()

                    out.register_hook(_sv)
                return _hk

            handles.append(mod.register_forward_hook(make_tca_hk()))

        try:
            logits9 = model(inputs_embeds=inp_emb, use_cache=False).logits[:, -1, digit_ids_t].float()
            zc = logits9[0, tgt]
            other = logits9[0].clone()
            other[tgt] = float("-inf")
            margin = zc - torch.logsumexp(other, dim=0)
            margin.backward()

            for mi, L in enumerate(tca_layers):
                u_mat, s_vec, v_mat = facs[L]
                din, dout = v_mat.shape[0], u_mat.shape[0]
                xb = cap_x[mi].float()
                ab = cap_g[mi].float()
                xv = xb @ v_mat
                au = ab @ u_mat
                delta = -s_vec * (xv * au).sum(dim=1)
                xtau = torch.einsum("bpi,bpr->bir", xb, au)
                atxv = torch.einsum("bpo,bpr->bor", ab, xv)
                sdr = s_vec * xtau.norm(dim=1) / math.sqrt(din)
                sdw = s_vec * atxv.norm(dim=1) / math.sqrt(dout)
                zr = (delta / sdr.clamp_min(1e-12)).abs()
                zw = (delta / sdw.clamp_min(1e-12)).abs()
                zr_gt3[mi] += int((zr > 3.0).sum().item())
                zw_gt3[mi] += int((zw > 3.0).sum().item())
                x_sub, a_sub = xb[:, -32:, :], ab[:, -32:, :]
                frag[mi, loc_j] = float(
                    ((x_sub @ x_sub.transpose(1, 2)) * (a_sub @ a_sub.transpose(1, 2)))
                    .sum().clamp_min(0).sqrt().item()
                )
        finally:
            for hk in handles:
                hk.remove()
            del cap_x, cap_g

    tgt_all = np.array([int(it["target_digit"]) - 1 for it in items], dtype=np.int64)
    denom = float(len(s1_indices) * TOP_SVD)
    return {
        "profile": profile,
        "model": MODEL_LOCKS[profile]["repo"],
        "revision": MODEL_LOCKS[profile]["revision"],
        "num_layers": n_layers,
        "mid_layer_index": mid_layer + 1,
        "bundle_sha256": bundle["bundle_sha256"],
        "items_target_correct": (probs_final.argmax(axis=-1) == tgt_all).astype(int).tolist(),
        "items_mid_lens_correct": (probs_mid_lens.argmax(axis=-1) == tgt_all).astype(int).tolist(),
        "items_target_mean_logprob": float(
            np.log(np.clip(probs_final[np.arange(n), tgt_all], 1e-12, 1.0)).mean()
        ),
        "tca_s1": {
            "layers": [f"L{L + 1}" for L in tca_layers],
            "frag_mean": frag.mean(axis=1).tolist(),
            "zr_gt3_share": (zr_gt3 / denom).tolist(),
            "zw_gt3_share": (zw_gt3 / denom).tolist(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("nanbeige", "qwen4b", "qwen3_4b_inst"), required=True)
    parser.add_argument("--out", type=Path, default=Path("mechinterp/loopscope02/results"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mem-fraction", type=float, default=0.72)
    parser.add_argument("--max-loops", type=int, default=8)
    parser.add_argument("--min-watts", type=float, default=40.0)
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        watts = power_limit_watts()
        if watts is not None and watts < args.min_watts:
            print(f"[power-guard] GPU enforced limit {watts}W < {args.min_watts}W; pausing (75).")
            return EXIT_POWER_CAP
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
        torch.backends.cuda.matmul.allow_tf32 = False

    args.out.mkdir(parents=True, exist_ok=True)
    bundle_path = args.out / "benchmark_bundle.json"
    if bundle_path.exists():
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    else:
        bundle = SP.build_benchmark_bundle()
        bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    t0 = time.time()
    root = snapshot_path(MODEL_LOCKS[args.profile])
    if args.profile == "nanbeige":
        model, enc = load_nanbeige(root, args.device, bundle)
        res = run_nanbeige_suite(model, enc, bundle, max_loops=args.max_loops, min_watts=args.min_watts)
        res["elapsed_sec"] = time.time() - t0
        out_file = args.out / "nanbeige_results.json"
        out_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
        c1 = np.mean(res["items_target_correct_by_loop"][0])
        c2 = np.mean(res["items_target_correct_by_loop"][1])
        c8 = np.mean(res["items_target_correct_by_loop"][-1])
        print(
            f"[NANBEIGE DONE] {res['elapsed_sec']:.1f}s | acc loop1={c1:.3f} loop2={c2:.3f} loop{args.max_loops}={c8:.3f} "
            f"| r_pass1_pass2={res['mean_r_pass1_vs_pass2']:.3f}"
        )
    else:
        model, enc = load_qwen(args.profile, root, args.device, bundle)
        res = run_qwen_suite(args.profile, model, enc, bundle, min_watts=args.min_watts)
        res["elapsed_sec"] = time.time() - t0
        out_file = args.out / f"{args.profile}_results.json"
        out_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(
            f"[{args.profile.upper()} DONE] {res['elapsed_sec']:.1f}s | "
            f"acc={np.mean(res['items_target_correct']):.3f} (mid={np.mean(res['items_mid_lens_correct']):.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
