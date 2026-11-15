#!/usr/bin/env python3
"""ARG-03-N: local Nanbeige context-fiber gym reusing the pinned ARG-02 adapter.

Readout-preserving causal patches operate on context positions BEFORE the
three scored positions. Produces model-only answer tapes; labels/futures stay
with the separate EvalMath scorer. This runner never imports EvalMath.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from .core import (
    arg02_program_scores,
    nanbeige_context_swap,
    nanbeige_norm_matched_random_context_swap,
    ranking_comparison,
    sha256_file,
    write_json,
)

HERE = Path(__file__).resolve().parent
EXTENDED_TEMPS = (0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
SURFACE_VARIANTS = ("canonical", "swap", "space", "swap_space")


def assert_oracle_isolated() -> None:
    leaked = [
        name
        for name in sys.modules
        if name == "spiral_ln"
        or name.startswith("spiral_ln.")
        or "algebraic_reasoning_oracle" in name
    ]
    if leaked:
        raise RuntimeError(f"oracle isolation violated in model runner: {leaked}")


def public_rows(path: Path) -> list[dict]:
    required = {"case_id", "topology", "observed_degree", "demos", "query", "public_sha256"}
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != required:
            raise ValueError(f"line {i}: runner received hidden or missing fields: {sorted(set(row) ^ required)}")
        if len(row["demos"]) != 2 or row["observed_degree"] != 12:
            raise ValueError(f"line {i}: unexpected ARG-02 record")
        rows.append(row)
    if len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate case IDs")
    return rows


def _reinit_rotary_buffers(module: torch.nn.Module, target_dev: torch.device) -> None:
    for sub in module.modules():
        if hasattr(sub, "inv_freq") and hasattr(sub, "base") and hasattr(sub, "dim"):
            with torch.device(target_dev):
                inv_freq = 1.0 / (
                    sub.base ** (torch.arange(0, sub.dim, 2, dtype=torch.int64).float() / sub.dim)
                )
            sub.register_buffer("inv_freq", inv_freq, persistent=False)


class StreamedNanbeigeLoop:
    """Bit-exact Nanbeige full-stack loop supporting low-VRAM CUDA layer streaming.

    All layer, RMSNorm, RoPE, SDPA, and `lm_head` arithmetic executes on `exec_device`
    in the checkpoint's native dtype (`bfloat16` with `float32` readout softmax),
    producing bit-for-bit identical outputs to keeping all 22 layers resident on GPU
    while respecting shared-GPU VRAM budgets via a single reusable CUDA staging layer.
    """

    def __init__(self, model, exec_device: torch.device, n_gpu_layers: int, stage_layer=None):
        self.model = model
        self.base = model.model
        self.layers = self.base.layers
        self.cfg = model.config
        self.n = len(self.layers)
        self.block = (0, self.n)
        self._exec_device = exec_device
        self.n_gpu_layers = int(n_gpu_layers)
        self.stage_layer = stage_layer

    @property
    def device(self) -> torch.device:
        return self._exec_device

    def _load_stage(self, cpu_layer: torch.nn.Module) -> torch.nn.Module:
        if self.stage_layer is None:
            raise RuntimeError("offloaded layer encountered without initialized CUDA stage_layer")
        for (k_dst, p_dst), (k_src, p_src) in zip(
            self.stage_layer.named_parameters(), cpu_layer.named_parameters(), strict=True
        ):
            if k_dst != k_src:
                raise RuntimeError(f"stage parameter mismatch: {k_dst} != {k_src}")
            p_dst.data.copy_(p_src.data, non_blocking=False)
        if hasattr(self.stage_layer, "self_attn") and hasattr(cpu_layer, "self_attn"):
            self.stage_layer.self_attn.layer_idx = cpu_layer.self_attn.layer_idx
        return self.stage_layer

    @torch.no_grad()
    def prepare(self, ids: torch.Tensor):
        emb_dev = self.base.embed_tokens.weight.device
        emb = self.base.embed_tokens(ids.to(emb_dev)).to(self._exec_device)
        rows, length = ids.shape
        cache_position = torch.arange(length, device=self._exec_device)
        pos = cache_position.unsqueeze(0).expand(rows, -1)
        mask = self.base._update_causal_mask(None, emb, cache_position, None, False)
        return emb, {"mask": mask, "pos": pos, "cache_position": cache_position}

    @torch.no_grad()
    def run_pairs(self, items: list[tuple[torch.Tensor, dict]]) -> list[torch.Tensor]:
        if not items:
            return []
        cur = list(items)
        for idx in range(self.n):
            active = self.layers[idx] if idx < self.n_gpu_layers else self._load_stage(self.layers[idx])
            for k, (h_k, ctx_k) in enumerate(cur):
                cur[k] = (
                    active(
                        h_k,
                        attention_mask=ctx_k["mask"],
                        position_ids=ctx_k["pos"],
                        past_key_value=None,
                        output_attentions=False,
                        use_cache=False,
                        cache_position=ctx_k["cache_position"],
                        loop_idx=0,
                    )[0],
                    ctx_k,
                )
        return [self.base.norm(h_k) for h_k, _ in cur]

    @torch.no_grad()
    def run(self, h: torch.Tensor, lo: int, hi: int, ctx: dict) -> torch.Tensor:
        if lo == hi:
            return h
        if (lo, hi) != (0, self.n):
            raise ValueError("Nanbeige runs whole loops only")
        return self.run_pairs([(h, ctx)])[0]

    @torch.no_grad()
    def readout(self, h: torch.Tensor, ctx: dict) -> torch.Tensor:
        return torch.log_softmax(self.model.lm_head(h[:, -3:, :]).float(), dim=-1)


def _true_cuda_free_and_total_bytes(cuda_dev: torch.device) -> tuple[int, int]:
    """Query global cross-process free VRAM via NVML/nvidia-smi (WDDM cudaMemGetInfo is per-process)."""
    idx = 0 if cuda_dev.index is None else int(cuda_dev.index)
    rt_free, rt_total = torch.cuda.mem_get_info(cuda_dev)
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={idx}",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            parts = [p.strip() for p in out.stdout.splitlines()[0].split(",")]
            nvml_free = int(float(parts[0]) * (1024**2))
            nvml_total = int(float(parts[1]) * (1024**2))
            return min(int(rt_free), nvml_free), nvml_total
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return int(rt_free), int(rt_total)


def _select_gpu_layer_count(
    num_layers: int,
    free_bytes: int,
    *,
    vram_reserve_gib: float,
    requested_gpu_layers: int,
) -> int:
    if requested_gpu_layers >= 0:
        return min(num_layers, requested_gpu_layers)
    free_gib = free_bytes / (1024**3)
    # Full model weights (~7.77 GiB) + activations (~0.35 GiB) + reserve fit without streaming
    if free_gib >= 8.25 + vram_reserve_gib:
        return num_layers
    # Reserve cushion for co-tenant GPU processes + lm_head (0.95 GiB) + 1 stage layer (0.27 GiB) + activations (0.35 GiB)
    usable_for_layers_gib = free_gib - vram_reserve_gib - 1.57
    n_keep = max(1, min(num_layers - 1, int(usable_for_layers_gib / 0.27)))
    return n_keep


def build_nanbeige_model_and_loop(
    snapshot_root: Path,
    *,
    exec_device: torch.device,
    dtype: torch.dtype,
    attn: str = "sdpa",
    n_gpu_layers: int,
    config_overrides: dict | None = None,
    random_init: bool = False,
):
    from transformers import AutoConfig, AutoModelForCausalLM

    snapshot_root = Path(snapshot_root)
    config = AutoConfig.from_pretrained(str(snapshot_root), trust_remote_code=True, local_files_only=True)
    for k, v in (config_overrides or {}).items():
        setattr(config, k, v)
    config._attn_implementation = attn
    n_total = int(config.num_hidden_layers)
    n_gpu = max(0, min(n_total, int(n_gpu_layers)))
    offload_active = n_gpu < n_total

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        if random_init:
            with torch.device("cpu"):
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation=attn)
        else:
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation=attn)

        emb_dev = torch.device("cpu") if offload_active else exec_device
        if not random_init:
            model.model.embed_tokens.to_empty(device=emb_dev)
            for i in range(n_total):
                layer_dev = exec_device if i < n_gpu else torch.device("cpu")
                model.model.layers[i].to_empty(device=layer_dev)
                _reinit_rotary_buffers(model.model.layers[i], layer_dev)
            model.model.norm.to_empty(device=exec_device)
            model.lm_head.to_empty(device=exec_device)
        else:
            model.model.embed_tokens.to(device=emb_dev)
            for i in range(n_total):
                layer_dev = exec_device if i < n_gpu else torch.device("cpu")
                model.model.layers[i].to(device=layer_dev)
                _reinit_rotary_buffers(model.model.layers[i], layer_dev)
            model.model.norm.to(device=exec_device)
            model.lm_head.to(device=exec_device)

        stage_layer = None
        if offload_active:
            layer_cls = type(model.model.layers[0])
            with torch.device("meta"):
                stage_layer = layer_cls(config, 0)
            stage_layer.to_empty(device=exec_device)
            _reinit_rotary_buffers(stage_layer, exec_device)
            stage_layer.eval()
    finally:
        torch.set_default_dtype(previous)

    if not random_init:
        index = json.loads((snapshot_root / "model.safetensors.index.json").read_text(encoding="utf-8"))
        params = model.state_dict()
        dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
        loaded = set()
        for shard in sorted(set(index["weight_map"].values())):
            with open(snapshot_root / shard, "rb") as f:
                size = int.from_bytes(f.read(8), "little")
                header = json.loads(f.read(size))
                base = 8 + size
                for key, spec in header.items():
                    if key == "__metadata__":
                        continue
                    start, end = spec["data_offsets"]
                    f.seek(base + start)
                    data = bytearray(f.read(end - start))
                    tensor = torch.frombuffer(data, dtype=dtypes[spec["dtype"]]).reshape(spec["shape"])
                    with torch.no_grad():
                        target_p = params[key]
                        target_p.copy_(tensor.to(device=target_p.device, dtype=target_p.dtype))
                    loaded.add(key)
        missing = sorted(k for k in set(params) - loaded if "rotary" not in k)
        if missing:
            raise AssertionError(f"weights not loaded: {missing[:5]}")

    model.eval()
    return model, StreamedNanbeigeLoop(model, exec_device=exec_device, n_gpu_layers=n_gpu, stage_layer=stage_layer)


def load_engine(
    root: Path,
    device: str,
    *,
    requested_gpu_layers: int = -1,
    vram_reserve_gib: float = 2.2,
):
    assert_oracle_isolated()
    sys.path.insert(0, str(root.resolve()))
    from research.algernon.algebraic_reasoning_gym.arg02 import nanbeige as arg02_nb
    from research.algernon.algebraic_reasoning_gym.arg02 import runner
    from research.algernon.algebraic_reasoning_gym.arg02.settings import load_lock

    entry = load_lock()["nanbeige"]
    snap_root = runner.verify_snapshot(entry)
    exec_dev = torch.device(device)
    config_obj = json.loads((snap_root / "config.json").read_text(encoding="utf-8"))
    n_total = int(config_obj["num_hidden_layers"])
    if exec_dev.type == "cuda":
        if exec_dev.index is None:
            exec_dev = torch.device("cuda", torch.cuda.current_device())
        free_b, _ = _true_cuda_free_and_total_bytes(exec_dev)
        n_gpu = _select_gpu_layer_count(
            n_total,
            free_b,
            vram_reserve_gib=vram_reserve_gib,
            requested_gpu_layers=requested_gpu_layers,
        )
    else:
        n_gpu = n_total

    dtype = getattr(torch, entry["dtype"])
    attn = entry.get("attn_implementation", "sdpa")
    model, loop = build_nanbeige_model_and_loop(
        snap_root,
        exec_device=exec_dev,
        dtype=dtype,
        attn=attn,
        n_gpu_layers=n_gpu,
    )
    arg02_nb.check_supported(model.config)
    tok = arg02_nb.load_tokenizer(snap_root, entry["bos_token_id"])
    engine = runner.Engine(model, tok, block=(0, len(model.model.layers)), shots=[], device=str(exec_dev))
    engine.loop = loop
    assert_oracle_isolated()
    return engine, entry


def _raw_tail_logits(loop, h: torch.Tensor, ctx: dict) -> torch.Tensor | None:
    """Extract raw pre-softmax [16, 3, vocab] logits when running on a real Nanbeige loop."""
    model = getattr(loop, "model", None)
    if model is not None and hasattr(model, "lm_head"):
        return model.lm_head(h[:, -3:, :]).float()
    return None


def _run_pairs(loop, items: list[tuple[torch.Tensor, dict]]) -> list[torch.Tensor]:
    if hasattr(loop, "run_pairs"):
        return loop.run_pairs(items)
    return [loop.run(h_k, 0, loop.n, ctx_k) for h_k, ctx_k in items]


def _prepare_surface_variant_inputs(engine, row: dict) -> list[tuple[str, torch.Tensor, dict]]:
    sig = inspect.signature(engine.rows)
    if "variant" not in sig.parameters:
        return []
    loop = engine.loop
    prepared = []
    for v in SURFACE_VARIANTS[1:]:
        ids_v, _ = engine.rows(row, variant=v)
        emb_v, ctx_v = loop.prepare(ids_v)
        prepared.append((v, emb_v, ctx_v))
    return prepared


def _assemble_equal_compute_scores(
    engine,
    lp1_canonical: torch.Tensor,
    variant_h1_pairs: list[tuple[str, torch.Tensor, dict]],
) -> dict[str, list[float]]:
    base_scores = arg02_program_scores(lp1_canonical, engine.op_ids)
    if not variant_h1_pairs:
        return {str(d): list(base_scores) for d in (1, 2, 4, 8)}

    loop = engine.loop
    by_var: dict[str, list[float]] = {"canonical": base_scores}
    for v, h1_v, ctx_v in variant_h1_pairs:
        lp1_v = loop.readout(h1_v, ctx_v)
        by_var[v] = arg02_program_scores(lp1_v, engine.op_ids)

    def _mean_first_m(var_seq: tuple[str, ...]) -> list[float]:
        m = len(var_seq)
        return [sum(by_var[v][i] for v in var_seq) / m for i in range(64)]

    return {
        "1": list(by_var["canonical"]),
        "2": _mean_first_m(SURFACE_VARIANTS[:2]),
        "4": _mean_first_m(SURFACE_VARIANTS[:4]),
        "8": _mean_first_m(SURFACE_VARIANTS[:4] + SURFACE_VARIANTS[:4]),
    }


@torch.inference_mode()
def evaluate_case(engine, row: dict, *, sigma: float, width: int, seed: int, depths=(1, 2, 4, 8)):
    assert_oracle_isolated()
    t_case0 = time.perf_counter()
    loop = engine.loop
    ids, prompt_tokens = engine.rows(row)
    emb, ctx = loop.prepare(ids)
    if emb.device.type == "cuda":
        torch.cuda.synchronize(emb.device)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(tuple(emb.shape), generator=gen, dtype=torch.float32).to(emb.device)
    emb_scale = emb.float().square().mean().sqrt()
    donor_emb = (emb.float() + sigma * emb_scale * noise).to(emb.dtype)

    var_inputs = _prepare_surface_variant_inputs(engine, row)
    step1_items = [(emb, ctx), (donor_emb, ctx)] + [(emb_v, ctx_v) for _, emb_v, ctx_v in var_inputs]
    step1_outs = _run_pairs(loop, step1_items)
    h1 = step1_outs[0]
    donor = step1_outs[1]
    var_h1_pairs = [
        (var_inputs[i][0], step1_outs[2 + i], var_inputs[i][2]) for i in range(len(var_inputs))
    ]

    raw1 = _raw_tail_logits(loop, h1, ctx)
    lp1 = torch.log_softmax(raw1, dim=-1) if raw1 is not None else loop.readout(h1, ctx)

    patched, patch_meta = nanbeige_context_swap(h1, donor, width=width, readout_tail=3)
    rand_patched, rand_meta = nanbeige_norm_matched_random_context_swap(
        h1, patched, width=width, readout_tail=3, seed=seed + 424242
    )
    self_patched, self_meta = nanbeige_context_swap(h1, h1, width=width, readout_tail=3)

    lp1_after = loop.readout(patched, ctx)
    lp1_rand_after = loop.readout(rand_patched, ctx)
    lp1_self_after = loop.readout(self_patched, ctx)
    lp1_donor_full = loop.readout(donor, ctx)

    leak = float((lp1 - lp1_after).abs().max().item())
    leak_rand = float((lp1 - lp1_rand_after).abs().max().item())
    leak_self = float((lp1 - lp1_self_after).abs().max().item())
    if max(leak, leak_rand, leak_self) > 1e-6:
        raise AssertionError(
            f"full-vocabulary output changed before continuation: donor={leak}, rand={leak_rand}, self={leak_self}"
        )

    temp_source = raw1 if raw1 is not None else lp1.float()
    eq_comp = _assemble_equal_compute_scores(engine, lp1, var_h1_pairs)
    base1_scores = arg02_program_scores(lp1, engine.op_ids)

    # Compact [16, 3, 4] operator raw logits / logprobs for exact dev-only per-position calibration
    tok_idx = [int(engine.op_ids[c]) for c in "INSR"]
    op_site_raw = temp_source[:, :, tok_idx].detach().cpu().tolist()

    row_stats = {
        "case_id": row["case_id"],
        "public_sha256": row["public_sha256"],
        "prompt_tokens": prompt_tokens,
        "baseline": {"1": base1_scores},
        "context_swap": {"1": arg02_program_scores(lp1_after, engine.op_ids)},
        "random_context_norm_matched": {"1": arg02_program_scores(lp1_rand_after, engine.op_ids)},
        "self_context_control": {"1": arg02_program_scores(lp1_self_after, engine.op_ids)},
        "donor_clean_perturbed": {"1": arg02_program_scores(lp1_donor_full, engine.op_ids)},
        "open_loop": {"1": list(base1_scores)},
        "equal_compute": {k: v for k, v in eq_comp.items() if int(k) in depths},
        "temperature_controls": {
            f"T{t:g}": arg02_program_scores(torch.log_softmax(temp_source / t, dim=-1), engine.op_ids)
            for t in (0.5, 0.75, 1.25, 2.0)
        },
        "temperature_grid": {
            f"T{t:g}": arg02_program_scores(torch.log_softmax(temp_source / t, dim=-1), engine.op_ids)
            for t in EXTENDED_TEMPS
        },
        "loop1_operator_site_logits": op_site_raw,
        "patch": patch_meta,
        "random_patch": rand_meta,
        "self_patch": self_meta,
        "max_precontinuation_logprob_error": leak,
        "max_precontinuation_random_logprob_error": leak_rand,
        "max_precontinuation_self_logprob_error": leak_self,
        "hidden_state_norms": {
            "1": {
                "clean_l2": float(h1.float().norm().item()),
                "context_swap_delta_l2": float((patched.float() - h1.float()).norm().item()),
                "random_context_delta_l2": float((rand_patched.float() - h1.float()).norm().item()),
                "donor_full_delta_l2": float((donor.float() - h1.float()).norm().item()),
            }
        },
        "postloop_readout_distance": {},
        "postloop_random_readout_distance": {},
        "ranking_changes_vs_clean": {},
    }

    clean = h1
    intervention = patched
    rand_interv = rand_patched
    donor_cur = donor
    for t in range(2, max(depths) + 1):
        clean, intervention, rand_interv, donor_cur = _run_pairs(
            loop,
            [(clean, ctx), (intervention, ctx), (rand_interv, ctx), (donor_cur, ctx)],
        )
        if t in depths:
            lc = loop.readout(clean, ctx)
            li = loop.readout(intervention, ctx)
            lr = loop.readout(rand_interv, ctx)
            ld = loop.readout(donor_cur, ctx)
            open_state = (h1.float() + (t - 1) * (h1.float() - emb.float())).to(h1.dtype)
            lo = loop.readout(open_state, ctx)

            sc_clean = arg02_program_scores(lc, engine.op_ids)
            sc_swap = arg02_program_scores(li, engine.op_ids)
            sc_rand = arg02_program_scores(lr, engine.op_ids)
            sc_don = arg02_program_scores(ld, engine.op_ids)
            sc_open = arg02_program_scores(lo, engine.op_ids)

            key = str(t)
            row_stats["open_loop"][key] = sc_open
            row_stats["baseline"][key] = sc_clean
            row_stats["context_swap"][key] = sc_swap
            row_stats["random_context_norm_matched"][key] = sc_rand
            row_stats["donor_clean_perturbed"][key] = sc_don
            row_stats["postloop_readout_distance"][key] = float((lc.float() - li.float()).norm().item())
            row_stats["postloop_random_readout_distance"][key] = float((lc.float() - lr.float()).norm().item())
            row_stats["hidden_state_norms"][key] = {
                "clean_l2": float(clean.float().norm().item()),
                "context_swap_delta_l2": float((intervention.float() - clean.float()).norm().item()),
                "random_context_delta_l2": float((rand_interv.float() - clean.float()).norm().item()),
                "donor_full_delta_l2": float((donor_cur.float() - clean.float()).norm().item()),
            }
            row_stats["ranking_changes_vs_clean"][key] = {
                "context_swap": ranking_comparison(sc_clean, sc_swap),
                "random_context_norm_matched": ranking_comparison(sc_clean, sc_rand),
                "open_loop": ranking_comparison(sc_clean, sc_open),
            }

    if emb.device.type == "cuda":
        torch.cuda.synchronize(emb.device)
    row_stats["case_elapsed_seconds"] = round(time.perf_counter() - t_case0, 4)
    return row_stats


def main(argv=None):
    assert_oracle_isolated()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--public-jsonl", type=Path, required=True, help="Fresh ARG-03 public-only split")
    p.add_argument("--compactification-root", type=Path, required=True, help="Checkout containing arg02 runner/lock")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=4, help="Local pilot count; 0 means entire supplied split")
    p.add_argument("--sigma", type=float, default=0.02, help="Donor embedding-noise RMS scale")
    p.add_argument("--width", type=int, default=8, help="Context positions immediately before scored tail")
    p.add_argument("--seed", type=int, default=20261115)
    p.add_argument("--min-free-gib", type=float, default=4.2,
                   help="Minimum free GPU VRAM (GiB) required before running bit-exact CUDA Nanbeige4.2-3B")
    p.add_argument("--vram-reserve-gib", type=float, default=2.2,
                   help="Untouched free GPU VRAM cushion (GiB) preserved for co-tenant processes")
    p.add_argument("--gpu-layers", type=int, default=-1,
                   help="Number of permanent GPU layers (-1 selects automatically from free VRAM)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--permit-heldout-test", action="store_true",
                   help="Explicit gate after protocol registration; never use during dev tuning")
    p.add_argument("--allow-small-vram", action="store_true")
    args = p.parse_args(argv)
    if args.limit < 0 or args.width < 1 or not (0 < args.sigma < 1):
        p.error("invalid limit, width, or sigma")
    rows = public_rows(args.public_jsonl)
    if args.public_jsonl.name == "test.jsonl" and not args.permit_heldout_test:
        raise RuntimeError("held-out test sealed: pass --permit-heldout-test only after protocol lock")
    chosen = rows[: args.limit] if args.limit else rows
    metadata = {
        "schema": "darkfibers.arg03.nanbeige.local.v2",
        "public_jsonl_sha256": sha256_file(args.public_jsonl),
        "source_runner": args.compactification_root.name,
        "chosen_count": len(chosen),
        "total_public_count": len(rows),
        "sigma": args.sigma,
        "width": args.width,
        "seed": args.seed,
        "device": args.device,
        "code_sha256": sha256_file(Path(__file__)),
        "oracle_isolated": True,
        "status": "dry-run" if args.dry_run else "running",
    }
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return 0
    old_manifest = args.out_dir / "manifest.json"
    if old_manifest.is_file():
        prev = json.loads(old_manifest.read_text(encoding="utf-8"))
        for field in (
            "public_jsonl_sha256",
            "sigma",
            "width",
            "seed",
            "code_sha256",
            "chosen_count",
            "device",
        ):
            if prev.get(field) != metadata[field]:
                raise RuntimeError(f"refusing mixed-config tape; manifest field changed: {field}")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable; use --dry-run or --device cpu")
        torch.cuda.init()
        cuda_dev = torch.device(args.device)
        if cuda_dev.index is None:
            cuda_dev = torch.device("cuda", torch.cuda.current_device())
        free_mem, total_mem = _true_cuda_free_and_total_bytes(cuda_dev)
        metadata["initial_free_vram_bytes"] = int(free_mem)
        metadata["total_vram_bytes"] = int(total_mem)
        if total_mem < 12 * 1024**3 and not args.allow_small_vram:
            raise RuntimeError("BF16 Nanbeige adapter targets RTX 5080 (16 GiB); RTX 3050 4GB: run TRM gym instead")
        if free_mem < args.min_free_gib * (1024**3) and not args.allow_small_vram:
            raise RuntimeError(
                f"Insufficient free GPU VRAM for BF16 Nanbeige4.2-3B: "
                f"{free_mem / (1024**3):.2f} GiB free < {args.min_free_gib:.2f} GiB required "
                f"(another process is holding {(total_mem - free_mem) / (1024**3):.2f} GiB)."
            )
        torch.cuda.reset_peak_memory_stats(cuda_dev)

    engine, entry = load_engine(
        args.compactification_root,
        args.device,
        requested_gpu_layers=args.gpu_layers,
        vram_reserve_gib=args.vram_reserve_gib,
    )
    metadata["checkpoint_revision"] = entry["revision"]
    metadata["checkpoint_lock_sha256"] = sha256_file(
        args.compactification_root / "research/algernon/algebraic_reasoning_gym/arg02/config/checkpoint_lock.json"
    )
    metadata["n_gpu_layers"] = int(engine.loop.n_gpu_layers)
    metadata["n_total_layers"] = int(engine.loop.n)
    metadata["layer_streaming"] = bool(engine.loop.n_gpu_layers < engine.loop.n)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "manifest.json", metadata)
    tape = args.out_dir / "answer_tape.jsonl"
    existing = {}
    if tape.is_file():
        for line in tape.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                existing[record["case_id"]] = record
    start = time.perf_counter()
    for idx, row in enumerate(chosen, 1):
        if row["case_id"] in existing:
            if existing[row["case_id"]]["public_sha256"] != row["public_sha256"]:
                raise ValueError("resume identity mismatch")
            continue
        stable_seed = args.seed ^ int(row["public_sha256"][:8], 16)
        result = evaluate_case(engine, row, sigma=args.sigma, width=args.width, seed=stable_seed)
        with tape.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n")
        print(
            f"{idx}/{len(chosen)} {row['case_id']}: parity={result['max_precontinuation_logprob_error']:.2e} "
            f"d2_l2={result['postloop_readout_distance']['2']:.2f} "
            f"t={result['case_elapsed_seconds']:.2f}s",
            flush=True,
        )
    assert_oracle_isolated()
    metadata["status"] = "completed"
    metadata["elapsed_seconds"] = round(time.perf_counter() - start, 2)
    if args.device.startswith("cuda"):
        metadata["peak_allocated_vram_bytes"] = int(torch.cuda.max_memory_allocated(args.device))
    metadata["answer_tape_sha256"] = sha256_file(tape)
    write_json(args.out_dir / "manifest.json", metadata)
    print("wrote", tape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
