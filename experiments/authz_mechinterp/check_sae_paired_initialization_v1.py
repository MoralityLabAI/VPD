"""Q2 initialization-only contract and tiny fixture check; no fitting or forward pass."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from experiments.nanbeige_looped import sae as sae_module
from experiments.authz_mechinterp import bind_sae_paired_activation_rows_v1 as binding
from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256, sha256_file, write_json

ARMS = ("Q2-N", "Q2-B")


def initialize_pair(common_pre_bias, *, width, features, top_k, seed):
    if (any(type(v) is not int for v in (width, features, top_k, seed))
            or not 1 <= width <= 3072 or not 1 <= top_k <= features <= 6144 or not 0 <= seed < 2**64):
        raise ValueError("bounded explicit integer dimensions and seed required")
    bias = np.asarray(common_pre_bias)
    if bias.shape != (width,) or bias.dtype != np.float32 or not np.isfinite(bias).all():
        raise ValueError("finite float32 common bias with declared width required")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("float32 constructor default required; no silent dtype override")
    # Seed only the CPU generator inside a restoring context; do not touch CUDA RNGs.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.default_generator.manual_seed(seed)
        template = sae_module.SharedSAE(width, features, top_k)
        with torch.no_grad():
            template.pre_bias.copy_(torch.tensor(bias.copy(), dtype=torch.float32, device="cpu"))
        template.eval().requires_grad_(False)
        models = {arm: copy.deepcopy(template) for arm in ARMS}
    return models


def check_pair(models, common_pre_bias, *, width, features, top_k):
    if set(models) != set(ARMS):
        raise ValueError("exactly two Q2 arms required")
    expected_shapes = {"pre_bias": (width,), "encoder.weight": (features, width),
                       "encoder.bias": (features,), "decoder.weight": (width, features)}
    hashes, storage = {}, {}
    expected_bias = torch.tensor(np.array(common_pre_bias, copy=True), dtype=torch.float32, device="cpu")
    for arm in ARMS:
        model = models[arm]
        state = model.state_dict()
        if set(state) != set(expected_shapes) or model.top_k != top_k or model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("unexpected architecture or non-frozen initialization")
        entries = {}
        for name, tensor in state.items():
            if (tuple(tensor.shape) != expected_shapes[name] or tensor.device.type != "cpu"
                    or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all()):
                raise ValueError("initial tensor shape/device/dtype/finite contract differs")
            entries[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                             "array_content_sha256": binding.array_digest(tensor.numpy())}
        if not torch.equal(state["pre_bias"], expected_bias) or torch.count_nonzero(state["encoder.bias"]).item() != 0:
            raise ValueError("common pre-bias or zero encoder bias differs")
        norms = state["decoder.weight"].norm(dim=0)
        initial_decoder = state["encoder.weight"].T
        initial_decoder = initial_decoder/initial_decoder.norm(dim=0, keepdim=True).clamp_min(1e-8)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-6, rtol=0) or not torch.allclose(state["decoder.weight"], initial_decoder, atol=1e-7, rtol=1e-6):
            raise ValueError("initial decoder normalization/orientation differs")
        hashes[arm] = {"parameters": entries, "state_sha256": content_sha256({"top_k": top_k, "parameters": entries})}
        storage[arm] = {p.untyped_storage().data_ptr() for p in model.parameters()}
    if hashes[ARMS[0]] != hashes[ARMS[1]]:
        raise ValueError("paired initial parameter values differ")
    if storage[ARMS[0]] & storage[ARMS[1]]:
        raise ValueError("paired models share parameter storage")
    return {"arms": hashes, "exact_parameter_equality": True, "disjoint_parameter_storage": True,
            "frozen_cpu_float32": True, "common_pre_bias_sha256": binding.array_digest(np.asarray(common_pre_bias))}


def build_report(binding_path):
    source = json.loads(binding_path.read_text(encoding="utf-8"))
    if source != binding.build_report(Path(source["source"]["schedule_path"])):
        raise ValueError("synthetic binding receipt does not reproduce")
    bound = binding.bind_rows(*binding.synthetic_fixture(), expected_width=5)
    rng_before = torch.get_rng_state().clone()
    results = []
    for site in binding.SITES:
        for seed in (1707, 250829, 260829):
            models = initialize_pair(bound["common_pre_bias"][site], width=5, features=10, top_k=2, seed=seed)
            result = check_pair(models, bound["common_pre_bias"][site], width=5, features=10, top_k=2)
            results.append({"site": site, "initialization_seed": seed, **result})
    if not torch.equal(rng_before, torch.get_rng_state()):
        raise ValueError("initialization changed the caller CPU RNG state")
    for site in binding.SITES:
        if len({r["arms"][ARMS[0]]["parameters"]["encoder.weight"]["array_content_sha256"] for r in results if r["site"] == site}) != 3:
            raise ValueError("distinct initialization seeds did not change the encoder")
    body = {"schema": "sch.vpd-sae-paired-initialization-benchmark.v1", "status": "tiny_initialization_contract_verified",
            "source": {"binding_path": str(binding_path.resolve()), "sha256": sha256_file(binding_path), "report_id": source["report_id"]},
            "fixture": {"width": 5, "features": 10, "top_k": 2, "sites": list(binding.SITES)},
            "results": results, "caller_cpu_rng_preserved": True,
            "design": "Construct one existing SharedSAE template per site/seed, copy the complete-pool bias, freeze it, then deepcopy independently for each arm.",
            "limitations": ["Tiny synthetic fixtures only; no 3072-by-6144 research initialization or fresh activation pack was verified.",
                            "No forward pass, feature interpretation, loss, backward pass, optimizer, or trained checkpoint export occurs.",
                            "The historical fit_shared_sae helper recomputes bias from its input rows and must not independently initialize the two Q2 arms.",
                            "Returned models are frozen CPU float32 instances. A later authorized trainer must explicitly enable training and verify this contract on its real runtime."],
            "boundary": {"tiny_synthetic_initializations_created": True, "research_weights_modified": False, "training_performed": False,
                         "model_forward_performed": False, "optimizer_steps": 0, "research_initialization_verified": False,
                         "paid_compute_launched": False, "launch_authorized": False, "operative_admissions": 0},
            "provenance": {"runner_sha256": sha256_file(Path(__file__)), "shared_sae_sha256": sha256_file(Path(sae_module.__file__)),
                           "binding_runner_sha256": sha256_file(Path(binding.__file__)), "torch_version": torch.__version__}}
    return {**body, "report_id": content_sha256(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        observed = json.loads(args.verify.read_text(encoding="utf-8"))
        report = build_report(Path(observed["source"]["binding_path"]))
        if observed != report:
            raise ValueError("initialization receipt does not reproduce")
    else:
        if not args.binding:
            parser.error("--binding required")
        if args.output.exists():
            raise ValueError("refusing to overwrite existing receipt")
        report = build_report(args.binding)
        write_json(args.output, report)
    print(json.dumps({"report_id": report["report_id"], "status": report["status"],
                      "verified_tiny_pairs": len(report["results"]), "optimizer_steps": 0}, indent=2))


if __name__ == "__main__":
    main()
