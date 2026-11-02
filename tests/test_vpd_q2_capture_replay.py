"""Tensor/hook simulation only: no checkpoint, LM inference, weights or GPU."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from experiments.authz_mechinterp.cluster_vpd_for_sae import content_sha256
from experiments.nanbeige_reasoning.positions import capture_positions
from experiments.nanbeige_reasoning.q2_replay import replay_q2, SITES, WIDTH


class TensorBackbone:
    def __init__(self):
        self.layers = [torch.nn.Identity() for _ in range(22)]
        self.norm = torch.nn.Identity()
        self.calls = 0
        self.passes = 2
        self.output_dtype = torch.bfloat16
        self.nonfinite = False

    def __call__(self, tokens, *, use_cache):
        assert use_cache is False
        assert not torch.is_grad_enabled()
        self.calls += 1
        # A position/pass/site sentinel, not a language-model computation.
        for p in range(self.passes):
            h = torch.arange(tokens.shape[1], dtype=self.output_dtype)[None, :, None].expand(1, -1, WIDTH)
            if self.nonfinite:
                h = h.clone()
                h[:, -1] = float("nan")
            self.layers[11](h + p * 64)
            self.norm(h + p * 64 + 16)


def fixture():
    tokens = list(range(40))
    positions, regions = capture_positions(8, 40)
    record = dict(id="sentinel", family="synthetic", split="train", prompt_tokens=8,
                  generated_tokens=32, positions=positions,
                  regions={k: list(v) for k, v in regions.items()},
                  token_ids_content_sha256=content_sha256(tokens))
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=22, num_loops=2,
                           hidden_size=WIDTH, loop_loss_weights=[], vocab_size=100),
                           model=TensorBackbone(), training=False, dtype=torch.bfloat16,
                           device=torch.device("cpu"))
    return model, tokens, record


def assert_no_hooks(model):
    assert all(not m._forward_hooks for m in model.model.layers)
    assert not model.model.norm._forward_hooks


def test_exact_rows_sites_passes_single_replay_and_no_mutation():
    model, tokens, record = fixture()
    before = deepcopy(record)
    config_before = vars(model.config).copy()
    result = replay_q2(model, tokens, record, max_sequence_tokens=40)
    assert model.model.calls == 1
    assert record == before and vars(model.config) == config_before
    assert set(result["arrays"]) == {f"{s}__pass_{p}" for s in SITES for p in (0, 1)}
    for si, site in enumerate(SITES):
        for p in (0, 1):
            values = result["arrays"][f"{site}__pass_{p}"]
            assert values.shape == (32, WIDTH) and str(values.dtype) == "float32"
            assert (values == torch.tensor(record["positions"])[:, None].numpy() + 64*p + 16*si).all()
    assert result["record"]["calls"] == {s: 2 for s in SITES}
    assert_no_hooks(model)


@pytest.mark.parametrize("change", ["short", "hash", "length", "role", "bool_token", "vocab", "bound", "loops", "training", "dtype", "quantized"])
def test_reject_before_forward(change):
    model, tokens, record = fixture()
    bound = 40
    if change == "short": record["generated_tokens"] = 8
    if change == "hash": record["token_ids_content_sha256"] = "0" * 64
    if change == "length": tokens = tokens[:-1]
    if change == "role": record["split"] = "unknown"
    if change == "bool_token": tokens[0] = False
    if change == "vocab": model.config.vocab_size = 20
    if change == "bound": bound = 39
    if change == "loops": model.config.num_loops = 1
    if change == "training": model.training = True
    if change == "dtype": model.dtype = torch.float32
    if change == "quantized": model.is_quantized = True
    with pytest.raises(ValueError):
        replay_q2(model, tokens, record, max_sequence_tokens=bound)
    assert model.model.calls == 0
    assert_no_hooks(model)


@pytest.mark.parametrize("passes", [1, 3])
def test_wrong_invocation_count_fails_and_removes_hooks(passes):
    model, tokens, record = fixture()
    model.model.passes = passes
    with pytest.raises(RuntimeError):
        replay_q2(model, tokens, record, max_sequence_tokens=40)
    assert_no_hooks(model)


@pytest.mark.parametrize("mode", ["dtype", "nonfinite"])
def test_bad_physical_output_fails_and_removes_hooks(mode):
    model, tokens, record = fixture()
    if mode == "dtype": model.model.output_dtype = torch.float32
    else: model.model.nonfinite = True
    with pytest.raises(ValueError):
        replay_q2(model, tokens, record, max_sequence_tokens=40)
    assert_no_hooks(model)
