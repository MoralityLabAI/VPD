"""Causal and KV-cache controls for answer-generating component interventions."""
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.nanbeige_looped.run_multipass import architecture_canary
from experiments.nanbeige_reasoning.pass_interventions import PassComponentIntervention


@pytest.fixture
def setup():
    torch.set_num_threads(1)
    torch.manual_seed(102)
    model, _ = architecture_canary(Path('models/Nanbeige4.2-3B-tokenizer'))
    path = 'model.layers.11.mlp.down_proj'
    weight = model.get_submodule(path).weight
    v = torch.randn(weight.shape[1], 3) * .1
    u = torch.randn(3, weight.shape[0]) * .1
    intervention = PassComponentIntervention(model, path, v, u, component=1)
    return model, path, v, u, intervention, torch.tensor([[2, 5, 11, 6, 1, 9, 4]])


def test_both_passes_match_actual_shared_weight_edit_and_restore(setup):
    model, path, v, u, intervention, tokens = setup
    with torch.inference_mode():
        original = model(tokens, use_cache=False).logits
        with intervention.apply(.1, [0, 1]):
            hooked = model(tokens, use_cache=False).logits
        weight = model.get_submodule(path).weight
        saved = weight.clone()
        try:
            weight.add_(.1 * (v[:, 1:2] @ u[1:2]).T)
            edited = model(tokens, use_cache=False).logits
        finally:
            weight.copy_(saved)
        torch.testing.assert_close(hooked, edited, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(model(tokens, use_cache=False).logits, original, atol=0, rtol=0)
        assert torch.max(torch.abs(hooked - original)) > 1e-5


@pytest.mark.parametrize('selected', [[0], [1], [0, 1]])
def test_cached_decode_matches_full_prefix_with_intervention(setup, selected):
    model, _, _, _, intervention, tokens = setup
    with torch.inference_mode(), intervention.apply(-.1, selected):
        full = model(tokens, use_cache=False).logits[:, -1]
        prefix = model(tokens[:, :-2], use_cache=True)
        cache = prefix.past_key_values
        for end in (tokens.shape[1] - 1, tokens.shape[1]):
            output = model(tokens[:, end-1:end], past_key_values=cache,
                           attention_mask=torch.ones_like(tokens[:, :end]), use_cache=True)
            cache = output.past_key_values
        assert len(cache) == 44
        assert intervention.model_calls == 4
        torch.testing.assert_close(output.logits[:, -1], full, atol=1e-6, rtol=1e-5)


def test_second_only_preserves_entire_first_pass_and_cleanup_after_error(setup):
    model, _, _, _, intervention, tokens = setup
    captures = []
    handle = model.model.norm.register_forward_hook(lambda m, a, o: captures.append(o.detach().clone()))
    try:
        with torch.inference_mode():
            model(tokens, use_cache=False)
            baseline = captures[:]
            captures.clear()
            with intervention.apply(.1, [1]):
                model(tokens, use_cache=False)
            torch.testing.assert_close(captures[0], baseline[0], atol=0, rtol=0)
            assert torch.max(torch.abs(captures[1] - baseline[1])) > 1e-5
            with pytest.raises(RuntimeError, match='deliberate'):
                with intervention.apply(.1, [0]):
                    raise RuntimeError('deliberate interrupted generation')
            captures.clear()
            model(tokens, use_cache=False)
            for current, original in zip(captures, baseline, strict=True):
                torch.testing.assert_close(current, original, atol=0, rtol=0)
    finally:
        handle.remove()
    assert not intervention.active


def test_invalid_scope_and_factor_dimensions_fail_before_generation(setup):
    model, path, v, u, intervention, _ = setup
    with pytest.raises(ValueError):
        PassComponentIntervention(model, path, v[:-1], u, component=1)
    for alpha, selected in [(np.nan, [0]), (.1, []), (.1, [2]), (.1, [False])]:
        with pytest.raises(ValueError):
            with intervention.apply(alpha, selected):
                pass


def test_greedy_generation_matches_uncached_manual_decode_and_resets_each_call(setup):
    model, _, _, _, intervention, tokens = setup
    with torch.inference_mode(), intervention.apply(.1, [1]):
        generated = model.generate(tokens, attention_mask=torch.ones_like(tokens), do_sample=False,
            max_new_tokens=4, use_cache=True, eos_token_id=None, pad_token_id=0)
        assert intervention.model_calls == 4
    manual = tokens
    with torch.inference_mode(), intervention.apply(.1, [1]):
        for _ in range(4):
            predicted = model(manual, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
            manual = torch.cat((manual, predicted), dim=1)
    torch.testing.assert_close(generated, manual, atol=0, rtol=0)
