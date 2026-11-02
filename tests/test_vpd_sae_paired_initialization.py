import numpy as np
import pytest
import torch

from experiments.authz_mechinterp.check_sae_paired_initialization_v1 import initialize_pair, check_pair


def make_pair(seed=1707):
    bias = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    models = initialize_pair(bias, width=5, features=10, top_k=2, seed=seed)
    return models, bias


def test_exact_independent_frozen_pair_and_rng_restoration():
    before = torch.get_rng_state().clone()
    models, bias = make_pair()
    result = check_pair(models, bias, width=5, features=10, top_k=2)
    assert result["exact_parameter_equality"] and result["disjoint_parameter_storage"]
    assert torch.equal(before, torch.get_rng_state())


def test_same_seed_same_state_and_other_seed_changes_matrix():
    a, bias = make_pair()
    b, _ = make_pair()
    c, _ = make_pair(250829)
    assert check_pair(a, bias, width=5, features=10, top_k=2) == check_pair(b, bias, width=5, features=10, top_k=2)
    assert not torch.equal(a["Q2-N"].encoder.weight, c["Q2-N"].encoder.weight)


def test_changing_input_bias_after_initialization_cannot_change_models():
    models, bias = make_pair()
    bias[:] = 99
    assert models["Q2-N"].pre_bias.tolist() == [1, 2, 3, 4, 5]
    assert models["Q2-B"].pre_bias.tolist() == [1, 2, 3, 4, 5]


def test_parameter_alias_is_rejected():
    models, bias = make_pair()
    models["Q2-B"] = models["Q2-N"]
    with pytest.raises(ValueError, match="share parameter storage"):
        check_pair(models, bias, width=5, features=10, top_k=2)


@pytest.mark.parametrize("mutation", ["bias", "encoder_bias", "decoder", "nonfinite", "unfreeze"])
def test_disposable_fixture_mutations_rejected(mutation):
    models, bias = make_pair()
    model = models["Q2-B"]
    with torch.no_grad():
        if mutation == "bias":
            model.pre_bias[0] += 1
        elif mutation == "encoder_bias":
            model.encoder.bias[0] += 1
        elif mutation == "decoder":
            model.decoder.weight.mul_(2)
        elif mutation == "nonfinite":
            model.encoder.weight[0, 0] = float("nan")
        else:
            model.requires_grad_(True)
    with pytest.raises(ValueError):
        check_pair(models, bias, width=5, features=10, top_k=2)


@pytest.mark.parametrize("kwargs", [{"width": True}, {"features": 0}, {"top_k": 11}, {"seed": -1}])
def test_invalid_dimensions_and_seed_rejected(kwargs):
    options = {"width": 5, "features": 10, "top_k": 2, "seed": 1707, **kwargs}
    with pytest.raises(ValueError):
        initialize_pair(np.zeros(5, dtype=np.float32), **options)


def test_bias_dtype_and_global_dtype_are_explicit():
    with pytest.raises(ValueError, match="float32 common bias"):
        initialize_pair(np.zeros(5, dtype=np.float64), width=5, features=10, top_k=2, seed=1707)
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError, match="constructor default"):
            make_pair()
    finally:
        torch.set_default_dtype(original)
