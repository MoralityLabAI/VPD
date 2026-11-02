"""Capture baseline and signed component effects using exact token alignment."""
from __future__ import annotations

from contextlib import nullcontext
from functools import partial

import numpy as np
import torch

from experiments.nanbeige_looped import vpd_math as vpd
from experiments.nanbeige_looped.capture import RepeatedForwardCapture, torch_tail_rows
from experiments.nanbeige_looped.sae import encode_rows


def capture_forward(target, tokens, sites, *, rows=16):
    if tokens.ndim != 2 or tokens.shape[0] != 1 or rows < 1:
        raise ValueError("Capture requires one unpadded sequence and positive tail rows")
    modules = {name: target.model.get_submodule(name) for name in sites}
    capture = RepeatedForwardCapture(modules, expected_calls=target.passes,
                                    extract=partial(torch_tail_rows, rows=rows))
    with torch.inference_mode(), target.autocast(), capture:
        logits = target(tokens)[:, -1].float().cpu().numpy().copy()
    return capture.values, logits


def measure_component(target, tokens, sites, dictionaries, *, path, component,
                      alpha=0.05, rows=16):
    """All-pass effects preserve sharing. Single-pass effects break sharing diagnostically.

    Finite differences are measured through the full nonlinear SAE encoder;
    decoder projection of delta_h is deliberately not substituted for them.
    """
    if target.passes != 2 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("This paired diagnostic requires two passes and positive finite alpha")
    if set(sites) != set(dictionaries):
        raise ValueError("Exactly one common dictionary per capture site is required")
    vpd.clear_wrapper_masks(target.views)
    arrays, quality, metrics = {}, {}, {}
    states = {}
    scopes = {"baseline": None, "first": [0], "second": [1], "both": [0, 1]}
    baseline_logits = None
    for scope, selected in scopes.items():
        for sign, coefficient in (("zero", 0.0),) if selected is None else (("plus", alpha), ("minus", -alpha)):
            context = (target.intervene(path, component, coefficient, selected)
                       if selected is not None else nullcontext())
            with context:
                activations, logits = capture_forward(target, tokens, sites, rows=rows)
            label = "baseline" if selected is None else f"{scope}_{sign}"
            states[label] = activations
            arrays[f"{label}__last_logits"] = logits
            if selected is None:
                baseline_logits = torch.from_numpy(logits)
            candidate = int(baseline_logits[0].argmax())
            metrics[label] = {
                "reference_top_token_id": candidate,
                "reference_top_token_logit_delta": float(logits[0, candidate] - baseline_logits[0, candidate]),
                "next_token_kl": float(vpd.kl_logits(torch.from_numpy(logits), baseline_logits)),
            }
            for (site, invocation), hidden in activations.items():
                key = f"{site}__pass_{invocation}"
                z, q = encode_rows(dictionaries[site], hidden)
                arrays[f"{label}__{key}__h"] = hidden
                arrays[f"{label}__{key}__z"] = z
                quality[f"{label}__{key}"] = q
    for site in sites:
        for kind in ("h", "z"):
            arrays[f"baseline__{site}__pass_delta__{kind}"] = (
                arrays[f"baseline__{site}__pass_1__{kind}"]
                - arrays[f"baseline__{site}__pass_0__{kind}"])
            for invocation in range(2):
                key = f"{site}__pass_{invocation}__{kind}"
                base = arrays[f"baseline__{key}"]
                for scope in ("first", "second", "both"):
                    plus = arrays[f"{scope}_plus__{key}"]
                    minus = arrays[f"{scope}_minus__{key}"]
                    arrays[f"{scope}__{key}__central_difference"] = (plus - minus) / (2 * alpha)
                    arrays[f"{scope}__{key}__positive_effect"] = plus - base
                arrays[f"interaction__{key}"] = (arrays[f"both_plus__{key}"]
                    - arrays[f"first_plus__{key}"] - arrays[f"second_plus__{key}"] + base)
        # A second-pass-only intervention must never change first-pass captures.
        np.testing.assert_array_equal(states["baseline"][(site, 0)],
                                      states["second_plus"][(site, 0)])
        np.testing.assert_array_equal(states["baseline"][(site, 0)],
                                      states["second_minus"][(site, 0)])
    arrays["token_positions"] = np.arange(max(0, tokens.shape[1] - rows), tokens.shape[1])
    arrays["token_ids"] = tokens[0, -rows:].detach().cpu().numpy()
    return arrays, {"alpha": alpha, "component": component, "physical_module": path,
                    "reconstruction_by_condition_and_pass": quality, "output_proxies": metrics,
                    "endpoint_note": "Next-token distribution proxies; not task accuracy",
                    "interaction_definition": "both(+a) - first(+a) - second(+a) + baseline",
                    "finite_difference_note": "Finite alpha; SAE top-k support can change",
                    "precision_note": "Interventions add factorized component contributions at each selected invocation. "
                        "Shared-pass changes equal a shared matrix update in real arithmetic; "
                        "materializing and rounding an edited BF16 weight matrix can give different numerical results."}
