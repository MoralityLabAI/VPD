"""Pass-specific component interventions compatible with autoregressive KV caches.

Hooks reset invocation counts at every model call, including every decode step.
No shared weight is edited. Single-pass conditions deliberately break sharing.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch


class PassComponentIntervention:
    def __init__(self, model, path, v, u, *, component, passes=2):
        module = model.get_submodule(path)
        if not isinstance(module, torch.nn.Linear):
            raise ValueError('Interventions require an original linear module')
        if (v.ndim != 2 or u.ndim != 2 or v.shape[1] != u.shape[0]
                or (u.shape[1], v.shape[0]) != tuple(module.weight.shape)
                or type(component) is not int or not 0 <= component < v.shape[1]
                or type(passes) is not int or passes < 1
                or not torch.isfinite(v).all() or not torch.isfinite(u).all()):
            raise ValueError('Invalid factor dimensions, component or pass count')
        self.model, self.module, self.path = model, module, path
        self.v = v[:, component:component + 1].detach().to(module.weight.device)
        self.u = u[component:component + 1].detach().to(module.weight.device)
        self.passes = passes
        self.active = False
        self.calls = self.model_calls = 0

    @contextmanager
    def apply(self, alpha, selected):
        selected = frozenset(selected)
        if (not math.isfinite(alpha) or not selected
                or any(type(n) is not int or not 0 <= n < self.passes for n in selected)):
            raise ValueError('Invalid intervention scope or coefficient')
        if self.active:
            raise RuntimeError('Nested intervention contexts are unsupported')
        handles = []
        self.active = True
        self.model_calls = 0

        def begin(module, args):
            self.calls = 0

        def change(module, args, output):
            invocation = self.calls
            self.calls += 1
            if invocation >= self.passes:
                raise RuntimeError('Unexpected repeated invocation')
            if invocation in selected and alpha != 0:
                x = args[0]
                # Matches RepeatedLinear's factorized finite-coefficient update.
                output = output + alpha * ((x @ self.v) @ self.u).to(output.dtype)
            return output

        def finish(module, args, output):
            if self.calls != self.passes:
                raise RuntimeError(f'Incomplete recurrent forward: {self.calls}/{self.passes}')
            self.model_calls += 1

        try:
            handles.append(self.model.register_forward_pre_hook(begin))
            handles.append(self.module.register_forward_hook(change))
            handles.append(self.model.register_forward_hook(finish))
            yield self
        finally:
            for handle in reversed(handles):
                handle.remove()
            self.active = False
