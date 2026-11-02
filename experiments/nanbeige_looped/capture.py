"""Capture each invocation of a shared module without overwriting earlier passes."""
from __future__ import annotations

from typing import Any, Callable


class RepeatedForwardCapture:
    """One inference forward per context; invocation indices are zero-based.

    This records physical module outputs, before any normalization outside that
    module. It must not span generation or gradient-checkpoint recomputation.
    """

    def __init__(self, modules: dict[str, Any], *, expected_calls: int,
                 extract: Callable[[Any], Any]):
        if expected_calls < 1 or not modules:
            raise ValueError("At least one module and one call are required")
        self.modules = modules
        self.expected_calls = expected_calls
        self.extract = extract
        self.values: dict[tuple[str, int], Any] = {}
        self.counts: dict[str, int] = {}
        self.handles: list[Any] = []

    def __enter__(self) -> "RepeatedForwardCapture":
        if self.handles:
            raise RuntimeError("Capture is already active")
        self.values.clear()
        self.counts = {name: 0 for name in self.modules}
        for name, module in self.modules.items():
            def hook(_module: Any, _inputs: Any, output: Any, *, name: str = name) -> None:
                invocation = self.counts[name]
                if invocation >= self.expected_calls:
                    raise RuntimeError(f"Unexpected repeated call for {name}; span only one forward")
                self.values[(name, invocation)] = self.extract(output)
                self.counts[name] += 1
            self.handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if exc_type is None and any(count != self.expected_calls for count in self.counts.values()):
            raise RuntimeError(f"Incomplete recurrent capture: {self.counts}")


def torch_tail_rows(output: Any, *, rows: int = 32) -> Any:
    import torch
    if rows < 1:
        raise ValueError("Capture rows must be positive")
    hidden = output[0] if isinstance(output, tuple) else output
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError("Expected one unpadded sequence: [1, tokens, hidden]")
    return hidden[0, -rows:].detach().to(device="cpu", dtype=torch.float32).numpy().copy()
