"""Report finite-value coverage for a PyTorch VPD state-dict checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def group_for_key(key: str) -> str:
    if key.startswith("_components"):
        return "components"
    if key.startswith("ci_fn"):
        return "ci_fn"
    if key.startswith("target_model"):
        return "target_model"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"expected a state dict, got {type(state)!r}")

    groups: dict[str, dict[str, int]] = {}
    bad_tensors: list[dict[str, int | str]] = []
    for key, value in state.items():
        if not torch.is_tensor(value) or not (
            value.is_floating_point() or value.is_complex()
        ):
            continue
        group = group_for_key(key)
        record = groups.setdefault(
            group, {"tensor_count": 0, "element_count": 0, "nonfinite_count": 0}
        )
        nonfinite = int((~torch.isfinite(value)).sum().item())
        record["tensor_count"] += 1
        record["element_count"] += value.numel()
        record["nonfinite_count"] += nonfinite
        if nonfinite:
            bad_tensors.append(
                {"key": key, "element_count": value.numel(), "nonfinite_count": nonfinite}
            )

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "groups": groups,
        "bad_tensor_count": len(bad_tensors),
        "bad_tensors": bad_tensors,
    }
    print(json.dumps(result, indent=2))
    return 1 if bad_tensors else 0


if __name__ == "__main__":
    raise SystemExit(main())
