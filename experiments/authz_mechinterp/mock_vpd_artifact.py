"""Create a shape-valid VPD artifact for pipeline tests only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ids = np.asarray([f"mock-component-{i}" for i in range(args.components)])
    delta_h = rng.normal(size=(args.components, args.hidden_size)).astype(np.float32)
    np.savez_compressed(
        args.output_dir / "activation_effects.npz",
        delta_h=delta_h,
        component_ids=ids,
    )
    with (args.output_dir / "components.jsonl").open("w", encoding="utf-8") as handle:
        for index, component_id in enumerate(ids.tolist()):
            handle.write(
                json.dumps(
                    {
                        "component_id": component_id,
                        "contrast": "mock",
                        "layer": 0,
                        "rank": 1,
                        "score": float(index),
                        "mock": True,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
