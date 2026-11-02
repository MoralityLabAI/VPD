from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--effects", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    components = list(iter_jsonl(args.components))
    archive = np.load(args.effects, allow_pickle=True)
    required = {"delta_h", "component_ids"}
    missing = required - set(archive.files)
    if missing:
        raise ValueError(f"VPD effects archive missing keys: {sorted(missing)}")
    delta_h = archive["delta_h"]
    ids = [str(value) for value in archive["component_ids"].tolist()]
    if delta_h.ndim != 2:
        raise ValueError(f"delta_h must be rank 2, got {delta_h.shape}")
    if delta_h.shape != (len(ids), args.hidden_size):
        raise ValueError(
            f"delta_h shape {delta_h.shape} does not match ({len(ids)}, {args.hidden_size})"
        )
    component_ids = [str(row["component_id"]) for row in components]
    if component_ids != ids:
        raise ValueError("components.jsonl order must match component_ids in effects archive")
    finite = np.isfinite(delta_h).all(axis=1)
    norms = np.linalg.norm(delta_h.astype(np.float64), axis=1)
    report = {
        "components": len(ids),
        "hidden_size": args.hidden_size,
        "all_finite": bool(finite.all()),
        "zero_norm_components": [ids[i] for i, value in enumerate(norms) if value <= 1e-12],
        "mean_effect_norm": float(norms.mean()) if len(norms) else 0.0,
        "max_effect_norm": float(norms.max()) if len(norms) else 0.0,
        "valid": bool(finite.all() and np.all(norms > 1e-12)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
