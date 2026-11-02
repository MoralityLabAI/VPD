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


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae-decoder", type=Path, required=True)
    parser.add_argument("--sae-scores", type=Path, required=True)
    parser.add_argument("--vpd-components", type=Path, required=True)
    parser.add_argument("--vpd-effects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-sae-significance", type=float, default=0.05)
    args = parser.parse_args()

    decoder = np.load(args.sae_decoder).astype(np.float32)
    vpd_archive = np.load(args.vpd_effects, allow_pickle=True)
    delta_h = vpd_archive["delta_h"].astype(np.float32)
    component_ids = [str(x) for x in vpd_archive["component_ids"].tolist()]
    if decoder.ndim != 2 or delta_h.ndim != 2:
        raise ValueError("decoder and delta_h must both be rank-2")
    if decoder.shape[1] != delta_h.shape[1]:
        raise ValueError(
            f"hidden dimension mismatch: SAE {decoder.shape[1]}, VPD {delta_h.shape[1]}"
        )

    component_meta = {row["component_id"]: row for row in iter_jsonl(args.vpd_components)}
    sae_scores = [
        row
        for row in iter_jsonl(args.sae_scores)
        if float(row.get("shuffle_p", 1.0)) <= args.min_sae_significance
    ]
    candidate_features = sorted({int(row["feature_index"]) for row in sae_scores})
    if not candidate_features:
        raise ValueError("no significant SAE feature candidates")

    decoder_norm = normalize_rows(decoder[candidate_features])
    delta_norm = normalize_rows(delta_h)
    cosine = delta_norm @ decoder_norm.T
    pairs: list[dict[str, Any]] = []
    score_by_feature: dict[int, list[dict[str, Any]]] = {}
    for row in sae_scores:
        score_by_feature.setdefault(int(row["feature_index"]), []).append(row)

    for component_row, component_id in enumerate(component_ids):
        ordering = np.argsort(np.abs(cosine[component_row]))[::-1][: args.top_k]
        for local_feature_index in ordering:
            feature_index = candidate_features[int(local_feature_index)]
            value = float(cosine[component_row, local_feature_index])
            pairs.append(
                {
                    "component_id": component_id,
                    "component": component_meta.get(component_id, {}),
                    "feature_index": feature_index,
                    "cosine": value,
                    "absolute_cosine": abs(value),
                    "sae_behavior_scores": score_by_feature.get(feature_index, []),
                    "passes_alignment_screen": abs(value) >= 0.2,
                }
            )

    pairs.sort(key=lambda row: row["absolute_cosine"], reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
