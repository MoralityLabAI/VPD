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


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average
        start = end
    return ranks


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(scores)
    rank_sum = ranks[labels].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def rank_columns(values: np.ndarray) -> np.ndarray:
    ranks = np.empty(values.shape, dtype=np.float64)
    for feature_index in range(values.shape[1]):
        ranks[:, feature_index] = average_ranks(values[:, feature_index])
    return ranks


def aucs_from_ranks(labels: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return np.full(ranks.shape[1], np.nan, dtype=np.float64)
    rank_sums = ranks[labels].sum(axis=0)
    return (rank_sums - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fit_residualizer(y: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(y)), covariates])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return beta


def apply_residualizer(
    y: np.ndarray, covariates: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    design = np.column_stack([np.ones(len(y)), covariates])
    return y - design @ beta


def read_label(row: dict[str, Any], label: str) -> int:
    if label in row.get("factors", {}):
        return int(bool(row["factors"][label]))
    if label in row.get("labels", {}):
        return int(bool(row["labels"][label]))
    if label == "gate_allowed":
        return int(row.get("observed_gate_decision") == "allow")
    if label == "actual_execution":
        return int(bool(row.get("actual_execution")))
    raise KeyError(f"label not found: {label}")


def encode_topk(
    x: np.ndarray,
    *,
    pre_bias: np.ndarray,
    encoder_weight: np.ndarray,
    encoder_bias: np.ndarray,
    top_k: int,
) -> np.ndarray:
    dense = np.maximum(0.0, (x - pre_bias) @ encoder_weight.T + encoder_bias)
    if top_k >= dense.shape[1]:
        return dense
    indices = np.argpartition(dense, -top_k, axis=1)[:, -top_k:]
    sparse = np.zeros_like(dense)
    rows = np.arange(len(dense))[:, None]
    sparse[rows, indices] = dense[rows, indices]
    return sparse


def permute_within_blocks(
    labels: np.ndarray, blocks: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    shuffled = labels.copy()
    for block in np.unique(blocks):
        mask = blocks == block
        shuffled[mask] = rng.permutation(shuffled[mask])
    return shuffled


def cross_validated_auc(
    labels: np.ndarray,
    feature: np.ndarray,
    covariates: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[str, float], float, float]:
    fold_aucs: dict[str, float] = {}
    for group in sorted(set(groups.tolist())):
        test = groups == group
        train = ~test
        if train.sum() < 4 or test.sum() < 2:
            continue
        beta = fit_residualizer(feature[train], covariates[train])
        train_scores = apply_residualizer(feature[train], covariates[train], beta)
        test_scores = apply_residualizer(feature[test], covariates[test], beta)
        train_auc = roc_auc(labels[train], train_scores)
        test_auc = roc_auc(labels[test], test_scores)
        if not np.isfinite(train_auc) or not np.isfinite(test_auc):
            continue
        oriented = test_auc if train_auc >= 0.5 else 1.0 - test_auc
        fold_aucs[str(group)] = float(oriented)
    values = np.asarray(list(fold_aucs.values()), dtype=np.float64)
    if len(values) == 0:
        return fold_aucs, float("nan"), float("nan")
    return fold_aucs, float(values.mean()), float(values.min())


def add_bh_q_values(rows: list[dict[str, Any]]) -> None:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row["label"]), []).append(row)
    for label_rows in by_label.values():
        ordered = sorted(label_rows, key=lambda row: float(row["shuffle_p"]))
        m = len(ordered)
        running = 1.0
        for reverse_rank, row in enumerate(reversed(ordered), start=1):
            rank = m - reverse_rank + 1
            q = min(running, float(row["shuffle_p"]) * m / rank)
            row["shuffle_q_bh"] = float(q)
            running = q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--sae-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--labels",
        default=(
            "semantic_authorization,host_grant_valid,scope_match,"
            "trusted_visible_provenance,pre_gate_proposal_ready,"
            "normative_commit_ready,gate_allowed"
        ),
    )
    parser.add_argument("--phase", choices=["pre_gate", "post_gate"])
    parser.add_argument("--subset-conditions", default="")
    parser.add_argument("--shuffle-permutations", type=int, default=500)
    parser.add_argument("--top-features", type=int, default=64)
    parser.add_argument(
        "--permutation-block",
        choices=["pair_id", "template_group", "none"],
        default="pair_id",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("feature scoring requires torch to read the checkpoint") from exc

    checkpoint = torch.load(args.sae_checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    key = str(checkpoint["activation_key"])
    source_rows = np.asarray(checkpoint["source_rows"], dtype=np.int64)
    archive = np.load(args.activations)
    raw = archive[key].astype(np.float32)
    x = raw[source_rows]
    metadata_all = list(iter_jsonl(args.metadata))
    metadata = [metadata_all[int(index)] for index in source_rows]

    subset_conditions = {
        value.strip() for value in args.subset_conditions.split(",") if value.strip()
    }
    keep = np.asarray(
        [
            (args.phase is None or row.get("phase") == args.phase)
            and (not subset_conditions or row.get("condition") in subset_conditions)
            for row in metadata
        ],
        dtype=bool,
    )
    x = x[keep]
    metadata = [row for row, retain in zip(metadata, keep.tolist()) if retain]
    if len(metadata) < 8:
        raise ValueError(f"analysis subset has only {len(metadata)} rows")

    features = encode_topk(
        x,
        pre_bias=state["pre_bias"].numpy(),
        encoder_weight=state["encoder.weight"].numpy(),
        encoder_bias=state["encoder.bias"].numpy(),
        top_k=int(checkpoint["top_k"]),
    )
    anchor_name = key.split("__", 1)[1] if "__" in key else ""
    prompt_tokens = np.asarray([row["prompt_tokens"] for row in metadata], dtype=np.float64)
    anchor_positions = np.asarray(
        [
            row.get("positions", {}).get(anchor_name)
            if row.get("positions", {}).get(anchor_name) is not None
            else row["prompt_tokens"]
            for row in metadata
        ],
        dtype=np.float64,
    )
    covariates = np.column_stack([prompt_tokens, anchor_positions])
    groups = np.asarray([row["template_group"] for row in metadata])
    if args.permutation_block == "none":
        blocks = np.asarray(["all"] * len(metadata))
    else:
        blocks = np.asarray([row[args.permutation_block] for row in metadata])
    rng = np.random.default_rng(args.seed)
    label_names = [x.strip() for x in args.labels.split(",") if x.strip()]
    results: list[dict[str, Any]] = []

    beta_all = fit_residualizer(features, covariates)
    residual_features = apply_residualizer(features, covariates, beta_all)
    residual_ranks = rank_columns(residual_features)

    for label_name in label_names:
        labels = np.asarray([read_label(row, label_name) for row in metadata], dtype=np.int8)
        if labels.min() == labels.max():
            continue
        raw_aucs = np.asarray(
            [roc_auc(labels, features[:, feature]) for feature in range(features.shape[1])]
        )
        residual_aucs = np.asarray(
            [
                roc_auc(labels, residual_features[:, feature])
                for feature in range(features.shape[1])
            ]
        )
        strength = np.abs(residual_aucs - 0.5)
        candidate_indices = np.argsort(np.nan_to_num(strength, nan=-1.0))[::-1][
            : args.top_features
        ]

        null_by_feature: dict[int, list[float]] = {
            int(index): [] for index in candidate_indices
        }
        null_max_strengths: list[float] = []
        for _ in range(args.shuffle_permutations):
            shuffled = permute_within_blocks(labels, blocks, rng)
            null_aucs = aucs_from_ranks(shuffled, residual_ranks)
            permutation_strengths = np.abs(null_aucs - 0.5)
            for feature_index in candidate_indices:
                value = float(permutation_strengths[feature_index])
                null_by_feature[int(feature_index)].append(value)
            null_max_strengths.append(float(np.nanmax(permutation_strengths)))

        for feature_index in candidate_indices:
            observed = float(residual_aucs[feature_index])
            if not np.isfinite(observed):
                continue
            observed_strength = abs(observed - 0.5)
            null_strengths = null_by_feature[int(feature_index)]
            p_value = (1 + sum(x >= observed_strength for x in null_strengths)) / (
                1 + len(null_strengths)
            )
            max_p = (1 + sum(x >= observed_strength for x in null_max_strengths)) / (
                1 + len(null_max_strengths)
            )
            fold_aucs, cv_mean, cv_min = cross_validated_auc(
                labels,
                features[:, feature_index],
                covariates,
                groups,
            )
            active = features[:, feature_index] > 0
            results.append(
                {
                    "activation_key": key,
                    "phase_filter": args.phase,
                    "condition_filter": sorted(subset_conditions),
                    "label": label_name,
                    "feature_index": int(feature_index),
                    "raw_auc": float(raw_aucs[feature_index]),
                    "residualized_auc": observed,
                    "direction": "positive" if observed >= 0.5 else "negative",
                    "shuffle_p": float(p_value),
                    "shuffle_max_stat_p": float(max_p),
                    "cv_held_out_template_auc": fold_aucs,
                    "cv_mean_oriented_auc": cv_mean,
                    "cv_min_oriented_auc": cv_min,
                    "activation_prevalence": float(active.mean()),
                    "mean_positive": float(features[labels == 1, feature_index].mean()),
                    "mean_negative": float(features[labels == 0, feature_index].mean()),
                    "n_rows": int(len(metadata)),
                }
            )

    add_bh_q_values(results)
    results.sort(
        key=lambda row: (
            row["label"],
            row["shuffle_max_stat_p"],
            row["shuffle_p"],
            -abs(row["residualized_auc"] - 0.5),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
