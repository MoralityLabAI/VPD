from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    load_jsonl,
    sha256_file,
    stable_hash,
    write_json,
)
from experiments.authz_mechinterp.activation_categories import (
    CATEGORY_PROTOCOLS,
    COMPONENT_COMPARTMENT_PROTOCOLS,
    activation_category,
    category_contract,
)
from experiments.authz_mechinterp.merge_vpd_effect_shards import write_deterministic_npz
from experiments.authz_mechinterp.activation_shards import (
    SHARDED_ACTIVATION_SCHEMA,
    ShardedActivationAccessor,
    verify_sharded_activation_artifact,
)
from experiments.authz_mechinterp.verify_activation_capture_semantics import (
    verify_prompt_window_capture_semantics,
)


class ArrayActivationAccessor:
    def __init__(self, array: np.ndarray) -> None:
        if array.ndim != 2:
            raise ValueError("SAE activation array must be two-dimensional")
        self.array = array
        self.shape = array.shape

    def finite_mask(self) -> np.ndarray:
        return np.isfinite(self.array).all(axis=1)

    def read_rows(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(self.array[indices], dtype=np.float32)

    def mean_rows(self, indices: np.ndarray, *, chunk_rows: int = 4096) -> np.ndarray:
        del chunk_rows
        return np.asarray(self.array[indices], dtype=np.float32).mean(axis=0)


def load_capacity_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != "sch.vpd-compartment-sae-capacity-plan.v1":
        raise ValueError("unsupported compartment SAE capacity plan schema")
    body = dict(plan)
    plan_id = str(body.pop("capacity_plan_id", ""))
    if content_sha256(body) != plan_id:
        raise ValueError("compartment SAE capacity plan content address mismatch")
    return plan


def validate_prompt_window_metadata(
    metadata: list[dict[str, Any]], *, rows_per_episode: int
) -> dict[str, int]:
    if rows_per_episode < 1:
        raise ValueError("prompt-window rows per episode must be positive")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in metadata:
        episode_id = str(row.get("episode_id", ""))
        if not episode_id:
            raise ValueError("prompt-window metadata lacks an episode ID")
        grouped.setdefault(episode_id, []).append(row)
    if not grouped:
        raise ValueError("prompt-window metadata is empty")
    expected_offsets = list(range(1 - rows_per_episode, 1))
    for episode_id, rows in grouped.items():
        if len(rows) != rows_per_episode:
            raise ValueError(
                f"prompt-window episode {episode_id} does not have the exact row count"
            )
        offsets = sorted(
            int(row.get("token_offset_from_proposal_start", rows_per_episode))
            for row in rows
        )
        if offsets != expected_offsets:
            raise ValueError(
                f"prompt-window episode {episode_id} has incomplete token offsets"
            )
        if any(row.get("phase") != "pre_gate" for row in rows):
            raise ValueError("prompt-window metadata contains a non-pre_gate row")
        if sum(bool(row.get("is_proposal_start")) for row in rows) != 1:
            raise ValueError(
                f"prompt-window episode {episode_id} must contain one proposal_start row"
            )
        if len({str(row.get("pair_id", "")) for row in rows}) != 1:
            raise ValueError(
                f"prompt-window episode {episode_id} crosses prompt-pair identities"
            )
        if len({activation_category(row) for row in rows}) != 1:
            raise ValueError(
                f"prompt-window episode {episode_id} crosses activation categories"
            )
    return {"episodes": len(grouped), "rows": len(metadata)}


def authorize_compartment_capacity(
    *,
    plan: dict[str, Any],
    profile_id: str,
    capture_manifest: dict[str, Any],
    activation_category_name: str,
    input_dim: int,
    expansion: int,
    top_k: int,
    selected_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless the exact compartment capture satisfies a frozen plan."""
    split_provenance = capture_manifest["source"]["activation_category_split"]
    if plan["source"]["split_id"] != split_provenance.get(
        "activation_category_split_id"
    ):
        raise ValueError("capacity plan and SAE capture use different split IDs")
    if plan["source"]["split_manifest_sha256"] != split_provenance.get(
        "activation_category_split_manifest_sha256"
    ):
        raise ValueError("capacity plan and SAE capture use different split manifests")
    if int(plan["model_space"]["hidden_size"]) != input_dim:
        raise ValueError("capacity plan hidden size differs from SAE activation width")
    profiles = {str(row["id"]): row for row in plan["profiles"]}
    if profile_id not in profiles:
        raise ValueError(f"unknown compartment SAE capacity profile: {profile_id}")
    profile = profiles[profile_id]
    if int(profile["expansion"]) != expansion or int(profile["top_k"]) != top_k:
        raise ValueError("SAE hyperparameters differ from the selected capacity profile")
    support = (
        profile["shared_compartment"]
        if activation_category_name == "shared_cross_category"
        else profile["specific_compartments"].get(activation_category_name)
    )
    if not isinstance(support, dict):
        raise ValueError("capacity plan does not cover the selected activation compartment")
    if profile.get("status") != "ready" or support.get("status") != "ready":
        raise ValueError(
            "selected capacity profile is insufficient for compartment SAE training"
        )
    sampling = capture_manifest["capture"].get("sae_training_sampling")
    if not isinstance(sampling, dict):
        raise ValueError("SAE capture lacks a capacity-bound training sampling contract")
    if sampling.get("schema") != "sch.vpd-sae-training-sampling.v1":
        raise ValueError("unsupported SAE training sampling contract")
    if sampling.get("mode") != "prompt_context_window" or int(
        sampling.get("generated_tokens", -1)
    ) != 0:
        raise ValueError("compartment SAE capture must be prompt-only context windows")
    if int(sampling.get("activation_rows_per_episode", -1)) != int(
        profile["activation_rows_per_episode"]
    ):
        raise ValueError("capture rows per episode differ from the capacity profile")
    resource = capture_manifest["source"].get("capture_resource_plan")
    if not isinstance(resource, dict):
        raise ValueError("compartment SAE capture lacks a resource plan receipt")
    if resource.get("resource_plan_id") != sampling.get("resource_plan_id"):
        raise ValueError("capture sampling and resource plan receipts differ")
    if resource.get("capacity_plan_id") != plan["capacity_plan_id"]:
        raise ValueError("capture resource and SAE capacity plans differ")
    if resource.get("split_id") != plan["source"]["split_id"]:
        raise ValueError("capture resource plan uses a different split")
    resource_plan_sha256 = str(resource.get("resource_plan_sha256", ""))
    if len(resource_plan_sha256) != 64:
        raise ValueError("capture resource plan receipt lacks a SHA-256")
    if sampling.get("activation_storage") != "npy_row_shards":
        raise ValueError("compartment SAE capture must use sharded activation storage")
    if capture_manifest["artifacts"]["activations"].get("schema") != (
        SHARDED_ACTIVATION_SCHEMA
    ):
        raise ValueError("compartment SAE activation artifact is not sharded")
    rows_per_episode = int(sampling["activation_rows_per_episode"])
    if int(sampling.get("activation_row_count", -1)) != int(
        capture_manifest["capture"].get("rows", -2)
    ):
        raise ValueError("capture sampling contract and manifest row counts differ")
    observed_window = validate_prompt_window_metadata(
        selected_metadata, rows_per_episode=rows_per_episode
    )
    requirements = profile["requirements"]
    if len(selected_metadata) < int(
        requirements["required_total_observations_before_train_validation_split"]
    ):
        raise ValueError("captured compartment has fewer observations than required")
    unique_pairs = len({str(row["pair_id"]) for row in selected_metadata})
    minimum_pairs = int(
        plan["frozen_support_policy"][
            "minimum_unique_prompt_pairs_per_compartment"
        ]
    )
    if unique_pairs < minimum_pairs:
        raise ValueError("captured compartment has fewer unique prompt pairs than required")
    return {
        "schema": "sch.vpd-compartment-sae-capacity-authorization.v1",
        "status": "pass",
        "capacity_plan_id": plan["capacity_plan_id"],
        "profile_id": profile_id,
        "activation_category": activation_category_name,
        "selected_activation_rows": len(selected_metadata),
        "selected_episodes": observed_window["episodes"],
        "selected_unique_pairs": unique_pairs,
        "requirements": dict(requirements),
        "minimum_unique_prompt_pairs": minimum_pairs,
        "sampling": dict(sampling),
        "capture_resource_plan": dict(resource),
    }


def activation_compartment_selection(
    metadata: list[dict[str, Any]], category: str
) -> tuple[np.ndarray, dict[str, Any]]:
    if category not in CATEGORY_PROTOCOLS:
        raise ValueError(f"unknown SAE activation category: {category}")
    categories = [activation_category(row) for row in metadata]
    mask = np.asarray([value == category for value in categories], dtype=bool)
    compartment = {
        "schema": "sch.vpd-sae-activation-compartment.v1",
        "activation_category": category,
        "protocol": CATEGORY_PROTOCOLS[category],
        "category_contract_sha256": content_sha256(category_contract()),
        "source_rows": len(metadata),
        "selected_rows_before_finite_filter": int(mask.sum()),
        "all_category_counts": dict(
            sorted({value: categories.count(value) for value in set(categories)}.items())
        ),
    }
    return mask, compartment


def balanced_activation_compartment_selection(
    metadata: list[dict[str, Any]], *, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    categories = [activation_category(row) for row in metadata]
    grouped: dict[str, dict[str, list[int]]] = {}
    for index, category in enumerate(categories):
        episode_id = str(metadata[index].get("episode_id", index))
        grouped.setdefault(category, {}).setdefault(episode_id, []).append(index)
    if len(grouped) < 2:
        raise ValueError("balanced cross-category SAE training requires at least two categories")
    target_episodes_per_category = min(
        len(episodes) for episodes in grouped.values()
    )
    if target_episodes_per_category <= 0:
        raise ValueError("balanced cross-category SAE training has an empty category")
    selected: set[int] = set()
    selected_rows_by_category: dict[str, int] = {}
    for category, episode_rows in sorted(grouped.items()):
        ordered = sorted(
            episode_rows,
            key=lambda episode_id: stable_hash(
                seed,
                "sae-balanced-category-episode",
                category,
                episode_id,
            ),
        )
        chosen = ordered[:target_episodes_per_category]
        category_indices = {
            index for episode_id in chosen for index in episode_rows[episode_id]
        }
        selected.update(category_indices)
        selected_rows_by_category[category] = len(category_indices)
    mask = np.asarray([index in selected for index in range(len(metadata))], dtype=bool)
    compartment = {
        "schema": "sch.vpd-sae-activation-compartment.v1",
        "activation_category": "shared_cross_category",
        "protocol": COMPONENT_COMPARTMENT_PROTOCOLS["shared_cross_category"],
        "category_contract_sha256": content_sha256(category_contract()),
        "source_rows": len(metadata),
        "selected_rows_before_finite_filter": int(mask.sum()),
        "balance_rule": "equal_whole_episodes_per_observed_activation_category",
        "episodes_per_category": target_episodes_per_category,
        "rows_per_category": (
            next(iter(set(selected_rows_by_category.values())))
            if len(set(selected_rows_by_category.values())) == 1
            else None
        ),
        "selected_rows_by_category": selected_rows_by_category,
        "all_category_counts": {
            category: sum(len(indices) for indices in episodes.values())
            for category, episodes in sorted(grouped.items())
        },
        "all_category_episode_counts": {
            category: len(episodes)
            for category, episodes in sorted(grouped.items())
        },
    }
    return mask, compartment


def pair_atomic_train_validation_split(
    metadata: list[dict[str, Any]],
    selected_rows: np.ndarray,
    *,
    seed: int,
    training_fraction: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not 0.0 < training_fraction < 1.0:
        raise ValueError("training fraction must be in (0, 1)")
    grouped: dict[str, list[int]] = {}
    for row_index in np.asarray(selected_rows, dtype=np.int64):
        pair_id = str(metadata[int(row_index)].get("pair_id", ""))
        if not pair_id:
            raise ValueError("SAE metadata lacks a prompt-pair identity")
        grouped.setdefault(pair_id, []).append(int(row_index))
    if len(grouped) < 2:
        raise ValueError("pair-atomic SAE validation requires at least two pairs")
    ordered_pairs = sorted(
        grouped,
        key=lambda pair_id: stable_hash(seed, "sae-pair-atomic-split", pair_id),
    )
    split = max(1, min(len(ordered_pairs) - 1, int(training_fraction * len(grouped))))
    training_pairs = set(ordered_pairs[:split])
    validation_pairs = set(ordered_pairs[split:])
    train_rows = np.asarray(
        sorted(index for pair in training_pairs for index in grouped[pair]),
        dtype=np.int64,
    )
    validation_rows = np.asarray(
        sorted(index for pair in validation_pairs for index in grouped[pair]),
        dtype=np.int64,
    )
    if set(train_rows.tolist()) & set(validation_rows.tolist()):
        raise ValueError("pair-atomic SAE row split overlaps")
    receipt = {
        "schema": "sch.vpd-sae-pair-atomic-train-validation-split.v1",
        "seed": seed,
        "requested_training_fraction": training_fraction,
        "training_pairs": len(training_pairs),
        "validation_pairs": len(validation_pairs),
        "training_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "training_pair_id_set_sha256": content_sha256(sorted(training_pairs)),
        "validation_pair_id_set_sha256": content_sha256(sorted(validation_pairs)),
        "pair_sets_disjoint": True,
        "row_sets_disjoint": True,
        "exact_selected_row_partition": (
            len(train_rows) + len(validation_rows) == len(selected_rows)
        ),
    }
    if not receipt["exact_selected_row_partition"]:
        raise ValueError("pair-atomic SAE split is not an exact row partition")
    return train_rows, validation_rows, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path)
    parser.add_argument("--capacity-plan", type=Path)
    parser.add_argument("--capacity-profile")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--source-dataset", type=Path)
    parser.add_argument("--activation-category-split-manifest", type=Path)
    compartment_group = parser.add_mutually_exclusive_group()
    compartment_group.add_argument(
        "--activation-category", choices=sorted(CATEGORY_PROTOCOLS)
    )
    compartment_group.add_argument(
        "--balanced-across-activation-categories", action="store_true"
    )
    parser.add_argument("--key", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expansion", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=17,
        help="fixed data-selection and pair-atomic train/validation seed",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    capture_manifest: dict[str, Any] | None = None
    sharded_capture_artifact: dict[str, Any] | None = None
    capture_semantic_verification: dict[str, Any] | None = None
    if args.capture_manifest is not None:
        capture_manifest = json.loads(args.capture_manifest.read_text(encoding="utf-8"))
        if capture_manifest.get("schema") != "sch.vpd-activation-capture-manifest.v1":
            raise ValueError("unsupported activation capture manifest schema")
        capture_body = dict(capture_manifest)
        capture_id = str(capture_body.pop("capture_id"))
        if content_sha256(capture_body) != capture_id:
            raise ValueError("activation capture manifest content address mismatch")
        activation_artifact = capture_manifest["artifacts"]["activations"]
        if activation_artifact.get("schema") == SHARDED_ACTIVATION_SCHEMA:
            if not args.activations.is_dir():
                raise ValueError("sharded SAE activations require their capture directory")
            verify_sharded_activation_artifact(args.activations, activation_artifact)
            sharded_capture_artifact = activation_artifact
        elif sha256_file(args.activations) != str(activation_artifact["sha256"]):
            raise ValueError("SAE training activations differ from their manifest")
        if args.key not in set(capture_manifest["capture"]["activation_keys"]):
            raise ValueError("SAE activation key is absent from the capture manifest")
        if capture_manifest["model"].get("adapter_sha256") is not None:
            raise ValueError("fixed SAE dictionary must be trained on base-model activations")
        if args.metadata is not None and sha256_file(args.metadata) != str(
            capture_manifest["artifacts"]["metadata"]["sha256"]
        ):
            raise ValueError("SAE training metadata differs from its capture manifest")
    if (
        args.activation_category is not None
        or args.balanced_across_activation_categories
    ) and args.metadata is None:
        raise ValueError("activation-compartment SAE training requires --metadata")
    if args.activation_category is not None or args.balanced_across_activation_categories:
        if capture_manifest is None:
            raise ValueError(
                "activation-compartment SAE training requires a capture manifest"
            )
        split_provenance = capture_manifest["source"].get(
            "activation_category_split"
        )
        if not isinstance(split_provenance, dict) or split_provenance.get(
            "activation_category_split_verification_status"
        ) != "pass":
            raise ValueError("SAE training capture lacks verified split provenance")
        if split_provenance.get("activation_category_split_role") != "sae_training":
            raise ValueError("SAE dictionary must use the sae_training split role")
        if args.capacity_plan is None or args.capacity_profile is None:
            raise ValueError(
                "activation-compartment SAE training requires a capacity plan and profile"
            )
        if args.source_dataset is None or args.activation_category_split_manifest is None:
            raise ValueError(
                "activation-compartment SAE training requires source dataset and split manifest"
            )
        assert args.capture_manifest is not None
        capture_semantic_verification = verify_prompt_window_capture_semantics(
            capture_dir=args.activations,
            capture_manifest_path=args.capture_manifest,
            dataset_path=args.source_dataset,
            split_manifest_path=args.activation_category_split_manifest,
            expected_role="sae_training",
        )

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("SAE training requires torch") from exc

    if args.seed < 0 or args.split_seed < 0:
        raise ValueError("SAE initialization and split seeds must be nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if sharded_capture_artifact is None:
        archive = np.load(args.activations)
        if args.key not in archive:
            raise KeyError(f"activation key not found: {args.key}")
        accessor: ArrayActivationAccessor | ShardedActivationAccessor = (
            ArrayActivationAccessor(archive[args.key])
        )
    else:
        accessor = ShardedActivationAccessor(
            args.activations,
            sharded_capture_artifact,
            args.key,
            verify_artifact=False,
        )
    valid_mask = accessor.finite_mask()
    compartment: dict[str, Any] | None = None
    selected_metadata: list[dict[str, Any]] | None = None
    metadata: list[dict[str, Any]] | None = None
    if args.activation_category is not None or args.balanced_across_activation_categories:
        assert args.metadata is not None
        metadata = load_jsonl(args.metadata)
        if len(metadata) != accessor.shape[0]:
            raise ValueError("SAE activation and metadata rows do not align")
        if args.balanced_across_activation_categories:
            category_mask, compartment = balanced_activation_compartment_selection(
                metadata, seed=args.split_seed
            )
        else:
            assert args.activation_category is not None
            category_mask, compartment = activation_compartment_selection(
                metadata, args.activation_category
            )
        valid_mask &= category_mask
        selected_metadata = [
            row for row, selected in zip(metadata, valid_mask, strict=True) if selected
        ]
        compartment.update(
            {
                "metadata_sha256": sha256_file(args.metadata),
                "selected_rows": int(valid_mask.sum()),
            }
        )
    source_rows = np.flatnonzero(valid_mask)
    if len(source_rows) < 16:
        raise ValueError(
            f"need at least 16 valid activation rows, found {len(source_rows)}"
        )

    input_dim = int(accessor.shape[1])
    dict_dim = int(input_dim * args.expansion)
    top_k = min(args.top_k, dict_dim)
    capacity_authorization: dict[str, Any] | None = None
    if compartment is not None:
        assert args.capacity_plan is not None
        assert args.capacity_profile is not None
        assert capture_manifest is not None
        assert selected_metadata is not None
        capacity_authorization = authorize_compartment_capacity(
            plan=load_capacity_plan(args.capacity_plan),
            profile_id=args.capacity_profile,
            capture_manifest=capture_manifest,
            activation_category_name=str(compartment["activation_category"]),
            input_dim=input_dim,
            expansion=args.expansion,
            top_k=top_k,
            selected_metadata=selected_metadata,
        )
        compartment["capacity_authorization"] = capacity_authorization

    split_receipt: dict[str, Any] | None = None
    if metadata is not None:
        train_rows, val_rows, split_receipt = pair_atomic_train_validation_split(
            metadata, source_rows, seed=args.split_seed
        )
        if capacity_authorization is not None:
            if len(train_rows) < int(
                capacity_authorization["requirements"][
                    "required_training_observations"
                ]
            ):
                raise ValueError(
                    "pair-atomic training split falls below the capacity observation floor"
                )
            if int(split_receipt["training_pairs"]) < int(
                capacity_authorization["minimum_unique_prompt_pairs"]
            ):
                raise ValueError(
                    "pair-atomic training split falls below the capacity pair floor"
                )
            capacity_authorization["train_validation_split"] = split_receipt
    else:
        permutation = np.random.permutation(len(source_rows))
        split = max(1, int(0.9 * len(source_rows)))
        train_rows = source_rows[permutation[:split]]
        val_rows = source_rows[permutation[split:]]
        if len(val_rows) == 0:
            val_rows = train_rows[:1]

    class TopKSAE(nn.Module):
        def __init__(self, initial_pre_bias: np.ndarray) -> None:
            super().__init__()
            self.pre_bias = nn.Parameter(torch.from_numpy(initial_pre_bias))
            self.encoder = nn.Linear(input_dim, dict_dim)
            self.decoder = nn.Linear(dict_dim, input_dim, bias=False)
            nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
            with torch.no_grad():
                self.decoder.weight.copy_(self.encoder.weight.T)
                self.normalize_decoder()

        @torch.no_grad()
        def normalize_decoder(self) -> None:
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
            self.decoder.weight.div_(norms)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            pre = self.encoder(x - self.pre_bias)
            positive = torch.relu(pre)
            values, indices = torch.topk(positive, k=top_k, dim=-1)
            sparse = torch.zeros_like(positive)
            sparse.scatter_(-1, indices, values)
            return sparse

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            features = self.encode(x)
            reconstruction = self.decoder(features) + self.pre_bias
            return reconstruction, features

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = TopKSAE(accessor.mean_rows(train_rows)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, Any]] = []

    def evaluate_validation() -> tuple[float, float, float]:
        squared_error = 0.0
        feature_sum = np.zeros(input_dim, dtype=np.float64)
        feature_square_sum = np.zeros(input_dim, dtype=np.float64)
        active_features = 0.0
        evaluation_batch_rows = max(1, min(1024, args.batch_size * 4))
        with torch.inference_mode():
            for start in range(0, len(val_rows), evaluation_batch_rows):
                batch_np = accessor.read_rows(
                    val_rows[start : start + evaluation_batch_rows]
                )
                batch = torch.from_numpy(batch_np).to(device)
                reconstruction, features = model(batch)
                squared_error += float(
                    torch.sum((reconstruction - batch) ** 2).item()
                )
                active_features += float((features > 0).float().sum().item())
                feature_sum += batch_np.sum(axis=0, dtype=np.float64)
                feature_square_sum += np.square(
                    batch_np, dtype=np.float64
                ).sum(axis=0, dtype=np.float64)
        denominator = float(
            np.sum(
                feature_square_sum
                - np.square(feature_sum) / max(1, len(val_rows))
            )
        )
        val_mse = squared_error / (len(val_rows) * input_dim)
        explained = 1.0 - squared_error / max(denominator, 1e-8)
        mean_l0 = active_features / len(val_rows)
        return val_mse, explained, mean_l0

    for step in range(1, args.steps + 1):
        batch_positions = np.random.randint(0, len(train_rows), size=args.batch_size)
        batch = torch.from_numpy(
            accessor.read_rows(train_rows[batch_positions])
        ).to(device)
        reconstruction, features = model(batch)
        mse = torch.mean((reconstruction - batch) ** 2)
        loss = mse
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.normalize_decoder()

        if step == 1 or step % max(1, args.steps // 20) == 0 or step == args.steps:
            val_mse, explained, l0 = evaluate_validation()
            history.append(
                {
                    "step": step,
                    "train_mse": float(mse.item()),
                    "val_mse": float(val_mse),
                    "explained_variance": float(explained),
                    "mean_l0": float(l0),
                }
            )
            print(json.dumps(history[-1], sort_keys=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "dict_dim": dict_dim,
        "top_k": top_k,
        "activation_key": args.key,
        "source_rows": source_rows,
        "config": vars(args),
        "history": history,
    }
    torch.save(checkpoint, args.output_dir / "sae_checkpoint.pt")
    np.save(
        args.output_dir / "sae_decoder.npy",
        model.decoder.weight.detach().cpu().numpy().T.astype(np.float32),
    )
    np.save(
        args.output_dir / "sae_encoder.npy",
        model.encoder.weight.detach().cpu().numpy().astype(np.float32),
    )
    np.save(
        args.output_dir / "sae_pre_bias.npy",
        model.pre_bias.detach().cpu().numpy().astype(np.float32),
    )
    dictionary_path = args.output_dir / "sae_dictionary.npz"
    write_deterministic_npz(
        dictionary_path,
        {
            "activation_key": np.asarray(args.key),
            "decoder": model.decoder.weight.detach().cpu().numpy().T.astype(np.float32),
            "encoder_weight": model.encoder.weight.detach().cpu().numpy().astype(np.float32),
            "encoder_bias": model.encoder.bias.detach().cpu().numpy().astype(np.float32),
            "pre_bias": model.pre_bias.detach().cpu().numpy().astype(np.float32),
            "top_k": np.asarray(top_k, dtype=np.int64),
        },
    )
    activation_capture_digest = (
        content_sha256(sharded_capture_artifact)
        if sharded_capture_artifact is not None
        else sha256_file(args.activations)
    )
    dictionary_manifest_body = {
        "schema": "sch.vpd-fixed-sae-dictionary-manifest.v1",
        "source": {
            "activation_capture_sha256": activation_capture_digest,
            "capture_manifest_sha256": (
                sha256_file(args.capture_manifest) if args.capture_manifest else None
            ),
            "capture_id": capture_manifest.get("capture_id") if capture_manifest else None,
            "capture_semantic_verification": capture_semantic_verification,
            "runner_sha256": sha256_file(Path(__file__)),
            "activation_category_split": (
                capture_manifest["source"].get("activation_category_split")
                if capture_manifest
                else None
            ),
            "capacity_plan_sha256": (
                sha256_file(args.capacity_plan) if args.capacity_plan else None
            ),
            "capacity_authorization": capacity_authorization,
        },
        "model": dict(capture_manifest["model"]) if capture_manifest else None,
        "dictionary": {
            "activation_key": args.key,
            "input_dim": input_dim,
            "dict_dim": dict_dim,
            "top_k": top_k,
            "expansion": args.expansion,
            "steps": args.steps,
            "seed": args.seed,
            "data_selection_seed": args.split_seed,
            "activation_compartment": compartment,
            "train_validation_split": split_receipt,
        },
        "artifacts": {
            "dictionary": {
                "path": dictionary_path.name,
                "sha256": sha256_file(dictionary_path),
            }
        },
    }
    write_json(
        args.output_dir / "sae_dictionary_manifest.json",
        {
            **dictionary_manifest_body,
            "dictionary_id": content_sha256(dictionary_manifest_body),
        },
    )
    (args.output_dir / "sae_metrics.json").write_text(
        json.dumps(
            {
                "activation_key": args.key,
                "input_dim": input_dim,
                "dict_dim": dict_dim,
                "top_k": top_k,
                "valid_rows": int(len(source_rows)),
                "activation_compartment": compartment,
                "train_validation_split": split_receipt,
                "dictionary_path": dictionary_path.name,
                "dictionary_sha256": sha256_file(dictionary_path),
                "history": history,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if isinstance(accessor, ShardedActivationAccessor):
        accessor.close()


if __name__ == "__main__":
    main()
