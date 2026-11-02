from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.authz_mechinterp.effect_uncertainty import (
    CONDITION_ROBUSTNESS_ARRAYS,
    TEMPLATE_ROBUSTNESS_ARRAYS,
    UNCERTAINTY_ARRAYS,
    condition_robustness_metrics,
    leave_one_block_out_means,
    leave_one_block_out_strengths,
    pair_block_direction_metrics,
    template_robustness_metrics,
    validate_condition_robustness_arrays,
    validate_template_robustness_arrays,
    validate_uncertainty_arrays,
)

from experiments.authz_mechinterp.activation_categories import (
    COMPONENT_COMPARTMENT_PROTOCOLS,
    category_contract,
)


MODULE_RE = re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)\.(?P<family>.+)$")
DEFAULT_ANCHORS = ("auth_end", "proposal_start")
DEFAULT_MINIMUM_DOMINANT_CATEGORY_SHARE = 0.35
DEFAULT_MINIMUM_CATEGORY_DOMINANCE_MARGIN = 0.10
DEFAULT_MAXIMUM_CATEGORY_NORMALIZED_ENTROPY = 0.80
DEFAULT_MINIMUM_CATEGORY_REPLICATION_RATE = 0.90
DEFAULT_MINIMUM_STABLE_SPECIFIC_COMPONENTS = 32
DEFAULT_MINIMUM_CATEGORY_CALIBRATION_EPISODES = 8
DEFAULT_MINIMUM_CATEGORY_DIRECTION_COSINE = 0.50
DEFAULT_MINIMUM_CATEGORY_STRENGTH_RATIO = 0.50
DEFAULT_MINIMUM_CATEGORY_JACKKNIFE_STABILITY = 0.875
DEFAULT_MINIMUM_CATEGORY_CONDITION_COVERAGE = 0.75
DEFAULT_MINIMUM_CONDITION_DIRECTION_COSINE = 0.50
DEFAULT_MINIMUM_CONDITION_RELATIVE_NORM = 0.25
DEFAULT_MINIMUM_CATEGORY_TEMPLATE_COVERAGE = 0.75
DEFAULT_MINIMUM_TEMPLATE_DIRECTION_COSINE = 0.50
DEFAULT_MINIMUM_TEMPLATE_RELATIVE_NORM = 0.25
DEFAULT_MINIMUM_CATEGORY_ROBUSTNESS_SKETCH_DIMENSION = 256
DEFAULT_MINIMUM_CATEGORY_PAIR_BLOCK_DIRECTION_COVERAGE = 0.75
DEFAULT_MINIMUM_PAIR_BLOCK_DIRECTION_COSINE = 0.50
DEFAULT_MINIMUM_PAIR_BLOCK_RELATIVE_NORM = 0.25
ACTIVATION_CATEGORY_DIRECTION_WEIGHTING = "sqrt_category_strength_share"
DEFAULT_MINIMUM_CATEGORY_CLUSTER_PROJECTION_STABILITY = 0.90
DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE = 0.15
DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE_P10 = 0.05
DEFAULT_MINIMUM_CATEGORY_CLUSTER_BOUNDARY_GAP_RATIO = 1.50
DEFAULT_MINIMUM_CATEGORY_CLUSTER_SIZE = 4
DEFAULT_MAXIMUM_CLUSTERS_PER_CATEGORY = 4
DEFAULT_CATEGORY_CLUSTER_PROJECTION_REPLICATES = 8
DEFAULT_CATEGORY_CALIBRATION_GRID = {
    "minimum_dominant_share": [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
    "minimum_dominance_margin": [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60],
    "maximum_normalized_entropy": [0.40, 0.50, 0.60, 0.70, 0.80],
}


def json_normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_normalized(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        json_normalized(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_effect_merge_split_provenance(
    effects_path: Path, merge_manifest_path: Path, *, expected_role: str
) -> dict[str, Any]:
    manifest = json.loads(merge_manifest_path.read_text(encoding="utf-8"))
    body = dict(manifest)
    merge_id = str(body.pop("merge_id", ""))
    if content_sha256(body) != merge_id:
        raise ValueError("effect merge manifest content address mismatch")
    if manifest.get("schema") != "sch.vpd-effect-shard-merge.v1":
        raise ValueError("unexpected effect merge manifest schema")
    if manifest.get("status") != "complete":
        raise ValueError("effect merge manifest is not complete")
    artifact = manifest["artifacts"]["activation_effects"]
    expected_effects_path = (merge_manifest_path.parent / artifact["path"]).resolve()
    if effects_path.resolve() != expected_effects_path:
        raise ValueError("effect archive does not match its merge manifest")
    if sha256_file(effects_path) != artifact["sha256"]:
        raise ValueError("effect archive hash differs from merge manifest")
    identity = manifest.get("identity", {})
    required = {
        "activation_category_split_id",
        "activation_category_split_manifest_sha256",
        "activation_category_split_role",
        "activation_category_split_episode_id_set_sha256",
        "activation_category_split_pair_id_set_sha256",
        "activation_category_split_verification_status",
    }
    if not required.issubset(identity):
        raise ValueError("effect merge omits activation-category split provenance")
    if str(identity["activation_category_split_role"]) != expected_role:
        raise ValueError(f"effect merge does not use the {expected_role} split role")
    if str(identity["activation_category_split_verification_status"]) != "pass":
        raise ValueError("effect merge split provenance was not verified")
    digest_re = re.compile(r"^[0-9a-f]{64}$")
    for key in required - {
        "activation_category_split_role",
        "activation_category_split_verification_status",
    }:
        if digest_re.fullmatch(str(identity[key])) is None:
            raise ValueError(f"effect merge split digest is invalid: {key}")
    split_schema = str(
        identity.get(
            "activation_category_split_schema",
            "sch.vpd-activation-category-split-manifest.v1",
        )
    )
    semantic_status = str(
        identity.get(
            "activation_category_split_semantic_disjointness",
            "not_provided_v1",
        )
    )
    semantic: dict[str, Any] = {
        "activation_category_split_schema": split_schema,
        "activation_category_split_semantic_disjointness": semantic_status,
    }
    if split_schema in {
        "sch.vpd-activation-category-split-manifest.v2",
        "sch.vpd-activation-category-split-manifest.v3",
    }:
        template_hash = str(
            identity.get("activation_category_split_template_group_set_sha256", "")
        )
        if semantic_status != "pass" or digest_re.fullmatch(template_hash) is None:
            raise ValueError("effect merge semantic split provenance is invalid")
        semantic["activation_category_split_template_group_set_sha256"] = (
            template_hash
        )
    elif split_schema != "sch.vpd-activation-category-split-manifest.v1":
        raise ValueError("effect merge uses an unsupported split schema")
    return {
        "effect_merge_id": merge_id,
        "effect_merge_manifest_sha256": sha256_file(merge_manifest_path),
        **{key: identity[key] for key in sorted(required)},
        **semantic,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError(f"input contains null bytes: {path}")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("component inventory is empty")
    return rows


def load_hierarchy_plan(path: Path) -> list[dict[str, Any]]:
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError(f"hierarchy plan contains null bytes: {path}")
    plan = json.loads(data.decode("utf-8"))
    cells = [group for group in plan["groups"] if group.get("tier") == "family_x_depth"]
    if not cells:
        raise ValueError("hierarchy plan has no family-by-depth cells")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cell in cells:
        norm_by_id = {
            str(match["target_component_id"]): float(match["target_fro_norm"])
            for match in cell["matches"]
        }
        target_ids = [str(value) for value in cell["target_component_ids"]]
        if set(target_ids) != set(norm_by_id):
            raise ValueError(f"cell target IDs and norm receipts differ: {cell['group_id']}")
        for component_id in target_ids:
            if component_id in seen:
                raise ValueError(f"hierarchy cells overlap at {component_id}")
            seen.add(component_id)
            module, rank_text = component_id.rsplit(":rank", 1)
            match = MODULE_RE.search(module)
            if match is None:
                raise ValueError(f"cannot parse hierarchy component ID: {component_id}")
            rows.append(
                {
                    "schema": "sch.vpd-component-from-hierarchy-plan.v1",
                    "component_id": component_id,
                    "module": module,
                    "layer": int(match.group("layer")),
                    "rank_index": int(rank_text),
                    "projection_family": str(cell["projection_family"]),
                    "depth_band": int(cell["depth_band"]),
                    "component_fro_norm": norm_by_id[component_id],
                    "model_id": plan.get("model_id"),
                    "revision": plan.get("revision"),
                    "source_group_id": cell["group_id"],
                }
            )
    expected = int(plan.get("hierarchy", {}).get("cell_target_partition_component_count", 0))
    if expected and len(rows) != expected:
        raise ValueError(f"hierarchy plan expected {expected} components, reconstructed {len(rows)}")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def stable_hash(seed: int, *parts: str) -> str:
    return hashlib.sha256(":".join([str(seed), *parts]).encode("utf-8")).hexdigest()


def parse_identity(row: dict[str, Any]) -> dict[str, Any]:
    component_id = str(row["component_id"])
    module = str(row.get("module", component_id.split(":", 1)[0]))
    match = MODULE_RE.search(module)
    layer = int(row.get("layer", match.group("layer") if match else -1))
    family = str(
        row.get(
            "projection_family",
            match.group("family") if match else module.rsplit(".", 1)[-1],
        )
    )
    if layer < 0:
        raise ValueError(f"cannot determine layer for {component_id}")
    norm = float(row.get("component_fro_norm", row.get("fro_norm", 0.0)))
    if not math.isfinite(norm) or norm < 0.0:
        raise ValueError(f"invalid component norm for {component_id}")
    return {
        **row,
        "component_id": component_id,
        "module": module,
        "layer": layer,
        "projection_family": family,
        "component_fro_norm": norm,
    }


def nested_float(row: dict[str, Any], section: str, key: str) -> float:
    value = row.get(section)
    if not isinstance(value, dict):
        return 0.0
    result = float(value.get(key, 0.0))
    return result if math.isfinite(result) else 0.0


def standardize(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    return (matrix - mean) / np.where(std > 1e-12, std, 1.0)


def metadata_features(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    families = sorted({str(row["projection_family"]) for row in rows})
    family_index = {value: index for index, value in enumerate(families)}
    max_layer = max(int(row["layer"]) for row in rows)
    max_rank = max(int(row.get("rank_index", 0)) for row in rows)
    continuous: list[list[float]] = []
    one_hot = np.zeros((len(rows), len(families)), dtype=np.float64)
    for index, row in enumerate(rows):
        norm = float(row["component_fro_norm"])
        fraction = max(float(row.get("fraction_of_module_delta_fro_sq", 0.0)), 0.0)
        continuous.append(
            [
                int(row["layer"]) / max(max_layer, 1),
                int(row.get("rank_index", 0)) / max(max_rank, 1),
                math.log1p(norm),
                math.sqrt(fraction),
                nested_float(row, "discovery_cell_joint_effect", "target_invalid_margin_reduction"),
                nested_float(row, "discovery_cell_joint_effect", "target_control_advantage"),
                nested_float(row, "discovery_cell_joint_effect", "invalid_specificity_margin"),
            ]
        )
        one_hot[index, family_index[str(row["projection_family"])]] = 1.0
    continuous_names = [
        "layer",
        "rank_index",
        "log_component_fro_norm",
        "sqrt_module_fraction",
        "cell_target_effect",
        "cell_control_advantage",
        "cell_specificity",
    ]
    matrix = np.column_stack([standardize(np.asarray(continuous)), one_hot])
    return matrix, continuous_names + [f"family:{family}" for family in families]


def _project_effects(
    matrix: np.ndarray, *, seed: int, projection_dim: int
) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)
    target_dim = min(max(2, projection_dim), normalized.shape[1])
    if target_dim < normalized.shape[1]:
        rng = np.random.default_rng(seed)
        projection = rng.normal(
            0.0, 1.0 / math.sqrt(target_dim), size=(normalized.shape[1], target_dim)
        )
        normalized = normalized @ projection
    return standardize(normalized)


def weight_activation_category_direction_features(
    direction_features: np.ndarray,
    category_strength_shares: np.ndarray,
    *,
    mode: str = ACTIVATION_CATEGORY_DIRECTION_WEIGHTING,
) -> np.ndarray:
    """Bound each category direction's clustering influence by effect strength.

    Per-category direction projections are normalized independently, so leaving
    them unweighted lets arbitrarily small background effects dominate distance
    merely because there are several background categories.  Square-root share
    weighting gives every component one unit of total squared directional weight
    while retaining more secondary-category information than linear weighting.
    """

    features = np.asarray(direction_features, dtype=np.float64)
    shares = np.asarray(category_strength_shares, dtype=np.float64)
    if features.ndim != 3 or shares.shape != features.shape[:2]:
        raise ValueError("activation-category direction features and shares do not align")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(shares)):
        raise ValueError("activation-category direction features contain non-finite values")
    if np.any(shares < 0.0):
        raise ValueError("activation-category direction shares must be non-negative")
    if mode == "unweighted":
        weights = np.ones_like(shares)
    elif mode == ACTIVATION_CATEGORY_DIRECTION_WEIGHTING:
        weights = np.sqrt(shares)
    elif mode == "category_strength_share":
        weights = shares
    else:
        raise ValueError("unknown activation-category direction weighting mode")
    return features * weights[:, :, None]


def activation_category_profile_features(
    category_effects: np.ndarray,
    *,
    seed: int,
    projection_dim: int,
    direction_weighting: str = ACTIVATION_CATEGORY_DIRECTION_WEIGHTING,
) -> dict[str, Any]:
    values = np.asarray(category_effects, dtype=np.float64)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("activation-category effects must be a finite rank-three array")
    strengths = np.linalg.norm(values, axis=2)
    totals = strengths.sum(axis=1, keepdims=True)
    shares = strengths / np.maximum(totals, 1e-12)
    projected_by_category = _project_effects(
        values.reshape(-1, values.shape[2]),
        seed=seed + 1,
        projection_dim=projection_dim,
    ).reshape(values.shape[0], values.shape[1], -1)
    projected_by_category = weight_activation_category_direction_features(
        projected_by_category,
        shares,
        mode=direction_weighting,
    )
    direction_feature_dim = int(projected_by_category.shape[2])
    direction_features = projected_by_category.reshape(values.shape[0], -1)
    return {
        "features": np.column_stack(
            [
                direction_features,
                standardize(np.log1p(strengths)),
                standardize(shares),
            ]
        ),
        "strengths": strengths,
        "shares": shares,
        "direction_feature_dim_per_category": direction_feature_dim,
        "direction_feature_weighting": direction_weighting,
    }


def load_effects(
    path: Path, rows: list[dict[str, Any]], *, seed: int, projection_dim: int
) -> tuple[np.ndarray, dict[str, Any] | None]:
    archive = np.load(path, allow_pickle=False)
    missing = {"delta_h", "component_ids"} - set(archive.files)
    if missing:
        raise ValueError(f"effects archive is missing {sorted(missing)}")
    delta_h = np.asarray(archive["delta_h"], dtype=np.float64)
    effect_ids = [str(value) for value in archive["component_ids"].tolist()]
    if delta_h.ndim != 2 or delta_h.shape[0] != len(effect_ids):
        raise ValueError("effects rows and component IDs do not align")
    if len(effect_ids) != len(set(effect_ids)):
        raise ValueError("effects archive contains duplicate component IDs")
    index = {component_id: row_index for row_index, component_id in enumerate(effect_ids)}
    row_ids = [str(row["component_id"]) for row in rows]
    if set(row_ids) != set(effect_ids):
        raise ValueError("effects and component inventory IDs differ")
    ordered = delta_h[[index[component_id] for component_id in row_ids]]
    projected = _project_effects(ordered, seed=seed, projection_dim=projection_dim)
    category_arrays = {
        "category_names",
        "category_episode_counts",
        "delta_h_by_category",
    }
    present = category_arrays & set(archive.files)
    if not present:
        return projected, None
    if present != category_arrays:
        raise ValueError("effects archive has an incomplete activation-category payload")
    category_names = [str(value) for value in archive["category_names"].tolist()]
    category_counts = np.asarray(archive["category_episode_counts"])
    category_delta = np.asarray(archive["delta_h_by_category"], dtype=np.float64)
    if category_names != sorted(category_names) or len(category_names) != len(
        set(category_names)
    ):
        raise ValueError("activation category names must be sorted and unique")
    if category_counts.shape != (len(category_names),) or np.any(category_counts <= 0):
        raise ValueError("activation category episode counts are invalid")
    if category_delta.shape != (
        len(effect_ids),
        len(category_names),
        delta_h.shape[1],
    ):
        raise ValueError("activation-category effects do not align")
    if not np.all(np.isfinite(category_delta)):
        raise ValueError("activation-category effects contain non-finite values")
    category_ordered = category_delta[
        [index[component_id] for component_id in row_ids]
    ]
    category_geometry = activation_category_profile_features(
        category_ordered,
        seed=seed,
        projection_dim=projection_dim,
    )
    strengths = category_geometry["strengths"]
    shares = category_geometry["shares"]
    dominant_indices = np.argmax(strengths, axis=1)
    dominant_categories = [category_names[int(value)] for value in dominant_indices]
    profile_features = category_geometry["features"]
    uncertainty_profile: dict[str, Any] = {}
    present_uncertainty = UNCERTAINTY_ARRAYS & set(archive.files)
    if present_uncertainty and present_uncertainty != UNCERTAINTY_ARRAYS:
        raise ValueError("effects archive has an incomplete category uncertainty payload")
    if present_uncertainty:
        block_count = int(np.asarray(archive["uncertainty_block_count"]).item())
        sketch_dimension = int(
            np.asarray(archive["uncertainty_countsketch_dimension"]).item()
        )
        sketch_seed = int(np.asarray(archive["uncertainty_countsketch_seed"]).item())
        block_counts, block_sums = validate_uncertainty_arrays(
            block_counts=archive["category_uncertainty_block_counts"],
            block_sums=archive["delta_h_countsketch_block_sums"],
            component_count=len(effect_ids),
            category_count=len(category_names),
            block_count=block_count,
            sketch_dimension=sketch_dimension,
        )
        uncertainty_profile = {
            "uncertainty_block_counts": block_counts,
            "uncertainty_block_sums": block_sums[
                [index[component_id] for component_id in row_ids]
            ],
            "uncertainty_block_count": block_count,
            "uncertainty_countsketch_dimension": sketch_dimension,
            "uncertainty_countsketch_seed": sketch_seed,
        }
    condition_profile: dict[str, Any] = {}
    present_conditions = CONDITION_ROBUSTNESS_ARRAYS & set(archive.files)
    if present_conditions and present_conditions != CONDITION_ROBUSTNESS_ARRAYS:
        raise ValueError("effects archive has incomplete condition robustness arrays")
    if present_conditions:
        if not present_uncertainty:
            raise ValueError("condition robustness requires the category CountSketch contract")
        condition_names, condition_categories, condition_counts, condition_sums = (
            validate_condition_robustness_arrays(
                condition_names=archive["condition_names"],
                condition_category_names=archive["condition_category_names"],
                condition_counts=archive["condition_episode_counts"],
                condition_sums=archive["delta_h_countsketch_condition_sums"],
                component_count=len(effect_ids),
                category_names=category_names,
                sketch_dimension=uncertainty_profile[
                    "uncertainty_countsketch_dimension"
                ],
            )
        )
        condition_profile = {
            "condition_names": condition_names,
            "condition_categories": condition_categories,
            "condition_counts": condition_counts,
            "condition_sums": condition_sums[
                [index[component_id] for component_id in row_ids]
            ],
        }
    template_profile: dict[str, Any] = {}
    present_templates = TEMPLATE_ROBUSTNESS_ARRAYS & set(archive.files)
    if present_templates and present_templates != TEMPLATE_ROBUSTNESS_ARRAYS:
        raise ValueError("effects archive has incomplete template robustness arrays")
    if present_templates:
        if not present_uncertainty:
            raise ValueError("template robustness requires the category CountSketch contract")
        (
            template_cell_names,
            template_cell_categories,
            template_cell_groups,
            template_cell_counts,
            template_cell_sums,
        ) = validate_template_robustness_arrays(
            cell_names=archive["template_cell_names"],
            cell_category_names=archive["template_cell_category_names"],
            cell_group_names=archive["template_cell_group_names"],
            cell_counts=archive["template_cell_episode_counts"],
            cell_sums=archive["delta_h_countsketch_template_cell_sums"],
            component_count=len(effect_ids),
            category_names=category_names,
            sketch_dimension=uncertainty_profile[
                "uncertainty_countsketch_dimension"
            ],
        )
        template_profile = {
            "template_cell_names": template_cell_names,
            "template_cell_categories": template_cell_categories,
            "template_cell_groups": template_cell_groups,
            "template_cell_counts": template_cell_counts,
            "template_cell_sums": template_cell_sums[
                [index[component_id] for component_id in row_ids]
            ],
        }
    return np.column_stack([projected, profile_features]), {
        "category_names": category_names,
        "category_episode_counts": category_counts.astype(int).tolist(),
        "dominant_categories": dominant_categories,
        "strengths": strengths,
        "shares": shares,
        "category_effect_vectors": category_ordered,
        "direction_feature_weighting": category_geometry[
            "direction_feature_weighting"
        ],
        "direction_feature_dim_per_category": category_geometry[
            "direction_feature_dim_per_category"
        ],
        **uncertainty_profile,
        **condition_profile,
        **template_profile,
    }


def feature_matrix(
    rows: list[dict[str, Any]],
    *,
    effects_path: Path | None,
    seed: int,
    projection_dim: int,
    metadata_weight: float,
) -> tuple[np.ndarray, list[str], str, dict[str, Any] | None]:
    metadata, feature_names = metadata_features(rows)
    if effects_path is None:
        return metadata, feature_names, "metadata_and_joint_evidence", None
    effects, category_profile = load_effects(
        effects_path, rows, seed=seed, projection_dim=projection_dim
    )
    combined = np.column_stack([effects, metadata * metadata_weight])
    if category_profile is None:
        effect_names = [f"effect_projection:{i}" for i in range(effects.shape[1])]
        feature_mode = "effect_projection_plus_metadata"
    else:
        effect_names = [f"activation_category_effect:{i}" for i in range(effects.shape[1])]
        feature_mode = "share_weighted_activation_category_effects_plus_metadata"
    return combined, effect_names + feature_names, feature_mode, category_profile


def category_profile_feature_matrix(
    category_means: np.ndarray,
    category_counts: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    seed: int,
    projection_dim: int,
    metadata_weight: float,
) -> np.ndarray:
    """Rebuild the clustering matrix from category-level sufficient statistics."""

    means = np.asarray(category_means, dtype=np.float64)
    counts = np.asarray(category_counts, dtype=np.float64)
    if means.ndim != 3 or counts.shape != (means.shape[1],):
        raise ValueError("category means and counts do not align")
    if np.any(counts <= 0) or not np.all(np.isfinite(means)):
        raise ValueError("category sufficient statistics are invalid")
    aggregate_mean = np.sum(means * counts[None, :, None], axis=1) / float(
        np.sum(counts)
    )
    aggregate_features = _project_effects(
        aggregate_mean,
        seed=seed,
        projection_dim=projection_dim,
    )
    category_features = activation_category_profile_features(
        means,
        seed=seed,
        projection_dim=projection_dim,
    )["features"]
    metadata_matrix, _ = metadata_features(rows)
    return np.column_stack(
        [aggregate_features, category_features, metadata_matrix * metadata_weight]
    )


def initialize_centers(
    matrix: np.ndarray, component_ids: list[str], cluster_count: int, seed: int
) -> np.ndarray:
    first = min(range(len(component_ids)), key=lambda i: stable_hash(seed, component_ids[i]))
    selected = [first]
    nearest = np.sum((matrix - matrix[first]) ** 2, axis=1)
    while len(selected) < cluster_count:
        candidates = [index for index in range(len(component_ids)) if index not in selected]
        choice = min(
            candidates,
            key=lambda index: (-float(nearest[index]), component_ids[index]),
        )
        selected.append(choice)
        distance = np.sum((matrix - matrix[choice]) ** 2, axis=1)
        nearest = np.minimum(nearest, distance)
    return matrix[selected].copy()


def deterministic_kmeans(
    matrix: np.ndarray,
    component_ids: list[str],
    *,
    cluster_count: int,
    seed: int,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not 1 <= cluster_count <= len(component_ids):
        raise ValueError("cluster count must be between one and the component count")
    centers = initialize_centers(matrix, component_ids, cluster_count, seed)
    labels = np.full(len(component_ids), -1, dtype=np.int64)
    for iteration in range(1, max_iterations + 1):
        distances = np.sum((matrix[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, new_labels):
            return labels, centers, iteration - 1
        labels = new_labels
        new_centers = centers.copy()
        for cluster_index in range(cluster_count):
            members = matrix[labels == cluster_index]
            if len(members):
                new_centers[cluster_index] = members.mean(axis=0)
            else:
                assigned_distance = distances[np.arange(len(labels)), labels]
                candidates = sorted(
                    range(len(labels)),
                    key=lambda index: (-float(assigned_distance[index]), component_ids[index]),
                )
                replacement = next(
                    index
                    for index in candidates
                    if int(np.sum(labels == labels[index])) > 1
                )
                labels[replacement] = cluster_index
                new_centers[cluster_index] = matrix[replacement]
        centers = new_centers
    return labels, centers, max_iterations


def allocate_category_clusters(
    categories: list[str], cluster_count: int
) -> dict[str, int]:
    counts = Counter(categories)
    if cluster_count < len(counts):
        raise ValueError(
            "cluster count is smaller than the number of activation categories"
        )
    if cluster_count > len(categories):
        raise ValueError("cluster count exceeds component count")
    allocation = {category: 1 for category in sorted(counts)}
    while sum(allocation.values()) < cluster_count:
        eligible = [
            category
            for category in sorted(counts)
            if allocation[category] < counts[category]
        ]
        if not eligible:
            raise ValueError("cannot allocate requested activation-category clusters")
        selected = min(
            eligible,
            key=lambda category: (
                -counts[category] / (allocation[category] + 1),
                -counts[category],
                category,
            ),
        )
        allocation[selected] += 1
    return allocation


def deterministic_category_kmeans(
    matrix: np.ndarray,
    component_ids: list[str],
    categories: list[str],
    *,
    cluster_count: int,
    seed: int,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, int]]:
    allocation = allocate_category_clusters(categories, cluster_count)
    labels, centers, iterations = deterministic_category_kmeans_with_allocation(
        matrix,
        component_ids,
        categories,
        allocation=allocation,
        seed=seed,
        max_iterations=max_iterations,
    )
    return labels, centers, iterations, allocation


def deterministic_category_kmeans_with_allocation(
    matrix: np.ndarray,
    component_ids: list[str],
    categories: list[str],
    *,
    allocation: dict[str, int],
    seed: int,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    observed = Counter(categories)
    if set(allocation) != set(observed) or any(
        not 1 <= int(allocation[category]) <= observed[category]
        for category in observed
    ):
        raise ValueError("activation-category cluster allocation is invalid")
    labels = np.full(len(component_ids), -1, dtype=np.int64)
    centers: list[np.ndarray] = []
    iteration_counts: list[int] = []
    offset = 0
    for category in sorted(allocation):
        indices = [index for index, value in enumerate(categories) if value == category]
        local_seed = int(stable_hash(seed, "activation-category", category)[:8], 16)
        local_labels, local_centers, iterations = deterministic_kmeans(
            matrix[indices],
            [component_ids[index] for index in indices],
            cluster_count=allocation[category],
            seed=local_seed,
            max_iterations=max_iterations,
        )
        for local_index, source_index in enumerate(indices):
            labels[source_index] = offset + int(local_labels[local_index])
        centers.extend(local_centers)
        iteration_counts.append(iterations)
        offset += allocation[category]
    if np.any(labels < 0):
        raise ValueError("activation-category clustering left components unassigned")
    return labels, np.asarray(centers), max(iteration_counts, default=0)


def adjusted_rand_index(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=np.int64)
    second = np.asarray(right, dtype=np.int64)
    if first.ndim != 1 or second.shape != first.shape:
        raise ValueError("cluster label vectors do not align")
    if len(first) < 2:
        return 1.0
    first_values = sorted(set(first.tolist()))
    second_values = sorted(set(second.tolist()))
    contingency = np.zeros((len(first_values), len(second_values)), dtype=np.int64)
    first_index = {value: index for index, value in enumerate(first_values)}
    second_index = {value: index for index, value in enumerate(second_values)}
    for left_value, right_value in zip(first.tolist(), second.tolist()):
        contingency[first_index[left_value], second_index[right_value]] += 1

    def pair_count(values: np.ndarray) -> float:
        numbers = np.asarray(values, dtype=np.float64)
        return float(np.sum(numbers * (numbers - 1.0) / 2.0))

    joint_pairs = pair_count(contingency)
    first_pairs = pair_count(contingency.sum(axis=1))
    second_pairs = pair_count(contingency.sum(axis=0))
    total_pairs = len(first) * (len(first) - 1.0) / 2.0
    expected = first_pairs * second_pairs / total_pairs
    maximum = 0.5 * (first_pairs + second_pairs)
    denominator = maximum - expected
    if abs(denominator) <= 1e-12:
        same_partition = all(
            (first[left_index] == first[right_index])
            == (second[left_index] == second[right_index])
            for left_index in range(len(first))
            for right_index in range(left_index + 1, len(first))
        )
        return 1.0 if same_partition else 0.0
    return float((joint_pairs - expected) / denominator)


def mean_silhouette(matrix: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    assignments = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or assignments.shape != (len(values),):
        raise ValueError("silhouette matrix and labels do not align")
    if not np.all(np.isfinite(values)):
        raise ValueError("silhouette matrix contains non-finite values")
    unique_labels = sorted(set(assignments.tolist()))
    if len(unique_labels) < 2:
        return 0.0
    distances = np.sqrt(
        np.maximum(
            np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=2),
            0.0,
        )
    )
    silhouettes: list[float] = []
    for index, label in enumerate(assignments.tolist()):
        same = np.where(assignments == label)[0]
        same = same[same != index]
        if len(same) == 0:
            silhouettes.append(0.0)
            continue
        within = float(np.mean(distances[index, same]))
        between = min(
            float(np.mean(distances[index, assignments == other]))
            for other in unique_labels
            if other != label
        )
        scale = max(within, between)
        silhouettes.append((between - within) / scale if scale > 1e-12 else 0.0)
    return float(np.mean(silhouettes))


def select_category_cluster_counts(
    replicate_matrices: list[np.ndarray],
    component_ids: list[str],
    categories: list[str],
    *,
    seed: int,
    replicate_kinds: list[str] | None = None,
    maximum_clusters_per_category: int = DEFAULT_MAXIMUM_CLUSTERS_PER_CATEGORY,
    minimum_replicate_stability: float = DEFAULT_MINIMUM_CATEGORY_CLUSTER_PROJECTION_STABILITY,
    minimum_silhouette: float = DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE,
    minimum_cluster_size: int = DEFAULT_MINIMUM_CATEGORY_CLUSTER_SIZE,
    max_iterations: int = 100,
) -> dict[str, Any]:
    """Select nontrivial within-category counts only when geometry replicates.

    Each candidate is clustered independently in every projected or resampled
    feature space.
    A subdivision must retain adjusted-Rand agreement, positive separation, and
    minimum support in every projection.  If no nontrivial candidate passes, the
    category stays as one conservative screening cluster.
    """

    if len(replicate_matrices) < 2:
        raise ValueError("category cluster selection requires multiple projections")
    matrices = [np.asarray(matrix, dtype=np.float64) for matrix in replicate_matrices]
    if any(matrix.ndim != 2 or matrix.shape[0] != len(component_ids) for matrix in matrices):
        raise ValueError("category cluster projection matrices do not align")
    if any(not np.all(np.isfinite(matrix)) for matrix in matrices):
        raise ValueError("category cluster projection matrix contains non-finite values")
    if len(component_ids) != len(set(component_ids)) or len(categories) != len(
        component_ids
    ):
        raise ValueError("category cluster component identities do not align")
    if maximum_clusters_per_category < 1:
        raise ValueError("maximum clusters per category must be positive")
    if not 0.0 <= minimum_replicate_stability <= 1.0:
        raise ValueError("minimum cluster projection stability must be in [0, 1]")
    if not -1.0 <= minimum_silhouette <= 1.0:
        raise ValueError("minimum cluster silhouette must be in [-1, 1]")
    if minimum_cluster_size < 2:
        raise ValueError("minimum category cluster size must be at least two")
    kinds = (
        ["projection"] * len(matrices)
        if replicate_kinds is None
        else [str(value) for value in replicate_kinds]
    )
    if len(kinds) != len(matrices) or any(not value for value in kinds):
        raise ValueError("category cluster replicate kinds do not align")

    allocation: dict[str, int] = {}
    category_results: dict[str, Any] = {}
    for category in sorted(set(categories)):
        indices = [index for index, value in enumerate(categories) if value == category]
        local_ids = [component_ids[index] for index in indices]
        maximum_candidate = min(
            maximum_clusters_per_category,
            len(indices) // minimum_cluster_size,
        )
        candidates: list[dict[str, Any]] = [
            {
                "cluster_count": 1,
                "status": "pass_fallback",
                "minimum_replicate_adjusted_rand": 1.0,
                "minimum_replicate_silhouette": 0.0,
                "minimum_observed_cluster_size": len(indices),
            }
        ]
        for cluster_count in range(2, maximum_candidate + 1):
            local_seed = int(
                stable_hash(
                    seed,
                    "activation-category-count-selection",
                    category,
                    str(cluster_count),
                )[:8],
                16,
            )
            replicate_labels: list[np.ndarray] = []
            silhouettes: list[float] = []
            cluster_sizes: list[int] = []
            for matrix in matrices:
                labels, _, _ = deterministic_kmeans(
                    matrix[indices],
                    local_ids,
                    cluster_count=cluster_count,
                    seed=local_seed,
                    max_iterations=max_iterations,
                )
                replicate_labels.append(labels)
                silhouettes.append(mean_silhouette(matrix[indices], labels))
                cluster_sizes.append(
                    min(int(np.sum(labels == label)) for label in set(labels.tolist()))
                )
            stability_values = [
                adjusted_rand_index(replicate_labels[left], replicate_labels[right])
                for left in range(len(replicate_labels))
                for right in range(left + 1, len(replicate_labels))
            ]
            candidate = {
                "cluster_count": cluster_count,
                "status": "pass",
                "minimum_replicate_adjusted_rand": min(stability_values),
                "minimum_replicate_silhouette": min(silhouettes),
                "minimum_observed_cluster_size": min(cluster_sizes),
            }
            if candidate["minimum_replicate_adjusted_rand"] < minimum_replicate_stability:
                candidate["status"] = "reject_replicate_instability"
            elif candidate["minimum_replicate_silhouette"] < minimum_silhouette:
                candidate["status"] = "reject_weak_separation"
            elif candidate["minimum_observed_cluster_size"] < minimum_cluster_size:
                candidate["status"] = "reject_small_cluster"
            candidates.append(candidate)
        passing = [row for row in candidates if row["status"] == "pass"]
        selected = (
            max(
                passing,
                key=lambda row: (
                    float(row["minimum_replicate_silhouette"]),
                    float(row["minimum_replicate_adjusted_rand"]),
                    -int(row["cluster_count"]),
                ),
            )
            if passing
            else candidates[0]
        )
        allocation[category] = int(selected["cluster_count"])
        category_results[category] = {
            "component_count": len(indices),
            "candidates": candidates,
            "selection": selected,
        }
    return {
        "schema": "sch.vpd-category-cluster-count-selection.v1",
        "status": "pass",
        "policy": {
            "replicates": len(matrices),
            "replicate_kind_counts": dict(sorted(Counter(kinds).items())),
            "maximum_clusters_per_category": maximum_clusters_per_category,
            "minimum_replicate_adjusted_rand": minimum_replicate_stability,
            "minimum_replicate_silhouette": minimum_silhouette,
            "minimum_cluster_size": minimum_cluster_size,
            "no_passing_nontrivial_candidate_rule": "one_cluster_for_category",
            "selection_order": [
                "maximum_minimum_replicate_silhouette",
                "maximum_minimum_replicate_adjusted_rand",
                "minimum_cluster_count",
            ],
        },
        "allocation": allocation,
        "categories": category_results,
    }


def classify_activation_compartments(
    strengths: np.ndarray,
    category_names: list[str],
    *,
    minimum_dominant_share: float,
    minimum_dominance_margin: float,
    maximum_normalized_entropy: float,
) -> dict[str, Any]:
    values = np.asarray(strengths, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(category_names):
        raise ValueError("activation strengths do not align with category names")
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("activation strengths must be finite and non-negative")
    if not 0.0 <= minimum_dominant_share <= 1.0:
        raise ValueError("minimum dominant share must be in [0, 1]")
    if not 0.0 <= minimum_dominance_margin <= 1.0:
        raise ValueError("minimum dominance margin must be in [0, 1]")
    if not 0.0 <= maximum_normalized_entropy <= 1.0:
        raise ValueError("maximum normalized entropy must be in [0, 1]")
    totals = values.sum(axis=1, keepdims=True)
    shares = values / np.maximum(totals, 1e-12)
    ordering = np.argsort(-shares, axis=1, kind="stable")
    dominant_indices = ordering[:, 0]
    dominant_shares = shares[np.arange(len(values)), dominant_indices]
    runner_up_shares = (
        shares[np.arange(len(values)), ordering[:, 1]]
        if len(category_names) > 1
        else np.zeros(len(values), dtype=np.float64)
    )
    dominance_margins = dominant_shares - runner_up_shares
    if len(category_names) > 1:
        safe = np.where(shares > 0.0, shares, 1.0)
        entropy = -np.sum(np.where(shares > 0.0, shares * np.log(safe), 0.0), axis=1)
        normalized_entropy = entropy / math.log(len(category_names))
    else:
        normalized_entropy = np.zeros(len(values), dtype=np.float64)
    dominant_categories = [category_names[int(index)] for index in dominant_indices]
    compartment_categories = []
    for index, dominant in enumerate(dominant_categories):
        specific = (
            totals[index, 0] > 1e-12
            and dominant_shares[index] >= minimum_dominant_share
            and dominance_margins[index] >= minimum_dominance_margin
            and normalized_entropy[index] <= maximum_normalized_entropy
        )
        compartment_categories.append(dominant if specific else "shared_cross_category")
    return {
        "shares": shares,
        "dominant_categories": dominant_categories,
        "compartment_categories": compartment_categories,
        "dominant_shares": dominant_shares,
        "dominance_margins": dominance_margins,
        "normalized_entropy": normalized_entropy,
    }


def cross_archive_category_direction_cosines(
    discovery_effects: np.ndarray, calibration_effects: np.ndarray
) -> np.ndarray:
    discovery = np.asarray(discovery_effects, dtype=np.float64)
    calibration = np.asarray(calibration_effects, dtype=np.float64)
    if discovery.shape != calibration.shape or discovery.ndim != 3:
        raise ValueError("discovery and calibration category effect vectors do not align")
    if not np.all(np.isfinite(discovery)) or not np.all(np.isfinite(calibration)):
        raise ValueError("category effect vectors contain non-finite values")
    discovery_norms = np.linalg.norm(discovery, axis=2)
    calibration_norms = np.linalg.norm(calibration, axis=2)
    denominator = discovery_norms * calibration_norms
    dots = np.sum(discovery * calibration, axis=2)
    cosines = np.zeros_like(dots, dtype=np.float64)
    supported = denominator > 1e-12
    cosines[supported] = dots[supported] / denominator[supported]
    return np.clip(cosines, -1.0, 1.0)


def calibrate_activation_compartment_thresholds(
    discovery_strengths: np.ndarray,
    calibration_strengths: np.ndarray,
    category_names: list[str],
    *,
    direction_cosines: np.ndarray,
    minimum_replication_rate: float = DEFAULT_MINIMUM_CATEGORY_REPLICATION_RATE,
    minimum_stable_specific_components: int = DEFAULT_MINIMUM_STABLE_SPECIFIC_COMPONENTS,
    minimum_direction_cosine: float = DEFAULT_MINIMUM_CATEGORY_DIRECTION_COSINE,
    minimum_strength_ratio: float = DEFAULT_MINIMUM_CATEGORY_STRENGTH_RATIO,
    robustness_eligibility: np.ndarray | None = None,
    candidate_grid: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Select fail-closed specificity gates from an independent replicate.

    Discovery effects remain the only clustering features.  The independent
    calibration replicate is used solely to choose a threshold triple whose
    non-shared assignments reproduce symmetrically at the frozen minimum rate.
    """

    discovery = np.asarray(discovery_strengths, dtype=np.float64)
    calibration = np.asarray(calibration_strengths, dtype=np.float64)
    if discovery.shape != calibration.shape:
        raise ValueError("discovery and calibration strengths do not align")
    if discovery.ndim != 2 or discovery.shape[1] != len(category_names):
        raise ValueError("calibration strengths do not align with category names")
    if (
        not np.all(np.isfinite(discovery))
        or not np.all(np.isfinite(calibration))
        or np.any(discovery < 0.0)
        or np.any(calibration < 0.0)
    ):
        raise ValueError("calibration strengths must be finite and non-negative")
    cosines = np.asarray(direction_cosines, dtype=np.float64)
    if cosines.shape != discovery.shape or not np.all(np.isfinite(cosines)):
        raise ValueError("calibration direction cosines do not align")
    if np.any(cosines < -1.0000001) or np.any(cosines > 1.0000001):
        raise ValueError("calibration direction cosines must be in [-1, 1]")
    if len(category_names) < 2 or len(category_names) != len(set(category_names)):
        raise ValueError("calibration requires at least two unique categories")
    if not 0.0 < minimum_replication_rate <= 1.0:
        raise ValueError("minimum replication rate must be in (0, 1]")
    if minimum_stable_specific_components < 1:
        raise ValueError("minimum stable specific components must be positive")
    if not -1.0 <= minimum_direction_cosine <= 1.0:
        raise ValueError("minimum direction cosine must be in [-1, 1]")
    if not 0.0 < minimum_strength_ratio <= 1.0:
        raise ValueError("minimum strength ratio must be in (0, 1]")
    maximum_strengths = np.maximum(discovery, calibration)
    strength_ratios = np.zeros_like(maximum_strengths)
    supported_strengths = maximum_strengths > 1e-12
    strength_ratios[supported_strengths] = (
        np.minimum(discovery, calibration)[supported_strengths]
        / maximum_strengths[supported_strengths]
    )
    eligibility = (
        np.ones_like(discovery, dtype=bool)
        if robustness_eligibility is None
        else np.asarray(robustness_eligibility, dtype=bool)
    )
    if eligibility.shape != discovery.shape:
        raise ValueError("calibration robustness eligibility does not align")
    grid = json_normalized(candidate_grid or DEFAULT_CATEGORY_CALIBRATION_GRID)
    expected_grid_keys = {
        "minimum_dominant_share",
        "minimum_dominance_margin",
        "maximum_normalized_entropy",
    }
    if set(grid) != expected_grid_keys or any(not grid[key] for key in grid):
        raise ValueError("activation-category calibration grid is incomplete")
    for key, values in grid.items():
        normalized_values = [float(value) for value in values]
        if normalized_values != sorted(set(normalized_values)):
            raise ValueError(f"calibration grid values must be sorted and unique: {key}")
        if any(not 0.0 <= value <= 1.0 for value in normalized_values):
            raise ValueError(f"calibration grid values must be in [0, 1]: {key}")
        grid[key] = normalized_values

    candidates: list[dict[str, Any]] = []
    for minimum_share in grid["minimum_dominant_share"]:
        for minimum_margin in grid["minimum_dominance_margin"]:
            for maximum_entropy in grid["maximum_normalized_entropy"]:
                thresholds = {
                    "minimum_dominant_share": minimum_share,
                    "minimum_dominance_margin": minimum_margin,
                    "maximum_normalized_entropy": maximum_entropy,
                    "zero_total_effect_routes_to_shared": True,
                }
                discovery_result = classify_activation_compartments(
                    discovery,
                    category_names,
                    minimum_dominant_share=minimum_share,
                    minimum_dominance_margin=minimum_margin,
                    maximum_normalized_entropy=maximum_entropy,
                )
                calibration_result = classify_activation_compartments(
                    calibration,
                    category_names,
                    minimum_dominant_share=minimum_share,
                    minimum_dominance_margin=minimum_margin,
                    maximum_normalized_entropy=maximum_entropy,
                )
                discovery_labels = discovery_result["compartment_categories"]
                calibration_labels = calibration_result["compartment_categories"]
                discovery_specific = [
                    label != "shared_cross_category" for label in discovery_labels
                ]
                calibration_specific = [
                    label != "shared_cross_category" for label in calibration_labels
                ]
                stable_label_specific = [
                    left == right and left != "shared_cross_category"
                    for left, right in zip(discovery_labels, calibration_labels)
                ]
                direction_stable_specific = [
                    stable
                    and cosines[index, category_names.index(discovery_labels[index])]
                    >= minimum_direction_cosine
                    for index, stable in enumerate(stable_label_specific)
                ]
                magnitude_stable_specific = [
                    stable
                    and strength_ratios[
                        index, category_names.index(discovery_labels[index])
                    ]
                    >= minimum_strength_ratio
                    for index, stable in enumerate(direction_stable_specific)
                ]
                stable_specific = [
                    stable
                    and bool(
                        eligibility[
                            index, category_names.index(discovery_labels[index])
                        ]
                    )
                    for index, stable in enumerate(magnitude_stable_specific)
                ]
                discovery_specific_count = int(sum(discovery_specific))
                calibration_specific_count = int(sum(calibration_specific))
                stable_specific_count = int(sum(stable_specific))
                stable_label_specific_count = int(sum(stable_label_specific))
                direction_stable_specific_count = int(
                    sum(direction_stable_specific)
                )
                magnitude_stable_specific_count = int(
                    sum(magnitude_stable_specific)
                )
                union_specific_count = int(sum(
                    left or right
                    for left, right in zip(discovery_specific, calibration_specific)
                ))
                discovery_replication_rate = (
                    stable_specific_count / discovery_specific_count
                    if discovery_specific_count
                    else 0.0
                )
                symmetric_replication_rate = (
                    stable_specific_count / union_specific_count
                    if union_specific_count
                    else 0.0
                )
                exact_agreement_rate = sum(
                    left == right
                    for left, right in zip(discovery_labels, calibration_labels)
                ) / len(discovery_labels)
                decision_slacks: list[float] = []
                for index, stable in enumerate(stable_specific):
                    if not stable:
                        continue
                    target_index = category_names.index(discovery_labels[index])
                    decision_slacks.append(
                        min(
                            float(discovery_result["dominant_shares"][index])
                            - minimum_share,
                            float(calibration_result["dominant_shares"][index])
                            - minimum_share,
                            float(discovery_result["dominance_margins"][index])
                            - minimum_margin,
                            float(calibration_result["dominance_margins"][index])
                            - minimum_margin,
                            maximum_entropy
                            - float(discovery_result["normalized_entropy"][index]),
                            maximum_entropy
                            - float(calibration_result["normalized_entropy"][index]),
                            float(cosines[index, target_index])
                            - minimum_direction_cosine,
                            float(strength_ratios[index, target_index])
                            - minimum_strength_ratio,
                        )
                    )
                minimum_decision_slack = (
                    min(decision_slacks) if decision_slacks else 0.0
                )
                metrics = {
                    "component_count": len(discovery_labels),
                    "discovery_specific_count": discovery_specific_count,
                    "calibration_specific_count": calibration_specific_count,
                    "stable_specific_count": stable_specific_count,
                    "stable_label_specific_count": stable_label_specific_count,
                    "direction_rejected_specific_count": (
                        stable_label_specific_count - direction_stable_specific_count
                    ),
                    "magnitude_rejected_specific_count": (
                        direction_stable_specific_count
                        - magnitude_stable_specific_count
                    ),
                    "stratified_robustness_rejected_specific_count": (
                        magnitude_stable_specific_count - stable_specific_count
                    ),
                    "unstable_specific_union_count": (
                        union_specific_count - stable_specific_count
                    ),
                    "discovery_replication_rate": discovery_replication_rate,
                    "symmetric_replication_rate": symmetric_replication_rate,
                    "exact_compartment_agreement_rate": exact_agreement_rate,
                    "minimum_stable_specific_decision_slack": (
                        minimum_decision_slack
                    ),
                }
                if (
                    stable_specific_count >= minimum_stable_specific_components
                    and discovery_replication_rate >= minimum_replication_rate
                    and symmetric_replication_rate >= minimum_replication_rate
                ):
                    candidates.append({"thresholds": thresholds, "metrics": metrics})
    if not candidates:
        raise ValueError(
            "no activation-category threshold candidate meets the frozen "
            "independent replication policy"
        )
    selected = max(
        candidates,
        key=lambda row: (
            int(row["metrics"]["stable_specific_count"]),
            float(row["metrics"]["symmetric_replication_rate"]),
            float(row["metrics"]["exact_compartment_agreement_rate"]),
            float(row["metrics"]["minimum_stable_specific_decision_slack"]),
            float(row["thresholds"]["minimum_dominant_share"]),
            float(row["thresholds"]["minimum_dominance_margin"]),
            -float(row["thresholds"]["maximum_normalized_entropy"]),
        ),
    )
    return {
        "policy": {
            "minimum_replication_rate": minimum_replication_rate,
            "minimum_stable_specific_components": minimum_stable_specific_components,
            "minimum_direction_cosine": minimum_direction_cosine,
            "minimum_strength_ratio": minimum_strength_ratio,
            "cross_archive_stratified_robustness_required": (
                robustness_eligibility is not None
            ),
            "replication_rule": (
                "same_non_shared_category_signed_direction_cosine_and_dominant_"
                "strength_ratio_in_discovery_and_independent_calibration"
            ),
            "selection_order": [
                "maximum_stable_specific_count",
                "maximum_symmetric_replication_rate",
                "maximum_exact_compartment_agreement_rate",
                "maximum_minimum_stable_specific_decision_slack",
                "strictest_thresholds_lexicographic",
            ],
        },
        "candidate_grid": grid,
        "selection": {
            "evaluated_candidate_count": math.prod(len(values) for values in grid.values()),
            "passing_candidate_count": len(candidates),
            "selected_thresholds": selected["thresholds"],
            "selected_metrics": selected["metrics"],
        },
    }


def apply_cross_archive_component_admission(
    pre_admission_categories: list[str],
    calibration_strengths: np.ndarray,
    category_names: list[str],
    *,
    thresholds: dict[str, Any],
    direction_cosines: np.ndarray,
    strength_ratios: np.ndarray,
    minimum_direction_cosine: float,
    minimum_strength_ratio: float,
    robustness_eligibility: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fail closed for every category-specific component across calibration.

    Aggregate calibration decides whether a threshold policy is usable at all.
    This second gate applies that policy to each component: a discovery-specific
    component remains specific only when the independent calibration archive
    assigns the same category and independently preserves its direction,
    magnitude, and requested stratified robustness screens.
    """

    calibration = np.asarray(calibration_strengths, dtype=np.float64)
    cosines = np.asarray(direction_cosines, dtype=np.float64)
    ratios = np.asarray(strength_ratios, dtype=np.float64)
    if calibration.ndim != 2 or calibration.shape[1] != len(category_names):
        raise ValueError("component admission calibration strengths do not align")
    if calibration.shape[0] != len(pre_admission_categories):
        raise ValueError("component admission categories do not align")
    if cosines.shape != calibration.shape or ratios.shape != calibration.shape:
        raise ValueError("component admission cross-archive diagnostics do not align")
    if (
        not np.all(np.isfinite(calibration))
        or not np.all(np.isfinite(cosines))
        or not np.all(np.isfinite(ratios))
        or np.any(calibration < 0.0)
        or np.any(cosines < -1.0000001)
        or np.any(cosines > 1.0000001)
        or np.any(ratios < 0.0)
        or np.any(ratios > 1.0000001)
    ):
        raise ValueError("component admission diagnostics are invalid")
    if len(category_names) < 2 or len(category_names) != len(set(category_names)):
        raise ValueError("component admission category names are invalid")
    valid_categories = set(category_names) | {"shared_cross_category"}
    if any(category not in valid_categories for category in pre_admission_categories):
        raise ValueError("component admission contains an unknown discovery category")
    if not -1.0 <= minimum_direction_cosine <= 1.0:
        raise ValueError("component admission direction threshold is invalid")
    if not 0.0 < minimum_strength_ratio <= 1.0:
        raise ValueError("component admission strength threshold is invalid")
    eligibility = (
        np.ones_like(calibration, dtype=bool)
        if robustness_eligibility is None
        else np.asarray(robustness_eligibility, dtype=bool)
    )
    if eligibility.shape != calibration.shape:
        raise ValueError("component admission robustness eligibility does not align")

    calibration_result = classify_activation_compartments(
        calibration,
        category_names,
        minimum_dominant_share=float(thresholds["minimum_dominant_share"]),
        minimum_dominance_margin=float(thresholds["minimum_dominance_margin"]),
        maximum_normalized_entropy=float(thresholds["maximum_normalized_entropy"]),
    )
    calibration_categories = calibration_result["compartment_categories"]
    final_categories: list[str] = []
    admitted_specific: list[bool] = []
    rejection_reasons: list[str | None] = []
    for index, discovery_category in enumerate(pre_admission_categories):
        if discovery_category == "shared_cross_category":
            final_categories.append(discovery_category)
            admitted_specific.append(False)
            rejection_reasons.append("already_shared_upstream")
            continue
        category_index = category_names.index(discovery_category)
        if calibration_categories[index] != discovery_category:
            reason = "calibration_category_mismatch"
        elif cosines[index, category_index] < minimum_direction_cosine:
            reason = "cross_archive_direction_instability"
        elif ratios[index, category_index] < minimum_strength_ratio:
            reason = "cross_archive_magnitude_instability"
        elif not bool(eligibility[index, category_index]):
            reason = "cross_archive_stratified_robustness_failure"
        else:
            reason = None
        admitted = reason is None
        final_categories.append(
            discovery_category if admitted else "shared_cross_category"
        )
        admitted_specific.append(admitted)
        rejection_reasons.append(reason)
    return {
        "schema": "sch.vpd-cross-archive-component-admission.v1",
        "calibration_point_estimate_categories": calibration_categories,
        "final_categories": final_categories,
        "admitted_specific": admitted_specific,
        "rejection_reasons": rejection_reasons,
        "admitted_specific_count": int(sum(admitted_specific)),
        "rerouted_specific_count": int(
            sum(
                category != "shared_cross_category" and not admitted
                for category, admitted in zip(
                    pre_admission_categories, admitted_specific
                )
            )
        ),
    }


def category_profile_stratified_robustness(
    profile: dict[str, Any],
    *,
    minimum_pair_block_coverage: float,
    minimum_pair_block_direction_cosine: float,
    minimum_pair_block_relative_norm: float,
    require_pair_block: bool,
    minimum_condition_coverage: float,
    minimum_condition_direction_cosine: float,
    minimum_condition_relative_norm: float,
    require_condition: bool,
    minimum_template_coverage: float,
    minimum_template_direction_cosine: float,
    minimum_template_relative_norm: float,
    require_template: bool,
    minimum_sketch_dimension: int,
) -> dict[str, Any]:
    component_count, category_count = np.asarray(profile["strengths"]).shape
    eligibility = np.ones((component_count, category_count), dtype=bool)
    if minimum_sketch_dimension < 2:
        raise ValueError("minimum category robustness CountSketch dimension is invalid")
    observed_dimension = int(profile.get("uncertainty_countsketch_dimension", 0))
    if (require_pair_block or require_condition or require_template) and (
        observed_dimension < minimum_sketch_dimension
    ):
        raise ValueError("category robustness CountSketch dimension is underpowered")
    result: dict[str, Any] = {
        "eligible": eligibility,
        "pair_block": None,
        "condition": None,
        "template": None,
    }
    if "uncertainty_block_sums" in profile:
        pair_block = pair_block_direction_metrics(
            block_sums=profile["uncertainty_block_sums"],
            block_counts=profile["uncertainty_block_counts"],
            minimum_direction_cosine=minimum_pair_block_direction_cosine,
            minimum_relative_norm=minimum_pair_block_relative_norm,
        )
        eligibility &= pair_block["coverage"] >= minimum_pair_block_coverage
        result["pair_block"] = pair_block
    elif require_pair_block:
        raise ValueError("category pair-block direction evidence is required but absent")
    if "condition_sums" in profile:
        condition = condition_robustness_metrics(
            condition_sums=profile["condition_sums"],
            condition_counts=profile["condition_counts"],
            condition_categories=profile["condition_categories"],
            category_names=profile["category_names"],
            minimum_direction_cosine=minimum_condition_direction_cosine,
            minimum_relative_norm=minimum_condition_relative_norm,
        )
        eligibility &= condition["coverage"] >= minimum_condition_coverage
        result["condition"] = condition
    elif require_condition:
        raise ValueError("category condition robustness evidence is required but absent")
    if "template_cell_sums" in profile:
        template = template_robustness_metrics(
            cell_sums=profile["template_cell_sums"],
            cell_counts=profile["template_cell_counts"],
            cell_categories=profile["template_cell_categories"],
            category_names=profile["category_names"],
            minimum_direction_cosine=minimum_template_direction_cosine,
            minimum_relative_norm=minimum_template_relative_norm,
        )
        eligibility &= template["coverage"] >= minimum_template_coverage
        result["template"] = template
    elif require_template:
        raise ValueError("category template robustness evidence is required but absent")
    return result


def rendezvous_worker(packet_id: str, workers: list[str]) -> str | None:
    if not workers:
        return None
    return max(workers, key=lambda worker: stable_hash(0, packet_id, worker))


def summarize_cluster(
    rows: list[dict[str, Any]], matrix: np.ndarray, center: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    distances = np.sqrt(np.sum((matrix - center) ** 2, axis=1))
    ordering = sorted(
        range(len(rows)),
        key=lambda index: (
            float(distances[index]),
            -float(rows[index]["component_fro_norm"]),
            str(rows[index]["component_id"]),
        ),
    )
    representatives = []
    for index in ordering[: min(8, len(ordering))]:
        row = rows[index]
        representatives.append(
            {
                "component_id": row["component_id"],
                "layer": row["layer"],
                "projection_family": row["projection_family"],
                "component_fro_norm": row["component_fro_norm"],
                "distance_to_centroid": float(distances[index]),
            }
        )
    layers = Counter(int(row["layer"]) for row in rows)
    families = Counter(str(row["projection_family"]) for row in rows)
    capture_layers = [
        layer for layer, _ in sorted(layers.items(), key=lambda item: (-item[1], item[0]))[:4]
    ]
    summary = {
        "component_count": len(rows),
        "layers": dict(sorted(layers.items())),
        "projection_families": dict(sorted(families.items())),
        "component_fro_norm_rss": math.sqrt(
            sum(float(row["component_fro_norm"]) ** 2 for row in rows)
        ),
        "mean_distance_to_centroid": float(distances.mean()),
        "max_distance_to_centroid": float(distances.max()),
        "capture_layers": capture_layers,
    }
    return summary, representatives


def build_bundle(
    rows: list[dict[str, Any]],
    *,
    inventory_sha256: str,
    effects_path: Path | None,
    output_dir: Path,
    cluster_count: int,
    seed: int,
    projection_dim: int,
    metadata_weight: float,
    max_iterations: int,
    anchors: list[str],
    workers: list[str],
    source_kind: str = "component_inventory_jsonl",
    effect_space_id: str = "layer_35__proposal_start",
    category_compartments: bool = True,
    calibration_effects_path: Path | None = None,
    cluster_validation_effects_path: Path | None = None,
    effects_merge_manifest_path: Path | None = None,
    calibration_effects_merge_manifest_path: Path | None = None,
    cluster_validation_effects_merge_manifest_path: Path | None = None,
    require_calibration_split_provenance: bool = True,
    require_cluster_validation_split_provenance: bool = True,
    minimum_dominant_category_share: float = DEFAULT_MINIMUM_DOMINANT_CATEGORY_SHARE,
    minimum_category_dominance_margin: float = DEFAULT_MINIMUM_CATEGORY_DOMINANCE_MARGIN,
    maximum_category_normalized_entropy: float = DEFAULT_MAXIMUM_CATEGORY_NORMALIZED_ENTROPY,
    minimum_category_replication_rate: float = DEFAULT_MINIMUM_CATEGORY_REPLICATION_RATE,
    minimum_stable_specific_components: int = DEFAULT_MINIMUM_STABLE_SPECIFIC_COMPONENTS,
    minimum_category_calibration_episodes: int = DEFAULT_MINIMUM_CATEGORY_CALIBRATION_EPISODES,
    minimum_category_direction_cosine: float = DEFAULT_MINIMUM_CATEGORY_DIRECTION_COSINE,
    minimum_category_strength_ratio: float = DEFAULT_MINIMUM_CATEGORY_STRENGTH_RATIO,
    minimum_category_jackknife_stability: float = DEFAULT_MINIMUM_CATEGORY_JACKKNIFE_STABILITY,
    require_category_uncertainty: bool = False,
    minimum_category_pair_block_direction_coverage: float = DEFAULT_MINIMUM_CATEGORY_PAIR_BLOCK_DIRECTION_COVERAGE,
    minimum_pair_block_direction_cosine: float = DEFAULT_MINIMUM_PAIR_BLOCK_DIRECTION_COSINE,
    minimum_pair_block_relative_norm: float = DEFAULT_MINIMUM_PAIR_BLOCK_RELATIVE_NORM,
    require_category_pair_block_direction_robustness: bool = False,
    minimum_category_condition_coverage: float = DEFAULT_MINIMUM_CATEGORY_CONDITION_COVERAGE,
    minimum_condition_direction_cosine: float = DEFAULT_MINIMUM_CONDITION_DIRECTION_COSINE,
    minimum_condition_relative_norm: float = DEFAULT_MINIMUM_CONDITION_RELATIVE_NORM,
    require_category_condition_robustness: bool = False,
    minimum_category_template_coverage: float = DEFAULT_MINIMUM_CATEGORY_TEMPLATE_COVERAGE,
    minimum_template_direction_cosine: float = DEFAULT_MINIMUM_TEMPLATE_DIRECTION_COSINE,
    minimum_template_relative_norm: float = DEFAULT_MINIMUM_TEMPLATE_RELATIVE_NORM,
    require_category_template_robustness: bool = False,
    minimum_category_robustness_sketch_dimension: int = DEFAULT_MINIMUM_CATEGORY_ROBUSTNESS_SKETCH_DIMENSION,
    select_category_cluster_counts_by_projection: bool = False,
    category_cluster_projection_replicates: int = DEFAULT_CATEGORY_CLUSTER_PROJECTION_REPLICATES,
    maximum_clusters_per_category: int = DEFAULT_MAXIMUM_CLUSTERS_PER_CATEGORY,
    minimum_category_cluster_projection_stability: float = DEFAULT_MINIMUM_CATEGORY_CLUSTER_PROJECTION_STABILITY,
    minimum_category_cluster_silhouette: float = DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE,
    minimum_category_cluster_silhouette_p10: float = DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE_P10,
    minimum_category_cluster_boundary_gap_ratio: float = DEFAULT_MINIMUM_CATEGORY_CLUSTER_BOUNDARY_GAP_RATIO,
    minimum_category_cluster_size: int = DEFAULT_MINIMUM_CATEGORY_CLUSTER_SIZE,
    category_cluster_count_selector_version: str = "v1",
    require_category_cluster_sampling_stability: bool = False,
    require_category_cluster_cross_archive_stability: bool = False,
) -> dict[str, Any]:
    if category_cluster_count_selector_version not in {"v1", "v2"}:
        raise ValueError("category cluster count selector version must be v1 or v2")
    effect_space_match = re.fullmatch(r"layer_(\d+)__([A-Za-z0-9_]+)", effect_space_id)
    if effect_space_match is None:
        raise ValueError("effect_space_id must be layer_N__anchor")
    alignment_layer = int(effect_space_match.group(1))
    parsed = sorted((parse_identity(row) for row in rows), key=lambda row: row["component_id"])
    component_ids = [str(row["component_id"]) for row in parsed]
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("component IDs are not unique")
    matrix, feature_names, feature_mode, category_profile = feature_matrix(
        parsed,
        effects_path=effects_path,
        seed=seed,
        projection_dim=projection_dim,
        metadata_weight=metadata_weight,
    )
    compartment_enabled = category_compartments and category_profile is not None
    category_calibration: dict[str, Any] | None = None
    component_cross_archive_admission: dict[str, Any] | None = None
    category_cluster_count_selection: dict[str, Any] | None = None
    cluster_validation_matrix: np.ndarray | None = None
    cluster_validation_profile: dict[str, Any] | None = None
    cluster_validation_provenance: dict[str, Any] | None = None
    category_specificity_thresholds = {
        "minimum_dominant_share": minimum_dominant_category_share,
        "minimum_dominance_margin": minimum_category_dominance_margin,
        "maximum_normalized_entropy": maximum_category_normalized_entropy,
        "zero_total_effect_routes_to_shared": True,
    }
    pair_block_direction_screen = {
        "mode": (
            "pair_block_mean_countsketch_direction_and_relative_norm_coverage"
            if category_profile is not None
            and "uncertainty_block_sums" in category_profile
            else "unavailable_legacy_archive"
        ),
        "required": require_category_pair_block_direction_robustness,
        "minimum_coverage": minimum_category_pair_block_direction_coverage,
        "minimum_direction_cosine": minimum_pair_block_direction_cosine,
        "minimum_relative_norm": minimum_pair_block_relative_norm,
        "failed_specific_routes_to": "shared_cross_category",
        "minimum_countsketch_dimension": minimum_category_robustness_sketch_dimension,
        "observed_countsketch_dimension": (
            category_profile.get("uncertainty_countsketch_dimension")
            if category_profile is not None
            else None
        ),
        "block_count": (
            category_profile.get("uncertainty_block_count")
            if category_profile is not None
            else None
        ),
        "model_outcomes_used": False,
    }
    condition_robustness_screen = {
        "mode": (
            "condition_mean_countsketch_direction_and_relative_norm_coverage"
            if category_profile is not None and "condition_sums" in category_profile
            else "unavailable_legacy_archive"
        ),
        "required": require_category_condition_robustness,
        "minimum_coverage": minimum_category_condition_coverage,
        "minimum_direction_cosine": minimum_condition_direction_cosine,
        "minimum_relative_norm": minimum_condition_relative_norm,
        "failed_specific_routes_to": "shared_cross_category",
        "model_outcomes_used": False,
        "minimum_countsketch_dimension": minimum_category_robustness_sketch_dimension,
        "observed_countsketch_dimension": (
            category_profile.get("uncertainty_countsketch_dimension")
            if category_profile is not None
            else None
        ),
    }
    template_robustness_screen = {
        "mode": (
            "template_group_mean_countsketch_direction_and_relative_norm_coverage"
            if category_profile is not None and "template_cell_sums" in category_profile
            else "unavailable_legacy_archive"
        ),
        "required": require_category_template_robustness,
        "minimum_coverage": minimum_category_template_coverage,
        "minimum_direction_cosine": minimum_template_direction_cosine,
        "minimum_relative_norm": minimum_template_relative_norm,
        "failed_specific_routes_to": "shared_cross_category",
        "model_outcomes_used": False,
        "minimum_countsketch_dimension": minimum_category_robustness_sketch_dimension,
        "observed_countsketch_dimension": (
            category_profile.get("uncertainty_countsketch_dimension")
            if category_profile is not None
            else None
        ),
    }
    discovery_stratified_robustness = (
        None
        if category_profile is None
        else category_profile_stratified_robustness(
            category_profile,
            minimum_pair_block_coverage=minimum_category_pair_block_direction_coverage,
            minimum_pair_block_direction_cosine=minimum_pair_block_direction_cosine,
            minimum_pair_block_relative_norm=minimum_pair_block_relative_norm,
            require_pair_block=require_category_pair_block_direction_robustness,
            minimum_condition_coverage=minimum_category_condition_coverage,
            minimum_condition_direction_cosine=minimum_condition_direction_cosine,
            minimum_condition_relative_norm=minimum_condition_relative_norm,
            require_condition=require_category_condition_robustness,
            minimum_template_coverage=minimum_category_template_coverage,
            minimum_template_direction_cosine=minimum_template_direction_cosine,
            minimum_template_relative_norm=minimum_template_relative_norm,
            require_template=require_category_template_robustness,
            minimum_sketch_dimension=minimum_category_robustness_sketch_dimension,
        )
    )
    calibration_profile: dict[str, Any] | None = None
    calibration_stratified_robustness: dict[str, Any] | None = None
    calibration_direction_cosines: np.ndarray | None = None
    calibration_strength_ratios: np.ndarray | None = None
    cross_archive_stratified_eligibility: np.ndarray | None = None
    split_provenance: dict[str, Any] | None = None
    semantic_template_hashes_by_role: dict[str, str] | None = None
    if calibration_effects_path is not None:
        if not category_compartments:
            raise ValueError(
                "activation-category calibration cannot be used when compartments are disabled"
            )
        if effects_path is None or category_profile is None:
            raise ValueError(
                "activation-category calibration requires discovery category effects"
            )
        discovery_sha256 = sha256_file(effects_path)
        calibration_sha256 = sha256_file(calibration_effects_path)
        if discovery_sha256 == calibration_sha256:
            raise ValueError("discovery and calibration effect archives must be distinct")
        merge_manifest_paths = (
            effects_merge_manifest_path,
            calibration_effects_merge_manifest_path,
        )
        if require_calibration_split_provenance and any(
            path is None for path in merge_manifest_paths
        ):
            raise ValueError(
                "calibrated clustering requires discovery and calibration effect merge manifests"
            )
        if any(path is not None for path in merge_manifest_paths) and any(
            path is None for path in merge_manifest_paths
        ):
            raise ValueError("effect merge provenance must be supplied for both split roles")
        if all(path is not None for path in merge_manifest_paths):
            assert effects_merge_manifest_path is not None
            assert calibration_effects_merge_manifest_path is not None
            discovery_provenance = load_effect_merge_split_provenance(
                effects_path,
                effects_merge_manifest_path,
                expected_role="discovery",
            )
            calibration_provenance = load_effect_merge_split_provenance(
                calibration_effects_path,
                calibration_effects_merge_manifest_path,
                expected_role="calibration",
            )
            identity_keys = (
                "activation_category_split_id",
                "activation_category_split_manifest_sha256",
                "activation_category_split_schema",
                "activation_category_split_semantic_disjointness",
            )
            if any(
                discovery_provenance[key] != calibration_provenance[key]
                for key in identity_keys
            ):
                raise ValueError("effect archives come from different split contracts")
            for set_key in (
                "activation_category_split_episode_id_set_sha256",
                "activation_category_split_pair_id_set_sha256",
            ):
                if discovery_provenance[set_key] == calibration_provenance[set_key]:
                    raise ValueError("discovery and calibration split identities are not distinct")
            if discovery_provenance["activation_category_split_schema"] in {
                "sch.vpd-activation-category-split-manifest.v2",
                "sch.vpd-activation-category-split-manifest.v3",
            }:
                template_key = (
                    "activation_category_split_template_group_set_sha256"
                )
                if discovery_provenance.get(template_key) == calibration_provenance.get(
                    template_key
                ):
                    raise ValueError(
                        "discovery and calibration template-group sets are not distinct"
                    )
                semantic_template_hashes_by_role = {
                    "discovery": str(discovery_provenance[template_key]),
                    "calibration": str(calibration_provenance[template_key]),
                }
            split_provenance = {
                "schema": "sch.vpd-effect-split-provenance.v1",
                "status": "verified",
                "activation_category_split_id": discovery_provenance[
                    "activation_category_split_id"
                ],
                "activation_category_split_manifest_sha256": discovery_provenance[
                    "activation_category_split_manifest_sha256"
                ],
                "discovery": discovery_provenance,
                "calibration": calibration_provenance,
            }
        _, calibration_profile = load_effects(
            calibration_effects_path,
            parsed,
            seed=seed,
            projection_dim=projection_dim,
        )
        if calibration_profile is None:
            raise ValueError("calibration effects omit activation-category payload")
        if calibration_profile["category_names"] != category_profile["category_names"]:
            raise ValueError("discovery and calibration category identities differ")
        calibration_stratified_robustness = category_profile_stratified_robustness(
            calibration_profile,
            minimum_pair_block_coverage=minimum_category_pair_block_direction_coverage,
            minimum_pair_block_direction_cosine=minimum_pair_block_direction_cosine,
            minimum_pair_block_relative_norm=minimum_pair_block_relative_norm,
            require_pair_block=require_category_pair_block_direction_robustness,
            minimum_condition_coverage=minimum_category_condition_coverage,
            minimum_condition_direction_cosine=minimum_condition_direction_cosine,
            minimum_condition_relative_norm=minimum_condition_relative_norm,
            require_condition=require_category_condition_robustness,
            minimum_template_coverage=minimum_category_template_coverage,
            minimum_template_direction_cosine=minimum_template_direction_cosine,
            minimum_template_relative_norm=minimum_template_relative_norm,
            require_template=require_category_template_robustness,
            minimum_sketch_dimension=minimum_category_robustness_sketch_dimension,
        )
        if minimum_category_calibration_episodes < 1:
            raise ValueError("minimum category calibration episodes must be positive")
        if min(category_profile["category_episode_counts"]) < minimum_category_calibration_episodes:
            raise ValueError("discovery category effects have insufficient episode support")
        if min(calibration_profile["category_episode_counts"]) < minimum_category_calibration_episodes:
            raise ValueError("calibration category effects have insufficient episode support")
        calibration_direction_cosines = cross_archive_category_direction_cosines(
            category_profile["category_effect_vectors"],
            calibration_profile["category_effect_vectors"],
        )
        discovery_strengths = np.asarray(category_profile["strengths"], dtype=np.float64)
        calibration_strengths = np.asarray(calibration_profile["strengths"], dtype=np.float64)
        maximum_strengths = np.maximum(discovery_strengths, calibration_strengths)
        calibration_strength_ratios = np.zeros_like(maximum_strengths)
        supported_strengths = maximum_strengths > 1e-12
        calibration_strength_ratios[supported_strengths] = (
            np.minimum(discovery_strengths, calibration_strengths)[supported_strengths]
            / maximum_strengths[supported_strengths]
        )
        cross_archive_stratified_eligibility = (
            discovery_stratified_robustness["eligible"]
            & calibration_stratified_robustness["eligible"]
        )
        calibration_result = calibrate_activation_compartment_thresholds(
            category_profile["strengths"],
            calibration_profile["strengths"],
            category_profile["category_names"],
            direction_cosines=calibration_direction_cosines,
            minimum_replication_rate=minimum_category_replication_rate,
            minimum_stable_specific_components=minimum_stable_specific_components,
            minimum_direction_cosine=minimum_category_direction_cosine,
            minimum_strength_ratio=minimum_category_strength_ratio,
            robustness_eligibility=cross_archive_stratified_eligibility,
        )
        category_specificity_thresholds = calibration_result["selection"][
            "selected_thresholds"
        ]
        calibration_body = {
            "schema": "sch.vpd-activation-compartment-calibration.v1",
            "mode": "independent_effect_archive_stability_selection",
            "status": (
                "pass" if split_provenance is not None else "pass_unattested_pilot"
            ),
            "sources": {
                "discovery_effects_sha256": discovery_sha256,
                "calibration_effects_sha256": calibration_sha256,
            },
            "category_names": category_profile["category_names"],
            "discovery_category_episode_counts": category_profile[
                "category_episode_counts"
            ],
            "calibration_category_episode_counts": calibration_profile[
                "category_episode_counts"
            ],
            "source_requirements": {
                "minimum_episodes_per_category_per_archive": (
                    minimum_category_calibration_episodes
                ),
                "distinct_content_hashes_required": True,
                "same_component_and_category_identity_required": True,
                "independent_prompt_split_required_upstream": True,
                "minimum_dominant_category_strength_ratio": (
                    minimum_category_strength_ratio
                ),
                "same_stratified_robustness_gates_required_in_both_archives": True,
                "verified_split_provenance_required_for_empirical_claim": True,
            },
            "split_provenance": split_provenance,
            **calibration_result,
        }
        category_calibration = {
            **calibration_body,
            "calibration_id": content_sha256(calibration_body),
        }
    if require_category_cluster_cross_archive_stability and (
        cluster_validation_effects_path is None
    ):
        raise ValueError(
            "category cluster cross-archive stability requires an independent validation archive"
        )
    if (
        cluster_validation_effects_path is not None
        and not require_category_cluster_cross_archive_stability
    ):
        raise ValueError(
            "cluster validation effects require the cross-archive stability gate"
        )
    if cluster_validation_effects_path is not None:
        if not category_compartments or effects_path is None or category_profile is None:
            raise ValueError(
                "category cluster validation requires discovery category effects"
            )
        validation_sha256 = sha256_file(cluster_validation_effects_path)
        discovery_sha256 = sha256_file(effects_path)
        if validation_sha256 == discovery_sha256 or (
            calibration_effects_path is not None
            and validation_sha256 == sha256_file(calibration_effects_path)
        ):
            raise ValueError(
                "cluster validation effects must be distinct from discovery and calibration"
            )
        validation_manifest_paths = (
            effects_merge_manifest_path,
            cluster_validation_effects_merge_manifest_path,
        )
        if require_cluster_validation_split_provenance and any(
            path is None for path in validation_manifest_paths
        ):
            raise ValueError(
                "cluster validation requires discovery and validation effect merge manifests"
            )
        if any(path is not None for path in validation_manifest_paths) and any(
            path is None for path in validation_manifest_paths
        ):
            raise ValueError(
                "cluster validation effect merge provenance must be supplied for both roles"
            )
        if all(path is not None for path in validation_manifest_paths):
            assert effects_merge_manifest_path is not None
            assert cluster_validation_effects_merge_manifest_path is not None
            validation_discovery_provenance = load_effect_merge_split_provenance(
                effects_path,
                effects_merge_manifest_path,
                expected_role="discovery",
            )
            validation_role_provenance = load_effect_merge_split_provenance(
                cluster_validation_effects_path,
                cluster_validation_effects_merge_manifest_path,
                expected_role="cluster_validation",
            )
            identity_keys = (
                "activation_category_split_id",
                "activation_category_split_manifest_sha256",
                "activation_category_split_schema",
                "activation_category_split_semantic_disjointness",
            )
            if any(
                validation_discovery_provenance[key]
                != validation_role_provenance[key]
                for key in identity_keys
            ):
                raise ValueError(
                    "cluster validation effects come from a different split contract"
                )
            for set_key in (
                "activation_category_split_episode_id_set_sha256",
                "activation_category_split_pair_id_set_sha256",
            ):
                if validation_discovery_provenance[set_key] == (
                    validation_role_provenance[set_key]
                ):
                    raise ValueError(
                        "discovery and cluster validation split identities are not distinct"
                    )
            if validation_discovery_provenance[
                "activation_category_split_schema"
            ] in {
                "sch.vpd-activation-category-split-manifest.v2",
                "sch.vpd-activation-category-split-manifest.v3",
            }:
                template_key = "activation_category_split_template_group_set_sha256"
                if validation_discovery_provenance.get(template_key) == (
                    validation_role_provenance.get(template_key)
                ):
                    raise ValueError(
                        "discovery and cluster validation template-group sets are not distinct"
                    )
                if semantic_template_hashes_by_role is None:
                    semantic_template_hashes_by_role = {
                        "discovery": str(
                            validation_discovery_provenance[template_key]
                        )
                    }
                semantic_template_hashes_by_role["cluster_validation"] = str(
                    validation_role_provenance[template_key]
                )
            cluster_validation_provenance = {
                "schema": "sch.vpd-effect-split-provenance.v1",
                "status": "verified",
                "activation_category_split_id": validation_discovery_provenance[
                    "activation_category_split_id"
                ],
                "activation_category_split_manifest_sha256": (
                    validation_discovery_provenance[
                        "activation_category_split_manifest_sha256"
                    ]
                ),
                "discovery": validation_discovery_provenance,
                "cluster_validation": validation_role_provenance,
            }
            if split_provenance is not None:
                if split_provenance["activation_category_split_id"] != (
                    cluster_validation_provenance["activation_category_split_id"]
                ):
                    raise ValueError(
                        "calibration and cluster validation use different split contracts"
                    )
                calibration_role = split_provenance["calibration"]
                for set_key in (
                    "activation_category_split_episode_id_set_sha256",
                    "activation_category_split_pair_id_set_sha256",
                ):
                    if calibration_role[set_key] == validation_role_provenance[set_key]:
                        raise ValueError(
                            "calibration and cluster validation split identities are not distinct"
                        )
                template_key = "activation_category_split_template_group_set_sha256"
                if template_key in calibration_role and calibration_role.get(
                    template_key
                ) == validation_role_provenance.get(template_key):
                    raise ValueError(
                        "calibration and cluster validation template-group sets are not distinct"
                    )
        (
            cluster_validation_matrix,
            validation_feature_names,
            validation_feature_mode,
            cluster_validation_profile,
        ) = feature_matrix(
            parsed,
            effects_path=cluster_validation_effects_path,
            seed=seed,
            projection_dim=projection_dim,
            metadata_weight=metadata_weight,
        )
        if (
            cluster_validation_profile is None
            or cluster_validation_profile["category_names"]
            != category_profile["category_names"]
            or validation_feature_mode != feature_mode
            or validation_feature_names != feature_names
        ):
            raise ValueError(
                "discovery and cluster validation feature contracts differ"
            )
        if min(cluster_validation_profile["category_episode_counts"]) < (
            minimum_category_calibration_episodes
        ):
            raise ValueError(
                "cluster validation category effects have insufficient episode support"
            )
    effective_split_provenance = split_provenance or cluster_validation_provenance
    if not 0.0 < minimum_category_jackknife_stability <= 1.0:
        raise ValueError("minimum category jackknife stability must be in (0, 1]")
    if require_category_uncertainty and (
        category_profile is None or "uncertainty_block_sums" not in category_profile
    ):
        raise ValueError("category uncertainty evidence is required but absent")
    if not 0.0 < minimum_category_pair_block_direction_coverage <= 1.0:
        raise ValueError("minimum category pair-block direction coverage must be in (0, 1]")
    if not 0.0 <= minimum_pair_block_direction_cosine <= 1.0:
        raise ValueError("minimum pair-block direction cosine must be in [0, 1]")
    if not 0.0 <= minimum_pair_block_relative_norm <= 1.0:
        raise ValueError("minimum pair-block relative norm must be in [0, 1]")
    if require_category_pair_block_direction_robustness and (
        category_profile is None or "uncertainty_block_sums" not in category_profile
    ):
        raise ValueError("category pair-block direction evidence is required but absent")
    if not 0.0 < minimum_category_condition_coverage <= 1.0:
        raise ValueError("minimum category condition coverage must be in (0, 1]")
    if not 0.0 <= minimum_condition_direction_cosine <= 1.0:
        raise ValueError("minimum condition direction cosine must be in [0, 1]")
    if not 0.0 <= minimum_condition_relative_norm <= 1.0:
        raise ValueError("minimum condition relative norm must be in [0, 1]")
    if require_category_condition_robustness and (
        category_profile is None or "condition_sums" not in category_profile
    ):
        raise ValueError("category condition robustness evidence is required but absent")
    if minimum_category_robustness_sketch_dimension < 2:
        raise ValueError("minimum category robustness CountSketch dimension is invalid")
    if (
        require_category_pair_block_direction_robustness
        or require_category_condition_robustness
        or require_category_template_robustness
    ) and category_profile is not None and int(
        category_profile.get("uncertainty_countsketch_dimension", 0)
    ) < minimum_category_robustness_sketch_dimension:
        raise ValueError("category robustness CountSketch dimension is underpowered")
    if not 0.0 < minimum_category_template_coverage <= 1.0:
        raise ValueError("minimum category template coverage must be in (0, 1]")
    if not 0.0 <= minimum_template_direction_cosine <= 1.0:
        raise ValueError("minimum template direction cosine must be in [0, 1]")
    if not 0.0 <= minimum_template_relative_norm <= 1.0:
        raise ValueError("minimum template relative norm must be in [0, 1]")
    if require_category_template_robustness and (
        category_profile is None or "template_cell_sums" not in category_profile
    ):
        raise ValueError("category template robustness evidence is required but absent")
    if category_profile is not None:
        classification = classify_activation_compartments(
            category_profile["strengths"],
            category_profile["category_names"],
            minimum_dominant_share=float(
                category_specificity_thresholds["minimum_dominant_share"]
            ),
            minimum_dominance_margin=float(
                category_specificity_thresholds["minimum_dominance_margin"]
            ),
            maximum_normalized_entropy=float(
                category_specificity_thresholds["maximum_normalized_entropy"]
            ),
        )
        classification["point_estimate_compartment_categories"] = list(
            classification["compartment_categories"]
        )
        if "uncertainty_block_sums" in category_profile:
            replicate_strengths = leave_one_block_out_strengths(
                category_profile["uncertainty_block_sums"],
                category_profile["uncertainty_block_counts"],
            )
            replicate_labels: list[list[str]] = []
            for block_index in range(replicate_strengths.shape[1]):
                replicate = classify_activation_compartments(
                    replicate_strengths[:, block_index, :],
                    category_profile["category_names"],
                    minimum_dominant_share=float(
                        category_specificity_thresholds["minimum_dominant_share"]
                    ),
                    minimum_dominance_margin=float(
                        category_specificity_thresholds["minimum_dominance_margin"]
                    ),
                    maximum_normalized_entropy=float(
                        category_specificity_thresholds["maximum_normalized_entropy"]
                    ),
                )
                replicate_labels.append(replicate["compartment_categories"])
            labels_by_component = [
                [replicate_labels[block][component] for block in range(len(replicate_labels))]
                for component in range(len(parsed))
            ]
            stability = np.asarray(
                [
                    sum(label == classification["compartment_categories"][index] for label in labels)
                    / len(labels)
                    for index, labels in enumerate(labels_by_component)
                ],
                dtype=np.float64,
            )
            classification["jackknife_labels"] = labels_by_component
            classification["jackknife_stability"] = stability
            classification["compartment_categories"] = [
                point
                if point == "shared_cross_category"
                or stability[index] >= minimum_category_jackknife_stability
                else "shared_cross_category"
                for index, point in enumerate(
                    classification["point_estimate_compartment_categories"]
                )
            ]
        else:
            classification["jackknife_labels"] = None
            classification["jackknife_stability"] = None
        classification["post_uncertainty_compartment_categories"] = list(
            classification["compartment_categories"]
        )
        if "uncertainty_block_sums" in category_profile:
            block_direction = pair_block_direction_metrics(
                block_sums=category_profile["uncertainty_block_sums"],
                block_counts=category_profile["uncertainty_block_counts"],
                minimum_direction_cosine=minimum_pair_block_direction_cosine,
                minimum_relative_norm=minimum_pair_block_relative_norm,
            )
            classification["pair_block_direction_coverage"] = block_direction[
                "coverage"
            ]
            classification["pair_block_minimum_direction_cosine"] = block_direction[
                "minimum_direction_cosine"
            ]
            classification["pair_block_minimum_relative_norm"] = block_direction[
                "minimum_relative_norm"
            ]
            classification["compartment_categories"] = [
                label
                if label == "shared_cross_category"
                or block_direction["coverage"][
                    index, category_profile["category_names"].index(label)
                ]
                >= minimum_category_pair_block_direction_coverage
                else "shared_cross_category"
                for index, label in enumerate(
                    classification["post_uncertainty_compartment_categories"]
                )
            ]
        else:
            classification["pair_block_direction_coverage"] = None
            classification["pair_block_minimum_direction_cosine"] = None
            classification["pair_block_minimum_relative_norm"] = None
        classification["post_pair_block_direction_compartment_categories"] = list(
            classification["compartment_categories"]
        )
        if "condition_sums" in category_profile:
            robustness = condition_robustness_metrics(
                condition_sums=category_profile["condition_sums"],
                condition_counts=category_profile["condition_counts"],
                condition_categories=category_profile["condition_categories"],
                category_names=category_profile["category_names"],
                minimum_direction_cosine=minimum_condition_direction_cosine,
                minimum_relative_norm=minimum_condition_relative_norm,
            )
            classification["condition_coverage"] = robustness["coverage"]
            classification["condition_minimum_direction_cosine"] = robustness[
                "minimum_direction_cosine"
            ]
            classification["condition_minimum_relative_norm"] = robustness[
                "minimum_relative_norm"
            ]
            classification["compartment_categories"] = [
                label
                if label == "shared_cross_category"
                or robustness["coverage"][
                    index, category_profile["category_names"].index(label)
                ]
                >= minimum_category_condition_coverage
                else "shared_cross_category"
                for index, label in enumerate(
                    classification["post_pair_block_direction_compartment_categories"]
                )
            ]
        else:
            classification["condition_coverage"] = None
            classification["condition_minimum_direction_cosine"] = None
            classification["condition_minimum_relative_norm"] = None
        classification["post_condition_compartment_categories"] = list(
            classification["compartment_categories"]
        )
        if "template_cell_sums" in category_profile:
            template_robustness = template_robustness_metrics(
                cell_sums=category_profile["template_cell_sums"],
                cell_counts=category_profile["template_cell_counts"],
                cell_categories=category_profile["template_cell_categories"],
                category_names=category_profile["category_names"],
                minimum_direction_cosine=minimum_template_direction_cosine,
                minimum_relative_norm=minimum_template_relative_norm,
            )
            classification["template_coverage"] = template_robustness["coverage"]
            classification["template_minimum_direction_cosine"] = template_robustness[
                "minimum_direction_cosine"
            ]
            classification["template_minimum_relative_norm"] = template_robustness[
                "minimum_relative_norm"
            ]
            classification["compartment_categories"] = [
                label
                if label == "shared_cross_category"
                or template_robustness["coverage"][
                    index, category_profile["category_names"].index(label)
                ]
                >= minimum_category_template_coverage
                else "shared_cross_category"
                for index, label in enumerate(
                    classification["post_condition_compartment_categories"]
                )
            ]
        else:
            classification["template_coverage"] = None
            classification["template_minimum_direction_cosine"] = None
            classification["template_minimum_relative_norm"] = None
        classification["pre_cross_archive_compartment_categories"] = list(
            classification["compartment_categories"]
        )
        if calibration_profile is not None:
            if (
                calibration_direction_cosines is None
                or calibration_strength_ratios is None
                or cross_archive_stratified_eligibility is None
            ):
                raise ValueError(
                    "component cross-archive admission diagnostics are absent"
                )
            component_cross_archive_admission = apply_cross_archive_component_admission(
                classification["pre_cross_archive_compartment_categories"],
                calibration_profile["strengths"],
                category_profile["category_names"],
                thresholds=category_specificity_thresholds,
                direction_cosines=calibration_direction_cosines,
                strength_ratios=calibration_strength_ratios,
                minimum_direction_cosine=minimum_category_direction_cosine,
                minimum_strength_ratio=minimum_category_strength_ratio,
                robustness_eligibility=cross_archive_stratified_eligibility,
            )
            classification["calibration_point_estimate_compartment_categories"] = (
                component_cross_archive_admission[
                    "calibration_point_estimate_categories"
                ]
            )
            classification["cross_archive_specific_admitted"] = (
                component_cross_archive_admission["admitted_specific"]
            )
            classification["cross_archive_rejection_reasons"] = (
                component_cross_archive_admission["rejection_reasons"]
            )
            classification["compartment_categories"] = (
                component_cross_archive_admission["final_categories"]
            )
        else:
            classification["calibration_point_estimate_compartment_categories"] = None
            classification["cross_archive_specific_admitted"] = None
            classification["cross_archive_rejection_reasons"] = None
        category_profile.update(classification)
    component_cross_archive_screen = {
        "mode": (
            "same_category_direction_magnitude_and_stratified_robustness"
            if component_cross_archive_admission is not None
            else "unavailable_without_calibration_archive"
        ),
        "required": calibration_profile is not None,
        "minimum_direction_cosine": minimum_category_direction_cosine,
        "minimum_strength_ratio": minimum_category_strength_ratio,
        "unstable_specific_routes_to": "shared_cross_category",
        "admitted_specific_count": (
            component_cross_archive_admission["admitted_specific_count"]
            if component_cross_archive_admission is not None
            else None
        ),
        "rerouted_specific_count": (
            component_cross_archive_admission["rerouted_specific_count"]
            if component_cross_archive_admission is not None
            else None
        ),
        "model_outcomes_used": False,
    }
    effective_cluster_count = cluster_count
    if compartment_enabled:
        compartment_categories = category_profile["compartment_categories"]
        if select_category_cluster_counts_by_projection:
            if effects_path is None:
                raise ValueError("category cluster count selection requires effects")
            if category_cluster_projection_replicates < 2:
                raise ValueError(
                    "category cluster count selection requires multiple projections"
                )
            projection_seeds = [seed] + [
                int(
                    stable_hash(
                        seed,
                        "category-cluster-projection",
                        str(replicate_index),
                    )[:8],
                    16,
                )
                for replicate_index in range(1, category_cluster_projection_replicates)
            ]
            if len(projection_seeds) != len(set(projection_seeds)):
                raise ValueError("category cluster projection seeds are not unique")
            projection_matrices = [matrix]
            replicate_kinds = ["projection"]
            for projection_seed in projection_seeds[1:]:
                alternate_matrix, _, alternate_mode, alternate_profile = feature_matrix(
                    parsed,
                    effects_path=effects_path,
                    seed=projection_seed,
                    projection_dim=projection_dim,
                    metadata_weight=metadata_weight,
                )
                if alternate_mode != feature_mode or alternate_profile is None or (
                    alternate_profile["category_names"]
                    != category_profile["category_names"]
                ):
                    raise ValueError("category cluster projection feature contract differs")
                projection_matrices.append(alternate_matrix)
                replicate_kinds.append("projection")
            if require_category_cluster_cross_archive_stability:
                if (
                    cluster_validation_effects_path is None
                    or cluster_validation_matrix is None
                    or cluster_validation_profile is None
                ):
                    raise ValueError(
                        "category cluster cross-archive stability evidence is absent"
                    )
                for projection_seed in projection_seeds:
                    validation_matrix, _, validation_mode, validation_profile = (
                        feature_matrix(
                            parsed,
                            effects_path=cluster_validation_effects_path,
                            seed=projection_seed,
                            projection_dim=projection_dim,
                            metadata_weight=metadata_weight,
                        )
                    )
                    if (
                        validation_mode != feature_mode
                        or validation_profile is None
                        or validation_profile["category_names"]
                        != category_profile["category_names"]
                    ):
                        raise ValueError(
                            "category cluster validation projection contract differs"
                        )
                    projection_matrices.append(validation_matrix)
                    replicate_kinds.append(
                        "independent_cluster_validation_projection"
                    )
            pair_block_omissions: list[int] = []
            validation_pair_block_omissions: list[int] = []
            if require_category_cluster_sampling_stability:
                if "uncertainty_block_sums" not in category_profile:
                    raise ValueError(
                        "category cluster sampling stability requires pair-block evidence"
                    )
                omitted_means, omitted_counts = leave_one_block_out_means(
                    category_profile["uncertainty_block_sums"],
                    category_profile["uncertainty_block_counts"],
                )
                for block_index in range(omitted_means.shape[1]):
                    projection_matrices.append(
                        category_profile_feature_matrix(
                            omitted_means[:, block_index, :, :],
                            omitted_counts[block_index],
                            parsed,
                            seed=seed,
                            projection_dim=projection_dim,
                            metadata_weight=metadata_weight,
                        )
                    )
                    replicate_kinds.append("leave_one_pair_block_out")
                    pair_block_omissions.append(block_index)
                if require_category_cluster_cross_archive_stability:
                    assert cluster_validation_profile is not None
                    if "uncertainty_block_sums" not in cluster_validation_profile:
                        raise ValueError(
                            "category cluster validation sampling stability requires pair-block evidence"
                        )
                    validation_omitted_means, validation_omitted_counts = (
                        leave_one_block_out_means(
                            cluster_validation_profile["uncertainty_block_sums"],
                            cluster_validation_profile["uncertainty_block_counts"],
                        )
                    )
                    for block_index in range(validation_omitted_means.shape[1]):
                        projection_matrices.append(
                            category_profile_feature_matrix(
                                validation_omitted_means[:, block_index, :, :],
                                validation_omitted_counts[block_index],
                                parsed,
                                seed=seed,
                                projection_dim=projection_dim,
                                metadata_weight=metadata_weight,
                            )
                        )
                        replicate_kinds.append(
                            "independent_cluster_validation_leave_one_pair_block_out"
                        )
                        validation_pair_block_omissions.append(block_index)
            if category_cluster_count_selector_version == "v2":
                # Imported lazily because the V2 selector reuses deterministic
                # clustering primitives from this module.
                from experiments.authz_mechinterp.activation_category_cluster_count_selection_v2 import (
                    select_category_cluster_counts_v2,
                )

                selection = select_category_cluster_counts_v2(
                    projection_matrices,
                    component_ids,
                    compartment_categories,
                    seed=seed,
                    replicate_kinds=replicate_kinds,
                    maximum_clusters_per_category=maximum_clusters_per_category,
                    minimum_replicate_stability=(
                        minimum_category_cluster_projection_stability
                    ),
                    minimum_mean_silhouette=minimum_category_cluster_silhouette,
                    minimum_silhouette_p10=minimum_category_cluster_silhouette_p10,
                    minimum_boundary_gap_ratio=(
                        minimum_category_cluster_boundary_gap_ratio
                    ),
                    minimum_cluster_size=minimum_category_cluster_size,
                    max_iterations=max_iterations,
                )
            else:
                selection = select_category_cluster_counts(
                    projection_matrices,
                    component_ids,
                    compartment_categories,
                    seed=seed,
                    replicate_kinds=replicate_kinds,
                    maximum_clusters_per_category=maximum_clusters_per_category,
                    minimum_replicate_stability=(
                        minimum_category_cluster_projection_stability
                    ),
                    minimum_silhouette=minimum_category_cluster_silhouette,
                    minimum_cluster_size=minimum_category_cluster_size,
                    max_iterations=max_iterations,
                )
            category_allocation = selection["allocation"]
            effective_cluster_count = sum(category_allocation.values())
            labels, centers, iterations = (
                deterministic_category_kmeans_with_allocation(
                    matrix,
                    component_ids,
                    compartment_categories,
                    allocation=category_allocation,
                    seed=seed,
                    max_iterations=max_iterations,
                )
            )
            selection_body = {
                "schema": (
                    "sch.vpd-category-cluster-count-selection-receipt.v2"
                    if category_cluster_count_selector_version == "v2"
                    else "sch.vpd-category-cluster-count-selection-receipt.v1"
                ),
                "mode": (
                    "cross_archive_projection_and_pair_block_ensemble_stability_and_silhouette"
                    if require_category_cluster_cross_archive_stability
                    and require_category_cluster_sampling_stability
                    else (
                        "cross_archive_projection_ensemble_stability_and_silhouette"
                        if require_category_cluster_cross_archive_stability
                        else (
                            "projection_and_pair_block_ensemble_stability_and_silhouette"
                            if require_category_cluster_sampling_stability
                            else "projection_ensemble_stability_and_silhouette"
                        )
                    )
                ) + (
                    "_lower_tail_and_boundary_gap_v2"
                    if category_cluster_count_selector_version == "v2"
                    else ""
                ),
                "status": "pass",
                "source": {
                    "effects_sha256": sha256_file(effects_path),
                    "cluster_validation_effects_sha256": (
                        sha256_file(cluster_validation_effects_path)
                        if cluster_validation_effects_path is not None
                        else None
                    ),
                    "cluster_validation_effects_merge_manifest_sha256": (
                        sha256_file(cluster_validation_effects_merge_manifest_path)
                        if cluster_validation_effects_merge_manifest_path is not None
                        else None
                    ),
                    "component_id_set_sha256": content_sha256(
                        sorted(component_ids)
                    ),
                },
                "projection_seeds": projection_seeds,
                "pair_block_omissions": pair_block_omissions,
                "cluster_validation_pair_block_omissions": (
                    validation_pair_block_omissions
                ),
                "cluster_validation_split_provenance": (
                    cluster_validation_provenance
                ),
                "selection": selection,
            }
        else:
            labels, centers, iterations, category_allocation = (
                deterministic_category_kmeans(
                    matrix,
                    component_ids,
                    compartment_categories,
                    cluster_count=cluster_count,
                    seed=seed,
                    max_iterations=max_iterations,
                )
            )
            selection_body = {
                "schema": "sch.vpd-category-cluster-count-selection-receipt.v1",
                "mode": "manual_total_cluster_count_allocation",
                "status": "unvalidated_manual",
                "source": {
                    "effects_sha256": sha256_file(effects_path),
                    "component_id_set_sha256": content_sha256(
                        sorted(component_ids)
                    ),
                },
                "requested_total_cluster_count": cluster_count,
                "allocation": category_allocation,
            }
        category_cluster_count_selection = {
            **selection_body,
            "selection_id": content_sha256(selection_body),
        }
    else:
        if select_category_cluster_counts_by_projection:
            raise ValueError(
                "category cluster count selection requires category compartments"
            )
        if require_category_cluster_cross_archive_stability:
            raise ValueError(
                "category cluster cross-archive stability requires automatic count selection"
            )
        labels, centers, iterations = deterministic_kmeans(
            matrix,
            component_ids,
            cluster_count=cluster_count,
            seed=seed,
            max_iterations=max_iterations,
        )
        category_allocation = None
    output_dir.mkdir(parents=True, exist_ok=True)
    packet_dir = output_dir / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    assignment_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    packet_entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    clusters: list[tuple[str, int, list[int]]] = []
    for internal_index in range(effective_cluster_count):
        indices = np.where(labels == internal_index)[0].tolist()
        ids = sorted(component_ids[index] for index in indices)
        cluster_hash = content_sha256(ids)
        clusters.append((f"vpd-cluster-{cluster_hash[:16]}", internal_index, indices))
    for cluster_id, internal_index, indices in sorted(clusters):
        cluster_rows_source = [parsed[index] for index in indices]
        cluster_matrix = matrix[indices]
        component_set = sorted(str(row["component_id"]) for row in cluster_rows_source)
        component_set_sha256 = content_sha256(component_set)
        summary, representatives = summarize_cluster(
            cluster_rows_source, cluster_matrix, centers[internal_index]
        )
        cluster_categories: list[str] = []
        if category_profile is not None:
            cluster_categories = [
                category_profile["compartment_categories"][index] for index in indices
            ]
            summary["activation_categories"] = dict(
                sorted(Counter(cluster_categories).items())
            )
            if compartment_enabled and len(set(cluster_categories)) != 1:
                raise ValueError("activation-category compartment is not homogeneous")
            if len(set(cluster_categories)) == 1:
                summary["activation_category"] = cluster_categories[0]
        activation_keys = [
            f"layer_{layer}__{anchor}"
            for layer in summary["capture_layers"]
            for anchor in anchors
        ]
        if effect_space_id not in activation_keys:
            activation_keys.append(effect_space_id)
        activation_keys = sorted(set(activation_keys))
        member_ids = set(component_set)
        cluster_by_family: dict[str, list[float]] = {}
        for row in cluster_rows_source:
            cluster_by_family.setdefault(str(row["projection_family"]), []).append(
                math.log1p(float(row["component_fro_norm"]))
            )

        def control_distance(row: dict[str, Any]) -> tuple[float, float, str]:
            family = str(row["projection_family"])
            value = math.log1p(float(row["component_fro_norm"]))
            references = cluster_by_family.get(family)
            family_penalty = 0.0 if references else 1.0
            if references is None:
                references = [
                    math.log1p(float(source["component_fro_norm"]))
                    for source in cluster_rows_source
                ]
            return (
                family_penalty,
                min(abs(value - reference) for reference in references),
                str(row["component_id"]),
            )

        matched_control_component_ids = [
            str(row["component_id"])
            for row in sorted(
                (row for row in parsed if str(row["component_id"]) not in member_ids),
                key=control_distance,
            )[:32]
        ]
        activation_category = summary.get("activation_category")
        category_control_component_ids: list[str] = []
        if activation_category is not None and category_profile is not None:
            category_control_component_ids = [
                str(parsed[index]["component_id"])
                for index, category in enumerate(
                    category_profile["compartment_categories"]
                )
                if category == activation_category
                and str(parsed[index]["component_id"]) not in member_ids
            ][:32]
        packet_body = {
            "schema": "sch.vpd-sae-study-packet.v1",
            "cluster_id": cluster_id,
            "cluster_component_set_sha256": component_set_sha256,
            "source": {
                "component_inventory_sha256": inventory_sha256,
                "source_kind": source_kind,
                "effects_sha256": sha256_file(effects_path) if effects_path else None,
                "calibration_effects_sha256": (
                    sha256_file(calibration_effects_path)
                    if calibration_effects_path is not None
                    else None
                ),
                "cluster_validation_effects_sha256": (
                    sha256_file(cluster_validation_effects_path)
                    if cluster_validation_effects_path is not None
                    else None
                ),
                "effects_merge_manifest_sha256": (
                    sha256_file(effects_merge_manifest_path)
                    if effects_merge_manifest_path is not None
                    else None
                ),
                "calibration_effects_merge_manifest_sha256": (
                    sha256_file(calibration_effects_merge_manifest_path)
                    if calibration_effects_merge_manifest_path is not None
                    else None
                ),
                "cluster_validation_effects_merge_manifest_sha256": (
                    sha256_file(cluster_validation_effects_merge_manifest_path)
                    if cluster_validation_effects_merge_manifest_path is not None
                    else None
                ),
                "activation_category_split_id": (
                    effective_split_provenance["activation_category_split_id"]
                    if effective_split_provenance is not None
                    else None
                ),
                "activation_category_split_schema": (
                    effective_split_provenance["discovery"][
                        "activation_category_split_schema"
                    ]
                    if effective_split_provenance is not None
                    else None
                ),
                "activation_category_template_group_set_sha256_by_role": (
                    semantic_template_hashes_by_role
                ),
                "feature_mode": feature_mode,
            },
            "component_ids": component_set,
            "representatives": representatives,
            "focus": {
                **summary,
                "recommended_activation_keys": activation_keys,
            },
            "sae_study": {
                "alignment_activation_key": effect_space_id,
                "alignment_capture_layer": alignment_layer,
                "alignment_rule": (
                    "SAE decoder directions and VPD delta_h must be measured at the exact "
                    "same activation key."
                ),
                "dictionary_scope": (
                    "fixed_base_shared_across_packets"
                    if activation_category is None
                    else (
                        "fixed_base_balanced_cross_category"
                        if activation_category == "shared_cross_category"
                        else "fixed_base_per_activation_category"
                    )
                ),
                "activation_compartment": (
                    None
                    if activation_category is None
                    else {
                        "category": activation_category,
                        "protocol": COMPONENT_COMPARTMENT_PROTOCOLS[
                            activation_category
                        ],
                        "dictionary_scope": (
                            "fixed_base_balanced_cross_category"
                            if activation_category == "shared_cross_category"
                            else "fixed_base_per_activation_category"
                        ),
                        "episode_filter": (
                            "all_categories_balanced"
                            if activation_category == "shared_cross_category"
                            else activation_category
                        ),
                        "category_matched_control_component_ids": category_control_component_ids,
                        "cross_category_evaluation_required": True,
                        "specificity_thresholds": category_specificity_thresholds,
                        "uncertainty_screen": {
                            "mode": (
                                "leave_one_pair_block_out_countsketch"
                                if "uncertainty_block_sums" in category_profile
                                else "unavailable_legacy_archive"
                            ),
                            "required": require_category_uncertainty,
                            "minimum_stability": minimum_category_jackknife_stability,
                            "block_count": category_profile.get(
                                "uncertainty_block_count"
                            ),
                            "countsketch_dimension": category_profile.get(
                                "uncertainty_countsketch_dimension"
                            ),
                            "low_stability_specific_routes_to_shared": True,
                        },
                        "pair_block_direction_robustness_screen": pair_block_direction_screen,
                        "condition_robustness_screen": condition_robustness_screen,
                        "template_robustness_screen": template_robustness_screen,
                        "per_component_cross_archive_admission": (
                            component_cross_archive_screen
                        ),
                        "threshold_calibration": (
                            None
                            if category_calibration is None
                            else {
                                "schema": category_calibration["schema"],
                                "status": category_calibration["status"],
                                "mode": category_calibration["mode"],
                                "calibration_id": category_calibration["calibration_id"],
                            }
                        ),
                        "cluster_count_selection": (
                            None
                            if category_cluster_count_selection is None
                            else {
                                "schema": category_cluster_count_selection["schema"],
                                "status": category_cluster_count_selection["status"],
                                "mode": category_cluster_count_selection["mode"],
                                "selection_id": category_cluster_count_selection[
                                    "selection_id"
                                ],
                            }
                        ),
                    }
                ),
                "matched_control_component_ids": matched_control_component_ids,
                "matched_control_set_sha256": content_sha256(
                    matched_control_component_ids
                ),
                "stages": [
                    "capture_fixed_base_activations",
                    "train_or_reuse_fixed_dictionary_sae",
                    "score_held_out_behavior_features",
                    "align_cluster_effects_to_sae_decoder",
                    "validate_with_signed_activation_and_parameter_interventions",
                ],
                "required_controls": [
                    "matched_norm_component_control",
                    "held_out_prompt_families",
                    "length_and_position_residualization",
                    "max_stat_permutation_screen",
                    "guardrail_endpoint",
                    "same_activation_space_alignment",
                    "cross_activation_category_holdout",
                ],
                "claim_boundary": (
                    "Cluster membership is a screening result; individual causality requires "
                    "held-out signed intervention agreement."
                ),
            },
            "effect_refinement": {
                "effect_space_id": effect_space_id,
                "component_set_rule": "exact_packet_membership",
                "output_schema": "component_ids plus float32 rank-2 delta_h",
                "receipt_schema": (
                    "sch.vpd-effect-shard-receipt.v5"
                    if category_profile is not None
                    and "template_cell_sums" in category_profile
                    else (
                    "sch.vpd-effect-shard-receipt.v4"
                    if category_profile is not None
                    and "condition_sums" in category_profile
                    else (
                    "sch.vpd-effect-shard-receipt.v3"
                    if category_profile is not None
                    and "uncertainty_block_sums" in category_profile
                    else (
                        "sch.vpd-effect-shard-receipt.v2"
                        if category_profile is not None
                        else "sch.vpd-effect-shard-receipt.v1"
                    )
                    )
                    )
                ),
                "required_for_effect_space_reclustering": effects_path is None,
            },
            "distribution": {
                "algorithm": "highest-sha256 rendezvous hashing",
                "eligible_workers": workers,
            },
        }
        packet_id = content_sha256(packet_body)
        packet = {
            **packet_body,
            "packet_id": packet_id,
            "distribution": {
                **packet_body["distribution"],
                "assigned_worker": rendezvous_worker(packet_id, workers),
            },
        }
        packet_path = packet_dir / f"{cluster_id}.json"
        write_json(packet_path, packet)
        packet_sha256 = sha256_file(packet_path)
        packet_entries.append(
            {
                "cluster_id": cluster_id,
                "packet_path": packet_path.relative_to(output_dir).as_posix(),
                "packet_sha256": packet_sha256,
                "packet_id": packet_id,
            }
        )
        distances = np.sqrt(
            np.sum((cluster_matrix - centers[internal_index]) ** 2, axis=1)
        )
        rank_order = sorted(
            range(len(indices)),
            key=lambda local: (
                float(distances[local]),
                str(cluster_rows_source[local]["component_id"]),
            ),
        )
        ranks = {local: rank + 1 for rank, local in enumerate(rank_order)}
        for local_index, source_index in enumerate(indices):
            component_id = component_ids[source_index]
            if component_id in seen:
                raise ValueError(f"component appears in multiple clusters: {component_id}")
            seen.add(component_id)
            assignment_rows.append(
                {
                    "schema": "sch.vpd-cluster-assignment.v1",
                    "component_id": component_id,
                    "cluster_id": cluster_id,
                    "distance_to_centroid": float(distances[local_index]),
                    "centroid_rank": ranks[local_index],
                    **(
                        {}
                        if category_profile is None
                        else {
                            "activation_category": category_profile[
                                "compartment_categories"
                            ][source_index],
                            "point_estimate_activation_category": category_profile[
                                "point_estimate_compartment_categories"
                            ][source_index],
                            "post_uncertainty_activation_category": category_profile[
                                "post_uncertainty_compartment_categories"
                            ][source_index],
                            "post_pair_block_direction_activation_category": category_profile[
                                "post_pair_block_direction_compartment_categories"
                            ][source_index],
                            "post_condition_activation_category": category_profile[
                                "post_condition_compartment_categories"
                            ][source_index],
                            "pre_cross_archive_activation_category": category_profile[
                                "pre_cross_archive_compartment_categories"
                            ][source_index],
                            "calibration_point_estimate_activation_category": (
                                None
                                if category_profile[
                                    "calibration_point_estimate_compartment_categories"
                                ]
                                is None
                                else category_profile[
                                    "calibration_point_estimate_compartment_categories"
                                ][source_index]
                            ),
                            "cross_archive_activation_category_admitted": (
                                None
                                if category_profile[
                                    "cross_archive_specific_admitted"
                                ]
                                is None
                                else bool(
                                    category_profile[
                                        "cross_archive_specific_admitted"
                                    ][source_index]
                                )
                            ),
                            "cross_archive_activation_category_rejection_reason": (
                                None
                                if category_profile[
                                    "cross_archive_rejection_reasons"
                                ]
                                is None
                                else category_profile[
                                    "cross_archive_rejection_reasons"
                                ][source_index]
                            ),
                            "dominant_activation_category": category_profile[
                                "dominant_categories"
                            ][source_index],
                            "dominant_activation_category_share": float(
                                category_profile["dominant_shares"][source_index]
                            ),
                            "activation_category_dominance_margin": float(
                                category_profile["dominance_margins"][source_index]
                            ),
                            "activation_category_normalized_entropy": float(
                                category_profile["normalized_entropy"][source_index]
                            ),
                            "activation_category_strengths": {
                                name: float(
                                    category_profile["strengths"][source_index, cat_index]
                                )
                                for cat_index, name in enumerate(
                                    category_profile["category_names"]
                                )
                            },
                            "activation_category_shares": {
                                name: float(
                                    category_profile["shares"][source_index, cat_index]
                                )
                                for cat_index, name in enumerate(
                                    category_profile["category_names"]
                                )
                            },
                            "activation_category_jackknife_labels": (
                                None
                                if category_profile["jackknife_labels"] is None
                                else category_profile["jackknife_labels"][source_index]
                            ),
                            "activation_category_jackknife_stability": (
                                None
                                if category_profile["jackknife_stability"] is None
                                else float(
                                    category_profile["jackknife_stability"][source_index]
                                )
                            ),
                            "activation_category_pair_block_direction_coverage": (
                                None
                                if category_profile[
                                    "pair_block_direction_coverage"
                                ]
                                is None
                                else {
                                    name: float(
                                        category_profile[
                                            "pair_block_direction_coverage"
                                        ][source_index, cat_index]
                                    )
                                    for cat_index, name in enumerate(
                                        category_profile["category_names"]
                                    )
                                }
                            ),
                            "activation_category_pair_block_minimum_direction_cosine": (
                                None
                                if category_profile[
                                    "pair_block_minimum_direction_cosine"
                                ]
                                is None
                                else {
                                    name: float(
                                        category_profile[
                                            "pair_block_minimum_direction_cosine"
                                        ][source_index, cat_index]
                                    )
                                    for cat_index, name in enumerate(
                                        category_profile["category_names"]
                                    )
                                }
                            ),
                            "activation_category_pair_block_minimum_relative_norm": (
                                None
                                if category_profile[
                                    "pair_block_minimum_relative_norm"
                                ]
                                is None
                                else {
                                    name: float(
                                        category_profile[
                                            "pair_block_minimum_relative_norm"
                                        ][source_index, cat_index]
                                    )
                                    for cat_index, name in enumerate(
                                        category_profile["category_names"]
                                    )
                                }
                            ),
                            "activation_category_condition_coverage": (
                                None
                                if category_profile["condition_coverage"] is None
                                else {
                                    name: float(
                                        category_profile["condition_coverage"][
                                            source_index, cat_index
                                        ]
                                    )
                                    for cat_index, name in enumerate(
                                        category_profile["category_names"]
                                    )
                                }
                            ),
                            "activation_category_condition_minimum_direction_cosine": (
                                None
                                if category_profile[
                                    "condition_minimum_direction_cosine"
                                ]
                                is None
                                else {
                                    name: float(
                                        category_profile[
                                            "condition_minimum_direction_cosine"
                                        ][source_index, cat_index]
                                    )
                                    for cat_index, name in enumerate(
                                        category_profile["category_names"]
                                    )
                                }
                            ),
                            "activation_category_condition_minimum_relative_norm": (
                                None
                                if category_profile[
                                    "condition_minimum_relative_norm"
                                ]
                                is None
                                else {
                                    name: float(
                                        category_profile[
                                            "condition_minimum_relative_norm"
                                        ][source_index, cat_index]
                                    )
                                    for cat_index, name in enumerate(
                                        category_profile["category_names"]
                                    )
                                }
                            ),
                            "activation_category_template_coverage": (
                                None
                                if category_profile["template_coverage"] is None
                                else {
                                    name: float(
                                        category_profile["template_coverage"][
                                            source_index, cat_index
                                        ]
                                    )
                                    for cat_index, name in enumerate(
                                        category_profile["category_names"]
                                    )
                                }
                            ),
                            "activation_category_template_minimum_direction_cosine": (
                                None
                                if category_profile[
                                    "template_minimum_direction_cosine"
                                ]
                                is None
                                else {
                                    name: float(
                                        category_profile[
                                            "template_minimum_direction_cosine"
                                        ][source_index, cat_index]
                                    )
                                    for cat_index, name in enumerate(
                                        category_profile["category_names"]
                                    )
                                }
                            ),
                            "activation_category_template_minimum_relative_norm": (
                                None
                                if category_profile[
                                    "template_minimum_relative_norm"
                                ]
                                is None
                                else {
                                    name: float(
                                        category_profile[
                                            "template_minimum_relative_norm"
                                        ][source_index, cat_index]
                                    )
                                    for cat_index, name in enumerate(
                                        category_profile["category_names"]
                                    )
                                }
                            ),
                            **(
                                {}
                                if calibration_profile is None
                                else {
                                    "calibration_activation_category_strengths": {
                                        name: float(
                                            calibration_profile["strengths"][
                                                source_index, cat_index
                                            ]
                                        )
                                        for cat_index, name in enumerate(
                                            calibration_profile["category_names"]
                                        )
                                    },
                                    "cross_archive_activation_category_direction_cosines": {
                                        name: float(
                                            calibration_direction_cosines[
                                                source_index, cat_index
                                            ]
                                        )
                                        for cat_index, name in enumerate(
                                            calibration_profile["category_names"]
                                        )
                                    },
                                    "cross_archive_activation_category_strength_ratios": {
                                        name: float(
                                            calibration_strength_ratios[
                                                source_index, cat_index
                                            ]
                                        )
                                        for cat_index, name in enumerate(
                                            calibration_profile["category_names"]
                                        )
                                    },
                                    "calibration_activation_category_pair_block_direction_coverage": (
                                        None
                                        if calibration_stratified_robustness[
                                            "pair_block"
                                        ]
                                        is None
                                        else {
                                            name: float(
                                                calibration_stratified_robustness[
                                                    "pair_block"
                                                ]["coverage"][source_index, cat_index]
                                            )
                                            for cat_index, name in enumerate(
                                                calibration_profile["category_names"]
                                            )
                                        }
                                    ),
                                    "calibration_activation_category_condition_coverage": (
                                        None
                                        if calibration_stratified_robustness[
                                            "condition"
                                        ]
                                        is None
                                        else {
                                            name: float(
                                                calibration_stratified_robustness[
                                                    "condition"
                                                ]["coverage"][source_index, cat_index]
                                            )
                                            for cat_index, name in enumerate(
                                                calibration_profile["category_names"]
                                            )
                                        }
                                    ),
                                    "calibration_activation_category_template_coverage": (
                                        None
                                        if calibration_stratified_robustness[
                                            "template"
                                        ]
                                        is None
                                        else {
                                            name: float(
                                                calibration_stratified_robustness[
                                                    "template"
                                                ]["coverage"][source_index, cat_index]
                                            )
                                            for cat_index, name in enumerate(
                                                calibration_profile["category_names"]
                                            )
                                        }
                                    ),
                                    "cross_archive_activation_category_stratified_robustness_eligible": {
                                        name: bool(
                                            cross_archive_stratified_eligibility[
                                                source_index, cat_index
                                            ]
                                        )
                                        for cat_index, name in enumerate(
                                            calibration_profile["category_names"]
                                        )
                                    },
                                }
                            ),
                        }
                    ),
                }
            )
        cluster_rows.append(
            {
                "schema": "sch.vpd-cluster-summary.v1",
                "cluster_id": cluster_id,
                "cluster_component_set_sha256": component_set_sha256,
                **summary,
                "representative_component_ids": [
                    row["component_id"] for row in representatives
                ],
                "packet_path": packet_path.relative_to(output_dir).as_posix(),
                "packet_sha256": packet_sha256,
            }
        )
    if seen != set(component_ids):
        raise ValueError("cluster assignments do not cover the component inventory")
    assignments_path = output_dir / "assignments.jsonl"
    clusters_path = output_dir / "clusters.jsonl"
    write_jsonl(assignments_path, sorted(assignment_rows, key=lambda row: row["component_id"]))
    write_jsonl(clusters_path, sorted(cluster_rows, key=lambda row: row["cluster_id"]))
    manifest = {
        "schema": "sch.vpd-sae-cluster-manifest.v1",
        "status": "complete",
        "algorithm": {
            "name": "deterministic_farthest_seed_kmeans",
            "seed": seed,
            "cluster_count": effective_cluster_count,
            "requested_manual_cluster_count": (
                cluster_count
                if not select_category_cluster_counts_by_projection
                else None
            ),
            "iterations": iterations,
            "feature_mode": feature_mode,
            "feature_names": feature_names,
            "effect_projection_dim": projection_dim if effects_path else None,
            "metadata_weight": metadata_weight,
            "effect_space_id": effect_space_id,
            "activation_category_compartments": {
                "enabled": compartment_enabled,
                "contract_sha256": (
                    content_sha256(category_contract())
                    if category_profile is not None
                    else None
                ),
                "categories": (
                    category_profile["category_names"]
                    if category_profile is not None
                    else []
                ),
                "episode_counts": (
                    category_profile["category_episode_counts"]
                    if category_profile is not None
                    else []
                ),
                "cluster_allocation": category_allocation,
                "cluster_count_selection": category_cluster_count_selection,
                "observed_component_compartments": (
                    sorted(set(category_profile["compartment_categories"]))
                    if category_profile is not None
                    else []
                ),
                "specificity_thresholds": category_specificity_thresholds,
                "uncertainty_screen": {
                    "mode": (
                        "leave_one_pair_block_out_countsketch"
                        if category_profile is not None
                        and "uncertainty_block_sums" in category_profile
                        else "unavailable_legacy_archive"
                    ),
                    "required": require_category_uncertainty,
                    "minimum_stability": minimum_category_jackknife_stability,
                    "block_count": (
                        category_profile.get("uncertainty_block_count")
                        if category_profile is not None
                        else None
                    ),
                    "countsketch_dimension": (
                        category_profile.get("uncertainty_countsketch_dimension")
                        if category_profile is not None
                        else None
                    ),
                    "low_stability_specific_routes_to_shared": True,
                },
                "condition_robustness_screen": condition_robustness_screen,
                "pair_block_direction_robustness_screen": pair_block_direction_screen,
                "template_robustness_screen": template_robustness_screen,
                "per_component_cross_archive_admission": (
                    component_cross_archive_screen
                ),
                "threshold_calibration": category_calibration,
                "within_category_clustering_geometry": {
                    "mode": "share_weighted_direction_strength_profile_kmeans",
                    "direction_weighting": (
                        category_profile.get("direction_feature_weighting")
                        if category_profile is not None
                        else None
                    ),
                    "sum_squared_category_direction_weights_for_nonzero_effect_component": 1.0,
                    "arbitrarily_small_background_directions_have_arbitrarily_small_weight": True,
                    "metadata_weight": metadata_weight,
                },
            },
        },
        "source": {
            "component_inventory_sha256": inventory_sha256,
            "source_kind": source_kind,
            "effects_sha256": sha256_file(effects_path) if effects_path else None,
            "calibration_effects_sha256": (
                sha256_file(calibration_effects_path)
                if calibration_effects_path is not None
                else None
            ),
            "cluster_validation_effects_sha256": (
                sha256_file(cluster_validation_effects_path)
                if cluster_validation_effects_path is not None
                else None
            ),
            "effects_merge_manifest_sha256": (
                sha256_file(effects_merge_manifest_path)
                if effects_merge_manifest_path is not None
                else None
            ),
            "calibration_effects_merge_manifest_sha256": (
                sha256_file(calibration_effects_merge_manifest_path)
                if calibration_effects_merge_manifest_path is not None
                else None
            ),
            "cluster_validation_effects_merge_manifest_sha256": (
                sha256_file(cluster_validation_effects_merge_manifest_path)
                if cluster_validation_effects_merge_manifest_path is not None
                else None
            ),
            "activation_category_split_id": (
                effective_split_provenance["activation_category_split_id"]
                if effective_split_provenance is not None
                else None
            ),
            "activation_category_split_schema": (
                effective_split_provenance["discovery"][
                    "activation_category_split_schema"
                ]
                if effective_split_provenance is not None
                else None
            ),
            "activation_category_template_group_set_sha256_by_role": (
                semantic_template_hashes_by_role
            ),
        },
        "invariants": {
            "component_count": len(component_ids),
            "assigned_component_count": len(assignment_rows),
            "unique_assigned_component_count": len(seen),
            "exact_partition": True,
            "component_id_set_sha256": content_sha256(sorted(component_ids)),
        },
        "artifacts": {
            "assignments": {
                "path": assignments_path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(assignments_path),
            },
            "clusters": {
                "path": clusters_path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(clusters_path),
            },
            "packets": packet_entries,
        },
        "decentralized_execution": {
            "workers": workers,
            "assignment_algorithm": "highest-sha256 rendezvous hashing",
            "packet_content_addressed": True,
            "worker_results_must_reference_packet_id": True,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def verify_bundle(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    for key in ("assignments", "clusters"):
        entry = artifacts[key]
        path = output_dir / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{key} artifact hash mismatch")
    assignments = load_jsonl(output_dir / artifacts["assignments"]["path"])
    assignment_ids = [str(row["component_id"]) for row in assignments]
    assignment_by_id = {str(row["component_id"]): row for row in assignments}
    if len(assignment_ids) != len(set(assignment_ids)):
        raise ValueError("duplicate component assignment")
    packet_ids: list[str] = []
    packet_component_ids: list[str] = []
    packet_compartments: list[str] = []
    compartment_contract = manifest["algorithm"].get(
        "activation_category_compartments", {}
    )
    if bool(compartment_contract.get("enabled", False)):
        clustering_geometry = compartment_contract.get(
            "within_category_clustering_geometry", {}
        )
        if manifest["algorithm"].get("feature_mode") != (
            "share_weighted_activation_category_effects_plus_metadata"
        ) or clustering_geometry != {
            "mode": "share_weighted_direction_strength_profile_kmeans",
            "direction_weighting": ACTIVATION_CATEGORY_DIRECTION_WEIGHTING,
            "sum_squared_category_direction_weights_for_nonzero_effect_component": 1.0,
            "arbitrarily_small_background_directions_have_arbitrarily_small_weight": True,
            "metadata_weight": float(manifest["algorithm"]["metadata_weight"]),
        }:
            raise ValueError("activation category clustering geometry contract is invalid")
        cluster_count_receipt = compartment_contract.get("cluster_count_selection")
        if not isinstance(cluster_count_receipt, dict):
            raise ValueError("activation category cluster count receipt is absent")
        cluster_count_body = dict(cluster_count_receipt)
        cluster_count_selection_id = str(
            cluster_count_body.pop("selection_id", "")
        )
        if content_sha256(cluster_count_body) != cluster_count_selection_id:
            raise ValueError("activation category cluster count receipt address differs")
        cluster_allocation = compartment_contract.get("cluster_allocation")
        if not isinstance(cluster_allocation, dict) or sum(
            int(value) for value in cluster_allocation.values()
        ) != int(manifest["algorithm"]["cluster_count"]):
            raise ValueError("activation category cluster allocation total differs")
        if cluster_count_receipt.get("source", {}).get(
            "effects_sha256"
        ) != manifest["source"].get("effects_sha256") or cluster_count_receipt.get(
            "source", {}
        ).get("component_id_set_sha256") != manifest["invariants"].get(
            "component_id_set_sha256"
        ):
            raise ValueError("activation category cluster count sources differ")
        cluster_count_mode = str(cluster_count_receipt.get("mode"))
        cluster_count_schema = str(cluster_count_receipt.get("schema"))
        v2_mode_suffix = "_lower_tail_and_boundary_gap_v2"
        selector_v2 = cluster_count_mode.endswith(v2_mode_suffix)
        base_cluster_count_mode = (
            cluster_count_mode[: -len(v2_mode_suffix)]
            if selector_v2
            else cluster_count_mode
        )
        if cluster_count_mode == "manual_total_cluster_count_allocation":
            if cluster_count_schema != "sch.vpd-category-cluster-count-selection-receipt.v1":
                raise ValueError("manual activation category cluster count schema differs")
            if cluster_count_receipt.get("status") != "unvalidated_manual" or (
                cluster_count_receipt.get("allocation") != cluster_allocation
            ) or int(cluster_count_receipt.get("requested_total_cluster_count", 0)) != int(
                manifest["algorithm"].get("requested_manual_cluster_count", 0)
            ):
                raise ValueError("manual activation category cluster count differs")
        elif base_cluster_count_mode in {
            "projection_ensemble_stability_and_silhouette",
            "projection_and_pair_block_ensemble_stability_and_silhouette",
            "cross_archive_projection_ensemble_stability_and_silhouette",
            "cross_archive_projection_and_pair_block_ensemble_stability_and_silhouette",
        }:
            expected_receipt_schema = (
                "sch.vpd-category-cluster-count-selection-receipt.v2"
                if selector_v2
                else "sch.vpd-category-cluster-count-selection-receipt.v1"
            )
            expected_selection_schema = (
                "sch.vpd-category-cluster-count-selection.v2"
                if selector_v2
                else "sch.vpd-category-cluster-count-selection.v1"
            )
            if cluster_count_schema != expected_receipt_schema:
                raise ValueError("activation category cluster count receipt schema differs")
            if cluster_count_receipt.get("status") != "pass" or manifest[
                "algorithm"
            ].get("requested_manual_cluster_count") is not None:
                raise ValueError("selected activation category cluster count status differs")
            projection_seeds = cluster_count_receipt.get("projection_seeds")
            pair_block_omissions = cluster_count_receipt.get(
                "pair_block_omissions", []
            )
            validation_pair_block_omissions = cluster_count_receipt.get(
                "cluster_validation_pair_block_omissions", []
            )
            selection = cluster_count_receipt.get("selection")
            if not isinstance(projection_seeds, list) or len(projection_seeds) < 2 or len(
                projection_seeds
            ) != len(set(int(value) for value in projection_seeds)) or not isinstance(
                selection, dict
            ) or selection.get("allocation") != cluster_allocation:
                raise ValueError("activation category cluster count selection differs")
            if not isinstance(pair_block_omissions, list) or pair_block_omissions != list(
                range(len(pair_block_omissions))
            ):
                raise ValueError("activation category cluster pair-block omissions differ")
            if not isinstance(
                validation_pair_block_omissions, list
            ) or validation_pair_block_omissions != list(
                range(len(validation_pair_block_omissions))
            ):
                raise ValueError(
                    "activation category cluster validation pair-block omissions differ"
                )
            if selection.get("schema") != expected_selection_schema:
                raise ValueError("activation category cluster count selection schema differs")
            cross_archive_mode = base_cluster_count_mode.startswith("cross_archive_")
            pair_block_mode = "pair_block" in base_cluster_count_mode
            receipt_source = cluster_count_receipt.get("source", {})
            if cross_archive_mode:
                if receipt_source.get(
                    "cluster_validation_effects_sha256"
                ) != manifest["source"].get(
                    "cluster_validation_effects_sha256"
                ) or receipt_source.get(
                    "cluster_validation_effects_merge_manifest_sha256"
                ) != manifest["source"].get(
                    "cluster_validation_effects_merge_manifest_sha256"
                ) or receipt_source.get("cluster_validation_effects_sha256") is None:
                    raise ValueError(
                        "activation category cluster validation sources differ"
                    )
                validation_provenance = cluster_count_receipt.get(
                    "cluster_validation_split_provenance"
                )
                if validation_provenance is not None:
                    if validation_provenance.get("status") != "verified" or (
                        validation_provenance.get("activation_category_split_id")
                        != manifest["source"].get("activation_category_split_id")
                    ) or validation_provenance["discovery"].get(
                        "effect_merge_manifest_sha256"
                    ) != manifest["source"].get(
                        "effects_merge_manifest_sha256"
                    ) or validation_provenance["cluster_validation"].get(
                        "effect_merge_manifest_sha256"
                    ) != manifest["source"].get(
                        "cluster_validation_effects_merge_manifest_sha256"
                    ):
                        raise ValueError(
                            "activation category cluster validation provenance differs"
                        )
            elif receipt_source.get("cluster_validation_effects_sha256") is not None:
                raise ValueError(
                    "discovery-only cluster selection carries validation effects"
                )
            expected_kind_counts = {"projection": len(projection_seeds)}
            if cross_archive_mode:
                expected_kind_counts[
                    "independent_cluster_validation_projection"
                ] = len(projection_seeds)
            if pair_block_mode:
                if not pair_block_omissions:
                    raise ValueError(
                        "activation category cluster sampling stability is absent"
                    )
                expected_kind_counts["leave_one_pair_block_out"] = len(
                    pair_block_omissions
                )
                if cross_archive_mode:
                    if not validation_pair_block_omissions:
                        raise ValueError(
                            "activation category cluster validation sampling stability is absent"
                        )
                    expected_kind_counts[
                        "independent_cluster_validation_leave_one_pair_block_out"
                    ] = len(validation_pair_block_omissions)
                elif validation_pair_block_omissions:
                    raise ValueError(
                        "discovery-only cluster selection carries validation omissions"
                    )
            elif pair_block_omissions or validation_pair_block_omissions:
                raise ValueError(
                    "projection-only cluster selection carries pair-block omissions"
                )
            policy = selection.get("policy", {})
            if int(policy.get("replicates", 0)) != sum(
                expected_kind_counts.values()
            ) or policy.get("replicate_kind_counts") != dict(
                sorted(expected_kind_counts.items())
            ) or policy.get("no_passing_nontrivial_candidate_rule") != (
                "one_cluster_for_category"
            ):
                raise ValueError("activation category cluster count policy differs")
            category_selections = selection.get("categories")
            if not isinstance(category_selections, dict) or set(
                category_selections
            ) != set(cluster_allocation):
                raise ValueError("activation category cluster count categories differ")
            for category, category_result in category_selections.items():
                candidates = category_result.get("candidates")
                selected = category_result.get("selection")
                if not isinstance(candidates, list) or not isinstance(selected, dict) or (
                    selected not in candidates
                ) or int(selected.get("cluster_count", 0)) != int(
                    cluster_allocation[category]
                ):
                    raise ValueError("activation category cluster count candidate differs")
                candidate_counts = [
                    int(row.get("cluster_count", 0)) for row in candidates
                ]
                if candidate_counts != list(range(1, len(candidates) + 1)):
                    raise ValueError("activation category cluster count grid differs")
                for candidate in candidates:
                    candidate_count = int(candidate["cluster_count"])
                    if candidate_count == 1:
                        expected_status = "pass_fallback"
                    elif float(
                        candidate["minimum_replicate_adjusted_rand"]
                    ) < float(policy["minimum_replicate_adjusted_rand"]):
                        expected_status = "reject_replicate_instability"
                    elif selector_v2 and float(
                        candidate["minimum_replicate_mean_silhouette"]
                    ) < float(policy["minimum_replicate_mean_silhouette"]):
                        expected_status = "reject_weak_mean_separation"
                    elif selector_v2 and float(
                        candidate["minimum_replicate_silhouette_p10"]
                    ) < float(policy["minimum_replicate_silhouette_p10"]):
                        expected_status = "reject_weak_lower_tail_separation"
                    elif selector_v2 and float(
                        candidate["minimum_replicate_boundary_gap_ratio"]
                    ) < float(policy["minimum_replicate_boundary_gap_ratio"]):
                        expected_status = "reject_continuous_boundary"
                    elif not selector_v2 and float(
                        candidate["minimum_replicate_silhouette"]
                    ) < float(policy["minimum_replicate_silhouette"]):
                        expected_status = "reject_weak_separation"
                    elif int(candidate["minimum_observed_cluster_size"]) < int(
                        policy["minimum_cluster_size"]
                    ):
                        expected_status = "reject_small_cluster"
                    else:
                        expected_status = "pass"
                    if candidate.get("status") != expected_status:
                        raise ValueError(
                            "activation category cluster count candidate status differs"
                        )
                passing = [row for row in candidates if row.get("status") == "pass"]
                if passing:
                    if selector_v2:
                        expected_selected = max(
                            passing,
                            key=lambda row: (
                                float(row["minimum_replicate_boundary_gap_ratio"]),
                                float(row["minimum_replicate_silhouette_p10"]),
                                float(row["minimum_replicate_mean_silhouette"]),
                                float(row["minimum_replicate_adjusted_rand"]),
                                -int(row["cluster_count"]),
                            ),
                        )
                    else:
                        expected_selected = max(
                            passing,
                            key=lambda row: (
                                float(row["minimum_replicate_silhouette"]),
                                float(row["minimum_replicate_adjusted_rand"]),
                                -int(row["cluster_count"]),
                            ),
                        )
                    if selected != expected_selected:
                        raise ValueError("activation category cluster count choice differs")
                elif selected.get("status") != "pass_fallback" or int(
                    selected.get("cluster_count", 0)
                ) != 1:
                    raise ValueError("activation category cluster count fallback differs")
        else:
            raise ValueError("activation category cluster count mode is invalid")
        category_names = [str(value) for value in compartment_contract["categories"]]
        thresholds = compartment_contract["specificity_thresholds"]
        uncertainty_screen = compartment_contract.get("uncertainty_screen", {})
        uncertainty_mode = str(
            uncertainty_screen.get("mode", "unavailable_legacy_archive")
        )
        if uncertainty_mode not in {
            "leave_one_pair_block_out_countsketch",
            "unavailable_legacy_archive",
        }:
            raise ValueError("activation category uncertainty mode is invalid")
        if bool(uncertainty_screen.get("required", False)) and uncertainty_mode != (
            "leave_one_pair_block_out_countsketch"
        ):
            raise ValueError("required activation category uncertainty evidence is absent")
        pair_block_direction_screen = compartment_contract.get(
            "pair_block_direction_robustness_screen", {}
        )
        pair_block_direction_mode = str(
            pair_block_direction_screen.get("mode", "unavailable_legacy_archive")
        )
        if pair_block_direction_mode not in {
            "pair_block_mean_countsketch_direction_and_relative_norm_coverage",
            "unavailable_legacy_archive",
        }:
            raise ValueError("activation category pair-block direction mode is invalid")
        if bool(pair_block_direction_screen.get("required", False)) and (
            pair_block_direction_mode
            != "pair_block_mean_countsketch_direction_and_relative_norm_coverage"
        ):
            raise ValueError("required activation category pair-block direction evidence is absent")
        if bool(pair_block_direction_screen.get("required", False)) and int(
            pair_block_direction_screen.get("observed_countsketch_dimension", 0)
        ) < int(pair_block_direction_screen.get("minimum_countsketch_dimension", 0)):
            raise ValueError("activation category pair-block direction sketch is underpowered")
        minimum_pair_block_direction_coverage = float(
            pair_block_direction_screen.get("minimum_coverage", 0.0)
        )
        if not 0.0 < minimum_pair_block_direction_coverage <= 1.0:
            raise ValueError("activation category pair-block direction coverage is invalid")
        condition_screen = compartment_contract.get(
            "condition_robustness_screen", {}
        )
        condition_mode = str(
            condition_screen.get("mode", "unavailable_legacy_archive")
        )
        if condition_mode not in {
            "condition_mean_countsketch_direction_and_relative_norm_coverage",
            "unavailable_legacy_archive",
        }:
            raise ValueError("activation category condition robustness mode is invalid")
        if bool(condition_screen.get("required", False)) and condition_mode != (
            "condition_mean_countsketch_direction_and_relative_norm_coverage"
        ):
            raise ValueError("required activation category condition evidence is absent")
        if bool(condition_screen.get("required", False)) and int(
            condition_screen.get("observed_countsketch_dimension", 0)
        ) < int(condition_screen.get("minimum_countsketch_dimension", 0)):
            raise ValueError("activation category condition sketch is underpowered")
        minimum_condition_coverage = float(
            condition_screen.get("minimum_coverage", 0.0)
        )
        if not 0.0 < minimum_condition_coverage <= 1.0:
            raise ValueError("activation category condition coverage threshold is invalid")
        template_screen = compartment_contract.get(
            "template_robustness_screen", {}
        )
        template_mode = str(
            template_screen.get("mode", "unavailable_legacy_archive")
        )
        if template_mode not in {
            "template_group_mean_countsketch_direction_and_relative_norm_coverage",
            "unavailable_legacy_archive",
        }:
            raise ValueError("activation category template robustness mode is invalid")
        if bool(template_screen.get("required", False)) and template_mode != (
            "template_group_mean_countsketch_direction_and_relative_norm_coverage"
        ):
            raise ValueError("required activation category template evidence is absent")
        if bool(template_screen.get("required", False)) and int(
            template_screen.get("observed_countsketch_dimension", 0)
        ) < int(template_screen.get("minimum_countsketch_dimension", 0)):
            raise ValueError("activation category template sketch is underpowered")
        minimum_template_coverage = float(
            template_screen.get("minimum_coverage", 0.0)
        )
        if not 0.0 < minimum_template_coverage <= 1.0:
            raise ValueError("activation category template coverage threshold is invalid")
        cross_archive_screen = compartment_contract.get(
            "per_component_cross_archive_admission", {}
        )
        if not isinstance(cross_archive_screen, dict):
            raise ValueError("per-component cross-archive admission contract is invalid")
        cross_archive_mode = str(cross_archive_screen.get("mode", ""))
        if cross_archive_mode not in {
            "same_category_direction_magnitude_and_stratified_robustness",
            "unavailable_without_calibration_archive",
        } or cross_archive_screen.get("unstable_specific_routes_to") != (
            "shared_cross_category"
        ) or bool(cross_archive_screen.get("model_outcomes_used", True)):
            raise ValueError("per-component cross-archive admission policy is invalid")
        threshold_calibration = compartment_contract.get("threshold_calibration")
        if threshold_calibration is not None:
            if cross_archive_mode != (
                "same_category_direction_magnitude_and_stratified_robustness"
            ) or not bool(cross_archive_screen.get("required", False)):
                raise ValueError("calibrated bundle lacks per-component admission")
            if not isinstance(threshold_calibration, dict):
                raise ValueError("activation threshold calibration is invalid")
            calibration_body = dict(threshold_calibration)
            calibration_id = str(calibration_body.pop("calibration_id", ""))
            if content_sha256(calibration_body) != calibration_id:
                raise ValueError("activation threshold calibration content address mismatch")
            if threshold_calibration.get("category_names") != category_names:
                raise ValueError("activation threshold calibration categories differ")
            sources = threshold_calibration.get("sources", {})
            if sources.get("discovery_effects_sha256") != manifest["source"].get(
                "effects_sha256"
            ) or sources.get("calibration_effects_sha256") != manifest["source"].get(
                "calibration_effects_sha256"
            ):
                raise ValueError("activation threshold calibration source hashes differ")
            if sources.get("discovery_effects_sha256") == sources.get(
                "calibration_effects_sha256"
            ):
                raise ValueError("activation threshold calibration sources are not distinct")
            split_provenance = threshold_calibration.get("split_provenance")
            if split_provenance is None:
                if threshold_calibration.get("status") != "pass_unattested_pilot":
                    raise ValueError("activation threshold calibration lacks split provenance")
            else:
                if threshold_calibration.get("status") != "pass" or split_provenance.get(
                    "status"
                ) != "verified":
                    raise ValueError("activation threshold split provenance status is invalid")
                if split_provenance.get("activation_category_split_id") != manifest[
                    "source"
                ].get("activation_category_split_id"):
                    raise ValueError("activation threshold split ID differs from manifest")
                if split_provenance["discovery"].get(
                    "effect_merge_manifest_sha256"
                ) != manifest["source"].get("effects_merge_manifest_sha256") or split_provenance[
                    "calibration"
                ].get("effect_merge_manifest_sha256") != manifest["source"].get(
                    "calibration_effects_merge_manifest_sha256"
                ):
                    raise ValueError("activation threshold effect merge hashes differ")
            source_requirements = threshold_calibration.get("source_requirements", {})
            policy = threshold_calibration["policy"]
            robustness_required = bool(
                policy.get("cross_archive_stratified_robustness_required", False)
            )
            if bool(
                source_requirements.get(
                    "same_stratified_robustness_gates_required_in_both_archives",
                    False,
                )
            ) != robustness_required:
                raise ValueError(
                    "activation threshold calibration stratified robustness policy differs"
                )
            minimum_episodes = int(
                source_requirements.get(
                    "minimum_episodes_per_category_per_archive", 0
                )
            )
            if minimum_episodes < 1 or min(
                int(value)
                for value in threshold_calibration[
                    "discovery_category_episode_counts"
                ]
            ) < minimum_episodes or min(
                int(value)
                for value in threshold_calibration[
                    "calibration_category_episode_counts"
                ]
            ) < minimum_episodes:
                raise ValueError("activation threshold calibration episode support is invalid")
            discovery_strength_rows: list[list[float]] = []
            calibration_strength_rows: list[list[float]] = []
            direction_cosine_rows: list[list[float]] = []
            robustness_eligibility_rows: list[list[bool]] = []
            for assignment in assignments:
                discovery_strengths = assignment.get("activation_category_strengths")
                calibration_strengths = assignment.get(
                    "calibration_activation_category_strengths"
                )
                direction_cosines = assignment.get(
                    "cross_archive_activation_category_direction_cosines"
                )
                strength_ratios = assignment.get(
                    "cross_archive_activation_category_strength_ratios"
                )
                if not isinstance(discovery_strengths, dict) or not isinstance(
                    calibration_strengths, dict
                ) or not isinstance(direction_cosines, dict) or not isinstance(
                    strength_ratios, dict
                ):
                    raise ValueError("assignment omits calibration activation strengths")
                if set(discovery_strengths) != set(category_names) or set(
                    calibration_strengths
                ) != set(category_names) or set(direction_cosines) != set(
                    category_names
                ) or set(strength_ratios) != set(category_names):
                    raise ValueError("assignment calibration strengths differ from categories")
                calibration_coverage_fields = (
                    (
                        pair_block_direction_mode
                        == "pair_block_mean_countsketch_direction_and_relative_norm_coverage",
                        "activation_category_pair_block_direction_coverage",
                        "calibration_activation_category_pair_block_direction_coverage",
                        minimum_pair_block_direction_coverage,
                    ),
                    (
                        condition_mode
                        == "condition_mean_countsketch_direction_and_relative_norm_coverage",
                        "activation_category_condition_coverage",
                        "calibration_activation_category_condition_coverage",
                        minimum_condition_coverage,
                    ),
                    (
                        template_mode
                        == "template_group_mean_countsketch_direction_and_relative_norm_coverage",
                        "activation_category_template_coverage",
                        "calibration_activation_category_template_coverage",
                        minimum_template_coverage,
                    ),
                )
                stored_eligibility = assignment.get(
                    "cross_archive_activation_category_stratified_robustness_eligible"
                )
                if robustness_required:
                    if not isinstance(stored_eligibility, dict) or set(
                        stored_eligibility
                    ) != set(category_names):
                        raise ValueError(
                            "assignment omits cross-archive stratified robustness eligibility"
                        )
                    expected_eligibility = {
                        category: True for category in category_names
                    }
                    for available, discovery_field, calibration_field, minimum in (
                        calibration_coverage_fields
                    ):
                        discovery_coverage = assignment.get(discovery_field)
                        calibration_coverage = assignment.get(calibration_field)
                        if available:
                            if not isinstance(discovery_coverage, dict) or not isinstance(
                                calibration_coverage, dict
                            ) or set(discovery_coverage) != set(category_names) or set(
                                calibration_coverage
                            ) != set(category_names):
                                raise ValueError(
                                    "assignment calibration robustness coverage differs"
                                )
                            if any(
                                not 0.0 <= float(value) <= 1.0
                                for value in calibration_coverage.values()
                            ):
                                raise ValueError(
                                    "assignment calibration robustness coverage is invalid"
                                )
                            for category in category_names:
                                expected_eligibility[category] = bool(
                                    expected_eligibility[category]
                                    and float(discovery_coverage[category]) >= minimum
                                    and float(calibration_coverage[category]) >= minimum
                                )
                        elif calibration_coverage is not None:
                            raise ValueError(
                                "legacy calibration unexpectedly carries robustness coverage"
                            )
                    if any(
                        bool(stored_eligibility[category])
                        != expected_eligibility[category]
                        for category in category_names
                    ):
                        raise ValueError(
                            "assignment cross-archive stratified robustness is not reproducible"
                        )
                    robustness_eligibility_rows.append(
                        [expected_eligibility[name] for name in category_names]
                    )
                elif stored_eligibility is not None or any(
                    assignment.get(calibration_field) is not None
                    for _, _, calibration_field, _ in calibration_coverage_fields
                ):
                    raise ValueError(
                        "assignment unexpectedly carries calibration robustness evidence"
                    )
                for category in category_names:
                    discovery_value = float(discovery_strengths[category])
                    calibration_value = float(calibration_strengths[category])
                    denominator = max(discovery_value, calibration_value)
                    expected_ratio = (
                        min(discovery_value, calibration_value) / denominator
                        if denominator > 1e-12
                        else 0.0
                    )
                    if not math.isclose(
                        float(strength_ratios[category]),
                        expected_ratio,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    ):
                        raise ValueError("assignment calibration strength ratio differs")
                discovery_strength_rows.append(
                    [float(discovery_strengths[name]) for name in category_names]
                )
                calibration_strength_rows.append(
                    [float(calibration_strengths[name]) for name in category_names]
                )
                direction_cosine_rows.append(
                    [float(direction_cosines[name]) for name in category_names]
                )
            try:
                recomputed_calibration = calibrate_activation_compartment_thresholds(
                    np.asarray(discovery_strength_rows, dtype=np.float64),
                    np.asarray(calibration_strength_rows, dtype=np.float64),
                    category_names,
                    direction_cosines=np.asarray(
                        direction_cosine_rows, dtype=np.float64
                    ),
                    minimum_replication_rate=float(policy["minimum_replication_rate"]),
                    minimum_stable_specific_components=int(
                        policy["minimum_stable_specific_components"]
                    ),
                    minimum_direction_cosine=float(
                        policy["minimum_direction_cosine"]
                    ),
                    minimum_strength_ratio=float(
                        policy["minimum_strength_ratio"]
                    ),
                    robustness_eligibility=(
                        np.asarray(robustness_eligibility_rows, dtype=bool)
                        if robustness_required
                        else None
                    ),
                    candidate_grid=threshold_calibration["candidate_grid"],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "activation threshold calibration is not reproducible"
                ) from exc
            for key in ("policy", "candidate_grid", "selection"):
                if json_normalized(threshold_calibration.get(key)) != json_normalized(
                    recomputed_calibration[key]
                ):
                    raise ValueError(
                        f"activation threshold calibration is not reproducible: {key}"
                    )
            if thresholds != threshold_calibration["selection"][
                "selected_thresholds"
            ]:
                raise ValueError("selected activation thresholds differ from calibration")
            if not math.isclose(
                float(cross_archive_screen.get("minimum_direction_cosine")),
                float(policy["minimum_direction_cosine"]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ) or not math.isclose(
                float(cross_archive_screen.get("minimum_strength_ratio")),
                float(policy["minimum_strength_ratio"]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError("per-component cross-archive thresholds differ")
        elif cross_archive_mode != "unavailable_without_calibration_archive" or bool(
            cross_archive_screen.get("required", True)
        ):
            raise ValueError("uncalibrated bundle claims per-component admission")
        observed_cross_archive_admitted = 0
        observed_cross_archive_rerouted = 0
        for assignment in assignments:
            strengths = assignment.get("activation_category_strengths")
            if not isinstance(strengths, dict) or set(strengths) != set(category_names):
                raise ValueError("assignment activation strengths differ from category contract")
            classification = classify_activation_compartments(
                np.asarray(
                    [[float(strengths[name]) for name in category_names]],
                    dtype=np.float64,
                ),
                category_names,
                minimum_dominant_share=float(thresholds["minimum_dominant_share"]),
                minimum_dominance_margin=float(
                    thresholds["minimum_dominance_margin"]
                ),
                maximum_normalized_entropy=float(
                    thresholds["maximum_normalized_entropy"]
                ),
            )
            point_category = classification[
                "compartment_categories"
            ][0]
            if str(assignment.get("point_estimate_activation_category", point_category)) != (
                point_category
            ):
                raise ValueError("assignment point-estimate compartment is not reproducible")
            final_category = point_category
            jackknife_labels = assignment.get("activation_category_jackknife_labels")
            jackknife_stability = assignment.get(
                "activation_category_jackknife_stability"
            )
            if uncertainty_mode == "leave_one_pair_block_out_countsketch":
                block_count = int(uncertainty_screen.get("block_count", 0))
                if not isinstance(jackknife_labels, list) or len(jackknife_labels) != block_count:
                    raise ValueError("assignment jackknife labels do not match block count")
                valid_labels = set(category_names) | {"shared_cross_category"}
                if any(str(label) not in valid_labels for label in jackknife_labels):
                    raise ValueError("assignment jackknife label is invalid")
                expected_stability = sum(
                    str(label) == point_category for label in jackknife_labels
                ) / block_count
                if jackknife_stability is None or not math.isclose(
                    float(jackknife_stability),
                    expected_stability,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise ValueError("assignment jackknife stability is not reproducible")
                if point_category != "shared_cross_category" and expected_stability < float(
                    uncertainty_screen["minimum_stability"]
                ):
                    final_category = "shared_cross_category"
            elif jackknife_labels is not None or jackknife_stability is not None:
                raise ValueError("legacy assignment unexpectedly carries jackknife evidence")
            if str(
                assignment.get("post_uncertainty_activation_category", final_category)
            ) != final_category:
                raise ValueError("assignment post-uncertainty compartment is not reproducible")
            pair_block_fields = (
                "activation_category_pair_block_direction_coverage",
                "activation_category_pair_block_minimum_direction_cosine",
                "activation_category_pair_block_minimum_relative_norm",
            )
            if pair_block_direction_mode == (
                "pair_block_mean_countsketch_direction_and_relative_norm_coverage"
            ):
                pair_block_values = [
                    assignment.get(field) for field in pair_block_fields
                ]
                if any(
                    not isinstance(values, dict) or set(values) != set(category_names)
                    for values in pair_block_values
                ):
                    raise ValueError("assignment pair-block direction diagnostics differ")
                coverage, cosines, relative_norms = pair_block_values
                if any(
                    not 0.0 <= float(value) <= 1.0 for value in coverage.values()
                ) or any(
                    not -1.0 <= float(value) <= 1.0 for value in cosines.values()
                ) or any(float(value) < 0.0 for value in relative_norms.values()):
                    raise ValueError("assignment pair-block direction values are invalid")
                if final_category != "shared_cross_category" and float(
                    coverage[final_category]
                ) < minimum_pair_block_direction_coverage:
                    final_category = "shared_cross_category"
            elif any(assignment.get(field) is not None for field in pair_block_fields):
                raise ValueError("legacy assignment unexpectedly carries pair-block direction evidence")
            if str(
                assignment.get(
                    "post_pair_block_direction_activation_category", final_category
                )
            ) != final_category:
                raise ValueError("assignment post-pair-block compartment is not reproducible")
            condition_fields = (
                "activation_category_condition_coverage",
                "activation_category_condition_minimum_direction_cosine",
                "activation_category_condition_minimum_relative_norm",
            )
            if condition_mode == (
                "condition_mean_countsketch_direction_and_relative_norm_coverage"
            ):
                condition_values = [assignment.get(field) for field in condition_fields]
                if any(
                    not isinstance(values, dict) or set(values) != set(category_names)
                    for values in condition_values
                ):
                    raise ValueError("assignment condition robustness diagnostics differ")
                coverage, cosines, relative_norms = condition_values
                if any(
                    not 0.0 <= float(value) <= 1.0 for value in coverage.values()
                ) or any(
                    not -1.0 <= float(value) <= 1.0 for value in cosines.values()
                ) or any(float(value) < 0.0 for value in relative_norms.values()):
                    raise ValueError("assignment condition robustness values are invalid")
                if final_category != "shared_cross_category" and float(
                    coverage[final_category]
                ) < minimum_condition_coverage:
                    final_category = "shared_cross_category"
            elif any(assignment.get(field) is not None for field in condition_fields):
                raise ValueError("legacy assignment unexpectedly carries condition evidence")
            if str(
                assignment.get("post_condition_activation_category", final_category)
            ) != final_category:
                raise ValueError("assignment post-condition compartment is not reproducible")
            template_fields = (
                "activation_category_template_coverage",
                "activation_category_template_minimum_direction_cosine",
                "activation_category_template_minimum_relative_norm",
            )
            if template_mode == (
                "template_group_mean_countsketch_direction_and_relative_norm_coverage"
            ):
                template_values = [assignment.get(field) for field in template_fields]
                if any(
                    not isinstance(values, dict) or set(values) != set(category_names)
                    for values in template_values
                ):
                    raise ValueError("assignment template robustness diagnostics differ")
                coverage, cosines, relative_norms = template_values
                if any(
                    not 0.0 <= float(value) <= 1.0 for value in coverage.values()
                ) or any(
                    not -1.0 <= float(value) <= 1.0 for value in cosines.values()
                ) or any(float(value) < 0.0 for value in relative_norms.values()):
                    raise ValueError("assignment template robustness values are invalid")
                if final_category != "shared_cross_category" and float(
                    coverage[final_category]
                ) < minimum_template_coverage:
                    final_category = "shared_cross_category"
            elif any(assignment.get(field) is not None for field in template_fields):
                raise ValueError("legacy assignment unexpectedly carries template evidence")
            if str(
                assignment.get("pre_cross_archive_activation_category", final_category)
            ) != final_category:
                raise ValueError(
                    "assignment pre-cross-archive compartment is not reproducible"
                )
            if threshold_calibration is not None:
                calibration_strengths = assignment.get(
                    "calibration_activation_category_strengths"
                )
                direction_cosines = assignment.get(
                    "cross_archive_activation_category_direction_cosines"
                )
                strength_ratios = assignment.get(
                    "cross_archive_activation_category_strength_ratios"
                )
                stored_eligibility = assignment.get(
                    "cross_archive_activation_category_stratified_robustness_eligible"
                )
                if any(
                    not isinstance(values, dict)
                    or set(values) != set(category_names)
                    for values in (
                        calibration_strengths,
                        direction_cosines,
                        strength_ratios,
                    )
                ):
                    raise ValueError(
                        "assignment per-component admission diagnostics differ"
                    )
                if robustness_required and (
                    not isinstance(stored_eligibility, dict)
                    or set(stored_eligibility) != set(category_names)
                ):
                    raise ValueError(
                        "assignment per-component robustness eligibility differs"
                    )
                admission = apply_cross_archive_component_admission(
                    [final_category],
                    np.asarray(
                        [[float(calibration_strengths[name]) for name in category_names]],
                        dtype=np.float64,
                    ),
                    category_names,
                    thresholds=thresholds,
                    direction_cosines=np.asarray(
                        [[float(direction_cosines[name]) for name in category_names]],
                        dtype=np.float64,
                    ),
                    strength_ratios=np.asarray(
                        [[float(strength_ratios[name]) for name in category_names]],
                        dtype=np.float64,
                    ),
                    minimum_direction_cosine=float(
                        cross_archive_screen["minimum_direction_cosine"]
                    ),
                    minimum_strength_ratio=float(
                        cross_archive_screen["minimum_strength_ratio"]
                    ),
                    robustness_eligibility=(
                        np.asarray(
                            [[bool(stored_eligibility[name]) for name in category_names]],
                            dtype=bool,
                        )
                        if robustness_required
                        else None
                    ),
                )
                expected_calibration_category = admission[
                    "calibration_point_estimate_categories"
                ][0]
                expected_admitted = bool(admission["admitted_specific"][0])
                expected_reason = admission["rejection_reasons"][0]
                if assignment.get(
                    "calibration_point_estimate_activation_category"
                ) != expected_calibration_category or assignment.get(
                    "cross_archive_activation_category_admitted"
                ) is not expected_admitted or assignment.get(
                    "cross_archive_activation_category_rejection_reason"
                ) != expected_reason:
                    raise ValueError(
                        "assignment per-component cross-archive admission is not reproducible"
                    )
                if expected_admitted:
                    observed_cross_archive_admitted += 1
                elif final_category != "shared_cross_category":
                    observed_cross_archive_rerouted += 1
                final_category = admission["final_categories"][0]
            elif any(
                assignment.get(field) is not None
                for field in (
                    "calibration_point_estimate_activation_category",
                    "cross_archive_activation_category_admitted",
                    "cross_archive_activation_category_rejection_reason",
                )
            ):
                raise ValueError(
                    "uncalibrated assignment claims per-component admission"
                )
            if str(assignment.get("activation_category")) != final_category:
                raise ValueError("assignment activation compartment is not reproducible")
            if str(assignment.get("dominant_activation_category")) != classification[
                "dominant_categories"
            ][0]:
                raise ValueError("assignment dominant activation category is not reproducible")
            diagnostic_checks = (
                (
                    "dominant_activation_category_share",
                    float(classification["dominant_shares"][0]),
                ),
                (
                    "activation_category_dominance_margin",
                    float(classification["dominance_margins"][0]),
                ),
                (
                    "activation_category_normalized_entropy",
                    float(classification["normalized_entropy"][0]),
                ),
            )
            for field, expected_value in diagnostic_checks:
                if not math.isclose(
                    float(assignment.get(field)), expected_value, rel_tol=1e-9, abs_tol=1e-9
                ):
                    raise ValueError(f"assignment activation diagnostic differs: {field}")
        if threshold_calibration is not None:
            if int(cross_archive_screen.get("admitted_specific_count", -1)) != (
                observed_cross_archive_admitted
            ) or int(cross_archive_screen.get("rerouted_specific_count", -1)) != (
                observed_cross_archive_rerouted
            ):
                raise ValueError("per-component cross-archive admission counts differ")
        elif cross_archive_screen.get("admitted_specific_count") is not None or (
            cross_archive_screen.get("rerouted_specific_count") is not None
        ):
            raise ValueError("uncalibrated bundle carries per-component admission counts")
    for entry in artifacts["packets"]:
        path = output_dir / entry["packet_path"]
        if sha256_file(path) != entry["packet_sha256"]:
            raise ValueError(f"packet artifact hash mismatch: {entry['cluster_id']}")
        packet = json.loads(path.read_text(encoding="utf-8"))
        packet_id = str(packet.pop("packet_id"))
        distribution = dict(packet["distribution"])
        distribution.pop("assigned_worker", None)
        packet["distribution"] = distribution
        if content_sha256(packet) != packet_id or packet_id != entry["packet_id"]:
            raise ValueError(f"packet content address mismatch: {entry['cluster_id']}")
        for source_key in (
            "component_inventory_sha256",
            "source_kind",
            "effects_sha256",
            "calibration_effects_sha256",
            "cluster_validation_effects_sha256",
            "effects_merge_manifest_sha256",
            "calibration_effects_merge_manifest_sha256",
            "cluster_validation_effects_merge_manifest_sha256",
            "activation_category_split_id",
            "activation_category_split_schema",
            "activation_category_template_group_set_sha256_by_role",
        ):
            if packet["source"].get(source_key) != manifest["source"].get(source_key):
                raise ValueError(
                    f"packet source differs from manifest: {entry['cluster_id']}:{source_key}"
                )
        alignment_key = str(packet["sae_study"].get("alignment_activation_key"))
        effect_space_id = str(packet["effect_refinement"].get("effect_space_id"))
        if alignment_key != effect_space_id:
            raise ValueError(f"packet SAE/effect spaces differ: {entry['cluster_id']}")
        if alignment_key not in set(packet["focus"]["recommended_activation_keys"]):
            raise ValueError(f"packet omits aligned SAE activation key: {entry['cluster_id']}")
        required_controls = set(packet["sae_study"]["required_controls"])
        if "same_activation_space_alignment" not in required_controls:
            raise ValueError(f"packet omits same-space alignment control: {entry['cluster_id']}")
        if bool(compartment_contract.get("enabled", False)):
            compartment = packet["sae_study"].get("activation_compartment")
            if not isinstance(compartment, dict):
                raise ValueError(f"packet omits activation compartment: {entry['cluster_id']}")
            category = str(compartment.get("category"))
            packet_compartments.append(category)
            assigned_categories = {
                str(assignment_by_id[str(component_id)].get("activation_category"))
                for component_id in packet["component_ids"]
            }
            if assigned_categories != {category}:
                raise ValueError(
                    f"packet crosses activation categories: {entry['cluster_id']}"
                )
            if "cross_activation_category_holdout" not in required_controls:
                raise ValueError(
                    f"packet omits cross-category holdout: {entry['cluster_id']}"
                )
            if compartment.get("specificity_thresholds") != compartment_contract.get(
                "specificity_thresholds"
            ):
                raise ValueError(
                    f"packet activation thresholds differ from manifest: {entry['cluster_id']}"
                )
            if compartment.get("uncertainty_screen") != compartment_contract.get(
                "uncertainty_screen"
            ):
                raise ValueError(
                    f"packet uncertainty screen differs from manifest: {entry['cluster_id']}"
                )
            if compartment.get(
                "pair_block_direction_robustness_screen"
            ) != compartment_contract.get("pair_block_direction_robustness_screen"):
                raise ValueError(
                    f"packet pair-block direction screen differs from manifest: {entry['cluster_id']}"
                )
            if compartment.get(
                "condition_robustness_screen"
            ) != compartment_contract.get("condition_robustness_screen"):
                raise ValueError(
                    f"packet condition screen differs from manifest: {entry['cluster_id']}"
                )
            if compartment.get(
                "template_robustness_screen"
            ) != compartment_contract.get("template_robustness_screen"):
                raise ValueError(
                    f"packet template screen differs from manifest: {entry['cluster_id']}"
                )
            if compartment.get(
                "per_component_cross_archive_admission"
            ) != compartment_contract.get("per_component_cross_archive_admission"):
                raise ValueError(
                    f"packet per-component admission differs from manifest: {entry['cluster_id']}"
                )
            expected_calibration = (
                None
                if compartment_contract.get("threshold_calibration") is None
                else {
                    "schema": compartment_contract["threshold_calibration"]["schema"],
                    "status": compartment_contract["threshold_calibration"]["status"],
                    "mode": compartment_contract["threshold_calibration"]["mode"],
                    "calibration_id": compartment_contract["threshold_calibration"][
                        "calibration_id"
                    ],
                }
            )
            if compartment.get("threshold_calibration") != expected_calibration:
                raise ValueError(
                    f"packet activation calibration differs from manifest: {entry['cluster_id']}"
                )
            expected_cluster_count_selection = {
                "schema": cluster_count_receipt["schema"],
                "status": cluster_count_receipt["status"],
                "mode": cluster_count_receipt["mode"],
                "selection_id": cluster_count_receipt["selection_id"],
            }
            if compartment.get(
                "cluster_count_selection"
            ) != expected_cluster_count_selection:
                raise ValueError(
                    f"packet cluster count selection differs from manifest: {entry['cluster_id']}"
                )
        packet_ids.append(packet_id)
        packet_component_ids.extend(str(value) for value in packet["component_ids"])
        matched_controls = [
            str(value) for value in packet["sae_study"]["matched_control_component_ids"]
        ]
        if len(matched_controls) != len(set(matched_controls)):
            raise ValueError(f"packet has duplicate matched controls: {entry['cluster_id']}")
        if set(matched_controls) & set(packet["component_ids"]):
            raise ValueError(f"packet matched controls overlap its cluster: {entry['cluster_id']}")
        if not set(matched_controls).issubset(set(assignment_ids)):
            raise ValueError(f"packet matched controls are outside the VPD inventory: {entry['cluster_id']}")
        if content_sha256(matched_controls) != str(
            packet["sae_study"]["matched_control_set_sha256"]
        ):
            raise ValueError(f"packet matched control set hash mismatch: {entry['cluster_id']}")
    if sorted(packet_component_ids) != sorted(assignment_ids):
        raise ValueError("packet membership and assignments differ")
    if bool(compartment_contract.get("enabled", False)):
        if dict(sorted(Counter(packet_compartments).items())) != dict(
            sorted(compartment_contract["cluster_allocation"].items())
        ):
            raise ValueError("packet compartment counts differ from cluster allocation")
        if sorted(set(packet_compartments)) != sorted(
            compartment_contract["observed_component_compartments"]
        ):
            raise ValueError("packet compartments differ from manifest")
    expected = manifest["invariants"]
    if len(assignment_ids) != int(expected["component_count"]):
        raise ValueError("manifest component count mismatch")
    if content_sha256(sorted(assignment_ids)) != expected["component_id_set_sha256"]:
        raise ValueError("component ID set hash mismatch")
    return {
        "schema": "sch.vpd-sae-cluster-verification.v1",
        "status": "pass",
        "components": len(assignment_ids),
        "clusters": len(packet_ids),
        "packets": len(packet_ids),
        "exact_partition": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster VPD components into content-addressed focused SAE study packets."
    )
    parser.add_argument("--components", type=Path)
    parser.add_argument("--hierarchy-plan", type=Path)
    parser.add_argument("--effects", type=Path)
    parser.add_argument("--calibration-effects", type=Path)
    parser.add_argument("--cluster-validation-effects", type=Path)
    parser.add_argument("--effects-merge-manifest", type=Path)
    parser.add_argument("--calibration-effects-merge-manifest", type=Path)
    parser.add_argument("--cluster-validation-effects-merge-manifest", type=Path)
    parser.add_argument("--allow-unattested-calibration", action="store_true")
    parser.add_argument("--allow-unattested-cluster-validation", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-output", type=Path)
    parser.add_argument("--clusters", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1707)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--metadata-weight", type=float, default=0.35)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--anchors", default=",".join(DEFAULT_ANCHORS))
    parser.add_argument("--workers", default="")
    parser.add_argument("--effect-space-id", default="layer_35__proposal_start")
    parser.add_argument("--no-category-compartments", action="store_true")
    parser.add_argument(
        "--select-category-cluster-counts-by-projection", action="store_true"
    )
    parser.add_argument(
        "--category-cluster-count-selector-version",
        choices=("v1", "v2"),
        default="v1",
        help="V1 preserves historical receipts; V2 adds lower-tail and boundary-gap discreteness controls.",
    )
    parser.add_argument(
        "--require-category-cluster-sampling-stability", action="store_true"
    )
    parser.add_argument(
        "--require-category-cluster-cross-archive-stability", action="store_true"
    )
    parser.add_argument(
        "--category-cluster-projection-replicates",
        type=int,
        default=DEFAULT_CATEGORY_CLUSTER_PROJECTION_REPLICATES,
    )
    parser.add_argument(
        "--maximum-clusters-per-category",
        type=int,
        default=DEFAULT_MAXIMUM_CLUSTERS_PER_CATEGORY,
    )
    parser.add_argument(
        "--minimum-category-cluster-projection-stability",
        type=float,
        default=DEFAULT_MINIMUM_CATEGORY_CLUSTER_PROJECTION_STABILITY,
    )
    parser.add_argument(
        "--minimum-category-cluster-silhouette",
        type=float,
        default=DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE,
    )
    parser.add_argument(
        "--minimum-category-cluster-silhouette-p10",
        type=float,
        default=DEFAULT_MINIMUM_CATEGORY_CLUSTER_SILHOUETTE_P10,
    )
    parser.add_argument(
        "--minimum-category-cluster-boundary-gap-ratio",
        type=float,
        default=DEFAULT_MINIMUM_CATEGORY_CLUSTER_BOUNDARY_GAP_RATIO,
    )
    parser.add_argument(
        "--minimum-category-cluster-size",
        type=int,
        default=DEFAULT_MINIMUM_CATEGORY_CLUSTER_SIZE,
    )
    parser.add_argument(
        "--minimum-dominant-category-share",
        type=float,
        default=DEFAULT_MINIMUM_DOMINANT_CATEGORY_SHARE,
    )
    parser.add_argument(
        "--minimum-category-dominance-margin",
        type=float,
        default=DEFAULT_MINIMUM_CATEGORY_DOMINANCE_MARGIN,
    )
    parser.add_argument(
        "--maximum-category-normalized-entropy",
        type=float,
        default=DEFAULT_MAXIMUM_CATEGORY_NORMALIZED_ENTROPY,
    )
    parser.add_argument(
        "--minimum-category-replication-rate",
        type=float,
        default=DEFAULT_MINIMUM_CATEGORY_REPLICATION_RATE,
    )
    parser.add_argument(
        "--minimum-stable-specific-components",
        type=int,
        default=DEFAULT_MINIMUM_STABLE_SPECIFIC_COMPONENTS,
    )
    parser.add_argument(
        "--minimum-category-calibration-episodes",
        type=int,
        default=DEFAULT_MINIMUM_CATEGORY_CALIBRATION_EPISODES,
    )
    parser.add_argument(
        "--minimum-category-direction-cosine",
        type=float,
        default=DEFAULT_MINIMUM_CATEGORY_DIRECTION_COSINE,
    )
    parser.add_argument(
        "--minimum-category-strength-ratio",
        type=float,
        default=DEFAULT_MINIMUM_CATEGORY_STRENGTH_RATIO,
    )
    parser.add_argument(
        "--minimum-category-jackknife-stability",
        type=float,
        default=DEFAULT_MINIMUM_CATEGORY_JACKKNIFE_STABILITY,
    )
    parser.add_argument("--require-category-uncertainty", action="store_true")
    parser.add_argument(
        "--minimum-category-pair-block-direction-coverage",
        type=float,
        default=DEFAULT_MINIMUM_CATEGORY_PAIR_BLOCK_DIRECTION_COVERAGE,
    )
    parser.add_argument(
        "--minimum-pair-block-direction-cosine",
        type=float,
        default=DEFAULT_MINIMUM_PAIR_BLOCK_DIRECTION_COSINE,
    )
    parser.add_argument(
        "--minimum-pair-block-relative-norm",
        type=float,
        default=DEFAULT_MINIMUM_PAIR_BLOCK_RELATIVE_NORM,
    )
    parser.add_argument(
        "--require-category-pair-block-direction-robustness",
        action="store_true",
    )
    parser.add_argument(
        "--minimum-category-condition-coverage",
        type=float,
        default=DEFAULT_MINIMUM_CATEGORY_CONDITION_COVERAGE,
    )
    parser.add_argument(
        "--minimum-condition-direction-cosine",
        type=float,
        default=DEFAULT_MINIMUM_CONDITION_DIRECTION_COSINE,
    )
    parser.add_argument(
        "--minimum-condition-relative-norm",
        type=float,
        default=DEFAULT_MINIMUM_CONDITION_RELATIVE_NORM,
    )
    parser.add_argument(
        "--require-category-condition-robustness", action="store_true"
    )
    parser.add_argument(
        "--minimum-category-template-coverage",
        type=float,
        default=DEFAULT_MINIMUM_CATEGORY_TEMPLATE_COVERAGE,
    )
    parser.add_argument(
        "--minimum-template-direction-cosine",
        type=float,
        default=DEFAULT_MINIMUM_TEMPLATE_DIRECTION_COSINE,
    )
    parser.add_argument(
        "--minimum-template-relative-norm",
        type=float,
        default=DEFAULT_MINIMUM_TEMPLATE_RELATIVE_NORM,
    )
    parser.add_argument(
        "--require-category-template-robustness", action="store_true"
    )
    parser.add_argument(
        "--minimum-category-robustness-sketch-dimension",
        type=int,
        default=DEFAULT_MINIMUM_CATEGORY_ROBUSTNESS_SKETCH_DIMENSION,
    )
    args = parser.parse_args()
    if args.verify_output is not None:
        result = verify_bundle(args.verify_output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --verify-output is used")
    if (args.components is None) == (args.hierarchy_plan is None):
        parser.error("provide exactly one of --components or --hierarchy-plan")
    if args.metadata_weight < 0.0:
        raise ValueError("metadata weight must be non-negative")
    source_path = args.components or args.hierarchy_plan
    assert source_path is not None
    if args.components is not None:
        rows = load_jsonl(args.components)
        source_kind = "component_inventory_jsonl"
    else:
        rows = load_hierarchy_plan(args.hierarchy_plan)
        source_kind = "hierarchy_plan_exact_partition"
    manifest = build_bundle(
        rows,
        inventory_sha256=sha256_file(source_path),
        effects_path=args.effects,
        output_dir=args.output_dir,
        cluster_count=args.clusters,
        seed=args.seed,
        projection_dim=args.projection_dim,
        metadata_weight=args.metadata_weight,
        max_iterations=args.max_iterations,
        anchors=[value.strip() for value in args.anchors.split(",") if value.strip()],
        workers=sorted({value.strip() for value in args.workers.split(",") if value.strip()}),
        source_kind=source_kind,
        effect_space_id=args.effect_space_id,
        category_compartments=not args.no_category_compartments,
        calibration_effects_path=args.calibration_effects,
        cluster_validation_effects_path=args.cluster_validation_effects,
        effects_merge_manifest_path=args.effects_merge_manifest,
        calibration_effects_merge_manifest_path=args.calibration_effects_merge_manifest,
        cluster_validation_effects_merge_manifest_path=(
            args.cluster_validation_effects_merge_manifest
        ),
        require_calibration_split_provenance=not args.allow_unattested_calibration,
        require_cluster_validation_split_provenance=(
            not args.allow_unattested_cluster_validation
        ),
        minimum_dominant_category_share=args.minimum_dominant_category_share,
        minimum_category_dominance_margin=args.minimum_category_dominance_margin,
        maximum_category_normalized_entropy=args.maximum_category_normalized_entropy,
        minimum_category_replication_rate=args.minimum_category_replication_rate,
        minimum_stable_specific_components=args.minimum_stable_specific_components,
        minimum_category_calibration_episodes=args.minimum_category_calibration_episodes,
        minimum_category_direction_cosine=args.minimum_category_direction_cosine,
        minimum_category_strength_ratio=args.minimum_category_strength_ratio,
        minimum_category_jackknife_stability=(
            args.minimum_category_jackknife_stability
        ),
        require_category_uncertainty=args.require_category_uncertainty,
        minimum_category_pair_block_direction_coverage=(
            args.minimum_category_pair_block_direction_coverage
        ),
        minimum_pair_block_direction_cosine=(
            args.minimum_pair_block_direction_cosine
        ),
        minimum_pair_block_relative_norm=args.minimum_pair_block_relative_norm,
        require_category_pair_block_direction_robustness=(
            args.require_category_pair_block_direction_robustness
        ),
        minimum_category_condition_coverage=(
            args.minimum_category_condition_coverage
        ),
        minimum_condition_direction_cosine=args.minimum_condition_direction_cosine,
        minimum_condition_relative_norm=args.minimum_condition_relative_norm,
        require_category_condition_robustness=(
            args.require_category_condition_robustness
        ),
        minimum_category_template_coverage=(
            args.minimum_category_template_coverage
        ),
        minimum_template_direction_cosine=args.minimum_template_direction_cosine,
        minimum_template_relative_norm=args.minimum_template_relative_norm,
        require_category_template_robustness=(
            args.require_category_template_robustness
        ),
        minimum_category_robustness_sketch_dimension=(
            args.minimum_category_robustness_sketch_dimension
        ),
        select_category_cluster_counts_by_projection=(
            args.select_category_cluster_counts_by_projection
        ),
        category_cluster_projection_replicates=(
            args.category_cluster_projection_replicates
        ),
        maximum_clusters_per_category=args.maximum_clusters_per_category,
        minimum_category_cluster_projection_stability=(
            args.minimum_category_cluster_projection_stability
        ),
        minimum_category_cluster_silhouette=(
            args.minimum_category_cluster_silhouette
        ),
        minimum_category_cluster_silhouette_p10=(
            args.minimum_category_cluster_silhouette_p10
        ),
        minimum_category_cluster_boundary_gap_ratio=(
            args.minimum_category_cluster_boundary_gap_ratio
        ),
        minimum_category_cluster_size=args.minimum_category_cluster_size,
        category_cluster_count_selector_version=(
            args.category_cluster_count_selector_version
        ),
        require_category_cluster_sampling_stability=(
            args.require_category_cluster_sampling_stability
        ),
        require_category_cluster_cross_archive_stability=(
            args.require_category_cluster_cross_archive_stability
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
