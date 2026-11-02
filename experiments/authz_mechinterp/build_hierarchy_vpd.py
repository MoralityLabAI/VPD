from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

FAMILIES = (
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "self_attn.k_proj",
    "self_attn.o_proj",
    "self_attn.q_proj",
    "self_attn.v_proj",
)
MODULE_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\."
    r"(?P<family>mlp\.(?:down_proj|gate_proj|up_proj)|"
    r"self_attn\.(?:k_proj|o_proj|q_proj|v_proj))$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def stable_key(seed: int, *parts: str) -> str:
    return hashlib.sha256(":".join([str(seed), *parts]).encode("utf-8")).hexdigest()


def parse_component(row: dict[str, Any]) -> dict[str, Any]:
    match = MODULE_RE.fullmatch(str(row["module"]))
    if match is None:
        raise ValueError(f"unsupported adapter module: {row['module']}")
    layer = int(match.group("layer"))
    if not 0 <= layer < 36:
        raise ValueError(f"unexpected layer index: {layer}")
    parsed = dict(row)
    parsed["layer"] = layer
    parsed["projection_family"] = match.group("family")
    parsed["depth_band"] = layer // 9
    return parsed


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    norms = [float(row["component_fro_norm"]) for row in rows]
    return {
        "component_count": len(rows),
        "sum_component_fro_norm": sum(norms),
        "rss_component_fro_norm": math.sqrt(sum(value * value for value in norms)),
    }


def choose_controls(
    *,
    group_id: str,
    targets: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    candidate_filter: Callable[[dict[str, Any], dict[str, Any]], bool],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_ids = {str(row["component_id"]) for row in targets}
    selected: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    used: set[str] = set()
    for target in sorted(targets, key=lambda row: str(row["component_id"])):
        target_norm = float(target["component_fro_norm"])
        candidates = [
            row
            for row in inventory
            if str(row["component_id"]) not in target_ids
            and str(row["component_id"]) not in used
            and candidate_filter(target, row)
        ]
        if not candidates:
            raise ValueError(f"no unique control candidates for {target['component_id']}")
        within = [
            row
            for row in candidates
            if abs(float(row["component_fro_norm"]) - target_norm) / max(target_norm, 1e-12) <= 0.01
        ]
        if within:
            control = min(
                within,
                key=lambda row: stable_key(
                    seed,
                    group_id,
                    str(target["component_id"]),
                    str(row["component_id"]),
                ),
            )
            selection = "sha256_seeded_within_1pct"
        else:
            control = min(
                candidates,
                key=lambda row: (
                    abs(float(row["component_fro_norm"]) - target_norm),
                    stable_key(
                        seed,
                        group_id,
                        str(target["component_id"]),
                        str(row["component_id"]),
                    ),
                ),
            )
            selection = "nearest_norm_fallback"
        control_id = str(control["component_id"])
        used.add(control_id)
        selected.append(control)
        matches.append(
            {
                "target_component_id": target["component_id"],
                "control_component_id": control["component_id"],
                "target_fro_norm": target_norm,
                "control_fro_norm": float(control["component_fro_norm"]),
                "relative_norm_gap": abs(float(control["component_fro_norm"]) - target_norm)
                / max(target_norm, 1e-12),
                "selection": selection,
            }
        )
    return selected, matches


def make_group(
    *,
    group_id: str,
    tier: str,
    description: str,
    targets: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    candidate_filter: Callable[[dict[str, Any], dict[str, Any]], bool],
    seed: int,
    projection_family: str | None = None,
    depth_band: int | None = None,
) -> dict[str, Any]:
    controls, matches = choose_controls(
        group_id=group_id,
        targets=targets,
        inventory=inventory,
        candidate_filter=candidate_filter,
        seed=seed,
    )
    target_aggregate = aggregate(targets)
    control_aggregate = aggregate(controls)
    relative_rss_gap = abs(
        float(control_aggregate["rss_component_fro_norm"])
        - float(target_aggregate["rss_component_fro_norm"])
    ) / max(float(target_aggregate["rss_component_fro_norm"]), 1e-12)
    return {
        "group_id": group_id,
        "tier": tier,
        "description": description,
        "projection_family": projection_family,
        "depth_band": depth_band,
        "target_component_ids": [row["component_id"] for row in targets],
        "control_component_ids": [row["component_id"] for row in controls],
        "target_aggregate": target_aggregate,
        "control_aggregate": control_aggregate,
        "relative_rss_norm_gap": relative_rss_gap,
        "matches": matches,
    }


def build_group_plan(
    rows: list[dict[str, Any]], *, seed: int, inventory_sha256: str
) -> dict[str, Any]:
    inventory = [parse_component(row) for row in rows]
    if len(inventory) != 4032:
        raise ValueError(f"expected 4032 rank components, found {len(inventory)}")
    component_ids = [str(row["component_id"]) for row in inventory]
    if len(set(component_ids)) != len(component_ids):
        raise ValueError("component IDs are not unique")
    if {str(row["projection_family"]) for row in inventory} != set(FAMILIES):
        raise ValueError("projection family inventory does not match the frozen seven")
    if {int(row["depth_band"]) for row in inventory} != {0, 1, 2, 3}:
        raise ValueError("depth bands do not match four frozen nine-layer bands")

    groups: list[dict[str, Any]] = []
    for family in FAMILIES:
        targets = [row for row in inventory if row["projection_family"] == family]
        groups.append(
            make_group(
                group_id=f"family_{family.replace('.', '_')}",
                tier="projection_family",
                description=f"all ranks in {family} across layers 0-35",
                targets=targets,
                inventory=inventory,
                candidate_filter=lambda target, candidate: True,
                seed=seed,
                projection_family=family,
            )
        )
    for depth_band in range(4):
        targets = [row for row in inventory if row["depth_band"] == depth_band]
        groups.append(
            make_group(
                group_id=f"depth_{depth_band}",
                tier="depth_band",
                description=(f"all ranks in layers {depth_band * 9}-{depth_band * 9 + 8}"),
                targets=targets,
                inventory=inventory,
                candidate_filter=lambda target, candidate: True,
                seed=seed,
                depth_band=depth_band,
            )
        )
    for family in FAMILIES:
        for depth_band in range(4):
            targets = [
                row
                for row in inventory
                if row["projection_family"] == family and row["depth_band"] == depth_band
            ]
            groups.append(
                make_group(
                    group_id=(f"cell_{family.replace('.', '_')}_depth_{depth_band}"),
                    tier="family_x_depth",
                    description=(
                        f"all ranks in {family}, layers {depth_band * 9}-{depth_band * 9 + 8}"
                    ),
                    targets=targets,
                    inventory=inventory,
                    candidate_filter=lambda target, candidate: True,
                    seed=seed,
                    projection_family=family,
                    depth_band=depth_band,
                )
            )

    cells = [group for group in groups if group["tier"] == "family_x_depth"]
    cell_ids = [
        str(component_id) for group in cells for component_id in group["target_component_ids"]
    ]
    if len(cell_ids) != 4032 or set(cell_ids) != set(component_ids):
        raise ValueError("family-by-depth cells do not exactly partition the inventory")
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("a component appears in more than one family-by-depth cell")

    return {
        "schema": "bonsai8b_lora_hierarchy_group_plan_v1",
        "model_id": inventory[0]["model_id"],
        "revision": inventory[0]["revision"],
        "seed": seed,
        "component_inventory_sha256": inventory_sha256,
        "selection_source": (
            "architecture-only partition of every adapter rank component; no v5 "
            "intervention outcomes were available"
        ),
        "hierarchy": {
            "projection_families": list(FAMILIES),
            "depth_bands": {
                "0": [0, 8],
                "1": [9, 17],
                "2": [18, 26],
                "3": [27, 35],
            },
            "projection_family_groups": 7,
            "depth_band_groups": 4,
            "family_x_depth_cells": 28,
            "cell_target_partition_component_count": 4032,
        },
        "control_policy": {
            "unique_within_group": True,
            "norm_rule": (
                "SHA-256-seeded choice within 1% component Frobenius norm, "
                "else nearest-norm fallback"
            ),
            "eligible_pool": (
                "every component outside the target group; target components and "
                "already selected controls are excluded"
            ),
        },
        "groups": groups,
    }


def split_pairs(
    rows: list[dict[str, Any]], *, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pair_groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        pair_groups[str(row["pair_id"])].add(str(row["template_group"]))
    bad = {pair_id: values for pair_id, values in pair_groups.items() if len(values) != 1}
    if bad:
        raise ValueError(f"pair IDs span template groups: {bad}")

    by_template: dict[str, list[str]] = defaultdict(list)
    for pair_id, values in pair_groups.items():
        by_template[next(iter(values))].append(pair_id)
    discovery_ids: set[str] = set()
    validation_ids: set[str] = set()
    allocations: dict[str, Any] = {}
    for template_group in sorted(by_template):
        ordered = sorted(
            by_template[template_group],
            key=lambda pair_id: stable_key(seed, "split", template_group, pair_id),
        )
        if len(ordered) % 2:
            raise ValueError(f"template stratum has odd pair count: {template_group}")
        midpoint = len(ordered) // 2
        discovery = ordered[:midpoint]
        validation = ordered[midpoint:]
        discovery_ids.update(discovery)
        validation_ids.update(validation)
        allocations[template_group] = {
            "discovery_pair_ids": discovery,
            "validation_pair_ids": validation,
        }
    if discovery_ids & validation_ids:
        raise ValueError("discovery and validation pair IDs overlap")
    discovery_rows = [row for row in rows if str(row["pair_id"]) in discovery_ids]
    validation_rows = [row for row in rows if str(row["pair_id"]) in validation_ids]
    return (
        discovery_rows,
        validation_rows,
        {
            "seed": seed,
            "method": ("within-template-group SHA-256 ordering, exact 50/50 pair split"),
            "discovery_pair_count": len(discovery_ids),
            "validation_pair_count": len(validation_ids),
            "pair_overlap_count": 0,
            "template_allocations": allocations,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--task3", type=Path, required=True)
    parser.add_argument("--component-inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1707)
    args = parser.parse_args()

    development_rows = load_jsonl(args.development)
    task3_rows = load_jsonl(args.task3)
    discovery_rows, validation_rows, split = split_pairs(development_rows, seed=args.seed)
    task3_pair_ids = {str(row["pair_id"]) for row in task3_rows}
    development_pair_ids = {str(row["pair_id"]) for row in development_rows}
    if task3_pair_ids & development_pair_ids:
        raise ValueError("task-3 pair IDs overlap development pair IDs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    discovery_path = args.output_dir / "discovery_episodes.jsonl"
    validation_path = args.output_dir / "validation_episodes.jsonl"
    write_jsonl(discovery_path, discovery_rows)
    write_jsonl(validation_path, validation_rows)

    inventory_rows = load_jsonl(args.component_inventory)
    inventory_sha256 = sha256_file(args.component_inventory)
    group_plan = build_group_plan(inventory_rows, seed=args.seed, inventory_sha256=inventory_sha256)
    group_plan_path = args.output_dir / "hierarchy_group_plan.json"
    group_plan_path.write_text(
        json.dumps(group_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    split_manifest = {
        "schema": "bonsai8b_authz_nested_split_v1",
        "source_development_path": str(args.development),
        "source_development_sha256": sha256_file(args.development),
        "task3_path_local_only_never_stage_on_discovery_pod": str(args.task3),
        "task3_sha256": sha256_file(args.task3),
        "task3_pair_count": len(task3_pair_ids),
        "task3_pair_overlap_with_development": 0,
        "discovery_path": str(discovery_path),
        "discovery_sha256": sha256_file(discovery_path),
        "discovery_rows": len(discovery_rows),
        "validation_path": str(validation_path),
        "validation_sha256": sha256_file(validation_path),
        "validation_rows": len(validation_rows),
        "group_plan_path": str(group_plan_path),
        "group_plan_sha256": sha256_file(group_plan_path),
        "component_inventory_sha256": inventory_sha256,
        **split,
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
