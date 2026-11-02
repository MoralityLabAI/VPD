from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from experiments.authz_mechinterp.cluster_vpd_for_sae import stable_hash
from experiments.authz_mechinterp.generate_activation_category_pack_v2 import (
    generate_activation_category_episodes_v2,
)


ARTIFACT_VERSION = "activation-category-capacity-v3"


def _version_identity(row: dict[str, Any], *, seed: int) -> dict[str, Any]:
    result = deepcopy(row)
    old_pair_id = str(result["pair_id"])
    suffix = old_pair_id.removeprefix("semantic-v2-")
    pair_id = f"semantic-capacity-v3-s{seed}-{suffix}"
    condition = str(result["condition"])
    old_reference = str(result["activation_category_pack"]["case_reference"])
    reference = "ac-cap-v3-" + stable_hash(
        seed, "capacity-case-reference", pair_id
    )[:20]
    for message in result["messages"]:
        message["content"] = str(message["content"]).replace(
            old_reference, reference
        )
    result["pair_id"] = pair_id
    result["episode_id"] = f"{pair_id}-{condition}"
    result["activation_category_pack"]["case_reference"] = reference
    result["activation_category_pack"]["artifact_version"] = ARTIFACT_VERSION
    result["activation_category_pack"]["predecessor_artifact_version"] = (
        "activation-category-pack-v2"
    )
    result["activation_category_pack"]["identity_namespace_includes_seed"] = True
    return result


def generate_capacity_episodes_v3(
    *, seed: int = 320829, pairs_per_family: int = 143
) -> Iterable[dict[str, Any]]:
    if pairs_per_family < 2:
        raise ValueError("capacity pack needs at least two pairs per semantic family")
    for row in generate_activation_category_episodes_v2(
        seed=seed, pairs_per_family=pairs_per_family
    ):
        yield _version_identity(row, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a capacity-sized, identity-disjoint activation-category pack."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=320829)
    parser.add_argument("--pairs-per-family", type=int, default=143)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in generate_capacity_episodes_v3(
            seed=args.seed, pairs_per_family=args.pairs_per_family
        ):
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
