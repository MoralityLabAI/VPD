from __future__ import annotations

from typing import Any

from experiments.authz_mechinterp import compute_vpd_effect_shard as preserved_worker


SCHEMA = "sch.vpd-effect-worker-wrapper.v2"
CORRECTION = "convert_hex_countsketch_seed_to_integer_before_modulo"
_PRESERVED_STABLE_HASH = preserved_worker.stable_hash


def corrected_stable_hash(seed: int, *parts: str) -> Any:
    digest = _PRESERVED_STABLE_HASH(seed, *parts)
    if parts and parts[0] == "category-effect-countsketch":
        return int(digest, 16)
    return digest


def main() -> None:
    preserved_worker.stable_hash = corrected_stable_hash
    preserved_worker.main()


if __name__ == "__main__":
    main()
