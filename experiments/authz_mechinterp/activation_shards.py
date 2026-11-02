from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from experiments.authz_mechinterp.cluster_vpd_for_sae import sha256_file


SHARDED_ACTIVATION_SCHEMA = "sch.vpd-sharded-activation-artifact.v1"


class ActivationShardWriter:
    def __init__(
        self,
        *,
        output_dir: Path,
        key: str,
        hidden_size: int,
        dtype: np.dtype[Any],
        shard_rows: int,
    ) -> None:
        if hidden_size < 1 or shard_rows < 1:
            raise ValueError("activation shard dimensions must be positive")
        self.output_dir = output_dir
        self.key = key
        self.hidden_size = hidden_size
        self.dtype = np.dtype(dtype)
        self.shard_rows = shard_rows
        self.relative_dir = Path("activation_shards") / key
        self.directory = output_dir / self.relative_dir
        self.directory.mkdir(parents=True, exist_ok=True)
        self._pieces: list[np.ndarray] = []
        self._buffered_rows = 0
        self._total_rows = 0
        self._entries: list[dict[str, Any]] = []

    def append(self, matrix: np.ndarray) -> None:
        value = np.asarray(matrix, dtype=self.dtype)
        if value.ndim != 2 or value.shape[1] != self.hidden_size:
            raise ValueError("activation shard append has the wrong matrix shape")
        offset = 0
        while offset < len(value):
            take = min(self.shard_rows - self._buffered_rows, len(value) - offset)
            self._pieces.append(np.ascontiguousarray(value[offset : offset + take]))
            self._buffered_rows += take
            offset += take
            if self._buffered_rows == self.shard_rows:
                self._flush()

    def _flush(self) -> None:
        if self._buffered_rows == 0:
            return
        matrix = np.concatenate(self._pieces, axis=0)
        if matrix.shape != (self._buffered_rows, self.hidden_size):
            raise ValueError("activation shard buffer shape mismatch")
        shard_index = len(self._entries)
        relative_path = self.relative_dir / f"part-{shard_index:05d}.npy"
        path = self.output_dir / relative_path
        with path.open("wb") as handle:
            np.save(handle, matrix, allow_pickle=False)
        start_row = self._total_rows
        self._total_rows += len(matrix)
        self._entries.append(
            {
                "path": relative_path.as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "start_row": start_row,
                "end_row_exclusive": self._total_rows,
                "rows": len(matrix),
            }
        )
        self._pieces.clear()
        self._buffered_rows = 0

    def finalize(self) -> dict[str, Any]:
        self._flush()
        return {
            "dtype": self.dtype.name,
            "shape": [self._total_rows, self.hidden_size],
            "rows_per_full_shard": self.shard_rows,
            "shards": list(self._entries),
        }


def build_sharded_activation_artifact(
    key_entries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not key_entries:
        raise ValueError("sharded activation artifact has no keys")
    row_counts = {int(entry["shape"][0]) for entry in key_entries.values()}
    if len(row_counts) != 1:
        raise ValueError("sharded activation keys have different row counts")
    return {
        "schema": SHARDED_ACTIVATION_SCHEMA,
        "format": "npy_row_shards",
        "rows": next(iter(row_counts)),
        "activation_keys": sorted(key_entries),
        "keys": {key: key_entries[key] for key in sorted(key_entries)},
    }


def verify_sharded_activation_artifact(
    output_dir: Path, artifact: dict[str, Any]
) -> dict[str, Any]:
    if artifact.get("schema") != SHARDED_ACTIVATION_SCHEMA:
        raise ValueError("unsupported sharded activation artifact schema")
    keys = artifact.get("keys")
    if not isinstance(keys, dict) or sorted(keys) != artifact.get("activation_keys"):
        raise ValueError("sharded activation key inventory differs")
    expected_rows = int(artifact.get("rows", -1))
    hidden_sizes: set[int] = set()
    shard_total = 0
    for key, entry in sorted(keys.items()):
        shape = [int(value) for value in entry["shape"]]
        if len(shape) != 2 or shape[0] != expected_rows:
            raise ValueError(f"sharded activation shape differs: {key}")
        hidden_sizes.add(shape[1])
        expected_start = 0
        for shard in entry["shards"]:
            path = output_dir / str(shard["path"])
            if sha256_file(path) != shard["sha256"] or path.stat().st_size != int(
                shard["bytes"]
            ):
                raise ValueError(f"activation shard differs: {shard['path']}")
            if int(shard["start_row"]) != expected_start:
                raise ValueError("activation shard row ranges are not contiguous")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            try:
                rows = int(shard["rows"])
                if array.shape != (rows, shape[1]) or array.dtype.name != entry["dtype"]:
                    raise ValueError(
                        f"activation shard array contract differs: {shard['path']}"
                    )
            finally:
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()
            expected_start += rows
            if int(shard["end_row_exclusive"]) != expected_start:
                raise ValueError("activation shard row endpoint differs")
            shard_total += 1
        if expected_start != expected_rows:
            raise ValueError(f"activation shard partition is incomplete: {key}")
    if len(hidden_sizes) != 1:
        raise ValueError("sharded activation keys have different hidden sizes")
    return {
        "schema": "sch.vpd-sharded-activation-verification.v1",
        "status": "pass",
        "rows": expected_rows,
        "activation_keys": len(keys),
        "hidden_size": next(iter(hidden_sizes)),
        "shards": shard_total,
    }


class ShardedActivationAccessor:
    def __init__(
        self,
        output_dir: Path,
        artifact: dict[str, Any],
        key: str,
        *,
        verify_artifact: bool = True,
    ) -> None:
        if verify_artifact:
            verify_sharded_activation_artifact(output_dir, artifact)
        if key not in artifact["keys"]:
            raise KeyError(f"activation key not found: {key}")
        self.output_dir = output_dir
        self.entry = artifact["keys"][key]
        self.shape = tuple(int(value) for value in self.entry["shape"])
        self._arrays = [
            np.load(
                output_dir / str(shard["path"]),
                mmap_mode="r",
                allow_pickle=False,
            )
            for shard in self.entry["shards"]
        ]
        self._ranges = [
            (int(shard["start_row"]), int(shard["end_row_exclusive"]))
            for shard in self.entry["shards"]
        ]

    def finite_mask(self) -> np.ndarray:
        return np.concatenate(
            [np.isfinite(array).all(axis=1) for array in self._arrays]
        )

    def read_rows(self, indices: np.ndarray) -> np.ndarray:
        requested = np.asarray(indices, dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("activation row indices must be one-dimensional")
        if len(requested) == 0:
            return np.empty((0, self.shape[1]), dtype=np.float32)
        if requested.min() < 0 or requested.max() >= self.shape[0]:
            raise IndexError("activation row index is outside the shard artifact")
        result = np.empty((len(requested), self.shape[1]), dtype=np.float32)
        found = np.zeros(len(requested), dtype=bool)
        for array, (start, end) in zip(self._arrays, self._ranges, strict=True):
            positions = np.flatnonzero((requested >= start) & (requested < end))
            if len(positions) == 0:
                continue
            local = requested[positions] - start
            result[positions] = np.asarray(array[local], dtype=np.float32)
            found[positions] = True
        if not found.all():
            raise ValueError("activation shard row partition has a gap")
        return result

    def mean_rows(self, indices: np.ndarray, *, chunk_rows: int = 4096) -> np.ndarray:
        requested = np.asarray(indices, dtype=np.int64)
        if len(requested) == 0:
            raise ValueError("cannot average an empty activation selection")
        total = np.zeros(self.shape[1], dtype=np.float64)
        for start in range(0, len(requested), chunk_rows):
            batch = self.read_rows(requested[start : start + chunk_rows])
            total += batch.sum(axis=0, dtype=np.float64)
        return (total / len(requested)).astype(np.float32)

    def close(self) -> None:
        for array in self._arrays:
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._arrays.clear()

    def __enter__(self) -> "ShardedActivationAccessor":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()
