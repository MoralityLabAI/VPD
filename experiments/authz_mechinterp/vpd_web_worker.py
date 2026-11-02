from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    content_sha256,
    sha256_file,
    write_json,
)


class ExchangeHttpError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"exchange returned HTTP {status}: {message}")
        self.status = status


def normalize_exchange_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("exchange URL must be an absolute http:// or https:// URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("exchange URL must not embed credentials")
    return value.rstrip("/")


def request(
    exchange_url: str,
    path: str,
    *,
    method: str = "GET",
    value: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> tuple[bytes, dict[str, str]]:
    url = normalize_exchange_url(exchange_url) + path
    data = None if value is None else json.dumps(value, separators=(",", ":")).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read(), {name.lower(): val for name, val in response.headers.items()}
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            payload = json.loads(body.decode("utf-8"))
            message = str(payload.get("error", payload))
        except Exception:
            message = body.decode("utf-8", errors="replace")
        raise ExchangeHttpError(exc.code, message) from exc


def request_json(
    exchange_url: str,
    path: str,
    *,
    method: str = "GET",
    value: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    data, _ = request(exchange_url, path, method=method, value=value, timeout=timeout)
    result = json.loads(data.decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError("exchange JSON response was not an object")
    return result


def atomic_write_verified(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"existing synchronized file has different content: {path.name}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix="sync-", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(
                    f"existing synchronized file has different content: {path.name}"
                )
    finally:
        temporary_path.unlink(missing_ok=True)


def sync_packets(
    *,
    exchange_url: str,
    worker_id: str,
    output_dir: Path,
    timeout: float = 60.0,
) -> dict[str, Any]:
    network = request_json(exchange_url, "/v1/network", timeout=timeout)
    quoted_worker = urllib.parse.quote(worker_id, safe="")
    work = request_json(
        exchange_url, f"/v1/workers/{quoted_worker}/work", timeout=timeout
    )
    if str(work.get("network_id")) != str(network.get("network_id")):
        raise ValueError("worker view belongs to a different exchange network")
    packet_rows: list[dict[str, Any]] = []
    for row in work.get("packets", []):
        packet_id = str(row["packet_id"])
        digest = str(row["packet_sha256"])
        object_path = str(row["packet_object_url"])
        expected_path = f"/v1/objects/sha256/{digest}"
        if object_path != expected_path:
            raise ValueError("worker view returned a non-canonical packet object URL")
        packet_bytes, headers = request(exchange_url, object_path, timeout=timeout)
        if hashlib.sha256(packet_bytes).hexdigest() != digest:
            raise ValueError(f"downloaded packet hash mismatch: {packet_id}")
        packet = json.loads(packet_bytes.decode("utf-8"))
        if str(packet.get("packet_id")) != packet_id:
            raise ValueError("downloaded packet content identity mismatch")
        etag = headers.get("etag")
        if etag is not None and etag != f'"sha256:{digest}"':
            raise ValueError("packet object ETag does not match its SHA-256 address")
        path = output_dir / "packets" / f"{packet_id}.json"
        atomic_write_verified(path, packet_bytes)
        packet_rows.append(
            {
                "packet_id": packet_id,
                "cluster_id": str(row["cluster_id"]),
                "packet_sha256": digest,
                "path": path.relative_to(output_dir).as_posix(),
                "effect_status": str(row["effect_status"]),
                "sae_status": str(row["sae_status"]),
            }
        )
    packet_rows.sort(key=lambda row: row["packet_id"])
    body = {
        "schema": "sch.vpd-web-worker-sync.v1",
        "network_id": str(network["network_id"]),
        "worker_view_id": str(work["view_id"]),
        "worker_id": worker_id,
        "packets": packet_rows,
    }
    manifest = {**body, "sync_id": content_sha256(body)}
    write_json(output_dir / "sync_manifest.json", manifest)
    return manifest


def load_signed_json(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError(f"signed document contains null bytes: {path.name}")
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("signed document must be a JSON object")
    if "result_id" not in value or "signature" not in value:
        raise ValueError("document must be signed before web submission")
    return value


def submit_effect(
    *,
    exchange_url: str,
    receipt_path: Path,
    shard_path: Path,
    timeout: float = 300.0,
) -> dict[str, Any]:
    receipt = load_signed_json(receipt_path)
    shard_bytes = shard_path.read_bytes()
    if sha256_file(shard_path) != str(receipt.get("shard_sha256")):
        raise ValueError("local effect shard does not match its signed receipt")
    if shard_path.name != str(receipt.get("shard_filename")):
        raise ValueError("local effect shard filename does not match its signed receipt")
    return request_json(
        exchange_url,
        "/v1/submissions/effect",
        method="POST",
        value={
            "schema": "sch.vpd-effect-web-submission.v1",
            "receipt": receipt,
            "shard_base64": base64.b64encode(shard_bytes).decode("ascii"),
        },
        timeout=timeout,
    )


def submit_sae(
    *,
    exchange_url: str,
    result_path: Path,
    timeout: float = 60.0,
) -> dict[str, Any]:
    result = load_signed_json(result_path)
    return request_json(
        exchange_url,
        "/v1/submissions/sae",
        method="POST",
        value={"schema": "sch.vpd-sae-web-submission.v1", "result": result},
        timeout=timeout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize VPD packets and submit signed worker artifacts to a web exchange."
    )
    parser.add_argument("--exchange-url", required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--sync", action="store_true")
    actions.add_argument("--submit-effect-receipt", type=Path)
    actions.add_argument("--submit-sae-result", type=Path)
    parser.add_argument("--worker-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--effect-shard", type=Path)
    args = parser.parse_args()
    if args.sync:
        if args.worker_id is None or args.output_dir is None:
            parser.error("--sync requires --worker-id and --output-dir")
        result = sync_packets(
            exchange_url=args.exchange_url,
            worker_id=args.worker_id,
            output_dir=args.output_dir,
            timeout=args.timeout,
        )
    elif args.submit_effect_receipt is not None:
        if args.effect_shard is None:
            parser.error("effect submission requires --effect-shard")
        result = submit_effect(
            exchange_url=args.exchange_url,
            receipt_path=args.submit_effect_receipt,
            shard_path=args.effect_shard,
            timeout=max(args.timeout, 300.0),
        )
    else:
        result = submit_sae(
            exchange_url=args.exchange_url,
            result_path=args.submit_sae_result,
            timeout=args.timeout,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
