from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import re
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from experiments.authz_mechinterp.cluster_vpd_for_sae import (
    canonical_bytes,
    content_sha256,
    sha256_file,
)
from experiments.authz_mechinterp.merge_vpd_effect_shards import (
    merge_effect_shards,
    validate_shard_receipt,
)
from experiments.authz_mechinterp.merge_vpd_sae_results import (
    load_packets,
    load_policy,
    load_registry,
    merge_results,
    validate_result,
)


class SubmissionConflict(ValueError):
    """A worker already submitted different signed content for this packet."""


SAFE_WORKER_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
SHA256_ADDRESS = re.compile(r"^[0-9a-f]{64}$")


class VpdWebExchange:
    """Append-only web exchange for content-addressed VPD work and results.

    The exchange is deliberately not a scheduler. Packet assignment is frozen in
    the bundle, and admission depends only on packet content, the public worker
    registry, signatures, and the scientific validation contracts. Consequently,
    mirrors that admit the same signed objects derive the same snapshot ID.
    """

    def __init__(
        self,
        *,
        bundle_dir: Path,
        registry_path: Path,
        state_dir: Path,
        policy_path: Path | None = None,
        maximum_shard_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if maximum_shard_bytes <= 0:
            raise ValueError("maximum_shard_bytes must be positive")
        self.bundle_dir = bundle_dir.resolve()
        self.registry_path = registry_path.resolve()
        self.state_dir = state_dir.resolve()
        self.policy = load_policy(policy_path)
        self.maximum_shard_bytes = int(maximum_shard_bytes)
        self.packets, self.packet_hashes = load_packets(self.bundle_dir)
        self.workers = load_registry(self.registry_path)
        unsafe_workers = sorted(
            worker_id for worker_id in self.workers if SAFE_WORKER_ID.fullmatch(worker_id) is None
        )
        if unsafe_workers:
            raise ValueError(f"worker registry contains unsafe web IDs: {unsafe_workers}")
        unsafe_packets = sorted(
            packet_id for packet_id in self.packets if SHA256_ADDRESS.fullmatch(packet_id) is None
        )
        if unsafe_packets:
            raise ValueError("bundle contains packet IDs that are not SHA-256 addresses")
        assigned_workers = {
            str(packet["distribution"].get("assigned_worker"))
            for packet in self.packets.values()
        }
        unavailable_workers = sorted(
            worker_id
            for worker_id in assigned_workers
            if worker_id not in self.workers
            or not bool(self.workers[worker_id].get("enabled", False))
        )
        if unavailable_workers:
            raise ValueError(
                f"assigned packet workers are absent or disabled: {unavailable_workers}"
            )
        enabled_key_owners: dict[str, str] = {}
        for worker_id, worker in self.workers.items():
            if not bool(worker.get("enabled", False)):
                continue
            public_key = str(worker["public_key_base64"])
            prior_owner = enabled_key_owners.get(public_key)
            if prior_owner is not None:
                raise ValueError(
                    f"enabled workers share one signing key: {prior_owner}, {worker_id}"
                )
            enabled_key_owners[public_key] = worker_id
        self._lock = threading.RLock()
        self._objects_dir = self.state_dir / "objects" / "sha256"
        self._submissions_dir = self.state_dir / "submissions"
        self._tmp_dir = self.state_dir / "tmp"
        for path in (self._objects_dir, self._submissions_dir, self._tmp_dir):
            path.mkdir(parents=True, exist_ok=True)

        manifest_path = self.bundle_dir / "manifest.json"
        self.bundle_manifest_sha256 = sha256_file(manifest_path)
        self.registry_sha256 = sha256_file(self.registry_path)
        self.policy_sha256 = content_sha256(self.policy)
        self._source_objects: dict[str, Path] = {
            self.bundle_manifest_sha256: manifest_path,
            self.registry_sha256: self.registry_path,
        }
        self._atomic_write_once(
            self._objects_dir / self.policy_sha256, canonical_bytes(self.policy)
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        packet_entries = {
            str(row["packet_id"]): row for row in manifest["artifacts"]["packets"]
        }
        self._packet_paths: dict[str, Path] = {}
        for packet_id, digest in self.packet_hashes.items():
            path = self.bundle_dir / str(packet_entries[packet_id]["packet_path"])
            self._packet_paths[packet_id] = path
            self._source_objects[digest] = path

        network_body = {
            "schema": "sch.vpd-web-network.v1",
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "worker_registry_sha256": self.registry_sha256,
            "sae_policy_sha256": self.policy_sha256,
            "assignment_algorithm": str(
                manifest["decentralized_execution"]["assignment_algorithm"]
            ),
            "packet_count": len(self.packets),
            "protocols": [
                "sch.vpd-effect-web-submission.v1",
                "sch.vpd-sae-web-submission.v1",
            ],
            "maximum_shard_bytes": self.maximum_shard_bytes,
        }
        self._network = {**network_body, "network_id": content_sha256(network_body)}

    def network(self) -> dict[str, Any]:
        self.verify_sources()
        return dict(self._network)

    def verify_sources(self) -> None:
        if sha256_file(self.bundle_dir / "manifest.json") != self.bundle_manifest_sha256:
            raise ValueError("bundle manifest changed after exchange initialization")
        if sha256_file(self.registry_path) != self.registry_sha256:
            raise ValueError("worker registry changed after exchange initialization")
        for packet_id, path in self._packet_paths.items():
            if sha256_file(path) != self.packet_hashes[packet_id]:
                raise ValueError(f"packet changed after exchange initialization: {packet_id}")

    def packet_bytes(self, packet_id: str) -> tuple[str, bytes]:
        path = self._packet_paths.get(packet_id)
        if path is None:
            raise KeyError(f"unknown packet: {packet_id}")
        digest = self.packet_hashes[packet_id]
        data = path.read_bytes()
        if sha256_file(path) != digest:
            raise ValueError("packet changed after exchange initialization")
        return digest, data

    def object_bytes(self, digest: str) -> bytes:
        if SHA256_ADDRESS.fullmatch(digest) is None:
            raise KeyError("invalid SHA-256 object address")
        source = self._source_objects.get(digest)
        if source is not None:
            data = source.read_bytes()
            if sha256_file(source) != digest:
                raise ValueError("source object changed after exchange initialization")
            return data
        path = self._objects_dir / digest
        if not path.is_file():
            raise KeyError(f"unknown object: {digest}")
        data = path.read_bytes()
        if sha256_file(path) != digest:
            raise ValueError("stored object hash mismatch")
        return data

    def _record_path(self, kind: str, packet_id: str, worker_id: str) -> Path:
        return self._submissions_dir / kind / packet_id / f"{worker_id}.json"

    def _read_record(
        self, kind: str, packet_id: str, worker_id: str
    ) -> dict[str, Any] | None:
        path = self._record_path(kind, packet_id, worker_id)
        if not path.is_file():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        admission_id = str(record.pop("admission_id"))
        if content_sha256(record) != admission_id:
            raise ValueError("stored admission record content address mismatch")
        record["admission_id"] = admission_id
        return record

    @staticmethod
    def _atomic_write_once(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != data:
                raise SubmissionConflict(f"immutable path already contains different bytes: {path.name}")
            return
        descriptor, temporary_name = tempfile.mkstemp(prefix="admit-", dir=path.parent)
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
                    raise SubmissionConflict(
                        f"immutable path already contains different bytes: {path.name}"
                    )
        finally:
            temporary_path.unlink(missing_ok=True)

    def _admit(
        self,
        *,
        kind: str,
        packet_id: str,
        worker_id: str,
        result_id: str,
        signed_document: dict[str, Any],
        extra_objects: dict[str, bytes] | None = None,
        accepted_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        document_bytes = canonical_bytes(signed_document)
        receipt_sha256 = hashlib.sha256(document_bytes).hexdigest()
        objects = {receipt_sha256: document_bytes, **(extra_objects or {})}
        record = {
            "schema": "sch.vpd-web-admission-record.v1",
            "kind": kind,
            "packet_id": packet_id,
            "worker_id": worker_id,
            "result_id": result_id,
            "receipt_sha256": receipt_sha256,
            "object_sha256": sorted(objects),
            "accepted_summary": accepted_summary or {},
        }
        record["admission_id"] = content_sha256(record)
        record_path = self._record_path(kind, packet_id, worker_id)
        with self._lock:
            prior = self._read_record(kind, packet_id, worker_id)
            if prior is not None:
                if str(prior["result_id"]) != result_id:
                    raise SubmissionConflict(
                        f"worker already submitted a different {kind} result for packet"
                    )
                if prior != record:
                    raise SubmissionConflict(
                        f"result ID was reused with different {kind} admission content"
                    )
                status = "idempotent"
            else:
                for digest, data in objects.items():
                    if hashlib.sha256(data).hexdigest() != digest:
                        raise ValueError("object key does not match its SHA-256 content")
                    self._atomic_write_once(self._objects_dir / digest, data)
                self._atomic_write_once(record_path, canonical_bytes(record))
                status = "accepted"
        return {
            "schema": "sch.vpd-web-admission-response.v1",
            "status": status,
            "kind": kind,
            "packet_id": packet_id,
            "worker_id": worker_id,
            "result_id": result_id,
            "admission_id": record["admission_id"],
            "snapshot_id": self.snapshot()["snapshot_id"],
        }

    def submit_effect(self, submission: dict[str, Any]) -> dict[str, Any]:
        self.verify_sources()
        if submission.get("schema") != "sch.vpd-effect-web-submission.v1":
            raise ValueError("unsupported effect web submission schema")
        receipt = submission.get("receipt")
        if not isinstance(receipt, dict):
            raise ValueError("effect submission receipt must be an object")
        encoded = submission.get("shard_base64")
        if not isinstance(encoded, str):
            raise ValueError("effect submission shard_base64 must be a string")
        try:
            shard_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("effect shard is not valid base64") from exc
        if len(shard_bytes) > self.maximum_shard_bytes:
            raise ValueError("effect shard exceeds the exchange size limit")
        shard_filename = receipt.get("shard_filename")
        if not isinstance(shard_filename, str) or Path(shard_filename).name != shard_filename:
            raise ValueError("effect shard filename must be a basename")
        with tempfile.TemporaryDirectory(dir=self._tmp_dir) as directory:
            shard_path = Path(directory) / shard_filename
            shard_path.write_bytes(shard_bytes)
            accepted, _, _, _, _, _, _ = validate_shard_receipt(
                receipt,
                shard_dir=Path(directory),
                packets=self.packets,
                packet_hashes=self.packet_hashes,
                workers=self.workers,
            )
        shard_sha256 = str(receipt["shard_sha256"])
        return self._admit(
            kind="effect",
            packet_id=str(receipt["packet_id"]),
            worker_id=str(receipt["worker_id"]),
            result_id=str(receipt["result_id"]),
            signed_document=receipt,
            extra_objects={shard_sha256: shard_bytes},
            accepted_summary=accepted,
        )

    def submit_sae(self, submission: dict[str, Any]) -> dict[str, Any]:
        self.verify_sources()
        if submission.get("schema") != "sch.vpd-sae-web-submission.v1":
            raise ValueError("unsupported SAE web submission schema")
        result = submission.get("result")
        if not isinstance(result, dict):
            raise ValueError("SAE submission result must be an object")
        accepted, candidates = validate_result(
            result,
            packets=self.packets,
            packet_hashes=self.packet_hashes,
            workers=self.workers,
            policy=self.policy,
        )
        summary = {
            **accepted,
            "sae_screen_passed": sum(
                bool(candidate["passes_sae_screen"]) for candidate in candidates
            ),
            "causal_gate_passed": sum(
                bool(candidate["passes_causal_gate"]) for candidate in candidates
            ),
        }
        return self._admit(
            kind="sae",
            packet_id=str(result["packet_id"]),
            worker_id=str(result["worker_id"]),
            result_id=str(result["result_id"]),
            signed_document=result,
            accepted_summary=summary,
        )

    def work_for(self, worker_id: str) -> dict[str, Any]:
        self.verify_sources()
        worker = self.workers.get(worker_id)
        if worker is None or not bool(worker.get("enabled", False)):
            raise KeyError(f"worker is absent or disabled: {worker_id}")
        packets: list[dict[str, Any]] = []
        for packet_id, packet in self.packets.items():
            if str(packet["distribution"].get("assigned_worker")) != worker_id:
                continue
            effect = self._read_record("effect", packet_id, worker_id)
            sae = self._read_record("sae", packet_id, worker_id)
            packets.append(
                {
                    "packet_id": packet_id,
                    "cluster_id": str(packet["cluster_id"]),
                    "packet_sha256": self.packet_hashes[packet_id],
                    "packet_object_url": f"/v1/objects/sha256/{self.packet_hashes[packet_id]}",
                    "effect_status": "submitted" if effect else "pending",
                    "sae_status": "submitted" if sae else "pending",
                }
            )
        packets.sort(key=lambda row: row["packet_id"])
        body = {
            "schema": "sch.vpd-web-worker-view.v1",
            "network_id": self._network["network_id"],
            "worker_id": worker_id,
            "packets": packets,
        }
        return {**body, "view_id": content_sha256(body)}

    def _records(self, kind: str) -> list[dict[str, Any]]:
        root = self._submissions_dir / kind
        records: list[dict[str, Any]] = []
        if root.exists():
            for path in root.glob("*/*.json"):
                record = self._read_record(kind, path.parent.name, path.stem)
                if record is not None:
                    records.append(record)
        records.sort(
            key=lambda row: (str(row["packet_id"]), str(row["worker_id"]), str(row["result_id"]))
        )
        return records

    def reduce_effects(
        self, output_dir: Path, *, require_complete: bool = True
    ) -> dict[str, Any]:
        self.verify_sources()
        records = self._records("effect")
        if not records:
            raise ValueError("exchange contains no admitted effect shards")
        receipt_paths: list[Path] = []
        shard_paths: dict[str, Path] = {}
        for record in records:
            receipt_path = self._objects_dir / str(record["receipt_sha256"])
            self.object_bytes(str(record["receipt_sha256"]))
            shard_sha256 = str(record["accepted_summary"]["shard_sha256"])
            shard_path = self._objects_dir / shard_sha256
            self.object_bytes(shard_sha256)
            receipt_paths.append(receipt_path)
            shard_paths[str(record["result_id"])] = shard_path
        return merge_effect_shards(
            bundle_dir=self.bundle_dir,
            registry_path=self.registry_path,
            receipt_paths=receipt_paths,
            shard_dir=None,
            output_dir=output_dir,
            require_complete=require_complete,
            shard_paths_by_result_id=shard_paths,
        )

    def reduce_sae(
        self, output_dir: Path, *, require_complete: bool = True
    ) -> dict[str, Any]:
        self.verify_sources()
        records = self._records("sae")
        if not records:
            raise ValueError("exchange contains no admitted SAE results")
        result_paths: list[Path] = []
        for record in records:
            digest = str(record["receipt_sha256"])
            self.object_bytes(digest)
            result_paths.append(self._objects_dir / digest)
        return merge_results(
            bundle_dir=self.bundle_dir,
            registry_path=self.registry_path,
            result_paths=result_paths,
            output_dir=output_dir,
            policy_path=self._objects_dir / self.policy_sha256,
            require_complete=require_complete,
        )

    def snapshot(self) -> dict[str, Any]:
        self.verify_sources()
        admissions: list[dict[str, Any]] = []
        for kind in ("effect", "sae"):
            for record in self._records(kind):
                admissions.append(
                    {
                        "kind": kind,
                        "packet_id": str(record["packet_id"]),
                        "worker_id": str(record["worker_id"]),
                        "result_id": str(record["result_id"]),
                        "receipt_sha256": str(record["receipt_sha256"]),
                        "object_sha256": sorted(str(value) for value in record["object_sha256"]),
                        "admission_id": str(record["admission_id"]),
                    }
                )
        admissions.sort(
            key=lambda row: (row["kind"], row["packet_id"], row["worker_id"], row["result_id"])
        )
        effect_packets = {row["packet_id"] for row in admissions if row["kind"] == "effect"}
        sae_packets = {row["packet_id"] for row in admissions if row["kind"] == "sae"}
        expected = set(self.packets)
        body = {
            "schema": "sch.vpd-web-snapshot.v1",
            "network_id": self._network["network_id"],
            "coverage": {
                "expected_packets": len(expected),
                "effect_packets": len(effect_packets),
                "effect_missing_packet_ids": sorted(expected - effect_packets),
                "sae_packets": len(sae_packets),
                "sae_missing_packet_ids": sorted(expected - sae_packets),
            },
            "admissions": admissions,
        }
        return {**body, "snapshot_id": content_sha256(body)}


def make_handler(exchange: VpdWebExchange) -> type[BaseHTTPRequestHandler]:
    class ExchangeHandler(BaseHTTPRequestHandler):
        server_version = "VpdWebExchange/1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status: int, value: Any) -> None:
            data = canonical_bytes(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_bytes(self, status: int, data: bytes, digest: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("ETag", f'"sha256:{digest}"')
            self.send_header("Cache-Control", "public, immutable")
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, exc: Exception) -> None:
            self._send_json(
                status,
                {
                    "schema": "sch.vpd-web-error.v1",
                    "status": status,
                    "error": str(exc),
                },
            )

        def do_GET(self) -> None:  # noqa: N802
            try:
                path = unquote(urlsplit(self.path).path)
                if path == "/v1/network":
                    self._send_json(HTTPStatus.OK, exchange.network())
                    return
                if path == "/v1/snapshot":
                    self._send_json(HTTPStatus.OK, exchange.snapshot())
                    return
                prefix = "/v1/objects/sha256/"
                if path.startswith(prefix):
                    digest = path[len(prefix) :]
                    self._send_bytes(HTTPStatus.OK, exchange.object_bytes(digest), digest)
                    return
                worker_prefix = "/v1/workers/"
                if path.startswith(worker_prefix) and path.endswith("/work"):
                    worker_id = path[len(worker_prefix) : -len("/work")]
                    if "/" in worker_id or not worker_id:
                        raise KeyError("invalid worker ID")
                    self._send_json(HTTPStatus.OK, exchange.work_for(worker_id))
                    return
                raise KeyError("unknown endpoint")
            except KeyError as exc:
                self._error(HTTPStatus.NOT_FOUND, exc)
            except ValueError as exc:
                self._error(HTTPStatus.BAD_REQUEST, exc)

        def do_POST(self) -> None:  # noqa: N802
            try:
                path = unquote(urlsplit(self.path).path)
                maximum_body = exchange.maximum_shard_bytes * 2 + 1024 * 1024
                length_text = self.headers.get("Content-Length")
                if length_text is None:
                    raise ValueError("Content-Length is required")
                length = int(length_text)
                if length < 0 or length > maximum_body:
                    raise ValueError("request body exceeds the exchange size limit")
                data = self.rfile.read(length)
                submission = json.loads(data.decode("utf-8"))
                if not isinstance(submission, dict):
                    raise ValueError("submission must be a JSON object")
                if path == "/v1/submissions/effect":
                    response = exchange.submit_effect(submission)
                elif path == "/v1/submissions/sae":
                    response = exchange.submit_sae(submission)
                else:
                    raise KeyError("unknown endpoint")
                status = (
                    HTTPStatus.CREATED
                    if response["status"] == "accepted"
                    else HTTPStatus.OK
                )
                self._send_json(status, response)
            except SubmissionConflict as exc:
                self._error(HTTPStatus.CONFLICT, exc)
            except KeyError as exc:
                self._error(HTTPStatus.NOT_FOUND, exc)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, exc)

    return ExchangeHandler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve immutable VPD packets and admit signed decentralized worker results."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--maximum-shard-mib", type=int, default=64)
    parser.add_argument("--reduce-effects", type=Path)
    parser.add_argument("--reduce-sae", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    exchange = VpdWebExchange(
        bundle_dir=args.bundle,
        registry_path=args.registry,
        state_dir=args.state_dir,
        policy_path=args.policy,
        maximum_shard_bytes=args.maximum_shard_mib * 1024 * 1024,
    )
    if args.reduce_effects is not None:
        print(
            json.dumps(
                exchange.reduce_effects(
                    args.reduce_effects, require_complete=not args.allow_incomplete
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.reduce_sae is not None:
        print(
            json.dumps(
                exchange.reduce_sae(
                    args.reduce_sae, require_complete=not args.allow_incomplete
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    server = ThreadingHTTPServer((args.host, args.port), make_handler(exchange))
    print(
        json.dumps(
            {
                "schema": "sch.vpd-web-listener.v1",
                "host": args.host,
                "port": server.server_address[1],
                "network_id": exchange.network()["network_id"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
