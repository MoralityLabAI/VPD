"""Resume a public model download and verify its published SHA-256."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--bytes", type=int, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if args.output.exists():
        if digest(args.output) != args.sha256:
            raise SystemExit("Existing destination has a different hash; left unchanged.")
        print(json.dumps({"status": "already_verified", "path": str(args.output)}), flush=True)
        return
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == args.bytes:
            break
        if offset > args.bytes:
            raise SystemExit("Partial file exceeds expected length; left unchanged.")
        if shutil.disk_usage(args.output.parent).free < args.bytes - offset + (2 << 30):
            raise SystemExit("Download would leave less than 2 GiB free.")
        request = urllib.request.Request(args.url, headers={"User-Agent": "VPD-Nanbeige-Pilot/1.0"})
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status != 206:
                    raise RuntimeError("Server did not honor resume range")
                if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Resume range differs from local partial")
                last_report = 0.0
                with partial.open("ab") as output:
                    while block := response.read(4 << 20):
                        output.write(block)
                        offset += len(block)
                        if offset > args.bytes:
                            raise RuntimeError("Server exceeded expected download length")
                        now = time.monotonic()
                        if now - last_report >= 15:
                            print(json.dumps({"status": "downloading", "bytes": offset,
                                              "total": args.bytes, "percent": round(100 * offset / args.bytes, 1)}), flush=True)
                            last_report = now
            if offset != args.bytes:
                raise RuntimeError("Incomplete response")
            break
        except Exception as exc:
            print(json.dumps({"status": "retry", "attempt": attempt + 1, "error_type": type(exc).__name__}), flush=True)
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 16))
    actual = digest(partial)
    if actual != args.sha256:
        raise SystemExit("SHA-256 mismatch; unverified partial retained.")
    partial.replace(args.output)
    receipt = {"status": "verified", "path": str(args.output), "url": args.url,
               "bytes": args.bytes, "sha256": actual}
    args.output.with_suffix(args.output.suffix + ".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
