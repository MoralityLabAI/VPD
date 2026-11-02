"""Parallel ranged download of a public checkpoint, with a required final hash."""
import argparse
import concurrent.futures
import hashlib
import json
import shutil
import threading
import time
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("url")
p.add_argument("output", type=Path)
p.add_argument("--sha256", required=True)
p.add_argument("--bytes", type=int, required=True)
p.add_argument("--workers", type=int, default=4)
args = p.parse_args()
output = args.output.resolve()
partial = output.with_suffix(output.suffix + ".range.partial")
state_path = output.with_suffix(output.suffix + ".ranges.json")
sequential = output.with_suffix(output.suffix + ".partial")
if output.exists():
    raise SystemExit("Destination already exists; left unchanged")
if state_path.exists():
    state = json.loads(state_path.read_text())
    if (state["url"], state["bytes"], state["sha256"]) != (args.url, args.bytes, args.sha256):
        raise SystemExit("Saved range state does not match requested checkpoint")
else:
    if partial.exists():
        raise SystemExit("Partial range file exists without state; left unchanged")
    offset = sequential.stat().st_size if sequential.exists() else 0
    if not 0 <= offset <= args.bytes:
        raise SystemExit("Invalid existing prefix length")
    if shutil.disk_usage(output.parent).free < args.bytes - offset + (2 << 30):
        raise SystemExit("Insufficient free disk space")
    if sequential.exists():
        sequential.rename(partial)
    else:
        partial.touch()
    state = {"url": args.url, "bytes": args.bytes, "sha256": args.sha256,
             "prefix": offset, "chunk_bytes": 64 << 20, "complete": []}
    state_path.write_text(json.dumps(state))
    with partial.open("r+b") as stream:
        stream.truncate(args.bytes)
lock = threading.Lock()
last_report = 0


def fetch(bounds):
    global last_report
    start, end = bounds
    for attempt in range(5):
        try:
            request = urllib.request.Request(args.url, headers={
                "Range": f"bytes={start}-{end}", "User-Agent": "VPD-Nanbeige-Pilot/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                expected = f"bytes {start}-{end}/{args.bytes}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise RuntimeError("Server did not honor exact requested range")
                with partial.open("r+b") as stream:
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        block = response.read(min(4 << 20, remaining))
                        if not block:
                            raise RuntimeError("Truncated range response")
                        stream.write(block)
                        remaining -= len(block)
                    stream.flush()
            with lock:
                state["complete"].append(start)
                temporary = state_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(state))
                temporary.replace(state_path)
                done = state["prefix"] + sum(min(state["chunk_bytes"], args.bytes - i) for i in state["complete"])
                if time.monotonic() - last_report >= 15 or done == args.bytes:
                    print(json.dumps({"status": "downloading", "bytes": done,
                                      "total": args.bytes, "percent": round(100 * done / args.bytes, 1)}), flush=True)
                    last_report = time.monotonic()
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


ranges = [(start, min(start + state["chunk_bytes"], args.bytes) - 1)
          for start in range(state["prefix"], args.bytes, state["chunk_bytes"])
          if start not in state["complete"]]
with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
    for _ in pool.map(fetch, ranges):
        pass
h = hashlib.sha256()
with partial.open("rb") as stream:
    for block in iter(lambda: stream.read(8 << 20), b""):
        h.update(block)
if partial.stat().st_size != args.bytes or h.hexdigest() != args.sha256:
    raise SystemExit("Final checkpoint hash or length mismatch; unverified partial retained")
partial.replace(output)
receipt = {"status": "verified", "path": str(output), "url": args.url,
           "bytes": args.bytes, "sha256": h.hexdigest()}
output.with_suffix(output.suffix + ".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
