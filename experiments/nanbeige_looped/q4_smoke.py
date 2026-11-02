"""Bounded local Q4 smoke check, using official HF input token IDs.

This is a runtime/tokenizer check, not a reasoning benchmark or BF16 parity test.
The server is bound to localhost and is stopped when this process exits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from experiments.nanbeige_looped.preflight import MODEL_ID, REVISION, prompts

ROOT = Path(__file__).resolve().parents[2]
Q4_SHA256 = "01739fad13be318c5c8620f8de52731608bb750ae887e9445cc81cb4a189e45d"


def request(base: str, endpoint: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base + endpoint, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8096)
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--wait-for-download-minutes", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.examples <= len(prompts()):
        raise SystemExit("Invalid smoke example count")
    model = ROOT / "models" / "Nanbeige4.2-3B-Q4_K_M.gguf"
    deadline = time.monotonic() + 60 * args.wait_for_download_minutes
    while not model.exists():
        if time.monotonic() >= deadline:
            raise SystemExit("Verified Q4 file is not present; download must finish first")
        print(json.dumps({"status": "waiting_for_verified_download"}), flush=True)
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise SystemExit("Choose an empty output directory")
    if digest(model) != Q4_SHA256:
        raise SystemExit("Q4 checksum differs from the pinned file")
    from transformers import AutoTokenizer
    tokenizer_dir = ROOT / "models" / "Nanbeige4.2-3B-tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False, local_files_only=True)
    fast_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True, local_files_only=True)
    executable = ROOT / "tools" / "nanbeige" / "runtime" / "llama-server.exe"
    base = f"http://127.0.0.1:{args.port}"
    command = [str(executable), "-m", str(model), "-c", "1024", "-b", "128", "-ub", "128",
               "-ngl", "all", "--fit", "off", "-fa", "on", "-np", "1",
               "--host", "127.0.0.1", "--port", str(args.port)]
    started = time.monotonic()
    records = []
    with (args.output_dir / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError("llama-server exited; inspect server.log")
                try:
                    if request(base, "/health").get("status") == "ok":
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("llama-server did not become ready")
                time.sleep(1)
            print(json.dumps({"status": "q4_loaded", "context": 1024}), flush=True)
            for item in prompts()[:args.examples]:
                for thinking in (False, True):
                    rendered = tokenizer.apply_chat_template(
                        [{"role": "user", "content": item["text"]}], tokenize=False,
                        add_generation_prompt=True, enable_thinking=thinking,
                    )
                    ids = tokenizer.encode(rendered, add_special_tokens=False)
                    fast_ids = fast_tokenizer.encode(rendered, add_special_tokens=False)
                    # Observe conversion drift, but don't feed converted text tokenization to eval.
                    native_ids = request(base, "/tokenize", {"content": rendered,
                                         "add_special": False, "parse_special": True})["tokens"]
                    prefixed_ids = request(base, "/tokenize", {"content": " " + rendered,
                                           "add_special": False, "parse_special": True})["tokens"]
                    completion = request(base, "/completion", {
                        "prompt": ids, "n_predict": 128 if thinking else 16,
                        "temperature": 0, "seed": 42, "cache_prompt": False,
                        "return_tokens": True, "stream": False,
                    })
                    if not completion.get("tokens") or completion.get("truncated"):
                        raise RuntimeError("Smoke completion was empty or exceeded context")
                    row = {**item, "thinking": thinking, "rendered_prompt": rendered,
                           "prompt_token_ids": ids, "hf_fast_slow_match": fast_ids == ids,
                           "native_tokenizer_match": native_ids == ids,
                           "leading_space_tokenizer_match": prefixed_ids == ids,
                           "native_token_ids": native_ids, "leading_space_token_ids": prefixed_ids,
                           "hf_decoded_output": tokenizer.decode(completion["tokens"], skip_special_tokens=False),
                           "completion": completion}
                    records.append(row)
                    with (args.output_dir / "completions.jsonl").open("a", encoding="utf-8") as output:
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(json.dumps({"status": "q4_completed", "id": item["id"], "thinking": thinking,
                                      "tokens": len(completion["tokens"]), "stop_type": completion.get("stop_type")}), flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    result = {"status": "q4_smoke_passed", "source_model": MODEL_ID, "source_revision": REVISION,
              "q4_sha256": Q4_SHA256, "examples": len(records), "command": command,
              "elapsed_seconds": time.monotonic() - started, "server_stopped": process.poll() is not None,
              "all_hf_fast_slow_match": all(r["hf_fast_slow_match"] for r in records),
              "all_native_tokenizer_match": all(r["native_tokenizer_match"] for r in records),
              "bf16_parity_validated": False, "parameter_decomposition_trained": False}
    (args.output_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
