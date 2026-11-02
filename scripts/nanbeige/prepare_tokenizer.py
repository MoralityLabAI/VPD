"""Download the official pinned tokenizer/config/code, without BF16 weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL = "Nanbeige/Nanbeige4.2-3B"
REVISION = "451ed48c3273ecef7ea8faaa43c31ce529763bb1"


def main() -> None:
    destination = Path(__file__).resolve().parents[2] / "models" / "Nanbeige4.2-3B-tokenizer"
    snapshot_download(MODEL, revision=REVISION, local_dir=destination,
                      allow_patterns=["*.json", "*.py", "*.model", "*.txt"], max_workers=2)
    files = {str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in destination.iterdir() if path.is_file() and path.name != "provenance.json"}
    provenance = {"model_id": MODEL, "revision": REVISION, "contains_weights": False, "files": files}
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "tokenizer_downloaded", "path": str(destination), "files": len(files)}))


if __name__ == "__main__":
    main()
