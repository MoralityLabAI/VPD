"""Package only the Nanbeige preflight sources, never credentials or local models."""
import hashlib
import json
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
destination = root / "artifacts" / "nanbeige_looped" / "preflight_bundle.tar.gz"
sources = [root / relative for relative in (
    "experiments/nanbeige_looped/__init__.py",
    "experiments/nanbeige_looped/capture.py",
    "experiments/nanbeige_looped/preflight.py",
    "experiments/nanbeige_looped/requirements.txt",
    "scripts/nanbeige/run_preflight_remote.sh",
)]
destination.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(destination, "w:gz") as archive:
    for source in sources:
        archive.add(source, arcname=source.relative_to(root).as_posix(), recursive=False)
receipt = {"bundle": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
           "files": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sources}}
destination.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt))
