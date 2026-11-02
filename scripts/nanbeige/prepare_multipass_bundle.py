"""Bundle only the public-model experiment sources; exclude credentials and weights."""
import hashlib
import json
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
destination = root / "artifacts/nanbeige_looped/multipass_bundle.tar.gz"
names = ("__init__.py", "capture.py", "preflight.py", "requirements.txt", "recurrent.py",
         "vpd_math.py", "vpd_backend_provenance.json", "VPD_BACKEND_LICENSE", "sae.py",
         "measure.py", "run_multipass.py")
sources = [root / "experiments/nanbeige_looped" / name for name in names]
sources.append(root / "scripts/nanbeige/run_multipass_remote.sh")
destination.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(destination, "w:gz") as archive:
    for source in sources:
        archive.add(source, arcname=source.relative_to(root).as_posix(), recursive=False)
receipt = {"bundle": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
           "files": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sources}}
destination.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"bundle": str(destination), "files": len(sources), "sha256": receipt["sha256"]}))
