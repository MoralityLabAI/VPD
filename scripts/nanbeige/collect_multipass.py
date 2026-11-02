"""Extract and verify this pilot's retrieved artifacts inside its run directory."""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(run_dir):
    run_dir = run_dir.resolve()
    archive_path = run_dir / "results.tar.gz"
    destination = run_dir / "retrieved"
    if destination.exists():
        raise ValueError("Retrieval directory already exists")
    destination.mkdir()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if sum(member.size for member in members) > 4 * 1024 ** 3:
            raise ValueError("Retrieved artifacts exceed the 4 GiB calibration limit")
        for member in members:
            relative = PurePosixPath(member.name)
            target = (destination / member.name).resolve()
            if (relative.is_absolute() or ".." in relative.parts or "\\" in member.name
                    or not relative.parts or relative.parts[0] != "artifacts"
                    or not target.is_relative_to(destination)
                    or not (member.isdir() or member.isfile())):
                raise ValueError(f"Unsafe archive member: {member.name}")
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    root = destination / "artifacts/nanbeige_looped"
    summary = {"archive_sha256": sha256(archive_path), "verified_files": 0,
               "preflight_passed": False, "calibration_complete": False}
    preflight = root / "bf16_preflight"
    if (preflight / "manifest.json").exists():
        manifest = json.loads((preflight / "manifest.json").read_text())
        if manifest.get("status") != "preflight_passed":
            raise ValueError("Preflight manifest does not report success")
        for line in (preflight / "metadata.jsonl").read_text().splitlines():
            row = json.loads(line)
            path = (preflight / row["archive"]).resolve()
            if not path.is_relative_to(preflight) or sha256(path) != row["sha256"]:
                raise ValueError("Preflight artifact hash mismatch")
            summary["verified_files"] += 1
        summary["preflight_passed"] = True
    calibration = root / "bf16_multipass_calibration"
    if (calibration / "manifest.json").exists():
        manifest = json.loads((calibration / "manifest.json").read_text())
        if manifest.get("status") != "calibration_complete":
            raise ValueError("Calibration manifest does not report completion")
        for filename, expected in manifest["artifacts"].items():
            path = (calibration / filename).resolve()
            if not path.is_relative_to(calibration) or sha256(path) != expected:
                raise ValueError(f"Calibration artifact hash mismatch: {filename}")
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as arrays:
                    for key in arrays.files:
                        value = arrays[key]
                        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                            raise ValueError(f"Nonfinite capture: {filename}:{key}")
            summary["verified_files"] += 1
        summary["calibration_complete"] = True
        summary["report"] = str(calibration / "REPORT.md")
    (run_dir / "retrieval_verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args.run_dir), indent=2))
