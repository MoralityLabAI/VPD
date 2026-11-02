"""Run the existing WSL Prime CLI with the saved key, never in command arguments."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    env = os.environ.copy()
    arguments = sys.argv[1:]
    cached = bool(arguments and arguments[0] == "--cached")
    if cached:
        arguments = arguments[1:]
    source = Path("D:/research_engine/prime_lab/.env")
    key = None
    for line in source.read_text(encoding="utf-8-sig").splitlines() if source.exists() else []:
        name, sep, value = line.partition("=")
        if sep and name.strip() == "PRIME_API_KEY":
            key = value.strip().strip("\"'")
    if not key and not cached:
        raise SystemExit("Saved PRIME_API_KEY is absent")
    if cached:
        env.pop("PRIME_API_KEY", None)
    else:
        env["PRIME_API_KEY"] = key
    env["PRIME_DISABLE_VERSION_CHECK"] = "1"
    env["WSLENV"] = ":".join(filter(None, [env.get("WSLENV", ""), "PRIME_API_KEY", "PRIME_DISABLE_VERSION_CHECK"]))
    run_script = bool(arguments and arguments[0] == "--script")
    command = ["wsl.exe", "-d", "Ubuntu", "--exec"]
    command += ["/bin/bash", *arguments[1:]] if run_script else ["/home/duganist/.local/bin/prime", "--plain", *arguments]
    if run_script or (arguments and arguments[0] == "login"):
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line.replace(key, "[REDACTED]") if key else line, end="", flush=True)
            return process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
    result = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    sys.stdout.write(result.stdout.replace(key, "[REDACTED]") if key else result.stdout)
    sys.stderr.write(result.stderr.replace(key, "[REDACTED]") if key else result.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
