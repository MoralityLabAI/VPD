#!/usr/bin/env bash
set -euo pipefail
export NANBEIGE_REASONING_MODE=benchmark
export NANBEIGE_COMPUTE_PROFILE=runpod_ada
export NANBEIGE_REASONING_REGION=US
exec bash /mnt/c/projects/VPD/scripts/nanbeige/launch_reasoning.sh "$@"
