#!/usr/bin/env bash
set -euo pipefail
export NANBEIGE_REASONING_MODE=vpd-precision
export NANBEIGE_COMPUTE_PROFILE=runpod_ada
export NANBEIGE_REASONING_REGION=US
export NANBEIGE_LAUNCH_BOUND_SECONDS=7200
exec bash /mnt/c/projects/VPD/scripts/nanbeige/launch_reasoning.sh "$@"
