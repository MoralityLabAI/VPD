#!/usr/bin/env bash
set -euo pipefail
export NANBEIGE_REASONING_REGION=US
export NANBEIGE_COMPUTE_PROFILE=runpod_a100
export NANBEIGE_REASONING_MODE=cache-diagnostics
exec bash /mnt/c/projects/VPD/scripts/nanbeige/launch_reasoning.sh "$@"
