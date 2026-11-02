#!/usr/bin/env bash
set -euo pipefail
export NANBEIGE_REASONING_MODE=reasoning-vpd
export NANBEIGE_COMPUTE_PROFILE=lambda_a100
export NANBEIGE_REASONING_REGION=IN
exec bash /mnt/c/projects/VPD/scripts/nanbeige/launch_reasoning.sh "$@"
