#!/usr/bin/env bash
set -euo pipefail
export NANBEIGE_COMPUTE_PROFILE=lambda_a100 NANBEIGE_REASONING_REGION=US NANBEIGE_MAX_HOURLY=1.99 NANBEIGE_LAUNCH_BOUND_SECONDS=19800 NANBEIGE_IMAGE=cuda_12_4_pytorch_2_4
unset NANBEIGE_LAUNCHER_FROZEN
exec bash /mnt/c/projects/VPD/scripts/nanbeige/launch_logic_skill.sh
