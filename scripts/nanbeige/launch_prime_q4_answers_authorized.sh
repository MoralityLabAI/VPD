#!/usr/bin/env bash
set -euo pipefail
export NANBEIGE_REASONING_REGION=US
export NANBEIGE_COMPUTE_PROFILE=massed_ada
export NANBEIGE_MAX_HOURLY=0.75
export NANBEIGE_LAUNCH_BOUND_SECONDS=28800
unset NANBEIGE_LAUNCHER_FROZEN
exec bash /mnt/c/projects/VPD/scripts/nanbeige/launch_prime_q4_answers.sh
