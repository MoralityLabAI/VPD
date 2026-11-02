#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/ubuntu/nanbeige-reasoning
mkdir -p artifacts/nanbeige_reasoning/cloud_job
finish() {
  code=$?
  trap - EXIT INT TERM HUP
  printf '{"exit_code":%s}\n' "$code" > artifacts/nanbeige_reasoning/cloud_job/exit.json.tmp
  mv artifacts/nanbeige_reasoning/cloud_job/exit.json.tmp artifacts/nanbeige_reasoning/cloud_job/exit.json
  exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
bash scripts/nanbeige/run_logic_skill_remote.sh
