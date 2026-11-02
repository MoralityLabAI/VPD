#!/usr/bin/env bash
set -Eeuo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/workspace/hrm-base-conformance-v1}"
RESULT_DIR="${REMOTE_ROOT}/results"
mkdir -p "${RESULT_DIR}"

python3 -m venv --system-site-packages "${REMOTE_ROOT}/venv"
source "${REMOTE_ROOT}/venv/bin/activate"
python -m pip install --upgrade pip \
  > "${RESULT_DIR}/pip_upgrade_stdout.log" \
  2> "${RESULT_DIR}/pip_upgrade_stderr.log"
set +e
python -m pip install --requirement "${REMOTE_ROOT}/requirements.txt" \
  > "${RESULT_DIR}/install_stdout.log" \
  2> "${RESULT_DIR}/install_stderr.log"
INSTALL_EXIT=$?
set -e
python -c 'import json,sys; json.dump({"status":"passed" if int(sys.argv[1])==0 else "failed","exit_code":int(sys.argv[1])},open(sys.argv[2],"w"),indent=2,sort_keys=True)' \
  "${INSTALL_EXIT}" "${RESULT_DIR}/package_setup_receipt.json"
if [[ "${INSTALL_EXIT}" -ne 0 ]]; then
  exit "${INSTALL_EXIT}"
fi
python -m pip freeze > "${RESULT_DIR}/pip_freeze.txt"

set +e
timeout --signal=TERM --kill-after=60s 2700s \
  stdbuf -oL -eL \
  python "${REMOTE_ROOT}/remote_conformance.py" \
    --output-dir "${RESULT_DIR}" \
    --probe-pack "${REMOTE_ROOT}/probe_pack.json" \
    --model-code-dir "${REMOTE_ROOT}/fused_repaired_composite" \
    > "${RESULT_DIR}/runner_stdout.log" \
    2> "${RESULT_DIR}/runner_stderr.log"
RUN_EXIT=$?
set -e
printf '%s\n' "${RUN_EXIT}" > "${RESULT_DIR}/remote_exit_code.txt"

if [[ "${RUN_EXIT}" -ne 0 ]]; then
  exit "${RUN_EXIT}"
fi
